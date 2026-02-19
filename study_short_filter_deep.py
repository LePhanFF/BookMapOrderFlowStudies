"""
DEEP ANALYSIS: Why Do Short Fades Fail? What Makes Them Work?

Consolidates findings from ALL short studies:
  1. test_ibh_fade.py: Fading AT IBH (12 configs, ALL negative)
  2. study_failed_breakout.py: 94% of IBH breaks fail, best short = 50% WR
  3. study_ibh_fail_confluence.py: More HTF confluence = LESS reversion (counterintuitive)
  4. study_ibh_sweep_daytype.py: b_day sweeps = 42.9% WR (10 wins, 16 losses)

Central question: What filter separates the 10 WINNERS from the 16 LOSERS
in b_day IBH sweep SHORT trades?

Dimensions tested:
  A. Sweep mechanics: sweep size, wick ratio, close vs IBH distance
  B. Order flow: delta, delta_zscore, cumulative_delta, volume_spike, imbalance
  C. Market structure: IB range, time of day, bars since IB, prior bar momentum
  D. HTF levels: PDH proximity (wider buffers), overnight high, prior 3-day high
  E. Regime: EMA20 vs EMA50 bull/bear, 5-day rolling returns
  F. Composite filters: best 2-3 factor combinations
"""

import sys
from pathlib import Path
from collections import defaultdict
from datetime import time as _time, timedelta
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.instruments import get_instrument
from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from config.constants import IB_BARS_1MIN
from strategy.day_type import classify_trend_strength, classify_day_type
from engine.execution import ExecutionModel

# Load data
instrument = get_instrument('MNQ')
df_raw = load_csv('NQ')
df = filter_rth(df_raw)
df = compute_all_features(df)

if 'session_date' not in df.columns:
    df['session_date'] = df['timestamp'].dt.date

sessions = sorted(df['session_date'].unique())
execution = ExecutionModel(instrument, slippage_ticks=1)

# Compute prior day levels and overnight highs
prior_levels = {}
overnight_highs = {}
prev_high = None
prev_low = None
prev_close = None
rolling_closes = []

for session_date in sessions:
    sdf = df[df['session_date'] == session_date]
    if len(sdf) == 0:
        continue
    prior_levels[str(session_date)] = {
        'pdh': prev_high, 'pdl': prev_low, 'pdc': prev_close,
    }
    prev_high = sdf['high'].max()
    prev_low = sdf['low'].min()
    prev_close = sdf['close'].iloc[-1]
    rolling_closes.append((session_date, prev_close))

# Compute prior 3-day high
p3d_highs = {}
session_highs = []
for session_date in sessions:
    sdf = df[df['session_date'] == session_date]
    if len(sdf) > 0:
        session_highs.append((session_date, sdf['high'].max()))
    if len(session_highs) >= 4:
        p3d_highs[str(session_date)] = max(h for _, h in session_highs[-4:-1])

# Overnight (globex) highs
df_raw_ts = df_raw.copy()
if 'timestamp' not in df_raw_ts.columns and isinstance(df_raw_ts.index, pd.DatetimeIndex):
    df_raw_ts['timestamp'] = df_raw_ts.index
df_raw_ts['timestamp'] = pd.to_datetime(df_raw_ts['timestamp'])
df_raw_ts['time'] = df_raw_ts['timestamp'].dt.time

for session_date in sessions:
    prior_day = session_date - timedelta(days=1)
    eth_mask = (
        ((df_raw_ts['timestamp'].dt.date == prior_day) & (df_raw_ts['time'] >= _time(18, 0))) |
        ((df_raw_ts['timestamp'].dt.date == session_date) & (df_raw_ts['time'] < _time(9, 30)))
    )
    eth_df = df_raw_ts[eth_mask]
    overnight_highs[str(session_date)] = eth_df['high'].max() if len(eth_df) > 0 else None

# 5-day rolling returns for regime
rolling_returns_5d = {}
for i, session_date in enumerate(sessions):
    if i >= 5:
        close_5d_ago = None
        for _, c in rolling_closes:
            if _ == sessions[i - 5]:
                close_5d_ago = c
                break
        if close_5d_ago and prev_close:
            sdf = df[df['session_date'] == session_date]
            if len(sdf) > 0:
                current = sdf['close'].iloc[-1]
                rolling_returns_5d[str(session_date)] = (current - close_5d_ago) / close_5d_ago


SWEEP_MIN_PTS = 5.0

print("=" * 140)
print("  DEEP SHORT FILTER ANALYSIS — WHAT MAKES IBH SWEEP SHORTS WIN VS LOSE?")
print(f"  Data: {len(sessions)} sessions | Sweep threshold: {SWEEP_MIN_PTS}+ pts above IBH")
print("=" * 140)

# ============================================================================
# PHASE 1: COLLECT ALL B-DAY IBH SWEEP EVENTS WITH RICH FEATURES
# ============================================================================
print("\n" + "=" * 140)
print("  PHASE 1: ENRICH ALL B-DAY IBH SWEEP EVENTS WITH DEEP FEATURES")
print("=" * 140)

all_sweeps = []

for session_date in sessions:
    date_str = str(session_date)
    sdf = df[df['session_date'] == session_date].copy()
    if len(sdf) < IB_BARS_1MIN + 10:
        continue

    ib_df = sdf.head(IB_BARS_1MIN)
    post_ib = sdf.iloc[IB_BARS_1MIN:]

    ib_high = ib_df['high'].max()
    ib_low = ib_df['low'].min()
    ib_range = ib_high - ib_low
    ib_mid = (ib_high + ib_low) / 2

    if ib_range <= 0:
        continue

    pdh = prior_levels.get(date_str, {}).get('pdh')
    pdc = prior_levels.get(date_str, {}).get('pdc')
    ovn_high = overnight_highs.get(date_str)
    p3d_high = p3d_highs.get(date_str)

    session_high = ib_high
    sweep_taken = False

    for i, (idx, bar) in enumerate(post_ib.iterrows()):
        price = bar['close']
        bar_time = bar['timestamp'].time() if 'timestamp' in bar.index else None

        if bar['high'] > session_high:
            session_high = bar['high']

        if bar_time and (bar_time < _time(10, 30) or bar_time >= _time(14, 0)):
            continue

        sweep_pts = bar['high'] - ib_high
        if sweep_pts < SWEEP_MIN_PTS:
            continue
        if price >= ib_high:
            continue

        # Day type classification at sweep moment
        if price > ib_high:
            ib_dir = 'BULL'
            ext = (price - ib_mid) / ib_range
        elif price < ib_low:
            ib_dir = 'BEAR'
            ext = (ib_mid - price) / ib_range
        else:
            ib_dir = 'INSIDE'
            ext = 0.0

        strength = classify_trend_strength(ext)
        day_type = classify_day_type(ib_high, ib_low, price, ib_dir, strength)
        dt_val = day_type.value if hasattr(day_type, 'value') else str(day_type)

        # Only b_day
        if dt_val != 'b_day':
            continue

        # ----- DEEP FEATURE EXTRACTION -----

        # A. Sweep mechanics
        delta_bar = bar.get('delta', 0)
        if pd.isna(delta_bar):
            delta_bar = 0
        bar_range = bar['high'] - bar['low']
        upper_wick = bar['high'] - max(bar['open'], price)
        wick_ratio = upper_wick / bar_range if bar_range > 0 else 0
        close_vs_ibh = ib_high - price  # positive = closed below IBH
        close_pct_ib = close_vs_ibh / ib_range if ib_range > 0 else 0
        sweep_mult = sweep_pts / ib_range if ib_range > 0 else 0

        # B. Order flow features from the sweep bar
        delta_zscore = bar.get('delta_zscore', 0)
        if pd.isna(delta_zscore):
            delta_zscore = 0
        delta_pct = bar.get('delta_pct', 0)
        if pd.isna(delta_pct):
            delta_pct = 0
        cumulative_delta = bar.get('cumulative_delta', 0)
        if pd.isna(cumulative_delta):
            cumulative_delta = 0
        volume_spike = bar.get('volume_spike', 1)
        if pd.isna(volume_spike):
            volume_spike = 1
        imbalance = bar.get('imbalance_ratio', 1)
        if pd.isna(imbalance):
            imbalance = 1
        volume = bar.get('volume', 0)
        if pd.isna(volume):
            volume = 0

        # C. Prior bar momentum (5-bar lookback before sweep)
        lookback = 5
        start_idx = max(0, i - lookback)
        prior_bars = post_ib.iloc[start_idx:i]
        if len(prior_bars) >= 2:
            prior_delta_sum = prior_bars['delta'].sum() if 'delta' in prior_bars.columns else 0
            prior_price_change = prior_bars['close'].iloc[-1] - prior_bars['close'].iloc[0]
            prior_vol_avg = prior_bars['volume'].mean() if 'volume' in prior_bars.columns else 0
        else:
            prior_delta_sum = 0
            prior_price_change = 0
            prior_vol_avg = 0

        # D. HTF levels with wider buffers
        near_pdh_tight = False
        near_pdh_wide = False
        near_ovn_tight = False
        near_ovn_wide = False
        near_p3d = False
        at_htf_resistance = False

        zone_tight = ib_range * 0.10
        zone_wide = ib_range * 0.30

        if pdh is not None:
            near_pdh_tight = abs(bar['high'] - pdh) <= zone_tight
            near_pdh_wide = abs(bar['high'] - pdh) <= zone_wide
        if ovn_high is not None:
            near_ovn_tight = abs(bar['high'] - ovn_high) <= zone_tight
            near_ovn_wide = abs(bar['high'] - ovn_high) <= zone_wide
        if p3d_high is not None:
            near_p3d = abs(bar['high'] - p3d_high) <= zone_wide

        at_htf_resistance = near_pdh_wide or near_ovn_wide or near_p3d

        # E. Gap analysis — did we gap up today?
        gap_up = False
        if pdc is not None:
            session_open = sdf.iloc[0]['open']
            gap_up = session_open > pdc

        # F. EMA regime (use prior day close and 20/50 EMA)
        # Approximate: is current IB mid above/below a trailing average?
        ema_bull = None
        if len(rolling_closes) >= 20:
            closes_20 = [c for _, c in rolling_closes[-20:]]
            ema20 = np.mean(closes_20)
            ema_bull = ib_mid > ema20

        # G. Time bucket
        if bar_time:
            hour = bar_time.hour
            minute = bar_time.minute
            minutes_since_open = (hour - 9) * 60 + (minute - 30)
        else:
            minutes_since_open = 0

        time_bucket = 'early' if minutes_since_open <= 90 else ('mid' if minutes_since_open <= 180 else 'late')

        # H. IB range relative to recent ranges
        ib_range_rel = None

        # I. Sweep bar open vs close — did bar open above IBH? (gap probe)
        bar_opened_above = bar['open'] > ib_high

        # J. Distance from sweep high to any HTF level
        htf_dist = float('inf')
        for level in [pdh, ovn_high, p3d_high]:
            if level is not None:
                d = abs(bar['high'] - level)
                if d < htf_dist:
                    htf_dist = d
        if htf_dist == float('inf'):
            htf_dist = None

        sweep = {
            'date': date_str,
            'bar_idx': i,
            'bar_time': str(bar_time),
            'minutes_since_open': minutes_since_open,
            'time_bucket': time_bucket,
            # IB levels
            'ib_high': ib_high, 'ib_low': ib_low,
            'ib_range': ib_range, 'ib_mid': ib_mid,
            # Sweep mechanics
            'sweep_pts': sweep_pts,
            'sweep_mult': sweep_mult,
            'sweep_high': bar['high'],
            'close': price,
            'close_vs_ibh': close_vs_ibh,
            'close_pct_ib': close_pct_ib,
            'wick_ratio': wick_ratio,
            'bar_opened_above': bar_opened_above,
            # Order flow
            'delta': delta_bar,
            'delta_zscore': delta_zscore,
            'delta_pct': delta_pct,
            'cumulative_delta': cumulative_delta,
            'volume_spike': volume_spike,
            'imbalance': imbalance,
            'volume': volume,
            # Prior momentum
            'prior_delta_sum': prior_delta_sum,
            'prior_price_change': prior_price_change,
            'prior_vol_avg': prior_vol_avg,
            # HTF levels
            'near_pdh_tight': near_pdh_tight,
            'near_pdh_wide': near_pdh_wide,
            'near_ovn_tight': near_ovn_tight,
            'near_ovn_wide': near_ovn_wide,
            'near_p3d': near_p3d,
            'at_htf_resistance': at_htf_resistance,
            'htf_dist': htf_dist,
            # Context
            'gap_up': gap_up,
            'ema_bull': ema_bull,
            'first_sweep': not sweep_taken,
            'day_type': dt_val,
        }
        all_sweeps.append(sweep)

        if not sweep_taken:
            sweep_taken = True

first_sweeps = [s for s in all_sweeps if s['first_sweep']]
print(f"\nTotal b_day sweep events: {len(all_sweeps)}")
print(f"First-sweep-only per session: {len(first_sweeps)}")


# ============================================================================
# PHASE 2: SIMULATE ALL FIRST SWEEPS → CLASSIFY WINNERS VS LOSERS
# ============================================================================
print("\n\n" + "=" * 140)
print("  PHASE 2: SIMULATE & CLASSIFY — WINNERS VS LOSERS")
print("=" * 140)

trades_enriched = []

for sweep in first_sweeps:
    date_str = sweep['date']
    sdf = df[df['session_date'].astype(str) == date_str[:10]].copy()
    if len(sdf) < IB_BARS_1MIN + 10:
        continue

    post_ib = sdf.iloc[IB_BARS_1MIN:]
    ib_mid = sweep['ib_mid']

    entry_raw = sweep['close']
    entry_fill = execution.fill_entry('SHORT', entry_raw)

    stop_price = sweep['sweep_high'] + (sweep['ib_range'] * 0.15)
    stop_price = max(stop_price, entry_fill + 15.0)
    target_price = ib_mid

    risk = stop_price - entry_fill
    reward = entry_fill - target_price
    if reward <= 0 or risk <= 0:
        continue

    rr_ratio = reward / risk

    entry_bar = sweep['bar_idx']
    trade_result = None

    for j in range(entry_bar + 1, len(post_ib)):
        bar = post_ib.iloc[j]
        bar_time = bar['timestamp'].time() if 'timestamp' in bar.index else None

        if bar['high'] >= stop_price:
            exit_fill = execution.fill_exit('SHORT', stop_price)
            _, _, _, net = execution.calculate_net_pnl('SHORT', entry_fill, exit_fill, 5)
            trade_result = {'exit_reason': 'STOP', 'net_pnl': net, 'bars_held': j - entry_bar,
                           'exit_price': exit_fill, 'max_favorable': 0, 'max_adverse': 0}
            break

        if bar['low'] <= target_price:
            exit_fill = execution.fill_exit('SHORT', target_price)
            _, _, _, net = execution.calculate_net_pnl('SHORT', entry_fill, exit_fill, 5)
            trade_result = {'exit_reason': 'TARGET', 'net_pnl': net, 'bars_held': j - entry_bar,
                           'exit_price': exit_fill, 'max_favorable': 0, 'max_adverse': 0}
            break

        if bar_time and bar_time >= _time(15, 0):
            exit_fill = execution.fill_exit('SHORT', bar['close'])
            _, _, _, net = execution.calculate_net_pnl('SHORT', entry_fill, exit_fill, 5)
            trade_result = {'exit_reason': 'EOD', 'net_pnl': net, 'bars_held': j - entry_bar,
                           'exit_price': exit_fill, 'max_favorable': 0, 'max_adverse': 0}
            break

    if trade_result is None and entry_bar < len(post_ib) - 1:
        last_bar = post_ib.iloc[-1]
        exit_fill = execution.fill_exit('SHORT', last_bar['close'])
        _, _, _, net = execution.calculate_net_pnl('SHORT', entry_fill, exit_fill, 5)
        trade_result = {'exit_reason': 'EOD', 'net_pnl': net, 'bars_held': len(post_ib) - entry_bar,
                       'exit_price': exit_fill, 'max_favorable': 0, 'max_adverse': 0}

    if trade_result:
        # Calculate max favorable / adverse excursion
        mfe = 0  # max favorable (price dropped below entry = good for short)
        mae = 0  # max adverse (price rose above entry = bad for short)
        for j in range(entry_bar + 1, min(entry_bar + 1 + trade_result['bars_held'], len(post_ib))):
            bar = post_ib.iloc[j]
            favorable = entry_fill - bar['low']
            adverse = bar['high'] - entry_fill
            mfe = max(mfe, favorable)
            mae = max(mae, adverse)

        trade = {**sweep, **trade_result}
        trade['entry_fill'] = entry_fill
        trade['stop'] = stop_price
        trade['target'] = target_price
        trade['risk'] = risk
        trade['reward'] = reward
        trade['rr_ratio'] = rr_ratio
        trade['winner'] = trade_result['net_pnl'] > 0
        trade['mfe'] = mfe
        trade['mae'] = mae
        trades_enriched.append(trade)

winners = [t for t in trades_enriched if t['winner']]
losers = [t for t in trades_enriched if not t['winner']]

print(f"\nTotal trades: {len(trades_enriched)}")
print(f"Winners: {len(winners)} ({len(winners)/max(len(trades_enriched),1)*100:.0f}%)")
print(f"Losers:  {len(losers)} ({len(losers)/max(len(trades_enriched),1)*100:.0f}%)")
print(f"Net PnL: ${sum(t['net_pnl'] for t in trades_enriched):,.0f}")


# ============================================================================
# PHASE 3: HEAD-TO-HEAD FEATURE COMPARISON — WINNERS VS LOSERS
# ============================================================================
print("\n\n" + "=" * 140)
print("  PHASE 3: FEATURE COMPARISON — WHAT DISTINGUISHES WINNERS FROM LOSERS?")
print("=" * 140)

def compare_feature(feature_name, label=None):
    if not label:
        label = feature_name
    w_vals = [t[feature_name] for t in winners if t[feature_name] is not None and not (isinstance(t[feature_name], float) and np.isnan(t[feature_name]))]
    l_vals = [t[feature_name] for t in losers if t[feature_name] is not None and not (isinstance(t[feature_name], float) and np.isnan(t[feature_name]))]

    if not w_vals or not l_vals:
        print(f"  {label:<30s}: insufficient data")
        return

    w_mean = np.mean(w_vals)
    l_mean = np.mean(l_vals)
    w_med = np.median(w_vals)
    l_med = np.median(l_vals)
    w_std = np.std(w_vals)
    l_std = np.std(l_vals)

    diff_pct = abs(w_mean - l_mean) / max(abs(l_mean), 0.001) * 100

    # Separation score: how different are the distributions?
    pooled_std = np.sqrt((w_std**2 + l_std**2) / 2) if (w_std + l_std) > 0 else 0.001
    cohens_d = abs(w_mean - l_mean) / pooled_std

    signal = "***" if cohens_d > 0.8 else ("** " if cohens_d > 0.5 else ("*  " if cohens_d > 0.3 else "   "))

    print(f"  {signal} {label:<28s}: W={w_mean:>10.2f} (med {w_med:>8.2f})  L={l_mean:>10.2f} (med {l_med:>8.2f})  "
          f"d={cohens_d:.2f}  diff={diff_pct:.0f}%")

def compare_boolean(feature_name, label=None):
    if not label:
        label = feature_name
    w_pct = sum(1 for t in winners if t[feature_name]) / max(len(winners), 1) * 100
    l_pct = sum(1 for t in losers if t[feature_name]) / max(len(losers), 1) * 100
    diff = abs(w_pct - l_pct)
    signal = "***" if diff > 30 else ("** " if diff > 20 else ("*  " if diff > 10 else "   "))
    print(f"  {signal} {label:<28s}: W={w_pct:>5.0f}%  L={l_pct:>5.0f}%  gap={diff:.0f}pp")


print("\n--- A. SWEEP MECHANICS ---")
compare_feature('sweep_pts', 'Sweep size (pts)')
compare_feature('sweep_mult', 'Sweep as IB multiple')
compare_feature('wick_ratio', 'Upper wick ratio')
compare_feature('close_vs_ibh', 'Close below IBH (pts)')
compare_feature('close_pct_ib', 'Close below IBH (% IB)')

print("\n--- B. ORDER FLOW ON SWEEP BAR ---")
compare_feature('delta', 'Bar delta')
compare_feature('delta_zscore', 'Delta z-score')
compare_feature('delta_pct', 'Delta %')
compare_feature('cumulative_delta', 'Cumulative delta')
compare_feature('volume_spike', 'Volume spike ratio')
compare_feature('imbalance', 'Imbalance ratio')
compare_feature('volume', 'Volume')

print("\n--- C. PRIOR MOMENTUM (5-bar lookback) ---")
compare_feature('prior_delta_sum', 'Prior 5-bar delta sum')
compare_feature('prior_price_change', 'Prior 5-bar price change')
compare_feature('prior_vol_avg', 'Prior 5-bar avg volume')

print("\n--- D. MARKET STRUCTURE ---")
compare_feature('ib_range', 'IB range (pts)')
compare_feature('minutes_since_open', 'Minutes since open')
compare_feature('risk', 'Risk (stop distance)')
compare_feature('reward', 'Reward (target distance)')
compare_feature('rr_ratio', 'R:R ratio')
compare_feature('mfe', 'Max favorable excursion')
compare_feature('mae', 'Max adverse excursion')

print("\n--- E. HTF PROXIMITY ---")
compare_boolean('near_pdh_tight', 'Near PDH (tight)')
compare_boolean('near_pdh_wide', 'Near PDH (wide)')
compare_boolean('near_ovn_tight', 'Near overnight H (tight)')
compare_boolean('near_ovn_wide', 'Near overnight H (wide)')
compare_boolean('near_p3d', 'Near 3-day high')
compare_boolean('at_htf_resistance', 'Any HTF resistance')
compare_feature('htf_dist', 'Distance to nearest HTF')

print("\n--- F. CONTEXT ---")
compare_boolean('gap_up', 'Gap up day')
compare_boolean('bar_opened_above', 'Bar opened above IBH')
compare_boolean('ema_bull', 'EMA20 bull regime')

print("\n--- G. TIME BUCKET ---")
for bucket in ['early', 'mid', 'late']:
    w_pct = sum(1 for t in winners if t['time_bucket'] == bucket) / max(len(winners), 1) * 100
    l_pct = sum(1 for t in losers if t['time_bucket'] == bucket) / max(len(losers), 1) * 100
    diff = abs(w_pct - l_pct)
    signal = "***" if diff > 30 else ("** " if diff > 20 else ("*  " if diff > 10 else "   "))
    print(f"  {signal} {bucket + ' (<=90/<=180/>180 min)':<28s}: W={w_pct:>5.0f}%  L={l_pct:>5.0f}%  gap={diff:.0f}pp")


# ============================================================================
# PHASE 4: DETAILED TRADE-BY-TRADE COMPARISON
# ============================================================================
print("\n\n" + "=" * 140)
print("  PHASE 4: TRADE-BY-TRADE DETAIL")
print("=" * 140)

header = (f"  {'Date':<12s} {'W/L':>3s} {'PnL':>9s} {'Reason':<7s} {'SweepPts':>8s} "
          f"{'SwpMlt':>6s} {'Wick':>5s} {'ClsBelIBH':>9s} {'Delta':>8s} "
          f"{'DltZ':>6s} {'VolSpk':>6s} {'IbRng':>6s} {'Time':>8s} "
          f"{'PDH':>3s} {'OvnH':>4s} {'P3D':>3s} {'GapUp':>5s} {'R:R':>5s} "
          f"{'MFE':>7s} {'MAE':>7s} {'PriorDlt':>8s}")
print(header)
print("  " + "-" * (len(header) - 2))

for t in sorted(trades_enriched, key=lambda x: x['date']):
    wl = "W" if t['winner'] else "L"
    htf_dist_str = f"{t['htf_dist']:.0f}" if t['htf_dist'] is not None else "-"
    print(f"  {t['date']:<12s} {wl:>3s} ${t['net_pnl']:>7,.0f} {t['exit_reason']:<7s} "
          f"{t['sweep_pts']:>7.1f} {t['sweep_mult']:>6.3f} {t['wick_ratio']:>5.2f} "
          f"{t['close_vs_ibh']:>8.1f} {t['delta']:>+7.0f} "
          f"{t['delta_zscore']:>+5.1f} {t['volume_spike']:>6.1f} "
          f"{t['ib_range']:>6.1f} {t['bar_time'][-8:]:>8s} "
          f"{'Y' if t['near_pdh_wide'] else 'N':>3s} {'Y' if t['near_ovn_wide'] else 'N':>4s} "
          f"{'Y' if t['near_p3d'] else 'N':>3s} {'Y' if t['gap_up'] else 'N':>5s} "
          f"{t['rr_ratio']:>5.2f} {t['mfe']:>7.1f} {t['mae']:>7.1f} "
          f"{t['prior_delta_sum']:>+7.0f}")


# ============================================================================
# PHASE 5: TEST CANDIDATE FILTERS — FIND THE BEST DISCRIMINATOR
# ============================================================================
print("\n\n" + "=" * 140)
print("  PHASE 5: CANDIDATE FILTER TESTS")
print("=" * 140)

def test_filter(label, condition_fn):
    """Test a filter on all trades, return stats."""
    passed = [t for t in trades_enriched if condition_fn(t)]
    failed = [t for t in trades_enriched if not condition_fn(t)]

    if not passed:
        print(f"  {label:<60s}: 0 passed")
        return None

    p_wins = sum(1 for t in passed if t['winner'])
    p_wr = p_wins / len(passed) * 100
    p_pnl = sum(t['net_pnl'] for t in passed)

    f_wins = sum(1 for t in failed if t['winner'])
    f_wr = f_wins / max(len(failed), 1) * 100
    f_pnl = sum(t['net_pnl'] for t in failed)

    signal = ">>>" if p_wr >= 60 and len(passed) >= 5 else ("** " if p_wr >= 55 else "   ")

    print(f"  {signal} {label:<57s}: PASS {p_wins}/{len(passed)} = {p_wr:>5.1f}% WR ${p_pnl:>7,.0f}  |  "
          f"FAIL {f_wins}/{len(failed)} = {f_wr:>5.1f}% WR ${f_pnl:>7,.0f}")

    return {'label': label, 'passed': passed, 'wr': p_wr, 'pnl': p_pnl, 'n': len(passed)}


print("\n--- A. SWEEP SIZE FILTERS ---")
results = []
for thresh in [5, 7, 10, 15, 20, 25]:
    r = test_filter(f"Sweep >= {thresh} pts", lambda t, th=thresh: t['sweep_pts'] >= th)
    if r:
        results.append(r)
for thresh in [5, 7, 10, 15]:
    r = test_filter(f"Sweep <= {thresh} pts", lambda t, th=thresh: t['sweep_pts'] <= th)
    if r:
        results.append(r)

print("\n--- B. SWEEP AS IB MULTIPLE ---")
for thresh in [0.05, 0.10, 0.15, 0.20, 0.30]:
    r = test_filter(f"Sweep mult >= {thresh:.2f}x IB", lambda t, th=thresh: t['sweep_mult'] >= th)
    if r: results.append(r)
for thresh in [0.05, 0.10, 0.15, 0.20]:
    r = test_filter(f"Sweep mult <= {thresh:.2f}x IB", lambda t, th=thresh: t['sweep_mult'] <= th)
    if r: results.append(r)

print("\n--- C. WICK RATIO FILTERS ---")
for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
    r = test_filter(f"Wick ratio >= {thresh:.1f}", lambda t, th=thresh: t['wick_ratio'] >= th)
    if r: results.append(r)

print("\n--- D. CLOSE VS IBH DISTANCE ---")
for thresh in [5, 10, 15, 20, 30]:
    r = test_filter(f"Closed >= {thresh} pts below IBH",
                    lambda t, th=thresh: t['close_vs_ibh'] >= th)
    if r: results.append(r)

print("\n--- E. DELTA FILTERS ---")
r = test_filter("Delta > 0 (positive = exhaustion)", lambda t: t['delta'] > 0)
if r: results.append(r)
r = test_filter("Delta < 0 (negative = absorption)", lambda t: t['delta'] < 0)
if r: results.append(r)
r = test_filter("Delta > +100 (strong buy exhaustion)", lambda t: t['delta'] > 100)
if r: results.append(r)
r = test_filter("Delta < -100 (strong sell absorption)", lambda t: t['delta'] < -100)
if r: results.append(r)
r = test_filter("Delta zscore > 0 (above avg)", lambda t: t['delta_zscore'] > 0)
if r: results.append(r)
r = test_filter("Delta zscore < 0 (below avg)", lambda t: t['delta_zscore'] < 0)
if r: results.append(r)
r = test_filter("|Delta zscore| > 1 (extreme)", lambda t: abs(t['delta_zscore']) > 1)
if r: results.append(r)

print("\n--- F. VOLUME FILTERS ---")
r = test_filter("Volume spike > 1.5x", lambda t: t['volume_spike'] > 1.5)
if r: results.append(r)
r = test_filter("Volume spike > 2.0x", lambda t: t['volume_spike'] > 2.0)
if r: results.append(r)
r = test_filter("Volume spike < 1.0x (low vol sweep)", lambda t: t['volume_spike'] < 1.0)
if r: results.append(r)

print("\n--- G. IB RANGE FILTERS ---")
for thresh in [20, 30, 40, 50, 60, 80]:
    r = test_filter(f"IB range >= {thresh} pts", lambda t, th=thresh: t['ib_range'] >= th)
    if r: results.append(r)
for thresh in [30, 40, 50, 60]:
    r = test_filter(f"IB range <= {thresh} pts", lambda t, th=thresh: t['ib_range'] <= th)
    if r: results.append(r)

print("\n--- H. TIME FILTERS ---")
for hour_cutoff in [11, 12, 13]:
    r = test_filter(f"Time <= {hour_cutoff}:00 (early sweep)",
                    lambda t, h=hour_cutoff: t['minutes_since_open'] <= (h - 9) * 60 - 30)
    if r: results.append(r)
    r = test_filter(f"Time > {hour_cutoff}:00 (late sweep)",
                    lambda t, h=hour_cutoff: t['minutes_since_open'] > (h - 9) * 60 - 30)
    if r: results.append(r)

print("\n--- I. HTF PROXIMITY ---")
r = test_filter("Near PDH (tight)", lambda t: t['near_pdh_tight'])
if r: results.append(r)
r = test_filter("Near PDH (wide)", lambda t: t['near_pdh_wide'])
if r: results.append(r)
r = test_filter("NOT near PDH (wide)", lambda t: not t['near_pdh_wide'])
if r: results.append(r)
r = test_filter("Near overnight high (wide)", lambda t: t['near_ovn_wide'])
if r: results.append(r)
r = test_filter("Near 3-day high", lambda t: t['near_p3d'])
if r: results.append(r)
r = test_filter("Any HTF resistance", lambda t: t['at_htf_resistance'])
if r: results.append(r)
r = test_filter("NO HTF resistance (clean)", lambda t: not t['at_htf_resistance'])
if r: results.append(r)

print("\n--- J. CONTEXT ---")
r = test_filter("Gap up day", lambda t: t['gap_up'])
if r: results.append(r)
r = test_filter("No gap up", lambda t: not t['gap_up'])
if r: results.append(r)
r = test_filter("Bar opened above IBH", lambda t: t['bar_opened_above'])
if r: results.append(r)
r = test_filter("Bar opened inside IB", lambda t: not t['bar_opened_above'])
if r: results.append(r)

print("\n--- K. PRIOR MOMENTUM ---")
r = test_filter("Prior 5-bar delta > 0 (buying into sweep)", lambda t: t['prior_delta_sum'] > 0)
if r: results.append(r)
r = test_filter("Prior 5-bar delta < 0 (selling into sweep)", lambda t: t['prior_delta_sum'] < 0)
if r: results.append(r)
r = test_filter("Prior 5-bar price up > 5 pts", lambda t: t['prior_price_change'] > 5)
if r: results.append(r)
r = test_filter("Prior 5-bar price down < -5 pts", lambda t: t['prior_price_change'] < -5)
if r: results.append(r)

print("\n--- L. R:R RATIO ---")
for thresh in [0.3, 0.5, 0.7, 1.0]:
    r = test_filter(f"R:R >= {thresh:.1f}", lambda t, th=thresh: t['rr_ratio'] >= th)
    if r: results.append(r)


# ============================================================================
# PHASE 6: TOP SINGLE FILTERS — RANKED BY WR (min 5 trades)
# ============================================================================
print("\n\n" + "=" * 140)
print("  PHASE 6: TOP SINGLE FILTERS (min 5 trades, ranked by WR)")
print("=" * 140)

valid_results = [r for r in results if r and r['n'] >= 5]
valid_results.sort(key=lambda x: (-x['wr'], -x['pnl']))

print(f"\n  {'Rank':>4s} {'Filter':<60s} {'Trades':>6s} {'WR':>6s} {'Net PnL':>10s}")
print("  " + "-" * 90)
for i, r in enumerate(valid_results[:20]):
    print(f"  {i+1:>4d} {r['label']:<60s} {r['n']:>6d} {r['wr']:>5.1f}% ${r['pnl']:>9,.0f}")


# ============================================================================
# PHASE 7: COMBO FILTERS — BEST 2-FACTOR COMBINATIONS
# ============================================================================
print("\n\n" + "=" * 140)
print("  PHASE 7: BEST 2-FACTOR FILTER COMBINATIONS (min 4 trades)")
print("=" * 140)

# Define reusable filter functions
filter_fns = {
    'sweep_small': ('Sweep <= 15 pts', lambda t: t['sweep_pts'] <= 15),
    'sweep_big': ('Sweep >= 15 pts', lambda t: t['sweep_pts'] >= 15),
    'sweep_tiny': ('Sweep <= 10 pts', lambda t: t['sweep_pts'] <= 10),
    'wick_high': ('Wick >= 0.5', lambda t: t['wick_ratio'] >= 0.5),
    'wick_low': ('Wick < 0.5', lambda t: t['wick_ratio'] < 0.5),
    'delta_pos': ('Delta > 0', lambda t: t['delta'] > 0),
    'delta_neg': ('Delta < 0', lambda t: t['delta'] < 0),
    'vol_spike': ('VolSpike > 1.5', lambda t: t['volume_spike'] > 1.5),
    'vol_low': ('VolSpike < 1.0', lambda t: t['volume_spike'] < 1.0),
    'ib_wide': ('IB >= 50', lambda t: t['ib_range'] >= 50),
    'ib_narrow': ('IB < 50', lambda t: t['ib_range'] < 50),
    'early': ('Before 12:00', lambda t: t['minutes_since_open'] <= 150),
    'late': ('After 12:00', lambda t: t['minutes_since_open'] > 150),
    'htf_near': ('HTF resistance', lambda t: t['at_htf_resistance']),
    'htf_none': ('No HTF resistance', lambda t: not t['at_htf_resistance']),
    'gap_up': ('Gap up', lambda t: t['gap_up']),
    'no_gap': ('No gap up', lambda t: not t['gap_up']),
    'close_far': ('Close >= 10 below IBH', lambda t: t['close_vs_ibh'] >= 10),
    'close_near': ('Close < 10 below IBH', lambda t: t['close_vs_ibh'] < 10),
    'prior_buy': ('Prior delta > 0', lambda t: t['prior_delta_sum'] > 0),
    'prior_sell': ('Prior delta < 0', lambda t: t['prior_delta_sum'] < 0),
    'rr_ok': ('R:R >= 0.5', lambda t: t['rr_ratio'] >= 0.5),
    'bar_above': ('Bar opened above IBH', lambda t: t['bar_opened_above']),
    'bar_inside': ('Bar opened inside IB', lambda t: not t['bar_opened_above']),
}

combo_results = []
filter_keys = list(filter_fns.keys())

for i, k1 in enumerate(filter_keys):
    for k2 in filter_keys[i + 1:]:
        label1, fn1 = filter_fns[k1]
        label2, fn2 = filter_fns[k2]

        passed = [t for t in trades_enriched if fn1(t) and fn2(t)]
        if len(passed) < 4:
            continue

        p_wins = sum(1 for t in passed if t['winner'])
        p_wr = p_wins / len(passed) * 100
        p_pnl = sum(t['net_pnl'] for t in passed)

        combo_results.append({
            'label': f"{label1} + {label2}",
            'n': len(passed),
            'wins': p_wins,
            'wr': p_wr,
            'pnl': p_pnl,
        })

combo_results.sort(key=lambda x: (-x['wr'], -x['pnl']))

print(f"\n  {'Rank':>4s} {'Filter Combo':<55s} {'Trades':>6s} {'WR':>6s} {'Net PnL':>10s}")
print("  " + "-" * 85)
for i, r in enumerate(combo_results[:25]):
    print(f"  {i+1:>4d} {r['label']:<55s} {r['n']:>6d} {r['wr']:>5.1f}% ${r['pnl']:>9,.0f}")


# ============================================================================
# PHASE 8: BEST 3-FACTOR COMBINATIONS
# ============================================================================
print("\n\n" + "=" * 140)
print("  PHASE 8: BEST 3-FACTOR FILTER COMBINATIONS (min 4 trades)")
print("=" * 140)

triple_results = []

for i, k1 in enumerate(filter_keys):
    for j, k2 in enumerate(filter_keys[i + 1:], i + 1):
        for k3 in filter_keys[j + 1:]:
            _, fn1 = filter_fns[k1]
            _, fn2 = filter_fns[k2]
            _, fn3 = filter_fns[k3]

            passed = [t for t in trades_enriched if fn1(t) and fn2(t) and fn3(t)]
            if len(passed) < 4:
                continue

            p_wins = sum(1 for t in passed if t['winner'])
            p_wr = p_wins / len(passed) * 100
            p_pnl = sum(t['net_pnl'] for t in passed)

            label1 = filter_fns[k1][0]
            label2 = filter_fns[k2][0]
            label3 = filter_fns[k3][0]

            triple_results.append({
                'label': f"{label1} + {label2} + {label3}",
                'n': len(passed),
                'wins': p_wins,
                'wr': p_wr,
                'pnl': p_pnl,
            })

triple_results.sort(key=lambda x: (-x['wr'], -x['pnl']))

print(f"\n  {'Rank':>4s} {'Filter Combo':<75s} {'Trades':>6s} {'WR':>6s} {'Net PnL':>10s}")
print("  " + "-" * 105)
for i, r in enumerate(triple_results[:25]):
    print(f"  {i+1:>4d} {r['label']:<75s} {r['n']:>6d} {r['wr']:>5.1f}% ${r['pnl']:>9,.0f}")


# ============================================================================
# PHASE 9: ALSO TEST WITH VWAP TARGET
# ============================================================================
print("\n\n" + "=" * 140)
print("  PHASE 9: REPEAT TOP FILTERS WITH VWAP TARGET")
print("=" * 140)

# Re-simulate with VWAP target
vwap_trades = []
for sweep in first_sweeps:
    date_str = sweep['date']
    sdf = df[df['session_date'].astype(str) == date_str[:10]].copy()
    if len(sdf) < IB_BARS_1MIN + 10:
        continue

    post_ib = sdf.iloc[IB_BARS_1MIN:]

    entry_raw = sweep['close']
    entry_fill = execution.fill_entry('SHORT', entry_raw)

    stop_price = sweep['sweep_high'] + (sweep['ib_range'] * 0.15)
    stop_price = max(stop_price, entry_fill + 15.0)

    # VWAP target
    sweep_bar_in_post = post_ib.iloc[sweep['bar_idx']] if sweep['bar_idx'] < len(post_ib) else None
    if sweep_bar_in_post is not None:
        vwap = sweep_bar_in_post.get('vwap', sweep['ib_mid'])
        target_price = vwap if not pd.isna(vwap) else sweep['ib_mid']
    else:
        target_price = sweep['ib_mid']

    risk = stop_price - entry_fill
    reward = entry_fill - target_price
    if reward <= 0 or risk <= 0:
        continue

    entry_bar = sweep['bar_idx']
    trade_result = None

    for j in range(entry_bar + 1, len(post_ib)):
        bar = post_ib.iloc[j]
        bar_time = bar['timestamp'].time() if 'timestamp' in bar.index else None

        if bar['high'] >= stop_price:
            exit_fill = execution.fill_exit('SHORT', stop_price)
            _, _, _, net = execution.calculate_net_pnl('SHORT', entry_fill, exit_fill, 5)
            trade_result = {'exit_reason': 'STOP', 'net_pnl': net, 'bars_held': j - entry_bar}
            break
        if bar['low'] <= target_price:
            exit_fill = execution.fill_exit('SHORT', target_price)
            _, _, _, net = execution.calculate_net_pnl('SHORT', entry_fill, exit_fill, 5)
            trade_result = {'exit_reason': 'TARGET', 'net_pnl': net, 'bars_held': j - entry_bar}
            break
        if bar_time and bar_time >= _time(15, 0):
            exit_fill = execution.fill_exit('SHORT', bar['close'])
            _, _, _, net = execution.calculate_net_pnl('SHORT', entry_fill, exit_fill, 5)
            trade_result = {'exit_reason': 'EOD', 'net_pnl': net, 'bars_held': j - entry_bar}
            break

    if trade_result is None and entry_bar < len(post_ib) - 1:
        last_bar = post_ib.iloc[-1]
        exit_fill = execution.fill_exit('SHORT', last_bar['close'])
        _, _, _, net = execution.calculate_net_pnl('SHORT', entry_fill, exit_fill, 5)
        trade_result = {'exit_reason': 'EOD', 'net_pnl': net, 'bars_held': len(post_ib) - entry_bar}

    if trade_result:
        t = {**sweep, **trade_result, 'winner': trade_result['net_pnl'] > 0}
        vwap_trades.append(t)

vwap_wins = sum(1 for t in vwap_trades if t['winner'])
print(f"\nVWAP target baseline: {vwap_wins}/{len(vwap_trades)} = "
      f"{vwap_wins/max(len(vwap_trades),1)*100:.1f}% WR, "
      f"${sum(t['net_pnl'] for t in vwap_trades):,.0f}")

# Apply top filters from IB mid analysis to VWAP target
print(f"\n--- Top single filters with VWAP target ---")
top_filters_to_test = [
    ('Sweep <= 10 pts', lambda t: t['sweep_pts'] <= 10),
    ('Sweep <= 15 pts', lambda t: t['sweep_pts'] <= 15),
    ('Wick >= 0.5', lambda t: t['wick_ratio'] >= 0.5),
    ('Close >= 10 below IBH', lambda t: t['close_vs_ibh'] >= 10),
    ('Delta > 0', lambda t: t['delta'] > 0),
    ('IB >= 50', lambda t: t['ib_range'] >= 50),
    ('No HTF resistance', lambda t: not t['at_htf_resistance']),
    ('Before 12:00', lambda t: t['minutes_since_open'] <= 150),
    ('Bar opened inside IB', lambda t: not t['bar_opened_above']),
    ('R:R >= 0.5', lambda t: t.get('rr_ratio', t.get('reward', 0) / max(t.get('risk', 1), 1)) >= 0.5),
]

for label, fn in top_filters_to_test:
    passed = [t for t in vwap_trades if fn(t)]
    if len(passed) < 3:
        continue
    p_wins = sum(1 for t in passed if t['winner'])
    p_wr = p_wins / len(passed) * 100
    p_pnl = sum(t['net_pnl'] for t in passed)
    print(f"  {label:<40s}: {p_wins}/{len(passed)} = {p_wr:>5.1f}% WR  ${p_pnl:>8,.0f}")


# ============================================================================
# PHASE 10: VERDICT — THE FILTER THAT WORKS (OR DOESN'T)
# ============================================================================
print("\n\n" + "=" * 140)
print("  PHASE 10: CONSOLIDATED SHORT FADE VERDICT")
print("=" * 140)

print("""
  PRIOR STUDY FINDINGS (consolidated):
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. test_ibh_fade.py (fade AT IBH, before breakout):
     - 12 configs tested, ALL negative
     - NQ long bias creates 3.3x loss ratio on stops
     - Even best config (67.3% WR) was -$2,282

  2. study_failed_breakout.py (94% of IBH breaks fail):
     - 14 configs, best = 50% WR, $962 (marginal)
     - 83% of failures had POSITIVE delta = exhaustion, not absorption

  3. study_ibh_fail_confluence.py (HTF confluence):
     - COUNTERINTUITIVE: More confluence = LESS reversion
     - 0 confluences: 75% VWAP hit, 206 pts drop
     - 2 confluences: 29% VWAP hit, 52 pts drop
     - Bull regime failures revert DEEPER than bear regime

  4. study_ibh_sweep_daytype.py (day-type filtered):
     - b_day sweeps: 28 trades, 42.9% WR
     - Other thread's 4 dates confirmed as winners
     - Our broader b_day classification captures 24 extra (dilutes WR)

  CORE PROBLEM WITH NQ SHORTS:
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - NQ has structural long bias (S&P 500 / QQQ upward drift)
  - When targets hit: avg win ~$100-700 (small, consistent)
  - When stops hit: avg loss ~$400-650 (catastrophic, wipes multiple wins)
  - Even at 60-70% WR, the loss asymmetry makes net PnL negative
  - Shorts ONLY work with very tight stops OR very high WR (>75%)
""")

# Final assessment
best_single = valid_results[0] if valid_results else None
best_combo = combo_results[0] if combo_results else None
best_triple = triple_results[0] if triple_results else None

if best_single:
    # Calculate wins from the passed trades
    single_wins = sum(1 for t in best_single['passed'] if t['winner'])
    print(f"  BEST SINGLE FILTER: {best_single['label']}")
    print(f"    {single_wins}/{best_single['n']} trades = {best_single['wr']:.1f}% WR, ${best_single['pnl']:,.0f}")

if best_combo:
    print(f"\n  BEST 2-FACTOR COMBO: {best_combo['label']}")
    print(f"    {best_combo['wins']}/{best_combo['n']} trades = {best_combo['wr']:.1f}% WR, ${best_combo['pnl']:,.0f}")

if best_triple:
    print(f"\n  BEST 3-FACTOR COMBO: {best_triple['label']}")
    print(f"    {best_triple['wins']}/{best_triple['n']} trades = {best_triple['wr']:.1f}% WR, ${best_triple['pnl']:,.0f}")

# Combined playbook impact with best filter
print(f"\n  COMBINED PLAYBOOK IMPACT:")
print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━")
current_wr = 75.9
current_trades = 29
current_wins = 22
current_pnl = 3861

if best_combo and best_combo['wr'] >= 65:
    cb = best_combo
    combo_trades = current_trades + cb['n']
    combo_wins = current_wins + cb['wins']
    combo_pnl = current_pnl + cb['pnl']
    combo_wr = combo_wins / combo_trades * 100
    print(f"    Current LONG playbook:  {current_trades} trades, {current_wr:.1f}% WR, ${current_pnl:,.0f}")
    print(f"    + Best SHORT filter:    {cb['n']} trades, {cb['wr']:.1f}% WR, ${cb['pnl']:,.0f}")
    print(f"    = COMBINED:             {combo_trades} trades, {combo_wr:.1f}% WR, ${combo_pnl:,.0f}")
    if combo_wr >= 70:
        print(f"    >>> PASSES 70% WR threshold — CAN ADD")
    else:
        print(f"    >>> Below 70% WR — NOT RECOMMENDED for Lightning")

print(f"""
  RECOMMENDATION:
  ━━━━━━━━━━━━━━━
  Review the filter rankings above. If any filter combination achieves:
    - >= 65% WR with 5+ trades AND positive PnL → CAUTIOUSLY ADD to playbook
    - >= 75% WR with 4+ trades AND positive PnL → ADD to playbook as selective play
    - < 60% WR or negative PnL → DO NOT ADD, stick with LONG-only playbook

  The 70% WR LONG-only playbook (75.9% WR, $3,861, PF 8.86) should NOT be
  diluted unless the short filter is genuinely additive.
""")
