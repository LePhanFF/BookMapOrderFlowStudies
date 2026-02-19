"""
IBH Failed Breakout — HTF Confluence + Regime Study

Hypothesis: IBH failures that coincide with higher timeframe supply zones
(4H/Daily bearish FVG, PDH, London High) are higher-probability fades.

The idea: NQ sweeps IBH liquidity, runs into HTF supply/resistance,
and the double confluence triggers a reversion.

Confluence levels checked:
  1. Prior Day High (PDH) — major liquidity pool
  2. Prior 3-Day High — wider liquidity sweep
  3. 15-min bearish FVG zone near IBH area
  4. 1-min bearish FVG zone at breakout level
  5. Session opens with overnight high nearby (globex high)
  6. EMA regime (bear vs bull)
  7. ATR / volatility regime
  8. 4H FVG approximation (computed from 4H candles)

Also studies:
  - Do failed breakouts in bear regime revert deeper?
  - Does PDH confluence improve WR?
  - Which confluence combinations are most reliable?
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

# Load RTH data
instrument = get_instrument('MNQ')
df_raw = load_csv('NQ')  # Full data including ETH
df = filter_rth(df_raw)
df = compute_all_features(df)

if 'session_date' not in df.columns:
    df['session_date'] = df['timestamp'].dt.date

sessions = sorted(df['session_date'].unique())
execution = ExecutionModel(instrument, slippage_ticks=1)

# ============================================================================
# COMPUTE PRIOR DAY LEVELS + OVERNIGHT (GLOBEX) LEVELS
# ============================================================================

# Prior day high/low from RTH data
prior_day_levels = {}
prev_session_high = None
prev_session_low = None
prev_session_close = None
prev_3day_high = None
prev_3day_low = None

session_highs = []
session_lows = []

for session_date in sessions:
    sdf = df[df['session_date'] == session_date]
    if len(sdf) == 0:
        continue

    # Store prior day levels for this session
    prior_day_levels[str(session_date)] = {
        'pdh': prev_session_high,
        'pdl': prev_session_low,
        'pdc': prev_session_close,
        'p3dh': prev_3day_high,
        'p3dl': prev_3day_low,
    }

    # Compute this session's levels for next day
    prev_session_high = sdf['high'].max()
    prev_session_low = sdf['low'].min()
    prev_session_close = sdf.iloc[-1]['close']

    session_highs.append(prev_session_high)
    session_lows.append(prev_session_low)
    if len(session_highs) >= 3:
        prev_3day_high = max(session_highs[-3:])
        prev_3day_low = min(session_lows[-3:])

# Compute overnight (globex) high from ETH data
# ETH = 18:00 previous day to 09:29 current day
df_raw_ts = df_raw.copy()
if 'timestamp' not in df_raw_ts.columns and isinstance(df_raw_ts.index, pd.DatetimeIndex):
    df_raw_ts['timestamp'] = df_raw_ts.index
df_raw_ts['timestamp'] = pd.to_datetime(df_raw_ts['timestamp'])
df_raw_ts['time'] = df_raw_ts['timestamp'].dt.time

overnight_levels = {}
for session_date in sessions:
    # Get bars between 18:00 prior day and 09:29 current day
    rth_start = _time(9, 30)
    prior_day = session_date - timedelta(days=1)
    # Overnight = bars from prior day 18:00 to current day 09:29
    eth_mask = (
        ((df_raw_ts['timestamp'].dt.date == prior_day) & (df_raw_ts['time'] >= _time(18, 0))) |
        ((df_raw_ts['timestamp'].dt.date == session_date) & (df_raw_ts['time'] < rth_start))
    )
    eth_df = df_raw_ts[eth_mask]
    if len(eth_df) > 0:
        overnight_levels[str(session_date)] = {
            'globex_high': eth_df['high'].max(),
            'globex_low': eth_df['low'].min(),
        }
    else:
        overnight_levels[str(session_date)] = {
            'globex_high': None,
            'globex_low': None,
        }

# ============================================================================
# COMPUTE 4H FVG APPROXIMATION
# ============================================================================
# Create 4H candles and identify bear FVGs (supply zones)

def compute_4h_fvgs(df_full):
    """Compute 4H bearish FVG zones from full data."""
    df_4h = df_full.copy()
    df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'])
    df_4h = df_4h.set_index('timestamp')

    # Resample to 4H candles
    ohlc_4h = df_4h.resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        'volume': 'sum'
    }).dropna()

    # Identify bearish FVGs: candle[i-2].low > candle[i].high (gap down)
    bear_fvgs = []
    for i in range(2, len(ohlc_4h)):
        c0 = ohlc_4h.iloc[i - 2]  # Two candles ago
        c2 = ohlc_4h.iloc[i]       # Current candle

        if c0['low'] > c2['high']:
            bear_fvgs.append({
                'top': c0['low'],
                'bottom': c2['high'],
                'time': ohlc_4h.index[i],
            })

    # Also check for bullish FVGs for completeness
    bull_fvgs = []
    for i in range(2, len(ohlc_4h)):
        c0 = ohlc_4h.iloc[i - 2]
        c2 = ohlc_4h.iloc[i]

        if c0['high'] < c2['low']:
            bull_fvgs.append({
                'top': c2['low'],
                'bottom': c0['high'],
                'time': ohlc_4h.index[i],
            })

    return bear_fvgs, bull_fvgs

bear_fvgs_4h, bull_fvgs_4h = compute_4h_fvgs(df_raw)
print(f"\n4H FVGs detected: {len(bear_fvgs_4h)} bearish, {len(bull_fvgs_4h)} bullish")


# ============================================================================
# BUILD ENRICHED FAILED BREAKOUT DATA
# ============================================================================
print("\n" + "=" * 130)
print("  IBH FAILED BREAKOUT — HTF CONFLUENCE ANALYSIS")
print(f"  Data: {len(sessions)} sessions, PDH/Globex/4H-FVG confluence checked")
print("=" * 130)

enriched_fails = []

for session_date in sessions:
    date_str = str(session_date)
    sdf = df[df['session_date'] == session_date].copy()
    if len(sdf) < IB_BARS_1MIN + 20:
        continue

    ib_df = sdf.head(IB_BARS_1MIN)
    post_ib = sdf.iloc[IB_BARS_1MIN:]

    ib_high = ib_df['high'].max()
    ib_low = ib_df['low'].min()
    ib_range = ib_high - ib_low
    ib_mid = (ib_high + ib_low) / 2

    if ib_range <= 0:
        continue

    # Get prior day / overnight levels
    pdl = prior_day_levels.get(date_str, {})
    pdh = pdl.get('pdh')
    p3dh = pdl.get('p3dh')
    ovn = overnight_levels.get(date_str, {})
    globex_high = ovn.get('globex_high')

    # Get EMA regime from end-of-IB bar
    ib_end = ib_df.iloc[-1]
    ema20 = ib_end.get('ema20')
    ema50 = ib_end.get('ema50')
    ema_regime = None
    if ema20 is not None and ema50 is not None:
        if not pd.isna(ema20) and not pd.isna(ema50):
            ema_regime = 'BULL' if ema20 > ema50 else 'BEAR'
    # Fallback to mid-session
    if ema_regime is None and len(post_ib) > 30:
        mid = post_ib.iloc[30]
        e20 = mid.get('ema20')
        e50 = mid.get('ema50')
        if e20 is not None and e50 is not None and not pd.isna(e20) and not pd.isna(e50):
            ema_regime = 'BULL' if e20 > e50 else 'BEAR'

    # Track breakout state
    broke_above = False
    failed = False
    max_ext_pts = 0
    bars_above = 0
    breakout_delta_sum = 0
    fail_bar = None
    fail_price = None
    session_high = ib_high
    breakout_high = ib_high

    # Confluence flags at breakout level
    near_pdh = False
    near_p3dh = False
    near_globex_high = False
    in_bear_fvg_15m = False
    in_bear_fvg_1m = False
    in_bear_fvg_4h = False

    for i, (idx, bar) in enumerate(post_ib.iterrows()):
        price = bar['close']

        if bar['high'] > session_high:
            session_high = bar['high']

        if price > ib_high and not broke_above:
            broke_above = True
            breakout_high = bar['high']

        if broke_above and not failed:
            if price > ib_high:
                bars_above += 1
                ext = bar['high'] - ib_high
                if ext > max_ext_pts:
                    max_ext_pts = ext
                if bar['high'] > breakout_high:
                    breakout_high = bar['high']
                delta = bar.get('delta', 0)
                if not pd.isna(delta):
                    breakout_delta_sum += delta
            else:
                if bars_above >= 1:
                    failed = True
                    fail_bar = i
                    fail_price = price

                    # CHECK CONFLUENCE at breakout high level
                    zone_buffer = ib_range * 0.20  # 20% of IB range as buffer

                    # PDH confluence
                    if pdh is not None:
                        if abs(breakout_high - pdh) <= zone_buffer:
                            near_pdh = True

                    # Prior 3-day high
                    if p3dh is not None:
                        if abs(breakout_high - p3dh) <= zone_buffer:
                            near_p3dh = True

                    # Globex (overnight) high
                    if globex_high is not None:
                        if abs(breakout_high - globex_high) <= zone_buffer:
                            near_globex_high = True

                    # 15-min bearish FVG at breakout level
                    fvg_bear_top = bar.get('fvg_bear_15m_top')
                    if fvg_bear_top is not None and not pd.isna(fvg_bear_top):
                        if breakout_high >= fvg_bear_top - zone_buffer:
                            in_bear_fvg_15m = True

                    # 1-min bearish FVG
                    fvg_bear_1m_top = bar.get('fvg_bear_top')
                    if fvg_bear_1m_top is not None and not pd.isna(fvg_bear_1m_top):
                        if breakout_high >= fvg_bear_1m_top - zone_buffer:
                            in_bear_fvg_1m = True

                    # 4H bearish FVG
                    for fvg in bear_fvgs_4h:
                        if fvg['bottom'] <= breakout_high <= fvg['top'] + zone_buffer:
                            if fvg['time'].date() <= pd.Timestamp(session_date).date():
                                in_bear_fvg_4h = True
                                break

                    break

    if not failed:
        continue

    # Post-failure outcome
    post_fail_bars = post_ib.iloc[fail_bar:] if fail_bar is not None else pd.DataFrame()
    vwap_hit = False
    ib_mid_hit = False
    ibl_hit = False
    post_fail_min = fail_price

    for _, pbar in post_fail_bars.iterrows():
        if pbar['low'] < post_fail_min:
            post_fail_min = pbar['low']
        vwap_now = pbar.get('vwap', ib_mid)
        if not pd.isna(vwap_now) and pbar['low'] <= vwap_now:
            vwap_hit = True
        if pbar['low'] <= ib_mid:
            ib_mid_hit = True
        if pbar['low'] <= ib_low:
            ibl_hit = True

    session_close = sdf.iloc[-1]['close']

    # Count confluences
    confluence_count = sum([
        near_pdh, near_p3dh, near_globex_high,
        in_bear_fvg_15m, in_bear_fvg_1m, in_bear_fvg_4h,
    ])

    enriched_fails.append({
        'date': date_str,
        'ib_range': ib_range,
        'ib_high': ib_high,
        'ib_mid': ib_mid,
        'ib_low': ib_low,
        'max_ext_pts': max_ext_pts,
        'max_ext_mult': max_ext_pts / ib_range if ib_range > 0 else 0,
        'bars_above': bars_above,
        'breakout_delta': breakout_delta_sum,
        'breakout_high': breakout_high,
        'fail_price': fail_price,
        'session_close': session_close,
        'close_vs_ibh': session_close - ib_high,
        'close_vs_mid': session_close - ib_mid,
        # Confluence flags
        'near_pdh': near_pdh,
        'near_p3dh': near_p3dh,
        'near_globex_high': near_globex_high,
        'in_bear_fvg_15m': in_bear_fvg_15m,
        'in_bear_fvg_1m': in_bear_fvg_1m,
        'in_bear_fvg_4h': in_bear_fvg_4h,
        'confluence_count': confluence_count,
        # Regime
        'ema_regime': ema_regime,
        # Outcome
        'vwap_hit': vwap_hit,
        'ib_mid_hit': ib_mid_hit,
        'ibl_hit': ibl_hit,
        'post_fail_drop': fail_price - post_fail_min if post_fail_min else 0,
        'closed_below_ibh': session_close < ib_high,
        'closed_below_mid': session_close < ib_mid,
    })

print(f"\nTotal failed breakouts analyzed: {len(enriched_fails)}")

# ============================================================================
# CONFLUENCE ANALYSIS
# ============================================================================
print(f"\n--- CONFLUENCE PREVALENCE ---")
for flag in ['near_pdh', 'near_p3dh', 'near_globex_high',
             'in_bear_fvg_15m', 'in_bear_fvg_1m', 'in_bear_fvg_4h']:
    n = sum(1 for r in enriched_fails if r[flag])
    print(f"  {flag:<22s}: {n}/{len(enriched_fails)} ({n/len(enriched_fails)*100:.0f}%)")

print(f"\n--- CONFLUENCE COUNT DISTRIBUTION ---")
for c in range(max(r['confluence_count'] for r in enriched_fails) + 1):
    n = sum(1 for r in enriched_fails if r['confluence_count'] == c)
    if n > 0:
        sessions_at_c = [r for r in enriched_fails if r['confluence_count'] == c]
        vwap_pct = sum(1 for r in sessions_at_c if r['vwap_hit']) / n * 100
        mid_pct = sum(1 for r in sessions_at_c if r['ib_mid_hit']) / n * 100
        below_ibh_pct = sum(1 for r in sessions_at_c if r['closed_below_ibh']) / n * 100
        avg_drop = np.mean([r['post_fail_drop'] for r in sessions_at_c])
        print(f"  {c} confluences: {n} sessions → "
              f"VWAP hit {vwap_pct:.0f}% | IB mid hit {mid_pct:.0f}% | "
              f"Close<IBH {below_ibh_pct:.0f}% | Avg drop {avg_drop:.0f} pts")


# ============================================================================
# OUTCOME BY CONFLUENCE TYPE
# ============================================================================
print(f"\n\n--- OUTCOME BY SPECIFIC CONFLUENCE ---")

def analyze_confluence(label, subset, total):
    """Analyze reversion depth for a subset of failed breakouts."""
    if not subset:
        print(f"  {label:<35s}: 0 sessions")
        return
    n = len(subset)
    vwap_pct = sum(1 for r in subset if r['vwap_hit']) / n * 100
    mid_pct = sum(1 for r in subset if r['ib_mid_hit']) / n * 100
    ibl_pct = sum(1 for r in subset if r['ibl_hit']) / n * 100
    below_ibh = sum(1 for r in subset if r['closed_below_ibh']) / n * 100
    below_mid = sum(1 for r in subset if r['closed_below_mid']) / n * 100
    avg_drop = np.mean([r['post_fail_drop'] for r in subset])
    no_conf = [r for r in total if r not in subset]
    no_mid = sum(1 for r in no_conf if r['ib_mid_hit']) / len(no_conf) * 100 if no_conf else 0

    print(f"  {label:<35s}: {n} sess → VWAP {vwap_pct:.0f}% | Mid {mid_pct:.0f}% | "
          f"IBL {ibl_pct:.0f}% | Cl<IBH {below_ibh:.0f}% | Cl<Mid {below_mid:.0f}% | "
          f"AvgDrop {avg_drop:.0f}pts  (vs {no_mid:.0f}% mid w/o)")

analyze_confluence("ALL failed breakouts",
                   enriched_fails, enriched_fails)
analyze_confluence("Near PDH",
                   [r for r in enriched_fails if r['near_pdh']], enriched_fails)
analyze_confluence("Near Prior 3-Day High",
                   [r for r in enriched_fails if r['near_p3dh']], enriched_fails)
analyze_confluence("Near Globex (overnight) High",
                   [r for r in enriched_fails if r['near_globex_high']], enriched_fails)
analyze_confluence("In 15-min Bear FVG",
                   [r for r in enriched_fails if r['in_bear_fvg_15m']], enriched_fails)
analyze_confluence("In 1-min Bear FVG",
                   [r for r in enriched_fails if r['in_bear_fvg_1m']], enriched_fails)
analyze_confluence("In 4H Bear FVG",
                   [r for r in enriched_fails if r['in_bear_fvg_4h']], enriched_fails)
analyze_confluence("ANY confluence (count >= 1)",
                   [r for r in enriched_fails if r['confluence_count'] >= 1], enriched_fails)
analyze_confluence("STRONG confluence (count >= 2)",
                   [r for r in enriched_fails if r['confluence_count'] >= 2], enriched_fails)

# ============================================================================
# REGIME BREAKDOWN
# ============================================================================
print(f"\n\n--- OUTCOME BY EMA REGIME ---")
bear_fails = [r for r in enriched_fails if r['ema_regime'] == 'BEAR']
bull_fails = [r for r in enriched_fails if r['ema_regime'] == 'BULL']
unk_fails = [r for r in enriched_fails if r['ema_regime'] is None]

analyze_confluence("BEAR regime (EMA20 < EMA50)", bear_fails, enriched_fails)
analyze_confluence("BULL regime (EMA20 > EMA50)", bull_fails, enriched_fails)
if unk_fails:
    analyze_confluence("Unknown regime", unk_fails, enriched_fails)

# Regime + confluence combo
print(f"\n--- REGIME + CONFLUENCE COMBOS ---")
analyze_confluence("BEAR + any confluence",
                   [r for r in bear_fails if r['confluence_count'] >= 1], enriched_fails)
analyze_confluence("BEAR + near PDH",
                   [r for r in bear_fails if r['near_pdh']], enriched_fails)
analyze_confluence("BEAR + 15m bear FVG",
                   [r for r in bear_fails if r['in_bear_fvg_15m']], enriched_fails)
analyze_confluence("BEAR + neg delta breakout",
                   [r for r in bear_fails if r['breakout_delta'] < 0], enriched_fails)
analyze_confluence("BULL + any confluence",
                   [r for r in bull_fails if r['confluence_count'] >= 1], enriched_fails)
analyze_confluence("BULL + near PDH",
                   [r for r in bull_fails if r['near_pdh']], enriched_fails)


# ============================================================================
# SIMULATED TRADES WITH CONFLUENCE FILTER
# ============================================================================
print(f"\n\n{'=' * 130}")
print("  SIMULATED FAILED BREAKOUT SHORTS — WITH CONFLUENCE FILTERS")
print(f"{'=' * 130}")

def simulate_confluence_short(config_name, target_mode='vwap',
                               require_confluence=0, require_pdh=False,
                               require_bear_fvg=False, require_bear_regime=False,
                               require_neg_delta=False, min_bars=1, max_ext=0.5):
    """Simulate failed breakout shorts with confluence requirements."""
    trades = []

    for rec in enriched_fails:
        date_str = rec['date']

        # Apply filters
        if rec['confluence_count'] < require_confluence:
            continue
        if require_pdh and not rec['near_pdh']:
            continue
        if require_bear_fvg and not (rec['in_bear_fvg_15m'] or rec['in_bear_fvg_4h']):
            continue
        if require_bear_regime and rec['ema_regime'] != 'BEAR':
            continue
        if require_neg_delta and rec['breakout_delta'] >= 0:
            continue
        if rec['bars_above'] < min_bars:
            continue
        if rec['max_ext_mult'] > max_ext:
            continue

        # Simulate the trade
        ib_high = rec['ib_high']
        ib_mid = rec['ib_mid']
        ib_low = rec['ib_low']
        ib_range = rec['ib_range']

        sdf = df[df['session_date'].astype(str) == date_str[:10]].copy()
        if len(sdf) < IB_BARS_1MIN + 20:
            continue
        post_ib = sdf.iloc[IB_BARS_1MIN:]

        # Find the failure bar and enter
        entry_taken = False
        broke = False
        bars_up = 0
        session_high = ib_high

        for i, (idx, bar) in enumerate(post_ib.iterrows()):
            price = bar['close']
            bar_time = bar['timestamp'].time() if 'timestamp' in bar.index else None

            if bar['high'] > session_high:
                session_high = bar['high']

            if bar_time and bar_time >= _time(13, 30) and not entry_taken:
                break

            if entry_taken:
                # Manage position
                if bar['high'] >= stop_price:
                    exit_fill = execution.fill_exit('SHORT', stop_price)
                    _, _, _, net = execution.calculate_net_pnl('SHORT', entry_price, exit_fill, 5)
                    trades.append({'date': date_str, 'net_pnl': net, 'exit_reason': 'STOP',
                                   'bars_held': i - entry_bar_idx, 'confluence': rec['confluence_count']})
                    break
                if bar['low'] <= target_price:
                    exit_fill = execution.fill_exit('SHORT', target_price)
                    _, _, _, net = execution.calculate_net_pnl('SHORT', entry_price, exit_fill, 5)
                    trades.append({'date': date_str, 'net_pnl': net, 'exit_reason': 'TARGET',
                                   'bars_held': i - entry_bar_idx, 'confluence': rec['confluence_count']})
                    break
                if bar_time and bar_time >= _time(15, 0):
                    exit_fill = execution.fill_exit('SHORT', price)
                    _, _, _, net = execution.calculate_net_pnl('SHORT', entry_price, exit_fill, 5)
                    trades.append({'date': date_str, 'net_pnl': net, 'exit_reason': 'EOD',
                                   'bars_held': i - entry_bar_idx, 'confluence': rec['confluence_count']})
                    break
                continue

            # State machine
            if price > ib_high:
                broke = True
                bars_up += 1
            elif broke and bars_up >= min_bars:
                # Failed breakout confirmed — enter short
                entry_fill = execution.fill_entry('SHORT', price)
                entry_price = entry_fill
                stop_price = session_high + (ib_range * 0.10)
                stop_price = max(stop_price, entry_price + 15.0)

                vwap_now = bar.get('vwap', ib_mid)
                if pd.isna(vwap_now):
                    vwap_now = ib_mid

                if target_mode == 'vwap':
                    target_price = vwap_now
                elif target_mode == 'ib_mid':
                    target_price = ib_mid
                elif target_mode == 'ibl':
                    target_price = ib_low
                else:
                    target_price = ib_mid

                risk = stop_price - entry_price
                reward = entry_price - target_price
                if reward > 0 and risk > 0:
                    entry_taken = True
                    entry_bar_idx = i
                else:
                    break

        # EOD force close
        if entry_taken and not any(t['date'] == date_str for t in trades):
            last_bar = post_ib.iloc[-1]
            exit_fill = execution.fill_exit('SHORT', last_bar['close'])
            _, _, _, net = execution.calculate_net_pnl('SHORT', entry_price, exit_fill, 5)
            trades.append({'date': date_str, 'net_pnl': net, 'exit_reason': 'EOD',
                           'bars_held': len(post_ib) - entry_bar_idx, 'confluence': rec['confluence_count']})

    return trades

# Test configurations
trade_configs = [
    # (name, target, confluence, pdh, bear_fvg, bear_regime, neg_delta, min_bars, max_ext)
    ('A: No filter → VWAP',
     'vwap', 0, False, False, False, False, 1, 0.5),
    ('B: Any confluence (≥1) → VWAP',
     'vwap', 1, False, False, False, False, 1, 0.5),
    ('C: Strong confluence (≥2) → VWAP',
     'vwap', 2, False, False, False, False, 1, 0.5),
    ('D: Near PDH → VWAP',
     'vwap', 0, True, False, False, False, 1, 0.5),
    ('E: Bear FVG (15m/4H) → VWAP',
     'vwap', 0, False, True, False, False, 1, 0.5),
    ('F: Bear regime → VWAP',
     'vwap', 0, False, False, True, False, 1, 0.5),
    ('G: Bear regime + confluence → VWAP',
     'vwap', 1, False, False, True, False, 1, 0.5),
    ('H: Bear regime + PDH → VWAP',
     'vwap', 0, True, False, True, False, 1, 0.5),
    ('I: PDH + neg delta → VWAP',
     'vwap', 0, True, False, False, True, 1, 0.5),
    ('J: Any confluence → IB mid',
     'ib_mid', 1, False, False, False, False, 1, 0.5),
    ('K: PDH + bear FVG → VWAP',
     'vwap', 0, True, True, False, False, 1, 0.5),
    ('L: Bear regime + bear FVG → VWAP',
     'vwap', 0, False, True, True, False, 1, 0.5),
    ('M: Globex high proximity → VWAP',
     'vwap', 0, False, False, False, False, 1, 0.5),  # Will check via globex manually
    ('N: Bear regime + confluence + neg delta → VWAP',
     'vwap', 1, False, False, True, True, 1, 0.5),
]

print(f"\n{'Config':<50s} {'Trades':>6s} {'W':>3s} {'L':>3s} {'WR%':>6s} {'Net PnL':>10s} {'Expect':>8s} {'AvgWin':>8s} {'AvgLoss':>8s}")
print("-" * 110)

for name, target, conf, pdh_r, fvg_r, bear_r, neg_d, min_b, max_e in trade_configs:
    t_list = simulate_confluence_short(
        name, target_mode=target, require_confluence=conf,
        require_pdh=pdh_r, require_bear_fvg=fvg_r,
        require_bear_regime=bear_r, require_neg_delta=neg_d,
        min_bars=min_b, max_ext=max_e,
    )
    wins = [t for t in t_list if t['net_pnl'] > 0]
    losses = [t for t in t_list if t['net_pnl'] <= 0]
    pnl = sum(t['net_pnl'] for t in t_list)
    wr = len(wins) / len(t_list) * 100 if t_list else 0
    avg_w = np.mean([t['net_pnl'] for t in wins]) if wins else 0
    avg_l = np.mean([t['net_pnl'] for t in losses]) if losses else 0
    exp = pnl / len(t_list) if t_list else 0

    print(f"{name:<50s} {len(t_list):>6d} {len(wins):>3d} {len(losses):>3d} "
          f"{wr:>5.1f}% ${pnl:>8,.0f} ${exp:>6,.0f} ${avg_w:>6,.0f} ${avg_l:>6,.0f}")

print("-" * 110)


# ============================================================================
# DETAILED SESSION LOG FOR CONFLUENCE PATTERNS
# ============================================================================
print(f"\n\n{'=' * 130}")
print("  DETAILED FAILED BREAKOUT LOG — WITH CONFLUENCE FLAGS")
print(f"{'=' * 130}")

print(f"\n{'Date':<14s} {'ExtMult':>7s} {'Bars':>5s} {'BrkDelta':>9s} {'PDH':>4s} {'P3DH':>5s} "
      f"{'Globex':>6s} {'FVG15':>5s} {'FVG1m':>5s} {'FVG4H':>5s} "
      f"{'Conf#':>5s} {'Regime':>6s} {'→VWAP':>5s} {'→Mid':>5s} {'ClBelIBH':>8s} {'Drop':>6s}")
print("-" * 130)

for r in sorted(enriched_fails, key=lambda x: x['date']):
    print(f"{r['date']:<14s} {r['max_ext_mult']:>6.2f}x {r['bars_above']:>5d} "
          f"{r['breakout_delta']:>+8.0f} "
          f"{'Y' if r['near_pdh'] else '.':>4s} "
          f"{'Y' if r['near_p3dh'] else '.':>5s} "
          f"{'Y' if r['near_globex_high'] else '.':>6s} "
          f"{'Y' if r['in_bear_fvg_15m'] else '.':>5s} "
          f"{'Y' if r['in_bear_fvg_1m'] else '.':>5s} "
          f"{'Y' if r['in_bear_fvg_4h'] else '.':>5s} "
          f"{r['confluence_count']:>5d} "
          f"{(r['ema_regime'] or '?'):>6s} "
          f"{'Y' if r['vwap_hit'] else 'N':>5s} "
          f"{'Y' if r['ib_mid_hit'] else 'N':>5s} "
          f"{'Y' if r['closed_below_ibh'] else 'N':>8s} "
          f"{r['post_fail_drop']:>5.0f}")


# ============================================================================
# VERDICT
# ============================================================================
print(f"\n\n{'=' * 130}")
print("  VERDICT: HTF CONFLUENCE + FAILED BREAKOUT")
print(f"{'=' * 130}")

# Find configs with positive PnL
positive_configs = [(name, t_list) for (name, *_), t_list in zip(trade_configs, [
    simulate_confluence_short(n, t, c, p, f, b, nd, mb, me)
    for n, t, c, p, f, b, nd, mb, me in trade_configs
]) if sum(t['net_pnl'] for t in t_list) > 0 and len(t_list) >= 3]

if positive_configs:
    print(f"\n  PROFITABLE CONFIGS FOUND:")
    for name, trades in positive_configs:
        w = sum(1 for t in trades if t['net_pnl'] > 0)
        pnl = sum(t['net_pnl'] for t in trades)
        wr = w / len(trades) * 100
        print(f"    {name}: {len(trades)} trades, {wr:.0f}% WR, ${pnl:,.0f}")
else:
    print(f"\n  No confluence configuration produced consistent profits.")

# Key question: does confluence help the reversion depth?
conf_yes = [r for r in enriched_fails if r['confluence_count'] >= 1]
conf_no = [r for r in enriched_fails if r['confluence_count'] == 0]

if conf_yes and conf_no:
    print(f"\n  CONFLUENCE IMPACT ON REVERSION DEPTH:")
    print(f"    WITH confluence (≥1): {len(conf_yes)} sessions")
    print(f"      VWAP hit: {sum(1 for r in conf_yes if r['vwap_hit'])/len(conf_yes)*100:.0f}%")
    print(f"      IB mid hit: {sum(1 for r in conf_yes if r['ib_mid_hit'])/len(conf_yes)*100:.0f}%")
    print(f"      Avg drop: {np.mean([r['post_fail_drop'] for r in conf_yes]):.0f} pts")
    print(f"    WITHOUT confluence:    {len(conf_no)} sessions")
    print(f"      VWAP hit: {sum(1 for r in conf_no if r['vwap_hit'])/len(conf_no)*100:.0f}%")
    print(f"      IB mid hit: {sum(1 for r in conf_no if r['ib_mid_hit'])/len(conf_no)*100:.0f}%")
    print(f"      Avg drop: {np.mean([r['post_fail_drop'] for r in conf_no]):.0f} pts")
