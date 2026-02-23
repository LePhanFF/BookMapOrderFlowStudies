"""
20P Rule Study — The Failure of 80P (Trend Extension)
======================================================

The 80P Rule trades the return-to-balance: price opens outside VA, re-enters, rotates back.
The 20P is the OPPOSITE — the 20% of the time the breakout is REAL:
  - Price opens outside prior session VA
  - Instead of rotating back, it EXTENDS further away
  - IB extension confirms the trend (accept above IBH / below IBL)
  - We catch the extension move

Direction logic (opposite of 80P):
  - Open ABOVE VAH → price continues UP → LONG (with IBH extension)
  - Open BELOW VAL → price continues DOWN → SHORT (with IBL extension)

Study covers:
  PART 1: How often does 20P occur? (frequency & day characteristics)
  PART 2: IB extension as entry trigger (accept above IBH / below IBL)
  PART 3: VA boundary rejection entry (price touches VA edge, bounces away)
  PART 4: Pullback entry after IB extension (re-entry to IB boundary)
  PART 5: Time-based confirmation (N consecutive bars outside VA)
  PART 6: Order flow filters (delta, CVD at entry)
  PART 7: Target analysis (extension projections)
  PART 8: Head-to-head ranking of all models
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import time
from typing import Dict, List, Optional

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.instruments import get_instrument
from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from indicators.value_area import compute_session_value_areas

INSTRUMENT = get_instrument('MNQ')
TICK_VALUE = INSTRUMENT.tick_value / INSTRUMENT.tick_size  # $2/pt for MNQ
SLIPPAGE_PTS = 0.50
COMMISSION = 1.24
CONTRACTS = 5
MIN_VA_WIDTH = 25.0
ENTRY_CUTOFF = time(13, 0)
EOD_EXIT = time(15, 30)

# ============================================================================
# DATA
# ============================================================================
print("Loading data...")
df_raw = load_csv('NQ')
df_rth = filter_rth(df_raw)
df_rth = compute_all_features(df_rth)

if 'session_date' not in df_rth.columns:
    df_rth['session_date'] = df_rth['timestamp'].dt.date

df_full = df_raw.copy()
if 'session_date' not in df_full.columns:
    df_full['session_date'] = df_full['timestamp'].dt.date
    evening_mask = df_full['timestamp'].dt.time >= time(18, 0)
    df_full.loc[evening_mask, 'session_date'] = (
        pd.to_datetime(df_full.loc[evening_mask, 'session_date']) + pd.Timedelta(days=1)
    ).dt.date

sessions = sorted(df_rth['session_date'].unique())
n_sessions = len(sessions)
months = n_sessions / 22
print(f"Sessions: {n_sessions}, Months: {months:.1f}")

print("Computing ETH Value Areas...")
eth_va = compute_session_value_areas(df_full, tick_size=0.25, va_percent=0.70)


# ============================================================================
# HELPERS
# ============================================================================
def aggregate_bars(bars_df, period_min):
    """Aggregate 1-min bars into N-min OHLCV bars."""
    agg_bars = []
    for start in range(0, len(bars_df), period_min):
        end = min(start + period_min, len(bars_df))
        chunk = bars_df.iloc[start:end]
        if len(chunk) == 0:
            continue
        agg_bars.append({
            'bar_start': start,
            'bar_end': end - 1,
            'open': chunk.iloc[0]['open'],
            'high': chunk['high'].max(),
            'low': chunk['low'].min(),
            'close': chunk.iloc[-1]['close'],
            'volume': chunk['volume'].sum(),
            'vol_delta': chunk['vol_delta'].sum() if 'vol_delta' in chunk.columns else 0,
            'timestamp': chunk.iloc[-1]['timestamp'],
        })
    return agg_bars


def get_ib_levels(session_df):
    """Get IB high, low, range from first 60 bars (9:30-10:30)."""
    ib_bars = session_df.iloc[:60]
    if len(ib_bars) < 30:
        return None
    ibh = ib_bars['high'].max()
    ibl = ib_bars['low'].min()
    return {'ibh': ibh, 'ibl': ibl, 'ib_range': ibh - ibl, 'ib_mid': (ibh + ibl) / 2}


def replay_trade(post_ib, entry_bar_idx, entry_price, direction,
                 stop_price, target_price):
    """Replay a trade from a 1-min bar index with explicit stop/target."""
    risk_pts = abs(entry_price - stop_price)
    reward_pts = abs(target_price - entry_price)
    if risk_pts <= 0 or reward_pts <= 0:
        return None

    remaining = post_ib.iloc[entry_bar_idx:]
    mfe = 0.0
    mae = 0.0
    exit_price = None
    exit_reason = None
    bars_held = 0

    for i in range(len(remaining)):
        bar = remaining.iloc[i]
        bars_held += 1
        bt_time = bar.get('timestamp')
        bt = bt_time.time() if bt_time and hasattr(bt_time, 'time') else None
        if bt and bt >= EOD_EXIT:
            exit_price = bar['close']
            exit_reason = 'EOD'
            break

        fav = (bar['high'] - entry_price) if direction == 'LONG' else (entry_price - bar['low'])
        adv = (entry_price - bar['low']) if direction == 'LONG' else (bar['high'] - entry_price)
        mfe = max(mfe, fav)
        mae = max(mae, adv)

        # Check stop
        if direction == 'LONG' and bar['low'] <= stop_price:
            exit_price = stop_price
            exit_reason = 'STOP'
            break
        elif direction == 'SHORT' and bar['high'] >= stop_price:
            exit_price = stop_price
            exit_reason = 'STOP'
            break

        # Check target
        if direction == 'LONG' and bar['high'] >= target_price:
            exit_price = target_price
            exit_reason = 'TARGET'
            break
        elif direction == 'SHORT' and bar['low'] <= target_price:
            exit_price = target_price
            exit_reason = 'TARGET'
            break

    if exit_price is None:
        exit_price = remaining.iloc[-1]['close']
        exit_reason = 'EOD'

    pnl_pts = (exit_price - entry_price - SLIPPAGE_PTS) if direction == 'LONG' \
              else (entry_price - exit_price - SLIPPAGE_PTS)
    pnl_dollars = pnl_pts * TICK_VALUE * CONTRACTS - COMMISSION * CONTRACTS

    return {
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'pnl_pts': pnl_pts,
        'pnl_dollars': pnl_dollars,
        'risk_pts': risk_pts,
        'reward_pts': reward_pts,
        'rr_ratio': reward_pts / risk_pts if risk_pts > 0 else 0,
        'mfe_pts': mfe,
        'mae_pts': mae,
        'bars_held': bars_held,
        'is_winner': pnl_dollars > 0,
    }


def compute_target(entry_price, direction, target_mode, ib, va_width, risk_pts):
    """Compute target price based on mode."""
    ibh, ibl, ib_range = ib['ibh'], ib['ibl'], ib['ib_range']

    if target_mode == '1x_ib':
        return (ibh + ib_range) if direction == 'LONG' else (ibl - ib_range)
    elif target_mode == '1.5x_ib':
        return (ibh + 1.5 * ib_range) if direction == 'LONG' else (ibl - 1.5 * ib_range)
    elif target_mode == '2x_ib':
        return (ibh + 2.0 * ib_range) if direction == 'LONG' else (ibl - 2.0 * ib_range)
    elif target_mode == 'va_width':
        return (entry_price + va_width) if direction == 'LONG' else (entry_price - va_width)
    elif target_mode.endswith('R'):
        r_mult = float(target_mode[:-1])
        return (entry_price + risk_pts * r_mult) if direction == 'LONG' \
               else (entry_price - risk_pts * r_mult)
    else:
        return (ibh + ib_range) if direction == 'LONG' else (ibl - ib_range)


def compute_stop(entry_price, direction, stop_mode, ib, stop_buffer=10):
    """Compute stop price based on mode."""
    ibh, ibl = ib['ibh'], ib['ibl']

    if stop_mode == 'ib_boundary':
        # Stop at IB boundary that was extended through
        return (ibl - stop_buffer) if direction == 'LONG' else (ibh + stop_buffer)
    elif stop_mode == 'ib_mid':
        ib_mid = (ibh + ibl) / 2
        return (ib_mid - stop_buffer) if direction == 'LONG' else (ib_mid + stop_buffer)
    elif stop_mode == 'entry_atr':
        # Will be overridden externally with ATR value
        return None
    else:
        return (ibl - stop_buffer) if direction == 'LONG' else (ibh + stop_buffer)


def print_results_table(results_list, label, months_val):
    """Print a standard results table from a list of trade dicts."""
    if not results_list:
        print(f"    {label:<45s}    0  (no trades)")
        return

    df_r = pd.DataFrame(results_list)
    n = len(df_r)
    wr = df_r['is_winner'].mean() * 100
    gw = df_r[df_r['is_winner']]['pnl_dollars'].sum()
    gl = abs(df_r[~df_r['is_winner']]['pnl_dollars'].sum())
    pf = gw / gl if gl > 0 else float('inf')
    pm = df_r['pnl_dollars'].sum() / months_val
    avg_w = df_r[df_r['is_winner']]['pnl_dollars'].mean() if df_r['is_winner'].any() else 0
    avg_l = df_r[~df_r['is_winner']]['pnl_dollars'].mean() if (~df_r['is_winner']).any() else 0
    avg_risk = df_r['risk_pts'].mean()
    stopped = (df_r['exit_reason'] == 'STOP').sum()
    target_hit = (df_r['exit_reason'] == 'TARGET').sum()
    eod = (df_r['exit_reason'] == 'EOD').sum()
    avg_mfe = df_r['mfe_pts'].mean()
    avg_mae = df_r['mae_pts'].mean()

    print(f"    {label:<45s} {n:>4d} {n/months_val:>6.1f} "
          f"{wr:>5.1f}% {pf:>5.2f} ${pm:>8,.0f} ${avg_w:>7,.0f} ${avg_l:>7,.0f} "
          f"{avg_risk:>5.0f}p {avg_mfe:>5.0f}/{avg_mae:>4.0f} {stopped:>3d}/{target_hit:>3d}/{eod:>3d}")


def print_header():
    print(f"    {'Config':<45s} {'N':>4s} {'Trd/Mo':>6s} "
          f"{'WR':>6s} {'PF':>6s} {'$/Mo':>9s} {'AvgWin':>8s} {'AvgLoss':>8s} "
          f"{'Risk':>6s} {'MFE/MAE':>9s} {'S/T/E':>10s}")
    print(f"    {'-'*130}")


# ============================================================================
# BUILD SESSION DATABASE
# ============================================================================
print("\nBuilding session database...")

session_db = []
for i in range(1, len(sessions)):
    current = sessions[i]
    prior = sessions[i - 1]
    prior_key = str(prior)

    if prior_key not in eth_va:
        continue
    prior_va = eth_va[prior_key]
    if prior_va.va_width < MIN_VA_WIDTH:
        continue

    session_df = df_rth[df_rth['session_date'] == current].reset_index(drop=True)
    if len(session_df) < 90:
        continue

    open_price = session_df['open'].iloc[0]
    vah = prior_va.vah
    val = prior_va.val
    poc = prior_va.poc
    va_width = vah - val

    # Must open outside VA
    if val <= open_price <= vah:
        continue

    # 20P direction: CONTINUE the breakout (opposite of 80P)
    # Open above VAH → expect continuation UP → LONG
    # Open below VAL → expect continuation DOWN → SHORT
    if open_price > vah:
        direction = 'LONG'
        gap_distance = open_price - vah
    else:
        direction = 'SHORT'
        gap_distance = val - open_price

    ib = get_ib_levels(session_df)
    if ib is None or ib['ib_range'] < 10:
        continue

    post_ib = session_df.iloc[60:].reset_index(drop=True)
    if len(post_ib) < 60:
        continue

    # Compute session ATR (from first 60 bars)
    ib_bars = session_df.iloc[:60]
    atr_vals = []
    for j in range(1, len(ib_bars)):
        tr = max(
            ib_bars.iloc[j]['high'] - ib_bars.iloc[j]['low'],
            abs(ib_bars.iloc[j]['high'] - ib_bars.iloc[j-1]['close']),
            abs(ib_bars.iloc[j]['low'] - ib_bars.iloc[j-1]['close'])
        )
        atr_vals.append(tr)
    session_atr = np.mean(atr_vals[-14:]) if len(atr_vals) >= 14 else np.mean(atr_vals)

    # Track what happens during the session
    session_high = session_df['high'].max()
    session_low = session_df['low'].min()
    post_ib_high = post_ib['high'].max()
    post_ib_low = post_ib['low'].min()

    # Did the 80P fire? (price re-entered VA)
    price_entered_va = False
    for j in range(len(post_ib)):
        bar = post_ib.iloc[j]
        if direction == 'LONG' and bar['low'] <= vah:
            price_entered_va = True
            break
        elif direction == 'SHORT' and bar['high'] >= val:
            price_entered_va = True
            break

    # Did IB extend in our direction?
    ib_extended = False
    if direction == 'LONG' and post_ib_high > ib['ibh']:
        ib_extended = True
    elif direction == 'SHORT' and post_ib_low < ib['ibl']:
        ib_extended = True

    # How far did price extend from IB?
    if direction == 'LONG':
        max_extension_from_ib = post_ib_high - ib['ibh']
        max_extension_from_va = post_ib_high - vah
    else:
        max_extension_from_ib = ib['ibl'] - post_ib_low
        max_extension_from_va = val - post_ib_low

    session_db.append({
        'session_date': current,
        'direction': direction,
        'open_price': open_price,
        'vah': vah, 'val': val, 'poc': poc, 'va_width': va_width,
        'gap_distance': gap_distance,
        'ib': ib,
        'session_atr': session_atr,
        'post_ib': post_ib,
        'session_df': session_df,
        'price_entered_va': price_entered_va,
        'ib_extended': ib_extended,
        'max_extension_from_ib': max_extension_from_ib,
        'max_extension_from_va': max_extension_from_va,
        'session_high': session_high,
        'session_low': session_low,
    })

n_setups = len(session_db)
n_long = sum(1 for s in session_db if s['direction'] == 'LONG')
n_short = sum(1 for s in session_db if s['direction'] == 'SHORT')
n_extended = sum(1 for s in session_db if s['ib_extended'])
n_entered_va = sum(1 for s in session_db if s['price_entered_va'])
n_pure_20p = sum(1 for s in session_db if not s['price_entered_va'])

print(f"\n  Total open-outside-VA sessions: {n_setups}")
print(f"  LONG (open > VAH): {n_long},  SHORT (open < VAL): {n_short}")
print(f"  IB extended in 20P direction: {n_extended} ({n_extended/n_setups*100:.0f}%)")
print(f"  Price re-entered VA (80P fired): {n_entered_va} ({n_entered_va/n_setups*100:.0f}%)")
print(f"  Price NEVER entered VA (pure 20P): {n_pure_20p} ({n_pure_20p/n_setups*100:.0f}%)")


# ============================================================================
# PART 1: FREQUENCY & SESSION CHARACTERISTICS
# ============================================================================
print(f"\n{'='*130}")
print(f"  PART 1 — 20P FREQUENCY & SESSION CHARACTERISTICS")
print(f"  How often does the 20P occur? What do these sessions look like?")
print(f"{'='*130}")

# Extension statistics
extensions = [s['max_extension_from_ib'] for s in session_db if s['ib_extended']]
gap_dists = [s['gap_distance'] for s in session_db]
ib_ranges = [s['ib']['ib_range'] for s in session_db]
va_widths = [s['va_width'] for s in session_db]

print(f"\n  Setup Frequency:")
print(f"    Open outside VA: {n_setups} sessions ({n_setups/n_sessions*100:.0f}%) = "
      f"{n_setups/months:.1f}/month")
print(f"    IB extends in trend dir: {n_extended} ({n_extended/n_sessions*100:.0f}%) = "
      f"{n_extended/months:.1f}/month")
print(f"    Pure 20P (never re-enters VA): {n_pure_20p} ({n_pure_20p/n_sessions*100:.0f}%) = "
      f"{n_pure_20p/months:.1f}/month")

print(f"\n  Gap Distance (open to VA boundary):")
print(f"    Mean: {np.mean(gap_dists):.0f} pts  |  Median: {np.median(gap_dists):.0f} pts  |  "
      f"25th: {np.percentile(gap_dists, 25):.0f} pts  |  75th: {np.percentile(gap_dists, 75):.0f} pts")

print(f"\n  IB Range (these sessions):")
print(f"    Mean: {np.mean(ib_ranges):.0f} pts  |  Median: {np.median(ib_ranges):.0f} pts  |  "
      f"25th: {np.percentile(ib_ranges, 25):.0f} pts  |  75th: {np.percentile(ib_ranges, 75):.0f} pts")

if extensions:
    print(f"\n  Extension from IB (when it extends):")
    print(f"    Mean: {np.mean(extensions):.0f} pts  |  Median: {np.median(extensions):.0f} pts  |  "
          f"75th: {np.percentile(extensions, 75):.0f} pts  |  90th: {np.percentile(extensions, 90):.0f} pts")

    # Extension as multiple of IB range
    ext_multiples = [s['max_extension_from_ib'] / s['ib']['ib_range']
                     for s in session_db if s['ib_extended'] and s['ib']['ib_range'] > 0]
    if ext_multiples:
        print(f"\n  Extension as multiple of IB range:")
        print(f"    Mean: {np.mean(ext_multiples):.2f}x  |  Median: {np.median(ext_multiples):.2f}x  |  "
              f"75th: {np.percentile(ext_multiples, 75):.2f}x  |  90th: {np.percentile(ext_multiples, 90):.2f}x")

# By direction
print(f"\n  By Direction:")
for d in ['LONG', 'SHORT']:
    subset = [s for s in session_db if s['direction'] == d]
    n_ext = sum(1 for s in subset if s['ib_extended'])
    n_pure = sum(1 for s in subset if not s['price_entered_va'])
    exts = [s['max_extension_from_ib'] for s in subset if s['ib_extended']]
    print(f"    {d}: {len(subset)} setups, {n_ext} IB extended ({n_ext/len(subset)*100:.0f}%), "
          f"{n_pure} pure 20P ({n_pure/len(subset)*100:.0f}%)"
          f"{f', avg ext: {np.mean(exts):.0f} pts' if exts else ''}")


# ============================================================================
# PART 2: IB EXTENSION ENTRY (accept above IBH / below IBL)
# ============================================================================
print(f"\n{'='*130}")
print(f"  PART 2 — IB EXTENSION ENTRY")
print(f"  Enter when price accepts beyond IB in the trend direction")
print(f"  Acceptance = N consecutive 5-min closes beyond IB boundary")
print(f"{'='*130}")


def find_ib_extension_acceptance(post_ib, ib, direction, period_min=5, n_periods=2):
    """Find N consecutive period closes beyond IB boundary (extension confirmation)."""
    agg = aggregate_bars(post_ib, period_min)
    consecutive = 0

    for i, bar in enumerate(agg):
        bt = bar['timestamp']
        bt_time = bt.time() if hasattr(bt, 'time') else None
        if bt_time and bt_time >= ENTRY_CUTOFF:
            break

        if direction == 'LONG' and bar['close'] > ib['ibh']:
            consecutive += 1
        elif direction == 'SHORT' and bar['close'] < ib['ibl']:
            consecutive += 1
        else:
            consecutive = 0

        if consecutive >= n_periods:
            return {
                'agg_idx': i,
                'bar_end_1min': agg[i]['bar_end'],
                'entry_price': agg[i]['close'],
                'candle_high': max(agg[j]['high'] for j in range(i - n_periods + 1, i + 1)),
                'candle_low': min(agg[j]['low'] for j in range(i - n_periods + 1, i + 1)),
            }

    return None


# Test different acceptance periods and targets
ib_ext_configs = []

acceptance_periods = [
    (5, 1, '1x5min'),
    (5, 2, '2x5min'),
    (5, 3, '3x5min'),
    (15, 1, '1x15min'),
    (30, 1, '1x30min'),
]

target_modes_20p = ['1x_ib', '1.5x_ib', '2x_ib', 'va_width', '2R', '4R']
stop_modes_20p = ['ib_boundary']
stop_buffers_20p = [10, 20]

for acc_period, acc_n, acc_label in acceptance_periods:
    for target_mode in target_modes_20p:
        for stop_buf in stop_buffers_20p:
            results = []

            for s in session_db:
                ext = find_ib_extension_acceptance(
                    s['post_ib'], s['ib'], s['direction'], acc_period, acc_n)
                if ext is None:
                    continue

                entry_price = ext['entry_price']
                stop_price = compute_stop(entry_price, s['direction'], 'ib_boundary',
                                          s['ib'], stop_buf)
                risk_pts = abs(entry_price - stop_price)
                if risk_pts < 2:
                    continue

                target_price = compute_target(entry_price, s['direction'], target_mode,
                                              s['ib'], s['va_width'], risk_pts)
                reward_pts = abs(target_price - entry_price)
                if reward_pts < 2:
                    continue

                r = replay_trade(s['post_ib'], ext['bar_end_1min'], entry_price,
                                 s['direction'], stop_price, target_price)
                if r:
                    r['session_date'] = str(s['session_date'])
                    r['direction'] = s['direction']
                    r['entry_price'] = entry_price
                    r['gap_distance'] = s['gap_distance']
                    r['ib_range'] = s['ib']['ib_range']
                    results.append(r)

            if len(results) >= 3:
                ib_ext_configs.append({
                    'label': f"{acc_label} | stop=IB+{stop_buf}pt | tgt={target_mode}",
                    'acc_label': acc_label,
                    'target': target_mode,
                    'stop_buf': stop_buf,
                    'results': results,
                })

# Print results by acceptance period (stop=10pt)
for target_mode in target_modes_20p:
    print(f"\n  Target: {target_mode}")
    print_header()
    for _, _, acc_label in acceptance_periods:
        match = [c for c in ib_ext_configs if c['acc_label'] == acc_label
                 and c['target'] == target_mode and c['stop_buf'] == 10]
        if match:
            print_results_table(match[0]['results'], f"{acc_label} | IB+10pt", months)
    print()


# ============================================================================
# PART 3: VA BOUNDARY REJECTION ENTRY
# ============================================================================
print(f"\n{'='*130}")
print(f"  PART 3 — VA BOUNDARY REJECTION ENTRY")
print(f"  Price opens outside VA, approaches VA edge, gets REJECTED")
print(f"  Enter on the bounce away from VA (confirms the trend)")
print(f"{'='*130}")


def find_va_rejection(post_ib, direction, vah, val, ib, min_wick_pts=5):
    """
    Find a bar that approaches VA boundary and bounces off (rejection).
    For LONG (open > VAH): price dips toward VAH, bounces up
    For SHORT (open < VAL): price spikes toward VAL, bounces down
    """
    for bar_idx in range(len(post_ib)):
        bar = post_ib.iloc[bar_idx]
        bt_time = bar.get('timestamp')
        bt = bt_time.time() if bt_time and hasattr(bt_time, 'time') else None
        if bt and bt >= ENTRY_CUTOFF:
            break

        if direction == 'LONG':
            # Price dips near VAH but closes above it
            # Wick touches or goes below VAH, close stays above
            proximity_to_va = bar['low'] - vah
            if proximity_to_va <= min_wick_pts and bar['close'] > vah:
                wick_rejection = bar['close'] - bar['low']
                if wick_rejection >= min_wick_pts:
                    return {
                        'bar_idx': bar_idx,
                        'entry_price': bar['close'],
                        'rejection_wick': wick_rejection,
                        'proximity': proximity_to_va,
                    }
        else:  # SHORT
            # Price spikes near VAL but closes below it
            proximity_to_va = val - bar['high']
            if proximity_to_va <= min_wick_pts and bar['close'] < val:
                wick_rejection = bar['high'] - bar['close']
                if wick_rejection >= min_wick_pts:
                    return {
                        'bar_idx': bar_idx,
                        'entry_price': bar['close'],
                        'rejection_wick': wick_rejection,
                        'proximity': proximity_to_va,
                    }
    return None


va_rej_configs = []
for target_mode in target_modes_20p:
    for stop_buf in [10, 20]:
        for min_wick in [3, 5, 10]:
            results = []
            for s in session_db:
                rej = find_va_rejection(s['post_ib'], s['direction'],
                                        s['vah'], s['val'], s['ib'], min_wick)
                if rej is None:
                    continue

                entry_price = rej['entry_price']
                # Stop inside VA (below VAH for long, above VAL for short)
                if s['direction'] == 'LONG':
                    stop_price = s['vah'] - stop_buf
                else:
                    stop_price = s['val'] + stop_buf

                risk_pts = abs(entry_price - stop_price)
                if risk_pts < 2:
                    continue

                target_price = compute_target(entry_price, s['direction'], target_mode,
                                              s['ib'], s['va_width'], risk_pts)
                reward_pts = abs(target_price - entry_price)
                if reward_pts < 2:
                    continue

                r = replay_trade(s['post_ib'], rej['bar_idx'], entry_price,
                                 s['direction'], stop_price, target_price)
                if r:
                    r['session_date'] = str(s['session_date'])
                    r['direction'] = s['direction']
                    r['entry_price'] = entry_price
                    r['rejection_wick'] = rej['rejection_wick']
                    results.append(r)

            if len(results) >= 3:
                va_rej_configs.append({
                    'label': f"VA rej wick>={min_wick}pt | VA+{stop_buf}pt | tgt={target_mode}",
                    'min_wick': min_wick,
                    'target': target_mode,
                    'stop_buf': stop_buf,
                    'results': results,
                })

# Print best VA rejection configs
for target_mode in ['1x_ib', '2x_ib', '2R', '4R']:
    print(f"\n  Target: {target_mode}")
    print_header()
    for min_wick in [3, 5, 10]:
        for stop_buf in [10, 20]:
            match = [c for c in va_rej_configs if c['min_wick'] == min_wick
                     and c['target'] == target_mode and c['stop_buf'] == stop_buf]
            if match:
                print_results_table(match[0]['results'],
                                    f"wick>={min_wick}pt | VA+{stop_buf}pt", months)
    print()


# ============================================================================
# PART 4: PULLBACK ENTRY AFTER IB EXTENSION
# ============================================================================
print(f"\n{'='*130}")
print(f"  PART 4 — PULLBACK ENTRY AFTER IB EXTENSION")
print(f"  Wait for IB extension, then buy the pullback to IB boundary")
print(f"  (Like 80P retest entry but for extensions)")
print(f"{'='*130}")

pb_configs = []
for target_mode in target_modes_20p:
    for stop_buf in [10, 20]:
        for pullback_pct in [0.0, 0.25, 0.50, 0.75, 1.0]:
            # pullback_pct: how far back toward IB boundary
            # 0% = enter immediately on extension
            # 100% = wait for full pullback to IB boundary
            results = []

            for s in session_db:
                # First find IB extension (2x5min acceptance)
                ext = find_ib_extension_acceptance(
                    s['post_ib'], s['ib'], s['direction'], 5, 2)
                if ext is None:
                    continue

                ibh, ibl = s['ib']['ibh'], s['ib']['ibl']

                if pullback_pct == 0.0:
                    # Immediate entry on extension
                    entry_price = ext['entry_price']
                    entry_bar = ext['bar_end_1min']
                else:
                    # Wait for pullback
                    ext_price = ext['entry_price']
                    if s['direction'] == 'LONG':
                        ib_boundary = ibh
                        pb_target = ext_price - pullback_pct * (ext_price - ib_boundary)
                    else:
                        ib_boundary = ibl
                        pb_target = ext_price + pullback_pct * (ib_boundary - ext_price)

                    # Search for pullback fill
                    entry_bar = None
                    entry_price = pb_target
                    for bar_idx in range(ext['bar_end_1min'] + 1, len(s['post_ib'])):
                        bar = s['post_ib'].iloc[bar_idx]
                        bt_time = bar.get('timestamp')
                        bt = bt_time.time() if bt_time and hasattr(bt_time, 'time') else None
                        if bt and bt >= ENTRY_CUTOFF:
                            break
                        if s['direction'] == 'LONG' and bar['low'] <= pb_target:
                            entry_bar = bar_idx
                            break
                        elif s['direction'] == 'SHORT' and bar['high'] >= pb_target:
                            entry_bar = bar_idx
                            break

                    if entry_bar is None:
                        continue

                stop_price = compute_stop(entry_price, s['direction'], 'ib_boundary',
                                          s['ib'], stop_buf)
                risk_pts = abs(entry_price - stop_price)
                if risk_pts < 2:
                    continue

                target_price = compute_target(entry_price, s['direction'], target_mode,
                                              s['ib'], s['va_width'], risk_pts)
                reward_pts = abs(target_price - entry_price)
                if reward_pts < 2:
                    continue

                r = replay_trade(s['post_ib'], entry_bar, entry_price,
                                 s['direction'], stop_price, target_price)
                if r:
                    r['session_date'] = str(s['session_date'])
                    r['direction'] = s['direction']
                    r['entry_price'] = entry_price
                    results.append(r)

            if len(results) >= 3:
                pb_label = 'Immediate' if pullback_pct == 0.0 else f'{pullback_pct*100:.0f}% pullback'
                pb_configs.append({
                    'label': f"{pb_label} | IB+{stop_buf}pt | tgt={target_mode}",
                    'pullback_pct': pullback_pct,
                    'target': target_mode,
                    'stop_buf': stop_buf,
                    'results': results,
                })

for target_mode in ['1x_ib', '1.5x_ib', '2x_ib', '2R', '4R']:
    print(f"\n  Target: {target_mode}")
    print_header()
    for pb_pct in [0.0, 0.25, 0.50, 0.75, 1.0]:
        for stop_buf in [10]:
            match = [c for c in pb_configs if c['pullback_pct'] == pb_pct
                     and c['target'] == target_mode and c['stop_buf'] == stop_buf]
            if match:
                pb_label = 'Immediate' if pb_pct == 0.0 else f'{pb_pct*100:.0f}% pullback'
                print_results_table(match[0]['results'],
                                    f"{pb_label} | IB+10pt", months)
    print()


# ============================================================================
# PART 5: TIME-BASED CONFIRMATION (N bars outside VA)
# ============================================================================
print(f"\n{'='*130}")
print(f"  PART 5 — TIME-BASED CONFIRMATION")
print(f"  Enter after N consecutive 5-min bars close OUTSIDE VA (staying out)")
print(f"{'='*130}")

time_configs = []
for n_outside in [2, 3, 4, 5]:
    for target_mode in ['1x_ib', '2x_ib', '2R', '4R']:
        results = []

        for s in session_db:
            agg = aggregate_bars(s['post_ib'], 5)
            consecutive_outside = 0
            entry_info = None

            for idx, bar in enumerate(agg):
                bt = bar['timestamp']
                bt_time = bt.time() if hasattr(bt, 'time') else None
                if bt_time and bt_time >= ENTRY_CUTOFF:
                    break

                if s['direction'] == 'LONG' and bar['close'] > s['vah']:
                    consecutive_outside += 1
                elif s['direction'] == 'SHORT' and bar['close'] < s['val']:
                    consecutive_outside += 1
                else:
                    consecutive_outside = 0

                if consecutive_outside >= n_outside:
                    entry_info = {
                        'bar_end_1min': bar['bar_end'],
                        'entry_price': bar['close'],
                    }
                    break

            if entry_info is None:
                continue

            entry_price = entry_info['entry_price']
            stop_price = compute_stop(entry_price, s['direction'], 'ib_boundary',
                                      s['ib'], 10)
            risk_pts = abs(entry_price - stop_price)
            if risk_pts < 2:
                continue

            target_price = compute_target(entry_price, s['direction'], target_mode,
                                          s['ib'], s['va_width'], risk_pts)
            reward_pts = abs(target_price - entry_price)
            if reward_pts < 2:
                continue

            r = replay_trade(s['post_ib'], entry_info['bar_end_1min'], entry_price,
                             s['direction'], stop_price, target_price)
            if r:
                r['session_date'] = str(s['session_date'])
                r['direction'] = s['direction']
                results.append(r)

        if len(results) >= 3:
            time_configs.append({
                'label': f"{n_outside}x5min outside VA | IB+10pt | tgt={target_mode}",
                'n_outside': n_outside,
                'target': target_mode,
                'results': results,
            })

for target_mode in ['1x_ib', '2x_ib', '2R', '4R']:
    print(f"\n  Target: {target_mode}")
    print_header()
    for n_out in [2, 3, 4, 5]:
        match = [c for c in time_configs if c['n_outside'] == n_out
                 and c['target'] == target_mode]
        if match:
            print_results_table(match[0]['results'],
                                f"{n_out}x5min outside VA | IB+10pt", months)
    print()


# ============================================================================
# PART 6: ORDER FLOW FILTERS
# ============================================================================
print(f"\n{'='*130}")
print(f"  PART 6 — ORDER FLOW FILTERS")
print(f"  Does delta/CVD at entry improve the IB extension model?")
print(f"{'='*130}")

# Use 2x5min IB extension as base model
of_configs = []
for target_mode in ['1x_ib', '2x_ib', '2R']:
    for filter_name, filter_fn in [
        ('No filter', lambda bar, d: True),
        ('Delta aligns', lambda bar, d: (bar.get('vol_delta', 0) > 0 if d == 'LONG'
                                          else bar.get('vol_delta', 0) < 0)),
        ('Delta pctl>60', lambda bar, d: bar.get('delta_percentile', 50) > 60),
        ('Delta pctl>75', lambda bar, d: bar.get('delta_percentile', 50) > 75),
        ('Vol spike>1.2', lambda bar, d: bar.get('volume_spike', 0) > 1.2 if 'volume_spike' in bar.index else True),
    ]:
        results = []

        for s in session_db:
            ext = find_ib_extension_acceptance(
                s['post_ib'], s['ib'], s['direction'], 5, 2)
            if ext is None:
                continue

            entry_bar_idx = ext['bar_end_1min']
            entry_bar = s['post_ib'].iloc[entry_bar_idx]

            if not filter_fn(entry_bar, s['direction']):
                continue

            entry_price = ext['entry_price']
            stop_price = compute_stop(entry_price, s['direction'], 'ib_boundary',
                                      s['ib'], 10)
            risk_pts = abs(entry_price - stop_price)
            if risk_pts < 2:
                continue

            target_price = compute_target(entry_price, s['direction'], target_mode,
                                          s['ib'], s['va_width'], risk_pts)
            reward_pts = abs(target_price - entry_price)
            if reward_pts < 2:
                continue

            r = replay_trade(s['post_ib'], entry_bar_idx, entry_price,
                             s['direction'], stop_price, target_price)
            if r:
                r['session_date'] = str(s['session_date'])
                r['direction'] = s['direction']
                results.append(r)

        if len(results) >= 3:
            of_configs.append({
                'label': f"2x5min ext + {filter_name} | IB+10pt | tgt={target_mode}",
                'filter_name': filter_name,
                'target': target_mode,
                'results': results,
            })

for target_mode in ['1x_ib', '2x_ib', '2R']:
    print(f"\n  Target: {target_mode}")
    print_header()
    for fn in ['No filter', 'Delta aligns', 'Delta pctl>60', 'Delta pctl>75', 'Vol spike>1.2']:
        match = [c for c in of_configs if c['filter_name'] == fn and c['target'] == target_mode]
        if match:
            print_results_table(match[0]['results'], f"2x5min + {fn}", months)
    print()


# ============================================================================
# PART 7: TARGET ANALYSIS (MFE study — how far do 20P sessions extend?)
# ============================================================================
print(f"\n{'='*130}")
print(f"  PART 7 — EXTENSION DEPTH ANALYSIS (MFE Study)")
print(f"  How far do sessions extend when the 20P works?")
print(f"{'='*130}")

# For all sessions with IB extension, track max favorable excursion
ext_sessions = [s for s in session_db if s['ib_extended']]
if ext_sessions:
    ext_from_ib = [s['max_extension_from_ib'] for s in ext_sessions]
    ext_from_va = [s['max_extension_from_va'] for s in ext_sessions]
    ib_ranges_ext = [s['ib']['ib_range'] for s in ext_sessions]
    ext_mult = [e / r for e, r in zip(ext_from_ib, ib_ranges_ext) if r > 0]

    print(f"\n  Sessions with IB extension: {len(ext_sessions)}")

    print(f"\n  Max Extension from IB boundary:")
    for pctl in [25, 50, 75, 90, 95]:
        print(f"    {pctl}th percentile: {np.percentile(ext_from_ib, pctl):.0f} pts")

    print(f"\n  Max Extension from VA boundary:")
    for pctl in [25, 50, 75, 90, 95]:
        print(f"    {pctl}th percentile: {np.percentile(ext_from_va, pctl):.0f} pts")

    print(f"\n  Extension as Multiple of IB Range:")
    for pctl in [25, 50, 75, 90, 95]:
        print(f"    {pctl}th percentile: {np.percentile(ext_mult, pctl):.2f}x IB range")

    # What % of sessions reach each target level?
    print(f"\n  % of IB-extension sessions reaching target:")
    for mult in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        hit = sum(1 for e, r in zip(ext_from_ib, ib_ranges_ext) if e >= mult * r and r > 0)
        pct = hit / len(ext_sessions) * 100
        print(f"    {mult:.1f}x IB range: {hit}/{len(ext_sessions)} ({pct:.0f}%)")

    # By direction
    print(f"\n  By Direction:")
    for d in ['LONG', 'SHORT']:
        sub = [s for s in ext_sessions if s['direction'] == d]
        if sub:
            exts = [s['max_extension_from_ib'] for s in sub]
            print(f"    {d}: {len(sub)} sessions, median ext: {np.median(exts):.0f} pts, "
                  f"75th: {np.percentile(exts, 75):.0f} pts")


# ============================================================================
# PART 8: HEAD-TO-HEAD RANKING
# ============================================================================
print(f"\n{'='*130}")
print(f"  PART 8 — HEAD-TO-HEAD: ALL 20P MODELS RANKED BY $/MONTH")
print(f"{'='*130}")

all_20p_configs = ib_ext_configs + va_rej_configs + pb_configs + time_configs + of_configs

ranked = []
for c in all_20p_configs:
    df_r = pd.DataFrame(c['results'])
    n = len(df_r)
    if n < 3:
        continue
    wr = df_r['is_winner'].mean() * 100
    gw = df_r[df_r['is_winner']]['pnl_dollars'].sum()
    gl = abs(df_r[~df_r['is_winner']]['pnl_dollars'].sum())
    pf = gw / gl if gl > 0 else float('inf')
    pm = df_r['pnl_dollars'].sum() / months
    avg_risk = df_r['risk_pts'].mean()
    avg_mfe = df_r['mfe_pts'].mean()
    ranked.append({
        'label': c['label'],
        'n': n, 'tpm': n / months, 'wr': wr, 'pf': pf, 'pm': pm,
        'risk': avg_risk, 'mfe': avg_mfe,
    })

ranked.sort(key=lambda x: x['pm'], reverse=True)

print(f"\n  {'Rank':>4s} {'Config':<60s} {'N':>4s} {'Trd/Mo':>7s} {'WR':>6s} {'PF':>6s} "
      f"{'$/Mo':>9s} {'Risk':>5s} {'MFE':>5s}")
print(f"  {'-'*115}")

for rank, c in enumerate(ranked[:40], 1):
    marker = ' <--' if rank <= 3 else ''
    print(f"  {rank:>4d} {c['label']:<60s} {c['n']:>4d} {c['tpm']:>6.1f} "
          f"{c['wr']:>5.1f}% {c['pf']:>5.2f} ${c['pm']:>7,.0f} {c['risk']:>4.0f}p "
          f"{c['mfe']:>4.0f}p{marker}")


# ============================================================================
# PART 9: LONG vs SHORT BREAKDOWN (best configs)
# ============================================================================
print(f"\n{'='*130}")
print(f"  PART 9 — LONG vs SHORT BREAKDOWN (Top 5 configs)")
print(f"{'='*130}")

# Get top 5 from all_20p_configs
top5_labels = [r['label'] for r in ranked[:5]]
for label in top5_labels:
    match = [c for c in all_20p_configs if c['label'] == label]
    if not match:
        continue
    c = match[0]
    df_r = pd.DataFrame(c['results'])
    print(f"\n  {label}")

    for d in ['LONG', 'SHORT']:
        sub = df_r[df_r['direction'] == d]
        if len(sub) == 0:
            print(f"    {d}: 0 trades")
            continue
        n = len(sub)
        wr = sub['is_winner'].mean() * 100
        gw = sub[sub['is_winner']]['pnl_dollars'].sum()
        gl = abs(sub[~sub['is_winner']]['pnl_dollars'].sum())
        pf = gw / gl if gl > 0 else float('inf')
        pm = sub['pnl_dollars'].sum() / months
        print(f"    {d}: {n} trades, {wr:.1f}% WR, PF {pf:.2f}, ${pm:,.0f}/mo")


# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*130}")
print(f"  20P RULE STUDY — SUMMARY")
print(f"{'='*130}")

print(f"""
  SETUP FREQUENCY:
    Open outside VA:                {n_setups} sessions ({n_setups/n_sessions*100:.0f}%) = {n_setups/months:.1f}/month
    IB extends in trend direction:  {n_extended} ({n_extended/n_setups*100:.0f}% of setups) = {n_extended/months:.1f}/month
    Pure 20P (never re-enters VA):  {n_pure_20p} ({n_pure_20p/n_setups*100:.0f}% of setups) = {n_pure_20p/months:.1f}/month

  BEST CONFIGS (by $/month):""")

for rank, c in enumerate(ranked[:5], 1):
    print(f"    #{rank}: {c['label']}")
    print(f"        {c['n']} trades, {c['tpm']:.1f}/mo, {c['wr']:.1f}% WR, "
          f"PF {c['pf']:.2f}, ${c['pm']:,.0f}/mo")

print(f"""
  COMPARISON TO 80P:
    80P Baseline (1x30min + 4R):         60 trades, 38.3% WR, PF 1.70, $955/mo
    80P Best (Limit 50% VA + 4R):        47 trades, 44.7% WR, PF 2.57, $1,922/mo
    80P Double Top (100% retest + 2R):   35 trades, 65.7% WR, PF 3.45, $915/mo

  KEY QUESTIONS ANSWERED:
    1. How often does the 20P (trend continuation) occur?
    2. Is IB extension a reliable entry trigger?
    3. Does pullback to IB boundary improve entry?
    4. Do order flow filters help?
    5. How far do extensions typically run (target sizing)?
    6. LONG vs SHORT: is the NQ long bias still dominant?
""")

print(f"{'='*130}")
print(f"  20P RULE STUDY COMPLETE")
print(f"{'='*130}")
