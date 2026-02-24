"""
20P Rule Study v2 — Structure-Based Stops
==========================================

v1 used mechanical IB-boundary stops (~200 pts) — no human trader would do that.
v2 uses REAL stops against structure: swing points, FVGs, ATR, VWAP, POC, VAL.

The thesis: with tight structure-based stops (20-40 pts instead of 200),
the 20P breakout becomes a viable trade because R:R flips from 0.3R to 2-3R+.
Size accordingly to the stop — 1-2 ATR max.

Stop hierarchy tested:
  TIGHT (structure):
    1. Swing point — nearest swing low (LONG) / swing high (SHORT) before entry
    2. 15-min FVG — bottom of bull FVG (LONG) / top of bear FVG (SHORT)
    3. 1-min FVG — same but 1-min timeframe
    4. 1 ATR — entry ± 1 * ATR14
    5. 2 ATR — entry ± 2 * ATR14

  WIDE (still reasonable):
    6. VWAP — session VWAP at entry time
    7. POC — prior session POC
    8. VAL/VAH — prior session VA boundary (closer one)

  NEVER:
    IBL for IBH breakout — that's the full IB range (~200 pts), nonsensical
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import time
from typing import Dict, List, Optional, Tuple

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.instruments import get_instrument
from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from indicators.value_area import compute_session_value_areas
from indicators.smt_divergence import detect_swing_points

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


def find_ib_extension_acceptance(post_ib, ib, direction, period_min=5, n_periods=2):
    """Find N consecutive period closes beyond IB boundary."""
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


def replay_trade(post_ib, entry_bar_idx, entry_price, direction,
                 stop_price, target_price):
    """Replay a trade bar-by-bar with explicit stop/target."""
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


def print_results_table(results_list, label, months_val):
    """Print a standard results table from a list of trade dicts."""
    if not results_list:
        print(f"    {label:<55s}    0  (no trades)")
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
    avg_rr = df_r['rr_ratio'].mean()
    stopped = (df_r['exit_reason'] == 'STOP').sum()
    target_hit = (df_r['exit_reason'] == 'TARGET').sum()
    eod = (df_r['exit_reason'] == 'EOD').sum()
    avg_mfe = df_r['mfe_pts'].mean()
    avg_mae = df_r['mae_pts'].mean()

    print(f"    {label:<55s} {n:>4d} {n/months_val:>5.1f} "
          f"{wr:>5.1f}% {pf:>5.2f} ${pm:>8,.0f} ${avg_w:>7,.0f} ${avg_l:>7,.0f} "
          f"{avg_risk:>5.0f}p {avg_rr:>4.1f}R {avg_mfe:>5.0f}/{avg_mae:>4.0f} "
          f"{stopped:>3d}/{target_hit:>3d}/{eod:>3d}")


def print_header():
    print(f"    {'Config':<55s} {'N':>4s} {'T/Mo':>5s} "
          f"{'WR':>6s} {'PF':>6s} {'$/Mo':>9s} {'AvgWin':>8s} {'AvgLoss':>8s} "
          f"{'Risk':>6s} {'R:R':>4s} {'MFE/MAE':>9s} {'S/T/E':>10s}")
    print(f"    {'-'*145}")


# ============================================================================
# STRUCTURE-BASED STOP FUNCTIONS
# ============================================================================
def find_swing_stop(session_df, entry_bar_global_idx, direction, lookback=3):
    """
    Find nearest swing point before entry for stop placement.
    LONG: nearest swing LOW before entry → stop below it
    SHORT: nearest swing HIGH before entry → stop above it

    Uses lookback=3 for tighter pivots (faster detection, closer to price).
    """
    # Use bars up to and including entry area
    bars_before = session_df.iloc[:entry_bar_global_idx + 1]
    if len(bars_before) < lookback * 2 + 1:
        return None

    swings = detect_swing_points(
        bars_before['high'].values,
        bars_before['low'].values,
        lookback=lookback,
    )

    if direction == 'LONG':
        # Find nearest swing low before entry
        swing_lows = [s for s in swings if s.swing_type == 'LOW']
        if not swing_lows:
            return None
        # Take the most recent one
        nearest = swing_lows[-1]
        return nearest.price - 2  # 2pt buffer below swing low
    else:
        # Find nearest swing high before entry
        swing_highs = [s for s in swings if s.swing_type == 'HIGH']
        if not swing_highs:
            return None
        nearest = swing_highs[-1]
        return nearest.price + 2  # 2pt buffer above swing high


def find_fvg_stop(entry_bar_row, direction, timeframe='15m'):
    """
    Use FVG zone as stop.
    LONG: stop below bottom of nearest bull FVG (support)
    SHORT: stop above top of nearest bear FVG (resistance)
    """
    if timeframe == '15m':
        if direction == 'LONG':
            fvg_level = entry_bar_row.get('fvg_bull_15m_bottom', None)
            if fvg_level and not pd.isna(fvg_level) and fvg_level > 0:
                return fvg_level - 2  # 2pt buffer
        else:
            fvg_level = entry_bar_row.get('fvg_bear_15m_top', None)
            if fvg_level and not pd.isna(fvg_level) and fvg_level > 0:
                return fvg_level + 2
    else:  # 1min
        if direction == 'LONG':
            fvg_level = entry_bar_row.get('fvg_bull_bottom', None)
            if fvg_level and not pd.isna(fvg_level) and fvg_level > 0:
                return fvg_level - 2
        else:
            fvg_level = entry_bar_row.get('fvg_bear_top', None)
            if fvg_level and not pd.isna(fvg_level) and fvg_level > 0:
                return fvg_level + 2

    return None


def find_atr_stop(entry_price, direction, atr_value, multiplier=1.0):
    """Stop at N * ATR from entry."""
    if atr_value is None or pd.isna(atr_value) or atr_value <= 0:
        return None
    distance = atr_value * multiplier
    if direction == 'LONG':
        return entry_price - distance
    else:
        return entry_price + distance


def find_vwap_stop(entry_bar_row, direction):
    """Use session VWAP as stop level."""
    vwap = entry_bar_row.get('vwap', None)
    if vwap is None or pd.isna(vwap) or vwap <= 0:
        return None
    if direction == 'LONG':
        # VWAP should be below entry for longs
        return vwap - 2  # 2pt buffer
    else:
        return vwap + 2


def find_poc_stop(poc, direction):
    """Use prior session POC as stop."""
    if poc is None or pd.isna(poc) or poc <= 0:
        return None
    if direction == 'LONG':
        return poc - 2
    else:
        return poc + 2


def find_va_boundary_stop(vah, val, direction):
    """Use prior session VA boundary (closer one) as stop."""
    if direction == 'LONG':
        # For longs opening above VAH, VAH is the nearest support
        return vah - 2
    else:
        # For shorts opening below VAL, VAL is the nearest resistance
        return val + 2


def validate_stop(stop_price, entry_price, direction, max_risk_pts=None):
    """Validate stop is on correct side and within max risk."""
    if stop_price is None:
        return None

    if direction == 'LONG' and stop_price >= entry_price:
        return None  # Stop above entry for long = invalid
    if direction == 'SHORT' and stop_price <= entry_price:
        return None  # Stop below entry for short = invalid

    risk = abs(entry_price - stop_price)
    if risk < 3:
        return None  # Too tight, will get noise-stopped

    if max_risk_pts and risk > max_risk_pts:
        return None  # Too wide

    return stop_price


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

    # Track session stats
    session_high = session_df['high'].max()
    session_low = session_df['low'].min()
    post_ib_high = post_ib['high'].max()
    post_ib_low = post_ib['low'].min()

    price_entered_va = False
    for j in range(len(post_ib)):
        bar = post_ib.iloc[j]
        if direction == 'LONG' and bar['low'] <= vah:
            price_entered_va = True
            break
        elif direction == 'SHORT' and bar['high'] >= val:
            price_entered_va = True
            break

    ib_extended = False
    if direction == 'LONG' and post_ib_high > ib['ibh']:
        ib_extended = True
    elif direction == 'SHORT' and post_ib_low < ib['ibl']:
        ib_extended = True

    if direction == 'LONG':
        max_extension_from_ib = post_ib_high - ib['ibh']
    else:
        max_extension_from_ib = ib['ibl'] - post_ib_low

    session_db.append({
        'session_date': current,
        'direction': direction,
        'open_price': open_price,
        'vah': vah, 'val': val, 'poc': poc, 'va_width': va_width,
        'gap_distance': gap_distance,
        'ib': ib,
        'post_ib': post_ib,
        'session_df': session_df,
        'price_entered_va': price_entered_va,
        'ib_extended': ib_extended,
        'max_extension_from_ib': max_extension_from_ib,
    })

n_setups = len(session_db)
n_extended = sum(1 for s in session_db if s['ib_extended'])
print(f"  Open-outside-VA sessions: {n_setups}")
print(f"  IB extended in trend direction: {n_extended} ({n_extended/n_setups*100:.0f}%)")


# ============================================================================
# PART 1: STOP DISTANCE ANALYSIS
# ============================================================================
print(f"\n{'='*150}")
print(f"  PART 1 — STOP DISTANCE ANALYSIS")
print(f"  How far are structure-based stops vs the old IB-boundary stop?")
print(f"{'='*150}")

# For each session with IB extension + 3x5min acceptance, compute all stop types
stop_distances = {
    'swing_3bar': [], 'swing_5bar': [],
    'fvg_15m': [], 'fvg_1m': [],
    'atr_1x': [], 'atr_2x': [],
    'vwap': [], 'poc': [], 'va_boundary': [],
    'ib_boundary': [],  # The old dumb stop for comparison
}

for s in session_db:
    ext = find_ib_extension_acceptance(s['post_ib'], s['ib'], s['direction'], 5, 3)
    if ext is None:
        continue

    entry_price = ext['entry_price']
    entry_bar_idx = ext['bar_end_1min']
    entry_bar = s['post_ib'].iloc[entry_bar_idx]

    # Global index in session_df for swing detection (60 IB bars + post_ib offset)
    global_idx = 60 + entry_bar_idx

    # Swing point stops (3-bar and 5-bar pivots)
    for lb, key in [(3, 'swing_3bar'), (5, 'swing_5bar')]:
        sp = find_swing_stop(s['session_df'], global_idx, s['direction'], lookback=lb)
        sp = validate_stop(sp, entry_price, s['direction'])
        if sp:
            stop_distances[key].append(abs(entry_price - sp))

    # FVG stops
    for tf, key in [('15m', 'fvg_15m'), ('1m', 'fvg_1m')]:
        fvg_stop = find_fvg_stop(entry_bar, s['direction'], tf)
        fvg_stop = validate_stop(fvg_stop, entry_price, s['direction'])
        if fvg_stop:
            stop_distances[key].append(abs(entry_price - fvg_stop))

    # ATR stops
    atr_val = entry_bar.get('atr14', None)
    for mult, key in [(1.0, 'atr_1x'), (2.0, 'atr_2x')]:
        atr_stop = find_atr_stop(entry_price, s['direction'], atr_val, mult)
        atr_stop = validate_stop(atr_stop, entry_price, s['direction'])
        if atr_stop:
            stop_distances[key].append(abs(entry_price - atr_stop))

    # VWAP stop
    vwap_stop = find_vwap_stop(entry_bar, s['direction'])
    vwap_stop = validate_stop(vwap_stop, entry_price, s['direction'])
    if vwap_stop:
        stop_distances['vwap'].append(abs(entry_price - vwap_stop))

    # POC stop
    poc_stop = find_poc_stop(s['poc'], s['direction'])
    poc_stop = validate_stop(poc_stop, entry_price, s['direction'])
    if poc_stop:
        stop_distances['poc'].append(abs(entry_price - poc_stop))

    # VA boundary stop
    va_stop = find_va_boundary_stop(s['vah'], s['val'], s['direction'])
    va_stop = validate_stop(va_stop, entry_price, s['direction'])
    if va_stop:
        stop_distances['va_boundary'].append(abs(entry_price - va_stop))

    # Old IB boundary stop (for comparison)
    ibh, ibl = s['ib']['ibh'], s['ib']['ibl']
    ib_stop = (ibl - 10) if s['direction'] == 'LONG' else (ibh + 10)
    stop_distances['ib_boundary'].append(abs(entry_price - ib_stop))

print(f"\n  {'Stop Type':<20s} {'N':>5s} {'Mean':>8s} {'Median':>8s} {'25th':>8s} {'75th':>8s} {'vs IB':>8s}")
print(f"  {'-'*70}")

ib_median = np.median(stop_distances['ib_boundary']) if stop_distances['ib_boundary'] else 1
for key, label in [
    ('swing_3bar', 'Swing (3-bar)'),
    ('swing_5bar', 'Swing (5-bar)'),
    ('fvg_15m', '15m FVG'),
    ('fvg_1m', '1m FVG'),
    ('atr_1x', '1x ATR'),
    ('atr_2x', '2x ATR'),
    ('vwap', 'VWAP'),
    ('poc', 'POC'),
    ('va_boundary', 'VA boundary'),
    ('ib_boundary', 'IB boundary (v1)'),
]:
    dists = stop_distances[key]
    if not dists:
        print(f"  {label:<20s} {'0':>5s}   (no data)")
        continue
    med = np.median(dists)
    ratio = med / ib_median if ib_median > 0 else 0
    print(f"  {label:<20s} {len(dists):>5d} {np.mean(dists):>7.0f}p {med:>7.0f}p "
          f"{np.percentile(dists, 25):>7.0f}p {np.percentile(dists, 75):>7.0f}p "
          f"{ratio:>7.1%}")


# ============================================================================
# PART 2: STRUCTURE STOP BACKTEST — All stop types × R-multiple targets
# ============================================================================
print(f"\n{'='*150}")
print(f"  PART 2 — STRUCTURE STOP BACKTEST")
print(f"  Entry: 3x5min IB extension acceptance")
print(f"  Stops: Structure-based  |  Targets: R-multiples + fixed")
print(f"{'='*150}")

# Define stop models
STOP_MODELS = [
    ('swing_3bar', 'Swing 3-bar', None),
    ('swing_5bar', 'Swing 5-bar', None),
    ('fvg_15m', '15m FVG', None),
    ('fvg_1m', '1m FVG', None),
    ('atr_1x', '1x ATR', None),
    ('atr_2x', '2x ATR', None),
    ('vwap', 'VWAP', None),
    ('poc', 'POC', None),
    ('va_boundary', 'VA boundary', None),
]

TARGET_MODES = ['2R', '3R', '4R', '6R', '1x_ib']

all_configs = []


def get_structure_stop(stop_type, session, entry_price, entry_bar_idx, entry_bar_row,
                       direction, max_risk=None):
    """Compute stop price for a given stop type."""
    global_idx = 60 + entry_bar_idx

    if stop_type == 'swing_3bar':
        raw = find_swing_stop(session['session_df'], global_idx, direction, lookback=3)
    elif stop_type == 'swing_5bar':
        raw = find_swing_stop(session['session_df'], global_idx, direction, lookback=5)
    elif stop_type == 'fvg_15m':
        raw = find_fvg_stop(entry_bar_row, direction, '15m')
    elif stop_type == 'fvg_1m':
        raw = find_fvg_stop(entry_bar_row, direction, '1m')
    elif stop_type == 'atr_1x':
        atr_val = entry_bar_row.get('atr14', None)
        raw = find_atr_stop(entry_price, direction, atr_val, 1.0)
    elif stop_type == 'atr_2x':
        atr_val = entry_bar_row.get('atr14', None)
        raw = find_atr_stop(entry_price, direction, atr_val, 2.0)
    elif stop_type == 'vwap':
        raw = find_vwap_stop(entry_bar_row, direction)
    elif stop_type == 'poc':
        raw = find_poc_stop(session['poc'], direction)
    elif stop_type == 'va_boundary':
        raw = find_va_boundary_stop(session['vah'], session['val'], direction)
    else:
        return None

    return validate_stop(raw, entry_price, direction, max_risk)


def compute_target(entry_price, direction, target_mode, ib, va_width, risk_pts):
    """Compute target price based on mode."""
    ibh, ibl, ib_range = ib['ibh'], ib['ibl'], ib['ib_range']

    if target_mode == '1x_ib':
        return (ibh + ib_range) if direction == 'LONG' else (ibl - ib_range)
    elif target_mode == '1.5x_ib':
        return (ibh + 1.5 * ib_range) if direction == 'LONG' else (ibl - 1.5 * ib_range)
    elif target_mode.endswith('R'):
        r_mult = float(target_mode[:-1])
        return (entry_price + risk_pts * r_mult) if direction == 'LONG' \
               else (entry_price - risk_pts * r_mult)
    else:
        return (ibh + ib_range) if direction == 'LONG' else (ibl - ib_range)


# Run all stop × target combinations
for stop_type, stop_label, _ in STOP_MODELS:
    for target_mode in TARGET_MODES:
        results = []

        for s in session_db:
            ext = find_ib_extension_acceptance(
                s['post_ib'], s['ib'], s['direction'], 5, 3)
            if ext is None:
                continue

            entry_price = ext['entry_price']
            entry_bar_idx = ext['bar_end_1min']
            entry_bar = s['post_ib'].iloc[entry_bar_idx]

            stop_price = get_structure_stop(
                stop_type, s, entry_price, entry_bar_idx, entry_bar, s['direction'])
            if stop_price is None:
                continue

            risk_pts = abs(entry_price - stop_price)
            target_price = compute_target(
                entry_price, s['direction'], target_mode,
                s['ib'], s['va_width'], risk_pts)
            reward_pts = abs(target_price - entry_price)
            if reward_pts < 2:
                continue

            r = replay_trade(s['post_ib'], entry_bar_idx, entry_price,
                             s['direction'], stop_price, target_price)
            if r:
                r['session_date'] = str(s['session_date'])
                r['direction'] = s['direction']
                r['entry_price'] = entry_price
                r['stop_type'] = stop_type
                r['gap_distance'] = s['gap_distance']
                r['ib_range'] = s['ib']['ib_range']
                results.append(r)

        if len(results) >= 3:
            all_configs.append({
                'label': f"{stop_label} | tgt={target_mode}",
                'stop_type': stop_type,
                'stop_label': stop_label,
                'target': target_mode,
                'results': results,
            })

# Print results grouped by stop type
for stop_type, stop_label, _ in STOP_MODELS:
    configs = [c for c in all_configs if c['stop_type'] == stop_type]
    if not configs:
        continue
    print(f"\n  Stop: {stop_label}")
    print_header()
    for target_mode in TARGET_MODES:
        match = [c for c in configs if c['target'] == target_mode]
        if match:
            print_results_table(match[0]['results'],
                                f"{stop_label} | {target_mode}", months)
    print()


# ============================================================================
# PART 3: ATR-CAPPED STOPS (structure stop capped at 2 ATR max)
# ============================================================================
print(f"\n{'='*150}")
print(f"  PART 3 — ATR-CAPPED STRUCTURE STOPS")
print(f"  Use structure stop, but cap at 2x ATR max (size accordingly)")
print(f"  If structure stop > 2 ATR, skip the trade")
print(f"{'='*150}")

capped_configs = []

for stop_type, stop_label, _ in STOP_MODELS:
    if stop_type in ('atr_1x', 'atr_2x'):
        continue  # ATR stops don't need ATR capping

    for target_mode in TARGET_MODES:
        results = []

        for s in session_db:
            ext = find_ib_extension_acceptance(
                s['post_ib'], s['ib'], s['direction'], 5, 3)
            if ext is None:
                continue

            entry_price = ext['entry_price']
            entry_bar_idx = ext['bar_end_1min']
            entry_bar = s['post_ib'].iloc[entry_bar_idx]

            # Get ATR for max risk cap
            atr_val = entry_bar.get('atr14', None)
            if atr_val is None or pd.isna(atr_val) or atr_val <= 0:
                continue
            max_risk = 2.0 * atr_val

            stop_price = get_structure_stop(
                stop_type, s, entry_price, entry_bar_idx, entry_bar,
                s['direction'], max_risk=max_risk)
            if stop_price is None:
                continue

            risk_pts = abs(entry_price - stop_price)
            target_price = compute_target(
                entry_price, s['direction'], target_mode,
                s['ib'], s['va_width'], risk_pts)
            reward_pts = abs(target_price - entry_price)
            if reward_pts < 2:
                continue

            r = replay_trade(s['post_ib'], entry_bar_idx, entry_price,
                             s['direction'], stop_price, target_price)
            if r:
                r['session_date'] = str(s['session_date'])
                r['direction'] = s['direction']
                r['entry_price'] = entry_price
                r['stop_type'] = stop_type
                r['atr_at_entry'] = atr_val
                results.append(r)

        if len(results) >= 3:
            capped_configs.append({
                'label': f"{stop_label} (≤2ATR) | tgt={target_mode}",
                'stop_type': stop_type,
                'stop_label': stop_label,
                'target': target_mode,
                'results': results,
            })

for stop_type, stop_label, _ in STOP_MODELS:
    if stop_type in ('atr_1x', 'atr_2x'):
        continue
    configs = [c for c in capped_configs if c['stop_type'] == stop_type]
    if not configs:
        continue
    print(f"\n  Stop: {stop_label} (capped at 2x ATR)")
    print_header()
    for target_mode in TARGET_MODES:
        match = [c for c in configs if c['target'] == target_mode]
        if match:
            print_results_table(match[0]['results'],
                                f"{stop_label} ≤2ATR | {target_mode}", months)
    print()


# ============================================================================
# PART 4: ENTRY MODEL COMPARISON (2x5min vs 3x5min vs 1x15min)
# ============================================================================
print(f"\n{'='*150}")
print(f"  PART 4 — ENTRY MODEL × STOP MODEL (best target per combo)")
print(f"  Testing acceptance periods with top structure stops")
print(f"{'='*150}")

acceptance_periods = [
    (5, 2, '2x5min'),
    (5, 3, '3x5min'),
    (15, 1, '1x15min'),
    (30, 1, '1x30min'),
]

top_stops = ['swing_3bar', 'fvg_15m', 'atr_1x', 'atr_2x']
entry_configs = []

for acc_period, acc_n, acc_label in acceptance_periods:
    for stop_type in top_stops:
        stop_label = [sl for st, sl, _ in STOP_MODELS if st == stop_type][0]
        for target_mode in ['2R', '3R', '4R']:
            results = []

            for s in session_db:
                ext = find_ib_extension_acceptance(
                    s['post_ib'], s['ib'], s['direction'], acc_period, acc_n)
                if ext is None:
                    continue

                entry_price = ext['entry_price']
                entry_bar_idx = ext['bar_end_1min']
                entry_bar = s['post_ib'].iloc[entry_bar_idx]

                stop_price = get_structure_stop(
                    stop_type, s, entry_price, entry_bar_idx, entry_bar, s['direction'])
                if stop_price is None:
                    continue

                risk_pts = abs(entry_price - stop_price)
                target_price = compute_target(
                    entry_price, s['direction'], target_mode,
                    s['ib'], s['va_width'], risk_pts)
                reward_pts = abs(target_price - entry_price)
                if reward_pts < 2:
                    continue

                r = replay_trade(s['post_ib'], entry_bar_idx, entry_price,
                                 s['direction'], stop_price, target_price)
                if r:
                    r['session_date'] = str(s['session_date'])
                    r['direction'] = s['direction']
                    r['entry_price'] = entry_price
                    results.append(r)

            if len(results) >= 3:
                entry_configs.append({
                    'label': f"{acc_label} | {stop_label} | {target_mode}",
                    'acc_label': acc_label,
                    'stop_type': stop_type,
                    'stop_label': stop_label,
                    'target': target_mode,
                    'results': results,
                })

for stop_type in top_stops:
    stop_label = [sl for st, sl, _ in STOP_MODELS if st == stop_type][0]
    print(f"\n  Stop: {stop_label}")
    print_header()
    for acc_period, acc_n, acc_label in acceptance_periods:
        for target_mode in ['2R', '3R', '4R']:
            match = [c for c in entry_configs if c['acc_label'] == acc_label
                     and c['stop_type'] == stop_type and c['target'] == target_mode]
            if match:
                print_results_table(match[0]['results'],
                                    f"{acc_label} | {target_mode}", months)
    print()


# ============================================================================
# PART 5: LONG vs SHORT BREAKDOWN (top configs)
# ============================================================================
print(f"\n{'='*150}")
print(f"  PART 5 — LONG vs SHORT BREAKDOWN")
print(f"{'='*150}")

# Rank all configs by $/month
all_ranked_configs = all_configs + capped_configs + entry_configs
ranked = []
for c in all_ranked_configs:
    df_r = pd.DataFrame(c['results'])
    n = len(df_r)
    if n < 5:
        continue
    wr = df_r['is_winner'].mean() * 100
    gw = df_r[df_r['is_winner']]['pnl_dollars'].sum()
    gl = abs(df_r[~df_r['is_winner']]['pnl_dollars'].sum())
    pf = gw / gl if gl > 0 else float('inf')
    pm = df_r['pnl_dollars'].sum() / months
    avg_risk = df_r['risk_pts'].mean()
    avg_rr = df_r['rr_ratio'].mean()
    ranked.append({
        'label': c['label'],
        'n': n, 'tpm': n / months, 'wr': wr, 'pf': pf, 'pm': pm,
        'risk': avg_risk, 'rr': avg_rr,
        'results': c['results'],
    })

ranked.sort(key=lambda x: x['pm'], reverse=True)

# Show top 15 with long/short breakdown
for rank, c in enumerate(ranked[:15], 1):
    df_r = pd.DataFrame(c['results'])
    print(f"\n  #{rank}: {c['label']}")
    print(f"       ALL: {c['n']} trades, {c['tpm']:.1f}/mo, {c['wr']:.1f}% WR, "
          f"PF {c['pf']:.2f}, ${c['pm']:,.0f}/mo, avg risk {c['risk']:.0f}p, "
          f"avg R:R {c['rr']:.1f}")

    for d in ['LONG', 'SHORT']:
        sub = df_r[df_r['direction'] == d]
        if len(sub) == 0:
            continue
        n_d = len(sub)
        wr_d = sub['is_winner'].mean() * 100
        gw_d = sub[sub['is_winner']]['pnl_dollars'].sum()
        gl_d = abs(sub[~sub['is_winner']]['pnl_dollars'].sum())
        pf_d = gw_d / gl_d if gl_d > 0 else float('inf')
        pm_d = sub['pnl_dollars'].sum() / months
        print(f"       {d}: {n_d} trades, {wr_d:.1f}% WR, PF {pf_d:.2f}, ${pm_d:,.0f}/mo")


# ============================================================================
# PART 6: HEAD-TO-HEAD RANKING — ALL CONFIGS
# ============================================================================
print(f"\n{'='*150}")
print(f"  PART 6 — HEAD-TO-HEAD: ALL CONFIGS RANKED BY $/MONTH")
print(f"{'='*150}")

print(f"\n  {'Rank':>4s} {'Config':<65s} {'N':>4s} {'T/Mo':>5s} {'WR':>6s} {'PF':>6s} "
      f"{'$/Mo':>9s} {'Risk':>5s} {'R:R':>4s}")
print(f"  {'-'*115}")

for rank, c in enumerate(ranked[:40], 1):
    marker = ' <--' if rank <= 5 else ''
    print(f"  {rank:>4d} {c['label']:<65s} {c['n']:>4d} {c['tpm']:>4.1f} "
          f"{c['wr']:>5.1f}% {c['pf']:>5.2f} ${c['pm']:>7,.0f} {c['risk']:>4.0f}p "
          f"{c['rr']:>4.1f}R{marker}")


# ============================================================================
# PART 7: COMPARISON TO v1 AND 80P
# ============================================================================
print(f"\n{'='*150}")
print(f"  PART 7 — COMPARISON: v2 (Structure Stops) vs v1 (IB Boundary) vs 80P")
print(f"{'='*150}")

# v1 baseline: 3x5min + IB boundary stop + 1x_ib target
v1_results = []
for s in session_db:
    ext = find_ib_extension_acceptance(s['post_ib'], s['ib'], s['direction'], 5, 3)
    if ext is None:
        continue
    entry_price = ext['entry_price']
    entry_bar_idx = ext['bar_end_1min']
    ibh, ibl = s['ib']['ibh'], s['ib']['ibl']
    stop_price = (ibl - 20) if s['direction'] == 'LONG' else (ibh + 20)
    risk_pts = abs(entry_price - stop_price)
    if risk_pts < 2:
        continue
    target_price = compute_target(entry_price, s['direction'], '1x_ib',
                                  s['ib'], s['va_width'], risk_pts)
    reward_pts = abs(target_price - entry_price)
    if reward_pts < 2:
        continue
    r = replay_trade(s['post_ib'], entry_bar_idx, entry_price,
                     s['direction'], stop_price, target_price)
    if r:
        r['direction'] = s['direction']
        v1_results.append(r)

print(f"\n  {'Strategy':<65s} {'N':>4s} {'T/Mo':>5s} {'WR':>6s} {'PF':>6s} "
      f"{'$/Mo':>9s} {'Risk':>5s} {'R:R':>4s}")
print(f"  {'-'*110}")

# v1 baseline
if v1_results:
    df_v1 = pd.DataFrame(v1_results)
    n = len(df_v1)
    wr = df_v1['is_winner'].mean() * 100
    gw = df_v1[df_v1['is_winner']]['pnl_dollars'].sum()
    gl = abs(df_v1[~df_v1['is_winner']]['pnl_dollars'].sum())
    pf = gw / gl if gl > 0 else float('inf')
    pm = df_v1['pnl_dollars'].sum() / months
    print(f"  {'v1: 3x5min + IB boundary + 1x_ib (OLD)':<65s} {n:>4d} {n/months:>4.1f} "
          f"{wr:>5.1f}% {pf:>5.2f} ${pm:>7,.0f} {df_v1['risk_pts'].mean():>4.0f}p "
          f"{df_v1['rr_ratio'].mean():>4.1f}R")

# Top 5 v2 configs
for rank, c in enumerate(ranked[:5], 1):
    print(f"  {'v2 #' + str(rank) + ': ' + c['label']:<65s} {c['n']:>4d} {c['tpm']:>4.1f} "
          f"{c['wr']:>5.1f}% {c['pf']:>5.2f} ${c['pm']:>7,.0f} {c['risk']:>4.0f}p "
          f"{c['rr']:>4.1f}R")

# 80P references
print(f"  {'80P Baseline (1x30min + 4R)':<65s} {'60':>4s} {'5.1':>5s} "
      f"{'38.3':>5s}% {'1.70':>5s} ${'955':>7s} {'60':>5s}p")
print(f"  {'80P Best (Limit 50% VA + 4R)':<65s} {'47':>4s} {'4.0':>5s} "
      f"{'44.7':>5s}% {'2.57':>5s} ${'1,922':>7s} {'60':>5s}p")


# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*150}")
print(f"  20P v2 STUDY — STRUCTURE-BASED STOPS SUMMARY")
print(f"{'='*150}")

print(f"""
  THESIS: v1 used IB-boundary stops (~200 pts risk) which killed the R:R.
          v2 uses structure-based stops (swing points, FVGs, ATR, VWAP, POC)
          which dramatically reduce risk per trade and improve R:R.

  STOP DISTANCE COMPARISON (median):""")

for key, label in [
    ('swing_3bar', 'Swing 3-bar'),
    ('fvg_15m', '15m FVG'),
    ('atr_1x', '1x ATR'),
    ('atr_2x', '2x ATR'),
    ('vwap', 'VWAP'),
    ('poc', 'POC'),
    ('va_boundary', 'VA boundary'),
    ('ib_boundary', 'IB boundary (v1)'),
]:
    dists = stop_distances[key]
    if dists:
        print(f"    {label:<20s}: {np.median(dists):>6.0f} pts median risk")

print(f"""
  TOP 5 CONFIGS (by $/month):""")

for rank, c in enumerate(ranked[:5], 1):
    print(f"    #{rank}: {c['label']}")
    print(f"        {c['n']} trades, {c['tpm']:.1f}/mo, {c['wr']:.1f}% WR, "
          f"PF {c['pf']:.2f}, ${c['pm']:,.0f}/mo, risk {c['risk']:.0f}p, R:R {c['rr']:.1f}")

print(f"""
  KEY QUESTION ANSWERED:
    Does using structure-based stops (like a real trader) make the 20P viable?

  IMPLEMENTATION NOTE:
    Stop hierarchy for live trading:
      1. Look for nearest swing low/high (3-bar pivot) before entry
      2. If no swing point, use 15-min FVG zone boundary
      3. If no FVG, use 1x ATR as fallback
      4. NEVER exceed 2x ATR as max risk — skip the trade if structure is too far
      5. Size position to risk 1-2% of account per trade
""")

print(f"{'='*150}")
print(f"  20P v2 STUDY COMPLETE")
print(f"{'='*150}")
