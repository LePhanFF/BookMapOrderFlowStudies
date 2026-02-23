"""
80P-CVD: VA Boundary Sweep + CVD Entry Model
=============================================
Codename: 80p-cvd

Skip the 30-min acceptance candle entirely. Instead:
  1. Price opens outside prior session VA (same setup condition)
  2. After IB, scan 1-min bars for price interacting with VA boundary
  3. Detect sweep pattern at the VA boundary (poke + pullback + re-entry)
  4. Measure CVD/delta at the sweep moment (tight 1-5 bar window)
  5. Enter on reversal back inside VA
  6. Stop beyond the sweep extreme

Two entry models tested:

  MODEL A — "Immediate + CVD":
    Price first touches VA boundary → check CVD over last N bars → enter
    No waiting. Fastest possible entry.

  MODEL B — "VA Boundary Sweep":
    Price enters VA → pulls back OUT → re-enters (double bottom/top at VA edge)
    Wait for the sweep, then enter. Slower but higher confirmation.

  MODEL C — "VA Boundary Sweep + CVD":
    Same as B, but only enter if CVD diverges at the sweep bar.
    Tightest filter = fewest trades but highest conviction.
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
from indicators.value_area import compute_session_value_areas, ValueAreaLevels

INSTRUMENT = get_instrument('MNQ')
TICK_VALUE = INSTRUMENT.tick_value / INSTRUMENT.tick_size
SLIPPAGE_PTS = 0.50
COMMISSION = 1.24
CONTRACTS = 5
MIN_VA_WIDTH = 25.0
ENTRY_CUTOFF = time(13, 0)

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
# FIND SESSIONS WITH OPEN OUTSIDE VA
# ============================================================================
def find_sessions_outside_va(df_rth, va_by_session):
    """Find all sessions where RTH open is outside prior session's VA."""
    candidates = []

    for i in range(1, len(sessions)):
        current = sessions[i]
        prior = sessions[i - 1]
        prior_key = str(prior)

        if prior_key not in va_by_session:
            continue
        prior_va = va_by_session[prior_key]
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

        if open_price < val:
            direction = 'LONG'
        elif open_price > vah:
            direction = 'SHORT'
        else:
            continue

        # Get post-IB bars (after first 60 min)
        post_ib = session_df.iloc[60:].reset_index(drop=True)
        if len(post_ib) < 60:
            continue

        # Compute prior session 1-min ATR
        prior_df = df_rth[df_rth['session_date'] == prior]
        prior_atr = (prior_df['high'] - prior_df['low']).mean() if len(prior_df) > 30 else 15.0

        candidates.append({
            'session_date': str(current),
            'direction': direction,
            'open_price': open_price,
            'vah': vah,
            'val': val,
            'poc': poc,
            'va_width': va_width,
            'prior_atr': prior_atr,
            'post_ib': post_ib,
        })

    return candidates


# ============================================================================
# MODEL A: IMMEDIATE ENTRY AT VA TOUCH + CVD FILTER
# ============================================================================
def model_a_immediate(candidate, cvd_window=5, require_cvd=False,
                       stop_buffer=10.0, target_mode='opposite_va'):
    """
    Enter on first 1-min bar that touches/crosses the VA boundary.
    Optionally filter by CVD over the last N bars.

    LONG: First bar where high >= VAL → enter at VAL (limit)
    SHORT: First bar where low <= VAH → enter at VAH (limit)
    """
    direction = candidate['direction']
    vah = candidate['vah']
    val = candidate['val']
    poc = candidate['poc']
    va_width = candidate['va_width']
    post_ib = candidate['post_ib']

    boundary = val if direction == 'LONG' else vah
    entry_bar = None
    entry_price = boundary  # limit at boundary

    # Running CVD from IB end
    cvd_running = 0.0

    for bar_idx in range(len(post_ib)):
        bar = post_ib.iloc[bar_idx]
        bt_time = bar.get('timestamp')
        bt = bt_time.time() if bt_time and hasattr(bt_time, 'time') else None
        if bt and bt >= ENTRY_CUTOFF:
            break

        # Update running CVD
        delta = bar.get('vol_delta', 0) or 0
        cvd_running += delta

        # Check if price touches the VA boundary
        if direction == 'LONG' and bar['high'] >= val:
            # CVD check: over the last N bars, is delta positive (buyers absorbing)?
            if require_cvd:
                start = max(0, bar_idx - cvd_window + 1)
                window_bars = post_ib.iloc[start:bar_idx + 1]
                window_delta = window_bars['vol_delta'].sum() if 'vol_delta' in window_bars.columns else 0
                if window_delta <= 0:  # Need positive delta for LONG
                    continue
            entry_bar = bar_idx
            break

        elif direction == 'SHORT' and bar['low'] <= vah:
            if require_cvd:
                start = max(0, bar_idx - cvd_window + 1)
                window_bars = post_ib.iloc[start:bar_idx + 1]
                window_delta = window_bars['vol_delta'].sum() if 'vol_delta' in window_bars.columns else 0
                if window_delta >= 0:  # Need negative delta for SHORT
                    continue
            entry_bar = bar_idx
            break

    if entry_bar is None:
        return None

    # Stop
    if direction == 'LONG':
        stop_price = val - stop_buffer
    else:
        stop_price = vah + stop_buffer

    return _replay_trade(candidate, entry_bar, entry_price, stop_price, target_mode,
                         extra={'model': 'A_immediate', 'cvd_filtered': require_cvd,
                                'cvd_at_entry': cvd_running})


# ============================================================================
# MODEL B: VA BOUNDARY SWEEP (DOUBLE TOP/BOTTOM AT VA EDGE)
# ============================================================================
def model_b_sweep(candidate, min_sweep_pts=0.0, max_sweep_pts=50.0,
                   stop_buffer=5.0, target_mode='opposite_va',
                   require_cvd=False, cvd_window=5):
    """
    1. Price enters VA (crosses boundary)
    2. Price pulls back OUT of VA (the sweep)
    3. Price re-enters VA (double bottom/top confirmed)
    4. Enter on re-entry

    LONG: high >= VAL (enter VA) → low < VAL (pullback sweep) → high >= VAL (re-entry)
    SHORT: low <= VAH (enter VA) → high > VAH (pullback bounce) → low <= VAH (re-entry)

    min_sweep_pts: Minimum overshoot beyond boundary to count as sweep
    stop_buffer: Buffer beyond the sweep extreme for stop
    """
    direction = candidate['direction']
    vah = candidate['vah']
    val = candidate['val']
    poc = candidate['poc']
    va_width = candidate['va_width']
    post_ib = candidate['post_ib']

    boundary = val if direction == 'LONG' else vah

    # State machine
    state = 'WAITING_TOUCH'  # → INSIDE_VA → PULLED_BACK → RE_ENTERED
    first_touch_bar = None
    pullback_extreme = None  # Low for LONG (how far below VAL), High for SHORT (above VAH)
    entry_bar = None
    entry_price = None

    cvd_at_sweep = 0.0
    cvd_running = 0.0

    for bar_idx in range(len(post_ib)):
        bar = post_ib.iloc[bar_idx]
        bt_time = bar.get('timestamp')
        bt = bt_time.time() if bt_time and hasattr(bt_time, 'time') else None
        if bt and bt >= ENTRY_CUTOFF:
            break

        delta = bar.get('vol_delta', 0) or 0
        cvd_running += delta

        if state == 'WAITING_TOUCH':
            # Wait for price to touch/enter VA
            if direction == 'LONG' and bar['high'] >= val:
                state = 'INSIDE_VA'
                first_touch_bar = bar_idx
            elif direction == 'SHORT' and bar['low'] <= vah:
                state = 'INSIDE_VA'
                first_touch_bar = bar_idx

        elif state == 'INSIDE_VA':
            # Wait for price to pull back OUT of VA
            if direction == 'LONG' and bar['low'] < val:
                state = 'PULLED_BACK'
                pullback_extreme = bar['low']
            elif direction == 'SHORT' and bar['high'] > vah:
                state = 'PULLED_BACK'
                pullback_extreme = bar['high']
            # If price just stays inside VA for too long without pullback, keep waiting
            # Allow some bars inside before requiring pullback
            elif bar_idx - first_touch_bar > 120:  # 2 hours max wait
                break

        elif state == 'PULLED_BACK':
            # Track the sweep extreme
            if direction == 'LONG':
                pullback_extreme = min(pullback_extreme, bar['low'])
                sweep_distance = val - pullback_extreme
            else:
                pullback_extreme = max(pullback_extreme, bar['high'])
                sweep_distance = pullback_extreme - vah

            # Check if sweep is deep enough
            if sweep_distance < min_sweep_pts:
                # Still need more sweep — but check if price re-enters VA
                if direction == 'LONG' and bar['high'] >= val:
                    # Re-entered but sweep wasn't deep enough — skip
                    if sweep_distance < min_sweep_pts:
                        # Reset: this wasn't a real sweep
                        state = 'INSIDE_VA'
                        first_touch_bar = bar_idx
                        pullback_extreme = None
                        continue
                elif direction == 'SHORT' and bar['low'] <= vah:
                    if sweep_distance < min_sweep_pts:
                        state = 'INSIDE_VA'
                        first_touch_bar = bar_idx
                        pullback_extreme = None
                        continue

            # Check sweep isn't too deep (would indicate real breakdown)
            if sweep_distance > max_sweep_pts:
                break  # Too much — this is a real breakdown, not a sweep

            # Wait for re-entry into VA (the confirmation)
            if direction == 'LONG' and bar['close'] >= val:
                cvd_at_sweep = cvd_running
                # CVD filter: at the sweep moment, check recent delta
                if require_cvd:
                    start = max(0, bar_idx - cvd_window + 1)
                    window_bars = post_ib.iloc[start:bar_idx + 1]
                    window_delta = window_bars['vol_delta'].sum() if 'vol_delta' in window_bars.columns else 0
                    if window_delta <= 0:
                        # No confirming delta at sweep — but keep scanning
                        # Maybe next re-entry will have better delta
                        continue
                entry_bar = bar_idx
                entry_price = val  # enter at boundary
                break

            elif direction == 'SHORT' and bar['close'] <= vah:
                cvd_at_sweep = cvd_running
                if require_cvd:
                    start = max(0, bar_idx - cvd_window + 1)
                    window_bars = post_ib.iloc[start:bar_idx + 1]
                    window_delta = window_bars['vol_delta'].sum() if 'vol_delta' in window_bars.columns else 0
                    if window_delta >= 0:
                        continue
                entry_bar = bar_idx
                entry_price = vah  # enter at boundary
                break

    if entry_bar is None:
        return None

    # Stop beyond the sweep extreme
    if direction == 'LONG':
        stop_price = pullback_extreme - stop_buffer
    else:
        stop_price = pullback_extreme + stop_buffer

    risk_pts = abs(entry_price - stop_price)
    if risk_pts <= 0 or risk_pts > va_width:  # sanity check
        return None

    # Capture sweep stats
    sweep_distance = (val - pullback_extreme) if direction == 'LONG' else (pullback_extreme - vah)
    bars_to_sweep = entry_bar - first_touch_bar if first_touch_bar is not None else 0

    # Delta at the exact sweep bar (the re-entry bar)
    sweep_bar_delta = post_ib.iloc[entry_bar].get('vol_delta', 0) or 0

    # Delta over the pullback phase (from first touch to re-entry)
    if first_touch_bar is not None:
        pullback_bars = post_ib.iloc[first_touch_bar:entry_bar + 1]
        pullback_delta = pullback_bars['vol_delta'].sum() if 'vol_delta' in pullback_bars.columns else 0
        pullback_volume = pullback_bars['volume'].sum()
    else:
        pullback_delta = 0
        pullback_volume = 0

    return _replay_trade(candidate, entry_bar, entry_price, stop_price, target_mode,
                         extra={
                             'model': 'B_sweep',
                             'cvd_filtered': require_cvd,
                             'sweep_pts': sweep_distance,
                             'bars_to_sweep': bars_to_sweep,
                             'cvd_at_sweep': cvd_at_sweep,
                             'sweep_bar_delta': sweep_bar_delta,
                             'pullback_delta': pullback_delta,
                             'pullback_volume': pullback_volume,
                             'stop_is_sweep': True,
                         })


# ============================================================================
# SHARED TRADE REPLAY
# ============================================================================
def _replay_trade(candidate, entry_bar, entry_price, stop_price, target_mode, extra=None):
    """Replay from entry bar to exit."""
    direction = candidate['direction']
    vah = candidate['vah']
    val = candidate['val']
    poc = candidate['poc']
    va_width = candidate['va_width']
    post_ib = candidate['post_ib']

    risk_pts = abs(entry_price - stop_price)
    if risk_pts <= 0:
        return None

    # Target
    if target_mode == 'opposite_va':
        target = vah if direction == 'LONG' else val
    elif target_mode == 'poc':
        target = poc
    elif target_mode.endswith('R'):
        r_mult = float(target_mode[:-1])
        target = entry_price + risk_pts * r_mult if direction == 'LONG' else entry_price - risk_pts * r_mult
    else:
        target = vah if direction == 'LONG' else val

    reward = (target - entry_price) if direction == 'LONG' else (entry_price - target)
    if reward <= 0:
        return None

    # Replay
    remaining = post_ib.iloc[entry_bar:]
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
        if bt and bt >= time(15, 30):
            exit_price = bar['close']
            exit_reason = 'EOD'
            break

        fav = (bar['high'] - entry_price) if direction == 'LONG' else (entry_price - bar['low'])
        adv = (entry_price - bar['low']) if direction == 'LONG' else (bar['high'] - entry_price)
        mfe = max(mfe, fav)
        mae = max(mae, adv)

        # Stop
        if direction == 'LONG' and bar['low'] <= stop_price:
            exit_price = stop_price
            exit_reason = 'STOP'
            break
        elif direction == 'SHORT' and bar['high'] >= stop_price:
            exit_price = stop_price
            exit_reason = 'STOP'
            break

        # Target
        if direction == 'LONG' and bar['high'] >= target:
            exit_price = target
            exit_reason = 'TARGET'
            break
        elif direction == 'SHORT' and bar['low'] <= target:
            exit_price = target
            exit_reason = 'TARGET'
            break

    if exit_price is None:
        exit_price = remaining.iloc[-1]['close']
        exit_reason = 'EOD'

    pnl_pts = (exit_price - entry_price - SLIPPAGE_PTS) if direction == 'LONG' else (entry_price - exit_price - SLIPPAGE_PTS)
    pnl_dollars = pnl_pts * TICK_VALUE * CONTRACTS - COMMISSION * CONTRACTS

    result = {
        'session_date': candidate['session_date'],
        'direction': direction,
        'entry_price': entry_price,
        'stop_price': stop_price,
        'target': target,
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'pnl_pts': pnl_pts,
        'pnl_dollars': pnl_dollars,
        'risk_pts': risk_pts,
        'reward_pts': reward,
        'rr_ratio': reward / risk_pts if risk_pts > 0 else 0,
        'mfe_pts': mfe,
        'mae_pts': mae,
        'bars_held': bars_held,
        'is_winner': pnl_dollars > 0,
        'entry_bar': entry_bar,
        'va_width': va_width,
    }
    if extra:
        result.update(extra)
    return result


def print_stats(results, label, months):
    """Print standard stats for a set of trade results."""
    if not results:
        print(f"    {label:<45s}    0 trades")
        return
    df = pd.DataFrame(results)
    n = len(df)
    wr = df['is_winner'].mean() * 100
    gw = df[df['is_winner']]['pnl_dollars'].sum()
    gl = abs(df[~df['is_winner']]['pnl_dollars'].sum())
    pf = gw / gl if gl > 0 else float('inf')
    pm = df['pnl_dollars'].sum() / months
    avg_w = df[df['is_winner']]['pnl_dollars'].mean() if df['is_winner'].any() else 0
    avg_l = df[~df['is_winner']]['pnl_dollars'].mean() if (~df['is_winner']).any() else 0
    stopped = (df['exit_reason'] == 'STOP').sum()
    target_hit = (df['exit_reason'] == 'TARGET').sum()
    eod = (df['exit_reason'] == 'EOD').sum()
    avg_risk = df['risk_pts'].mean()
    avg_rr = df['rr_ratio'].mean()

    print(f"    {label:<45s} {n:>4d} {n/months:>5.1f}/mo "
          f"{wr:>5.1f}% {pf:>5.2f} ${pm:>7,.0f} "
          f"${avg_w:>6,.0f} ${avg_l:>6,.0f} "
          f"{avg_risk:>4.0f}p {avg_rr:>4.1f}R "
          f"S={stopped} T={target_hit} E={eod}")


# ============================================================================
# MAIN
# ============================================================================
print("\nFinding sessions with open outside VA...")
candidates = find_sessions_outside_va(df_rth, eth_va)
print(f"Found {len(candidates)} sessions (open outside prior ETH VA)")


# ============================================================================
# MODEL A: IMMEDIATE ENTRY AT VA TOUCH
# ============================================================================
print(f"\n{'='*120}")
print(f"  MODEL A — IMMEDIATE ENTRY AT VA TOUCH")
print(f"  Enter the moment price first touches VAL (LONG) or VAH (SHORT)")
print(f"  Stop: VA boundary ± buffer. No waiting for candles or retests.")
print(f"{'='*120}")

header = (f"    {'Config':<45s} {'N':>4s} {'Freq':>6s} "
          f"{'WR':>6s} {'PF':>6s} {'$/Mo':>8s} "
          f"{'AvgWin':>8s} {'AvgLoss':>8s} "
          f"{'Risk':>5s} {'R:R':>5s} {'Exits':>12s}")

for target_mode in ['opposite_va', '2R', '4R']:
    print(f"\n  ━━ Target: {target_mode} ━━")
    print(header)
    print(f"    {'-'*120}")

    for stop_buf in [10, 15, 20]:
        # Without CVD
        results = [r for r in [model_a_immediate(c, stop_buffer=stop_buf, target_mode=target_mode)
                                for c in candidates] if r]
        print_stats(results, f"VA touch, stop={stop_buf}pt", months)

        # With CVD filter (various windows)
        for cvd_win in [3, 5, 10]:
            results = [r for r in [model_a_immediate(c, cvd_window=cvd_win, require_cvd=True,
                                                      stop_buffer=stop_buf, target_mode=target_mode)
                                    for c in candidates] if r]
            print_stats(results, f"VA touch + CVD({cvd_win}bar), stop={stop_buf}pt", months)


# ============================================================================
# MODEL B: VA BOUNDARY SWEEP (DOUBLE TOP/BOTTOM)
# ============================================================================
print(f"\n{'='*120}")
print(f"  MODEL B — VA BOUNDARY SWEEP (DOUBLE TOP/BOTTOM AT VA EDGE)")
print(f"  Price enters VA → pulls back out → re-enters = sweep confirmed")
print(f"  Stop: below/above the sweep extreme + buffer")
print(f"{'='*120}")

for target_mode in ['opposite_va', '2R', '4R']:
    print(f"\n  ━━ Target: {target_mode} ━━")
    print(header)
    print(f"    {'-'*120}")

    for stop_buf in [5, 10]:
        for min_sweep in [0, 2, 5, 10]:
            # Without CVD
            results = [r for r in [model_b_sweep(c, min_sweep_pts=min_sweep,
                                                  stop_buffer=stop_buf,
                                                  target_mode=target_mode)
                                    for c in candidates] if r]
            print_stats(results, f"Sweep>={min_sweep}pt, stop=sweep+{stop_buf}pt", months)

    # With CVD (best stop from above)
    print(f"\n    --- With CVD filter ---")
    for stop_buf in [5, 10]:
        for min_sweep in [0, 5]:
            for cvd_win in [3, 5]:
                results = [r for r in [model_b_sweep(c, min_sweep_pts=min_sweep,
                                                      stop_buffer=stop_buf,
                                                      target_mode=target_mode,
                                                      require_cvd=True, cvd_window=cvd_win)
                                        for c in candidates] if r]
                print_stats(results, f"Sweep>={min_sweep}pt + CVD({cvd_win}), stop=sw+{stop_buf}pt", months)


# ============================================================================
# SWEEP QUALITY ANALYSIS
# ============================================================================
print(f"\n{'='*120}")
print(f"  SWEEP QUALITY — What makes a good sweep?")
print(f"{'='*120}")

# Get all sweep trades (opposite_va target, stop=sweep+5pt)
all_sweep = [r for r in [model_b_sweep(c, min_sweep_pts=0, stop_buffer=5,
                                        target_mode='opposite_va')
                          for c in candidates] if r]

if all_sweep:
    df_sw = pd.DataFrame(all_sweep)
    print(f"\n  {len(df_sw)} sweep trades found")

    # Sweep distance analysis
    print(f"\n  Sweep distance (how far price goes beyond VA boundary):")
    print(f"    Mean:   {df_sw['sweep_pts'].mean():.1f} pts")
    print(f"    Median: {df_sw['sweep_pts'].median():.1f} pts")
    for pct in [25, 50, 75, 90]:
        print(f"    {pct}th pctile: {np.percentile(df_sw['sweep_pts'], pct):.1f} pts")

    # Time to sweep
    print(f"\n  Bars from first VA touch to sweep completion:")
    print(f"    Mean:   {df_sw['bars_to_sweep'].mean():.0f} bars")
    print(f"    Median: {df_sw['bars_to_sweep'].median():.0f} bars")

    # Segment by sweep depth
    print(f"\n  Performance by sweep depth:")
    sweep_bins = [(0, 5, '0-5pt'), (5, 10, '5-10pt'), (10, 20, '10-20pt'), (20, 100, '20+pt')]
    print(f"    {'Sweep':>10s} {'N':>4s} {'WR':>6s} {'PF':>6s} {'$/Mo':>9s} {'Risk':>5s}")
    print(f"    {'-'*50}")
    for low, high, label in sweep_bins:
        subset = df_sw[(df_sw['sweep_pts'] >= low) & (df_sw['sweep_pts'] < high)]
        if len(subset) < 2:
            print(f"    {label:>10s} {len(subset):>4d}  (too few)")
            continue
        wr = subset['is_winner'].mean() * 100
        gw = subset[subset['is_winner']]['pnl_dollars'].sum()
        gl = abs(subset[~subset['is_winner']]['pnl_dollars'].sum())
        pf = gw / gl if gl > 0 else float('inf')
        pm = subset['pnl_dollars'].sum() / months
        avg_risk = subset['risk_pts'].mean()
        print(f"    {label:>10s} {len(subset):>4d} {wr:>5.1f}% {pf:>5.2f} ${pm:>7,.0f} {avg_risk:>4.0f}p")

    # Delta at the sweep bar
    print(f"\n  Delta at the sweep bar (re-entry bar):")
    print(f"    Mean:   {df_sw['sweep_bar_delta'].mean():.0f}")
    print(f"    Median: {df_sw['sweep_bar_delta'].median():.0f}")

    # Confirming vs non-confirming delta at sweep
    df_sw['delta_confirms'] = df_sw.apply(
        lambda r: (r['direction'] == 'LONG' and r['sweep_bar_delta'] > 0) or
                  (r['direction'] == 'SHORT' and r['sweep_bar_delta'] < 0), axis=1)

    conf = df_sw[df_sw['delta_confirms'] == True]
    anti = df_sw[df_sw['delta_confirms'] == False]

    print(f"\n    {'Delta at sweep':>20s} {'N':>4s} {'WR':>6s} {'PF':>6s} {'$/Mo':>9s}")
    print(f"    {'-'*55}")
    for label, subset in [('Confirming', conf), ('Against', anti)]:
        if len(subset) < 2:
            print(f"    {label:>20s} {len(subset):>4d}  (too few)")
            continue
        wr = subset['is_winner'].mean() * 100
        gw = subset[subset['is_winner']]['pnl_dollars'].sum()
        gl = abs(subset[~subset['is_winner']]['pnl_dollars'].sum())
        pf = gw / gl if gl > 0 else float('inf')
        pm = subset['pnl_dollars'].sum() / months
        print(f"    {label:>20s} {len(subset):>4d} {wr:>5.1f}% {pf:>5.2f} ${pm:>7,.0f}")

    # Pullback delta (total delta during the sweep phase)
    print(f"\n  Pullback delta (total delta from VA touch to sweep completion):")
    print(f"    Mean:   {df_sw['pullback_delta'].mean():.0f}")

    df_sw['pullback_diverges'] = df_sw.apply(
        lambda r: (r['direction'] == 'LONG' and r['pullback_delta'] > 0) or
                  (r['direction'] == 'SHORT' and r['pullback_delta'] < 0), axis=1)

    div = df_sw[df_sw['pullback_diverges'] == True]
    nodiv = df_sw[df_sw['pullback_diverges'] == False]

    print(f"\n    {'Pullback CVD':>25s} {'N':>4s} {'WR':>6s} {'PF':>6s} {'$/Mo':>9s}")
    print(f"    {'-'*60}")
    for label, subset in [('Diverging (good)', div), ('Confirming (bad)', nodiv)]:
        if len(subset) < 2:
            print(f"    {label:>25s} {len(subset):>4d}  (too few)")
            continue
        wr = subset['is_winner'].mean() * 100
        gw = subset[subset['is_winner']]['pnl_dollars'].sum()
        gl = abs(subset[~subset['is_winner']]['pnl_dollars'].sum())
        pf = gw / gl if gl > 0 else float('inf')
        pm = subset['pnl_dollars'].sum() / months
        print(f"    {label:>25s} {len(subset):>4d} {wr:>5.1f}% {pf:>5.2f} ${pm:>7,.0f}")


# ============================================================================
# HEAD-TO-HEAD: ALL MODELS COMPARED
# ============================================================================
print(f"\n{'='*120}")
print(f"  HEAD-TO-HEAD COMPARISON — All Entry Models")
print(f"{'='*120}")

comparisons = []

for target_mode in ['opposite_va', '2R', '4R']:
    print(f"\n  ━━ Target: {target_mode} ━━")
    print(f"    {'Model':<55s} {'N':>4s} {'Freq':>6s} "
          f"{'WR':>6s} {'PF':>6s} {'$/Mo':>8s} {'Risk':>5s}")
    print(f"    {'-'*100}")

    configs = []

    # Model A: Immediate
    r = [x for x in [model_a_immediate(c, stop_buffer=10, target_mode=target_mode)
                      for c in candidates] if x]
    configs.append(('A: Immediate (no filter)', r))

    r = [x for x in [model_a_immediate(c, stop_buffer=10, target_mode=target_mode,
                                         require_cvd=True, cvd_window=5)
                      for c in candidates] if x]
    configs.append(('A: Immediate + CVD(5bar)', r))

    # Model B: Sweep
    r = [x for x in [model_b_sweep(c, min_sweep_pts=0, stop_buffer=5,
                                     target_mode=target_mode)
                      for c in candidates] if x]
    configs.append(('B: Sweep>=0pt, stop=sweep+5pt', r))

    r = [x for x in [model_b_sweep(c, min_sweep_pts=5, stop_buffer=5,
                                     target_mode=target_mode)
                      for c in candidates] if x]
    configs.append(('B: Sweep>=5pt, stop=sweep+5pt', r))

    r = [x for x in [model_b_sweep(c, min_sweep_pts=0, stop_buffer=10,
                                     target_mode=target_mode)
                      for c in candidates] if x]
    configs.append(('B: Sweep>=0pt, stop=sweep+10pt', r))

    # Model C: Sweep + CVD
    r = [x for x in [model_b_sweep(c, min_sweep_pts=0, stop_buffer=5,
                                     target_mode=target_mode,
                                     require_cvd=True, cvd_window=5)
                      for c in candidates] if x]
    configs.append(('C: Sweep + CVD(5bar), stop=sw+5pt', r))

    r = [x for x in [model_b_sweep(c, min_sweep_pts=0, stop_buffer=5,
                                     target_mode=target_mode,
                                     require_cvd=True, cvd_window=3)
                      for c in candidates] if x]
    configs.append(('C: Sweep + CVD(3bar), stop=sw+5pt', r))

    for label, results in configs:
        if not results:
            print(f"    {label:<55s}    0 trades")
            continue
        df_c = pd.DataFrame(results)
        n = len(df_c)
        wr = df_c['is_winner'].mean() * 100
        gw = df_c[df_c['is_winner']]['pnl_dollars'].sum()
        gl = abs(df_c[~df_c['is_winner']]['pnl_dollars'].sum())
        pf = gw / gl if gl > 0 else float('inf')
        pm = df_c['pnl_dollars'].sum() / months
        avg_risk = df_c['risk_pts'].mean()
        print(f"    {label:<55s} {n:>4d} {n/months:>5.1f}/mo "
              f"{wr:>5.1f}% {pf:>5.2f} ${pm:>7,.0f} {avg_risk:>4.0f}p")

        comparisons.append({
            'model': label, 'target': target_mode,
            'n': n, 'wr': wr, 'pf': pf, 'pm': pm, 'risk': avg_risk,
        })


# ============================================================================
# TRADE LOG — Best sweep model
# ============================================================================
print(f"\n{'='*120}")
print(f"  TRADE LOG — Model B: Sweep>=0pt, stop=sweep+5pt, opposite_va target")
print(f"{'='*120}")

best_results = [r for r in [model_b_sweep(c, min_sweep_pts=0, stop_buffer=5,
                                           target_mode='opposite_va')
                             for c in candidates] if r]

if best_results:
    df_best = pd.DataFrame(best_results)
    print(f"\n  {'Date':12s} {'Dir':5s} {'Entry':>8s} {'Stop':>8s} {'Target':>8s} "
          f"{'Exit':>8s} {'Rsn':5s} {'P&L':>8s} {'Risk':>5s} {'Sweep':>5s} "
          f"{'Bars':>4s} {'SwDlt':>6s} {'PBDlt':>7s}")
    print(f"  {'-'*115}")

    for _, t in df_best.sort_values('session_date').iterrows():
        sweep_label = f"{t.get('sweep_pts', 0):.0f}p"
        swdlt = f"{t.get('sweep_bar_delta', 0):.0f}"
        pbdlt = f"{t.get('pullback_delta', 0):.0f}"
        bars = f"{t.get('bars_to_sweep', 0):.0f}"
        print(f"  {t['session_date']:12s} {t['direction']:5s} "
              f"{t['entry_price']:>8.1f} {t['stop_price']:>8.1f} {t['target']:>8.1f} "
              f"{t['exit_price']:>8.1f} {t['exit_reason']:5s} "
              f"${t['pnl_dollars']:>7,.0f} {t['risk_pts']:>4.0f}p {sweep_label:>5s} "
              f"{bars:>4s} {swdlt:>6s} {pbdlt:>7s}")


# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*120}")
print(f"  SUMMARY — 80P-CVD ANALYSIS")
print(f"{'='*120}")

if comparisons:
    df_comp = pd.DataFrame(comparisons)
    best_pm = df_comp.loc[df_comp['pm'].idxmax()]
    best_wr = df_comp.loc[df_comp['wr'].idxmax()]
    best_pf = df_comp[df_comp['n'] >= 10].loc[df_comp[df_comp['n'] >= 10]['pf'].idxmax()] if len(df_comp[df_comp['n'] >= 10]) > 0 else None

    print(f"""
  BEST $/MONTH:  {best_pm['model']} ({best_pm['target']})
                 {best_pm['n']:.0f} trades, {best_pm['wr']:.1f}% WR, PF {best_pm['pf']:.2f}, ${best_pm['pm']:,.0f}/mo

  BEST WIN RATE: {best_wr['model']} ({best_wr['target']})
                 {best_wr['n']:.0f} trades, {best_wr['wr']:.1f}% WR, PF {best_wr['pf']:.2f}, ${best_wr['pm']:,.0f}/mo
""")
    if best_pf is not None:
        print(f"  BEST PF (≥10 trades): {best_pf['model']} ({best_pf['target']})")
        print(f"                 {best_pf['n']:.0f} trades, {best_pf['wr']:.1f}% WR, PF {best_pf['pf']:.2f}, ${best_pf['pm']:,.0f}/mo")

print(f"""
  KEY QUESTIONS ANSWERED:

  1. SKIP THE 30-MIN CANDLE?
     Model A (immediate entry at VA touch) answers this directly.
     Compare its $/mo vs the 30-min acceptance model ($955/mo with 4R).

  2. CVD AT THE BOUNDARY?
     CVD filter on Model A shows whether delta confirmation at VA touch adds edge.

  3. SWEEP / DOUBLE TOP-BOTTOM AT VA EDGE?
     Model B captures the exact pattern: VA touch → pullback → re-entry.
     Stop is at the sweep extreme — the most logical stop level.

  4. SWEEP + CVD?
     Model C combines the sweep with tight CVD confirmation.

  COMPARE TO PRIOR MODELS:
     Current (acceptance + 4R):      60 trades, 38.3% WR, PF 1.70, $955/mo
     100% retest (prev analysis):    35 trades, 62.9% WR, PF 3.25, $865/mo
     Limit 50% VA + 4R:             47 trades, 44.7% WR, PF 2.57, $1,922/mo
""")

print(f"{'='*120}")
print(f"  80P-CVD ANALYSIS COMPLETE")
print(f"{'='*120}")
