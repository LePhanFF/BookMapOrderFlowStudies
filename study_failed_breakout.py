"""
IBH Failed Breakout / Liquidity Sweep Study

NQ PATTERN: Price pokes above IBH to claim liquidity (stop runs), then FAILS
to hold, closes back below IBH, and reverts to the mean (VWAP/POC/IB mid).

This is fundamentally different from "fading at IBH" (which we tested before):
  - IBH Fade: SHORT before the break happens → bad WR on NQ (long bias rips through)
  - Failed Breakout: Let it BREAK above IBH, wait for it to FAIL, THEN trade

Study phases:
  1. How often does NQ break above IBH? (frequency)
  2. Of those breaks, how many FAIL vs EXTEND? (classification)
  3. What does "fail" look like? (time above, max extension, delta profile)
  4. After failure, where does price go? (VWAP, POC, IB mid, IBL, VAL)
  5. Can we trade this as a SHORT entry after confirmed failure?
  6. Can we use this as a FILTER (don't go long after failed breakout)?
  7. Test across day types (b_day, neutral, p_day, trend)
"""

import sys
from pathlib import Path
from collections import defaultdict
from datetime import time as _time
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
from strategy.day_type import classify_trend_strength, classify_day_type, DayType
from engine.execution import ExecutionModel

# Load data
instrument = get_instrument('MNQ')
df = load_csv('NQ')
df = filter_rth(df)
df = compute_all_features(df)

if 'session_date' not in df.columns:
    df['session_date'] = df['timestamp'].dt.date

sessions = sorted(df['session_date'].unique())
execution = ExecutionModel(instrument, slippage_ticks=1)

print("=" * 130)
print("  IBH FAILED BREAKOUT / LIQUIDITY SWEEP STUDY")
print(f"  Data: {len(sessions)} sessions (Nov 18, 2025 - Feb 16, 2026)")
print("  Pattern: Price breaks above IBH → fails to hold → reverts to mean")
print("=" * 130)


# ============================================================================
# PHASE 1: CLASSIFY EVERY SESSION'S IBH BREAKOUT BEHAVIOR
# ============================================================================
print("\n" + "=" * 130)
print("  PHASE 1: IBH BREAKOUT CLASSIFICATION (every session)")
print("=" * 130)

session_data = []

for session_date in sessions:
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

    # Track every bar's relationship to IBH
    broke_above = False
    first_break_bar = None
    first_break_time = None
    max_extension_pts = 0          # Max pts above IBH
    max_extension_bars_above = 0   # Consecutive bars with close > IBH
    current_bars_above = 0
    total_bars_above = 0           # Total bars that closed above IBH
    bars_above_streak_max = 0
    failed_back_below = False
    fail_bar = None
    fail_time = None
    fail_price = None

    # Delta/volume during the breakout
    breakout_deltas = []           # Delta values while above IBH
    breakout_volumes = []

    # After-failure tracking
    post_fail_min_price = None     # How far did it go after failing?
    post_fail_vwap_hit = False
    post_fail_ib_mid_hit = False
    post_fail_ibl_hit = False

    # Session end data
    session_close = post_ib.iloc[-1]['close']
    vwap_end = post_ib.iloc[-1].get('vwap', ib_mid)
    if pd.isna(vwap_end):
        vwap_end = ib_mid

    for i, (idx, bar) in enumerate(post_ib.iterrows()):
        price = bar['close']
        bar_time = bar['timestamp'].time() if 'timestamp' in bar.index else None

        # Track break above IBH
        if price > ib_high and not broke_above:
            broke_above = True
            first_break_bar = i
            first_break_time = bar_time

        if broke_above and not failed_back_below:
            if price > ib_high:
                current_bars_above += 1
                total_bars_above += 1
                bars_above_streak_max = max(bars_above_streak_max, current_bars_above)
                ext_pts = bar['high'] - ib_high
                if ext_pts > max_extension_pts:
                    max_extension_pts = ext_pts

                delta = bar.get('delta', 0)
                if not pd.isna(delta):
                    breakout_deltas.append(delta)
                vol = bar.get('volume', 0)
                if not pd.isna(vol):
                    breakout_volumes.append(vol)
            else:
                # Price closed back below IBH
                if current_bars_above > 0:
                    # This is a potential failure — price was above, now below
                    failed_back_below = True
                    fail_bar = i
                    fail_time = bar_time
                    fail_price = price
                    post_fail_min_price = price
                current_bars_above = 0

        # Track post-failure behavior
        if failed_back_below:
            if post_fail_min_price is None or bar['low'] < post_fail_min_price:
                post_fail_min_price = bar['low']

            vwap_now = bar.get('vwap', ib_mid)
            if not pd.isna(vwap_now) and bar['low'] <= vwap_now:
                post_fail_vwap_hit = True
            if bar['low'] <= ib_mid:
                post_fail_ib_mid_hit = True
            if bar['low'] <= ib_low:
                post_fail_ibl_hit = True

    # Classify this session
    if not broke_above:
        classification = 'NO_BREAK'
    elif broke_above and not failed_back_below:
        # Never came back below IBH
        ext_mult = max_extension_pts / ib_range if ib_range > 0 else 0
        if ext_mult >= 1.0:
            classification = 'STRONG_EXTENSION'
        else:
            classification = 'HELD_ABOVE'
    else:
        # Broke above then failed back below
        ext_mult = max_extension_pts / ib_range if ib_range > 0 else 0
        if ext_mult < 0.3:
            classification = 'QUICK_FAKE'  # Barely poked above
        elif ext_mult < 0.7:
            classification = 'MODERATE_SWEEP'  # Decent poke then fail
        else:
            classification = 'DEEP_SWEEP_FAIL'  # Extended far then failed

    # Final day type at close
    if session_close > ib_high:
        ib_dir = 'BULL'
        ext = (session_close - ib_mid) / ib_range
    elif session_close < ib_low:
        ib_dir = 'BEAR'
        ext = (ib_mid - session_close) / ib_range
    else:
        ib_dir = 'INSIDE'
        ext = 0.0

    strength = classify_trend_strength(ext)
    day_type = classify_day_type(ib_high, ib_low, session_close, ib_dir, strength)
    dt_val = day_type.value if hasattr(day_type, 'value') else str(day_type)

    # Breakout delta profile
    total_breakout_delta = sum(breakout_deltas) if breakout_deltas else 0
    avg_breakout_delta = np.mean(breakout_deltas) if breakout_deltas else 0

    record = {
        'date': str(session_date),
        'ib_range': ib_range,
        'ib_high': ib_high,
        'ib_low': ib_low,
        'ib_mid': ib_mid,
        'classification': classification,
        'broke_above': broke_above,
        'failed': failed_back_below,
        'max_ext_pts': max_extension_pts,
        'max_ext_mult': max_extension_pts / ib_range if ib_range > 0 else 0,
        'total_bars_above': total_bars_above,
        'max_streak_above': bars_above_streak_max,
        'break_bar': first_break_bar,
        'break_time': first_break_time,
        'fail_bar': fail_bar,
        'fail_time': fail_time,
        'fail_price': fail_price,
        'breakout_delta_total': total_breakout_delta,
        'breakout_delta_avg': avg_breakout_delta,
        'session_close': session_close,
        'close_vs_ibh': session_close - ib_high,
        'close_vs_ib_mid': session_close - ib_mid,
        'close_vs_vwap': session_close - vwap_end,
        'post_fail_min': post_fail_min_price,
        'post_fail_vwap_hit': post_fail_vwap_hit,
        'post_fail_ib_mid_hit': post_fail_ib_mid_hit,
        'post_fail_ibl_hit': post_fail_ibl_hit,
        'day_type': dt_val,
    }
    session_data.append(record)

# ============================================================================
# PRINT CLASSIFICATION SUMMARY
# ============================================================================
class_counts = defaultdict(int)
for r in session_data:
    class_counts[r['classification']] += 1

total = len(session_data)
print(f"\nSession Classification ({total} total):")
for cls in ['NO_BREAK', 'QUICK_FAKE', 'MODERATE_SWEEP', 'DEEP_SWEEP_FAIL',
            'HELD_ABOVE', 'STRONG_EXTENSION']:
    n = class_counts.get(cls, 0)
    print(f"  {cls:<20s}: {n:>3d} ({n/total*100:>5.1f}%)")

# Failed breakouts combined
failed_sessions = [r for r in session_data if r['failed']]
non_failed_breaks = [r for r in session_data if r['broke_above'] and not r['failed']]

print(f"\n  Broke above IBH:    {len(failed_sessions) + len(non_failed_breaks)} / {total} ({(len(failed_sessions)+len(non_failed_breaks))/total*100:.0f}%)")
print(f"  Of those, FAILED:   {len(failed_sessions)} / {len(failed_sessions)+len(non_failed_breaks)} ({len(failed_sessions)/(len(failed_sessions)+len(non_failed_breaks))*100:.0f}%)")
print(f"  Of those, EXTENDED: {len(non_failed_breaks)} / {len(failed_sessions)+len(non_failed_breaks)} ({len(non_failed_breaks)/(len(failed_sessions)+len(non_failed_breaks))*100:.0f}%)")


# ============================================================================
# PHASE 2: DEEP DIVE ON FAILED BREAKOUTS
# ============================================================================
print("\n\n" + "=" * 130)
print("  PHASE 2: FAILED BREAKOUT DEEP DIVE")
print("=" * 130)

if failed_sessions:
    print(f"\n  {len(failed_sessions)} failed breakout sessions:")
    print(f"\n  {'Date':<14s} {'Class':<18s} {'Ext(pts)':>8s} {'Ext(xIB)':>8s} {'BarsAbv':>7s} "
          f"{'FailTime':>9s} {'BrkDelta':>9s} {'CloseVsIBH':>10s} {'CloseVsMid':>10s} "
          f"{'→VWAP':>5s} {'→Mid':>5s} {'→IBL':>5s} {'DayType':>12s}")
    print("  " + "-" * 128)

    for r in sorted(failed_sessions, key=lambda x: x['date']):
        print(f"  {r['date']:<14s} {r['classification']:<18s} {r['max_ext_pts']:>7.1f} "
              f"{r['max_ext_mult']:>7.2f}x {r['total_bars_above']:>7d} "
              f"{str(r['fail_time'] or 'N/A'):>9s} {r['breakout_delta_total']:>+8.0f} "
              f"{r['close_vs_ibh']:>+9.1f} {r['close_vs_ib_mid']:>+9.1f} "
              f"{'Y' if r['post_fail_vwap_hit'] else 'N':>5s} "
              f"{'Y' if r['post_fail_ib_mid_hit'] else 'N':>5s} "
              f"{'Y' if r['post_fail_ibl_hit'] else 'N':>5s} "
              f"{r['day_type']:>12s}")

    # Statistics
    ext_pts_list = [r['max_ext_pts'] for r in failed_sessions]
    ext_mult_list = [r['max_ext_mult'] for r in failed_sessions]
    bars_above_list = [r['total_bars_above'] for r in failed_sessions]
    delta_list = [r['breakout_delta_total'] for r in failed_sessions]

    print(f"\n  Extension above IBH before failure:")
    print(f"    Mean: {np.mean(ext_pts_list):.1f} pts ({np.mean(ext_mult_list):.2f}x IB)")
    print(f"    Median: {np.median(ext_pts_list):.1f} pts ({np.median(ext_mult_list):.2f}x IB)")
    print(f"    Max: {max(ext_pts_list):.1f} pts ({max(ext_mult_list):.2f}x IB)")

    print(f"\n  Bars spent above IBH:")
    print(f"    Mean: {np.mean(bars_above_list):.1f}  Median: {np.median(bars_above_list):.0f}  Max: {max(bars_above_list)}")

    print(f"\n  Breakout delta (while above IBH):")
    print(f"    Mean: {np.mean(delta_list):+.0f}  Positive: {sum(1 for d in delta_list if d > 0)}/{len(delta_list)}")

    # Where did failed breakouts end up?
    vwap_hit_pct = sum(1 for r in failed_sessions if r['post_fail_vwap_hit']) / len(failed_sessions) * 100
    mid_hit_pct = sum(1 for r in failed_sessions if r['post_fail_ib_mid_hit']) / len(failed_sessions) * 100
    ibl_hit_pct = sum(1 for r in failed_sessions if r['post_fail_ibl_hit']) / len(failed_sessions) * 100

    closed_below_ibh = sum(1 for r in failed_sessions if r['session_close'] < r['ib_high'])
    closed_below_mid = sum(1 for r in failed_sessions if r['session_close'] < r['ib_mid'])
    closed_below_ibl = sum(1 for r in failed_sessions if r['session_close'] < r['ib_low'])

    print(f"\n  After failure, price reached:")
    print(f"    Hit VWAP:     {vwap_hit_pct:.0f}%")
    print(f"    Hit IB mid:   {mid_hit_pct:.0f}%")
    print(f"    Hit IBL:      {ibl_hit_pct:.0f}%")

    print(f"\n  Session close:")
    print(f"    Below IBH:    {closed_below_ibh}/{len(failed_sessions)} ({closed_below_ibh/len(failed_sessions)*100:.0f}%)")
    print(f"    Below IB mid: {closed_below_mid}/{len(failed_sessions)} ({closed_below_mid/len(failed_sessions)*100:.0f}%)")
    print(f"    Below IBL:    {closed_below_ibl}/{len(failed_sessions)} ({closed_below_ibl/len(failed_sessions)*100:.0f}%)")

    # Day types of failed breakouts
    fail_day_types = defaultdict(int)
    for r in failed_sessions:
        fail_day_types[r['day_type']] += 1
    print(f"\n  Day types after failed breakout: {dict(fail_day_types)}")

    # Delta signal: failed breakouts with negative delta while above
    neg_delta_fails = [r for r in failed_sessions if r['breakout_delta_total'] < 0]
    pos_delta_fails = [r for r in failed_sessions if r['breakout_delta_total'] >= 0]
    print(f"\n  Delta during breakout:")
    print(f"    Negative delta (sellers absorbing): {len(neg_delta_fails)} — early warning signal")
    if neg_delta_fails:
        mid_pct = sum(1 for r in neg_delta_fails if r['post_fail_ib_mid_hit']) / len(neg_delta_fails) * 100
        print(f"      → {mid_pct:.0f}% reached IB mid after failure")
    print(f"    Positive delta (buyers driving):    {len(pos_delta_fails)}")
    if pos_delta_fails:
        mid_pct = sum(1 for r in pos_delta_fails if r['post_fail_ib_mid_hit']) / len(pos_delta_fails) * 100
        print(f"      → {mid_pct:.0f}% reached IB mid after failure")


# ============================================================================
# PHASE 3: SIMULATE FAILED BREAKOUT SHORT ENTRIES
# ============================================================================
print("\n\n" + "=" * 130)
print("  PHASE 3: SIMULATED FAILED BREAKOUT SHORT TRADES")
print("=" * 130)

def simulate_failed_breakout(config_name, target_mode='ib_mid',
                              min_bars_above=2, max_extension_mult=0.5,
                              require_neg_delta_breakout=False,
                              require_close_below_ibh=True,
                              require_rejection_candle=False,
                              allowed_day_types=None,
                              max_contracts=5):
    """
    Simulate shorting after a confirmed IBH breakout failure.

    Entry: Price broke above IBH, spent time above, then closes back below IBH.
    This is the CONFIRMED failure, not a fade at IBH.
    """
    trades = []

    for session_date in sessions:
        sdf = df[df['session_date'] == session_date].copy()
        if len(sdf) < IB_BARS_1MIN + 20:
            continue

        ib_df = sdf.head(IB_BARS_1MIN)
        post_ib = sdf.iloc[IB_BARS_1MIN:]

        ib_high = ib_df['high'].max()
        ib_low = ib_df['low'].min()
        ib_range = ib_high - ib_low
        ib_mid = (ib_high + ib_low) / 2

        if ib_range <= 0 or ib_range > 500:
            continue

        # State machine
        state = 'WAITING'  # WAITING -> ABOVE_IBH -> CONFIRMED_FAIL -> IN_TRADE
        bars_above_count = 0
        max_ext_above = 0
        breakout_delta_sum = 0
        entry_taken = False
        entry_price = None
        stop_price = None
        target_price = None
        entry_bar_idx = None
        session_high = ib_high

        for i, (idx, bar) in enumerate(post_ib.iterrows()):
            price = bar['close']
            bar_time = bar['timestamp'].time() if 'timestamp' in bar.index else None

            # No entries after 1:30 PM (need time for reversion)
            if bar_time and bar_time >= _time(13, 30) and not entry_taken:
                break

            # Force close at 3:00 PM
            if bar_time and bar_time >= _time(15, 0) and entry_taken:
                exit_fill = execution.fill_exit('SHORT', price)
                gross, comm, slip, net = execution.calculate_net_pnl(
                    'SHORT', entry_price, exit_fill, max_contracts)
                trades.append({
                    'date': str(session_date), 'entry': entry_price,
                    'exit': exit_fill, 'net_pnl': net,
                    'exit_reason': 'EOD', 'bars_held': i - entry_bar_idx,
                    'ext_mult': max_ext_above / ib_range if ib_range > 0 else 0,
                    'bars_above': bars_above_count,
                    'breakout_delta': breakout_delta_sum,
                })
                entry_taken = False
                break

            if bar['high'] > session_high:
                session_high = bar['high']

            # --- MANAGE OPEN POSITION ---
            if entry_taken:
                # Check stop
                if bar['high'] >= stop_price:
                    exit_fill = execution.fill_exit('SHORT', stop_price)
                    gross, comm, slip, net = execution.calculate_net_pnl(
                        'SHORT', entry_price, exit_fill, max_contracts)
                    trades.append({
                        'date': str(session_date), 'entry': entry_price,
                        'exit': exit_fill, 'net_pnl': net,
                        'exit_reason': 'STOP', 'bars_held': i - entry_bar_idx,
                        'ext_mult': max_ext_above / ib_range if ib_range > 0 else 0,
                        'bars_above': bars_above_count,
                        'breakout_delta': breakout_delta_sum,
                    })
                    entry_taken = False
                    break  # One trade per session

                # Check target
                if bar['low'] <= target_price:
                    exit_fill = execution.fill_exit('SHORT', target_price)
                    gross, comm, slip, net = execution.calculate_net_pnl(
                        'SHORT', entry_price, exit_fill, max_contracts)
                    trades.append({
                        'date': str(session_date), 'entry': entry_price,
                        'exit': exit_fill, 'net_pnl': net,
                        'exit_reason': 'TARGET', 'bars_held': i - entry_bar_idx,
                        'ext_mult': max_ext_above / ib_range if ib_range > 0 else 0,
                        'bars_above': bars_above_count,
                        'breakout_delta': breakout_delta_sum,
                    })
                    entry_taken = False
                    break  # One trade per session

                continue

            # --- STATE MACHINE ---
            if state == 'WAITING':
                if price > ib_high:
                    state = 'ABOVE_IBH'
                    bars_above_count = 1
                    max_ext_above = bar['high'] - ib_high
                    delta = bar.get('delta', 0)
                    breakout_delta_sum = delta if not pd.isna(delta) else 0

            elif state == 'ABOVE_IBH':
                if price > ib_high:
                    bars_above_count += 1
                    ext = bar['high'] - ib_high
                    if ext > max_ext_above:
                        max_ext_above = ext
                    delta = bar.get('delta', 0)
                    if not pd.isna(delta):
                        breakout_delta_sum += delta
                else:
                    # Price closed back below IBH — potential failure
                    ext_mult = max_ext_above / ib_range if ib_range > 0 else 0

                    # --- FAILURE CONFIRMATION CHECKS ---
                    pass_checks = True

                    # Min bars above (was it a real breakout or just a wick?)
                    if bars_above_count < min_bars_above:
                        pass_checks = False

                    # Max extension (if it went too far, it might be a legit trend day pullback)
                    if ext_mult > max_extension_mult:
                        pass_checks = False

                    # Require close below IBH
                    if require_close_below_ibh and price >= ib_high:
                        pass_checks = False

                    # Negative delta during breakout (sellers absorbing = trap)
                    if require_neg_delta_breakout and breakout_delta_sum >= 0:
                        pass_checks = False

                    # Rejection candle: close near bar low (strong selling)
                    if require_rejection_candle:
                        bar_range = bar['high'] - bar['low']
                        if bar_range > 0:
                            close_position = (price - bar['low']) / bar_range
                            if close_position > 0.35:  # Not a strong rejection
                                pass_checks = False

                    # Day type filter
                    if allowed_day_types:
                        if price > ib_high:
                            ib_dir = 'BULL'
                            ext_c = (price - ib_mid) / ib_range
                        elif price < ib_low:
                            ib_dir = 'BEAR'
                            ext_c = (ib_mid - price) / ib_range
                        else:
                            ib_dir = 'INSIDE'
                            ext_c = 0.0
                        strength = classify_trend_strength(ext_c)
                        dt = classify_day_type(ib_high, ib_low, price, ib_dir, strength)
                        dt_val = dt.value if hasattr(dt, 'value') else str(dt)
                        if dt_val not in allowed_day_types:
                            pass_checks = False

                    if pass_checks:
                        # === ENTER SHORT ===
                        entry_raw = price
                        entry_fill = execution.fill_entry('SHORT', entry_raw)
                        entry_price = entry_fill

                        # Stop: above the failed breakout high + buffer
                        stop_price = session_high + (ib_range * 0.10)
                        stop_price = max(stop_price, entry_price + 15.0)

                        # Target
                        vwap_now = bar.get('vwap', ib_mid)
                        if pd.isna(vwap_now):
                            vwap_now = ib_mid

                        if target_mode == 'ib_mid':
                            target_price = ib_mid
                        elif target_mode == 'vwap':
                            target_price = vwap_now
                        elif target_mode == 'ibl':
                            target_price = ib_low
                        elif target_mode == 'val':
                            target_price = ib_low + (ib_range * 0.15)
                        else:
                            target_price = ib_mid

                        # R:R check
                        risk = stop_price - entry_price
                        reward = entry_price - target_price
                        if reward > 0 and risk > 0 and risk < reward * 3:
                            entry_taken = True
                            entry_bar_idx = i
                            state = 'IN_TRADE'
                        else:
                            state = 'DONE'
                    else:
                        state = 'DONE'  # Failed checks, skip this session

        # Force close at EOD if still open
        if entry_taken and len(post_ib) > 0:
            last_bar = post_ib.iloc[-1]
            exit_fill = execution.fill_exit('SHORT', last_bar['close'])
            gross, comm, slip, net = execution.calculate_net_pnl(
                'SHORT', entry_price, exit_fill, max_contracts)
            trades.append({
                'date': str(session_date), 'entry': entry_price,
                'exit': exit_fill, 'net_pnl': net,
                'exit_reason': 'EOD', 'bars_held': len(post_ib) - entry_bar_idx,
                'ext_mult': max_ext_above / ib_range if ib_range > 0 else 0,
                'bars_above': bars_above_count,
                'breakout_delta': breakout_delta_sum,
            })

    return trades


# Test multiple configurations
configs = [
    # (name, target, min_bars, max_ext, neg_delta, close_below, rejection, day_types)
    ('A: Basic (2+ bars above, ext<0.5x) → IB mid',
     'ib_mid', 2, 0.5, False, True, False, None),
    ('B: Basic → VWAP',
     'vwap', 2, 0.5, False, True, False, None),
    ('C: Basic → IBL',
     'ibl', 2, 0.5, False, True, False, None),
    ('D: Quick fake (1+ bar, ext<0.3x) → IB mid',
     'ib_mid', 1, 0.3, False, True, False, None),
    ('E: Moderate sweep (3+ bars, ext<0.7x) → IB mid',
     'ib_mid', 3, 0.7, False, True, False, None),
    ('F: Neg delta breakout (sellers absorb) → IB mid',
     'ib_mid', 2, 0.5, True, True, False, None),
    ('G: Neg delta breakout → VWAP',
     'vwap', 2, 0.5, True, True, False, None),
    ('H: Rejection candle + neg delta → IB mid',
     'ib_mid', 2, 0.5, True, True, True, None),
    ('I: Balance days only (b_day/neutral) → IB mid',
     'ib_mid', 2, 0.5, False, True, False, ['b_day', 'neutral']),
    ('J: Balance + neg delta → IB mid',
     'ib_mid', 2, 0.5, True, True, False, ['b_day', 'neutral']),
    ('K: All day types, neg delta → VWAP',
     'vwap', 2, 0.7, True, True, False, None),
    ('L: 1+ bar, ext<0.5x, neg delta → VWAP',
     'vwap', 1, 0.5, True, True, False, None),
    ('M: Strict (3+ bars, rejection, neg delta) → IB mid',
     'ib_mid', 3, 0.5, True, True, True, None),
    ('N: Quick fake + neg delta → VWAP',
     'vwap', 1, 0.3, True, True, False, None),
]

all_results = []

print(f"\n{'Config':<55s} {'Trades':>6s} {'W':>3s} {'L':>3s} {'WR%':>6s} {'Net PnL':>10s} {'Expect':>8s} {'AvgWin':>8s} {'AvgLoss':>8s}")
print("-" * 120)

for name, target, min_b, max_e, neg_d, close_b, rej, dt_filter in configs:
    t_list = simulate_failed_breakout(
        name, target_mode=target, min_bars_above=min_b,
        max_extension_mult=max_e, require_neg_delta_breakout=neg_d,
        require_close_below_ibh=close_b, require_rejection_candle=rej,
        allowed_day_types=dt_filter,
    )
    wins = [t for t in t_list if t['net_pnl'] > 0]
    losses = [t for t in t_list if t['net_pnl'] <= 0]
    pnl = sum(t['net_pnl'] for t in t_list)
    wr = len(wins) / len(t_list) * 100 if t_list else 0
    avg_w = np.mean([t['net_pnl'] for t in wins]) if wins else 0
    avg_l = np.mean([t['net_pnl'] for t in losses]) if losses else 0
    exp = pnl / len(t_list) if t_list else 0

    print(f"{name:<55s} {len(t_list):>6d} {len(wins):>3d} {len(losses):>3d} "
          f"{wr:>5.1f}% ${pnl:>8,.0f} ${exp:>6,.0f} ${avg_w:>6,.0f} ${avg_l:>6,.0f}")

    all_results.append({
        'name': name, 'trades': len(t_list), 'wins': len(wins),
        'losses': len(losses), 'wr': wr, 'pnl': pnl, 'exp': exp,
        'trade_list': t_list,
    })

print("-" * 120)


# ============================================================================
# PHASE 4: DETAILED TRADE LOG FOR BEST CONFIG
# ============================================================================
profitable = [r for r in all_results if r['pnl'] > 0 and r['trades'] >= 3]
if profitable:
    best = max(profitable, key=lambda x: x['pnl'])
else:
    best = max(all_results, key=lambda x: x['pnl']) if all_results else None

if best and best['trade_list']:
    print(f"\n\n{'=' * 130}")
    print(f"  BEST CONFIG: {best['name']}")
    print(f"  {best['trades']} trades | {best['wr']:.1f}% WR | ${best['pnl']:,.0f} net | ${best['exp']:,.0f}/trade")
    print(f"{'=' * 130}")

    print(f"\n  {'Date':<14s} {'Entry':>10s} {'Exit':>10s} {'Net PnL':>10s} {'Reason':<8s} "
          f"{'Bars':>5s} {'ExtMult':>7s} {'BarsAbv':>7s} {'BrkDelta':>9s}")
    print("  " + "-" * 90)

    for t in sorted(best['trade_list'], key=lambda x: x['date']):
        print(f"  {t['date']:<14s} {t['entry']:>10.2f} {t['exit']:>10.2f} "
              f"${t['net_pnl']:>8,.2f} {t['exit_reason']:<8s} "
              f"{t['bars_held']:>5d} {t['ext_mult']:>6.2f}x {t['bars_above']:>7d} "
              f"{t['breakout_delta']:>+8.0f}")

    # Exit reason analysis
    exit_summary = defaultdict(list)
    for t in best['trade_list']:
        exit_summary[t['exit_reason']].append(t['net_pnl'])

    print(f"\n  Exit reasons:")
    for reason, pnls in sorted(exit_summary.items()):
        w = sum(1 for p in pnls if p > 0)
        print(f"    {reason:<8s}: {len(pnls)} trades, {w}/{len(pnls)} wins, ${sum(pnls):>8,.0f}")


# ============================================================================
# PHASE 5: COMPARISON WITH LONG-ONLY PLAYBOOK
# ============================================================================
print(f"\n\n{'=' * 130}")
print("  PHASE 5: CAN FAILED BREAKOUT SHORT ADD TO THE 70% WR PLAYBOOK?")
print(f"{'=' * 130}")

if best:
    print(f"""
  CURRENT PLAYBOOK (Core + MeanRev LONG only):
    75.9% WR | 29 trades | $3,861 net | $133/trade

  BEST FAILED BREAKOUT SHORT:
    {best['wr']:.1f}% WR | {best['trades']} trades | ${best['pnl']:,.0f} net | ${best['exp']:,.0f}/trade
""")

    if best['pnl'] > 0 and best['wr'] >= 40:
        # Simulate combined portfolio
        combined_trades = 29 + best['trades']
        combined_wins = 22 + best['wins']
        combined_pnl = 3861 + best['pnl']
        combined_wr = combined_wins / combined_trades * 100

        print(f"  COMBINED PORTFOLIO:")
        print(f"    {combined_wr:.1f}% WR | {combined_trades} trades | ${combined_pnl:,.0f} net | ${combined_pnl/combined_trades:,.0f}/trade")

        if combined_wr >= 70:
            print(f"\n    COMBINED WR >= 70% — CAN ADD to Lightning playbook!")
            print(f"    Adding failed breakout shorts would contribute +${best['pnl']:,.0f} additional profit")
            print(f"    and +{best['trades']} more trades for consistency rule compliance.")
        elif combined_wr >= 65:
            print(f"\n    COMBINED WR 65-70% — MARGINAL. Adds profit but risks WR target.")
        else:
            print(f"\n    COMBINED WR BELOW 65% — NOT RECOMMENDED for Lightning 70% target.")
    elif best['pnl'] > 0:
        print(f"  Failed breakout has positive PnL but low WR ({best['wr']:.1f}%).")
        print(f"  Risk: Would dilute portfolio WR significantly.")
    else:
        print(f"  Failed breakout is NET NEGATIVE even as best config.")
        print(f"  NQ long bias still dominates: failed breakouts often re-break higher.")
else:
    print("  No configurations produced trades.")


# ============================================================================
# PHASE 6: FAILED BREAKOUT AS A FILTER (not a trade)
# ============================================================================
print(f"\n\n{'=' * 130}")
print("  PHASE 6: FAILED BREAKOUT AS A LONG ENTRY FILTER")
print("  (Don't go long when breakout is failing — wait for confirmation)")
print(f"{'=' * 130}")

# Analysis: Of the sessions where breakout failed, did our LONG strategies
# that session perform better or worse?
fail_dates = set(r['date'] for r in failed_sessions)
non_fail_break_dates = set(r['date'] for r in session_data if r['broke_above'] and not r['failed'])
no_break_dates = set(r['date'] for r in session_data if not r['broke_above'])

print(f"\n  Session categories:")
print(f"    No break above IBH:       {len(no_break_dates)} sessions")
print(f"    Broke above + held/ext:   {len(non_fail_break_dates)} sessions")
print(f"    Broke above + FAILED:     {len(fail_dates)} sessions")

print(f"""
  INSIGHT: If we detect a breakout is failing (negative delta while above IBH,
  price spending time but not accepting), we could:
    1. SKIP new long entries that session (avoid buying into a failing breakout)
    2. TIGHTEN stops on existing longs (breakout failure = bearish signal)
    3. Consider SHORT only after confirmed failure (this study's main finding)

  This filter concept requires real-time detection of breakout quality —
  monitoring delta, acceptance bars, and extension while price is above IBH.
  When delta is negative and extension is weak (<0.3x IB), flag as likely failure.
""")
