"""
Opening Range Strategy v3 - Coverage Gap Analysis + New Models
==============================================================

Goals:
1. For ALL 62 sessions, classify WHY each session has/doesn't have an OR trade
2. Fix bugs in v1/v2: London levels missing, level priority order, threshold too tight
3. Test new models: OR Breakout continuation, Silver Bullet (10:00-11:00 FVG entry)
4. Test dynamic proximity threshold
5. Find the sessions we're MISSING and what model could catch them

New entry models tested:
A. OR_REVERSAL_V2 (improved): Fix level priority + add London H/L + dynamic threshold
B. OR_BREAKOUT: Price breaks OR H/L and CONTINUES (not reverses). For trend-from-open days.
C. SILVER_BULLET: 10:00-11:00 FVG entry after liquidity sweep with structural displacement
D. NR7_BREAKOUT: After narrow prior-day range, ORB continuation next day
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path
import sys

proj_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(proj_root))

from data.loader import load_csv
from data.session import filter_rth

# Reuse v1 functions
from diagnostic_opening_range_smt import (
    compute_session_levels,
    detect_smt_divergence,
    detect_fvg_5min,
    detect_fvg_15min,
    calc_metrics,
    print_metrics,
)


def classify_session_gap(nq, sd, levels):
    """
    For a session that has ZERO raw trades from v1 detect_opening_reversal(),
    figure out exactly WHY.

    Returns dict with failure reason and diagnostics.
    """
    sd_ts = pd.Timestamp(sd)

    # Get bars 9:00-11:00
    mask = (
        (nq['timestamp'] >= sd_ts + timedelta(hours=9)) &
        (nq['timestamp'] <= sd_ts + timedelta(hours=11))
    )
    bars = nq[mask].copy().reset_index(drop=True)

    if len(bars) < 30:
        return {'reason': 'INSUFFICIENT_DATA', 'details': f'Only {len(bars)} bars'}

    # Find RTH start
    rth_start = None
    for i, bar in bars.iterrows():
        if bar['timestamp'].hour == 9 and bar['timestamp'].minute >= 30:
            rth_start = i
            break
        elif bar['timestamp'].hour >= 10:
            rth_start = i
            break
    if rth_start is None:
        return {'reason': 'NO_RTH_START', 'details': 'Cannot find 9:30 bar'}

    rth_bars = bars.loc[rth_start:].copy().reset_index(drop=True)
    if len(rth_bars) < 30:
        return {'reason': 'INSUFFICIENT_RTH', 'details': f'Only {len(rth_bars)} RTH bars'}

    # OR: first 15 bars
    or_bars = rth_bars.iloc[:15]
    or_high = or_bars['high'].max()
    or_low = or_bars['low'].min()
    or_mid = (or_high + or_low) / 2
    or_range = or_high - or_low

    # EOR: first 30 bars
    eor_bars = rth_bars.iloc[:30]
    eor_high = eor_bars['high'].max()
    eor_low = eor_bars['low'].min()
    eor_range = eor_high - eor_low

    # All reference levels (including London!)
    all_highs = {}
    all_lows = {}
    for lname in ['overnight_high', 'asia_high', 'london_high', 'pdh']:
        val = levels.get(lname, np.nan)
        if not np.isnan(val):
            all_highs[lname] = val
    for lname in ['overnight_low', 'asia_low', 'london_low', 'pdl']:
        val = levels.get(lname, np.nan)
        if not np.isnan(val):
            all_lows[lname] = val

    # Compute distance from EOR high/low to each level
    high_distances = {name: abs(eor_high - val) for name, val in all_highs.items()}
    low_distances = {name: abs(eor_low - val) for name, val in all_lows.items()}

    closest_high_name = min(high_distances, key=high_distances.get) if high_distances else None
    closest_high_dist = high_distances[closest_high_name] if closest_high_name else 999
    closest_low_name = min(low_distances, key=low_distances.get) if low_distances else None
    closest_low_dist = low_distances[closest_low_name] if closest_low_name else 999

    # Gate 1: Is the EOR extreme near any level (30 pt threshold)?
    high_near = closest_high_dist < 30
    low_near = closest_low_dist < 30

    # Dynamic threshold: 15% of overnight range
    on_range = levels.get('overnight_high', 0) - levels.get('overnight_low', 0)
    dynamic_threshold = max(30, min(50, 0.15 * on_range))
    high_near_dynamic = closest_high_dist < dynamic_threshold
    low_near_dynamic = closest_low_dist < dynamic_threshold

    if not high_near and not low_near:
        # Check if dynamic threshold would help
        if high_near_dynamic or low_near_dynamic:
            recovered = closest_high_name if high_near_dynamic else closest_low_name
            recovered_dist = closest_high_dist if high_near_dynamic else closest_low_dist
            return {
                'reason': 'GATE1_FAIL_RECOVERABLE',
                'details': f'Nearest level: {recovered} at {recovered_dist:.0f} pts (threshold=30, dynamic={dynamic_threshold:.0f})',
                'closest_high': closest_high_name,
                'closest_high_dist': closest_high_dist,
                'closest_low': closest_low_name,
                'closest_low_dist': closest_low_dist,
                'eor_high': eor_high,
                'eor_low': eor_low,
                'or_range': or_range,
                'eor_range': eor_range,
                'dynamic_threshold': dynamic_threshold,
            }
        return {
            'reason': 'GATE1_FAIL_NO_LEVEL',
            'details': f'Closest high: {closest_high_name}={closest_high_dist:.0f}pts, Closest low: {closest_low_name}={closest_low_dist:.0f}pts',
            'closest_high': closest_high_name,
            'closest_high_dist': closest_high_dist,
            'closest_low': closest_low_name,
            'closest_low_dist': closest_low_dist,
            'eor_high': eor_high,
            'eor_low': eor_low,
            'or_range': or_range,
            'eor_range': eor_range,
        }

    # Gate 2: Does price reverse through OR midpoint?
    # Check high sweep reversal
    high_reversed = False
    if high_near:
        high_bar_idx = eor_bars['high'].idxmax()
        post_high = rth_bars.loc[high_bar_idx:]
        for j in range(1, min(30, len(post_high))):
            if post_high.iloc[j]['close'] < or_mid:
                high_reversed = True
                break

    low_reversed = False
    if low_near:
        low_bar_idx = eor_bars['low'].idxmin()
        post_low = rth_bars.loc[low_bar_idx:]
        for j in range(1, min(30, len(post_low))):
            if post_low.iloc[j]['close'] > or_mid:
                low_reversed = True
                break

    if not high_reversed and not low_reversed:
        # Check if this is a trend-from-open (breakout continuation)
        eod_bar = rth_bars.iloc[-1] if len(rth_bars) > 60 else rth_bars.iloc[min(60, len(rth_bars)-1)]
        eod_close = eod_bar['close']

        direction = 'NONE'
        if high_near and eod_close > eor_high:
            direction = 'BREAKOUT_UP'
        elif low_near and eod_close < eor_low:
            direction = 'BREAKOUT_DOWN'
        elif high_near:
            direction = 'HIGH_SWEEP_NO_REVERSE'
        elif low_near:
            direction = 'LOW_SWEEP_NO_REVERSE'

        return {
            'reason': 'GATE2_FAIL_NO_REVERSAL',
            'details': f'Sweep found but no reversal through OR mid. Direction: {direction}',
            'direction': direction,
            'sweep_side': 'HIGH' if high_near else 'LOW',
            'closest_level': closest_high_name if high_near else closest_low_name,
            'level_dist': closest_high_dist if high_near else closest_low_dist,
            'eor_high': eor_high,
            'eor_low': eor_low,
            'or_range': or_range,
            'eor_range': eor_range,
            'eod_close': eod_close,
        }

    return {
        'reason': 'SHOULD_HAVE_TRADE',
        'details': 'Passed Gate 1 and Gate 2 but detect_opening_reversal() missed it?',
        'high_near': high_near,
        'low_near': low_near,
        'high_reversed': high_reversed,
        'low_reversed': low_reversed,
    }


def detect_or_breakout(rth_bars, levels, threshold=30):
    """
    OR Breakout Continuation model.

    Instead of fading the Judas swing, this model RIDES it:
    - First 30 min breaks above/below a key level
    - Price CONTINUES in the breakout direction (no reversal)
    - Enter on pullback to OR edge or FVG after the breakout

    Returns list of setup dicts.
    """
    setups = []
    if len(rth_bars) < 60:
        return setups

    # OR: first 30 bars
    eor_bars = rth_bars.iloc[:30]
    eor_high = eor_bars['high'].max()
    eor_low = eor_bars['low'].min()
    or_range = eor_high - eor_low

    if or_range < 20:
        return setups  # Too narrow to trade

    # Check for breakout above: EOR high near a level AND price keeps going up
    all_highs = {}
    for lname in ['overnight_high', 'asia_high', 'london_high', 'pdh']:
        val = levels.get(lname, np.nan)
        if not np.isnan(val):
            all_highs[lname] = val

    all_lows = {}
    for lname in ['overnight_low', 'asia_low', 'london_low', 'pdl']:
        val = levels.get(lname, np.nan)
        if not np.isnan(val):
            all_lows[lname] = val

    # Bullish breakout: EOR high sweeps above a high level AND price stays above
    for lname, lval in sorted(all_highs.items(), key=lambda x: abs(eor_high - x[1])):
        if abs(eor_high - lval) < threshold:
            # Found a sweep of a high level. Check continuation (bars 30-60)
            if len(rth_bars) < 60:
                break
            post_eor = rth_bars.iloc[30:60]
            # Continuation = majority of bars close above the level
            bars_above = sum(post_eor['close'] > lval)
            if bars_above >= len(post_eor) * 0.6:
                # This is a breakout, not a reversal. Enter on pullback to EOR high
                for j in range(30, min(90, len(rth_bars))):
                    bar = rth_bars.iloc[j]
                    # Pullback: price comes back near EOR high (within 15 pts)
                    if bar['low'] <= eor_high + 15 and bar['close'] > eor_high - 10:
                        risk = bar['close'] - (eor_high - 0.15 * or_range)
                        if 10 < risk < 200:
                            setups.append({
                                'model': 'OR_BREAKOUT_LONG',
                                'entry_idx': j,
                                'entry_time': bar['timestamp'],
                                'entry_price': bar['close'],
                                'sweep_level': lname,
                                'direction': 'LONG',
                                'stop': eor_high - 0.15 * or_range,
                                'or_high': eor_high,
                                'or_low': eor_low,
                            })
                            break
            break  # Only check closest level

    # Bearish breakout: EOR low sweeps below a low level AND price stays below
    for lname, lval in sorted(all_lows.items(), key=lambda x: abs(eor_low - x[1])):
        if abs(eor_low - lval) < threshold:
            if len(rth_bars) < 60:
                break
            post_eor = rth_bars.iloc[30:60]
            bars_below = sum(post_eor['close'] < lval)
            if bars_below >= len(post_eor) * 0.6:
                for j in range(30, min(90, len(rth_bars))):
                    bar = rth_bars.iloc[j]
                    if bar['high'] >= eor_low - 15 and bar['close'] < eor_low + 10:
                        risk = (eor_low + 0.15 * or_range) - bar['close']
                        if 10 < risk < 200:
                            setups.append({
                                'model': 'OR_BREAKOUT_SHORT',
                                'entry_idx': j,
                                'entry_time': bar['timestamp'],
                                'entry_price': bar['close'],
                                'sweep_level': lname,
                                'direction': 'SHORT',
                                'stop': eor_low + 0.15 * or_range,
                                'or_high': eor_high,
                                'or_low': eor_low,
                            })
                            break
            break

    return setups


def detect_silver_bullet(rth_bars, levels, nq_full, sd):
    """
    ICT Silver Bullet: 10:00-11:00 AM entry.

    Model:
    1. During 9:30-10:00, a liquidity sweep occurs (any pre-market level)
    2. Displacement candle creates FVG on 5-min bars between 10:00-11:00
    3. Price retraces into FVG zone
    4. Enter on FVG retrace with stop beyond sweep extreme

    Returns list of setup dicts.
    """
    setups = []
    sd_ts = pd.Timestamp(sd)

    # Get 10:00-11:00 bars
    sb_mask = (
        (nq_full['timestamp'] >= sd_ts + timedelta(hours=10)) &
        (nq_full['timestamp'] <= sd_ts + timedelta(hours=11))
    )
    sb_bars = nq_full[sb_mask].copy().reset_index(drop=True)
    if len(sb_bars) < 30:
        return setups

    # Need the 9:30-10:00 bars for context (sweep detection)
    or_mask = (
        (nq_full['timestamp'] >= sd_ts + timedelta(hours=9, minutes=30)) &
        (nq_full['timestamp'] < sd_ts + timedelta(hours=10))
    )
    or_bars = nq_full[or_mask].copy().reset_index(drop=True)
    if len(or_bars) < 15:
        return setups

    or_high = or_bars['high'].max()
    or_low = or_bars['low'].min()

    # Detect if a sweep happened in the OR period
    all_levels = {}
    for lname in ['overnight_high', 'overnight_low', 'asia_high', 'asia_low',
                   'london_high', 'london_low', 'pdh', 'pdl']:
        val = levels.get(lname, np.nan)
        if not np.isnan(val):
            all_levels[lname] = val

    sweep_high_level = None
    sweep_low_level = None
    for lname, lval in all_levels.items():
        if 'high' in lname or lname == 'pdh':
            if or_high >= lval - 5:  # Swept above
                if sweep_high_level is None or abs(or_high - lval) < abs(or_high - all_levels.get(sweep_high_level, 0)):
                    sweep_high_level = lname
        elif 'low' in lname or lname == 'pdl':
            if or_low <= lval + 5:  # Swept below
                if sweep_low_level is None or abs(or_low - lval) < abs(or_low - all_levels.get(sweep_low_level, 0)):
                    sweep_low_level = lname

    if not sweep_high_level and not sweep_low_level:
        return setups  # No sweep in OR period

    # Resample 10:00-11:00 bars to 5-min
    sb_5m = sb_bars.set_index('timestamp').resample('5min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        'volume': 'sum', 'vol_delta': 'sum',
    }).dropna()

    if len(sb_5m) < 3:
        return setups

    # Detect FVGs in 5-min bars
    for i in range(len(sb_5m) - 2):
        b1_high = sb_5m['high'].iloc[i]
        b1_low = sb_5m['low'].iloc[i]
        b2_ts = sb_5m.index[i + 1]
        b3_high = sb_5m['high'].iloc[i + 2]
        b3_low = sb_5m['low'].iloc[i + 2]

        # Bullish FVG after sweep LOW (price reversed up, gap left)
        if sweep_low_level and b3_low > b1_high + 2:
            fvg_top = b3_low
            fvg_bottom = b1_high
            # Look for retrace into FVG on 1-min bars
            post_fvg_mask = sb_bars['timestamp'] > b2_ts
            post_fvg = sb_bars[post_fvg_mask]
            for _, bar in post_fvg.iterrows():
                if bar['low'] <= fvg_top and bar['close'] >= fvg_bottom:
                    entry_price = bar['close']
                    stop = or_low - 0.10 * (or_high - or_low)
                    risk = entry_price - stop
                    if 10 < risk < 200:
                        setups.append({
                            'model': 'SILVER_BULLET_LONG',
                            'entry_idx': bar.name if hasattr(bar, 'name') else 0,
                            'entry_time': bar['timestamp'],
                            'entry_price': entry_price,
                            'sweep_level': sweep_low_level,
                            'direction': 'LONG',
                            'stop': stop,
                            'or_high': or_high,
                            'or_low': or_low,
                            'fvg_top': fvg_top,
                            'fvg_bottom': fvg_bottom,
                        })
                        return setups  # One per session

        # Bearish FVG after sweep HIGH
        if sweep_high_level and b3_high < b1_low - 2:
            fvg_top = b1_low
            fvg_bottom = b3_high
            post_fvg_mask = sb_bars['timestamp'] > b2_ts
            post_fvg = sb_bars[post_fvg_mask]
            for _, bar in post_fvg.iterrows():
                if bar['high'] >= fvg_bottom and bar['close'] <= fvg_top:
                    entry_price = bar['close']
                    stop = or_high + 0.10 * (or_high - or_low)
                    risk = stop - entry_price
                    if 10 < risk < 200:
                        setups.append({
                            'model': 'SILVER_BULLET_SHORT',
                            'entry_idx': bar.name if hasattr(bar, 'name') else 0,
                            'entry_time': bar['timestamp'],
                            'entry_price': entry_price,
                            'sweep_level': sweep_high_level,
                            'direction': 'SHORT',
                            'stop': stop,
                            'or_high': or_high,
                            'or_low': or_low,
                            'fvg_top': fvg_top,
                            'fvg_bottom': fvg_bottom,
                        })
                        return setups

    return setups


def detect_or_reversal_v2(bars, levels_dict):
    """
    Improved OR Reversal with fixes:
    1. Add London H/L as valid sweep levels
    2. Check ALL levels and pick closest (not first match)
    3. Use dynamic proximity threshold based on overnight range
    """
    setups = []
    if len(bars) < 30:
        return setups

    # Find RTH start
    rth_start = None
    for i, bar in bars.iterrows():
        if bar['timestamp'].hour == 9 and bar['timestamp'].minute >= 30:
            rth_start = i
            break
        elif bar['timestamp'].hour >= 10:
            rth_start = i
            break
    if rth_start is None:
        return setups

    rth_bars = bars.loc[rth_start:].copy().reset_index(drop=True)
    if len(rth_bars) < 30:
        return setups

    or_bars = rth_bars.iloc[:15]
    or_high = or_bars['high'].max()
    or_low = or_bars['low'].min()
    or_mid = (or_high + or_low) / 2

    eor_bars = rth_bars.iloc[:30]
    eor_high = eor_bars['high'].max()
    eor_low = eor_bars['low'].min()

    # Dynamic threshold based on overnight range
    on_h = levels_dict.get('overnight_high', np.nan)
    on_l = levels_dict.get('overnight_low', np.nan)
    on_range = on_h - on_l if not np.isnan(on_h) and not np.isnan(on_l) else 180
    threshold = max(30, min(50, 0.15 * on_range))

    # All high levels (including London!)
    high_levels = []
    for lname in ['overnight_high', 'asia_high', 'london_high', 'pdh']:
        val = levels_dict.get(lname, np.nan)
        if not np.isnan(val):
            high_levels.append((lname, val, abs(eor_high - val)))

    # All low levels (including London!)
    low_levels = []
    for lname in ['overnight_low', 'asia_low', 'london_low', 'pdl']:
        val = levels_dict.get(lname, np.nan)
        if not np.isnan(val):
            low_levels.append((lname, val, abs(eor_low - val)))

    # Sort by distance, pick closest
    high_levels.sort(key=lambda x: x[2])
    low_levels.sort(key=lambda x: x[2])

    # Check SHORT setup (high near level + reversal down)
    if high_levels and high_levels[0][2] < threshold:
        best_high = high_levels[0]
        high_bar_idx = eor_bars['high'].idxmax()
        post_high = rth_bars.loc[high_bar_idx:]
        for j in range(1, min(30, len(post_high))):
            bar = post_high.iloc[j]
            if bar['close'] < or_mid:
                setups.append({
                    'model': 'OR_REVERSAL_V2_SHORT',
                    'entry_idx': post_high.index[j],
                    'entry_time': bar['timestamp'],
                    'entry_price': bar['close'],
                    'sweep_extreme': eor_high,
                    'sweep_level': best_high[0],
                    'level_dist': best_high[2],
                    'or_high': or_high,
                    'or_low': or_low,
                    'or_mid': or_mid,
                    'direction': 'SHORT',
                    'stop': eor_high + 0.15 * (eor_high - eor_low),
                    'threshold_used': threshold,
                })
                break

    # Check LONG setup (low near level + reversal up)
    if low_levels and low_levels[0][2] < threshold:
        best_low = low_levels[0]
        low_bar_idx = eor_bars['low'].idxmin()
        post_low = rth_bars.loc[low_bar_idx:]
        for j in range(1, min(30, len(post_low))):
            bar = post_low.iloc[j]
            if bar['close'] > or_mid:
                setups.append({
                    'model': 'OR_REVERSAL_V2_LONG',
                    'entry_idx': post_low.index[j],
                    'entry_time': bar['timestamp'],
                    'entry_price': bar['close'],
                    'sweep_extreme': eor_low,
                    'sweep_level': best_low[0],
                    'level_dist': best_low[2],
                    'or_high': or_high,
                    'or_low': or_low,
                    'or_mid': or_mid,
                    'direction': 'LONG',
                    'stop': eor_low - 0.15 * (eor_high - eor_low),
                    'threshold_used': threshold,
                })
                break

    return setups


def simulate_trade(nq_rth, setup, mnq_mult=2.0):
    """Simulate a single trade with 2R target and stop."""
    entry_price = setup['entry_price']
    entry_time = setup['entry_time']
    stop_price = setup['stop']
    direction = setup['direction']

    if direction == 'LONG':
        risk = entry_price - stop_price
    else:
        risk = stop_price - entry_price

    if risk <= 0 or risk > 200:
        return None

    target_2r = entry_price + 2 * risk if direction == 'LONG' else entry_price - 2 * risk

    post_entry = nq_rth[nq_rth['timestamp'] > entry_time]
    exit_price = None
    exit_reason = None
    mfe = 0
    mae = 0

    for _, bar in post_entry.iterrows():
        if direction == 'LONG':
            mfe = max(mfe, bar['high'] - entry_price)
            mae = min(mae, bar['low'] - entry_price)
            if bar['low'] <= stop_price:
                exit_price, exit_reason = stop_price, 'STOP'
                break
            if bar['high'] >= target_2r:
                exit_price, exit_reason = target_2r, 'TARGET_2R'
                break
        else:
            mfe = max(mfe, entry_price - bar['low'])
            mae = min(mae, entry_price - bar['high'])
            if bar['high'] >= stop_price:
                exit_price, exit_reason = stop_price, 'STOP'
                break
            if bar['low'] <= target_2r:
                exit_price, exit_reason = target_2r, 'TARGET_2R'
                break

    if exit_price is None and len(post_entry) > 0:
        last = post_entry.iloc[-1]
        exit_price, exit_reason = last['close'], 'EOD'

    if exit_price is None:
        return None

    pnl_pts = (exit_price - entry_price) if direction == 'LONG' else (entry_price - exit_price)
    pnl_dollar = pnl_pts * mnq_mult
    r_mult = pnl_pts / risk if risk > 0 else 0

    return {
        'pnl_pts': round(pnl_pts, 1),
        'pnl_dollar': round(pnl_dollar, 2),
        'r_multiple': round(r_mult, 2),
        'exit_reason': exit_reason,
        'exit_price': round(exit_price, 2),
        'risk_pts': round(risk, 1),
        'mfe_pts': round(mfe, 1),
        'mae_pts': round(mae, 1),
    }


def main():
    print('=' * 70)
    print('  OPENING RANGE v3: COVERAGE GAP ANALYSIS')
    print('=' * 70)

    nq = load_csv('NQ')
    es = load_csv('ES')
    ym = load_csv('YM')

    rth = filter_rth(nq)
    sessions = sorted(rth['session_date'].dt.date.unique())
    print(f'\n  Sessions: {len(sessions)}')

    # ================================================================
    print('\n' + '=' * 70)
    print('  STUDY A: SESSION-BY-SESSION GAP ANALYSIS')
    print('  Why does each session fail to produce an OR Reversal trade?')
    print('=' * 70)

    # First, run v1 detection to find which sessions HAVE trades
    from diagnostic_opening_range_smt import detect_opening_reversal

    sessions_with_v1 = set()
    sessions_with_v1_optimized = set()

    for sd in sessions:
        sd_ts = pd.Timestamp(sd)
        levels = compute_session_levels(nq, sd)
        if np.isnan(levels.get('overnight_high', np.nan)):
            continue

        mask = (
            (nq['timestamp'] >= sd_ts + timedelta(hours=9)) &
            (nq['timestamp'] <= sd_ts + timedelta(hours=11))
        )
        nq_entry = nq[mask].copy().reset_index(drop=True)
        if len(nq_entry) < 15:
            continue

        setups = detect_opening_reversal(nq_entry, levels)
        if setups:
            sessions_with_v1.add(sd)
            # Check if any are overnight sweep + VWAP aligned
            for s in setups:
                if s.get('sweep_level', '') in ['overnight_high', 'overnight_low']:
                    if 'vwap' in nq_entry.columns:
                        idx = s['entry_idx']
                        if idx < len(nq_entry):
                            vw = nq_entry.iloc[idx]['vwap']
                            ep = s['entry_price']
                            if not np.isnan(vw):
                                if (s['direction'] == 'LONG' and ep <= vw + 20) or \
                                   (s['direction'] == 'SHORT' and ep >= vw - 20):
                                    sessions_with_v1_optimized.add(sd)

    no_trade_sessions = [sd for sd in sessions if sd not in sessions_with_v1]

    print(f'\n  Sessions WITH v1 raw trades: {len(sessions_with_v1)}/{len(sessions)}')
    print(f'  Sessions WITH v1 optimized trades: {len(sessions_with_v1_optimized)}/{len(sessions)}')
    print(f'  Sessions with ZERO v1 trades: {len(no_trade_sessions)}/{len(sessions)}')

    # Classify each no-trade session
    gap_reasons = {}
    print(f'\n  --- NO-TRADE SESSION ANALYSIS ({len(no_trade_sessions)} sessions) ---\n')
    print(f'  {"Date":<12s} {"Reason":<30s} {"Closest H":<20s} {"Dist":>5s} {"Closest L":<20s} {"Dist":>5s} {"OR Rng":>7s}')
    print(f'  {"-"*12} {"-"*30} {"-"*20} {"-"*5} {"-"*20} {"-"*5} {"-"*7}')

    for sd in no_trade_sessions:
        levels = compute_session_levels(nq, sd)
        gap = classify_session_gap(nq, sd, levels)
        reason = gap['reason']
        gap_reasons[reason] = gap_reasons.get(reason, 0) + 1

        ch = gap.get('closest_high', '-')
        ch_d = gap.get('closest_high_dist', 0)
        cl = gap.get('closest_low', '-')
        cl_d = gap.get('closest_low_dist', 0)
        or_r = gap.get('or_range', 0)

        print(f'  {str(sd):<12s} {reason:<30s} {str(ch):<20s} {ch_d:>5.0f} {str(cl):<20s} {cl_d:>5.0f} {or_r:>7.0f}')

    print(f'\n  --- FAILURE REASON SUMMARY ---')
    for reason, count in sorted(gap_reasons.items(), key=lambda x: -x[1]):
        print(f'  {reason:<35s}: {count:3d} sessions ({100*count/len(no_trade_sessions):.0f}%)')

    # ================================================================
    print('\n' + '=' * 70)
    print('  STUDY B: IMPROVED OR REVERSAL V2')
    print('  (London H/L, closest-level priority, dynamic threshold)')
    print('=' * 70)

    v2_trades = []
    for sd in sessions:
        sd_ts = pd.Timestamp(sd)
        levels = compute_session_levels(nq, sd)
        if np.isnan(levels.get('overnight_high', np.nan)):
            continue

        mask = (
            (nq['timestamp'] >= sd_ts + timedelta(hours=9)) &
            (nq['timestamp'] <= sd_ts + timedelta(hours=11))
        )
        nq_entry = nq[mask].copy().reset_index(drop=True)
        if len(nq_entry) < 15:
            continue

        rth_mask = (
            (nq['timestamp'] >= sd_ts + timedelta(hours=9, minutes=30)) &
            (nq['timestamp'] <= sd_ts + timedelta(hours=16))
        )
        nq_rth = nq[rth_mask].copy().reset_index(drop=True)
        if len(nq_rth) < 30:
            continue

        setups = detect_or_reversal_v2(nq_entry, levels)
        seen = set()
        for setup in setups:
            if setup['model'] in seen:
                continue
            seen.add(setup['model'])
            result = simulate_trade(nq_rth, setup)
            if result:
                trade = {
                    'session_date': sd,
                    'model': setup['model'],
                    'direction': setup['direction'],
                    'level_swept': setup['sweep_level'],
                    'level_dist': round(setup['level_dist'], 1),
                    'threshold_used': round(setup['threshold_used'], 1),
                    'entry_price': round(setup['entry_price'], 2),
                    'entry_time': setup['entry_time'],
                }
                trade.update(result)
                v2_trades.append(trade)

    v2_df = pd.DataFrame(v2_trades) if v2_trades else pd.DataFrame()
    if len(v2_df) > 0:
        print(f'\n  OR Reversal V2: {len(v2_df)} trades across {v2_df["session_date"].nunique()} sessions')
        print_metrics(calc_metrics(v2_df, 'OR_REVERSAL_V2'), '  ')

        # Breakdown by level type
        print('\n  --- By Level Swept ---')
        for level in v2_df['level_swept'].unique():
            sub = v2_df[v2_df['level_swept'] == level]
            m = calc_metrics(sub, level)
            if m and m['trades'] >= 2:
                print_metrics(m, '  ')

        # New levels that v1 missed
        new_levels = v2_df[v2_df['level_swept'].isin(['london_high', 'london_low'])]
        if len(new_levels) > 0:
            print(f'\n  NEW from London levels: {len(new_levels)} trades')
            print_metrics(calc_metrics(new_levels, 'London H/L'), '  ')

        # Dynamic threshold recoveries
        dyn_recoveries = v2_df[v2_df['level_dist'] >= 30]
        if len(dyn_recoveries) > 0:
            print(f'\n  Dynamic threshold recoveries (>30pt): {len(dyn_recoveries)} trades')
            print_metrics(calc_metrics(dyn_recoveries, 'Dynamic threshold'), '  ')
    else:
        print('\n  No OR Reversal V2 trades.')

    # ================================================================
    print('\n' + '=' * 70)
    print('  STUDY C: OR BREAKOUT CONTINUATION')
    print('  (For trend-from-open days that OR Reversal misses)')
    print('=' * 70)

    breakout_trades = []
    for sd in sessions:
        sd_ts = pd.Timestamp(sd)
        levels = compute_session_levels(nq, sd)
        if np.isnan(levels.get('overnight_high', np.nan)):
            continue

        rth_mask = (
            (nq['timestamp'] >= sd_ts + timedelta(hours=9, minutes=30)) &
            (nq['timestamp'] <= sd_ts + timedelta(hours=16))
        )
        nq_rth = nq[rth_mask].copy().reset_index(drop=True)
        if len(nq_rth) < 60:
            continue

        setups = detect_or_breakout(nq_rth, levels, threshold=40)
        seen = set()
        for setup in setups:
            if setup['model'] in seen:
                continue
            seen.add(setup['model'])
            result = simulate_trade(nq_rth, setup)
            if result:
                trade = {
                    'session_date': sd,
                    'model': setup['model'],
                    'direction': setup['direction'],
                    'level_swept': setup['sweep_level'],
                    'entry_price': round(setup['entry_price'], 2),
                    'entry_time': setup['entry_time'],
                }
                trade.update(result)
                breakout_trades.append(trade)

    bo_df = pd.DataFrame(breakout_trades) if breakout_trades else pd.DataFrame()
    if len(bo_df) > 0:
        print(f'\n  OR Breakout: {len(bo_df)} trades across {bo_df["session_date"].nunique()} sessions')
        print_metrics(calc_metrics(bo_df, 'OR_BREAKOUT'), '  ')

        for model in bo_df['model'].unique():
            sub = bo_df[bo_df['model'] == model]
            m = calc_metrics(sub, model)
            if m:
                print_metrics(m, '  ')

        # Which sessions did breakout catch that reversal missed?
        bo_sessions = set(bo_df['session_date'].unique())
        reversal_missed = bo_sessions - sessions_with_v1
        print(f'\n  Breakout catches {len(reversal_missed)} sessions that Reversal missed')
    else:
        print('\n  No OR Breakout trades.')

    # ================================================================
    print('\n' + '=' * 70)
    print('  STUDY D: SILVER BULLET (10:00-11:00 FVG ENTRY)')
    print('=' * 70)

    sb_trades = []
    for sd in sessions:
        sd_ts = pd.Timestamp(sd)
        levels = compute_session_levels(nq, sd)
        if np.isnan(levels.get('overnight_high', np.nan)):
            continue

        rth_mask = (
            (nq['timestamp'] >= sd_ts + timedelta(hours=9, minutes=30)) &
            (nq['timestamp'] <= sd_ts + timedelta(hours=16))
        )
        nq_rth = nq[rth_mask].copy().reset_index(drop=True)
        if len(nq_rth) < 60:
            continue

        setups = detect_silver_bullet(nq_rth, levels, nq, sd)
        for setup in setups:
            result = simulate_trade(nq_rth, setup)
            if result:
                trade = {
                    'session_date': sd,
                    'model': setup['model'],
                    'direction': setup['direction'],
                    'level_swept': setup['sweep_level'],
                    'entry_price': round(setup['entry_price'], 2),
                    'entry_time': setup['entry_time'],
                }
                trade.update(result)
                sb_trades.append(trade)

    sb_df = pd.DataFrame(sb_trades) if sb_trades else pd.DataFrame()
    if len(sb_df) > 0:
        print(f'\n  Silver Bullet: {len(sb_df)} trades across {sb_df["session_date"].nunique()} sessions')
        print_metrics(calc_metrics(sb_df, 'SILVER_BULLET'), '  ')

        for model in sb_df['model'].unique():
            sub = sb_df[sb_df['model'] == model]
            m = calc_metrics(sub, model)
            if m:
                print_metrics(m, '  ')

        sb_sessions = set(sb_df['session_date'].unique())
        sb_new = sb_sessions - sessions_with_v1
        print(f'\n  Silver Bullet catches {len(sb_new)} sessions that OR Reversal missed')
    else:
        print('\n  No Silver Bullet trades.')

    # ================================================================
    print('\n' + '=' * 70)
    print('  STUDY E: COMBINED MODEL COMPARISON')
    print('=' * 70)

    # Combine all models
    all_models = {}
    if len(v2_df) > 0:
        all_models['OR_REVERSAL_V2'] = v2_df
    if len(bo_df) > 0:
        all_models['OR_BREAKOUT'] = bo_df
    if len(sb_df) > 0:
        all_models['SILVER_BULLET'] = sb_df

    # Total unique sessions covered
    all_session_sets = {}
    for name, df in all_models.items():
        all_session_sets[name] = set(df['session_date'].unique())

    union_sessions = set()
    for s in all_session_sets.values():
        union_sessions |= s

    print(f'\n  Sessions covered:')
    print(f'    V1 OR Reversal (current):   {len(sessions_with_v1)}/{len(sessions)} ({100*len(sessions_with_v1)/len(sessions):.0f}%)')
    print(f'    V1 optimized (ON+VWAP):     {len(sessions_with_v1_optimized)}/{len(sessions)} ({100*len(sessions_with_v1_optimized)/len(sessions):.0f}%)')
    for name, sess in all_session_sets.items():
        print(f'    {name:<30s}: {len(sess)}/{len(sessions)} ({100*len(sess)/len(sessions):.0f}%)')
    print(f'    ALL MODELS COMBINED:        {len(union_sessions)}/{len(sessions)} ({100*len(union_sessions)/len(sessions):.0f}%)')
    print(f'    STILL UNCOVERED:            {len(sessions) - len(union_sessions)}/{len(sessions)} ({100*(len(sessions) - len(union_sessions))/len(sessions):.0f}%)')

    # Best-per-session: pick the model with highest expectancy for each session
    print(f'\n  --- BEST MODEL PER SESSION (pick highest expectancy) ---')
    combined = pd.concat([df for df in all_models.values()], ignore_index=True)
    if len(combined) > 0:
        # For sessions with multiple models, keep the one with highest pnl_dollar
        best = combined.sort_values('pnl_dollar', ascending=False).groupby('session_date').first().reset_index()
        print(f'\n  Best-per-session: {len(best)} trades across {best["session_date"].nunique()} sessions')
        print_metrics(calc_metrics(best, 'BEST_PER_SESSION'), '  ')

        # By model source
        for model in best['model'].unique():
            sub = best[best['model'] == model]
            print(f'    {model}: {len(sub)} sessions selected')

    # ================================================================
    print('\n' + '=' * 70)
    print('  SUMMARY: PORTFOLIO IMPACT')
    print('=' * 70)

    print(f'\n  Existing portfolio: 52 intraday + 20 OR Reversal = 72 trades, ~$17,513')
    if len(combined) > 0:
        best_net = best['pnl_dollar'].sum()
        best_trades = len(best)
        best_wr = 100 * sum(best['pnl_dollar'] > 0) / len(best)
        best_exp = best_net / best_trades
        print(f'  V3 best-per-session: {best_trades} OR trades, {best_wr:.0f}% WR, ${best_exp:.0f}/trade, ${best_net:+,.0f} net')
        print(f'  Potential combined: 52 intraday + {best_trades} OR = {52 + best_trades} trades, ~${13706 + best_net:+,.0f}')

    print('\n  DONE')


if __name__ == '__main__':
    main()
