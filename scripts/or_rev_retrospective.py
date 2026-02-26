"""
OR Reversal Retrospective Analysis -- Last 90 Days (Nov 25 2025 - Feb 24 2026)

For each session in the window:
  1. RTH range (high - low)
  2. Did OR Rev fire? Result?
  3. If NO signal, was there a big directional move we missed?
  4. For each missed big move (RTH range > 200pts), compute ideal entry/stop/target

Goals:
  - Identify 300+ pt move days with no signal
  - Propose additional entry models
  - Characterize big moves we miss

Usage:
    E:/anaconda/python.exe scripts/or_rev_retrospective.py
"""

import sys
from pathlib import Path
from datetime import time as _time, datetime, timedelta
from typing import Optional, Tuple, List, Dict
import pandas as pd
import numpy as np

# Add project root
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_order_flow_features

# ── Strategy constants (from strategy/or_reversal.py) ──
OR_BARS = 15
EOR_BARS = 30
SWEEP_THRESHOLD_RATIO = 0.17
VWAP_ALIGNED_RATIO = 0.17
OR_STOP_BUFFER = 0.15
MIN_RISK_RATIO = 0.03
MAX_RISK_RATIO = 1.3
DRIVE_THRESHOLD = 0.4

MNQ_POINT_VALUE = 0.50  # $/pt for 1 MNQ


# ═══════════════════════════════════════════════════════════════
# HELPER: find closest swept level (same logic as strategy)
# ═══════════════════════════════════════════════════════════════
def _find_closest_swept_level(eor_extreme, candidates, sweep_threshold, eor_range):
    best_level = None
    best_name = None
    best_dist = float('inf')
    for name, lvl in candidates:
        if lvl is None:
            continue
        dist = abs(eor_extreme - lvl)
        if dist < sweep_threshold and dist <= eor_range and dist < best_dist:
            best_dist = dist
            best_level = lvl
            best_name = name
    return best_level, best_name


# ═══════════════════════════════════════════════════════════════
# HELPER: compute overnight levels from full df
# ═══════════════════════════════════════════════════════════════
def compute_overnight_levels(full_df, session_date_ts):
    """Compute ON/London/PDH/PDL for a session date."""
    sd = pd.Timestamp(session_date_ts)
    prev_day = sd - timedelta(days=1)
    if prev_day.weekday() == 5:
        prev_day -= timedelta(days=1)
    elif prev_day.weekday() == 6:
        prev_day -= timedelta(days=2)

    ts = full_df['timestamp']
    levels = {}

    # Overnight: 18:00 prev day to 9:29 session day
    on_mask = (ts >= prev_day + timedelta(hours=18)) & (ts < sd + timedelta(hours=9, minutes=30))
    on_bars = full_df[on_mask]
    if len(on_bars) > 0:
        levels['overnight_high'] = float(on_bars['high'].max())
        levels['overnight_low'] = float(on_bars['low'].min())
    else:
        levels['overnight_high'] = None
        levels['overnight_low'] = None

    # London: 02:00-05:00
    london_mask = (ts >= sd + timedelta(hours=2)) & (ts < sd + timedelta(hours=5))
    london = full_df[london_mask]
    if len(london) > 0:
        levels['london_high'] = float(london['high'].max())
        levels['london_low'] = float(london['low'].min())
    else:
        levels['london_high'] = None
        levels['london_low'] = None

    # PDH/PDL
    prior_rth_mask = (ts >= prev_day + timedelta(hours=9, minutes=30)) & (ts <= prev_day + timedelta(hours=16))
    prior_rth = full_df[prior_rth_mask]
    if len(prior_rth) > 0:
        levels['pdh'] = float(prior_rth['high'].max())
        levels['pdl'] = float(prior_rth['low'].min())
    else:
        levels['pdh'] = None
        levels['pdl'] = None

    return levels


# ═══════════════════════════════════════════════════════════════
# CORE: Diagnose OR Rev for a single session
# ═══════════════════════════════════════════════════════════════
def diagnose_session(session_df, ib_bars, levels, trade_log_row=None):
    """
    Full diagnostic for one session.
    Returns a dict with all analysis fields.
    """
    result = {}

    # --- RTH range ---
    rth_high = float(session_df['high'].max())
    rth_low = float(session_df['low'].min())
    rth_range = rth_high - rth_low
    result['rth_high'] = rth_high
    result['rth_low'] = rth_low
    result['rth_range'] = rth_range

    # --- IB stats ---
    ib_high = float(ib_bars['high'].max()) if len(ib_bars) >= 60 else None
    ib_low = float(ib_bars['low'].min()) if len(ib_bars) >= 60 else None
    ib_range = ib_high - ib_low if ib_high and ib_low else 0
    result['ib_high'] = ib_high
    result['ib_low'] = ib_low
    result['ib_range'] = ib_range

    # --- IB Regime ---
    if ib_range < 100:
        result['ib_regime'] = 'LOW'
    elif ib_range < 150:
        result['ib_regime'] = 'MED'
    elif ib_range < 250:
        result['ib_regime'] = 'NORMAL'
    else:
        result['ib_regime'] = 'HIGH'

    if len(ib_bars) < EOR_BARS:
        result['signal_fired'] = False
        result['block_reason'] = 'INSUFFICIENT_BARS'
        return result

    # --- OR / EOR ---
    or_bars = ib_bars.iloc[:OR_BARS]
    or_high = float(or_bars['high'].max())
    or_low = float(or_bars['low'].min())
    or_mid = (or_high + or_low) / 2

    eor_bars = ib_bars.iloc[:EOR_BARS]
    eor_high = float(eor_bars['high'].max())
    eor_low = float(eor_bars['low'].min())
    eor_range = eor_high - eor_low

    result['or_high'] = or_high
    result['or_low'] = or_low
    result['or_mid'] = or_mid
    result['eor_high'] = eor_high
    result['eor_low'] = eor_low
    result['eor_range'] = eor_range

    if eor_range < 10:
        result['signal_fired'] = False
        result['block_reason'] = 'EOR_RANGE_TOO_SMALL'
        return result

    sweep_threshold = eor_range * SWEEP_THRESHOLD_RATIO
    vwap_threshold = eor_range * VWAP_ALIGNED_RATIO

    # --- Opening drive ---
    first_5 = ib_bars.iloc[:5]
    open_price = float(first_5.iloc[0]['open'])
    close_5th = float(first_5.iloc[4]['close'])
    drive_range = float(first_5['high'].max()) - float(first_5['low'].min())
    if drive_range > 0:
        drive_pct = (close_5th - open_price) / drive_range
    else:
        drive_pct = 0

    if drive_pct > DRIVE_THRESHOLD:
        opening_drive = 'DRIVE_UP'
    elif drive_pct < -DRIVE_THRESHOLD:
        opening_drive = 'DRIVE_DOWN'
    else:
        opening_drive = 'ROTATION'
    result['opening_drive'] = opening_drive
    result['drive_pct'] = drive_pct

    # --- Sweep detection ---
    on_high = levels.get('overnight_high')
    on_low = levels.get('overnight_low')
    pdh = levels.get('pdh')
    pdl = levels.get('pdl')
    ldn_high = levels.get('london_high')
    ldn_low = levels.get('london_low')

    if on_high is None or on_low is None:
        result['signal_fired'] = False
        result['block_reason'] = 'NO_OVERNIGHT_LEVELS'
        return result

    high_candidates = [('ON_HIGH', on_high)]
    if pdh: high_candidates.append(('PDH', pdh))
    if ldn_high: high_candidates.append(('LDN_HIGH', ldn_high))

    low_candidates = [('ON_LOW', on_low)]
    if pdl: low_candidates.append(('PDL', pdl))
    if ldn_low: low_candidates.append(('LDN_LOW', ldn_low))

    swept_high_level, swept_high_name = _find_closest_swept_level(
        eor_high, high_candidates, sweep_threshold, eor_range)
    swept_low_level, swept_low_name = _find_closest_swept_level(
        eor_low, low_candidates, sweep_threshold, eor_range)

    result['swept_high_level'] = swept_high_level
    result['swept_high_name'] = swept_high_name
    result['swept_low_level'] = swept_low_level
    result['swept_low_name'] = swept_low_name

    no_sweep = swept_high_level is None and swept_low_level is None
    result['no_sweep'] = no_sweep

    # Dual sweep: keep deeper
    if swept_high_level is not None and swept_low_level is not None:
        high_depth = eor_high - swept_high_level
        low_depth = swept_low_level - eor_low
        if high_depth >= low_depth:
            swept_low_level = None
            swept_low_name = None
        else:
            swept_high_level = None
            swept_high_name = None

    # --- CVD / delta computation ---
    if 'delta' in ib_bars.columns:
        deltas = ib_bars['delta'].fillna(0)
    elif 'vol_delta' in ib_bars.columns:
        deltas = ib_bars['vol_delta'].fillna(0)
    else:
        deltas = pd.Series(0, index=ib_bars.index)
    cvd_series = deltas.cumsum()

    high_bar_idx = eor_bars['high'].idxmax()
    low_bar_idx = eor_bars['low'].idxmin()

    # ═══════════════════════════════════════════════════════════
    # Try SHORT setup (same logic as strategy)
    # ═══════════════════════════════════════════════════════════
    short_block_reasons = []
    short_signal = None

    if swept_high_level is not None:
        if opening_drive == 'DRIVE_UP':
            short_block_reasons.append('FADE_FILTER (DRIVE_UP + SHORT)')
        else:
            cvd_at_extreme = cvd_series.loc[high_bar_idx]
            post_high = ib_bars.loc[high_bar_idx:]
            found_entry = False
            for j in range(1, min(30, len(post_high))):
                bar = post_high.iloc[j]
                price = float(bar['close'])

                if price >= or_mid:
                    continue

                # VWAP check
                vwap = bar.get('vwap', np.nan)
                if pd.isna(vwap) or abs(price - vwap) > vwap_threshold:
                    if not found_entry:
                        short_block_reasons.append(f'VWAP_NOT_ALIGNED (price={price:.1f}, vwap={vwap if not pd.isna(vwap) else "N/A"})')
                    continue

                # Delta / CVD check
                delta = bar.get('delta', bar.get('vol_delta', 0))
                if pd.isna(delta): delta = 0
                cvd_at_entry = cvd_series.loc[post_high.index[j]]
                cvd_declining = cvd_at_entry < cvd_at_extreme
                if delta >= 0 and not cvd_declining:
                    if not found_entry:
                        short_block_reasons.append(f'DELTA_CVD_FILTER (delta={delta:.0f}, cvd_decline={cvd_declining})')
                    continue

                # Risk check
                stop = swept_high_level + eor_range * OR_STOP_BUFFER
                risk = stop - price
                max_risk = eor_range * MAX_RISK_RATIO
                if risk < eor_range * MIN_RISK_RATIO or risk > max_risk:
                    if not found_entry:
                        short_block_reasons.append(f'RISK_OUT_OF_RANGE (risk={risk:.1f}, min={eor_range*MIN_RISK_RATIO:.1f}, max={max_risk:.1f})')
                    continue

                target = price - 2 * risk
                short_signal = {
                    'direction': 'SHORT',
                    'entry': price,
                    'stop': stop,
                    'target': target,
                    'risk': risk,
                    'level_swept': swept_high_name,
                }
                found_entry = True
                break

            if not found_entry and not short_block_reasons:
                short_block_reasons.append('NO_REVERSAL_PAST_OR_MID')

    # ═══════════════════════════════════════════════════════════
    # Try LONG setup (same logic as strategy)
    # ═══════════════════════════════════════════════════════════
    long_block_reasons = []
    long_signal = None

    if swept_low_level is not None and short_signal is None:
        if opening_drive == 'DRIVE_DOWN':
            long_block_reasons.append('FADE_FILTER (DRIVE_DOWN + LONG)')
        else:
            cvd_at_extreme = cvd_series.loc[low_bar_idx]
            post_low = ib_bars.loc[low_bar_idx:]
            found_entry = False
            for j in range(1, min(30, len(post_low))):
                bar = post_low.iloc[j]
                price = float(bar['close'])

                if price <= or_mid:
                    continue

                vwap = bar.get('vwap', np.nan)
                if pd.isna(vwap) or abs(price - vwap) > vwap_threshold:
                    if not found_entry:
                        long_block_reasons.append(f'VWAP_NOT_ALIGNED (price={price:.1f})')
                    continue

                delta = bar.get('delta', bar.get('vol_delta', 0))
                if pd.isna(delta): delta = 0
                cvd_at_entry = cvd_series.loc[post_low.index[j]]
                cvd_rising = cvd_at_entry > cvd_at_extreme
                if delta <= 0 and not cvd_rising:
                    if not found_entry:
                        long_block_reasons.append(f'DELTA_CVD_FILTER (delta={delta:.0f}, cvd_rise={cvd_rising})')
                    continue

                stop = swept_low_level - eor_range * OR_STOP_BUFFER
                risk = price - stop
                max_risk = eor_range * MAX_RISK_RATIO
                if risk < eor_range * MIN_RISK_RATIO or risk > max_risk:
                    if not found_entry:
                        long_block_reasons.append(f'RISK_OUT_OF_RANGE (risk={risk:.1f})')
                    continue

                target = price + 2 * risk
                long_signal = {
                    'direction': 'LONG',
                    'entry': price,
                    'stop': stop,
                    'target': target,
                    'risk': risk,
                    'level_swept': swept_low_name,
                }
                found_entry = True
                break

            if not found_entry and not long_block_reasons:
                long_block_reasons.append('NO_REVERSAL_PAST_OR_MID')

    # --- Did signal fire? ---
    signal = short_signal or long_signal
    result['signal_fired'] = signal is not None

    if signal:
        result['signal_direction'] = signal['direction']
        result['signal_entry'] = signal['entry']
        result['signal_stop'] = signal['stop']
        result['signal_target'] = signal['target']
        result['signal_risk'] = signal['risk']
        result['signal_level_swept'] = signal['level_swept']
    else:
        all_blocks = short_block_reasons + long_block_reasons
        if no_sweep:
            all_blocks.insert(0, 'NO_SWEEP')
        result['block_reason'] = ' | '.join(all_blocks) if all_blocks else 'UNKNOWN'

    # --- Regime filter check ---
    if signal and result['ib_regime'] == 'LOW':
        result['regime_blocked'] = True
        result['block_reason'] = 'REGIME_FILTER (LOW IB < 100)'
        result['signal_fired'] = False  # Would have been filtered
    else:
        result['regime_blocked'] = False

    # --- Actual move after EOR extreme ---
    # Compute max favorable excursion from EOR extreme (reversal direction)
    post_eor = session_df.iloc[EOR_BARS:]  # All bars after first 30 min

    if len(post_eor) > 0:
        # After high sweep, favorable = downside from eor_high
        mfe_from_high = eor_high - float(post_eor['low'].min())
        # After low sweep, favorable = upside from eor_low
        mfe_from_low = float(post_eor['high'].max()) - eor_low

        result['mfe_from_high'] = mfe_from_high
        result['mfe_from_low'] = mfe_from_low

        # Which direction had the bigger move?
        if mfe_from_high > mfe_from_low:
            result['dominant_reversal'] = 'DOWN'
            result['dominant_reversal_pts'] = mfe_from_high
        else:
            result['dominant_reversal'] = 'UP'
            result['dominant_reversal_pts'] = mfe_from_low
    else:
        result['mfe_from_high'] = 0
        result['mfe_from_low'] = 0
        result['dominant_reversal'] = 'NONE'
        result['dominant_reversal_pts'] = 0

    # --- "Relaxed entry" simulation for missed sessions ---
    # Just require: (1) sweep of any level, (2) reversal past OR mid, NO delta/CVD/VWAP filter
    if not result['signal_fired']:
        relaxed = _simulate_relaxed_entry(ib_bars, session_df, eor_high, eor_low, eor_range,
                                           or_mid, opening_drive, swept_high_level, swept_high_name,
                                           swept_low_level, swept_low_name, high_candidates, low_candidates,
                                           sweep_threshold)
        result.update(relaxed)

    # --- "Second chance" entry: scan bars 30-120 (10:00-11:30) for late reversal ---
    second_chance = _scan_second_chance(session_df, ib_bars, eor_high, eor_low, eor_range, or_mid)
    result.update(second_chance)

    return result


def _simulate_relaxed_entry(ib_bars, session_df, eor_high, eor_low, eor_range,
                             or_mid, opening_drive, swept_high_level, swept_high_name,
                             swept_low_level, swept_low_name, high_candidates, low_candidates,
                             sweep_threshold):
    """Simulate a relaxed OR Rev entry: sweep + reversal past mid, no delta/VWAP filter."""
    result = {}
    result['relaxed_signal'] = False

    # Try wider sweep threshold (2x)
    wide_threshold = eor_range * SWEEP_THRESHOLD_RATIO * 2.0

    sh_level, sh_name = _find_closest_swept_level(eor_high, high_candidates, wide_threshold, eor_range * 1.5)
    sl_level, sl_name = _find_closest_swept_level(eor_low, low_candidates, wide_threshold, eor_range * 1.5)

    if sh_level is None and sl_level is None:
        result['relaxed_block'] = 'NO_SWEEP_EVEN_WIDE'
        return result

    # Dual sweep: deeper wins
    if sh_level is not None and sl_level is not None:
        high_depth = eor_high - sh_level
        low_depth = sl_level - eor_low
        if high_depth >= low_depth:
            sl_level = None
            sl_name = None
        else:
            sh_level = None
            sh_name = None

    # Try SHORT (relaxed)
    if sh_level is not None:
        # Still exclude fades
        if opening_drive != 'DRIVE_UP':
            post_eor = session_df.iloc[EOR_BARS:]
            for j in range(len(post_eor)):
                bar = post_eor.iloc[j]
                price = float(bar['close'])
                if price < or_mid:
                    stop = sh_level + eor_range * OR_STOP_BUFFER
                    risk = stop - price
                    if risk > 0 and risk < eor_range * 2:
                        target = price - 2 * risk
                        # Simulate P&L: what actually happened?
                        remaining = post_eor.iloc[j+1:]
                        pnl, exit_reason = _simulate_trade(price, stop, target, 'SHORT', remaining)
                        result['relaxed_signal'] = True
                        result['relaxed_direction'] = 'SHORT'
                        result['relaxed_entry'] = price
                        result['relaxed_stop'] = stop
                        result['relaxed_target'] = target
                        result['relaxed_risk'] = risk
                        result['relaxed_pnl_pts'] = pnl
                        result['relaxed_pnl_dollar'] = pnl * MNQ_POINT_VALUE
                        result['relaxed_exit_reason'] = exit_reason
                        result['relaxed_level_swept'] = sh_name
                        return result
                    break

    # Try LONG (relaxed)
    if sl_level is not None:
        if opening_drive != 'DRIVE_DOWN':
            post_eor = session_df.iloc[EOR_BARS:]
            for j in range(len(post_eor)):
                bar = post_eor.iloc[j]
                price = float(bar['close'])
                if price > or_mid:
                    stop = sl_level - eor_range * OR_STOP_BUFFER
                    risk = price - stop
                    if risk > 0 and risk < eor_range * 2:
                        target = price + 2 * risk
                        remaining = post_eor.iloc[j+1:]
                        pnl, exit_reason = _simulate_trade(price, stop, target, 'LONG', remaining)
                        result['relaxed_signal'] = True
                        result['relaxed_direction'] = 'LONG'
                        result['relaxed_entry'] = price
                        result['relaxed_stop'] = stop
                        result['relaxed_target'] = target
                        result['relaxed_risk'] = risk
                        result['relaxed_pnl_pts'] = pnl
                        result['relaxed_pnl_dollar'] = pnl * MNQ_POINT_VALUE
                        result['relaxed_exit_reason'] = exit_reason
                        result['relaxed_level_swept'] = sl_name
                        return result

    result['relaxed_block'] = 'NO_REVERSAL_PAST_MID_EVEN_RELAXED'
    return result


def _scan_second_chance(session_df, ib_bars, eor_high, eor_low, eor_range, or_mid):
    """Scan bars 60-120 (10:30-11:30) for 'second chance' reversal entry."""
    result = {}
    result['second_chance'] = False

    if len(session_df) < 120:
        return result

    # Look for reversal from EOR extreme in the 10:30-11:30 window
    window = session_df.iloc[60:120]  # bars 60-120

    # Check if price returns to EOR high zone then reverses down
    best_entry = None

    for i in range(len(window)):
        bar = window.iloc[i]
        h = float(bar['high'])
        l = float(bar['low'])
        c = float(bar['close'])

        # Retest of EOR high and reversal
        if h >= eor_high - eor_range * 0.1 and c < eor_high - eor_range * 0.15:
            stop = h + eor_range * 0.1
            risk = stop - c
            if risk > 0 and risk < eor_range:
                target = c - 2 * risk
                remaining = session_df.iloc[60 + i + 1:]
                pnl, exit_reason = _simulate_trade(c, stop, target, 'SHORT', remaining)
                result['second_chance'] = True
                result['sc_direction'] = 'SHORT'
                result['sc_entry'] = c
                result['sc_stop'] = stop
                result['sc_target'] = target
                result['sc_risk'] = risk
                result['sc_pnl_pts'] = pnl
                result['sc_pnl_dollar'] = pnl * MNQ_POINT_VALUE
                result['sc_exit_reason'] = exit_reason
                result['sc_bar_idx'] = 60 + i
                return result

        # Retest of EOR low and reversal
        if l <= eor_low + eor_range * 0.1 and c > eor_low + eor_range * 0.15:
            stop = l - eor_range * 0.1
            risk = c - stop
            if risk > 0 and risk < eor_range:
                target = c + 2 * risk
                remaining = session_df.iloc[60 + i + 1:]
                pnl, exit_reason = _simulate_trade(c, stop, target, 'LONG', remaining)
                result['second_chance'] = True
                result['sc_direction'] = 'LONG'
                result['sc_entry'] = c
                result['sc_stop'] = stop
                result['sc_target'] = target
                result['sc_risk'] = risk
                result['sc_pnl_pts'] = pnl
                result['sc_pnl_dollar'] = pnl * MNQ_POINT_VALUE
                result['sc_exit_reason'] = exit_reason
                result['sc_bar_idx'] = 60 + i
                return result

    return result


def _simulate_trade(entry, stop, target, direction, remaining_bars):
    """Simulate a trade on remaining bars. Returns (pnl_pts, exit_reason)."""
    for i in range(len(remaining_bars)):
        bar = remaining_bars.iloc[i]
        h = float(bar['high'])
        l = float(bar['low'])
        bar_time = bar.get('time', None)
        if bar_time is None and 'timestamp' in bar.index:
            bar_time = bar['timestamp'].time() if hasattr(bar['timestamp'], 'time') else None

        if direction == 'SHORT':
            # Check stop first (conservative)
            if h >= stop:
                return -(stop - entry), 'STOP'
            if l <= target:
                return entry - target, 'TARGET'
        else:  # LONG
            if l <= stop:
                return -(entry - stop), 'STOP'
            if h >= target:
                return target - entry, 'TARGET'

        # EOD cutoff at 15:30
        if bar_time and bar_time >= _time(15, 30):
            c = float(bar['close'])
            if direction == 'SHORT':
                return entry - c, 'EOD'
            else:
                return c - entry, 'EOD'

    # End of data
    c = float(remaining_bars.iloc[-1]['close']) if len(remaining_bars) > 0 else entry
    if direction == 'SHORT':
        return entry - c, 'EOD'
    else:
        return c - entry, 'EOD'


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 80)
    print("OR REVERSAL RETROSPECTIVE -- Last 90 Days (Nov 25 2025 - Feb 24 2026)")
    print("=" * 80)

    # --- Load data ---
    full_df = load_csv('NQ')
    full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])

    # Compute order flow features for delta
    full_df = compute_order_flow_features(full_df)

    rth_df = filter_rth(full_df)

    if 'session_date' not in rth_df.columns:
        rth_df['session_date'] = rth_df['timestamp'].dt.date

    # --- Date window ---
    start_date = pd.Timestamp('2025-11-25').date()
    end_date = pd.Timestamp('2026-02-24').date()

    sessions = sorted(rth_df['session_date'].unique())
    sessions = [s for s in sessions if start_date <= (s if not isinstance(s, pd.Timestamp) else s.date()) <= end_date]
    # Handle Timestamp vs date
    sessions_ts = []
    for s in sessions:
        if isinstance(s, pd.Timestamp):
            sessions_ts.append(s.date())
        else:
            sessions_ts.append(s)
    sessions = sessions_ts

    print(f"\nSessions in window: {len(sessions)}")
    print(f"  From: {sessions[0]}  To: {sessions[-1]}")

    # --- Load trade log ---
    trade_log_path = project_root / 'output' / 'trade_log.csv'
    trade_log = None
    if trade_log_path.exists():
        trade_log = pd.read_csv(trade_log_path)
        trade_log['session_date'] = pd.to_datetime(trade_log['session_date']).dt.date
        trade_log = trade_log[trade_log['strategy_name'] == 'Opening Range Rev']
        print(f"  OR Rev trades in log: {len(trade_log)}")
    else:
        print("  WARNING: No trade_log.csv found, using strategy-level diagnostics only")

    # --- Process each session ---
    all_results = []

    for session_date in sessions:
        sd_ts = pd.Timestamp(session_date)
        mask = rth_df['session_date'] == session_date
        if not mask.any():
            # Try with Timestamp
            mask = rth_df['session_date'] == sd_ts
        session_bars = rth_df[mask].sort_values('timestamp')

        if len(session_bars) < 60:
            continue

        ib_bars = session_bars.head(60)

        # Compute overnight levels
        levels = compute_overnight_levels(full_df, sd_ts)

        # Get trade log row if exists
        tl_row = None
        if trade_log is not None:
            tl_matches = trade_log[trade_log['session_date'] == session_date]
            if len(tl_matches) > 0:
                tl_row = tl_matches.iloc[0]

        # Diagnose
        diag = diagnose_session(session_bars, ib_bars, levels, tl_row)
        diag['date'] = session_date

        # Annotate with actual trade result
        if tl_row is not None:
            diag['actual_trade'] = True
            diag['actual_direction'] = tl_row['direction']
            diag['actual_net_pnl'] = float(tl_row['net_pnl'])
            diag['actual_exit_reason'] = tl_row['exit_reason']
        else:
            diag['actual_trade'] = False

        all_results.append(diag)

    df_results = pd.DataFrame(all_results)

    # ═══════════════════════════════════════════════════════════
    # REPORT SECTION 1: Session-by-session summary
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("SECTION 1: SESSION-BY-SESSION SUMMARY (last 90 days)")
    print("=" * 100)
    print(f"\n{'Date':<12} {'RTH Rng':>8} {'IB Rng':>7} {'Regime':<7} {'Signal':>7} {'Trade?':>7} "
          f"{'Drive':<10} {'Dir':>5} {'P&L':>8} {'Block Reason'}")
    print("-" * 120)

    for _, row in df_results.iterrows():
        date = row['date']
        rth_rng = row['rth_range']
        ib_rng = row.get('ib_range', 0)
        regime = row.get('ib_regime', '?')
        sig = 'YES' if row['signal_fired'] else 'NO'
        trade = 'YES' if row.get('actual_trade', False) else 'NO'
        drive = row.get('opening_drive', '?')
        direction = row.get('signal_direction', row.get('actual_direction', '-'))
        pnl = f"${row['actual_net_pnl']:+.0f}" if row.get('actual_trade') else '-'
        block = row.get('block_reason', '-') if not row['signal_fired'] else '-'
        # Mark big move days
        marker = ' ***' if rth_rng >= 300 and not row.get('actual_trade', False) else ''
        marker = ' **' if rth_rng >= 200 and not row.get('actual_trade', False) and not marker else marker
        print(f"{str(date):<12} {rth_rng:>8.0f} {ib_rng:>7.0f} {regime:<7} {sig:>7} {trade:>7} "
              f"{drive:<10} {direction if direction else '-':>5} {pnl:>8} {block[:60]}{marker}")

    # ═══════════════════════════════════════════════════════════
    # REPORT SECTION 2: Big move days analysis
    # ═══════════════════════════════════════════════════════════
    print("\n\n" + "=" * 100)
    print("SECTION 2: BIG MOVE DAYS (RTH Range >= 250 pts)")
    print("=" * 100)

    big_days = df_results[df_results['rth_range'] >= 250].copy()
    big_days_traded = big_days[big_days['actual_trade'] == True]
    big_days_missed = big_days[big_days['actual_trade'] == False]

    total_big = len(big_days)
    traded_big = len(big_days_traded)
    missed_big = len(big_days_missed)

    print(f"\nTotal big move days (>=250 pts): {total_big}")
    print(f"  Captured (had a trade): {traded_big} ({traded_big/total_big*100:.0f}%)" if total_big else "")
    print(f"  Missed (no trade):      {missed_big} ({missed_big/total_big*100:.0f}%)" if total_big else "")

    if traded_big > 0:
        avg_pnl_traded = big_days_traded['actual_net_pnl'].mean()
        total_pnl_traded = big_days_traded['actual_net_pnl'].sum()
        print(f"  Avg P&L on captured: ${avg_pnl_traded:+.2f}")
        print(f"  Total P&L captured:  ${total_pnl_traded:+.2f}")

    # 300+ pt days
    very_big = df_results[df_results['rth_range'] >= 300]
    very_big_missed = very_big[very_big['actual_trade'] == False]
    print(f"\nTotal 300+ pt days: {len(very_big)}")
    print(f"  Missed: {len(very_big_missed)}")

    # ═══════════════════════════════════════════════════════════
    # REPORT SECTION 3: Top 10 Missed Opportunities
    # ═══════════════════════════════════════════════════════════
    print("\n\n" + "=" * 100)
    print("SECTION 3: TOP 10 BIGGEST MISSED OPPORTUNITIES")
    print("  (No OR Rev trade, ranked by RTH range)")
    print("=" * 100)

    missed = df_results[df_results['actual_trade'] == False].copy()
    missed = missed.sort_values('rth_range', ascending=False).head(10)

    for rank, (_, row) in enumerate(missed.iterrows(), 1):
        date = row['date']
        rth_range = row['rth_range']
        print(f"\n{'-'*80}")
        print(f"#{rank}  {date}  |  RTH Range: {rth_range:.0f} pts  |  IB: {row.get('ib_range',0):.0f} ({row.get('ib_regime','?')})")
        print(f"{'-'*80}")
        print(f"  RTH: High={row['rth_high']:.2f}  Low={row['rth_low']:.2f}")
        print(f"  EOR: High={row.get('eor_high','?')}  Low={row.get('eor_low','?')}  Range={row.get('eor_range','?')}")
        print(f"  Drive: {row.get('opening_drive', '?')} (pct={row.get('drive_pct', 0):.2f})")
        print(f"  Sweep High: {row.get('swept_high_name', 'None')} @ {row.get('swept_high_level', 'None')}")
        print(f"  Sweep Low:  {row.get('swept_low_name', 'None')} @ {row.get('swept_low_level', 'None')}")
        print(f"  Block Reason: {row.get('block_reason', '?')}")
        print(f"  MFE from EOR High (DOWN): {row.get('mfe_from_high', 0):.0f} pts = ${row.get('mfe_from_high', 0)*MNQ_POINT_VALUE:.0f}")
        print(f"  MFE from EOR Low (UP):    {row.get('mfe_from_low', 0):.0f} pts = ${row.get('mfe_from_low', 0)*MNQ_POINT_VALUE:.0f}")
        print(f"  Dominant reversal: {row.get('dominant_reversal', '?')} {row.get('dominant_reversal_pts', 0):.0f} pts")

        # Relaxed entry result
        if row.get('relaxed_signal', False):
            print(f"\n  >>> RELAXED ENTRY (no delta/VWAP filter):")
            print(f"      Direction: {row.get('relaxed_direction')}")
            print(f"      Entry: {row.get('relaxed_entry', 0):.2f}  Stop: {row.get('relaxed_stop', 0):.2f}  Target: {row.get('relaxed_target', 0):.2f}")
            print(f"      Risk: {row.get('relaxed_risk', 0):.0f} pts  |  P&L: {row.get('relaxed_pnl_pts', 0):+.0f} pts = ${row.get('relaxed_pnl_dollar', 0):+.0f}")
            print(f"      Exit: {row.get('relaxed_exit_reason', '?')}  |  Level: {row.get('relaxed_level_swept', '?')}")
        else:
            print(f"\n  >>> RELAXED ENTRY: Would not fire ({row.get('relaxed_block', '?')})")

        # Second chance
        if row.get('second_chance', False):
            print(f"\n  >>> SECOND CHANCE (10:30-11:30 retest):")
            print(f"      Direction: {row.get('sc_direction')}")
            print(f"      Entry: {row.get('sc_entry', 0):.2f}  Stop: {row.get('sc_stop', 0):.2f}  Target: {row.get('sc_target', 0):.2f}")
            print(f"      Risk: {row.get('sc_risk', 0):.0f} pts  |  P&L: {row.get('sc_pnl_pts', 0):+.0f} pts = ${row.get('sc_pnl_dollar', 0):+.0f}")
            print(f"      Exit: {row.get('sc_exit_reason', '?')}  |  Bar idx: {row.get('sc_bar_idx', '?')}")

        # What ideal entry would have been
        if not row.get('actual_trade') and rth_range >= 200:
            dom = row.get('dominant_reversal', 'NONE')
            if dom == 'DOWN':
                ideal_entry = row.get('eor_high', 0)
                ideal_target = row['rth_low']
                ideal_stop = ideal_entry + row.get('eor_range', 100) * 0.15
                ideal_pnl = (ideal_entry - ideal_target) * MNQ_POINT_VALUE
                ideal_risk = (ideal_stop - ideal_entry) * MNQ_POINT_VALUE
            elif dom == 'UP':
                ideal_entry = row.get('eor_low', 0)
                ideal_target = row['rth_high']
                ideal_stop = ideal_entry - row.get('eor_range', 100) * 0.15
                ideal_pnl = (ideal_target - ideal_entry) * MNQ_POINT_VALUE
                ideal_risk = (ideal_entry - ideal_stop) * MNQ_POINT_VALUE
            else:
                ideal_pnl = 0
                ideal_risk = 0
                ideal_entry = 0
                ideal_target = 0
                ideal_stop = 0

            if ideal_pnl > 0:
                print(f"\n  >>> IDEAL (hindsight) at 1 MNQ:")
                print(f"      {'SHORT' if dom == 'DOWN' else 'LONG'} @ {ideal_entry:.2f} → {ideal_target:.2f}")
                print(f"      Stop: {ideal_stop:.2f}")
                print(f"      Max profit: ${ideal_pnl:.0f}  Risk: ${ideal_risk:.0f}  R:R = {ideal_pnl/ideal_risk:.1f}" if ideal_risk > 0 else "")

    # ═══════════════════════════════════════════════════════════
    # REPORT SECTION 4: Block reason distribution
    # ═══════════════════════════════════════════════════════════
    print("\n\n" + "=" * 100)
    print("SECTION 4: WHY SIGNALS DON'T FIRE (Block Reason Distribution)")
    print("=" * 100)

    no_signal = df_results[df_results['signal_fired'] == False].copy()
    if len(no_signal) > 0:
        # Parse primary block reason
        def primary_reason(r):
            reasons = str(r).split(' | ')
            for reason in reasons:
                if 'SWEEP' in reason:
                    return 'NO_SWEEP'
                elif 'FADE' in reason:
                    return 'FADE_FILTER'
                elif 'DELTA' in reason or 'CVD' in reason:
                    return 'DELTA_CVD_FILTER'
                elif 'VWAP' in reason:
                    return 'VWAP_FILTER'
                elif 'RISK' in reason:
                    return 'RISK_OUT_OF_RANGE'
                elif 'REVERSAL' in reason or 'NO_REVERSAL' in reason:
                    return 'NO_REVERSAL_PAST_MID'
                elif 'REGIME' in reason:
                    return 'REGIME_FILTER'
                elif 'OVERNIGHT' in reason:
                    return 'NO_OVERNIGHT_DATA'
                elif 'INSUFFICIENT' in reason:
                    return 'INSUFFICIENT_BARS'
            return 'OTHER'

        no_signal['primary_block'] = no_signal['block_reason'].apply(primary_reason)
        block_counts = no_signal['primary_block'].value_counts()

        print(f"\nTotal sessions without OR Rev signal: {len(no_signal)}/{len(df_results)}")
        print(f"\n{'Block Reason':<30} {'Count':>6} {'%':>6} {'Avg RTH Range':>14}")
        print("-" * 60)
        for reason, count in block_counts.items():
            pct = count / len(no_signal) * 100
            avg_rng = no_signal[no_signal['primary_block'] == reason]['rth_range'].mean()
            print(f"{reason:<30} {count:>6} {pct:>5.1f}% {avg_rng:>13.0f}")

    # ═══════════════════════════════════════════════════════════
    # REPORT SECTION 5: FADE-blocked days analysis
    # ═══════════════════════════════════════════════════════════
    print("\n\n" + "=" * 100)
    print("SECTION 5: FADE-BLOCKED DAYS -- Do They Continue or Reverse?")
    print("=" * 100)

    fade_blocked = no_signal[no_signal['block_reason'].str.contains('FADE', na=False)].copy() if len(no_signal) > 0 else pd.DataFrame()

    if len(fade_blocked) > 0:
        print(f"\nTotal FADE-blocked sessions: {len(fade_blocked)}")

        # For each fade-blocked day, check if the move continued or reversed
        continues = 0
        reverses = 0
        reverse_sizes = []
        continue_sizes = []

        print(f"\n{'Date':<12} {'Drive':<10} {'Would-be':>10} {'RTH Rng':>8} {'MFE Rev':>8} {'MFE Cont':>9} {'Verdict':<12}")
        print("-" * 80)

        for _, row in fade_blocked.iterrows():
            drive = row.get('opening_drive', '?')
            # If DRIVE_UP + SHORT blocked, check if price actually went down (reverse) or up (continue)
            if 'DRIVE_UP' in str(row.get('block_reason', '')):
                would_be = 'SHORT'
                mfe_rev = row.get('mfe_from_high', 0)  # how far DOWN from EOR high
                mfe_cont = row.get('mfe_from_low', 0)  # how far UP
            else:
                would_be = 'LONG'
                mfe_rev = row.get('mfe_from_low', 0)
                mfe_cont = row.get('mfe_from_high', 0)

            if mfe_rev > mfe_cont * 0.5 and mfe_rev > 50:
                verdict = 'REVERSED'
                reverses += 1
                reverse_sizes.append(mfe_rev)
            else:
                verdict = 'CONTINUED'
                continues += 1
                continue_sizes.append(mfe_cont)

            print(f"{str(row['date']):<12} {drive:<10} {would_be:>10} {row['rth_range']:>8.0f} "
                  f"{mfe_rev:>8.0f} {mfe_cont:>9.0f} {verdict:<12}")

        print(f"\nSummary:")
        print(f"  Continued in drive direction: {continues} ({continues/(continues+reverses)*100:.0f}%)")
        print(f"  Reversed (we would have been right): {reverses} ({reverses/(continues+reverses)*100:.0f}%)")
        if reverse_sizes:
            print(f"  Avg reversal size when right: {np.mean(reverse_sizes):.0f} pts")
        if continue_sizes:
            print(f"  Avg continuation when wrong:  {np.mean(continue_sizes):.0f} pts")
    else:
        print("\nNo FADE-blocked sessions in this window.")

    # ═══════════════════════════════════════════════════════════
    # REPORT SECTION 6: Second Chance Entries
    # ═══════════════════════════════════════════════════════════
    print("\n\n" + "=" * 100)
    print("SECTION 6: SECOND CHANCE ENTRIES (10:30-11:30 EOR Retest)")
    print("=" * 100)

    sc_df = df_results[df_results['second_chance'] == True].copy()
    sc_no_trade = sc_df[sc_df['actual_trade'] == False]

    print(f"\nTotal sessions with second chance entry: {len(sc_df)}")
    print(f"  Of which we had NO original trade: {len(sc_no_trade)}")

    if len(sc_no_trade) > 0:
        sc_wins = sc_no_trade[sc_no_trade['sc_pnl_pts'] > 0]
        sc_losses = sc_no_trade[sc_no_trade['sc_pnl_pts'] <= 0]
        print(f"  Winners: {len(sc_wins)}, Losers: {len(sc_losses)}")
        if len(sc_no_trade) > 0:
            wr = len(sc_wins) / len(sc_no_trade) * 100
            total_pnl = sc_no_trade['sc_pnl_dollar'].sum()
            avg_pnl = sc_no_trade['sc_pnl_dollar'].mean()
            print(f"  Win Rate: {wr:.1f}%")
            print(f"  Total P&L: ${total_pnl:+.0f}")
            print(f"  Avg P&L: ${avg_pnl:+.0f}")

        print(f"\n{'Date':<12} {'Dir':>5} {'Entry':>10} {'Stop':>10} {'Target':>10} {'P&L':>8} {'Exit':<8}")
        print("-" * 70)
        for _, row in sc_no_trade.sort_values('sc_pnl_pts', ascending=False).iterrows():
            print(f"{str(row['date']):<12} {row.get('sc_direction','?'):>5} {row.get('sc_entry',0):>10.2f} "
                  f"{row.get('sc_stop',0):>10.2f} {row.get('sc_target',0):>10.2f} "
                  f"${row.get('sc_pnl_dollar',0):>+7.0f} {row.get('sc_exit_reason','?'):<8}")

    # ═══════════════════════════════════════════════════════════
    # REPORT SECTION 7: Relaxed Entry Simulation
    # ═══════════════════════════════════════════════════════════
    print("\n\n" + "=" * 100)
    print("SECTION 7: RELAXED ENTRY SIMULATION (sweep + reversal, NO delta/VWAP filter)")
    print("=" * 100)

    relaxed = df_results[(df_results['relaxed_signal'] == True) & (df_results['actual_trade'] == False)].copy()

    if len(relaxed) > 0:
        r_wins = relaxed[relaxed['relaxed_pnl_pts'] > 0]
        r_losses = relaxed[relaxed['relaxed_pnl_pts'] <= 0]
        total_pnl = relaxed['relaxed_pnl_dollar'].sum()
        wr = len(r_wins) / len(relaxed) * 100

        print(f"\nRelaxed entries that would have fired (no original trade): {len(relaxed)}")
        print(f"  Winners: {len(r_wins)}, Losers: {len(r_losses)}")
        print(f"  Win Rate: {wr:.1f}%")
        print(f"  Total P&L: ${total_pnl:+.0f}")
        print(f"  Avg P&L: ${total_pnl/len(relaxed):+.0f}")

        print(f"\n{'Date':<12} {'Dir':>5} {'Entry':>10} {'Stop':>10} {'Target':>10} {'P&L pts':>8} {'P&L $':>8} {'Exit':<8} {'Level'}")
        print("-" * 95)
        for _, row in relaxed.sort_values('relaxed_pnl_pts', ascending=False).iterrows():
            print(f"{str(row['date']):<12} {row.get('relaxed_direction','?'):>5} {row.get('relaxed_entry',0):>10.2f} "
                  f"{row.get('relaxed_stop',0):>10.2f} {row.get('relaxed_target',0):>10.2f} "
                  f"{row.get('relaxed_pnl_pts',0):>+8.0f} ${row.get('relaxed_pnl_dollar',0):>+7.0f} "
                  f"{row.get('relaxed_exit_reason','?'):<8} {row.get('relaxed_level_swept','?')}")
    else:
        print("\nNo additional relaxed entries would fire.")

    # ═══════════════════════════════════════════════════════════
    # REPORT SECTION 8: Overall Capture Rate by RTH Range Bucket
    # ═══════════════════════════════════════════════════════════
    print("\n\n" + "=" * 100)
    print("SECTION 8: CAPTURE RATE BY RTH RANGE BUCKET")
    print("=" * 100)

    def bucket(rng):
        if rng < 100: return '<100'
        if rng < 150: return '100-150'
        if rng < 200: return '150-200'
        if rng < 250: return '200-250'
        if rng < 300: return '250-300'
        if rng < 400: return '300-400'
        return '400+'

    df_results['rth_bucket'] = df_results['rth_range'].apply(bucket)
    buckets = ['<100', '100-150', '150-200', '200-250', '250-300', '300-400', '400+']

    print(f"\n{'Bucket':<12} {'Total':>6} {'Traded':>7} {'Missed':>7} {'Capture%':>9} {'Avg RTH':>8} {'Signal%':>8}")
    print("-" * 65)
    for b in buckets:
        subset = df_results[df_results['rth_bucket'] == b]
        if len(subset) == 0:
            continue
        total = len(subset)
        traded = len(subset[subset['actual_trade'] == True])
        missed = total - traded
        sig_fired = len(subset[subset['signal_fired'] == True])
        avg_rng = subset['rth_range'].mean()
        print(f"{b:<12} {total:>6} {traded:>7} {missed:>7} {traded/total*100:>8.0f}% {avg_rng:>8.0f} {sig_fired/total*100:>7.0f}%")

    # ═══════════════════════════════════════════════════════════
    # REPORT SECTION 9: Key Findings and Recommendations
    # ═══════════════════════════════════════════════════════════
    print("\n\n" + "=" * 100)
    print("SECTION 9: KEY FINDINGS AND RECOMMENDATIONS")
    print("=" * 100)

    total_sessions = len(df_results)
    signal_sessions = len(df_results[df_results['signal_fired'] == True])
    traded_sessions = len(df_results[df_results['actual_trade'] == True])

    print(f"\n--- Summary Statistics (90-day window) ---")
    print(f"Total sessions:                {total_sessions}")
    print(f"Sessions with OR Rev signal:   {signal_sessions} ({signal_sessions/total_sessions*100:.0f}%)")
    print(f"Sessions with OR Rev trade:    {traded_sessions} ({traded_sessions/total_sessions*100:.0f}%)")
    print(f"Big move days (>=250):         {len(big_days)}")
    print(f"  Captured:                    {traded_big} ({traded_big/total_big*100:.0f}%)" if total_big else "")
    print(f"  Missed:                      {missed_big} ({missed_big/total_big*100:.0f}%)" if total_big else "")

    # Block reason summary for big days
    big_missed = df_results[(df_results['rth_range'] >= 250) & (df_results['actual_trade'] == False)]
    if len(big_missed) > 0:
        print(f"\n--- Block reasons on big move days (>=250 pts) ---")
        for _, row in big_missed.iterrows():
            print(f"  {row['date']}: {row.get('block_reason', '?')[:80]}")

    # Fade analysis summary
    if len(fade_blocked) > 0 and (continues + reverses) > 0:
        print(f"\n--- Fade Filter Analysis ---")
        print(f"Fade-blocked days where reversal DID happen: {reverses}/{continues+reverses} ({reverses/(continues+reverses)*100:.0f}%)")
        if reverses > continues:
            print(f"  FINDING: Fade filter is too aggressive -- majority of blocked trades would have won!")
            print(f"  RECOMMENDATION: Consider allowing fades when IB regime is NORMAL or HIGH")
        else:
            print(f"  FINDING: Fade filter is correctly blocking -- majority continued in drive direction")

    # Relaxed entry value
    if len(relaxed) > 0:
        r_total = relaxed['relaxed_pnl_dollar'].sum()
        r_wr = len(relaxed[relaxed['relaxed_pnl_pts'] > 0]) / len(relaxed) * 100
        print(f"\n--- Relaxed Entry Value ---")
        print(f"Additional trades from removing delta/VWAP filters: {len(relaxed)}")
        print(f"Win rate: {r_wr:.0f}%  |  Total P&L: ${r_total:+.0f}")
        if r_total > 0 and r_wr >= 50:
            print(f"  FINDING: Relaxed filters ADD value -- consider a 'wide net' mode")
        else:
            print(f"  FINDING: Relaxed filters do NOT add value -- current filters are optimal")

    # Second chance value
    if len(sc_no_trade) > 0:
        sc_total = sc_no_trade['sc_pnl_dollar'].sum()
        sc_wr = len(sc_no_trade[sc_no_trade['sc_pnl_pts'] > 0]) / len(sc_no_trade) * 100
        print(f"\n--- Second Chance Entry Value ---")
        print(f"Late reversals (10:30-11:30) on missed days: {len(sc_no_trade)}")
        print(f"Win rate: {sc_wr:.0f}%  |  Total P&L: ${sc_total:+.0f}")
        if sc_total > 0 and sc_wr >= 55:
            print(f"  FINDING: Second chance entries ADD value -- worth implementing as a new model")
        else:
            print(f"  FINDING: Second chance entries are marginal -- not worth the added complexity")

    print("\n" + "=" * 100)
    print("END OF RETROSPECTIVE")
    print("=" * 100)


if __name__ == '__main__':
    main()
