"""
Diagnostic: Analyze ALL 62 sessions for new entry model opportunities.

User's key insight: "Not all trend days work, price might attempt to trade above or
below IBH/IBL and fail, becomes a balance day. Reclaim and reclaim failure - reclaim
failure IBH or IBL are juicy. VWAP failure or VWAP reclaim are both trades, but VWAP
reclaim failure is even better. Morph might happen neutral/balance becomes breakout in PM."

Entry Models to Detect:
1. IBH RECLAIM FAILURE (long): Price breaks above IBH, comes back inside IB,
   then fails to reclaim IBH -> short signal (failed breakout)
   IBH RECLAIM FAILURE (short = fade the failure): When IBH reclaim fails,
   price reverting to IB mid = long from IBL or short from failed IBH

2. IBL RECLAIM FAILURE: Price breaks below IBL, comes back inside IB,
   then fails to reclaim IBL -> fade back to IB mid (LONG from above IBL)

3. VWAP RECLAIM: Price drops below VWAP, then reclaims back above -> LONG
4. VWAP FAILURE: Price at VWAP rejects, can't get through -> SHORT (or fade direction)
5. VWAP RECLAIM FAILURE: Price tries to reclaim VWAP from below, fails -> SHORT
   (This is the "even better trade" per user)

6. PM MORPH: Session starts as neutral/balance, then breaks out in PM (after 13:00)
   -> enter on the breakout with momentum

7. EDGE FADE (B-Day/P-Day/Neutral): Fade from IB edges toward IB midpoint
   - Long at IBL touch with rejection
   - Short at IBH touch with rejection (risky on NQ)

For each session we track:
- Price action relative to IBH/IBL throughout the day
- VWAP interactions (reclaims, failures, reclaim failures)
- Failed breakout patterns
- Edge touches and rejections
- PM morph breakouts
- Potential P&L for each entry model
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from config.instruments import get_instrument
from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from config.constants import (
    IB_BARS_1MIN, ACCEPTANCE_MIN_BARS, PM_SESSION_START, EOD_CUTOFF,
)
from strategy.day_type import classify_trend_strength, classify_day_type


@dataclass
class EntryOpportunity:
    """A potential trade entry detected by the diagnostic."""
    session_date: str
    bar_time: str
    bar_idx: int
    entry_type: str         # e.g. 'IBH_RECLAIM_FAILURE', 'VWAP_RECLAIM', etc.
    direction: str          # 'LONG' or 'SHORT'
    entry_price: float
    stop_price: float
    target_price: float
    day_type: str
    # MFE/MAE computed post-entry
    mfe_points: float = 0.0    # Max favorable excursion
    mae_points: float = 0.0    # Max adverse excursion
    eod_price: float = 0.0
    eod_pnl_points: float = 0.0
    hit_target: bool = False
    hit_stop: bool = False
    delta_at_entry: float = 0.0
    volume_spike_at_entry: float = 0.0
    of_quality: int = 0         # 2-of-3 quality gate score


def compute_mfe_mae(entries: List[EntryOpportunity], post_entry_bars: pd.DataFrame,
                    eod_price: float):
    """Compute MFE, MAE, and EOD P&L for each entry opportunity."""
    for entry in entries:
        remaining_bars = post_entry_bars[post_entry_bars.index >= entry.bar_idx]
        if len(remaining_bars) == 0:
            continue

        entry.eod_price = eod_price

        if entry.direction == 'LONG':
            entry.mfe_points = remaining_bars['high'].max() - entry.entry_price
            entry.mae_points = entry.entry_price - remaining_bars['low'].min()
            entry.eod_pnl_points = eod_price - entry.entry_price
            entry.hit_target = remaining_bars['high'].max() >= entry.target_price
            entry.hit_stop = remaining_bars['low'].min() <= entry.stop_price
        else:  # SHORT
            entry.mfe_points = entry.entry_price - remaining_bars['low'].min()
            entry.mae_points = remaining_bars['high'].max() - entry.entry_price
            entry.eod_pnl_points = entry.entry_price - eod_price
            entry.hit_target = remaining_bars['low'].min() <= entry.target_price
            entry.hit_stop = remaining_bars['high'].max() >= entry.stop_price


def analyze_session(session_df: pd.DataFrame, session_date: str) -> dict:
    """Analyze a single session for all new entry model opportunities."""

    ib_df = session_df.head(IB_BARS_1MIN)
    ib_high = ib_df['high'].max()
    ib_low = ib_df['low'].min()
    ib_range = ib_high - ib_low
    ib_mid = (ib_high + ib_low) / 2

    if ib_range <= 0:
        return {'session_date': session_date, 'entries': [], 'ib_range': 0}

    post_ib = session_df.iloc[IB_BARS_1MIN:].copy()
    if len(post_ib) == 0:
        return {'session_date': session_date, 'entries': [], 'ib_range': ib_range}

    post_ib = post_ib.reset_index(drop=True)
    eod_price = post_ib['close'].iloc[-1]

    entries: List[EntryOpportunity] = []

    # State tracking
    was_above_ibh = False      # Price was above IBH at some point
    was_below_ibl = False      # Price was below IBL at some point
    returned_inside = False    # After being above IBH, came back inside IB
    ibh_reclaim_attempt = False
    ibl_reclaim_attempt = False

    # VWAP state tracking
    was_above_vwap = True      # Start assuming above (post-IB usually near VWAP)
    vwap_cross_below = False   # Price crossed below VWAP
    vwap_reclaim_attempt = False

    # Edge touch tracking
    ibh_touch_count = 0
    ibl_touch_count = 0
    last_ibh_touch_bar = -30
    last_ibl_touch_bar = -30

    # PM morph tracking
    am_high = ib_high
    am_low = ib_low
    pm_breakout_detected = False

    # Track consecutive bars for acceptance/reclaim
    consecutive_above_ibh = 0
    consecutive_below_ibl = 0
    consecutive_inside = 0

    # IBH reclaim failure tracking
    ibh_break_bar = -999       # When price first broke above IBH
    ibh_return_bar = -999      # When price returned inside IB
    ibh_reclaim_fail_bar = -999

    # IBL break tracking
    ibl_break_bar = -999
    ibl_return_bar = -999

    # Cooldowns for entries
    last_entry_bar = -30
    ENTRY_COOLDOWN = 15

    for i in range(len(post_ib)):
        bar = post_ib.iloc[i]
        ts = bar.get('timestamp')
        bar_time = ts.time() if hasattr(ts, 'time') else None

        if bar_time and bar_time >= time(15, 25):
            break  # Don't enter in last 5 min

        current_price = bar['close']
        delta = bar.get('delta', 0) if not pd.isna(bar.get('delta', 0)) else 0
        vol_spike = bar.get('volume_spike', 1.0) if not pd.isna(bar.get('volume_spike', 1.0)) else 1.0
        delta_pctl = bar.get('delta_percentile', 50) if not pd.isna(bar.get('delta_percentile', 50)) else 50
        imbalance = bar.get('imbalance_ratio', 1.0) if not pd.isna(bar.get('imbalance_ratio', 1.0)) else 1.0
        vwap = bar.get('vwap')
        if vwap is None or pd.isna(vwap):
            vwap = ib_mid  # fallback

        # Compute day type at this bar
        if current_price > ib_high:
            ib_dir = 'BULL'
            ext = (current_price - ib_mid) / ib_range
        elif current_price < ib_low:
            ib_dir = 'BEAR'
            ext = (ib_mid - current_price) / ib_range
        else:
            ib_dir = 'INSIDE'
            ext = 0.0

        strength = classify_trend_strength(ext)
        day_type = classify_day_type(ib_high, ib_low, current_price, ib_dir, strength)

        # OF quality score
        of_quality = sum([
            delta_pctl >= 60,
            imbalance > 1.0,
            vol_spike >= 1.0,
        ])

        # Track AM high/low (before 13:00)
        if bar_time and bar_time < PM_SESSION_START:
            am_high = max(am_high, bar['high'])
            am_low = min(am_low, bar['low'])

        # === Track price relative to IBH ===
        if current_price > ib_high:
            consecutive_above_ibh += 1
            consecutive_inside = 0
            if not was_above_ibh:
                was_above_ibh = True
                ibh_break_bar = i
            if returned_inside:
                # Price reclaimed IBH after being inside
                ibh_reclaim_attempt = True
                returned_inside = False
        elif current_price < ib_low:
            consecutive_below_ibl += 1
            consecutive_above_ibh = 0
            consecutive_inside = 0
            if not was_below_ibl:
                was_below_ibl = True
                ibl_break_bar = i
        else:
            # Inside IB
            consecutive_inside += 1
            if was_above_ibh and consecutive_above_ibh > 0:
                returned_inside = True
                ibh_return_bar = i
            consecutive_above_ibh = 0
            consecutive_below_ibl = 0

        # === 1. IBH RECLAIM FAILURE ===
        # Pattern: Price broke above IBH, came back inside IB (2+ bars),
        # tried to reclaim IBH but FAILED (didn't sustain 2 bars above).
        # This is a powerful short/fade signal.
        if (returned_inside and consecutive_inside >= 2 and
            bar['high'] >= ib_high - ib_range * 0.05 and  # Tests IBH area
            current_price < ib_high and  # But closes below
            i - last_entry_bar >= ENTRY_COOLDOWN):

            # IBH reclaim failure -> fade back to IB mid (LONG from lower,
            # or SHORT if price is near IBH)
            # The trade is: SHORT from near IBH, target IB mid
            if current_price > ib_mid:
                stop = ib_high + ib_range * 0.15
                target = ib_mid
                entry = EntryOpportunity(
                    session_date=session_date,
                    bar_time=str(bar_time),
                    bar_idx=i,
                    entry_type='IBH_RECLAIM_FAILURE',
                    direction='SHORT',
                    entry_price=current_price,
                    stop_price=stop,
                    target_price=target,
                    day_type=day_type.value,
                    delta_at_entry=delta,
                    volume_spike_at_entry=vol_spike,
                    of_quality=of_quality,
                )
                entries.append(entry)
                last_entry_bar = i
                returned_inside = False  # Reset to avoid re-triggering

        # === 2. IBL TOUCH FADE (expanded B-Day logic for all day types) ===
        # Fade at IBL: any day type where price touches IBL and rejects
        # Not just b_day - also p_day, neutral days
        if (bar['low'] <= ib_low + ib_range * 0.02 and  # Touch IBL area
            current_price > ib_low and  # Close above IBL (rejection)
            delta > 0 and  # Buyer presence
            i - last_entry_bar >= ENTRY_COOLDOWN):

            # Track touch count
            if i - last_ibl_touch_bar <= 5:
                ibl_touch_count += 1
            else:
                ibl_touch_count = 1
            last_ibl_touch_bar = i

            stop = ib_low - ib_range * 0.10
            target = ib_mid
            entry = EntryOpportunity(
                session_date=session_date,
                bar_time=str(bar_time),
                bar_idx=i,
                entry_type='IBL_FADE_EXPANDED',
                direction='LONG',
                entry_price=current_price,
                stop_price=stop,
                target_price=target,
                day_type=day_type.value,
                delta_at_entry=delta,
                volume_spike_at_entry=vol_spike,
                of_quality=of_quality,
            )
            entries.append(entry)
            last_entry_bar = i

        # === 3. IBH TOUCH FADE (test on NQ despite long bias concerns) ===
        # Track IBH touches for research
        if (bar['high'] >= ib_high - ib_range * 0.02 and  # Touch IBH area
            current_price < ib_high and  # Close below IBH (rejection)
            delta < 0 and  # Seller presence
            i - last_entry_bar >= ENTRY_COOLDOWN):

            if i - last_ibh_touch_bar <= 5:
                ibh_touch_count += 1
            else:
                ibh_touch_count = 1
            last_ibh_touch_bar = i

            stop = ib_high + ib_range * 0.10
            target = ib_mid
            entry = EntryOpportunity(
                session_date=session_date,
                bar_time=str(bar_time),
                bar_idx=i,
                entry_type='IBH_FADE_EXPANDED',
                direction='SHORT',
                entry_price=current_price,
                stop_price=stop,
                target_price=target,
                day_type=day_type.value,
                delta_at_entry=delta,
                volume_spike_at_entry=vol_spike,
                of_quality=of_quality,
            )
            entries.append(entry)
            last_entry_bar = i

        # === 4. VWAP RECLAIM (price drops below VWAP, then reclaims above) ===
        if vwap:
            if current_price < vwap:
                vwap_cross_below = True
                was_above_vwap = False

            if (vwap_cross_below and current_price > vwap and
                bar['low'] <= vwap + ib_range * 0.05 and  # Was near/below VWAP this bar
                delta > 0 and
                i - last_entry_bar >= ENTRY_COOLDOWN):

                stop = vwap - ib_range * 0.30
                target = vwap + ib_range * 0.80
                entry = EntryOpportunity(
                    session_date=session_date,
                    bar_time=str(bar_time),
                    bar_idx=i,
                    entry_type='VWAP_RECLAIM',
                    direction='LONG',
                    entry_price=current_price,
                    stop_price=stop,
                    target_price=target,
                    day_type=day_type.value,
                    delta_at_entry=delta,
                    volume_spike_at_entry=vol_spike,
                    of_quality=of_quality,
                )
                entries.append(entry)
                last_entry_bar = i
                vwap_cross_below = False  # Reset

        # === 5. VWAP REJECTION (price approaches VWAP from below, fails to reclaim) ===
        if vwap:
            if (not was_above_vwap and
                bar['high'] >= vwap - ib_range * 0.02 and  # Tests VWAP
                current_price < vwap and  # But fails to close above
                delta < 0 and  # Sellers reject
                i - last_entry_bar >= ENTRY_COOLDOWN):

                stop = vwap + ib_range * 0.20
                target = vwap - ib_range * 0.60
                entry = EntryOpportunity(
                    session_date=session_date,
                    bar_time=str(bar_time),
                    bar_idx=i,
                    entry_type='VWAP_RECLAIM_FAILURE',
                    direction='SHORT',
                    entry_price=current_price,
                    stop_price=stop,
                    target_price=target,
                    day_type=day_type.value,
                    delta_at_entry=delta,
                    volume_spike_at_entry=vol_spike,
                    of_quality=of_quality,
                )
                entries.append(entry)
                last_entry_bar = i

        # === 6. VWAP FAILURE (price at VWAP from above, breaks below) ===
        if vwap:
            if (was_above_vwap and
                bar['low'] <= vwap + ib_range * 0.02 and  # Tests VWAP from above
                current_price < vwap - ib_range * 0.01 and  # Breaks below
                delta < 0 and
                i - last_entry_bar >= ENTRY_COOLDOWN):

                was_above_vwap = False
                vwap_cross_below = True
                # Don't take this as entry yet - wait for VWAP reclaim failure
                # This is informational tracking

        # === 7. PM MORPH BREAKOUT ===
        # Session was inside IB in AM, then breaks out in PM
        if (bar_time and bar_time >= PM_SESSION_START and
            not pm_breakout_detected and
            i - last_entry_bar >= ENTRY_COOLDOWN):

            # Bullish PM morph: breaks above AM high
            if current_price > am_high + ib_range * 0.05 and delta > 0:
                stop = am_high - ib_range * 0.15
                target = current_price + ib_range * 0.80
                entry = EntryOpportunity(
                    session_date=session_date,
                    bar_time=str(bar_time),
                    bar_idx=i,
                    entry_type='PM_MORPH_BULL',
                    direction='LONG',
                    entry_price=current_price,
                    stop_price=stop,
                    target_price=target,
                    day_type=day_type.value,
                    delta_at_entry=delta,
                    volume_spike_at_entry=vol_spike,
                    of_quality=of_quality,
                )
                entries.append(entry)
                last_entry_bar = i
                pm_breakout_detected = True

            # Bearish PM morph: breaks below AM low
            elif current_price < am_low - ib_range * 0.05 and delta < 0:
                stop = am_low + ib_range * 0.15
                target = current_price - ib_range * 0.80
                entry = EntryOpportunity(
                    session_date=session_date,
                    bar_time=str(bar_time),
                    bar_idx=i,
                    entry_type='PM_MORPH_BEAR',
                    direction='SHORT',
                    entry_price=current_price,
                    stop_price=stop,
                    target_price=target,
                    day_type=day_type.value,
                    delta_at_entry=delta,
                    volume_spike_at_entry=vol_spike,
                    of_quality=of_quality,
                )
                entries.append(entry)
                last_entry_bar = i
                pm_breakout_detected = True

    # Compute MFE/MAE for all entries
    compute_mfe_mae(entries, post_ib, eod_price)

    return {
        'session_date': session_date,
        'entries': entries,
        'ib_range': ib_range,
        'ib_high': ib_high,
        'ib_low': ib_low,
        'ib_mid': ib_mid,
        'eod_price': eod_price,
        'was_above_ibh': was_above_ibh,
        'was_below_ibl': was_below_ibl,
        'ibh_touch_count': ibh_touch_count,
        'ibl_touch_count': ibl_touch_count,
    }


def simulate_entry(entry: EntryOpportunity, point_value: float = 2.0,
                   commission_rt: float = 0.62) -> Tuple[float, str]:
    """
    Simulate a single entry: which hits first - stop or target?
    Returns (net_pnl_points, exit_reason)
    For simplicity, use EOD exit if neither stop nor target hit.
    """
    if entry.hit_stop and entry.hit_target:
        # Both hit - conservative: assume stop hit first (worse case)
        # In reality would need bar-by-bar sim
        return (-(abs(entry.entry_price - entry.stop_price)), 'STOP')

    if entry.hit_target:
        if entry.direction == 'LONG':
            pts = entry.target_price - entry.entry_price
        else:
            pts = entry.entry_price - entry.target_price
        return (pts, 'TARGET')

    if entry.hit_stop:
        if entry.direction == 'LONG':
            pts = entry.stop_price - entry.entry_price
        else:
            pts = entry.entry_price - entry.stop_price  # negative
        return (pts, 'STOP')

    # EOD exit
    return (entry.eod_pnl_points, 'EOD')


def main():
    instrument = get_instrument('MNQ')
    point_value = instrument.point_value  # $2 for MNQ

    df = load_csv('NQ')
    df = filter_rth(df)
    df = compute_all_features(df)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    if 'session_date' not in df.columns:
        df['session_date'] = df['timestamp'].dt.date

    sessions = sorted(df['session_date'].unique())

    print(f"\n{'='*130}")
    print(f"  NEW ENTRY MODEL DIAGNOSTIC: 62 Sessions")
    print(f"  Detecting: Reclaim Failures, VWAP Reclaim/Failure, Edge Fades, PM Morphs")
    print(f"{'='*130}\n")

    all_entries: List[EntryOpportunity] = []
    session_summaries = []

    for session_date in sessions:
        session_df = df[df['session_date'] == session_date].copy()
        if len(session_df) < IB_BARS_1MIN:
            continue

        result = analyze_session(session_df, str(session_date))
        all_entries.extend(result['entries'])

        if result['entries']:
            session_summaries.append(result)

    # =============================================
    # RESULTS BY ENTRY TYPE
    # =============================================
    entry_types = {}
    for e in all_entries:
        entry_types.setdefault(e.entry_type, []).append(e)

    print(f"\n{'='*130}")
    print(f"  RESULTS BY ENTRY TYPE")
    print(f"{'='*130}\n")

    print(f"{'Entry Type':<25} {'Count':>6} {'WR':>7} {'Avg MFE':>9} {'Avg MAE':>9} "
          f"{'Avg EOD':>9} {'Hit Tgt':>8} {'Hit Stop':>9} {'Net Pts':>10} {'Net $':>10}")
    print(f"{'-'*25} {'-'*6} {'-'*7} {'-'*9} {'-'*9} {'-'*9} {'-'*8} {'-'*9} {'-'*10} {'-'*10}")

    for etype in sorted(entry_types.keys()):
        elist = entry_types[etype]
        n = len(elist)

        # Simulate each entry
        results = [simulate_entry(e) for e in elist]
        pnl_pts = [r[0] for r in results]
        winners = sum(1 for p in pnl_pts if p > 0)
        wr = winners / n * 100 if n > 0 else 0

        avg_mfe = np.mean([e.mfe_points for e in elist])
        avg_mae = np.mean([e.mae_points for e in elist])
        avg_eod = np.mean([e.eod_pnl_points for e in elist])
        hit_tgt = sum(1 for e in elist if e.hit_target)
        hit_stop = sum(1 for e in elist if e.hit_stop)
        total_pts = sum(pnl_pts)
        total_dollars = total_pts * point_value

        print(f"{etype:<25} {n:>6} {wr:>6.1f}% {avg_mfe:>8.1f} {avg_mae:>8.1f} "
              f"{avg_eod:>8.1f} {hit_tgt:>7}/{n:<2} {hit_stop:>7}/{n:<2} "
              f"{total_pts:>9.1f} ${total_dollars:>8.0f}")

    # =============================================
    # RESULTS FILTERED BY OF QUALITY >= 2
    # =============================================
    print(f"\n{'='*130}")
    print(f"  RESULTS FILTERED BY ORDER FLOW QUALITY >= 2 (2-of-3 gate)")
    print(f"{'='*130}\n")

    print(f"{'Entry Type':<25} {'Count':>6} {'WR':>7} {'Avg MFE':>9} {'Avg MAE':>9} "
          f"{'Avg EOD':>9} {'Net Pts':>10} {'Net $':>10}")
    print(f"{'-'*25} {'-'*6} {'-'*7} {'-'*9} {'-'*9} {'-'*9} {'-'*10} {'-'*10}")

    for etype in sorted(entry_types.keys()):
        elist = [e for e in entry_types[etype] if e.of_quality >= 2]
        if not elist:
            continue
        n = len(elist)

        results = [simulate_entry(e) for e in elist]
        pnl_pts = [r[0] for r in results]
        winners = sum(1 for p in pnl_pts if p > 0)
        wr = winners / n * 100 if n > 0 else 0

        avg_mfe = np.mean([e.mfe_points for e in elist])
        avg_mae = np.mean([e.mae_points for e in elist])
        avg_eod = np.mean([e.eod_pnl_points for e in elist])
        total_pts = sum(pnl_pts)
        total_dollars = total_pts * point_value

        print(f"{etype:<25} {n:>6} {wr:>6.1f}% {avg_mfe:>8.1f} {avg_mae:>8.1f} "
              f"{avg_eod:>8.1f} {total_pts:>9.1f} ${total_dollars:>8.0f}")

    # =============================================
    # RESULTS BY DAY TYPE
    # =============================================
    print(f"\n{'='*130}")
    print(f"  RESULTS BY DAY TYPE (which days produce the best edge fades?)")
    print(f"{'='*130}\n")

    day_type_entries = {}
    for e in all_entries:
        day_type_entries.setdefault(e.day_type, []).append(e)

    print(f"{'Day Type':<20} {'Count':>6} {'WR':>7} {'Avg EOD':>9} {'Net Pts':>10} {'Net $':>10}")
    print(f"{'-'*20} {'-'*6} {'-'*7} {'-'*9} {'-'*10} {'-'*10}")

    for dtype in sorted(day_type_entries.keys()):
        elist = day_type_entries[dtype]
        n = len(elist)
        results = [simulate_entry(e) for e in elist]
        pnl_pts = [r[0] for r in results]
        winners = sum(1 for p in pnl_pts if p > 0)
        wr = winners / n * 100 if n > 0 else 0
        avg_eod = np.mean([e.eod_pnl_points for e in elist])
        total_pts = sum(pnl_pts)
        total_dollars = total_pts * point_value
        print(f"{dtype:<20} {n:>6} {wr:>6.1f}% {avg_eod:>8.1f} {total_pts:>9.1f} ${total_dollars:>8.0f}")

    # =============================================
    # SESSIONS WITH MOST OPPORTUNITIES
    # =============================================
    print(f"\n{'='*130}")
    print(f"  SESSION-LEVEL TRADE COUNT")
    print(f"{'='*130}\n")

    session_trade_counts = {}
    for e in all_entries:
        session_trade_counts[e.session_date] = session_trade_counts.get(e.session_date, 0) + 1

    total_sessions = len(sessions)
    sessions_with_trades = len(session_trade_counts)
    total_trades = len(all_entries)
    avg_trades = total_trades / total_sessions if total_sessions > 0 else 0

    print(f"Total sessions:           {total_sessions}")
    print(f"Sessions with trades:     {sessions_with_trades} ({sessions_with_trades/total_sessions*100:.0f}%)")
    print(f"Total trade opportunities: {total_trades}")
    print(f"Average trades/session:   {avg_trades:.1f}")
    print()

    # Distribution
    count_dist = {}
    for c in session_trade_counts.values():
        count_dist[c] = count_dist.get(c, 0) + 1

    print(f"Trade count distribution:")
    for tc in sorted(count_dist.keys()):
        print(f"  {tc} trades/session: {count_dist[tc]} sessions")
    no_trade_sessions = total_sessions - sessions_with_trades
    print(f"  0 trades/session: {no_trade_sessions} sessions")

    # =============================================
    # COMBINED PORTFOLIO: Best entries per session
    # =============================================
    print(f"\n{'='*130}")
    print(f"  COMBINED PORTFOLIO: Best entry models per session")
    print(f"{'='*130}\n")

    # Take best (highest EOD PnL) entry per session for each long/short
    best_entries = []
    for s in session_summaries:
        session_entries = s['entries']
        if not session_entries:
            continue
        # Take up to 3 best entries per session (diversified)
        sorted_entries = sorted(session_entries, key=lambda x: x.eod_pnl_points, reverse=True)
        for e in sorted_entries[:3]:
            best_entries.append(e)

    if best_entries:
        results = [simulate_entry(e) for e in best_entries]
        pnl_pts = [r[0] for r in results]
        winners = sum(1 for p in pnl_pts if p > 0)
        n = len(best_entries)
        wr = winners / n * 100

        total_pts = sum(pnl_pts)
        total_dollars = total_pts * point_value

        print(f"Best 3 entries/session: {n} trades across {len(session_summaries)} sessions")
        print(f"Win Rate: {wr:.1f}%")
        print(f"Total Net Points: {total_pts:.1f}")
        print(f"Total Net $: ${total_dollars:,.0f}")
        print(f"Avg per trade: {total_pts/n:.1f} pts (${total_dollars/n:,.0f})")

    # =============================================
    # LONG-ONLY PORTFOLIO (NQ long bias)
    # =============================================
    print(f"\n{'='*130}")
    print(f"  LONG-ONLY PORTFOLIO (leveraging NQ long bias)")
    print(f"{'='*130}\n")

    long_entries = [e for e in all_entries if e.direction == 'LONG']
    short_entries = [e for e in all_entries if e.direction == 'SHORT']

    for label, elist in [('LONG', long_entries), ('SHORT', short_entries)]:
        if not elist:
            print(f"  {label}: No entries")
            continue
        n = len(elist)
        results = [simulate_entry(e) for e in elist]
        pnl_pts = [r[0] for r in results]
        winners = sum(1 for p in pnl_pts if p > 0)
        wr = winners / n * 100
        total_pts = sum(pnl_pts)
        total_dollars = total_pts * point_value
        print(f"  {label}: {n} trades, {wr:.1f}% WR, {total_pts:.1f} pts (${total_dollars:,.0f})")

    # Per-type breakdown for longs
    print(f"\n  LONG entries by type:")
    for etype in sorted(set(e.entry_type for e in long_entries)):
        elist = [e for e in long_entries if e.entry_type == etype]
        n = len(elist)
        results = [simulate_entry(e) for e in elist]
        pnl_pts = [r[0] for r in results]
        winners = sum(1 for p in pnl_pts if p > 0)
        wr = winners / n * 100
        total_pts = sum(pnl_pts)
        total_dollars = total_pts * point_value
        print(f"    {etype:<25} {n:>4} trades, {wr:>5.1f}% WR, {total_pts:>8.1f} pts (${total_dollars:>7,.0f})")

    if short_entries:
        print(f"\n  SHORT entries by type:")
        for etype in sorted(set(e.entry_type for e in short_entries)):
            elist = [e for e in short_entries if e.entry_type == etype]
            n = len(elist)
            results = [simulate_entry(e) for e in elist]
            pnl_pts = [r[0] for r in results]
            winners = sum(1 for p in pnl_pts if p > 0)
            wr = winners / n * 100
            total_pts = sum(pnl_pts)
            total_dollars = total_pts * point_value
            print(f"    {etype:<25} {n:>4} trades, {wr:>5.1f}% WR, {total_pts:>8.1f} pts (${total_dollars:>7,.0f})")

    # =============================================
    # DETAILED SESSION LOG
    # =============================================
    print(f"\n{'='*130}")
    print(f"  DETAILED SESSION LOG")
    print(f"{'='*130}\n")

    for s in session_summaries:
        entries = s['entries']
        n = len(entries)
        results = [simulate_entry(e) for e in entries]
        pnl_pts = [r[0] for r in results]
        total_pts = sum(pnl_pts)
        total_dollars = total_pts * point_value
        winners = sum(1 for p in pnl_pts if p > 0)
        wr = winners / n * 100 if n > 0 else 0

        print(f"--- {s['session_date']} --- IB: {s['ib_range']:.0f} pts | "
              f"{n} entries | {wr:.0f}% WR | ${total_dollars:>+7,.0f}")

        for e, (pts, reason) in zip(entries, results):
            dollars = pts * point_value
            marker = "WIN" if pts > 0 else "LOSS"
            print(f"  {e.bar_time:>8} {e.entry_type:<25} {e.direction:>5} "
                  f"@{e.entry_price:>9.1f} -> {reason:<7} "
                  f"{pts:>+7.1f}pts (${dollars:>+7.0f}) "
                  f"MFE={e.mfe_points:>5.1f} MAE={e.mae_points:>5.1f} "
                  f"OF={e.of_quality} [{marker}]")
        print()


if __name__ == '__main__':
    main()
