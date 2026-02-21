"""
Diagnostic v2: Refined entry models with proper structural gates.

V1 findings:
- Raw signals are too noisy (337 entries, 32% WR overall)
- All shorts on NQ are negative (long bias confirmed again)
- IBL_FADE_EXPANDED is the only positive raw signal (+$1,117)
- Stops are too tight, too many false triggers

V2 refinements:
- LONG-ONLY focus (NQ long bias)
- Structural gates: require day_type context, multi-bar confirmation
- Wider stops with better R:R
- Quality filters: OF quality >= 2, delta confirmation
- Focus on entries that pass eval quickly: high frequency + positive expectancy

Entry models to test:
1. IBL_FADE_QUALITY: IBL touch + rejection + delta + multi-confirmation
   - Any day type (not just b_day) where price is inside or near IB
   - Stop: IBL - wider buffer (0.15x IB)
   - Target: IB mid (mean reversion)

2. VWAP_RECLAIM_QUALITY: Price drops below VWAP then reclaims with delta
   - Requires: was below VWAP for 3+ bars, then reclaims with delta > 0
   - OF quality >= 2
   - Stop: below recent swing low or VWAP - 0.3x IB
   - Target: VWAP + 0.6x IB

3. FAILED_BREAKDOWN: Price breaks below IBL, reverses back inside IB
   - More aggressive version of IBL fade
   - Requires: close below IBL for 2+ bars, then close back above IBL
   - This is the "IBL reclaim" the user described
   - Stop: session low - buffer
   - Target: IB mid

4. PM_VWAP_RECLAIM: Afternoon VWAP reclaim (after 13:00)
   - Price was below VWAP in AM, reclaims in PM
   - Requires delta confirmation + volume
   - Stop: VWAP - buffer
   - Target: session high or VWAP + IB

5. EXISTING PLAYBOOK (for comparison): Our current v13 signals
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
from typing import List, Tuple

from config.instruments import get_instrument
from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from config.constants import IB_BARS_1MIN, PM_SESSION_START
from strategy.day_type import classify_trend_strength, classify_day_type


@dataclass
class Trade:
    session_date: str
    bar_time: str
    bar_idx: int
    entry_type: str
    direction: str
    entry_price: float
    stop_price: float
    target_price: float
    day_type: str
    exit_price: float = 0.0
    exit_reason: str = ''
    pnl_points: float = 0.0
    mfe_points: float = 0.0
    mae_points: float = 0.0
    delta_at_entry: float = 0.0
    of_quality: int = 0


def simulate_trade(trade: Trade, remaining_bars: pd.DataFrame, eod_price: float):
    """Bar-by-bar simulation: stop first, then target, else EOD."""
    for _, bar in remaining_bars.iterrows():
        if trade.direction == 'LONG':
            # Check stop first (conservative)
            if bar['low'] <= trade.stop_price:
                trade.exit_price = trade.stop_price
                trade.exit_reason = 'STOP'
                trade.pnl_points = trade.stop_price - trade.entry_price
                break
            # Check target
            if bar['high'] >= trade.target_price:
                trade.exit_price = trade.target_price
                trade.exit_reason = 'TARGET'
                trade.pnl_points = trade.target_price - trade.entry_price
                break
            # Track MFE/MAE
            trade.mfe_points = max(trade.mfe_points, bar['high'] - trade.entry_price)
            trade.mae_points = max(trade.mae_points, trade.entry_price - bar['low'])
        else:
            if bar['high'] >= trade.stop_price:
                trade.exit_price = trade.stop_price
                trade.exit_reason = 'STOP'
                trade.pnl_points = trade.entry_price - trade.stop_price
                break
            if bar['low'] <= trade.target_price:
                trade.exit_price = trade.target_price
                trade.exit_reason = 'TARGET'
                trade.pnl_points = trade.entry_price - trade.target_price
                break
            trade.mfe_points = max(trade.mfe_points, trade.entry_price - bar['low'])
            trade.mae_points = max(trade.mae_points, bar['high'] - trade.entry_price)
    else:
        # EOD exit
        trade.exit_price = eod_price
        trade.exit_reason = 'EOD'
        if trade.direction == 'LONG':
            trade.pnl_points = eod_price - trade.entry_price
            trade.mfe_points = max(trade.mfe_points, remaining_bars['high'].max() - trade.entry_price)
            trade.mae_points = max(trade.mae_points, trade.entry_price - remaining_bars['low'].min())
        else:
            trade.pnl_points = trade.entry_price - eod_price
            trade.mfe_points = max(trade.mfe_points, trade.entry_price - remaining_bars['low'].min())
            trade.mae_points = max(trade.mae_points, remaining_bars['high'].max() - trade.entry_price)


def analyze_session(session_df: pd.DataFrame, session_date: str) -> List[Trade]:
    """Analyze session with refined, structural entry models."""

    ib_df = session_df.head(IB_BARS_1MIN)
    ib_high = ib_df['high'].max()
    ib_low = ib_df['low'].min()
    ib_range = ib_high - ib_low
    ib_mid = (ib_high + ib_low) / 2

    if ib_range <= 0:
        return []

    post_ib = session_df.iloc[IB_BARS_1MIN:].copy()
    if len(post_ib) == 0:
        return []

    post_ib = post_ib.reset_index(drop=True)
    eod_price = post_ib['close'].iloc[-1]
    trades: List[Trade] = []

    # State tracking
    bars_below_vwap = 0
    bars_below_ibl = 0
    bars_above_ibh = 0
    session_low = ib_low
    recent_swing_low = ib_low

    # Cooldowns per entry type
    last_entry_bars = {}  # entry_type -> last bar index
    COOLDOWN = 20

    # Delta history for momentum
    delta_history = []

    # Track IBL touches
    ibl_touch_count = 0
    last_ibl_touch_bar = -30

    # Track if price was below IBL at any point (for failed breakdown)
    ever_below_ibl = False
    below_ibl_start = -1

    for i in range(len(post_ib)):
        bar = post_ib.iloc[i]
        ts = bar.get('timestamp')
        bar_time = ts.time() if hasattr(ts, 'time') else None

        if bar_time and bar_time >= time(15, 20):
            break

        current_price = bar['close']
        delta = bar.get('delta', 0) if not pd.isna(bar.get('delta', 0)) else 0
        vol_spike = bar.get('volume_spike', 1.0) if not pd.isna(bar.get('volume_spike', 1.0)) else 1.0
        delta_pctl = bar.get('delta_percentile', 50) if not pd.isna(bar.get('delta_percentile', 50)) else 50
        imbalance = bar.get('imbalance_ratio', 1.0) if not pd.isna(bar.get('imbalance_ratio', 1.0)) else 1.0

        vwap = bar.get('vwap')
        if vwap is None or pd.isna(vwap):
            vwap = ib_mid

        # Track session low
        if bar['low'] < session_low:
            session_low = bar['low']

        # Delta history
        delta_history.append(delta)
        if len(delta_history) > 10:
            delta_history.pop(0)
        pre_delta = sum(delta_history[:-1]) if len(delta_history) > 1 else 0

        # OF quality score (for LONGs: bullish signals)
        of_quality = sum([
            delta_pctl >= 60,
            imbalance > 1.0,
            vol_spike >= 1.0,
        ])

        # Day type classification
        if current_price > ib_high:
            ib_dir = 'BULL'
            ext = (current_price - ib_mid) / ib_range
            bars_above_ibh += 1
            bars_below_ibl = 0
        elif current_price < ib_low:
            ib_dir = 'BEAR'
            ext = (ib_mid - current_price) / ib_range
            bars_below_ibl += 1
            bars_above_ibh = 0
            if not ever_below_ibl:
                ever_below_ibl = True
                below_ibl_start = i
        else:
            ib_dir = 'INSIDE'
            ext = 0.0
            bars_above_ibh = 0
            bars_below_ibl = 0

        strength = classify_trend_strength(ext)
        day_type = classify_day_type(ib_high, ib_low, current_price, ib_dir, strength)

        # Track VWAP relationship
        if current_price < vwap:
            bars_below_vwap += 1
        else:
            bars_below_vwap = 0

        # Track IBL touches
        if bar['low'] <= ib_low + ib_range * 0.02:
            if i - last_ibl_touch_bar > 5:
                ibl_touch_count += 1
            last_ibl_touch_bar = i

        # Track recent swing low (for stops)
        if current_price > vwap:
            recent_swing_low = session_low

        def can_enter(etype):
            return i - last_entry_bars.get(etype, -COOLDOWN) >= COOLDOWN

        # ================================================================
        # ENTRY MODEL 1: IBL_FADE_QUALITY (LONG)
        # Price touches IBL area and rejects back above with buyer confirmation
        # Works on any day type where price is near IB (b_day, p_day, neutral)
        # ================================================================
        if (can_enter('IBL_FADE_QUALITY') and
            bar['low'] <= ib_low + ib_range * 0.02 and  # Touch IBL
            current_price > ib_low and                   # Close above (rejection)
            current_price < ib_mid + ib_range * 0.2 and  # Not already at IB mid
            delta > 0 and                                 # Buyer present
            of_quality >= 2 and                           # OF confirmation
            ib_range <= 400):                             # Not stupidly wide IB

            stop = ib_low - ib_range * 0.15
            target = ib_mid

            # Must have reasonable R:R
            risk = current_price - stop
            reward = target - current_price
            if risk > 0 and reward / risk >= 0.8:
                t = Trade(
                    session_date=session_date, bar_time=str(bar_time), bar_idx=i,
                    entry_type='IBL_FADE_QUALITY', direction='LONG',
                    entry_price=current_price, stop_price=stop, target_price=target,
                    day_type=day_type.value, delta_at_entry=delta, of_quality=of_quality,
                )
                remaining = post_ib.iloc[i+1:]
                simulate_trade(t, remaining, eod_price)
                trades.append(t)
                last_entry_bars['IBL_FADE_QUALITY'] = i

        # ================================================================
        # ENTRY MODEL 2: VWAP_RECLAIM_QUALITY (LONG)
        # Price was below VWAP for 5+ bars, then reclaims above with delta
        # The key structural shift: VWAP acts as magnet. Reclaim = buyers won.
        # ================================================================
        if (can_enter('VWAP_RECLAIM_QUALITY') and
            bars_below_vwap == 0 and  # Just reclaimed (currently above VWAP)
            current_price > vwap and
            current_price < vwap + ib_range * 0.3 and   # Not too far above VWAP
            bar['low'] <= vwap + ib_range * 0.05 and    # Bar touched near VWAP
            delta > 0 and
            of_quality >= 2 and
            pre_delta > -500):  # Not heavy selling before

            # Check that we WERE below VWAP recently (structural shift)
            lookback = post_ib.iloc[max(0, i-10):i]
            bars_were_below = sum(1 for _, b in lookback.iterrows()
                                  if b['close'] < vwap) if len(lookback) > 0 else 0
            if bars_were_below >= 3:
                stop = vwap - ib_range * 0.30
                target = vwap + ib_range * 0.80

                risk = current_price - stop
                reward = target - current_price
                if risk > 0 and reward / risk >= 0.8:
                    t = Trade(
                        session_date=session_date, bar_time=str(bar_time), bar_idx=i,
                        entry_type='VWAP_RECLAIM_QUALITY', direction='LONG',
                        entry_price=current_price, stop_price=stop, target_price=target,
                        day_type=day_type.value, delta_at_entry=delta, of_quality=of_quality,
                    )
                    remaining = post_ib.iloc[i+1:]
                    simulate_trade(t, remaining, eod_price)
                    trades.append(t)
                    last_entry_bars['VWAP_RECLAIM_QUALITY'] = i

        # ================================================================
        # ENTRY MODEL 3: FAILED_BREAKDOWN (LONG)
        # Price broke below IBL (2+ bars below), then reclaims back above IBL
        # This is the "failed breakdown" / "IBL reclaim" pattern
        # Shorts were trapped below IBL, now covering -> upside momentum
        # ================================================================
        if (can_enter('FAILED_BREAKDOWN') and
            ever_below_ibl and
            current_price > ib_low and                    # Back above IBL
            current_price < ib_mid and                    # Still in lower half
            i - below_ibl_start >= 3 and                  # Was below for a few bars
            delta > 0 and
            of_quality >= 2):

            # Verify we were actually below IBL recently
            lookback = post_ib.iloc[max(0, i-10):i]
            bars_were_below_ibl = sum(1 for _, b in lookback.iterrows()
                                       if b['close'] < ib_low) if len(lookback) > 0 else 0
            if bars_were_below_ibl >= 2:
                stop = session_low - ib_range * 0.05
                target = ib_mid

                risk = current_price - stop
                reward = target - current_price
                if risk > 0 and reward / risk >= 0.5:  # Wider stop, accept lower R:R
                    t = Trade(
                        session_date=session_date, bar_time=str(bar_time), bar_idx=i,
                        entry_type='FAILED_BREAKDOWN', direction='LONG',
                        entry_price=current_price, stop_price=stop, target_price=target,
                        day_type=day_type.value, delta_at_entry=delta, of_quality=of_quality,
                    )
                    remaining = post_ib.iloc[i+1:]
                    simulate_trade(t, remaining, eod_price)
                    trades.append(t)
                    last_entry_bars['FAILED_BREAKDOWN'] = i

        # ================================================================
        # ENTRY MODEL 4: PM_VWAP_RECLAIM (LONG)
        # After 13:00, price that was below VWAP reclaims above with delta
        # PM momentum shift — morning sellers exhausted, afternoon buyers step in
        # ================================================================
        if (bar_time and bar_time >= PM_SESSION_START and
            can_enter('PM_VWAP_RECLAIM') and
            current_price > vwap and
            bar['low'] <= vwap + ib_range * 0.05 and
            delta > 0 and
            of_quality >= 2):

            # Must have been below VWAP recently in the afternoon
            lookback = post_ib.iloc[max(0, i-15):i]
            bars_below_recent = sum(1 for _, b in lookback.iterrows()
                                    if b['close'] < vwap) if len(lookback) > 0 else 0
            if bars_below_recent >= 5:
                stop = vwap - ib_range * 0.25
                target = vwap + ib_range * 0.60

                risk = current_price - stop
                reward = target - current_price
                if risk > 0 and reward / risk >= 0.7:
                    t = Trade(
                        session_date=session_date, bar_time=str(bar_time), bar_idx=i,
                        entry_type='PM_VWAP_RECLAIM', direction='LONG',
                        entry_price=current_price, stop_price=stop, target_price=target,
                        day_type=day_type.value, delta_at_entry=delta, of_quality=of_quality,
                    )
                    remaining = post_ib.iloc[i+1:]
                    simulate_trade(t, remaining, eod_price)
                    trades.append(t)
                    last_entry_bars['PM_VWAP_RECLAIM'] = i

        # ================================================================
        # ENTRY MODEL 5: EDGE_TO_MID_NEUTRAL (LONG)
        # On neutral/b_day: price near IBL, fade toward IB mid
        # Wider stop, simple mean reversion
        # Requires price to be in lower 25% of IB range
        # ================================================================
        if (can_enter('EDGE_TO_MID') and
            day_type.value in ('b_day', 'neutral') and
            current_price > ib_low and
            current_price < ib_low + ib_range * 0.25 and  # Lower quartile
            delta > 0 and
            of_quality >= 2 and
            ib_range <= 350 and  # Tighter range for this trade
            ib_range >= 50):     # Not too narrow

            stop = ib_low - ib_range * 0.15
            target = ib_mid

            risk = current_price - stop
            reward = target - current_price
            if risk > 0 and reward / risk >= 1.0:
                t = Trade(
                    session_date=session_date, bar_time=str(bar_time), bar_idx=i,
                    entry_type='EDGE_TO_MID', direction='LONG',
                    entry_price=current_price, stop_price=stop, target_price=target,
                    day_type=day_type.value, delta_at_entry=delta, of_quality=of_quality,
                )
                remaining = post_ib.iloc[i+1:]
                simulate_trade(t, remaining, eod_price)
                trades.append(t)
                last_entry_bars['EDGE_TO_MID'] = i

    return trades


def print_results(all_trades: List[Trade], point_value: float, total_sessions: int):
    """Print comprehensive results."""

    print(f"\n{'='*120}")
    print(f"  REFINED ENTRY MODEL RESULTS (v2 - LONG ONLY with structural gates)")
    print(f"{'='*120}\n")

    # Overall
    n = len(all_trades)
    if n == 0:
        print("No trades generated!")
        return

    winners = sum(1 for t in all_trades if t.pnl_points > 0)
    total_pts = sum(t.pnl_points for t in all_trades)
    total_dollars = total_pts * point_value

    sessions_with_trades = len(set(t.session_date for t in all_trades))

    print(f"Total trades:          {n}")
    print(f"Sessions with trades:  {sessions_with_trades} / {total_sessions} ({sessions_with_trades/total_sessions*100:.0f}%)")
    print(f"Trades/session:        {n/total_sessions:.1f}")
    print(f"Win Rate:              {winners/n*100:.1f}%")
    print(f"Total Net Points:      {total_pts:.1f}")
    print(f"Total Net $:           ${total_dollars:,.0f}")
    print(f"Avg PnL/trade:         {total_pts/n:.1f} pts (${total_dollars/n:,.0f})")
    print(f"Expectancy/trade:      ${total_dollars/n:,.0f}")

    # By entry type
    print(f"\n--- BY ENTRY TYPE ---\n")
    print(f"{'Entry Type':<25} {'N':>5} {'WR':>7} {'Avg PnL':>9} {'Total $':>10} {'Avg MFE':>9} {'Avg MAE':>9} {'T/S/E':>8}")
    print(f"{'-'*25} {'-'*5} {'-'*7} {'-'*9} {'-'*10} {'-'*9} {'-'*9} {'-'*8}")

    entry_types = {}
    for t in all_trades:
        entry_types.setdefault(t.entry_type, []).append(t)

    for etype in sorted(entry_types.keys()):
        tlist = entry_types[etype]
        en = len(tlist)
        ew = sum(1 for t in tlist if t.pnl_points > 0)
        ewr = ew / en * 100
        epts = sum(t.pnl_points for t in tlist)
        e_dollars = epts * point_value
        avg_pnl = epts / en
        avg_mfe = np.mean([t.mfe_points for t in tlist])
        avg_mae = np.mean([t.mae_points for t in tlist])

        # Exit type breakdown
        targets = sum(1 for t in tlist if t.exit_reason == 'TARGET')
        stops = sum(1 for t in tlist if t.exit_reason == 'STOP')
        eods = sum(1 for t in tlist if t.exit_reason == 'EOD')

        print(f"{etype:<25} {en:>5} {ewr:>6.1f}% {avg_pnl:>8.1f} ${e_dollars:>8,.0f} "
              f"{avg_mfe:>8.1f} {avg_mae:>8.1f} {targets}/{stops}/{eods}")

    # By day type
    print(f"\n--- BY DAY TYPE ---\n")
    day_types = {}
    for t in all_trades:
        day_types.setdefault(t.day_type, []).append(t)

    for dtype in sorted(day_types.keys()):
        tlist = day_types[dtype]
        en = len(tlist)
        ew = sum(1 for t in tlist if t.pnl_points > 0)
        ewr = ew / en * 100
        epts = sum(t.pnl_points for t in tlist)
        e_dollars = epts * point_value
        print(f"  {dtype:<15}: {en:>4} trades, {ewr:>5.1f}% WR, ${e_dollars:>8,.0f}")

    # Session-level distribution
    print(f"\n--- TRADES PER SESSION ---\n")
    session_counts = {}
    for t in all_trades:
        session_counts[t.session_date] = session_counts.get(t.session_date, 0) + 1

    dist = {}
    for c in session_counts.values():
        dist[c] = dist.get(c, 0) + 1

    no_trade = total_sessions - sessions_with_trades
    print(f"  0 trades: {no_trade} sessions")
    for tc in sorted(dist.keys()):
        print(f"  {tc} trades: {dist[tc]} sessions")

    # Combined with existing playbook
    print(f"\n{'='*120}")
    print(f"  COMPARISON: New entries vs Existing Playbook (v13)")
    print(f"{'='*120}\n")
    print(f"  Existing Playbook:   20 trades, 80.0% WR, $4,434 net, $222/trade")
    print(f"  New entries (this):  {n} trades, {winners/n*100:.1f}% WR, ${total_dollars:,.0f} net, ${total_dollars/n:,.0f}/trade")
    combined_trades = n + 20
    combined_dollars = total_dollars + 4434
    print(f"  COMBINED:            {combined_trades} trades, ${combined_dollars:,.0f} net, ${combined_dollars/combined_trades:,.0f}/trade")
    print(f"  Trades/session:      {combined_trades/total_sessions:.1f} (target: 2-3)")

    # Per-session P&L
    print(f"\n--- SESSION DETAIL ---\n")
    session_pnl = {}
    for t in all_trades:
        session_pnl.setdefault(t.session_date, []).append(t)

    for sdate in sorted(session_pnl.keys()):
        tlist = session_pnl[sdate]
        sn = len(tlist)
        sw = sum(1 for t in tlist if t.pnl_points > 0)
        spts = sum(t.pnl_points for t in tlist)
        s_dollars = spts * point_value
        wr = sw / sn * 100 if sn > 0 else 0

        print(f"  {sdate} | {sn} trades | {wr:>5.0f}% WR | ${s_dollars:>+8,.0f}")
        for t in tlist:
            dollars = t.pnl_points * point_value
            marker = 'W' if t.pnl_points > 0 else 'L'
            print(f"    {t.bar_time:>8} {t.entry_type:<25} {t.direction:>5} "
                  f"@{t.entry_price:>9.1f} -> {t.exit_reason:<7} "
                  f"{t.pnl_points:>+7.1f}pts (${dollars:>+6,.0f}) "
                  f"MFE={t.mfe_points:>5.1f} MAE={t.mae_points:>5.1f} [{marker}]")


def main():
    instrument = get_instrument('MNQ')
    point_value = instrument.point_value

    df = load_csv('NQ')
    df = filter_rth(df)
    df = compute_all_features(df)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    if 'session_date' not in df.columns:
        df['session_date'] = df['timestamp'].dt.date

    sessions = sorted(df['session_date'].unique())
    total_sessions = len(sessions)

    all_trades: List[Trade] = []

    for session_date in sessions:
        session_df = df[df['session_date'] == session_date].copy()
        if len(session_df) < IB_BARS_1MIN:
            continue
        trades = analyze_session(session_df, str(session_date))
        all_trades.extend(trades)

    print_results(all_trades, point_value, total_sessions)


if __name__ == '__main__':
    main()
