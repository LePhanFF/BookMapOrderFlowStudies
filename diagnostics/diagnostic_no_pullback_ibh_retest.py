"""
Diagnostic 7: For sessions that accepted but had NO VWAP pullback,
check if an IBH retest entry model could work.

When price breaks above IBH and NEVER comes back to VWAP, it's often
because it's a very strong trend day. These sessions often have
an IBH retest (brief dip to IBH as support) that could be an entry.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import time

from config.instruments import get_instrument
from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from config.constants import IB_BARS_1MIN, ACCEPTANCE_MIN_BARS
from strategy.day_type import classify_trend_strength, classify_day_type
from strategy.day_confidence import DayTypeConfidenceScorer


def analyze():
    df = load_csv('NQ')
    df = filter_rth(df)
    df = compute_all_features(df)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    if 'session_date' not in df.columns:
        df['session_date'] = df['timestamp'].dt.date

    sessions = sorted(df['session_date'].unique())

    # Identify sessions with acceptance but no VWAP pullback (from prior diagnostic)
    no_pullback_sessions = [
        '2025-11-24', '2025-12-02', '2025-12-18', '2025-12-24',
        '2025-12-30', '2026-01-05', '2026-01-06', '2026-01-09',
        '2026-01-12', '2026-01-20', '2026-01-21', '2026-01-26',
        '2026-02-02', '2026-02-06',
    ]

    print(f"\n{'='*120}")
    print(f"  NO PULLBACK SESSIONS: IBH RETEST + CUMULATIVE DELTA ENTRY ANALYSIS")
    print(f"{'='*120}\n")

    for session_date in sessions:
        session_str = str(session_date)[:10]
        if session_str not in no_pullback_sessions:
            continue

        session_df = df[df['session_date'] == session_date].copy()
        if len(session_df) < IB_BARS_1MIN:
            continue

        ib_df = session_df.head(IB_BARS_1MIN)
        ib_high = ib_df['high'].max()
        ib_low = ib_df['low'].min()
        ib_range = ib_high - ib_low
        ib_mid = (ib_high + ib_low) / 2

        if ib_range <= 0:
            continue

        post_ib_df = session_df.iloc[IB_BARS_1MIN:]
        if len(post_ib_df) == 0:
            continue

        scorer = DayTypeConfidenceScorer()
        atr = ib_df.iloc[-1].get('atr14', 0.0)
        scorer.on_session_start(ib_high, ib_low, ib_range, atr)

        consecutive_above = 0
        acceptance_confirmed = False
        acceptance_bar_idx = None
        acceptance_bar_time = None
        session_high = ib_high

        # Track IBH retest events
        ibh_retest_opportunities = []

        # Track cumulative delta trend
        delta_history = []

        for bar_idx in range(len(post_ib_df)):
            bar = post_ib_df.iloc[bar_idx]
            timestamp = bar['timestamp'] if 'timestamp' in bar.index else None
            bar_time = timestamp.time() if timestamp and hasattr(timestamp, 'time') else None
            current_price = bar['close']
            delta = bar.get('delta', 0)

            if bar['high'] > session_high:
                session_high = bar['high']

            delta_history.append(delta if not pd.isna(delta) else 0)
            if len(delta_history) > 10:
                delta_history.pop(0)

            if current_price > ib_high:
                ib_direction = 'BULL'
                ext = (current_price - ib_mid) / ib_range
            elif current_price < ib_low:
                ib_direction = 'BEAR'
                ext = (ib_mid - current_price) / ib_range
            else:
                ib_direction = 'INSIDE'
                ext = 0.0

            strength = classify_trend_strength(ext)
            day_type = classify_day_type(ib_high, ib_low, current_price, ib_direction, strength)
            day_conf = scorer.update(bar, bar_idx)

            if not acceptance_confirmed:
                if current_price > ib_high:
                    consecutive_above += 1
                else:
                    consecutive_above = 0
                if consecutive_above >= ACCEPTANCE_MIN_BARS:
                    acceptance_confirmed = True
                    acceptance_bar_idx = bar_idx
                    acceptance_bar_time = bar_time
                continue

            # Look for IBH retest: bar low touches or dips below IBH, close above IBH
            if bar['low'] <= ib_high + (ib_range * 0.05) and current_price > ib_high:
                if delta > 0 and bar_time and bar_time < time(13, 0):
                    pre_delta = sum(delta_history[:-1]) if len(delta_history) > 1 else 0

                    # What would happen with entry at close, stop at IBH - buffer?
                    entry_price = current_price
                    stop_price = ib_high - (ib_range * 0.15)  # tighter stop for IBH retest

                    # Get session result
                    eod_close = post_ib_df.iloc[-1]['close']
                    remaining = post_ib_df.iloc[bar_idx + 1:]

                    hit_stop = False
                    min_after = entry_price
                    for _, future_bar in remaining.iterrows():
                        if future_bar['low'] < min_after:
                            min_after = future_bar['low']
                        if future_bar['low'] <= stop_price:
                            hit_stop = True
                            break

                    pnl_pts = (stop_price if hit_stop else eod_close) - entry_price
                    result = 'WIN' if pnl_pts > 0 else 'LOSS'
                    mae_pts = entry_price - min_after

                    ibh_retest_opportunities.append({
                        'bar_time': bar_time,
                        'price': entry_price,
                        'delta': delta,
                        'pre_delta': pre_delta,
                        'pnl_pts': pnl_pts,
                        'result': result,
                        'hit_stop': hit_stop,
                        'mae': mae_pts,
                        'tb_conf': day_conf.trend_bull,
                        'ext': ext,
                        'day_type': day_type.value,
                    })

        session_high_ext = (session_high - ib_mid) / ib_range if ib_range > 0 else 0
        eod_close = post_ib_df.iloc[-1]['close']
        eod_pnl = eod_close - ib_high

        print(f"--- {session_str} ---")
        print(f"  IB Range: {ib_range:.1f} | Acceptance at: {acceptance_bar_time}")
        print(f"  Session High: {session_high:.1f} ({session_high_ext:.2f}x IB) | EOD: {eod_close:.1f}")
        print(f"  EOD P&L from IBH: {eod_pnl:+.1f} pts")

        if ibh_retest_opportunities:
            for opp in ibh_retest_opportunities[:3]:  # Show first 3
                print(f"  IBH Retest: time={opp['bar_time']}, entry={opp['price']:.1f}, "
                      f"delta={opp['delta']:.0f}, pre_delta={opp['pre_delta']:.0f}, "
                      f"P&L={opp['pnl_pts']:+.1f} pts [{opp['result']}], "
                      f"MAE={opp['mae']:.1f} pts, tb_conf={opp['tb_conf']:.2f}")
        else:
            print(f"  No IBH retest opportunities found (price never came back near IBH)")

        print()


if __name__ == '__main__':
    analyze()
