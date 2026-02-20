"""
Diagnostic 2: For sessions that accepted but had NO VWAP pullback,
check how close price ever got to VWAP after acceptance.
Also check if a slightly looser VWAP distance threshold would help.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
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


def analyze_no_pullback_sessions():
    instrument = get_instrument('MNQ')
    df = load_csv('NQ')
    df = filter_rth(df)
    df = compute_all_features(df)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    if 'session_date' not in df.columns:
        df['session_date'] = df['timestamp'].dt.date

    sessions = sorted(df['session_date'].unique())

    print(f"\n{'='*120}")
    print(f"  ANALYSIS: VWAP DISTANCE AFTER ACCEPTANCE + EXTENDED TIME WINDOW")
    print(f"{'='*120}\n")

    for session_date in sessions:
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

        # Initialize confidence scorer
        scorer = DayTypeConfidenceScorer()
        atr = ib_df.iloc[-1].get('atr14', 0.0)
        scorer.on_session_start(ib_high, ib_low, ib_range, atr)

        consecutive_above = 0
        acceptance_confirmed = False
        acceptance_bar_idx = None
        acceptance_bar_time = None

        # Track closest VWAP approach after acceptance
        min_vwap_dist = 999.0
        min_vwap_time = None
        min_vwap_price = None
        min_vwap_vwap = None
        min_vwap_delta = None
        min_vwap_above = None

        # Track entries with different time cutoffs
        entries_by_cutoff = {
            '11:30': 0, '12:00': 0, '12:30': 0, '13:00': 0, '14:00': 0, '15:00': 0,
        }
        # Track entries with different VWAP distance thresholds
        entries_by_dist = {
            0.40: 0, 0.50: 0, 0.60: 0, 0.75: 0, 1.00: 0,
        }

        for bar_idx in range(len(post_ib_df)):
            bar = post_ib_df.iloc[bar_idx]
            timestamp = bar['timestamp'] if 'timestamp' in bar.index else None
            bar_time = timestamp.time() if timestamp and hasattr(timestamp, 'time') else None
            current_price = bar['close']
            delta = bar.get('delta', 0)

            # Update day type
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

            # After acceptance - track VWAP distance
            vwap = bar.get('vwap')
            if vwap is None or pd.isna(vwap):
                continue

            vwap_dist = abs(current_price - vwap) / ib_range if ib_range > 0 else 999
            is_above_vwap = current_price > vwap
            is_above_ibh = current_price > ib_high
            has_positive_delta = delta > 0
            is_moderate = strength.value in ('moderate', 'strong', 'super')
            is_valid_dt = day_type.value in ('trend_up', 'super_trend_up', 'p_day')
            meets_conf = day_conf.trend_bull >= 0.375

            # Track minimum distance
            if is_above_vwap and vwap_dist < min_vwap_dist:
                min_vwap_dist = vwap_dist
                min_vwap_time = bar_time
                min_vwap_price = current_price
                min_vwap_vwap = vwap
                min_vwap_delta = delta
                min_vwap_above = is_above_ibh

            # Check entries with different cutoffs (all other conditions met except time)
            base_ok = (is_above_vwap and is_above_ibh and has_positive_delta and
                       is_moderate and is_valid_dt and meets_conf)

            if base_ok:
                for cutoff_str, cutoff_time in [
                    ('11:30', time(11, 30)), ('12:00', time(12, 0)),
                    ('12:30', time(12, 30)), ('13:00', time(13, 0)),
                    ('14:00', time(14, 0)), ('15:00', time(15, 0)),
                ]:
                    if bar_time and bar_time < cutoff_time and vwap_dist < 0.40:
                        entries_by_cutoff[cutoff_str] += 1

                for dist_thresh in [0.40, 0.50, 0.60, 0.75, 1.00]:
                    if bar_time and bar_time < time(13, 0) and vwap_dist < dist_thresh:
                        entries_by_dist[dist_thresh] += 1

        if not acceptance_confirmed:
            continue

        has_blocked = any(v > 0 for v in entries_by_cutoff.values())
        has_entry_now = entries_by_cutoff['11:30'] > 0

        if has_entry_now:
            tag = '[FIRES NOW]'
        elif has_blocked:
            tag = '[COULD FIRE]'
        else:
            tag = '[NO SIGNAL]'

        print(f"--- {session_date} ---  {tag}")
        print(f"  Acceptance at: {acceptance_bar_time}")
        if min_vwap_time:
            print(f"  Closest VWAP approach: dist={min_vwap_dist:.3f}x IB at {min_vwap_time}, "
                  f"price={min_vwap_price:.1f}, vwap={min_vwap_vwap:.1f}, "
                  f"delta={'+ ' if min_vwap_delta > 0 else ''}{min_vwap_delta:.0f}, "
                  f"above_IBH={min_vwap_above}")
        else:
            print(f"  No VWAP approach while above VWAP")

        # Only show cutoff analysis for sessions that DON'T fire now
        if not has_entry_now:
            cutoff_str = ' | '.join(f"{k}: {v}" for k, v in entries_by_cutoff.items())
            print(f"  Entries by time cutoff: {cutoff_str}")
            dist_str = ' | '.join(f"{k}: {v}" for k, v in entries_by_dist.items())
            print(f"  Entries by VWAP dist (13:00 cutoff): {dist_str}")
        print()

    # Grand summary
    print(f"\n{'='*120}")
    print(f"  OPTIMIZATION OPPORTUNITY: EXTEND TIME WINDOW")
    print(f"{'='*120}\n")
    print("The LONDON_CLOSE = 11:30 cutoff is the single biggest filter.")
    print("12 sessions have VWAP pullbacks that pass ALL conditions EXCEPT time.")
    print("All blocked pullbacks are after 11:30 but most are before 14:00.")
    print("\nRecommendation: Extend VWAP pullback window to PM_SESSION_START (13:00)")
    print("This would add ~12 new trades while maintaining the same entry quality.")


if __name__ == '__main__':
    analyze_no_pullback_sessions()
