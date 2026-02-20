"""
Diagnostic 3: Check sessions where VWAP pullback is near but price is below IBH.
If we relax the "must be above IBH" check, how many more entries do we get?
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


def analyze():
    df = load_csv('NQ')
    df = filter_rth(df)
    df = compute_all_features(df)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    if 'session_date' not in df.columns:
        df['session_date'] = df['timestamp'].dt.date

    sessions = sorted(df['session_date'].unique())

    print(f"\n{'='*120}")
    print(f"  RELAXED IBH REQUIREMENT: What if we allow VWAP entries below IBH after acceptance?")
    print(f"{'='*120}\n")

    total_new_entries = 0
    session_new_entries = {}

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

        scorer = DayTypeConfidenceScorer()
        atr = ib_df.iloc[-1].get('atr14', 0.0)
        scorer.on_session_start(ib_high, ib_low, ib_range, atr)

        consecutive_above = 0
        acceptance_confirmed = False

        # Count entries under different rules
        # Current: above_ibh=True, time < 11:30, vwap_dist < 0.40
        # New V1: above_ibh=True, time < 13:00, vwap_dist < 0.40
        # New V2: above_ibh relaxed (above ib_mid), time < 13:00, vwap_dist < 0.40
        # New V3: above_ibh relaxed (above ib_mid), time < 13:00, vwap_dist < 0.50

        entry_counts = {'current': 0, 'v1_time_13': 0, 'v2_relaxed_ibh': 0, 'v3_relaxed_both': 0}
        first_entries = {'current': None, 'v1_time_13': None, 'v2_relaxed_ibh': None, 'v3_relaxed_both': None}

        for bar_idx in range(len(post_ib_df)):
            bar = post_ib_df.iloc[bar_idx]
            timestamp = bar['timestamp'] if 'timestamp' in bar.index else None
            bar_time = timestamp.time() if timestamp and hasattr(timestamp, 'time') else None
            current_price = bar['close']
            delta = bar.get('delta', 0)

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
                continue

            vwap = bar.get('vwap')
            if vwap is None or pd.isna(vwap):
                continue

            vwap_dist = abs(current_price - vwap) / ib_range if ib_range > 0 else 999
            is_above_vwap = current_price > vwap
            has_positive_delta = delta > 0
            is_moderate = strength.value in ('moderate', 'strong', 'super')
            is_valid_dt = day_type.value in ('trend_up', 'super_trend_up', 'p_day')
            meets_conf = day_conf.trend_bull >= 0.375
            is_above_ibh = current_price > ib_high
            is_above_ib_mid = current_price > ib_mid

            # Base conditions (excluding time, price level, and dist)
            base = is_above_vwap and has_positive_delta and is_moderate and meets_conf

            # Current entry
            if base and is_valid_dt and is_above_ibh and bar_time and bar_time < time(11, 30) and vwap_dist < 0.40:
                entry_counts['current'] += 1
                if first_entries['current'] is None:
                    first_entries['current'] = (bar_time, current_price, vwap, delta, day_type.value, ext, vwap_dist)

            # V1: Extend time to 13:00
            if base and is_valid_dt and is_above_ibh and bar_time and bar_time < time(13, 0) and vwap_dist < 0.40:
                entry_counts['v1_time_13'] += 1
                if first_entries['v1_time_13'] is None:
                    first_entries['v1_time_13'] = (bar_time, current_price, vwap, delta, day_type.value, ext, vwap_dist)

            # V2: Relax above IBH to above IB mid + extend time
            # Also accept neutral day type (price returned inside IB)
            valid_dt_relaxed = day_type.value in ('trend_up', 'super_trend_up', 'p_day', 'neutral', 'b_day')
            if base and valid_dt_relaxed and is_above_ib_mid and bar_time and bar_time < time(13, 0) and vwap_dist < 0.40:
                entry_counts['v2_relaxed_ibh'] += 1
                if first_entries['v2_relaxed_ibh'] is None:
                    first_entries['v2_relaxed_ibh'] = (bar_time, current_price, vwap, delta, day_type.value, ext, vwap_dist)

            # V3: Relax both + slightly wider VWAP distance
            if base and valid_dt_relaxed and is_above_ib_mid and bar_time and bar_time < time(13, 0) and vwap_dist < 0.50:
                entry_counts['v3_relaxed_both'] += 1
                if first_entries['v3_relaxed_both'] is None:
                    first_entries['v3_relaxed_both'] = (bar_time, current_price, vwap, delta, day_type.value, ext, vwap_dist)

        if not acceptance_confirmed:
            continue

        # Only show sessions where something changes
        if entry_counts['v2_relaxed_ibh'] > entry_counts['current']:
            gained = entry_counts['v2_relaxed_ibh'] - entry_counts['current']
            total_new_entries += 1 if gained > 0 else 0

            print(f"--- {session_date} ---")
            for version, count in entry_counts.items():
                first = first_entries[version]
                if first:
                    print(f"  {version:20s}: {count:3d} bars | first: time={first[0]}, price={first[1]:.1f}, "
                          f"vwap={first[2]:.1f}, delta={first[3]:.0f}, dt={first[4]}, ext={first[5]:.2f}, "
                          f"vwap_dist={first[6]:.3f}")
                else:
                    print(f"  {version:20s}: {count:3d} bars | (no entry)")
            print()

    print(f"\nSessions gaining entries with V2 (relaxed IBH + 13:00 cutoff): {total_new_entries}")


if __name__ == '__main__':
    analyze()
