"""
Diagnostic: Analyze every session to understand WHY entries fire or don't fire.
Identifies the exact bar-level conditions that block/allow VWAP pullback entries.
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
from config.constants import (
    IB_BARS_1MIN, ACCEPTANCE_MIN_BARS, LONDON_CLOSE,
)
from strategy.day_type import classify_trend_strength, classify_day_type
from strategy.day_confidence import DayTypeConfidenceScorer


def analyze_all_sessions():
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
    print(f"  SESSION-BY-SESSION DIAGNOSTIC: WHY ENTRIES FIRE OR DON'T")
    print(f"{'='*120}\n")

    session_results = []

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

        # Track acceptance
        consecutive_above = 0
        acceptance_confirmed = False
        acceptance_bar_idx = None
        acceptance_bar_time = None

        # Track VWAP pullback opportunities after acceptance
        vwap_pullback_bars = []
        vwap_pullback_reasons_blocked = []

        # Track session extremes and day type evolution
        session_high = ib_high
        session_low = ib_low
        final_day_type = 'neutral'
        final_strength = 'weak'
        final_ext = 0.0
        max_tb_conf = 0.0
        max_pb_conf = 0.0
        max_bday_conf = 0.0

        # Track IBL touch events for B-Day
        ibl_touch_count = 0
        ibl_touches_with_delta = 0

        for bar_idx in range(len(post_ib_df)):
            bar = post_ib_df.iloc[bar_idx]
            timestamp = bar['timestamp'] if 'timestamp' in bar.index else None
            bar_time = timestamp.time() if timestamp and hasattr(timestamp, 'time') else None
            current_price = bar['close']
            delta = bar.get('delta', 0)

            if bar['high'] > session_high:
                session_high = bar['high']
            if bar['low'] < session_low:
                session_low = bar['low']

            # Update day type dynamically
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

            # Update confidence scorer
            day_conf = scorer.update(bar, bar_idx)
            max_tb_conf = max(max_tb_conf, day_conf.trend_bull)
            max_pb_conf = max(max_pb_conf, day_conf.p_day_bull)
            max_bday_conf = max(max_bday_conf, day_conf.b_day)

            final_day_type = day_type.value
            final_strength = strength.value
            final_ext = ext

            # Track IBL touches
            if bar['low'] <= ib_low:
                ibl_touch_count += 1
                if delta > 0:
                    ibl_touches_with_delta += 1

            # Track acceptance
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

            # After acceptance: look for VWAP pullback opportunities
            vwap = bar.get('vwap')
            if vwap is None or pd.isna(vwap):
                continue

            vwap_dist = abs(current_price - vwap) / ib_range if ib_range > 0 else 999

            # Check all the conditions for VWAP pullback entry
            is_near_vwap = vwap_dist < 0.40
            is_above_vwap = current_price > vwap
            has_positive_delta = delta > 0
            is_above_ibh = current_price > ib_high
            is_before_london = bar_time is None or bar_time < LONDON_CLOSE
            is_moderate_plus = strength.value in ('moderate', 'strong', 'super')
            is_valid_daytype = day_type.value in ('trend_up', 'super_trend_up', 'p_day')
            tb_conf = day_conf.trend_bull
            meets_confidence = tb_conf >= 0.375

            # Check ALL conditions
            if is_near_vwap and is_above_vwap and has_positive_delta and is_above_ibh:
                if is_before_london and is_moderate_plus and is_valid_daytype and meets_confidence:
                    vwap_pullback_bars.append({
                        'bar_idx': bar_idx,
                        'bar_time': str(bar_time),
                        'price': current_price,
                        'vwap': vwap,
                        'vwap_dist': vwap_dist,
                        'delta': delta,
                        'day_type': day_type.value,
                        'strength': strength.value,
                        'tb_conf': tb_conf,
                        'ext': ext,
                    })
                else:
                    # Record why it was blocked
                    reasons = []
                    if not is_before_london:
                        reasons.append('AFTER_LONDON_CLOSE')
                    if not is_moderate_plus:
                        reasons.append(f'WEAK_STRENGTH(ext={ext:.2f})')
                    if not is_valid_daytype:
                        reasons.append(f'DAY_TYPE={day_type.value}')
                    if not meets_confidence:
                        reasons.append(f'LOW_TB_CONF({tb_conf:.2f})')
                    vwap_pullback_reasons_blocked.append({
                        'bar_idx': bar_idx,
                        'bar_time': str(bar_time),
                        'price': current_price,
                        'vwap': vwap,
                        'vwap_dist': vwap_dist,
                        'delta': delta,
                        'reasons': reasons,
                        'day_type': day_type.value,
                        'strength': strength.value,
                        'ext': ext,
                        'tb_conf': tb_conf,
                    })

        # Compute session summary
        max_ext_up = (session_high - ib_mid) / ib_range if ib_range > 0 else 0
        max_ext_down = (ib_mid - session_low) / ib_range if ib_range > 0 else 0

        result = {
            'session_date': str(session_date),
            'ib_range': ib_range,
            'max_ext_up': max_ext_up,
            'max_ext_down': max_ext_down,
            'final_day_type': final_day_type,
            'final_strength': final_strength,
            'acceptance': acceptance_confirmed,
            'acceptance_bar_time': str(acceptance_bar_time) if acceptance_bar_time else 'N/A',
            'max_tb_conf': max_tb_conf,
            'max_pb_conf': max_pb_conf,
            'max_bday_conf': max_bday_conf,
            'vwap_entries_available': len(vwap_pullback_bars),
            'vwap_entries_blocked': len(vwap_pullback_reasons_blocked),
            'ibl_touches': ibl_touch_count,
            'ibl_touches_with_delta': ibl_touches_with_delta,
        }
        session_results.append(result)

        # Print detailed analysis for each session
        has_trade_potential = (
            acceptance_confirmed and (len(vwap_pullback_bars) > 0 or len(vwap_pullback_reasons_blocked) > 0)
        ) or ibl_touch_count > 0

        tag = ''
        if len(vwap_pullback_bars) > 0:
            tag = '  [ENTRY FIRES]'
        elif acceptance_confirmed and len(vwap_pullback_reasons_blocked) > 0:
            tag = '  [BLOCKED]'
        elif acceptance_confirmed:
            tag = '  [ACCEPTED, NO PULLBACK]'
        elif max_ext_up > 0.5:
            tag = '  [NO ACCEPTANCE]'
        elif ibl_touch_count > 0:
            tag = '  [B-DAY CANDIDATE]'
        else:
            tag = '  [NO OPPORTUNITY]'

        print(f"--- {session_date} ---{tag}")
        print(f"  IB Range: {ib_range:.1f} pts | Max Ext Up: {max_ext_up:.2f}x | Max Ext Down: {max_ext_down:.2f}x")
        print(f"  Final: {final_day_type} / {final_strength} | max_tb_conf={max_tb_conf:.2f} | max_pb_conf={max_pb_conf:.2f} | max_bday_conf={max_bday_conf:.2f}")
        print(f"  Acceptance: {'YES at ' + str(acceptance_bar_time) if acceptance_confirmed else 'NO'}")
        print(f"  IBL Touches: {ibl_touch_count} (w/ delta+: {ibl_touches_with_delta})")

        if vwap_pullback_bars:
            first = vwap_pullback_bars[0]
            print(f"  VWAP ENTRY AVAILABLE: {len(vwap_pullback_bars)} bars (first at {first['bar_time']}, "
                  f"price={first['price']:.1f}, vwap={first['vwap']:.1f}, dist={first['vwap_dist']:.2f}, "
                  f"delta={first['delta']:.0f}, dt={first['day_type']}, str={first['strength']}, "
                  f"tb_conf={first['tb_conf']:.2f})")

        if vwap_pullback_reasons_blocked:
            # Group by reason
            reason_counts = {}
            for blocked in vwap_pullback_reasons_blocked:
                for r in blocked['reasons']:
                    reason_counts[r] = reason_counts.get(r, 0) + 1
            print(f"  BLOCKED VWAP PULLBACKS: {len(vwap_pullback_reasons_blocked)} bars")
            for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
                print(f"    - {reason}: {count} bars")
            # Show first blocked example
            first_blocked = vwap_pullback_reasons_blocked[0]
            print(f"    First blocked: time={first_blocked['bar_time']}, price={first_blocked['price']:.1f}, "
                  f"vwap={first_blocked['vwap']:.1f}, ext={first_blocked['ext']:.2f}, "
                  f"dt={first_blocked['day_type']}, str={first_blocked['strength']}, "
                  f"tb_conf={first_blocked['tb_conf']:.2f}")

        print()

    # Summary table
    print(f"\n{'='*120}")
    print(f"  SUMMARY")
    print(f"{'='*120}\n")

    accepted = [s for s in session_results if s['acceptance']]
    with_vwap = [s for s in session_results if s['vwap_entries_available'] > 0]
    blocked = [s for s in session_results if s['acceptance'] and s['vwap_entries_available'] == 0 and s['vwap_entries_blocked'] > 0]
    no_pullback = [s for s in session_results if s['acceptance'] and s['vwap_entries_available'] == 0 and s['vwap_entries_blocked'] == 0]
    bday_candidates = [s for s in session_results if s['ibl_touches'] > 0 and s['max_bday_conf'] >= 0.5]

    print(f"Total sessions:        {len(session_results)}")
    print(f"With acceptance:       {len(accepted)}")
    print(f"VWAP entry fires:      {len(with_vwap)}")
    print(f"VWAP entry blocked:    {len(blocked)}")
    print(f"Accepted, no pullback: {len(no_pullback)}")
    print(f"B-Day candidates:      {len(bday_candidates)}")
    print()

    # Blocked reason analysis
    if blocked:
        print("BLOCKED SESSION ANALYSIS (accepted but VWAP entry didn't fire):")
        all_reasons = {}
        for s in session_results:
            if s['acceptance'] and s['vwap_entries_available'] == 0 and s['vwap_entries_blocked'] > 0:
                # This session had pullbacks that were blocked
                print(f"  {s['session_date']}: final_dt={s['final_day_type']}, ext_up={s['max_ext_up']:.2f}, "
                      f"tb_conf={s['max_tb_conf']:.2f}, pb_conf={s['max_pb_conf']:.2f}")

    # No pullback analysis - sessions that accepted but price never came back near VWAP
    if no_pullback:
        print(f"\nACCEPTED BUT NO VWAP PULLBACK ({len(no_pullback)} sessions):")
        for s in no_pullback:
            print(f"  {s['session_date']}: accepted at {s['acceptance_bar_time']}, "
                  f"ext_up={s['max_ext_up']:.2f}, final_dt={s['final_day_type']}, "
                  f"tb_conf={s['max_tb_conf']:.2f}")


if __name__ == '__main__':
    analyze_all_sessions()
