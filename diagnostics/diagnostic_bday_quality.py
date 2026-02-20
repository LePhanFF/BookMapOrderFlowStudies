"""
Diagnostic 5: Analyze B-Day entry quality for winners vs losers.
Find order flow patterns that could filter out the 2 losing B-Day trades.
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
from config.constants import IB_BARS_1MIN, BDAY_COOLDOWN_BARS, BDAY_STOP_IB_BUFFER
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
    print(f"  B-DAY ENTRY QUALITY ANALYSIS: ORDER FLOW AT IBL FADE ENTRIES")
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

        scorer = DayTypeConfidenceScorer()
        atr = ib_df.iloc[-1].get('atr14', 0.0)
        scorer.on_session_start(ib_high, ib_low, ib_range, atr)

        # Simulate B-Day strategy logic
        val_fade_taken = False
        last_entry_bar = -999
        ibl_touch_count = 0
        ibl_last_touch_bar = -999

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

            # Check B-Day conditions
            if day_type.value != 'b_day':
                continue
            if strength.value != 'weak':
                continue
            if day_conf.b_day < 0.5:
                continue
            if bar_idx - last_entry_bar < BDAY_COOLDOWN_BARS:
                continue
            if val_fade_taken:
                continue

            # Track touches
            if bar['low'] <= ib_low:
                if bar_idx - ibl_last_touch_bar <= 3:
                    ibl_touch_count += 1
                else:
                    ibl_touch_count = 1
                ibl_last_touch_bar = bar_idx

            # Check IBL fade conditions
            if bar['low'] <= ib_low and current_price > ib_low:
                has_fvg = bar.get('ifvg_bull_entry', False) or bar.get('fvg_bull', False)
                has_fvg_15m = bar.get('fvg_bull_15m', False)
                has_delta_rejection = delta > 0
                has_multi_touch = ibl_touch_count >= 2
                has_volume_spike = bar.get('volume_spike', 1.0) > 1.3

                quality_count = sum([
                    bool(has_fvg or has_fvg_15m),
                    has_delta_rejection,
                    has_multi_touch,
                    has_volume_spike,
                ])

                if quality_count >= 2 and has_delta_rejection:
                    # This is a B-Day entry — analyze it
                    entry_price = current_price
                    stop_price = ib_low - (ib_range * BDAY_STOP_IB_BUFFER)
                    target_price = ib_mid

                    # Get session outcome
                    remaining_bars = post_ib_df.iloc[bar_idx + 1:]

                    # Check stop and target
                    hit_stop = False
                    hit_target = False
                    eod_price = post_ib_df.iloc[-1]['close']
                    exit_price = eod_price

                    for future_idx in range(len(remaining_bars)):
                        future_bar = remaining_bars.iloc[future_idx]
                        if future_bar['low'] <= stop_price:
                            hit_stop = True
                            exit_price = stop_price
                            break
                        if future_bar['high'] >= target_price:
                            hit_target = True
                            exit_price = target_price
                            break

                    pnl_pts = exit_price - entry_price
                    result_tag = 'WIN' if pnl_pts > 0 else 'LOSS'

                    # Order flow analysis
                    cum_delta = bar.get('cumulative_delta', 0)
                    delta_zscore = bar.get('delta_zscore', 0)
                    delta_pctl = bar.get('delta_percentile', 50)
                    imbalance = bar.get('imbalance_ratio', 0)
                    volume_spike = bar.get('volume_spike', 1.0)

                    # Pre-entry delta momentum
                    pre_start = max(0, bar_idx - 10)
                    pre_bars = post_ib_df.iloc[pre_start:bar_idx]
                    pre_delta_sum = sum(pre_bars['delta'].fillna(0)) if 'delta' in pre_bars.columns else 0

                    # Post-IBL-touch delta flow (3 bars after touch)
                    post_start = bar_idx + 1
                    post_end = min(len(post_ib_df), bar_idx + 4)
                    post_bars = post_ib_df.iloc[post_start:post_end]
                    post_delta_sum = sum(post_bars['delta'].fillna(0)) if 'delta' in post_bars.columns and len(post_bars) > 0 else 0

                    # How far price dropped into IBL before rejecting
                    penetration = ib_low - bar['low']

                    # Bid volume vs ask volume on this bar
                    vol_bid = bar.get('vol_bid', 0)
                    vol_ask = bar.get('vol_ask', 0)

                    print(f"--- {session_date} ---  [{result_tag}: {pnl_pts:+.1f} pts]")
                    print(f"  Entry: time={bar_time}, price={entry_price:.1f}, stop={stop_price:.1f}, target={target_price:.1f}")
                    print(f"  IBL penetration: {penetration:.1f} pts")
                    print(f"  IB Range: {ib_range:.1f} | b_day_conf={day_conf.b_day:.2f}")
                    print(f"  Delta: {delta:.0f} | zscore={delta_zscore:.2f} | pctl={delta_pctl:.0f}th")
                    print(f"  CumDelta: {cum_delta:.0f}")
                    print(f"  Imbalance: {imbalance:.3f} | VolSpike={volume_spike:.2f}")
                    print(f"  Vol_Ask={vol_ask:.0f} | Vol_Bid={vol_bid:.0f}")
                    print(f"  FVG: bull={has_fvg or has_fvg_15m} | MultiTouch={has_multi_touch} (count={ibl_touch_count})")
                    print(f"  Quality: {quality_count} (fvg={has_fvg or has_fvg_15m}, delta+={has_delta_rejection}, "
                          f"multi={has_multi_touch}, volSpike={has_volume_spike})")
                    print(f"  Pre-entry (10-bar) delta: {pre_delta_sum:.0f}")
                    print(f"  Post-entry (3-bar) delta: {post_delta_sum:.0f}")
                    print(f"  Exit: {'STOP' if hit_stop else 'TARGET' if hit_target else 'EOD'} at {exit_price:.1f}")
                    print()

                    val_fade_taken = True
                    last_entry_bar = bar_idx


if __name__ == '__main__':
    analyze()
