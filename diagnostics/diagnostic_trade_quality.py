"""
Diagnostic 4: Analyze order flow quality at entry bar for each trade.
Identify which order flow signals distinguish winners from losers.
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

    print(f"\n{'='*120}")
    print(f"  ORDER FLOW QUALITY AT VWAP PULLBACK ENTRY BARS")
    print(f"{'='*120}\n")

    # For each session that has acceptance, find the first VWAP pullback entry bar
    # and analyze its order flow characteristics
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
        entry_bar = None

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

            # Check VWAP pullback conditions (matching the actual strategy)
            vwap = bar.get('vwap')
            if vwap is None or pd.isna(vwap):
                continue

            vwap_dist = abs(current_price - vwap) / ib_range if ib_range > 0 else 999

            if (vwap_dist < 0.40 and current_price > vwap and delta > 0 and
                current_price > ib_high and
                strength.value in ('moderate', 'strong', 'super') and
                day_type.value in ('trend_up', 'super_trend_up', 'p_day') and
                day_conf.trend_bull >= 0.375 and
                bar_time and bar_time < time(13, 0)):

                entry_bar = bar
                entry_bar_time = bar_time
                entry_ext = ext

                # Compute the session result (did price close above entry?)
                eod_bar = post_ib_df.iloc[-1]
                eod_close = eod_bar['close']
                pnl_pts = eod_close - current_price

                # Get order flow metrics for this bar and surrounding bars
                # Look at the 5-bar window around entry
                start_idx = max(0, bar_idx - 2)
                end_idx = min(len(post_ib_df) - 1, bar_idx + 2)
                window = post_ib_df.iloc[start_idx:end_idx + 1]

                # Cumulative delta at entry
                cum_delta = bar.get('cumulative_delta', 0)
                delta_zscore = bar.get('delta_zscore', 0)
                delta_pctl = bar.get('delta_percentile', 50)
                imbalance = bar.get('imbalance_ratio', 0)
                volume_spike = bar.get('volume_spike', 1.0)
                fvg_bull = bar.get('fvg_bull', False)
                fvg_bull_15m = bar.get('fvg_bull_15m', False)
                ifvg_bull = bar.get('ifvg_bull_entry', False)

                # 5-bar delta sum
                window_delta = sum(window.get('delta', pd.Series(dtype=float)).fillna(0))
                window_vol_avg = window['volume'].mean() if 'volume' in window.columns else 0

                # Check: was the delta momentum building before entry?
                pre_bars = post_ib_df.iloc[max(0, bar_idx - 10):bar_idx]
                if len(pre_bars) > 0:
                    pre_delta_sum = sum(pre_bars.get('delta', pd.Series(dtype=float)).fillna(0))
                    pre_delta_avg = pre_delta_sum / len(pre_bars)
                else:
                    pre_delta_sum = 0
                    pre_delta_avg = 0

                result_tag = 'WIN' if pnl_pts > 5 else ('FLAT' if pnl_pts > -5 else 'LOSS')

                print(f"--- {session_date} ---  [{result_tag}: {pnl_pts:+.1f} pts]")
                print(f"  Entry: time={entry_bar_time}, price={current_price:.1f}, ext={entry_ext:.2f}")
                print(f"  VWAP: {vwap:.1f} (dist={vwap_dist:.3f}x IB)")
                print(f"  Delta: {delta:.0f} | zscore={delta_zscore:.2f} | pctl={delta_pctl:.0f}th | "
                      f"cumDelta={cum_delta:.0f}")
                print(f"  Imbalance: {imbalance:.3f} | VolSpike={volume_spike:.2f}")
                print(f"  FVG: bull={fvg_bull} | bull_15m={fvg_bull_15m} | ifvg={ifvg_bull}")
                print(f"  Window (5-bar): delta_sum={window_delta:.0f}")
                print(f"  Pre-entry (10-bar): delta_sum={pre_delta_sum:.0f} | delta_avg={pre_delta_avg:.0f}")
                print(f"  tb_conf={day_conf.trend_bull:.2f} | pb_conf={day_conf.p_day_bull:.2f}")
                print()

                break  # Only first entry per session


if __name__ == '__main__':
    analyze()
