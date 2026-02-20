"""
Diagnostic 9: Study order flow features at VWAP pullback moments on sessions
where we currently DON'T take trades.

Can better order flow understanding capture more sessions?
For each session with acceptance, check every bar that gets "close" to a VWAP
pullback entry and report what order flow features look like there.

Also: Check if loosening delta > 0 requirement to delta_percentile >= 60
or imbalance_ratio > 1.0 (net buying pressure) could capture more entries.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import time

from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from config.constants import IB_BARS_1MIN, ACCEPTANCE_MIN_BARS, PM_SESSION_START
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

    # Sessions that already have trades (from trade log)
    trade_log = pd.read_csv(Path(project_root) / 'output' / 'trade_log.csv')
    traded_sessions = set(trade_log['session_date'].str[:10].unique())

    print(f"\n{'='*140}")
    print(f"  MISSED SESSION ORDER FLOW: VWAP PULLBACK NEAR-MISSES")
    print(f"{'='*140}\n")

    near_miss_count = 0
    captured_with_relaxed = 0

    for session_date in sessions:
        session_str = str(session_date)[:10]

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

        delta_history = []
        near_miss_bars = []

        for bar_idx in range(len(post_ib_df)):
            bar = post_ib_df.iloc[bar_idx]
            timestamp = bar['timestamp'] if 'timestamp' in bar.index else None
            bar_time = timestamp.time() if timestamp and hasattr(timestamp, 'time') else None
            current_price = bar['close']
            delta = bar.get('delta', 0)

            # Track delta history
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
                continue

            # After acceptance: look for near-VWAP bars
            if bar_time and bar_time >= PM_SESSION_START:
                continue

            vwap = bar.get('vwap')
            if vwap is None or pd.isna(vwap):
                continue

            vwap_dist = abs(current_price - vwap) / ib_range if ib_range > 0 else 999

            # Check if this bar is near VWAP (within 0.5x IB — slightly wider than 0.4x)
            if vwap_dist < 0.50 and current_price > vwap:
                is_valid_dt = day_type.value in ('trend_up', 'super_trend_up', 'p_day')
                meets_strength = strength.value in ('moderate', 'strong', 'super')
                meets_conf = day_conf.trend_bull >= 0.375
                above_ibh = current_price > ib_high

                pre_delta_sum = sum(delta_history[:-1]) if len(delta_history) > 1 else 0

                # Current filter: delta > 0
                current_passes = (delta > 0 and vwap_dist < 0.40 and is_valid_dt
                                  and meets_strength and meets_conf and above_ibh
                                  and pre_delta_sum >= -500)

                # Get all order flow features
                delta_pctl = bar.get('delta_percentile', 50)
                delta_zs = bar.get('delta_zscore', 0)
                imb = bar.get('imbalance_ratio', 1.0)
                vol_spike = bar.get('volume_spike', 1.0)
                cvd = bar.get('cumulative_delta', 0)
                cvd_ma = bar.get('cumulative_delta_ma', 0)
                cvd_above = (cvd > cvd_ma) if not (pd.isna(cvd) or pd.isna(cvd_ma)) else None
                fvg_bull = bar.get('fvg_bull', False)
                ifvg = bar.get('ifvg_bull_entry', False)
                fvg_15m = bar.get('fvg_bull_15m', False)

                # Relaxed filters:
                # A: delta_percentile >= 60 (instead of delta > 0)
                relaxed_a = (delta_pctl >= 60 and vwap_dist < 0.40 and is_valid_dt
                            and meets_strength and meets_conf and above_ibh
                            and pre_delta_sum >= -500)

                # B: imbalance_ratio > 1.0 (net buying pressure)
                relaxed_b = (imb > 1.0 and vwap_dist < 0.40 and is_valid_dt
                            and meets_strength and meets_conf and above_ibh
                            and pre_delta_sum >= -500)

                # C: volume_spike > 1.2 + delta_percentile >= 50
                relaxed_c = (vol_spike > 1.2 and delta_pctl >= 50 and vwap_dist < 0.40
                            and is_valid_dt and meets_strength and meets_conf and above_ibh
                            and pre_delta_sum >= -500)

                # EOD outcome
                eod_close = post_ib_df.iloc[-1]['close']
                eod_pnl = eod_close - current_price

                near_miss_bars.append({
                    'bar_time': bar_time,
                    'price': current_price,
                    'vwap': vwap,
                    'vwap_dist': vwap_dist,
                    'delta': delta,
                    'delta_pctl': delta_pctl,
                    'delta_zs': delta_zs,
                    'imb': imb,
                    'vol_spike': vol_spike,
                    'cvd_above': cvd_above,
                    'fvg_bull': fvg_bull,
                    'ifvg': ifvg,
                    'fvg_15m': fvg_15m,
                    'pre_delta': pre_delta_sum,
                    'day_type': day_type.value,
                    'strength': strength.value,
                    'tb_conf': day_conf.trend_bull,
                    'above_ibh': above_ibh,
                    'is_valid_dt': is_valid_dt,
                    'meets_strength': meets_strength,
                    'meets_conf': meets_conf,
                    'current_passes': current_passes,
                    'relaxed_a': relaxed_a,
                    'relaxed_b': relaxed_b,
                    'relaxed_c': relaxed_c,
                    'eod_pnl': eod_pnl,
                })

        if not acceptance_confirmed:
            continue

        if not near_miss_bars:
            continue

        # Only show sessions that have near misses but no current entry
        has_current_entry = any(b['current_passes'] for b in near_miss_bars)
        has_relaxed_entry = any(b['relaxed_a'] or b['relaxed_b'] for b in near_miss_bars)

        if session_str in traded_sessions and has_current_entry:
            continue  # Skip — already traded and captured

        eod_close = post_ib_df.iloc[-1]['close']
        eod_from_ibh = eod_close - ib_high

        print(f"--- {session_str} --- {'TRADED' if session_str in traded_sessions else 'NOT TRADED'} | EOD from IBH: {eod_from_ibh:+.1f}")

        # Show first few near-miss bars
        for b in near_miss_bars[:5]:
            flags = []
            if b['current_passes']: flags.append('CURRENT')
            if b['relaxed_a']: flags.append('RELAX_A')
            if b['relaxed_b']: flags.append('RELAX_B')
            if b['relaxed_c']: flags.append('RELAX_C')
            flag_str = ' | '.join(flags) if flags else 'NO_PASS'

            fail_reasons = []
            if not b['is_valid_dt']: fail_reasons.append(f"dt={b['day_type']}")
            if not b['meets_strength']: fail_reasons.append(f"str={b['strength']}")
            if not b['meets_conf']: fail_reasons.append(f"conf={b['tb_conf']:.3f}")
            if not b['above_ibh']: fail_reasons.append('below_IBH')
            if b['vwap_dist'] >= 0.40: fail_reasons.append(f"vwap_dist={b['vwap_dist']:.3f}")
            if b['delta'] <= 0: fail_reasons.append(f"delta={b['delta']:.0f}")
            if b['pre_delta'] < -500: fail_reasons.append(f"pre_delta={b['pre_delta']:.0f}")
            fail_str = ' | '.join(fail_reasons) if fail_reasons else 'ALL_PASS'

            print(f"  {b['bar_time']}: price={b['price']:.1f} vwap_dist={b['vwap_dist']:.3f} "
                  f"delta={b['delta']:.0f} pctl={b['delta_pctl']:.0f} zs={b['delta_zs']:.2f} "
                  f"imb={b['imb']:.3f} vol_sp={b['vol_spike']:.2f} cvd>ma={b['cvd_above']} "
                  f"fvg={b['fvg_bull']} eod_pnl={b['eod_pnl']:+.1f}")
            print(f"    [{flag_str}] Blocks: {fail_str}")

        if not has_current_entry and has_relaxed_entry:
            near_miss_count += 1
            # Would the first relaxed entry be a winner?
            first_relaxed = next((b for b in near_miss_bars if b['relaxed_a'] or b['relaxed_b']), None)
            if first_relaxed and first_relaxed['eod_pnl'] > 0:
                captured_with_relaxed += 1
                print(f"  >>> WOULD CAPTURE WITH RELAXED FILTER: eod_pnl={first_relaxed['eod_pnl']:+.1f}")
            elif first_relaxed:
                print(f"  >>> RELAXED WOULD ADD LOSER: eod_pnl={first_relaxed['eod_pnl']:+.1f}")

        print()

    print(f"\n{'='*70}")
    print(f"Sessions with near-misses captured by relaxed filters: {near_miss_count}")
    print(f"Of those, would be winners: {captured_with_relaxed}")
    print(f"{'='*70}")


if __name__ == '__main__':
    analyze()
