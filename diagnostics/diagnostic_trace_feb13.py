"""
Trace the 2026-02-13 entry to see why the OF quality gate didn't filter it.
The deep OF study showed: delta=-99, pctl=25, imb=0.871, vol_spike=0.77.
But that was based on matching by entry_price from the trade log.
Let's trace bar-by-bar to find the actual entry bar the strategy sees.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
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

df = load_csv('NQ')
df = filter_rth(df)
df = compute_all_features(df)

if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
if 'session_date' not in df.columns:
    df['session_date'] = df['timestamp'].dt.date

# Focus on 2026-02-13
session_df = df[df['session_date'].astype(str) == '2026-02-13'].copy()
print(f"Session rows: {len(session_df)}")
if len(session_df) == 0:
    # Try string match on timestamp
    session_df = df[df['timestamp'].dt.strftime('%Y-%m-%d') == '2026-02-13'].copy()
    print(f"After timestamp match: {len(session_df)}")

ib_df = session_df.head(IB_BARS_1MIN)
ib_high = ib_df['high'].max()
ib_low = ib_df['low'].min()
ib_range = ib_high - ib_low
ib_mid = (ib_high + ib_low) / 2

print(f"IB: high={ib_high}, low={ib_low}, range={ib_range}")

post_ib = session_df.iloc[IB_BARS_1MIN:]

scorer = DayTypeConfidenceScorer()
atr = ib_df.iloc[-1].get('atr14', 0.0)
scorer.on_session_start(ib_high, ib_low, ib_range, atr)

consecutive_above = 0
acceptance_confirmed = False
delta_history = []

for bar_idx in range(len(post_ib)):
    bar = post_ib.iloc[bar_idx]
    ts = bar['timestamp']
    bar_time = ts.time()
    current_price = bar['close']
    delta = bar.get('delta', 0)

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
            print(f"Acceptance at bar_idx={bar_idx}, time={bar_time}, price={current_price}")
        continue

    # Check VWAP pullback conditions
    vwap = bar.get('vwap')
    if vwap is None or pd.isna(vwap):
        continue

    vwap_dist = abs(current_price - vwap) / ib_range if ib_range > 0 else 999

    if bar_time >= PM_SESSION_START:
        continue

    # Check if this bar would be a VWAP pullback entry
    if (vwap_dist < 0.40 and current_price > vwap and delta > 0
        and current_price > ib_high
        and strength.value in ('moderate', 'strong', 'super')
        and day_type.value in ('trend_up', 'super_trend_up', 'p_day')
        and day_conf.trend_bull >= 0.375):

        pre_delta_sum = sum(delta_history[:-1]) if len(delta_history) > 1 else 0

        delta_pctl = bar.get('delta_percentile', 50)
        imbalance = bar.get('imbalance_ratio', 1.0)
        vol_spike = bar.get('volume_spike', 1.0)

        of_quality = sum([
            (delta_pctl >= 60) if not pd.isna(delta_pctl) else True,
            (imbalance > 1.0) if not pd.isna(imbalance) else True,
            (vol_spike >= 1.0) if not pd.isna(vol_spike) else True,
        ])

        print(f"\n*** VWAP ENTRY CANDIDATE at {bar_time} ***")
        print(f"  price={current_price}, vwap={vwap}, dist={vwap_dist:.3f}")
        print(f"  delta={delta:.0f}, pctl={delta_pctl:.0f}, zscore={bar.get('delta_zscore', 0):.2f}")
        print(f"  imbalance={imbalance:.3f}, vol_spike={vol_spike:.2f}")
        print(f"  pre_delta_sum={pre_delta_sum:.0f}")
        print(f"  OF quality={of_quality} (pctl>60: {delta_pctl >= 60}, imb>1.0: {imbalance > 1.0}, vol>=1.0: {vol_spike >= 1.0})")
        print(f"  day_type={day_type.value}, strength={strength.value}, tb_conf={day_conf.trend_bull:.3f}")
        print(f"  Would pass pre_delta: {pre_delta_sum >= -500}")
        print(f"  Would pass OF gate: {of_quality >= 2}")
