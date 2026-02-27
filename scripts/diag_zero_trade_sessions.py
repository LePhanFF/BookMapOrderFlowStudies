"""Diagnose WHY each session has zero trades for specific strategies.

For each session x strategy combination, trace through the entry criteria
and identify the exact gate that blocks the signal.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from collections import defaultdict
from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from config.instruments import get_instrument
from config.constants import IB_BARS_1MIN
from strategy.day_type import classify_day_type, classify_trend_strength, DayType, TrendStrength

full_df = load_csv('NQ')
df = filter_rth(full_df)
df = compute_all_features(df)

sessions = df.groupby('session_date')
dates = sorted(sessions.groups.keys())
print(f"Analyzing {len(dates)} sessions...")

# For each session, trace the day type classification at every bar
# and identify what prevents each strategy from firing

results = []
for date in dates:
    sdf = sessions.get_group(date)
    if len(sdf) < IB_BARS_1MIN:
        continue

    ib_df = sdf.iloc[:IB_BARS_1MIN]
    ib_high = ib_df['high'].max()
    ib_low = ib_df['low'].min()
    ib_range = ib_high - ib_low
    ib_mid = (ib_high + ib_low) / 2

    if ib_range <= 0:
        continue

    post_ib = sdf.iloc[IB_BARS_1MIN:]
    if len(post_ib) == 0:
        continue

    # Track dynamic day type at each bar
    day_types_seen = set()
    max_ext_up = 0
    max_ext_down = 0
    bars_above_ib = 0
    bars_below_ib = 0
    bars_inside = 0

    # Track VWAP alignment opportunities
    vwap_aligned_bull = 0
    vwap_aligned_bear = 0
    delta_positive_bars = 0
    delta_negative_bars = 0

    # Track acceptance bars
    consec_above_ibh = 0
    max_consec_above = 0
    consec_below_ibl = 0
    max_consec_below = 0

    # Edge fade: price near IB edge opportunities
    near_ibh = 0
    near_ibl = 0

    for i in range(len(post_ib)):
        bar = post_ib.iloc[i]
        close = bar['close']
        vwap = bar.get('vwap', np.nan)
        delta = bar.get('delta', bar.get('vol_delta', 0))
        if pd.isna(delta):
            delta = 0

        # Extension
        if close > ib_mid:
            ext = (close - ib_mid) / ib_range
            max_ext_up = max(max_ext_up, ext)
        else:
            ext = (ib_mid - close) / ib_range
            max_ext_down = max(max_ext_down, ext)

        # IB position
        if close > ib_high:
            bars_above_ib += 1
            consec_above_ibh += 1
            max_consec_above = max(max_consec_above, consec_above_ibh)
            consec_below_ibl = 0
        elif close < ib_low:
            bars_below_ib += 1
            consec_below_ibl += 1
            max_consec_below = max(max_consec_below, consec_below_ibl)
            consec_above_ibh = 0
        else:
            bars_inside += 1
            consec_above_ibh = 0
            consec_below_ibl = 0

        # VWAP alignment
        if not pd.isna(vwap):
            vwap_dist = abs(close - vwap)
            if close > ib_high and vwap_dist < ib_range * 0.3:
                vwap_aligned_bull += 1
            if close < ib_low and vwap_dist < ib_range * 0.3:
                vwap_aligned_bear += 1

        # Delta
        if delta > 0:
            delta_positive_bars += 1
        elif delta < 0:
            delta_negative_bars += 1

        # Near IB edges (for edge fade)
        if abs(close - ib_high) < ib_range * 0.15:
            near_ibh += 1
        if abs(close - ib_low) < ib_range * 0.15:
            near_ibl += 1

        # Dynamic day type
        if close > ib_high:
            direction = 'BULL'
        elif close < ib_low:
            direction = 'BEAR'
        else:
            direction = 'INSIDE'
        ts = classify_trend_strength(ext)
        dt = classify_day_type(ib_high, ib_low, close, direction, ts)
        day_types_seen.add(dt.value)

    # Session-level stats
    session_high = sdf['high'].max()
    session_low = sdf['low'].min()
    session_range = session_high - session_low
    total_bars = len(post_ib)

    # EOD day type
    eod_ext_up = (session_high - ib_high) / ib_range
    eod_ext_down = (ib_low - session_low) / ib_range

    month = str(date)[:7]

    results.append({
        'date': str(date)[:10],
        'month': month,
        'ib_range': ib_range,
        'session_range': session_range,
        'max_ext_up': max_ext_up,
        'max_ext_down': max_ext_down,
        'eod_ext_up': eod_ext_up,
        'eod_ext_down': eod_ext_down,
        'bars_above_ib': bars_above_ib,
        'bars_below_ib': bars_below_ib,
        'bars_inside': bars_inside,
        'total_bars': total_bars,
        'day_types_seen': day_types_seen,
        'vwap_aligned_bull': vwap_aligned_bull,
        'vwap_aligned_bear': vwap_aligned_bear,
        'delta_pos': delta_positive_bars,
        'delta_neg': delta_negative_bars,
        'max_consec_above': max_consec_above,
        'max_consec_below': max_consec_below,
        'near_ibh': near_ibh,
        'near_ibl': near_ibl,
    })

rdf = pd.DataFrame(results)

# === ANALYSIS 1: How often does each day type appear dynamically? ===
print("\n" + "=" * 80)
print("DYNAMIC DAY TYPE APPEARANCE (at any point during session)")
print("=" * 80)
dt_counts = defaultdict(int)
for r in results:
    for dt in r['day_types_seen']:
        dt_counts[dt] += 1
for dt, count in sorted(dt_counts.items(), key=lambda x: -x[1]):
    print(f"  {dt:20s}: {count:>4d} / {len(results)} sessions ({count/len(results)*100:.0f}%)")

# === ANALYSIS 2: Why Trend Bull misses trend_up sessions ===
print("\n" + "=" * 80)
print("TREND BULL: Why 40+ trend_up sessions have NO trade")
print("=" * 80)
trend_up_sessions = [r for r in results if r['eod_ext_up'] > 1.0 and r['eod_ext_down'] < 0.3]
print(f"EOD trend_up sessions: {len(trend_up_sessions)}")

# Check each potential gate
no_dynamic_trend = 0
no_vwap = 0
no_acceptance = 0
low_ib = 0
for r in trend_up_sessions:
    issues = []
    if 'trend_up' not in r['day_types_seen'] and 'super_trend_up' not in r['day_types_seen']:
        no_dynamic_trend += 1
        issues.append('never_classified_trend')
    if r['vwap_aligned_bull'] == 0:
        no_vwap += 1
        issues.append('no_vwap_alignment')
    if r['max_consec_above'] < 3:
        no_acceptance += 1
        issues.append(f'acceptance={r["max_consec_above"]}')
    if r['ib_range'] < 100:
        low_ib += 1
        issues.append(f'ib={r["ib_range"]:.0f}')
    if issues:
        print(f"  {r['date']}: IB={r['ib_range']:.0f}, max_ext={r['max_ext_up']:.2f}x, "
              f"vwap_bars={r['vwap_aligned_bull']}, accept={r['max_consec_above']}, "
              f"issues=[{', '.join(issues)}]")

print(f"\nSummary of kill reasons (not mutually exclusive):")
print(f"  Never classified as trend_up dynamically: {no_dynamic_trend}")
print(f"  No VWAP-aligned bars: {no_vwap}")
print(f"  Acceptance < 3 bars: {no_acceptance}")
print(f"  IB range < 100: {low_ib}")

# === ANALYSIS 3: Why B-Day misses b_day sessions ===
print("\n" + "=" * 80)
print("B-DAY: Why 35+ balance sessions have NO trade")
print("=" * 80)
bday_sessions = [r for r in results if
    max(r['eod_ext_up'], r['eod_ext_down']) < 0.5 and
    max(r['eod_ext_up'], r['eod_ext_down']) > 0.1]
print(f"EOD b_day-like sessions: {len(bday_sessions)}")

no_dynamic_bday = 0
no_edge = 0
low_ib_b = 0
for r in bday_sessions:
    issues = []
    if 'b_day' not in r['day_types_seen']:
        no_dynamic_bday += 1
        issues.append('never_b_day')
    if r['near_ibh'] == 0 and r['near_ibl'] == 0:
        no_edge += 1
        issues.append('no_edge_touch')
    if r['ib_range'] < 100:
        low_ib_b += 1
        issues.append(f'ib={r["ib_range"]:.0f}')

print(f"  Never classified as b_day dynamically: {no_dynamic_bday}")
print(f"  No IB edge touch: {no_edge}")
print(f"  IB range < 100: {low_ib_b}")

# === ANALYSIS 4: Edge Fade coverage gaps ===
print("\n" + "=" * 80)
print("EDGE FADE: Coverage by IB range bucket")
print("=" * 80)
for lo, hi in [(0, 80), (80, 120), (120, 160), (160, 200), (200, 999)]:
    bucket = [r for r in results if lo <= r['ib_range'] < hi]
    has_bday = [r for r in bucket if 'b_day' in r['day_types_seen'] or 'neutral' in r['day_types_seen']]
    has_edge = [r for r in bucket if r['near_ibh'] > 0 or r['near_ibl'] > 0]
    print(f"  IB {lo}-{hi}: {len(bucket)} sessions, "
          f"{len(has_bday)} see b_day/neutral, "
          f"{len(has_edge)} have edge touch")

# === ANALYSIS 5: Sessions with ZERO trades possible ===
print("\n" + "=" * 80)
print("SESSIONS WHERE NO STRATEGY COULD FIRE")
print("=" * 80)
# A session is "dead" if: no directional extension AND no IB edge touch AND no OR sweep
dead_sessions = [r for r in results
    if r['max_ext_up'] < 0.3 and r['max_ext_down'] < 0.3
    and r['near_ibh'] == 0 and r['near_ibl'] == 0]
print(f"Truly dead sessions (no extension, no edge touch): {len(dead_sessions)}")
for r in dead_sessions[:10]:
    print(f"  {r['date']}: IB={r['ib_range']:.0f}, range={r['session_range']:.0f}, "
          f"ext_up={r['max_ext_up']:.2f}, ext_dn={r['max_ext_down']:.2f}")

# === ANALYSIS 6: Monthly coverage ===
print("\n" + "=" * 80)
print("SESSIONS WITH AT LEAST ONE STRATEGY OPPORTUNITY")
print("=" * 80)
print(f"{'Month':>10s} {'Sessions':>8s} {'Has trend':>10s} {'Has bday':>10s} {'Has OR':>8s} {'Dead':>6s}")
for month in sorted(rdf['month'].unique()):
    mdf = rdf[rdf['month'] == month]
    n = len(mdf)
    mresults = [r for r in results if r['month'] == month]
    has_trend = sum(1 for r in mresults if r['max_ext_up'] > 0.5 or r['max_ext_down'] > 0.5)
    has_bday = sum(1 for r in mresults if 'b_day' in r['day_types_seen'] or 'neutral' in r['day_types_seen'])
    has_or = sum(1 for r in mresults if r['max_ext_up'] > 0.2 or r['max_ext_down'] > 0.2)
    dead = sum(1 for r in mresults if r['max_ext_up'] < 0.3 and r['max_ext_down'] < 0.3
               and r['near_ibh'] == 0 and r['near_ibl'] == 0)
    print(f"{month:>10s} {n:>8d} {has_trend:>10d} {has_bday:>10d} {has_or:>8d} {dead:>6d}")
