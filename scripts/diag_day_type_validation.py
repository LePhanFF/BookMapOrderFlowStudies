"""Validate day-type classification across 259 sessions.

Tests:
1. Classification at 11:00 AM vs 1:00 PM vs EOD (3:30 PM)
2. PM morph detection (day type changes after 1:00 PM)
3. ATR regime correlation (low/med/high/extreme volatility)
4. Classification accuracy vs Dalton framework expectations
5. Per-month breakdown showing classification stability

Key question: Does classification break during high-vol (Apr-Jun tariff period)?
"""
import sys, warnings, io
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import time
from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from config.instruments import get_instrument
from config.constants import IB_BARS_1MIN, IB_EXT_WEAK, IB_EXT_MODERATE, IB_EXT_STRONG
from strategy.day_type import (
    classify_day_type, classify_trend_strength, DayType, TrendStrength
)

full_df = load_csv('NQ')
df = filter_rth(full_df)
df = compute_all_features(df)
instrument = get_instrument('MNQ')
point_value = instrument.point_value

sessions = sorted(df['session_date'].unique())
n_sessions = len(sessions)
print(f"Total sessions: {n_sessions}")
print(f"Date range: {sessions[0]} to {sessions[-1]}")

# Checkpoints: bars after session start
# 9:30 + 90min = 11:00 AM (bar 90), 9:30 + 210min = 1:00 PM (bar 210), 9:30 + 360min = 3:30 PM (bar 360)
CHECKPOINTS = {
    '11:00': 90,    # 60 IB bars + 30 post-IB
    '13:00': 210,   # 3.5 hrs into session
    'EOD':   None,   # last bar
}

# Storage
session_data = []

for session_date in sessions:
    sdf = df[df['session_date'] == session_date].copy()
    if len(sdf) < IB_BARS_1MIN + 10:
        continue

    # IB computation
    ib_df = sdf.head(IB_BARS_1MIN)
    ib_high = ib_df['high'].max()
    ib_low = ib_df['low'].min()
    ib_range = ib_high - ib_low
    ib_mid = (ib_high + ib_low) / 2

    if ib_range <= 0:
        continue

    # ATR proxy: mean of bar ranges in first 60 bars (IB period)
    bar_ranges = (ib_df['high'] - ib_df['low']).values
    avg_bar_range = np.mean(bar_ranges)
    max_bar_range = np.max(bar_ranges)

    # Full session ATR: average bar range across ALL RTH bars
    all_bar_ranges = (sdf['high'] - sdf['low']).values
    full_atr = np.mean(all_bar_ranges)
    full_max_bar = np.max(all_bar_ranges)

    # Session high/low for range
    session_high = sdf['high'].max()
    session_low = sdf['low'].min()
    session_range = session_high - session_low

    # Classify at each checkpoint
    checkpoints = {}
    post_ib = sdf.iloc[IB_BARS_1MIN:]

    for cp_name, cp_bar in CHECKPOINTS.items():
        if cp_name == 'EOD':
            idx = len(post_ib) - 1
        else:
            idx = min(cp_bar - IB_BARS_1MIN, len(post_ib) - 1)

        if idx < 0:
            continue

        bar = post_ib.iloc[idx]
        price = bar['close']

        if price > ib_high:
            ib_dir = 'BULL'
            ext = (price - ib_mid) / ib_range
        elif price < ib_low:
            ib_dir = 'BEAR'
            ext = (ib_mid - price) / ib_range
        else:
            ib_dir = 'INSIDE'
            ext = 0.0

        strength = classify_trend_strength(ext)
        dtype = classify_day_type(ib_high, ib_low, price, ib_dir, strength)
        checkpoints[cp_name] = {
            'day_type': dtype.value,
            'ext': ext,
            'strength': strength.value,
            'direction': ib_dir,
            'price': price,
        }

    # Session peak extension (max extension at any point)
    peak_ext_up = 0.0
    peak_ext_down = 0.0
    for i in range(len(post_ib)):
        bar = post_ib.iloc[i]
        # Check highs for bull extension
        h_ext = (bar['high'] - ib_mid) / ib_range
        l_ext = (ib_mid - bar['low']) / ib_range
        peak_ext_up = max(peak_ext_up, h_ext)
        peak_ext_down = max(peak_ext_down, l_ext)

    # Check if day type MORPHED (changed between 11AM and EOD)
    dt_11 = checkpoints.get('11:00', {}).get('day_type', '?')
    dt_13 = checkpoints.get('13:00', {}).get('day_type', '?')
    dt_eod = checkpoints.get('EOD', {}).get('day_type', '?')

    morphed_am_pm = dt_11 != dt_eod
    morphed_13_eod = dt_13 != dt_eod

    # Dalton "expected" type based on session range vs IB
    # A true trend day: session range >> IB, closes at extreme
    range_ratio = session_range / ib_range if ib_range > 0 else 0
    close_position = (sdf.iloc[-1]['close'] - session_low) / session_range if session_range > 0 else 0.5

    # Dalton expectation (using end-of-session data = "ideal" classification)
    eod_ext = checkpoints.get('EOD', {}).get('ext', 0)
    if eod_ext > 2.0:
        dalton_expected = 'super_trend'
    elif eod_ext > 1.0:
        dalton_expected = 'trend'
    elif eod_ext > 0.5:
        dalton_expected = 'p_day'
    elif eod_ext < 0.2:
        dalton_expected = 'b_day'
    else:
        dalton_expected = 'neutral'

    # Does engine agree at EOD?
    eod_type = dt_eod
    if eod_type in ('trend_up', 'trend_down'):
        engine_category = 'trend'
    elif eod_type in ('super_trend_up', 'super_trend_down'):
        engine_category = 'super_trend'
    else:
        engine_category = eod_type

    classification_match = engine_category == dalton_expected

    month = str(session_date)[:7]

    session_data.append({
        'date': session_date,
        'month': month,
        'ib_range': ib_range,
        'avg_bar_atr': avg_bar_range,
        'max_bar': max_bar_range,
        'full_atr': full_atr,
        'full_max_bar': full_max_bar,
        'session_range': session_range,
        'range_ratio': range_ratio,
        'close_pos': close_position,
        'peak_ext_up': peak_ext_up,
        'peak_ext_down': peak_ext_down,
        'dt_11': dt_11,
        'dt_13': dt_13,
        'dt_eod': dt_eod,
        'eod_ext': eod_ext,
        'eod_dir': checkpoints.get('EOD', {}).get('direction', '?'),
        'morphed_am_pm': morphed_am_pm,
        'morphed_13_eod': morphed_13_eod,
        'dalton_expected': dalton_expected,
        'engine_category': engine_category,
        'classification_match': classification_match,
    })

sdata = pd.DataFrame(session_data)

# ================================================================
# SECTION 1: Overall Classification Distribution
# ================================================================
print(f"\n{'='*120}")
print(f"  SECTION 1: DAY TYPE DISTRIBUTION (at each checkpoint)")
print(f"{'='*120}")

for cp in ['11:00', '13:00', 'EOD']:
    col = f'dt_{cp.replace(":", "")}' if cp != 'EOD' else 'dt_eod'
    if cp == '11:00':
        col = 'dt_11'
    elif cp == '13:00':
        col = 'dt_13'
    counts = sdata[col].value_counts()
    print(f"\n  At {cp}:")
    for dtype, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        pct = cnt / len(sdata) * 100
        bar = '#' * int(pct / 2)
        print(f"    {dtype:20s} {cnt:>4d} ({pct:>5.1f}%) {bar}")

# ================================================================
# SECTION 2: Classification Morphing (AM->PM changes)
# ================================================================
print(f"\n\n{'='*120}")
print(f"  SECTION 2: DAY TYPE MORPHING (classification changes over time)")
print(f"{'='*120}")

morph_11_eod = sdata[sdata['morphed_am_pm']].copy()
morph_13_eod = sdata[sdata['morphed_13_eod']].copy()

print(f"\n  Classification changes 11:00 -> EOD:  {len(morph_11_eod)} / {len(sdata)} ({len(morph_11_eod)/len(sdata)*100:.1f}%)")
print(f"  Classification changes 13:00 -> EOD:  {len(morph_13_eod)} / {len(sdata)} ({len(morph_13_eod)/len(sdata)*100:.1f}%)")

# Morph matrix: what changes to what
print(f"\n  MORPH MATRIX (11:00 -> EOD):")
print(f"  {'From\\To':15s}", end='')
eod_types = sorted(sdata['dt_eod'].unique())
for t in eod_types:
    print(f" {t:>12s}", end='')
print()
print("  " + "-" * (15 + 13 * len(eod_types)))

for from_type in sorted(sdata['dt_11'].unique()):
    subset = sdata[sdata['dt_11'] == from_type]
    print(f"  {from_type:15s}", end='')
    for to_type in eod_types:
        cnt = len(subset[subset['dt_eod'] == to_type])
        if cnt > 0:
            print(f" {cnt:>12d}", end='')
        else:
            print(f" {'':>12s}", end='')
    print()

# Morph matrix: 13:00 -> EOD
print(f"\n  MORPH MATRIX (13:00 -> EOD):")
print(f"  {'From\\To':15s}", end='')
for t in eod_types:
    print(f" {t:>12s}", end='')
print()
print("  " + "-" * (15 + 13 * len(eod_types)))

for from_type in sorted(sdata['dt_13'].unique()):
    subset = sdata[sdata['dt_13'] == from_type]
    print(f"  {from_type:15s}", end='')
    for to_type in eod_types:
        cnt = len(subset[subset['dt_eod'] == to_type])
        if cnt > 0:
            print(f" {cnt:>12d}", end='')
        else:
            print(f" {'':>12s}", end='')
    print()

# ================================================================
# SECTION 3: ATR / Volatility Regime Analysis
# ================================================================
print(f"\n\n{'='*120}")
print(f"  SECTION 3: VOLATILITY REGIME vs CLASSIFICATION")
print(f"{'='*120}")

# Define vol regimes by avg 1-min bar range
# User said: recent months 17-36 pts, April-June up to 120 pts
vol_bins = [0, 5, 10, 20, 40, 80, 200]
vol_labels = ['Ultra Low (<5)', 'Low (5-10)', 'Normal (10-20)', 'Elevated (20-40)', 'High (40-80)', 'Extreme (>80)']
sdata['vol_regime'] = pd.cut(sdata['avg_bar_atr'], bins=vol_bins, labels=vol_labels, right=False)

print(f"\n  Volatility = avg 1-min bar range during IB (high-low per bar)")
print(f"\n  {'Regime':20s} {'Sessions':>8s} {'IB Range':>10s} {'Max Bar':>10s} {'SessRange':>10s}")
print(f"  {'-'*65}")

for regime in vol_labels:
    sub = sdata[sdata['vol_regime'] == regime]
    if len(sub) == 0:
        continue
    print(f"  {regime:20s} {len(sub):>8d} {sub['ib_range'].mean():>8.0f}pts {sub['max_bar'].mean():>8.1f}pts {sub['session_range'].mean():>8.0f}pts")

# Classification distribution per vol regime
print(f"\n  DAY TYPE DISTRIBUTION BY VOLATILITY REGIME (at EOD):")
print(f"  {'Regime':20s}", end='')
day_types_all = sorted(sdata['dt_eod'].unique())
for dt in day_types_all:
    print(f" {dt:>12s}", end='')
print()
print(f"  {'-'*120}")

for regime in vol_labels:
    sub = sdata[sdata['vol_regime'] == regime]
    if len(sub) == 0:
        continue
    print(f"  {regime:20s}", end='')
    for dt in day_types_all:
        cnt = len(sub[sub['dt_eod'] == dt])
        if cnt > 0:
            pct = cnt / len(sub) * 100
            print(f" {cnt:>3d}({pct:>4.0f}%)", end='')
        else:
            print(f" {'':>12s}", end='')
    print(f"  n={len(sub)}")

# ================================================================
# SECTION 4: Classification Accuracy Check
# ================================================================
print(f"\n\n{'='*120}")
print(f"  SECTION 4: CLASSIFICATION ACCURACY (engine vs Dalton expectation)")
print(f"{'='*120}")

total_match = sdata['classification_match'].sum()
total = len(sdata)
print(f"\n  Overall match: {total_match}/{total} ({total_match/total*100:.1f}%)")

# Per day type
print(f"\n  {'Expected':15s} {'Correct':>8s} {'Total':>8s} {'Accuracy':>10s}")
print(f"  {'-'*50}")
for expected in sorted(sdata['dalton_expected'].unique()):
    sub = sdata[sdata['dalton_expected'] == expected]
    correct = sub['classification_match'].sum()
    print(f"  {expected:15s} {correct:>8d} {len(sub):>8d} {correct/len(sub)*100:>8.1f}%")

# Mismatches detail
mismatches = sdata[~sdata['classification_match']]
if len(mismatches) > 0:
    print(f"\n  MISMATCHES ({len(mismatches)} sessions):")
    print(f"  {'Date':12s} {'Expected':12s} {'Engine':12s} {'EOD Ext':>8s} {'IB Range':>8s} {'Avg ATR':>8s}")
    print(f"  {'-'*70}")
    for _, row in mismatches.iterrows():
        print(f"  {str(row['date'])[:10]:12s} {row['dalton_expected']:12s} {row['engine_category']:12s} "
              f"{row['eod_ext']:>7.2f}x {row['ib_range']:>7.0f} {row['avg_bar_atr']:>7.1f}")

# ================================================================
# SECTION 5: Monthly Classification Stability
# ================================================================
print(f"\n\n{'='*120}")
print(f"  SECTION 5: MONTHLY CLASSIFICATION BREAKDOWN")
print(f"{'='*120}")

months = sorted(sdata['month'].unique())
print(f"\n  {'Month':10s} {'Sess':>5s} {'IB_Rng':>8s} {'AvgATR':>8s} {'MaxBar':>8s}", end='')
for dt in ['trend_up', 'trend_down', 'p_day', 'b_day', 'neutral']:
    print(f" {dt:>10s}", end='')
print(f" {'Morphs':>8s} {'Accuracy':>10s}")
print(f"  {'-'*120}")

for month in months:
    sub = sdata[sdata['month'] == month]
    n = len(sub)
    avg_ib = sub['ib_range'].mean()
    avg_atr = sub['avg_bar_atr'].mean()
    max_bar = sub['full_max_bar'].mean()
    morphs = sub['morphed_am_pm'].sum()
    acc = sub['classification_match'].mean() * 100

    print(f"  {month:10s} {n:>5d} {avg_ib:>7.0f}pt {avg_atr:>7.1f}pt {max_bar:>7.1f}pt", end='')
    for dt in ['trend_up', 'trend_down', 'p_day', 'b_day', 'neutral']:
        cnt = len(sub[sub['dt_eod'] == dt])
        if cnt > 0:
            pct = cnt / n * 100
            print(f" {cnt:>3d}({pct:>3.0f}%)", end='')
        else:
            print(f" {'':>10s}", end='')
    print(f" {morphs:>8d} {acc:>8.1f}%")

# ================================================================
# SECTION 6: Problem Sessions (high vol + classification issues)
# ================================================================
print(f"\n\n{'='*120}")
print(f"  SECTION 6: PROBLEM SESSIONS (extreme vol or classification instability)")
print(f"{'='*120}")

# Sessions where 1-min bar range > 60 pts (extreme volatility)
extreme_bars = sdata[sdata['full_max_bar'] > 60].copy()
print(f"\n  Sessions with max 1-min bar > 60 pts: {len(extreme_bars)}")
if len(extreme_bars) > 0:
    print(f"  {'Date':12s} {'MaxBar':>8s} {'IB_Rng':>8s} {'SessRng':>10s} {'DT@11':>12s} {'DT@13':>12s} {'DT@EOD':>12s} {'Morph':>6s} {'Ext':>6s}")
    print(f"  {'-'*95}")
    for _, row in extreme_bars.sort_values('full_max_bar', ascending=False).iterrows():
        print(f"  {str(row['date'])[:10]:12s} {row['full_max_bar']:>7.1f}pt {row['ib_range']:>7.0f}pt {row['session_range']:>9.0f}pt "
              f"{row['dt_11']:>12s} {row['dt_13']:>12s} {row['dt_eod']:>12s} "
              f"{'YES' if row['morphed_am_pm'] else 'no':>6s} {row['eod_ext']:>5.2f}x")

# Sessions where classification seems wrong
# e.g., session range > 3x IB but classified as b_day
suspicious = sdata[
    ((sdata['range_ratio'] > 3.0) & (sdata['dt_eod'].isin(['b_day', 'neutral']))) |
    ((sdata['range_ratio'] < 1.2) & (sdata['dt_eod'].isin(['trend_up', 'trend_down', 'super_trend_up', 'super_trend_down'])))
].copy()

print(f"\n  SUSPICIOUS CLASSIFICATIONS: {len(suspicious)}")
print(f"  (Session range > 3x IB but b_day/neutral, OR range < 1.2x IB but trend)")
if len(suspicious) > 0:
    print(f"  {'Date':12s} {'Rng/IB':>8s} {'IB_Rng':>8s} {'SessRng':>10s} {'DT@EOD':>12s} {'Ext':>6s} {'ClosePos':>10s}")
    print(f"  {'-'*75}")
    for _, row in suspicious.sort_values('range_ratio', ascending=False).iterrows():
        print(f"  {str(row['date'])[:10]:12s} {row['range_ratio']:>7.2f}x {row['ib_range']:>7.0f}pt {row['session_range']:>9.0f}pt "
              f"{row['dt_eod']:>12s} {row['eod_ext']:>5.2f}x {row['close_pos']:>8.2f}")

# ================================================================
# SECTION 7: IB Range vs Classification Quality
# ================================================================
print(f"\n\n{'='*120}")
print(f"  SECTION 7: IB RANGE SIZE IMPACT ON CLASSIFICATION")
print(f"{'='*120}")

# When IB is very wide (>300pts), classification becomes problematic because
# extension thresholds are based on IB range -- a 300pt IB needs 150pt extension
# for P-day and 300pt for trend day
ib_bins = [0, 50, 100, 150, 200, 300, 500, 1000]
ib_labels = ['<50', '50-100', '100-150', '150-200', '200-300', '300-500', '>500']
sdata['ib_bucket'] = pd.cut(sdata['ib_range'], bins=ib_bins, labels=ib_labels, right=False)

print(f"\n  When IB is wide, extension thresholds scale UP (harder to classify as trend):")
print(f"  P-Day requires 0.5x IB from midpoint, Trend requires 1.0x from midpoint")
print(f"\n  {'IB Bucket':12s} {'Sess':>5s} {'Avg IB':>8s} {'Avg Ext':>8s} {'trend':>8s} {'p_day':>8s} {'b_day':>8s} {'neutral':>8s} {'Match%':>8s}")
print(f"  {'-'*85}")

for bucket in ib_labels:
    sub = sdata[sdata['ib_bucket'] == bucket]
    if len(sub) == 0:
        continue
    n = len(sub)
    trend_n = len(sub[sub['dt_eod'].isin(['trend_up', 'trend_down', 'super_trend_up', 'super_trend_down'])])
    p_n = len(sub[sub['dt_eod'] == 'p_day'])
    b_n = len(sub[sub['dt_eod'] == 'b_day'])
    neut_n = len(sub[sub['dt_eod'] == 'neutral'])
    match_pct = sub['classification_match'].mean() * 100

    print(f"  {bucket:12s} {n:>5d} {sub['ib_range'].mean():>7.0f}pt {sub['eod_ext'].mean():>7.2f}x "
          f"{trend_n:>3d}({trend_n/n*100:>3.0f}%) {p_n:>3d}({p_n/n*100:>3.0f}%) "
          f"{b_n:>3d}({b_n/n*100:>3.0f}%) {neut_n:>3d}({neut_n/n*100:>3.0f}%) {match_pct:>7.1f}%")

# ================================================================
# SECTION 8: Peak Extension vs EOD (do we classify too late/early?)
# ================================================================
print(f"\n\n{'='*120}")
print(f"  SECTION 8: PEAK EXTENSION vs EOD CLASSIFICATION")
print(f"{'='*120}")

# Sessions where peak extension was much higher than EOD
# This means the day LOOKED like a trend at peak but reverted
sdata['peak_ext'] = sdata[['peak_ext_up', 'peak_ext_down']].max(axis=1)
sdata['ext_diff'] = sdata['peak_ext'] - sdata['eod_ext']

print(f"\n  Peak ext vs EOD ext (positive = price reverted from extreme):")
print(f"  Mean peak ext: {sdata['peak_ext'].mean():.2f}x | Mean EOD ext: {sdata['eod_ext'].mean():.2f}x")
print(f"  Mean reversion: {sdata['ext_diff'].mean():.2f}x")

# Sessions with big reversion (peak > 1.0x but EOD < 0.5x)
big_reversions = sdata[(sdata['peak_ext'] > 1.0) & (sdata['eod_ext'] < 0.5)]
print(f"\n  Sessions: peak > 1.0x but EOD < 0.5x (trend fakeout): {len(big_reversions)}")
if len(big_reversions) > 0:
    print(f"  {'Date':12s} {'PeakExt':>8s} {'EODExt':>8s} {'DT@11':>12s} {'DT@13':>12s} {'DT@EOD':>12s} {'IB_Rng':>8s} {'AvgATR':>8s}")
    print(f"  {'-'*90}")
    for _, row in big_reversions.sort_values('ext_diff', ascending=False).head(20).iterrows():
        print(f"  {str(row['date'])[:10]:12s} {row['peak_ext']:>7.2f}x {row['eod_ext']:>7.2f}x "
              f"{row['dt_11']:>12s} {row['dt_13']:>12s} {row['dt_eod']:>12s} "
              f"{row['ib_range']:>7.0f}pt {row['avg_bar_atr']:>7.1f}pt")

# ================================================================
# SECTION 9: April-June vs Nov-Feb Comparison
# ================================================================
print(f"\n\n{'='*120}")
print(f"  SECTION 9: TARIFF PERIOD (Apr-Jun) vs RECENT (Nov-Feb) COMPARISON")
print(f"{'='*120}")

tariff = sdata[sdata['month'].isin(['2025-04', '2025-05', '2025-06'])]
recent = sdata[sdata['month'].isin(['2025-11', '2025-12', '2026-01', '2026-02'])]

for label, sub in [('Apr-Jun 2025 (Tariff)', tariff), ('Nov-Feb 2025-26 (Recent)', recent)]:
    n = len(sub)
    if n == 0:
        continue
    print(f"\n  {label}: {n} sessions")
    print(f"    Avg IB Range:     {sub['ib_range'].mean():>7.0f} pts")
    print(f"    Avg 1-min ATR:    {sub['avg_bar_atr'].mean():>7.1f} pts")
    print(f"    Max 1-min bar:    {sub['full_max_bar'].max():>7.1f} pts")
    print(f"    Avg Session Rng:  {sub['session_range'].mean():>7.0f} pts")
    print(f"    Avg EOD Extension:{sub['eod_ext'].mean():>7.2f}x")
    print(f"    Morph rate:       {sub['morphed_am_pm'].mean()*100:>5.1f}%")
    print(f"    Classification accuracy: {sub['classification_match'].mean()*100:.1f}%")

    print(f"    Day type distribution (EOD):")
    for dt in sorted(sub['dt_eod'].unique()):
        cnt = len(sub[sub['dt_eod'] == dt])
        print(f"      {dt:20s} {cnt:>3d} ({cnt/n*100:>5.1f}%)")

# ================================================================
# SECTION 10: Does classification work when VIX proxy is high?
# ================================================================
print(f"\n\n{'='*120}")
print(f"  SECTION 10: HIGH VOLATILITY CLASSIFICATION QUALITY")
print(f"{'='*120}")

# Use avg bar range as VIX proxy (since we don't have VIX data)
# User mentioned: recent = 17-36 pts bar range, tariff = up to 120 pts
print(f"\n  Using avg 1-min bar range as VIX proxy:")
print(f"    Median: {sdata['avg_bar_atr'].median():.1f} pts")
print(f"    p25: {sdata['avg_bar_atr'].quantile(0.25):.1f} pts")
print(f"    p75: {sdata['avg_bar_atr'].quantile(0.75):.1f} pts")
print(f"    p95: {sdata['avg_bar_atr'].quantile(0.95):.1f} pts")
print(f"    Max: {sdata['avg_bar_atr'].max():.1f} pts")

# Split into "normal" (< p75) vs "high" (> p75) vs "extreme" (> p95)
p75 = sdata['avg_bar_atr'].quantile(0.75)
p95 = sdata['avg_bar_atr'].quantile(0.95)

normal_vol = sdata[sdata['avg_bar_atr'] <= p75]
high_vol = sdata[(sdata['avg_bar_atr'] > p75) & (sdata['avg_bar_atr'] <= p95)]
extreme_vol = sdata[sdata['avg_bar_atr'] > p95]

print(f"\n  {'Vol Level':18s} {'Sessions':>8s} {'Acc%':>8s} {'Morph%':>8s} {'Avg IB':>8s} {'Avg Ext':>8s} {'trend%':>8s} {'b_day%':>8s}")
print(f"  {'-'*85}")

for label, sub in [('Normal (<p75)', normal_vol), ('High (p75-p95)', high_vol), ('Extreme (>p95)', extreme_vol)]:
    n = len(sub)
    if n == 0:
        continue
    acc = sub['classification_match'].mean() * 100
    morph = sub['morphed_am_pm'].mean() * 100
    trend_pct = len(sub[sub['dt_eod'].isin(['trend_up', 'trend_down', 'super_trend_up', 'super_trend_down'])]) / n * 100
    bday_pct = len(sub[sub['dt_eod'] == 'b_day']) / n * 100

    print(f"  {label:18s} {n:>8d} {acc:>7.1f}% {morph:>7.1f}% {sub['ib_range'].mean():>7.0f}pt {sub['eod_ext'].mean():>7.2f}x {trend_pct:>7.1f}% {bday_pct:>7.1f}%")

# Key question: in extreme vol, do we get MORE b_day/neutral (wrong) or MORE trend (right)?
print(f"\n  KEY QUESTION: Does extreme vol break classification?")
if len(extreme_vol) > 0:
    print(f"  Extreme vol ({len(extreme_vol)} sessions):")
    for dt in sorted(extreme_vol['dt_eod'].unique()):
        cnt = len(extreme_vol[extreme_vol['dt_eod'] == dt])
        print(f"    {dt:20s} {cnt:>3d} ({cnt/len(extreme_vol)*100:>5.1f}%)")

    # In extreme vol, B-day should be RARE (wide IB + high ATR = directional)
    bday_extreme = len(extreme_vol[extreme_vol['dt_eod'] == 'b_day'])
    neutral_extreme = len(extreme_vol[extreme_vol['dt_eod'] == 'neutral'])
    print(f"\n  B-day in extreme vol: {bday_extreme} ({bday_extreme/len(extreme_vol)*100:.0f}%)")
    print(f"  Neutral in extreme vol: {neutral_extreme} ({neutral_extreme/len(extreme_vol)*100:.0f}%)")

    if bday_extreme / len(extreme_vol) > 0.3:
        print(f"  ** WARNING: >30% b_day in extreme vol -- classification may be failing")
        print(f"  ** Wide IB absorbs extension -- price can move 200pts but still inside IB")
    else:
        print(f"  OK: b_day rate is reasonable in extreme vol")

# ================================================================
# SUMMARY
# ================================================================
print(f"\n\n{'='*120}")
print(f"  SUMMARY")
print(f"{'='*120}")
print(f"\n  Total sessions analyzed: {len(sdata)}")
print(f"  Overall classification accuracy: {sdata['classification_match'].mean()*100:.1f}%")
print(f"  AM->EOD morph rate: {sdata['morphed_am_pm'].mean()*100:.1f}%")
print(f"  13:00->EOD morph rate: {sdata['morphed_13_eod'].mean()*100:.1f}%")
print(f"  Mean peak extension: {sdata['peak_ext'].mean():.2f}x | Mean EOD extension: {sdata['eod_ext'].mean():.2f}x")
print(f"  Trend fakeouts (peak>1.0x, EOD<0.5x): {len(big_reversions)} ({len(big_reversions)/len(sdata)*100:.1f}%)")
print(f"\n  Avg 1-min bar range: {sdata['avg_bar_atr'].mean():.1f} pts")
print(f"  Avg IB range: {sdata['ib_range'].mean():.0f} pts")
print(f"  Avg session range: {sdata['session_range'].mean():.0f} pts")

print(f"\n{'='*120}")
print(f"  END OF DAY TYPE VALIDATION")
print(f"{'='*120}")
