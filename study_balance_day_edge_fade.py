"""
Balance Day IB Edge Fade Study
=================================

Comprehensive study of IB edge rejection patterns on balance days (B-day, P-day,
Neutral, Wide IB). Tests acceptance models, delta confirmation, volume profile
shape, and VWAP sweep-fail patterns.

Key question: Can we reliably fade IB edges on balance days with high win rate?

Prior findings to build on:
  - Delta at structural levels = 83% WR (bracket breakout study)
  - B-day IBL fade = 71.4% WR but tiny sample (N=7)
  - NT8 BalanceSignal = 12.5% WR on B-days (faded wrong levels — VA not IB)
  - NQ has structural long bias (shorts fail on trend days)

Parts:
  1. Day type foundation — frequency, IB width, early identification, C-period
  2. IB edge touch & rejection analysis
  3. Volume profile shape — POC/VWAP as skew filter
  4. Acceptance model comparison (2x1min, 2x5min, 30min, delta-only)
  5. Entry model backtests with dynamic targets
  5B. VWAP sweep-fail pattern
  6. Wide IB balance days
  7. Combined optimal filter + strategy card
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import time
from typing import Dict, List, Optional, Tuple

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.instruments import get_instrument
from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from indicators.value_area import calculate_value_area

# ── Config ──────────────────────────────────────────────────────
INSTRUMENT = get_instrument('MNQ')
TICK_VALUE = INSTRUMENT.tick_value / INSTRUMENT.tick_size  # $/pt
SLIPPAGE_PTS = 0.50
COMMISSION = 1.24
CONTRACTS = 5
IB_END_TIME = time(10, 30)
C_PERIOD_END = time(11, 0)
RTH_START = time(9, 30)
NOON = time(12, 0)
PM_START = time(13, 0)
EOD_EXIT = time(15, 30)
ENTRY_CUTOFF = time(14, 0)

# Touch proximity — how close price needs to be to count as "touching" IB edge
TOUCH_PROXIMITY = 5.0  # points

# ── Load Data ───────────────────────────────────────────────────
print("=" * 70)
print("BALANCE DAY IB EDGE FADE STUDY")
print("=" * 70)

print("\nLoading data...")
df_raw = load_csv('NQ')
df_rth = filter_rth(df_raw)
df_rth = compute_all_features(df_rth)

sessions = sorted(df_rth['session_date'].unique())
n_sessions = len(sessions)
months = n_sessions / 22
print(f"Sessions: {n_sessions}, Months: {months:.1f}")


def pnl_pts_to_dollars(pts, contracts=CONTRACTS):
    gross = pts * TICK_VALUE * contracts
    costs = (SLIPPAGE_PTS * TICK_VALUE * 2 + COMMISSION * 2) * contracts
    return gross - costs


def get_session_bars(session_date):
    return df_rth[df_rth['session_date'] == session_date].copy()


# ── Pre-compute per-session data ────────────────────────────────
print("Pre-computing per-session IB + profile data...")

session_data = []
for session_date in sessions:
    bars = get_session_bars(session_date)
    if len(bars) < 60:
        continue

    ib_bars = bars.head(60)
    post_ib_bars = bars.iloc[60:]

    ib_high = ib_bars['high'].max()
    ib_low = ib_bars['low'].min()
    ib_range = ib_high - ib_low
    ib_mid = (ib_high + ib_low) / 2

    if ib_range < 10:
        continue

    # IB POC using volume profile
    ib_va = calculate_value_area(
        ib_bars['high'].values, ib_bars['low'].values,
        ib_bars['volume'].values, tick_size=0.25, va_percent=0.70
    )
    ib_poc = ib_va.poc if ib_va else ib_mid

    # POC location classification
    ib_third = ib_range / 3
    if ib_poc >= ib_low + 2 * ib_third:
        poc_shape = 'P_SHAPE'  # POC in upper third
    elif ib_poc <= ib_low + ib_third:
        poc_shape = 'B_SHAPE'  # POC in lower third
    else:
        poc_shape = 'D_SHAPE'  # POC in center

    # EOD day type (from compute_all_features)
    eod_day_type = bars['day_type'].iloc[-1] if 'day_type' in bars.columns else 'UNKNOWN'
    eod_direction = bars['day_direction'].iloc[-1] if 'day_direction' in bars.columns else 'UNKNOWN'

    # Session extremes
    session_high = bars['high'].max()
    session_low = bars['low'].min()
    session_close = bars['close'].iloc[-1]
    session_range = session_high - session_low

    # ATR for reference
    atr = bars['atr14'].iloc[59] if 'atr14' in bars.columns else ib_range

    # VWAP at IB close
    vwap_at_ib = ib_bars['vwap'].iloc[-1] if 'vwap' in ib_bars.columns else ib_mid

    # C-period analysis (bars 60-89, 10:30-11:00)
    c_bars = bars.iloc[60:90] if len(bars) >= 90 else pd.DataFrame()
    c_close = c_bars['close'].iloc[-1] if len(c_bars) > 0 else None
    c_close_location = None
    if c_close is not None:
        if c_close > ib_high:
            c_close_location = 'ABOVE_IBH'
        elif c_close < ib_low:
            c_close_location = 'BELOW_IBL'
        else:
            c_close_location = 'INSIDE_IB'

    # Did price extend beyond IB after IB close?
    if len(post_ib_bars) > 0:
        post_ib_high = post_ib_bars['high'].max()
        post_ib_low = post_ib_bars['low'].min()
        max_ext_above = max(0, post_ib_high - ib_high)
        max_ext_below = max(0, ib_low - post_ib_low)
    else:
        max_ext_above = 0
        max_ext_below = 0

    # Extension multiple (for day type prediction at IB close)
    ib_close_price = ib_bars['close'].iloc[-1]
    ib_close_ext = 0
    if ib_close_price > ib_mid:
        ib_close_ext = (ib_close_price - ib_mid) / ib_range if ib_range > 0 else 0
    else:
        ib_close_ext = -(ib_mid - ib_close_price) / ib_range if ib_range > 0 else 0

    # EOD extension for day type classification
    eod_ext_above = max(0, session_high - ib_high) / ib_range if ib_range > 0 else 0
    eod_ext_below = max(0, ib_low - session_low) / ib_range if ib_range > 0 else 0
    eod_max_ext = max(eod_ext_above, eod_ext_below)

    # Is this a "balance day" by our definition?
    # Balance = max extension < 0.5x IB range (price stays mostly inside IB)
    is_balance = eod_max_ext < 0.5
    # Wide balance = IB range > 200 pts AND balance
    is_wide_balance = is_balance and ib_range > 200

    session_data.append({
        'session': session_date,
        'ib_high': ib_high,
        'ib_low': ib_low,
        'ib_range': ib_range,
        'ib_mid': ib_mid,
        'ib_poc': ib_poc,
        'poc_shape': poc_shape,
        'eod_day_type': eod_day_type,
        'eod_direction': eod_direction,
        'session_high': session_high,
        'session_low': session_low,
        'session_close': session_close,
        'session_range': session_range,
        'atr': atr if not pd.isna(atr) else ib_range,
        'vwap_at_ib': vwap_at_ib,
        'c_close': c_close,
        'c_close_location': c_close_location,
        'max_ext_above': max_ext_above,
        'max_ext_below': max_ext_below,
        'eod_ext_above': eod_ext_above,
        'eod_ext_below': eod_ext_below,
        'eod_max_ext': eod_max_ext,
        'ib_close_ext': ib_close_ext,
        'is_balance': is_balance,
        'is_wide_balance': is_wide_balance,
    })

sdf = pd.DataFrame(session_data)
print(f"Valid sessions with IB data: {len(sdf)}")


# ============================================================================
# PART 1: DAY TYPE FOUNDATION
# ============================================================================
print("\n\n" + "=" * 70)
print("PART 1: DAY TYPE FOUNDATION — FREQUENCY & EARLY IDENTIFICATION")
print("=" * 70)

# 1. Day type distribution
print("\n### 1.1: Day Type Distribution (259 sessions)\n")

day_type_counts = sdf['eod_day_type'].value_counts()
print(f"| Day Type | Count | % | Per Month | Avg IB Range | Median IB Range |")
print(f"|----------|-------|---|-----------|-------------|----------------|")
for dt in ['TREND', 'P_DAY', 'B_DAY', 'NEUTRAL']:
    sub = sdf[sdf['eod_day_type'] == dt]
    cnt = len(sub)
    pct = cnt / len(sdf) * 100
    per_mo = cnt / months
    avg_ib = sub['ib_range'].mean()
    med_ib = sub['ib_range'].median()
    print(f"| {dt} | {cnt} | {pct:.0f}% | {per_mo:.1f} | {avg_ib:.0f} pts | {med_ib:.0f} pts |")

# Balance day count
balance_count = sdf['is_balance'].sum()
wide_balance_count = sdf['is_wide_balance'].sum()
print(f"\n**Balance days (ext < 0.5x IB):** {balance_count} ({balance_count/len(sdf)*100:.0f}%)")
print(f"**Wide balance (IB > 200 + ext < 0.5x):** {wide_balance_count} ({wide_balance_count/len(sdf)*100:.0f}%)")

# 2. IB width distribution
print("\n### 1.2: IB Width Distribution\n")

width_buckets = [
    ('Narrow (<100)', sdf[sdf['ib_range'] < 100]),
    ('Normal (100-200)', sdf[(sdf['ib_range'] >= 100) & (sdf['ib_range'] < 200)]),
    ('Wide (200-300)', sdf[(sdf['ib_range'] >= 200) & (sdf['ib_range'] < 300)]),
    ('Very Wide (300-400)', sdf[(sdf['ib_range'] >= 300) & (sdf['ib_range'] < 400)]),
    ('Extreme (>400)', sdf[sdf['ib_range'] >= 400]),
]

print(f"| IB Width | Count | % | Avg EOD Ext | % Balance |")
print(f"|----------|-------|---|-------------|-----------|")
for label, sub in width_buckets:
    if len(sub) == 0:
        print(f"| {label} | 0 | 0% | — | — |")
        continue
    cnt = len(sub)
    pct = cnt / len(sdf) * 100
    avg_ext = sub['eod_max_ext'].mean()
    bal_pct = sub['is_balance'].mean() * 100
    print(f"| {label} | {cnt} | {pct:.0f}% | {avg_ext:.1f}x | {bal_pct:.0f}% |")

# 3. Early identification accuracy
print("\n### 1.3: Can We Predict Day Type at IB Close (10:30)?\n")

# Predict based on extension at IB close
def predict_day_type_at_ib(ext):
    abs_ext = abs(ext)
    if abs_ext > 0.5:
        return 'TREND'
    elif abs_ext > 0.25:
        return 'P_DAY'
    elif abs_ext < 0.1:
        return 'B_DAY'
    else:
        return 'NEUTRAL'

sdf['predicted_at_ib'] = sdf['ib_close_ext'].apply(predict_day_type_at_ib)

# Accuracy matrix
print(f"| Predicted at IB | Actual TREND | Actual P_DAY | Actual B_DAY | Actual NEUTRAL | Accuracy |")
print(f"|----------------|-------------|-------------|-------------|---------------|----------|")
for pred in ['TREND', 'P_DAY', 'B_DAY', 'NEUTRAL']:
    sub = sdf[sdf['predicted_at_ib'] == pred]
    if len(sub) == 0:
        continue
    correct = (sub['eod_day_type'] == pred).sum()
    acc = correct / len(sub) * 100
    trend_n = (sub['eod_day_type'] == 'TREND').sum()
    p_n = (sub['eod_day_type'] == 'P_DAY').sum()
    b_n = (sub['eod_day_type'] == 'B_DAY').sum()
    n_n = (sub['eod_day_type'] == 'NEUTRAL').sum()
    print(f"| {pred} | {trend_n} | {p_n} | {b_n} | {n_n} | {acc:.0f}% |")

# 4. C-period rule validation
print("\n### 1.4: C-Period Rule Validation (10:30-11:00)\n")
print("Dalton claims: C-period close above IBH = 70-75% continuation up")
print("               C-period close below IBL = 70-75% continuation down")
print("               C-period close inside IB = 70-75% reversal to opposite\n")

c_valid = sdf[sdf['c_close_location'].notna()]
print(f"| C-Period Close | N | Cont. Same Dir | Reversal | Cont. Rate |")
print(f"|----------------|---|----------------|----------|-----------|")

for loc in ['ABOVE_IBH', 'BELOW_IBL', 'INSIDE_IB']:
    sub = c_valid[c_valid['c_close_location'] == loc]
    if len(sub) == 0:
        continue

    if loc == 'ABOVE_IBH':
        # Continuation = session closes above IBH
        cont = (sub['session_close'] > sub['ib_high']).sum()
    elif loc == 'BELOW_IBL':
        cont = (sub['session_close'] < sub['ib_low']).sum()
    else:  # INSIDE_IB
        # "Reversal to opposite extreme" — stays inside or goes opposite from IB close direction
        # If IB closed in upper half, reversal = session closes in lower half (below IB mid)
        cont_mask = (
            ((sub['ib_close_ext'] > 0) & (sub['session_close'] < sub['ib_mid'])) |
            ((sub['ib_close_ext'] <= 0) & (sub['session_close'] > sub['ib_mid']))
        )
        cont = cont_mask.sum()

    rev = len(sub) - cont
    rate = cont / len(sub) * 100 if len(sub) > 0 else 0
    print(f"| {loc} | {len(sub)} | {cont} | {rev} | {rate:.0f}% |")


# ============================================================================
# PART 2: IB EDGE TOUCH & REJECTION ANALYSIS
# ============================================================================
print("\n\n" + "=" * 70)
print("PART 2: IB EDGE TOUCH & REJECTION ANALYSIS")
print("=" * 70)

# For each session, find all post-IB touches of IBH/IBL
all_touches = []

for _, row in sdf.iterrows():
    session_date = row['session']
    ib_high = row['ib_high']
    ib_low = row['ib_low']
    ib_range = row['ib_range']
    ib_mid = row['ib_mid']
    eod_day_type = row['eod_day_type']

    bars = get_session_bars(session_date)
    post_ib = bars.iloc[60:]  # After IB

    if len(post_ib) < 30:
        continue

    # Track IBH touches
    ibh_touch_count = 0
    ibl_touch_count = 0
    last_ibh_touch_idx = -30  # cooldown
    last_ibl_touch_idx = -30

    for i, (idx, bar) in enumerate(post_ib.iterrows()):
        bar_idx = 60 + i
        bar_time = bar['time']

        if bar_time >= ENTRY_CUTOFF:
            break

        # IBH touch
        if bar['high'] >= ib_high - TOUCH_PROXIMITY and (bar_idx - last_ibh_touch_idx) >= 15:
            ibh_touch_count += 1
            last_ibh_touch_idx = bar_idx

            # Track what happens after the touch
            penetration = max(0, bar['high'] - ib_high)
            delta_at_touch = bar.get('delta', 0)
            vol_spike = bar.get('volume_spike', 1.0) if 'volume_spike' in bar.index else 1.0
            vwap = bar.get('vwap', ib_mid)

            # Check for acceptance back inside (2x1min, 2x5min, 30min)
            future_bars = post_ib.iloc[i+1:i+61] if i+1 < len(post_ib) else pd.DataFrame()
            accept_2x1min = False
            accept_2x5min = False
            accept_30min = False
            accept_2x1min_bar = None
            accept_2x5min_bar = None
            accept_30min_bar = None

            if len(future_bars) >= 2:
                # 2x1min: 2 consecutive closes below IBH
                consec = 0
                for fi, (fidx, fbar) in enumerate(future_bars.iterrows()):
                    if fbar['close'] < ib_high:
                        consec += 1
                        if consec >= 2 and not accept_2x1min:
                            accept_2x1min = True
                            accept_2x1min_bar = bar_idx + fi + 1
                    else:
                        consec = 0

                # 2x5min: 2 consecutive 5-bar period closes below IBH
                period_consec = 0
                for period_start in range(0, len(future_bars), 5):
                    period_end = min(period_start + 5, len(future_bars))
                    period_close = future_bars.iloc[period_end - 1]['close']
                    if period_close < ib_high:
                        period_consec += 1
                        if period_consec >= 2 and not accept_2x5min:
                            accept_2x5min = True
                            accept_2x5min_bar = bar_idx + period_end
                    else:
                        period_consec = 0

                # 30min: 1 x 30-bar period close below IBH
                if len(future_bars) >= 30:
                    if future_bars.iloc[29]['close'] < ib_high:
                        accept_30min = True
                        accept_30min_bar = bar_idx + 30

            # Delta alignment (for SHORT at IBH: delta < 0 = aligned)
            delta_aligned = delta_at_touch < 0

            # Did the fade work? (reached IB mid within 60 bars)
            reached_mid = False
            reached_vwap = False
            max_adverse = 0
            exit_pnl = None

            for fi, (fidx, fbar) in enumerate(future_bars.iterrows()):
                adverse = max(0, fbar['high'] - ib_high)
                if adverse > max_adverse:
                    max_adverse = adverse
                if fbar['low'] <= ib_mid and not reached_mid:
                    reached_mid = True
                if vwap and not pd.isna(vwap) and fbar['low'] <= vwap and not reached_vwap:
                    reached_vwap = True

            # BPR / FVG at touch
            has_bpr = bar.get('in_bpr', False) if 'in_bpr' in bar.index else False
            has_fvg = bar.get('fvg_bear', False) if 'fvg_bear' in bar.index else False

            all_touches.append({
                'session': session_date,
                'edge': 'IBH',
                'fade_dir': 'SHORT',
                'touch_num': ibh_touch_count,
                'touch_bar': bar_idx,
                'touch_time': bar_time,
                'penetration': penetration,
                'delta_at_touch': delta_at_touch,
                'delta_aligned': delta_aligned,
                'vol_spike': vol_spike,
                'vwap': vwap,
                'vwap_vs_mid': 'above' if (vwap and not pd.isna(vwap) and vwap > ib_mid) else 'below',
                'accept_2x1min': accept_2x1min,
                'accept_2x5min': accept_2x5min,
                'accept_30min': accept_30min,
                'accept_2x1min_bar': accept_2x1min_bar,
                'accept_2x5min_bar': accept_2x5min_bar,
                'accept_30min_bar': accept_30min_bar,
                'reached_mid': reached_mid,
                'reached_vwap': reached_vwap,
                'max_adverse': max_adverse,
                'has_bpr': has_bpr,
                'has_fvg': has_fvg,
                'ib_range': ib_range,
                'ib_mid': ib_mid,
                'ib_high': ib_high,
                'ib_low': ib_low,
                'poc_shape': row['poc_shape'],
                'eod_day_type': eod_day_type,
                'is_balance': row['is_balance'],
                'is_wide_balance': row['is_wide_balance'],
            })

        # IBL touch
        if bar['low'] <= ib_low + TOUCH_PROXIMITY and (bar_idx - last_ibl_touch_idx) >= 15:
            ibl_touch_count += 1
            last_ibl_touch_idx = bar_idx

            penetration = max(0, ib_low - bar['low'])
            delta_at_touch = bar.get('delta', 0)
            vol_spike = bar.get('volume_spike', 1.0) if 'volume_spike' in bar.index else 1.0
            vwap = bar.get('vwap', ib_mid)

            future_bars = post_ib.iloc[i+1:i+61] if i+1 < len(post_ib) else pd.DataFrame()
            accept_2x1min = False
            accept_2x5min = False
            accept_30min = False
            accept_2x1min_bar = None
            accept_2x5min_bar = None
            accept_30min_bar = None

            if len(future_bars) >= 2:
                consec = 0
                for fi, (fidx, fbar) in enumerate(future_bars.iterrows()):
                    if fbar['close'] > ib_low:
                        consec += 1
                        if consec >= 2 and not accept_2x1min:
                            accept_2x1min = True
                            accept_2x1min_bar = bar_idx + fi + 1
                    else:
                        consec = 0

                period_consec = 0
                for period_start in range(0, len(future_bars), 5):
                    period_end = min(period_start + 5, len(future_bars))
                    period_close = future_bars.iloc[period_end - 1]['close']
                    if period_close > ib_low:
                        period_consec += 1
                        if period_consec >= 2 and not accept_2x5min:
                            accept_2x5min = True
                            accept_2x5min_bar = bar_idx + period_end
                    else:
                        period_consec = 0

                if len(future_bars) >= 30:
                    if future_bars.iloc[29]['close'] > ib_low:
                        accept_30min = True
                        accept_30min_bar = bar_idx + 30

            delta_aligned = delta_at_touch > 0  # LONG at IBL: positive delta = aligned

            reached_mid = False
            reached_vwap = False
            max_adverse = 0

            for fi, (fidx, fbar) in enumerate(future_bars.iterrows()):
                adverse = max(0, ib_low - fbar['low'])
                if adverse > max_adverse:
                    max_adverse = adverse
                if fbar['high'] >= ib_mid and not reached_mid:
                    reached_mid = True
                if vwap and not pd.isna(vwap) and fbar['high'] >= vwap and not reached_vwap:
                    reached_vwap = True

            has_bpr = bar.get('in_bpr', False) if 'in_bpr' in bar.index else False
            has_fvg = bar.get('fvg_bull', False) if 'fvg_bull' in bar.index else False

            all_touches.append({
                'session': session_date,
                'edge': 'IBL',
                'fade_dir': 'LONG',
                'touch_num': ibl_touch_count,
                'touch_bar': bar_idx,
                'touch_time': bar_time,
                'penetration': penetration,
                'delta_at_touch': delta_at_touch,
                'delta_aligned': delta_aligned,
                'vol_spike': vol_spike,
                'vwap': vwap,
                'vwap_vs_mid': 'above' if (vwap and not pd.isna(vwap) and vwap > ib_mid) else 'below',
                'accept_2x1min': accept_2x1min,
                'accept_2x5min': accept_2x5min,
                'accept_30min': accept_30min,
                'accept_2x1min_bar': accept_2x1min_bar,
                'accept_2x5min_bar': accept_2x5min_bar,
                'accept_30min_bar': accept_30min_bar,
                'reached_mid': reached_mid,
                'reached_vwap': reached_vwap,
                'max_adverse': max_adverse,
                'has_bpr': has_bpr,
                'has_fvg': has_fvg,
                'ib_range': ib_range,
                'ib_mid': ib_mid,
                'ib_high': ib_high,
                'ib_low': ib_low,
                'poc_shape': row['poc_shape'],
                'eod_day_type': eod_day_type,
                'is_balance': row['is_balance'],
                'is_wide_balance': row['is_wide_balance'],
            })

tdf = pd.DataFrame(all_touches)
print(f"\nTotal IB edge touches found: {len(tdf)}")
print(f"  IBH touches: {(tdf['edge'] == 'IBH').sum()}")
print(f"  IBL touches: {(tdf['edge'] == 'IBL').sum()}")

# 5. Touch frequency on balance days
print("\n### 2.1: IB Edge Touch Frequency\n")

balance_touches = tdf[tdf['is_balance'] == True]
all_day_touches = tdf

print(f"| Day Filter | IBH Touches | IBL Touches | Total | Sessions | Touches/Session |")
print(f"|-----------|-------------|-------------|-------|----------|----------------|")
for label, sub in [
    ('All sessions', all_day_touches),
    ('Balance days only', balance_touches),
    ('Wide balance', tdf[tdf['is_wide_balance'] == True]),
]:
    ibh = (sub['edge'] == 'IBH').sum()
    ibl = (sub['edge'] == 'IBL').sum()
    unique_sessions = sub['session'].nunique()
    per_session = len(sub) / unique_sessions if unique_sessions > 0 else 0
    print(f"| {label} | {ibh} | {ibl} | {len(sub)} | {unique_sessions} | {per_session:.1f} |")

# 6. Rejection success rate
print("\n### 2.2: IB Edge Fade Success Rate (Reached IB Mid)\n")

print(f"| Filter | N | Reached Mid | Reached VWAP | Fade Success % | Avg Max Adverse |")
print(f"|--------|---|------------|-------------|----------------|----------------|")
for label, sub in [
    ('All touches', tdf),
    ('Balance days', balance_touches),
    ('IBL LONG (all)', tdf[tdf['fade_dir'] == 'LONG']),
    ('IBH SHORT (all)', tdf[tdf['fade_dir'] == 'SHORT']),
    ('IBL LONG (balance)', balance_touches[balance_touches['fade_dir'] == 'LONG']),
    ('IBH SHORT (balance)', balance_touches[balance_touches['fade_dir'] == 'SHORT']),
]:
    if len(sub) == 0:
        continue
    mid_cnt = sub['reached_mid'].sum()
    vwap_cnt = sub['reached_vwap'].sum()
    mid_pct = mid_cnt / len(sub) * 100
    avg_adv = sub['max_adverse'].mean()
    print(f"| {label} | {len(sub)} | {mid_cnt} ({mid_pct:.0f}%) | {vwap_cnt} ({vwap_cnt/len(sub)*100:.0f}%) | {mid_pct:.0f}% | {avg_adv:.0f} pts |")

# By touch number
print("\n### 2.3: First Touch vs Second Touch vs Third\n")
print(f"| Touch # | N | Fade Success % | Avg Max Adverse |")
print(f"|---------|---|----------------|----------------|")
for touch_n in [1, 2, 3]:
    sub = balance_touches[balance_touches['touch_num'] == touch_n]
    if len(sub) == 0:
        continue
    mid_pct = sub['reached_mid'].mean() * 100
    avg_adv = sub['max_adverse'].mean()
    print(f"| Touch #{touch_n} | {len(sub)} | {mid_pct:.0f}% | {avg_adv:.0f} pts |")

# 7. What differentiates successful vs failed rejections?
print("\n### 2.4: What Differentiates Successful vs Failed Rejections?\n")

for label, sub in [('Balance day touches', balance_touches)]:
    if len(sub) < 10:
        continue
    wins = sub[sub['reached_mid'] == True]
    losses = sub[sub['reached_mid'] == False]
    if len(wins) == 0 or len(losses) == 0:
        continue

    print(f"**{label}** (N={len(sub)}, {len(wins)} wins, {len(losses)} losses)\n")
    print(f"| Factor | Winners (N={len(wins)}) | Losers (N={len(losses)}) | Edge |")
    print(f"|--------|----------------------|----------------------|------|")

    # Delta alignment
    w_delta = wins['delta_aligned'].mean() * 100
    l_delta = losses['delta_aligned'].mean() * 100
    print(f"| Delta aligned | {w_delta:.0f}% | {l_delta:.0f}% | {w_delta - l_delta:+.0f}pp |")

    # Penetration depth
    w_pen = wins['penetration'].mean()
    l_pen = losses['penetration'].mean()
    print(f"| Avg penetration | {w_pen:.0f} pts | {l_pen:.0f} pts | {w_pen - l_pen:+.0f} pts |")

    # Volume spike
    w_vol = wins['vol_spike'].mean()
    l_vol = losses['vol_spike'].mean()
    print(f"| Avg vol spike | {w_vol:.1f}x | {l_vol:.1f}x | {w_vol - l_vol:+.1f}x |")

    # BPR/FVG
    w_bpr = wins['has_bpr'].mean() * 100
    l_bpr = losses['has_bpr'].mean() * 100
    print(f"| Has BPR | {w_bpr:.0f}% | {l_bpr:.0f}% | {w_bpr - l_bpr:+.0f}pp |")

    w_fvg = wins['has_fvg'].mean() * 100
    l_fvg = losses['has_fvg'].mean() * 100
    print(f"| Has FVG | {w_fvg:.0f}% | {l_fvg:.0f}% | {w_fvg - l_fvg:+.0f}pp |")

    # 2x5min acceptance
    w_acc = wins['accept_2x5min'].mean() * 100
    l_acc = losses['accept_2x5min'].mean() * 100
    print(f"| 2x5min accepted | {w_acc:.0f}% | {l_acc:.0f}% | {w_acc - l_acc:+.0f}pp |")


# ============================================================================
# PART 3: VOLUME PROFILE SHAPE — POC/VWAP AS SKEW FILTER
# ============================================================================
print("\n\n" + "=" * 70)
print("PART 3: VOLUME PROFILE SHAPE — POC/VWAP AS SKEW FILTER")
print("=" * 70)

# 8. IB POC shape vs EOD day type
print("\n### 3.1: Does IB POC Shape Predict Day Type?\n")

print(f"| IB POC Shape | N | TREND | P_DAY | B_DAY | NEUTRAL | % Balance |")
print(f"|-------------|---|-------|-------|-------|---------|-----------|")
for shape in ['P_SHAPE', 'D_SHAPE', 'B_SHAPE']:
    sub = sdf[sdf['poc_shape'] == shape]
    if len(sub) == 0:
        continue
    trend = (sub['eod_day_type'] == 'TREND').sum()
    p = (sub['eod_day_type'] == 'P_DAY').sum()
    b = (sub['eod_day_type'] == 'B_DAY').sum()
    n = (sub['eod_day_type'] == 'NEUTRAL').sum()
    bal = sub['is_balance'].mean() * 100
    print(f"| {shape} | {len(sub)} | {trend} ({trend/len(sub)*100:.0f}%) | {p} ({p/len(sub)*100:.0f}%) | {b} ({b/len(sub)*100:.0f}%) | {n} ({n/len(sub)*100:.0f}%) | {bal:.0f}% |")

# 9. POC alignment with fade direction
print("\n### 3.2: POC Shape Alignment with Fade Direction\n")
print("b-shape + IBL LONG = aligned (fading toward where POC is)")
print("P-shape + IBH SHORT = aligned\n")

if len(balance_touches) > 0:
    bt = balance_touches.copy()
    bt['poc_aligned'] = False
    bt.loc[(bt['poc_shape'] == 'B_SHAPE') & (bt['fade_dir'] == 'LONG'), 'poc_aligned'] = True
    bt.loc[(bt['poc_shape'] == 'P_SHAPE') & (bt['fade_dir'] == 'SHORT'), 'poc_aligned'] = True
    bt.loc[(bt['poc_shape'] == 'D_SHAPE'), 'poc_aligned'] = True  # Neutral shape = always ok

    aligned = bt[bt['poc_aligned'] == True]
    misaligned = bt[bt['poc_aligned'] == False]

    print(f"| POC Alignment | N | Fade Success | Avg Max Adverse |")
    print(f"|--------------|---|-------------|----------------|")
    for label, sub in [('Aligned', aligned), ('Misaligned', misaligned)]:
        if len(sub) == 0:
            continue
        succ = sub['reached_mid'].mean() * 100
        adv = sub['max_adverse'].mean()
        print(f"| {label} | {len(sub)} | {succ:.0f}% | {adv:.0f} pts |")

# 10. VWAP as demarcation
print("\n### 3.3: VWAP Position at Touch Time\n")

if len(balance_touches) > 0:
    print(f"| VWAP vs IB Mid | Fade Dir | N | Fade Success | Hypothesis |")
    print(f"|---------------|---------|---|-------------|-----------|")

    for vwap_pos in ['above', 'below']:
        for fade_dir in ['LONG', 'SHORT']:
            sub = balance_touches[
                (balance_touches['vwap_vs_mid'] == vwap_pos) &
                (balance_touches['fade_dir'] == fade_dir)
            ]
            if len(sub) == 0:
                continue
            succ = sub['reached_mid'].mean() * 100
            hypothesis = ''
            if vwap_pos == 'above' and fade_dir == 'LONG':
                hypothesis = 'ALIGNED (bullish bias + IBL long)'
            elif vwap_pos == 'below' and fade_dir == 'SHORT':
                hypothesis = 'ALIGNED (bearish bias + IBH short)'
            else:
                hypothesis = 'MISALIGNED'
            print(f"| VWAP {vwap_pos} mid | {fade_dir} | {len(sub)} | {succ:.0f}% | {hypothesis} |")


# ============================================================================
# PART 4: ACCEPTANCE MODEL COMPARISON
# ============================================================================
print("\n\n" + "=" * 70)
print("PART 4: ACCEPTANCE MODEL COMPARISON")
print("=" * 70)

print("\n### 4.1: Which Acceptance Model Works Best for IB Edge Fades?\n")

# Only use balance day touches
bt = balance_touches.copy() if len(balance_touches) > 0 else pd.DataFrame()

if len(bt) > 0:
    print(f"| Acceptance Model | N (accepted) | Fade Success | Not Accepted Fade % | Filter Value |")
    print(f"|-----------------|-------------|-------------|--------------------|--------------| ")

    for model, col in [
        ('2x1min consecutive', 'accept_2x1min'),
        ('2x5min consecutive', 'accept_2x5min'),
        ('1x30min close', 'accept_30min'),
        ('Delta aligned only', 'delta_aligned'),
    ]:
        accepted = bt[bt[col] == True]
        rejected = bt[bt[col] == False]
        if len(accepted) == 0:
            print(f"| {model} | 0 | — | — | — |")
            continue
        acc_succ = accepted['reached_mid'].mean() * 100
        rej_succ = rejected['reached_mid'].mean() * 100 if len(rejected) > 0 else 0
        filter_val = acc_succ - rej_succ
        print(f"| {model} | {len(accepted)} | {acc_succ:.0f}% | {rej_succ:.0f}% | +{filter_val:.0f}pp |")

    # 12. Combined: acceptance + delta
    print("\n### 4.2: Acceptance + Delta Combined\n")

    print(f"| Combo | N | Fade Success | Avg Max Adverse |")
    print(f"|-------|---|-------------|----------------|")
    for model_label, acc_col in [('2x5min', 'accept_2x5min'), ('2x1min', 'accept_2x1min'), ('30min', 'accept_30min')]:
        combo = bt[(bt[acc_col] == True) & (bt['delta_aligned'] == True)]
        if len(combo) == 0:
            print(f"| {model_label} + delta | 0 | — | — |")
            continue
        succ = combo['reached_mid'].mean() * 100
        adv = combo['max_adverse'].mean()
        print(f"| {model_label} + delta | {len(combo)} | {succ:.0f}% | {adv:.0f} pts |")


# ============================================================================
# PART 5: ENTRY MODEL BACKTESTS
# ============================================================================
print("\n\n" + "=" * 70)
print("PART 5: ENTRY MODEL BACKTESTS")
print("=" * 70)


def backtest_fade(touches_df, entry_filter, stop_buffer_pct, target_mode, label):
    """Backtest an IB edge fade entry model."""
    trades = []

    for _, touch in touches_df.iterrows():
        if not entry_filter(touch):
            continue

        session_date = touch['session']
        fade_dir = touch['fade_dir']
        ib_high = touch['ib_high']
        ib_low = touch['ib_low']
        ib_range = touch['ib_range']
        ib_mid = touch['ib_mid']
        vwap = touch['vwap']

        # Entry bar
        entry_bar_idx = touch['touch_bar']
        if fade_dir == 'LONG':
            # Enter at IBL touch, expecting bounce
            accept_bar = touch.get('accept_2x5min_bar')
            if accept_bar and pd.notna(accept_bar):
                entry_bar_idx = int(accept_bar)

        elif fade_dir == 'SHORT':
            accept_bar = touch.get('accept_2x5min_bar')
            if accept_bar and pd.notna(accept_bar):
                entry_bar_idx = int(accept_bar)

        bars = get_session_bars(session_date)
        if entry_bar_idx >= len(bars):
            continue

        entry_price = bars.iloc[entry_bar_idx]['close']

        # Stop
        if fade_dir == 'LONG':
            stop = ib_low - ib_range * stop_buffer_pct
        else:
            stop = ib_high + ib_range * stop_buffer_pct

        risk = abs(entry_price - stop)
        if risk <= 0:
            continue

        # Target
        if target_mode == 'ib_mid':
            target = ib_mid
        elif target_mode == 'vwap':
            target = vwap if vwap and not pd.isna(vwap) else ib_mid
        elif target_mode == 'dynamic':
            # Dynamic based on POC shape
            poc_shape = touch['poc_shape']
            if poc_shape == 'D_SHAPE':
                target = ib_mid
            elif fade_dir == 'LONG' and poc_shape == 'B_SHAPE':
                target = vwap if (vwap and not pd.isna(vwap) and vwap > ib_low) else ib_mid
            elif fade_dir == 'SHORT' and poc_shape == 'P_SHAPE':
                target = vwap if (vwap and not pd.isna(vwap) and vwap < ib_high) else ib_mid
            else:
                target = ib_mid
        elif target_mode == '2R':
            if fade_dir == 'LONG':
                target = entry_price + 2 * risk
            else:
                target = entry_price - 2 * risk
        else:
            target = ib_mid

        # Check target is in the right direction
        if fade_dir == 'LONG' and target <= entry_price:
            target = entry_price + risk  # minimum 1R
        if fade_dir == 'SHORT' and target >= entry_price:
            target = entry_price - risk

        # Walk remaining bars
        remaining = bars.iloc[entry_bar_idx:]
        outcome = 'EOD'
        exit_price = remaining['close'].iloc[-1]

        for _, bar in remaining.iterrows():
            if bar['time'] >= EOD_EXIT:
                exit_price = bar['close']
                outcome = 'EOD'
                break

            if fade_dir == 'LONG':
                if bar['low'] <= stop:
                    exit_price = stop
                    outcome = 'STOP'
                    break
                if bar['high'] >= target:
                    exit_price = target
                    outcome = 'TARGET'
                    break
            else:
                if bar['high'] >= stop:
                    exit_price = stop
                    outcome = 'STOP'
                    break
                if bar['low'] <= target:
                    exit_price = target
                    outcome = 'TARGET'
                    break

        if fade_dir == 'LONG':
            pnl_pts = exit_price - entry_price
        else:
            pnl_pts = entry_price - exit_price

        trades.append({
            'session': session_date,
            'direction': fade_dir,
            'entry': entry_price,
            'stop': stop,
            'target': target,
            'exit': exit_price,
            'pnl_pts': pnl_pts,
            'pnl_dollars': pnl_pts_to_dollars(pnl_pts),
            'outcome': outcome,
            'risk_pts': risk,
            'r_multiple': pnl_pts / risk if risk > 0 else 0,
        })

    if not trades:
        return None

    tdf_result = pd.DataFrame(trades)
    n = len(tdf_result)
    wins = (tdf_result['pnl_pts'] > 0).sum()
    wr = wins / n * 100
    total_pnl = tdf_result['pnl_dollars'].sum()
    monthly_pnl = total_pnl / months
    gross_win = tdf_result[tdf_result['pnl_pts'] > 0]['pnl_dollars'].sum()
    gross_loss = abs(tdf_result[tdf_result['pnl_pts'] <= 0]['pnl_dollars'].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

    return {
        'label': label,
        'n': n,
        'trades_mo': n / months,
        'wr': wr,
        'pf': pf,
        'monthly': monthly_pnl,
        'avg_risk': tdf_result['risk_pts'].mean(),
        'trades_df': tdf_result,
    }


# Run backtests
print("\n### 5.1: Entry Model Comparison (Balance Days)\n")

configs = [
    # (filter_fn, stop_buffer, target_mode, label)
    (lambda t: t['accept_2x5min'] == True,
     0.10, 'ib_mid', 'A: 2x5min accept → IB mid'),
    (lambda t: t['accept_2x5min'] == True and t['delta_aligned'] == True,
     0.10, 'ib_mid', 'B: 2x5min + delta → IB mid'),
    (lambda t: t['accept_2x5min'] == True and t['delta_aligned'] == True and (t['has_bpr'] == True or t['has_fvg'] == True),
     0.10, 'ib_mid', 'C: 2x5min + delta + BPR/FVG → IB mid'),
    (lambda t: t['accept_2x5min'] == True and t.get('poc_aligned', False) == True,
     0.10, 'ib_mid', 'D: 2x5min + POC aligned → IB mid'),
    (lambda t: t['accept_2x5min'] == True,
     0.10, 'vwap', 'E: 2x5min accept → VWAP'),
    (lambda t: t['accept_2x5min'] == True and t['delta_aligned'] == True,
     0.10, 'dynamic', 'F: 2x5min + delta → dynamic target'),
    (lambda t: t['delta_aligned'] == True,
     0.10, 'ib_mid', 'G: Delta-only (no accept) → IB mid'),
    (lambda t: t['accept_30min'] == True,
     0.10, 'ib_mid', 'H: 30min accept → IB mid'),
    (lambda t: t['accept_2x1min'] == True,
     0.10, 'ib_mid', 'I: 2x1min accept → IB mid'),
    (lambda t: t['accept_2x5min'] == True and t['delta_aligned'] == True,
     0.10, '2R', 'J: 2x5min + delta → 2R'),
]

# Add POC alignment to balance touches
if len(balance_touches) > 0:
    bt_bt = balance_touches.copy()
    bt_bt['poc_aligned'] = False
    bt_bt.loc[(bt_bt['poc_shape'] == 'B_SHAPE') & (bt_bt['fade_dir'] == 'LONG'), 'poc_aligned'] = True
    bt_bt.loc[(bt_bt['poc_shape'] == 'P_SHAPE') & (bt_bt['fade_dir'] == 'SHORT'), 'poc_aligned'] = True
    bt_bt.loc[(bt_bt['poc_shape'] == 'D_SHAPE'), 'poc_aligned'] = True
else:
    bt_bt = pd.DataFrame()

print(f"| Config | N | T/Mo | WR | PF | $/Mo | Avg Risk |")
print(f"|--------|---|------|-----|-----|------|---------|")

best_monthly = -float('inf')
best_config = None

for filter_fn, stop_buf, target, label in configs:
    result = backtest_fade(bt_bt, filter_fn, stop_buf, target, label) if len(bt_bt) > 0 else None
    if result is None:
        print(f"| {label} | 0 | — | — | — | — | — |")
        continue
    print(f"| {label} | {result['n']} | {result['trades_mo']:.1f} | {result['wr']:.0f}% | {result['pf']:.2f} | ${result['monthly']:.0f} | {result['avg_risk']:.0f}p |")
    if result['monthly'] > best_monthly:
        best_monthly = result['monthly']
        best_config = result

# LONG vs SHORT breakdown for best config
if best_config and best_config['n'] >= 5:
    print(f"\n**Best: {best_config['label']}** — LONG vs SHORT:")
    for d in ['LONG', 'SHORT']:
        ddf = best_config['trades_df'][best_config['trades_df']['direction'] == d]
        if len(ddf) == 0:
            continue
        dw = (ddf['pnl_pts'] > 0).sum()
        dwr = dw / len(ddf) * 100
        dtotal = ddf['pnl_dollars'].sum()
        print(f"  {d}: {len(ddf)} trades, WR {dwr:.0f}%, ${dtotal/months:.0f}/mo")


# ============================================================================
# PART 5B: VWAP SWEEP FAIL PATTERN
# ============================================================================
print("\n\n" + "=" * 70)
print("PART 5B: VWAP SWEEP-FAIL PATTERN")
print("=" * 70)

print("\nThe user's concept: price sweeps above/below VWAP and fails back")
print("B-day: rally to VWAP, sweep above, fail → SHORT back to IBL")
print("P-day: drop to VWAP, sweep below, fail → LONG back to IBH\n")

vwap_sweeps = []

for _, row in sdf.iterrows():
    if not row['is_balance']:
        continue

    session_date = row['session']
    ib_high = row['ib_high']
    ib_low = row['ib_low']
    ib_mid = row['ib_mid']
    ib_range = row['ib_range']

    bars = get_session_bars(session_date)
    post_ib = bars.iloc[60:]
    if len(post_ib) < 30:
        continue

    for i, (idx, bar) in enumerate(post_ib.iterrows()):
        bar_time = bar['time']
        if bar_time >= ENTRY_CUTOFF:
            break

        vwap = bar.get('vwap')
        if vwap is None or pd.isna(vwap):
            continue

        # Sweep ABOVE VWAP and fail → SHORT
        if bar['high'] > vwap + 3 and bar['close'] < vwap:
            # Check 2x5min acceptance below VWAP
            future = post_ib.iloc[i+1:i+11]
            if len(future) >= 10:
                p1_close = future.iloc[4]['close']
                p2_close = future.iloc[9]['close']
                if p1_close < vwap and p2_close < vwap:
                    entry_price = future.iloc[9]['close']
                    stop = bar['high'] + ib_range * 0.10
                    target = ib_low + ib_range * 0.25  # lower quarter

                    risk = stop - entry_price
                    if risk <= 0:
                        continue

                    # Walk remaining bars for outcome
                    remain = post_ib.iloc[i+10:]
                    outcome = 'EOD'
                    exit_price = bars['close'].iloc[-1]

                    for _, rbar in remain.iterrows():
                        if rbar['time'] >= EOD_EXIT:
                            exit_price = rbar['close']
                            break
                        if rbar['high'] >= stop:
                            exit_price = stop
                            outcome = 'STOP'
                            break
                        if rbar['low'] <= target:
                            exit_price = target
                            outcome = 'TARGET'
                            break

                    pnl = entry_price - exit_price
                    vwap_sweeps.append({
                        'session': session_date,
                        'direction': 'SHORT',
                        'pattern': 'VWAP_SWEEP_ABOVE_FAIL',
                        'entry': entry_price,
                        'stop': stop,
                        'target': target,
                        'exit': exit_price,
                        'pnl_pts': pnl,
                        'pnl_dollars': pnl_pts_to_dollars(pnl),
                        'outcome': outcome,
                        'risk_pts': risk,
                        'day_type': row['eod_day_type'],
                    })

        # Sweep BELOW VWAP and fail → LONG
        if bar['low'] < vwap - 3 and bar['close'] > vwap:
            future = post_ib.iloc[i+1:i+11]
            if len(future) >= 10:
                p1_close = future.iloc[4]['close']
                p2_close = future.iloc[9]['close']
                if p1_close > vwap and p2_close > vwap:
                    entry_price = future.iloc[9]['close']
                    stop = bar['low'] - ib_range * 0.10
                    target = ib_high - ib_range * 0.25

                    risk = entry_price - stop
                    if risk <= 0:
                        continue

                    remain = post_ib.iloc[i+10:]
                    outcome = 'EOD'
                    exit_price = bars['close'].iloc[-1]

                    for _, rbar in remain.iterrows():
                        if rbar['time'] >= EOD_EXIT:
                            exit_price = rbar['close']
                            break
                        if rbar['low'] <= stop:
                            exit_price = stop
                            outcome = 'STOP'
                            break
                        if rbar['high'] >= target:
                            exit_price = target
                            outcome = 'TARGET'
                            break

                    pnl = exit_price - entry_price
                    vwap_sweeps.append({
                        'session': session_date,
                        'direction': 'LONG',
                        'pattern': 'VWAP_SWEEP_BELOW_FAIL',
                        'entry': entry_price,
                        'stop': stop,
                        'target': target,
                        'exit': exit_price,
                        'pnl_pts': pnl,
                        'pnl_dollars': pnl_pts_to_dollars(pnl),
                        'outcome': outcome,
                        'risk_pts': risk,
                        'day_type': row['eod_day_type'],
                    })

vsdf = pd.DataFrame(vwap_sweeps) if vwap_sweeps else pd.DataFrame()

if len(vsdf) > 0:
    wins = (vsdf['pnl_pts'] > 0).sum()
    wr = wins / len(vsdf) * 100
    total_pnl = vsdf['pnl_dollars'].sum()
    monthly = total_pnl / months
    print(f"VWAP Sweep-Fail Results: {len(vsdf)} trades, WR {wr:.0f}%, ${monthly:.0f}/mo\n")

    print(f"| Direction | N | WR | Avg P&L | $/Mo |")
    print(f"|-----------|---|------|---------|------|")
    for d in ['LONG', 'SHORT']:
        sub = vsdf[vsdf['direction'] == d]
        if len(sub) == 0:
            continue
        w = (sub['pnl_pts'] > 0).sum()
        wr = w / len(sub) * 100
        avg = sub['pnl_pts'].mean()
        mo = sub['pnl_dollars'].sum() / months
        print(f"| {d} | {len(sub)} | {wr:.0f}% | {avg:+.1f} pts | ${mo:.0f} |")
else:
    print("No VWAP sweep-fail patterns detected on balance days.")


# ============================================================================
# PART 6: WIDE IB BALANCE DAYS
# ============================================================================
print("\n\n" + "=" * 70)
print("PART 6: WIDE IB BALANCE DAYS (IB > 200 pts)")
print("=" * 70)

wide_balance = sdf[sdf['is_wide_balance'] == True]
print(f"\nWide IB balance days: {len(wide_balance)} ({len(wide_balance)/len(sdf)*100:.0f}%)")
print(f"  Per month: {len(wide_balance)/months:.1f}")

if len(wide_balance) > 0:
    print(f"\n  IB Range: mean {wide_balance['ib_range'].mean():.0f}, median {wide_balance['ib_range'].median():.0f}")
    print(f"  Session Range: mean {wide_balance['session_range'].mean():.0f}")
    print(f"  Day types: {wide_balance['eod_day_type'].value_counts().to_dict()}")

    # Wide balance edge fades
    wide_touches = tdf[tdf['is_wide_balance'] == True] if len(tdf) > 0 else pd.DataFrame()
    if len(wide_touches) > 0:
        print(f"\n### 6.1: Wide IB Edge Fades\n")
        print(f"| Metric | Wide IB Balance | All Balance | Difference |")
        print(f"|--------|----------------|-------------|-----------|")

        wide_succ = wide_touches['reached_mid'].mean() * 100
        all_succ = balance_touches['reached_mid'].mean() * 100 if len(balance_touches) > 0 else 0
        print(f"| Fade success (→ mid) | {wide_succ:.0f}% | {all_succ:.0f}% | {wide_succ - all_succ:+.0f}pp |")

        wide_adv = wide_touches['max_adverse'].mean()
        all_adv = balance_touches['max_adverse'].mean() if len(balance_touches) > 0 else 0
        print(f"| Avg max adverse | {wide_adv:.0f} pts | {all_adv:.0f} pts | — |")

        # Backtest on wide IB
        if len(wide_touches) > 0:
            wide_bt = wide_touches.copy()
            wide_bt['poc_aligned'] = True  # Simplified for wide IB
            result = backtest_fade(
                wide_bt,
                lambda t: t['accept_2x5min'] == True and t['delta_aligned'] == True,
                0.10, 'ib_mid', 'Wide IB: 2x5min + delta → IB mid'
            )
            if result:
                print(f"\n  Wide IB backtest: {result['n']} trades, WR {result['wr']:.0f}%, PF {result['pf']:.2f}, ${result['monthly']:.0f}/mo")
    else:
        print("\n  No wide IB edge touches found.")
else:
    print("  No wide IB balance days found in dataset.")


# ============================================================================
# PART 7: COMBINED OPTIMAL + STRATEGY CARD
# ============================================================================
print("\n\n" + "=" * 70)
print("PART 7: COMBINED OPTIMAL FILTER + STRATEGY COMPARISON")
print("=" * 70)

# Stack best filters: 2x5min + delta + POC aligned + balance day
if len(bt_bt) > 0:
    print("\n### 7.1: Filter Stacking (Cumulative)\n")

    stacks = [
        ('Baseline: all balance touches', lambda t: True),
        ('+ 2x5min accepted', lambda t: t['accept_2x5min'] == True),
        ('+ delta aligned', lambda t: t['accept_2x5min'] == True and t['delta_aligned'] == True),
        ('+ POC shape aligned', lambda t: t['accept_2x5min'] == True and t['delta_aligned'] == True and t.get('poc_aligned', False) == True),
        ('LONG only (2x5min + delta)', lambda t: t['accept_2x5min'] == True and t['delta_aligned'] == True and t['fade_dir'] == 'LONG'),
    ]

    print(f"| Filter Stack | N | WR | PF | $/Mo | T/Mo |")
    print(f"|-------------|---|------|-----|------|------|")

    for label, filter_fn in stacks:
        result = backtest_fade(bt_bt, filter_fn, 0.10, 'ib_mid', label)
        if result is None:
            print(f"| {label} | 0 | — | — | — | — |")
            continue
        print(f"| {label} | {result['n']} | {result['wr']:.0f}% | {result['pf']:.2f} | ${result['monthly']:.0f} | {result['trades_mo']:.1f} |")

# Comparison to existing strategies
print("\n### 7.2: Comparison to Existing Strategies\n")

print(f"| Strategy | WR | PF | $/Mo | Risk | Freq |")
print(f"|----------|-----|-----|------|------|------|")
print(f"| 80P (Limit 50% VA + 4R) | 44.7% | 2.57 | $1,922 | 60p | 4.0/mo |")
print(f"| Bracket Retrace + Delta | 83% | ~5.0 | $360 | 15p | 2.0/mo |")
print(f"| 20P (3x5min + 2xATR + 2R) | 45.5% | 1.78 | $496 | 32p | 3.7/mo |")
if best_config:
    print(f"| **Balance Day (best)** | **{best_config['wr']:.0f}%** | **{best_config['pf']:.2f}** | **${best_config['monthly']:.0f}** | **{best_config['avg_risk']:.0f}p** | **{best_config['trades_mo']:.1f}/mo** |")


# ============================================================================
# SUMMARY
# ============================================================================
print("\n\n" + "=" * 70)
print("SUMMARY — KEY FINDINGS")
print("=" * 70)

print("""
PART 1: Day Type Foundation
  - Day type frequency distribution across 259 sessions
  - IB width by day type
  - Early identification accuracy at IB close and C-period
  - C-period rule validation (vs claimed 70-75%)

PART 2: IB Edge Rejection Analysis
  - Touch frequency on balance days
  - Fade success rates (reached IB mid) by direction and day type
  - Differentiating factors: delta, penetration, volume, BPR/FVG, acceptance

PART 3: Volume Profile Shape
  - IB POC shape (P/D/b) vs EOD day type prediction
  - POC alignment improves fade WR
  - VWAP position relative to IB mid as directional bias

PART 4: Acceptance Model Comparison
  - 2x1min vs 2x5min vs 30min vs delta-only
  - Combined acceptance + delta

PART 5: Entry Model Backtests
  - 10 configurations tested with different filters and targets
  - LONG vs SHORT directional bias on balance days
  - Dynamic target based on developing profile shape

PART 5B: VWAP Sweep-Fail Pattern
  - Price sweeps above/below VWAP and fails back on balance days

PART 6: Wide IB Balance
  - IB > 200 pts that stays inside — different behavior?

PART 7: Combined Optimal
  - Best filter stack for maximum edge
  - Comparison to 80P, 20P, bracket retrace

See tables above for detailed findings.
""")

# ============================================================================
# PART 8: P-DAY vs B-DAY vs NEUTRAL SEGMENTATION
# ============================================================================
print("\n\n" + "=" * 70)
print("PART 8: P-DAY vs B-DAY vs NEUTRAL — DIRECTION-SPECIFIC ANALYSIS")
print("=" * 70)

print("\nKey question: Should we go LONG-only on P-days and B-days (NQ long bias)?")
print("Does shorting at the 'seam' of B-days work? Should we skip shorts on P-days?\n")

# 8.1: Day type breakdown of IB edge touches
print("### 8.1: IB Edge Touches by EOD Day Type\n")

if len(tdf) > 0:
    print(f"| Day Type | IBH Touches | IBL Touches | Total | Fade→Mid % (IBL LONG) | Fade→Mid % (IBH SHORT) |")
    print(f"|----------|-------------|-------------|-------|----------------------|----------------------|")
    for dt in ['P_DAY', 'B_DAY', 'NEUTRAL', 'TREND']:
        sub = tdf[tdf['eod_day_type'] == dt]
        if len(sub) == 0:
            continue
        ibh = sub[sub['edge'] == 'IBH']
        ibl = sub[sub['edge'] == 'IBL']
        ibl_succ = ibl['reached_mid'].mean() * 100 if len(ibl) > 0 else 0
        ibh_succ = ibh['reached_mid'].mean() * 100 if len(ibh) > 0 else 0
        print(f"| {dt} | {len(ibh)} | {len(ibl)} | {len(sub)} | {ibl_succ:.0f}% (N={len(ibl)}) | {ibh_succ:.0f}% (N={len(ibh)}) |")

# 8.2: Backtest by day type — LONG-only vs SHORT-only vs BOTH
print("\n### 8.2: 30-Min Acceptance Backtest by Day Type\n")

if len(bt_bt) > 0:
    print(f"| Day Type | Dir | N | T/Mo | WR | PF | $/Mo | Avg Risk |")
    print(f"|----------|-----|---|------|-----|-----|------|---------|")

    for dt in ['P_DAY', 'B_DAY', 'NEUTRAL', 'TREND']:
        for direction in ['LONG', 'SHORT', 'BOTH']:
            if direction == 'BOTH':
                filter_fn = lambda t, d=dt: t['accept_30min'] == True and t['eod_day_type'] == d
            else:
                filter_fn = lambda t, d=dt, dr=direction: t['accept_30min'] == True and t['eod_day_type'] == d and t['fade_dir'] == dr

            result = backtest_fade(bt_bt, filter_fn, 0.10, 'ib_mid', f'{dt} {direction}')
            if result is None or result['n'] < 2:
                print(f"| {dt} | {direction} | {result['n'] if result else 0} | — | — | — | — | — |")
                continue
            print(f"| {dt} | {direction} | {result['n']} | {result['trades_mo']:.1f} | {result['wr']:.0f}% | {result['pf']:.2f} | ${result['monthly']:.0f} | {result['avg_risk']:.0f}p |")

# 8.3: B-Day specific — LONG at IBL (the low of the B)
print("\n### 8.3: B-Day LONG at IBL — 'Long at the Low of the B'\n")
print("b-shape (POC in lower third) = heavy volume at the low → support")
print("After price breaks below IBL and accepts back in → LONG to IB mid\n")

if len(bt_bt) > 0:
    # B-day LONG with various filters
    bday_configs = [
        ('B-day LONG (30min accept)', lambda t: t['accept_30min'] == True and t['eod_day_type'] == 'B_DAY' and t['fade_dir'] == 'LONG'),
        ('B-day LONG (2x5min accept)', lambda t: t['accept_2x5min'] == True and t['eod_day_type'] == 'B_DAY' and t['fade_dir'] == 'LONG'),
        ('B-day LONG + delta', lambda t: t['accept_2x5min'] == True and t['eod_day_type'] == 'B_DAY' and t['fade_dir'] == 'LONG' and t['delta_aligned'] == True),
        ('B-day LONG + VWAP above mid', lambda t: t['accept_2x5min'] == True and t['eod_day_type'] == 'B_DAY' and t['fade_dir'] == 'LONG' and t['vwap_vs_mid'] == 'above'),
        ('b-shape LONG (any day type)', lambda t: t['accept_30min'] == True and t['poc_shape'] == 'B_SHAPE' and t['fade_dir'] == 'LONG'),
    ]

    print(f"| Config | N | T/Mo | WR | PF | $/Mo | Avg Risk |")
    print(f"|--------|---|------|-----|-----|------|---------|")
    for label, filter_fn in bday_configs:
        result = backtest_fade(bt_bt, filter_fn, 0.10, 'ib_mid', label)
        if result is None or result['n'] == 0:
            print(f"| {label} | 0 | — | — | — | — | — |")
            continue
        print(f"| {label} | {result['n']} | {result['trades_mo']:.1f} | {result['wr']:.0f}% | {result['pf']:.2f} | ${result['monthly']:.0f} | {result['avg_risk']:.0f}p |")

# 8.4: P-Day specific — LONG at IBL or VWAP, NO SHORTS
print("\n### 8.4: P-Day LONG Only — 'Go Long if Opportunity Exists'\n")
print("P-day = bullish skew (POC migrating up). Don't short P-day — NQ long bias.\n")

if len(bt_bt) > 0:
    pday_configs = [
        ('P-day LONG (30min accept)', lambda t: t['accept_30min'] == True and t['eod_day_type'] == 'P_DAY' and t['fade_dir'] == 'LONG'),
        ('P-day LONG (2x5min accept)', lambda t: t['accept_2x5min'] == True and t['eod_day_type'] == 'P_DAY' and t['fade_dir'] == 'LONG'),
        ('P-day LONG + delta', lambda t: t['accept_2x5min'] == True and t['eod_day_type'] == 'P_DAY' and t['fade_dir'] == 'LONG' and t['delta_aligned'] == True),
        ('P-day SHORT (for comparison)', lambda t: t['accept_30min'] == True and t['eod_day_type'] == 'P_DAY' and t['fade_dir'] == 'SHORT'),
        ('P-shape LONG (any day type)', lambda t: t['accept_30min'] == True and t['poc_shape'] == 'P_SHAPE' and t['fade_dir'] == 'LONG'),
    ]

    print(f"| Config | N | T/Mo | WR | PF | $/Mo | Avg Risk |")
    print(f"|--------|---|------|-----|-----|------|---------|")
    for label, filter_fn in pday_configs:
        result = backtest_fade(bt_bt, filter_fn, 0.10, 'ib_mid', label)
        if result is None or result['n'] == 0:
            print(f"| {label} | 0 | — | — | — | — | — |")
            continue
        print(f"| {label} | {result['n']} | {result['trades_mo']:.1f} | {result['wr']:.0f}% | {result['pf']:.2f} | ${result['monthly']:.0f} | {result['avg_risk']:.0f}p |")

# 8.5: Neutral Day — Both directions
print("\n### 8.5: Neutral Day — Both Directions\n")
print("Neutral = POC at center, balanced rotation. Both edges should fade equally.\n")

if len(bt_bt) > 0:
    neutral_configs = [
        ('Neutral LONG (30min)', lambda t: t['accept_30min'] == True and t['eod_day_type'] == 'NEUTRAL' and t['fade_dir'] == 'LONG'),
        ('Neutral SHORT (30min)', lambda t: t['accept_30min'] == True and t['eod_day_type'] == 'NEUTRAL' and t['fade_dir'] == 'SHORT'),
        ('Neutral BOTH (30min)', lambda t: t['accept_30min'] == True and t['eod_day_type'] == 'NEUTRAL'),
        ('D-shape LONG (any type)', lambda t: t['accept_30min'] == True and t['poc_shape'] == 'D_SHAPE' and t['fade_dir'] == 'LONG'),
    ]

    print(f"| Config | N | T/Mo | WR | PF | $/Mo | Avg Risk |")
    print(f"|--------|---|------|-----|-----|------|---------|")
    for label, filter_fn in neutral_configs:
        result = backtest_fade(bt_bt, filter_fn, 0.10, 'ib_mid', label)
        if result is None or result['n'] == 0:
            print(f"| {label} | 0 | — | — | — | — | — |")
            continue
        print(f"| {label} | {result['n']} | {result['trades_mo']:.1f} | {result['wr']:.0f}% | {result['pf']:.2f} | ${result['monthly']:.0f} | {result['avg_risk']:.0f}p |")

# 8.6: LONG-only across all balance day types combined
print("\n### 8.6: LONG-Only Strategy — Best Combinations\n")
print("NQ long bias: LONG IBL fades dominate. What's the best LONG-only combo?\n")

if len(bt_bt) > 0:
    long_configs = [
        ('ALL days LONG (30min)', lambda t: t['accept_30min'] == True and t['fade_dir'] == 'LONG'),
        ('Balance LONG (30min)', lambda t: t['accept_30min'] == True and t['fade_dir'] == 'LONG' and t['is_balance'] == True),
        ('B+P+N LONG (30min)', lambda t: t['accept_30min'] == True and t['fade_dir'] == 'LONG' and t['eod_day_type'] in ('B_DAY', 'P_DAY', 'NEUTRAL')),
        ('P+B LONG only (30min)', lambda t: t['accept_30min'] == True and t['fade_dir'] == 'LONG' and t['eod_day_type'] in ('P_DAY', 'B_DAY')),
        ('LONG + VWAP above mid (30min)', lambda t: t['accept_30min'] == True and t['fade_dir'] == 'LONG' and t['vwap_vs_mid'] == 'above'),
        ('LONG + first touch (30min)', lambda t: t['accept_30min'] == True and t['fade_dir'] == 'LONG' and t['touch_num'] == 1),
        ('LONG + first touch + VWAP above', lambda t: t['accept_30min'] == True and t['fade_dir'] == 'LONG' and t['touch_num'] == 1 and t['vwap_vs_mid'] == 'above'),
    ]

    print(f"| Config | N | T/Mo | WR | PF | $/Mo | Avg Risk |")
    print(f"|--------|---|------|-----|-----|------|---------|")
    for label, filter_fn in long_configs:
        result = backtest_fade(bt_bt, filter_fn, 0.10, 'ib_mid', label)
        if result is None or result['n'] == 0:
            print(f"| {label} | 0 | — | — | — | — | — |")
            continue
        print(f"| {label} | {result['n']} | {result['trades_mo']:.1f} | {result['wr']:.0f}% | {result['pf']:.2f} | ${result['monthly']:.0f} | {result['avg_risk']:.0f}p |")

# 8.7: Does shorting at the "seam" of B-day work?
print("\n### 8.7: B-Day SHORT at IBH — 'Shorting the Seam'\n")
print("User question: does shorting at the seam of the B-day work?\n")

if len(bt_bt) > 0:
    bday_short_configs = [
        ('B-day SHORT (30min)', lambda t: t['accept_30min'] == True and t['eod_day_type'] == 'B_DAY' and t['fade_dir'] == 'SHORT'),
        ('B-day SHORT (2x5min)', lambda t: t['accept_2x5min'] == True and t['eod_day_type'] == 'B_DAY' and t['fade_dir'] == 'SHORT'),
        ('B-day SHORT + delta', lambda t: t['accept_2x5min'] == True and t['eod_day_type'] == 'B_DAY' and t['fade_dir'] == 'SHORT' and t['delta_aligned'] == True),
    ]

    print(f"| Config | N | T/Mo | WR | PF | $/Mo | Avg Risk |")
    print(f"|--------|---|------|-----|-----|------|---------|")
    for label, filter_fn in bday_short_configs:
        result = backtest_fade(bt_bt, filter_fn, 0.10, 'ib_mid', label)
        if result is None or result['n'] == 0:
            print(f"| {label} | 0 | — | — | — | — | — |")
            continue
        print(f"| {label} | {result['n']} | {result['trades_mo']:.1f} | {result['wr']:.0f}% | {result['pf']:.2f} | ${result['monthly']:.0f} | {result['avg_risk']:.0f}p |")


# ============================================================================
# PART 9: COMPLETE EDGE FADE MATRIX — SWEEPS + VAH/VAL + ALL DAY TYPES
# ============================================================================
print("\n\n" + "=" * 70)
print("PART 9: COMPLETE EDGE FADE MATRIX — SWEEPS, VAH/VAL, ALL DAY TYPES")
print("=" * 70)

print("""
Testing every edge × every day type × both directions:
  P-day: LONG at IBL (buy low of P), SHORT at IBH sweep (short high of P)
  B-day: LONG at IBL (long B low), SHORT at IBH (short B seam)
  Neutral/Wide: LONG at VAL/IBL after sweep, SHORT at VAH/IBH after sweep
  IB range guides stops/targets, but market skews P or B or neutral
""")

# ── 9.0: Compute IB VAH/VAL + add prior-day VA to session data ────────────
print("Computing IB VAH/VAL and sweep detection for all sessions...\n")

sweep_touches = []   # sweep-based entries at IB edges + VA levels
SWEEP_MIN = 3.0      # minimum points beyond level to count as sweep

for _, row in sdf.iterrows():
    session_date = row['session']
    ib_high = row['ib_high']
    ib_low = row['ib_low']
    ib_range = row['ib_range']
    ib_mid = row['ib_mid']
    eod_day_type = row['eod_day_type']
    poc_shape = row['poc_shape']

    bars = get_session_bars(session_date)
    if len(bars) < 90:
        continue

    ib_bars = bars.head(60)
    post_ib = bars.iloc[60:]

    # Compute IB VAH/VAL
    ib_va = calculate_value_area(
        ib_bars['high'].values, ib_bars['low'].values,
        ib_bars['volume'].values, tick_size=0.25, va_percent=0.70
    )
    if ib_va:
        ib_vah = ib_va.vah
        ib_val = ib_va.val
    else:
        ib_vah = ib_high - ib_range * 0.15
        ib_val = ib_low + ib_range * 0.15

    # Prior-day VA levels (already in bars from compute_all_features)
    prior_vah = bars['prior_va_vah'].iloc[0] if 'prior_va_vah' in bars.columns and not pd.isna(bars['prior_va_vah'].iloc[0]) else None
    prior_val = bars['prior_va_val'].iloc[0] if 'prior_va_val' in bars.columns and not pd.isna(bars['prior_va_val'].iloc[0]) else None

    # VWAP at IB close
    vwap_at_ib = ib_bars['vwap'].iloc[-1] if 'vwap' in ib_bars.columns else ib_mid

    # Define all levels to test
    levels = [
        ('IBH', ib_high, 'SHORT'),      # fade: sweep above IBH → short
        ('IBL', ib_low, 'LONG'),         # fade: sweep below IBL → long
        ('IB_VAH', ib_vah, 'SHORT'),     # fade: sweep above IB VAH → short
        ('IB_VAL', ib_val, 'LONG'),      # fade: sweep below IB VAL → long
    ]
    if prior_vah is not None:
        levels.append(('PRIOR_VAH', prior_vah, 'SHORT'))
    if prior_val is not None:
        levels.append(('PRIOR_VAL', prior_val, 'LONG'))

    # Track cooldowns per level
    last_touch_by_level = {}

    for i, (idx, bar) in enumerate(post_ib.iterrows()):
        bar_idx = 60 + i
        bar_time = bar['time']
        if bar_time >= ENTRY_CUTOFF:
            break

        delta = bar.get('delta', 0)
        vwap = bar.get('vwap', ib_mid)

        for level_name, level_price, fade_dir in levels:
            # Cooldown
            last = last_touch_by_level.get(level_name, -30)
            if bar_idx - last < 15:
                continue

            is_sweep = False
            penetration = 0

            if fade_dir == 'LONG':
                # Sweep below: bar low goes below level, close comes back above
                if bar['low'] < level_price - SWEEP_MIN:
                    penetration = level_price - bar['low']
                    # Check if price sweeps and rejects (close back near/above level)
                    if bar['close'] > level_price - ib_range * 0.05:
                        is_sweep = True
                    # Or: touch-based (just got close enough)
                    elif bar['low'] <= level_price + TOUCH_PROXIMITY:
                        is_sweep = False  # just a touch, not a sweep
            else:  # SHORT
                if bar['high'] > level_price + SWEEP_MIN:
                    penetration = bar['high'] - level_price
                    if bar['close'] < level_price + ib_range * 0.05:
                        is_sweep = True
                    elif bar['high'] >= level_price - TOUCH_PROXIMITY:
                        is_sweep = False

            # Also detect plain touch (for comparison)
            is_touch = False
            if fade_dir == 'LONG' and bar['low'] <= level_price + TOUCH_PROXIMITY:
                is_touch = True
                if penetration == 0:
                    penetration = max(0, level_price - bar['low'])
            elif fade_dir == 'SHORT' and bar['high'] >= level_price - TOUCH_PROXIMITY:
                is_touch = True
                if penetration == 0:
                    penetration = max(0, bar['high'] - level_price)

            if not is_touch and not is_sweep:
                continue

            last_touch_by_level[level_name] = bar_idx

            # Check acceptance models on future bars
            future_bars = post_ib.iloc[i+1:i+61] if i+1 < len(post_ib) else pd.DataFrame()
            accept_2x5min = False
            accept_30min = False
            accept_30min_bar = None

            if len(future_bars) >= 2:
                # 2x5min acceptance back inside
                period_consec = 0
                for ps in range(0, len(future_bars), 5):
                    pe = min(ps + 5, len(future_bars))
                    pc = future_bars.iloc[pe - 1]['close']
                    if fade_dir == 'LONG':
                        inside = pc > level_price
                    else:
                        inside = pc < level_price
                    if inside:
                        period_consec += 1
                        if period_consec >= 2 and not accept_2x5min:
                            accept_2x5min = True
                    else:
                        period_consec = 0

                # 30min acceptance
                if len(future_bars) >= 30:
                    fc = future_bars.iloc[29]['close']
                    if fade_dir == 'LONG' and fc > level_price:
                        accept_30min = True
                        accept_30min_bar = bar_idx + 30
                    elif fade_dir == 'SHORT' and fc < level_price:
                        accept_30min = True
                        accept_30min_bar = bar_idx + 30

            # Did price reach IB mid within 60 bars?
            reached_mid = False
            reached_vwap = False
            max_adverse = 0

            for fi, (fidx, fbar) in enumerate(future_bars.iterrows()):
                if fade_dir == 'LONG':
                    adverse = max(0, level_price - fbar['low'])
                    if fbar['high'] >= ib_mid:
                        reached_mid = True
                    if vwap and not pd.isna(vwap) and fbar['high'] >= vwap:
                        reached_vwap = True
                else:
                    adverse = max(0, fbar['high'] - level_price)
                    if fbar['low'] <= ib_mid:
                        reached_mid = True
                    if vwap and not pd.isna(vwap) and fbar['low'] <= vwap:
                        reached_vwap = True
                if adverse > max_adverse:
                    max_adverse = adverse

            delta_aligned = (delta > 0) if fade_dir == 'LONG' else (delta < 0)

            sweep_touches.append({
                'session': session_date,
                'level': level_name,
                'level_price': level_price,
                'fade_dir': fade_dir,
                'is_sweep': is_sweep,
                'is_touch': is_touch,
                'penetration': penetration,
                'touch_bar': bar_idx,
                'touch_time': bar_time,
                'delta': delta,
                'delta_aligned': delta_aligned,
                'vwap': vwap,
                'vwap_vs_mid': 'above' if (vwap and not pd.isna(vwap) and vwap > ib_mid) else 'below',
                'accept_2x5min': accept_2x5min,
                'accept_30min': accept_30min,
                'accept_30min_bar': accept_30min_bar,
                'reached_mid': reached_mid,
                'reached_vwap': reached_vwap,
                'max_adverse': max_adverse,
                'eod_day_type': eod_day_type,
                'poc_shape': poc_shape,
                'ib_high': ib_high,
                'ib_low': ib_low,
                'ib_range': ib_range,
                'ib_mid': ib_mid,
                'ib_vah': ib_vah,
                'ib_val': ib_val,
                'is_balance': row['is_balance'],
                'is_wide_balance': row['is_wide_balance'],
            })

stdf = pd.DataFrame(sweep_touches)
print(f"Total level touches/sweeps detected: {len(stdf)}")
print(f"  Sweeps (penetration > {SWEEP_MIN} pts + close rejects): {stdf['is_sweep'].sum()}")
for lv in ['IBH', 'IBL', 'IB_VAH', 'IB_VAL', 'PRIOR_VAH', 'PRIOR_VAL']:
    sub = stdf[stdf['level'] == lv]
    if len(sub) > 0:
        print(f"  {lv}: {len(sub)} touches, {sub['is_sweep'].sum()} sweeps")


# ── 9.1: Backtest helper for sweep-based entries ──────────────────────────
def backtest_level_fade(touches, entry_filter, stop_buffer_pct, target_mode, label):
    """Backtest a level fade with sweep/acceptance filter."""
    trades = []

    for _, touch in touches.iterrows():
        if not entry_filter(touch):
            continue

        session_date = touch['session']
        fade_dir = touch['fade_dir']
        level_price = touch['level_price']
        ib_high = touch['ib_high']
        ib_low = touch['ib_low']
        ib_range = touch['ib_range']
        ib_mid = touch['ib_mid']
        vwap = touch['vwap']

        # Entry at acceptance bar or touch bar
        entry_bar_idx = touch['touch_bar']
        if touch['accept_30min'] and touch.get('accept_30min_bar') and pd.notna(touch['accept_30min_bar']):
            entry_bar_idx = int(touch['accept_30min_bar'])

        bars_session = get_session_bars(session_date)
        if entry_bar_idx >= len(bars_session):
            continue
        entry_price = bars_session.iloc[entry_bar_idx]['close']

        # Stop: beyond the level by buffer
        if fade_dir == 'LONG':
            stop = level_price - ib_range * stop_buffer_pct
        else:
            stop = level_price + ib_range * stop_buffer_pct

        risk = abs(entry_price - stop)
        if risk <= 0:
            continue

        # Target
        if target_mode == 'ib_mid':
            target = ib_mid
        elif target_mode == 'vwap':
            target = vwap if vwap and not pd.isna(vwap) else ib_mid
        elif target_mode == 'opposite_edge':
            target = ib_high if fade_dir == 'LONG' else ib_low
        elif target_mode == 'ib_vah_val':
            target = touch['ib_vah'] if fade_dir == 'LONG' else touch['ib_val']
        else:
            target = ib_mid

        # Ensure target in right direction
        if fade_dir == 'LONG' and target <= entry_price:
            target = entry_price + risk
        if fade_dir == 'SHORT' and target >= entry_price:
            target = entry_price - risk

        # Walk bars
        remaining = bars_session.iloc[entry_bar_idx:]
        outcome = 'EOD'
        exit_price = remaining['close'].iloc[-1]

        for _, bar in remaining.iterrows():
            if bar['time'] >= EOD_EXIT:
                exit_price = bar['close']
                outcome = 'EOD'
                break
            if fade_dir == 'LONG':
                if bar['low'] <= stop:
                    exit_price = stop
                    outcome = 'STOP'
                    break
                if bar['high'] >= target:
                    exit_price = target
                    outcome = 'TARGET'
                    break
            else:
                if bar['high'] >= stop:
                    exit_price = stop
                    outcome = 'STOP'
                    break
                if bar['low'] <= target:
                    exit_price = target
                    outcome = 'TARGET'
                    break

        pnl_pts = (exit_price - entry_price) if fade_dir == 'LONG' else (entry_price - exit_price)

        trades.append({
            'session': session_date,
            'direction': fade_dir,
            'level': touch['level'],
            'entry': entry_price,
            'stop': stop,
            'target': target,
            'exit': exit_price,
            'pnl_pts': pnl_pts,
            'pnl_dollars': pnl_pts_to_dollars(pnl_pts),
            'outcome': outcome,
            'risk_pts': risk,
        })

    if not trades:
        return None
    tdf_r = pd.DataFrame(trades)
    n = len(tdf_r)
    wins = (tdf_r['pnl_pts'] > 0).sum()
    wr = wins / n * 100
    total_pnl = tdf_r['pnl_dollars'].sum()
    monthly_pnl = total_pnl / months
    gw = tdf_r[tdf_r['pnl_pts'] > 0]['pnl_dollars'].sum()
    gl = abs(tdf_r[tdf_r['pnl_pts'] <= 0]['pnl_dollars'].sum())
    pf = gw / gl if gl > 0 else float('inf')
    return {'label': label, 'n': n, 'trades_mo': n / months, 'wr': wr,
            'pf': pf, 'monthly': monthly_pnl, 'avg_risk': tdf_r['risk_pts'].mean(),
            'trades_df': tdf_r}


def print_result(result):
    if result is None or result['n'] == 0:
        return "0 | — | — | — | — | — | —"
    return f"{result['n']} | {result['trades_mo']:.1f} | {result['wr']:.0f}% | {result['pf']:.2f} | ${result['monthly']:.0f} | {result['avg_risk']:.0f}p"


# ── 9.2: P-DAY EDGES ─────────────────────────────────────────────────────
print("\n\n### 9.2: P-Day Edge Fades — Buy Low of P, Short High of P Sweep\n")
print("P-day = bullish skew (POC migrating up)")
print("  LONG: buy at IBL or IB VAL after price sweeps below and rejects")
print("  SHORT: short at IBH after price sweeps above IBH and fails\n")

pday = stdf[stdf['eod_day_type'] == 'P_DAY']

print(f"| P-Day Edge | Dir | N | T/Mo | WR | PF | $/Mo | Risk |")
print(f"|------------|-----|---|------|-----|-----|------|------|")

pday_tests = [
    # Buy low of P
    ('IBL touch+30min', 'LONG', lambda t: t['level'] == 'IBL' and t['accept_30min']),
    ('IBL sweep+30min', 'LONG', lambda t: t['level'] == 'IBL' and t['is_sweep'] and t['accept_30min']),
    ('IBL touch+2x5min', 'LONG', lambda t: t['level'] == 'IBL' and t['accept_2x5min']),
    ('IB_VAL touch+30min', 'LONG', lambda t: t['level'] == 'IB_VAL' and t['accept_30min']),
    ('IB_VAL sweep+30min', 'LONG', lambda t: t['level'] == 'IB_VAL' and t['is_sweep'] and t['accept_30min']),
    ('PRIOR_VAL touch+30min', 'LONG', lambda t: t['level'] == 'PRIOR_VAL' and t['accept_30min']),
    # Short high of P sweep
    ('IBH touch+30min', 'SHORT', lambda t: t['level'] == 'IBH' and t['accept_30min']),
    ('IBH sweep+30min', 'SHORT', lambda t: t['level'] == 'IBH' and t['is_sweep'] and t['accept_30min']),
    ('IBH sweep+2x5min', 'SHORT', lambda t: t['level'] == 'IBH' and t['is_sweep'] and t['accept_2x5min']),
    ('IB_VAH touch+30min', 'SHORT', lambda t: t['level'] == 'IB_VAH' and t['accept_30min']),
    ('IB_VAH sweep+30min', 'SHORT', lambda t: t['level'] == 'IB_VAH' and t['is_sweep'] and t['accept_30min']),
]

for name, direction, filt in pday_tests:
    result = backtest_level_fade(pday, filt, 0.10, 'ib_mid', name)
    n_str = print_result(result)
    print(f"| {name} | {direction} | {n_str} |")


# ── 9.3: B-DAY EDGES ─────────────────────────────────────────────────────
print("\n\n### 9.3: B-Day Edge Fades — Long B Low, Short B Seam\n")
print("B-day = bearish skew (POC in lower third)")
print("  LONG: buy at IBL (long the low of the B) after sweep/acceptance")
print("  SHORT: short at IBH (short the seam of B) after sweep/acceptance\n")

bday = stdf[stdf['eod_day_type'] == 'B_DAY']

print(f"| B-Day Edge | Dir | N | T/Mo | WR | PF | $/Mo | Risk |")
print(f"|------------|-----|---|------|-----|-----|------|------|")

bday_tests = [
    # Long B low
    ('IBL touch+30min', 'LONG', lambda t: t['level'] == 'IBL' and t['accept_30min']),
    ('IBL sweep+30min', 'LONG', lambda t: t['level'] == 'IBL' and t['is_sweep'] and t['accept_30min']),
    ('IBL touch+2x5min', 'LONG', lambda t: t['level'] == 'IBL' and t['accept_2x5min']),
    ('IBL sweep+delta', 'LONG', lambda t: t['level'] == 'IBL' and t['is_sweep'] and t['delta_aligned']),
    ('IB_VAL touch+30min', 'LONG', lambda t: t['level'] == 'IB_VAL' and t['accept_30min']),
    ('IB_VAL sweep+30min', 'LONG', lambda t: t['level'] == 'IB_VAL' and t['is_sweep'] and t['accept_30min']),
    # Short B seam (IBH)
    ('IBH touch+30min', 'SHORT', lambda t: t['level'] == 'IBH' and t['accept_30min']),
    ('IBH sweep+30min', 'SHORT', lambda t: t['level'] == 'IBH' and t['is_sweep'] and t['accept_30min']),
    ('IBH touch+2x5min', 'SHORT', lambda t: t['level'] == 'IBH' and t['accept_2x5min']),
    ('IB_VAH touch+30min', 'SHORT', lambda t: t['level'] == 'IB_VAH' and t['accept_30min']),
    ('IB_VAH sweep+30min', 'SHORT', lambda t: t['level'] == 'IB_VAH' and t['is_sweep'] and t['accept_30min']),
]

for name, direction, filt in bday_tests:
    result = backtest_level_fade(bday, filt, 0.10, 'ib_mid', name)
    n_str = print_result(result)
    print(f"| {name} | {direction} | {n_str} |")


# ── 9.4: NEUTRAL DAY EDGES ───────────────────────────────────────────────
print("\n\n### 9.4: Neutral Day Edge Fades — Short VAH, Long VAL After Sweep\n")
print("Neutral = POC at center, perfect rotation.")
print("  IB range guides stops/targets. Sweep above VAH → short, sweep below VAL → long.\n")

neutral = stdf[stdf['eod_day_type'] == 'NEUTRAL']

print(f"| Neutral Edge | Dir | N | T/Mo | WR | PF | $/Mo | Risk |")
print(f"|--------------|-----|---|------|-----|-----|------|------|")

neutral_tests = [
    # Long at VAL / IBL
    ('IBL touch+30min', 'LONG', lambda t: t['level'] == 'IBL' and t['accept_30min']),
    ('IBL sweep+30min', 'LONG', lambda t: t['level'] == 'IBL' and t['is_sweep'] and t['accept_30min']),
    ('IB_VAL touch+30min', 'LONG', lambda t: t['level'] == 'IB_VAL' and t['accept_30min']),
    ('IB_VAL sweep+30min', 'LONG', lambda t: t['level'] == 'IB_VAL' and t['is_sweep'] and t['accept_30min']),
    ('PRIOR_VAL touch+30min', 'LONG', lambda t: t['level'] == 'PRIOR_VAL' and t['accept_30min']),
    # Short at VAH / IBH
    ('IBH touch+30min', 'SHORT', lambda t: t['level'] == 'IBH' and t['accept_30min']),
    ('IBH sweep+30min', 'SHORT', lambda t: t['level'] == 'IBH' and t['is_sweep'] and t['accept_30min']),
    ('IB_VAH touch+30min', 'SHORT', lambda t: t['level'] == 'IB_VAH' and t['accept_30min']),
    ('IB_VAH sweep+30min', 'SHORT', lambda t: t['level'] == 'IB_VAH' and t['is_sweep'] and t['accept_30min']),
    ('PRIOR_VAH touch+30min', 'SHORT', lambda t: t['level'] == 'PRIOR_VAH' and t['accept_30min']),
]

for name, direction, filt in neutral_tests:
    result = backtest_level_fade(neutral, filt, 0.10, 'ib_mid', name)
    n_str = print_result(result)
    print(f"| {name} | {direction} | {n_str} |")


# ── 9.5: WIDE RANGE / WIDE IB BALANCE ────────────────────────────────────
print("\n\n### 9.5: Wide IB Balance Day — Short IBH/VAH, Long IBL/VAL\n")
print("Wide IB (>200 pts) balance day: never trades outside IB.")
print("Big range = big mean reversion. Fade edges with IB as the guide.\n")

wide = stdf[stdf['is_wide_balance'] == True]

print(f"| Wide IB Edge | Dir | N | T/Mo | WR | PF | $/Mo | Risk |")
print(f"|--------------|-----|---|------|-----|-----|------|------|")

wide_tests = [
    ('IBL touch+30min', 'LONG', lambda t: t['level'] == 'IBL' and t['accept_30min']),
    ('IBL sweep+30min', 'LONG', lambda t: t['level'] == 'IBL' and t['is_sweep'] and t['accept_30min']),
    ('IB_VAL touch+30min', 'LONG', lambda t: t['level'] == 'IB_VAL' and t['accept_30min']),
    ('IB_VAL sweep+30min', 'LONG', lambda t: t['level'] == 'IB_VAL' and t['is_sweep'] and t['accept_30min']),
    ('IBH touch+30min', 'SHORT', lambda t: t['level'] == 'IBH' and t['accept_30min']),
    ('IBH sweep+30min', 'SHORT', lambda t: t['level'] == 'IBH' and t['is_sweep'] and t['accept_30min']),
    ('IB_VAH touch+30min', 'SHORT', lambda t: t['level'] == 'IB_VAH' and t['accept_30min']),
    ('IB_VAH sweep+30min', 'SHORT', lambda t: t['level'] == 'IB_VAH' and t['is_sweep'] and t['accept_30min']),
]

for name, direction, filt in wide_tests:
    result = backtest_level_fade(wide, filt, 0.10, 'ib_mid', name)
    n_str = print_result(result)
    print(f"| {name} | {direction} | {n_str} |")


# ── 9.6: BEST EDGE PER DAY TYPE — SUMMARY MATRIX ─────────────────────────
print("\n\n### 9.6: Best Edge Per Day Type — Summary Matrix\n")
print("For each day type, which edge + direction gives the best edge?\n")

print(f"| Day Type | Best LONG | WR | PF | $/Mo | Best SHORT | WR | PF | $/Mo |")
print(f"|----------|-----------|-----|-----|------|------------|-----|-----|------|")

for dt_name, dt_data in [('P_DAY', pday), ('B_DAY', bday), ('NEUTRAL', neutral), ('WIDE_IB', wide)]:
    # Best LONG
    best_long = None
    best_long_monthly = -float('inf')
    long_levels = ['IBL', 'IB_VAL', 'PRIOR_VAL']
    for lv in long_levels:
        for accept in ['accept_30min']:
            filt = lambda t, l=lv, a=accept: t['level'] == l and t[a] == True
            r = backtest_level_fade(dt_data, filt, 0.10, 'ib_mid', f'{lv}')
            if r and r['n'] >= 3 and r['monthly'] > best_long_monthly:
                best_long_monthly = r['monthly']
                best_long = r
                best_long['label'] = lv

    # Best SHORT
    best_short = None
    best_short_monthly = -float('inf')
    short_levels = ['IBH', 'IB_VAH', 'PRIOR_VAH']
    for lv in short_levels:
        for accept in ['accept_30min']:
            filt = lambda t, l=lv, a=accept: t['level'] == l and t[a] == True
            r = backtest_level_fade(dt_data, filt, 0.10, 'ib_mid', f'{lv}')
            if r and r['n'] >= 3 and r['monthly'] > best_short_monthly:
                best_short_monthly = r['monthly']
                best_short = r
                best_short['label'] = lv

    bl = best_long
    bs = best_short
    bl_str = f"{bl['label']} | {bl['wr']:.0f}% | {bl['pf']:.2f} | ${bl['monthly']:.0f}" if bl else "— | — | — | —"
    bs_str = f"{bs['label']} | {bs['wr']:.0f}% | {bs['pf']:.2f} | ${bs['monthly']:.0f}" if bs else "— | — | — | —"
    print(f"| {dt_name} | {bl_str} | {bs_str} |")


# ── 9.7: Sweep vs Plain Touch Comparison ──────────────────────────────────
print("\n\n### 9.7: Does Sweep Improve Over Plain Touch?\n")
print("Sweep = price penetrates beyond the level by 3+ pts then close rejects back")
print("Touch = price just reaches the level (within 5 pts proximity)\n")

print(f"| Day Type | Level | Touch WR (N) | Sweep WR (N) | Sweep Edge |")
print(f"|----------|-------|-------------|-------------|-----------|")

for dt in ['P_DAY', 'B_DAY', 'NEUTRAL']:
    dt_sub = stdf[stdf['eod_day_type'] == dt]
    for lv in ['IBH', 'IBL']:
        lv_sub = dt_sub[dt_sub['level'] == lv]
        touch_only = lv_sub[lv_sub['accept_30min'] == True]
        sweep_only = lv_sub[(lv_sub['is_sweep'] == True) & (lv_sub['accept_30min'] == True)]
        if len(touch_only) == 0:
            continue
        t_wr = touch_only['reached_mid'].mean() * 100
        s_wr = sweep_only['reached_mid'].mean() * 100 if len(sweep_only) > 0 else 0
        s_n = len(sweep_only)
        edge = s_wr - t_wr if s_n > 0 else 0
        print(f"| {dt} | {lv} | {t_wr:.0f}% (N={len(touch_only)}) | {s_wr:.0f}% (N={s_n}) | {edge:+.0f}pp |")


# ── 9.8: Alternate Targets — IB VAH/VAL, VWAP, Opposite Edge ─────────────
print("\n\n### 9.8: Target Comparison — IB Mid vs IB VAH/VAL vs VWAP vs Opposite Edge\n")
print("On B-day LONG at IBL (the strongest setup), which target is best?\n")

bday_ibl_30 = bday[(bday['level'] == 'IBL') & (bday['accept_30min'] == True)]

print(f"| Target | N | WR | PF | $/Mo |")
print(f"|--------|---|-----|-----|------|")

for target_name, target_mode in [
    ('IB Midpoint', 'ib_mid'),
    ('IB VAH', 'ib_vah_val'),
    ('VWAP', 'vwap'),
    ('Opposite Edge (IBH)', 'opposite_edge'),
]:
    filt = lambda t: True
    r = backtest_level_fade(bday_ibl_30, filt, 0.10, target_mode, target_name)
    if r and r['n'] > 0:
        print(f"| {target_name} | {r['n']} | {r['wr']:.0f}% | {r['pf']:.2f} | ${r['monthly']:.0f} |")
    else:
        print(f"| {target_name} | 0 | — | — | — |")


# ── 9.9: Using IB POC Shape as Real-Time Proxy ───────────────────────────
print("\n\n### 9.9: IB POC Shape as Real-Time Day Type Proxy\n")
print("EOD day type can't be known in real-time. But IB POC shape is known at 10:30.")
print("b-shape → likely B-day (54%). P-shape → likely P-day (42%). D-shape → could be anything.\n")

print(f"| POC Shape | Edge | Dir | N | T/Mo | WR | PF | $/Mo |")
print(f"|-----------|------|-----|---|------|-----|-----|------|")

shape_tests = [
    # b-shape: long the low
    ('B_SHAPE', 'IBL', 'LONG', lambda t: t['poc_shape'] == 'B_SHAPE' and t['level'] == 'IBL' and t['accept_30min']),
    ('B_SHAPE', 'IBH', 'SHORT', lambda t: t['poc_shape'] == 'B_SHAPE' and t['level'] == 'IBH' and t['accept_30min']),
    # P-shape: short the high sweep, long the low
    ('P_SHAPE', 'IBL', 'LONG', lambda t: t['poc_shape'] == 'P_SHAPE' and t['level'] == 'IBL' and t['accept_30min']),
    ('P_SHAPE', 'IBH', 'SHORT', lambda t: t['poc_shape'] == 'P_SHAPE' and t['level'] == 'IBH' and t['accept_30min']),
    # D-shape: both edges
    ('D_SHAPE', 'IBL', 'LONG', lambda t: t['poc_shape'] == 'D_SHAPE' and t['level'] == 'IBL' and t['accept_30min']),
    ('D_SHAPE', 'IBH', 'SHORT', lambda t: t['poc_shape'] == 'D_SHAPE' and t['level'] == 'IBH' and t['accept_30min']),
]

for shape, level, direction, filt in shape_tests:
    result = backtest_level_fade(stdf, filt, 0.10, 'ib_mid', f'{shape} {level}')
    n_str = print_result(result)
    print(f"| {shape} | {level} | {direction} | {n_str} |")


print("\n\nPart 9 complete.")


# ============================================================================
# PART 10: DPOC MIGRATION AS REAL-TIME ENTRY FILTER
# ============================================================================
print("\n\n" + "=" * 70)
print("PART 10: DPOC MIGRATION — REAL-TIME ENTRY FILTER FOR EDGE FADES")
print("=" * 70)

print("""
DPOC = Developing Point of Control. Unlike EOD day type (known only after close),
DPOC migration direction and velocity are observable in real-time.

Key question: Does DPOC migration direction at the time of an IB edge touch
improve the win rate of the fade entry?

Hypothesis:
  - LONG at IBL + DPOC migrating DOWN = price building value lower → fade may fail
  - LONG at IBL + DPOC migrating UP or FLAT = value holding → fade should work
  - SHORT at IBH + DPOC migrating UP = value building higher → fade may fail
  - SHORT at IBH + DPOC migrating DOWN or FLAT = value holding → fade should work

We compute DPOC migration at each touch bar using 30-min volume-weighted POC slices,
the same methodology as the rockit-framework DPOC migration module.
""")

# ── 10.0: Compute DPOC migration at each touch point ─────────────────────
print("Computing DPOC migration at each touch point...\n")

from collections import Counter as DPOCCounter

def compute_dpoc_at_bar(bars, touch_bar_idx, ib_range):
    """
    Compute DPOC migration metrics at a specific bar index within a session.
    Uses 30-min slices of post-IB bars up to the touch bar.
    Returns dict with direction, net_migration, velocity, regime.
    """
    # Only use bars from IB close (bar 60) to the touch bar
    post_ib = bars.iloc[60:touch_bar_idx + 1]
    if len(post_ib) < 10:  # Need at least 10 bars post-IB
        return None

    # ATR-aware thresholds (use IB range as proxy)
    atr_proxy = ib_range / 6  # rough ATR from IB range
    cluster_threshold = max(15, atr_proxy * 0.6)
    velocity_strong = max(25, atr_proxy * 0.8)
    velocity_low = max(8, atr_proxy * 0.3)
    exhausted_threshold = max(50, atr_proxy * 3)

    # Divide post-IB bars into 30-bar (30-min) slices
    completed_dpocs = []
    for slice_start in range(0, len(post_ib), 30):
        slice_end = min(slice_start + 30, len(post_ib))
        slice_bars = post_ib.iloc[slice_start:slice_end]

        if len(slice_bars) < 4:
            continue  # developing slice, skip

        # Volume-weighted POC for this slice
        price_vol = DPOCCounter()
        for _, sbar in slice_bars.iterrows():
            vwap_price = sbar.get('vwap', sbar['close'])
            if pd.isna(vwap_price):
                vwap_price = sbar['close']
            price = round(vwap_price * 4) / 4.0  # round to 0.25 tick
            vol = int(sbar.get('volume', 1))
            price_vol[price] += vol

        if price_vol:
            poc = max(price_vol, key=price_vol.get)
        else:
            poc = slice_bars['close'].mean()
        completed_dpocs.append(round(poc, 2))

    if len(completed_dpocs) < 2:
        # Not enough slices for migration analysis
        if len(completed_dpocs) == 1:
            return {
                'dpoc_direction': 'insufficient',
                'dpoc_net_migration': 0,
                'dpoc_velocity': 0,
                'dpoc_regime': 'insufficient',
                'dpoc_stabilizing': False,
            }
        return None

    first_dpoc = completed_dpocs[0]
    current_dpoc = completed_dpocs[-1]
    net_migration = current_dpoc - first_dpoc

    # Direction
    if net_migration > cluster_threshold / 2:
        direction = 'up'
    elif net_migration < -cluster_threshold / 2:
        direction = 'down'
    else:
        direction = 'flat'

    # Velocity (avg pts per 30-min slice, last 3 slices)
    deltas = [completed_dpocs[i+1] - completed_dpocs[i] for i in range(len(completed_dpocs) - 1)]
    recent_deltas = deltas[-min(3, len(deltas)):]
    avg_velocity = np.mean(recent_deltas) if recent_deltas else 0.0
    abs_velocity = abs(avg_velocity)

    # Retention
    peak = max(completed_dpocs) if direction != "down" else min(completed_dpocs)
    excursion = abs(peak - first_dpoc)
    if excursion > 0:
        retain_pct = abs(net_migration) / excursion * 100
    else:
        retain_pct = 100.0

    # Stabilization (last 4 slices cluster)
    recent_cluster = completed_dpocs[-min(4, len(completed_dpocs)):]
    cluster_range = max(recent_cluster) - min(recent_cluster) if len(recent_cluster) > 1 else 0
    is_stabilizing = cluster_range <= cluster_threshold and abs_velocity <= velocity_low

    # Acceleration
    decelerating = False
    if len(recent_deltas) >= 2:
        accel = recent_deltas[-1] - recent_deltas[-2]
        if abs(accel) >= 8 and np.sign(avg_velocity) == -np.sign(accel):
            decelerating = True

    # Regime classification (simplified from rockit module)
    if abs_velocity >= velocity_strong and not decelerating and retain_pct >= 70:
        regime = 'trending_strong'
    elif abs_velocity >= velocity_low and (decelerating or retain_pct < 60):
        regime = 'trending_fading'
    elif is_stabilizing:
        regime = 'stabilizing'
    else:
        regime = 'transitional'

    return {
        'dpoc_direction': direction,
        'dpoc_net_migration': round(net_migration, 2),
        'dpoc_velocity': round(avg_velocity, 2),
        'dpoc_abs_velocity': round(abs_velocity, 2),
        'dpoc_regime': regime,
        'dpoc_stabilizing': is_stabilizing,
        'dpoc_retain_pct': round(retain_pct, 1),
    }


# Enrich the sweep touches dataframe with DPOC data at each touch
dpoc_enriched = []
dpoc_computed = 0
dpoc_failed = 0

for _, touch in stdf.iterrows():
    session_date = touch['session']
    touch_bar = touch['touch_bar']
    ib_range = touch['ib_range']

    bars = get_session_bars(session_date)
    dpoc = compute_dpoc_at_bar(bars, int(touch_bar), ib_range)

    row = touch.to_dict()
    if dpoc is not None:
        row.update(dpoc)
        dpoc_computed += 1
    else:
        row['dpoc_direction'] = 'unknown'
        row['dpoc_net_migration'] = 0
        row['dpoc_velocity'] = 0
        row['dpoc_abs_velocity'] = 0
        row['dpoc_regime'] = 'unknown'
        row['dpoc_stabilizing'] = False
        row['dpoc_retain_pct'] = 0
        dpoc_failed += 1

    # Derived: is DPOC direction aligned with the fade?
    # LONG fade + DPOC up/flat = aligned (value not migrating against the long)
    # SHORT fade + DPOC down/flat = aligned
    if row['fade_dir'] == 'LONG':
        row['dpoc_aligned'] = row['dpoc_direction'] in ('up', 'flat')
        row['dpoc_contra'] = row['dpoc_direction'] == 'down'
    else:
        row['dpoc_aligned'] = row['dpoc_direction'] in ('down', 'flat')
        row['dpoc_contra'] = row['dpoc_direction'] == 'up'

    dpoc_enriched.append(row)

dpoc_df = pd.DataFrame(dpoc_enriched)
print(f"DPOC computed for {dpoc_computed} / {len(stdf)} touches ({dpoc_failed} insufficient data)")

# Distribution of DPOC direction at touch points
print(f"\nDPOC direction distribution at edge touches:")
for d in ['up', 'down', 'flat', 'insufficient', 'unknown']:
    n = (dpoc_df['dpoc_direction'] == d).sum()
    pct = n / len(dpoc_df) * 100 if len(dpoc_df) > 0 else 0
    print(f"  {d}: {n} ({pct:.0f}%)")

print(f"\nDPOC regime distribution at edge touches:")
for r in ['trending_strong', 'trending_fading', 'stabilizing', 'transitional', 'insufficient', 'unknown']:
    n = (dpoc_df['dpoc_regime'] == r).sum()
    pct = n / len(dpoc_df) * 100 if len(dpoc_df) > 0 else 0
    print(f"  {r}: {n} ({pct:.0f}%)")


# ── 10.1: DPOC Direction as Filter — All Day Types ──────────────────────
print("\n\n### 10.1: DPOC Direction as Entry Filter — All Day Types\n")
print("Does filtering by DPOC direction at entry time improve the edge?\n")

# Use only touches with valid DPOC and 30min acceptance
valid_dpoc = dpoc_df[(dpoc_df['dpoc_direction'] != 'unknown') & (dpoc_df['dpoc_direction'] != 'insufficient')]

print(f"| Filter | Level | Dir | N | T/Mo | WR | PF | $/Mo | Risk |")
print(f"|--------|-------|-----|---|------|-----|-----|------|------|")

dpoc_filter_tests = [
    # Baseline: no DPOC filter
    ('No DPOC filter', 'IBL', 'LONG',
     lambda t: t['level'] == 'IBL' and t['accept_30min']),
    # DPOC aligned (up/flat for LONG)
    ('DPOC aligned', 'IBL', 'LONG',
     lambda t: t['level'] == 'IBL' and t['accept_30min'] and t['dpoc_aligned']),
    # DPOC contra (down for LONG — should be worse)
    ('DPOC contra', 'IBL', 'LONG',
     lambda t: t['level'] == 'IBL' and t['accept_30min'] and t['dpoc_contra']),
    # DPOC flat only (value stationary = classic balance)
    ('DPOC flat only', 'IBL', 'LONG',
     lambda t: t['level'] == 'IBL' and t['accept_30min'] and t['dpoc_direction'] == 'flat'),
    # DPOC up only (bullish migration + long = strong alignment)
    ('DPOC up only', 'IBL', 'LONG',
     lambda t: t['level'] == 'IBL' and t['accept_30min'] and t['dpoc_direction'] == 'up'),

    # SHORT at IBH
    ('No DPOC filter', 'IBH', 'SHORT',
     lambda t: t['level'] == 'IBH' and t['accept_30min']),
    ('DPOC aligned', 'IBH', 'SHORT',
     lambda t: t['level'] == 'IBH' and t['accept_30min'] and t['dpoc_aligned']),
    ('DPOC contra', 'IBH', 'SHORT',
     lambda t: t['level'] == 'IBH' and t['accept_30min'] and t['dpoc_contra']),
    ('DPOC flat only', 'IBH', 'SHORT',
     lambda t: t['level'] == 'IBH' and t['accept_30min'] and t['dpoc_direction'] == 'flat'),
    ('DPOC down only', 'IBH', 'SHORT',
     lambda t: t['level'] == 'IBH' and t['accept_30min'] and t['dpoc_direction'] == 'down'),
]

for name, level, direction, filt in dpoc_filter_tests:
    result = backtest_level_fade(valid_dpoc, filt, 0.10, 'ib_mid', f'{name} {level}')
    n_str = print_result(result)
    print(f"| {name} | {level} | {direction} | {n_str} |")


# ── 10.2: DPOC Regime as Filter — Best Setups per Regime ──────────────────
print("\n\n### 10.2: DPOC Regime as Entry Filter — By Regime\n")
print("Which DPOC regime produces the best edge fade results?\n")
print("  trending_strong  = DPOC moving fast with retention (continuation)")
print("  trending_fading   = DPOC moving but decelerating (potential reversal)")
print("  stabilizing      = DPOC cluster forming (balance, mean reversion)")
print("  transitional     = unclear regime\n")

print(f"| Regime | Level | Dir | N | T/Mo | WR | PF | $/Mo | Risk |")
print(f"|--------|-------|-----|---|------|-----|-----|------|------|")

for regime in ['trending_strong', 'trending_fading', 'stabilizing', 'transitional']:
    regime_sub = valid_dpoc[valid_dpoc['dpoc_regime'] == regime]
    if len(regime_sub) < 3:
        print(f"| {regime} | IBL | LONG | 0 | — | — | — | — | — |")
        continue

    for level, direction, filt_extra in [
        ('IBL', 'LONG', lambda t: t['level'] == 'IBL' and t['accept_30min']),
        ('IBH', 'SHORT', lambda t: t['level'] == 'IBH' and t['accept_30min']),
    ]:
        result = backtest_level_fade(regime_sub, filt_extra, 0.10, 'ib_mid', f'{regime} {level}')
        n_str = print_result(result)
        print(f"| {regime} | {level} | {direction} | {n_str} |")


# ── 10.3: DPOC × Day Type — Does DPOC Add Edge Within Day Types? ─────────
print("\n\n### 10.3: DPOC × Day Type — Incremental Edge Over Day Type Alone\n")
print("Within each day type, does DPOC alignment add edge over baseline?\n")

print(f"| Day Type | DPOC Filter | Level | Dir | N | WR | PF | $/Mo | Delta vs Base |")
print(f"|----------|-------------|-------|-----|---|-----|-----|------|--------------|")

for dt in ['P_DAY', 'B_DAY', 'NEUTRAL']:
    dt_sub = valid_dpoc[valid_dpoc['eod_day_type'] == dt]
    if len(dt_sub) < 5:
        continue

    for level, direction in [('IBL', 'LONG'), ('IBH', 'SHORT')]:
        # Baseline: day type + level + 30min acceptance, no DPOC filter
        base_filt = lambda t, l=level: t['level'] == l and t['accept_30min']
        base_r = backtest_level_fade(dt_sub, base_filt, 0.10, 'ib_mid', 'base')
        base_wr = base_r['wr'] if base_r and base_r['n'] >= 3 else None
        base_monthly = base_r['monthly'] if base_r and base_r['n'] >= 3 else None

        for dpoc_label, dpoc_filt in [
            ('None (baseline)', lambda t, l=level: t['level'] == l and t['accept_30min']),
            ('DPOC aligned', lambda t, l=level: t['level'] == l and t['accept_30min'] and t['dpoc_aligned']),
            ('DPOC contra', lambda t, l=level: t['level'] == l and t['accept_30min'] and t['dpoc_contra']),
            ('Stabilizing', lambda t, l=level: t['level'] == l and t['accept_30min'] and t['dpoc_stabilizing']),
        ]:
            r = backtest_level_fade(dt_sub, dpoc_filt, 0.10, 'ib_mid', dpoc_label)
            if r and r['n'] >= 2:
                delta_wr = f"{r['wr'] - base_wr:+.0f}pp" if base_wr is not None else "—"
                print(f"| {dt} | {dpoc_label} | {level} | {direction} | {r['n']} | {r['wr']:.0f}% | {r['pf']:.2f} | ${r['monthly']:.0f} | {delta_wr} |")
            else:
                print(f"| {dt} | {dpoc_label} | {level} | {direction} | 0 | — | — | — | — |")


# ── 10.4: DPOC Velocity Buckets — Low/Med/High Velocity at Entry ─────────
print("\n\n### 10.4: DPOC Velocity Buckets — Does Speed of Migration Matter?\n")
print("Bucketing DPOC absolute velocity at entry time into low/med/high.\n")

# Define velocity buckets based on data distribution
vel_data = valid_dpoc[valid_dpoc['dpoc_abs_velocity'] > 0]['dpoc_abs_velocity']
if len(vel_data) > 10:
    vel_33 = vel_data.quantile(0.33)
    vel_67 = vel_data.quantile(0.67)
else:
    vel_33 = 10
    vel_67 = 25

print(f"Velocity terciles: Low < {vel_33:.0f} pts/30min, Med {vel_33:.0f}-{vel_67:.0f}, High > {vel_67:.0f}\n")

print(f"| Velocity | Level | Dir | N | T/Mo | WR | PF | $/Mo |")
print(f"|----------|-------|-----|---|------|-----|-----|------|")

for vel_label, vel_filt in [
    ('Low', lambda t: t['dpoc_abs_velocity'] <= vel_33),
    ('Medium', lambda t: t['dpoc_abs_velocity'] > vel_33 and t['dpoc_abs_velocity'] <= vel_67),
    ('High', lambda t: t['dpoc_abs_velocity'] > vel_67),
    ('Zero/Flat', lambda t: t['dpoc_abs_velocity'] == 0),
]:
    for level, direction in [('IBL', 'LONG'), ('IBH', 'SHORT')]:
        combined_filt = lambda t, vf=vel_filt, l=level: t['level'] == l and t['accept_30min'] and vf(t)
        result = backtest_level_fade(valid_dpoc, combined_filt, 0.10, 'ib_mid', f'{vel_label} {level}')
        if result and result['n'] >= 2:
            print(f"| {vel_label} | {level} | {direction} | {result['n']} | {result['trades_mo']:.1f} | {result['wr']:.0f}% | {result['pf']:.2f} | ${result['monthly']:.0f} |")
        else:
            print(f"| {vel_label} | {level} | {direction} | 0 | — | — | — | — |")


# ── 10.5: DPOC Net Migration Magnitude — Strong vs Weak ──────────────────
print("\n\n### 10.5: DPOC Net Migration Magnitude at Entry\n")
print("How much has the DPOC moved from first post-IB slice to entry time?\n")

print(f"| Migration | Level | Dir | N | WR | PF | $/Mo |")
print(f"|-----------|-------|-----|---|-----|-----|------|")

mig_data = valid_dpoc['dpoc_net_migration'].abs()
mig_25 = mig_data.quantile(0.25) if len(mig_data) > 10 else 5
mig_75 = mig_data.quantile(0.75) if len(mig_data) > 10 else 30

for mig_label, mig_filt in [
    (f'Small (<{mig_25:.0f}pts)', lambda t: abs(t['dpoc_net_migration']) < mig_25),
    (f'Medium ({mig_25:.0f}-{mig_75:.0f}pts)', lambda t: abs(t['dpoc_net_migration']) >= mig_25 and abs(t['dpoc_net_migration']) <= mig_75),
    (f'Large (>{mig_75:.0f}pts)', lambda t: abs(t['dpoc_net_migration']) > mig_75),
]:
    for level, direction in [('IBL', 'LONG'), ('IBH', 'SHORT')]:
        combined_filt = lambda t, mf=mig_filt, l=level: t['level'] == l and t['accept_30min'] and mf(t)
        result = backtest_level_fade(valid_dpoc, combined_filt, 0.10, 'ib_mid', f'{mig_label} {level}')
        if result and result['n'] >= 2:
            print(f"| {mig_label} | {level} | {direction} | {result['n']} | {result['wr']:.0f}% | {result['pf']:.2f} | ${result['monthly']:.0f} |")
        else:
            print(f"| {mig_label} | {level} | {direction} | 0 | — | — | — | — |")


# ── 10.6: Best Combined Filter — DPOC + POC Shape + Day Type ─────────────
print("\n\n### 10.6: Best Combined Filter — DPOC + POC Shape + Acceptance\n")
print("Combining the best real-time filters: POC shape + DPOC regime + 30min acceptance\n")

print(f"| Combo | N | T/Mo | WR | PF | $/Mo | Risk |")
print(f"|-------|---|------|-----|-----|------|------|")

combo_tests = [
    # B-shape + DPOC aligned LONG at IBL (strongest from previous parts)
    ('B-shape + DPOC aligned + IBL LONG',
     lambda t: t['poc_shape'] == 'B_SHAPE' and t['level'] == 'IBL' and t['accept_30min'] and t['dpoc_aligned']),
    ('B-shape + DPOC stabilizing + IBL LONG',
     lambda t: t['poc_shape'] == 'B_SHAPE' and t['level'] == 'IBL' and t['accept_30min'] and t['dpoc_stabilizing']),
    ('B-shape + DPOC NOT contra + IBL LONG',
     lambda t: t['poc_shape'] == 'B_SHAPE' and t['level'] == 'IBL' and t['accept_30min'] and not t['dpoc_contra']),

    # P-shape + DPOC aligned SHORT at IBH
    ('P-shape + DPOC aligned + IBH SHORT',
     lambda t: t['poc_shape'] == 'P_SHAPE' and t['level'] == 'IBH' and t['accept_30min'] and t['dpoc_aligned']),
    ('P-shape + DPOC stabilizing + IBH SHORT',
     lambda t: t['poc_shape'] == 'P_SHAPE' and t['level'] == 'IBH' and t['accept_30min'] and t['dpoc_stabilizing']),

    # D-shape (neutral POC) + stabilizing DPOC = classic balance
    ('D-shape + stabilizing + IBL LONG',
     lambda t: t['poc_shape'] == 'D_SHAPE' and t['level'] == 'IBL' and t['accept_30min'] and t['dpoc_stabilizing']),
    ('D-shape + stabilizing + IBH SHORT',
     lambda t: t['poc_shape'] == 'D_SHAPE' and t['level'] == 'IBH' and t['accept_30min'] and t['dpoc_stabilizing']),

    # Any shape + DPOC aligned (universal filter)
    ('Any shape + DPOC aligned + IBL LONG',
     lambda t: t['level'] == 'IBL' and t['accept_30min'] and t['dpoc_aligned']),
    ('Any shape + DPOC aligned + IBH SHORT',
     lambda t: t['level'] == 'IBH' and t['accept_30min'] and t['dpoc_aligned']),

    # Strongest from Part 8: B-day LONG, now with DPOC
    ('B-day + DPOC aligned + IBL LONG',
     lambda t: t['eod_day_type'] == 'B_DAY' and t['level'] == 'IBL' and t['accept_30min'] and t['dpoc_aligned']),
    ('B-day + DPOC aligned + sweep + IBL LONG',
     lambda t: t['eod_day_type'] == 'B_DAY' and t['level'] == 'IBL' and t['accept_30min'] and t['is_sweep'] and t['dpoc_aligned']),
]

for name, filt in combo_tests:
    result = backtest_level_fade(valid_dpoc, filt, 0.10, 'ib_mid', name)
    n_str = print_result(result)
    print(f"| {name} | {n_str} |")


# ── 10.7: DPOC Retention % — High Retention vs Fading ────────────────────
print("\n\n### 10.7: DPOC Retention % — High vs Low Retention\n")
print("Retention = how much of peak DPOC migration is preserved at entry time.")
print("High retention = strong directional conviction. Low = fading/exhausting.\n")

print(f"| Retention | Level | Dir | N | WR | PF | $/Mo |")
print(f"|-----------|-------|-----|---|-----|-----|------|")

for ret_label, ret_filt in [
    ('High (>70%)', lambda t: t['dpoc_retain_pct'] > 70),
    ('Medium (40-70%)', lambda t: t['dpoc_retain_pct'] >= 40 and t['dpoc_retain_pct'] <= 70),
    ('Low (<40%)', lambda t: t['dpoc_retain_pct'] < 40),
]:
    for level, direction in [('IBL', 'LONG'), ('IBH', 'SHORT')]:
        combined_filt = lambda t, rf=ret_filt, l=level: t['level'] == l and t['accept_30min'] and rf(t)
        result = backtest_level_fade(valid_dpoc, combined_filt, 0.10, 'ib_mid', f'{ret_label} {level}')
        if result and result['n'] >= 2:
            print(f"| {ret_label} | {level} | {direction} | {result['n']} | {result['wr']:.0f}% | {result['pf']:.2f} | ${result['monthly']:.0f} |")
        else:
            print(f"| {ret_label} | {level} | {direction} | 0 | — | — | — | — |")


# ── 10.8: Summary — Does DPOC Migration Improve the Edge? ────────────────
print("\n\n### 10.8: SUMMARY — Does DPOC Migration Add Edge?\n")
print("Comparing best baseline (no DPOC) vs best DPOC-filtered configs:\n")

print(f"| Comparison | N | WR | PF | $/Mo | Verdict |")
print(f"|------------|---|-----|-----|------|---------|")

# Baseline: IBL LONG + 30min acceptance (no DPOC)
base_long = backtest_level_fade(valid_dpoc,
    lambda t: t['level'] == 'IBL' and t['accept_30min'], 0.10, 'ib_mid', 'base')
# DPOC aligned
aligned_long = backtest_level_fade(valid_dpoc,
    lambda t: t['level'] == 'IBL' and t['accept_30min'] and t['dpoc_aligned'], 0.10, 'ib_mid', 'aligned')
# DPOC contra
contra_long = backtest_level_fade(valid_dpoc,
    lambda t: t['level'] == 'IBL' and t['accept_30min'] and t['dpoc_contra'], 0.10, 'ib_mid', 'contra')

for label, r in [('IBL LONG baseline', base_long), ('IBL LONG + DPOC aligned', aligned_long), ('IBL LONG + DPOC contra', contra_long)]:
    if r and r['n'] >= 2:
        verdict = ""
        if base_long and r['wr'] > base_long['wr'] + 3:
            verdict = "BETTER"
        elif base_long and r['wr'] < base_long['wr'] - 3:
            verdict = "WORSE"
        else:
            verdict = "SIMILAR"
        print(f"| {label} | {r['n']} | {r['wr']:.0f}% | {r['pf']:.2f} | ${r['monthly']:.0f} | {verdict} |")
    else:
        print(f"| {label} | 0 | — | — | — | — |")

# Same for SHORT
base_short = backtest_level_fade(valid_dpoc,
    lambda t: t['level'] == 'IBH' and t['accept_30min'], 0.10, 'ib_mid', 'base')
aligned_short = backtest_level_fade(valid_dpoc,
    lambda t: t['level'] == 'IBH' and t['accept_30min'] and t['dpoc_aligned'], 0.10, 'ib_mid', 'aligned')
contra_short = backtest_level_fade(valid_dpoc,
    lambda t: t['level'] == 'IBH' and t['accept_30min'] and t['dpoc_contra'], 0.10, 'ib_mid', 'contra')

for label, r in [('IBH SHORT baseline', base_short), ('IBH SHORT + DPOC aligned', aligned_short), ('IBH SHORT + DPOC contra', contra_short)]:
    if r and r['n'] >= 2:
        verdict = ""
        if base_short and r['wr'] > base_short['wr'] + 3:
            verdict = "BETTER"
        elif base_short and r['wr'] < base_short['wr'] - 3:
            verdict = "WORSE"
        else:
            verdict = "SIMILAR"
        print(f"| {label} | {r['n']} | {r['wr']:.0f}% | {r['pf']:.2f} | ${r['monthly']:.0f} | {verdict} |")
    else:
        print(f"| {label} | 0 | — | — | — | — |")


print("\n\nPart 10 complete.")
print("\n\nStudy complete.")
