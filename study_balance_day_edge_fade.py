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

print("Study complete.")
