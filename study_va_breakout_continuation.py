"""
VA Breakout Continuation & Range Formation Study
==================================================

Two core questions from auction theory:

PART A — PM CONTINUATION
  When price breaks above/below the composite VA (weekly) and holds
  through midday (noon), does it statistically continue in PM or stall?
  Sub-questions:
    1. Breakout holds midday but NO IBH/IBL extension by noon — neutral or continuation?
    2. Does delta/CVD at midday predict PM direction?
    3. Time-segmented returns: AM vs Lunch vs PM for breakout days

PART B — POST-BREAKOUT RANGE FORMATION
  After a bracket breakout, does price establish a NEW value area?
    1. How many sessions until a new composite VA forms that doesn't overlap the old one?
    2. How wide is the new range vs old range?
    3. Does the breakout level (old VAH/VAL) become new support/resistance?
    4. Multi-day continuation: does the breakout direction persist for D+1, D+2, D+3?

PART C — BRACKET BREAKOUT ENHANCEMENTS (from prior study follow-ups)
    1. Double breakout filter (bracket + single-day VA alignment)
    2. Clearance filter optimization
    3. Order flow confirmation at retracement (delta/CVD at VA edge touch)
    4. Time-of-day filter for retracement entries

Data: 259 RTH sessions, NQ/MNQ 1-min volumetric bars
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import time, timedelta
from typing import Dict, List, Optional, Tuple

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.instruments import get_instrument
from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from indicators.value_area import (
    compute_session_value_areas,
    compute_composite_value_area,
    ValueAreaLevels,
)

# ── Config ──────────────────────────────────────────────────────
INSTRUMENT = get_instrument('MNQ')
TICK_VALUE = INSTRUMENT.tick_value / INSTRUMENT.tick_size  # $/pt
SLIPPAGE_PTS = 0.50
COMMISSION = 1.24
CONTRACTS = 5
LOOKBACK = 5  # 5-day = "weekly" composite VA
MIN_COMP_VA_WIDTH = 30.0
IB_END_TIME = time(10, 30)
RTH_START = time(9, 30)
NOON = time(12, 0)
LUNCH_START = time(11, 30)
LUNCH_END = time(13, 30)
PM_START = time(13, 0)
PM_END = time(15, 30)
EOD = time(16, 0)

# ── Load Data ───────────────────────────────────────────────────
print("=" * 70)
print("VA BREAKOUT CONTINUATION & RANGE FORMATION STUDY")
print("=" * 70)

print("\nLoading data...")
df_raw = load_csv('NQ')
df_rth = filter_rth(df_raw)
df_rth = compute_all_features(df_rth)

# Full data for composite VA computation
df_full = df_raw.copy()
if 'session_date' not in df_full.columns:
    df_full['session_date'] = df_full['timestamp'].dt.date
    evening_mask = df_full['timestamp'].dt.time >= time(18, 0)
    df_full.loc[evening_mask, 'session_date'] = (
        pd.to_datetime(df_full.loc[evening_mask, 'session_date']) + pd.Timedelta(days=1)
    ).dt.date

sessions = sorted(df_rth['session_date'].unique())
n_sessions = len(sessions)
months = n_sessions / 22

print(f"Sessions: {n_sessions}, Months: {months:.1f}")

# Compute value areas
print("Computing single-day VAs...")
single_va = compute_session_value_areas(df_full, tick_size=0.25, va_percent=0.70)

print(f"Computing {LOOKBACK}-day composite VAs...")
comp_vas = {}
for session_date in sessions:
    cva = compute_composite_value_area(
        df_full, session_date, lookback_days=LOOKBACK, tick_size=0.25, va_percent=0.70
    )
    if cva is not None:
        comp_vas[str(session_date)] = cva


def get_session_bars(df_rth, session_date):
    """Get all RTH bars for a session."""
    return df_rth[df_rth['session_date'] == session_date].copy()


def pnl_pts_to_dollars(pts, contracts=CONTRACTS):
    """Convert point P&L to dollars after slippage + commission."""
    gross = pts * TICK_VALUE * contracts
    costs = (SLIPPAGE_PTS * TICK_VALUE * 2 + COMMISSION * 2) * contracts
    return gross - costs


# ============================================================================
# PART A: PM CONTINUATION AFTER VA BREAKOUT
# ============================================================================
print("\n\n" + "=" * 70)
print("PART A: PM CONTINUATION AFTER VA BREAKOUT HOLDS THROUGH MIDDAY")
print("=" * 70)

# For each session: identify breakout direction, track price at key times,
# measure AM/Lunch/PM returns

breakout_sessions = []

for session_date in sessions:
    key = str(session_date)
    if key not in comp_vas:
        continue
    cva = comp_vas[key]
    if cva.va_width < MIN_COMP_VA_WIDTH:
        continue

    bars = get_session_bars(df_rth, session_date)
    if len(bars) < 60:
        continue

    open_price = bars['open'].iloc[0]

    # Classify open vs composite VA
    if open_price > cva.vah:
        direction = 'LONG'
        breakout_level = cva.vah
    elif open_price < cva.val:
        direction = 'SHORT'
        breakout_level = cva.val
    else:
        continue  # Inside VA, not a breakout

    # IB levels
    ib_bars = bars[bars['time'] <= IB_END_TIME]
    if len(ib_bars) == 0:
        continue
    ib_high = ib_bars['high'].max()
    ib_low = ib_bars['low'].min()
    ib_range = ib_high - ib_low

    # Key time checkpoints
    am_bars = bars[(bars['time'] >= RTH_START) & (bars['time'] < NOON)]
    noon_bars = bars[(bars['time'] >= time(11, 55)) & (bars['time'] <= time(12, 5))]
    lunch_bars = bars[(bars['time'] >= LUNCH_START) & (bars['time'] < PM_START)]
    pm_bars = bars[bars['time'] >= PM_START]

    if len(noon_bars) == 0 or len(pm_bars) == 0:
        continue

    # Prices at key times
    noon_price = noon_bars['close'].iloc[-1] if len(noon_bars) > 0 else None
    noon_high = am_bars['high'].max()
    noon_low = am_bars['low'].min()

    pm_high = pm_bars['high'].max()
    pm_low = pm_bars['low'].min()
    pm_close = pm_bars['close'].iloc[-1]
    session_close = bars['close'].iloc[-1]
    session_high = bars['high'].max()
    session_low = bars['low'].min()

    # Check: does breakout hold at noon?
    if direction == 'LONG':
        breakout_holds_noon = noon_price > cva.vah
        ib_extended_by_noon = noon_high > ib_high
        # AM return = open to noon
        am_return = noon_price - open_price
        # PM return = noon to close
        pm_return = session_close - noon_price
        # Did it make new highs in PM?
        pm_new_high = pm_high > noon_high
        pm_extension = pm_high - noon_high if pm_new_high else 0
        # Clearance from VA edge
        clearance = open_price - cva.vah
    else:  # SHORT
        breakout_holds_noon = noon_price < cva.val
        ib_extended_by_noon = noon_low < ib_low
        am_return = open_price - noon_price  # positive = in breakout dir
        pm_return = noon_price - session_close
        pm_new_high = pm_low < noon_low  # "new high" = new low for shorts
        pm_extension = noon_low - pm_low if pm_new_high else 0
        clearance = cva.val - open_price

    # Delta/CVD at noon
    noon_delta = am_bars['delta'].sum() if 'delta' in am_bars.columns else 0
    noon_cvd = am_bars['cumulative_delta'].iloc[-1] if 'cumulative_delta' in am_bars.columns and len(am_bars) > 0 else 0

    # PM delta
    pm_delta = pm_bars['delta'].sum() if 'delta' in pm_bars.columns else 0

    # Lunch return
    lunch_return_val = 0
    if len(lunch_bars) > 0:
        lunch_open = lunch_bars['open'].iloc[0]
        lunch_close_price = lunch_bars['close'].iloc[-1]
        if direction == 'LONG':
            lunch_return_val = lunch_close_price - lunch_open
        else:
            lunch_return_val = lunch_open - lunch_close_price

    breakout_sessions.append({
        'session': session_date,
        'direction': direction,
        'open': open_price,
        'comp_vah': cva.vah,
        'comp_val': cva.val,
        'comp_poc': cva.poc,
        'comp_width': cva.va_width,
        'clearance': clearance,
        'ib_high': ib_high,
        'ib_low': ib_low,
        'ib_range': ib_range,
        'noon_price': noon_price,
        'breakout_holds_noon': breakout_holds_noon,
        'ib_extended_by_noon': ib_extended_by_noon,
        'am_return': am_return,
        'pm_return': pm_return,
        'lunch_return': lunch_return_val,
        'pm_new_extreme': pm_new_high,
        'pm_extension': pm_extension,
        'session_close': session_close,
        'session_high': session_high,
        'session_low': session_low,
        'noon_delta': noon_delta,
        'pm_delta': pm_delta,
        'breakout_held_eod': (session_close > cva.vah) if direction == 'LONG' else (session_close < cva.val),
    })

bdf = pd.DataFrame(breakout_sessions)
print(f"\nTotal breakout sessions (5-day): {len(bdf)}")
print(f"  LONG: {(bdf['direction'] == 'LONG').sum()}")
print(f"  SHORT: {(bdf['direction'] == 'SHORT').sum()}")

# ── A1: Overall PM continuation after breakout holds noon ──
print("\n### A1: Does the Breakout Continue in the PM?\n")

holds_noon = bdf[bdf['breakout_holds_noon'] == True]
fails_noon = bdf[bdf['breakout_holds_noon'] == False]

print(f"| Noon Status | N | PM Continues (same dir) | PM Reverses | PM New Extreme | Avg PM Return |")
print(f"|-------------|---|-------------------------|-------------|----------------|---------------|")

for label, sub in [('Holds at noon', holds_noon), ('Fails at noon', fails_noon)]:
    if len(sub) == 0:
        continue
    pm_continues = (sub['pm_return'] > 0).sum()
    pm_reverses = (sub['pm_return'] <= 0).sum()
    pm_new = sub['pm_new_extreme'].sum()
    avg_pm = sub['pm_return'].mean()
    print(f"| {label} | {len(sub)} | {pm_continues} ({pm_continues/len(sub)*100:.0f}%) | {pm_reverses} ({pm_reverses/len(sub)*100:.0f}%) | {pm_new} ({pm_new/len(sub)*100:.0f}%) | {avg_pm:+.0f} pts |")

# ── A2: KEY SCENARIO — Breakout holds noon BUT no IB extension ──
print("\n### A2: The Key Scenario — Breakout Holds Noon BUT No IB Extension\n")
print("(Price broke out of weekly VA but is stuck in the IB by noon)")
print("This is the 'quiet conviction' scenario — is it neutral or does it resolve in PM?\n")

holds_no_ext = bdf[(bdf['breakout_holds_noon'] == True) & (bdf['ib_extended_by_noon'] == False)]
holds_with_ext = bdf[(bdf['breakout_holds_noon'] == True) & (bdf['ib_extended_by_noon'] == True)]

print(f"| Scenario | N | PM Continues | PM New Extreme | Avg PM Ret | BO Held EOD |")
print(f"|----------|---|--------------|----------------|------------|-------------|")

for label, sub in [
    ('Holds noon + IB extended', holds_with_ext),
    ('**Holds noon + NO IB ext**', holds_no_ext),
]:
    if len(sub) == 0:
        print(f"| {label} | 0 | — | — | — | — |")
        continue
    pm_cont = (sub['pm_return'] > 0).sum()
    pm_new = sub['pm_new_extreme'].sum()
    avg_pm = sub['pm_return'].mean()
    held_eod = sub['breakout_held_eod'].sum()
    print(f"| {label} | {len(sub)} | {pm_cont} ({pm_cont/len(sub)*100:.0f}%) | {pm_new} ({pm_new/len(sub)*100:.0f}%) | {avg_pm:+.0f} pts | {held_eod} ({held_eod/len(sub)*100:.0f}%) |")

# LONG vs SHORT breakdown of the key scenario
for d in ['LONG', 'SHORT']:
    sub = holds_no_ext[holds_no_ext['direction'] == d]
    if len(sub) < 3:
        continue
    pm_cont = (sub['pm_return'] > 0).sum()
    pm_new = sub['pm_new_extreme'].sum()
    avg_pm = sub['pm_return'].mean()
    med_pm = sub['pm_return'].median()
    held_eod = sub['breakout_held_eod'].sum()
    print(f"\n  {d} (holds noon, no IB ext, N={len(sub)}):")
    print(f"    PM continues: {pm_cont}/{len(sub)} ({pm_cont/len(sub)*100:.0f}%)")
    print(f"    PM new extreme: {pm_new}/{len(sub)} ({pm_new/len(sub)*100:.0f}%)")
    print(f"    Avg PM return: {avg_pm:+.1f} pts | Median: {med_pm:+.1f} pts")
    print(f"    Breakout held EOD: {held_eod}/{len(sub)} ({held_eod/len(sub)*100:.0f}%)")
    print(f"    Avg PM extension (when it happens): {sub[sub['pm_new_extreme']]['pm_extension'].mean():.0f} pts" if pm_new > 0 else "")

# ── A3: Time-Segmented Returns ──
print("\n### A3: Time-Segmented Returns — AM vs Lunch vs PM\n")

holds = bdf[bdf['breakout_holds_noon'] == True]
if len(holds) > 0:
    print(f"| Period | Avg Return | Median | Positive % | Contribution to Day |")
    print(f"|--------|-----------|--------|-----------|---------------------|")
    total_day = holds['am_return'] + holds['lunch_return'] + holds['pm_return']
    for period, col in [('AM (open-noon)', 'am_return'), ('Lunch (11:30-13:00)', 'lunch_return'), ('PM (13:00-close)', 'pm_return')]:
        avg = holds[col].mean()
        med = holds[col].median()
        pos = (holds[col] > 0).sum() / len(holds) * 100
        contrib = avg / total_day.mean() * 100 if total_day.mean() != 0 else 0
        print(f"| {period} | {avg:+.0f} pts | {med:+.0f} pts | {pos:.0f}% | {contrib:.0f}% |")

# ── A4: Delta/CVD at Noon as PM Predictor ──
print("\n### A4: Does Noon Delta Predict PM Direction?\n")

if len(holds) > 0:
    # Split by noon delta direction
    positive_delta = holds[holds['noon_delta'] > 0]
    negative_delta = holds[holds['noon_delta'] <= 0]

    print(f"| AM Delta Direction | N | PM Continues | Avg PM Ret | PM New Extreme |")
    print(f"|--------------------|---|-------------|-----------|----------------|")
    for label, sub in [('AM delta positive', positive_delta), ('AM delta negative', negative_delta)]:
        if len(sub) == 0:
            continue
        pm_cont = (sub['pm_return'] > 0).sum()
        avg_pm = sub['pm_return'].mean()
        pm_new = sub['pm_new_extreme'].sum()
        print(f"| {label} | {len(sub)} | {pm_cont} ({pm_cont/len(sub)*100:.0f}%) | {avg_pm:+.0f} pts | {pm_new} ({pm_new/len(sub)*100:.0f}%) |")

    # Split by whether AM delta aligns with breakout direction
    aligned = holds.copy()
    aligned['delta_aligned'] = False
    long_mask = aligned['direction'] == 'LONG'
    short_mask = aligned['direction'] == 'SHORT'
    aligned.loc[long_mask & (aligned['noon_delta'] > 0), 'delta_aligned'] = True
    aligned.loc[short_mask & (aligned['noon_delta'] < 0), 'delta_aligned'] = True

    delta_match = aligned[aligned['delta_aligned'] == True]
    delta_conflict = aligned[aligned['delta_aligned'] == False]

    print(f"\n| Delta vs Breakout | N | PM Continues | Avg PM Ret | PM New Extreme |")
    print(f"|-------------------|---|-------------|-----------|----------------|")
    for label, sub in [('Delta ALIGNS with breakout', delta_match), ('Delta CONFLICTS with breakout', delta_conflict)]:
        if len(sub) == 0:
            continue
        pm_cont = (sub['pm_return'] > 0).sum()
        avg_pm = sub['pm_return'].mean()
        pm_new = sub['pm_new_extreme'].sum()
        print(f"| {label} | {len(sub)} | {pm_cont} ({pm_cont/len(sub)*100:.0f}%) | {avg_pm:+.0f} pts | {pm_new} ({pm_new/len(sub)*100:.0f}%) |")


# ============================================================================
# PART B: POST-BREAKOUT RANGE FORMATION
# ============================================================================
print("\n\n" + "=" * 70)
print("PART B: POST-BREAKOUT RANGE FORMATION — NEW VALUE AFTER BREAKOUT")
print("=" * 70)

# For each breakout session, look at the NEXT 1-5 sessions:
# - Does the next day's VA overlap the old composite VA?
# - How quickly does a new non-overlapping VA form?
# - Does the old VA boundary (VAH/VAL) act as support/resistance?

print("\n### B1: Multi-Day Continuation After Breakout\n")

continuation_data = []
session_idx_map = {s: i for i, s in enumerate(sessions)}

for row in breakout_sessions:
    session_date = row['session']
    direction = row['direction']
    comp_vah = row['comp_vah']
    comp_val = row['comp_val']
    comp_poc = row['comp_poc']

    idx = session_idx_map.get(session_date)
    if idx is None:
        continue

    entry = {
        'session': session_date,
        'direction': direction,
        'comp_vah': comp_vah,
        'comp_val': comp_val,
        'breakout_held_d0': row['breakout_held_eod'],
        'clearance': row['clearance'],
    }

    # Look forward D+1 through D+5
    for fwd in range(1, 6):
        fwd_idx = idx + fwd
        if fwd_idx >= len(sessions):
            entry[f'd{fwd}_continues'] = None
            entry[f'd{fwd}_touches_old_va'] = None
            entry[f'd{fwd}_inside_old_va'] = None
            entry[f'd{fwd}_return'] = None
            continue

        fwd_session = sessions[fwd_idx]
        fwd_bars = get_session_bars(df_rth, fwd_session)
        if len(fwd_bars) < 30:
            entry[f'd{fwd}_continues'] = None
            entry[f'd{fwd}_touches_old_va'] = None
            entry[f'd{fwd}_inside_old_va'] = None
            entry[f'd{fwd}_return'] = None
            continue

        fwd_open = fwd_bars['open'].iloc[0]
        fwd_close = fwd_bars['close'].iloc[-1]
        fwd_high = fwd_bars['high'].max()
        fwd_low = fwd_bars['low'].min()

        if direction == 'LONG':
            entry[f'd{fwd}_continues'] = fwd_close > comp_vah
            entry[f'd{fwd}_touches_old_va'] = fwd_low <= comp_vah
            entry[f'd{fwd}_inside_old_va'] = fwd_close <= comp_vah and fwd_close >= comp_val
            entry[f'd{fwd}_return'] = fwd_close - comp_vah  # positive = still above
        else:
            entry[f'd{fwd}_continues'] = fwd_close < comp_val
            entry[f'd{fwd}_touches_old_va'] = fwd_high >= comp_val
            entry[f'd{fwd}_inside_old_va'] = fwd_close >= comp_val and fwd_close <= comp_vah
            entry[f'd{fwd}_return'] = comp_val - fwd_close

    continuation_data.append(entry)

cdf = pd.DataFrame(continuation_data)

# Multi-day continuation rates
print(f"| Day | Still Beyond VA | Touches Old VA | Returns Inside VA | Avg Return vs VA Edge |")
print(f"|-----|-----------------|----------------|-------------------|----------------------|")

for fwd in range(0, 6):
    col_cont = f'd{fwd}_continues' if fwd > 0 else 'breakout_held_d0'
    col_touch = f'd{fwd}_touches_old_va' if fwd > 0 else None
    col_inside = f'd{fwd}_inside_old_va' if fwd > 0 else None
    col_ret = f'd{fwd}_return' if fwd > 0 else None

    valid = cdf[cdf[col_cont].notna()] if fwd > 0 else cdf
    if len(valid) == 0:
        continue

    cont = valid[col_cont].sum()
    cont_pct = cont / len(valid) * 100

    if col_touch and col_touch in valid.columns:
        touch = valid[col_touch].sum()
        inside = valid[col_inside].sum() if col_inside else 0
        avg_ret = valid[col_ret].mean() if col_ret else 0
        label = f"D+{fwd}" if fwd > 0 else "D+0"
        print(f"| {label} | {cont} ({cont_pct:.0f}%) | {touch} ({touch/len(valid)*100:.0f}%) | {inside} ({inside/len(valid)*100:.0f}%) | {avg_ret:+.0f} pts |")
    else:
        label = "D+0"
        print(f"| {label} | {cont} ({cont_pct:.0f}%) | — | — | — |")

# ── B2: New VA Formation ──
print("\n### B2: New Composite VA Formation After Breakout\n")
print("Does the new 5-day composite VA separate from the old one?\n")

new_va_data = []
for row in breakout_sessions:
    session_date = row['session']
    direction = row['direction']
    old_vah = row['comp_vah']
    old_val = row['comp_val']
    old_width = row['comp_width']
    idx = session_idx_map.get(session_date)
    if idx is None:
        continue

    # Check composite VA at D+1 through D+5
    entry = {
        'session': session_date,
        'direction': direction,
        'old_vah': old_vah,
        'old_val': old_val,
        'old_width': old_width,
    }

    for fwd in range(1, 6):
        fwd_idx = idx + fwd
        if fwd_idx >= len(sessions):
            entry[f'd{fwd}_new_vah'] = None
            entry[f'd{fwd}_new_val'] = None
            entry[f'd{fwd}_overlaps'] = None
            entry[f'd{fwd}_shift'] = None
            continue

        fwd_session = sessions[fwd_idx]
        fwd_key = str(fwd_session)

        # The new composite VA is computed at the forward session (uses prior 5 days)
        new_cva = compute_composite_value_area(
            df_full, fwd_session, lookback_days=LOOKBACK, tick_size=0.25, va_percent=0.70
        )
        if new_cva is None:
            entry[f'd{fwd}_new_vah'] = None
            entry[f'd{fwd}_new_val'] = None
            entry[f'd{fwd}_overlaps'] = None
            entry[f'd{fwd}_shift'] = None
            continue

        entry[f'd{fwd}_new_vah'] = new_cva.vah
        entry[f'd{fwd}_new_val'] = new_cva.val

        # Does new VA overlap old VA?
        overlaps = new_cva.val <= old_vah and new_cva.vah >= old_val
        entry[f'd{fwd}_overlaps'] = overlaps

        # How much has the VA shifted?
        if direction == 'LONG':
            entry[f'd{fwd}_shift'] = new_cva.poc - (old_vah + old_val) / 2
        else:
            entry[f'd{fwd}_shift'] = (old_vah + old_val) / 2 - new_cva.poc

        entry[f'd{fwd}_new_width'] = new_cva.va_width

    new_va_data.append(entry)

ndf = pd.DataFrame(new_va_data)

print(f"| Day After BO | New VA Overlaps Old | VA Shifted in BO Dir | Avg VA Shift | Avg New Width |")
print(f"|-------------|--------------------|-----------------------|-------------|---------------|")

for fwd in range(1, 6):
    col_overlap = f'd{fwd}_overlaps'
    col_shift = f'd{fwd}_shift'
    col_width = f'd{fwd}_new_width'

    valid = ndf[ndf[col_overlap].notna()]
    if len(valid) == 0:
        continue

    overlaps = valid[col_overlap].sum()
    shifted_right = (valid[col_shift] > 0).sum()
    avg_shift = valid[col_shift].mean()
    avg_width = valid[col_width].mean() if col_width in valid.columns else 0

    print(f"| D+{fwd} | {overlaps}/{len(valid)} ({overlaps/len(valid)*100:.0f}%) | {shifted_right}/{len(valid)} ({shifted_right/len(valid)*100:.0f}%) | {avg_shift:+.0f} pts | {avg_width:.0f} pts |")


# ── B3: Old VA Boundary as Support/Resistance ──
print("\n### B3: Does the Breakout Level Become Support/Resistance?\n")
print("After a LONG breakout (price > old VAH), does old VAH become support?")
print("After a SHORT breakout (price < old VAL), does old VAL become resistance?\n")

sr_data = []
for row in breakout_sessions:
    if not row['breakout_held_eod']:
        continue  # Only look at held breakouts

    session_date = row['session']
    direction = row['direction']
    idx = session_idx_map.get(session_date)
    if idx is None:
        continue

    for fwd in range(1, 6):
        fwd_idx = idx + fwd
        if fwd_idx >= len(sessions):
            continue
        fwd_session = sessions[fwd_idx]
        fwd_bars = get_session_bars(df_rth, fwd_session)
        if len(fwd_bars) < 30:
            continue

        fwd_low = fwd_bars['low'].min()
        fwd_high = fwd_bars['high'].max()
        fwd_close = fwd_bars['close'].iloc[-1]

        if direction == 'LONG':
            # Old VAH should act as support
            tested = fwd_low <= row['comp_vah'] + 5  # within 5 pts
            held_as_support = tested and fwd_close > row['comp_vah']
            broke_through = fwd_low < row['comp_val']  # fell back into VA
        else:
            tested = fwd_high >= row['comp_val'] - 5
            held_as_support = tested and fwd_close < row['comp_val']
            broke_through = fwd_high > row['comp_vah']

        sr_data.append({
            'direction': direction,
            'day_offset': fwd,
            'tested': tested,
            'held': held_as_support,
            'broke_through': broke_through,
        })

srdf = pd.DataFrame(sr_data) if sr_data else pd.DataFrame()

if len(srdf) > 0:
    print(f"| Day | Tested Level | Held as S/R | Broke Through |")
    print(f"|-----|-------------|-------------|---------------|")
    for fwd in range(1, 6):
        sub = srdf[srdf['day_offset'] == fwd]
        if len(sub) == 0:
            continue
        tested = sub['tested'].sum()
        held = sub['held'].sum()
        broke = sub['broke_through'].sum()
        held_when_tested = held / tested * 100 if tested > 0 else 0
        print(f"| D+{fwd} | {tested}/{len(sub)} ({tested/len(sub)*100:.0f}%) | {held}/{tested} ({held_when_tested:.0f}%) | {broke}/{len(sub)} ({broke/len(sub)*100:.0f}%) |")


# ============================================================================
# PART C: BRACKET BREAKOUT ENHANCEMENTS
# ============================================================================
print("\n\n" + "=" * 70)
print("PART C: BRACKET BREAKOUT ENHANCEMENTS — FILTER OPTIMIZATION")
print("=" * 70)

# ── C1: Double Breakout Filter ──
print("\n### C1: Double Breakout Filter (Bracket + Single-Day VA)\n")
print("Do sessions where BOTH the composite VA and single-day VA are breached produce")
print("stronger continuation and better retracement entries?\n")

# For each breakout session, check if it's also outside the prior single-day VA
double_bo_data = []
for row in breakout_sessions:
    session_date = row['session']
    direction = row['direction']
    idx = session_idx_map.get(session_date)
    if idx is None or idx == 0:
        continue

    prior_session = sessions[idx - 1]
    prior_key = str(prior_session)
    if prior_key not in single_va:
        continue

    sva = single_va[prior_key]
    open_price = row['open']

    if direction == 'LONG':
        also_outside_single = open_price > sva.vah
    else:
        also_outside_single = open_price < sva.val

    bars = get_session_bars(df_rth, session_date)
    if len(bars) < 60:
        continue

    # Check for retracement to composite VA edge
    comp_vah = row['comp_vah']
    comp_val = row['comp_val']
    retrace_entry_price = None
    retrace_bar_time = None

    for _, bar in bars.iterrows():
        if bar['time'] >= PM_START:
            break
        if direction == 'LONG' and bar['low'] <= comp_vah + 5:
            retrace_entry_price = comp_vah + 5
            retrace_bar_time = bar['time']
            break
        elif direction == 'SHORT' and bar['high'] >= comp_val - 5:
            retrace_entry_price = comp_val - 5
            retrace_bar_time = bar['time']
            break

    # If retracement entry happened, compute outcome
    retrace_won = None
    retrace_pnl = None
    if retrace_entry_price is not None:
        stop = comp_vah - 10 if direction == 'LONG' else comp_val + 10
        risk = abs(retrace_entry_price - stop)
        if risk <= 0:
            risk = 15
        target = retrace_entry_price + 2 * risk if direction == 'LONG' else retrace_entry_price - 2 * risk

        # Walk bars after entry
        entry_found = False
        for _, bar in bars.iterrows():
            if bar['time'] < retrace_bar_time:
                continue
            if not entry_found:
                entry_found = True
                continue

            if direction == 'LONG':
                if bar['low'] <= stop:
                    retrace_pnl = stop - retrace_entry_price
                    retrace_won = False
                    break
                if bar['high'] >= target:
                    retrace_pnl = target - retrace_entry_price
                    retrace_won = True
                    break
            else:
                if bar['high'] >= stop:
                    retrace_pnl = retrace_entry_price - stop
                    retrace_won = False
                    break
                if bar['low'] <= target:
                    retrace_pnl = retrace_entry_price - target
                    retrace_won = True
                    break

        if retrace_pnl is None:
            # EOD exit
            eod_price = bars['close'].iloc[-1]
            if direction == 'LONG':
                retrace_pnl = eod_price - retrace_entry_price
            else:
                retrace_pnl = retrace_entry_price - eod_price
            retrace_won = retrace_pnl > 0

    # Order flow at retracement point
    retrace_delta = None
    if retrace_bar_time is not None:
        retrace_area_bars = bars[(bars['time'] >= retrace_bar_time)]
        if len(retrace_area_bars) >= 3:
            retrace_delta = retrace_area_bars.head(3)['delta'].sum()

    double_bo_data.append({
        'session': session_date,
        'direction': direction,
        'double_breakout': also_outside_single,
        'clearance': row['clearance'],
        'breakout_held_eod': row['breakout_held_eod'],
        'pm_return': row['pm_return'],
        'retraced': retrace_entry_price is not None,
        'retrace_won': retrace_won,
        'retrace_pnl': retrace_pnl,
        'retrace_time': retrace_bar_time,
        'retrace_delta': retrace_delta,
    })

dbdf = pd.DataFrame(double_bo_data)

double = dbdf[dbdf['double_breakout'] == True]
single_only = dbdf[dbdf['double_breakout'] == False]

print(f"| Filter | N | BO Held EOD | Retraced | Retrace WR | Avg Retrace P&L |")
print(f"|--------|---|------------|---------|-----------|----------------|")

for label, sub in [('Double breakout (both)', double), ('Bracket only (not single)', single_only)]:
    if len(sub) == 0:
        print(f"| {label} | 0 | — | — | — | — |")
        continue
    held = sub['breakout_held_eod'].sum()
    retraced = sub['retraced'].sum()
    retrace_trades = sub[sub['retrace_won'].notna()]
    wr = retrace_trades['retrace_won'].mean() * 100 if len(retrace_trades) > 0 else 0
    avg_pnl = retrace_trades['retrace_pnl'].mean() if len(retrace_trades) > 0 else 0
    print(f"| {label} | {len(sub)} | {held} ({held/len(sub)*100:.0f}%) | {retraced} ({retraced/len(sub)*100:.0f}%) | {wr:.0f}% | {avg_pnl:+.1f} pts |")

# Direction breakdown
for d in ['LONG', 'SHORT']:
    d_double = double[double['direction'] == d]
    if len(d_double) < 3:
        continue
    retrace_trades = d_double[d_double['retrace_won'].notna()]
    if len(retrace_trades) == 0:
        continue
    wr = retrace_trades['retrace_won'].mean() * 100
    avg_pnl = retrace_trades['retrace_pnl'].mean()
    held = d_double['breakout_held_eod'].sum()
    print(f"\n  {d} double breakout (N={len(d_double)}):")
    print(f"    Breakout held EOD: {held}/{len(d_double)} ({held/len(d_double)*100:.0f}%)")
    print(f"    Retrace entry WR: {wr:.0f}% (N={len(retrace_trades)})")
    print(f"    Avg retrace P&L: {avg_pnl:+.1f} pts")


# ── C2: Clearance Filter Optimization ──
print("\n\n### C2: Clearance Filter — Optimal Threshold\n")

# Test different clearance thresholds
clearance_thresholds = [50, 100, 150, 200, 250, 300]

print(f"| Max Clearance | N (retraced) | Retrace WR | Avg P&L | BO Held EOD |")
print(f"|---------------|-------------|-----------|---------|-------------|")

for thresh in clearance_thresholds:
    sub = dbdf[(dbdf['clearance'] <= thresh) & (dbdf['retrace_won'].notna())]
    if len(sub) == 0:
        print(f"| ≤ {thresh} pts | 0 | — | — | — |")
        continue
    wr = sub['retrace_won'].mean() * 100
    avg_pnl = sub['retrace_pnl'].mean()
    all_in_clearance = dbdf[dbdf['clearance'] <= thresh]
    held = all_in_clearance['breakout_held_eod'].sum()
    print(f"| ≤ {thresh} pts | {len(sub)} | {wr:.0f}% | {avg_pnl:+.1f} pts | {held}/{len(all_in_clearance)} ({held/len(all_in_clearance)*100:.0f}%) |")


# ── C3: Order Flow at Retracement ──
print("\n\n### C3: Order Flow Confirmation at Retracement Touch\n")
print("When price retraces to the composite VA edge, does the delta at that touch")
print("predict whether the retracement bounce succeeds?\n")

retrace_trades_all = dbdf[dbdf['retrace_won'].notna() & dbdf['retrace_delta'].notna()].copy()
if len(retrace_trades_all) > 0:
    # Split by whether delta aligns with breakout direction at the retracement
    retrace_trades_all['delta_aligned'] = False
    long_mask = retrace_trades_all['direction'] == 'LONG'
    short_mask = retrace_trades_all['direction'] == 'SHORT'
    retrace_trades_all.loc[long_mask & (retrace_trades_all['retrace_delta'] > 0), 'delta_aligned'] = True
    retrace_trades_all.loc[short_mask & (retrace_trades_all['retrace_delta'] < 0), 'delta_aligned'] = True

    aligned = retrace_trades_all[retrace_trades_all['delta_aligned'] == True]
    conflicting = retrace_trades_all[retrace_trades_all['delta_aligned'] == False]

    print(f"| Delta at Retracement | N | Retrace WR | Avg P&L |")
    print(f"|---------------------|---|-----------|---------|")
    for label, sub in [('Delta ALIGNS (confirms bounce)', aligned), ('Delta CONFLICTS (warns of failure)', conflicting)]:
        if len(sub) == 0:
            print(f"| {label} | 0 | — | — |")
            continue
        wr = sub['retrace_won'].mean() * 100
        avg_pnl = sub['retrace_pnl'].mean()
        print(f"| {label} | {len(sub)} | {wr:.0f}% | {avg_pnl:+.1f} pts |")
else:
    print("Insufficient data for order flow at retracement analysis.")


# ── C4: Time-of-Day Filter for Retracement Entries ──
print("\n\n### C4: Time-of-Day Filter for Retracement Entries\n")
print("66% of retracements happen during IB. Should we only enter during IB?\n")

retrace_with_time = dbdf[dbdf['retrace_time'].notna() & dbdf['retrace_won'].notna()].copy()
if len(retrace_with_time) > 0:
    # Convert time to comparable format
    retrace_with_time['during_ib'] = retrace_with_time['retrace_time'].apply(lambda t: t <= IB_END_TIME)
    retrace_with_time['before_noon'] = retrace_with_time['retrace_time'].apply(lambda t: t <= NOON)

    print(f"| Time Filter | N | Retrace WR | Avg P&L |")
    print(f"|------------|---|-----------|---------|")

    for label, mask in [
        ('IB only (≤10:30)', retrace_with_time['during_ib']),
        ('Before noon (≤12:00)', retrace_with_time['before_noon']),
        ('After IB (>10:30)', ~retrace_with_time['during_ib']),
        ('All times', pd.Series(True, index=retrace_with_time.index)),
    ]:
        sub = retrace_with_time[mask]
        if len(sub) == 0:
            print(f"| {label} | 0 | — | — |")
            continue
        wr = sub['retrace_won'].mean() * 100
        avg_pnl = sub['retrace_pnl'].mean()
        print(f"| {label} | {len(sub)} | {wr:.0f}% | {avg_pnl:+.1f} pts |")


# ============================================================================
# PART D: COMBINED OPTIMAL FILTER — PUTTING IT ALL TOGETHER
# ============================================================================
print("\n\n" + "=" * 70)
print("PART D: COMBINED OPTIMAL FILTER")
print("=" * 70)

print("\nApplying best filters simultaneously to find the highest-edge setup:\n")

# Best combo: double breakout + clearance filter + IB retracement + delta aligned
best_setup = dbdf.copy()
best_setup = best_setup[best_setup['retrace_won'].notna()]

filters_applied = []

# Filter 1: Double breakout
f1 = best_setup[best_setup['double_breakout'] == True]
# Filter 2: Clearance ≤ 200 pts
f2 = best_setup[best_setup['clearance'] <= 200]
# Combined 1+2
f12 = best_setup[(best_setup['double_breakout'] == True) & (best_setup['clearance'] <= 200)]

# Add time filter (before noon)
if 'retrace_time' in best_setup.columns:
    best_setup['before_noon'] = best_setup['retrace_time'].apply(
        lambda t: t <= NOON if pd.notna(t) else False
    )
    f123 = best_setup[
        (best_setup['double_breakout'] == True) &
        (best_setup['clearance'] <= 200) &
        (best_setup['before_noon'] == True)
    ]
else:
    f123 = f12

# Add delta alignment
if 'retrace_delta' in best_setup.columns:
    best_setup['delta_aligned'] = False
    long_mask = best_setup['direction'] == 'LONG'
    short_mask = best_setup['direction'] == 'SHORT'
    best_setup.loc[long_mask & (best_setup['retrace_delta'] > 0), 'delta_aligned'] = True
    best_setup.loc[short_mask & (best_setup['retrace_delta'] < 0), 'delta_aligned'] = True

    f1234 = best_setup[
        (best_setup['double_breakout'] == True) &
        (best_setup['clearance'] <= 200) &
        (best_setup['before_noon'] == True) &
        (best_setup['delta_aligned'] == True)
    ]
else:
    f1234 = f123

print(f"| Filter Stack | N | WR | Avg P&L | $/Mo (5 MNQ) |")
print(f"|-------------|---|------|---------|-------------|")

for label, sub in [
    ('Baseline (all retracements)', best_setup),
    ('+ Double breakout', f1),
    ('+ Clearance ≤ 200', f2),
    ('+ Double BO + Clearance ≤ 200', f12),
    ('+ Double BO + Clear + Before noon', f123),
    ('+ Double BO + Clear + Noon + Delta', f1234),
]:
    if len(sub) == 0:
        print(f"| {label} | 0 | — | — | — |")
        continue
    wr = sub['retrace_won'].mean() * 100
    avg_pnl = sub['retrace_pnl'].mean()
    monthly_trades = len(sub) / months if months > 0 else 0
    monthly_dollars = pnl_pts_to_dollars(avg_pnl) * monthly_trades if monthly_trades > 0 else 0
    print(f"| {label} | {len(sub)} | {wr:.0f}% | {avg_pnl:+.1f} pts | ${monthly_dollars:.0f} ({monthly_trades:.1f}/mo) |")


# ============================================================================
# SUMMARY
# ============================================================================
print("\n\n" + "=" * 70)
print("SUMMARY — KEY FINDINGS")
print("=" * 70)

print("""
PART A — PM CONTINUATION:
  1. When the VA breakout HOLDS at noon, does PM continue?
  2. The "quiet conviction" scenario (holds noon, no IB extension) — neutral or continuation?
  3. Time-segmented returns show where the money is made
  4. Delta alignment at noon as PM predictor

PART B — POST-BREAKOUT RANGE FORMATION:
  1. Multi-day continuation: how many days does the breakout persist?
  2. New composite VA formation: overlap analysis D+1 through D+5
  3. Old VA boundary as support/resistance after breakout

PART C — BRACKET BREAKOUT ENHANCEMENTS:
  1. Double breakout filter (bracket + single-day) effect on retracement WR
  2. Clearance threshold optimization
  3. Order flow (delta) confirmation at retracement touch
  4. Time-of-day filter for retracement entries

PART D — COMBINED OPTIMAL FILTER:
  Stacking all filters to find the highest-edge retracement entry

See detailed tables above for findings.
""")

print("Study complete.")
