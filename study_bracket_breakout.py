"""
Bracket Breakout Study — Multi-Day Composite VA Breakout Analysis
==================================================================

Dalton's "bracket" = multi-session balance area where value overlaps.
When price breaks OUT of a multi-day composite Value Area during the
opening range, we study:

1. How often does this happen? (frequency per lookback)
2. Does the breakout retrace into the composite VA before resuming?
3. What's the max excursion in the breakout direction?
4. Is the retracement entry tradeable (better than chasing)?
5. How does this interact with single-day 80P/20P?
6. Optimal lookback: 3, 5, 7, 10 days?

This is NOT the 20P (single-day VA breakout). This is a higher-timeframe
structural breakout from a multi-session consolidation.
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
LOOKBACKS = [3, 5, 7, 10]
MIN_COMP_VA_WIDTH = 30.0  # Composite VA must be meaningful
ENTRY_CUTOFF = time(13, 0)
EOD_EXIT = time(15, 30)
IB_END_TIME = time(10, 30)
RTH_START = time(9, 30)

# ── Load Data ───────────────────────────────────────────────────
print("=" * 70)
print("BRACKET BREAKOUT STUDY — Multi-Day Composite VA Breakout")
print("=" * 70)

print("\nLoading data...")
df_raw = load_csv('NQ')
df_rth = filter_rth(df_raw)
df_rth = compute_all_features(df_rth)

# Full data for ETH VA computation
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

# Compute single-day VAs (for 80P/20P interaction analysis)
print("Computing single-day VAs...")
single_va = compute_session_value_areas(df_full, tick_size=0.25, va_percent=0.70)


def pnl_pts_to_dollars(pts, contracts=CONTRACTS):
    """Convert point P&L to dollars after slippage + commission."""
    gross = pts * TICK_VALUE * contracts
    costs = (SLIPPAGE_PTS * TICK_VALUE * 2 + COMMISSION * 2) * contracts
    return gross - costs


# ============================================================================
# SECTION 1: COMPOSITE VA COMPUTATION & BREAKOUT IDENTIFICATION
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 1: BREAKOUT FREQUENCY BY LOOKBACK")
print("=" * 70)


def get_session_open(df_rth, session_date):
    """Get the opening price for a session."""
    sdf = df_rth[df_rth['session_date'] == session_date]
    if len(sdf) == 0:
        return None
    return sdf['open'].iloc[0]


def get_session_bars(df_rth, session_date):
    """Get all RTH bars for a session."""
    return df_rth[df_rth['session_date'] == session_date].copy()


def get_ib_bars(session_bars):
    """Get IB period bars (9:30-10:30)."""
    if len(session_bars) == 0:
        return session_bars
    mask = session_bars['time'] <= IB_END_TIME
    return session_bars[mask]


# Compute composite VAs for all lookbacks
comp_vas = {}  # {lookback: {session_date_str: ValueAreaLevels}}
for lb in LOOKBACKS:
    print(f"  Computing {lb}-day composite VAs...")
    comp_vas[lb] = {}
    for session_date in sessions:
        cva = compute_composite_value_area(
            df_full, session_date, lookback_days=lb, tick_size=0.25, va_percent=0.70
        )
        if cva is not None:
            comp_vas[lb][str(session_date)] = cva


# Identify breakouts for each lookback
def classify_open_vs_comp_va(open_price, cva):
    """Classify open relative to composite VA."""
    if open_price > cva.vah:
        return 'ABOVE'
    elif open_price < cva.val:
        return 'BELOW'
    else:
        return 'INSIDE'


print("\n### Breakout Frequency\n")
print(f"| Lookback | Valid Sessions | Open Above CVA | Open Below CVA | Open Inside | Breakout % | Per Month |")
print(f"|----------|---------------|----------------|----------------|-------------|------------|-----------|")

breakout_data = {}  # {lookback: [session_info_dicts]}

for lb in LOOKBACKS:
    above = 0
    below = 0
    inside = 0
    valid = 0
    breakouts = []

    for session_date in sessions:
        key = str(session_date)
        if key not in comp_vas[lb]:
            continue
        cva = comp_vas[lb][key]
        if cva.va_width < MIN_COMP_VA_WIDTH:
            continue

        open_price = get_session_open(df_rth, session_date)
        if open_price is None:
            continue

        valid += 1
        classification = classify_open_vs_comp_va(open_price, cva)

        if classification == 'ABOVE':
            above += 1
            breakouts.append({
                'session': session_date,
                'direction': 'LONG',
                'open': open_price,
                'comp_vah': cva.vah,
                'comp_val': cva.val,
                'comp_poc': cva.poc,
                'comp_width': cva.va_width,
                'gap_pts': open_price - cva.vah,
            })
        elif classification == 'BELOW':
            below += 1
            breakouts.append({
                'session': session_date,
                'direction': 'SHORT',
                'open': open_price,
                'comp_vah': cva.vah,
                'comp_val': cva.val,
                'comp_poc': cva.poc,
                'comp_width': cva.va_width,
                'gap_pts': cva.val - open_price,
            })
        else:
            inside += 1

    total_bo = above + below
    pct = total_bo / valid * 100 if valid > 0 else 0
    per_mo = total_bo / months if months > 0 else 0
    print(f"| {lb}-day | {valid} | {above} ({above/valid*100:.0f}%) | {below} ({below/valid*100:.0f}%) | {inside} ({inside/valid*100:.0f}%) | {pct:.0f}% | {per_mo:.1f} |")

    breakout_data[lb] = breakouts

# ============================================================================
# SECTION 2: WHAT HAPPENS AFTER BREAKOUT? RETRACEMENT ANALYSIS
# ============================================================================
print("\n\n" + "=" * 70)
print("SECTION 2: POST-BREAKOUT BEHAVIOR — RETRACEMENT ANALYSIS")
print("=" * 70)


def analyze_session_behavior(df_rth, session_info, comp_va_dict, lookback):
    """
    For a breakout session, track:
    - Did price retrace back INTO the composite VA?
    - How deep was the retracement?
    - Max excursion in breakout direction
    - Max adverse excursion
    - Did it eventually continue in breakout direction?
    - What happened relative to single-day VA too?
    """
    session_date = session_info['session']
    direction = session_info['direction']
    open_price = session_info['open']
    comp_vah = session_info['comp_vah']
    comp_val = session_info['comp_val']
    comp_poc = session_info['comp_poc']

    session_bars = get_session_bars(df_rth, session_date)
    if len(session_bars) < 60:
        return None

    ib_bars = get_ib_bars(session_bars)
    post_ib_bars = session_bars[session_bars['time'] > IB_END_TIME]
    all_bars = session_bars

    ib_high = ib_bars['high'].max() if len(ib_bars) > 0 else open_price
    ib_low = ib_bars['low'].min() if len(ib_bars) > 0 else open_price
    ib_range = ib_high - ib_low

    # Session extremes
    session_high = all_bars['high'].max()
    session_low = all_bars['low'].min()
    session_close = all_bars['close'].iloc[-1]

    result = {
        'session': session_date,
        'direction': direction,
        'open': open_price,
        'comp_vah': comp_vah,
        'comp_val': comp_val,
        'comp_poc': comp_poc,
        'comp_width': session_info['comp_width'],
        'gap_pts': session_info['gap_pts'],
        'ib_high': ib_high,
        'ib_low': ib_low,
        'ib_range': ib_range,
        'session_high': session_high,
        'session_low': session_low,
        'session_close': session_close,
    }

    if direction == 'LONG':
        # Breakout above composite VAH
        # Max favorable excursion = how far above VAH price went
        result['max_excursion'] = session_high - comp_vah
        # Deepest retracement = how far below open price dipped
        result['max_retrace_from_open'] = open_price - session_low
        # Did it retrace into composite VA?
        result['retrace_into_va'] = session_low <= comp_vah
        # Did it retrace to POC?
        result['retrace_to_poc'] = session_low <= comp_poc
        # Did it retrace to VAL?
        result['retrace_to_val'] = session_low <= comp_val
        # Retracement depth: how far below VAH did it go (if it retested)
        if session_low <= comp_vah:
            result['retrace_depth_into_va'] = comp_vah - session_low
        else:
            result['retrace_depth_into_va'] = 0
        # Did it close above VAH (breakout held)?
        result['breakout_held'] = session_close > comp_vah
        # Did it close above open (trending day)?
        result['closed_above_open'] = session_close > open_price
        # Extension from VAH
        result['close_vs_vah'] = session_close - comp_vah

        # Track bar-by-bar for retracement timing
        first_retrace_bar = None
        for idx, bar in session_bars.iterrows():
            if bar['low'] <= comp_vah and first_retrace_bar is None:
                first_retrace_bar = idx
                break
        result['retraced_during_ib'] = False
        if first_retrace_bar is not None:
            retrace_time = session_bars.loc[first_retrace_bar, 'time']
            result['retraced_during_ib'] = retrace_time <= IB_END_TIME
            result['retrace_time'] = retrace_time
        else:
            result['retrace_time'] = None

    else:  # SHORT
        result['max_excursion'] = comp_val - session_low
        result['max_retrace_from_open'] = session_high - open_price
        result['retrace_into_va'] = session_high >= comp_val
        result['retrace_to_poc'] = session_high >= comp_poc
        result['retrace_to_val'] = session_high >= comp_vah
        if session_high >= comp_val:
            result['retrace_depth_into_va'] = session_high - comp_val
        else:
            result['retrace_depth_into_va'] = 0
        result['breakout_held'] = session_close < comp_val
        result['closed_above_open'] = session_close < open_price  # "above" = in breakout dir
        result['close_vs_vah'] = comp_val - session_close

        first_retrace_bar = None
        for idx, bar in session_bars.iterrows():
            if bar['high'] >= comp_val and first_retrace_bar is None:
                first_retrace_bar = idx
                break
        result['retraced_during_ib'] = False
        if first_retrace_bar is not None:
            retrace_time = session_bars.loc[first_retrace_bar, 'time']
            result['retraced_during_ib'] = retrace_time <= IB_END_TIME
            result['retrace_time'] = retrace_time
        else:
            result['retrace_time'] = None

    # Single-day VA interaction
    prior_session_idx = sessions.index(session_date) - 1 if session_date in sessions else -1
    if prior_session_idx >= 0:
        prior_key = str(sessions[prior_session_idx])
        if prior_key in single_va:
            sva = single_va[prior_key]
            result['single_vah'] = sva.vah
            result['single_val'] = sva.val
            if direction == 'LONG':
                result['open_vs_single_va'] = 'ABOVE' if open_price > sva.vah else ('BELOW' if open_price < sva.val else 'INSIDE')
            else:
                result['open_vs_single_va'] = 'BELOW' if open_price < sva.val else ('ABOVE' if open_price > sva.vah else 'INSIDE')
        else:
            result['single_vah'] = None
            result['single_val'] = None
            result['open_vs_single_va'] = None
    else:
        result['single_vah'] = None
        result['single_val'] = None
        result['open_vs_single_va'] = None

    return result


# Analyze each lookback
for lb in LOOKBACKS:
    breakouts = breakout_data[lb]
    if not breakouts:
        continue

    print(f"\n### {lb}-Day Composite VA Breakout ({len(breakouts)} sessions)")

    results = []
    for bo in breakouts:
        r = analyze_session_behavior(df_rth, bo, comp_vas[lb], lb)
        if r is not None:
            results.append(r)

    if not results:
        print("  No valid results.")
        continue

    rdf = pd.DataFrame(results)

    # --- Retracement rates ---
    retrace_rate = rdf['retrace_into_va'].mean() * 100
    retrace_poc = rdf['retrace_to_poc'].mean() * 100
    bo_held = rdf['breakout_held'].mean() * 100
    closed_trend = rdf['closed_above_open'].mean() * 100

    print(f"\n| Behavior | Count | % |")
    print(f"|----------|-------|---|")
    print(f"| Retraced into composite VA | {rdf['retrace_into_va'].sum()}/{len(rdf)} | **{retrace_rate:.0f}%** |")
    print(f"| Retraced to composite POC | {rdf['retrace_to_poc'].sum()}/{len(rdf)} | {retrace_poc:.0f}% |")
    print(f"| Breakout held (close beyond VA) | {rdf['breakout_held'].sum()}/{len(rdf)} | **{bo_held:.0f}%** |")
    print(f"| Closed in breakout direction | {rdf['closed_above_open'].sum()}/{len(rdf)} | {closed_trend:.0f}% |")

    # Retracement depth stats
    retracers = rdf[rdf['retrace_into_va'] == True]
    if len(retracers) > 0:
        print(f"\n**Retracement Depth (into VA):** (N={len(retracers)})")
        depth = retracers['retrace_depth_into_va']
        print(f"  Mean: {depth.mean():.0f} pts | Median: {depth.median():.0f} pts | 75th: {depth.quantile(0.75):.0f} pts")
        # As % of composite VA width
        depth_pct = retracers['retrace_depth_into_va'] / retracers['comp_width'] * 100
        print(f"  As % of VA width — Mean: {depth_pct.mean():.0f}% | Median: {depth_pct.median():.0f}%")

        # Did retracement happen during IB?
        ib_retrace = retracers['retraced_during_ib'].mean() * 100
        print(f"  Retraced during IB: {retracers['retraced_during_ib'].sum()}/{len(retracers)} ({ib_retrace:.0f}%)")

    # Max excursion in breakout direction
    print(f"\n**Max Excursion (breakout direction):** (N={len(rdf)})")
    exc = rdf['max_excursion']
    print(f"  Mean: {exc.mean():.0f} pts | Median: {exc.median():.0f} pts | 75th: {exc.quantile(0.75):.0f} pts | 90th: {exc.quantile(0.90):.0f} pts")

    # Gap distance stats
    gap = rdf['gap_pts']
    print(f"\n**Gap Distance (open to VA boundary):**")
    print(f"  Mean: {gap.mean():.0f} pts | Median: {gap.median():.0f} pts")

    # LONG vs SHORT breakdown
    for d in ['LONG', 'SHORT']:
        ddf = rdf[rdf['direction'] == d]
        if len(ddf) == 0:
            continue
        print(f"\n**{d} Breakouts** (N={len(ddf)}):")
        print(f"  Retrace into VA: {ddf['retrace_into_va'].sum()}/{len(ddf)} ({ddf['retrace_into_va'].mean()*100:.0f}%)")
        print(f"  Breakout held: {ddf['breakout_held'].sum()}/{len(ddf)} ({ddf['breakout_held'].mean()*100:.0f}%)")
        print(f"  Max excursion median: {ddf['max_excursion'].median():.0f} pts")

    # Single-day VA interaction
    has_sva = rdf[rdf['open_vs_single_va'].notna()]
    if len(has_sva) > 0:
        print(f"\n**Interaction with Single-Day VA** (N={len(has_sva)}):")
        for cls in ['ABOVE', 'BELOW', 'INSIDE']:
            cnt = (has_sva['open_vs_single_va'] == cls).sum()
            if cnt > 0:
                sub = has_sva[has_sva['open_vs_single_va'] == cls]
                held_pct = sub['breakout_held'].mean() * 100
                print(f"  Open {cls} single-day VA: {cnt} ({cnt/len(has_sva)*100:.0f}%) — breakout held: {held_pct:.0f}%")


# ============================================================================
# SECTION 3: ENTRY MODEL BACKTESTS
# ============================================================================
print("\n\n" + "=" * 70)
print("SECTION 3: ENTRY MODEL BACKTESTS")
print("=" * 70)


def backtest_entry(df_rth, results_list, entry_mode, stop_mode, target_mode, label):
    """
    Backtest an entry model on bracket breakout sessions.

    entry_mode:
      'immediate_open' - enter at open price
      'ib_breakout' - enter when IB extends in breakout direction
      'retracement_va_edge' - enter on retracement to composite VA edge
      'retracement_poc' - enter on retracement to composite POC
      'acceptance_2x5min' - 2 consecutive 5-min closes beyond VA
      'acceptance_3x5min' - 3 consecutive 5-min closes beyond VA

    stop_mode:
      'va_edge' - stop at composite VA edge + buffer
      '2x_atr' - stop at 2x ATR from entry
      'ib_boundary' - stop at IB boundary opposite breakout

    target_mode:
      'opposite_va' - target = opposite VA edge (full traverse)
      '1x_va_width' - target = 1x composite VA width from entry
      '0.5x_va_width' - target = 0.5x VA width from entry
      '2R' - target = 2x risk
      '3R' - target = 3x risk
    """
    trades = []
    STOP_BUFFER = 10.0

    for r in results_list:
        session_date = r['session']
        direction = r['direction']
        comp_vah = r['comp_vah']
        comp_val = r['comp_val']
        comp_poc = r['comp_poc']
        comp_width = r['comp_width']
        open_price = r['open']

        session_bars = get_session_bars(df_rth, session_date)
        if len(session_bars) < 60:
            continue

        # --- Determine entry price and bar ---
        entry_price = None
        entry_bar_idx = None

        if entry_mode == 'immediate_open':
            entry_price = open_price
            entry_bar_idx = 0

        elif entry_mode == 'ib_breakout':
            # Wait for IB to extend in breakout direction
            ib_bars = get_ib_bars(session_bars)
            if len(ib_bars) == 0:
                continue
            ib_high = ib_bars['high'].max()
            ib_low = ib_bars['low'].min()

            # Post-IB: look for extension
            post_ib = session_bars[session_bars['time'] > IB_END_TIME]
            for i, (idx, bar) in enumerate(post_ib.iterrows()):
                if bar['time'] >= ENTRY_CUTOFF:
                    break
                if direction == 'LONG' and bar['close'] > ib_high:
                    entry_price = bar['close']
                    entry_bar_idx = session_bars.index.get_loc(idx)
                    break
                elif direction == 'SHORT' and bar['close'] < ib_low:
                    entry_price = bar['close']
                    entry_bar_idx = session_bars.index.get_loc(idx)
                    break

        elif entry_mode == 'retracement_va_edge':
            # Wait for price to retrace to composite VA edge, then enter
            for i, (idx, bar) in enumerate(session_bars.iterrows()):
                if bar['time'] >= ENTRY_CUTOFF:
                    break
                if direction == 'LONG' and bar['low'] <= comp_vah + 5:
                    # Price touched near VAH, enter long
                    entry_price = comp_vah + 5  # slight above VAH
                    entry_bar_idx = i
                    break
                elif direction == 'SHORT' and bar['high'] >= comp_val - 5:
                    entry_price = comp_val - 5
                    entry_bar_idx = i
                    break

        elif entry_mode == 'retracement_poc':
            for i, (idx, bar) in enumerate(session_bars.iterrows()):
                if bar['time'] >= ENTRY_CUTOFF:
                    break
                if direction == 'LONG' and bar['low'] <= comp_poc + 5:
                    entry_price = comp_poc + 5
                    entry_bar_idx = i
                    break
                elif direction == 'SHORT' and bar['high'] >= comp_poc - 5:
                    entry_price = comp_poc - 5
                    entry_bar_idx = i
                    break

        elif entry_mode.startswith('acceptance_'):
            # N consecutive 5-min closes beyond VA in breakout direction
            n_periods = int(entry_mode.split('_')[1].replace('x5min', ''))
            consecutive = 0
            period_bars = 5  # 5x 1-min bars = 5 min period
            bar_list = list(session_bars.iterrows())

            for i in range(0, len(bar_list), period_bars):
                chunk = bar_list[i:i+period_bars]
                if not chunk:
                    break
                last_bar_idx, last_bar = chunk[-1]
                if last_bar['time'] >= ENTRY_CUTOFF:
                    break

                if direction == 'LONG' and last_bar['close'] > comp_vah:
                    consecutive += 1
                elif direction == 'SHORT' and last_bar['close'] < comp_val:
                    consecutive += 1
                else:
                    consecutive = 0

                if consecutive >= n_periods:
                    entry_price = last_bar['close']
                    entry_bar_idx = session_bars.index.get_loc(last_bar_idx)
                    break

        if entry_price is None or entry_bar_idx is None:
            continue

        # --- Determine stop price ---
        atr = session_bars['atr14'].iloc[min(entry_bar_idx, len(session_bars)-1)]
        if pd.isna(atr) or atr <= 0:
            atr = 20.0

        if stop_mode == 'va_edge':
            if direction == 'LONG':
                stop_price = comp_vah - STOP_BUFFER
            else:
                stop_price = comp_val + STOP_BUFFER
        elif stop_mode == '2x_atr':
            if direction == 'LONG':
                stop_price = entry_price - 2 * atr
            else:
                stop_price = entry_price + 2 * atr
        elif stop_mode == 'ib_boundary':
            ib_bars = get_ib_bars(session_bars)
            if len(ib_bars) == 0:
                continue
            ib_high = ib_bars['high'].max()
            ib_low = ib_bars['low'].min()
            if direction == 'LONG':
                stop_price = ib_low - STOP_BUFFER
            else:
                stop_price = ib_high + STOP_BUFFER

        risk = abs(entry_price - stop_price)
        if risk <= 0:
            continue

        # --- Determine target price ---
        if target_mode == 'opposite_va':
            if direction == 'LONG':
                target_price = comp_vah + comp_width  # VAH + full width above
            else:
                target_price = comp_val - comp_width
        elif target_mode == '1x_va_width':
            if direction == 'LONG':
                target_price = entry_price + comp_width
            else:
                target_price = entry_price - comp_width
        elif target_mode == '0.5x_va_width':
            if direction == 'LONG':
                target_price = entry_price + comp_width * 0.5
            else:
                target_price = entry_price - comp_width * 0.5
        elif target_mode == '2R':
            if direction == 'LONG':
                target_price = entry_price + 2 * risk
            else:
                target_price = entry_price - 2 * risk
        elif target_mode == '3R':
            if direction == 'LONG':
                target_price = entry_price + 3 * risk
            else:
                target_price = entry_price - 3 * risk

        # --- Simulate trade bar-by-bar from entry ---
        remaining_bars = session_bars.iloc[entry_bar_idx:]
        outcome = 'EOD'
        exit_price = remaining_bars['close'].iloc[-1]  # default = EOD

        for _, bar in remaining_bars.iterrows():
            if bar['time'] >= EOD_EXIT:
                exit_price = bar['close']
                outcome = 'EOD'
                break

            if direction == 'LONG':
                if bar['low'] <= stop_price:
                    exit_price = stop_price
                    outcome = 'STOP'
                    break
                if bar['high'] >= target_price:
                    exit_price = target_price
                    outcome = 'TARGET'
                    break
            else:
                if bar['high'] >= stop_price:
                    exit_price = stop_price
                    outcome = 'STOP'
                    break
                if bar['low'] <= target_price:
                    exit_price = target_price
                    outcome = 'TARGET'
                    break

        # P&L
        if direction == 'LONG':
            pnl_pts = exit_price - entry_price
        else:
            pnl_pts = entry_price - exit_price

        pnl_dollars = pnl_pts_to_dollars(pnl_pts)

        trades.append({
            'session': session_date,
            'direction': direction,
            'entry': entry_price,
            'stop': stop_price,
            'target': target_price,
            'exit': exit_price,
            'risk_pts': risk,
            'pnl_pts': pnl_pts,
            'pnl_dollars': pnl_dollars,
            'outcome': outcome,
            'r_multiple': pnl_pts / risk if risk > 0 else 0,
        })

    if not trades:
        return None

    tdf = pd.DataFrame(trades)
    n = len(tdf)
    wins = (tdf['pnl_pts'] > 0).sum()
    wr = wins / n * 100 if n > 0 else 0
    total_pnl = tdf['pnl_dollars'].sum()
    monthly = total_pnl / months if months > 0 else 0
    gross_win = tdf[tdf['pnl_pts'] > 0]['pnl_dollars'].sum()
    gross_loss = abs(tdf[tdf['pnl_pts'] <= 0]['pnl_dollars'].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
    avg_risk = tdf['risk_pts'].mean()
    stops = (tdf['outcome'] == 'STOP').sum()
    targets = (tdf['outcome'] == 'TARGET').sum()
    eods = (tdf['outcome'] == 'EOD').sum()

    return {
        'label': label,
        'n': n,
        'trades_mo': n / months,
        'wr': wr,
        'pf': pf,
        'monthly': monthly,
        'avg_risk': avg_risk,
        'stops': stops,
        'targets': targets,
        'eods': eods,
        'trades_df': tdf,
    }


# Run backtests for each lookback
for lb in LOOKBACKS:
    breakouts = breakout_data[lb]
    if not breakouts:
        continue

    # Compute results for this lookback
    results = []
    for bo in breakouts:
        r = analyze_session_behavior(df_rth, bo, comp_vas[lb], lb)
        if r is not None:
            results.append(r)

    if not results:
        continue

    print(f"\n### {lb}-Day Lookback ({len(results)} breakout sessions)\n")

    configs = [
        # (entry_mode, stop_mode, target_mode, label)
        ('immediate_open', 'va_edge', '2R', 'Open + VA edge stop + 2R'),
        ('immediate_open', '2x_atr', '2R', 'Open + 2xATR stop + 2R'),
        ('immediate_open', '2x_atr', '3R', 'Open + 2xATR stop + 3R'),
        ('acceptance_2x5min', '2x_atr', '2R', '2x5min accept + 2xATR + 2R'),
        ('acceptance_3x5min', '2x_atr', '2R', '3x5min accept + 2xATR + 2R'),
        ('ib_breakout', '2x_atr', '2R', 'IB break + 2xATR + 2R'),
        ('ib_breakout', '2x_atr', '3R', 'IB break + 2xATR + 3R'),
        ('ib_breakout', 'ib_boundary', '2R', 'IB break + IB stop + 2R'),
        ('retracement_va_edge', 'va_edge', '2R', 'Retrace to VA + VA stop + 2R'),
        ('retracement_va_edge', '2x_atr', '2R', 'Retrace to VA + 2xATR + 2R'),
        ('retracement_va_edge', '2x_atr', '3R', 'Retrace to VA + 2xATR + 3R'),
        ('retracement_poc', '2x_atr', '2R', 'Retrace to POC + 2xATR + 2R'),
        ('retracement_poc', '2x_atr', '3R', 'Retrace to POC + 2xATR + 3R'),
    ]

    print(f"| Config | N | T/Mo | WR | PF | $/Mo | Risk | S/T/E |")
    print(f"|--------|---|------|-----|-----|------|------|-------|")

    best_monthly = -float('inf')
    best_config = None

    for entry_m, stop_m, target_m, label in configs:
        bt = backtest_entry(df_rth, results, entry_m, stop_m, target_m, label)
        if bt is None:
            print(f"| {label} | 0 | — | — | — | — | — | — |")
            continue

        print(f"| {label} | {bt['n']} | {bt['trades_mo']:.1f} | {bt['wr']:.1f}% | {bt['pf']:.2f} | ${bt['monthly']:.0f} | {bt['avg_risk']:.0f}p | {bt['stops']}/{bt['targets']}/{bt['eods']} |")

        if bt['monthly'] > best_monthly:
            best_monthly = bt['monthly']
            best_config = bt

    # Best config LONG vs SHORT breakdown
    if best_config is not None and best_config['n'] >= 10:
        tdf = best_config['trades_df']
        print(f"\n**Best: {best_config['label']}** — LONG vs SHORT:")
        for d in ['LONG', 'SHORT']:
            ddf = tdf[tdf['direction'] == d]
            if len(ddf) == 0:
                continue
            dw = (ddf['pnl_pts'] > 0).sum()
            dwr = dw / len(ddf) * 100
            dtotal = ddf['pnl_dollars'].sum()
            print(f"  {d}: {len(ddf)} trades, WR {dwr:.0f}%, ${dtotal/months:.0f}/mo")


# ============================================================================
# SECTION 4: CONFLUENCE — BRACKET + SINGLE-DAY VA
# ============================================================================
print("\n\n" + "=" * 70)
print("SECTION 4: CONFLUENCE — BRACKET BREAKOUT + SINGLE-DAY 80P/20P")
print("=" * 70)

# For the best-performing lookback, check overlap with 80P/20P
for lb in LOOKBACKS:
    breakouts = breakout_data[lb]
    if not breakouts:
        continue

    results = []
    for bo in breakouts:
        r = analyze_session_behavior(df_rth, bo, comp_vas[lb], lb)
        if r is not None:
            results.append(r)

    rdf = pd.DataFrame(results) if results else pd.DataFrame()
    if len(rdf) == 0:
        continue

    has_sva = rdf[rdf['open_vs_single_va'].notna()]
    if len(has_sva) == 0:
        continue

    print(f"\n### {lb}-Day Lookback — Single-Day VA Overlap ({len(has_sva)} sessions)")

    # When we have a bracket breakout AND the open is also outside single-day VA
    # → that's a "double breakout" (both timeframes agree)
    if 'LONG' in has_sva['direction'].values:
        long_above_sva = has_sva[(has_sva['direction'] == 'LONG') & (has_sva['open_vs_single_va'] == 'ABOVE')]
        long_inside_sva = has_sva[(has_sva['direction'] == 'LONG') & (has_sva['open_vs_single_va'] == 'INSIDE')]
        long_below_sva = has_sva[(has_sva['direction'] == 'LONG') & (has_sva['open_vs_single_va'] == 'BELOW')]

        print(f"\n  LONG bracket breakout + single-day VA classification:")
        for label, sub in [('Also above single VA (double BO)', long_above_sva),
                           ('Inside single VA', long_inside_sva),
                           ('Below single VA (conflict)', long_below_sva)]:
            if len(sub) > 0:
                held = sub['breakout_held'].mean() * 100
                exc = sub['max_excursion'].median()
                print(f"    {label}: {len(sub)} — held {held:.0f}%, median excursion {exc:.0f}pts")

    if 'SHORT' in has_sva['direction'].values:
        short_below_sva = has_sva[(has_sva['direction'] == 'SHORT') & (has_sva['open_vs_single_va'] == 'BELOW')]
        short_inside_sva = has_sva[(has_sva['direction'] == 'SHORT') & (has_sva['open_vs_single_va'] == 'INSIDE')]
        short_above_sva = has_sva[(has_sva['direction'] == 'SHORT') & (has_sva['open_vs_single_va'] == 'ABOVE')]

        print(f"\n  SHORT bracket breakout + single-day VA classification:")
        for label, sub in [('Also below single VA (double BO)', short_below_sva),
                           ('Inside single VA', short_inside_sva),
                           ('Above single VA (conflict)', short_above_sva)]:
            if len(sub) > 0:
                held = sub['breakout_held'].mean() * 100
                exc = sub['max_excursion'].median()
                print(f"    {label}: {len(sub)} — held {held:.0f}%, median excursion {exc:.0f}pts")


# ============================================================================
# SECTION 5: BRACKET TIGHTNESS — DO NARROW BRACKETS BREAK HARDER?
# ============================================================================
print("\n\n" + "=" * 70)
print("SECTION 5: BRACKET CHARACTERISTICS — WIDTH, GAP, TIGHTNESS")
print("=" * 70)

for lb in [5]:  # Focus on 5-day as the "weekly" proxy
    breakouts = breakout_data[lb]
    if not breakouts:
        continue

    results = []
    for bo in breakouts:
        r = analyze_session_behavior(df_rth, bo, comp_vas[lb], lb)
        if r is not None:
            results.append(r)

    if not results:
        continue

    rdf = pd.DataFrame(results)

    print(f"\n### {lb}-Day Bracket Characteristics ({len(rdf)} breakouts)\n")

    # Width percentiles
    width = rdf['comp_width']
    print(f"**Composite VA Width:**")
    print(f"  25th: {width.quantile(0.25):.0f} pts | Median: {width.median():.0f} pts | 75th: {width.quantile(0.75):.0f} pts | Max: {width.max():.0f} pts")

    # Split by narrow vs wide bracket
    median_width = width.median()
    narrow = rdf[rdf['comp_width'] <= median_width]
    wide = rdf[rdf['comp_width'] > median_width]

    print(f"\n| Bracket Type | N | Retrace % | BO Held % | Median Excursion |")
    print(f"|-------------|---|-----------|-----------|-----------------|")
    for label, sub in [('Narrow (< median)', narrow), ('Wide (> median)', wide)]:
        if len(sub) > 0:
            ret = sub['retrace_into_va'].mean() * 100
            held = sub['breakout_held'].mean() * 100
            exc = sub['max_excursion'].median()
            print(f"| {label} | {len(sub)} | {ret:.0f}% | {held:.0f}% | {exc:.0f} pts |")

    # Gap size analysis
    gap = rdf['gap_pts']
    print(f"\n**Gap Distance (open to VA boundary):**")
    print(f"  25th: {gap.quantile(0.25):.0f} pts | Median: {gap.median():.0f} pts | 75th: {gap.quantile(0.75):.0f} pts")

    median_gap = gap.median()
    small_gap = rdf[rdf['gap_pts'] <= median_gap]
    big_gap = rdf[rdf['gap_pts'] > median_gap]

    print(f"\n| Gap Size | N | Retrace % | BO Held % | Median Excursion |")
    print(f"|----------|---|-----------|-----------|-----------------|")
    for label, sub in [('Small gap (< median)', small_gap), ('Big gap (> median)', big_gap)]:
        if len(sub) > 0:
            ret = sub['retrace_into_va'].mean() * 100
            held = sub['breakout_held'].mean() * 100
            exc = sub['max_excursion'].median()
            print(f"| {label} | {len(sub)} | {ret:.0f}% | {held:.0f}% | {exc:.0f} pts |")


# ============================================================================
# SUMMARY
# ============================================================================
print("\n\n" + "=" * 70)
print("SUMMARY — KEY FINDINGS")
print("=" * 70)

print("""
Key questions answered:
1. Breakout frequency by lookback
2. Retracement rates (does it pull back into the bracket?)
3. Best entry model for trading the breakout or the retracement
4. Confluence with single-day 80P/20P
5. Does bracket width or gap size affect outcome?

See sections above for detailed findings.
""")

print("Study complete.")
