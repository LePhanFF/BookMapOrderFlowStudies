"""
80P Rejection Candle Quality Analysis
======================================

Tests whether the QUALITY of the rejection candle and the QUALITY of the retest
predict trade outcome. Specifically:

1. CANDLE CLOSE DEPTH — How deep does the rejection candle close inside the VA?
   If it closes >40% into VA, is the mean reversion stronger?

2. CVD DIVERGENCE AT RETEST — When price retests the candle extreme, is CVD
   diverging? (price retests high but CVD declining = bearish divergence for shorts)

3. SWEEP / DOUBLE TOP-BOTTOM — Does the retest sweep slightly beyond the candle
   extreme (liquidity grab) before reversing? Or does it respect it exactly?

4. BOUNCE STRENGTH — Is a weak retest bounce (low volume, slow) better than a
   strong bounce? Weak bounce = bigger continuation move expected.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import time
from typing import Dict, List, Optional

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.instruments import get_instrument
from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from indicators.value_area import compute_session_value_areas, ValueAreaLevels

INSTRUMENT = get_instrument('MNQ')
TICK_VALUE = INSTRUMENT.tick_value / INSTRUMENT.tick_size
SLIPPAGE_PTS = 0.50
COMMISSION = 1.24
CONTRACTS = 5
MIN_VA_WIDTH = 25.0
STOP_BUFFER = 10.0
ENTRY_CUTOFF = time(13, 0)
PERIOD_BARS = 30

# ============================================================================
# DATA
# ============================================================================
print("Loading data...")
df_raw = load_csv('NQ')
df_rth = filter_rth(df_raw)
df_rth = compute_all_features(df_rth)

if 'session_date' not in df_rth.columns:
    df_rth['session_date'] = df_rth['timestamp'].dt.date

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

print("Computing ETH Value Areas...")
eth_va = compute_session_value_areas(df_full, tick_size=0.25, va_percent=0.70)


# ============================================================================
# FIND SETUPS WITH FULL 1-MIN BAR DATA
# ============================================================================
def find_setups_detailed(df_rth, va_by_session, directions='BOTH'):
    """Find 80P setups capturing rejection candle + full 1-min bar data for analysis."""
    setups = []

    for i in range(1, len(sessions)):
        current = sessions[i]
        prior = sessions[i - 1]
        prior_key = str(prior)

        if prior_key not in va_by_session:
            continue
        prior_va = va_by_session[prior_key]
        if prior_va.va_width < MIN_VA_WIDTH:
            continue

        session_df = df_rth[df_rth['session_date'] == current].reset_index(drop=True)
        if len(session_df) < 90:
            continue

        open_price = session_df['open'].iloc[0]
        vah = prior_va.vah
        val = prior_va.val
        poc = prior_va.poc
        va_width = vah - val

        if open_price < val:
            direction = 'LONG'
        elif open_price > vah:
            direction = 'SHORT'
        else:
            continue

        if directions == 'LONG_ONLY' and direction == 'SHORT':
            continue

        post_ib = session_df.iloc[60:].reset_index(drop=True)
        if len(post_ib) < PERIOD_BARS * 2:
            continue

        # Find the first 30-min period where close is inside VA
        acceptance_period_end = None
        for bar_idx in range(len(post_ib)):
            period_end = ((bar_idx + 1) % PERIOD_BARS == 0)
            if not period_end:
                continue

            close = post_ib.iloc[bar_idx]['close']
            bar_time = post_ib.iloc[bar_idx]['timestamp']
            bt = bar_time.time() if hasattr(bar_time, 'time') else None
            if bt and bt >= ENTRY_CUTOFF:
                break

            is_inside = val <= close <= vah
            if is_inside:
                acceptance_period_end = bar_idx
                break

        if acceptance_period_end is None:
            continue

        # Get the 30-min rejection candle OHLC
        period_start = acceptance_period_end - PERIOD_BARS + 1
        candle_bars = post_ib.iloc[period_start:acceptance_period_end + 1]

        candle_open = candle_bars.iloc[0]['open']
        candle_high = candle_bars['high'].max()
        candle_low = candle_bars['low'].min()
        candle_close = candle_bars.iloc[-1]['close']
        candle_range = candle_high - candle_low
        candle_volume = candle_bars['volume'].sum()
        candle_delta = candle_bars['vol_delta'].sum() if 'vol_delta' in candle_bars.columns else 0

        if candle_range < 2:
            continue

        # Candle close depth within VA (0 = at VA edge, 1 = at opposite VA)
        if direction == 'LONG':
            # Candle pushes UP from below VAL into VA
            candle_close_depth = (candle_close - val) / va_width if va_width > 0 else 0
        else:
            # Candle pushes DOWN from above VAH into VA
            candle_close_depth = (vah - candle_close) / va_width if va_width > 0 else 0

        # Prior session ATR
        prior_df = df_rth[df_rth['session_date'] == prior]
        prior_atr = (prior_df['high'] - prior_df['low']).mean() if len(prior_df) > 30 else 15.0

        setups.append({
            'session_date': str(current),
            'direction': direction,
            'open_price': open_price,
            'vah': vah,
            'val': val,
            'poc': poc,
            'va_width': va_width,
            'candle_open': candle_open,
            'candle_high': candle_high,
            'candle_low': candle_low,
            'candle_close': candle_close,
            'candle_range': candle_range,
            'candle_volume': candle_volume,
            'candle_delta': candle_delta,
            'candle_close_depth': candle_close_depth,
            'acceptance_bar': acceptance_period_end,
            'prior_atr': prior_atr,
            'post_ib': post_ib,
        })

    return setups


# ============================================================================
# REPLAY WITH QUALITY METRICS
# ============================================================================
def replay_with_quality(setup, retracement_pct=1.0, stop_mode='va_edge',
                        target_mode='opposite_va', sweep_tolerance=5.0):
    """
    Replay a retest entry and capture quality metrics:
    - CVD from acceptance to retest
    - Did the retest sweep beyond candle extreme?
    - Retest bounce velocity and volume
    """
    direction = setup['direction']
    vah = setup['vah']
    val = setup['val']
    poc = setup['poc']
    va_width = setup['va_width']
    post_ib = setup['post_ib']
    acc_bar = setup['acceptance_bar']
    prior_atr = setup['prior_atr']

    ch = setup['candle_high']
    cl = setup['candle_low']
    cr = setup['candle_range']
    cc = setup['candle_close']

    # Compute limit entry price
    if retracement_pct == 0.0:
        entry_price = cc
        entry_bar = acc_bar
    else:
        if direction == 'LONG':
            limit_price = ch - retracement_pct * cr
        else:
            limit_price = cl + retracement_pct * cr

        entry_bar = None
        entry_price = limit_price

        for bar_idx in range(acc_bar + 1, len(post_ib)):
            bar = post_ib.iloc[bar_idx]
            bt_time = bar.get('timestamp')
            bt = bt_time.time() if bt_time and hasattr(bt_time, 'time') else None
            if bt and bt >= time(15, 30):
                break

            if direction == 'LONG' and bar['low'] <= limit_price:
                entry_bar = bar_idx
                break
            elif direction == 'SHORT' and bar['high'] >= limit_price:
                entry_bar = bar_idx
                break

        if entry_bar is None:
            return None

    # ---- QUALITY METRICS ----

    # 1. CVD from acceptance candle close to retest entry
    cvd_bars = post_ib.iloc[acc_bar + 1:entry_bar + 1] if entry_bar > acc_bar else pd.DataFrame()
    if len(cvd_bars) > 0 and 'vol_delta' in cvd_bars.columns:
        cvd_sum = cvd_bars['vol_delta'].sum()
        cvd_volume = cvd_bars['volume'].sum()
    else:
        cvd_sum = 0
        cvd_volume = 0

    # CVD divergence: for SHORT, price goes UP (retest high) but CVD should be negative
    # for LONG, price goes DOWN (retest low) but CVD should be positive
    if direction == 'SHORT':
        cvd_divergence = cvd_sum < 0  # price up but delta negative = bearish divergence
    else:
        cvd_divergence = cvd_sum > 0  # price down but delta positive = bullish divergence

    # 2. Sweep analysis: did price go BEYOND the candle extreme?
    sweep_pts = 0.0
    if entry_bar > acc_bar:
        between_bars = post_ib.iloc[acc_bar + 1:entry_bar + 1]
        if direction == 'LONG':
            # Retest of candle low — did price sweep below candle low?
            min_low = between_bars['low'].min()
            if min_low < cl:
                sweep_pts = cl - min_low  # how far below candle low
        else:
            # Retest of candle high — did price sweep above candle high?
            max_high = between_bars['high'].max()
            if max_high > ch:
                sweep_pts = max_high - ch  # how far above candle high

    did_sweep = sweep_pts > sweep_tolerance

    # 3. Bounce strength: volume and velocity of the retest move
    bars_to_retest = entry_bar - acc_bar if entry_bar > acc_bar else 1
    retest_velocity = cr / bars_to_retest if bars_to_retest > 0 else 0

    # Volume on retest bars vs candle volume
    retest_bars = post_ib.iloc[acc_bar + 1:entry_bar + 1] if entry_bar > acc_bar else pd.DataFrame()
    retest_volume = retest_bars['volume'].sum() if len(retest_bars) > 0 else 0
    vol_ratio = retest_volume / setup['candle_volume'] if setup['candle_volume'] > 0 else 0

    # Retest delta (total delta during the retest move)
    retest_delta = retest_bars['vol_delta'].sum() if len(retest_bars) > 0 and 'vol_delta' in retest_bars.columns else 0

    # ---- STOP ----
    if stop_mode == 'va_edge':
        if direction == 'LONG':
            stop_price = val - STOP_BUFFER
        else:
            stop_price = vah + STOP_BUFFER
    elif stop_mode == 'candle_extreme':
        if direction == 'LONG':
            stop_price = cl - STOP_BUFFER
        else:
            stop_price = ch + STOP_BUFFER
    else:
        if direction == 'LONG':
            stop_price = val - STOP_BUFFER
        else:
            stop_price = vah + STOP_BUFFER

    risk_pts = abs(entry_price - stop_price)
    if risk_pts <= 0:
        return None

    # ---- TARGET ----
    if target_mode == 'opposite_va':
        target = vah if direction == 'LONG' else val
    elif target_mode.endswith('R'):
        r_mult = float(target_mode[:-1])
        if direction == 'LONG':
            target = entry_price + risk_pts * r_mult
        else:
            target = entry_price - risk_pts * r_mult
    else:
        target = vah if direction == 'LONG' else val

    if direction == 'LONG':
        reward = target - entry_price
    else:
        reward = entry_price - target
    if reward <= 0:
        return None

    # ---- REPLAY ----
    remaining = post_ib.iloc[entry_bar:]
    mfe = 0.0
    mae = 0.0
    exit_price = None
    exit_reason = None
    bars_held = 0

    for i in range(len(remaining)):
        bar = remaining.iloc[i]
        bars_held += 1
        bt_time = bar.get('timestamp')
        bt = bt_time.time() if bt_time and hasattr(bt_time, 'time') else None
        if bt and bt >= time(15, 30):
            exit_price = bar['close']
            exit_reason = 'EOD'
            break

        if direction == 'LONG':
            fav = bar['high'] - entry_price
            adv = entry_price - bar['low']
        else:
            fav = entry_price - bar['low']
            adv = bar['high'] - entry_price
        mfe = max(mfe, fav)
        mae = max(mae, adv)

        if direction == 'LONG' and bar['low'] <= stop_price:
            exit_price = stop_price
            exit_reason = 'STOP'
            break
        elif direction == 'SHORT' and bar['high'] >= stop_price:
            exit_price = stop_price
            exit_reason = 'STOP'
            break

        if direction == 'LONG' and bar['high'] >= target:
            exit_price = target
            exit_reason = 'TARGET'
            break
        elif direction == 'SHORT' and bar['low'] <= target:
            exit_price = target
            exit_reason = 'TARGET'
            break

    if exit_price is None:
        exit_price = remaining.iloc[-1]['close']
        exit_reason = 'EOD'

    if direction == 'LONG':
        pnl_pts = exit_price - entry_price - SLIPPAGE_PTS
    else:
        pnl_pts = entry_price - exit_price - SLIPPAGE_PTS
    pnl_dollars = pnl_pts * TICK_VALUE * CONTRACTS - COMMISSION * CONTRACTS

    return {
        'session_date': setup['session_date'],
        'direction': direction,
        'entry_price': entry_price,
        'stop_price': stop_price,
        'target': target,
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'pnl_pts': pnl_pts,
        'pnl_dollars': pnl_dollars,
        'risk_pts': risk_pts,
        'reward_pts': reward,
        'mfe_pts': mfe,
        'mae_pts': mae,
        'bars_held': bars_held,
        'is_winner': pnl_dollars > 0,

        # Quality metrics
        'candle_close_depth': setup['candle_close_depth'],
        'candle_range': setup['candle_range'],
        'candle_volume': setup['candle_volume'],
        'candle_delta': setup['candle_delta'],
        'cvd_to_retest': cvd_sum,
        'cvd_divergence': cvd_divergence,
        'did_sweep': did_sweep,
        'sweep_pts': sweep_pts,
        'bars_to_retest': bars_to_retest,
        'retest_velocity': retest_velocity,
        'retest_volume': retest_volume,
        'retest_delta': retest_delta,
        'vol_ratio': vol_ratio,
    }


# ============================================================================
# MAIN
# ============================================================================
print("\nFinding 80P setups with candle quality data...")
setups = find_setups_detailed(df_rth, eth_va, directions='BOTH')
print(f"Found {len(setups)} setups")

# ============================================================================
# 1. CANDLE CLOSE DEPTH — Does deeper close = better trade?
# ============================================================================
print(f"\n{'='*100}")
print(f"  1. REJECTION CANDLE CLOSE DEPTH IN VA")
print(f"     Does deeper penetration into VA predict better outcomes?")
print(f"{'='*100}")

# Show distribution
depths = [s['candle_close_depth'] for s in setups]
print(f"\n  Candle close depth distribution (0=VA edge, 1=opposite VA):")
print(f"    Mean:   {np.mean(depths):.2f}")
print(f"    Median: {np.median(depths):.2f}")
for pct in [10, 25, 50, 75, 90]:
    print(f"    {pct}th pctile: {np.percentile(depths, pct):.2f}")

# Segment by depth and compare outcomes
for target_mode in ['opposite_va', '2R']:
    print(f"\n  ── Target: {target_mode} ──")

    # First get ALL retest results (100% retest, VA edge stop)
    all_results = []
    for setup in setups:
        r = replay_with_quality(setup, retracement_pct=1.0, stop_mode='va_edge',
                                target_mode=target_mode)
        if r:
            all_results.append(r)

    if not all_results:
        print("    No trades.")
        continue

    df_r = pd.DataFrame(all_results)
    n_total = len(df_r)

    # Also do 0% (current model) for comparison
    all_current = []
    for setup in setups:
        r = replay_with_quality(setup, retracement_pct=0.0, stop_mode='va_edge',
                                target_mode=target_mode)
        if r:
            all_current.append(r)
    df_current = pd.DataFrame(all_current)

    print(f"\n    Total 100% retest trades: {n_total}")
    print(f"    Total current model trades: {len(df_current)}")

    # Segment by candle close depth
    bins = [0.0, 0.20, 0.40, 0.60, 1.01]
    labels = ['0-20%', '20-40%', '40-60%', '60%+']

    print(f"\n    {'Depth':>10s} {'N':>4s} {'WR':>6s} {'PF':>6s} {'$/Mo':>9s} "
          f"{'AvgWin':>8s} {'AvgLoss':>8s} {'Stopped':>7s} {'Target':>7s}")
    print(f"    {'-'*80}")

    for i in range(len(labels)):
        low, high = bins[i], bins[i + 1]
        subset = df_r[(df_r['candle_close_depth'] >= low) & (df_r['candle_close_depth'] < high)]
        if len(subset) < 2:
            print(f"    {labels[i]:>10s} {len(subset):>4d}  (too few)")
            continue
        wr = subset['is_winner'].mean() * 100
        gw = subset[subset['is_winner']]['pnl_dollars'].sum()
        gl = abs(subset[~subset['is_winner']]['pnl_dollars'].sum())
        pf = gw / gl if gl > 0 else float('inf')
        pm = subset['pnl_dollars'].sum() / months
        avg_w = subset[subset['is_winner']]['pnl_dollars'].mean() if subset['is_winner'].any() else 0
        avg_l = subset[~subset['is_winner']]['pnl_dollars'].mean() if (~subset['is_winner']).any() else 0
        stopped = (subset['exit_reason'] == 'STOP').sum()
        target_hit = (subset['exit_reason'] == 'TARGET').sum()
        print(f"    {labels[i]:>10s} {len(subset):>4d} {wr:>5.1f}% {pf:>5.2f} "
              f"${pm:>7,.0f} ${avg_w:>6,.0f} ${avg_l:>6,.0f} "
              f"{stopped:>6d} {target_hit:>6d}")

    # Also test the current model (0% retest) segmented by depth
    print(f"\n    --- Current model (candle close entry) by depth ---")
    print(f"    {'Depth':>10s} {'N':>4s} {'WR':>6s} {'PF':>6s} {'$/Mo':>9s}")
    print(f"    {'-'*50}")
    for i in range(len(labels)):
        low, high = bins[i], bins[i + 1]
        subset = df_current[(df_current['candle_close_depth'] >= low) & (df_current['candle_close_depth'] < high)]
        if len(subset) < 2:
            print(f"    {labels[i]:>10s} {len(subset):>4d}  (too few)")
            continue
        wr = subset['is_winner'].mean() * 100
        gw = subset[subset['is_winner']]['pnl_dollars'].sum()
        gl = abs(subset[~subset['is_winner']]['pnl_dollars'].sum())
        pf = gw / gl if gl > 0 else float('inf')
        pm = subset['pnl_dollars'].sum() / months
        print(f"    {labels[i]:>10s} {len(subset):>4d} {wr:>5.1f}% {pf:>5.2f} ${pm:>7,.0f}")


# ============================================================================
# 2. CVD DIVERGENCE AT RETEST
# ============================================================================
print(f"\n{'='*100}")
print(f"  2. CVD DIVERGENCE AT RETEST")
print(f"     From acceptance to retest: is delta confirming or diverging?")
print(f"     SHORT: price goes UP to retest high, CVD should be NEGATIVE (bearish div)")
print(f"     LONG: price goes DOWN to retest low, CVD should be POSITIVE (bullish div)")
print(f"{'='*100}")

for target_mode in ['opposite_va', '2R']:
    all_results = []
    for setup in setups:
        r = replay_with_quality(setup, retracement_pct=1.0, stop_mode='va_edge',
                                target_mode=target_mode)
        if r:
            all_results.append(r)

    if not all_results:
        continue

    df_r = pd.DataFrame(all_results)
    print(f"\n  ── Target: {target_mode} ({len(df_r)} trades) ──")

    # CVD divergence vs no divergence
    div = df_r[df_r['cvd_divergence'] == True]
    no_div = df_r[df_r['cvd_divergence'] == False]

    print(f"\n    CVD stats:")
    print(f"      Mean CVD to retest: {df_r['cvd_to_retest'].mean():.0f}")
    print(f"      Median CVD:         {df_r['cvd_to_retest'].median():.0f}")

    print(f"\n    {'Filter':>25s} {'N':>4s} {'WR':>6s} {'PF':>6s} {'$/Mo':>9s} "
          f"{'AvgWin':>8s} {'AvgLoss':>8s}")
    print(f"    {'-'*75}")

    for label, subset in [('All trades', df_r),
                          ('CVD divergence (YES)', div),
                          ('CVD divergence (NO)', no_div)]:
        if len(subset) < 2:
            print(f"    {label:>25s} {len(subset):>4d}  (too few)")
            continue
        wr = subset['is_winner'].mean() * 100
        gw = subset[subset['is_winner']]['pnl_dollars'].sum()
        gl = abs(subset[~subset['is_winner']]['pnl_dollars'].sum())
        pf = gw / gl if gl > 0 else float('inf')
        pm = subset['pnl_dollars'].sum() / months
        avg_w = subset[subset['is_winner']]['pnl_dollars'].mean() if subset['is_winner'].any() else 0
        avg_l = subset[~subset['is_winner']]['pnl_dollars'].mean() if (~subset['is_winner']).any() else 0
        print(f"    {label:>25s} {len(subset):>4d} {wr:>5.1f}% {pf:>5.2f} "
              f"${pm:>7,.0f} ${avg_w:>6,.0f} ${avg_l:>6,.0f}")

    # Also break out by direction
    for dir_label in ['LONG', 'SHORT']:
        dir_subset = df_r[df_r['direction'] == dir_label]
        if len(dir_subset) < 3:
            continue
        dir_div = dir_subset[dir_subset['cvd_divergence'] == True]
        dir_no = dir_subset[dir_subset['cvd_divergence'] == False]
        print(f"\n    {dir_label} trades:")
        for label, subset in [(f'  {dir_label} + CVD div', dir_div),
                              (f'  {dir_label} + no div', dir_no)]:
            if len(subset) < 2:
                print(f"    {label:>25s} {len(subset):>4d}  (too few)")
                continue
            wr = subset['is_winner'].mean() * 100
            gw = subset[subset['is_winner']]['pnl_dollars'].sum()
            gl = abs(subset[~subset['is_winner']]['pnl_dollars'].sum())
            pf = gw / gl if gl > 0 else float('inf')
            pm = subset['pnl_dollars'].sum() / months
            print(f"    {label:>25s} {len(subset):>4d} {wr:>5.1f}% {pf:>5.2f} ${pm:>7,.0f}")


# ============================================================================
# 3. SWEEP ANALYSIS — Does the retest sweep beyond candle extreme?
# ============================================================================
print(f"\n{'='*100}")
print(f"  3. SWEEP ANALYSIS")
print(f"     Does the retest sweep beyond the candle extreme (liquidity grab)?")
print(f"     Sweep = price goes >5pt beyond candle high/low before filling limit")
print(f"{'='*100}")

for target_mode in ['opposite_va', '2R']:
    all_results = []
    for setup in setups:
        r = replay_with_quality(setup, retracement_pct=1.0, stop_mode='va_edge',
                                target_mode=target_mode)
        if r:
            all_results.append(r)

    if not all_results:
        continue

    df_r = pd.DataFrame(all_results)
    print(f"\n  ── Target: {target_mode} ({len(df_r)} trades) ──")

    swept = df_r[df_r['did_sweep'] == True]
    not_swept = df_r[df_r['did_sweep'] == False]

    print(f"\n    Sweep stats:")
    print(f"      Trades with sweep (>5pt beyond): {len(swept)} ({len(swept)/len(df_r)*100:.0f}%)")
    print(f"      Trades without sweep:             {len(not_swept)} ({len(not_swept)/len(df_r)*100:.0f}%)")
    if len(swept) > 0:
        print(f"      Mean sweep distance: {swept['sweep_pts'].mean():.1f} pts")
        print(f"      Max sweep distance:  {swept['sweep_pts'].max():.1f} pts")

    print(f"\n    {'Filter':>20s} {'N':>4s} {'WR':>6s} {'PF':>6s} {'$/Mo':>9s}")
    print(f"    {'-'*55}")
    for label, subset in [('All trades', df_r), ('Swept (>5pt)', swept), ('Not swept', not_swept)]:
        if len(subset) < 2:
            print(f"    {label:>20s} {len(subset):>4d}  (too few)")
            continue
        wr = subset['is_winner'].mean() * 100
        gw = subset[subset['is_winner']]['pnl_dollars'].sum()
        gl = abs(subset[~subset['is_winner']]['pnl_dollars'].sum())
        pf = gw / gl if gl > 0 else float('inf')
        pm = subset['pnl_dollars'].sum() / months
        print(f"    {label:>20s} {len(subset):>4d} {wr:>5.1f}% {pf:>5.2f} ${pm:>7,.0f}")

    # Test different sweep tolerances
    print(f"\n    Sweep tolerance comparison:")
    print(f"    {'Tolerance':>12s} {'N swept':>8s} {'WR swept':>10s} {'WR not':>8s} {'PF swept':>10s} {'PF not':>8s}")
    print(f"    {'-'*65}")
    for tol in [0, 2, 5, 10, 15, 20]:
        s_yes = df_r[df_r['sweep_pts'] > tol]
        s_no = df_r[df_r['sweep_pts'] <= tol]
        wr_yes = s_yes['is_winner'].mean() * 100 if len(s_yes) > 1 else 0
        wr_no = s_no['is_winner'].mean() * 100 if len(s_no) > 1 else 0
        gw_y = s_yes[s_yes['is_winner']]['pnl_dollars'].sum() if len(s_yes) > 0 else 0
        gl_y = abs(s_yes[~s_yes['is_winner']]['pnl_dollars'].sum()) if len(s_yes) > 0 else 0.01
        pf_y = gw_y / gl_y if gl_y > 0 else 0
        gw_n = s_no[s_no['is_winner']]['pnl_dollars'].sum() if len(s_no) > 0 else 0
        gl_n = abs(s_no[~s_no['is_winner']]['pnl_dollars'].sum()) if len(s_no) > 0 else 0.01
        pf_n = gw_n / gl_n if gl_n > 0 else 0
        print(f"    {tol:>10d}pt {len(s_yes):>7d} {wr_yes:>9.1f}% {wr_no:>7.1f}% {pf_y:>9.2f} {pf_n:>7.2f}")


# ============================================================================
# 4. BOUNCE STRENGTH — Weak vs strong retest
# ============================================================================
print(f"\n{'='*100}")
print(f"  4. RETEST BOUNCE STRENGTH")
print(f"     Is a weak bounce (slow, low volume) better than a strong bounce?")
print(f"     Weak bounce = more likely continuation; Strong bounce = breakout risk")
print(f"{'='*100}")

for target_mode in ['opposite_va', '2R']:
    all_results = []
    for setup in setups:
        r = replay_with_quality(setup, retracement_pct=1.0, stop_mode='va_edge',
                                target_mode=target_mode)
        if r:
            all_results.append(r)

    if not all_results:
        continue

    df_r = pd.DataFrame(all_results)
    print(f"\n  ── Target: {target_mode} ({len(df_r)} trades) ──")

    # a) Time to retest
    print(f"\n    Bars from acceptance to retest (time to form double top/bottom):")
    print(f"      Mean:   {df_r['bars_to_retest'].mean():.0f} bars")
    print(f"      Median: {df_r['bars_to_retest'].median():.0f} bars")
    for pct in [25, 50, 75, 90]:
        print(f"      {pct}th pctile: {np.percentile(df_r['bars_to_retest'], pct):.0f} bars")

    # Segment by speed: fast retest (<30 bars) vs slow (>30 bars)
    fast = df_r[df_r['bars_to_retest'] <= 30]
    med = df_r[(df_r['bars_to_retest'] > 30) & (df_r['bars_to_retest'] <= 60)]
    slow = df_r[df_r['bars_to_retest'] > 60]

    print(f"\n    {'Speed':>15s} {'N':>4s} {'WR':>6s} {'PF':>6s} {'$/Mo':>9s}")
    print(f"    {'-'*50}")
    for label, subset in [('Fast (≤30 bars)', fast), ('Med (31-60 bars)', med),
                          ('Slow (>60 bars)', slow)]:
        if len(subset) < 2:
            print(f"    {label:>15s} {len(subset):>4d}  (too few)")
            continue
        wr = subset['is_winner'].mean() * 100
        gw = subset[subset['is_winner']]['pnl_dollars'].sum()
        gl = abs(subset[~subset['is_winner']]['pnl_dollars'].sum())
        pf = gw / gl if gl > 0 else float('inf')
        pm = subset['pnl_dollars'].sum() / months
        print(f"    {label:>15s} {len(subset):>4d} {wr:>5.1f}% {pf:>5.2f} ${pm:>7,.0f}")

    # b) Volume ratio (retest volume / candle volume)
    print(f"\n    Volume ratio (retest volume / candle volume):")
    print(f"      Mean:   {df_r['vol_ratio'].mean():.2f}")
    print(f"      Median: {df_r['vol_ratio'].median():.2f}")

    low_vol = df_r[df_r['vol_ratio'] <= df_r['vol_ratio'].median()]
    high_vol = df_r[df_r['vol_ratio'] > df_r['vol_ratio'].median()]

    print(f"\n    {'Volume':>20s} {'N':>4s} {'WR':>6s} {'PF':>6s} {'$/Mo':>9s}")
    print(f"    {'-'*55}")
    for label, subset in [('Low vol retest', low_vol), ('High vol retest', high_vol)]:
        if len(subset) < 2:
            print(f"    {label:>20s} {len(subset):>4d}  (too few)")
            continue
        wr = subset['is_winner'].mean() * 100
        gw = subset[subset['is_winner']]['pnl_dollars'].sum()
        gl = abs(subset[~subset['is_winner']]['pnl_dollars'].sum())
        pf = gw / gl if gl > 0 else float('inf')
        pm = subset['pnl_dollars'].sum() / months
        print(f"    {label:>20s} {len(subset):>4d} {wr:>5.1f}% {pf:>5.2f} ${pm:>7,.0f}")

    # c) Delta on the retest move (is the bounce absorbed by aggressive sellers?)
    print(f"\n    Retest delta (total delta during retest move):")
    print(f"      Mean:   {df_r['retest_delta'].mean():.0f}")
    print(f"      Median: {df_r['retest_delta'].median():.0f}")

    # For SHORT: negative retest delta = sellers absorbing the bounce = good
    # For LONG: positive retest delta = buyers absorbing the pullback = good
    confirming = df_r.apply(lambda r: (r['direction'] == 'SHORT' and r['retest_delta'] < 0) or
                                       (r['direction'] == 'LONG' and r['retest_delta'] > 0), axis=1)
    conf_trades = df_r[confirming]
    anti_trades = df_r[~confirming]

    print(f"\n    {'Delta Filter':>25s} {'N':>4s} {'WR':>6s} {'PF':>6s} {'$/Mo':>9s}")
    print(f"    {'-'*60}")
    for label, subset in [('Confirming delta', conf_trades),
                          ('Against delta', anti_trades)]:
        if len(subset) < 2:
            print(f"    {label:>25s} {len(subset):>4d}  (too few)")
            continue
        wr = subset['is_winner'].mean() * 100
        gw = subset[subset['is_winner']]['pnl_dollars'].sum()
        gl = abs(subset[~subset['is_winner']]['pnl_dollars'].sum())
        pf = gw / gl if gl > 0 else float('inf')
        pm = subset['pnl_dollars'].sum() / months
        print(f"    {label:>25s} {len(subset):>4d} {wr:>5.1f}% {pf:>5.2f} ${pm:>7,.0f}")


# ============================================================================
# 5. COMBINED FILTER — Best quality combo
# ============================================================================
print(f"\n{'='*100}")
print(f"  5. COMBINED QUALITY FILTERS")
print(f"     Testing combinations of depth, CVD, sweep, and bounce strength")
print(f"{'='*100}")

for target_mode in ['opposite_va', '2R']:
    all_results = []
    for setup in setups:
        r = replay_with_quality(setup, retracement_pct=1.0, stop_mode='va_edge',
                                target_mode=target_mode)
        if r:
            all_results.append(r)

    if not all_results:
        continue

    df_r = pd.DataFrame(all_results)
    print(f"\n  ── Target: {target_mode} ({len(df_r)} trades) ──")

    # Build confirming delta flag
    df_r['confirming_delta'] = df_r.apply(
        lambda r: (r['direction'] == 'SHORT' and r['retest_delta'] < 0) or
                  (r['direction'] == 'LONG' and r['retest_delta'] > 0), axis=1)

    combos = [
        ('No filter (baseline)', df_r),
        ('Depth >= 0.30', df_r[df_r['candle_close_depth'] >= 0.30]),
        ('Depth >= 0.40', df_r[df_r['candle_close_depth'] >= 0.40]),
        ('CVD divergence', df_r[df_r['cvd_divergence'] == True]),
        ('Confirming delta', df_r[df_r['confirming_delta'] == True]),
        ('Sweep (>5pt)', df_r[df_r['did_sweep'] == True]),
        ('Depth>=0.30 + CVD div', df_r[(df_r['candle_close_depth'] >= 0.30) & (df_r['cvd_divergence'] == True)]),
        ('Depth>=0.40 + CVD div', df_r[(df_r['candle_close_depth'] >= 0.40) & (df_r['cvd_divergence'] == True)]),
        ('Depth>=0.30 + conf delta', df_r[(df_r['candle_close_depth'] >= 0.30) & (df_r['confirming_delta'] == True)]),
        ('Depth>=0.40 + conf delta', df_r[(df_r['candle_close_depth'] >= 0.40) & (df_r['confirming_delta'] == True)]),
        ('Sweep + CVD div', df_r[(df_r['did_sweep'] == True) & (df_r['cvd_divergence'] == True)]),
        ('Depth>=0.30 + sweep', df_r[(df_r['candle_close_depth'] >= 0.30) & (df_r['did_sweep'] == True)]),
        ('Depth>=0.40 + sweep + CVD', df_r[(df_r['candle_close_depth'] >= 0.40) &
                                            (df_r['did_sweep'] == True) &
                                            (df_r['cvd_divergence'] == True)]),
    ]

    print(f"\n    {'Filter':>35s} {'N':>4s} {'Trd/Mo':>7s} {'WR':>6s} {'PF':>6s} "
          f"{'$/Mo':>9s} {'Stopped':>7s} {'Target':>7s}")
    print(f"    {'-'*90}")

    for label, subset in combos:
        if len(subset) < 2:
            print(f"    {label:>35s} {len(subset):>4d}  (too few)")
            continue
        wr = subset['is_winner'].mean() * 100
        gw = subset[subset['is_winner']]['pnl_dollars'].sum()
        gl = abs(subset[~subset['is_winner']]['pnl_dollars'].sum())
        pf = gw / gl if gl > 0 else float('inf')
        pm = subset['pnl_dollars'].sum() / months
        stopped = (subset['exit_reason'] == 'STOP').sum()
        tgt = (subset['exit_reason'] == 'TARGET').sum()
        print(f"    {label:>35s} {len(subset):>4d} {len(subset)/months:>6.1f} "
              f"{wr:>5.1f}% {pf:>5.2f} ${pm:>7,.0f} {stopped:>6d} {tgt:>6d}")


# ============================================================================
# 6. TRADE LOG — Best combo
# ============================================================================
print(f"\n{'='*100}")
print(f"  6. TRADE LOG — 100% retest + VA edge stop + opposite_va (all trades)")
print(f"{'='*100}")

all_results = []
for setup in setups:
    r = replay_with_quality(setup, retracement_pct=1.0, stop_mode='va_edge',
                            target_mode='opposite_va')
    if r:
        all_results.append(r)

df_r = pd.DataFrame(all_results)
print(f"\n  {'Date':12s} {'Dir':5s} {'Entry':>8s} {'Stop':>8s} {'Target':>8s} "
      f"{'Exit':>8s} {'Rsn':5s} {'P&L':>8s} {'Depth':>5s} {'CVDdiv':>6s} "
      f"{'Sweep':>5s} {'RstDlt':>7s} {'Bars':>4s} {'VolR':>5s}")
print(f"  {'-'*115}")

for _, t in df_r.sort_values('session_date').iterrows():
    cvd_flag = 'YES' if t['cvd_divergence'] else 'no'
    sweep_flag = f"{t['sweep_pts']:.0f}p" if t['did_sweep'] else '-'
    print(f"  {t['session_date']:12s} {t['direction']:5s} "
          f"{t['entry_price']:>8.1f} {t['stop_price']:>8.1f} {t['target']:>8.1f} "
          f"{t['exit_price']:>8.1f} {t['exit_reason']:5s} "
          f"${t['pnl_dollars']:>7,.0f} {t['candle_close_depth']:>4.2f} {cvd_flag:>6s} "
          f"{sweep_flag:>5s} {t['retest_delta']:>6.0f} {t['bars_to_retest']:>4.0f} "
          f"{t['vol_ratio']:>4.2f}")


# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*100}")
print(f"  SUMMARY — CANDLE QUALITY ANALYSIS")
print(f"{'='*100}")

print(f"""
  DATA: 259 sessions, {len(setups)} 80P setups, 100% retest entry with VA edge stop

  1. CANDLE CLOSE DEPTH:
     - Mean depth: {np.mean(depths):.2f} (0=VA edge, 1=opposite VA)
     - Deeper close into VA = candle has more conviction
     - Key question: Does depth > 0.40 filter improve WR?

  2. CVD DIVERGENCE:
     - Measures if order flow confirms or diverges from price during the retest
     - SHORT: price bounces up to retest high, but CVD declining = sellers absorbing
     - LONG: price pulls back to retest low, but CVD rising = buyers absorbing

  3. SWEEP:
     - Does the retest slightly overshoot the candle extreme (liquidity grab)?
     - Sweep = smart money grabbing stops before reversing

  4. BOUNCE STRENGTH:
     - Velocity (pts/bar), volume ratio, and delta on the retest move
     - Weak bounce with confirming delta = best setup

  See combined filter section for actionable trading rules.
""")

print(f"{'='*100}")
print(f"  ANALYSIS COMPLETE")
print(f"{'='*100}")
