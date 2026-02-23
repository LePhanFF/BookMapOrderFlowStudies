"""
80P 5-Min Acceptance Model
===========================

Price overshoots previous day VA, gets rejected, closes back inside.
Instead of waiting for 1x 30-min acceptance (slow, misses the move):
  - Use 2x 5-min consecutive closes inside VA (fast confirmation)
  - Enter immediately with tight stop
  - Target: POC / mid VA / opposite VA

Compare:
  A) Immediate entry on 5-min acceptance (1x, 2x, 3x 5-min)
  B) Wait for shallow retracement + double top/sweep after acceptance
  C) Current 30-min acceptance model (baseline)

Also test different acceptance periods: 1x5min, 2x5min, 3x5min, 1x15min, 2x15min
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
from indicators.value_area import compute_session_value_areas

INSTRUMENT = get_instrument('MNQ')
TICK_VALUE = INSTRUMENT.tick_value / INSTRUMENT.tick_size
SLIPPAGE_PTS = 0.50
COMMISSION = 1.24
CONTRACTS = 5
MIN_VA_WIDTH = 25.0
ENTRY_CUTOFF = time(13, 0)

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
# BUILD AGGREGATED BARS (5-min, 15-min) FROM 1-MIN DATA
# ============================================================================
def aggregate_bars(post_ib, period_min):
    """Aggregate 1-min bars into N-min OHLCV bars."""
    agg_bars = []
    for start in range(0, len(post_ib), period_min):
        end = min(start + period_min, len(post_ib))
        chunk = post_ib.iloc[start:end]
        if len(chunk) == 0:
            continue
        agg_bars.append({
            'bar_start': start,
            'bar_end': end - 1,
            'open': chunk.iloc[0]['open'],
            'high': chunk['high'].max(),
            'low': chunk['low'].min(),
            'close': chunk.iloc[-1]['close'],
            'volume': chunk['volume'].sum(),
            'vol_delta': chunk['vol_delta'].sum() if 'vol_delta' in chunk.columns else 0,
            'timestamp': chunk.iloc[-1]['timestamp'],
        })
    return agg_bars


# ============================================================================
# FIND ACCEPTANCE ON N-MIN BARS
# ============================================================================
def find_acceptance(agg_bars, val, vah, direction, n_periods=2):
    """
    Find first N consecutive period closes inside VA.
    Returns the bar index of the last acceptance bar, or None.
    """
    consecutive = 0
    for i, bar in enumerate(agg_bars):
        bt = bar['timestamp']
        bt_time = bt.time() if hasattr(bt, 'time') else None
        if bt_time and bt_time >= ENTRY_CUTOFF:
            break

        close = bar['close']
        is_inside = val <= close <= vah

        if is_inside:
            consecutive += 1
        else:
            consecutive = 0

        if consecutive >= n_periods:
            return i

    return None


# ============================================================================
# REPLAY TRADE FROM ACCEPTANCE
# ============================================================================
def replay_from_acceptance(post_ib, entry_bar_1min, entry_price, direction,
                           vah, val, poc, va_width, stop_buffer, target_mode):
    """Replay a trade from a 1-min bar index."""
    # Stop
    if direction == 'LONG':
        stop_price = val - stop_buffer
    else:
        stop_price = vah + stop_buffer

    risk_pts = abs(entry_price - stop_price)
    if risk_pts <= 0:
        return None

    # Target
    mid_va = (vah + val) / 2.0
    if target_mode == 'opposite_va':
        target = vah if direction == 'LONG' else val
    elif target_mode == 'poc':
        target = poc
    elif target_mode == 'mid_va':
        target = mid_va
    elif target_mode.endswith('R'):
        r_mult = float(target_mode[:-1])
        target = entry_price + risk_pts * r_mult if direction == 'LONG' else entry_price - risk_pts * r_mult
    else:
        target = poc

    reward = (target - entry_price) if direction == 'LONG' else (entry_price - target)
    if reward <= 0:
        return None

    # Replay from 1-min bar
    remaining = post_ib.iloc[entry_bar_1min:]
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

        fav = (bar['high'] - entry_price) if direction == 'LONG' else (entry_price - bar['low'])
        adv = (entry_price - bar['low']) if direction == 'LONG' else (bar['high'] - entry_price)
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

    pnl_pts = (exit_price - entry_price - SLIPPAGE_PTS) if direction == 'LONG' else (entry_price - exit_price - SLIPPAGE_PTS)
    pnl_dollars = pnl_pts * TICK_VALUE * CONTRACTS - COMMISSION * CONTRACTS

    return {
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'pnl_pts': pnl_pts,
        'pnl_dollars': pnl_dollars,
        'risk_pts': risk_pts,
        'reward_pts': reward,
        'rr_ratio': reward / risk_pts if risk_pts > 0 else 0,
        'mfe_pts': mfe,
        'mae_pts': mae,
        'bars_held': bars_held,
        'is_winner': pnl_dollars > 0,
    }


# ============================================================================
# REPLAY RETRACEMENT ENTRY AFTER ACCEPTANCE
# ============================================================================
def replay_retracement_after_acceptance(post_ib, acc_bar_1min, direction,
                                         vah, val, poc, va_width,
                                         acceptance_candle_high, acceptance_candle_low,
                                         acceptance_candle_range,
                                         retracement_pct, stop_buffer, target_mode):
    """
    After acceptance, wait for price to retrace the acceptance candle.
    Enter on limit at retracement level.
    Stop at VA edge (not candle extreme — we know that fails).
    """
    ch = acceptance_candle_high
    cl = acceptance_candle_low
    cr = acceptance_candle_range

    if cr < 2:
        return None

    # Compute limit
    if direction == 'LONG':
        limit_price = ch - retracement_pct * cr
    else:
        limit_price = cl + retracement_pct * cr

    # Search for fill after acceptance
    entry_bar = None
    for bar_idx in range(acc_bar_1min + 1, len(post_ib)):
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

    return replay_from_acceptance(post_ib, entry_bar, limit_price, direction,
                                   vah, val, poc, va_width, stop_buffer, target_mode)


# ============================================================================
# MAIN ANALYSIS
# ============================================================================
print("\nFinding all sessions with open outside prior VA...")

all_configs = []

# Acceptance period configs: (period_minutes, n_periods, label)
acceptance_configs = [
    (5, 1, '1x5min'),
    (5, 2, '2x5min'),
    (5, 3, '3x5min'),
    (15, 1, '1x15min'),
    (15, 2, '2x15min'),
    (30, 1, '1x30min'),
]

target_modes = ['poc', 'mid_va', 'opposite_va', '1R', '2R', '4R']
stop_buffers = [5, 10, 15, 20]


# ============================================================================
# PART 1: IMMEDIATE ENTRY ON ACCEPTANCE
# ============================================================================
print(f"\n{'='*130}")
print(f"  PART 1 — IMMEDIATE ENTRY ON N-MIN ACCEPTANCE")
print(f"  Enter at the acceptance candle close. No retracement. Tight stop.")
print(f"{'='*130}")

for acc_period, acc_n, acc_label in acceptance_configs:
    for target_mode in target_modes:
        for stop_buf in stop_buffers:
            results = []

            for i in range(1, len(sessions)):
                current = sessions[i]
                prior = sessions[i - 1]
                prior_key = str(prior)

                if prior_key not in eth_va:
                    continue
                prior_va = eth_va[prior_key]
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

                post_ib = session_df.iloc[60:].reset_index(drop=True)
                if len(post_ib) < 60:
                    continue

                # Aggregate bars
                agg = aggregate_bars(post_ib, acc_period)

                # Find acceptance
                acc_idx = find_acceptance(agg, val, vah, direction, acc_n)
                if acc_idx is None:
                    continue

                # Entry at the acceptance candle close (1-min bar index)
                acc_bar_end = agg[acc_idx]['bar_end']
                entry_price = agg[acc_idx]['close']

                r = replay_from_acceptance(post_ib, acc_bar_end, entry_price, direction,
                                           vah, val, poc, va_width, stop_buf, target_mode)
                if r:
                    r['session_date'] = str(current)
                    r['direction'] = direction
                    r['entry_price'] = entry_price
                    r['acc_label'] = acc_label
                    results.append(r)

            if len(results) >= 3:
                all_configs.append({
                    'label': f"{acc_label} | stop={stop_buf}pt | tgt={target_mode}",
                    'acc_label': acc_label,
                    'target': target_mode,
                    'stop_buf': stop_buf,
                    'results': results,
                })


# Print summary by acceptance period
print(f"\n  Acceptance Period Comparison (stop=10pt):")
print(f"\n    {'Acceptance':<12s} {'Target':<14s} {'N':>4s} {'Trd/Mo':>7s} {'WR':>6s} {'PF':>6s} "
      f"{'$/Mo':>9s} {'AvgWin':>8s} {'AvgLoss':>8s} {'Risk':>5s} {'S/T/E':>10s}")
print(f"    {'-'*110}")

for target_mode in ['poc', 'mid_va', 'opposite_va', '2R', '4R']:
    for acc_period, acc_n, acc_label in acceptance_configs:
        match = [c for c in all_configs if c['acc_label'] == acc_label
                 and c['target'] == target_mode and c['stop_buf'] == 10]
        if not match:
            continue
        c = match[0]
        df_r = pd.DataFrame(c['results'])
        n = len(df_r)
        wr = df_r['is_winner'].mean() * 100
        gw = df_r[df_r['is_winner']]['pnl_dollars'].sum()
        gl = abs(df_r[~df_r['is_winner']]['pnl_dollars'].sum())
        pf = gw / gl if gl > 0 else float('inf')
        pm = df_r['pnl_dollars'].sum() / months
        avg_w = df_r[df_r['is_winner']]['pnl_dollars'].mean() if df_r['is_winner'].any() else 0
        avg_l = df_r[~df_r['is_winner']]['pnl_dollars'].mean() if (~df_r['is_winner']).any() else 0
        avg_risk = df_r['risk_pts'].mean()
        stopped = (df_r['exit_reason'] == 'STOP').sum()
        target_hit = (df_r['exit_reason'] == 'TARGET').sum()
        eod = (df_r['exit_reason'] == 'EOD').sum()

        print(f"    {acc_label:<12s} {target_mode:<14s} {n:>4d} {n/months:>6.1f} "
              f"{wr:>5.1f}% {pf:>5.2f} ${pm:>7,.0f} ${avg_w:>6,.0f} ${avg_l:>6,.0f} "
              f"{avg_risk:>4.0f}p {stopped:>3d}/{target_hit:>3d}/{eod:>3d}")
    print()


# ============================================================================
# PART 2: BEST STOP BUFFER FOR EACH ACCEPTANCE PERIOD
# ============================================================================
print(f"\n{'='*130}")
print(f"  PART 2 — STOP BUFFER OPTIMIZATION (by acceptance period)")
print(f"{'='*130}")

for acc_label_filter in ['2x5min', '1x30min']:
    for target_mode in ['poc', 'mid_va', 'opposite_va', '2R']:
        print(f"\n  {acc_label_filter} | {target_mode}:")
        print(f"    {'Stop':>6s} {'N':>4s} {'WR':>6s} {'PF':>6s} {'$/Mo':>9s} {'Risk':>5s}")
        print(f"    {'-'*45}")
        for stop_buf in stop_buffers:
            match = [c for c in all_configs if c['acc_label'] == acc_label_filter
                     and c['target'] == target_mode and c['stop_buf'] == stop_buf]
            if not match:
                continue
            df_r = pd.DataFrame(match[0]['results'])
            n = len(df_r)
            wr = df_r['is_winner'].mean() * 100
            gw = df_r[df_r['is_winner']]['pnl_dollars'].sum()
            gl = abs(df_r[~df_r['is_winner']]['pnl_dollars'].sum())
            pf = gw / gl if gl > 0 else float('inf')
            pm = df_r['pnl_dollars'].sum() / months
            avg_risk = df_r['risk_pts'].mean()
            print(f"    {stop_buf:>4d}pt {n:>4d} {wr:>5.1f}% {pf:>5.2f} ${pm:>7,.0f} {avg_risk:>4.0f}p")


# ============================================================================
# PART 3: RETRACEMENT AFTER 5-MIN ACCEPTANCE
# ============================================================================
print(f"\n{'='*130}")
print(f"  PART 3 — RETRACEMENT ENTRY AFTER 2x5MIN ACCEPTANCE")
print(f"  After 2x5min acceptance, wait for retracement of the acceptance candle pair")
print(f"  Compare: immediate vs 50%/75%/100% retracement")
print(f"{'='*130}")

for target_mode in ['poc', 'mid_va', 'opposite_va', '2R']:
    print(f"\n  ━━ Target: {target_mode} ━━")
    print(f"    {'Entry':<25s} {'N':>4s} {'Trd/Mo':>7s} {'WR':>6s} {'PF':>6s} "
          f"{'$/Mo':>9s} {'AvgWin':>8s} {'AvgLoss':>8s} {'Risk':>5s}")
    print(f"    {'-'*95}")

    for retr_pct_val in [0.0, 0.50, 0.75, 1.0]:
        for stop_buf in [10]:
            results = []

            for i in range(1, len(sessions)):
                current = sessions[i]
                prior = sessions[i - 1]
                prior_key = str(prior)

                if prior_key not in eth_va:
                    continue
                prior_va = eth_va[prior_key]
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

                post_ib = session_df.iloc[60:].reset_index(drop=True)
                if len(post_ib) < 60:
                    continue

                # 2x5min acceptance
                agg = aggregate_bars(post_ib, 5)
                acc_idx = find_acceptance(agg, val, vah, direction, 2)
                if acc_idx is None:
                    continue

                acc_bar_end = agg[acc_idx]['bar_end']

                if retr_pct_val == 0.0:
                    # Immediate entry
                    entry_price = agg[acc_idx]['close']
                    r = replay_from_acceptance(post_ib, acc_bar_end, entry_price, direction,
                                               vah, val, poc, va_width, stop_buf, target_mode)
                else:
                    # Get the acceptance candle range (last 2 x 5-min = 10 min)
                    acc_start_bar = agg[acc_idx - 1]['bar_start'] if acc_idx > 0 else agg[acc_idx]['bar_start']
                    candle_bars = post_ib.iloc[acc_start_bar:acc_bar_end + 1]
                    ch = candle_bars['high'].max()
                    cl = candle_bars['low'].min()
                    cr = ch - cl

                    r = replay_retracement_after_acceptance(
                        post_ib, acc_bar_end, direction, vah, val, poc, va_width,
                        ch, cl, cr, retr_pct_val, stop_buf, target_mode)

                if r:
                    r['session_date'] = str(current)
                    r['direction'] = direction
                    results.append(r)

            if len(results) < 2:
                label = f"{'Immediate' if retr_pct_val == 0 else f'{retr_pct_val*100:.0f}% retest'}"
                print(f"    {label:<25s} {len(results):>4d}  (too few)")
                continue

            df_r = pd.DataFrame(results)
            n = len(df_r)
            wr = df_r['is_winner'].mean() * 100
            gw = df_r[df_r['is_winner']]['pnl_dollars'].sum()
            gl = abs(df_r[~df_r['is_winner']]['pnl_dollars'].sum())
            pf = gw / gl if gl > 0 else float('inf')
            pm = df_r['pnl_dollars'].sum() / months
            avg_w = df_r[df_r['is_winner']]['pnl_dollars'].mean() if df_r['is_winner'].any() else 0
            avg_l = df_r[~df_r['is_winner']]['pnl_dollars'].mean() if (~df_r['is_winner']).any() else 0
            avg_risk = df_r['risk_pts'].mean()

            label = f"{'Immediate' if retr_pct_val == 0 else f'{retr_pct_val*100:.0f}% retest'}"
            print(f"    {label:<25s} {n:>4d} {n/months:>6.1f} "
                  f"{wr:>5.1f}% {pf:>5.2f} ${pm:>7,.0f} ${avg_w:>6,.0f} ${avg_l:>6,.0f} "
                  f"{avg_risk:>4.0f}p")


# ============================================================================
# PART 4: ENTRY TIMING — HOW FAST IS EACH ACCEPTANCE?
# ============================================================================
print(f"\n{'='*130}")
print(f"  PART 4 — ENTRY TIMING: When does each acceptance trigger?")
print(f"{'='*130}")

for acc_period, acc_n, acc_label in acceptance_configs:
    entry_bars = []
    for i in range(1, len(sessions)):
        current = sessions[i]
        prior = sessions[i - 1]
        prior_key = str(prior)

        if prior_key not in eth_va:
            continue
        prior_va = eth_va[prior_key]
        if prior_va.va_width < MIN_VA_WIDTH:
            continue

        session_df = df_rth[df_rth['session_date'] == current].reset_index(drop=True)
        if len(session_df) < 90:
            continue

        open_price = session_df['open'].iloc[0]
        if open_price >= prior_va.val and open_price <= prior_va.vah:
            continue  # inside VA

        direction = 'LONG' if open_price < prior_va.val else 'SHORT'
        post_ib = session_df.iloc[60:].reset_index(drop=True)
        if len(post_ib) < 60:
            continue

        agg = aggregate_bars(post_ib, acc_period)
        acc_idx = find_acceptance(agg, prior_va.val, prior_va.vah, direction, acc_n)
        if acc_idx is not None:
            entry_bar_1min = agg[acc_idx]['bar_end']
            entry_bars.append(entry_bar_1min)

    if entry_bars:
        arr = np.array(entry_bars)
        print(f"\n  {acc_label}: {len(entry_bars)} triggers")
        print(f"    Entry time (bars after IB): mean={arr.mean():.0f}, median={np.median(arr):.0f}, "
              f"25th={np.percentile(arr, 25):.0f}, 75th={np.percentile(arr, 75):.0f}")
        print(f"    That's ~{arr.mean():.0f} min after IB end (10:30 ET), "
              f"so average entry at ~{10*60+30+arr.mean():.0f} min = "
              f"{int((10*60+30+arr.mean())//60)}:{int((10*60+30+arr.mean())%60):02d} ET")


# ============================================================================
# PART 5: HEAD-TO-HEAD — ALL MODELS RANKED
# ============================================================================
print(f"\n{'='*130}")
print(f"  PART 5 — HEAD-TO-HEAD: All Models Ranked by $/Month")
print(f"{'='*130}")

# Collect best configs
ranked = []
for c in all_configs:
    df_r = pd.DataFrame(c['results'])
    n = len(df_r)
    wr = df_r['is_winner'].mean() * 100
    gw = df_r[df_r['is_winner']]['pnl_dollars'].sum()
    gl = abs(df_r[~df_r['is_winner']]['pnl_dollars'].sum())
    pf = gw / gl if gl > 0 else float('inf')
    pm = df_r['pnl_dollars'].sum() / months
    avg_risk = df_r['risk_pts'].mean()
    ranked.append({
        'label': c['label'],
        'n': n, 'tpm': n / months, 'wr': wr, 'pf': pf, 'pm': pm, 'risk': avg_risk,
    })

ranked.sort(key=lambda x: x['pm'], reverse=True)

print(f"\n  {'Rank':>4s} {'Config':<50s} {'N':>4s} {'Trd/Mo':>7s} {'WR':>6s} {'PF':>6s} "
      f"{'$/Mo':>9s} {'Risk':>5s}")
print(f"  {'-'*100}")

for rank, c in enumerate(ranked[:30], 1):
    print(f"  {rank:>4d} {c['label']:<50s} {c['n']:>4d} {c['tpm']:>6.1f} "
          f"{c['wr']:>5.1f}% {c['pf']:>5.2f} ${c['pm']:>7,.0f} {c['risk']:>4.0f}p")


# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*130}")
print(f"  SUMMARY")
print(f"{'='*130}")

# Find best 2x5min config
best_2x5 = [c for c in ranked if '2x5min' in c['label']]
best_1x30 = [c for c in ranked if '1x30min' in c['label']]

if best_2x5:
    b = best_2x5[0]
    print(f"\n  BEST 2x5min config:")
    print(f"    {b['label']}")
    print(f"    {b['n']} trades, {b['tpm']:.1f}/mo, {b['wr']:.1f}% WR, PF {b['pf']:.2f}, ${b['pm']:,.0f}/mo")

if best_1x30:
    b = best_1x30[0]
    print(f"\n  BEST 1x30min config (baseline):")
    print(f"    {b['label']}")
    print(f"    {b['n']} trades, {b['tpm']:.1f}/mo, {b['wr']:.1f}% WR, PF {b['pf']:.2f}, ${b['pm']:,.0f}/mo")

print(f"""
  COMPARE TO PRIOR MODELS:
    Current (1x30min + 4R + 10pt stop):  60 trades, 38.3% WR, PF 1.70, $955/mo
    Immediate VA touch (10pt stop):      80 trades, 31.2% WR, PF 4.53, $1,836/mo
    Limit 50% VA + 4R:                   47 trades, 44.7% WR, PF 2.57, $1,922/mo

  KEY QUESTIONS:
    1. Does 2x5min beat 1x30min? (faster entry, catch more of the move)
    2. Does targeting POC/mid work better than opposite_va? (realistic target)
    3. Immediate vs retracement after 5-min acceptance?
    4. How many minutes earlier does 2x5min fire vs 1x30min?
""")

print(f"{'='*130}")
print(f"  5-MIN ACCEPTANCE ANALYSIS COMPLETE")
print(f"{'='*130}")
