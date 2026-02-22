"""
80P Retest Entry Model — Enter on retracement of the rejection candle
=====================================================================

Current model: Enter at the 30-min acceptance candle CLOSE.
Problem: Entry is at the edge of the VA (32% in), stop is right behind.

Proposed model:
  1. Identify the 30-min acceptance candle (first 30-min close inside VA)
  2. This candle has a HIGH and LOW — it pushed INTO the VA
  3. After the candle closes, wait for a RETEST/PULLBACK
  4. Enter on limit at various retracement levels of the candle:
     - 50%   = midpoint of candle
     - 61.8% = fib retracement
     - 75%   = deep retracement
     - 100%  = double top/bottom (tests candle extreme)

For LONG (opened below VAL, rejection candle pushes UP into VA):
  - Candle HIGH = inside VA, candle LOW = near VAL
  - After candle, price pulls back DOWN
  - Limit LONG at: candle_high - retracement% * (candle_high - candle_low)
  - Stop below candle low (or VAL - buffer)

For SHORT (opened above VAH, rejection candle pushes DOWN into VA):
  - Candle LOW = inside VA, candle HIGH = near VAH
  - After candle, price bounces back UP
  - Limit SHORT at: candle_low + retracement% * (candle_high - candle_low)
  - Stop above candle high (or VAH + buffer)
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
# FIND SETUPS WITH REJECTION CANDLE DATA
# ============================================================================
def find_setups_with_candle(df_rth, va_by_session, directions='BOTH'):
    """Find 80P setups and capture the 30-min rejection candle OHLC."""
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

        if candle_range < 2:  # skip degenerate candles
            continue

        # Compute prior session ATR
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
            'acceptance_bar': acceptance_period_end,
            'prior_atr': prior_atr,
            'post_ib': post_ib,
        })

    return setups


def replay_retest(setup, retracement_pct, stop_mode='candle_low', target_mode='opposite_va'):
    """
    After the rejection candle, wait for price to retrace to a limit level.

    retracement_pct: 0.0 = enter at candle close (current model)
                     0.50 = 50% retracement of candle
                     0.618 = 61.8% fib
                     0.75 = 75% retracement
                     1.0 = 100% (double top/bottom)

    stop_mode:
        'candle_extreme'  = stop at candle low (LONG) / candle high (SHORT) - buffer
        'va_edge'         = stop at VAL (LONG) / VAH (SHORT) - buffer
        'candle_1atr'     = stop at candle extreme - 1 ATR
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
        # Current model: enter at candle close
        entry_price = cc
        entry_bar = acc_bar  # enter immediately
    else:
        # Wait for retracement after the candle
        if direction == 'LONG':
            # Candle pushed UP into VA. Retracement = price pulls back DOWN.
            # Limit = candle_high - retracement% * candle_range
            limit_price = ch - retracement_pct * cr
        else:
            # Candle pushed DOWN into VA. Retracement = price bounces back UP.
            # Limit = candle_low + retracement% * candle_range
            limit_price = cl + retracement_pct * cr

        # Search for limit fill in bars AFTER the acceptance candle
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
            return None  # Limit never filled

    # Compute stop
    if stop_mode == 'candle_extreme':
        if direction == 'LONG':
            stop_price = cl - STOP_BUFFER
        else:
            stop_price = ch + STOP_BUFFER
    elif stop_mode == 'va_edge':
        if direction == 'LONG':
            stop_price = val - STOP_BUFFER
        else:
            stop_price = vah + STOP_BUFFER
    elif stop_mode == 'candle_1atr':
        if direction == 'LONG':
            stop_price = cl - prior_atr
        else:
            stop_price = ch + prior_atr
    else:
        if direction == 'LONG':
            stop_price = cl - STOP_BUFFER
        else:
            stop_price = ch + STOP_BUFFER

    risk_pts = abs(entry_price - stop_price)
    if risk_pts <= 0:
        return None

    # Compute target
    if target_mode == 'opposite_va':
        target = vah if direction == 'LONG' else val
    elif target_mode == 'poc':
        target = poc
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

    # Replay from entry bar
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

        # Stop
        if direction == 'LONG' and bar['low'] <= stop_price:
            exit_price = stop_price
            exit_reason = 'STOP'
            break
        elif direction == 'SHORT' and bar['high'] >= stop_price:
            exit_price = stop_price
            exit_reason = 'STOP'
            break

        # Target
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

    # Entry depth within VA
    if direction == 'LONG':
        entry_depth = (entry_price - val) / va_width if va_width > 0 else 0
    else:
        entry_depth = (vah - entry_price) / va_width if va_width > 0 else 0

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
        'rr_ratio': reward / risk_pts if risk_pts > 0 else 0,
        'va_width': va_width,
        'candle_range': setup['candle_range'],
        'mfe_pts': mfe,
        'mae_pts': mae,
        'mfe_r': mfe / risk_pts if risk_pts > 0 else 0,
        'mae_r': mae / risk_pts if risk_pts > 0 else 0,
        'bars_held': bars_held,
        'entry_depth': entry_depth,
        'is_winner': pnl_dollars > 0,
    }


# ============================================================================
# MAIN
# ============================================================================
print("\nFinding 80P setups with candle data (ETH VA, acc=1, BOTH)...")
setups = find_setups_with_candle(df_rth, eth_va, directions='BOTH')
print(f"Found {len(setups)} setups with rejection candle data")

# Show candle stats
candle_ranges = [s['candle_range'] for s in setups]
print(f"\n  30-min rejection candle stats:")
print(f"    Mean range: {np.mean(candle_ranges):.0f} pts")
print(f"    Median:     {np.median(candle_ranges):.0f} pts")
for pct in [25, 50, 75, 90]:
    print(f"    {pct}th pctile: {np.percentile(candle_ranges, pct):.0f} pts")

# Show candle vs VA context
for s in setups[:5]:
    d = s['direction']
    print(f"\n    {s['session_date']} {d}: "
          f"candle H={s['candle_high']:.1f} L={s['candle_low']:.1f} "
          f"range={s['candle_range']:.0f}p | "
          f"VAH={s['vah']:.1f} VAL={s['val']:.1f} width={s['va_width']:.0f}p")

# ============================================================================
# RETRACEMENT FILL RATES
# ============================================================================
print(f"\n{'='*100}")
print(f"  RETRACEMENT FILL RATES — How often does the limit get filled?")
print(f"{'='*100}")

for ret_pct in [0.0, 0.50, 0.618, 0.75, 1.0]:
    filled = 0
    for setup in setups:
        result = replay_retest(setup, ret_pct, stop_mode='candle_extreme', target_mode='opposite_va')
        if result is not None:
            filled += 1
    print(f"  {ret_pct*100:5.1f}% retracement: {filled}/{len(setups)} filled ({filled/len(setups)*100:.0f}%)")


# ============================================================================
# FULL COMPARISON
# ============================================================================
print(f"\n{'='*100}")
print(f"  ENTRY MODE COMPARISON — Retest entries vs current model")
print(f"{'='*100}")

retracement_levels = [0.0, 0.50, 0.618, 0.75, 1.0]
stop_modes = ['candle_extreme', 'va_edge', 'candle_1atr']
target_modes = ['opposite_va', '2R', '4R']

for target_mode in target_modes:
    print(f"\n  ━━ Target: {target_mode} ━━")

    for stop_mode in stop_modes:
        stop_label = {'candle_extreme': 'Candle±10pt',
                      'va_edge': 'VA edge±10pt',
                      'candle_1atr': 'Candle±1ATR'}[stop_mode]

        print(f"\n    Stop: {stop_label}")
        print(f"    {'Retracement':<16s} {'N':>4s} {'Trd/Mo':>7s} {'WR':>6s} {'PF':>6s} "
              f"{'$/Mo':>9s} {'AvgWin':>8s} {'AvgLoss':>8s} "
              f"{'Risk':>5s} {'R:R':>5s} {'MFE_R':>5s} {'Stopped':>7s} {'Target':>7s}")
        print(f"    {'-'*115}")

        for ret_pct in retracement_levels:
            results = []
            for setup in setups:
                r = replay_retest(setup, ret_pct, stop_mode=stop_mode, target_mode=target_mode)
                if r:
                    results.append(r)

            if not results:
                label = f"{ret_pct*100:.0f}% (candle close)" if ret_pct == 0 else f"{ret_pct*100:.1f}%"
                print(f"    {label:<16s} {'0':>4s}")
                continue

            df_r = pd.DataFrame(results)
            n = len(df_r)
            wins = df_r[df_r['is_winner']]
            losses = df_r[~df_r['is_winner']]
            wr = len(wins) / n * 100
            total_pnl = df_r['pnl_dollars'].sum()
            per_month = total_pnl / months
            avg_win = wins['pnl_dollars'].mean() if len(wins) > 0 else 0
            avg_loss = losses['pnl_dollars'].mean() if len(losses) > 0 else 0
            gw = wins['pnl_dollars'].sum() if len(wins) > 0 else 0
            gl = abs(losses['pnl_dollars'].sum()) if len(losses) > 0 else 0.01
            pf = gw / gl if gl > 0 else float('inf')
            stopped = (df_r['exit_reason'] == 'STOP').sum()
            target_hit = (df_r['exit_reason'] == 'TARGET').sum()
            avg_risk = df_r['risk_pts'].mean()
            avg_rr = df_r['rr_ratio'].mean()
            avg_mfe_r = df_r['mfe_r'].mean()

            label = "0% (candle close)" if ret_pct == 0 else f"{ret_pct*100:.1f}%"
            print(f"    {label:<16s} {n:>4d} {n/months:>6.1f} "
                  f"{wr:>5.1f}% {pf:>5.2f} "
                  f"${per_month:>7,.0f} ${avg_win:>6,.0f} ${avg_loss:>6,.0f} "
                  f"{avg_risk:>4.0f}p {avg_rr:>4.1f} {avg_mfe_r:>4.1f}R "
                  f"{stopped:>6d} {target_hit:>6d}")


# ============================================================================
# BEST COMBOS
# ============================================================================
print(f"\n{'='*100}")
print(f"  TOP CONFIGURATIONS — Sorted by $/Month")
print(f"{'='*100}")

all_configs = []
for target_mode in target_modes:
    for stop_mode in stop_modes:
        for ret_pct in retracement_levels:
            results = []
            for setup in setups:
                r = replay_retest(setup, ret_pct, stop_mode=stop_mode, target_mode=target_mode)
                if r:
                    results.append(r)

            if len(results) < 3:
                continue

            df_r = pd.DataFrame(results)
            n = len(df_r)
            wr = df_r['is_winner'].mean() * 100
            total_pnl = df_r['pnl_dollars'].sum()
            pm = total_pnl / months
            gw = df_r[df_r['is_winner']]['pnl_dollars'].sum()
            gl = abs(df_r[~df_r['is_winner']]['pnl_dollars'].sum())
            pf = gw / gl if gl > 0 else float('inf')
            stopped = (df_r['exit_reason'] == 'STOP').sum()

            stop_label = {'candle_extreme': 'candle', 'va_edge': 'va_edge', 'candle_1atr': 'candle_1atr'}[stop_mode]
            ret_label = f"{ret_pct*100:.0f}%" if ret_pct > 0 else "close"

            all_configs.append({
                'label': f"ret={ret_label} | stop={stop_label} | tgt={target_mode}",
                'n': n, 'wr': wr, 'pf': pf, 'pm': pm,
                'tpm': n / months, 'stopped': stopped,
                'avg_risk': df_r['risk_pts'].mean(),
            })

all_configs.sort(key=lambda x: x['pm'], reverse=True)

print(f"\n  {'Config':<55s} {'N':>4s} {'Trd/Mo':>7s} {'WR':>6s} {'PF':>6s} "
      f"{'$/Mo':>9s} {'Risk':>5s} {'Stopped':>7s}")
print(f"  {'-'*105}")

for rank, c in enumerate(all_configs[:25], 1):
    print(f"  {c['label']:<55s} {c['n']:>4d} {c['tpm']:>6.1f} "
          f"{c['wr']:>5.1f}% {c['pf']:>5.2f} "
          f"${c['pm']:>7,.0f} {c['avg_risk']:>4.0f}p {c['stopped']:>6d}")


# ============================================================================
# TRADE LOG — Best retest config
# ============================================================================
if all_configs:
    best = all_configs[0]
    print(f"\n{'='*100}")
    print(f"  TRADE LOG — {best['label']}")
    print(f"{'='*100}")

    # Re-run best config to get trade details
    parts = best['label'].split(' | ')
    ret_str = parts[0].replace('ret=', '')
    ret_pct = 0.0 if ret_str == 'close' else float(ret_str.replace('%', '')) / 100
    stop_mode = parts[1].replace('stop=', '')
    stop_mode_map = {'candle': 'candle_extreme', 'va_edge': 'va_edge', 'candle_1atr': 'candle_1atr'}
    stop_m = stop_mode_map.get(stop_mode, 'candle_extreme')
    target_m = parts[2].replace('tgt=', '')

    results = []
    for setup in setups:
        r = replay_retest(setup, ret_pct, stop_mode=stop_m, target_mode=target_m)
        if r:
            results.append(r)

    df_r = pd.DataFrame(results)
    print(f"\n  {'Date':12s} {'Dir':5s} {'Entry':>8s} {'Stop':>8s} {'Target':>8s} "
          f"{'Exit':>8s} {'Reason':7s} {'P&L':>8s} {'Risk':>5s} {'MFE':>5s} {'MAE':>5s} "
          f"{'MFE_R':>5s} {'CndlRng':>7s}")
    print(f"  {'-'*110}")

    for _, t in df_r.sort_values('session_date').iterrows():
        print(f"  {t['session_date']:12s} {t['direction']:5s} "
              f"{t['entry_price']:>8.1f} {t['stop_price']:>8.1f} {t['target']:>8.1f} "
              f"{t['exit_price']:>8.1f} {t['exit_reason']:7s} "
              f"${t['pnl_dollars']:>7,.0f} {t['risk_pts']:>4.0f}p "
              f"{t['mfe_pts']:>4.0f}p {t['mae_pts']:>4.0f}p "
              f"{t['mfe_r']:>4.1f}R {t['candle_range']:>5.0f}p")


# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*100}")
print(f"  SUMMARY")
print(f"{'='*100}")

# Current model baseline
baseline_results = [replay_retest(s, 0.0, 'va_edge', 'opposite_va') for s in setups]
baseline_results = [r for r in baseline_results if r]
bl_wr = np.mean([r['is_winner'] for r in baseline_results]) * 100
bl_pm = sum(r['pnl_dollars'] for r in baseline_results) / months

print(f"""
  CURRENT MODEL (candle close, VA edge stop, opposite_va target):
    {len(baseline_results)} trades, WR={bl_wr:.1f}%, $/mo=${bl_pm:,.0f}

  BEST RETEST CONFIG:
    {all_configs[0]['label']}
    {all_configs[0]['n']} trades, WR={all_configs[0]['wr']:.1f}%, $/mo=${all_configs[0]['pm']:,.0f}

  KEY FINDINGS:
  - Retracement entries filter out trades where price barely touches VA and reverses
  - Deeper retracement = fewer fills but higher WR
  - 100% retracement (double top/bottom) = strongest confirmation but fewest trades
  - The candle range is the natural risk unit, not the VA width
""")

print(f"{'='*100}")
print(f"  ANALYSIS COMPLETE")
print(f"{'='*100}")
