"""
Option 3 Conservative Portfolio — Drawdown & Prop Firm Sizing Analysis
======================================================================

Portfolio:
  A) 80P: 100% Retest + VA edge stop + 2R  → $915/mo, 65.7% WR, PF 3.45
  B) VA Edge Fade: Limit edge + Swing + 1R (2nd test) → $813/mo, 70.0% WR, PF 2.25

Question: Can this run on a 1x $150K prop account with $4,500 trailing drawdown?

Analysis:
  1. Replay actual trades chronologically, build equity curve
  2. Compute max drawdown, max consecutive losses, worst day/week
  3. Monte Carlo simulation (10,000 paths) for drawdown distribution
  4. Prop firm sizing: what contract count keeps DD < $4,500 at 95%/99% confidence
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import time, date
from typing import Dict, List, Optional

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.instruments import get_instrument
from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from indicators.value_area import compute_session_value_areas
from indicators.smt_divergence import detect_swing_points

INSTRUMENT = get_instrument('MNQ')
TICK_VALUE = INSTRUMENT.tick_value / INSTRUMENT.tick_size  # $2/pt
SLIPPAGE_PTS = 0.50
COMMISSION = 1.24
CONTRACTS = 5
MIN_VA_WIDTH = 25.0
ENTRY_CUTOFF = time(13, 0)
EOD_EXIT = time(15, 30)

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
# HELPERS
# ============================================================================
def aggregate_bars(bars_df, period_min):
    agg_bars = []
    for start in range(0, len(bars_df), period_min):
        end = min(start + period_min, len(bars_df))
        chunk = bars_df.iloc[start:end]
        if len(chunk) == 0:
            continue
        agg_bars.append({
            'bar_start': start, 'bar_end': end - 1,
            'open': chunk.iloc[0]['open'],
            'high': chunk['high'].max(), 'low': chunk['low'].min(),
            'close': chunk.iloc[-1]['close'],
            'volume': chunk['volume'].sum(),
            'timestamp': chunk.iloc[-1]['timestamp'],
        })
    return agg_bars


# ============================================================================
# STRATEGY A: 80P — 100% Retest + VA Edge Stop + 2R
# ============================================================================
def run_80p_retest(sessions_list, df_rth, eth_va):
    """Replay 80P 100% retest trades and return chronological trade list."""
    trades = []

    for i in range(1, len(sessions_list)):
        current = sessions_list[i]
        prior = sessions_list[i - 1]
        prior_key = str(prior)

        if prior_key not in eth_va:
            continue
        va = eth_va[prior_key]
        if va.va_width < MIN_VA_WIDTH:
            continue

        vah, val, poc = va.vah, va.val, va.poc
        va_width = vah - val

        session_df = df_rth[df_rth['session_date'] == current].reset_index(drop=True)
        if len(session_df) < 90:
            continue

        rth_open = session_df.iloc[0]['open']

        # Must open outside VA
        if val <= rth_open <= vah:
            continue

        direction = 'LONG' if rth_open < val else 'SHORT'

        # Build 30-min bars for acceptance
        bars_30m = aggregate_bars(session_df, 30)

        # Find acceptance: 1x 30-min close inside VA
        acceptance_idx = None
        for bi, bar in enumerate(bars_30m):
            bt = bar['timestamp']
            bt_time = bt.time() if hasattr(bt, 'time') else None
            if bt_time and bt_time >= ENTRY_CUTOFF:
                break
            if val <= bar['close'] <= vah:
                acceptance_idx = bi
                break

        if acceptance_idx is None:
            continue

        acc_bar = bars_30m[acceptance_idx]

        # 100% retest: wait for price to return to candle extreme
        if direction == 'LONG':
            retest_price = acc_bar['low']  # double bottom
        else:
            retest_price = acc_bar['high']  # double top

        # Find fill of limit order
        start_1m = acc_bar['bar_end'] + 1
        fill_bar = None
        for bi in range(start_1m, len(session_df)):
            bar = session_df.iloc[bi]
            bt_time = bar.get('timestamp')
            bt = bt_time.time() if bt_time and hasattr(bt_time, 'time') else None
            if bt and bt >= EOD_EXIT:
                break
            if direction == 'LONG' and bar['low'] <= retest_price:
                fill_bar = bi
                break
            elif direction == 'SHORT' and bar['high'] >= retest_price:
                fill_bar = bi
                break

        if fill_bar is None:
            continue

        entry_price = retest_price

        # Stop: VA edge + 10pt
        if direction == 'LONG':
            stop_price = val - 10
        else:
            stop_price = vah + 10

        risk_pts = abs(entry_price - stop_price)
        if risk_pts <= 0:
            continue

        # Target: 2R
        if direction == 'LONG':
            target_price = entry_price + 2 * risk_pts
        else:
            target_price = entry_price - 2 * risk_pts

        # Replay
        remaining = session_df.iloc[fill_bar:]
        exit_price = None
        exit_reason = None
        for idx in range(len(remaining)):
            bar = remaining.iloc[idx]
            bt_time = bar.get('timestamp')
            bt = bt_time.time() if bt_time and hasattr(bt_time, 'time') else None
            if bt and bt >= EOD_EXIT:
                exit_price = bar['close']
                exit_reason = 'EOD'
                break
            if direction == 'LONG' and bar['low'] <= stop_price:
                exit_price = stop_price
                exit_reason = 'STOP'
                break
            elif direction == 'SHORT' and bar['high'] >= stop_price:
                exit_price = stop_price
                exit_reason = 'STOP'
                break
            if direction == 'LONG' and bar['high'] >= target_price:
                exit_price = target_price
                exit_reason = 'TARGET'
                break
            elif direction == 'SHORT' and bar['low'] <= target_price:
                exit_price = target_price
                exit_reason = 'TARGET'
                break

        if exit_price is None:
            exit_price = remaining.iloc[-1]['close']
            exit_reason = 'EOD'

        pnl_pts = (exit_price - entry_price - SLIPPAGE_PTS) if direction == 'LONG' \
                  else (entry_price - exit_price - SLIPPAGE_PTS)
        pnl_dollars = pnl_pts * TICK_VALUE * CONTRACTS - COMMISSION * CONTRACTS

        trades.append({
            'session_date': str(current),
            'strategy': '80P_retest',
            'direction': direction,
            'entry_price': entry_price,
            'stop_price': stop_price,
            'target_price': target_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'risk_pts': risk_pts,
            'pnl_pts': pnl_pts,
            'pnl_dollars': pnl_dollars,
            'is_winner': pnl_dollars > 0,
        })

    return trades


# ============================================================================
# STRATEGY B: VA Edge Fade — Limit Edge + Swing + 1R (2nd Test)
# ============================================================================
def run_va_edge_fade(sessions_list, df_rth, eth_va):
    """Replay VA Edge Fade limit edge + swing stop + 1R trades."""
    trades = []

    for i in range(1, len(sessions_list)):
        current = sessions_list[i]
        prior = sessions_list[i - 1]
        prior_key = str(prior)

        if prior_key not in eth_va:
            continue
        va = eth_va[prior_key]
        if va.va_width < MIN_VA_WIDTH:
            continue

        vah, val, poc = va.vah, va.val, va.poc

        session_df = df_rth[df_rth['session_date'] == current].reset_index(drop=True)
        if len(session_df) < 90:
            continue

        bars_5m = aggregate_bars(session_df, 5)

        # Track pokes per edge
        vah_pokes = []
        val_pokes = []

        def is_inversion_short(bar):
            return bar['high'] > vah and bar['close'] < vah and bar['close'] < bar['open']

        def is_inversion_long(bar):
            return bar['low'] < val and bar['close'] > val and bar['close'] > bar['open']

        for bi, bar5 in enumerate(bars_5m):
            bt = bar5['timestamp']
            bt_time = bt.time() if hasattr(bt, 'time') else None
            if bt_time and bt_time >= ENTRY_CUTOFF:
                break

            # VAH poke (SHORT)
            if bar5['high'] > vah:
                is_inv = is_inversion_short(bar5)
                is_2x5 = False
                if bi + 1 < len(bars_5m):
                    nb = bars_5m[bi + 1]
                    nt = nb['timestamp'].time() if hasattr(nb['timestamp'], 'time') else None
                    if bar5['close'] < vah and nb['close'] < vah and (not nt or nt < ENTRY_CUTOFF):
                        is_2x5 = True
                if is_inv or is_2x5:
                    vah_pokes.append({
                        'bar_idx_5m': bi, 'bar_end_1m': bar5['bar_end'],
                        'direction': 'SHORT', 'edge': 'VAH',
                        'bar_high': bar5['high'], 'bar_low': bar5['low'],
                        'poke_number': len(vah_pokes) + 1,
                    })

            # VAL poke (LONG)
            if bar5['low'] < val:
                is_inv = is_inversion_long(bar5)
                is_2x5 = False
                if bi + 1 < len(bars_5m):
                    nb = bars_5m[bi + 1]
                    nt = nb['timestamp'].time() if hasattr(nb['timestamp'], 'time') else None
                    if bar5['close'] > val and nb['close'] > val and (not nt or nt < ENTRY_CUTOFF):
                        is_2x5 = True
                if is_inv or is_2x5:
                    val_pokes.append({
                        'bar_idx_5m': bi, 'bar_end_1m': bar5['bar_end'],
                        'direction': 'LONG', 'edge': 'VAL',
                        'bar_high': bar5['high'], 'bar_low': bar5['low'],
                        'poke_number': len(val_pokes) + 1,
                    })

        # Only trade 2nd poke (poke_number == 2)
        second_pokes = [p for p in vah_pokes if p['poke_number'] == 2] + \
                       [p for p in val_pokes if p['poke_number'] == 2]

        for poke in second_pokes:
            direction = poke['direction']

            # Limit at VA edge entry
            if direction == 'SHORT':
                entry_price = vah
            else:
                entry_price = val

            # Find limit fill
            fill_bar = None
            for bi in range(poke['bar_end_1m'] + 1, len(session_df)):
                bar = session_df.iloc[bi]
                bt_time = bar.get('timestamp')
                bt = bt_time.time() if bt_time and hasattr(bt_time, 'time') else None
                if bt and bt >= EOD_EXIT:
                    break
                if direction == 'SHORT' and bar['high'] >= entry_price:
                    fill_bar = bi
                    break
                elif direction == 'LONG' and bar['low'] <= entry_price:
                    fill_bar = bi
                    break

            if fill_bar is None:
                continue

            # Swing stop: 5-min candle extreme + 5pt
            if direction == 'SHORT':
                stop_price = poke['bar_high'] + 5
            else:
                stop_price = poke['bar_low'] - 5

            risk_pts = abs(entry_price - stop_price)
            if risk_pts <= 0 or risk_pts > 200:
                continue

            # Target: 1R
            if direction == 'SHORT':
                target_price = entry_price - risk_pts
            else:
                target_price = entry_price + risk_pts

            # Replay
            remaining = session_df.iloc[fill_bar:]
            exit_price = None
            exit_reason = None
            for idx in range(len(remaining)):
                bar = remaining.iloc[idx]
                bt_time = bar.get('timestamp')
                bt = bt_time.time() if bt_time and hasattr(bt_time, 'time') else None
                if bt and bt >= EOD_EXIT:
                    exit_price = bar['close']
                    exit_reason = 'EOD'
                    break
                if direction == 'SHORT' and bar['high'] >= stop_price:
                    exit_price = stop_price
                    exit_reason = 'STOP'
                    break
                elif direction == 'LONG' and bar['low'] <= stop_price:
                    exit_price = stop_price
                    exit_reason = 'STOP'
                    break
                if direction == 'SHORT' and bar['low'] <= target_price:
                    exit_price = target_price
                    exit_reason = 'TARGET'
                    break
                elif direction == 'LONG' and bar['high'] >= target_price:
                    exit_price = target_price
                    exit_reason = 'TARGET'
                    break

            if exit_price is None:
                exit_price = remaining.iloc[-1]['close']
                exit_reason = 'EOD'

            pnl_pts = (exit_price - entry_price - SLIPPAGE_PTS) if direction == 'LONG' \
                      else (entry_price - exit_price - SLIPPAGE_PTS)
            pnl_dollars = pnl_pts * TICK_VALUE * CONTRACTS - COMMISSION * CONTRACTS

            trades.append({
                'session_date': str(current),
                'strategy': 'VA_edge_fade',
                'direction': direction,
                'entry_price': entry_price,
                'stop_price': stop_price,
                'target_price': target_price,
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'risk_pts': risk_pts,
                'pnl_pts': pnl_pts,
                'pnl_dollars': pnl_dollars,
                'is_winner': pnl_dollars > 0,
            })

    return trades


# ============================================================================
# RUN BOTH STRATEGIES
# ============================================================================
print("\nRunning 80P 100% Retest + 2R...")
trades_80p = run_80p_retest(sessions, df_rth, eth_va)
print(f"  80P trades: {len(trades_80p)}")

print("Running VA Edge Fade Limit Edge + Swing + 1R (2nd test)...")
trades_vaf = run_va_edge_fade(sessions, df_rth, eth_va)
print(f"  VA Edge Fade trades: {len(trades_vaf)}")

# Combine and sort chronologically
all_trades = trades_80p + trades_vaf
all_trades.sort(key=lambda t: t['session_date'])
df_all = pd.DataFrame(all_trades)

print(f"\n  Combined trades: {len(df_all)}")
print(f"  Sessions with trades: {df_all['session_date'].nunique()}")


# ============================================================================
# SECTION 1: INDIVIDUAL STRATEGY STATS
# ============================================================================
print(f"\n\n{'='*100}")
print(f"  OPTION 3 CONSERVATIVE PORTFOLIO — DRAWDOWN ANALYSIS")
print(f"{'='*100}")

for strat_name, strat_trades in [('80P Retest', trades_80p), ('VA Edge Fade', trades_vaf), ('COMBINED', all_trades)]:
    if not strat_trades:
        continue
    df_s = pd.DataFrame(strat_trades)
    n = len(df_s)
    wr = df_s['is_winner'].mean() * 100
    gw = df_s[df_s['is_winner']]['pnl_dollars'].sum()
    gl = abs(df_s[~df_s['is_winner']]['pnl_dollars'].sum())
    pf = gw / gl if gl > 0 else float('inf')
    total_pnl = df_s['pnl_dollars'].sum()
    pm = total_pnl / months
    avg_w = df_s[df_s['is_winner']]['pnl_dollars'].mean() if df_s['is_winner'].any() else 0
    avg_l = df_s[~df_s['is_winner']]['pnl_dollars'].mean() if (~df_s['is_winner']).any() else 0
    max_single_win = df_s['pnl_dollars'].max()
    max_single_loss = df_s['pnl_dollars'].min()
    avg_risk = df_s['risk_pts'].mean()

    print(f"\n  ━━━ {strat_name} ━━━")
    print(f"    Trades:         {n} ({n/months:.1f}/mo)")
    print(f"    Win Rate:       {wr:.1f}%")
    print(f"    Profit Factor:  {pf:.2f}")
    print(f"    Total P&L:      ${total_pnl:,.0f}")
    print(f"    Monthly P&L:    ${pm:,.0f}")
    print(f"    Avg Winner:     ${avg_w:,.0f}")
    print(f"    Avg Loser:      ${avg_l:,.0f}")
    print(f"    Max Single Win: ${max_single_win:,.0f}")
    print(f"    Max Single Loss:${max_single_loss:,.0f}")
    print(f"    Avg Risk:       {avg_risk:.0f} pts")


# ============================================================================
# SECTION 2: EQUITY CURVE & DRAWDOWN (ACTUAL TRADES)
# ============================================================================
print(f"\n\n{'='*100}")
print(f"  EQUITY CURVE & MAX DRAWDOWN — Actual Trades (5 MNQ)")
print(f"{'='*100}")

equity = [0.0]
for t in all_trades:
    equity.append(equity[-1] + t['pnl_dollars'])

equity = np.array(equity)
running_max = np.maximum.accumulate(equity)
drawdowns = equity - running_max

max_dd = drawdowns.min()
max_dd_idx = np.argmin(drawdowns)

# Find peak before max DD
peak_idx = np.argmax(equity[:max_dd_idx + 1])
peak_val = equity[peak_idx]
trough_val = equity[max_dd_idx]

# Find recovery point
recovery_idx = None
for ri in range(max_dd_idx, len(equity)):
    if equity[ri] >= peak_val:
        recovery_idx = ri
        break

print(f"\n  Max Drawdown:     ${max_dd:,.0f}")
print(f"  Peak (trade #{peak_idx}):  ${peak_val:,.0f}")
print(f"  Trough (trade #{max_dd_idx}): ${trough_val:,.0f}")
if recovery_idx:
    print(f"  Recovery (trade #{recovery_idx}): ${equity[recovery_idx]:,.0f} ({recovery_idx - max_dd_idx} trades to recover)")
else:
    print(f"  Recovery: NOT YET RECOVERED")

# All drawdown periods
dd_periods = []
in_dd = False
dd_start = 0
for i in range(len(drawdowns)):
    if drawdowns[i] < 0 and not in_dd:
        in_dd = True
        dd_start = i
    elif drawdowns[i] >= 0 and in_dd:
        in_dd = False
        dd_periods.append({
            'start': dd_start,
            'end': i,
            'depth': drawdowns[dd_start:i].min(),
            'length': i - dd_start,
        })

if in_dd:
    dd_periods.append({
        'start': dd_start,
        'end': len(drawdowns) - 1,
        'depth': drawdowns[dd_start:].min(),
        'length': len(drawdowns) - dd_start,
    })

dd_periods.sort(key=lambda x: x['depth'])

print(f"\n  Top 5 Drawdown Periods:")
print(f"    {'Depth':>8s}  {'Length':>7s}  {'Start':>8s}  {'End':>8s}")
for dp in dd_periods[:5]:
    print(f"    ${dp['depth']:>7,.0f}  {dp['length']:>5d} tr  #{dp['start']:>6d}  #{dp['end']:>6d}")


# ============================================================================
# SECTION 3: DAILY P&L ANALYSIS
# ============================================================================
print(f"\n\n{'='*100}")
print(f"  DAILY P&L ANALYSIS")
print(f"{'='*100}")

daily_pnl = {}
for t in all_trades:
    d = t['session_date']
    if d not in daily_pnl:
        daily_pnl[d] = 0
    daily_pnl[d] += t['pnl_dollars']

daily_vals = list(daily_pnl.values())
daily_dates = list(daily_pnl.keys())

daily_equity = np.cumsum(daily_vals)
daily_running_max = np.maximum.accumulate(daily_equity)
daily_dd = daily_equity - daily_running_max

print(f"\n  Trading days with trades: {len(daily_vals)}")
print(f"  Winning days:   {sum(1 for v in daily_vals if v > 0)} ({sum(1 for v in daily_vals if v > 0)/len(daily_vals)*100:.1f}%)")
print(f"  Losing days:    {sum(1 for v in daily_vals if v < 0)} ({sum(1 for v in daily_vals if v < 0)/len(daily_vals)*100:.1f}%)")
print(f"  Flat days:      {sum(1 for v in daily_vals if v == 0)}")
print(f"\n  Best day:       ${max(daily_vals):,.0f}")
print(f"  Worst day:      ${min(daily_vals):,.0f}")
print(f"  Avg day:        ${np.mean(daily_vals):,.0f}")
print(f"  Median day:     ${np.median(daily_vals):,.0f}")
print(f"\n  Max daily DD:   ${daily_dd.min():,.0f}")

# Consecutive losing days
max_consec_loss_days = 0
current_consec = 0
for v in daily_vals:
    if v < 0:
        current_consec += 1
        max_consec_loss_days = max(max_consec_loss_days, current_consec)
    else:
        current_consec = 0

print(f"  Max consecutive losing days: {max_consec_loss_days}")

# Weekly P&L
weekly_pnl = {}
for d, pnl in zip(daily_dates, daily_vals):
    # Group by week (use ISO week)
    dt = pd.to_datetime(d)
    week_key = f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"
    if week_key not in weekly_pnl:
        weekly_pnl[week_key] = 0
    weekly_pnl[week_key] += pnl

weekly_vals = list(weekly_pnl.values())
print(f"\n  Weekly P&L:")
print(f"    Best week:     ${max(weekly_vals):,.0f}")
print(f"    Worst week:    ${min(weekly_vals):,.0f}")
print(f"    Avg week:      ${np.mean(weekly_vals):,.0f}")
print(f"    Winning weeks: {sum(1 for v in weekly_vals if v > 0)}/{len(weekly_vals)} ({sum(1 for v in weekly_vals if v > 0)/len(weekly_vals)*100:.1f}%)")


# ============================================================================
# SECTION 4: CONSECUTIVE LOSSES ANALYSIS
# ============================================================================
print(f"\n\n{'='*100}")
print(f"  CONSECUTIVE LOSSES ANALYSIS")
print(f"{'='*100}")

# Per trade
consec_losses = []
current_streak = 0
max_streak = 0
max_streak_pnl = 0
current_streak_pnl = 0

for t in all_trades:
    if not t['is_winner']:
        current_streak += 1
        current_streak_pnl += t['pnl_dollars']
        if current_streak > max_streak:
            max_streak = current_streak
            max_streak_pnl = current_streak_pnl
    else:
        if current_streak > 0:
            consec_losses.append({'streak': current_streak, 'pnl': current_streak_pnl})
        current_streak = 0
        current_streak_pnl = 0

if current_streak > 0:
    consec_losses.append({'streak': current_streak, 'pnl': current_streak_pnl})

consec_losses.sort(key=lambda x: x['pnl'])

print(f"\n  Max consecutive losses: {max_streak} trades (${max_streak_pnl:,.0f})")
print(f"\n  Top 5 Losing Streaks:")
print(f"    {'Streak':>6s}  {'P&L':>8s}")
for cl in consec_losses[:5]:
    print(f"    {cl['streak']:>4d} tr  ${cl['pnl']:>7,.0f}")

# Distribution of losing streaks
streak_counts = {}
for cl in consec_losses:
    s = cl['streak']
    if s not in streak_counts:
        streak_counts[s] = 0
    streak_counts[s] += 1

print(f"\n  Losing Streak Distribution:")
for s in sorted(streak_counts.keys()):
    print(f"    {s} in a row: {streak_counts[s]} times")


# ============================================================================
# SECTION 5: MONTE CARLO SIMULATION
# ============================================================================
print(f"\n\n{'='*100}")
print(f"  MONTE CARLO SIMULATION — 10,000 Paths")
print(f"{'='*100}")

# Use actual trade P&L distribution (both strategies)
pnl_pool = [t['pnl_dollars'] for t in all_trades]
n_trades = len(pnl_pool)

# Simulate 10,000 paths, each with same number of trades
N_SIMS = 10000
np.random.seed(42)

max_dds = []
final_pnls = []
max_consec_losses_sim = []

for sim in range(N_SIMS):
    # Bootstrap: sample with replacement
    path_pnl = np.random.choice(pnl_pool, size=n_trades, replace=True)
    path_equity = np.cumsum(path_pnl)
    path_running_max = np.maximum.accumulate(path_equity)
    path_dd = path_equity - path_running_max
    max_dds.append(path_dd.min())
    final_pnls.append(path_equity[-1])

    # Consecutive losses
    streak = 0
    max_s = 0
    for p in path_pnl:
        if p < 0:
            streak += 1
            max_s = max(max_s, streak)
        else:
            streak = 0
    max_consec_losses_sim.append(max_s)

max_dds = np.array(max_dds)
final_pnls = np.array(final_pnls)
max_consec_sim = np.array(max_consec_losses_sim)

print(f"\n  Max Drawdown Distribution (5 MNQ contracts):")
for pct in [50, 75, 90, 95, 99]:
    print(f"    {pct}th percentile: ${np.percentile(max_dds, 100-pct):,.0f}")

print(f"\n  Final P&L Distribution:")
for pct in [5, 25, 50, 75, 95]:
    print(f"    {pct}th percentile: ${np.percentile(final_pnls, pct):,.0f}")

print(f"\n  Max Consecutive Losses:")
for pct in [50, 75, 90, 95, 99]:
    print(f"    {pct}th percentile: {np.percentile(max_consec_sim, pct):.0f} trades")

print(f"\n  Probability of ruin (DD > $4,500 at 5 MNQ): {(max_dds < -4500).mean()*100:.1f}%")
print(f"  Probability of DD > $3,000 at 5 MNQ: {(max_dds < -3000).mean()*100:.1f}%")
print(f"  Probability of DD > $2,000 at 5 MNQ: {(max_dds < -2000).mean()*100:.1f}%")
print(f"  Probability of DD > $1,500 at 5 MNQ: {(max_dds < -1500).mean()*100:.1f}%")


# ============================================================================
# SECTION 6: PROP FIRM SIZING — What Contract Count Fits $4,500 DD?
# ============================================================================
print(f"\n\n{'='*100}")
print(f"  PROP FIRM SIZING — $150K Account, $4,500 Trailing Drawdown")
print(f"{'='*100}")

# Scale the Monte Carlo results for different contract counts
# Current is 5 MNQ. Test 1-10 MNQ.
print(f"\n  Monthly P&L and Drawdown by Contract Count:")
print(f"  {'Cts':>4s}  {'$/Mo':>8s}  {'$/Year':>9s}  {'50% DD':>8s}  {'90% DD':>8s}  {'95% DD':>8s}  {'99% DD':>8s}  {'P(DD>$4.5K)':>12s}  {'Safe?':>6s}")
print(f"  {'-'*90}")

base_contracts = 5
for cts in [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20]:
    scale = cts / base_contracts
    scaled_dds = max_dds * scale
    scaled_final = final_pnls * scale
    monthly = np.median(scaled_final) / months

    dd_50 = np.percentile(scaled_dds, 50)
    dd_90 = np.percentile(scaled_dds, 10)
    dd_95 = np.percentile(scaled_dds, 5)
    dd_99 = np.percentile(scaled_dds, 1)

    prob_ruin = (scaled_dds < -4500).mean() * 100
    safe = "YES" if prob_ruin < 5 else ("MAYBE" if prob_ruin < 15 else "NO")

    yearly = monthly * 12
    print(f"  {cts:>4d}  ${monthly:>7,.0f}  ${yearly:>8,.0f}  "
          f"${dd_50:>7,.0f}  ${dd_90:>7,.0f}  ${dd_95:>7,.0f}  ${dd_99:>7,.0f}  "
          f"{prob_ruin:>10.1f}%  {safe:>6s}")


# ============================================================================
# SECTION 7: OPTIMAL SIZING RECOMMENDATION
# ============================================================================
print(f"\n\n{'='*100}")
print(f"  SIZING RECOMMENDATION")
print(f"{'='*100}")

# Find max contracts where 95th percentile DD < $4,500
for cts in range(1, 30):
    scale = cts / base_contracts
    dd_95 = np.percentile(max_dds * scale, 5)
    if dd_95 < -4500:
        optimal_cts = cts - 1
        break
else:
    optimal_cts = 29

# Also find aggressive (90th pct) and conservative (99th pct)
for cts in range(1, 30):
    scale = cts / base_contracts
    dd_90 = np.percentile(max_dds * scale, 10)
    if dd_90 < -4500:
        aggressive_cts = cts - 1
        break
else:
    aggressive_cts = 29

for cts in range(1, 30):
    scale = cts / base_contracts
    dd_99 = np.percentile(max_dds * scale, 1)
    if dd_99 < -4500:
        conservative_cts = cts - 1
        break
else:
    conservative_cts = 29

scale_opt = optimal_cts / base_contracts
monthly_opt = np.median(final_pnls * scale_opt) / months

scale_agg = aggressive_cts / base_contracts
monthly_agg = np.median(final_pnls * scale_agg) / months

scale_con = conservative_cts / base_contracts
monthly_con = np.median(final_pnls * scale_con) / months

print(f"""
  $150K Account, $4,500 Trailing Drawdown:

  CONSERVATIVE (99% confidence — 1 in 100 paths breach DD):
    Contracts:  {conservative_cts} MNQ
    Monthly:    ${monthly_con:,.0f}
    Annual:     ${monthly_con * 12:,.0f}
    DD limit usage: keeps 99% of paths within $4,500

  RECOMMENDED (95% confidence — 1 in 20 paths breach DD):
    Contracts:  {optimal_cts} MNQ
    Monthly:    ${monthly_opt:,.0f}
    Annual:     ${monthly_opt * 12:,.0f}
    DD limit usage: keeps 95% of paths within $4,500

  AGGRESSIVE (90% confidence — 1 in 10 paths breach DD):
    Contracts:  {aggressive_cts} MNQ
    Monthly:    ${monthly_agg:,.0f}
    Annual:     ${monthly_agg * 12:,.0f}
    DD limit usage: keeps 90% of paths within $4,500

  MICRO BUFFER PROTOCOL:
    Start with {conservative_cts} MNQ contracts
    After building ${1500} buffer above trailing DD:
      Scale to {optimal_cts} MNQ
    After building ${2500} buffer:
      Scale to {aggressive_cts} MNQ
    After any losing week: drop back to {conservative_cts} MNQ for 3 days
""")


# ============================================================================
# SECTION 8: CONSISTENCY RULE CHECK
# ============================================================================
print(f"\n{'='*100}")
print(f"  CONSISTENCY RULE CHECK (35-40% Rule)")
print(f"{'='*100}")

# Check if any single day exceeds 35% of total profit
total_profit = sum(v for v in daily_vals if v > 0)
total_pnl_all = sum(daily_vals)

print(f"\n  Total profit (winning days only): ${total_profit:,.0f}")
print(f"  Total net P&L:                   ${total_pnl_all:,.0f}")

best_day = max(daily_vals)
best_day_pct = best_day / total_profit * 100 if total_profit > 0 else 0

print(f"  Best single day: ${best_day:,.0f} ({best_day_pct:.1f}% of total profit)")

# Check 35% consistency
consistency_violations = sum(1 for v in daily_vals if v > 0 and v / total_profit > 0.35)
print(f"  Days exceeding 35% of total profit: {consistency_violations}")

# Top 5 winning days as % of total
winning_days_sorted = sorted([v for v in daily_vals if v > 0], reverse=True)
print(f"\n  Top 5 Winning Days:")
for i, v in enumerate(winning_days_sorted[:5], 1):
    pct = v / total_profit * 100
    print(f"    {i}. ${v:,.0f} ({pct:.1f}% of total profit)")

# Number of profitable days needed to spread profit
target_days = total_pnl_all / (total_pnl_all * 0.35) if total_pnl_all > 0 else 0
print(f"\n  Min winning days needed for 35% rule: {max(3, int(np.ceil(1/0.35)))}")
print(f"  Actual winning days: {sum(1 for v in daily_vals if v > 0)}")
print(f"  Consistency: {'PASS' if consistency_violations == 0 else f'FAIL ({consistency_violations} violations)'}")


# ============================================================================
# SECTION 9: MONTHLY BREAKDOWN
# ============================================================================
print(f"\n\n{'='*100}")
print(f"  MONTHLY BREAKDOWN")
print(f"{'='*100}")

monthly_pnl = {}
monthly_trades = {}
for t in all_trades:
    dt = pd.to_datetime(t['session_date'])
    month_key = f"{dt.year}-{dt.month:02d}"
    if month_key not in monthly_pnl:
        monthly_pnl[month_key] = 0
        monthly_trades[month_key] = 0
    monthly_pnl[month_key] += t['pnl_dollars']
    monthly_trades[month_key] += 1

print(f"\n  {'Month':>8s}  {'Trades':>6s}  {'P&L':>8s}  {'Cumulative':>11s}")
print(f"  {'-'*45}")
cumulative = 0
for month in sorted(monthly_pnl.keys()):
    cumulative += monthly_pnl[month]
    print(f"  {month:>8s}  {monthly_trades[month]:>6d}  ${monthly_pnl[month]:>7,.0f}  ${cumulative:>10,.0f}")

profitable_months = sum(1 for v in monthly_pnl.values() if v > 0)
total_months_actual = len(monthly_pnl)
print(f"\n  Profitable months: {profitable_months}/{total_months_actual} ({profitable_months/total_months_actual*100:.0f}%)")
print(f"  Worst month: ${min(monthly_pnl.values()):,.0f}")
print(f"  Best month:  ${max(monthly_pnl.values()):,.0f}")


print(f"\n\n{'='*100}")
print(f"  ANALYSIS COMPLETE")
print(f"{'='*100}")
