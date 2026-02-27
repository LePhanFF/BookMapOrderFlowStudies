"""Comprehensive per-strategy trade management study.

For each strategy, simulate different management approaches using actual bar data:
1. Fixed stop/target (current baseline)
2. Trail stop at various R-multiples (0.3R, 0.5R, 0.75R, 1.0R)
3. Breakeven trigger at various R-multiples
4. Combined: BE trigger + trail
5. After trail stop is hit: how far does price continue?

Uses actual bar-by-bar data (not MFE estimates) for accurate simulation.
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
from engine.backtest import BacktestEngine
from engine.execution import ExecutionModel
from engine.position import PositionManager
from strategy import get_core_strategies

full_df = load_csv('NQ')
df = filter_rth(full_df)
df = compute_all_features(df)
instrument = get_instrument('MNQ')
point_value = instrument.point_value  # $2.0 for MNQ
commission_rt = 1.24 * 2  # round-trip commission
slippage_pts = 0.5  # 1 tick each side = 0.5 pts total cost

# Run baseline to get trades with entry/exit info
strategies = get_core_strategies()
engine = BacktestEngine(
    instrument=instrument, strategies=strategies, filters=None,
    execution=ExecutionModel(instrument),
    position_mgr=PositionManager(max_drawdown=999999),
    full_df=full_df,
)
result = engine.run(df, verbose=False)
trades = result.trades
print(f"Baseline: {len(trades)} trades")

sessions = df.groupby('session_date')


def simulate_bar_by_bar(trade, post_entry_bars, management_type, params):
    """Simulate a trade with given management on actual bar data.

    Returns: (exit_price, exit_bar_idx, exit_reason, bars_held)
    """
    entry = trade.entry_price
    direction = trade.direction
    stop = trade.stop_price
    target = trade.target_price
    risk = abs(entry - stop)

    if risk <= 0:
        return entry, 0, 'INVALID', 0

    current_stop = stop
    current_target = target
    be_activated = False
    trail_activated = False
    peak_favorable = 0.0

    be_trigger_r = params.get('be_trigger', None)
    trail_r = params.get('trail_r', None)
    trail_from_peak = params.get('trail_from_peak', None)  # trail distance from peak

    for i in range(len(post_entry_bars)):
        bar = post_entry_bars.iloc[i]
        bar_high = bar['high']
        bar_low = bar['low']
        bar_close = bar['close']

        # Calculate current favorable excursion
        if direction == 'LONG':
            favorable = bar_high - entry
            adverse = entry - bar_low
        else:
            favorable = entry - bar_low
            adverse = bar_high - entry

        peak_favorable = max(peak_favorable, favorable)

        # === Management logic ===

        # Breakeven trigger
        if be_trigger_r is not None and not be_activated:
            if peak_favorable >= be_trigger_r * risk:
                be_activated = True
                if direction == 'LONG':
                    current_stop = max(current_stop, entry)
                else:
                    current_stop = min(current_stop, entry)

        # Trail stop activation
        if trail_r is not None and not trail_activated:
            if peak_favorable >= trail_r * risk:
                trail_activated = True

        # Update trailing stop
        if trail_activated and trail_from_peak is not None:
            trail_dist = trail_from_peak * risk
            if direction == 'LONG':
                trail_level = (entry + peak_favorable) - trail_dist
                current_stop = max(current_stop, trail_level)
            else:
                trail_level = (entry - peak_favorable) + trail_dist
                current_stop = min(current_stop, trail_level)

        # === Check exits ===

        # Stop hit
        if direction == 'LONG':
            if bar_low <= current_stop:
                return current_stop, i, 'STOP' if not be_activated else 'BE_STOP', i + 1
        else:
            if bar_high >= current_stop:
                return current_stop, i, 'STOP' if not be_activated else 'BE_STOP', i + 1

        # Target hit (only if management includes fixed target)
        if params.get('use_target', True):
            if direction == 'LONG' and bar_high >= current_target:
                return current_target, i, 'TARGET', i + 1
            elif direction == 'SHORT' and bar_low <= current_target:
                return current_target, i, 'TARGET', i + 1

        # EOD exit (bar 330 = ~15:59)
        bar_time = bar.get('timestamp', None)
        if bar_time is not None and hasattr(bar_time, 'time'):
            if bar_time.time() >= pd.Timestamp('15:59').time():
                return bar_close, i, 'EOD', i + 1

    # End of available data
    return post_entry_bars.iloc[-1]['close'], len(post_entry_bars) - 1, 'EOD', len(post_entry_bars)


def analyze_post_trail_continuation(trade, post_entry_bars, trail_exit_idx, trail_exit_price):
    """After trail stop is hit, how far does price continue in original direction?"""
    if trail_exit_idx >= len(post_entry_bars) - 1:
        return 0, 0

    remaining = post_entry_bars.iloc[trail_exit_idx + 1:]
    if len(remaining) == 0:
        return 0, 0

    direction = trade.direction
    if direction == 'LONG':
        further_high = remaining['high'].max()
        continuation = further_high - trail_exit_price
        further_low = remaining['low'].min()
        reversal = trail_exit_price - further_low
    else:
        further_low = remaining['low'].min()
        continuation = trail_exit_price - further_low
        further_high = remaining['high'].max()
        reversal = further_high - trail_exit_price

    return max(0, continuation), max(0, reversal)


def get_post_entry_bars(trade):
    """Get all bars from entry to EOD for a trade."""
    session_date = trade.session_date
    try:
        sdf = sessions.get_group(pd.Timestamp(session_date).date())
    except (KeyError, ValueError):
        try:
            sdf = sessions.get_group(session_date)
        except KeyError:
            return None

    entry_ts = trade.entry_time
    mask = sdf['timestamp'] >= pd.Timestamp(entry_ts)
    post_entry = sdf[mask]
    return post_entry if len(post_entry) > 0 else None


def calc_pnl(direction, entry, exit_price):
    """Calculate net P&L for a trade."""
    if direction == 'LONG':
        gross = (exit_price - entry) * point_value
    else:
        gross = (entry - exit_price) * point_value
    return gross - commission_rt - slippage_pts * point_value


# === MANAGEMENT CONFIGURATIONS TO TEST ===
configs = {
    'BASELINE (fixed)': {'use_target': True},
    'BE@0.3R': {'be_trigger': 0.3, 'use_target': True},
    'BE@0.5R': {'be_trigger': 0.5, 'use_target': True},
    'BE@0.75R': {'be_trigger': 0.75, 'use_target': True},
    'BE@1.0R': {'be_trigger': 1.0, 'use_target': True},
    'Trail@0.3R': {'trail_r': 0.3, 'trail_from_peak': 0.5, 'use_target': False},
    'Trail@0.5R': {'trail_r': 0.5, 'trail_from_peak': 0.5, 'use_target': False},
    'Trail@0.75R': {'trail_r': 0.75, 'trail_from_peak': 0.75, 'use_target': False},
    'Trail@1.0R': {'trail_r': 1.0, 'trail_from_peak': 1.0, 'use_target': False},
    'BE@0.3+Trail@0.5': {'be_trigger': 0.3, 'trail_r': 0.5, 'trail_from_peak': 0.5, 'use_target': False},
    'BE@0.5+Trail@0.75': {'be_trigger': 0.5, 'trail_r': 0.75, 'trail_from_peak': 0.75, 'use_target': False},
    'BE@0.3+Target': {'be_trigger': 0.3, 'use_target': True},
    'Trail@0.5+Target': {'trail_r': 0.5, 'trail_from_peak': 0.5, 'use_target': True},
}

# === RUN STUDY PER STRATEGY ===
strategy_names = sorted(set(t.strategy_name for t in trades))

# Collect all results
all_results = {}  # config_name -> strategy -> {trades, wins, pnl, ...}

for config_name, params in configs.items():
    all_results[config_name] = {}
    for strat_name in strategy_names:
        strat_trades = [t for t in trades if t.strategy_name == strat_name]
        wins = 0
        total_pnl = 0
        be_exits = 0
        trail_exits = 0
        continuation_pts = []
        reversal_pts = []

        for trade in strat_trades:
            post_bars = get_post_entry_bars(trade)
            if post_bars is None or len(post_bars) < 2:
                total_pnl += trade.net_pnl
                if trade.net_pnl > 0:
                    wins += 1
                continue

            exit_price, exit_idx, exit_reason, bars = simulate_bar_by_bar(
                trade, post_bars, config_name, params)

            pnl = calc_pnl(trade.direction, trade.entry_price, exit_price)
            total_pnl += pnl
            if pnl > 0:
                wins += 1
            if 'BE' in exit_reason:
                be_exits += 1
            if exit_reason == 'STOP' and params.get('trail_r') is not None:
                trail_exits += 1
                # Check continuation after trail
                cont, rev = analyze_post_trail_continuation(
                    trade, post_bars, exit_idx, exit_price)
                continuation_pts.append(cont)
                reversal_pts.append(rev)

        n = len(strat_trades)
        wr = wins / n * 100 if n else 0
        exp = total_pnl / n if n else 0

        all_results[config_name][strat_name] = {
            'n': n, 'wins': wins, 'wr': wr, 'pnl': total_pnl, 'exp': exp,
            'be_exits': be_exits, 'trail_exits': trail_exits,
            'avg_continuation': np.mean(continuation_pts) if continuation_pts else 0,
            'avg_reversal': np.mean(reversal_pts) if reversal_pts else 0,
        }

# === PRINT RESULTS ===

# Per-strategy comparison across all configs
for strat_name in strategy_names:
    print(f"\n{'=' * 100}")
    print(f"  {strat_name}")
    print(f"{'=' * 100}")
    print(f"{'Config':25s} {'Trades':>6s} {'Wins':>5s} {'WR':>6s} {'Net P&L':>10s} {'$/trade':>8s} {'BE':>4s} {'Trail':>5s} {'Cont':>6s} {'Rev':>6s}")
    print('-' * 100)

    best_pnl = -999999
    best_config = ''

    for config_name in configs:
        r = all_results[config_name][strat_name]
        print(f"{config_name:25s} {r['n']:>6d} {r['wins']:>5d} {r['wr']:>5.1f}% ${r['pnl']:>8,.0f} ${r['exp']:>7,.0f} {r['be_exits']:>4d} {r['trail_exits']:>5d} {r['avg_continuation']:>5.0f}p {r['avg_reversal']:>5.0f}p")
        if r['pnl'] > best_pnl:
            best_pnl = r['pnl']
            best_config = config_name

    print(f"\n  >>> BEST: {best_config} (${best_pnl:,.0f})")

# === TOTAL PORTFOLIO COMPARISON ===
print(f"\n\n{'=' * 100}")
print(f"  TOTAL PORTFOLIO — ALL STRATEGIES COMBINED")
print(f"{'=' * 100}")
print(f"{'Config':25s} {'Trades':>6s} {'Wins':>5s} {'WR':>6s} {'Net P&L':>10s} {'$/trade':>8s}")
print('-' * 70)

config_totals = []
for config_name in configs:
    total_n = sum(all_results[config_name][s]['n'] for s in strategy_names)
    total_wins = sum(all_results[config_name][s]['wins'] for s in strategy_names)
    total_pnl = sum(all_results[config_name][s]['pnl'] for s in strategy_names)
    wr = total_wins / total_n * 100 if total_n else 0
    exp = total_pnl / total_n if total_n else 0
    config_totals.append((config_name, total_n, total_wins, wr, total_pnl, exp))
    print(f"{config_name:25s} {total_n:>6d} {total_wins:>5d} {wr:>5.1f}% ${total_pnl:>8,.0f} ${exp:>7,.0f}")

best = max(config_totals, key=lambda x: x[4])
print(f"\n  >>> BEST OVERALL: {best[0]} (${best[4]:,.0f}, {best[3]:.0f}% WR)")

# === OPTIMAL CONFIG PER STRATEGY ===
print(f"\n\n{'=' * 100}")
print(f"  OPTIMAL MANAGEMENT PER STRATEGY (mix-and-match)")
print(f"{'=' * 100}")
print(f"{'Strategy':25s} {'Best Config':25s} {'Trades':>6s} {'WR':>6s} {'Net P&L':>10s} {'$/trade':>8s} {'vs Base':>10s}")
print('-' * 100)

total_optimal_pnl = 0
total_baseline_pnl = 0
for strat_name in strategy_names:
    best_cfg = ''
    best_pnl = -999999
    for config_name in configs:
        r = all_results[config_name][strat_name]
        if r['pnl'] > best_pnl:
            best_pnl = r['pnl']
            best_cfg = config_name

    r = all_results[best_cfg][strat_name]
    base = all_results['BASELINE (fixed)'][strat_name]
    delta = r['pnl'] - base['pnl']
    total_optimal_pnl += r['pnl']
    total_baseline_pnl += base['pnl']
    print(f"{strat_name:25s} {best_cfg:25s} {r['n']:>6d} {r['wr']:>5.1f}% ${r['pnl']:>8,.0f} ${r['exp']:>7,.0f} ${delta:>+8,.0f}")

print(f"\n  OPTIMAL MIX TOTAL: ${total_optimal_pnl:,.0f} (vs baseline ${total_baseline_pnl:,.0f}, delta ${total_optimal_pnl-total_baseline_pnl:>+,.0f})")

# === POST-TRAIL CONTINUATION ANALYSIS ===
print(f"\n\n{'=' * 100}")
print(f"  AFTER TRAIL STOP HIT: Does price keep going?")
print(f"{'=' * 100}")

trail_config = 'Trail@0.5R'
for strat_name in strategy_names:
    strat_trades_list = [t for t in trades if t.strategy_name == strat_name]
    continuations = []
    reversals = []

    for trade in strat_trades_list:
        post_bars = get_post_entry_bars(trade)
        if post_bars is None or len(post_bars) < 2:
            continue

        params = configs[trail_config]
        exit_price, exit_idx, exit_reason, bars = simulate_bar_by_bar(
            trade, post_bars, trail_config, params)

        if exit_idx < len(post_bars) - 1:
            cont, rev = analyze_post_trail_continuation(
                trade, post_bars, exit_idx, exit_price)
            continuations.append(cont)
            reversals.append(rev)

    if continuations:
        avg_cont = np.mean(continuations)
        avg_rev = np.mean(reversals)
        cont_gt_20 = sum(1 for c in continuations if c > 20)
        print(f"  {strat_name:25s}: avg continuation={avg_cont:>6.0f}pts, avg reversal={avg_rev:>6.0f}pts, "
              f"continues>20pts={cont_gt_20}/{len(continuations)} ({cont_gt_20/len(continuations)*100:.0f}%)")
