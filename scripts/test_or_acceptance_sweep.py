#!/usr/bin/env python
"""
OR Acceptance Parameter Sweep

Tests multiple acceptance configurations and reports trades, WR, PF, net P&L.
Modifies the module-level constants in or_acceptance.py for each config,
then runs a fresh backtest.
"""

import sys
from pathlib import Path
from datetime import time

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import strategy.or_acceptance as ora_module
from config.instruments import get_instrument
from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from engine.backtest import BacktestEngine
from engine.execution import ExecutionModel
from engine.position import PositionManager
from strategy.or_acceptance import ORAcceptanceStrategy


def run_config(df, full_df, instrument, config):
    """Run a single backtest with the given OR Acceptance config."""
    # Patch module-level constants
    ora_module.MAX_WICK_VIOLATIONS = config['wicks']
    ora_module.CLOSE_ACCEPTANCE_PCT = config['close_pct']
    ora_module.ACCEPTANCE_BUFFER_PCT = config['buffer']
    ora_module.ACCEPTANCE_WINDOW = config['window']

    strategy = ORAcceptanceStrategy()
    execution = ExecutionModel(instrument, slippage_ticks=1)
    position_mgr = PositionManager(account_size=150000, max_drawdown=999999)

    engine = BacktestEngine(
        instrument=instrument,
        strategies=[strategy],
        filters=None,
        execution=execution,
        position_mgr=position_mgr,
        risk_per_trade=400,
        max_contracts=30,
        full_df=full_df,
    )

    result = engine.run(df, verbose=False)
    trades = result.trades

    n = len(trades)
    if n == 0:
        return {'trades': 0, 'wr': 0, 'pf': 0, 'net_pnl': 0, 'avg_pnl': 0,
                'max_dd': 0, 'long_n': 0, 'long_wr': 0, 'short_n': 0, 'short_wr': 0,
                'target_pct': 0, 'stop_pct': 0, 'eod_pct': 0}

    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl <= 0]
    net_pnl = sum(t.net_pnl for t in trades)
    gross_win = sum(t.net_pnl for t in wins) if wins else 0
    gross_loss = abs(sum(t.net_pnl for t in losses)) if losses else 1
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

    longs = [t for t in trades if t.direction == 'LONG']
    shorts = [t for t in trades if t.direction == 'SHORT']
    long_wins = [t for t in longs if t.net_pnl > 0]
    short_wins = [t for t in shorts if t.net_pnl > 0]

    # Exit reason breakdown
    exits = {}
    for t in trades:
        reason = t.exit_reason if hasattr(t, 'exit_reason') else 'UNKNOWN'
        exits[reason] = exits.get(reason, 0) + 1

    target_n = exits.get('TARGET', 0)
    stop_n = exits.get('STOP', 0)
    eod_n = exits.get('EOD', 0)

    # Max drawdown from equity curve
    equity = 150000
    peak = equity
    max_dd = 0
    for t in trades:
        equity += t.net_pnl
        peak = max(peak, equity)
        dd = peak - equity
        max_dd = max(max_dd, dd)

    return {
        'trades': n,
        'wr': len(wins) / n * 100,
        'pf': pf,
        'net_pnl': net_pnl,
        'avg_pnl': net_pnl / n,
        'max_dd': max_dd,
        'long_n': len(longs),
        'long_wr': len(long_wins) / max(1, len(longs)) * 100,
        'short_n': len(shorts),
        'short_wr': len(short_wins) / max(1, len(shorts)) * 100,
        'target_pct': target_n / n * 100,
        'stop_pct': stop_n / n * 100,
        'eod_pct': eod_n / n * 100,
    }


def main():
    print("=" * 80)
    print("  OR ACCEPTANCE PARAMETER SWEEP")
    print("=" * 80)

    # Load data once
    instrument = get_instrument('MNQ')
    full_df = load_csv('NQ')
    df = filter_rth(full_df)
    df = compute_all_features(df)

    configs = [
        {'name': 'Strict (baseline)',    'wicks': 0,  'close_pct': 1.00, 'buffer': 0.00, 'window': 'EOR'},
        {'name': 'Conservative',         'wicks': 2,  'close_pct': 0.85, 'buffer': 0.05, 'window': 'EOR'},
        {'name': 'Moderate',             'wicks': 3,  'close_pct': 0.80, 'buffer': 0.05, 'window': 'EOR'},
        {'name': 'Aggressive (default)', 'wicks': 5,  'close_pct': 0.70, 'buffer': 0.10, 'window': 'EOR'},
        {'name': 'Ultra-aggressive',     'wicks': 8,  'close_pct': 0.60, 'buffer': 0.15, 'window': 'EOR'},
        {'name': 'OR-window-agg',        'wicks': 5,  'close_pct': 0.70, 'buffer': 0.10, 'window': 'OR'},
        {'name': 'IB-window-agg',        'wicks': 5,  'close_pct': 0.70, 'buffer': 0.10, 'window': 'IB'},
        {'name': 'Close-only 80%',       'wicks': 99, 'close_pct': 0.80, 'buffer': 0.20, 'window': 'EOR'},
        {'name': 'Close-only 70%',       'wicks': 99, 'close_pct': 0.70, 'buffer': 0.20, 'window': 'EOR'},
    ]

    results = []
    for cfg in configs:
        print(f"\n  Running: {cfg['name']} (wicks={cfg['wicks']}, close={cfg['close_pct']:.0%}, "
              f"buffer={cfg['buffer']:.0%}, window={cfg['window']})...")
        r = run_config(df, full_df, instrument, cfg)
        r['name'] = cfg['name']
        results.append(r)
        print(f"    → {r['trades']} trades, WR={r['wr']:.1f}%, PF={r['pf']:.2f}, "
              f"Net=${r['net_pnl']:,.0f}, MaxDD=${r['max_dd']:,.0f}")

    # Print comparison table
    print(f"\n{'=' * 120}")
    print(f"  PARAMETER SWEEP RESULTS")
    print(f"{'=' * 120}")
    print(f"{'Config':<22} {'Trades':>6} {'WR%':>6} {'PF':>6} {'Net P&L':>10} {'$/trade':>8} "
          f"{'MaxDD':>8} {'L/WR':>6} {'S/WR':>6} {'Tgt%':>5} {'Stp%':>5} {'EOD%':>5}")
    print("-" * 120)
    for r in results:
        print(f"{r['name']:<22} {r['trades']:>6} {r['wr']:>5.1f}% {r['pf']:>5.2f} "
              f"${r['net_pnl']:>9,.0f} ${r['avg_pnl']:>7,.0f} "
              f"${r['max_dd']:>7,.0f} {r['long_wr']:>5.1f}% {r['short_wr']:>5.1f}% "
              f"{r['target_pct']:>4.0f}% {r['stop_pct']:>4.0f}% {r['eod_pct']:>4.0f}%")

    # Find best config by risk-adjusted return (PF × trades/month / maxDD)
    print(f"\n--- Best Config by Criteria ---")
    months = 12  # ~12 months of data
    for r in results:
        if r['trades'] > 0:
            r['trades_per_month'] = r['trades'] / months
            r['monthly_pnl'] = r['net_pnl'] / months
            r['score'] = r['pf'] * r['trades_per_month'] if r['max_dd'] < 4000 else 0
        else:
            r['trades_per_month'] = 0
            r['monthly_pnl'] = 0
            r['score'] = 0

    best = max(results, key=lambda x: x['score'])
    print(f"  Best overall: {best['name']} — score={best['score']:.1f} "
          f"({best['trades']} trades, PF={best['pf']:.2f}, ${best['monthly_pnl']:,.0f}/mo)")

    best_wr = max([r for r in results if r['trades'] >= 10], key=lambda x: x['wr'], default=None)
    if best_wr:
        print(f"  Best WR (≥10 trades): {best_wr['name']} — {best_wr['wr']:.1f}% WR, "
              f"PF={best_wr['pf']:.2f}, {best_wr['trades']} trades")

    best_pf = max([r for r in results if r['trades'] >= 10], key=lambda x: x['pf'], default=None)
    if best_pf:
        print(f"  Best PF (≥10 trades): {best_pf['name']} — PF={best_pf['pf']:.2f}, "
              f"WR={best_pf['wr']:.1f}%, {best_pf['trades']} trades")

    best_pnl = max([r for r in results if r['trades'] >= 10], key=lambda x: x['net_pnl'], default=None)
    if best_pnl:
        print(f"  Best Net P&L (≥10 trades): {best_pnl['name']} — ${best_pnl['net_pnl']:,.0f}, "
              f"WR={best_pnl['wr']:.1f}%, PF={best_pnl['pf']:.2f}")


if __name__ == '__main__':
    main()
