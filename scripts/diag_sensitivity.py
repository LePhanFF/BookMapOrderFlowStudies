"""Sensitivity analysis: sweep Tier 1 constants +/-25% to measure P&L stability."""
import sys, pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from config.instruments import get_instrument
from engine.backtest import BacktestEngine
from engine.execution import ExecutionModel
from engine.position import PositionManager
from strategy import get_core_strategies

full_df = load_csv('NQ')
df = filter_rth(full_df)
df = compute_all_features(df)
instrument = get_instrument('MNQ')

def run_backtest():
    strategies = get_core_strategies()
    engine = BacktestEngine(
        instrument=instrument, strategies=strategies, filters=None,
        execution=ExecutionModel(instrument),
        position_mgr=PositionManager(max_drawdown=999999),
        full_df=full_df,
    )
    result = engine.run(df, verbose=False)
    trades = result.trades
    pnl = sum(t.net_pnl for t in trades)
    wr = sum(1 for t in trades if t.net_pnl > 0) / len(trades) * 100 if trades else 0
    return len(trades), wr, pnl


def sweep_constant(module_name, attr_name, base_val, label):
    """Sweep a constant at -25%, base, +25% and measure P&L."""
    import importlib
    mod = importlib.import_module(module_name)

    results = []
    for mult, tag in [(0.75, '-25%'), (1.0, 'BASE'), (1.25, '+25%')]:
        setattr(mod, attr_name, base_val * mult)
        # Re-import strategies to pick up changes
        trades, wr, pnl = run_backtest()
        results.append((tag, base_val * mult, trades, wr, pnl))

    # Restore base value
    setattr(mod, attr_name, base_val)

    # Calculate stability
    base_pnl = results[1][4]
    pnl_range = max(r[4] for r in results) - min(r[4] for r in results)
    pct_change = pnl_range / abs(base_pnl) * 100 if base_pnl != 0 else 999

    print(f'\n{"="*80}')
    print(f'{label} ({module_name}.{attr_name})')
    print(f'{"="*80}')
    print(f'{"Sweep":>8s} {"Value":>10s} {"Trades":>7s} {"WR":>7s} {"Net P&L":>12s} {"Delta":>10s}')
    for tag, val, trades, wr, pnl in results:
        delta = pnl - base_pnl
        print(f'{tag:>8s} {val:>10.3f} {trades:>7d} {wr:>6.1f}% ${pnl:>10,.0f} ${delta:>+9,.0f}')

    flag = 'OVER-FIT' if pct_change > 10 else 'STABLE'
    print(f'  P&L range: ${pnl_range:,.0f} ({pct_change:.1f}% of base) -> {flag}')
    return label, pct_change, flag


# --- Baseline ---
print('Computing baseline...')
base_trades, base_wr, base_pnl = run_backtest()
print(f'BASELINE: {base_trades}t, {base_wr:.1f}% WR, ${base_pnl:,.0f}\n')

# --- Sweep Tier 1 constants ---
results = []

# 1. SWEEP_THRESHOLD_RATIO (OR Reversal)
results.append(sweep_constant(
    'strategy.or_reversal', 'SWEEP_THRESHOLD_RATIO', 0.17,
    'OR Sweep Threshold'))

# 2. VWAP_ALIGNED_RATIO (OR Reversal)
results.append(sweep_constant(
    'strategy.or_reversal', 'VWAP_ALIGNED_RATIO', 0.17,
    'OR VWAP Alignment'))

# 3. EDGE_FADE_IB_EXPANSION_RATIO (Edge Fade)
results.append(sweep_constant(
    'strategy.edge_fade', 'EDGE_FADE_IB_EXPANSION_RATIO', 1.2,
    'Edge Fade IB Expansion'))

# 4. EDGE_FADE_MAX_BEARISH_EXT (Edge Fade)
results.append(sweep_constant(
    'strategy.edge_fade', 'EDGE_FADE_MAX_BEARISH_EXT', 0.3,
    'Edge Fade Max Bearish Ext'))

# 5. STOP_VWAP_BUFFER (Trend Bull)
results.append(sweep_constant(
    'strategy.trend_bull', 'STOP_VWAP_BUFFER', 0.40,
    'Trend Bull Stop VWAP Buffer'))

# 6. PRE_DELTA_RATIO (Trend Bull)
results.append(sweep_constant(
    'strategy.trend_bull', 'PRE_DELTA_RATIO', 3.0,
    'Trend Bull Pre-Delta Ratio'))

# 7. BEAR_ACCEPT_MIN_EXT (Bear Accept)
results.append(sweep_constant(
    'strategy.bear_accept', 'BEAR_ACCEPT_MIN_EXT', 0.4,
    'Bear Accept Min Extension'))

# 8. STOP_VWAP_BUFFER (P-Day)
results.append(sweep_constant(
    'strategy.p_day', 'STOP_VWAP_BUFFER', 0.40,
    'P-Day Stop VWAP Buffer'))

# --- Summary ---
print('\n' + '=' * 80)
print('SENSITIVITY SUMMARY')
print('=' * 80)
print(f'{"Constant":40s} {"P&L Swing %":>12s} {"Verdict":>10s}')
print('-' * 65)
for label, pct, flag in results:
    print(f'{label:40s} {pct:>11.1f}% {flag:>10s}')

overfit = [r for r in results if r[2] == 'OVER-FIT']
print(f'\n{len(overfit)}/{len(results)} constants flagged as OVER-FIT (>10% P&L swing on +/-25%)')
