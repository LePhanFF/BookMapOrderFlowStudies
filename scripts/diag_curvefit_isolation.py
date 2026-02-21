"""Test which curve-fit change caused regression by reverting each one."""
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

def run(label):
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
    nov_trades = [t for t in trades if pd.Timestamp(t.session_date) >= pd.Timestamp('2025-11-18')]
    nov_pnl = sum(t.net_pnl for t in nov_trades)
    print(f'{label:50s} {len(trades):4d}t {wr:5.1f}% ${pnl:>10,.0f}  Nov-Feb: ${nov_pnl:>8,.0f}')

    # Show per-strategy
    by_s = {}
    for t in trades:
        by_s.setdefault(t.strategy_name, []).append(t)
    for sn in sorted(by_s.keys()):
        st = by_s[sn]
        w = sum(1 for t in st if t.net_pnl > 0)
        p = sum(t.net_pnl for t in st)
        print(f'  {sn:25s} {len(st):3d}t {w / len(st) * 100:5.1f}% ${p:>8,.0f}')
    print()

print('=' * 100)
print('ISOLATION TEST: Current state (all changes applied)')
print('=' * 100)
run('All changes')
