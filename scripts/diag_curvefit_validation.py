"""Validate curve-fit hardening changes: no regression on 142 sessions."""
import sys, pandas as pd
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

strategies = get_core_strategies()
engine = BacktestEngine(
    instrument=instrument, strategies=strategies, filters=None,
    execution=ExecutionModel(instrument),
    position_mgr=PositionManager(max_drawdown=999999),
    full_df=full_df,
)
result = engine.run(df, verbose=False)

nov18 = pd.Timestamp('2025-11-18')
all_t = result.trades
report_t = [t for t in all_t if pd.Timestamp(t.session_date) >= nov18]

total_pnl = sum(t.net_pnl for t in all_t)
total_w = sum(1 for t in all_t if t.net_pnl > 0)
total_wr = total_w / len(all_t) * 100 if all_t else 0

r_pnl = sum(t.net_pnl for t in report_t)
r_w = sum(1 for t in report_t if t.net_pnl > 0)
r_wr = r_w / len(report_t) * 100 if report_t else 0

print(f'FULL: {len(all_t)}t, {total_wr:.1f}% WR, ${total_pnl:,.0f}')
print(f'NOV-FEB: {len(report_t)}t, {r_wr:.1f}% WR, ${r_pnl:,.0f}')

by_s = {}
for t in all_t:
    by_s.setdefault(t.strategy_name, []).append(t)
print()
for sn in sorted(by_s.keys()):
    st = by_s[sn]
    w = sum(1 for t in st if t.net_pnl > 0)
    p = sum(t.net_pnl for t in st)
    wr = w / len(st) * 100
    print(f'{sn:25s} {len(st):3d}t {wr:5.1f}% ${p:>9,.0f}')

# Monthly
months = {}
for t in all_t:
    m = pd.Timestamp(t.session_date).strftime('%Y-%m')
    months.setdefault(m, []).append(t)
print()
for m in sorted(months.keys()):
    mt = months[m]
    w = sum(1 for t in mt if t.net_pnl > 0)
    p = sum(t.net_pnl for t in mt)
    wr = w / len(mt) * 100
    print(f'{m:>8s} {len(mt):3d}t {wr:5.1f}% ${p:>9,.0f}')
