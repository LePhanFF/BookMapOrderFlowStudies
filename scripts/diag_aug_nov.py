"""Diagnostic: Why Aug-Nov loses money."""
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

strategies = get_core_strategies()
engine = BacktestEngine(
    instrument=instrument, strategies=strategies, filters=None,
    execution=ExecutionModel(instrument),
    position_mgr=PositionManager(max_drawdown=999999),
    full_df=full_df,
)
result = engine.run(df, verbose=False)

# AUGUST-ONLY deep dive
aug = [t for t in result.trades if pd.Timestamp(t.session_date).strftime('%Y-%m') == '2025-08']

print('=' * 70)
print('AUGUST 2025 - EVERY TRADE')
print('=' * 70)
print('IB median: 96 pts (VERY low vol)')
print()

for t in sorted(aug, key=lambda x: x.session_date):
    marker = 'W' if t.net_pnl > 0 else 'L'
    risk = abs(t.entry_price - t.stop_price)
    reward = abs(t.target_price - t.entry_price)
    rr = reward / risk if risk > 0 else 0
    print(f'{t.session_date[:10]} {marker} {t.strategy_name:25s} ${t.net_pnl:>8,.0f}  '
          f'{t.exit_reason:12s}  risk={risk:.0f}pts rwd={reward:.0f}pts R:R={rr:.1f}  {t.contracts}ct')

# Strategy performance by regime
print()
print('=' * 70)
print('STRATEGY PERFORMANCE BY REGIME')
print('=' * 70)

low_vol = [t for t in result.trades
           if pd.Timestamp(t.session_date).strftime('%Y-%m') in ('2025-08', '2025-09')]
high_vol = [t for t in result.trades
            if pd.Timestamp(t.session_date) >= pd.Timestamp('2025-11-18')]

print('\nLow-Vol (Aug-Sep, IB median ~100):')
by_s = {}
for t in low_vol:
    by_s.setdefault(t.strategy_name, []).append(t)
for sn, st in sorted(by_s.items(), key=lambda x: sum(t.net_pnl for t in x[1])):
    w = sum(1 for t in st if t.net_pnl > 0)
    p = sum(t.net_pnl for t in st)
    print(f'  {sn:25s} {len(st):2d}t, {w:2d}W, {w / len(st) * 100:5.1f}% WR, ${p:>8,.0f}')

print('\nHigh-Vol (Nov-Feb, IB median ~200):')
by_s2 = {}
for t in high_vol:
    by_s2.setdefault(t.strategy_name, []).append(t)
for sn, st in sorted(by_s2.items(), key=lambda x: sum(t.net_pnl for t in x[1])):
    w = sum(1 for t in st if t.net_pnl > 0)
    p = sum(t.net_pnl for t in st)
    print(f'  {sn:25s} {len(st):2d}t, {w:2d}W, {w / len(st) * 100:5.1f}% WR, ${p:>8,.0f}')

# Stop distance analysis
print()
print('=' * 70)
print('STOP DISTANCE & EXIT ANALYSIS')
print('=' * 70)

for pname, pt in [('Aug-Sep (low-vol)', low_vol), ('Nov-Feb (high-vol)', high_vol)]:
    stop_hits = [t for t in pt if t.exit_reason == 'STOP']
    target_hits = [t for t in pt if t.exit_reason == 'TARGET']
    eod_exits = [t for t in pt if t.exit_reason == 'EOD']
    vwap_exits = [t for t in pt if t.exit_reason == 'VWAP_BREACH_PM']

    stop_dists = [abs(t.entry_price - t.stop_price) for t in pt]
    tgt_dists = [abs(t.target_price - t.entry_price) for t in pt]

    print(f'\n  {pname} ({len(pt)} trades):')
    print(f'    Avg stop distance:   {np.mean(stop_dists):6.0f} pts')
    print(f'    Avg target distance: {np.mean(tgt_dists):6.0f} pts')
    print(f'    Avg contracts/trade: {np.mean([t.contracts for t in pt]):5.1f}')
    print(f'    STOP exits:   {len(stop_hits):3d} ({len(stop_hits) / len(pt) * 100:4.0f}%) -> ALL losses, '
          f'avg ${np.mean([t.net_pnl for t in stop_hits]):,.0f}' if stop_hits else '')
    print(f'    TARGET exits: {len(target_hits):3d} ({len(target_hits) / len(pt) * 100:4.0f}%) -> '
          f'avg ${np.mean([t.net_pnl for t in target_hits]):,.0f}' if target_hits else '')
    print(f'    EOD exits:    {len(eod_exits):3d} ({len(eod_exits) / len(pt) * 100:4.0f}%) -> '
          f'avg ${np.mean([t.net_pnl for t in eod_exits]):,.0f}' if eod_exits else '')
    if vwap_exits:
        print(f'    VWAP exits:   {len(vwap_exits):3d} ({len(vwap_exits) / len(pt) * 100:4.0f}%) -> '
              f'avg ${np.mean([t.net_pnl for t in vwap_exits]):,.0f}')

# Contract sizing analysis - are we over-leveraged in low-vol?
print()
print('=' * 70)
print('CONTRACT SIZING - LEVERAGE ANALYSIS')
print('=' * 70)
for pname, pt in [('Aug-Sep', low_vol), ('Nov-Feb', high_vol)]:
    contracts = [t.contracts for t in pt]
    risks_pts = [abs(t.entry_price - t.stop_price) for t in pt]
    print(f'\n  {pname}:')
    print(f'    Avg contracts: {np.mean(contracts):.1f}')
    print(f'    Avg stop pts:  {np.mean(risks_pts):.0f}')
    print(f'    Avg $ at risk: ${np.mean(risks_pts) * np.mean(contracts) * 0.50:,.0f} (MNQ $0.50/pt)')

    # Per strategy
    by_s = {}
    for t in pt:
        by_s.setdefault(t.strategy_name, []).append(t)
    for sn, st in sorted(by_s.items()):
        avg_ct = np.mean([t.contracts for t in st])
        avg_risk = np.mean([abs(t.entry_price - t.stop_price) for t in st])
        print(f'    {sn:25s}: {avg_ct:5.1f} ct, {avg_risk:5.0f} pts stop')
