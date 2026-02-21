"""A/B test regime gates on all 142 sessions."""
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

nov18 = pd.Timestamp('2025-11-18')

def run_test(label, patch_fn=None):
    strategies = get_core_strategies()
    if patch_fn:
        patch_fn(strategies)
    engine = BacktestEngine(
        instrument=instrument, strategies=strategies, filters=None,
        execution=ExecutionModel(instrument),
        position_mgr=PositionManager(max_drawdown=999999),
        full_df=full_df,
    )
    result = engine.run(df, verbose=False)
    trades = result.trades
    early = [t for t in trades if pd.Timestamp(t.session_date) < nov18]
    late = [t for t in trades if pd.Timestamp(t.session_date) >= nov18]

    e_pnl = sum(t.net_pnl for t in early)
    l_pnl = sum(t.net_pnl for t in late)
    total = sum(t.net_pnl for t in trades)
    wr = sum(1 for t in trades if t.net_pnl > 0) / len(trades) * 100 if trades else 0

    print(f'{label:45s} {len(trades):4d}t {wr:5.1f}% ${total:>10,.0f}  Aug-Nov: ${e_pnl:>8,.0f}  Nov-Feb: ${l_pnl:>8,.0f}')
    return trades

def disable_all_gates(strategies):
    """Disable all regime gates."""
    for s in strategies:
        orig_start = s.on_session_start
        def make_patched(orig):
            def patched(self, session_date, ib_high, ib_low, ib_range, session_context):
                orig(session_date, ib_high, ib_low, ib_range, session_context)
                if hasattr(self, '_regime_allows'):
                    self._regime_allows = True
                if hasattr(self, '_regime_allows_bday'):
                    self._regime_allows_bday = True
            return patched
        s.on_session_start = make_patched(orig_start).__get__(s)

def disable_trend_pday_gates(strategies):
    """Only disable Trend+PDay gates, keep BDay gate."""
    for s in strategies:
        if s.name in ('Trend Day Bull', 'P-Day'):
            orig_start = s.on_session_start
            def make_patched(orig):
                def patched(self, session_date, ib_high, ib_low, ib_range, session_context):
                    orig(session_date, ib_high, ib_low, ib_range, session_context)
                    self._regime_allows = True
                return patched
            s.on_session_start = make_patched(orig_start).__get__(s)

def disable_bday_gate(strategies):
    """Only disable BDay gate, keep Trend+PDay gates."""
    for s in strategies:
        if s.name == 'B-Day':
            orig_start = s.on_session_start
            def make_patched(orig):
                def patched(self, session_date, ib_high, ib_low, ib_range, session_context):
                    orig(session_date, ib_high, ib_low, ib_range, session_context)
                    self._regime_allows_bday = True
                return patched
            s.on_session_start = make_patched(orig_start).__get__(s)

print('=' * 120)
print('A/B TEST: REGIME GATE COMBINATIONS (142 sessions)')
print('=' * 120)
print(f'{"Config":45s} {"Tr":>4s} {"WR":>5s} {"Total":>10s}  {"Aug-Nov":>10s}  {"Nov-Feb":>10s}')
print('-' * 120)

# 1. All gates ON (current code)
t1 = run_test('All gates ON (B-Day+Trend+PDay)')

# 2. All gates OFF (baseline)
t2 = run_test('All gates OFF (baseline)', disable_all_gates)

# 3. Only B-Day gate (no Trend/PDay gates)
t3 = run_test('B-Day gate only', disable_trend_pday_gates)

# 4. Only Trend+PDay gates (no B-Day gate)
t4 = run_test('Trend+PDay gates only', disable_bday_gate)

# Delta analysis
print()
print('DELTAS vs baseline (all gates OFF):')
base_pnl = sum(t.net_pnl for t in t2)
for label, trades in [('All gates ON', t1), ('B-Day only', t3), ('Trend+PDay only', t4)]:
    pnl = sum(t.net_pnl for t in trades)
    delta = pnl - base_pnl
    removed = len(t2) - len(trades)
    print(f'  {label:30s}: ${delta:>+8,.0f} ({removed:+d} trades)')
