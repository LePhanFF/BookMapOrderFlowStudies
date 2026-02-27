"""Diagnose what kills trades: regime gates vs day type classification vs entry criteria."""
import sys
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
from collections import defaultdict

full_df = load_csv('NQ')
df = filter_rth(full_df)
df = compute_all_features(df)
instrument = get_instrument('MNQ')


def run_and_report(label, strategies):
    engine = BacktestEngine(
        instrument=instrument, strategies=strategies, filters=None,
        execution=ExecutionModel(instrument),
        position_mgr=PositionManager(max_drawdown=999999),
        full_df=full_df,
    )
    result = engine.run(df, verbose=False)
    trades = result.trades
    pnl = sum(t.net_pnl for t in trades)
    wins = sum(1 for t in trades if t.net_pnl > 0)
    wr = wins / len(trades) * 100 if trades else 0

    strat_stats = defaultdict(lambda: [0, 0, 0.0])
    for t in trades:
        s = strat_stats[t.strategy_name]
        s[0] += 1
        if t.net_pnl > 0:
            s[1] += 1
        s[2] += t.net_pnl

    print(f'\n{"=" * 80}')
    print(f'{label}')
    print(f'{"=" * 80}')
    print(f'TOTAL: {len(trades)} trades, {wr:.1f}% WR, ${pnl:,.0f}')
    print(f'{"Strategy":25s} {"Trades":>7s} {"WR":>7s} {"Net P&L":>12s}')
    print('-' * 55)
    for name in sorted(strat_stats):
        n, w, p = strat_stats[name]
        swr = w / n * 100 if n else 0
        print(f'{name:25s} {n:>7d} {swr:>6.1f}% ${p:>10,.0f}')
    return len(trades), pnl


# ===== TEST 0: BASELINE (current code) =====
print('Computing baseline...')
t0, p0 = run_and_report('TEST 0: BASELINE (current code)', get_core_strategies())

# ===== TEST 1: REGIME GATES REMOVED =====
# Patch strategies to always allow regime
import strategy.trend_bull as tb_mod
import strategy.p_day as pd_mod
import strategy.b_day as bd_mod

# Patch on_session_start to force _regime_allows = True
orig_tb_start = tb_mod.TrendDayBull.on_session_start
def tb_start_patched(self, *a, **kw):
    orig_tb_start(self, *a, **kw)
    self._regime_allows = True
tb_mod.TrendDayBull.on_session_start = tb_start_patched

orig_pd_start = pd_mod.PDayStrategy.on_session_start
def pd_start_patched(self, *a, **kw):
    orig_pd_start(self, *a, **kw)
    self._regime_allows = True
pd_mod.PDayStrategy.on_session_start = pd_start_patched

# B-Day checks regime in on_bar
orig_bd_bar = bd_mod.BDayStrategy.on_bar
def bd_bar_patched(self, bar, bar_index, session_context):
    ctx = dict(session_context)
    ctx['regime_volatility'] = 'normal'
    return orig_bd_bar(self, bar, bar_index, ctx)
bd_mod.BDayStrategy.on_bar = bd_bar_patched

t1, p1 = run_and_report('TEST 1: REGIME GATES REMOVED', get_core_strategies())

# Restore
tb_mod.TrendDayBull.on_session_start = orig_tb_start
pd_mod.PDayStrategy.on_session_start = orig_pd_start
bd_mod.BDayStrategy.on_bar = orig_bd_bar

# ===== TEST 2: DAY TYPE RELAXED (use confidence-based instead of hard gate) =====
# Patch classify_day_type to use lower thresholds
import strategy.day_type as dt_mod
orig_classify = dt_mod.classify_day_type

def relaxed_classify(ib_high, ib_low, current_price, ib_direction='INSIDE', trend_strength=None):
    """Relaxed: trend at 0.7x instead of 1.0x, p_day at 0.3x instead of 0.5x."""
    ib_range = ib_high - ib_low
    if ib_range <= 0:
        return dt_mod.DayType.NEUTRAL
    ib_mid = (ib_high + ib_low) / 2
    if current_price > ib_mid:
        extension = (current_price - ib_mid) / ib_range
        direction = 'BULL'
    else:
        extension = (ib_mid - current_price) / ib_range
        direction = 'BEAR'

    if trend_strength == dt_mod.TrendStrength.SUPER:
        return dt_mod.DayType.SUPER_TREND_UP if direction == 'BULL' else dt_mod.DayType.SUPER_TREND_DOWN
    if extension > 0.7:  # was 1.0
        return dt_mod.DayType.TREND_UP if direction == 'BULL' else dt_mod.DayType.TREND_DOWN
    if extension > 0.3:  # was 0.5
        return dt_mod.DayType.P_DAY
    if extension < 0.15:  # was 0.2
        return dt_mod.DayType.B_DAY
    return dt_mod.DayType.NEUTRAL

dt_mod.classify_day_type = relaxed_classify
t2, p2 = run_and_report('TEST 2: RELAXED DAY TYPE (trend@0.7x, p@0.3x)', get_core_strategies())
dt_mod.classify_day_type = orig_classify

# ===== TEST 3: BOTH REGIME + DAY TYPE RELAXED =====
tb_mod.TrendDayBull.on_session_start = tb_start_patched
pd_mod.PDayStrategy.on_session_start = pd_start_patched
bd_mod.BDayStrategy.on_bar = bd_bar_patched
dt_mod.classify_day_type = relaxed_classify

t3, p3 = run_and_report('TEST 3: BOTH REMOVED (regime off + relaxed day type)', get_core_strategies())

# Restore all
tb_mod.TrendDayBull.on_session_start = orig_tb_start
pd_mod.PDayStrategy.on_session_start = orig_pd_start
bd_mod.BDayStrategy.on_bar = orig_bd_bar
dt_mod.classify_day_type = orig_classify

# ===== TEST 4: Confidence-based gating (use TPO hint instead of extension) =====
# Patch strategies to use confidence scores instead of day_type string matching
def confidence_classify(ib_high, ib_low, current_price, ib_direction='INSIDE', trend_strength=None):
    """Use confidence-based: if ANY directional check passes, classify accordingly."""
    ib_range = ib_high - ib_low
    if ib_range <= 0:
        return dt_mod.DayType.NEUTRAL
    ib_mid = (ib_high + ib_low) / 2
    if current_price > ib_mid:
        extension = (current_price - ib_mid) / ib_range
        direction = 'BULL'
    else:
        extension = (ib_mid - current_price) / ib_range
        direction = 'BEAR'

    if trend_strength == dt_mod.TrendStrength.SUPER:
        return dt_mod.DayType.SUPER_TREND_UP if direction == 'BULL' else dt_mod.DayType.SUPER_TREND_DOWN

    # More aggressive: trend at 0.5x (same as old p_day!)
    if extension > 0.5:
        if direction == 'BULL':
            return dt_mod.DayType.TREND_UP
        else:
            return dt_mod.DayType.TREND_DOWN

    # P_DAY at very small extension
    if extension > 0.2:
        return dt_mod.DayType.P_DAY

    # B_DAY for tight
    if ib_direction == 'INSIDE' or extension < 0.1:
        return dt_mod.DayType.B_DAY

    return dt_mod.DayType.NEUTRAL

# Remove regime gates too
tb_mod.TrendDayBull.on_session_start = tb_start_patched
pd_mod.PDayStrategy.on_session_start = pd_start_patched
bd_mod.BDayStrategy.on_bar = bd_bar_patched
dt_mod.classify_day_type = confidence_classify

t4, p4 = run_and_report('TEST 4: AGGRESSIVE (trend@0.5x, p@0.2x, no regime)', get_core_strategies())

# Restore all
tb_mod.TrendDayBull.on_session_start = orig_tb_start
pd_mod.PDayStrategy.on_session_start = orig_pd_start
bd_mod.BDayStrategy.on_bar = orig_bd_bar
dt_mod.classify_day_type = orig_classify

# ===== SUMMARY =====
print(f'\n\n{"=" * 80}')
print('SUMMARY: Impact of Each Gate')
print(f'{"=" * 80}')
print(f'{"Test":50s} {"Trades":>7s} {"Net P&L":>12s} {"vs Base":>10s}')
print('-' * 80)
print(f'{"TEST 0: Baseline (current)":50s} {t0:>7d} ${p0:>10,.0f} {"---":>10s}')
print(f'{"TEST 1: Regime gates removed":50s} {t1:>7d} ${p1:>10,.0f} ${p1-p0:>+9,.0f}')
print(f'{"TEST 2: Relaxed day type (0.7x/0.3x)":50s} {t2:>7d} ${p2:>10,.0f} ${p2-p0:>+9,.0f}')
print(f'{"TEST 3: Both (regime + relaxed day type)":50s} {t3:>7d} ${p3:>10,.0f} ${p3-p0:>+9,.0f}')
print(f'{"TEST 4: Aggressive (0.5x/0.2x, no regime)":50s} {t4:>7d} ${p4:>10,.0f} ${p4-p0:>+9,.0f}')
