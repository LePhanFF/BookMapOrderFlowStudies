"""When-to-avoid analysis: Would sitting out bad conditions improve the portfolio?

Tests:
1. Filter by IB range (too low, too high)
2. Filter by prior-day loss (don't trade after big loss day)
3. Filter by month/seasonal patterns
4. Filter by regime (LOW vol months)
5. Filter by strategy (drop losing strategies entirely)
6. Combined filters
7. What-if: remove worst N days, what's the improvement?
"""
import sys, warnings, io
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from collections import defaultdict
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
point_value = instrument.point_value

strategies = get_core_strategies()
engine = BacktestEngine(
    instrument=instrument, strategies=strategies, filters=None,
    execution=ExecutionModel(instrument),
    position_mgr=PositionManager(max_drawdown=999999),
    full_df=full_df,
)
result = engine.run(df, verbose=False)
trades = result.trades

# Compute session IB ranges
session_ib = {}
for session_date, sdf in df.groupby('session_date'):
    sd = str(session_date)
    ib_bars = sdf.head(60)
    if len(ib_bars) >= 10:
        session_ib[sd] = ib_bars['high'].max() - ib_bars['low'].min()

# Build trade data
trade_data = []
for t in trades:
    td = {
        'trade': t,
        'date': str(t.session_date),
        'month': str(t.session_date)[:7],
        'strategy': t.strategy_name,
        'net_pnl': t.net_pnl,
        'ib_range': session_ib.get(str(t.session_date), 150),
        'risk_pts': abs(t.entry_price - t.stop_price),
    }
    trade_data.append(td)

# Daily P&L
daily_pnl = defaultdict(float)
for td in trade_data:
    daily_pnl[td['date']] += td['net_pnl']

# ================================================================
def evaluate(filtered_trades, label):
    """Compute stats for a filtered set of trades."""
    n = len(filtered_trades)
    if n == 0:
        return None
    wins = sum(1 for td in filtered_trades if td['net_pnl'] > 0)
    net = sum(td['net_pnl'] for td in filtered_trades)
    wr = wins / n * 100

    # Max DD
    eq = 0
    peak = 0
    max_dd = 0
    for td in sorted(filtered_trades, key=lambda x: (x['date'], str(x['trade'].entry_time))):
        eq += td['net_pnl']
        peak = max(peak, eq)
        max_dd = max(max_dd, peak - eq)

    removed = len(trade_data) - n
    removed_pnl = sum(td['net_pnl'] for td in trade_data) - net

    return {
        'label': label,
        'trades': n,
        'removed': removed,
        'wins': wins,
        'wr': wr,
        'net': net,
        'exp': net / n,
        'max_dd': max_dd,
        'removed_pnl': removed_pnl,
    }

def print_result(r):
    if r is None:
        return
    delta = r['net'] - baseline['net']
    dd_delta = r['max_dd'] - baseline['max_dd']
    print("  {:45s} {:>4d}t (-{:>3d}) {:>5.1f}%WR ${:>8,.0f} ${:>5,.0f}/t DD${:>5,.0f} | delta ${:>+7,.0f} DD${:>+5,.0f}".format(
        r['label'], r['trades'], r['removed'], r['wr'], r['net'], r['exp'], r['max_dd'],
        delta, dd_delta))

baseline = evaluate(trade_data, 'BASELINE (all trades)')

print("=" * 120)
print("  WHEN TO AVOID TRADING -- FILTER ANALYSIS")
print("  259 sessions, 283 trades, $19,462 net")
print("=" * 120)

print("\n  BASELINE:")
print_result(baseline)

# ================================================================
# TEST 1: Filter by IB range
# ================================================================
print("\n" + "-" * 120)
print("  TEST 1: FILTER BY IB RANGE (too high = chaos, too low = no opportunity)")
print("-" * 120)

ib_thresholds = [
    ('IB > 50 (skip ultra-low)', lambda td: td['ib_range'] > 50),
    ('IB > 75 (skip low vol)', lambda td: td['ib_range'] > 75),
    ('IB > 100 (skip very low)', lambda td: td['ib_range'] > 100),
    ('IB < 400 (skip extreme)', lambda td: td['ib_range'] < 400),
    ('IB < 350 (skip very high)', lambda td: td['ib_range'] < 350),
    ('IB < 300 (skip high ATR)', lambda td: td['ib_range'] < 300),
    ('IB < 250 (normal only)', lambda td: td['ib_range'] < 250),
    ('IB 75-350 (goldilocks)', lambda td: 75 < td['ib_range'] < 350),
    ('IB 100-300 (sweet spot)', lambda td: 100 < td['ib_range'] < 300),
    ('IB 100-250 (conservative)', lambda td: 100 < td['ib_range'] < 250),
    ('IB 125-275 (narrow band)', lambda td: 125 < td['ib_range'] < 275),
]

for label, filt in ib_thresholds:
    filtered = [td for td in trade_data if filt(td)]
    r = evaluate(filtered, label)
    print_result(r)

# ================================================================
# TEST 2: Filter by prior-day loss
# ================================================================
print("\n" + "-" * 120)
print("  TEST 2: SIT OUT AFTER BAD DAY (loss > threshold)")
print("-" * 120)

sorted_dates = sorted(set(td['date'] for td in trade_data))
date_index = {d: i for i, d in enumerate(sorted_dates)}

for threshold in [500, 750, 1000, 1500]:
    # Build set of dates to skip (day after a bad day)
    skip_dates = set()
    for i, date in enumerate(sorted_dates):
        if daily_pnl.get(date, 0) < -threshold:
            if i + 1 < len(sorted_dates):
                skip_dates.add(sorted_dates[i + 1])

    filtered = [td for td in trade_data if td['date'] not in skip_dates]
    r = evaluate(filtered, f'Skip after day loss > ${threshold}')
    print_result(r)

# ================================================================
# TEST 3: Filter by month (seasonal)
# ================================================================
print("\n" + "-" * 120)
print("  TEST 3: SEASONAL FILTER (skip specific months)")
print("-" * 120)

month_combos = [
    ('Skip March', ['2025-03']),
    ('Skip Jun', ['2025-06']),
    ('Skip Mar+Jun', ['2025-03', '2025-06']),
    ('Skip Mar-Jun (tariff chaos)', ['2025-03', '2025-04', '2025-05', '2025-06']),
    ('Skip May-Jul (summer)', ['2025-05', '2025-06', '2025-07']),
    ('Only Nov-Feb (best period)', None),  # special
]

for label, skip_months in month_combos:
    if skip_months is not None:
        filtered = [td for td in trade_data if td['month'] not in skip_months]
    else:
        # Only Nov-Feb
        filtered = [td for td in trade_data if td['month'][-2:] in ['11', '12', '01', '02']]
    r = evaluate(filtered, label)
    print_result(r)

# ================================================================
# TEST 4: Drop losing strategies
# ================================================================
print("\n" + "-" * 120)
print("  TEST 4: DROP LOSING STRATEGIES")
print("-" * 120)

strategy_combos = [
    ('Drop Bear Accept', ['BearAcceptShort']),
    ('Drop P-Day', ['PDayStrategy']),
    ('Drop Trend Bull', ['TrendDayBull']),
    ('Drop IBH Sweep', ['IBHSweepFail']),
    ('Drop Bear+P-Day+Trend+IBH', ['BearAcceptShort', 'PDayStrategy', 'TrendDayBull', 'IBHSweepFail']),
    ('Drop Bear+P-Day+Trend+IBH+B-Day', ['BearAcceptShort', 'PDayStrategy', 'TrendDayBull', 'IBHSweepFail', 'BDayStrategy']),
    ('Only OR Rev + Edge Fade', None),  # special
]

for label, drop_strats in strategy_combos:
    if drop_strats is not None:
        filtered = [td for td in trade_data if td['strategy'] not in drop_strats]
    else:
        filtered = [td for td in trade_data if td['strategy'] in ['OpeningRangeReversal', 'EdgeFadeStrategy']]
    r = evaluate(filtered, label)
    print_result(r)

# ================================================================
# TEST 5: Combined filters
# ================================================================
print("\n" + "-" * 120)
print("  TEST 5: COMBINED FILTERS (best of each)")
print("-" * 120)

combos = [
    ('Drop Bear + IB>75',
     lambda td: td['strategy'] != 'BearAcceptShort' and td['ib_range'] > 75),
    ('Drop losers + IB 75-350',
     lambda td: td['strategy'] not in ['BearAcceptShort', 'PDayStrategy', 'TrendDayBull', 'IBHSweepFail'] and 75 < td['ib_range'] < 350),
    ('OR+Edge only + IB 75-350',
     lambda td: td['strategy'] in ['OpeningRangeReversal', 'EdgeFadeStrategy'] and 75 < td['ib_range'] < 350),
    ('OR+Edge+BDay + IB>75',
     lambda td: td['strategy'] in ['OpeningRangeReversal', 'EdgeFadeStrategy', 'BDayStrategy'] and td['ib_range'] > 75),
    ('All strats + IB 100-300 + skip after $750 loss',
     lambda td: 100 < td['ib_range'] < 300 and td['date'] not in skip_after_750),
    ('Drop losers + IB 100-300',
     lambda td: td['strategy'] not in ['BearAcceptShort', 'PDayStrategy', 'TrendDayBull', 'IBHSweepFail'] and 100 < td['ib_range'] < 300),
]

# Pre-compute skip-after-750
skip_after_750 = set()
for i, date in enumerate(sorted_dates):
    if daily_pnl.get(date, 0) < -750:
        if i + 1 < len(sorted_dates):
            skip_after_750.add(sorted_dates[i + 1])

for label, filt in combos:
    filtered = [td for td in trade_data if filt(td)]
    r = evaluate(filtered, label)
    print_result(r)

# ================================================================
# TEST 6: What if we remove worst N days?
# ================================================================
print("\n" + "-" * 120)
print("  TEST 6: WHAT IF WE REMOVE WORST N TRADING DAYS?")
print("-" * 120)

sorted_days = sorted(daily_pnl.items(), key=lambda x: x[1])
for n_remove in [1, 3, 5, 10, 15, 20]:
    worst_dates = set(d for d, _ in sorted_days[:n_remove])
    filtered = [td for td in trade_data if td['date'] not in worst_dates]
    r = evaluate(filtered, f'Remove worst {n_remove} days')
    print_result(r)

# Also: remove best N days to show fragility
print("\n  (For perspective: removing BEST days)")
sorted_days_best = sorted(daily_pnl.items(), key=lambda x: x[1], reverse=True)
for n_remove in [1, 3, 5, 10]:
    best_dates = set(d for d, _ in sorted_days_best[:n_remove])
    filtered = [td for td in trade_data if td['date'] not in best_dates]
    r = evaluate(filtered, f'Remove best {n_remove} days')
    print_result(r)

# ================================================================
# TEST 7: Per-strategy avoidance by IB regime
# ================================================================
print("\n" + "-" * 120)
print("  TEST 7: PER-STRATEGY IB REGIME FILTER (which strategy to run in which vol?)")
print("-" * 120)

abbrev = {
    'BDayStrategy': 'B-Day',
    'BearAcceptShort': 'BearAcc',
    'EdgeFadeStrategy': 'EdgeFade',
    'IBHSweepFail': 'IBHSwp',
    'OpeningRangeReversal': 'OR_Rev',
    'PDayStrategy': 'P-Day',
    'TrendDayBull': 'TrdBull',
}

all_strats = sorted(set(td['strategy'] for td in trade_data))
regimes = [
    ('LOW', lambda ib: ib < 100),
    ('MED', lambda ib: 100 <= ib < 150),
    ('NORMAL', lambda ib: 150 <= ib < 250),
    ('HIGH', lambda ib: ib >= 250),
]

for sname in all_strats:
    ab = abbrev.get(sname, sname[:12])
    st = [td for td in trade_data if td['strategy'] == sname]
    if len(st) < 3:
        continue

    total_pnl = sum(td['net_pnl'] for td in st)
    print(f"\n  {ab} ({len(st)}t, ${total_pnl:,.0f}):")

    for regime_name, regime_filter in regimes:
        regime_trades = [td for td in st if regime_filter(td['ib_range'])]
        if not regime_trades:
            print(f"    {regime_name:8s}: --- (0 trades)")
            continue
        n = len(regime_trades)
        w = sum(1 for td in regime_trades if td['net_pnl'] > 0)
        pnl = sum(td['net_pnl'] for td in regime_trades)
        wr = w / n * 100
        avg_r = np.mean([td['net_pnl'] / (td['risk_pts'] * point_value) for td in regime_trades if td['risk_pts'] > 0])
        marker = " <-- AVOID" if pnl < -200 and n >= 3 else " <-- BEST" if pnl > 500 else ""
        print(f"    {regime_name:8s}: {n:>3d}t {wr:>5.0f}%WR ${pnl:>7,.0f} ({avg_r:>+.2f}R avg){marker}")

    # Optimal regime filter
    best_regimes = []
    for regime_name, regime_filter in regimes:
        regime_trades = [td for td in st if regime_filter(td['ib_range'])]
        if regime_trades:
            pnl = sum(td['net_pnl'] for td in regime_trades)
            if pnl > 0:
                best_regimes.append(regime_name)

    if best_regimes and len(best_regimes) < len(regimes):
        print(f"    >> Optimal: trade only in {'+'.join(best_regimes)}")

# ================================================================
# SUMMARY
# ================================================================
print("\n\n" + "=" * 120)
print("  SUMMARY: BEST AVOIDANCE FILTERS")
print("=" * 120)

# Find top improvements
all_filters = []

# Re-run key filters and collect
key_filters = [
    ('BASELINE', lambda td: True),
    ('Drop Bear Accept only', lambda td: td['strategy'] != 'BearAcceptShort'),
    ('Drop all losers (Bear+PDay+TrdBull+IBH)', lambda td: td['strategy'] not in ['BearAcceptShort', 'PDayStrategy', 'TrendDayBull', 'IBHSweepFail']),
    ('OR+Edge+BDay only', lambda td: td['strategy'] in ['OpeningRangeReversal', 'EdgeFadeStrategy', 'BDayStrategy']),
    ('OR+Edge only', lambda td: td['strategy'] in ['OpeningRangeReversal', 'EdgeFadeStrategy']),
    ('All + IB 75-350', lambda td: 75 < td['ib_range'] < 350),
    ('Drop losers + IB 75-350', lambda td: td['strategy'] not in ['BearAcceptShort', 'PDayStrategy', 'TrendDayBull', 'IBHSweepFail'] and 75 < td['ib_range'] < 350),
    ('OR+Edge + IB 75-350', lambda td: td['strategy'] in ['OpeningRangeReversal', 'EdgeFadeStrategy'] and 75 < td['ib_range'] < 350),
]

print("\n  Ranked by Net P&L:")
results = []
for label, filt in key_filters:
    filtered = [td for td in trade_data if filt(td)]
    r = evaluate(filtered, label)
    if r:
        results.append(r)

results.sort(key=lambda x: x['net'], reverse=True)
print("  {:45s} {:>5s} {:>6s} {:>10s} {:>8s} {:>8s}".format(
    "Filter", "Trd", "WR", "Net P&L", "$/trade", "Max DD"))
print("  " + "-" * 90)
for r in results:
    marker = " <--" if r['label'] == 'BASELINE' else ""
    print("  {:45s} {:>5d} {:>5.1f}% ${:>8,.0f} ${:>6,.0f} ${:>6,.0f}{}".format(
        r['label'], r['trades'], r['wr'], r['net'], r['exp'], r['max_dd'], marker))

print("\n  Ranked by Expectancy ($/trade):")
results.sort(key=lambda x: x['exp'], reverse=True)
for r in results:
    marker = " <--" if r['label'] == 'BASELINE' else ""
    print("  {:45s} {:>5d} {:>5.1f}% ${:>8,.0f} ${:>6,.0f} ${:>6,.0f}{}".format(
        r['label'], r['trades'], r['wr'], r['net'], r['exp'], r['max_dd'], marker))

print("\n  Ranked by Max DD (lowest risk):")
results.sort(key=lambda x: x['max_dd'])
for r in results:
    marker = " <--" if r['label'] == 'BASELINE' else ""
    print("  {:45s} {:>5d} {:>5.1f}% ${:>8,.0f} ${:>6,.0f} ${:>6,.0f}{}".format(
        r['label'], r['trades'], r['wr'], r['net'], r['exp'], r['max_dd'], marker))

print("\n" + "=" * 120)
print("  END OF AVOIDANCE ANALYSIS")
print("=" * 120)
