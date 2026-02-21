"""Run each strategy in its own isolated account.

Each strategy gets its own BacktestEngine + PositionManager so there's
zero interaction between strategies. This shows true standalone performance
within the risk parameters.
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
from strategy.base import StrategyBase

full_df = load_csv('NQ')
df = filter_rth(full_df)
df = compute_all_features(df)
instrument = get_instrument('MNQ')
point_value = instrument.point_value

# Get all strategy instances
all_strategies = get_core_strategies()

abbrev = {
    'BDayStrategy': 'B-Day',
    'BearAcceptShort': 'BearAcc',
    'EdgeFadeStrategy': 'EdgeFade',
    'IBHSweepFail': 'IBHSwp',
    'OpeningRangeReversal': 'OR_Rev',
    'PDayStrategy': 'P-Day',
    'TrendDayBull': 'TrdBull',
}

print("=" * 120)
print("  SOLO STRATEGY ACCOUNTS -- Each strategy runs independently")
print("  259 sessions | MNQ ($2/pt) | max_drawdown=999999 (no DD limit)")
print("=" * 120)

# ── Run each strategy solo ──
solo_results = {}

for strat in all_strategies:
    name = strat.__class__.__name__
    ab = abbrev.get(name, name[:12])

    engine = BacktestEngine(
        instrument=instrument,
        strategies=[strat],
        filters=None,
        execution=ExecutionModel(instrument),
        position_mgr=PositionManager(max_drawdown=999999),
        full_df=full_df,
    )
    result = engine.run(df, verbose=False)
    trades = result.trades

    if not trades:
        print(f"\n  {ab}: 0 trades")
        solo_results[name] = {'trades': [], 'n': 0}
        continue

    n = len(trades)
    wins = sum(1 for t in trades if t.net_pnl > 0)
    net = sum(t.net_pnl for t in trades)
    wr = wins / n * 100
    gross_w = sum(t.net_pnl for t in trades if t.net_pnl > 0)
    gross_l = sum(t.net_pnl for t in trades if t.net_pnl <= 0)
    pf = abs(gross_w / gross_l) if gross_l else float('inf')
    avg_risk = np.mean([abs(t.entry_price - t.stop_price) for t in trades])

    # Equity curve and drawdown
    equity = 0
    peak = 0
    max_dd = 0
    eq_curve = []
    for t in sorted(trades, key=lambda x: (str(x.session_date), str(x.entry_time))):
        equity += t.net_pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
        eq_curve.append(equity)

    # Monthly breakdown
    monthly = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0})
    for t in trades:
        m = monthly[str(t.session_date)[:7]]
        m['trades'] += 1
        m['pnl'] += t.net_pnl
        if t.net_pnl > 0:
            m['wins'] += 1

    # Sessions traded
    sessions_traded = len(set(str(t.session_date) for t in trades))

    # Consecutive losses
    max_consec = 0
    curr = 0
    for t in sorted(trades, key=lambda x: (str(x.session_date), str(x.entry_time))):
        if t.net_pnl <= 0:
            curr += 1
            max_consec = max(max_consec, curr)
        else:
            curr = 0

    # Prop firm analysis
    cumul = 0
    sessions_to_9k = None
    breaches_dd = False
    for t in sorted(trades, key=lambda x: (str(x.session_date), str(x.entry_time))):
        cumul += t.net_pnl
    if max_dd > 4500:
        breaches_dd = True

    solo_results[name] = {
        'trades': trades, 'n': n, 'wins': wins, 'wr': wr,
        'net': net, 'pf': pf, 'max_dd': max_dd, 'avg_risk': avg_risk,
        'monthly': dict(monthly), 'sessions': sessions_traded,
        'max_consec_loss': max_consec, 'final_equity': equity,
        'breaches_dd': breaches_dd,
    }

    print(f"\n  {'='*100}")
    print(f"  {ab} -- SOLO ACCOUNT")
    print(f"  {'='*100}")
    print(f"  Trades: {n} | Sessions: {sessions_traded}/259 | WR: {wr:.1f}%")
    print(f"  Net P&L: ${net:,.0f} | PF: {pf:.2f} | Exp: ${net/n:,.0f}/trade")
    print(f"  Avg Risk: {avg_risk:.0f} pts (${avg_risk*point_value:,.0f})")
    print(f"  Avg Winner: ${gross_w/wins:,.0f} | Avg Loser: ${gross_l/(n-wins):,.0f}" if wins and n-wins else "")
    print(f"  Max DD: ${max_dd:,.0f} | Max Consec Loss: {max_consec}")
    print(f"  Prop Firm DD Breach ($4,500): {'YES -- WOULD BLOW' if breaches_dd else 'NO -- safe'}")

    # Monthly P&L
    months = sorted(monthly.keys())
    losing_months = [m for m in months if monthly[m]['pnl'] < 0]
    winning_months = [m for m in months if monthly[m]['pnl'] > 0]
    print(f"  Winning Months: {len(winning_months)}/{len(months)} | Losing Months: {len(losing_months)}/{len(months)}")

    print(f"\n  Monthly:")
    for month in months:
        m = monthly[month]
        mwr = m['wins'] / m['trades'] * 100 if m['trades'] else 0
        bar_len = int(abs(m['pnl']) / 500)
        if m['pnl'] >= 0:
            bar = '+' * min(bar_len, 40)
        else:
            bar = '-' * min(bar_len, 40)
        print(f"    {month}: {m['trades']:>3d}t {mwr:>3.0f}%WR ${m['pnl']:>7,.0f} {bar}")

# ══════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ══════════════════════════════════════════════════════════════
print(f"\n\n{'='*120}")
print(f"  COMPARISON: SOLO vs PORTFOLIO")
print(f"{'='*120}")

# Run portfolio for comparison
portfolio_engine = BacktestEngine(
    instrument=instrument,
    strategies=get_core_strategies(),
    filters=None,
    execution=ExecutionModel(instrument),
    position_mgr=PositionManager(max_drawdown=999999),
    full_df=full_df,
)
portfolio_result = portfolio_engine.run(df, verbose=False)

# Portfolio per-strategy
port_stats = defaultdict(lambda: {'n': 0, 'wins': 0, 'pnl': 0})
for t in portfolio_result.trades:
    s = port_stats[t.strategy_name]
    s['n'] += 1
    s['pnl'] += t.net_pnl
    if t.net_pnl > 0:
        s['wins'] += 1

fmt = "  {:12s} {:>5s} {:>6s} {:>10s} {:>8s} {:>7s} {:>5s} | {:>5s} {:>6s} {:>10s} | {:>8s}"
print(fmt.format("Strategy", "Trd", "WR", "Net P&L", "$/trade", "MaxDD", "CL",
                  "Trd_P", "WR_P", "P&L_P", "Delta"))
print("  " + "-" * 110)

total_solo = 0
total_port = 0
for strat in all_strategies:
    name = strat.__class__.__name__
    ab = abbrev.get(name, name[:12])
    sr = solo_results.get(name, {})
    pr = port_stats.get(name, {'n': 0, 'wins': 0, 'pnl': 0})

    if sr.get('n', 0) == 0 and pr['n'] == 0:
        continue

    sn = sr.get('n', 0)
    sw = sr.get('wins', 0)
    swr = sr.get('wr', 0)
    spnl = sr.get('net', 0)
    sexp = spnl / sn if sn else 0
    sdd = sr.get('max_dd', 0)
    scl = sr.get('max_consec_loss', 0)

    pn = pr['n']
    pwr = pr['wins'] / pn * 100 if pn else 0
    ppnl = pr['pnl']

    delta = spnl - ppnl
    total_solo += spnl
    total_port += ppnl

    print("  {:12s} {:>5d} {:>5.1f}% ${:>8,.0f} ${:>6,.0f} ${:>5,.0f} {:>5d} | {:>5d} {:>5.1f}% ${:>8,.0f} | ${:>+6,.0f}".format(
        ab, sn, swr, spnl, sexp, sdd, scl,
        pn, pwr, ppnl, delta))

print("  " + "-" * 110)
print("  {:12s} {:>5d} {:>6s} ${:>8,.0f} {:>8s} {:>7s} {:>5s} | {:>5d} {:>6s} ${:>8,.0f} | ${:>+6,.0f}".format(
    "TOTAL",
    sum(sr.get('n', 0) for sr in solo_results.values()),
    "",
    total_solo, "", "", "",
    len(portfolio_result.trades), "",
    total_port,
    total_solo - total_port))

# ══════════════════════════════════════════════════════════════
# PROP FIRM VIABILITY PER STRATEGY
# ══════════════════════════════════════════════════════════════
print(f"\n\n{'='*120}")
print(f"  PROP FIRM VIABILITY (Tradeify $150K Select Flex: $9K target, $4.5K trailing DD)")
print(f"{'='*120}")

fmt = "  {:12s} {:>10s} {:>8s} {:>8s} {:>10s} {:>12s}"
print(fmt.format("Strategy", "Net P&L", "Max DD", "DD%", "Hit $9K?", "Verdict"))
print("  " + "-" * 70)

for strat in all_strategies:
    name = strat.__class__.__name__
    ab = abbrev.get(name, name[:12])
    sr = solo_results.get(name, {})
    if sr.get('n', 0) == 0:
        continue

    net = sr['net']
    dd = sr['max_dd']
    dd_pct = dd / 4500 * 100

    if net >= 9000 and dd < 4500:
        verdict = "PASSES"
    elif net >= 9000 and dd >= 4500:
        verdict = "BLOWS DD"
    elif net > 0 and dd < 4500:
        verdict = "TOO SLOW"
    else:
        verdict = "LOSES"

    print("  {:12s} ${:>8,.0f} ${:>6,.0f} {:>6.0f}% {:>10s} {:>12s}".format(
        ab, net, dd, dd_pct, "YES" if net >= 9000 else "NO", verdict))

# Combined best strategies
print(f"\n  COMBINED ACCOUNTS (sum of solo strategies):")
combos = [
    ('OR_Rev + EdgeFade', ['OpeningRangeReversal', 'EdgeFadeStrategy']),
    ('OR_Rev + EdgeFade + B-Day', ['OpeningRangeReversal', 'EdgeFadeStrategy', 'BDayStrategy']),
    ('All 7', list(solo_results.keys())),
]

for label, strats in combos:
    combo_net = sum(solo_results.get(s, {}).get('net', 0) for s in strats)
    # For DD: we need to simulate combined equity curve
    all_combo_trades = []
    for s in strats:
        for t in solo_results.get(s, {}).get('trades', []):
            all_combo_trades.append(t)
    all_combo_trades.sort(key=lambda x: (str(x.session_date), str(x.entry_time)))

    eq = 0
    pk = 0
    mdd = 0
    for t in all_combo_trades:
        eq += t.net_pnl
        pk = max(pk, eq)
        mdd = max(mdd, pk - eq)

    dd_pct = mdd / 4500 * 100
    n_trades = len(all_combo_trades)

    if combo_net >= 9000 and mdd < 4500:
        verdict = "PASSES"
    elif combo_net >= 9000 and mdd >= 4500:
        verdict = "BLOWS DD"
    else:
        verdict = "TOO SLOW" if combo_net > 0 else "LOSES"

    print("  {:30s} {:>4d}t ${:>8,.0f} DD${:>5,.0f} ({:.0f}%) {:>10s}".format(
        label, n_trades, combo_net, mdd, dd_pct, verdict))

print(f"\n{'='*120}")
print(f"  END OF SOLO STRATEGY ANALYSIS")
print(f"{'='*120}")
