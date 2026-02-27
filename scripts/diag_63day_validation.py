"""Validate 63-day window (Nov 18 2025 - Feb 16 2026) against v14 report.

Report target: 72 trades, ~82% WR, ~$17,513 net, PF ~12, MaxDD -$407
Previous engine: 77 trades, 71.4% WR, $17,798 net

Runs both:
1. Portfolio (all strategies shared engine)
2. Solo (each strategy in own engine)
"""
import sys, warnings, io
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
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

# Filter to 63-day window: Nov 18 2025 - Feb 16 2026
start_date = pd.Timestamp('2025-11-18')
end_date = pd.Timestamp('2026-02-16')
mask = (df['session_date'] >= start_date) & (df['session_date'] <= end_date)
df_63 = df[mask].copy()
sessions_63 = df_63['session_date'].nunique()
print(f"63-day window: {start_date.date()} to {end_date.date()}")
print(f"Sessions: {sessions_63}")
print()

# Also filter full_df for ETH data in this window
full_mask = (full_df['session_date'] >= start_date) & (full_df['session_date'] <= end_date)
full_df_63 = full_df[full_mask].copy()

abbrev = {
    'BDayStrategy': 'B-Day',
    'BearAcceptShort': 'BearAcc',
    'EdgeFadeStrategy': 'EdgeFade',
    'IBHSweepFail': 'IBHSwp',
    'OpeningRangeReversal': 'OR_Rev',
    'PDayStrategy': 'P-Day',
    'TrendDayBull': 'TrdBull',
}

# v14 report targets
report = {
    'TrendDayBull': {'trades': 8, 'wr': 75, 'pnl': 1074},
    'PDayStrategy': {'trades': 8, 'wr': 75, 'pnl': 1075},
    'BDayStrategy': {'trades': 4, 'wr': 100, 'pnl': 2285},
    'EdgeFadeStrategy': {'trades': 17, 'wr': 94, 'pnl': 7696},
    'IBHSweepFail': {'trades': 4, 'wr': 100, 'pnl': 582},
    'BearAcceptShort': {'trades': 11, 'wr': 64, 'pnl': 995},
    'OpeningRangeReversal': {'trades': 20, 'wr': 80, 'pnl': 3807},
}

# ================================================================
# RUN 1: PORTFOLIO (all strategies shared engine)
# ================================================================
print("=" * 120)
print("  RUN 1: PORTFOLIO MODE (shared engine, 63-day window)")
print("=" * 120)

strategies = get_core_strategies()
engine = BacktestEngine(
    instrument=instrument, strategies=strategies, filters=None,
    execution=ExecutionModel(instrument),
    position_mgr=PositionManager(max_drawdown=999999),
    full_df=full_df,  # Pass FULL df for overnight levels
)
result = engine.run(df_63, verbose=False)
trades = result.trades

total = len(trades)
winners = sum(1 for t in trades if t.net_pnl > 0)
net = sum(t.net_pnl for t in trades)
wr = winners / total * 100 if total else 0
gross_w = sum(t.net_pnl for t in trades if t.net_pnl > 0)
gross_l = sum(t.net_pnl for t in trades if t.net_pnl <= 0)
pf = abs(gross_w / gross_l) if gross_l else float('inf')

print(f"\n  Trades: {total} | Winners: {winners} | WR: {wr:.1f}%")
print(f"  Net P&L: ${net:,.0f} | PF: {pf:.2f}")
print(f"  Expectancy: ${net/total:,.0f}/trade")
print(f"  Max DD: ${engine.position_mgr.max_drawdown_seen:,.0f}")

# Per strategy with report comparison
print(f"\n  {'Strategy':18s} {'Eng':>5s} {'Rep':>5s} {'E_WR':>6s} {'R_WR':>6s} {'E_P&L':>10s} {'R_P&L':>10s} {'Match':>8s}")
print("  " + "-" * 80)

strat_stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0})
for t in trades:
    s = strat_stats[t.strategy_name]
    s['trades'] += 1
    s['pnl'] += t.net_pnl
    if t.net_pnl > 0:
        s['wins'] += 1

for sname in sorted(strat_stats.keys()):
    s = strat_stats[sname]
    ab = abbrev.get(sname, sname[:12])
    ewr = s['wins'] / s['trades'] * 100 if s['trades'] else 0
    r = report.get(sname, {'trades': '?', 'wr': '?', 'pnl': '?'})

    # Match assessment
    trade_match = abs(s['trades'] - r['trades']) <= 2 if isinstance(r['trades'], int) else '?'
    pnl_close = abs(s['pnl'] - r['pnl']) < 1000 if isinstance(r['pnl'], (int, float)) else '?'
    if trade_match and pnl_close:
        match = "CLOSE"
    elif trade_match:
        match = "trd OK"
    elif pnl_close:
        match = "pnl OK"
    else:
        match = "GAP"

    print("  {:18s} {:>5d} {:>5} {:>5.1f}% {:>5}% ${:>8,.0f} ${:>8} {:>8s}".format(
        ab, s['trades'], r['trades'], ewr, r['wr'], s['pnl'], r['pnl'], match))

print("  " + "-" * 80)
print("  {:18s} {:>5d} {:>5d} {:>5.1f}% {:>5s}% ${:>8,.0f} ${:>8,}".format(
    "TOTAL", total, 72, wr, "~82", net, 17513))

# ================================================================
# RUN 2: SOLO (each strategy in own engine)
# ================================================================
print(f"\n\n{'='*120}")
print(f"  RUN 2: SOLO MODE (each strategy in own engine, 63-day window)")
print(f"{'='*120}")

all_strategies = get_core_strategies()
solo_results = {}

for strat in all_strategies:
    name = strat.__class__.__name__
    ab = abbrev.get(name, name[:12])

    solo_engine = BacktestEngine(
        instrument=instrument,
        strategies=[strat],
        filters=None,
        execution=ExecutionModel(instrument),
        position_mgr=PositionManager(max_drawdown=999999),
        full_df=full_df,
    )
    solo_result = solo_engine.run(df_63, verbose=False)
    solo_trades = solo_result.trades

    n = len(solo_trades)
    w = sum(1 for t in solo_trades if t.net_pnl > 0)
    pnl = sum(t.net_pnl for t in solo_trades)
    swr = w / n * 100 if n else 0

    # DD
    eq = 0; pk = 0; mdd = 0
    for t in sorted(solo_trades, key=lambda x: (str(x.session_date), str(x.entry_time))):
        eq += t.net_pnl
        pk = max(pk, eq)
        mdd = max(mdd, pk - eq)

    solo_results[name] = {'n': n, 'wins': w, 'wr': swr, 'pnl': pnl, 'dd': mdd, 'trades': solo_trades}

# Print solo vs report comparison
print(f"\n  {'Strategy':12s} {'Solo':>5s} {'Rep':>5s} {'S_WR':>6s} {'R_WR':>6s} {'S_P&L':>10s} {'R_P&L':>10s} {'S_DD':>8s} {'Match':>8s}")
print("  " + "-" * 85)

total_solo_pnl = 0
total_solo_trades = 0
for sname in sorted(solo_results.keys()):
    sr = solo_results[sname]
    ab = abbrev.get(sname, sname[:12])
    r = report.get(sname, {'trades': '?', 'wr': '?', 'pnl': '?'})

    trade_match = abs(sr['n'] - r['trades']) <= 2 if isinstance(r['trades'], int) else '?'
    pnl_close = abs(sr['pnl'] - r['pnl']) < 1000 if isinstance(r['pnl'], (int, float)) else '?'
    if trade_match and pnl_close:
        match = "CLOSE"
    elif trade_match:
        match = "trd OK"
    elif pnl_close:
        match = "pnl OK"
    else:
        match = "GAP"

    print("  {:12s} {:>5d} {:>5} {:>5.1f}% {:>5}% ${:>8,.0f} ${:>8} ${:>6,.0f} {:>8s}".format(
        ab, sr['n'], r['trades'], sr['wr'], r['wr'], sr['pnl'], r['pnl'], sr['dd'], match))

    total_solo_pnl += sr['pnl']
    total_solo_trades += sr['n']

print("  " + "-" * 85)
print("  {:12s} {:>5d} {:>5d} {:>6s} {:>6s} ${:>8,.0f} ${:>8,}".format(
    "TOTAL", total_solo_trades, 72, "", "", total_solo_pnl, 17513))

# ================================================================
# RUN 3: Portfolio vs Solo comparison
# ================================================================
print(f"\n\n{'='*120}")
print(f"  PORTFOLIO vs SOLO COMPARISON (63-day window)")
print(f"{'='*120}")

print(f"\n  {'Strategy':12s} {'Port':>5s} {'Solo':>5s} {'P_P&L':>10s} {'S_P&L':>10s} {'Delta':>8s}")
print("  " + "-" * 55)

for sname in sorted(strat_stats.keys()):
    ps = strat_stats[sname]
    sr = solo_results.get(sname, {'n': 0, 'pnl': 0})
    ab = abbrev.get(sname, sname[:12])
    delta = ps['pnl'] - sr['pnl']
    print("  {:12s} {:>5d} {:>5d} ${:>8,.0f} ${:>8,.0f} ${:>+6,.0f}".format(
        ab, ps['trades'], sr['n'], ps['pnl'], sr['pnl'], delta))

print("  " + "-" * 55)
port_total = sum(s['pnl'] for s in strat_stats.values())
solo_total = sum(sr['pnl'] for sr in solo_results.values())
print("  {:12s} {:>5d} {:>5d} ${:>8,.0f} ${:>8,.0f} ${:>+6,.0f}".format(
    "TOTAL", total, total_solo_trades, port_total, solo_total, port_total - solo_total))

# ================================================================
# Detailed trade log for gap analysis
# ================================================================
print(f"\n\n{'='*120}")
print(f"  GAP ANALYSIS: Where engine differs from report")
print(f"{'='*120}")

for sname in sorted(strat_stats.keys()):
    ps = strat_stats[sname]
    r = report.get(sname, {'trades': 0})
    ab = abbrev.get(sname, sname[:12])

    if isinstance(r['trades'], int) and abs(ps['trades'] - r['trades']) > 2:
        print(f"\n  ** {ab}: Engine {ps['trades']}t vs Report {r['trades']}t (gap: {ps['trades'] - r['trades']:+d})")

        # Show all trades for this strategy
        strat_trades = [t for t in trades if t.strategy_name == sname]
        for t in sorted(strat_trades, key=lambda x: str(x.session_date)):
            direction = t.direction
            result_str = "WIN" if t.net_pnl > 0 else "LOSS" if t.net_pnl < -10 else "SCRATCH"
            exit_r = getattr(t, 'exit_reason', '?')
            print(f"    {str(t.session_date)[:10]} {direction:5s} entry={t.entry_price:.0f} "
                  f"stop={t.stop_price:.0f} exit={t.exit_price:.0f} "
                  f"${t.net_pnl:>+7,.0f} {result_str} ({exit_r})")

# ================================================================
# Monthly breakdown within 63-day window
# ================================================================
print(f"\n\n{'='*120}")
print(f"  MONTHLY BREAKDOWN (63-day window)")
print(f"{'='*120}")

monthly = defaultdict(lambda: defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0}))
for t in trades:
    m = monthly[str(t.session_date)[:7]][t.strategy_name]
    m['trades'] += 1
    m['pnl'] += t.net_pnl
    if t.net_pnl > 0:
        m['wins'] += 1

months = sorted(monthly.keys())
all_strat_names = sorted(strat_stats.keys())

header = "  {:>8s}".format("Month")
for s in all_strat_names:
    header += " | {:>12s}".format(abbrev.get(s, s[:8]))
header += " | {:>10s}".format("TOTAL")
print(header)
print("  " + "-" * len(header))

for month in months:
    row = "  {:>8s}".format(month)
    month_total = 0
    for s in all_strat_names:
        m = monthly[month][s]
        if m['trades'] > 0:
            row += " | {:>2d}t ${:>6,.0f}".format(m['trades'], m['pnl'])
        else:
            row += " | {:>12s}".format("---")
        month_total += m['pnl']
    row += " | ${:>8,.0f}".format(month_total)
    print(row)

# Prop firm check
print(f"\n\n  PROP FIRM CHECK (63-day window):")
print(f"  Net P&L: ${net:,.0f} (target: $9,000) -- {'PASSES' if net >= 9000 else 'FAILS'}")
print(f"  Max DD: ${engine.position_mgr.max_drawdown_seen:,.0f} (limit: $4,500) -- {'SAFE' if engine.position_mgr.max_drawdown_seen < 4500 else 'BREACHES'}")

print(f"\n{'='*120}")
print(f"  END OF 63-DAY VALIDATION")
print(f"{'='*120}")
