"""Monthly P&L report by strategy — 259-session full-year backtest.

Produces:
1. Overall summary
2. Per-strategy breakdown
3. Monthly P&L grid (strategy x month)
4. Monthly totals with cumulative equity curve
"""
import sys, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collections import defaultdict
from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from config.instruments import get_instrument
from engine.backtest import BacktestEngine
from engine.execution import ExecutionModel
from engine.position import PositionManager
from strategy import get_core_strategies

# Load data
full_df = load_csv('NQ')
df = filter_rth(full_df)
df = compute_all_features(df)
instrument = get_instrument('MNQ')

# Run backtest
strategies = get_core_strategies()
engine = BacktestEngine(
    instrument=instrument, strategies=strategies, filters=None,
    execution=ExecutionModel(instrument),
    position_mgr=PositionManager(max_drawdown=999999),
    full_df=full_df,
)
result = engine.run(df, verbose=False)
trades = result.trades
sessions = df['session_date'].nunique()
max_dd = engine.position_mgr.max_drawdown_seen

# ============================================================
# 1. OVERALL SUMMARY
# ============================================================
print("=" * 80)
print("  259-SESSION FULL-YEAR BACKTEST REPORT")
print("  Date: 2026-02-21 | Instrument: MNQ | Dataset: Feb 2025 - Feb 2026")
print("=" * 80)

total = len(trades)
winners = sum(1 for t in trades if t.net_pnl > 0)
losers = total - winners
net = sum(t.net_pnl for t in trades)
wr = winners / total * 100 if total else 0
gross_win = sum(t.net_pnl for t in trades if t.net_pnl > 0)
gross_loss = sum(t.net_pnl for t in trades if t.net_pnl <= 0)
pf = abs(gross_win / gross_loss) if gross_loss != 0 else float('inf')
avg_win = gross_win / winners if winners else 0
avg_loss = gross_loss / losers if losers else 0

print(f"\nSessions: {sessions}")
print(f"Total Trades: {total}")
print(f"Winners: {winners} | Losers: {losers} | Win Rate: {wr:.1f}%")
print(f"Net P&L: ${net:,.0f}")
print(f"Gross Win: ${gross_win:,.0f} | Gross Loss: ${gross_loss:,.0f}")
print(f"Profit Factor: {pf:.2f}")
print(f"Avg Winner: ${avg_win:,.0f} | Avg Loser: ${avg_loss:,.0f}")
print(f"Max Drawdown: ${max_dd:,.0f}")
print(f"Expectancy: ${net/total:,.0f}/trade")

# ============================================================
# 2. PER-STRATEGY BREAKDOWN
# ============================================================
print("\n" + "=" * 80)
print("  PER-STRATEGY BREAKDOWN")
print("=" * 80)

strat_stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0, 'gross_w': 0.0, 'gross_l': 0.0})
for t in trades:
    s = strat_stats[t.strategy_name]
    s['trades'] += 1
    s['pnl'] += t.net_pnl
    if t.net_pnl > 0:
        s['wins'] += 1
        s['gross_w'] += t.net_pnl
    else:
        s['gross_l'] += t.net_pnl

fmt = "{:25s} {:>6s} {:>5s} {:>6s} {:>10s} {:>8s} {:>6s}"
print(fmt.format("Strategy", "Trades", "Wins", "WR", "Net P&L", "$/trade", "PF"))
print("-" * 75)
for name in sorted(strat_stats):
    s = strat_stats[name]
    wr2 = s['wins'] / s['trades'] * 100 if s['trades'] else 0
    exp = s['pnl'] / s['trades'] if s['trades'] else 0
    pf2 = abs(s['gross_w'] / s['gross_l']) if s['gross_l'] != 0 else float('inf')
    print("{:25s} {:>6d} {:>5d} {:>5.1f}% ${:>8,.0f} ${:>6,.0f} {:>5.2f}".format(
        name, s['trades'], s['wins'], wr2, s['pnl'], exp, pf2))

print("-" * 75)
print("{:25s} {:>6d} {:>5d} {:>5.1f}% ${:>8,.0f} ${:>6,.0f} {:>5.2f}".format(
    "TOTAL", total, winners, wr, net, net/total, pf))

# ============================================================
# 3. MONTHLY P&L GRID
# ============================================================
print("\n" + "=" * 120)
print("  MONTHLY P&L BY STRATEGY (trades / net P&L)")
print("=" * 120)

monthly = defaultdict(lambda: defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0}))
for t in trades:
    month = str(t.session_date)[:7]
    m = monthly[month][t.strategy_name]
    m['trades'] += 1
    m['pnl'] += t.net_pnl
    if t.net_pnl > 0:
        m['wins'] += 1

all_strats = sorted(strat_stats.keys())

# Use abbreviated strategy names
abbrev = {
    'BDayStrategy': 'B-Day',
    'BearAcceptShort': 'BearAcc',
    'EdgeFadeStrategy': 'EdgeFade',
    'IBHSweepFail': 'IBHSwp',
    'OpeningRangeReversal': 'OR Rev',
    'PDayStrategy': 'P-Day',
    'TrendDayBull': 'TrdBull',
}

header = "{:>8s}".format("Month")
for s in all_strats:
    ab = abbrev.get(s, s[:8])
    header += " | {:>12s}".format(ab)
header += " | {:>10s} | {:>10s}".format("TOTAL", "Cumul")
print(header)
print("-" * len(header))

months = sorted(monthly.keys())
cumulative = 0
for month in months:
    row = "{:>8s}".format(month)
    month_total = 0
    for s in all_strats:
        m = monthly[month][s]
        if m['trades'] > 0:
            row += " | {:>2d}t ${:>6,.0f}".format(m['trades'], m['pnl'])
        else:
            row += " | {:>12s}".format("---")
        month_total += m['pnl']
    cumulative += month_total
    row += " | ${:>8,.0f} | ${:>8,.0f}".format(month_total, cumulative)
    print(row)

# Totals
print("-" * len(header))
row = "{:>8s}".format("TOTAL")
for s in all_strats:
    st = strat_stats[s]
    row += " | {:>2d}t ${:>6,.0f}".format(st['trades'], st['pnl'])
row += " | ${:>8,.0f} |".format(net)
print(row)

# ============================================================
# 4. MONTHLY WIN RATE GRID
# ============================================================
print("\n" + "=" * 100)
print("  MONTHLY WIN RATE BY STRATEGY")
print("=" * 100)

header = "{:>8s}".format("Month")
for s in all_strats:
    ab = abbrev.get(s, s[:8])
    header += " | {:>8s}".format(ab)
header += " | {:>8s}".format("TOTAL")
print(header)
print("-" * len(header))

for month in months:
    row = "{:>8s}".format(month)
    month_trades = 0
    month_wins = 0
    for s in all_strats:
        m = monthly[month][s]
        if m['trades'] > 0:
            wr3 = m['wins'] / m['trades'] * 100
            row += " | {:>5.0f}%/{:d}".format(wr3, m['trades'])
        else:
            row += " | {:>8s}".format("---")
        month_trades += m['trades']
        month_wins += m['wins']
    total_wr = month_wins / month_trades * 100 if month_trades else 0
    row += " | {:>5.0f}%/{:d}".format(total_wr, month_trades)
    print(row)

# ============================================================
# 5. BEST/WORST DAYS
# ============================================================
print("\n" + "=" * 80)
print("  BEST AND WORST TRADING DAYS")
print("=" * 80)

daily_pnl = defaultdict(float)
daily_trades = defaultdict(int)
for t in trades:
    daily_pnl[str(t.session_date)] += t.net_pnl
    daily_trades[str(t.session_date)] += 1

sorted_days = sorted(daily_pnl.items(), key=lambda x: x[1], reverse=True)
print("\nTop 10 Best Days:")
for date, pnl in sorted_days[:10]:
    print("  {} : ${:>8,.0f} ({} trades)".format(date, pnl, daily_trades[date]))

print("\nTop 10 Worst Days:")
for date, pnl in sorted_days[-10:]:
    print("  {} : ${:>8,.0f} ({} trades)".format(date, pnl, daily_trades[date]))

# ============================================================
# 6. DRAWDOWN ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("  EQUITY CURVE & DRAWDOWN")
print("=" * 80)

equity = 0
peak = 0
max_dd = 0
dd_start = None
max_dd_start = None
max_dd_end = None

equity_curve = []
for t in sorted(trades, key=lambda x: str(x.session_date) + str(x.entry_time)):
    equity += t.net_pnl
    equity_curve.append((str(t.session_date), equity))
    if equity > peak:
        peak = equity
        dd_start = str(t.session_date)
    dd = peak - equity
    if dd > max_dd:
        max_dd = dd
        max_dd_start = dd_start
        max_dd_end = str(t.session_date)

print(f"\nPeak Equity: ${peak:,.0f}")
print(f"Final Equity: ${equity:,.0f}")
print(f"Max Drawdown: ${max_dd:,.0f}")
if max_dd_start and max_dd_end:
    print(f"DD Period: {max_dd_start} to {max_dd_end}")

# Consecutive losses
max_consec = 0
current_consec = 0
for t in sorted(trades, key=lambda x: str(x.session_date) + str(x.entry_time)):
    if t.net_pnl <= 0:
        current_consec += 1
        max_consec = max(max_consec, current_consec)
    else:
        current_consec = 0
print(f"Max Consecutive Losses: {max_consec}")

print("\n" + "=" * 80)
print("  END OF REPORT")
print("=" * 80)
