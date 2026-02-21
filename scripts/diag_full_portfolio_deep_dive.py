"""Full portfolio deep-dive report.

Covers:
1. Monthly breakdown by strategy (trades, sessions, regime, key findings)
2. Profit distribution (histogram, percentiles)
3. Risk per trade, expectancy, win rate
4. Fixed target comparison (1R, 1.5R, 2R, 3R) vs trailing
5. Losing month deep-dive (why we lose, strategy weakness)
6. Drawdown analysis + position sizing (scale down vs compound)
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

# --Load & run ──────────────────────────────────────────────
full_df = load_csv('NQ')
df = filter_rth(full_df)
df = compute_all_features(df)
instrument = get_instrument('MNQ')
point_value = instrument.point_value  # $2 for MNQ
commission_rt = 1.24 * 2
slippage_pts = 0.5

strategies = get_core_strategies()
engine = BacktestEngine(
    instrument=instrument, strategies=strategies, filters=None,
    execution=ExecutionModel(instrument),
    position_mgr=PositionManager(max_drawdown=999999),
    full_df=full_df,
)
result = engine.run(df, verbose=False)
trades = result.trades
print(f"Loaded {len(trades)} trades across {df['session_date'].nunique()} sessions\n")

# --Helpers ─────────────────────────────────────────────────
sessions_by_month = defaultdict(set)
for _, row in df.iterrows():
    d = str(row['session_date'])
    sessions_by_month[d[:7]].add(d)

# IB range per session for regime classification
session_ib = {}
for session_date, sdf in df.groupby('session_date'):
    sd = str(session_date)
    ib_bars = sdf.head(60)
    if len(ib_bars) >= 10:
        session_ib[sd] = ib_bars['high'].max() - ib_bars['low'].min()

def regime_label(ib_range):
    if ib_range < 100: return 'LOW'
    elif ib_range < 150: return 'MED'
    elif ib_range < 250: return 'NORMAL'
    else: return 'HIGH'

abbrev = {
    'BDayStrategy': 'B-Day',
    'BearAcceptShort': 'BearAcc',
    'EdgeFadeStrategy': 'EdgeFade',
    'IBHSweepFail': 'IBHSwp',
    'OpeningRangeReversal': 'OR_Rev',
    'PDayStrategy': 'P-Day',
    'TrendDayBull': 'TrdBull',
}

all_strats = sorted(set(t.strategy_name for t in trades))

# Pre-compute per-trade data
trade_data = []
for t in trades:
    risk_pts = abs(t.entry_price - t.stop_price)
    if t.direction == 'LONG':
        gross_pts = t.exit_price - t.entry_price
    else:
        gross_pts = t.entry_price - t.exit_price
    r_multiple = gross_pts / risk_pts if risk_pts > 0 else 0
    risk_dollars = risk_pts * point_value
    trade_data.append({
        'trade': t,
        'month': str(t.session_date)[:7],
        'date': str(t.session_date),
        'strategy': t.strategy_name,
        'direction': t.direction,
        'risk_pts': risk_pts,
        'risk_dollars': risk_dollars,
        'gross_pts': gross_pts,
        'r_multiple': r_multiple,
        'net_pnl': t.net_pnl,
        'exit_reason': getattr(t, 'exit_reason', 'UNKNOWN'),
    })

# ════════════════════════════════════════════════════════════
# SECTION 1: OVERALL SUMMARY
# ════════════════════════════════════════════════════════════
print("=" * 100)
print("  SECTION 1: OVERALL PORTFOLIO SUMMARY")
print("=" * 100)

total = len(trades)
winners = [td for td in trade_data if td['net_pnl'] > 0]
losers = [td for td in trade_data if td['net_pnl'] <= 0]
net = sum(td['net_pnl'] for td in trade_data)
gross_w = sum(td['net_pnl'] for td in winners)
gross_l = sum(td['net_pnl'] for td in losers)
pf = abs(gross_w / gross_l) if gross_l else float('inf')

avg_risk_pts = np.mean([td['risk_pts'] for td in trade_data])
avg_risk_dollars = np.mean([td['risk_dollars'] for td in trade_data])
avg_r_win = np.mean([td['r_multiple'] for td in winners]) if winners else 0
avg_r_loss = np.mean([td['r_multiple'] for td in losers]) if losers else 0

print(f"\n  Trades: {total} | Winners: {len(winners)} | Losers: {len(losers)}")
print(f"  Win Rate: {len(winners)/total*100:.1f}%")
print(f"  Net P&L: ${net:,.0f}")
print(f"  Profit Factor: {pf:.2f}")
print(f"  Expectancy: ${net/total:,.0f}/trade ({net/total/avg_risk_dollars:.2f}R/trade)")
print(f"\n  Avg Risk: {avg_risk_pts:.0f} pts (${avg_risk_dollars:,.0f})")
print(f"  Avg Winner: ${np.mean([td['net_pnl'] for td in winners]):,.0f} ({avg_r_win:.2f}R)")
print(f"  Avg Loser: ${np.mean([td['net_pnl'] for td in losers]):,.0f} ({avg_r_loss:.2f}R)")
print(f"  Max Winner: ${max(td['net_pnl'] for td in trade_data):,.0f}")
print(f"  Max Loser: ${min(td['net_pnl'] for td in trade_data):,.0f}")

# ════════════════════════════════════════════════════════════
# SECTION 2: PER-STRATEGY RISK & EXPECTANCY
# ════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("  SECTION 2: PER-STRATEGY RISK, EXPECTANCY & WIN RATE")
print("=" * 100)

fmt = "{:18s} {:>5s} {:>5s} {:>6s} {:>8s} {:>8s} {:>7s} {:>7s} {:>7s} {:>7s} {:>5s}"
print(fmt.format("Strategy", "Trd", "Win", "WR", "Net", "$/trd", "AvgRsk", "AvgW_R", "AvgL_R", "Expect", "PF"))
print("-" * 100)

for sname in all_strats:
    st = [td for td in trade_data if td['strategy'] == sname]
    sw = [td for td in st if td['net_pnl'] > 0]
    sl = [td for td in st if td['net_pnl'] <= 0]
    n = len(st)
    nw = len(sw)
    wr = nw / n * 100 if n else 0
    pnl = sum(td['net_pnl'] for td in st)
    exp = pnl / n if n else 0
    avg_rsk = np.mean([td['risk_pts'] for td in st])
    avg_w_r = np.mean([td['r_multiple'] for td in sw]) if sw else 0
    avg_l_r = np.mean([td['r_multiple'] for td in sl]) if sl else 0
    gw = sum(td['net_pnl'] for td in sw)
    gl = sum(td['net_pnl'] for td in sl)
    spf = abs(gw / gl) if gl else float('inf')
    ab = abbrev.get(sname, sname[:12])
    print("{:18s} {:>5d} {:>5d} {:>5.1f}% ${:>6,.0f} ${:>6,.0f} {:>5.0f}pt {:>+5.2f}R {:>+5.2f}R {:>+5.2f}R {:>5.2f}".format(
        ab, n, nw, wr, pnl, exp, avg_rsk, avg_w_r, avg_l_r, exp / (avg_rsk * point_value) if avg_rsk > 0 else 0, spf))

# ════════════════════════════════════════════════════════════
# SECTION 3: PROFIT DISTRIBUTION
# ════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("  SECTION 3: PROFIT DISTRIBUTION")
print("=" * 100)

pnls = [td['net_pnl'] for td in trade_data]
r_mults = [td['r_multiple'] for td in trade_data]

percentiles = [5, 10, 25, 50, 75, 90, 95]
print("\n  P&L Percentiles:")
for p in percentiles:
    val = np.percentile(pnls, p)
    print(f"    {p:3d}th: ${val:>8,.0f}")

print(f"\n  P&L Standard Deviation: ${np.std(pnls):,.0f}")
skew = np.mean(((np.array(pnls) - np.mean(pnls)) / np.std(pnls))**3) if np.std(pnls) > 0 else 0
print(f"  P&L Skewness: {skew:.2f}")

# P&L histogram (text-based)
print("\n  P&L Distribution (histogram):")
bins = [-2000, -1000, -500, -250, -100, 0, 100, 250, 500, 1000, 2000, 5000]
hist, _ = np.histogram(pnls, bins=bins)
max_bar = max(hist) if max(hist) > 0 else 1
for i in range(len(bins) - 1):
    bar_len = int(hist[i] / max_bar * 40)
    bar = "#" * bar_len
    print(f"  ${bins[i]:>6,} to ${bins[i+1]:>6,}: {hist[i]:>3d} {bar}")

# R-multiple distribution
print("\n  R-Multiple Distribution:")
r_bins = [-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 5, 10]
r_hist, _ = np.histogram(r_mults, bins=r_bins)
max_bar = max(r_hist) if max(r_hist) > 0 else 1
for i in range(len(r_bins) - 1):
    bar_len = int(r_hist[i] / max_bar * 40)
    bar = "#" * bar_len
    print(f"  {r_bins[i]:>+5.1f}R to {r_bins[i+1]:>+5.1f}R: {r_hist[i]:>3d} {bar}")

# Per-strategy P&L distribution
print("\n  Per-Strategy P&L Spread (P5 / P25 / Median / P75 / P95):")
for sname in all_strats:
    st_pnls = [td['net_pnl'] for td in trade_data if td['strategy'] == sname]
    if len(st_pnls) < 3:
        continue
    ab = abbrev.get(sname, sname[:12])
    p5, p25, p50, p75, p95 = [np.percentile(st_pnls, p) for p in [5, 25, 50, 75, 95]]
    print(f"    {ab:18s}: ${p5:>7,.0f} / ${p25:>7,.0f} / ${p50:>7,.0f} / ${p75:>7,.0f} / ${p95:>7,.0f}")

# ════════════════════════════════════════════════════════════
# SECTION 4: FIXED TARGET COMPARISON (1R, 1.5R, 2R, 3R) vs ACTUAL
# ════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("  SECTION 4: WHAT-IF TARGET COMPARISON (1R, 1.5R, 2R, 3R) vs TRAILING")
print("=" * 100)

# We need bar-by-bar sim for each target level
sessions_grouped = df.groupby('session_date')
import pandas as pd

def sim_fixed_target(trade, post_bars, target_r):
    """Simulate trade with fixed R-multiple target."""
    entry = trade.entry_price
    stop = trade.stop_price
    risk = abs(entry - stop)
    if risk <= 0:
        return trade.net_pnl, 'INVALID'

    direction = trade.direction
    if direction == 'LONG':
        target = entry + target_r * risk
    else:
        target = entry - target_r * risk

    for i in range(len(post_bars)):
        bar = post_bars.iloc[i]
        # Check stop first
        if direction == 'LONG':
            if bar['low'] <= stop:
                gross = (stop - entry) * point_value
                return gross - commission_rt - slippage_pts * point_value, 'STOP'
            if bar['high'] >= target:
                gross = (target - entry) * point_value
                return gross - commission_rt - slippage_pts * point_value, 'TARGET'
        else:
            if bar['high'] >= stop:
                gross = (entry - stop) * point_value
                return gross - commission_rt - slippage_pts * point_value, 'STOP'
            if bar['low'] <= target:
                gross = (entry - target) * point_value
                return gross - commission_rt - slippage_pts * point_value, 'TARGET'

        # EOD check
        if hasattr(bar.get('timestamp', None), 'time'):
            if bar['timestamp'].time() >= pd.Timestamp('15:59').time():
                if direction == 'LONG':
                    gross = (bar['close'] - entry) * point_value
                else:
                    gross = (entry - bar['close']) * point_value
                return gross - commission_rt - slippage_pts * point_value, 'EOD'

    # End of data
    last = post_bars.iloc[-1]
    if direction == 'LONG':
        gross = (last['close'] - entry) * point_value
    else:
        gross = (entry - last['close']) * point_value
    return gross - commission_rt - slippage_pts * point_value, 'EOD'

def sim_trail(trade, post_bars, trigger_r, trail_r):
    """Simulate trade with R-based trailing stop (no fixed target)."""
    entry = trade.entry_price
    stop = trade.stop_price
    risk = abs(entry - stop)
    if risk <= 0:
        return trade.net_pnl, 'INVALID'

    direction = trade.direction
    current_stop = stop
    peak_fav = 0.0
    trail_active = False

    for i in range(len(post_bars)):
        bar = post_bars.iloc[i]

        if direction == 'LONG':
            fav = bar['high'] - entry
            peak_fav = max(peak_fav, fav)
            if peak_fav >= trigger_r * risk and not trail_active:
                trail_active = True
            if trail_active:
                trail_level = (entry + peak_fav) - trail_r * risk
                current_stop = max(current_stop, trail_level)
            if bar['low'] <= current_stop:
                gross = (current_stop - entry) * point_value
                return gross - commission_rt - slippage_pts * point_value, 'TRAIL' if trail_active else 'STOP'
        else:
            fav = entry - bar['low']
            peak_fav = max(peak_fav, fav)
            if peak_fav >= trigger_r * risk and not trail_active:
                trail_active = True
            if trail_active:
                trail_level = (entry - peak_fav) + trail_r * risk
                current_stop = min(current_stop, trail_level)
            if bar['high'] >= current_stop:
                gross = (entry - current_stop) * point_value
                return gross - commission_rt - slippage_pts * point_value, 'TRAIL' if trail_active else 'STOP'

        if hasattr(bar.get('timestamp', None), 'time'):
            if bar['timestamp'].time() >= pd.Timestamp('15:59').time():
                if direction == 'LONG':
                    gross = (bar['close'] - entry) * point_value
                else:
                    gross = (entry - bar['close']) * point_value
                return gross - commission_rt - slippage_pts * point_value, 'EOD'

    last = post_bars.iloc[-1]
    if direction == 'LONG':
        gross = (last['close'] - entry) * point_value
    else:
        gross = (entry - last['close']) * point_value
    return gross - commission_rt - slippage_pts * point_value, 'EOD'

def get_post_bars(trade):
    session_date = trade.session_date
    try:
        sdf = sessions_grouped.get_group(pd.Timestamp(session_date).date())
    except (KeyError, ValueError):
        try:
            sdf = sessions_grouped.get_group(session_date)
        except KeyError:
            return None
    entry_ts = trade.entry_time
    mask = sdf['timestamp'] >= pd.Timestamp(entry_ts)
    post = sdf[mask]
    return post if len(post) > 0 else None

# Run what-if for all trades
target_levels = [1.0, 1.5, 2.0, 3.0]
trail_configs = [
    ('Trail@0.5R', 0.5, 0.5),
    ('Trail@0.75R', 0.75, 0.75),
    ('Trail@1.0R', 1.0, 1.0),
]

# Results: {config_name: {strategy: {pnl, wins, trades}}}
whatif_results = defaultdict(lambda: defaultdict(lambda: {'pnl': 0, 'wins': 0, 'trades': 0}))

print("\n  Running what-if simulations...", end='', flush=True)

for td in trade_data:
    t = td['trade']
    post_bars = get_post_bars(t)

    # Fixed targets
    for tr in target_levels:
        config_name = f"Fixed {tr:.1f}R"
        if post_bars is not None and len(post_bars) >= 2:
            pnl, reason = sim_fixed_target(t, post_bars, tr)
        else:
            pnl = td['net_pnl']
        r = whatif_results[config_name][td['strategy']]
        r['pnl'] += pnl
        r['trades'] += 1
        if pnl > 0:
            r['wins'] += 1

    # Trail configs
    for config_name, trigger, trail in trail_configs:
        if post_bars is not None and len(post_bars) >= 2:
            pnl, reason = sim_trail(t, post_bars, trigger, trail)
        else:
            pnl = td['net_pnl']
        r = whatif_results[config_name][td['strategy']]
        r['pnl'] += pnl
        r['trades'] += 1
        if pnl > 0:
            r['wins'] += 1

# Actual baseline
for td in trade_data:
    r = whatif_results['ACTUAL'][td['strategy']]
    r['pnl'] += td['net_pnl']
    r['trades'] += 1
    if td['net_pnl'] > 0:
        r['wins'] += 1

print(" done!")

# Print comparison table
configs_ordered = ['ACTUAL'] + [f"Fixed {tr:.1f}R" for tr in target_levels] + [c[0] for c in trail_configs]

print("\n  PORTFOLIO-LEVEL COMPARISON:")
fmt = "  {:15s} {:>6s} {:>5s} {:>6s} {:>10s} {:>8s}"
print(fmt.format("Config", "Trades", "Wins", "WR", "Net P&L", "$/trade"))
print("  " + "-" * 60)

for config in configs_ordered:
    n = sum(whatif_results[config][s]['trades'] for s in all_strats)
    w = sum(whatif_results[config][s]['wins'] for s in all_strats)
    p = sum(whatif_results[config][s]['pnl'] for s in all_strats)
    wr = w / n * 100 if n else 0
    exp = p / n if n else 0
    marker = " <-- current" if config == 'ACTUAL' else ""
    print("  {:15s} {:>6d} {:>5d} {:>5.1f}% ${:>8,.0f} ${:>6,.0f}{}".format(
        config, n, w, wr, p, exp, marker))

# Per-strategy comparison
print("\n  PER-STRATEGY COMPARISON (Net P&L by config):")
header_row = "  {:<15s}".format("Config")
for s in all_strats:
    header_row += " | {:>10s}".format(abbrev.get(s, s[:8]))
header_row += " | {:>10s}".format("TOTAL")
print(header_row)
print("  " + "-" * (15 + (len(all_strats) + 1) * 13))

for config in configs_ordered:
    row = "  {:15s}".format(config)
    total_cfg = 0
    for s in all_strats:
        r = whatif_results[config][s]
        row += " | ${:>8,.0f}".format(r['pnl'])
        total_cfg += r['pnl']
    row += " | ${:>8,.0f}".format(total_cfg)
    if config == 'ACTUAL':
        row += " <--"
    print(row)

# ════════════════════════════════════════════════════════════
# SECTION 5: MONTHLY DEEP DIVE - WHY WE LOSE
# ════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("  SECTION 5: MONTHLY DEEP DIVE - REGIME, SESSIONS, KEY FINDINGS")
print("=" * 100)

months = sorted(set(td['month'] for td in trade_data))

for month in months:
    month_trades = [td for td in trade_data if td['month'] == month]
    month_winners = [td for td in month_trades if td['net_pnl'] > 0]
    month_pnl = sum(td['net_pnl'] for td in month_trades)
    month_wr = len(month_winners) / len(month_trades) * 100 if month_trades else 0

    # Sessions this month
    total_sessions = len(sessions_by_month[month])
    traded_sessions = len(set(td['date'] for td in month_trades))

    # Regime distribution
    month_ibs = [session_ib.get(td['date'], 0) for td in month_trades]
    avg_ib = np.mean(month_ibs) if month_ibs else 0

    regime_counts = defaultdict(int)
    for td in month_trades:
        ib = session_ib.get(td['date'], 150)
        regime_counts[regime_label(ib)] += 1

    regime_str = ", ".join(f"{k}:{v}" for k, v in sorted(regime_counts.items()))

    status = "PROFIT" if month_pnl > 0 else "LOSS"
    marker = " *** LOSING MONTH ***" if month_pnl < 0 else ""

    print(f"\n  --{month} --{status}{marker}")
    print(f"  Sessions: {traded_sessions}/{total_sessions} traded | Trades: {len(month_trades)} | WR: {month_wr:.0f}% | Net: ${month_pnl:,.0f}")
    print(f"  Avg IB Range: {avg_ib:.0f} pts | Regimes: {regime_str}")

    # Per-strategy breakdown this month
    for sname in all_strats:
        st = [td for td in month_trades if td['strategy'] == sname]
        if not st:
            continue
        sw = [td for td in st if td['net_pnl'] > 0]
        sl = [td for td in st if td['net_pnl'] <= 0]
        spnl = sum(td['net_pnl'] for td in st)
        swr = len(sw) / len(st) * 100 if st else 0
        ab = abbrev.get(sname, sname[:12])

        # Show individual trades for losing strategies
        detail = ""
        if spnl < -200 and len(st) <= 6:
            # Show each trade
            trade_details = []
            for td in st:
                t = td['trade']
                trade_details.append(f"{td['date'][-5:]} {td['direction'][0]} {td['r_multiple']:+.1f}R ${td['net_pnl']:+,.0f}")
            detail = " | " + " ; ".join(trade_details)

        marker2 = " <-- DRAG" if spnl < -200 else ""
        print(f"    {ab:12s}: {len(st):>2d}t {swr:>3.0f}%WR ${spnl:>7,.0f}{marker2}{detail}")

    # Key findings for losing months
    if month_pnl < 0:
        # Identify biggest losers
        worst = sorted(month_trades, key=lambda x: x['net_pnl'])[:3]
        print(f"  ** WORST TRADES:")
        for td in worst:
            t = td['trade']
            ab = abbrev.get(td['strategy'], td['strategy'][:12])
            print(f"     {td['date']} {ab} {td['direction']} entry={t.entry_price:.0f} "
                  f"stop={t.stop_price:.0f} risk={td['risk_pts']:.0f}pt "
                  f"exit={t.exit_price:.0f} {td['exit_reason']} "
                  f"R={td['r_multiple']:+.2f} ${td['net_pnl']:+,.0f}")

        # Why did we lose?
        losing_strats = defaultdict(float)
        for td in month_trades:
            if td['net_pnl'] < 0:
                losing_strats[td['strategy']] += td['net_pnl']
        worst_strat = min(losing_strats, key=losing_strats.get) if losing_strats else None
        if worst_strat:
            ab = abbrev.get(worst_strat, worst_strat[:12])
            print(f"  ** PRIMARY DRAG: {ab} (${losing_strats[worst_strat]:,.0f})")

# ════════════════════════════════════════════════════════════
# SECTION 6: STRATEGY WEAKNESSES
# ════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("  SECTION 6: STRATEGY WEAKNESS ANALYSIS")
print("=" * 100)

for sname in all_strats:
    st = [td for td in trade_data if td['strategy'] == sname]
    if len(st) < 3:
        continue

    ab = abbrev.get(sname, sname[:12])
    sw = [td for td in st if td['net_pnl'] > 0]
    sl = [td for td in st if td['net_pnl'] <= 0]
    spnl = sum(td['net_pnl'] for td in st)

    print(f"\n  --{ab} ({len(st)} trades, ${spnl:,.0f}) ──")

    # Win streaks and loss streaks
    sorted_trades = sorted(st, key=lambda x: (x['date'], str(x['trade'].entry_time)))
    max_win_streak = 0
    max_loss_streak = 0
    curr_streak = 0
    for td in sorted_trades:
        if td['net_pnl'] > 0:
            if curr_streak > 0:
                curr_streak += 1
            else:
                curr_streak = 1
            max_win_streak = max(max_win_streak, curr_streak)
        else:
            if curr_streak < 0:
                curr_streak -= 1
            else:
                curr_streak = -1
            max_loss_streak = max(max_loss_streak, abs(curr_streak))

    # By direction
    longs = [td for td in st if td['direction'] == 'LONG']
    shorts = [td for td in st if td['direction'] == 'SHORT']
    long_pnl = sum(td['net_pnl'] for td in longs)
    short_pnl = sum(td['net_pnl'] for td in shorts)
    long_wr = sum(1 for td in longs if td['net_pnl'] > 0) / len(longs) * 100 if longs else 0
    short_wr = sum(1 for td in shorts if td['net_pnl'] > 0) / len(shorts) * 100 if shorts else 0

    print(f"    Direction: LONG {len(longs)}t {long_wr:.0f}%WR ${long_pnl:,.0f} | SHORT {len(shorts)}t {short_wr:.0f}%WR ${short_pnl:,.0f}")
    print(f"    Streaks: max win={max_win_streak}, max loss={max_loss_streak}")

    # By regime
    regime_perf = defaultdict(lambda: {'trades': 0, 'pnl': 0, 'wins': 0})
    for td in st:
        ib = session_ib.get(td['date'], 150)
        r = regime_perf[regime_label(ib)]
        r['trades'] += 1
        r['pnl'] += td['net_pnl']
        if td['net_pnl'] > 0:
            r['wins'] += 1

    regime_str = " | ".join(
        f"{k}: {v['trades']}t {v['wins']/v['trades']*100:.0f}%WR ${v['pnl']:,.0f}"
        for k, v in sorted(regime_perf.items())
    )
    print(f"    By Regime: {regime_str}")

    # By exit reason
    exit_perf = defaultdict(lambda: {'trades': 0, 'pnl': 0})
    for td in st:
        exit_perf[td['exit_reason']]['trades'] += 1
        exit_perf[td['exit_reason']]['pnl'] += td['net_pnl']

    exit_str = " | ".join(
        f"{k}: {v['trades']}t ${v['pnl']:,.0f}"
        for k, v in sorted(exit_perf.items(), key=lambda x: x[1]['pnl'], reverse=True)
    )
    print(f"    By Exit: {exit_str}")

    # R:R analysis
    avg_r_w = np.mean([td['r_multiple'] for td in sw]) if sw else 0
    avg_r_l = np.mean([td['r_multiple'] for td in sl]) if sl else 0
    print(f"    R:R: avg winner={avg_r_w:+.2f}R, avg loser={avg_r_l:+.2f}R, ratio={abs(avg_r_w/avg_r_l) if avg_r_l else float('inf'):.2f}")

    # Weakness identification
    weaknesses = []
    if len(sw) / len(st) * 100 < 50:
        weaknesses.append(f"Low WR ({len(sw)/len(st)*100:.0f}%)")
    if spnl < 0:
        weaknesses.append("NEGATIVE P&L")
    if abs(avg_r_l) > avg_r_w and avg_r_w > 0:
        weaknesses.append(f"Inverted R:R (win {avg_r_w:.1f}R < loss {abs(avg_r_l):.1f}R)")
    if max_loss_streak >= 4:
        weaknesses.append(f"Long loss streaks ({max_loss_streak})")

    # Check if only works in certain regimes
    for regime, perf in regime_perf.items():
        if perf['trades'] >= 3 and perf['pnl'] < -500:
            weaknesses.append(f"Loses in {regime} vol (${perf['pnl']:,.0f})")

    if weaknesses:
        print(f"    ** WEAKNESSES: {' | '.join(weaknesses)}")
    else:
        print(f"    ** No major weaknesses identified")

# ════════════════════════════════════════════════════════════
# SECTION 7: DRAWDOWN ANALYSIS & POSITION SIZING
# ════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("  SECTION 7: DRAWDOWN ANALYSIS & POSITION SIZING")
print("=" * 100)

# Build equity curve
sorted_all = sorted(trade_data, key=lambda x: (x['date'], str(x['trade'].entry_time)))
equity = 0
peak = 0
drawdowns = []
dd_start_date = None
dd_start_eq = 0
in_dd = False

equity_points = []
for td in sorted_all:
    equity += td['net_pnl']
    equity_points.append({'date': td['date'], 'equity': equity, 'pnl': td['net_pnl']})

    if equity > peak:
        if in_dd:
            # DD ended
            dd_depth = peak - min(ep['equity'] for ep in equity_points if ep['date'] >= dd_start_date)
            drawdowns.append({
                'start': dd_start_date,
                'end': td['date'],
                'depth': dd_depth,
                'peak': dd_start_eq,
            })
        peak = equity
        dd_start_eq = equity
        in_dd = False
    else:
        if not in_dd:
            dd_start_date = td['date']
            in_dd = True

# If still in DD at end
if in_dd and dd_start_date:
    dd_depth = peak - min(ep['equity'] for ep in equity_points if ep['date'] >= dd_start_date)
    drawdowns.append({
        'start': dd_start_date,
        'end': sorted_all[-1]['date'],
        'depth': dd_depth,
        'peak': dd_start_eq,
    })

max_dd = max(drawdowns, key=lambda x: x['depth']) if drawdowns else None

print(f"\n  Final Equity: ${equity:,.0f}")
print(f"  Peak Equity: ${peak:,.0f}")
if max_dd:
    print(f"  Max Drawdown: ${max_dd['depth']:,.0f} ({max_dd['start']} to {max_dd['end']})")

# All drawdowns > $500
print(f"\n  Significant Drawdowns (> $500):")
for dd in sorted(drawdowns, key=lambda x: x['depth'], reverse=True):
    if dd['depth'] > 500:
        print(f"    ${dd['depth']:>6,.0f}  {dd['start']} to {dd['end']}")

# Daily P&L distribution
daily_pnl = defaultdict(float)
daily_trades_count = defaultdict(int)
for td in trade_data:
    daily_pnl[td['date']] += td['net_pnl']
    daily_trades_count[td['date']] += 1

daily_vals = list(daily_pnl.values())
losing_days = [v for v in daily_vals if v < 0]
winning_days = [v for v in daily_vals if v > 0]

print(f"\n  Trading Days: {len(daily_vals)}")
print(f"  Winning Days: {len(winning_days)} ({len(winning_days)/len(daily_vals)*100:.0f}%)")
print(f"  Losing Days: {len(losing_days)} ({len(losing_days)/len(daily_vals)*100:.0f}%)")
print(f"  Avg Winning Day: ${np.mean(winning_days):,.0f}" if winning_days else "")
print(f"  Avg Losing Day: ${np.mean(losing_days):,.0f}" if losing_days else "")
print(f"  Worst Day: ${min(daily_vals):,.0f}")
print(f"  Best Day: ${max(daily_vals):,.0f}")

# Monte Carlo drawdown simulation
print(f"\n  --Monte Carlo Drawdown Analysis (10,000 shuffles) ──")
np.random.seed(42)
trade_pnls = np.array([td['net_pnl'] for td in trade_data])
mc_drawdowns = []

for _ in range(10000):
    shuffled = np.random.permutation(trade_pnls)
    cum = np.cumsum(shuffled)
    peak_arr = np.maximum.accumulate(cum)
    dd = (peak_arr - cum).max()
    mc_drawdowns.append(dd)

mc_dd = np.array(mc_drawdowns)
print(f"  Monte Carlo Max Drawdown Distribution:")
for p in [50, 75, 90, 95, 99]:
    print(f"    {p}th percentile: ${np.percentile(mc_dd, p):,.0f}")

# --Position Sizing Recommendations ──
print(f"\n  --POSITION SIZING RECOMMENDATIONS ──")
print(f"\n  Based on Monte Carlo 95th percentile DD = ${np.percentile(mc_dd, 95):,.0f}")

account_sizes = [50000, 100000, 150000]
risk_pcts = [0.5, 1.0, 1.5, 2.0]
mc_95_dd = np.percentile(mc_dd, 95)

print(f"\n  {'Account':>10s}", end='')
for rp in risk_pcts:
    print(f" | {rp:.1f}% risk", end='')
print()
print("  " + "-" * 60)

for acct in account_sizes:
    print(f"  ${acct:>9,}", end='')
    for rp in risk_pcts:
        risk_per_trade = acct * rp / 100
        # Scale: how many MNQ contracts at avg risk
        contracts = risk_per_trade / avg_risk_dollars
        # Scaled DD
        scaled_dd = mc_95_dd * contracts
        dd_pct = scaled_dd / acct * 100
        print(f" | {contracts:.1f}ct DD${scaled_dd:>5,.0f}({dd_pct:.0f}%)", end='')
    print()

# Compounding analysis
print(f"\n  --COMPOUNDING vs FIXED SIZING ──")
print(f"\n  Scenario: Start with $150K, {len(trades)} trades over 259 sessions")
print(f"  Fixed 1 MNQ contract vs compounding (risk 1% of equity per trade)")

# Fixed sizing
fixed_equity = 0
fixed_curve = []
for td in sorted_all:
    fixed_equity += td['net_pnl']
    fixed_curve.append(fixed_equity)

# Compounding: scale contracts by equity
compound_equity = 150000
compound_curve = []
compound_start = 150000
for td in sorted_all:
    # How many contracts at 1% risk
    risk_budget = compound_equity * 0.01
    contracts = max(1, int(risk_budget / td['risk_dollars'])) if td['risk_dollars'] > 0 else 1
    scaled_pnl = td['net_pnl'] * contracts
    compound_equity += scaled_pnl
    compound_curve.append(compound_equity - compound_start)

# Also track DD
compound_peak = compound_start
compound_max_dd = 0
compound_eq = compound_start
for td in sorted_all:
    risk_budget = compound_eq * 0.01
    contracts = max(1, int(risk_budget / td['risk_dollars'])) if td['risk_dollars'] > 0 else 1
    compound_eq += td['net_pnl'] * contracts
    compound_peak = max(compound_peak, compound_eq)
    compound_max_dd = max(compound_max_dd, compound_peak - compound_eq)

print(f"\n  Fixed (1 MNQ):    Final P&L: ${fixed_curve[-1]:>10,.0f} | Max DD: ${max_dd['depth'] if max_dd else 0:>6,.0f}")
print(f"  Compound (1% risk): Final P&L: ${compound_curve[-1]:>10,.0f} | Max DD: ${compound_max_dd:>6,.0f}")
print(f"  Compound growth: {compound_curve[-1] / fixed_curve[-1]:.1f}x vs fixed" if fixed_curve[-1] > 0 else "")

# Prop firm sizing
print(f"\n  --PROP FIRM SIZING (Tradeify $150K Select Flex) ──")
print(f"  Rules: $9K profit target, $4.5K trailing DD (EOD)")
print(f"\n  At 1 MNQ contract:")
print(f"    Expected P&L over 259 sessions: ${net:,.0f}")
print(f"    Max drawdown seen: ${max_dd['depth'] if max_dd else 0:,.0f}")
print(f"    DD/Limit ratio: {(max_dd['depth'] if max_dd else 0)/4500*100:.0f}% of $4,500 limit")
print(f"    Passes target: {'YES' if net >= 9000 else 'NO'} (need $9K, have ${net:,.0f})")

# How many sessions to hit $9K target
cumul = 0
for i, td in enumerate(sorted_all):
    cumul += td['net_pnl']
    if cumul >= 9000:
        print(f"    Hits $9K target after: {i+1} trades (~session {td['date']})")
        break

# What if we size up to 2 contracts
print(f"\n  At 2 MNQ contracts:")
scaled_dd = max_dd['depth'] * 2 if max_dd else 0
print(f"    Expected P&L: ${net*2:,.0f}")
print(f"    Expected Max DD: ${scaled_dd:,.0f}")
print(f"    DD/Limit ratio: {scaled_dd/4500*100:.0f}% of $4,500 limit")
print(f"    SAFE? {'YES - room for error' if scaled_dd < 3500 else 'RISKY' if scaled_dd < 4500 else 'NO - exceeds limit'}")

print("\n" + "=" * 100)
print("  END OF DEEP DIVE REPORT")
print("=" * 100)
