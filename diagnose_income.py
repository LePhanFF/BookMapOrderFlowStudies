"""
INCOME DIAGNOSTIC: Why is $760/month per account so low at 5 MNQ?

Investigation:
  1. Trade volume — are we too selective?
  2. Profit per trade — are targets too conservative?
  3. Monte Carlo inputs — are they stale (pre-IBH sweep)?
  4. Compare to benchmarks — what does $2k/month per account require?
  5. Test relaxed filters to increase volume while keeping >70% WR
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.instruments import get_instrument
from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from config.constants import IB_BARS_1MIN
from engine.execution import ExecutionModel
from engine.backtest import BacktestEngine
from engine.position import PositionManager
from strategy import (
    TrendDayBull, SuperTrendBull, PDayStrategy, BDayStrategy, MeanReversionVWAP,
)
from filters.regime_filter import SimpleRegimeFilter

instrument = get_instrument('MNQ')
df_raw = load_csv('NQ')
df = filter_rth(df_raw)
df = compute_all_features(df)

if 'session_date' not in df.columns:
    df['session_date'] = df['timestamp'].dt.date

sessions = sorted(df['session_date'].unique())
n_sessions = len(sessions)
months = n_sessions / 22  # approx trading days per month

print("=" * 120)
print("  INCOME DIAGNOSTIC: WHY $760/MONTH PER ACCOUNT?")
print(f"  Data: {n_sessions} sessions ({months:.1f} months)")
print("=" * 120)


# ============================================================================
# 1. CURRENT PLAYBOOK — RAW NUMBERS
# ============================================================================
print("\n\n" + "=" * 120)
print("  1. CURRENT PLAYBOOK — PER-TRADE ECONOMICS")
print("=" * 120)

regime_filter = SimpleRegimeFilter(
    longs_in_bull=True, longs_in_bear=True,
    shorts_in_bull=False, shorts_in_bear=False,
)

exec_m = ExecutionModel(instrument, slippage_ticks=1)
pos_m = PositionManager(account_size=150000)

strategies = [TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(), MeanReversionVWAP()]

engine = BacktestEngine(
    instrument=instrument,
    strategies=strategies,
    filters=regime_filter,
    execution=exec_m,
    position_mgr=pos_m,
    risk_per_trade=400,
    max_contracts=5,
)

result = engine.run(df, verbose=False)
trades = result.trades

print(f"\nTotal trades: {len(trades)}")
print(f"Sessions: {n_sessions}")
print(f"Trades/session: {len(trades)/n_sessions:.2f}")
print(f"Trades/month (22 days): {len(trades)/n_sessions * 22:.1f}")

active_dates = set(str(t.session_date) for t in trades)
print(f"Active days: {len(active_dates)} / {n_sessions} ({len(active_dates)/n_sessions*100:.0f}%)")
print(f"Trades/active day: {len(trades)/max(len(active_dates),1):.2f}")

wins = [t for t in trades if t.net_pnl > 0]
losses = [t for t in trades if t.net_pnl <= 0]
total_pnl = sum(t.net_pnl for t in trades)

print(f"\nWin rate: {len(wins)}/{len(trades)} = {len(wins)/max(len(trades),1)*100:.1f}%")
print(f"Total P&L: ${total_pnl:,.0f}")
print(f"Per month: ${total_pnl/months:,.0f}/month")
print(f"Per trade: ${total_pnl/max(len(trades),1):,.0f}/trade")

avg_win = np.mean([t.net_pnl for t in wins]) if wins else 0
avg_loss = np.mean([t.net_pnl for t in losses]) if losses else 0
max_win = max(t.net_pnl for t in trades)
max_loss = min(t.net_pnl for t in trades)

print(f"\nAvg winner: ${avg_win:,.0f}")
print(f"Avg loser: ${avg_loss:,.0f}")
print(f"Max single win: ${max_win:,.0f}")
print(f"Max single loss: ${max_loss:,.0f}")
print(f"Win/Loss ratio: {abs(avg_win/avg_loss) if avg_loss != 0 else 0:.2f}")


# ============================================================================
# 2. PER-STRATEGY BREAKDOWN
# ============================================================================
print("\n\n" + "=" * 120)
print("  2. PER-STRATEGY BREAKDOWN")
print("=" * 120)

by_strat = {}
for t in trades:
    name = t.setup_type
    if name not in by_strat:
        by_strat[name] = []
    by_strat[name].append(t)

print(f"\n  {'Strategy':<30s} {'Trades':>6s} {'WR':>6s} {'Total PnL':>10s} {'Avg PnL':>9s} "
      f"{'AvgWin':>8s} {'AvgLoss':>8s} {'MaxWin':>8s} {'MaxLoss':>8s}")
print("  " + "-" * 100)

for name, strades in sorted(by_strat.items()):
    n = len(strades)
    w = [t for t in strades if t.net_pnl > 0]
    l = [t for t in strades if t.net_pnl <= 0]
    wr = len(w) / n * 100
    pnl = sum(t.net_pnl for t in strades)
    avg = pnl / n
    aw = np.mean([t.net_pnl for t in w]) if w else 0
    al = np.mean([t.net_pnl for t in l]) if l else 0
    mw = max(t.net_pnl for t in strades)
    ml = min(t.net_pnl for t in strades)
    print(f"  {name:<30s} {n:>6d} {wr:>5.1f}% ${pnl:>9,.0f} ${avg:>8,.0f} "
          f"${aw:>7,.0f} ${al:>7,.0f} ${mw:>7,.0f} ${ml:>7,.0f}")


# ============================================================================
# 3. TRADE-BY-TRADE LOG WITH CONTRACT SIZE AND POINTS
# ============================================================================
print("\n\n" + "=" * 120)
print("  3. FULL TRADE LOG — CHECK CONTRACTS AND POINT CAPTURE")
print("=" * 120)

print(f"\n  {'Date':<14s} {'Strategy':<25s} {'Dir':<6s} {'Entry':>10s} {'Exit':>10s} "
      f"{'Pts':>8s} {'Ctrs':>4s} {'Gross':>9s} {'Net':>9s} {'Reason':<8s}")
print("  " + "-" * 110)

for t in sorted(trades, key=lambda x: str(x.session_date)):
    pts = abs(t.exit_price - t.entry_price)
    if t.direction == 'LONG':
        pts = t.exit_price - t.entry_price
    else:
        pts = t.entry_price - t.exit_price
    print(f"  {str(t.session_date):<14s} {t.setup_type:<25s} {t.direction:<6s} "
          f"{t.entry_price:>10.2f} {t.exit_price:>10.2f} "
          f"{pts:>+7.1f} {t.contracts:>4d} ${t.gross_pnl:>8,.0f} ${t.net_pnl:>8,.0f} "
          f"{t.exit_reason:<8s}")


# ============================================================================
# 4. BENCHMARK: WHAT DOES $2K/MONTH REQUIRE?
# ============================================================================
print("\n\n" + "=" * 120)
print("  4. BENCHMARK: WHAT DOES $2,000/MONTH PER ACCOUNT REQUIRE AT 5 MNQ?")
print("=" * 120)

print("""
  TARGET: $2,000/month = $91/day = $24,000/year

  AT 5 MNQ ($10/point per trade = 5 contracts x $2/pt):
    $91/day ÷ $10/pt = 9.1 points/day net capture needed

  SCENARIO A: 1 trade/day, 70% WR
    Need: avg win to offset 30% losses
    If avg loss = 20 pts ($200): need avg win = (91 + 0.3*200) / 0.7 = $216 = 21.6 pts
    That's ~21.6 pt target, 20 pt stop → 1.08 R:R at 70% WR

  SCENARIO B: 2 trades/day, 70% WR
    Need: $45.50/trade avg
    If avg loss = 15 pts ($150): need avg win = (45.5 + 0.3*150) / 0.7 = $129 = 12.9 pts
    That's ~12.9 pt target, 15 pt stop → 0.86 R:R at 70% WR

  SCENARIO C: 3 trades/day, 65% WR
    Need: $30.33/trade avg
    If avg loss = 15 pts ($150): need avg win = (30.33 + 0.35*150) / 0.65 = $128 = 12.8 pts
    That's ~12.8 pt target, 15 pt stop → 0.85 R:R at 65% WR

  OUR CURRENT REALITY:
    Trades/day: {trades_per_day:.2f} (need 1-3)
    Active days: {active_pct:.0f}% (need 70-100%)
    Avg win: ${avg_win_val:,.0f} ({avg_win_pts:.1f} pts)
    Avg loss: ${avg_loss_val:,.0f} ({avg_loss_pts:.1f} pts)
""".format(
    trades_per_day=len(trades) / n_sessions,
    active_pct=len(active_dates) / n_sessions * 100,
    avg_win_val=avg_win,
    avg_win_pts=avg_win / 10 if avg_win > 0 else 0,  # $10/pt at 5 MNQ
    avg_loss_val=avg_loss,
    avg_loss_pts=abs(avg_loss) / 10 if avg_loss < 0 else 0,
))


# ============================================================================
# 5. WHERE ARE THE MISSING TRADES? WHAT SIGNALS DO WE SKIP?
# ============================================================================
print("\n\n" + "=" * 120)
print("  5. SIGNAL ANALYSIS: WHERE ARE WE LEAVING TRADES ON THE TABLE?")
print("=" * 120)

# Run each strategy WITHOUT regime filter to see raw signal count
for strat_class, name in [
    (TrendDayBull, "TrendDayBull"),
    (SuperTrendBull, "SuperTrendBull"),
    (PDayStrategy, "P-Day"),
    (BDayStrategy, "B-Day"),
    (MeanReversionVWAP, "MeanRevVWAP"),
]:
    # No filter
    eng_nofilter = BacktestEngine(
        instrument=instrument,
        strategies=[strat_class()],
        filters=None,
        execution=exec_m,
        position_mgr=PositionManager(account_size=150000),
        risk_per_trade=400,
        max_contracts=5,
    )
    res_nf = eng_nofilter.run(df, verbose=False)

    # With regime filter (LONG only)
    eng_filtered = BacktestEngine(
        instrument=instrument,
        strategies=[strat_class()],
        filters=regime_filter,
        execution=exec_m,
        position_mgr=PositionManager(account_size=150000),
        risk_per_trade=400,
        max_contracts=5,
    )
    res_f = eng_filtered.run(df, verbose=False)

    n_raw = len(res_nf.trades)
    n_filt = len(res_f.trades)
    w_raw = sum(1 for t in res_nf.trades if t.net_pnl > 0)
    w_filt = sum(1 for t in res_f.trades if t.net_pnl > 0)
    pnl_raw = sum(t.net_pnl for t in res_nf.trades)
    pnl_filt = sum(t.net_pnl for t in res_f.trades)
    wr_raw = w_raw / max(n_raw, 1) * 100
    wr_filt = w_filt / max(n_filt, 1) * 100

    blocked = n_raw - n_filt
    print(f"\n  {name:<20s}:")
    print(f"    Raw (no filter):    {n_raw:>3d} trades, {wr_raw:>5.1f}% WR, ${pnl_raw:>8,.0f}")
    print(f"    LONG only filter:   {n_filt:>3d} trades, {wr_filt:>5.1f}% WR, ${pnl_filt:>8,.0f}")
    print(f"    Blocked by filter:  {blocked:>3d} trades")

    if blocked > 0:
        blocked_trades = [t for t in res_nf.trades if t.direction == 'SHORT']
        if blocked_trades:
            bw = sum(1 for t in blocked_trades if t.net_pnl > 0)
            bp = sum(t.net_pnl for t in blocked_trades)
            print(f"    Blocked SHORT trades: {len(blocked_trades)}, {bw}/{len(blocked_trades)} wins, ${bp:,.0f}")


# ============================================================================
# 6. TEST EXPANDED CONFIGS — MORE TRADES WHILE KEEPING WR > 65%
# ============================================================================
print("\n\n" + "=" * 120)
print("  6. EXPANDED CONFIGS — CAN WE ADD MORE TRADES?")
print("=" * 120)

# Config A: Current optimal (baseline)
configs = []

# Baseline
eng_base = BacktestEngine(
    instrument=instrument,
    strategies=[TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(), MeanReversionVWAP()],
    filters=regime_filter,
    execution=exec_m,
    position_mgr=PositionManager(account_size=150000),
    risk_per_trade=400,
    max_contracts=5,
)
res_base = eng_base.run(df, verbose=False)
configs.append(('A: Current (LONG only)', res_base))

# Config B: All strategies, NO regime filter (allow shorts too)
eng_all = BacktestEngine(
    instrument=instrument,
    strategies=[TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(), MeanReversionVWAP()],
    filters=None,
    execution=exec_m,
    position_mgr=PositionManager(account_size=150000),
    risk_per_trade=400,
    max_contracts=5,
)
res_all = eng_all.run(df, verbose=False)
configs.append(('B: All signals (no filter)', res_all))

# Config C: LONG only but lower quality threshold
# We'll try with all available strategies
from strategy import TrendDayBear, SuperTrendBear, LiquiditySweep, EMATrendFollow, ORBVwapBreakout, NeutralDayStrategy

# Config C: Add ORB and EMA strategies
try:
    eng_expanded = BacktestEngine(
        instrument=instrument,
        strategies=[TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(),
                    MeanReversionVWAP(), LiquiditySweep(), EMATrendFollow()],
        filters=regime_filter,
        execution=exec_m,
        position_mgr=PositionManager(account_size=150000),
        risk_per_trade=400,
        max_contracts=5,
    )
    res_expanded = eng_expanded.run(df, verbose=False)
    configs.append(('C: + LiqSweep + EMA (LONG only)', res_expanded))
except Exception as e:
    print(f"  Config C error: {e}")

# Config D: Add ORB and EMA without filter
try:
    eng_expanded_nf = BacktestEngine(
        instrument=instrument,
        strategies=[TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(),
                    MeanReversionVWAP(), LiquiditySweep(), EMATrendFollow()],
        filters=None,
        execution=exec_m,
        position_mgr=PositionManager(account_size=150000),
        risk_per_trade=400,
        max_contracts=5,
    )
    res_expanded_nf = eng_expanded_nf.run(df, verbose=False)
    configs.append(('D: + LiqSweep + EMA (no filter)', res_expanded_nf))
except Exception as e:
    print(f"  Config D error: {e}")

# Config E: Add ORB too
try:
    eng_orb = BacktestEngine(
        instrument=instrument,
        strategies=[TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(),
                    MeanReversionVWAP(), LiquiditySweep(), EMATrendFollow(), ORBVwapBreakout()],
        filters=regime_filter,
        execution=exec_m,
        position_mgr=PositionManager(account_size=150000),
        risk_per_trade=400,
        max_contracts=5,
    )
    res_orb = eng_orb.run(df, verbose=False)
    configs.append(('E: + ORB (LONG only)', res_orb))
except Exception as e:
    print(f"  Config E error: {e}")

# Config F: All 9 strategies including bear + neutral, no filter
try:
    eng_full = BacktestEngine(
        instrument=instrument,
        strategies=[TrendDayBull(), SuperTrendBull(), TrendDayBear(), SuperTrendBear(),
                    PDayStrategy(), BDayStrategy(), NeutralDayStrategy(), MeanReversionVWAP(),
                    LiquiditySweep(), EMATrendFollow()],
        filters=None,
        execution=exec_m,
        position_mgr=PositionManager(account_size=150000),
        risk_per_trade=400,
        max_contracts=5,
    )
    res_full = eng_full.run(df, verbose=False)
    configs.append(('F: ALL 9 strategies (no filter)', res_full))
except Exception as e:
    print(f"  Config E error: {e}")

print(f"\n  {'Config':<45s} {'Trades':>6s} {'Tr/Sess':>8s} {'WR':>6s} {'Total PnL':>10s} "
      f"{'$/Trade':>8s} {'$/Month':>9s} {'MaxDD':>8s}")
print("  " + "-" * 105)

for label, res in configs:
    t = res.trades
    n = len(t)
    w = sum(1 for tr in t if tr.net_pnl > 0)
    pnl = sum(tr.net_pnl for tr in t)
    wr = w / max(n, 1) * 100
    per_trade = pnl / max(n, 1)
    per_month = pnl / months

    # Max daily drawdown
    daily_pnl = {}
    for tr in t:
        d = str(tr.session_date)
        daily_pnl[d] = daily_pnl.get(d, 0) + tr.net_pnl
    max_dd = min(daily_pnl.values()) if daily_pnl else 0

    print(f"  {label:<45s} {n:>6d} {n/n_sessions:>7.2f} {wr:>5.1f}% ${pnl:>9,.0f} "
          f"${per_trade:>7,.0f} ${per_month:>8,.0f} ${max_dd:>7,.0f}")


# ============================================================================
# 7. TEST HIGHER CONTRACT SIZES
# ============================================================================
print("\n\n" + "=" * 120)
print("  7. CONTRACT SIZE SCALING — WHAT IF WE USE MORE CONTRACTS?")
print("=" * 120)

for max_ctrs in [5, 7, 10, 15, 20]:
    eng_scale = BacktestEngine(
        instrument=instrument,
        strategies=[TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(), MeanReversionVWAP()],
        filters=regime_filter,
        execution=exec_m,
        position_mgr=PositionManager(account_size=150000),
        risk_per_trade=400,
        max_contracts=max_ctrs,
    )
    res_scale = eng_scale.run(df, verbose=False)
    t = res_scale.trades
    pnl = sum(tr.net_pnl for tr in t)
    per_month = pnl / months

    daily_pnl = {}
    for tr in t:
        d = str(tr.session_date)
        daily_pnl[d] = daily_pnl.get(d, 0) + tr.net_pnl
    max_dd = min(daily_pnl.values()) if daily_pnl else 0
    dd_ratio = 4500 / abs(max_dd) if max_dd < 0 else 999

    print(f"  {max_ctrs:>2d} MNQ: ${pnl:>8,.0f} total, ${per_month:>7,.0f}/month, "
          f"max daily loss ${max_dd:>7,.0f}, DD buffer {dd_ratio:.1f}x")


# ============================================================================
# 8. TEST INCREASED RISK PER TRADE
# ============================================================================
print("\n\n" + "=" * 120)
print("  8. RISK PER TRADE SCALING — WHAT IF WE RISK MORE PER TRADE?")
print("=" * 120)

for risk in [400, 600, 800, 1000, 1500]:
    eng_risk = BacktestEngine(
        instrument=instrument,
        strategies=[TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(), MeanReversionVWAP()],
        filters=regime_filter,
        execution=exec_m,
        position_mgr=PositionManager(account_size=150000),
        risk_per_trade=risk,
        max_contracts=5,
    )
    res_risk = eng_risk.run(df, verbose=False)
    t = res_risk.trades
    pnl = sum(tr.net_pnl for tr in t)
    per_month = pnl / months
    avg_ctrs = np.mean([tr.contracts for tr in t]) if t else 0

    daily_pnl = {}
    for tr in t:
        d = str(tr.session_date)
        daily_pnl[d] = daily_pnl.get(d, 0) + tr.net_pnl
    max_dd = min(daily_pnl.values()) if daily_pnl else 0
    dd_ratio = 4500 / abs(max_dd) if max_dd < 0 else 999

    print(f"  Risk ${risk:>5,}: ${pnl:>8,.0f} total, ${per_month:>7,.0f}/month, "
          f"avg {avg_ctrs:.1f} ctrs, max daily loss ${max_dd:>7,.0f}, DD buffer {dd_ratio:.1f}x")


# ============================================================================
# 9. DAILY P&L DISTRIBUTION
# ============================================================================
print("\n\n" + "=" * 120)
print("  9. DAILY P&L DISTRIBUTION (current config)")
print("=" * 120)

daily_pnl_list = {}
for t in trades:
    d = str(t.session_date)
    daily_pnl_list[d] = daily_pnl_list.get(d, 0) + t.net_pnl

# Include zero days
all_daily = []
for s in sessions:
    d = str(s)
    all_daily.append(daily_pnl_list.get(d, 0))

positive_days = [p for p in all_daily if p > 0]
negative_days = [p for p in all_daily if p < 0]
zero_days = [p for p in all_daily if p == 0]

print(f"\n  Total days: {len(all_daily)}")
print(f"  Win days: {len(positive_days)} ({len(positive_days)/len(all_daily)*100:.0f}%)")
print(f"  Loss days: {len(negative_days)} ({len(negative_days)/len(all_daily)*100:.0f}%)")
print(f"  No-trade days: {len(zero_days)} ({len(zero_days)/len(all_daily)*100:.0f}%)")

if positive_days:
    print(f"\n  Win days avg: ${np.mean(positive_days):,.0f}")
    print(f"  Win days max: ${max(positive_days):,.0f}")
if negative_days:
    print(f"  Loss days avg: ${np.mean(negative_days):,.0f}")
    print(f"  Loss days max: ${min(negative_days):,.0f}")

print(f"\n  Overall daily avg (incl zero): ${np.mean(all_daily):,.0f}")
print(f"  Overall daily avg (active only): ${np.mean([p for p in all_daily if p != 0]):,.0f}")
print(f"  Monthly projection (active-only avg x 22): ${np.mean([p for p in all_daily if p != 0]) * 22:,.0f}")
print(f"  Monthly projection (overall avg x 22): ${np.mean(all_daily) * 22:,.0f}")


# ============================================================================
# 10. THE REAL PROBLEM — SUMMARY
# ============================================================================
print("\n\n" + "=" * 120)
print("  10. DIAGNOSIS: WHERE IS THE BOTTLENECK?")
print("=" * 120)

trades_per_month = len(trades) / months
monthly_pnl = total_pnl / months

print(f"""
  CURRENT STATE:
    Trades per month:     {trades_per_month:.1f}
    Active days/month:    {len(active_dates)/months:.1f} out of 22
    Monthly P&L (1 acct): ${monthly_pnl:,.0f}
    Monthly P&L (5 acct): ${monthly_pnl * 5:,.0f} (copy trade, minus costs)

  THE BOTTLENECK:
    1. TRADE VOLUME: {len(trades)/n_sessions:.2f} trades/session is TOO LOW
       - Target: 1-2 trades per session
       - Current: {len(trades)/n_sessions:.2f} (70% of days produce ZERO trades)

    2. SELECTIVITY: Our quality filters are extremely strict
       - Trend Bull: needs acceptance + pullback + delta + quality gate + confidence
       - P-Day: needs acceptance + pullback + delta + quality gate
       - B-Day: needs rejection + delta + quality >= 2 + R:R check
       - MeanRev: needs deviation + RSI + volume + delta + reversal candle

    3. WHAT TO DO ABOUT IT:
       - Option A: Relax quality gates (lower from 2/3 to 1/3)
       - Option B: Widen entry windows (allow after 1:00 PM)
       - Option C: Add more strategies (ORB, Liquidity Sweep, EMA)
       - Option D: Increase contract size (5 → 10 MNQ)
       - Option E: Increase risk per trade ($400 → $800)
       - Option F: Allow selective shorts (IBH sweep already added)
""")
