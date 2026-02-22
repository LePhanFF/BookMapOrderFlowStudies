"""
80% Rule (80P) Strategy Backtest
==================================

Tests the Dalton 80% Rule:
  - Price opens outside prior day's Value Area
  - Re-enters and is accepted (2x 30-min closes inside VA)
  - Targets the opposite VA boundary

User's target scenario:
  - 80% WR, 1:4 R:R (70pt risk : 270pt target)
  - ~410 pt IBR days
  - 5 occurrences/month (4W, 1L)
  - Need to validate if our data supports this frequency

Analysis includes:
  1. How often does price open outside prior VA?
  2. How often does it re-enter and get accepted?
  3. What's the actual WR and R:R?
  4. What's the actual trade frequency?
  5. Monte Carlo projections
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.instruments import get_instrument
from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from engine.execution import ExecutionModel
from engine.backtest import BacktestEngine
from engine.position import PositionManager
from strategy import (
    TrendDayBull, SuperTrendBull, PDayStrategy, BDayStrategy, MeanReversionVWAP,
)
from strategy.eighty_percent_rule import EightyPercentRule
from filters.regime_filter import SimpleRegimeFilter

# ============================================================================
# LOAD DATA
# ============================================================================
print("Loading data...")
instrument = get_instrument('MNQ')
df_raw = load_csv('NQ')
df = filter_rth(df_raw)
df = compute_all_features(df)

if 'session_date' not in df.columns:
    df['session_date'] = df['timestamp'].dt.date

sessions = sorted(df['session_date'].unique())
n_sessions = len(sessions)
months = n_sessions / 22

# ============================================================================
# VALUE AREA DISTRIBUTION ANALYSIS
# ============================================================================
print(f"\n{'='*90}")
print(f"  VALUE AREA DISTRIBUTION ANALYSIS ({n_sessions} sessions)")
print(f"{'='*90}")

if 'open_vs_va' in df.columns:
    session_first = df.groupby('session_date').first()

    # Open location distribution
    open_dist = session_first['open_vs_va'].value_counts(dropna=False)
    print(f"\n  Open Location Relative to Prior Day VA:")
    for loc, count in open_dist.items():
        label = loc if loc is not None and not pd.isna(loc) else 'N/A (no prior VA)'
        print(f"    {label:20s}: {count} sessions ({count/n_sessions*100:.0f}%)")

    # VA width stats
    va_widths = session_first['prior_va_width'].dropna()
    if len(va_widths) > 0:
        print(f"\n  Prior Day VA Width Statistics:")
        print(f"    Mean:   {va_widths.mean():.1f} pts")
        print(f"    Median: {va_widths.median():.1f} pts")
        print(f"    Min:    {va_widths.min():.1f} pts")
        print(f"    Max:    {va_widths.max():.1f} pts")
        print(f"    Std:    {va_widths.std():.1f} pts")

        # How many have VA width >= 25 (our minimum)
        viable = (va_widths >= 25).sum()
        print(f"    VA >= 25 pts: {viable}/{len(va_widths)} ({viable/len(va_widths)*100:.0f}%)")
        viable50 = (va_widths >= 50).sum()
        print(f"    VA >= 50 pts: {viable50}/{len(va_widths)} ({viable50/len(va_widths)*100:.0f}%)")

    # IB range stats for context
    ib_ranges = session_first['ib_range'].dropna()
    if len(ib_ranges) > 0:
        print(f"\n  IB Range Statistics (for context):")
        print(f"    Mean:   {ib_ranges.mean():.1f} pts")
        print(f"    Median: {ib_ranges.median():.1f} pts")
        print(f"    Min:    {ib_ranges.min():.1f} pts")
        print(f"    Max:    {ib_ranges.max():.1f} pts")

    # How many sessions have BOTH: open outside VA AND VA width >= 25?
    outside_va = session_first[
        (session_first['open_vs_va'].isin(['ABOVE_VAH', 'BELOW_VAL'])) &
        (session_first['prior_va_width'] >= 25)
    ]
    print(f"\n  80P Setup Frequency:")
    print(f"    Sessions with open outside VA + VA >= 25pt: {len(outside_va)}/{n_sessions}")
    print(f"    Per month: {len(outside_va)/months:.1f} potential setups")

    # Of those, how many have wide VA (>50) for the user's 270pt target scenario?
    wide_va = outside_va[outside_va['prior_va_width'] >= 200]
    print(f"    VA >= 200pt (for ~270pt target): {len(wide_va)}/{n_sessions} ({len(wide_va)/months:.1f}/mo)")

    # Per-direction
    below_val = session_first[session_first['open_vs_va'] == 'BELOW_VAL']
    above_vah = session_first[session_first['open_vs_va'] == 'ABOVE_VAH']
    print(f"\n    Open below VAL (LONG setup): {len(below_val)} sessions ({len(below_val)/months:.1f}/mo)")
    print(f"    Open above VAH (SHORT setup): {len(above_vah)} sessions ({len(above_vah)/months:.1f}/mo)")

# ============================================================================
# RUN 80P BACKTEST
# ============================================================================
print(f"\n{'='*90}")
print(f"  RUNNING 80P BACKTESTS")
print(f"{'='*90}")

rf_long_only = SimpleRegimeFilter(
    longs_in_bull=True, longs_in_bear=True,
    shorts_in_bull=False, shorts_in_bear=False,
)

rf_allow_shorts = SimpleRegimeFilter(
    longs_in_bull=True, longs_in_bear=True,
    shorts_in_bull=False, shorts_in_bear=True,
)


def run_backtest(strategies, rf, risk=800, max_ctrs=5):
    exec_m = ExecutionModel(instrument, slippage_ticks=1)
    pos_m = PositionManager(account_size=150000)
    engine = BacktestEngine(
        instrument=instrument,
        strategies=strategies,
        filters=rf,
        execution=exec_m,
        position_mgr=pos_m,
        risk_per_trade=risk,
        max_contracts=max_ctrs,
    )
    return engine.run(df, verbose=False)


configs = {}

# 80P standalone (LONG only)
res = run_backtest([EightyPercentRule()], rf_long_only)
configs['80P LONG only'] = res.trades

# 80P standalone (LONG + SHORT)
res = run_backtest([EightyPercentRule()], rf_allow_shorts)
configs['80P LONG+SHORT'] = res.trades

# Core 5 baseline
res = run_backtest(
    [TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(), MeanReversionVWAP()],
    rf_long_only,
)
configs['Core 5 ($800)'] = res.trades

# Core 5 + 80P (LONG only)
res = run_backtest(
    [TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(),
     MeanReversionVWAP(), EightyPercentRule()],
    rf_long_only,
)
configs['Core5 + 80P (LONG)'] = res.trades

# Core 5 + 80P (LONG + SHORT)
res = run_backtest(
    [TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(),
     MeanReversionVWAP(), EightyPercentRule()],
    rf_allow_shorts,
)
configs['Core5 + 80P (+SHORT)'] = res.trades

# ============================================================================
# RESULTS TABLE
# ============================================================================
print(f"\n{'='*120}")
print(f"  BACKTEST RESULTS COMPARISON")
print(f"{'='*120}")

print(f"\n  {'Config':<30s} {'Trades':>7s} {'WR':>6s} {'PF':>6s} "
      f"{'$/Month':>9s} {'AvgWin':>8s} {'AvgLoss':>8s} {'R:R':>5s} {'MaxDayDD':>9s}")
print(f"  {'-'*100}")

for name, trades in configs.items():
    n = len(trades)
    if n == 0:
        print(f"  {name:<30s} {0:>7d}   No trades")
        continue

    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl <= 0]
    wr = len(wins) / n * 100
    total_pnl = sum(t.net_pnl for t in trades)
    per_month = total_pnl / months

    avg_win = np.mean([t.net_pnl for t in wins]) if wins else 0
    avg_loss = np.mean([t.net_pnl for t in losses]) if losses else 0
    gw = sum(t.net_pnl for t in wins)
    gl = abs(sum(t.net_pnl for t in losses))
    pf = gw / gl if gl > 0 else float('inf')
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    daily_pnl = {}
    for t in trades:
        d = str(t.session_date)
        daily_pnl[d] = daily_pnl.get(d, 0) + t.net_pnl
    max_dd = min(daily_pnl.values()) if daily_pnl else 0

    print(f"  {name:<30s} {n:>7d} {wr:>5.1f}% {pf:>5.2f} "
          f"${per_month:>7,.0f} ${avg_win:>6,.0f} ${avg_loss:>6,.0f} {rr:>4.2f} ${max_dd:>7,.0f}")


# ============================================================================
# 80P DETAILED TRADE ANALYSIS
# ============================================================================
eighty_p_trades = configs.get('80P LONG+SHORT', configs.get('80P LONG only', []))

if eighty_p_trades:
    print(f"\n{'='*90}")
    print(f"  80P RULE — DETAILED TRADE ANALYSIS ({len(eighty_p_trades)} trades)")
    print(f"{'='*90}")

    # Trade frequency
    print(f"\n  Trade Frequency:")
    print(f"    Total trades: {len(eighty_p_trades)}")
    print(f"    Trades/month: {len(eighty_p_trades)/months:.1f}")
    print(f"    Data months: {months:.1f}")

    # By direction
    longs = [t for t in eighty_p_trades if t.direction == 'LONG']
    shorts = [t for t in eighty_p_trades if t.direction == 'SHORT']
    print(f"\n    LONG trades: {len(longs)}")
    print(f"    SHORT trades: {len(shorts)}")

    # Win rate
    wins = [t for t in eighty_p_trades if t.net_pnl > 0]
    losses = [t for t in eighty_p_trades if t.net_pnl <= 0]
    print(f"\n  Win Rate: {len(wins)}/{len(eighty_p_trades)} = {len(wins)/len(eighty_p_trades)*100:.1f}%")

    # R:R analysis
    if wins:
        print(f"\n  Winner Analysis:")
        print(f"    Avg win: ${np.mean([t.net_pnl for t in wins]):,.0f}")
        print(f"    Avg win pts: {np.mean([t.exit_price - t.entry_price if t.direction == 'LONG' else t.entry_price - t.exit_price for t in wins]):.1f}")
        print(f"    Avg risk pts: {np.mean([t.risk_points for t in wins]):.1f}")
        print(f"    Avg reward pts: {np.mean([t.reward_points for t in wins]):.1f}")

    if losses:
        print(f"\n  Loser Analysis:")
        print(f"    Avg loss: ${np.mean([t.net_pnl for t in losses]):,.0f}")
        print(f"    Avg loss pts: {np.mean([abs(t.exit_price - t.entry_price) for t in losses]):.1f}")

    # VA width of 80P trades
    va_widths = [t.metadata.get('va_width', 0) for t in eighty_p_trades if t.metadata]
    if va_widths:
        print(f"\n  VA Width of Triggered Trades:")
        print(f"    Mean: {np.mean(va_widths):.1f} pts")
        print(f"    Median: {np.median(va_widths):.1f} pts")
        print(f"    Min: {np.min(va_widths):.1f} pts")
        print(f"    Max: {np.max(va_widths):.1f} pts")

    # Individual trade log
    print(f"\n  Individual Trades:")
    print(f"  {'Date':12s} {'Setup':12s} {'Dir':6s} {'Ctrs':>5s} {'Entry':>10s} "
          f"{'Exit':>10s} {'P&L':>8s} {'ExitRsn':10s} {'VA_W':>6s} {'Risk':>6s} {'Rwrd':>6s}")
    print(f"  {'-'*105}")

    for t in sorted(eighty_p_trades, key=lambda x: str(x.session_date)):
        va_w = t.metadata.get('va_width', 0) if t.metadata else 0
        print(f"  {str(t.session_date):12s} {t.setup_type:12s} {t.direction:6s} "
              f"{t.contracts:>5d} {t.entry_price:>10.2f} {t.exit_price:>10.2f} "
              f"${t.net_pnl:>7,.0f} {t.exit_reason:10s} {va_w:>5.0f}p "
              f"{t.risk_points:>5.0f}p {t.reward_points:>5.0f}p")

else:
    print(f"\n  80P Rule: 0 trades generated.")
    print(f"  This may mean:")
    print(f"    - No sessions opened outside prior VA with acceptance")
    print(f"    - VA width filter eliminated all candidates")
    print(f"    - Need to check VA computation and open classification")


# ============================================================================
# USER'S SCENARIO VALIDATION
# ============================================================================
print(f"\n{'='*90}")
print(f"  USER SCENARIO VALIDATION")
print(f"  Target: 80% WR, 1:4 R:R (70pt risk : 270pt target), 5 trades/month")
print(f"{'='*90}")

print(f"""
  YOUR SCENARIO MATH:
    Win rate: 80%  →  4 wins / 1 loss per 5 trades
    Risk: 70 pts × $2/pt × contracts
    Target: 270 pts × $2/pt × contracts
    At 5 MNQ contracts:
      Per win:  270 × $2 × 5 = $2,700
      Per loss: 70 × $2 × 5  = $700
      Monthly:  (4 × $2,700) - (1 × $700) = $10,100/month per account
      5 accounts: $50,500/month

  THIS REQUIRES:
    1. VA width ~340+ pts (target of 270 = ~80% of VA width)
    2. IB range ~410 pts (your specified condition)
    3. Setup frequency: 5/month (every ~4.4 trading days)
""")

# Check how many sessions meet the user's specific criteria
if 'prior_va_width' in df.columns:
    session_first = df.groupby('session_date').first()

    # Sessions with VA width >= 200 and open outside VA
    big_va = session_first[
        (session_first['open_vs_va'].isin(['ABOVE_VAH', 'BELOW_VAL'])) &
        (session_first['prior_va_width'] >= 200)
    ]
    print(f"  DATA CHECK:")
    print(f"    Sessions with open outside VA + VA >= 200pt: {len(big_va)}/{n_sessions}")
    print(f"    Per month: {len(big_va)/months:.1f}")

    # Sessions with IB range >= 300
    big_ib = session_first[session_first['ib_range'] >= 300]
    print(f"    Sessions with IB range >= 300pt: {len(big_ib)}/{n_sessions} ({len(big_ib)/months:.1f}/mo)")

    big_ib_400 = session_first[session_first['ib_range'] >= 400]
    print(f"    Sessions with IB range >= 400pt: {len(big_ib_400)}/{n_sessions} ({len(big_ib_400)/months:.1f}/mo)")


# ============================================================================
# MONTE CARLO — 80P
# ============================================================================
if eighty_p_trades:
    EVAL_MAX_DD = 4500
    N_SIMS = 10000
    FUNDED_DAYS = 252
    np.random.seed(42)

    daily_pnl_map = {}
    for t in eighty_p_trades:
        d = str(t.session_date)
        daily_pnl_map[d] = daily_pnl_map.get(d, 0) + t.net_pnl

    all_daily = [daily_pnl_map.get(str(s), 0) for s in sessions]
    active_daily = np.array([p for p in all_daily if p != 0])
    trade_freq = len([p for p in all_daily if p != 0]) / len(all_daily)

    print(f"\n{'='*90}")
    print(f"  MONTE CARLO: 80P Rule ({N_SIMS} sims)")
    print(f"{'='*90}")

    print(f"\n  Active days: {len(active_daily)}/{len(all_daily)} ({trade_freq:.1%})")
    if len(active_daily) > 0:
        print(f"  Active day P&L: mean=${np.mean(active_daily):,.0f}, "
              f"median=${np.median(active_daily):,.0f}")

    # Funded survival
    survived = 0
    withdrawn_list = []
    for _ in range(N_SIMS):
        balance = 0.0
        hwm = 0.0
        dd_locked = False
        win_days = 0
        total_out = 0.0
        blown = False
        for day in range(1, FUNDED_DAYS + 1):
            pnl = np.random.choice(active_daily) if np.random.random() < trade_freq else 0.0
            balance += pnl
            if balance > hwm:
                hwm = balance
            if not dd_locked:
                dd_floor = hwm - EVAL_MAX_DD
                if balance >= 4600:
                    dd_locked = True
                    dd_floor = 100
            else:
                dd_floor = 100
            if balance <= dd_floor:
                blown = True
                break
            if pnl >= 250:
                win_days += 1
            if win_days >= 5 and balance > 0:
                pay = min(balance * 0.50 * 0.90, 5000)
                if pay > 100:
                    total_out += pay
                    balance -= pay / 0.90
                    win_days = 0
        if not blown:
            survived += 1
        withdrawn_list.append(total_out)

    surv_pct = survived / N_SIMS * 100
    avg_wd = np.mean(withdrawn_list)
    mc_monthly = avg_wd / 12

    print(f"\n  Funded survival: {surv_pct:.1f}%")
    print(f"  Avg annual withdrawal: ${avg_wd:,.0f}")
    print(f"  MC monthly income: ${mc_monthly:,.0f}")
    print(f"  5 accounts: ${mc_monthly * 5:,.0f}/month")


# ============================================================================
# COMBINED: Core 5 + 80P
# ============================================================================
combo_key = 'Core5 + 80P (LONG)' if 'Core5 + 80P (LONG)' in configs else 'Core5 + 80P (+SHORT)'
combo_trades = configs.get(combo_key, [])

if combo_trades:
    print(f"\n{'='*90}")
    print(f"  COMBINED: {combo_key}")
    print(f"{'='*90}")

    p80_in_combo = [t for t in combo_trades if '80P' in t.setup_type]
    core_in_combo = [t for t in combo_trades if '80P' not in t.setup_type]

    for label, subset in [("Core 5", core_in_combo), ("80P Rule", p80_in_combo)]:
        n = len(subset)
        if n == 0:
            print(f"  {label}: 0 trades")
            continue
        wins = sum(1 for t in subset if t.net_pnl > 0)
        wr = wins / n * 100
        total = sum(t.net_pnl for t in subset)
        print(f"  {label}: {n} trades, {wr:.1f}% WR, ${total:,.0f} ({total/months:,.0f}/mo)")


print(f"\n{'='*90}")
print(f"  DONE — 80P Rule Analysis Complete")
print(f"{'='*90}")
