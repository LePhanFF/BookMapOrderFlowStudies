"""
Enhanced ORB Strategy Backtest + Monte Carlo Simulation
========================================================

Tests the redesigned ORB Enhanced strategy with:
  1. IB width-adaptive targets (narrow/normal/wide)
  2. C-period close location rule
  3. FVG confluence entries
  4. Multi-bar sweep detection
  5. SMT divergence confirmation

Compares against:
  - Core 5 only (current production config)
  - Core 5 + ORB Enhanced
  - ORB Enhanced standalone
  - Original ORB VWAP Breakout (baseline)

Runs 10,000-sim Monte Carlo on each config to project funded survival
and monthly income on Tradeify Lightning $150K accounts.
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
    ORBVwapBreakout, ORBEnhanced,
)
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
# IB WIDTH DISTRIBUTION ANALYSIS
# ============================================================================
print(f"\n{'='*90}")
print(f"  IB WIDTH DISTRIBUTION ANALYSIS")
print(f"{'='*90}")

if 'ib_width_class' in df.columns:
    # Get one row per session
    session_first = df.groupby('session_date').first()
    width_counts = session_first['ib_width_class'].value_counts()
    print(f"\n  IB Width Classification across {n_sessions} sessions:")
    for wc, count in width_counts.items():
        print(f"    {wc:8s}: {count} sessions ({count/n_sessions*100:.0f}%)")

    # IB/ATR ratio stats
    ratios = session_first['ib_atr_ratio'].dropna()
    print(f"\n  IB/ATR Ratio Statistics:")
    print(f"    Mean:   {ratios.mean():.2f}")
    print(f"    Median: {ratios.median():.2f}")
    print(f"    Min:    {ratios.min():.2f}")
    print(f"    Max:    {ratios.max():.2f}")

    # C-period bias distribution
    if 'c_period_bias' in session_first.columns:
        c_bias = session_first['c_period_bias'].value_counts(dropna=False)
        print(f"\n  C-Period Bias Distribution:")
        for bias, count in c_bias.items():
            label = bias if bias is not None and not pd.isna(bias) else 'INSIDE (no bias)'
            print(f"    {label:20s}: {count} sessions ({count/n_sessions*100:.0f}%)")

    # NR4/NR7 detection
    nr4 = session_first['is_nr4'].sum() if 'is_nr4' in session_first.columns else 0
    nr7 = session_first['is_nr7'].sum() if 'is_nr7' in session_first.columns else 0
    print(f"\n  Narrow Range Days:")
    print(f"    NR4 (4-day narrow): {nr4} sessions")
    print(f"    NR7 (7-day narrow): {nr7} sessions")

# ============================================================================
# REGIME FILTER
# ============================================================================
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


# ============================================================================
# RUN BACKTESTS
# ============================================================================
print(f"\n{'='*90}")
print(f"  RUNNING BACKTESTS")
print(f"{'='*90}")

configs = {}

# Config 1: Core 5 only (production baseline)
res = run_backtest(
    [TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(), MeanReversionVWAP()],
    rf_long_only,
)
configs['Core 5 ($800, LONG only)'] = res.trades

# Config 2: ORB Enhanced standalone (LONG only)
res = run_backtest([ORBEnhanced()], rf_long_only)
configs['ORB Enhanced (LONG only)'] = res.trades

# Config 3: ORB Enhanced with short sweeps allowed
res = run_backtest([ORBEnhanced()], rf_allow_shorts)
configs['ORB Enhanced (+sweepSHORT)'] = res.trades

# Config 4: Core 5 + ORB Enhanced (LONG only)
res = run_backtest(
    [TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(),
     MeanReversionVWAP(), ORBEnhanced()],
    rf_long_only,
)
configs['Core5 + ORBv2 (LONG)'] = res.trades

# Config 5: Core 5 + ORB Enhanced (with short sweeps)
res = run_backtest(
    [TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(),
     MeanReversionVWAP(), ORBEnhanced()],
    rf_allow_shorts,
)
configs['Core5 + ORBv2 (+SHORT)'] = res.trades

# Config 6: Original ORB baseline
res = run_backtest([ORBVwapBreakout()], rf_long_only)
configs['Original ORB (baseline)'] = res.trades


# ============================================================================
# RESULTS SUMMARY
# ============================================================================
print(f"\n{'='*120}")
print(f"  BACKTEST RESULTS COMPARISON")
print(f"{'='*120}")

print(f"\n  {'Config':<35s} {'Trades':>7s} {'WR':>6s} {'PF':>6s} "
      f"{'$/Month':>9s} {'AvgWin':>8s} {'AvgLoss':>8s} {'R:R':>5s} "
      f"{'MaxDD':>8s} {'Setup Types':>30s}")
print(f"  {'-'*115}")

for name, trades in configs.items():
    n = len(trades)
    if n == 0:
        print(f"  {name:<35s} {0:>7d} {'N/A':>6s}")
        continue

    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl <= 0]
    wr = len(wins) / n * 100
    total_pnl = sum(t.net_pnl for t in trades)
    per_month = total_pnl / months

    avg_win = np.mean([t.net_pnl for t in wins]) if wins else 0
    avg_loss = np.mean([t.net_pnl for t in losses]) if losses else 0
    gross_win = sum(t.net_pnl for t in wins)
    gross_loss = abs(sum(t.net_pnl for t in losses))
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    # Max daily drawdown
    daily_pnl = {}
    for t in trades:
        d = str(t.session_date)
        daily_pnl[d] = daily_pnl.get(d, 0) + t.net_pnl
    max_dd = min(daily_pnl.values()) if daily_pnl else 0

    # Setup types
    setups = set(t.setup_type for t in trades)
    setup_str = ', '.join(sorted(setups))[:30]

    print(f"  {name:<35s} {n:>7d} {wr:>5.1f}% {pf:>5.2f} "
          f"${per_month:>7,.0f} ${avg_win:>6,.0f} ${avg_loss:>6,.0f} {rr:>4.2f} "
          f"${max_dd:>6,.0f} {setup_str:>30s}")


# ============================================================================
# DETAILED TRADE ANALYSIS: ORB Enhanced
# ============================================================================
orb_trades = configs.get('ORB Enhanced (+sweepSHORT)', configs.get('ORB Enhanced (LONG only)', []))

if orb_trades:
    print(f"\n{'='*90}")
    print(f"  ORB ENHANCED — DETAILED ANALYSIS")
    print(f"{'='*90}")

    # By entry type
    print(f"\n  By Entry Type:")
    print(f"  {'Setup':30s} {'N':>4s} {'WR':>6s} {'PF':>6s} {'$/Mo':>8s} {'AvgWin':>8s} {'AvgLoss':>8s}")
    print(f"  {'-'*75}")

    for setup in sorted(set(t.setup_type for t in orb_trades)):
        strades = [t for t in orb_trades if t.setup_type == setup]
        n = len(strades)
        wins = [t for t in strades if t.net_pnl > 0]
        losses = [t for t in strades if t.net_pnl <= 0]
        wr = len(wins) / n * 100 if n > 0 else 0
        total = sum(t.net_pnl for t in strades)
        gw = sum(t.net_pnl for t in wins)
        gl = abs(sum(t.net_pnl for t in losses))
        pf = gw / gl if gl > 0 else float('inf')
        avg_w = np.mean([t.net_pnl for t in wins]) if wins else 0
        avg_l = np.mean([t.net_pnl for t in losses]) if losses else 0

        print(f"  {setup:30s} {n:>4d} {wr:>5.1f}% {pf:>5.2f} ${total/months:>6,.0f} "
              f"${avg_w:>6,.0f} ${avg_l:>6,.0f}")

    # By IB width class
    print(f"\n  By IB Width Class:")
    for width in ['NARROW', 'NORMAL', 'WIDE']:
        wtrades = [t for t in orb_trades if t.metadata.get('ib_width') == width]
        if not wtrades:
            continue
        n = len(wtrades)
        wins = sum(1 for t in wtrades if t.net_pnl > 0)
        wr = wins / n * 100
        total = sum(t.net_pnl for t in wtrades)
        print(f"    {width:8s}: {n} trades, {wr:.1f}% WR, ${total:,.0f} total (${total/months:,.0f}/mo)")

    # Individual trade log
    print(f"\n  Individual Trades:")
    print(f"  {'Date':12s} {'Setup':30s} {'Dir':6s} {'Ctrs':>5s} "
          f"{'Entry':>10s} {'Exit':>10s} {'P&L':>8s} {'ExitRsn':12s} {'IB Width':>8s}")
    print(f"  {'-'*105}")

    for t in sorted(orb_trades, key=lambda x: str(x.session_date)):
        ib_w = t.metadata.get('ib_width', '?') if t.metadata else '?'
        print(f"  {str(t.session_date):12s} {t.setup_type:30s} {t.direction:6s} "
              f"{t.contracts:>5d} {t.entry_price:>10.2f} {t.exit_price:>10.2f} "
              f"${t.net_pnl:>7,.0f} {t.exit_reason:12s} {ib_w:>8s}")


# ============================================================================
# MONTE CARLO SIMULATION
# ============================================================================
EVAL_PROFIT_TARGET = 9000
EVAL_MAX_DD = 4500
EVAL_CONSISTENCY_PCT = 0.40
TRADING_DAYS_PER_MONTH = 22
N_SIMS = 10000
MAX_EVAL_DAYS = 500
FUNDED_DAYS = 252

np.random.seed(42)


def build_daily_pnl(trades):
    daily = {}
    for t in trades:
        d = str(t.session_date)
        daily[d] = daily.get(d, 0) + t.net_pnl
    all_daily = [daily.get(str(s), 0) for s in sessions]
    return all_daily


def simulate_eval(daily_pnl_pool, trade_freq, n_sims=N_SIMS):
    active_pool = np.array([p for p in daily_pnl_pool if p != 0])
    if len(active_pool) == 0:
        return {'pct': 0, 'days': []}

    results = {'passed': 0, 'failed': 0, 'days': []}

    for _ in range(n_sims):
        balance = 0.0
        hwm = 0.0
        daily_pnls = []

        for day in range(1, MAX_EVAL_DAYS + 1):
            pnl = np.random.choice(active_pool) if np.random.random() < trade_freq else 0.0
            daily_pnls.append(pnl)
            balance += pnl
            if balance > hwm:
                hwm = balance
            if balance <= hwm - EVAL_MAX_DD:
                results['failed'] += 1
                break
            if balance >= EVAL_PROFIT_TARGET and day >= 3:
                total_pos = sum(p for p in daily_pnls if p > 0)
                max_day = max(daily_pnls) if daily_pnls else 0
                if total_pos > 0 and max_day / total_pos > EVAL_CONSISTENCY_PCT:
                    continue
                results['passed'] += 1
                results['days'].append(day)
                break
        else:
            results['failed'] += 1

    results['pct'] = results['passed'] / n_sims * 100
    return results


def simulate_funded(daily_pnl_pool, trade_freq, n_sims=N_SIMS):
    active_pool = np.array([p for p in daily_pnl_pool if p != 0])
    if len(active_pool) == 0:
        return {'survived': 0, 'blown': n_sims, 'withdrawn': [0]*n_sims, 'monthly_income': [0]*n_sims}

    results = {'survived': 0, 'blown': 0, 'withdrawn': [], 'monthly_income': []}

    for _ in range(n_sims):
        balance = 0.0
        hwm = 0.0
        dd_locked = False
        win_days = 0
        total_out = 0.0
        blown = False

        for day in range(1, FUNDED_DAYS + 1):
            pnl = np.random.choice(active_pool) if np.random.random() < trade_freq else 0.0
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

        if blown:
            results['blown'] += 1
        else:
            results['survived'] += 1
        results['withdrawn'].append(total_out)
        results['monthly_income'].append(total_out / 12)

    return results


print(f"\n\n{'='*120}")
print(f"  MONTE CARLO SIMULATION: 10,000 sims × {len(configs)} configs")
print(f"{'='*120}")

print(f"\n  {'Config':<35s} {'Eval%':>6s} {'MedDays':>8s} {'Surv%':>6s} "
      f"{'AvgW/D':>10s} {'MC$/Mo':>9s} {'5 Accts':>10s}")
print(f"  {'-'*100}")

EVAL_COST = 99

for name, trades in configs.items():
    daily = build_daily_pnl(trades)
    trade_freq = len([p for p in daily if p != 0]) / len(daily)

    ev = simulate_eval(daily, trade_freq)
    fu = simulate_funded(daily, trade_freq)

    pass_rate = ev['pct'] / 100
    med_days = np.median(ev['days']) if ev['days'] else 0
    surv = fu['survived'] / N_SIMS * 100
    avg_wd = np.mean(fu['withdrawn'])

    if pass_rate > 0:
        attempts = 1.0 / pass_rate
        avg_eval_days = np.mean(ev['days']) if ev['days'] else 200
        eval_cost = attempts * (avg_eval_days / TRADING_DAYS_PER_MONTH) * EVAL_COST
    else:
        eval_cost = 10 * 12 * EVAL_COST

    net_year = avg_wd * (surv / 100) - eval_cost
    net_month = net_year / 12

    print(f"  {name:<35s} {ev['pct']:>5.1f}% {med_days:>6.0f} d {surv:>5.1f}% "
          f"${avg_wd:>8,.0f} ${net_month:>7,.0f} ${net_month*5:>8,.0f}")


# ============================================================================
# ENHANCED ORB SPECIFIC — Sweep vs Breakout Performance
# ============================================================================
combo_trades = configs.get('Core5 + ORBv2 (+SHORT)', configs.get('Core5 + ORBv2 (LONG)', []))

if combo_trades:
    orb_in_combo = [t for t in combo_trades if 'ORB' in t.strategy_name]
    other = [t for t in combo_trades if 'ORB' not in t.strategy_name]

    print(f"\n{'='*90}")
    print(f"  COMBINED PORTFOLIO BREAKDOWN")
    print(f"{'='*90}")

    for label, subset in [("Core 5 Strategies", other), ("ORB Enhanced", orb_in_combo)]:
        n = len(subset)
        if n == 0:
            print(f"\n  {label}: 0 trades")
            continue
        wins = sum(1 for t in subset if t.net_pnl > 0)
        wr = wins / n * 100
        total = sum(t.net_pnl for t in subset)
        print(f"\n  {label}: {n} trades, {wr:.1f}% WR, ${total:,.0f} total (${total/months:,.0f}/mo)")


# ============================================================================
# VERDICT
# ============================================================================
print(f"\n\n{'='*90}")
print(f"  VERDICT")
print(f"{'='*90}")

# Find best config by Monte Carlo monthly income
best_name = None
best_mc = -999999
for name, trades in configs.items():
    daily = build_daily_pnl(trades)
    trade_freq = len([p for p in daily if p != 0]) / len(daily)
    ev = simulate_eval(daily, trade_freq)
    fu = simulate_funded(daily, trade_freq)
    pass_rate = ev['pct'] / 100
    surv = fu['survived'] / N_SIMS
    avg_wd = np.mean(fu['withdrawn'])
    if pass_rate > 0:
        eval_cost = (1/pass_rate) * (np.mean(ev['days']) if ev['days'] else 200) / 22 * 99
    else:
        eval_cost = 10 * 12 * 99
    net_month = (avg_wd * surv - eval_cost) / 12

    if net_month > best_mc and fu['survived'] / N_SIMS >= 0.50:
        best_mc = net_month
        best_name = name

n_trades = len(configs.get(best_name, []))
if best_name:
    print(f"""
  BEST CONFIG: {best_name}
    Trades: {n_trades}
    Monte Carlo $/Month: ${best_mc:,.0f}
    5 Accounts: ${best_mc * 5:,.0f}/month

  KEY IMPROVEMENTS IN ORB Enhanced v2:
    1. IB Width Classification → adaptive targets (narrow=2x, normal=1.5x, wide=1x)
    2. C-Period Close Location → 70-75% directional edge
    3. FVG Confluence Entries → tighter stops, better R:R
    4. Multi-Bar Sweep Detection → catch institutional stop hunts
    5. Skip breakouts on WIDE IB days → reduce false breakout losses
    6. SMT Divergence Confirmation → filter fake breakouts
""")
else:
    print("\n  No config achieved >50% funded survival.\n")

print("Done.")
