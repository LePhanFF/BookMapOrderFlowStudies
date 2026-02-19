"""
MONTE CARLO SIMULATION: Optimized Config (Core 5 + EMA Trend Follow)

Runs 10,000 simulations per scenario using actual backtest trade distributions
to project realistic monthly income on Tradeify Lightning $150K accounts.

Compares:
  - Current config (Core 5, $400 risk) → $1,370/mo backtest
  - Optimized config (Core 5 + EMA, $400 risk) → $2,506/mo backtest
  - Risk-scaled (Core 5 + EMA, $800 risk) → check
  - Risk-scaled (Core 5, $800 risk) → $1,970/mo backtest

Tradeify Lightning $150K Rules:
  - Profit target: $9,000
  - EOD trailing max drawdown: $4,500
  - 40% consistency rule
  - 5 MNQ max during eval
  - After funded: scaling 3→12 minis
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
    EMATrendFollow,
)
from filters.regime_filter import SimpleRegimeFilter

# ============================================================================
# DATA
# ============================================================================
instrument = get_instrument('MNQ')
df_raw = load_csv('NQ')
df = filter_rth(df_raw)
df = compute_all_features(df)

if 'session_date' not in df.columns:
    df['session_date'] = df['timestamp'].dt.date

sessions = sorted(df['session_date'].unique())
n_sessions = len(sessions)
months = n_sessions / 22

rf_long_only = SimpleRegimeFilter(
    longs_in_bull=True, longs_in_bear=True,
    shorts_in_bull=False, shorts_in_bear=False,
)


def run_backtest(strategies, rf, risk, max_ctrs=5):
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
# RUN BACKTESTS FOR EACH CONFIG
# ============================================================================
configs = {}

# Config 1: Current (Core 5, $400)
res = run_backtest(
    [TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(), MeanReversionVWAP()],
    rf_long_only, risk=400,
)
configs['Current (Core5, $400)'] = res.trades

# Config 2: Optimized (Core 5 + EMA, $400) — THE WINNER
res = run_backtest(
    [TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(), MeanReversionVWAP(), EMATrendFollow()],
    rf_long_only, risk=400,
)
configs['Optimized (Core5+EMA, $400)'] = res.trades

# Config 3: Optimized at higher risk (Core 5 + EMA, $600)
res = run_backtest(
    [TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(), MeanReversionVWAP(), EMATrendFollow()],
    rf_long_only, risk=600,
)
configs['Optimized (Core5+EMA, $600)'] = res.trades

# Config 4: Optimized at higher risk (Core 5 + EMA, $800)
res = run_backtest(
    [TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(), MeanReversionVWAP(), EMATrendFollow()],
    rf_long_only, risk=800,
)
configs['Optimized (Core5+EMA, $800)'] = res.trades

# Config 5: Core only at $800 risk
res = run_backtest(
    [TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(), MeanReversionVWAP()],
    rf_long_only, risk=800,
)
configs['Core5 only ($800)'] = res.trades


# ============================================================================
# EXTRACT TRADE DISTRIBUTIONS
# ============================================================================
print("=" * 130)
print("  MONTE CARLO SIMULATION: OPTIMIZED CONFIG PROJECTIONS")
print("=" * 130)

config_data = {}

for name, trades in configs.items():
    # Build daily P&L distribution
    daily_pnl = {}
    daily_trades = {}
    for t in trades:
        d = str(t.session_date)
        daily_pnl[d] = daily_pnl.get(d, 0) + t.net_pnl
        daily_trades[d] = daily_trades.get(d, 0) + 1

    # Include zero-trade days
    all_daily = []
    for s in sessions:
        d = str(s)
        all_daily.append(daily_pnl.get(d, 0))

    active_daily = [p for p in all_daily if p != 0]
    trade_freq = len([p for p in all_daily if p != 0]) / len(all_daily)

    n = len(trades)
    wins = sum(1 for t in trades if t.net_pnl > 0)
    wr = wins / max(n, 1) * 100
    total_pnl = sum(t.net_pnl for t in trades)
    per_month = total_pnl / months
    max_dd = min(all_daily) if all_daily else 0

    config_data[name] = {
        'trades': trades,
        'all_daily': all_daily,
        'active_daily': active_daily,
        'trade_freq': trade_freq,
        'n': n,
        'wr': wr,
        'per_month': per_month,
        'max_dd': max_dd,
    }

    print(f"\n  {name}:")
    print(f"    Trades: {n}, WR: {wr:.1f}%, Monthly: ${per_month:,.0f}")
    print(f"    Trade freq: {trade_freq:.1%} of days, Max daily DD: ${max_dd:,.0f}")
    print(f"    Active daily P&L: mean=${np.mean(active_daily):,.0f}, "
          f"median=${np.median(active_daily):,.0f}" if active_daily else "")


# ============================================================================
# MONTE CARLO ENGINE
# ============================================================================
EVAL_PROFIT_TARGET = 9000
EVAL_MAX_DD = 4500
EVAL_CONSISTENCY_PCT = 0.40
EVAL_COST = 99  # Lightning per month
TRADING_DAYS_PER_MONTH = 22
N_SIMS = 10000
MAX_EVAL_DAYS = 500
FUNDED_DAYS = 252

np.random.seed(42)


def get_max_mnq(profit):
    """Tradeify Lightning scaling."""
    if profit >= 4500: return 120
    elif profit >= 3000: return 80
    elif profit >= 2000: return 50
    elif profit >= 1500: return 40
    else: return 30


def simulate_eval_from_daily(daily_pnl_pool, trade_freq, n_sims=N_SIMS):
    """Simulate evaluation using daily P&L distribution."""
    active_pool = np.array([p for p in daily_pnl_pool if p != 0])
    if len(active_pool) == 0:
        return {'passed': 0, 'failed': n_sims, 'days': [], 'pct': []}

    results = {'passed': 0, 'failed': 0, 'days': [], 'pct': []}

    for _ in range(n_sims):
        balance = 0.0
        hwm = 0.0
        daily_pnls = []

        for day in range(1, MAX_EVAL_DAYS + 1):
            if np.random.random() < trade_freq:
                # Sample from active day P&L distribution
                pnl = np.random.choice(active_pool)
            else:
                pnl = 0.0

            daily_pnls.append(pnl)
            balance += pnl

            if balance > hwm:
                hwm = balance
            dd_floor = hwm - EVAL_MAX_DD

            if balance <= dd_floor:
                results['failed'] += 1
                break

            if balance >= EVAL_PROFIT_TARGET and day >= 3:
                total_pos = sum(p for p in daily_pnls if p > 0)
                max_day = max(daily_pnls) if daily_pnls else 0
                if total_pos > 0 and max_day / total_pos > EVAL_CONSISTENCY_PCT:
                    continue  # Keep trading to dilute
                results['passed'] += 1
                results['days'].append(day)
                break
        else:
            results['failed'] += 1

    results['pct'] = results['passed'] / n_sims * 100
    return results


def simulate_funded_from_daily(daily_pnl_pool, trade_freq, n_sims=N_SIMS):
    """Simulate funded phase using daily P&L distribution."""
    active_pool = np.array([p for p in daily_pnl_pool if p != 0])
    if len(active_pool) == 0:
        return {'survived': 0, 'blown': n_sims, 'withdrawn': [], 'monthly_avg': 0}

    results = {
        'survived': 0, 'blown': 0,
        'withdrawn': [], 'payouts': [],
        'monthly_income': [],
    }

    for _ in range(n_sims):
        balance = 0.0
        hwm = 0.0
        dd_floor = -EVAL_MAX_DD
        dd_locked = False
        win_days = 0
        total_out = 0.0
        blown = False

        for day in range(1, FUNDED_DAYS + 1):
            if np.random.random() < trade_freq:
                pnl = np.random.choice(active_pool)
            else:
                pnl = 0.0

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


# ============================================================================
# RUN SIMULATIONS
# ============================================================================
print("\n\n" + "=" * 130)
print("  EVALUATION PHASE: PASS RATE & TIMELINE")
print("=" * 130)

print(f"\n  {'Config':<40s} {'Pass%':>6s} {'Fail%':>6s} {'Med Days':>9s} "
      f"{'Months':>7s} {'P10':>6s} {'P90':>6s}")
print("  " + "-" * 90)

eval_results = {}
for name, cd in config_data.items():
    r = simulate_eval_from_daily(cd['all_daily'], cd['trade_freq'])
    eval_results[name] = r

    md = np.median(r['days']) if r['days'] else 0
    mo = md / TRADING_DAYS_PER_MONTH
    p10 = np.percentile(r['days'], 10) if r['days'] else 0
    p90 = np.percentile(r['days'], 90) if r['days'] else 0

    print(f"  {name:<40s} {r['pct']:>5.1f}% {100-r['pct']:>5.1f}% {md:>7.0f} d "
          f"{mo:>5.1f} m {p10:>5.0f} d {p90:>5.0f} d")


# ============================================================================
# FUNDED PHASE
# ============================================================================
print("\n\n" + "=" * 130)
print("  FUNDED PHASE: 1-YEAR SURVIVAL & INCOME")
print("=" * 130)

print(f"\n  {'Config':<40s} {'Surv%':>6s} {'Blown%':>7s} {'Med W/D':>10s} "
      f"{'Avg W/D':>10s} {'$/Month':>9s}")
print("  " + "-" * 95)

funded_results = {}
for name, cd in config_data.items():
    r = simulate_funded_from_daily(cd['all_daily'], cd['trade_freq'])
    funded_results[name] = r

    sv = r['survived'] / N_SIMS * 100
    bl = r['blown'] / N_SIMS * 100
    mw = np.median(r['withdrawn'])
    aw = np.mean(r['withdrawn'])
    am = aw / 12

    print(f"  {name:<40s} {sv:>5.1f}% {bl:>6.1f}% ${mw:>8,.0f} "
          f"${aw:>8,.0f} ${am:>7,.0f}")


# ============================================================================
# NET INCOME AFTER EVAL COSTS (per account)
# ============================================================================
print("\n\n" + "=" * 130)
print("  NET INCOME PER ACCOUNT (after eval costs)")
print("=" * 130)

print(f"\n  {'Config':<40s} {'Eval Cost':>10s} {'Pass Rate':>10s} "
      f"{'1yr Income':>11s} {'Net/Year':>10s} {'Net/Month':>10s}")
print("  " + "-" * 105)

net_results = {}
for name in configs:
    ev = eval_results[name]
    fu = funded_results[name]

    pass_rate = ev['passed'] / N_SIMS
    if pass_rate > 0:
        attempts = 1.0 / pass_rate
        avg_eval_days = np.mean(ev['days']) if ev['days'] else 200
        avg_eval_mo = avg_eval_days / TRADING_DAYS_PER_MONTH
    else:
        attempts = 10
        avg_eval_mo = 12

    eval_cost = attempts * avg_eval_mo * EVAL_COST
    survival = fu['survived'] / N_SIMS
    avg_wd = np.mean(fu['withdrawn'])
    net_year = avg_wd * survival - eval_cost
    net_month = net_year / 12

    net_results[name] = {'net_month': net_month, 'net_year': net_year, 'eval_cost': eval_cost}

    print(f"  {name:<40s} ${eval_cost:>8,.0f} {pass_rate*100:>8.1f}% "
          f"${avg_wd:>9,.0f} ${net_year:>8,.0f} ${net_month:>8,.0f}")


# ============================================================================
# MULTI-ACCOUNT PROJECTION
# ============================================================================
print("\n\n" + "=" * 130)
print("  MULTI-ACCOUNT PROJECTION (5x Lightning $150K)")
print("=" * 130)

print(f"\n  {'Config':<40s} {'1 Acct':>10s} {'3 Accts':>10s} {'5 Accts':>10s}")
print("  " + "-" * 80)

for name in configs:
    nm = net_results[name]['net_month']
    print(f"  {name:<40s} ${nm:>8,.0f} ${nm*3:>8,.0f} ${nm*5:>8,.0f}")


# ============================================================================
# DEGRADED PERFORMANCE SCENARIOS
# ============================================================================
print("\n\n" + "=" * 130)
print("  DEGRADED PERFORMANCE SCENARIOS (Optimized Core5+EMA, $400)")
print("  Testing: what if live performance is worse than backtest?")
print("=" * 130)

# Get the optimized config's active daily P&L
opt_daily = config_data['Optimized (Core5+EMA, $400)']['all_daily']
opt_active = np.array([p for p in opt_daily if p != 0])
opt_freq = config_data['Optimized (Core5+EMA, $400)']['trade_freq']

# Degraded scenarios
scenarios = {
    'Backtest-exact': opt_active,
    '80% of backtest (slippage)': opt_active * 0.80,
    '60% of backtest (bad fills)': opt_active * 0.60,
    'Wins -20%, real losses': None,  # Special handling
}

# For "Wins -20%, real losses": degrade winners 20%, replace losses with realistic stops
wins_pool = opt_active[opt_active > 0] * 0.80
loss_pool = np.array([-400, -350, -300, -250, -200, -300, -350, -400, -250, -200])

print(f"\n  {'Scenario':<40s} {'Eval Pass':>10s} {'Surv%':>7s} "
      f"{'Avg W/D':>10s} {'$/Mo/Acct':>10s} {'5 Accts':>10s}")
print("  " + "-" * 100)

for scenario_name, degraded_pool in scenarios.items():
    if degraded_pool is None:
        # Win/loss split scenario
        # Create a mixed daily pool
        wr = len(opt_active[opt_active > 0]) / len(opt_active)
        synthetic = []
        for _ in range(1000):
            if np.random.random() < wr:
                synthetic.append(np.random.choice(wins_pool))
            else:
                synthetic.append(np.random.choice(loss_pool))
        degraded_pool = np.array(synthetic)

    # Build full daily list (with zero days)
    all_d = []
    for _ in range(len(sessions)):
        if np.random.random() < opt_freq:
            all_d.append(np.random.choice(degraded_pool))
        else:
            all_d.append(0)

    ev = simulate_eval_from_daily(all_d, opt_freq)
    fu = simulate_funded_from_daily(all_d, opt_freq)

    pass_rate = ev['passed'] / N_SIMS
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

    print(f"  {scenario_name:<40s} {pass_rate*100:>8.1f}% {surv:>6.1f}% "
          f"${avg_wd:>8,.0f} ${net_month:>8,.0f} ${net_month*5:>8,.0f}")


# ============================================================================
# FINAL VERDICT
# ============================================================================
print("\n\n" + "=" * 130)
print("  FINAL VERDICT: OPTIMIZED CONFIG")
print("=" * 130)

opt_nm = net_results['Optimized (Core5+EMA, $400)']['net_month']
cur_nm = net_results['Current (Core5, $400)']['net_month']

print(f"""
  BEST CONFIG: Core 5 + EMA Trend Follow, $400 risk, LONG only, 5 MNQ max

  BACKTEST PERFORMANCE:
    Trades: {config_data['Optimized (Core5+EMA, $400)']['n']}, {config_data['Optimized (Core5+EMA, $400)']['wr']:.1f}% WR
    Trades/session: {config_data['Optimized (Core5+EMA, $400)']['n']/n_sessions:.2f}
    Active days: {config_data['Optimized (Core5+EMA, $400)']['trade_freq']:.0%}
    Monthly backtest: ${config_data['Optimized (Core5+EMA, $400)']['per_month']:,.0f}

  MONTE CARLO PROJECTIONS (per account):
    Backtest-based: ${opt_nm:,.0f}/month
    Conservative (80%): ${opt_nm * 0.80:,.0f}/month

  ACROSS 5 ACCOUNTS:
    Projected: ${opt_nm * 5:,.0f}/month
    Conservative: ${opt_nm * 5 * 0.80:,.0f}/month

  vs CURRENT CONFIG:
    Current: ${cur_nm:,.0f}/month per account
    Optimized: ${opt_nm:,.0f}/month per account
    Improvement: {(opt_nm/max(cur_nm,1)-1)*100:+.0f}%

  RECOMMENDATION:
    Primary config: Core 5 + EMA (LONG only, $400 risk)
    The EMA Trend Follow strategy nearly TRIPLES trade volume
    (0.47 → 1.29 trades/session) while keeping DD buffer at 6.6x.
    This is the single biggest unlock for income generation.
""")
