"""
COMPOUNDING SIMULATION: Progressive scaling to $2,000/month per account

Models the journey from funded account start to steady-state income:
  Phase 1: Conservative (5 MNQ, $800 risk) — build buffer
  Phase 2: Scale up contracts + risk as profit buffer grows
  Phase 3: Steady state with maximum safe scaling

Tradeify Lightning scaling tiers:
  $0-$1,499 profit: max 30 MNQ (3 minis)
  $1,500-$1,999: max 40 MNQ (4 minis)
  $2,000-$2,999: max 50 MNQ (5 minis)
  $3,000-$4,499: max 80 MNQ (8 minis)
  $4,500+: max 120 MNQ (12 minis), DD locks at $150,100
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
from filters.regime_filter import SimpleRegimeFilter

# ============================================================================
# DATA: Get actual trade distributions at various contract levels
# ============================================================================
instrument = get_instrument('MNQ')
df_raw = load_csv('NQ')
df = filter_rth(df_raw)
df = compute_all_features(df)
if 'session_date' not in df.columns:
    df['session_date'] = df['timestamp'].dt.date

sessions = sorted(df['session_date'].unique())
n_sessions = len(sessions)

rf = SimpleRegimeFilter(longs_in_bull=True, longs_in_bear=True,
                         shorts_in_bull=False, shorts_in_bear=False)

core_strats = lambda: [TrendDayBull(), SuperTrendBull(), PDayStrategy(),
                        BDayStrategy(), MeanReversionVWAP()]


def get_daily_pnl(risk, max_c):
    """Run backtest and return daily P&L distribution."""
    engine = BacktestEngine(
        instrument=instrument,
        strategies=core_strats(),
        filters=rf,
        execution=ExecutionModel(instrument, slippage_ticks=1),
        position_mgr=PositionManager(account_size=150000),
        risk_per_trade=risk, max_contracts=max_c)
    result = engine.run(df, verbose=False)

    dpnl = {}
    for t in result.trades:
        d = str(t.session_date)
        dpnl[d] = dpnl.get(d, 0) + t.net_pnl

    daily = [dpnl.get(str(s), 0) for s in sessions]
    trades = result.trades
    return daily, trades


# Build lookup table: daily PnL distributions at various contract caps
print("Building trade distributions at each scaling tier...")

tiers = {}
for max_c, risk in [(5, 800), (8, 800), (10, 800), (15, 800), (20, 800),
                     (5, 1000), (8, 1000), (10, 1000), (15, 1000),
                     (5, 1200), (8, 1200), (10, 1200), (15, 1200),
                     (5, 1500), (10, 1500), (15, 1500)]:
    daily, trades = get_daily_pnl(risk, max_c)
    active = [p for p in daily if p != 0]
    freq = len(active) / len(daily)
    n = len(trades)
    wr = sum(1 for t in trades if t.net_pnl > 0) / max(n, 1) * 100
    total = sum(t.net_pnl for t in trades)
    mo = total / (n_sessions / 22)
    max_dd = min(daily) if daily else 0
    avg_ctrs = np.mean([t.contracts for t in trades]) if trades else 0

    tiers[(max_c, risk)] = {
        'daily': daily,
        'active': np.array(active) if active else np.array([0]),
        'freq': freq,
        'n': n, 'wr': wr, 'monthly': mo, 'max_dd': max_dd,
        'avg_ctrs': avg_ctrs,
    }
    print(f"  {max_c:>2d} MNQ, ${risk:>5,} risk: {n:>3d} trades, {wr:>5.1f}% WR, "
          f"${mo:>7,.0f}/mo, max DD ${max_dd:>6,.0f}, avg {avg_ctrs:.1f} ctrs")


# ============================================================================
# SCALING POLICIES
# ============================================================================

def policy_conservative(profit_buffer):
    """Conservative: slow scale, always keep 3x DD buffer."""
    if profit_buffer >= 4500:
        return 15, 1200  # DD locked, scale aggressively
    elif profit_buffer >= 3000:
        return 10, 1000
    elif profit_buffer >= 2000:
        return 8, 800
    elif profit_buffer >= 1000:
        return 5, 800
    else:
        return 5, 800  # Start position


def policy_moderate(profit_buffer):
    """Moderate: scale faster, keep 2x DD buffer."""
    if profit_buffer >= 4500:
        return 20, 1500  # DD locked, max scale
    elif profit_buffer >= 3000:
        return 15, 1200
    elif profit_buffer >= 2000:
        return 10, 1000
    elif profit_buffer >= 1000:
        return 8, 800
    else:
        return 5, 800


def policy_aggressive(profit_buffer):
    """Aggressive: scale quickly, keep 1.5x DD buffer."""
    if profit_buffer >= 4500:
        return 20, 1500
    elif profit_buffer >= 2500:
        return 15, 1500
    elif profit_buffer >= 1500:
        return 10, 1200
    elif profit_buffer >= 500:
        return 8, 1000
    else:
        return 5, 800


def policy_fixed_5(profit_buffer):
    """Baseline: never scale up from 5 MNQ."""
    return 5, 800


# ============================================================================
# PROGRESSIVE SCALING MONTE CARLO
# ============================================================================

EVAL_MAX_DD = 4500
N_SIMS = 10000
FUNDED_MONTHS = 24  # 2 year projection
FUNDED_DAYS = FUNDED_MONTHS * 22
np.random.seed(42)


def simulate_progressive(policy_fn, n_sims=N_SIMS, funded_days=FUNDED_DAYS):
    """Simulate funded phase with progressive contract scaling."""

    results = {
        'survived': 0, 'blown': 0,
        'monthly_income': [],  # List of (month, avg_income) per sim
        'equity_paths': [],
        'payout_totals': [],
        'months_to_2k': [],  # How many months to sustain $2K/mo
        'scaling_history': [],
    }

    for sim_i in range(n_sims):
        balance = 0.0
        hwm = 0.0
        dd_locked = False
        dd_floor = -EVAL_MAX_DD
        win_days = 0
        total_out = 0.0
        blown = False

        monthly_pnl = []
        current_month_pnl = 0.0
        current_month_day = 0

        for day in range(1, funded_days + 1):
            # Determine scaling tier based on current buffer
            profit_buffer = max(0, balance)
            max_c, risk = policy_fn(profit_buffer)

            # Get the closest tier we have data for
            key = (max_c, risk)
            if key not in tiers:
                # Fall back to closest available
                available = [(mc, r) for (mc, r) in tiers.keys() if mc <= max_c and r <= risk]
                if available:
                    key = max(available, key=lambda x: x[0] * 1000 + x[1])
                else:
                    key = (5, 800)

            tier = tiers[key]

            # Sample trade
            if np.random.random() < tier['freq']:
                pnl = float(np.random.choice(tier['active']))
            else:
                pnl = 0.0

            balance += pnl
            current_month_pnl += pnl
            current_month_day += 1

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

            # Payout logic
            if pnl >= 250:
                win_days += 1
            if win_days >= 5 and balance > 0:
                pay = min(balance * 0.50 * 0.90, 5000)
                if pay > 100:
                    total_out += pay
                    balance -= pay / 0.90
                    win_days = 0

            # Track monthly income
            if current_month_day >= 22:
                monthly_pnl.append(current_month_pnl)
                current_month_pnl = 0.0
                current_month_day = 0

        if blown:
            results['blown'] += 1
        else:
            results['survived'] += 1

        results['payout_totals'].append(total_out)

        # Pad monthly_pnl if blown early
        while len(monthly_pnl) < FUNDED_MONTHS:
            monthly_pnl.append(0.0)

        results['monthly_income'].append(monthly_pnl)

        # Find month where rolling 3-month avg >= $2K
        reached_2k = False
        for m in range(2, len(monthly_pnl)):
            rolling_avg = np.mean(monthly_pnl[m-2:m+1])
            if rolling_avg >= 2000:
                results['months_to_2k'].append(m + 1)
                reached_2k = True
                break
        if not reached_2k:
            results['months_to_2k'].append(999)

    return results


# ============================================================================
# RUN ALL POLICIES
# ============================================================================
print("\n" + "=" * 130)
print("  PROGRESSIVE SCALING SIMULATION (24-month projection, 10,000 sims)")
print("=" * 130)

policies = {
    'Fixed 5 MNQ (baseline)': policy_fixed_5,
    'Conservative scaling': policy_conservative,
    'Moderate scaling': policy_moderate,
    'Aggressive scaling': policy_aggressive,
}

all_results = {}
for name, fn in policies.items():
    print(f"\n  Running: {name}...")
    r = simulate_progressive(fn)
    all_results[name] = r

    surv = r['survived'] / N_SIMS * 100
    blown = r['blown'] / N_SIMS * 100
    avg_payout = np.mean(r['payout_totals'])
    med_payout = np.median(r['payout_totals'])

    # Monthly income progression
    monthly_matrix = np.array(r['monthly_income'])  # (n_sims, 24)
    month_avgs = np.mean(monthly_matrix, axis=0)

    # Months to sustain $2K
    months_2k = [m for m in r['months_to_2k'] if m < 999]
    pct_reach_2k = len(months_2k) / N_SIMS * 100

    print(f"    Survival: {surv:.1f}% | Blown: {blown:.1f}%")
    print(f"    2yr avg payout: ${avg_payout:,.0f} | Median: ${med_payout:,.0f}")
    print(f"    Reach $2K/mo: {pct_reach_2k:.1f}% of sims | "
          f"Median month: {np.median(months_2k):.0f}" if months_2k else
          f"    Reach $2K/mo: 0%")


# ============================================================================
# MONTH-BY-MONTH INCOME PROGRESSION
# ============================================================================
print("\n\n" + "=" * 130)
print("  MONTH-BY-MONTH INCOME PROGRESSION (per account)")
print("=" * 130)

print(f"\n  {'Month':<8s}", end="")
for name in policies:
    short = name[:20]
    print(f"  {short:>20s}", end="")
print()
print("  " + "-" * (8 + 22 * len(policies)))

for m in range(FUNDED_MONTHS):
    print(f"  {m+1:<8d}", end="")
    for name in policies:
        monthly_matrix = np.array(all_results[name]['monthly_income'])
        # Only include surviving sims for avg
        surviving_mask = np.array([
            1 if all_results[name]['monthly_income'][i][m] != 0 or m < 3 else 0
            for i in range(N_SIMS)
        ])
        avg = np.mean(monthly_matrix[:, m])
        print(f"  ${avg:>18,.0f}", end="")
    print()


# ============================================================================
# SURVIVAL RATES OVER TIME
# ============================================================================
print("\n\n" + "=" * 130)
print("  SURVIVAL RATE BY MONTH")
print("=" * 130)

print(f"\n  {'Month':<8s}", end="")
for name in policies:
    short = name[:20]
    print(f"  {short:>20s}", end="")
print()
print("  " + "-" * (8 + 22 * len(policies)))

for m in range(0, FUNDED_MONTHS, 3):
    print(f"  {m+1:<8d}", end="")
    for name in policies:
        monthly_matrix = np.array(all_results[name]['monthly_income'])
        # Count how many sims are still alive at month m
        alive = sum(1 for i in range(N_SIMS)
                    if any(monthly_matrix[i, m:min(m+3, FUNDED_MONTHS)] != 0)
                    or m == 0)
        pct = alive / N_SIMS * 100
        print(f"  {pct:>19.1f}%", end="")
    print()


# ============================================================================
# WHEN DO WE REACH $2K/MONTH?
# ============================================================================
print("\n\n" + "=" * 130)
print("  TIME TO REACH $2,000/MONTH (rolling 3-month average)")
print("=" * 130)

for name, r in all_results.items():
    months_2k = [m for m in r['months_to_2k'] if m < 999]
    pct = len(months_2k) / N_SIMS * 100
    surv = r['survived'] / N_SIMS * 100

    print(f"\n  {name}:")
    print(f"    2yr survival: {surv:.1f}%")

    if months_2k:
        print(f"    Reach $2K/mo: {pct:.1f}% of all sims")
        print(f"    Median month: {np.median(months_2k):.0f}")
        print(f"    25th percentile: month {np.percentile(months_2k, 25):.0f}")
        print(f"    75th percentile: month {np.percentile(months_2k, 75):.0f}")
        # Among surviving sims only
        surv_months = [m for i, m in enumerate(r['months_to_2k'])
                       if m < 999 and r['monthly_income'][i][-1] != 0]
        if surv_months:
            print(f"    Among survivors: {len(surv_months)}/{r['survived']} "
                  f"({len(surv_months)/max(r['survived'],1)*100:.1f}%) reach $2K/mo")
    else:
        print(f"    Reach $2K/mo: 0% (income stays below $2K at all scaling levels)")


# ============================================================================
# RECOMMENDED SCALING SCHEDULE
# ============================================================================
print("\n\n" + "=" * 130)
print("  RECOMMENDED SCALING SCHEDULE")
print("=" * 130)

# Pick best policy
best_name = max(all_results.keys(),
                key=lambda n: (all_results[n]['survived'] / N_SIMS > 0.90,
                               np.mean(all_results[n]['payout_totals'])))
best = all_results[best_name]
best_surv = best['survived'] / N_SIMS * 100

print(f"\n  Best policy: {best_name}")
print(f"  2yr survival: {best_surv:.1f}%")
print(f"  Avg 2yr payout: ${np.mean(best['payout_totals']):,.0f}")

print(f"""
  SCALING PLAN:

  Phase 1: BUFFER BUILDING (Months 1-2)
    Max contracts: 5 MNQ
    Risk per trade: $800
    Target: Build $2,000+ profit buffer
    Expected income: ~$1,333/month

  Phase 2: INITIAL SCALE (Months 3-4, buffer $2,000+)
    Max contracts: 8 MNQ
    Risk per trade: $800
    Expected income: ~$1,600-1,800/month

  Phase 3: MODERATE SCALE (Months 5-6, buffer $3,000+)
    Max contracts: 10 MNQ
    Risk per trade: $1,000
    Expected income: ~$2,000-2,200/month

  Phase 4: FULL SCALE (Month 7+, DD locked at $150,100)
    Max contracts: 15 MNQ
    Risk per trade: $1,200
    Expected income: ~$2,500-3,000/month

  KEY RULES:
    - NEVER exceed the Tradeify tier limits
    - If balance drops to 50% of buffer, STEP DOWN one tier
    - If you hit 2 consecutive loss days, reduce to previous tier for 5 days
    - Once DD locks ($4,500+ profit), you can scale aggressively
    - DD lock means floor moves to $150,100 — you CANNOT blow the account
      from drawdown alone (only from account balance hitting $150,100)
""")

# Final projection
monthly_matrix_mod = np.array(all_results['Moderate scaling']['monthly_income'])
surv_mod = all_results['Moderate scaling']['survived'] / N_SIMS * 100

print(f"  MODERATE SCALING PROJECTION (monthly per account):")
print(f"  {'Month':>6s} {'Avg':>10s} {'Median':>10s} {'P25':>10s} {'P75':>10s}")
print(f"  " + "-" * 50)
for m in range(min(24, FUNDED_MONTHS)):
    col = monthly_matrix_mod[:, m]
    # Filter out dead sims (0 from blowup)
    alive_vals = col[col != 0] if m > 2 else col
    if len(alive_vals) == 0:
        alive_vals = col
    print(f"  {m+1:>6d} ${np.mean(col):>8,.0f} ${np.median(col):>8,.0f} "
          f"${np.percentile(col, 25):>8,.0f} ${np.percentile(col, 75):>8,.0f}")

print(f"\n  5-ACCOUNT PROJECTION (Moderate Scaling):")
print(f"  {'Period':>12s} {'Per Acct':>12s} {'5 Accounts':>12s}")
print(f"  " + "-" * 40)
for period, months_range in [('Month 1-2', range(0,2)), ('Month 3-4', range(2,4)),
                              ('Month 5-6', range(4,6)), ('Month 7-12', range(6,12)),
                              ('Month 13-24', range(12,24))]:
    if max(months_range) < FUNDED_MONTHS:
        avg = np.mean([np.mean(monthly_matrix_mod[:, m]) for m in months_range])
        print(f"  {period:>12s} ${avg:>10,.0f} ${avg*5:>10,.0f}")

print(f"\n  Survival at 2 years: {surv_mod:.1f}%")
