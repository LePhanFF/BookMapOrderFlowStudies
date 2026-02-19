"""
Monte Carlo Simulation: Tradeify Select Flex $150K Prop Account

Uses ACTUAL backtest trade distribution from v13 Playbook + OF Quality Gate
to simulate:
  1. Evaluation phase: How long to reach $9,000 profit target
  2. Funded phase: Payout timeline and sustainability
  3. Risk of ruin (hitting $4,500 EOD trailing drawdown)

THREE SCENARIOS: Backtest-exact, Realistic (degraded), and Stress-test

Tradeify Select Flex $150K Rules:
  EVALUATION:
    - Profit target: $9,000
    - EOD trailing max drawdown: $4,500
    - No daily loss limit
    - 40% consistency rule (no single day > 40% of total profit)
    - Min 3 trading days
    - Contract limit: 12 minis / 120 micros

  FUNDED (Select Flex):
    - EOD trailing drawdown: $4,500 (locks at $150,100 when profit > $4,600)
    - No daily loss limit, no consistency rule
    - Payout: 90/10 split, every 5 winning days ($250+ min per day)
    - Max $5,000 per payout, up to 50% of total profit
    - Scaling: 3 minis at $0-$1499, 4 at $1500-$1999, 5 at $2000-$2999,
               8 at $3000-$4499, 12 at $4500+
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

project_root = Path(__file__).resolve().parent

# ============================================================================
#  ACTUAL TRADE DATA FROM v13 BACKTEST
# ============================================================================

trade_log = pd.read_csv(project_root / 'output' / 'trade_log.csv')

# Deduplicate to unique sessions (Trend Bull + P-Day fire on same bar)
seen_sessions = {}
for _, trade in trade_log.iterrows():
    session = trade['session_date'][:10]
    if session not in seen_sessions:
        seen_sessions[session] = trade
    else:
        if trade['net_pnl'] > seen_sessions[session]['net_pnl']:
            seen_sessions[session] = trade

session_trades = list(seen_sessions.values())

print("=" * 110)
print("  TRADEIFY SELECT FLEX $150K - MONTE CARLO SIMULATION")
print("=" * 110)

# Normalize to points per contract
trade_points = []
for t in session_trades:
    contracts = t['contracts']
    gross = t['gross_pnl']
    pts = gross / (contracts * 2.0)  # $2/pt/contract for MNQ
    trade_points.append(pts)

print(f"\nActual backtest: {len(trade_points)} unique trade sessions / 62 total sessions")
print(f"Points/contract: mean={np.mean(trade_points):+.1f}, median={np.median(trade_points):+.1f}, "
      f"min={min(trade_points):+.1f}, max={max(trade_points):+.1f}")

TRADE_FREQUENCY = len(trade_points) / 62.0
TRADING_DAYS_PER_MONTH = 22
print(f"Trade frequency: {TRADE_FREQUENCY:.1%} ({TRADE_FREQUENCY * TRADING_DAYS_PER_MONTH:.1f} trades/month)")

# ============================================================================
#  PROP ACCOUNT PARAMETERS
# ============================================================================

EVAL_PROFIT_TARGET = 9000
EVAL_MAX_DRAWDOWN = 4500
EVAL_CONSISTENCY_PCT = 0.40
EVAL_MIN_DAYS = 3
EVAL_COST_PER_MONTH = 359

FUNDED_DRAWDOWN = 4500
FUNDED_LOCK_THRESHOLD = 4600
FUNDED_PAYOUT_DAYS = 5
FUNDED_MIN_WIN_DAY = 250
FUNDED_MAX_PAYOUT = 5000
FUNDED_PROFIT_SPLIT = 0.90
FUNDED_PAYOUT_PCT = 0.50


def get_max_mnq(equity_above_start):
    if equity_above_start >= 4500: return 120
    elif equity_above_start >= 3000: return 80
    elif equity_above_start >= 2000: return 50
    elif equity_above_start >= 1500: return 40
    else: return 30


# ============================================================================
#  THREE PERFORMANCE SCENARIOS
# ============================================================================

# Scenario A: Backtest-exact (80% WR, losses are -$2.74 breakeven stops)
backtest_pts = trade_points.copy()

# Scenario B: Realistic (70% WR, real losses of -40 to -80 pts, winners degraded 20%)
# Live trading has slippage, hesitation, wider stops, partial fills
# Model: 70% of the time draw from winners (degraded 20%), 30% draw from loss pool
realistic_winners = [p * 0.80 for p in trade_points if p > 0]  # 20% haircut
realistic_losses = [-60, -50, -40, -70, -55, -45, -65, -80, -35, -50]  # realistic stop-outs

# Scenario C: Stress test (60% WR, bigger losses, full stop-outs)
stress_winners = [p * 0.65 for p in trade_points if p > 0]  # 35% haircut
stress_losses = [-80, -70, -90, -60, -100, -75, -85, -65, -95, -110]

scenarios_perf = {
    'Backtest-Exact (80% WR)': {
        'pool': backtest_pts,
        'mode': 'pool',  # Sample directly from pool
    },
    'Realistic Live (70% WR)': {
        'win_pool': realistic_winners,
        'loss_pool': realistic_losses,
        'win_rate': 0.70,
        'mode': 'split',
    },
    'Stress Test (60% WR)': {
        'win_pool': stress_winners,
        'loss_pool': stress_losses,
        'win_rate': 0.60,
        'mode': 'split',
    },
}


def sample_trade(scenario):
    if scenario['mode'] == 'pool':
        return np.random.choice(scenario['pool'])
    else:
        if np.random.random() < scenario['win_rate']:
            return np.random.choice(scenario['win_pool'])
        else:
            return np.random.choice(scenario['loss_pool'])


# ============================================================================
#  CONTRACT SIZING OPTIONS
# ============================================================================

sizing_options = {
    'Conservative (2 MNQ)': 2,
    'Moderate (4 MNQ)': 4,
    'Prop-Optimal (6 MNQ)': 6,
    'Aggressive (10 MNQ)': 10,
}


# ============================================================================
#  SIMULATION ENGINES
# ============================================================================

N_SIMS = 10000
MAX_EVAL_DAYS = 500
FUNDED_DAYS = 252

np.random.seed(42)


def simulate_eval(perf_scenario, target_contracts, n_sims=N_SIMS):
    results = {'passed': 0, 'failed': 0, 'days': [], 'trades': [], 'max_dd_seen': []}

    for _ in range(n_sims):
        balance = 0.0
        hwm = 0.0
        dd_floor = -EVAL_MAX_DRAWDOWN
        daily_pnls = []
        trades = 0

        for day in range(1, MAX_EVAL_DAYS + 1):
            if np.random.random() < TRADE_FREQUENCY:
                pts = sample_trade(perf_scenario)
                max_c = get_max_mnq(max(0, balance))
                c = min(target_contracts, max_c)
                pnl = pts * c * 2.0 - c * 1.24
                trades += 1
            else:
                pnl = 0.0

            daily_pnls.append(pnl)
            balance += pnl

            if balance > hwm:
                hwm = balance
            dd_floor = hwm - EVAL_MAX_DRAWDOWN

            if balance <= dd_floor:
                results['failed'] += 1
                results['max_dd_seen'].append(hwm - balance)
                break

            if balance >= EVAL_PROFIT_TARGET and trades >= EVAL_MIN_DAYS:
                # Check 40% consistency
                total_pos = sum(p for p in daily_pnls if p > 0)
                max_day = max(daily_pnls) if daily_pnls else 0
                if total_pos > 0 and max_day / total_pos > EVAL_CONSISTENCY_PCT:
                    continue  # Keep trading to dilute
                results['passed'] += 1
                results['days'].append(day)
                results['trades'].append(trades)
                break
        else:
            results['failed'] += 1  # Timed out

    return results


def simulate_funded(perf_scenario, target_contracts, n_sims=N_SIMS):
    results = {
        'survived': 0, 'blown': 0,
        'payouts': [], 'withdrawn': [],
        'first_payout_months': [],
        'balance_1yr': [],
    }

    for _ in range(n_sims):
        balance = 0.0
        hwm = 0.0
        dd_floor = -FUNDED_DRAWDOWN
        dd_locked = False
        dd_lock_val = None
        win_days = 0
        total_out = 0.0
        n_payouts = 0
        first_pay_day = None
        blown = False

        for day in range(1, FUNDED_DAYS + 1):
            if np.random.random() < TRADE_FREQUENCY:
                pts = sample_trade(perf_scenario)
                max_c = get_max_mnq(max(0, balance))
                c = min(target_contracts, max_c)
                pnl = pts * c * 2.0 - c * 1.24
            else:
                pnl = 0.0

            balance += pnl

            if balance > hwm:
                hwm = balance

            if not dd_locked:
                dd_floor = hwm - FUNDED_DRAWDOWN
                if balance >= FUNDED_LOCK_THRESHOLD:
                    dd_locked = True
                    dd_lock_val = 100
                    dd_floor = dd_lock_val
            else:
                dd_floor = dd_lock_val

            if balance <= dd_floor:
                blown = True
                break

            if pnl >= FUNDED_MIN_WIN_DAY:
                win_days += 1

            if win_days >= FUNDED_PAYOUT_DAYS and balance > 0:
                pay = min(balance * FUNDED_PAYOUT_PCT * FUNDED_PROFIT_SPLIT, FUNDED_MAX_PAYOUT)
                if pay > 100:
                    total_out += pay
                    balance -= pay / FUNDED_PROFIT_SPLIT
                    n_payouts += 1
                    win_days = 0
                    if first_pay_day is None:
                        first_pay_day = day

        if blown:
            results['blown'] += 1
        else:
            results['survived'] += 1

        results['payouts'].append(n_payouts)
        results['withdrawn'].append(total_out)
        results['balance_1yr'].append(balance if not blown else 0)
        if first_pay_day is not None:
            results['first_payout_months'].append(first_pay_day / TRADING_DAYS_PER_MONTH)

    return results


# ============================================================================
#  RUN ALL SIMULATIONS
# ============================================================================

all_eval = {}
all_funded = {}

for perf_name, perf_cfg in scenarios_perf.items():
    for size_name, contracts in sizing_options.items():
        key = (perf_name, size_name)
        all_eval[key] = simulate_eval(perf_cfg, contracts)
        all_funded[key] = simulate_funded(perf_cfg, contracts)


# ============================================================================
#  PRINT RESULTS
# ============================================================================

print("\n" + "=" * 110)
print("  EVALUATION PHASE: PASS RATE & TIMELINE")
print("=" * 110)
print()
print(f"{'Performance Scenario':<28s} {'Sizing':<24s} {'Pass%':>6s} {'Fail%':>6s} "
      f"{'Med Days':>9s} {'Med Mo':>7s} {'Trades':>7s} {'10th':>6s} {'90th':>6s}")
print("-" * 110)

for perf_name in scenarios_perf:
    for size_name, contracts in sizing_options.items():
        r = all_eval[(perf_name, size_name)]
        pr = r['passed'] / N_SIMS * 100
        fr = r['failed'] / N_SIMS * 100
        md = np.median(r['days']) if r['days'] else 0
        mt = np.median(r['trades']) if r['trades'] else 0
        mo = md / TRADING_DAYS_PER_MONTH
        p10 = np.percentile(r['days'], 10) if r['days'] else 0
        p90 = np.percentile(r['days'], 90) if r['days'] else 0
        print(f"{perf_name:<28s} {size_name:<24s} {pr:>5.1f}% {fr:>5.1f}% "
              f"{md:>7.0f} d {mo:>5.1f} m {mt:>5.0f} t {p10:>5.0f} d {p90:>5.0f} d")
    print()


print("\n" + "=" * 110)
print("  FUNDED PHASE: 1-YEAR SURVIVAL & PAYOUTS")
print("=" * 110)
print()
print(f"{'Performance Scenario':<28s} {'Sizing':<24s} {'Surv%':>6s} {'Blown%':>7s} "
      f"{'Payouts':>8s} {'Med W/D':>10s} {'Avg W/D':>10s} {'1st Pay':>8s}")
print("-" * 110)

for perf_name in scenarios_perf:
    for size_name, contracts in sizing_options.items():
        r = all_funded[(perf_name, size_name)]
        sv = r['survived'] / N_SIMS * 100
        bl = r['blown'] / N_SIMS * 100
        mp = np.median(r['payouts'])
        mw = np.median(r['withdrawn'])
        aw = np.mean(r['withdrawn'])
        fp = np.median(r['first_payout_months']) if r['first_payout_months'] else 0
        print(f"{perf_name:<28s} {size_name:<24s} {sv:>5.1f}% {bl:>6.1f}% "
              f"{mp:>6.1f} p ${mw:>8,.0f} ${aw:>8,.0f} {fp:>5.1f} mo")
    print()


# ============================================================================
#  COMBINED: NET ANNUAL INCOME AFTER EVAL COSTS
# ============================================================================

print("\n" + "=" * 110)
print("  NET ANNUAL INCOME (after eval costs, per account)")
print("=" * 110)
print()
print(f"{'Performance Scenario':<28s} {'Sizing':<24s} {'Eval Cost':>10s} "
      f"{'1yr W/D':>10s} {'Net/Year':>10s} {'Net/Month':>10s}")
print("-" * 110)

for perf_name in scenarios_perf:
    for size_name, contracts in sizing_options.items():
        ev = all_eval[(perf_name, size_name)]
        fu = all_funded[(perf_name, size_name)]

        pass_rate = ev['passed'] / N_SIMS
        if pass_rate > 0:
            attempts = 1 / pass_rate
            avg_eval_mo = np.mean(ev['days']) / TRADING_DAYS_PER_MONTH if ev['days'] else 12
        else:
            attempts = 10
            avg_eval_mo = 12

        eval_cost = attempts * avg_eval_mo * EVAL_COST_PER_MONTH
        survival = fu['survived'] / N_SIMS
        avg_wd = np.mean(fu['withdrawn'])
        net_annual = avg_wd * survival - eval_cost
        net_monthly = net_annual / 12

        print(f"{perf_name:<28s} {size_name:<24s} ${eval_cost:>8,.0f} "
              f"${avg_wd:>8,.0f} ${net_annual:>8,.0f} ${net_monthly:>8,.0f}")
    print()


# ============================================================================
#  $9,000 TARGET ANALYSIS: FASTEST PATH
# ============================================================================

print("\n" + "=" * 110)
print("  FASTEST PATH TO $9,000 EVAL TARGET")
print("=" * 110)
print()

# For each scenario, compute expected $ per trade day
for perf_name, perf_cfg in scenarios_perf.items():
    print(f"  {perf_name}:")

    if perf_cfg['mode'] == 'pool':
        pool = perf_cfg['pool']
        avg_pts = np.mean(pool)
    else:
        wr = perf_cfg['win_rate']
        avg_win = np.mean(perf_cfg['win_pool'])
        avg_loss = np.mean(perf_cfg['loss_pool'])
        avg_pts = wr * avg_win + (1 - wr) * avg_loss

    for size_name, c in sizing_options.items():
        pnl_per_trade = avg_pts * c * 2.0 - c * 1.24
        trades_needed = EVAL_PROFIT_TARGET / pnl_per_trade if pnl_per_trade > 0 else 999
        days_needed = trades_needed / TRADE_FREQUENCY
        months = days_needed / TRADING_DAYS_PER_MONTH
        print(f"    {size_name:<24s}: ${pnl_per_trade:>+7.0f}/trade, "
              f"~{trades_needed:.0f} trades, ~{days_needed:.0f} days ({months:.1f} months)")
    print()


# ============================================================================
#  PAYOUT SCHEDULE DETAIL (Moderate 4 MNQ, Realistic)
# ============================================================================

print("\n" + "=" * 110)
print("  PAYOUT SCHEDULE: MODERATE (4 MNQ) + REALISTIC LIVE (70% WR)")
print("=" * 110)
print()

# Run a single detailed sim
np.random.seed(123)
perf = scenarios_perf['Realistic Live (70% WR)']
c = 4
balance = 0.0
hwm = 0.0
dd_floor = -FUNDED_DRAWDOWN
dd_locked = False
dd_lock_val = None
win_days = 0
payouts_log = []
equity_log = []

for day in range(1, 253):
    if np.random.random() < TRADE_FREQUENCY:
        pts = sample_trade(perf)
        max_c = get_max_mnq(max(0, balance))
        ac = min(c, max_c)
        pnl = pts * ac * 2.0 - ac * 1.24
        traded = True
    else:
        pnl = 0.0
        traded = False

    balance += pnl
    if balance > hwm:
        hwm = balance
    if not dd_locked:
        dd_floor = hwm - FUNDED_DRAWDOWN
        if balance >= FUNDED_LOCK_THRESHOLD:
            dd_locked = True
            dd_lock_val = 100
            dd_floor = dd_lock_val
    else:
        dd_floor = dd_lock_val

    if balance <= dd_floor:
        print(f"  *** BLOWN on day {day} (balance ${balance:,.0f}, floor ${dd_floor:,.0f}) ***")
        break

    if pnl >= FUNDED_MIN_WIN_DAY:
        win_days += 1

    if win_days >= FUNDED_PAYOUT_DAYS and balance > 0:
        pay = min(balance * FUNDED_PAYOUT_PCT * FUNDED_PROFIT_SPLIT, FUNDED_MAX_PAYOUT)
        if pay > 100:
            payouts_log.append({
                'day': day,
                'month': day / TRADING_DAYS_PER_MONTH,
                'balance_before': balance,
                'payout': pay,
                'balance_after': balance - pay / FUNDED_PROFIT_SPLIT,
            })
            balance -= pay / FUNDED_PROFIT_SPLIT
            win_days = 0

    equity_log.append({'day': day, 'balance': balance, 'dd_floor': dd_floor, 'traded': traded})

print(f"{'Payout':>7s} {'Day':>5s} {'Month':>6s} {'Balance Before':>15s} {'Payout (90%)':>14s} {'Balance After':>14s}")
print("-" * 70)
total_paid = 0
for i, p in enumerate(payouts_log):
    total_paid += p['payout']
    print(f"  #{i+1:<4d} {p['day']:>5d} {p['month']:>5.1f} ${p['balance_before']:>12,.0f} "
          f"${p['payout']:>11,.0f} ${p['balance_after']:>11,.0f}")

print(f"\n  Total payouts: {len(payouts_log)}")
print(f"  Total withdrawn: ${total_paid:,.0f}")
print(f"  Final balance: ${balance:,.0f}")
print(f"  DD locked: {dd_locked}")


# ============================================================================
#  MULTI-ACCOUNT PROJECTION
# ============================================================================

print("\n\n" + "=" * 110)
print("  MULTI-ACCOUNT STRATEGY (same signals, multiple accounts)")
print("=" * 110)
print()

best_perf = 'Realistic Live (70% WR)'
best_size = 'Prop-Optimal (6 MNQ)'
best_c = 6

ev = all_eval[(best_perf, best_size)]
fu = all_funded[(best_perf, best_size)]
pass_rate = ev['passed'] / N_SIMS
attempts = 1 / pass_rate if pass_rate > 0 else 10
avg_mo = np.mean(ev['days']) / TRADING_DAYS_PER_MONTH if ev['days'] else 12
eval_cost = attempts * avg_mo * EVAL_COST_PER_MONTH
survival = fu['survived'] / N_SIMS
avg_wd = np.mean(fu['withdrawn'])

for n in [1, 2, 3, 5]:
    net = (avg_wd * survival - eval_cost) * n
    print(f"  {n} account(s): ${net:>10,.0f}/year  (${net/12:>8,.0f}/month)")

print()
print("  All accounts take identical trades -- one screen, one decision.")
print("  Eval cost is per-account but signal generation is shared.")
