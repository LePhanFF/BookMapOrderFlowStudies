"""
Monte Carlo Evaluation Risk Scaling Study

Simulates prop firm evaluation at different risk multipliers (1.0x to 2.0x)
to find the optimal balance between speed-to-pass and blowup probability.

Uses actual trade P&L distribution from v14 backtest trade log.
Accounts for 1 allowed reset per evaluation ($219 each).

Tradeify $150K Select Flex rules:
  - Profit target: $9,000
  - Trailing drawdown: $4,500 (EOD)
  - No daily loss limit
  - No time limit
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from collections import defaultdict

# -- Load actual trade P&L from v14 backtest ------------------------------
TRADE_LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'output', 'trade_log_v14.csv')

def load_trade_pnls():
    """Load net P&L per trade from the v14 trade log."""
    df = pd.read_csv(TRADE_LOG)
    return df['net_pnl'].values

# -- Tradeify rules -------------------------------------------------------
STARTING_BALANCE = 150_000.0
PROFIT_TARGET = 9_000.0
MAX_TRAILING_DD = 4_500.0
EVAL_COST = 219.0

# -- Simulation parameters ------------------------------------------------
NUM_SIMULATIONS = 50_000
MAX_TRADING_DAYS = 252  # 1 year max per attempt
RISK_MULTIPLIERS = [1.0, 1.25, 1.5, 1.75, 2.0]

# From v14 backtest: 38 of 62 sessions had trades, avg ~1.77 trades/active session
TRADE_PROBABILITY = 38 / 62  # 61.3% of days have trades
MULTI_TRADE_PROB = 0.10       # 10% chance of 2nd trade on active days


def simulate_single_eval(trade_pnls: np.ndarray, risk_mult: float, rng: np.random.Generator) -> dict:
    """
    Simulate a single evaluation attempt.

    Returns dict with: passed, blown, days, trades, peak_dd, final_equity
    """
    equity = STARTING_BALANCE
    hwm = STARTING_BALANCE
    max_dd_seen = 0.0
    total_trades = 0

    for day in range(1, MAX_TRADING_DAYS + 1):
        daily_pnl = 0.0

        # Does this day have trades?
        if rng.random() < TRADE_PROBABILITY:
            # First trade
            trade_pnl = rng.choice(trade_pnls) * risk_mult
            equity += trade_pnl
            daily_pnl += trade_pnl
            total_trades += 1

            # Possible second trade
            if rng.random() < MULTI_TRADE_PROB:
                trade_pnl = rng.choice(trade_pnls) * risk_mult
                equity += trade_pnl
                daily_pnl += trade_pnl
                total_trades += 1

        # EOD: update HWM
        if equity > hwm:
            hwm = equity

        # EOD: check trailing drawdown
        current_dd = hwm - equity
        max_dd_seen = max(max_dd_seen, current_dd)

        if current_dd >= MAX_TRAILING_DD:
            return {
                'passed': False, 'blown': True, 'days': day,
                'trades': total_trades, 'peak_dd': max_dd_seen,
                'final_equity': equity
            }

        # EOD: check if passed
        if (equity - STARTING_BALANCE) >= PROFIT_TARGET:
            return {
                'passed': True, 'blown': False, 'days': day,
                'trades': total_trades, 'peak_dd': max_dd_seen,
                'final_equity': equity
            }

    # Timed out (didn't pass or blow in 252 days)
    return {
        'passed': False, 'blown': False, 'days': MAX_TRADING_DAYS,
        'trades': total_trades, 'peak_dd': max_dd_seen,
        'final_equity': equity
    }


def simulate_with_one_reset(trade_pnls: np.ndarray, risk_mult: float, rng: np.random.Generator) -> dict:
    """
    Simulate evaluation with 1 allowed reset.
    If first attempt blows, try again (costs $219 + $219 = $438 total).
    """
    attempt1 = simulate_single_eval(trade_pnls, risk_mult, rng)

    if attempt1['passed']:
        return {
            'passed': True, 'attempts': 1, 'total_days': attempt1['days'],
            'total_trades': attempt1['trades'], 'peak_dd': attempt1['peak_dd'],
            'cost': EVAL_COST, 'final_equity': attempt1['final_equity']
        }

    if attempt1['blown']:
        # Reset and try again
        attempt2 = simulate_single_eval(trade_pnls, risk_mult, rng)

        if attempt2['passed']:
            return {
                'passed': True, 'attempts': 2,
                'total_days': attempt1['days'] + attempt2['days'],
                'total_trades': attempt1['trades'] + attempt2['trades'],
                'peak_dd': max(attempt1['peak_dd'], attempt2['peak_dd']),
                'cost': EVAL_COST * 2, 'final_equity': attempt2['final_equity']
            }
        else:
            return {
                'passed': False, 'attempts': 2,
                'total_days': attempt1['days'] + attempt2['days'],
                'total_trades': attempt1['trades'] + attempt2['trades'],
                'peak_dd': max(attempt1['peak_dd'], attempt2['peak_dd']),
                'cost': EVAL_COST * 2, 'final_equity': attempt2['final_equity']
            }

    # Timed out on first attempt (no blowup, just slow)
    return {
        'passed': False, 'attempts': 1, 'total_days': attempt1['days'],
        'total_trades': attempt1['trades'], 'peak_dd': attempt1['peak_dd'],
        'cost': EVAL_COST, 'final_equity': attempt1['final_equity']
    }


def run_study():
    """Run the full risk scaling study."""
    trade_pnls = load_trade_pnls()
    rng = np.random.default_rng(seed=42)

    print("=" * 80)
    print("PROP FIRM EVALUATION RISK SCALING STUDY")
    print("=" * 80)
    print(f"\nTrade log: {len(trade_pnls)} trades from v14 backtest")
    print(f"  Win rate: {(trade_pnls > 0).mean():.1%}")
    print(f"  Avg win:  ${trade_pnls[trade_pnls > 0].mean():,.2f}")
    print(f"  Avg loss: ${trade_pnls[trade_pnls < 0].mean():,.2f}")
    print(f"  Expectancy: ${trade_pnls.mean():,.2f}/trade")
    print(f"\nSimulations per multiplier: {NUM_SIMULATIONS:,}")
    print(f"Tradeify $150K: target=$9,000, trailing DD=$4,500")
    print(f"Allowed resets: 1 per evaluation\n")

    results = {}

    for mult in RISK_MULTIPLIERS:
        print(f"Running {mult:.2f}x risk... ", end='', flush=True)

        sim_results = []
        for _ in range(NUM_SIMULATIONS):
            result = simulate_with_one_reset(trade_pnls, mult, rng)
            sim_results.append(result)

        passed = [r for r in sim_results if r['passed']]
        blown_both = [r for r in sim_results if not r['passed'] and r['attempts'] == 2]

        # Single attempt stats
        single_attempt_results = []
        for _ in range(NUM_SIMULATIONS):
            single_attempt_results.append(simulate_single_eval(trade_pnls, mult, rng))

        single_pass_rate = sum(1 for r in single_attempt_results if r['passed']) / NUM_SIMULATIONS
        single_blow_rate = sum(1 for r in single_attempt_results if r['blown']) / NUM_SIMULATIONS

        stats = {
            'risk_mult': mult,
            'scaled_expectancy': trade_pnls.mean() * mult,
            'scaled_avg_loss': trade_pnls[trade_pnls < 0].mean() * mult,
            'scaled_max_loss': trade_pnls.min() * mult,
            # Single attempt
            'single_pass_rate': single_pass_rate,
            'single_blow_rate': single_blow_rate,
            # With 1 reset
            'pass_rate': len(passed) / NUM_SIMULATIONS,
            'blow_both_rate': len(blown_both) / NUM_SIMULATIONS,
            'timeout_rate': sum(1 for r in sim_results if not r['passed'] and r['attempts'] == 1) / NUM_SIMULATIONS,
            'avg_days_to_pass': np.mean([r['total_days'] for r in passed]) if passed else float('inf'),
            'median_days_to_pass': np.median([r['total_days'] for r in passed]) if passed else float('inf'),
            'p25_days': np.percentile([r['total_days'] for r in passed], 25) if passed else float('inf'),
            'p75_days': np.percentile([r['total_days'] for r in passed], 75) if passed else float('inf'),
            'avg_trades_to_pass': np.mean([r['total_trades'] for r in passed]) if passed else float('inf'),
            'avg_peak_dd': np.mean([r['peak_dd'] for r in sim_results]),
            'p95_peak_dd': np.percentile([r['peak_dd'] for r in sim_results], 95),
            'avg_cost': np.mean([r['cost'] for r in sim_results]),
            'needed_2_attempts': sum(1 for r in passed if r['attempts'] == 2) / max(len(passed), 1),
        }
        results[mult] = stats
        print(f"done (pass={stats['pass_rate']:.1%}, blow={stats['single_blow_rate']:.1%} per attempt)")

    # -- Print summary table ----------------------------------------------
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n{'-'*80}")
    print(f"{'Multiplier':>12} {'Exp/Trade':>10} {'MaxLoss':>10} | {'Pass%':>7} {'Blow%':>7} | "
          f"{'MedDays':>8} {'AvgDays':>8} | {'P95 DD':>8} {'Cost':>7}")
    print(f"{'-'*80}")

    for mult in RISK_MULTIPLIERS:
        s = results[mult]
        print(f"  {mult:.2f}x      "
              f"${s['scaled_expectancy']:>7,.0f}  "
              f"${s['scaled_max_loss']:>8,.0f} | "
              f"{s['pass_rate']:>6.1%} "
              f"{s['single_blow_rate']:>6.1%} | "
              f"{s['median_days_to_pass']:>7.0f}  "
              f"{s['avg_days_to_pass']:>7.0f}  | "
              f"${s['p95_peak_dd']:>7,.0f} "
              f"${s['avg_cost']:>5,.0f}")

    print(f"{'-'*80}")

    # -- Detailed per-multiplier analysis ---------------------------------
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS PER RISK LEVEL")
    print("=" * 80)

    for mult in RISK_MULTIPLIERS:
        s = results[mult]
        print(f"\n{'-' * 60}")
        print(f"  {mult:.2f}x RISK MULTIPLIER")
        print(f"{'-' * 60}")
        print(f"  Scaled expectancy:     ${s['scaled_expectancy']:>8,.2f} / trade")
        print(f"  Scaled avg loss:       ${s['scaled_avg_loss']:>8,.2f}")
        print(f"  Scaled worst loss:     ${s['scaled_max_loss']:>8,.2f}")
        print(f"")
        print(f"  SINGLE ATTEMPT:")
        print(f"    Pass rate:           {s['single_pass_rate']:>8.1%}")
        print(f"    Blowup rate:         {s['single_blow_rate']:>8.1%}")
        print(f"")
        print(f"  WITH 1 RESET (2 attempts max):")
        print(f"    Pass rate:           {s['pass_rate']:>8.1%}")
        print(f"    Blow both attempts:  {s['blow_both_rate']:>8.1%}")
        print(f"    Timeout (no blow):   {s['timeout_rate']:>8.1%}")
        print(f"    Needed 2nd attempt:  {s['needed_2_attempts']:>8.1%}")
        print(f"")
        print(f"  TIMELINE (when passed):")
        print(f"    25th percentile:     {s['p25_days']:>5.0f} trading days")
        print(f"    Median:              {s['median_days_to_pass']:>5.0f} trading days")
        print(f"    75th percentile:     {s['p75_days']:>5.0f} trading days")
        print(f"    Avg trades to pass:  {s['avg_trades_to_pass']:>5.0f}")
        print(f"")
        print(f"  RISK:")
        print(f"    Avg peak drawdown:   ${s['avg_peak_dd']:>8,.2f}")
        print(f"    95th pctl peak DD:   ${s['p95_peak_dd']:>8,.2f}")
        print(f"    Avg eval cost:       ${s['avg_cost']:>8,.2f}")

    # -- Recommendation ---------------------------------------------------
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    # Find best risk-adjusted multiplier: maximize (pass_rate / days) while keeping blow_both < 10%
    best = None
    for mult in RISK_MULTIPLIERS:
        s = results[mult]
        if s['pass_rate'] > 0.80 and s['blow_both_rate'] < 0.15:
            if best is None or s['median_days_to_pass'] < results[best]['median_days_to_pass']:
                best = mult

    if best:
        s = results[best]
        print(f"\n  RECOMMENDED: {best:.2f}x risk multiplier")
        print(f"  - Pass rate (with 1 reset): {s['pass_rate']:.1%}")
        print(f"  - Median days to pass: {s['median_days_to_pass']:.0f}")
        print(f"  - Blowup rate (both attempts): {s['blow_both_rate']:.1%}")
        print(f"  - Expected cost: ${s['avg_cost']:,.0f}")
    else:
        print("\n  No multiplier meets the criteria (>80% pass, <15% double-blow)")
        print("  Recommend sticking with 1.0x and being patient")

    # -- Speed vs Risk tradeoff table -------------------------------------
    print(f"\n{'-' * 60}")
    print("  SPEED vs RISK TRADEOFF:")
    print(f"{'-' * 60}")
    for mult in RISK_MULTIPLIERS:
        s = results[mult]
        bar_len = int(s['single_blow_rate'] * 50)
        risk_bar = '#' * bar_len + '.' * (50 - bar_len)
        print(f"  {mult:.2f}x | {s['median_days_to_pass']:>3.0f}d median | blow={s['single_blow_rate']:.1%} [{risk_bar}]")

    return results


if __name__ == '__main__':
    results = run_study()
