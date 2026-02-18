"""
Aggressive Prop Firm Monte Carlo Simulation

Sequential eval strategy with multi-account funded pipeline.
Compares Rotation vs Copy Trade execution modes at 3/6/9/12 month horizons.

Usage:
    python -m prop.sim_aggressive
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from prop import rules
from prop.account import PropAccount, Phase
from prop.sizer import PropSizer
from prop.pipeline import PropPipeline


# ============================================================================
#  TRADE DATA FROM v14 BACKTEST (Edge Fade + Playbook)
# ============================================================================

def load_trade_data():
    """
    Load and deduplicate v14 trade data.

    v14 includes Edge Fade (EDGE_TO_MID) which fires multiple times per session
    on b_day/neutral days. Deduplication rules:
    - Trend Bull + P-Day fire on same bar -> keep one (identical signals)
    - B-Day always kept (different setup type)
    - Edge Fade: keep first entry per session (representative for sim pool)
    - Multiple Edge Fade entries modeled via MULTI_TRADE_PROBABILITY in sim
    """
    # Try v14 log first, fall back to v13
    v14_path = project_root / 'output' / 'trade_log_v14.csv'
    v13_path = project_root / 'output' / 'trade_log.csv'
    trade_log_path = v14_path if v14_path.exists() else v13_path
    trade_log = pd.read_csv(trade_log_path)

    # Deduplicate: keep one representative trade per (session, strategy_group)
    seen_sessions = {}
    for _, trade in trade_log.iterrows():
        session = trade['session_date'][:10]
        strategy = trade['strategy_name']

        # Group Trend Bull + P-Day together (identical signals)
        if strategy in ('Trend Day Bull', 'P-Day'):
            key = (session, 'TrendPDay')
        else:
            key = (session, strategy)

        if key not in seen_sessions:
            seen_sessions[key] = trade

    # Build deduplicated list preserving all strategy types
    session_trades = list(seen_sessions.values())
    return session_trades


def build_trade_pool(session_trades):
    """
    Build trade pool with setup classification and points-per-contract.

    Returns list of dicts with:
      - points: P&L in NQ points per 1 MNQ contract
      - strategy_name: for Type A/B classification
      - setup_type: for Type A/B classification
      - confidence: from trade data (or inferred)
      - stop_distance_pts: distance from entry to stop in points
      - source: 'playbook' or 'edge_fade' for sim weighting
    """
    sizer = PropSizer()
    pool = []

    for t in session_trades:
        contracts = t['contracts']
        gross = t['gross_pnl']
        pts = gross / (contracts * rules.MNQ_POINT_VALUE)

        # Calculate stop distance
        entry = t['entry_price']
        stop = t['stop_price']
        stop_dist = abs(entry - stop)
        if stop_dist < 1.0:
            stop_dist = 50.0  # Default if stop is at entry (breakeven stop)

        strategy_name = t['strategy_name']
        setup_type = t['setup_type']

        # Infer confidence from the data
        # B-Day always high confidence; Trend Bull with moderate strength = medium
        if strategy_name == 'B-Day':
            confidence = 'high'
        elif t.get('trend_strength', 'moderate') == 'strong':
            confidence = 'high'
        else:
            confidence = 'medium'

        grade = sizer.classify_setup(strategy_name, setup_type, confidence)

        # Source tracking for sim weighting
        source = 'edge_fade' if strategy_name == 'Edge Fade' else 'playbook'

        pool.append({
            'points': pts,
            'strategy_name': strategy_name,
            'setup_type': setup_type,
            'confidence': confidence,
            'grade': grade,
            'stop_distance_pts': stop_dist,
            'is_winner': t['is_winner'],
            'source': source,
        })

    return pool


# ============================================================================
#  PERFORMANCE SCENARIOS
# ============================================================================

def build_scenarios(trade_pool):
    """
    Build 3 performance scenarios from the v14 trade pool.

    v14 portfolio is split: Playbook (80% WR, $222/trade) + Edge Fade (53% WR, $86/trade).
    Scenarios model degradation from backtest to live trading.
    """
    # Separate by source for proper modeling
    playbook_trades = [t for t in trade_pool if t['source'] == 'playbook']
    edge_fade_trades = [t for t in trade_pool if t['source'] == 'edge_fade']

    playbook_winner_pts = [t['points'] for t in playbook_trades if t['is_winner']]
    edge_fade_winner_pts = [t['points'] for t in edge_fade_trades if t['is_winner']]
    all_winner_pts = [t['points'] for t in trade_pool if t['is_winner']]

    # Type A/B breakdown
    type_a_trades = [t for t in trade_pool if t['grade'] == 'A']
    type_b_trades = [t for t in trade_pool if t['grade'] == 'B']

    avg_stop_a = np.mean([t['stop_distance_pts'] for t in type_a_trades]) if type_a_trades else 50.0
    avg_stop_b = np.mean([t['stop_distance_pts'] for t in type_b_trades]) if type_b_trades else 50.0

    type_a_frac = len(type_a_trades) / len(trade_pool) if trade_pool else 0.1

    # Edge Fade fraction of all trades (for blended WR modeling)
    edge_fade_frac = len(edge_fade_trades) / len(trade_pool) if trade_pool else 0.7

    # Edge Fade loss points from actual data
    edge_fade_loss_pts = [t['points'] for t in edge_fade_trades if not t['is_winner']]
    playbook_loss_pts = [t['points'] for t in playbook_trades if not t['is_winner']]

    scenarios = {
        'Backtest-Exact (58% WR)': {
            'win_rate_playbook': 0.80,
            'win_rate_edge_fade': 0.533,
            'playbook_winner_pts': playbook_winner_pts,
            'edge_fade_winner_pts': edge_fade_winner_pts,
            'all_winner_pts': all_winner_pts,
            'winner_haircut': 1.0,
            'edge_fade_loss_pts': edge_fade_loss_pts if edge_fade_loss_pts else [-50],
            'playbook_loss_pts': playbook_loss_pts if playbook_loss_pts else [-0.25],
            'type_a_frac': type_a_frac,
            'avg_stop_a': avg_stop_a,
            'avg_stop_b': avg_stop_b,
            'edge_fade_frac': edge_fade_frac,
        },
        'Realistic Live (50% WR)': {
            'win_rate_playbook': 0.70,
            'win_rate_edge_fade': 0.45,
            'playbook_winner_pts': playbook_winner_pts,
            'edge_fade_winner_pts': edge_fade_winner_pts,
            'all_winner_pts': all_winner_pts,
            'winner_haircut': 0.80,
            'edge_fade_loss_pts': edge_fade_loss_pts if edge_fade_loss_pts else [-50],
            'playbook_loss_pts': [-80, -40],  # Real stop-outs for playbook
            'type_a_frac': type_a_frac,
            'avg_stop_a': avg_stop_a,
            'avg_stop_b': avg_stop_b,
            'edge_fade_frac': edge_fade_frac,
        },
        'Stress Test (42% WR)': {
            'win_rate_playbook': 0.60,
            'win_rate_edge_fade': 0.38,
            'playbook_winner_pts': playbook_winner_pts,
            'edge_fade_winner_pts': edge_fade_winner_pts,
            'all_winner_pts': all_winner_pts,
            'winner_haircut': 0.65,
            'edge_fade_loss_pts': edge_fade_loss_pts if edge_fade_loss_pts else [-50],
            'playbook_loss_pts': [-110, -60],  # Full stop-outs
            'type_a_frac': type_a_frac,
            'avg_stop_a': avg_stop_a,
            'avg_stop_b': avg_stop_b,
            'edge_fade_frac': edge_fade_frac,
        },
    }
    return scenarios


def sample_trade(scenario, rng):
    """
    Sample a single trade from a scenario.

    v14 model: first determine if trade is Edge Fade or Playbook,
    then apply source-specific WR and P&L distributions.

    Returns (points_per_contract, grade, stop_distance_pts)
    """
    # Determine if Edge Fade or Playbook trade
    is_edge_fade = rng.random() < scenario['edge_fade_frac']

    # Determine setup grade
    if is_edge_fade:
        grade = 'B'  # Edge Fade is always Type B
        stop_dist = scenario['avg_stop_b']
        win_rate = scenario['win_rate_edge_fade']
    else:
        grade = 'A' if rng.random() < scenario['type_a_frac'] / (1 - scenario['edge_fade_frac'] + 1e-9) else 'B'
        stop_dist = scenario['avg_stop_a'] if grade == 'A' else scenario['avg_stop_b']
        win_rate = scenario['win_rate_playbook']

    # Win or lose
    if rng.random() < win_rate:
        if is_edge_fade and scenario['edge_fade_winner_pts']:
            pts = rng.choice(scenario['edge_fade_winner_pts']) * scenario['winner_haircut']
        elif not is_edge_fade and scenario['playbook_winner_pts']:
            pts = rng.choice(scenario['playbook_winner_pts']) * scenario['winner_haircut']
        else:
            pts = rng.choice(scenario['all_winner_pts']) * scenario['winner_haircut']
    else:
        if is_edge_fade:
            pts = rng.choice(scenario['edge_fade_loss_pts'])
        else:
            loss_pts = scenario['playbook_loss_pts']
            if isinstance(loss_pts, list) and len(loss_pts) == 2 and all(isinstance(v, (int, float)) for v in loss_pts):
                # Range format [-hi, -lo]
                pts = rng.uniform(loss_pts[0], loss_pts[1])
            else:
                pts = rng.choice(loss_pts) if loss_pts else -50.0

    return pts, grade, stop_dist


# ============================================================================
#  PIPELINE SIMULATION
# ============================================================================

def _execute_sim_trade(pipeline, scenario, rng):
    """Execute a single simulated trade through the pipeline."""
    pts, grade, stop_dist = sample_trade(scenario, rng)

    # Map grade to strategy/setup for the pipeline
    if grade == 'A':
        strategy_name = 'B-Day'
        setup_type = 'B_DAY_IBL_FADE'
        confidence = 'high'
    else:
        strategy_name = 'Edge Fade'
        setup_type = 'EDGE_TO_MID'
        confidence = 'medium'

    pipeline.on_signal(
        strategy_name=strategy_name,
        setup_type=setup_type,
        confidence=confidence,
        stop_distance_pts=stop_dist,
        trade_points=pts,
    )


def run_pipeline_sim(scenario, mode, n_sims=rules.MONTE_CARLO_RUNS,
                     sim_days=rules.SIM_DURATION_DAYS, seed=42,
                     eval_type_a_risk=None, eval_type_b_risk=None):
    """
    Run Monte Carlo simulation of the full pipeline.

    Args:
        scenario: Performance scenario dict
        mode: 'rotation' or 'copy_trade'
        n_sims: Number of Monte Carlo runs
        sim_days: Trading days to simulate
        seed: Random seed
        eval_type_a_risk: Override Type A eval risk
        eval_type_b_risk: Override Type B eval risk

    Returns:
        Dict of aggregated results with milestone snapshots
    """
    rng = np.random.default_rng(seed)

    # Collect results across all sims
    milestone_data = defaultdict(list)  # day -> list of snapshot dicts
    final_results = []

    for sim_idx in range(n_sims):
        pipeline = PropPipeline(
            mode=mode, max_funded=rules.MAX_FUNDED_ACCOUNTS,
            eval_type_a_risk=eval_type_a_risk,
            eval_type_b_risk=eval_type_b_risk,
        )
        pipeline.start_new_eval(day=0)

        for day in range(1, sim_days + 1):
            # Roll for trade occurrence (61.3% per session in v14)
            if rng.random() < rules.TRADE_FREQUENCY:
                # First trade of the day
                _execute_sim_trade(pipeline, scenario, rng)

                # Roll for second trade (Edge Fade cooldown allows ~10% multi-trade days)
                if rng.random() < rules.MULTI_TRADE_PROBABILITY:
                    _execute_sim_trade(pipeline, scenario, rng)

            # End of day
            pipeline.process_eod(day)

        # Collect final results
        summary = pipeline.get_final_summary()
        final_results.append(summary)

        # Collect milestone data
        for mday, snap in summary['milestones'].items():
            milestone_data[mday].append(snap)

    return _aggregate_results(final_results, milestone_data, mode)


def _aggregate_results(final_results, milestone_data, mode):
    """Aggregate Monte Carlo results into summary statistics."""
    agg = {
        'mode': mode,
        'n_sims': len(final_results),
        'milestones': {},
        'final': {},
    }

    # Final summary stats
    funded_active = [r['funded_active'] for r in final_results]
    net_incomes = [r['net_income'] for r in final_results]
    eval_costs = [r['total_eval_cost'] for r in final_results]
    total_payouts = [r['total_payouts'] for r in final_results]
    reset_rates = [r['reset_rate'] for r in final_results]
    evals_started = [r['evals_started'] for r in final_results]
    evals_passed = [r['evals_passed'] for r in final_results]
    funded_blown = [r['funded_blown'] for r in final_results]

    # Conditional: sims that passed at least 1 eval
    passed_sims = [r for r in final_results if r['evals_passed'] > 0]
    passed_nets = [r['net_income'] for r in passed_sims]
    passed_payouts = [r['total_payouts'] for r in passed_sims]
    passed_funded = [r['funded_active'] for r in passed_sims]

    agg['final'] = {
        'funded_active': _stats(funded_active),
        'net_income': _stats(net_incomes),
        'eval_cost': _stats(eval_costs),
        'total_payouts': _stats(total_payouts),
        'reset_rate': _stats(reset_rates),
        'evals_started': _stats(evals_started),
        'evals_passed': _stats(evals_passed),
        'funded_blown': _stats(funded_blown),
        'profitable_pct': sum(1 for n in net_incomes if n > 0) / len(net_incomes) * 100,
        'eval_pass_pct': len(passed_sims) / len(final_results) * 100,
        # Conditional stats (IF you pass at least 1 eval)
        'cond_net_income': _stats(passed_nets) if passed_nets else _stats([0]),
        'cond_payouts': _stats(passed_payouts) if passed_payouts else _stats([0]),
        'cond_funded': _stats(passed_funded) if passed_funded else _stats([0]),
    }

    # Milestone stats
    for mday, snaps in milestone_data.items():
        funded_counts = [s['funded_active'] for s in snaps]
        nets = [s['net_income'] for s in snaps]
        eval_c = [s['total_eval_cost'] for s in snaps]
        pay_t = [s['total_payouts'] for s in snaps]
        resets = [s['reset_rate'] for s in snaps]
        evals_s = [s['evals_started'] for s in snaps]
        evals_p = [s['evals_passed'] for s in snaps]

        # Conditional for milestones too
        passed_at_milestone = [s for s in snaps if s['evals_passed'] > 0]
        cond_nets = [s['net_income'] for s in passed_at_milestone]

        agg['milestones'][mday] = {
            'month': snaps[0]['month'],
            'funded_active': _stats(funded_counts),
            'net_income': _stats(nets),
            'eval_cost': _stats(eval_c),
            'total_payouts': _stats(pay_t),
            'reset_rate': _stats(resets),
            'evals_started': _stats(evals_s),
            'evals_passed': _stats(evals_p),
            'profitable_pct': sum(1 for n in nets if n > 0) / len(nets) * 100 if nets else 0,
            'eval_pass_pct': len(passed_at_milestone) / len(snaps) * 100 if snaps else 0,
            'cond_net_income': _stats(cond_nets) if cond_nets else _stats([0]),
        }

    return agg


def _stats(values):
    """Compute summary stats for a list of values."""
    if not values:
        return {'median': 0, 'mean': 0, 'p25': 0, 'p75': 0, 'p10': 0, 'p90': 0}
    arr = np.array(values)
    return {
        'median': float(np.median(arr)),
        'mean': float(np.mean(arr)),
        'p25': float(np.percentile(arr, 25)),
        'p75': float(np.percentile(arr, 75)),
        'p10': float(np.percentile(arr, 10)),
        'p90': float(np.percentile(arr, 90)),
    }


# ============================================================================
#  REPORTING
# ============================================================================

def print_results(all_results):
    """Print comparison tables for all scenarios and modes."""
    print("\n" + "=" * 120)
    print("  AGGRESSIVE PROP FIRM PIPELINE - MONTE CARLO SIMULATION")
    print(f"  Budget: {rules.MAX_FUNDED_ACCOUNTS} account slots x {rules.MAX_RESETS_PER_SLOT} resets = "
          f"{rules.MAX_TOTAL_EVAL_ATTEMPTS} max eval attempts")
    print(f"  Eval: Type A ${rules.EVAL_TYPE_A_RISK:,.0f} risk, Type B ${rules.EVAL_TYPE_B_RISK:,.0f} risk")
    print(f"  Funded: Phase 1 ${rules.FUNDED_PHASE1_RISK:,.0f}, Phase 2 ${rules.FUNDED_PHASE2_RISK:,.0f}, "
          f"Phase 3 ${rules.FUNDED_PHASE3_TYPE_B_RISK:,.0f}/${rules.FUNDED_PHASE3_TYPE_A_RISK:,.0f}")
    print(f"  Simulations: {rules.MONTE_CARLO_RUNS:,} runs x {rules.SIM_DURATION_DAYS} days (2 years)")
    print("=" * 120)

    for scenario_name, modes in all_results.items():
        _print_scenario(scenario_name, modes)

    # Cross-scenario comparison
    _print_cross_scenario_comparison(all_results)


def _print_scenario(scenario_name, modes):
    """Print detailed results for one scenario."""
    print(f"\n{'='*120}")
    print(f"  SCENARIO: {scenario_name}")
    print(f"{'='*120}")

    for mode_name in ['rotation', 'copy_trade']:
        mode_data = modes[mode_name]
        label = 'ROTATION' if mode_name == 'rotation' else 'COPY TRADE'
        final = mode_data['final']
        milestones = mode_data['milestones']

        print(f"\n  --- {label} MODE ---")

        # Summary box
        print(f"  Eval pass rate: {final['eval_pass_pct']:.1f}% of sims pass at least 1 eval")
        print(f"  Evals started (mean): {final['evals_started']['mean']:.1f}, "
              f"passed: {final['evals_passed']['mean']:.1f}, "
              f"blown: {final['evals_started']['mean'] - final['evals_passed']['mean']:.1f}")
        print(f"  Funded blown (mean): {final['funded_blown']['mean']:.1f}")
        print(f"  Profitable sims: {final['profitable_pct']:.1f}%")
        print()

        # Milestone table header
        sorted_mdays = sorted(milestones.keys())
        months = [milestones[d]['month'] for d in sorted_mdays]
        header = f"  {'Metric':<30s}"
        for m in months:
            header += f" {m:>4d}mo"
        header += f" {'2yr':>8s}"
        print(header)
        print("  " + "-" * (30 + 7 * len(months) + 10))

        # ALL SIMS - Unconditional
        print("  [All Sims]")
        _print_row('  Funded Active (mean)', sorted_mdays, milestones, final,
                    'funded_active', stat='mean', fmt='.1f')
        _print_row('  Evals Passed (mean)', sorted_mdays, milestones, final,
                    'evals_passed', stat='mean', fmt='.1f')
        _print_row('  Eval Cost (mean)', sorted_mdays, milestones, final,
                    'eval_cost', stat='mean', fmt='$')
        _print_row('  Total Payouts (mean)', sorted_mdays, milestones, final,
                    'total_payouts', stat='mean', fmt='$')
        _print_row('  Net Income (mean)', sorted_mdays, milestones, final,
                    'net_income', stat='mean', fmt='$')
        _print_row('  % Profitable', sorted_mdays, milestones, final,
                    'profitable_pct', stat='direct', fmt='pct')
        _print_row('  % Passed >= 1 Eval', sorted_mdays, milestones, final,
                    'eval_pass_pct', stat='direct', fmt='pct')
        print()

        # CONDITIONAL - Only sims that passed at least 1 eval
        print("  [If Passed Eval (conditional)]")
        _print_row('  Net Income (mean)', sorted_mdays, milestones, final,
                    'cond_net_income', stat='mean', fmt='$')
        _print_row('  Net Income (median)', sorted_mdays, milestones, final,
                    'cond_net_income', stat='median', fmt='$')
        _print_row('  Net Income (p25)', sorted_mdays, milestones, final,
                    'cond_net_income', stat='p25', fmt='$')
        _print_row('  Net Income (p75)', sorted_mdays, milestones, final,
                    'cond_net_income', stat='p75', fmt='$')
        print()

        # Distribution highlights
        print(f"  DISTRIBUTION (2-year final):")
        print(f"    All sims:       mean ${final['net_income']['mean']:>10,.0f} | "
              f"median ${final['net_income']['median']:>10,.0f} | "
              f"p10 ${final['net_income']['p10']:>10,.0f} | "
              f"p90 ${final['net_income']['p90']:>10,.0f}")
        print(f"    If passed eval: mean ${final['cond_net_income']['mean']:>10,.0f} | "
              f"median ${final['cond_net_income']['median']:>10,.0f} | "
              f"p10 ${final['cond_net_income']['p10']:>10,.0f} | "
              f"p90 ${final['cond_net_income']['p90']:>10,.0f}")
        print()


def _print_row(label, sorted_mdays, milestones, final, metric, stat='mean', fmt='.1f'):
    """Print one row of the milestone table."""
    print(f"{label:<30s}", end='')

    for mday in sorted_mdays:
        snap = milestones[mday]
        val = _get_val(snap, metric, stat)
        print(f" {_fmt(val, fmt):>6s}", end='')

    # Final column
    val = _get_val(final, metric, stat)
    print(f" {_fmt(val, fmt):>8s}")


def _get_val(data, metric, stat):
    """Extract a value from milestone/final data."""
    if stat == 'direct':
        return data.get(metric, 0)
    elif isinstance(data.get(metric), dict):
        return data[metric].get(stat, 0)
    else:
        return data.get(metric, 0)


def _fmt(val, fmt):
    if fmt == '$':
        if abs(val) >= 1000:
            return f"${val:,.0f}"
        return f"${val:.0f}"
    elif fmt == 'pct':
        return f"{val:.0f}%"
    elif fmt == '.1f':
        return f"{val:.1f}"
    elif fmt == '.0f':
        return f"{val:.0f}"
    else:
        return f"{val}"


def _print_cross_scenario_comparison(all_results):
    """Print cross-scenario comparison focusing on key metrics."""
    print("\n" + "=" * 120)
    print("  CROSS-SCENARIO COMPARISON: ROTATION vs COPY TRADE")
    print("=" * 120)

    print(f"\n  {'Scenario':<28s} {'Mode':<12s} {'Pass%':>6s} {'Funded':>7s} "
          f"{'Net (all)':>12s} {'Net (cond)':>12s} {'Payout':>10s} {'Profit%':>8s}")
    print("  " + "-" * 100)

    for scenario_name, modes in all_results.items():
        for mode_name in ['rotation', 'copy_trade']:
            f = modes[mode_name]['final']
            label = 'Rotate' if mode_name == 'rotation' else 'Copy'
            print(f"  {scenario_name:<28s} {label:<12s} "
                  f"{f['eval_pass_pct']:>5.0f}% "
                  f"{f['funded_active']['mean']:>6.1f} "
                  f"${f['net_income']['mean']:>10,.0f} "
                  f"${f['cond_net_income']['mean']:>10,.0f} "
                  f"${f['total_payouts']['mean']:>8,.0f} "
                  f"{f['profitable_pct']:>6.1f}%")
        print()

    # Monthly income at realistic
    print("\n  MONTHLY INCOME ESTIMATE (Realistic Live, if eval passes):")
    realistic = all_results.get('Realistic Live (50% WR)')
    if realistic:
        for mode_name, label in [('rotation', 'Rotation'), ('copy_trade', 'Copy Trade')]:
            f = realistic[mode_name]['final']
            annual = f['cond_net_income']['mean']
            monthly = annual / 24  # 2-year sim
            print(f"    {label}: ${monthly:,.0f}/month (${annual:,.0f} over 2 years)")

    # Recommendation
    print(f"\n  RECOMMENDATION (v14 Edge Fade + Playbook):")
    print(f"  1. v14 trades at ~14/month (3.2x more than v13's 4/month)")
    print(f"  2. Blended WR ~58% backtest; target 50%+ live for positive expectancy")
    print(f"  3. Edge Fade fills 76% of sessions that Playbook misses (b_day/neutral)")
    print(f"  4. Start with Copy Trade for faster payout accumulation")
    print(f"  5. Budget 5 evals + 10 resets = $3,285 max eval cost")
    print(f"  6. Higher trade freq = faster eval pass but also faster DD accumulation")


# ============================================================================
#  MAIN
# ============================================================================

def main():
    print("Loading trade data from v14 backtest (Edge Fade + Playbook)...")
    session_trades = load_trade_data()
    trade_pool = build_trade_pool(session_trades)

    # Print trade pool summary
    type_a = [t for t in trade_pool if t['grade'] == 'A']
    type_b = [t for t in trade_pool if t['grade'] == 'B']
    playbook = [t for t in trade_pool if t['source'] == 'playbook']
    edge_fade = [t for t in trade_pool if t['source'] == 'edge_fade']
    print(f"\nTrade pool: {len(trade_pool)} deduplicated trades")
    print(f"  Playbook (Trend/P-Day/B-Day): {len(playbook)} trades, "
          f"WR={sum(1 for t in playbook if t['is_winner'])/len(playbook)*100:.0f}%")
    print(f"  Edge Fade (EDGE_TO_MID):      {len(edge_fade)} trades, "
          f"WR={sum(1 for t in edge_fade if t['is_winner'])/len(edge_fade)*100:.0f}%")
    print(f"  Type A (high conviction): {len(type_a)} trades "
          f"({len(type_a)/len(trade_pool)*100:.0f}%)")
    for t in type_a:
        print(f"    - {t['strategy_name']}/{t['setup_type']}: "
              f"{t['points']:+.1f} pts, stop={t['stop_distance_pts']:.0f} pts")
    print(f"  Type B (standard): {len(type_b)} trades "
          f"({len(type_b)/len(trade_pool)*100:.0f}%)")
    eff_trades_per_month = (rules.TRADE_FREQUENCY * (1 + rules.MULTI_TRADE_PROBABILITY)
                            * rules.TRADING_DAYS_PER_MONTH)
    print(f"\nTrade frequency: {rules.TRADE_FREQUENCY:.1%} sessions/day "
          f"(+{rules.MULTI_TRADE_PROBABILITY:.0%} multi-trade)")
    print(f"Effective: ~{eff_trades_per_month:.0f} trades/month")

    # Build scenarios
    scenarios = build_scenarios(trade_pool)

    # ================================================================
    #  PART 1: Sizing scheme comparison at Realistic 50% WR
    # ================================================================
    print("\n" + "=" * 120)
    print("  PART 1: EVAL SIZING SCHEME COMPARISON (Realistic Live 50% WR, Copy Trade)")
    print("=" * 120)

    realistic = scenarios['Realistic Live (50% WR)']
    sizing_results = {}

    for scheme_name, scheme in rules.EVAL_SIZING_SCHEMES.items():
        print(f"\n  Testing: {scheme_name} (A=${scheme['type_a']:,}, B=${scheme['type_b']:,})...")
        result = run_pipeline_sim(
            realistic, 'copy_trade',
            n_sims=rules.MONTE_CARLO_RUNS,
            sim_days=rules.SIM_DURATION_DAYS,
            eval_type_a_risk=scheme['type_a'],
            eval_type_b_risk=scheme['type_b'],
        )
        sizing_results[scheme_name] = result

    # Print sizing comparison
    print(f"\n{'='*120}")
    print(f"  EVAL SIZING COMPARISON (Realistic 50% WR, Copy Trade, 2 years)")
    print(f"{'='*120}")
    print(f"\n  {'Scheme':<32s} {'Pass%':>6s} {'Mean$':>8s} {'Funded':>7s} {'Reset%':>7s} "
          f"{'EvCost':>8s} {'Payout':>9s} {'Net$':>9s} {'Cond$':>9s} {'Profit%':>8s}")
    print("  " + "-" * 105)

    for scheme_name, result in sizing_results.items():
        f = result['final']
        print(f"  {scheme_name:<32s} "
              f"{f['eval_pass_pct']:>5.0f}% "
              f"${f['net_income']['mean']:>6,.0f} "
              f"{f['funded_active']['mean']:>6.1f} "
              f"{f['reset_rate']['mean']:>5.0%} "
              f"${f['eval_cost']['mean']:>6,.0f} "
              f"${f['total_payouts']['mean']:>7,.0f} "
              f"${f['net_income']['mean']:>7,.0f} "
              f"${f['cond_net_income']['mean']:>7,.0f} "
              f"{f['profitable_pct']:>6.1f}%")

    # Find optimal scheme
    best_scheme = max(sizing_results.items(),
                      key=lambda x: x[1]['final']['net_income']['mean'])
    print(f"\n  OPTIMAL SCHEME: {best_scheme[0]}")
    print(f"  Mean net income: ${best_scheme[1]['final']['net_income']['mean']:,.0f}")

    # ================================================================
    #  PART 2: Full scenario comparison with optimal sizing
    # ================================================================
    print(f"\n\n{'='*120}")
    print(f"  PART 2: FULL SCENARIO x MODE COMPARISON")
    print(f"  (Using optimal eval sizing from Part 1)")
    print(f"{'='*120}")

    # Find the best scheme
    best_scheme_name = best_scheme[0]
    best_a = rules.EVAL_SIZING_SCHEMES[best_scheme_name]['type_a']
    best_b = rules.EVAL_SIZING_SCHEMES[best_scheme_name]['type_b']
    print(f"  Eval sizing: A=${best_a:,}, B=${best_b:,}")

    all_results = {}
    for scenario_name, scenario in scenarios.items():
        print(f"\nRunning {scenario_name}...")
        modes = {}
        for mode in ['rotation', 'copy_trade']:
            print(f"  Mode: {mode}... ", end='', flush=True)
            modes[mode] = run_pipeline_sim(
                scenario, mode,
                n_sims=rules.MONTE_CARLO_RUNS,
                sim_days=rules.SIM_DURATION_DAYS,
                eval_type_a_risk=best_a,
                eval_type_b_risk=best_b,
            )
            print("done")
        all_results[scenario_name] = modes

    # Print full results
    print_results(all_results)

    return all_results, sizing_results


if __name__ == '__main__':
    main()
