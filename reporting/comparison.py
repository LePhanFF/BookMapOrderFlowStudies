"""
Strategy comparison tables.
Shows side-by-side metrics for each strategy in the backtest.
"""

from typing import List, Dict
from engine.trade import Trade
from reporting.metrics import compute_metrics


def compare_strategies(
    trades: List[Trade],
    initial_equity: float = 150_000,
) -> Dict[str, Dict]:
    """
    Group trades by strategy and compute metrics for each.

    Args:
        trades: List of all trades from the backtest.
        initial_equity: Starting equity for metric computation.

    Returns:
        Dict mapping strategy_name -> metrics dict.
    """
    grouped = {}
    for t in trades:
        grouped.setdefault(t.strategy_name, []).append(t)

    results = {}
    for name, strades in sorted(grouped.items()):
        results[name] = compute_metrics(strades, initial_equity)

    return results


def compare_by_day_type(
    trades: List[Trade],
    initial_equity: float = 150_000,
) -> Dict[str, Dict]:
    """
    Group trades by day_type and compute metrics for each.
    """
    grouped = {}
    for t in trades:
        grouped.setdefault(t.day_type, []).append(t)

    results = {}
    for name, strades in sorted(grouped.items()):
        results[name] = compute_metrics(strades, initial_equity)

    return results


def compare_by_setup(
    trades: List[Trade],
    initial_equity: float = 150_000,
) -> Dict[str, Dict]:
    """
    Group trades by setup_type and compute metrics for each.
    """
    grouped = {}
    for t in trades:
        grouped.setdefault(t.setup_type, []).append(t)

    results = {}
    for name, strades in sorted(grouped.items()):
        results[name] = compute_metrics(strades, initial_equity)

    return results


def print_comparison_table(
    comparison: Dict[str, Dict],
    title: str = "Strategy Comparison",
) -> None:
    """Pretty-print a comparison table."""
    if not comparison:
        print("No data to compare.")
        return

    print(f"\n{'='*110}")
    print(f"  {title}")
    print(f"{'='*110}")

    header = (f"{'Name':<25} {'Trades':>6} {'WR%':>6} {'PF':>6} "
              f"{'Net P&L':>12} {'AvgWin':>10} {'AvgLoss':>10} "
              f"{'Sharpe':>7} {'MaxDD%':>7} {'Expect':>10}")
    print(header)
    print(f"{'-'*110}")

    for name, m in comparison.items():
        print(f"{name:<25} {m['total_trades']:>6d} "
              f"{m['win_rate']:>5.1f}% "
              f"{m['profit_factor']:>6.2f} "
              f"${m['total_net_pnl']:>10,.2f} "
              f"${m['avg_win']:>8,.2f} "
              f"${m['avg_loss']:>8,.2f} "
              f"{m['sharpe_ratio']:>7.2f} "
              f"{m['max_drawdown_pct']:>6.2f}% "
              f"${m['expectancy_per_trade']:>8,.2f}")

    # Totals row
    all_trades = sum(m['total_trades'] for m in comparison.values())
    all_pnl = sum(m['total_net_pnl'] for m in comparison.values())
    print(f"{'-'*110}")
    print(f"{'TOTAL':<25} {all_trades:>6d} {'':>6} {'':>6} "
          f"${all_pnl:>10,.2f}")
    print(f"{'='*110}\n")
