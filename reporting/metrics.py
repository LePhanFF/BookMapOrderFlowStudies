"""
Performance metrics for backtest results.
Sharpe, Sortino, Calmar, max DD, profit factor, expectancy, and more.
"""

import numpy as np
from typing import List, Dict, Any
from engine.trade import Trade


def compute_metrics(trades: List[Trade], initial_equity: float = 150_000) -> Dict[str, Any]:
    """
    Compute comprehensive performance metrics from a list of trades.

    All P&L values are in dollars (already computed by ExecutionModel).

    Args:
        trades: List of completed Trade objects.
        initial_equity: Starting account balance.

    Returns:
        Dict of named metrics.
    """
    if not trades:
        return _empty_metrics()

    pnl_list = [t.net_pnl for t in trades]
    gross_list = [t.gross_pnl for t in trades]
    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl <= 0]

    total_pnl = sum(pnl_list)
    total_gross = sum(gross_list)
    total_commission = sum(t.commission for t in trades)
    total_slippage = sum(t.slippage_cost for t in trades)

    win_count = len(wins)
    loss_count = len(losses)
    win_rate = win_count / len(trades) * 100

    avg_win = np.mean([t.net_pnl for t in wins]) if wins else 0.0
    avg_loss = np.mean([t.net_pnl for t in losses]) if losses else 0.0
    largest_win = max((t.net_pnl for t in wins), default=0.0)
    largest_loss = min((t.net_pnl for t in losses), default=0.0)

    gross_profit = sum(t.net_pnl for t in wins)
    gross_loss = abs(sum(t.net_pnl for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    expectancy = total_pnl / len(trades)

    # Payoff ratio (avg win / avg loss)
    payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

    # Equity curve for drawdown and ratio calculations
    equity_curve = _build_equity_curve(trades, initial_equity)
    max_dd = _max_drawdown(equity_curve)
    max_dd_pct = (max_dd / max(equity_curve)) * 100 if max(equity_curve) > 0 else 0.0

    # Daily P&L for Sharpe/Sortino
    daily_pnl = _daily_pnl(trades)
    sharpe = _sharpe_ratio(daily_pnl) if len(daily_pnl) > 1 else 0.0
    sortino = _sortino_ratio(daily_pnl) if len(daily_pnl) > 1 else 0.0

    # Calmar = annualized return / max drawdown
    total_days = len(daily_pnl)
    annualized_return = (total_pnl / initial_equity) * (252 / max(total_days, 1))
    calmar = annualized_return / (max_dd_pct / 100) if max_dd_pct > 0 else float('inf')

    # Consecutive stats
    max_consec_wins, max_consec_losses = _consecutive_streaks(trades)

    # Average bars held
    avg_bars = np.mean([t.bars_held for t in trades])
    avg_bars_win = np.mean([t.bars_held for t in wins]) if wins else 0
    avg_bars_loss = np.mean([t.bars_held for t in losses]) if losses else 0

    # R-multiple stats
    r_multiples = [t.r_multiple for t in trades if t.risk_points > 0]
    avg_r = np.mean(r_multiples) if r_multiples else 0.0
    expectancy_r = avg_r

    # Exit reason breakdown
    exit_reasons = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    return {
        # Counts
        'total_trades': len(trades),
        'winning_trades': win_count,
        'losing_trades': loss_count,
        'win_rate': round(win_rate, 2),

        # P&L
        'total_net_pnl': round(total_pnl, 2),
        'total_gross_pnl': round(total_gross, 2),
        'total_commission': round(total_commission, 2),
        'total_slippage': round(total_slippage, 2),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'largest_win': round(largest_win, 2),
        'largest_loss': round(largest_loss, 2),
        'expectancy_per_trade': round(expectancy, 2),

        # Ratios
        'profit_factor': round(profit_factor, 2),
        'payoff_ratio': round(payoff_ratio, 2),
        'sharpe_ratio': round(sharpe, 2),
        'sortino_ratio': round(sortino, 2),
        'calmar_ratio': round(calmar, 2),

        # Drawdown
        'max_drawdown': round(max_dd, 2),
        'max_drawdown_pct': round(max_dd_pct, 2),

        # R-multiples
        'avg_r_multiple': round(avg_r, 2),
        'expectancy_r': round(expectancy_r, 2),

        # Streaks
        'max_consecutive_wins': max_consec_wins,
        'max_consecutive_losses': max_consec_losses,

        # Duration
        'avg_bars_held': round(avg_bars, 1),
        'avg_bars_win': round(avg_bars_win, 1),
        'avg_bars_loss': round(avg_bars_loss, 1),

        # Equity
        'initial_equity': initial_equity,
        'final_equity': round(initial_equity + total_pnl, 2),
        'total_return_pct': round(total_pnl / initial_equity * 100, 2),
        'annualized_return_pct': round(annualized_return * 100, 2),

        # Breakdown
        'exit_reasons': exit_reasons,
    }


def _empty_metrics() -> Dict[str, Any]:
    return {
        'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
        'win_rate': 0.0, 'total_net_pnl': 0.0, 'total_gross_pnl': 0.0,
        'total_commission': 0.0, 'total_slippage': 0.0,
        'avg_win': 0.0, 'avg_loss': 0.0, 'largest_win': 0.0, 'largest_loss': 0.0,
        'expectancy_per_trade': 0.0, 'profit_factor': 0.0, 'payoff_ratio': 0.0,
        'sharpe_ratio': 0.0, 'sortino_ratio': 0.0, 'calmar_ratio': 0.0,
        'max_drawdown': 0.0, 'max_drawdown_pct': 0.0,
        'avg_r_multiple': 0.0, 'expectancy_r': 0.0,
        'max_consecutive_wins': 0, 'max_consecutive_losses': 0,
        'avg_bars_held': 0.0, 'avg_bars_win': 0.0, 'avg_bars_loss': 0.0,
        'initial_equity': 0, 'final_equity': 0, 'total_return_pct': 0.0,
        'annualized_return_pct': 0.0, 'exit_reasons': {},
    }


def _build_equity_curve(trades: List[Trade], initial: float) -> List[float]:
    curve = [initial]
    for t in trades:
        curve.append(curve[-1] + t.net_pnl)
    return curve


def _max_drawdown(equity_curve: List[float]) -> float:
    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _daily_pnl(trades: List[Trade]) -> List[float]:
    """Group trades by session_date, sum daily P&L."""
    daily = {}
    for t in trades:
        key = t.session_date
        daily[key] = daily.get(key, 0.0) + t.net_pnl
    return list(daily.values())


def _sharpe_ratio(daily_pnl: List[float], risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio from daily P&L."""
    if len(daily_pnl) < 2:
        return 0.0
    arr = np.array(daily_pnl)
    mean_ret = arr.mean() - risk_free_rate / 252
    std_ret = arr.std(ddof=1)
    if std_ret == 0:
        return 0.0
    return (mean_ret / std_ret) * np.sqrt(252)


def _sortino_ratio(daily_pnl: List[float], risk_free_rate: float = 0.0) -> float:
    """Annualized Sortino ratio (downside deviation only)."""
    if len(daily_pnl) < 2:
        return 0.0
    arr = np.array(daily_pnl)
    mean_ret = arr.mean() - risk_free_rate / 252
    downside = arr[arr < 0]
    if len(downside) == 0:
        return float('inf')
    downside_std = downside.std(ddof=1)
    if downside_std == 0:
        return 0.0
    return (mean_ret / downside_std) * np.sqrt(252)


def _consecutive_streaks(trades: List[Trade]) -> tuple:
    """Return (max consecutive wins, max consecutive losses)."""
    max_wins = max_losses = 0
    cur_wins = cur_losses = 0
    for t in trades:
        if t.net_pnl > 0:
            cur_wins += 1
            cur_losses = 0
            max_wins = max(max_wins, cur_wins)
        else:
            cur_losses += 1
            cur_wins = 0
            max_losses = max(max_losses, cur_losses)
    return max_wins, max_losses


def print_metrics(metrics: Dict[str, Any]) -> None:
    """Pretty-print a metrics dict."""
    print(f"\n{'='*70}")
    print(f"  PERFORMANCE REPORT")
    print(f"{'='*70}")

    print(f"\n--- Trade Summary ---")
    print(f"  Total Trades:        {metrics['total_trades']}")
    print(f"  Win Rate:            {metrics['win_rate']:.1f}%"
          f"  ({metrics['winning_trades']}W / {metrics['losing_trades']}L)")
    print(f"  Profit Factor:       {metrics['profit_factor']:.2f}")
    print(f"  Payoff Ratio:        {metrics['payoff_ratio']:.2f}")
    print(f"  Expectancy:          ${metrics['expectancy_per_trade']:,.2f} / trade")

    print(f"\n--- P&L ---")
    print(f"  Gross P&L:           ${metrics['total_gross_pnl']:>12,.2f}")
    print(f"  Commissions:         ${metrics['total_commission']:>12,.2f}")
    print(f"  Slippage:            ${metrics['total_slippage']:>12,.2f}")
    print(f"  Net P&L:             ${metrics['total_net_pnl']:>12,.2f}")

    print(f"\n--- Win/Loss ---")
    print(f"  Avg Win:             ${metrics['avg_win']:>10,.2f}")
    print(f"  Avg Loss:            ${metrics['avg_loss']:>10,.2f}")
    print(f"  Largest Win:         ${metrics['largest_win']:>10,.2f}")
    print(f"  Largest Loss:        ${metrics['largest_loss']:>10,.2f}")

    print(f"\n--- Risk Ratios ---")
    print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio:       {metrics['sortino_ratio']:.2f}")
    print(f"  Calmar Ratio:        {metrics['calmar_ratio']:.2f}")
    print(f"  Avg R-Multiple:      {metrics['avg_r_multiple']:.2f}")

    print(f"\n--- Drawdown ---")
    print(f"  Max Drawdown:        ${metrics['max_drawdown']:>10,.2f}")
    print(f"  Max Drawdown %:      {metrics['max_drawdown_pct']:.2f}%")

    print(f"\n--- Equity ---")
    print(f"  Initial:             ${metrics['initial_equity']:>12,.2f}")
    print(f"  Final:               ${metrics['final_equity']:>12,.2f}")
    print(f"  Total Return:        {metrics['total_return_pct']:.2f}%")
    print(f"  Annualized Return:   {metrics['annualized_return_pct']:.2f}%")

    print(f"\n--- Duration ---")
    print(f"  Avg Bars Held:       {metrics['avg_bars_held']:.1f}")
    print(f"  Avg Bars (Win):      {metrics['avg_bars_win']:.1f}")
    print(f"  Avg Bars (Loss):     {metrics['avg_bars_loss']:.1f}")

    print(f"\n--- Streaks ---")
    print(f"  Max Consec Wins:     {metrics['max_consecutive_wins']}")
    print(f"  Max Consec Losses:   {metrics['max_consecutive_losses']}")

    if metrics.get('exit_reasons'):
        print(f"\n--- Exit Reasons ---")
        for reason, count in sorted(metrics['exit_reasons'].items(),
                                     key=lambda x: -x[1]):
            pct = count / metrics['total_trades'] * 100
            print(f"  {reason:20s}: {count:4d}  ({pct:5.1f}%)")

    print(f"{'='*70}\n")
