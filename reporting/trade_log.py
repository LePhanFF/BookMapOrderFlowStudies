"""
Trade log export: per-trade CSV for analysis in Excel/Google Sheets.
"""

import csv
from pathlib import Path
from typing import List
from engine.trade import Trade


TRADE_LOG_COLUMNS = [
    'strategy_name', 'setup_type', 'day_type', 'trend_strength',
    'session_date', 'entry_time', 'exit_time', 'bars_held',
    'direction', 'contracts',
    'signal_price', 'entry_price', 'exit_price', 'stop_price', 'target_price',
    'gross_pnl', 'commission', 'slippage_cost', 'net_pnl',
    'exit_reason', 'r_multiple', 'is_winner',
]


def export_trade_log(trades: List[Trade], filepath: str = 'trade_log.csv') -> str:
    """
    Export trades to a CSV file.

    Args:
        trades: List of Trade objects.
        filepath: Output CSV path.

    Returns:
        Absolute path to the written file.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=TRADE_LOG_COLUMNS)
        writer.writeheader()

        for t in trades:
            row = {
                'strategy_name': t.strategy_name,
                'setup_type': t.setup_type,
                'day_type': t.day_type,
                'trend_strength': t.trend_strength,
                'session_date': t.session_date,
                'entry_time': str(t.entry_time) if t.entry_time else '',
                'exit_time': str(t.exit_time) if t.exit_time else '',
                'bars_held': t.bars_held,
                'direction': t.direction,
                'contracts': t.contracts,
                'signal_price': round(t.signal_price, 2),
                'entry_price': round(t.entry_price, 2),
                'exit_price': round(t.exit_price, 2),
                'stop_price': round(t.stop_price, 2),
                'target_price': round(t.target_price, 2),
                'gross_pnl': round(t.gross_pnl, 2),
                'commission': round(t.commission, 2),
                'slippage_cost': round(t.slippage_cost, 2),
                'net_pnl': round(t.net_pnl, 2),
                'exit_reason': t.exit_reason,
                'r_multiple': round(t.r_multiple, 2),
                'is_winner': 1 if t.is_winner else 0,
            }
            writer.writerow(row)

    abs_path = str(path.resolve())
    print(f"Trade log exported: {abs_path} ({len(trades)} trades)")
    return abs_path


def print_trade_summary(trades: List[Trade], max_rows: int = 50) -> None:
    """Print a summary table of recent trades to stdout."""
    if not trades:
        print("No trades to display.")
        return

    print(f"\n{'-'*120}")
    print(f"{'#':>4}  {'Strategy':<20} {'Setup':<22} {'Dir':>5} {'Entry':>10} "
          f"{'Exit':>10} {'Net P&L':>10} {'R':>5} {'Exit':>12} {'Date':>12}")
    print(f"{'-'*120}")

    display = trades[-max_rows:] if len(trades) > max_rows else trades

    for i, t in enumerate(display, 1):
        pnl_str = f"${t.net_pnl:>8,.2f}"
        marker = '+' if t.is_winner else '-'
        print(f"{i:>4}  {t.strategy_name:<20} {t.setup_type:<22} {t.direction:>5} "
              f"{t.entry_price:>10.2f} {t.exit_price:>10.2f} "
              f"{marker}{pnl_str:>9} {t.r_multiple:>5.2f}R "
              f"{t.exit_reason:>12} {t.session_date:>12}")

    print(f"{'-'*120}")
    if len(trades) > max_rows:
        print(f"  (showing last {max_rows} of {len(trades)} trades)")
