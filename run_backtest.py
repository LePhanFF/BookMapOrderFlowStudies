"""
Main backtest entry point.

Usage:
    python run_backtest.py                    # NQ, all strategies, default filters
    python run_backtest.py --instrument MNQ   # Different instrument
    python run_backtest.py --no-filters       # Run without filters
    python run_backtest.py --strategies "Trend Day Bull,B-Day"
    python run_backtest.py --export           # Also export NinjaTrader strategy
"""

import sys
import argparse
from pathlib import Path
from datetime import time

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.instruments import get_instrument
from config.constants import DEFAULT_ACCOUNT_SIZE, DEFAULT_MAX_RISK_PER_TRADE
from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from engine.backtest import BacktestEngine
from engine.execution import ExecutionModel
from engine.position import PositionManager
from strategy import get_all_strategies, get_core_strategies, get_strategies_by_name
from filters.composite import CompositeFilter
from filters.time_filter import TimeFilter
from filters.volatility_filter import VolatilityFilter
from filters.trend_filter import TrendFilter
from filters.order_flow_filter import DeltaFilter, CVDFilter, VolumeFilter
from reporting.metrics import compute_metrics, print_metrics
from reporting.trade_log import export_trade_log, print_trade_summary
from reporting.comparison import (
    compare_strategies, compare_by_day_type, compare_by_setup,
    print_comparison_table,
)
from reporting.day_analyzer import analyze_sessions, print_day_analysis


def build_default_filters():
    """Build the default composable filter chain."""
    return CompositeFilter([
        TimeFilter(start=time(10, 30), end=time(15, 30)),
        VolatilityFilter(min_atr=5.0, max_atr=80.0),
    ])


def build_strict_filters():
    """Build a stricter filter chain with order flow requirements.

    Note: CVDFilter was REMOVED after deep order flow study (diagnostic_deep_orderflow.py)
    showed it HURTS performance — CVD > MA is 67% in winners but 100% in losers.
    B-Day entries naturally have CVD < MA (balance day), so CVDFilter kills B-Day trades.

    DeltaFilter(pctl>=60) and VolumeFilter(spike>=1.0) are the data-proven filters:
      - DeltaFilter(pctl>=60): 11/12 winners pass, 1/2 losers rejected → 91.7% WR
      - VolumeFilter(spike>=1.0): 12/12 winners pass, 1/2 losers rejected
    """
    return CompositeFilter([
        TimeFilter(start=time(10, 30), end=time(15, 0)),
        VolatilityFilter(min_atr=8.0, max_atr=60.0),
        DeltaFilter(min_percentile=60.0),
        VolumeFilter(min_spike=1.0),
    ])


def main():
    parser = argparse.ArgumentParser(description='Dalton Playbook Backtest Engine')
    parser.add_argument('--instrument', '-i', default='MNQ',
                        help='Instrument symbol (NQ, MNQ, ES, MES, YM, MYM)')
    parser.add_argument('--csv-dir', default=None,
                        help='Path to CSV directory (auto-discovered if omitted)')
    parser.add_argument('--strategies', '-s', default='core',
                        help='Comma-separated strategy names, "core" (default), or "all"')
    parser.add_argument('--no-filters', action='store_true',
                        help='Disable all filters')
    parser.add_argument('--strict-filters', action='store_true',
                        help='Use strict filters (incl. order flow)')
    parser.add_argument('--account-size', type=float, default=DEFAULT_ACCOUNT_SIZE,
                        help=f'Starting account size (default: ${DEFAULT_ACCOUNT_SIZE:,.0f})')
    parser.add_argument('--risk-per-trade', type=float, default=DEFAULT_MAX_RISK_PER_TRADE,
                        help=f'Max risk per trade in dollars (default: ${DEFAULT_MAX_RISK_PER_TRADE})')
    parser.add_argument('--max-contracts', type=int, default=30,
                        help='Max contracts per trade')
    parser.add_argument('--slippage-ticks', type=int, default=None,
                        help='Slippage ticks per side (default: instrument default)')
    parser.add_argument('--zero-cost', action='store_true',
                        help='Zero slippage and commission (for comparison)')
    parser.add_argument('--export', action='store_true',
                        help='Export NinjaTrader strategy after backtest')
    parser.add_argument('--trade-log', default='output/trade_log.csv',
                        help='Path for trade log CSV export')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress per-session output')
    parser.add_argument('--day-analysis', action='store_true',
                        help='Run day type confidence analysis report')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  DALTON PLAYBOOK BACKTEST ENGINE")
    print(f"{'='*70}\n")

    # --- 1. Load instrument ---
    # Map CSV data to the right instrument for pricing
    # NQ CSV can be used with MNQ point_value
    csv_symbol = args.instrument.upper()
    if csv_symbol.startswith('M'):
        csv_file_symbol = csv_symbol[1:]  # MNQ -> NQ, MES -> ES, MYM -> YM
    else:
        csv_file_symbol = csv_symbol

    instrument = get_instrument(args.instrument)
    print(f"Instrument: {instrument.symbol}")
    print(f"  Point value: ${instrument.point_value}")
    print(f"  Tick size: {instrument.tick_size}")
    print(f"  Commission: ${instrument.commission}/side")

    # --- 2. Load data ---
    print(f"\n--- Loading Data ---")
    csv_dir = Path(args.csv_dir) if args.csv_dir else None
    df = load_csv(csv_file_symbol, csv_dir)
    df = filter_rth(df)
    df = compute_all_features(df)

    # --- 3. Build strategies ---
    print(f"\n--- Strategies ---")
    if args.strategies.lower() == 'all':
        strategies = get_all_strategies()
    elif args.strategies.lower() == 'core':
        strategies = get_core_strategies()
    else:
        names = [n.strip() for n in args.strategies.split(',')]
        strategies = get_strategies_by_name(*names)

    for s in strategies:
        print(f"  - {s.name} ({', '.join(s.applicable_day_types)})")

    # --- 4. Build filters ---
    if args.no_filters:
        filters = None
        print(f"\nFilters: NONE (disabled)")
    elif args.strict_filters:
        filters = build_strict_filters()
        print(f"\nFilters: STRICT ({len(filters._filters)} filters)")
    else:
        filters = build_default_filters()
        print(f"\nFilters: DEFAULT ({len(filters._filters)} filters)")

    # --- 5. Build execution model ---
    if args.zero_cost:
        execution = ExecutionModel(instrument, slippage_ticks=0, commission_per_side=0.0)
        print(f"Execution: ZERO COST (no slippage, no commission)")
    else:
        slippage = args.slippage_ticks if args.slippage_ticks is not None else instrument.slippage_ticks
        execution = ExecutionModel(instrument, slippage_ticks=slippage)

    # --- 6. Build position manager ---
    position_mgr = PositionManager(account_size=args.account_size)

    # --- 7. Run backtest ---
    engine = BacktestEngine(
        instrument=instrument,
        strategies=strategies,
        filters=filters,
        execution=execution,
        position_mgr=position_mgr,
        risk_per_trade=args.risk_per_trade,
        max_contracts=args.max_contracts,
    )

    result = engine.run(df, verbose=not args.quiet)

    # --- 8. Compute and print metrics ---
    metrics = compute_metrics(result.trades, args.account_size)
    print_metrics(metrics)

    # --- 9. Strategy comparison ---
    if len(strategies) > 1:
        strat_comp = compare_strategies(result.trades, args.account_size)
        print_comparison_table(strat_comp, "Strategy Comparison")

    day_comp = compare_by_day_type(result.trades, args.account_size)
    if len(day_comp) > 1:
        print_comparison_table(day_comp, "Day Type Comparison")

    setup_comp = compare_by_setup(result.trades, args.account_size)
    if len(setup_comp) > 1:
        print_comparison_table(setup_comp, "Setup Type Comparison")

    # --- 10. Print trade log summary ---
    print_trade_summary(result.trades, max_rows=30)

    # --- 11. Export trade log ---
    output_dir = Path(args.trade_log).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    export_trade_log(result.trades, args.trade_log)

    # --- 12. Day type analysis ---
    if args.day_analysis:
        print(f"\n--- Day Type Confidence Analysis ---")
        analyses = analyze_sessions(df, verbose=False)
        print_day_analysis(analyses)

    # --- 13. NinjaTrader export ---
    if args.export:
        from export.ninjatrader import NinjaTraderExporter, export_backtest_trades_to_ninjatrader
        print(f"\n--- NinjaTrader Export ---")
        exporter = NinjaTraderExporter({'instrument': args.instrument})
        exporter.export_all('output/ninjatrader')
        export_backtest_trades_to_ninjatrader(result.trades, 'output/ninjatrader')

    print(f"\nDone. Processed {result.sessions_processed} sessions, "
          f"{len(result.trades)} trades.")

    return result


if __name__ == '__main__':
    main()
