"""
Regime Filter Comparison: With vs Without regime gating.

Answers: Does gating longs to bull regime and shorts to bear regime
improve performance? How does it affect the combined portfolio?

Runs 4 configurations:
  1. Core strategies only (no filters)
  2. Research strategies, NO regime filter (both dirs always)
  3. Research strategies, WITH regime filter (directionally gated)
  4. ALL strategies (core + research), WITH regime filter
"""

import sys
from pathlib import Path
from datetime import time

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.instruments import get_instrument
from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from engine.backtest import BacktestEngine
from engine.execution import ExecutionModel
from engine.position import PositionManager
from strategy import get_core_strategies, get_research_strategies
from filters.composite import CompositeFilter
from filters.regime_filter import SimpleRegimeFilter
from reporting.metrics import compute_metrics


def run_backtest(strategies, filters, label, df, instrument):
    """Run a single backtest and return summary dict."""
    execution = ExecutionModel(instrument, slippage_ticks=1)
    position_mgr = PositionManager(account_size=150000)

    engine = BacktestEngine(
        instrument=instrument,
        strategies=strategies,
        filters=filters,
        execution=execution,
        position_mgr=position_mgr,
        risk_per_trade=400,
        max_contracts=30,
    )

    result = engine.run(df, verbose=False)
    metrics = compute_metrics(result.trades, 150000)

    trades = result.trades
    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl <= 0]
    long_trades = [t for t in trades if t.direction == 'LONG']
    short_trades = [t for t in trades if t.direction == 'SHORT']
    long_wins = [t for t in long_trades if t.net_pnl > 0]
    short_wins = [t for t in short_trades if t.net_pnl > 0]

    total_pnl = sum(t.net_pnl for t in trades)
    long_pnl = sum(t.net_pnl for t in long_trades)
    short_pnl = sum(t.net_pnl for t in short_trades)

    return {
        'label': label,
        'trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'wr': len(wins) / len(trades) * 100 if trades else 0,
        'pf': metrics.get('profit_factor', 0),
        'net_pnl': total_pnl,
        'expectancy': total_pnl / len(trades) if trades else 0,
        'max_dd': metrics.get('max_drawdown', 0),
        'sharpe': metrics.get('sharpe_ratio', 0),
        'long_trades': len(long_trades),
        'long_wr': len(long_wins) / len(long_trades) * 100 if long_trades else 0,
        'long_pnl': long_pnl,
        'short_trades': len(short_trades),
        'short_wr': len(short_wins) / len(short_trades) * 100 if short_trades else 0,
        'short_pnl': short_pnl,
        'signals_gen': result.signals_generated,
        'signals_filt': result.signals_filtered,
        'result': result,
    }


def main():
    # Load data
    instrument = get_instrument('MNQ')
    df = load_csv('NQ')
    df = filter_rth(df)
    df = compute_all_features(df)

    print("=" * 130)
    print("  REGIME FILTER COMPARISON")
    print("  Data: NQ, Nov 18 2025 - Feb 16 2026 (62 sessions)")
    print("  Market was CHOPPY: -0.9% net, 50/50 up/down days, 61% bear EMA regime")
    print("=" * 130)

    # --- Config 1: Core only ---
    r1 = run_backtest(
        strategies=get_core_strategies(),
        filters=None,
        label='Core Only (no filters)',
        df=df, instrument=instrument,
    )

    # --- Config 2: Research, NO regime filter ---
    r2 = run_backtest(
        strategies=get_research_strategies(),
        filters=None,
        label='Research (no regime filter)',
        df=df, instrument=instrument,
    )

    # --- Config 3: Research, WITH regime filter ---
    regime_filter = CompositeFilter([
        SimpleRegimeFilter(
            longs_in_bull=True,   # Longs when EMA20 > EMA50
            shorts_in_bear=True,  # Shorts when EMA20 < EMA50
            longs_in_bear=False,  # Block longs in bear regime
            shorts_in_bull=False, # Block shorts in bull regime
        ),
    ])
    r3 = run_backtest(
        strategies=get_research_strategies(),
        filters=regime_filter,
        label='Research + Regime Filter',
        df=df, instrument=instrument,
    )

    # --- Config 3b: Research, RELAXED regime (allow longs always, shorts only in bear) ---
    regime_relaxed = CompositeFilter([
        SimpleRegimeFilter(
            longs_in_bull=True,
            shorts_in_bear=True,
            longs_in_bear=True,   # Allow longs in bear too (NQ has long bias)
            shorts_in_bull=False, # Block shorts in bull
        ),
    ])
    r3b = run_backtest(
        strategies=get_research_strategies(),
        filters=regime_relaxed,
        label='Research + Relaxed Regime (longs always)',
        df=df, instrument=instrument,
    )

    # --- Config 4: ALL strategies (core + research) + regime filter ---
    r4 = run_backtest(
        strategies=get_core_strategies() + get_research_strategies(),
        filters=regime_filter,
        label='Core + Research + Regime Filter',
        df=df, instrument=instrument,
    )

    # --- Config 4b: ALL + relaxed regime ---
    r4b = run_backtest(
        strategies=get_core_strategies() + get_research_strategies(),
        filters=regime_relaxed,
        label='Core + Research + Relaxed Regime',
        df=df, instrument=instrument,
    )

    configs = [r1, r2, r3, r3b, r4, r4b]

    # Print comparison
    print(f"\n{'─' * 130}")
    print(f"{'Config':<42s} {'Trades':>6s} {'WR%':>6s} {'PF':>6s} {'Net P&L':>10s} "
          f"{'Expect':>8s} {'Sharpe':>7s} {'Long':>5s} {'L-WR%':>6s} {'L-PnL':>9s} "
          f"{'Short':>5s} {'S-WR%':>6s} {'S-PnL':>9s} {'Filt':>5s}")
    print(f"{'─' * 130}")

    for r in configs:
        print(
            f"{r['label']:<42s} {r['trades']:>6d} {r['wr']:>5.1f}% {r['pf']:>6.2f} "
            f"${r['net_pnl']:>8,.0f} ${r['expectancy']:>6,.0f} {r['sharpe']:>7.2f} "
            f"{r['long_trades']:>5d} {r['long_wr']:>5.1f}% ${r['long_pnl']:>7,.0f} "
            f"{r['short_trades']:>5d} {r['short_wr']:>5.1f}% ${r['short_pnl']:>7,.0f} "
            f"{r['signals_filt']:>5d}"
        )

    print(f"{'─' * 130}")

    # Per-strategy breakdown for best config
    print(f"\n{'=' * 130}")
    print(f"  DETAILED BREAKDOWN: Core + Research + Relaxed Regime")
    print(f"{'=' * 130}")

    for r in [r4b]:
        trades = r['result'].trades
        from collections import defaultdict
        strat_data = defaultdict(list)
        for t in trades:
            strat_data[t.strategy_name].append(t)

        print(f"\n{'Strategy':<25s} {'Trades':>6s} {'WR%':>6s} {'Net P&L':>10s} {'Expect':>8s} "
              f"{'Long':>5s} {'Short':>5s} {'L-WR%':>6s} {'S-WR%':>6s}")
        print(f"{'─' * 95}")

        for sname in sorted(strat_data.keys()):
            st = strat_data[sname]
            w = sum(1 for t in st if t.net_pnl > 0)
            wr = w / len(st) * 100
            pnl = sum(t.net_pnl for t in st)
            exp = pnl / len(st)
            longs = [t for t in st if t.direction == 'LONG']
            shorts = [t for t in st if t.direction == 'SHORT']
            lw = sum(1 for t in longs if t.net_pnl > 0) / len(longs) * 100 if longs else 0
            sw = sum(1 for t in shorts if t.net_pnl > 0) / len(shorts) * 100 if shorts else 0
            print(
                f"{sname:<25s} {len(st):>6d} {wr:>5.1f}% ${pnl:>8,.0f} ${exp:>6,.0f} "
                f"{len(longs):>5d} {len(shorts):>5d} {lw:>5.1f}% {sw:>5.1f}%"
            )

    # Setup type breakdown
    print(f"\n{'=' * 130}")
    print(f"  SETUP TYPE BREAKDOWN: Core + Research + Relaxed Regime")
    print(f"{'=' * 130}")

    for r in [r4b]:
        trades = r['result'].trades
        setup_data = defaultdict(list)
        for t in trades:
            setup_data[t.setup_type].append(t)

        print(f"\n{'Setup':<28s} {'Dir':>5s} {'Trades':>6s} {'WR%':>6s} {'Net P&L':>10s} {'Expect':>8s}")
        print(f"{'─' * 70}")

        for setup in sorted(setup_data.keys()):
            st = setup_data[setup]
            dirs = set(t.direction for t in st)
            d_str = '/'.join(dirs)
            w = sum(1 for t in st if t.net_pnl > 0)
            wr = w / len(st) * 100
            pnl = sum(t.net_pnl for t in st)
            exp = pnl / len(st)
            print(f"{setup:<28s} {d_str:>5s} {len(st):>6d} {wr:>5.1f}% ${pnl:>8,.0f} ${exp:>6,.0f}")


if __name__ == '__main__':
    main()
