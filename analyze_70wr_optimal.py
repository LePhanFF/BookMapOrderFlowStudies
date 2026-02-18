"""
70% Win Rate Optimization Analysis

Goal: Find the exact strategy + filter combination that delivers 70%+ WR
at 5 MNQ max contracts for maximum Lightning account survival.

Tests multiple configurations and outputs the definitive playbook.
"""

import sys
from pathlib import Path
from collections import defaultdict

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
from strategy import (
    get_core_strategies, get_research_strategies,
    TrendDayBull, SuperTrendBull, PDayStrategy, BDayStrategy,
    LiquiditySweep, MeanReversionVWAP, EMATrendFollow, ORBVwapBreakout,
)
from filters.composite import CompositeFilter
from filters.regime_filter import SimpleRegimeFilter
from reporting.metrics import compute_metrics


def run_config(strategies, filters, label, df, instrument, max_contracts=5):
    """Run a single backtest config and return detailed results."""
    execution = ExecutionModel(instrument, slippage_ticks=1)
    position_mgr = PositionManager(account_size=150000)

    engine = BacktestEngine(
        instrument=instrument,
        strategies=strategies,
        filters=filters,
        execution=execution,
        position_mgr=position_mgr,
        risk_per_trade=400,
        max_contracts=max_contracts,
    )

    result = engine.run(df, verbose=False)
    metrics = compute_metrics(result.trades, 150000)
    trades = result.trades

    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl <= 0]
    long_trades = [t for t in trades if t.direction == 'LONG']
    short_trades = [t for t in trades if t.direction == 'SHORT']

    total_pnl = sum(t.net_pnl for t in trades)

    # Win days / loss days
    daily_pnl = defaultdict(float)
    for t in trades:
        daily_pnl[t.session_date] += t.net_pnl
    win_days = sum(1 for v in daily_pnl.values() if v > 0)
    loss_days = sum(1 for v in daily_pnl.values() if v <= 0)
    active_days = len(daily_pnl)

    # Max daily loss
    max_daily_loss = min(daily_pnl.values()) if daily_pnl else 0

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
        'short_trades': len(short_trades),
        'win_days': win_days,
        'loss_days': loss_days,
        'active_days': active_days,
        'win_day_rate': win_days / active_days * 100 if active_days else 0,
        'max_daily_loss': max_daily_loss,
        'signals_gen': result.signals_generated,
        'signals_filt': result.signals_filtered,
        'trade_list': trades,
        'daily_pnl': dict(daily_pnl),
    }


def main():
    # Load data
    instrument = get_instrument('MNQ')
    df = load_csv('NQ')
    df = filter_rth(df)
    df = compute_all_features(df)

    print("=" * 130)
    print("  70% WIN RATE OPTIMIZATION — 5 MNQ MAX")
    print("  Data: NQ, Nov 18 2025 - Feb 16 2026 (62 sessions)")
    print("=" * 130)

    # ============================================================
    # CONFIGURATIONS TO TEST
    # ============================================================
    configs = []

    # A: Core only (baseline)
    configs.append(run_config(
        strategies=get_core_strategies(),
        filters=None,
        label='A: Core Only',
        df=df, instrument=instrument,
    ))

    # B: Core LONG only (no bear strategies already excluded)
    configs.append(run_config(
        strategies=[TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy()],
        filters=None,
        label='B: Core LONG only',
        df=df, instrument=instrument,
    ))

    # C: Core + Liquidity Sweep (LONG only via regime)
    regime_long_only = CompositeFilter([
        SimpleRegimeFilter(
            longs_in_bull=True, longs_in_bear=True,
            shorts_in_bull=False, shorts_in_bear=False,
        ),
    ])
    configs.append(run_config(
        strategies=[TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(),
                    LiquiditySweep()],
        filters=regime_long_only,
        label='C: Core + Sweep (LONG only)',
        df=df, instrument=instrument,
    ))

    # D: Core + Mean Reversion VWAP (LONG only)
    configs.append(run_config(
        strategies=[TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(),
                    MeanReversionVWAP()],
        filters=regime_long_only,
        label='D: Core + MeanRev (LONG only)',
        df=df, instrument=instrument,
    ))

    # E: Core + Sweep + MeanRev (LONG only) -- previous best
    configs.append(run_config(
        strategies=[TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(),
                    LiquiditySweep(), MeanReversionVWAP()],
        filters=regime_long_only,
        label='E: Core + Sweep + MeanRev (LONG)',
        df=df, instrument=instrument,
    ))

    # F: Core + EMA Trend (LONG only)
    configs.append(run_config(
        strategies=[TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(),
                    EMATrendFollow()],
        filters=regime_long_only,
        label='F: Core + EMA (LONG only)',
        df=df, instrument=instrument,
    ))

    # G: Core + All Research (LONG only)
    configs.append(run_config(
        strategies=[TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(),
                    LiquiditySweep(), MeanReversionVWAP(), EMATrendFollow(), ORBVwapBreakout()],
        filters=regime_long_only,
        label='G: Core + All Research (LONG)',
        df=df, instrument=instrument,
    ))

    # H: Core + Sweep + MeanRev (relaxed regime: longs always, shorts in bear)
    regime_relaxed = CompositeFilter([
        SimpleRegimeFilter(
            longs_in_bull=True, longs_in_bear=True,
            shorts_in_bull=False, shorts_in_bear=True,
        ),
    ])
    configs.append(run_config(
        strategies=[TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(),
                    LiquiditySweep(), MeanReversionVWAP()],
        filters=regime_relaxed,
        label='H: Core+Sweep+MeanRev (relax regime)',
        df=df, instrument=instrument,
    ))

    # I: Just TrendDayBull + PDayStrategy (highest WR core)
    configs.append(run_config(
        strategies=[TrendDayBull(), PDayStrategy()],
        filters=None,
        label='I: TrendBull + PDay only',
        df=df, instrument=instrument,
    ))

    # J: TrendBull + PDay + Sweep (LONG)
    configs.append(run_config(
        strategies=[TrendDayBull(), PDayStrategy(), LiquiditySweep()],
        filters=regime_long_only,
        label='J: TrendBull+PDay+Sweep (LONG)',
        df=df, instrument=instrument,
    ))

    # ============================================================
    # PRINT COMPARISON TABLE
    # ============================================================
    print()
    hdr = (f"{'Config':<42s} {'Trades':>6s} {'W':>3s} {'L':>3s} {'WR%':>6s} "
           f"{'PF':>6s} {'Net PnL':>10s} {'Expect':>8s} {'MaxDD':>8s} "
           f"{'WinDays':>7s} {'LossDays':>8s} {'WinDay%':>7s} {'MaxDLoss':>9s}")
    print(hdr)
    print("-" * 130)

    for r in configs:
        print(
            f"{r['label']:<42s} {r['trades']:>6d} {r['wins']:>3d} {r['losses']:>3d} "
            f"{r['wr']:>5.1f}% {r['pf']:>6.2f} ${r['net_pnl']:>8,.0f} "
            f"${r['expectancy']:>6,.0f} ${r['max_dd']:>7,.0f} "
            f"{r['win_days']:>4d}/{r['active_days']:<2d} "
            f"{r['loss_days']:>5d}/{r['active_days']:<2d} "
            f"{r['win_day_rate']:>5.1f}% ${r['max_daily_loss']:>7,.0f}"
        )

    print("-" * 130)

    # ============================================================
    # DETAILED BREAKDOWN OF BEST 70%+ WR CONFIG
    # ============================================================
    # Find best config with 70%+ WR
    best_70 = [r for r in configs if r['wr'] >= 70.0]
    if not best_70:
        best_70 = sorted(configs, key=lambda x: x['wr'], reverse=True)[:1]

    # Sort by: WR >= 70 first, then by net_pnl
    best_70.sort(key=lambda x: (x['wr'] >= 70, x['net_pnl']), reverse=True)
    best = best_70[0]

    print(f"\n{'=' * 130}")
    print(f"  BEST 70%+ WR CONFIG: {best['label']}")
    print(f"  WR: {best['wr']:.1f}% | Trades: {best['trades']} | PnL: ${best['net_pnl']:,.0f} | "
          f"Expect: ${best['expectancy']:,.0f} | PF: {best['pf']:.2f}")
    print(f"{'=' * 130}")

    trades = best['trade_list']

    # Per-strategy breakdown
    strat_data = defaultdict(list)
    for t in trades:
        strat_data[t.strategy_name].append(t)

    print(f"\n--- Per Strategy ---")
    print(f"{'Strategy':<25s} {'Trades':>6s} {'WR%':>6s} {'Net PnL':>10s} {'Expect':>8s} "
          f"{'AvgWin':>8s} {'AvgLoss':>8s} {'Long':>5s} {'Short':>5s}")
    print(f"{'-' * 95}")

    for sname in sorted(strat_data.keys()):
        st = strat_data[sname]
        w = [t for t in st if t.net_pnl > 0]
        l = [t for t in st if t.net_pnl <= 0]
        wr = len(w) / len(st) * 100
        pnl = sum(t.net_pnl for t in st)
        exp = pnl / len(st)
        avg_w = sum(t.net_pnl for t in w) / len(w) if w else 0
        avg_l = sum(t.net_pnl for t in l) / len(l) if l else 0
        longs = sum(1 for t in st if t.direction == 'LONG')
        shorts = sum(1 for t in st if t.direction == 'SHORT')
        print(f"{sname:<25s} {len(st):>6d} {wr:>5.1f}% ${pnl:>8,.0f} ${exp:>6,.0f} "
              f"${avg_w:>6,.0f} ${avg_l:>6,.0f} {longs:>5d} {shorts:>5d}")

    # Per setup type breakdown
    print(f"\n--- Per Setup Type ---")
    setup_data = defaultdict(list)
    for t in trades:
        setup_data[t.setup_type].append(t)

    print(f"{'Setup':<28s} {'Dir':>5s} {'Trades':>6s} {'WR%':>6s} {'Net PnL':>10s} {'Expect':>8s}")
    print(f"{'-' * 75}")

    for setup in sorted(setup_data.keys()):
        st = setup_data[setup]
        dirs = set(t.direction for t in st)
        d_str = '/'.join(sorted(dirs))
        w = sum(1 for t in st if t.net_pnl > 0)
        wr = w / len(st) * 100
        pnl = sum(t.net_pnl for t in st)
        exp = pnl / len(st)
        print(f"{setup:<28s} {d_str:>5s} {len(st):>6d} {wr:>5.1f}% ${pnl:>8,.0f} ${exp:>6,.0f}")

    # Exit reason breakdown
    print(f"\n--- Exit Reasons ---")
    exit_data = defaultdict(list)
    for t in trades:
        exit_data[t.exit_reason].append(t)

    print(f"{'Exit Reason':<20s} {'Trades':>6s} {'WR%':>6s} {'Net PnL':>10s} {'AvgPnL':>8s}")
    print(f"{'-' * 55}")

    for reason in sorted(exit_data.keys()):
        st = exit_data[reason]
        w = sum(1 for t in st if t.net_pnl > 0)
        wr = w / len(st) * 100
        pnl = sum(t.net_pnl for t in st)
        avg = pnl / len(st)
        print(f"{reason:<20s} {len(st):>6d} {wr:>5.1f}% ${pnl:>8,.0f} ${avg:>6,.0f}")

    # Bars held analysis
    print(f"\n--- Bars Held Analysis ---")
    wins_list = [t for t in trades if t.net_pnl > 0]
    losses_list = [t for t in trades if t.net_pnl <= 0]
    if wins_list:
        avg_bars_win = sum(t.bars_held for t in wins_list) / len(wins_list)
        print(f"Winners avg bars held: {avg_bars_win:.0f}")
    if losses_list:
        avg_bars_loss = sum(t.bars_held for t in losses_list) / len(losses_list)
        print(f"Losers avg bars held:  {avg_bars_loss:.0f}")

    # Daily P&L distribution
    print(f"\n--- Daily P&L Distribution ---")
    daily = best['daily_pnl']
    sorted_days = sorted(daily.items(), key=lambda x: x[1])
    print("Worst 5 days:")
    for day, pnl in sorted_days[:5]:
        day_trades = [t for t in trades if t.session_date == day]
        print(f"  {day}: ${pnl:>8,.0f} ({len(day_trades)} trades)")

    print("Best 5 days:")
    for day, pnl in sorted_days[-5:]:
        day_trades = [t for t in trades if t.session_date == day]
        print(f"  {day}: ${pnl:>8,.0f} ({len(day_trades)} trades)")

    # Trade frequency per day
    trades_per_day = defaultdict(int)
    for t in trades:
        trades_per_day[t.session_date] += 1

    if trades_per_day:
        avg_tpd = sum(trades_per_day.values()) / len(trades_per_day)
        max_tpd = max(trades_per_day.values())
        print(f"\nAvg trades/active day: {avg_tpd:.1f}")
        print(f"Max trades/day: {max_tpd}")

    # Consistency rule check (20% rule for Lightning)
    if trades:
        total_profit = sum(t.net_pnl for t in trades if t.net_pnl > 0)
        daily_profits = {d: p for d, p in daily.items() if p > 0}
        if total_profit > 0 and daily_profits:
            max_day_profit = max(daily_profits.values())
            consistency_pct = max_day_profit / total_profit * 100
            print(f"\n--- Consistency Rule Check ---")
            print(f"Total winning P&L: ${total_profit:,.0f}")
            print(f"Best single day:   ${max_day_profit:,.0f}")
            print(f"Best day % of total: {consistency_pct:.1f}% (need < 20% for Lightning)")
            if consistency_pct > 20:
                print(f"WARNING: Best day is {consistency_pct:.1f}% of total — need more winning days to pass 20% rule")
            else:
                print(f"PASS: Within 20% consistency rule")

    # ============================================================
    # COMPARISON: All configs that hit 70%+ WR
    # ============================================================
    print(f"\n{'=' * 130}")
    print(f"  ALL CONFIGS WITH 70%+ WIN RATE")
    print(f"{'=' * 130}")
    seventy_plus = [r for r in configs if r['wr'] >= 70.0]
    if seventy_plus:
        for r in sorted(seventy_plus, key=lambda x: x['net_pnl'], reverse=True):
            print(f"  {r['label']:<42s} WR:{r['wr']:>5.1f}% Trades:{r['trades']:>3d} "
                  f"PnL:${r['net_pnl']:>8,.0f} Expect:${r['expectancy']:>6,.0f} "
                  f"WinDays:{r['win_days']}/{r['active_days']} MaxDLoss:${r['max_daily_loss']:>7,.0f}")
    else:
        print("  No configs hit 70% WR — showing top 3:")
        for r in sorted(configs, key=lambda x: x['wr'], reverse=True)[:3]:
            print(f"  {r['label']:<42s} WR:{r['wr']:>5.1f}% Trades:{r['trades']:>3d} "
                  f"PnL:${r['net_pnl']:>8,.0f} Expect:${r['expectancy']:>6,.0f}")

    print(f"\n{'=' * 130}")
    print("  RECOMMENDATION FOR 70% WR / 5 MNQ / LIGHTNING SURVIVAL")
    print(f"{'=' * 130}")
    print(f"  Best config: {best['label']}")
    print(f"  Win Rate: {best['wr']:.1f}%")
    print(f"  Trades over 62 sessions: {best['trades']}")
    print(f"  Net P&L: ${best['net_pnl']:,.0f}")
    print(f"  Expectancy per trade: ${best['expectancy']:,.0f}")
    print(f"  Win day rate: {best['win_day_rate']:.1f}%")
    print(f"  Max daily loss: ${best['max_daily_loss']:,.0f}")


if __name__ == '__main__':
    main()
