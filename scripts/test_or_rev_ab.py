"""
A/B Test script for OR Reversal strategy improvements.

Tests three changes against the baseline (56 trades, 67.9% WR, $15,275 net):
  Test 1: Relax FADE filter - only skip fades on LOW/MED IB, allow on NORMAL/HIGH
  Test 2: Widen VWAP proximity to 30% of EOR range on HIGH IB (>=250)
  Test 3: Combined (Test 1 + Test 2)

Usage:
    E:\\anaconda\\python.exe scripts/test_or_rev_ab.py
"""

import sys
import copy
from pathlib import Path
from datetime import time
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
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
from strategy import get_strategies_by_name
from filters.composite import CompositeFilter
from filters.time_filter import TimeFilter
from filters.volatility_filter import VolatilityFilter
from filters.strategy_regime_filter import StrategyRegimeFilter
from reporting.metrics import compute_metrics
from reporting.trade_log import export_trade_log

# ---- constants mirroring or_reversal.py ----
OR_BARS = 15
EOR_BARS = 30
SWEEP_THRESHOLD_RATIO = 0.17
VWAP_ALIGNED_RATIO = 0.17
VWAP_ALIGNED_RATIO_HIGH = 0.30   # Test 2: widened for HIGH IB
OR_STOP_BUFFER = 0.15
MIN_RISK_RATIO = 0.03
MAX_RISK_RATIO = 1.3
DRIVE_THRESHOLD = 0.4

# IB regime thresholds
IB_HIGH_THRESHOLD = 250   # HIGH regime
IB_MED_THRESHOLD = 150    # MED regime


def _find_closest_swept_level(eor_extreme, candidates, sweep_threshold, eor_range):
    best_level = None
    best_name = None
    best_dist = float('inf')
    for name, lvl in candidates:
        if lvl is None:
            continue
        dist = abs(eor_extreme - lvl)
        if dist < sweep_threshold and dist <= eor_range and dist < best_dist:
            best_dist = dist
            best_level = lvl
            best_name = name
    return best_level, best_name


def _build_on_session_start(relax_fade: bool, widen_vwap: bool):
    """
    Factory: build an on_session_start method with the selected modifications.

    relax_fade: If True, only skip FADE trades on LOW/MED IB.
                On NORMAL/HIGH IB, allow fades (DRIVE_UP SHORT and DRIVE_DOWN LONG).
    widen_vwap: If True, use 30% VWAP threshold on HIGH IB (>=250) instead of 17%.
    """
    def on_session_start(self_strat, session_date, ib_high, ib_low, ib_range, session_context):
        self_strat._cached_signal = None
        self_strat._signal_emitted = False
        self_strat._ib_range = ib_range

        ib_bars = session_context.get('ib_bars')
        if ib_bars is None or len(ib_bars) < EOR_BARS:
            return

        or_bars = ib_bars.iloc[:OR_BARS]
        or_high = or_bars['high'].max()
        or_low = or_bars['low'].min()
        or_mid = (or_high + or_low) / 2

        eor_bars = ib_bars.iloc[:EOR_BARS]
        eor_high = eor_bars['high'].max()
        eor_low = eor_bars['low'].min()
        eor_range_val = eor_high - eor_low

        if eor_range_val < ib_range * 0.05 if ib_range > 0 else eor_range_val < 10:
            return

        sweep_threshold = eor_range_val * SWEEP_THRESHOLD_RATIO

        # Test 2: widen VWAP threshold on HIGH IB
        if widen_vwap and ib_range >= IB_HIGH_THRESHOLD:
            vwap_threshold = eor_range_val * VWAP_ALIGNED_RATIO_HIGH
        else:
            vwap_threshold = eor_range_val * VWAP_ALIGNED_RATIO

        max_risk = eor_range_val * MAX_RISK_RATIO

        # Opening drive classification from first 5 bars
        first_5 = ib_bars.iloc[:5]
        open_price = first_5.iloc[0]['open']
        close_5th = first_5.iloc[4]['close']
        drive_range = first_5['high'].max() - first_5['low'].min()
        if drive_range > 0:
            drive_pct = (close_5th - open_price) / drive_range
        else:
            drive_pct = 0

        if drive_pct > DRIVE_THRESHOLD:
            opening_drive = 'DRIVE_UP'
        elif drive_pct < -DRIVE_THRESHOLD:
            opening_drive = 'DRIVE_DOWN'
        else:
            opening_drive = 'ROTATION'

        overnight_high = session_context.get('overnight_high') or session_context.get('prior_session_high')
        overnight_low = session_context.get('overnight_low') or session_context.get('prior_session_low')
        if overnight_high is None or overnight_low is None:
            return

        pdh = session_context.get('pdh') or session_context.get('prior_session_high')
        pdl = session_context.get('pdl') or session_context.get('prior_session_low')
        asia_high = session_context.get('asia_high')
        asia_low = session_context.get('asia_low')
        london_high = session_context.get('london_high')
        london_low = session_context.get('london_low')

        high_candidates = [('ON_HIGH', overnight_high)]
        if pdh: high_candidates.append(('PDH', pdh))
        if asia_high: high_candidates.append(('ASIA_HIGH', asia_high))
        if london_high: high_candidates.append(('LDN_HIGH', london_high))

        low_candidates = [('ON_LOW', overnight_low)]
        if pdl: low_candidates.append(('PDL', pdl))
        if asia_low: low_candidates.append(('ASIA_LOW', asia_low))
        if london_low: low_candidates.append(('LDN_LOW', london_low))

        swept_high_level, swept_high_name = _find_closest_swept_level(
            eor_high, high_candidates, sweep_threshold, eor_range_val)
        swept_low_level, swept_low_name = _find_closest_swept_level(
            eor_low, low_candidates, sweep_threshold, eor_range_val)

        if swept_high_level is None and swept_low_level is None:
            return

        if swept_high_level is not None and swept_low_level is not None:
            high_depth = eor_high - swept_high_level
            low_depth = swept_low_level - eor_low
            if high_depth >= low_depth:
                swept_low_level = None
                swept_low_name = None
            else:
                swept_high_level = None
                swept_high_name = None

        high_bar_idx = eor_bars['high'].idxmax()
        low_bar_idx = eor_bars['low'].idxmin()

        deltas = ib_bars['delta'] if 'delta' in ib_bars.columns else ib_bars.get('vol_delta', pd.Series(dtype=float))
        if deltas is not None and len(deltas) > 0:
            deltas = deltas.fillna(0)
            cvd_series = deltas.cumsum()
        else:
            cvd_series = None

        # === SHORT SETUP ===
        if swept_high_level is not None:
            # Test 1: relax FADE filter
            # Baseline: always skip DRIVE_UP SHORT
            # Test 1: only skip if IB is LOW or MED (<150)
            if relax_fade:
                skip_fade = (opening_drive == 'DRIVE_UP' and ib_range < IB_MED_THRESHOLD)
            else:
                skip_fade = (opening_drive == 'DRIVE_UP')

            if not skip_fade:
                cvd_at_extreme = cvd_series.loc[high_bar_idx] if cvd_series is not None else None
                post_high = ib_bars.loc[high_bar_idx:]
                for j in range(1, min(30, len(post_high))):
                    bar = post_high.iloc[j]
                    price = bar['close']
                    if price >= or_mid:
                        continue
                    vwap = bar.get('vwap', np.nan)
                    if pd.isna(vwap) or abs(price - vwap) > vwap_threshold:
                        continue
                    delta = bar.get('delta', bar.get('vol_delta', 0))
                    if pd.isna(delta):
                        delta = 0
                    cvd_at_entry = cvd_series.loc[post_high.index[j]] if cvd_series is not None else None
                    cvd_declining = (cvd_at_entry is not None and cvd_at_extreme is not None
                                     and cvd_at_entry < cvd_at_extreme)
                    if delta >= 0 and not cvd_declining:
                        continue
                    stop = swept_high_level + eor_range_val * OR_STOP_BUFFER
                    risk = stop - price
                    if risk < eor_range_val * MIN_RISK_RATIO or risk > max_risk:
                        continue
                    target = price - 2 * risk
                    from strategy.signal import Signal
                    bar_ts = bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp')
                    self_strat._cached_signal = Signal(
                        timestamp=bar_ts,
                        direction='SHORT',
                        entry_price=price,
                        stop_price=stop,
                        target_price=target,
                        strategy_name=self_strat.name,
                        setup_type='OR_REVERSAL_SHORT',
                        day_type='neutral',
                        trend_strength='moderate',
                        confidence='high',
                        metadata={
                            'level_swept': swept_high_name,
                            'swept_level_price': swept_high_level,
                            'sweep_depth': eor_high - swept_high_level,
                            'vwap_aligned': True,
                            'opening_drive': opening_drive,
                            'cvd_declining': cvd_declining,
                        },
                    )
                    return

        # === LONG SETUP ===
        if swept_low_level is not None:
            # Test 1: relax FADE filter for LONG
            if relax_fade:
                skip_fade = (opening_drive == 'DRIVE_DOWN' and ib_range < IB_MED_THRESHOLD)
            else:
                skip_fade = (opening_drive == 'DRIVE_DOWN')

            if not skip_fade:
                cvd_at_extreme = cvd_series.loc[low_bar_idx] if cvd_series is not None else None
                post_low = ib_bars.loc[low_bar_idx:]
                for j in range(1, min(30, len(post_low))):
                    bar = post_low.iloc[j]
                    price = bar['close']
                    if price <= or_mid:
                        continue
                    vwap = bar.get('vwap', np.nan)
                    if pd.isna(vwap) or abs(price - vwap) > vwap_threshold:
                        continue
                    delta = bar.get('delta', bar.get('vol_delta', 0))
                    if pd.isna(delta):
                        delta = 0
                    cvd_at_entry = cvd_series.loc[post_low.index[j]] if cvd_series is not None else None
                    cvd_rising = (cvd_at_entry is not None and cvd_at_extreme is not None
                                  and cvd_at_entry > cvd_at_extreme)
                    if delta <= 0 and not cvd_rising:
                        continue
                    stop = swept_low_level - eor_range_val * OR_STOP_BUFFER
                    risk = price - stop
                    if risk < eor_range_val * MIN_RISK_RATIO or risk > max_risk:
                        continue
                    target = price + 2 * risk
                    from strategy.signal import Signal
                    bar_ts = bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp')
                    self_strat._cached_signal = Signal(
                        timestamp=bar_ts,
                        direction='LONG',
                        entry_price=price,
                        stop_price=stop,
                        target_price=target,
                        strategy_name=self_strat.name,
                        setup_type='OR_REVERSAL_LONG',
                        day_type='neutral',
                        trend_strength='moderate',
                        confidence='high',
                        metadata={
                            'level_swept': swept_low_name,
                            'swept_level_price': swept_low_level,
                            'sweep_depth': swept_low_level - eor_low,
                            'vwap_aligned': True,
                            'opening_drive': opening_drive,
                            'cvd_rising': cvd_rising,
                        },
                    )
                    return

    return on_session_start


def run_variant(label, df, full_df, instrument, filters, relax_fade, widen_vwap,
                account_size=DEFAULT_ACCOUNT_SIZE):
    """Run one backtest variant, return (metrics, trades list)."""
    from strategy.or_reversal import OpeningRangeReversal
    import types

    strat = OpeningRangeReversal()
    patched_fn = _build_on_session_start(relax_fade=relax_fade, widen_vwap=widen_vwap)
    strat.on_session_start = types.MethodType(patched_fn, strat)

    execution = ExecutionModel(instrument)
    position_mgr = PositionManager(account_size=account_size, max_drawdown=999999)

    engine = BacktestEngine(
        instrument=instrument,
        strategies=[strat],
        filters=filters,
        execution=execution,
        position_mgr=position_mgr,
        risk_per_trade=DEFAULT_MAX_RISK_PER_TRADE,
        max_contracts=30,
        full_df=full_df,
    )

    result = engine.run(df, verbose=False)
    metrics = compute_metrics(result.trades, account_size)
    return metrics, result.trades


def print_summary(label, metrics, trades):
    wins = [t for t in trades if t.net_pnl > 0]
    wr = len(wins) / len(trades) * 100 if trades else 0
    net = sum(t.net_pnl for t in trades)
    max_dd = metrics.get('max_drawdown', 0)
    pf = metrics.get('profit_factor', 0)
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Trades    : {len(trades)}")
    print(f"  Win Rate  : {wr:.1f}%")
    print(f"  Net P&L   : ${net:,.0f}")
    print(f"  Max DD    : ${max_dd:,.0f}")
    print(f"  Prof. Factor: {pf:.2f}")


def trades_to_df(trades):
    rows = []
    for t in trades:
        rows.append({
            'date': getattr(t, 'entry_time', None),
            'direction': getattr(t, 'direction', None),
            'entry': getattr(t, 'entry_price', None),
            'exit': getattr(t, 'exit_price', None),
            'pnl': getattr(t, 'net_pnl', None),
            'setup': getattr(t, 'setup_type', None),
        })
    return pd.DataFrame(rows)


def find_new_trades(baseline_trades, variant_trades):
    """Identify trades in variant but not in baseline (matched by date+direction)."""
    baseline_keys = set()
    for t in baseline_trades:
        dt = getattr(t, 'entry_time', None)
        if dt is not None:
            key = (str(dt)[:10], getattr(t, 'direction', None))
            baseline_keys.add(key)

    new_trades = []
    for t in variant_trades:
        dt = getattr(t, 'entry_time', None)
        if dt is not None:
            key = (str(dt)[:10], getattr(t, 'direction', None))
            if key not in baseline_keys:
                new_trades.append(t)
    return new_trades


def main():
    print("\n" + "="*60)
    print("  OR Reversal A/B Test Suite")
    print("="*60)

    # Load data
    print("\nLoading data...")
    instrument = get_instrument('MNQ')
    full_df = load_csv('NQ')
    df = filter_rth(full_df)
    df = compute_all_features(df)
    print(f"  Loaded {len(df)} bars")

    # Build filters (same as default run_backtest)
    filters = CompositeFilter([
        StrategyRegimeFilter(),
        TimeFilter(start=time(10, 30), end=time(15, 30)),
        VolatilityFilter(min_atr=5.0, max_atr=80.0),
    ])

    # ---- BASELINE ----
    print("\nRunning BASELINE...")
    base_metrics, base_trades = run_variant(
        "BASELINE", df, full_df, instrument, filters,
        relax_fade=False, widen_vwap=False)
    print_summary("BASELINE (current or_reversal.py)", base_metrics, base_trades)

    # ---- TEST 1: Relax FADE filter ----
    print("\nRunning TEST 1: Relax FADE filter (allow fades on NORMAL/HIGH IB)...")
    t1_metrics, t1_trades = run_variant(
        "TEST 1", df, full_df, instrument, filters,
        relax_fade=True, widen_vwap=False)
    print_summary("TEST 1: Relax FADE (NORMAL+HIGH allow)", t1_metrics, t1_trades)

    new_t1 = find_new_trades(base_trades, t1_trades)
    removed_t1 = find_new_trades(t1_trades, base_trades)  # in baseline but not in t1
    if new_t1:
        print(f"\n  +++ {len(new_t1)} NEW trades added vs baseline:")
        for t in new_t1:
            dt = str(getattr(t, 'entry_time', '?'))[:10]
            d = getattr(t, 'direction', '?')
            pnl = getattr(t, 'net_pnl', 0)
            print(f"      {dt}  {d}  ${pnl:+,.0f}")
    if removed_t1:
        print(f"\n  --- {len(removed_t1)} trades REMOVED vs baseline:")
        for t in removed_t1:
            dt = str(getattr(t, 'entry_time', '?'))[:10]
            d = getattr(t, 'direction', '?')
            pnl = getattr(t, 'net_pnl', 0)
            print(f"      {dt}  {d}  ${pnl:+,.0f}")

    # ---- TEST 2: Widen VWAP on HIGH IB ----
    print("\nRunning TEST 2: Widen VWAP proximity to 30% on HIGH IB (>=250)...")
    t2_metrics, t2_trades = run_variant(
        "TEST 2", df, full_df, instrument, filters,
        relax_fade=False, widen_vwap=True)
    print_summary("TEST 2: Widen VWAP (30% on HIGH)", t2_metrics, t2_trades)

    new_t2 = find_new_trades(base_trades, t2_trades)
    removed_t2 = find_new_trades(t2_trades, base_trades)
    if new_t2:
        print(f"\n  +++ {len(new_t2)} NEW trades added vs baseline:")
        for t in new_t2:
            dt = str(getattr(t, 'entry_time', '?'))[:10]
            d = getattr(t, 'direction', '?')
            pnl = getattr(t, 'net_pnl', 0)
            print(f"      {dt}  {d}  ${pnl:+,.0f}")
    if removed_t2:
        print(f"\n  --- {len(removed_t2)} trades REMOVED vs baseline:")
        for t in removed_t2:
            dt = str(getattr(t, 'entry_time', '?'))[:10]
            d = getattr(t, 'direction', '?')
            pnl = getattr(t, 'net_pnl', 0)
            print(f"      {dt}  {d}  ${pnl:+,.0f}")

    # ---- TEST 3: Combined ----
    print("\nRunning TEST 3: Combined (Test 1 + Test 2)...")
    t3_metrics, t3_trades = run_variant(
        "TEST 3", df, full_df, instrument, filters,
        relax_fade=True, widen_vwap=True)
    print_summary("TEST 3: Combined (relax FADE + widen VWAP)", t3_metrics, t3_trades)

    new_t3 = find_new_trades(base_trades, t3_trades)
    removed_t3 = find_new_trades(t3_trades, base_trades)
    if new_t3:
        print(f"\n  +++ {len(new_t3)} NEW trades added vs baseline:")
        for t in new_t3:
            dt = str(getattr(t, 'entry_time', '?'))[:10]
            d = getattr(t, 'direction', '?')
            pnl = getattr(t, 'net_pnl', 0)
            print(f"      {dt}  {d}  ${pnl:+,.0f}")
    if removed_t3:
        print(f"\n  --- {len(removed_t3)} trades REMOVED vs baseline:")
        for t in removed_t3:
            dt = str(getattr(t, 'entry_time', '?'))[:10]
            d = getattr(t, 'direction', '?')
            pnl = getattr(t, 'net_pnl', 0)
            print(f"      {dt}  {d}  ${pnl:+,.0f}")

    # ---- Final comparison table ----
    print("\n\n" + "="*60)
    print("  SUMMARY COMPARISON TABLE")
    print("="*60)
    print(f"  {'Variant':<30} {'Trades':>7} {'WR%':>7} {'Net P&L':>10} {'Max DD':>9} {'PF':>6}")
    print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*10} {'-'*9} {'-'*6}")

    def row(label, trades, metrics):
        wins = [t for t in trades if t.net_pnl > 0]
        wr = len(wins) / len(trades) * 100 if trades else 0
        net = sum(t.net_pnl for t in trades)
        mdd = metrics.get('max_drawdown', 0)
        pf = metrics.get('profit_factor', 0)
        print(f"  {label:<30} {len(trades):>7} {wr:>6.1f}% {net:>+10,.0f} {mdd:>9,.0f} {pf:>6.2f}")

    row("Baseline", base_trades, base_metrics)
    row("Test 1: Relax FADE", t1_trades, t1_metrics)
    row("Test 2: Widen VWAP (HIGH)", t2_trades, t2_metrics)
    row("Test 3: Combined", t3_trades, t3_metrics)

    print("\nDelta vs Baseline:")
    def delta_row(label, trades, metrics):
        base_net = sum(t.net_pnl for t in base_trades)
        var_net = sum(t.net_pnl for t in trades)
        base_trades_n = len(base_trades)
        var_trades_n = len(trades)
        base_wins = len([t for t in base_trades if t.net_pnl > 0])
        var_wins = len([t for t in trades if t.net_pnl > 0])
        base_wr = base_wins / base_trades_n * 100 if base_trades_n else 0
        var_wr = var_wins / var_trades_n * 100 if var_trades_n else 0
        d_trades = var_trades_n - base_trades_n
        d_wr = var_wr - base_wr
        d_net = var_net - base_net
        sign_t = '+' if d_trades >= 0 else ''
        sign_w = '+' if d_wr >= 0 else ''
        sign_n = '+' if d_net >= 0 else ''
        print(f"  {label:<30} {sign_t}{d_trades:>6}  {sign_w}{d_wr:>5.1f}%  {sign_n}${d_net:>+9,.0f}")

    print(f"  {'Variant':<30} {'dTrades':>7} {'dWR':>7} {'dNet':>11}")
    print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*11}")
    delta_row("Test 1: Relax FADE", t1_trades, t1_metrics)
    delta_row("Test 2: Widen VWAP (HIGH)", t2_trades, t2_metrics)
    delta_row("Test 3: Combined", t3_trades, t3_metrics)

    print("\nDone.")


if __name__ == '__main__':
    import io
    import contextlib

    output_path = Path(__file__).resolve().parent.parent / 'output' / 'ab_test_results.txt'
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main()
    result_text = buf.getvalue()
    print(result_text)  # also print to console
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(result_text)
