"""
Regime-Aware Directional Lean Analysis

Hypothesis: NQ's "long bias" isn't constant — it depends on the regime period.
During bear runs (EMA20 < EMA50, down-trending), shorts should work better.
During bull runs, longs dominate. If we can detect the regime and lean
directionally, we unlock shorts during bear periods without hurting long WR.

This analysis:
  1. Segments ALL trades by regime (bull vs bear)
  2. Tests IBH fade shorts specifically in bear regime
  3. Tests all strategies across regimes
  4. Looks at volatility (ATR-based) periods and directional bias
  5. Tests multi-signal regime detection (EMA + ATR + prior day + trend)
  6. Determines if a regime-aware system beats always-long
"""

import sys
from pathlib import Path
from collections import defaultdict
from datetime import time as _time
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.instruments import get_instrument
from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from config.constants import IB_BARS_1MIN
from strategy.day_type import classify_trend_strength, classify_day_type
from engine.backtest import BacktestEngine
from engine.execution import ExecutionModel
from engine.position import PositionManager
from strategy import (
    get_core_strategies, TrendDayBull, SuperTrendBull,
    PDayStrategy, BDayStrategy, MeanReversionVWAP,
    TrendDayBear, SuperTrendBear,
    LiquiditySweep, EMATrendFollow, ORBVwapBreakout,
)
from filters.composite import CompositeFilter
from filters.regime_filter import SimpleRegimeFilter
from reporting.metrics import compute_metrics

# Load data
instrument = get_instrument('MNQ')
df = load_csv('NQ')
df = filter_rth(df)
df = compute_all_features(df)

if 'session_date' not in df.columns:
    df['session_date'] = df['timestamp'].dt.date

sessions = sorted(df['session_date'].unique())

print("=" * 130)
print("  REGIME-AWARE DIRECTIONAL LEAN ANALYSIS")
print(f"  Data: {len(sessions)} sessions (Nov 18, 2025 - Feb 16, 2026)")
print("=" * 130)


# ============================================================================
# PHASE 1: MAP REGIME PER SESSION
# ============================================================================
print("\n" + "=" * 130)
print("  PHASE 1: SESSION-BY-SESSION REGIME MAPPING")
print("=" * 130)

session_regimes = {}  # date -> regime info

for session_date in sessions:
    sdf = df[df['session_date'] == session_date].copy()
    if len(sdf) < IB_BARS_1MIN + 10:
        continue

    ib_df = sdf.head(IB_BARS_1MIN)
    post_ib = sdf.iloc[IB_BARS_1MIN:]

    ib_high = ib_df['high'].max()
    ib_low = ib_df['low'].min()
    ib_range = ib_high - ib_low

    last_bar = sdf.iloc[-1]
    # Use end-of-IB bar for regime detection (EMAs need warmup)
    ib_end_bar = ib_df.iloc[-1]
    # Also try mid-session bar for more stable EMA
    mid_idx = min(len(post_ib) - 1, len(post_ib) // 2) if len(post_ib) > 0 else 0
    mid_bar = post_ib.iloc[mid_idx] if len(post_ib) > 0 else ib_end_bar

    # EMA regime — use end-of-IB bar (60 min in, EMAs have warmed up)
    ema_bull = None
    for check_bar in [ib_end_bar, mid_bar, last_bar]:
        ema20 = check_bar.get('ema20')
        ema50 = check_bar.get('ema50')
        if ema20 is not None and ema50 is not None:
            if not pd.isna(ema20) and not pd.isna(ema50):
                ema_bull = ema20 > ema50
                break

    # ATR — try multiple bars
    atr = 0
    for check_bar in [ib_end_bar, mid_bar, last_bar]:
        a = check_bar.get('atr14', 0)
        if a is not None and not pd.isna(a) and a > 0:
            atr = a
            break

    # Session outcome
    session_open = sdf.iloc[0]['open']
    session_close = last_bar['close']
    session_return = session_close - session_open
    session_return_pct = (session_return / session_open * 100) if session_open > 0 else 0

    # Session high/low for range analysis
    session_high = sdf['high'].max()
    session_low = sdf['low'].min()
    session_range = session_high - session_low

    # VWAP at close
    vwap_close = last_bar.get('vwap', session_close)
    close_vs_vwap = session_close - vwap_close if not pd.isna(vwap_close) else 0

    # Rolling 5-day return (momentum)
    idx = list(sessions).index(session_date)
    if idx >= 5:
        five_days_ago_date = sessions[idx - 5]
        five_ago_df = df[df['session_date'] == five_days_ago_date]
        if len(five_ago_df) > 0:
            five_ago_close = five_ago_df.iloc[-1]['close']
            rolling_5d_return = (session_close - five_ago_close) / five_ago_close * 100
        else:
            rolling_5d_return = 0
    else:
        rolling_5d_return = 0

    session_regimes[str(session_date)] = {
        'ema_bull': ema_bull,
        'atr': atr,
        'ib_range': ib_range,
        'session_return': session_return,
        'session_return_pct': session_return_pct,
        'close_vs_vwap': close_vs_vwap,
        'session_range': session_range,
        'rolling_5d_return': rolling_5d_return,
        'bullish_day': session_return > 0,
    }

# Print regime summary
bull_sessions = [d for d, r in session_regimes.items() if r['ema_bull'] is True]
bear_sessions = [d for d, r in session_regimes.items() if r['ema_bull'] is False]

unknown_sessions = [d for d, r in session_regimes.items() if r['ema_bull'] is None]

print(f"\nEMA20/EMA50 Regime:")
print(f"  Bull sessions: {len(bull_sessions)} ({len(bull_sessions)/max(len(session_regimes),1)*100:.0f}%)")
print(f"  Bear sessions: {len(bear_sessions)} ({len(bear_sessions)/max(len(session_regimes),1)*100:.0f}%)")
if unknown_sessions:
    print(f"  Unknown (NaN EMA): {len(unknown_sessions)}")

# Actual returns within each regime
bull_returns = [session_regimes[d]['session_return'] for d in bull_sessions]
bear_returns = [session_regimes[d]['session_return'] for d in bear_sessions]

if bull_returns:
    print(f"\n  Bull regime session returns:")
    print(f"    Mean: {np.mean(bull_returns):+.1f} pts  Median: {np.median(bull_returns):+.1f} pts")
    print(f"    Up days: {sum(1 for r in bull_returns if r > 0)}/{len(bull_returns)} ({sum(1 for r in bull_returns if r > 0)/len(bull_returns)*100:.0f}%)")
else:
    print(f"\n  Bull regime: no sessions detected")

if bear_returns:
    print(f"\n  Bear regime session returns:")
    print(f"    Mean: {np.mean(bear_returns):+.1f} pts  Median: {np.median(bear_returns):+.1f} pts")
    print(f"    Up days: {sum(1 for r in bear_returns if r > 0)}/{len(bear_returns)} ({sum(1 for r in bear_returns if r > 0)/len(bear_returns)*100:.0f}%)")
else:
    print(f"\n  Bear regime: no sessions detected")

# Volatility regimes
atrs = [r['atr'] for r in session_regimes.values() if r['atr'] > 0]
atr_median = np.median(atrs)
high_vol_sessions = [d for d, r in session_regimes.items() if r['atr'] > atr_median]
low_vol_sessions = [d for d, r in session_regimes.items() if r['atr'] <= atr_median and r['atr'] > 0]

print(f"\nVolatility Regime (ATR14 median = {atr_median:.1f}):")
print(f"  High vol sessions: {len(high_vol_sessions)}")
print(f"  Low vol sessions:  {len(low_vol_sessions)}")

hv_returns = [session_regimes[d]['session_return'] for d in high_vol_sessions]
lv_returns = [session_regimes[d]['session_return'] for d in low_vol_sessions]
if hv_returns:
    print(f"  High vol mean return: {np.mean(hv_returns):+.1f} pts (up {sum(1 for r in hv_returns if r > 0)/len(hv_returns)*100:.0f}%)")
if lv_returns:
    print(f"  Low vol mean return:  {np.mean(lv_returns):+.1f} pts (up {sum(1 for r in lv_returns if r > 0)/len(lv_returns)*100:.0f}%)")

# 5-day momentum regimes
momentum_bull = [d for d, r in session_regimes.items() if r['rolling_5d_return'] > 0.5]
momentum_bear = [d for d, r in session_regimes.items() if r['rolling_5d_return'] < -0.5]
momentum_neutral = [d for d, r in session_regimes.items()
                     if -0.5 <= r['rolling_5d_return'] <= 0.5]

print(f"\n5-Day Momentum Regime:")
print(f"  Bullish momentum (>+0.5%): {len(momentum_bull)} sessions")
print(f"  Bearish momentum (<-0.5%): {len(momentum_bear)} sessions")
print(f"  Neutral (-0.5% to +0.5%):  {len(momentum_neutral)} sessions")

if momentum_bull:
    mb_ret = [session_regimes[d]['session_return'] for d in momentum_bull]
    print(f"  Bull momentum → up {sum(1 for r in mb_ret if r > 0)/len(mb_ret)*100:.0f}%, mean {np.mean(mb_ret):+.1f} pts")
if momentum_bear:
    mb_ret = [session_regimes[d]['session_return'] for d in momentum_bear]
    print(f"  Bear momentum → up {sum(1 for r in mb_ret if r > 0)/len(mb_ret)*100:.0f}%, mean {np.mean(mb_ret):+.1f} pts")


# ============================================================================
# PHASE 2: RUN ALL STRATEGIES WITH NO FILTER — TAG TRADES BY REGIME
# ============================================================================
print("\n\n" + "=" * 130)
print("  PHASE 2: ALL STRATEGIES — TRADES SEGMENTED BY REGIME")
print("=" * 130)

# Run backtest with ALL strategies (including bears) and NO filter
all_strategies = [
    TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(),
    TrendDayBear(), SuperTrendBear(),
    MeanReversionVWAP(), LiquiditySweep(), EMATrendFollow(), ORBVwapBreakout(),
]

execution = ExecutionModel(instrument, slippage_ticks=1)
position_mgr = PositionManager(account_size=150000)

engine = BacktestEngine(
    instrument=instrument,
    strategies=all_strategies,
    filters=None,
    execution=execution,
    position_mgr=position_mgr,
    risk_per_trade=400,
    max_contracts=5,
)

result = engine.run(df, verbose=False)
trades = result.trades

print(f"\nTotal trades (all strategies, no filter): {len(trades)}")

# Tag each trade with its regime
for t in trades:
    date_str = str(t.session_date)[:10]
    regime = session_regimes.get(date_str, {})
    t._regime_ema = regime.get('ema_bull')
    t._regime_atr = regime.get('atr', 0)
    t._regime_5d = regime.get('rolling_5d_return', 0)

# Segment by direction AND regime
def analyze_segment(label, trade_list):
    """Print stats for a segment of trades."""
    if not trade_list:
        print(f"  {label:<55s} {'no trades':>10s}")
        return {'wr': 0, 'pnl': 0, 'n': 0, 'exp': 0}

    wins = [t for t in trade_list if t.net_pnl > 0]
    wr = len(wins) / len(trade_list) * 100
    pnl = sum(t.net_pnl for t in trade_list)
    exp = pnl / len(trade_list)
    avg_w = np.mean([t.net_pnl for t in wins]) if wins else 0
    avg_l = np.mean([t.net_pnl for t in trade_list if t.net_pnl <= 0]) if len(trade_list) > len(wins) else 0

    print(f"  {label:<55s} {len(trade_list):>3d} trades  {wr:>5.1f}% WR  ${pnl:>8,.0f}  ${exp:>6,.0f}/trade  "
          f"AvgW ${avg_w:>6,.0f}  AvgL ${avg_l:>6,.0f}")
    return {'wr': wr, 'pnl': pnl, 'n': len(trade_list), 'exp': exp}

# --- By Direction x Regime ---
print(f"\n--- Direction x EMA Regime ---")
long_bull = [t for t in trades if t.direction == 'LONG' and t._regime_ema is True]
long_bear = [t for t in trades if t.direction == 'LONG' and t._regime_ema is False]
short_bull = [t for t in trades if t.direction == 'SHORT' and t._regime_ema is True]
short_bear = [t for t in trades if t.direction == 'SHORT' and t._regime_ema is False]

analyze_segment("LONG  in BULL regime (EMA20 > EMA50)", long_bull)
analyze_segment("LONG  in BEAR regime (EMA20 < EMA50)", long_bear)
analyze_segment("SHORT in BULL regime (EMA20 > EMA50)", short_bull)
analyze_segment("SHORT in BEAR regime (EMA20 < EMA50)", short_bear)

# --- By Direction x 5-Day Momentum ---
print(f"\n--- Direction x 5-Day Momentum ---")
long_mom_bull = [t for t in trades if t.direction == 'LONG' and t._regime_5d > 0.5]
long_mom_bear = [t for t in trades if t.direction == 'LONG' and t._regime_5d < -0.5]
long_mom_neut = [t for t in trades if t.direction == 'LONG' and -0.5 <= t._regime_5d <= 0.5]
short_mom_bull = [t for t in trades if t.direction == 'SHORT' and t._regime_5d > 0.5]
short_mom_bear = [t for t in trades if t.direction == 'SHORT' and t._regime_5d < -0.5]
short_mom_neut = [t for t in trades if t.direction == 'SHORT' and -0.5 <= t._regime_5d <= 0.5]

analyze_segment("LONG  in bullish 5d momentum (>+0.5%)", long_mom_bull)
analyze_segment("LONG  in bearish 5d momentum (<-0.5%)", long_mom_bear)
analyze_segment("LONG  in neutral 5d momentum", long_mom_neut)
analyze_segment("SHORT in bullish 5d momentum (>+0.5%)", short_mom_bull)
analyze_segment("SHORT in bearish 5d momentum (<-0.5%)", short_mom_bear)
analyze_segment("SHORT in neutral 5d momentum", short_mom_neut)

# --- By Direction x Volatility ---
print(f"\n--- Direction x Volatility (ATR14) ---")
long_hv = [t for t in trades if t.direction == 'LONG' and t._regime_atr > atr_median]
long_lv = [t for t in trades if t.direction == 'LONG' and 0 < t._regime_atr <= atr_median]
short_hv = [t for t in trades if t.direction == 'SHORT' and t._regime_atr > atr_median]
short_lv = [t for t in trades if t.direction == 'SHORT' and 0 < t._regime_atr <= atr_median]

analyze_segment("LONG  in HIGH volatility (ATR > median)", long_hv)
analyze_segment("LONG  in LOW volatility (ATR <= median)", long_lv)
analyze_segment("SHORT in HIGH volatility (ATR > median)", short_hv)
analyze_segment("SHORT in LOW volatility (ATR <= median)", short_lv)


# ============================================================================
# PHASE 3: PER-STRATEGY BREAKDOWN BY REGIME
# ============================================================================
print("\n\n" + "=" * 130)
print("  PHASE 3: PER-STRATEGY x REGIME BREAKDOWN")
print("=" * 130)

strat_groups = defaultdict(list)
for t in trades:
    strat_groups[t.strategy_name].append(t)

for sname in sorted(strat_groups.keys()):
    strades = strat_groups[sname]
    print(f"\n  === {sname} ===")

    # By direction
    longs = [t for t in strades if t.direction == 'LONG']
    shorts = [t for t in strades if t.direction == 'SHORT']

    if longs:
        l_bull = [t for t in longs if t._regime_ema is True]
        l_bear = [t for t in longs if t._regime_ema is False]
        analyze_segment(f"  LONG  (all)", longs)
        if l_bull:
            analyze_segment(f"    LONG in BULL regime", l_bull)
        if l_bear:
            analyze_segment(f"    LONG in BEAR regime", l_bear)

    if shorts:
        s_bull = [t for t in shorts if t._regime_ema is True]
        s_bear = [t for t in shorts if t._regime_ema is False]
        analyze_segment(f"  SHORT (all)", shorts)
        if s_bull:
            analyze_segment(f"    SHORT in BULL regime", s_bull)
        if s_bear:
            analyze_segment(f"    SHORT in BEAR regime", s_bear)


# ============================================================================
# PHASE 4: IBH FADE SHORT — BEAR REGIME ONLY
# ============================================================================
print("\n\n" + "=" * 130)
print("  PHASE 4: IBH FADE SHORT — SPECIFICALLY IN BEAR REGIME")
print("=" * 130)

def simulate_ibh_fade_regime(config_name, allowed_day_types, target_mode,
                              regime_filter='bear_only', volatility_filter=None):
    """
    IBH fade with regime awareness.
    regime_filter: 'bear_only', 'bear_momentum', 'high_vol_bear', 'any'
    """
    trades_out = []

    for session_date in sessions:
        date_str = str(session_date)
        regime = session_regimes.get(date_str, {})

        # Regime gating
        if regime_filter == 'bear_only':
            if regime.get('ema_bull') is not False:
                continue
        elif regime_filter == 'bear_momentum':
            if regime.get('rolling_5d_return', 0) >= -0.5:
                continue
        elif regime_filter == 'high_vol_bear':
            if regime.get('ema_bull') is not False:
                continue
            if regime.get('atr', 0) <= atr_median:
                continue
        elif regime_filter == 'low_vol_bear':
            if regime.get('ema_bull') is not False:
                continue
            if regime.get('atr', 0) > atr_median:
                continue

        sdf = df[df['session_date'] == session_date].copy()
        if len(sdf) < IB_BARS_1MIN + 10:
            continue

        ib_df = sdf.head(IB_BARS_1MIN)
        post_ib = sdf.iloc[IB_BARS_1MIN:]

        ib_high = ib_df['high'].max()
        ib_low = ib_df['low'].min()
        ib_range = ib_high - ib_low
        ib_mid = (ib_high + ib_low) / 2

        if ib_range <= 0 or ib_range > 400:
            continue

        vah = ib_high - (ib_range * 0.15)
        val = ib_low + (ib_range * 0.15)

        entry_taken = False
        entry_price = None
        stop_price = None
        target_price = None
        entry_bar_idx = None
        session_high = ib_high
        exec_model = ExecutionModel(instrument, slippage_ticks=1)

        for i, (idx, bar) in enumerate(post_ib.iterrows()):
            price = bar['close']
            bar_time = bar['timestamp'].time() if 'timestamp' in bar.index else None

            if bar_time and bar_time >= _time(14, 0):
                if entry_taken:
                    exit_fill = exec_model.fill_exit('SHORT', price)
                    gross, comm, slip, net = exec_model.calculate_net_pnl(
                        'SHORT', entry_price, exit_fill, 5)
                    trades_out.append({
                        'date': date_str, 'entry': entry_price, 'exit': exit_fill,
                        'net_pnl': net, 'exit_reason': 'EOD', 'bars': i - entry_bar_idx,
                        'regime': 'bear', 'config': config_name,
                    })
                    entry_taken = False
                break

            if bar['high'] > session_high:
                session_high = bar['high']

            # Day type
            if price > ib_high:
                ib_dir = 'BULL'
                ext = (price - ib_mid) / ib_range
            elif price < ib_low:
                ib_dir = 'BEAR'
                ext = (ib_mid - price) / ib_range
            else:
                ib_dir = 'INSIDE'
                ext = 0.0

            strength = classify_trend_strength(ext)
            day_type = classify_day_type(ib_high, ib_low, price, ib_dir, strength)
            dt_val = day_type.value if hasattr(day_type, 'value') else str(day_type)

            if entry_taken:
                if bar['high'] >= stop_price:
                    exit_fill = exec_model.fill_exit('SHORT', stop_price)
                    gross, comm, slip, net = exec_model.calculate_net_pnl(
                        'SHORT', entry_price, exit_fill, 5)
                    trades_out.append({
                        'date': date_str, 'entry': entry_price, 'exit': exit_fill,
                        'net_pnl': net, 'exit_reason': 'STOP', 'bars': i - entry_bar_idx,
                        'regime': 'bear', 'config': config_name,
                    })
                    entry_taken = False
                    continue

                if bar['low'] <= target_price:
                    exit_fill = exec_model.fill_exit('SHORT', target_price)
                    gross, comm, slip, net = exec_model.calculate_net_pnl(
                        'SHORT', entry_price, exit_fill, 5)
                    trades_out.append({
                        'date': date_str, 'entry': entry_price, 'exit': exit_fill,
                        'net_pnl': net, 'exit_reason': 'TARGET', 'bars': i - entry_bar_idx,
                        'regime': 'bear', 'config': config_name,
                    })
                    entry_taken = False
                    continue
                continue

            if dt_val not in allowed_day_types:
                continue

            if session_high > ib_high + (ib_range * 0.3):
                continue

            # Rejection bar
            if not (bar['high'] >= ib_high - (ib_range * 0.05) and price < ib_high):
                continue

            bar_range_val = bar['high'] - bar['low']
            if bar_range_val > 0:
                upper_wick = bar['high'] - max(bar['open'], price)
                if upper_wick / bar_range_val < 0.25:
                    continue

            delta = bar.get('delta', 0)
            if pd.isna(delta):
                delta = 0
            if delta >= 0:
                continue

            entry_raw = price
            entry_fill = exec_model.fill_entry('SHORT', entry_raw)
            entry_price = entry_fill
            stop_price = ib_high + (ib_range * 0.15)
            stop_price = max(stop_price, entry_price + 15.0)

            if target_mode == 'ib_mid':
                target_price = ib_mid
            elif target_mode == 'vwap':
                vwap_val = bar.get('vwap', ib_mid)
                target_price = vwap_val if not pd.isna(vwap_val) else ib_mid
            elif target_mode == 'val':
                target_price = val
            elif target_mode == 'vah':
                target_price = vah
            else:
                target_price = ib_mid

            risk = stop_price - entry_price
            reward = entry_price - target_price
            if reward <= 0 or risk <= 0:
                entry_price = None
                continue

            entry_taken = True
            entry_bar_idx = i

        if entry_taken and len(post_ib) > 0:
            last_bar = post_ib.iloc[-1]
            exit_fill = exec_model.fill_exit('SHORT', last_bar['close'])
            gross, comm, slip, net = exec_model.calculate_net_pnl(
                'SHORT', entry_price, exit_fill, 5)
            trades_out.append({
                'date': date_str, 'entry': entry_price, 'exit': exit_fill,
                'net_pnl': net, 'exit_reason': 'EOD', 'bars': len(post_ib) - entry_bar_idx,
                'regime': 'bear', 'config': config_name,
            })

    return trades_out


# Test IBH fade in different regime contexts
ibh_configs = [
    # (name, day_types, target, regime_filter)
    ('A: Bear EMA, B-Day IBH→IB mid',
     ['b_day'], 'ib_mid', 'bear_only'),
    ('B: Bear EMA, B-Day+Neutral IBH→IB mid',
     ['b_day', 'neutral'], 'ib_mid', 'bear_only'),
    ('C: Bear EMA, B-Day IBH→VWAP',
     ['b_day'], 'vwap', 'bear_only'),
    ('D: Bear EMA, B-Day IBH→VAH (scalp)',
     ['b_day'], 'vah', 'bear_only'),
    ('E: Bear Momentum (<-0.5%), B-Day IBH→IB mid',
     ['b_day'], 'ib_mid', 'bear_momentum'),
    ('F: Bear Momentum, B-Day+Neutral IBH→IB mid',
     ['b_day', 'neutral'], 'ib_mid', 'bear_momentum'),
    ('G: High Vol Bear, B-Day IBH→IB mid',
     ['b_day'], 'ib_mid', 'high_vol_bear'),
    ('H: Low Vol Bear, B-Day IBH→IB mid',
     ['b_day'], 'ib_mid', 'low_vol_bear'),
    ('I: Bear EMA, All Balance IBH→VWAP',
     ['b_day', 'neutral', 'p_day'], 'vwap', 'bear_only'),
    ('J: Bear Momentum, All Balance IBH→VWAP',
     ['b_day', 'neutral', 'p_day'], 'vwap', 'bear_momentum'),
]

print(f"\n{'Config':<52s} {'Trades':>6s} {'W':>3s} {'L':>3s} {'WR%':>6s} {'Net PnL':>10s} {'AvgWin':>8s} {'AvgLoss':>8s} {'Expect':>8s}")
print("-" * 115)

ibh_results = []
for name, day_types, target, regime_f in ibh_configs:
    t_list = simulate_ibh_fade_regime(name, day_types, target, regime_f)
    wins = [t for t in t_list if t['net_pnl'] > 0]
    losses = [t for t in t_list if t['net_pnl'] <= 0]
    pnl = sum(t['net_pnl'] for t in t_list)
    wr = len(wins) / len(t_list) * 100 if t_list else 0
    avg_w = np.mean([t['net_pnl'] for t in wins]) if wins else 0
    avg_l = np.mean([t['net_pnl'] for t in losses]) if losses else 0
    exp = pnl / len(t_list) if t_list else 0

    print(f"{name:<52s} {len(t_list):>6d} {len(wins):>3d} {len(losses):>3d} "
          f"{wr:>5.1f}% ${pnl:>8,.0f} ${avg_w:>6,.0f} ${avg_l:>6,.0f} ${exp:>6,.0f}")

    ibh_results.append({
        'name': name, 'trades': len(t_list), 'wins': len(wins),
        'losses': len(losses), 'wr': wr, 'pnl': pnl, 'exp': exp,
        'trade_list': t_list,
    })

print("-" * 115)


# ============================================================================
# PHASE 5: REGIME-AWARE FULL PORTFOLIO COMPARISON
# ============================================================================
print("\n\n" + "=" * 130)
print("  PHASE 5: FULL PORTFOLIO — REGIME-AWARE DIRECTIONAL LEAN")
print("=" * 130)

def run_portfolio(label, strategies, filters, max_contracts=5):
    """Run a full portfolio backtest."""
    exec_m = ExecutionModel(instrument, slippage_ticks=1)
    pos_m = PositionManager(account_size=150000)
    eng = BacktestEngine(
        instrument=instrument, strategies=strategies, filters=filters,
        execution=exec_m, position_mgr=pos_m,
        risk_per_trade=400, max_contracts=max_contracts,
    )
    res = eng.run(df, verbose=False)
    t = res.trades
    wins = [x for x in t if x.net_pnl > 0]
    pnl = sum(x.net_pnl for x in t)
    wr = len(wins) / len(t) * 100 if t else 0

    daily_pnl = defaultdict(float)
    for x in t:
        daily_pnl[x.session_date] += x.net_pnl
    win_days = sum(1 for v in daily_pnl.values() if v > 0)
    active_days = len(daily_pnl)

    return {
        'label': label, 'trades': len(t), 'wins': len(wins),
        'wr': wr, 'pnl': pnl, 'exp': pnl / len(t) if t else 0,
        'win_days': win_days, 'active_days': active_days,
        'max_dd': pos_m.max_drawdown_seen,
    }

portfolio_configs = []

# 1) Current best: Core + MeanRev LONG only
portfolio_configs.append(run_portfolio(
    'A: Current Best (Core+MeanRev LONG only)',
    [TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(), MeanReversionVWAP()],
    CompositeFilter([SimpleRegimeFilter(
        longs_in_bull=True, longs_in_bear=True,
        shorts_in_bull=False, shorts_in_bear=False)]),
))

# 2) Regime-aware: Longs always, Shorts only in bear
portfolio_configs.append(run_portfolio(
    'B: Longs always + Shorts in BEAR only',
    [TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(),
     MeanReversionVWAP(), TrendDayBear(), SuperTrendBear()],
    CompositeFilter([SimpleRegimeFilter(
        longs_in_bull=True, longs_in_bear=True,
        shorts_in_bull=False, shorts_in_bear=True)]),
))

# 3) Strict regime: Longs in bull, Shorts in bear
portfolio_configs.append(run_portfolio(
    'C: Strict regime (Longs BULL, Shorts BEAR)',
    [TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(),
     MeanReversionVWAP(), TrendDayBear(), SuperTrendBear()],
    CompositeFilter([SimpleRegimeFilter(
        longs_in_bull=True, longs_in_bear=False,
        shorts_in_bull=False, shorts_in_bear=True)]),
))

# 4) All strategies, regime-gated
portfolio_configs.append(run_portfolio(
    'D: All 10 strats, regime-gated',
    [TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(),
     MeanReversionVWAP(), TrendDayBear(), SuperTrendBear(),
     LiquiditySweep(), EMATrendFollow(), ORBVwapBreakout()],
    CompositeFilter([SimpleRegimeFilter(
        longs_in_bull=True, longs_in_bear=True,
        shorts_in_bull=False, shorts_in_bear=True)]),
))

# 5) Longs always + all research, shorts in bear only
portfolio_configs.append(run_portfolio(
    'E: Core+Research LONG always, Shorts BEAR',
    [TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(),
     MeanReversionVWAP(), LiquiditySweep(), EMATrendFollow(), ORBVwapBreakout(),
     TrendDayBear(), SuperTrendBear()],
    CompositeFilter([SimpleRegimeFilter(
        longs_in_bull=True, longs_in_bear=True,
        shorts_in_bull=False, shorts_in_bear=True)]),
))

# 6) No filter at all (everything)
portfolio_configs.append(run_portfolio(
    'F: All strategies NO filter',
    [TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(),
     MeanReversionVWAP(), TrendDayBear(), SuperTrendBear(),
     LiquiditySweep(), EMATrendFollow(), ORBVwapBreakout()],
    None,
))

print(f"\n{'Config':<50s} {'Trades':>6s} {'WR%':>6s} {'Net PnL':>10s} {'Expect':>8s} {'WinDays':>10s} {'MaxDD':>8s}")
print("-" * 110)

for r in portfolio_configs:
    wd_str = f"{r['win_days']}/{r['active_days']}"
    print(f"{r['label']:<50s} {r['trades']:>6d} {r['wr']:>5.1f}% ${r['pnl']:>8,.0f} ${r['exp']:>6,.0f} "
          f"{wd_str:>10s} ${r['max_dd']:>7,.0f}")

print("-" * 110)


# ============================================================================
# PHASE 6: VERDICT
# ============================================================================
print("\n\n" + "=" * 130)
print("  VERDICT: DOES REGIME-AWARE DIRECTIONAL LEAN IMPROVE THE PORTFOLIO?")
print("=" * 130)

current_best = portfolio_configs[0]  # A: Current LONG only
regime_aware = portfolio_configs[1]  # B: Longs + Shorts in bear
strict_regime = portfolio_configs[2] # C: Strict regime

print(f"""
  CURRENT BEST (LONG only):
    {current_best['wr']:.1f}% WR | {current_best['trades']} trades | ${current_best['pnl']:,.0f} | ${current_best['exp']:,.0f}/trade | MaxDD ${current_best['max_dd']:,.0f}

  REGIME-AWARE (Longs always + Shorts in bear):
    {regime_aware['wr']:.1f}% WR | {regime_aware['trades']} trades | ${regime_aware['pnl']:,.0f} | ${regime_aware['exp']:,.0f}/trade | MaxDD ${regime_aware['max_dd']:,.0f}

  STRICT REGIME (Longs in bull only, Shorts in bear only):
    {strict_regime['wr']:.1f}% WR | {strict_regime['trades']} trades | ${strict_regime['pnl']:,.0f} | ${strict_regime['exp']:,.0f}/trade | MaxDD ${strict_regime['max_dd']:,.0f}
""")

if regime_aware['pnl'] > current_best['pnl'] and regime_aware['wr'] >= 65:
    pnl_diff = regime_aware['pnl'] - current_best['pnl']
    print(f"  REGIME-AWARE BEATS LONG-ONLY by ${pnl_diff:,.0f}")
    print(f"  Adding shorts in bear regime IS profitable on this data.")
    if regime_aware['wr'] >= 70:
        print(f"  WR still >= 70% — CAN add to Lightning playbook.")
    else:
        print(f"  BUT WR drops below 70% — risky for Lightning 70% target.")
else:
    print(f"  LONG-ONLY STILL WINS (or regime-aware has lower WR).")
    print(f"  NQ shorts hurt even in bear regime for this data period.")
    print(f"  The long bias is structural, not just cyclical.")

# Check if strict regime helps (only longs in bull, only shorts in bear)
if strict_regime['pnl'] > current_best['pnl']:
    print(f"\n  STRICT REGIME note: Restricting longs to BULL-only would give")
    print(f"  ${strict_regime['pnl']:,.0f} vs ${current_best['pnl']:,.0f} (always-long)")
    if strict_regime['trades'] < current_best['trades'] * 0.7:
        print(f"  BUT only {strict_regime['trades']} trades — too few for consistency rule.")
else:
    print(f"\n  STRICT REGIME LOSES: Blocking longs in bear regime costs")
    print(f"  ${current_best['pnl'] - strict_regime['pnl']:,.0f} — those bear-regime longs were winners!")
