"""
IBH Sweep + Failure SHORT — Day Type Filtered Study

Validating findings from parallel research thread:
  - b_day IBH sweep shorts: 4/4 = 100% WR, +291 pts
  - p_day IBH sweep shorts: 3/13 = 23% WR, -337 pts (death zone)
  - Day type IS the filter, not regime or HTF confluence

Sweep definition: Bar high goes 5+ pts above IBH, then bar CLOSES back below IBH.
This is a liquidity grab — stops above IBH get run, then price reverses.

Entry model:
  - SHORT at close of failure candle (bar that wicks above IBH but closes below)
  - Stop: sweep high + 15% of IB range
  - Target: IB midpoint
  - Max 1 trade per session, first sweep only
  - Time window: after IB (10:30) through 14:00
  - Day type filter: b_day only (primary), b_day + neutral (secondary)

This study:
  1. Finds ALL bar-level IBH sweep events (5+ pts above, close below)
  2. Tags each with the current day type classification
  3. Segments results by day type
  4. Simulates the exact entry model
  5. Checks HTF confluence (PDH, London/overnight high)
  6. Tests combined impact on the 70% WR playbook
"""

import sys
from pathlib import Path
from collections import defaultdict
from datetime import time as _time, timedelta
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
from engine.execution import ExecutionModel
from engine.backtest import BacktestEngine
from engine.position import PositionManager
from strategy import (
    TrendDayBull, SuperTrendBull, PDayStrategy, BDayStrategy, MeanReversionVWAP,
)
from filters.composite import CompositeFilter
from filters.regime_filter import SimpleRegimeFilter

# Load data
instrument = get_instrument('MNQ')
df_raw = load_csv('NQ')
df = filter_rth(df_raw)
df = compute_all_features(df)

if 'session_date' not in df.columns:
    df['session_date'] = df['timestamp'].dt.date

sessions = sorted(df['session_date'].unique())
execution = ExecutionModel(instrument, slippage_ticks=1)

# Compute prior day levels and overnight highs
prior_levels = {}
overnight_highs = {}
prev_high = None
prev_low = None

for session_date in sessions:
    sdf = df[df['session_date'] == session_date]
    if len(sdf) == 0:
        continue
    prior_levels[str(session_date)] = {
        'pdh': prev_high, 'pdl': prev_low,
    }
    prev_high = sdf['high'].max()
    prev_low = sdf['low'].min()

# Overnight (globex) highs
df_raw_ts = df_raw.copy()
if 'timestamp' not in df_raw_ts.columns and isinstance(df_raw_ts.index, pd.DatetimeIndex):
    df_raw_ts['timestamp'] = df_raw_ts.index
df_raw_ts['timestamp'] = pd.to_datetime(df_raw_ts['timestamp'])
df_raw_ts['time'] = df_raw_ts['timestamp'].dt.time

for session_date in sessions:
    prior_day = session_date - timedelta(days=1)
    eth_mask = (
        ((df_raw_ts['timestamp'].dt.date == prior_day) & (df_raw_ts['time'] >= _time(18, 0))) |
        ((df_raw_ts['timestamp'].dt.date == session_date) & (df_raw_ts['time'] < _time(9, 30)))
    )
    eth_df = df_raw_ts[eth_mask]
    overnight_highs[str(session_date)] = eth_df['high'].max() if len(eth_df) > 0 else None


SWEEP_MIN_PTS = 5.0  # Minimum extension above IBH to qualify as sweep

print("=" * 130)
print("  IBH SWEEP + FAILURE SHORT — DAY TYPE FILTERED STUDY")
print(f"  Data: {len(sessions)} sessions | Sweep threshold: {SWEEP_MIN_PTS}+ pts above IBH")
print("=" * 130)


# ============================================================================
# PHASE 1: FIND ALL BAR-LEVEL IBH SWEEP EVENTS
# ============================================================================
print("\n" + "=" * 130)
print("  PHASE 1: ALL IBH SWEEP EVENTS (bar-level)")
print("=" * 130)

all_sweeps = []

for session_date in sessions:
    date_str = str(session_date)
    sdf = df[df['session_date'] == session_date].copy()
    if len(sdf) < IB_BARS_1MIN + 10:
        continue

    ib_df = sdf.head(IB_BARS_1MIN)
    post_ib = sdf.iloc[IB_BARS_1MIN:]

    ib_high = ib_df['high'].max()
    ib_low = ib_df['low'].min()
    ib_range = ib_high - ib_low
    ib_mid = (ib_high + ib_low) / 2

    if ib_range <= 0:
        continue

    pdh = prior_levels.get(date_str, {}).get('pdh')
    ovn_high = overnight_highs.get(date_str)

    session_high = ib_high
    sweep_taken_this_session = False

    for i, (idx, bar) in enumerate(post_ib.iterrows()):
        price = bar['close']
        bar_time = bar['timestamp'].time() if 'timestamp' in bar.index else None

        if bar['high'] > session_high:
            session_high = bar['high']

        # Time window: after IB (10:30) through 14:00
        if bar_time and (bar_time < _time(10, 30) or bar_time >= _time(14, 0)):
            continue

        # SWEEP DETECTION: bar high goes 5+ pts above IBH, close stays below
        sweep_pts = bar['high'] - ib_high
        if sweep_pts < SWEEP_MIN_PTS:
            continue
        if price >= ib_high:
            continue  # Close must be below IBH for a failure

        # This is a sweep + failure bar
        # Classify current day type
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

        # Delta on the sweep bar
        delta = bar.get('delta', 0)
        if pd.isna(delta):
            delta = 0

        # Wick analysis
        bar_range = bar['high'] - bar['low']
        upper_wick = bar['high'] - max(bar['open'], price)
        wick_ratio = upper_wick / bar_range if bar_range > 0 else 0

        # HTF confluence
        near_pdh = False
        near_ovn_high = False
        zone_buffer = ib_range * 0.20

        if pdh is not None:
            if abs(bar['high'] - pdh) <= zone_buffer:
                near_pdh = True
        if ovn_high is not None:
            if abs(bar['high'] - ovn_high) <= zone_buffer:
                near_ovn_high = True

        # B-day confidence
        b_day_conf = 0.0
        day_conf = bar.get('day_confidence', None)
        if hasattr(day_conf, 'b_day'):
            b_day_conf = day_conf.b_day

        sweep = {
            'date': date_str,
            'bar_idx': i,
            'bar_time': str(bar_time),
            'ib_high': ib_high,
            'ib_low': ib_low,
            'ib_range': ib_range,
            'ib_mid': ib_mid,
            'sweep_pts': sweep_pts,
            'sweep_mult': sweep_pts / ib_range if ib_range > 0 else 0,
            'sweep_high': bar['high'],
            'close': price,
            'day_type': dt_val,
            'delta': delta,
            'wick_ratio': wick_ratio,
            'near_pdh': near_pdh,
            'near_ovn_high': near_ovn_high,
            'first_sweep': not sweep_taken_this_session,
            'session_high': session_high,
        }
        all_sweeps.append(sweep)

        if not sweep_taken_this_session:
            sweep_taken_this_session = True

# Summary
print(f"\nTotal IBH sweep events: {len(all_sweeps)}")
sessions_with_sweeps = len(set(s['date'] for s in all_sweeps))
print(f"Sessions with at least one sweep: {sessions_with_sweeps}/{len(sessions)}")

# By day type
dt_counts = defaultdict(list)
for s in all_sweeps:
    dt_counts[s['day_type']].append(s)

print(f"\n--- Sweeps by Day Type ---")
print(f"{'Day Type':<20s} {'Count':>6s} {'% of all':>8s} {'Avg Sweep':>10s} {'NearPDH':>8s} {'NearOvnH':>8s}")
print("-" * 65)
for dt in sorted(dt_counts.keys()):
    sweeps = dt_counts[dt]
    n = len(sweeps)
    avg_sweep = np.mean([s['sweep_pts'] for s in sweeps])
    pdh_pct = sum(1 for s in sweeps if s['near_pdh']) / n * 100
    ovn_pct = sum(1 for s in sweeps if s['near_ovn_high']) / n * 100
    print(f"{dt:<20s} {n:>6d} {n/len(all_sweeps)*100:>7.1f}% {avg_sweep:>9.1f} pts "
          f"{pdh_pct:>7.0f}% {ovn_pct:>7.0f}%")

# First sweep only per session
first_sweeps = [s for s in all_sweeps if s['first_sweep']]
print(f"\nFirst sweeps only (max 1/session): {len(first_sweeps)}")

dt_first = defaultdict(list)
for s in first_sweeps:
    dt_first[s['day_type']].append(s)

print(f"\n--- First Sweeps by Day Type ---")
for dt in sorted(dt_first.keys()):
    sweeps = dt_first[dt]
    print(f"  {dt:<18s}: {len(sweeps)}")


# ============================================================================
# PHASE 2: SIMULATE SHORT TRADES — BY DAY TYPE
# ============================================================================
print("\n\n" + "=" * 130)
print("  PHASE 2: SIMULATED SHORT TRADES — BY DAY TYPE")
print("=" * 130)

def simulate_sweep_short(sweep_list, label, target_mode='ib_mid',
                          max_contracts=5, first_only=True):
    """
    Simulate shorting on sweep + failure events.

    Entry: close of the sweep failure bar (price already closed below IBH)
    Stop: sweep high + 15% IB range
    Target: IB midpoint (or VWAP)
    """
    trades = []
    used_sessions = set()

    for sweep in sweep_list:
        if first_only and sweep['date'] in used_sessions:
            continue

        date_str = sweep['date']
        sdf = df[df['session_date'].astype(str) == date_str[:10]].copy()
        if len(sdf) < IB_BARS_1MIN + 10:
            continue

        post_ib = sdf.iloc[IB_BARS_1MIN:]
        ib_high = sweep['ib_high']
        ib_low = sweep['ib_low']
        ib_range = sweep['ib_range']
        ib_mid = sweep['ib_mid']

        # Entry price = close of sweep bar
        entry_raw = sweep['close']
        entry_fill = execution.fill_entry('SHORT', entry_raw)

        # Stop: sweep high + 15% IB range
        stop_price = sweep['sweep_high'] + (ib_range * 0.15)
        stop_price = max(stop_price, entry_fill + 15.0)

        # Target
        if target_mode == 'ib_mid':
            target_price = ib_mid
        elif target_mode == 'vwap':
            # Use VWAP at the sweep bar
            sweep_bar_in_post = post_ib.iloc[sweep['bar_idx']] if sweep['bar_idx'] < len(post_ib) else None
            if sweep_bar_in_post is not None:
                vwap = sweep_bar_in_post.get('vwap', ib_mid)
                target_price = vwap if not pd.isna(vwap) else ib_mid
            else:
                target_price = ib_mid
        else:
            target_price = ib_mid

        # R:R check
        risk = stop_price - entry_fill
        reward = entry_fill - target_price
        if reward <= 0 or risk <= 0:
            continue

        # Walk forward from sweep bar
        entry_bar = sweep['bar_idx']
        trade_result = None

        for j in range(entry_bar + 1, len(post_ib)):
            bar = post_ib.iloc[j]
            bar_time = bar['timestamp'].time() if 'timestamp' in bar.index else None

            # Check stop
            if bar['high'] >= stop_price:
                exit_fill = execution.fill_exit('SHORT', stop_price)
                _, _, _, net = execution.calculate_net_pnl(
                    'SHORT', entry_fill, exit_fill, max_contracts)
                trade_result = {
                    'date': date_str, 'day_type': sweep['day_type'],
                    'entry': entry_fill, 'exit': exit_fill, 'net_pnl': net,
                    'exit_reason': 'STOP', 'bars_held': j - entry_bar,
                    'sweep_pts': sweep['sweep_pts'],
                    'near_pdh': sweep['near_pdh'],
                    'near_ovn_high': sweep['near_ovn_high'],
                    'delta': sweep['delta'],
                    'risk': risk, 'reward': reward,
                }
                break

            # Check target
            if bar['low'] <= target_price:
                exit_fill = execution.fill_exit('SHORT', target_price)
                _, _, _, net = execution.calculate_net_pnl(
                    'SHORT', entry_fill, exit_fill, max_contracts)
                trade_result = {
                    'date': date_str, 'day_type': sweep['day_type'],
                    'entry': entry_fill, 'exit': exit_fill, 'net_pnl': net,
                    'exit_reason': 'TARGET', 'bars_held': j - entry_bar,
                    'sweep_pts': sweep['sweep_pts'],
                    'near_pdh': sweep['near_pdh'],
                    'near_ovn_high': sweep['near_ovn_high'],
                    'delta': sweep['delta'],
                    'risk': risk, 'reward': reward,
                }
                break

            # EOD close at 15:00
            if bar_time and bar_time >= _time(15, 0):
                exit_fill = execution.fill_exit('SHORT', bar['close'])
                _, _, _, net = execution.calculate_net_pnl(
                    'SHORT', entry_fill, exit_fill, max_contracts)
                trade_result = {
                    'date': date_str, 'day_type': sweep['day_type'],
                    'entry': entry_fill, 'exit': exit_fill, 'net_pnl': net,
                    'exit_reason': 'EOD', 'bars_held': j - entry_bar,
                    'sweep_pts': sweep['sweep_pts'],
                    'near_pdh': sweep['near_pdh'],
                    'near_ovn_high': sweep['near_ovn_high'],
                    'delta': sweep['delta'],
                    'risk': risk, 'reward': reward,
                }
                break

        if trade_result is None and entry_bar < len(post_ib) - 1:
            # Force close at last bar
            last_bar = post_ib.iloc[-1]
            exit_fill = execution.fill_exit('SHORT', last_bar['close'])
            _, _, _, net = execution.calculate_net_pnl(
                'SHORT', entry_fill, exit_fill, max_contracts)
            trade_result = {
                'date': date_str, 'day_type': sweep['day_type'],
                'entry': entry_fill, 'exit': exit_fill, 'net_pnl': net,
                'exit_reason': 'EOD', 'bars_held': len(post_ib) - entry_bar,
                'sweep_pts': sweep['sweep_pts'],
                'near_pdh': sweep['near_pdh'],
                'near_ovn_high': sweep['near_ovn_high'],
                'delta': sweep['delta'],
                'risk': risk, 'reward': reward,
            }

        if trade_result:
            trades.append(trade_result)
            used_sessions.add(date_str)

    return trades


def print_trade_results(label, trades):
    """Print results for a trade set."""
    if not trades:
        print(f"  {label:<50s}: 0 trades")
        return trades

    wins = [t for t in trades if t['net_pnl'] > 0]
    losses = [t for t in trades if t['net_pnl'] <= 0]
    pnl = sum(t['net_pnl'] for t in trades)
    wr = len(wins) / len(trades) * 100
    avg_w = np.mean([t['net_pnl'] for t in wins]) if wins else 0
    avg_l = np.mean([t['net_pnl'] for t in losses]) if losses else 0
    exp = pnl / len(trades)

    print(f"  {label:<50s}: {len(trades):>2d} trades  {wr:>5.1f}% WR  "
          f"${pnl:>8,.0f}  ${exp:>6,.0f}/trade  AvgW ${avg_w:>6,.0f}  AvgL ${avg_l:>6,.0f}")
    return trades


# Run by day type (first sweep only, target = IB mid)
print(f"\n--- First Sweep SHORT → IB Mid Target (1 trade/session) ---")
for dt_name in ['b_day', 'neutral', 'p_day', 'trend_up', 'trend_down',
                'super_trend_up', 'super_trend_down']:
    dt_sweeps = [s for s in first_sweeps if s['day_type'] == dt_name]
    if dt_sweeps:
        trades = simulate_sweep_short(dt_sweeps, f"{dt_name} → IB mid")
        print_trade_results(f"{dt_name} → IB mid", trades)

# Combined day type sets
print(f"\n--- Combined Day Type Sets ---")

bday_sweeps = [s for s in first_sweeps if s['day_type'] == 'b_day']
bday_trades = simulate_sweep_short(bday_sweeps, "b_day only")
print_trade_results("b_day only → IB mid", bday_trades)

bday_neutral = [s for s in first_sweeps if s['day_type'] in ('b_day', 'neutral')]
bn_trades = simulate_sweep_short(bday_neutral, "b_day + neutral")
print_trade_results("b_day + neutral → IB mid", bn_trades)

pday_sweeps = [s for s in first_sweeps if s['day_type'] == 'p_day']
pday_trades = simulate_sweep_short(pday_sweeps, "p_day only")
print_trade_results("p_day only → IB mid (death zone?)", pday_trades)

all_first_trades = simulate_sweep_short(first_sweeps, "ALL day types")
print_trade_results("ALL day types → IB mid", all_first_trades)

# Also test VWAP target
print(f"\n--- VWAP Target ---")
bday_vwap = simulate_sweep_short(bday_sweeps, "b_day → VWAP", target_mode='vwap')
print_trade_results("b_day → VWAP", bday_vwap)

bn_vwap = simulate_sweep_short(bday_neutral, "b_day+neutral → VWAP", target_mode='vwap')
print_trade_results("b_day + neutral → VWAP", bn_vwap)


# ============================================================================
# PHASE 3: DETAILED TRADE LOG — b_day
# ============================================================================
print(f"\n\n{'=' * 130}")
print("  PHASE 3: B-DAY IBH SWEEP SHORT — DETAILED TRADE LOG")
print(f"{'=' * 130}")

if bday_trades:
    print(f"\n  {'Date':<14s} {'DayType':<10s} {'SweepPts':>8s} {'Entry':>10s} {'Exit':>10s} "
          f"{'Net PnL':>10s} {'Reason':<8s} {'Bars':>5s} {'PDH':>4s} {'OvnH':>5s} {'Delta':>8s}")
    print("  " + "-" * 105)

    for t in sorted(bday_trades, key=lambda x: x['date']):
        print(f"  {t['date']:<14s} {t['day_type']:<10s} {t['sweep_pts']:>7.1f} "
              f"{t['entry']:>10.2f} {t['exit']:>10.2f} "
              f"${t['net_pnl']:>8,.2f} {t['exit_reason']:<8s} {t['bars_held']:>5d} "
              f"{'Y' if t['near_pdh'] else 'N':>4s} "
              f"{'Y' if t['near_ovn_high'] else 'N':>5s} "
              f"{t['delta']:>+7.0f}")

    exit_summary = defaultdict(list)
    for t in bday_trades:
        exit_summary[t['exit_reason']].append(t['net_pnl'])

    print(f"\n  Exit reasons:")
    for reason, pnls in sorted(exit_summary.items()):
        w = sum(1 for p in pnls if p > 0)
        print(f"    {reason:<8s}: {len(pnls)} trades, {w}/{len(pnls)} wins, ${sum(pnls):>8,.0f}")
else:
    print("\n  No b_day sweep trades found.")


# Also show b_day + neutral detail
print(f"\n\n{'=' * 130}")
print("  B-DAY + NEUTRAL IBH SWEEP SHORT — DETAILED TRADE LOG")
print(f"{'=' * 130}")

if bn_trades:
    print(f"\n  {'Date':<14s} {'DayType':<10s} {'SweepPts':>8s} {'Entry':>10s} {'Exit':>10s} "
          f"{'Net PnL':>10s} {'Reason':<8s} {'Bars':>5s} {'PDH':>4s} {'OvnH':>5s}")
    print("  " + "-" * 100)

    for t in sorted(bn_trades, key=lambda x: x['date']):
        print(f"  {t['date']:<14s} {t['day_type']:<10s} {t['sweep_pts']:>7.1f} "
              f"{t['entry']:>10.2f} {t['exit']:>10.2f} "
              f"${t['net_pnl']:>8,.2f} {t['exit_reason']:<8s} {t['bars_held']:>5d} "
              f"{'Y' if t['near_pdh'] else 'N':>4s} "
              f"{'Y' if t['near_ovn_high'] else 'N':>5s}")


# ============================================================================
# PHASE 4: COMBINED PLAYBOOK IMPACT
# ============================================================================
print(f"\n\n{'=' * 130}")
print("  PHASE 4: IMPACT ON 70% WR PLAYBOOK")
print(f"{'=' * 130}")

# Current playbook: Core + MeanRev LONG only = 75.9% WR, 29 trades, $3,861
current_wr = 75.9
current_trades = 29
current_wins = 22
current_pnl = 3861

for label, trade_set in [("b_day only", bday_trades), ("b_day + neutral", bn_trades)]:
    if not trade_set:
        continue

    n = len(trade_set)
    w = sum(1 for t in trade_set if t['net_pnl'] > 0)
    pnl = sum(t['net_pnl'] for t in trade_set)
    wr = w / n * 100

    combined_trades = current_trades + n
    combined_wins = current_wins + w
    combined_pnl = current_pnl + pnl
    combined_wr = combined_wins / combined_trades * 100

    print(f"\n  Adding IBH Sweep SHORT ({label}):")
    print(f"    Sweep trades: {n} trades, {wr:.1f}% WR, ${pnl:,.0f}")
    print(f"    Current playbook: {current_trades} trades, {current_wr:.1f}% WR, ${current_pnl:,.0f}")
    print(f"    COMBINED: {combined_trades} trades, {combined_wr:.1f}% WR, ${combined_pnl:,.0f}")

    if combined_wr >= 70:
        print(f"    >>> COMBINED WR >= 70% — CAN ADD TO LIGHTNING PLAYBOOK")
    elif combined_wr >= 65:
        print(f"    >>> COMBINED WR 65-70% — MARGINAL, needs more data")
    else:
        print(f"    >>> COMBINED WR < 65% — NOT RECOMMENDED")


# ============================================================================
# PHASE 5: VERIFY WITH FULL BACKTEST ENGINE
# ============================================================================
# Run the full backtest with existing B-Day strategy to cross-reference
# The B-Day strategy already has IBL fade LONG. Check if our sweep dates match.

print(f"\n\n{'=' * 130}")
print("  PHASE 5: CROSS-REFERENCE WITH EXISTING B-DAY STRATEGY")
print(f"{'=' * 130}")

# Check which sessions the current B-Day strategy trades
exec_m = ExecutionModel(instrument, slippage_ticks=1)
pos_m = PositionManager(account_size=150000)
from engine.backtest import BacktestEngine

engine = BacktestEngine(
    instrument=instrument,
    strategies=[BDayStrategy()],
    filters=None,
    execution=exec_m,
    position_mgr=pos_m,
    risk_per_trade=400,
    max_contracts=5,
)
bday_result = engine.run(df, verbose=False)

bday_existing = bday_result.trades
print(f"\nExisting B-Day strategy: {len(bday_existing)} trades")
for t in bday_existing:
    print(f"  {t.session_date} | {t.setup_type} | {t.direction} | ${t.net_pnl:,.2f} | {t.exit_reason}")

# Check for overlap: do sweep SHORT dates overlap with IBL LONG dates?
sweep_dates = set(t['date'] for t in bday_trades) if bday_trades else set()
existing_dates = set(str(t.session_date) for t in bday_existing)
overlap = sweep_dates & existing_dates

print(f"\n  B-Day IBL LONG dates:    {existing_dates}")
print(f"  B-Day IBH Sweep dates:   {sweep_dates}")
print(f"  Overlap (both signals):  {overlap}")
if overlap:
    print(f"  NOTE: On overlap days, you'd take BOTH the IBL fade LONG and IBH sweep SHORT")
    print(f"  This provides two-sided balance day trading")


# ============================================================================
# PHASE 6: VERDICT
# ============================================================================
print(f"\n\n{'=' * 130}")
print("  VERDICT: IBH SWEEP + FAILURE SHORT — DAY TYPE FILTERED")
print(f"{'=' * 130}")

best_set = bday_trades if bday_trades else bn_trades
best_label = "b_day" if bday_trades else "b_day + neutral"

if best_set:
    n = len(best_set)
    w = sum(1 for t in best_set if t['net_pnl'] > 0)
    pnl = sum(t['net_pnl'] for t in best_set)
    wr = w / n * 100

    combined_trades = current_trades + n
    combined_wins = current_wins + w
    combined_wr = combined_wins / combined_trades * 100

    print(f"""
  OTHER THREAD FINDINGS:
    b_day IBH sweep shorts: 4/4 = 100% WR, +291 pts
    p_day IBH sweep shorts: 3/13 = 23% WR (death zone)

  OUR DATA VALIDATION:
    b_day IBH sweep shorts: {w}/{n} = {wr:.0f}% WR, ${pnl:,.0f}
    (Compare to other thread: similar or different?)

  WHY B-DAY WORKS:
    - B-day = no extension, price trapped in IB range
    - Sweep above IBH = liquidity grab at PDH / overnight high
    - Failure to hold = longs trapped, shorts take profit → price returns to IB mid
    - On b_day, there's NO structural reason for price to hold above IBH

  WHY P-DAY FAILS:
    - P-day has directional momentum above IBH
    - The sweep is an acceptance ATTEMPT, not exhaustion
    - Shorting against p-day momentum = fighting the tape

  PROPOSED ENTRY MODEL:
    Day type:     b_day only (possibly b_day + neutral with more data)
    Entry:        SHORT at close of failure candle (high > IBH + 5pts, close < IBH)
    Stop:         Sweep high + 15% IB range
    Target:       IB midpoint
    Max trades:   1 per session, first sweep only
    Time window:  10:30 AM - 2:00 PM ET

  COMBINED PLAYBOOK:
    Core + MeanRev LONG + B-Day IBH Sweep SHORT
    {combined_wr:.1f}% WR | {combined_trades} trades | ${current_pnl + pnl:,.0f} net
""")

    if n < 10:
        print(f"  CAUTION: Only {n} trades — need 100+ sessions (15-20 trades) to confirm.")
        print(f"  The logic is sound, but statistical significance requires more data.")
else:
    print("\n  No sweep trades found in our data. Need more sessions or adjust sweep threshold.")
