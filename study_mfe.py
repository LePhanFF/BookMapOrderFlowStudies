"""
MFE (Maximum Favorable Excursion) Study
========================================

Analyzes how far winning trades moved in our favor BEFORE they were closed,
to determine if current targets leave money on the table.

For each trade:
  - Replay bars from entry to exit
  - Track the maximum favorable move (MFE) = best unrealized P&L
  - Compare MFE to actual exit P&L
  - The difference = money left on table

Key questions:
  1. What % of winners hit 2x, 3x, 4x the current target?
  2. What's the median MFE / actual P&L ratio?
  3. Would runner targets (partial exit at 1x, runner at 2-3x) help?
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.instruments import get_instrument
from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from engine.execution import ExecutionModel
from engine.backtest import BacktestEngine
from engine.position import PositionManager
from strategy import TrendDayBull, SuperTrendBull, PDayStrategy, BDayStrategy, MeanReversionVWAP
from filters.regime_filter import SimpleRegimeFilter

# ============================================================================
# LOAD DATA & RUN BACKTEST
# ============================================================================
instrument = get_instrument('MNQ')
df_raw = load_csv('NQ')
df = filter_rth(df_raw)
df = compute_all_features(df)

if 'session_date' not in df.columns:
    df['session_date'] = df['timestamp'].dt.date

sessions = sorted(df['session_date'].unique())

rf_long_only = SimpleRegimeFilter(
    longs_in_bull=True, longs_in_bear=True,
    shorts_in_bull=False, shorts_in_bear=False,
)

strategies = [
    TrendDayBull(), SuperTrendBull(), PDayStrategy(),
    BDayStrategy(), MeanReversionVWAP(),
]

exec_m = ExecutionModel(instrument, slippage_ticks=1)
pos_m = PositionManager(account_size=150000)
engine = BacktestEngine(
    instrument=instrument,
    strategies=strategies,
    filters=rf_long_only,
    execution=exec_m,
    position_mgr=pos_m,
    risk_per_trade=800,
    max_contracts=5,
)
result = engine.run(df, verbose=False)
trades = result.trades

print(f"\n{'='*90}")
print(f"  MFE (Maximum Favorable Excursion) STUDY")
print(f"{'='*90}")
print(f"  Trades analyzed: {len(trades)}")

# ============================================================================
# REPLAY EACH TRADE TO COMPUTE MFE
# ============================================================================
mfe_data = []

for trade in trades:
    session_date = trade.session_date
    session_df = df[df['session_date'].astype(str) == str(session_date)]

    if session_df.empty:
        continue

    entry_price = trade.entry_price
    direction = trade.direction

    # Find bars between entry and exit
    entry_ts = trade.entry_time
    exit_ts = trade.exit_time

    if entry_ts is None or exit_ts is None:
        continue

    trade_bars = session_df[
        (session_df['timestamp'] >= entry_ts) &
        (session_df['timestamp'] <= exit_ts)
    ]

    if trade_bars.empty:
        continue

    # Compute MFE (max favorable excursion in points)
    if direction == 'LONG':
        mfe_points = trade_bars['high'].max() - entry_price
        mae_points = entry_price - trade_bars['low'].min()
    else:
        mfe_points = entry_price - trade_bars['low'].min()
        mae_points = trade_bars['high'].max() - entry_price

    # Actual exit points
    if direction == 'LONG':
        actual_points = trade.exit_price - entry_price
    else:
        actual_points = entry_price - trade.exit_price

    # Risk in points
    risk_pts = trade.risk_points
    target_pts = trade.reward_points

    # IB range from metadata or trade context
    ib_range = trade.metadata.get('ib_range', 0) if trade.metadata else 0
    if ib_range == 0:
        # Estimate from session data
        ib_data = session_df.head(60)
        if len(ib_data) >= 60:
            ib_range = ib_data['high'].max() - ib_data['low'].min()

    mfe_data.append({
        'strategy': trade.strategy_name,
        'setup': trade.setup_type,
        'day_type': trade.day_type,
        'direction': direction,
        'entry_price': entry_price,
        'exit_price': trade.exit_price,
        'stop_price': trade.stop_price,
        'target_price': trade.target_price,
        'exit_reason': trade.exit_reason,
        'net_pnl': trade.net_pnl,
        'contracts': trade.contracts,
        'is_winner': trade.is_winner,
        'risk_pts': risk_pts,
        'target_pts': target_pts,
        'actual_pts': actual_points,
        'mfe_pts': mfe_points,
        'mae_pts': mae_points,
        'ib_range': ib_range,
        'bars_held': trade.bars_held,
        # Derived metrics
        'mfe_r': mfe_points / risk_pts if risk_pts > 0 else 0,
        'actual_r': actual_points / risk_pts if risk_pts > 0 else 0,
        'mfe_vs_actual': mfe_points - actual_points,
        'mfe_ib_mult': mfe_points / ib_range if ib_range > 0 else 0,
        'capture_ratio': actual_points / mfe_points if mfe_points > 0 else 0,
    })

mfe_df = pd.DataFrame(mfe_data)
winners = mfe_df[mfe_df['is_winner']].copy()
losers = mfe_df[~mfe_df['is_winner']].copy()

# ============================================================================
# OVERALL MFE ANALYSIS
# ============================================================================
print(f"\n--- Overall MFE Analysis ---")
print(f"  Winners: {len(winners)}, Losers: {len(losers)}")

if len(winners) > 0:
    print(f"\n  WINNERS:")
    print(f"    Avg MFE:         {winners['mfe_pts'].mean():.1f} pts ({winners['mfe_r'].mean():.2f}R)")
    print(f"    Avg Actual:      {winners['actual_pts'].mean():.1f} pts ({winners['actual_r'].mean():.2f}R)")
    print(f"    Avg Left on Table: {winners['mfe_vs_actual'].mean():.1f} pts")
    print(f"    Capture Ratio:   {winners['capture_ratio'].mean():.1%} (how much of the move was captured)")
    print(f"    MFE IB Multiple: {winners['mfe_ib_mult'].mean():.2f}x IB range")

    # Extension analysis
    print(f"\n  WINNERS — How far did they actually run?")
    for mult in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        pct = (winners['mfe_ib_mult'] >= mult).mean() * 100
        print(f"    >= {mult:.1f}x IB: {pct:.0f}% of winners ({(winners['mfe_ib_mult'] >= mult).sum()} trades)")

    # R-multiple analysis
    print(f"\n  WINNERS — MFE in R-multiples:")
    for r in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        pct = (winners['mfe_r'] >= r).mean() * 100
        print(f"    >= {r:.1f}R: {pct:.0f}% of winners ({(winners['mfe_r'] >= r).sum()} trades)")

    # Exit reason breakdown
    print(f"\n  EXIT REASON BREAKDOWN (Winners):")
    for reason in winners['exit_reason'].unique():
        subset = winners[winners['exit_reason'] == reason]
        print(f"    {reason:15s}: {len(subset)} trades, "
              f"avg capture={subset['capture_ratio'].mean():.1%}, "
              f"avg MFE={subset['mfe_pts'].mean():.0f} pts, "
              f"avg left={subset['mfe_vs_actual'].mean():.0f} pts")

if len(losers) > 0:
    print(f"\n  LOSERS:")
    print(f"    Avg MFE:         {losers['mfe_pts'].mean():.1f} pts ({losers['mfe_r'].mean():.2f}R)")
    print(f"    Avg MAE:         {losers['mae_pts'].mean():.1f} pts")
    print(f"    Avg Actual:      {losers['actual_pts'].mean():.1f} pts")
    # How many losers had significant MFE before hitting stop
    profitable_before_stop = losers[losers['mfe_pts'] > 10]
    print(f"    Had >10pt profit before losing: {len(profitable_before_stop)}/{len(losers)} "
          f"({len(profitable_before_stop)/len(losers)*100:.0f}%)")

# ============================================================================
# PER-STRATEGY MFE
# ============================================================================
print(f"\n{'='*90}")
print(f"  PER-STRATEGY MFE ANALYSIS")
print(f"{'='*90}")

for strat in mfe_df['strategy'].unique():
    strat_df = mfe_df[mfe_df['strategy'] == strat]
    strat_winners = strat_df[strat_df['is_winner']]

    if len(strat_winners) == 0:
        continue

    print(f"\n  {strat} ({len(strat_df)} trades, {len(strat_winners)} winners)")
    print(f"    Avg MFE:       {strat_winners['mfe_pts'].mean():.1f} pts ({strat_winners['mfe_r'].mean():.2f}R)")
    print(f"    Avg Actual:    {strat_winners['actual_pts'].mean():.1f} pts ({strat_winners['actual_r'].mean():.2f}R)")
    print(f"    Avg Left:      {strat_winners['mfe_vs_actual'].mean():.1f} pts")
    print(f"    Capture Ratio: {strat_winners['capture_ratio'].mean():.1%}")
    print(f"    MFE >= 2.0R:   {(strat_winners['mfe_r'] >= 2.0).mean()*100:.0f}%")
    print(f"    MFE >= 3.0R:   {(strat_winners['mfe_r'] >= 3.0).mean()*100:.0f}%")

# ============================================================================
# PER-DAY-TYPE MFE
# ============================================================================
print(f"\n{'='*90}")
print(f"  PER-DAY-TYPE MFE ANALYSIS")
print(f"{'='*90}")

for dt in mfe_df['day_type'].unique():
    dt_df = mfe_df[mfe_df['day_type'] == dt]
    dt_winners = dt_df[dt_df['is_winner']]

    if len(dt_winners) == 0:
        continue

    print(f"\n  {dt} ({len(dt_df)} trades, {len(dt_winners)} winners)")
    print(f"    Avg MFE:       {dt_winners['mfe_pts'].mean():.1f} pts ({dt_winners['mfe_r'].mean():.2f}R)")
    print(f"    Avg Actual:    {dt_winners['actual_pts'].mean():.1f} pts ({dt_winners['actual_r'].mean():.2f}R)")
    print(f"    Capture Ratio: {dt_winners['capture_ratio'].mean():.1%}")
    print(f"    MFE >= 2.0x IB: {(dt_winners['mfe_ib_mult'] >= 2.0).mean()*100:.0f}%")

# ============================================================================
# TARGET OPTIMIZATION: What if we used different target levels?
# ============================================================================
print(f"\n{'='*90}")
print(f"  TARGET OPTIMIZATION — Hypothetical Alternative Targets")
print(f"{'='*90}")

print(f"\n  Simulating alternative target levels using MFE data:")
print(f"  (Assumes we would exit at target OR at actual exit, whichever comes first)")

for target_r in [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
    simulated_pnl = 0
    sim_wins = 0
    sim_trades = 0

    for _, row in mfe_df.iterrows():
        sim_trades += 1
        risk = row['risk_pts']
        target_pts = target_r * risk

        if row['mfe_pts'] >= target_pts:
            # Would have hit new target
            pnl_pts = target_pts
            sim_wins += 1
        else:
            # Same outcome as actual
            pnl_pts = row['actual_pts']
            if pnl_pts > 0:
                sim_wins += 1

        # Convert to dollars (MNQ = $2/pt)
        pnl_dollars = pnl_pts * row['contracts'] * 2.0
        simulated_pnl += pnl_dollars

    wr = sim_wins / sim_trades * 100 if sim_trades > 0 else 0
    months = len(sessions) / 22
    per_month = simulated_pnl / months

    print(f"    Target {target_r:.1f}R: WR={wr:.1f}%, Net=${simulated_pnl:,.0f} "
          f"(${per_month:,.0f}/mo), {sim_wins}W/{sim_trades-sim_wins}L")

# ============================================================================
# PARTIAL EXIT SIMULATION: 50% at 1R, runner at 2-3R
# ============================================================================
print(f"\n{'='*90}")
print(f"  PARTIAL EXIT SIMULATION")
print(f"{'='*90}")

for first_target_r, runner_target_r in [(1.0, 2.0), (1.0, 3.0), (1.5, 2.5), (1.5, 3.0)]:
    total_pnl = 0
    sim_wins = 0

    for _, row in mfe_df.iterrows():
        risk = row['risk_pts']
        contracts = row['contracts']
        first_contracts = max(1, contracts // 2)
        runner_contracts = contracts - first_contracts

        first_target_pts = first_target_r * risk
        runner_target_pts = runner_target_r * risk

        # First portion
        if row['mfe_pts'] >= first_target_pts:
            first_pnl = first_target_pts * first_contracts * 2.0
        else:
            first_pnl = row['actual_pts'] * first_contracts * 2.0

        # Runner portion
        if runner_contracts > 0:
            if row['mfe_pts'] >= runner_target_pts:
                runner_pnl = runner_target_pts * runner_contracts * 2.0
            else:
                runner_pnl = row['actual_pts'] * runner_contracts * 2.0
        else:
            runner_pnl = 0

        trade_pnl = first_pnl + runner_pnl
        total_pnl += trade_pnl
        if trade_pnl > 0:
            sim_wins += 1

    months = len(sessions) / 22
    per_month = total_pnl / months
    wr = sim_wins / len(mfe_df) * 100 if len(mfe_df) > 0 else 0

    print(f"  50% at {first_target_r:.1f}R + 50% at {runner_target_r:.1f}R: "
          f"WR={wr:.1f}%, Net=${total_pnl:,.0f} (${per_month:,.0f}/mo)")

# ============================================================================
# IB WIDTH ANALYSIS: Narrow vs Wide IB → different extensions
# ============================================================================
print(f"\n{'='*90}")
print(f"  IB WIDTH → MFE RELATIONSHIP")
print(f"{'='*90}")

if len(winners) > 0:
    # Classify IB width
    atr_series = df.groupby('session_date').first()['atr14']
    median_ib = mfe_df['ib_range'].median()

    narrow = winners[winners['ib_range'] <= median_ib]
    wide = winners[winners['ib_range'] > median_ib]

    print(f"\n  Median IB Range: {median_ib:.1f} pts")
    print(f"\n  Narrow IB ({len(narrow)} winners):")
    if len(narrow) > 0:
        print(f"    Avg MFE: {narrow['mfe_pts'].mean():.1f} pts ({narrow['mfe_ib_mult'].mean():.2f}x IB)")
        print(f"    Capture: {narrow['capture_ratio'].mean():.1%}")
        print(f"    >= 2.0x IB: {(narrow['mfe_ib_mult'] >= 2.0).mean()*100:.0f}%")

    print(f"\n  Wide IB ({len(wide)} winners):")
    if len(wide) > 0:
        print(f"    Avg MFE: {wide['mfe_pts'].mean():.1f} pts ({wide['mfe_ib_mult'].mean():.2f}x IB)")
        print(f"    Capture: {wide['capture_ratio'].mean():.1%}")
        print(f"    >= 2.0x IB: {(wide['mfe_ib_mult'] >= 2.0).mean()*100:.0f}%")

# ============================================================================
# INDIVIDUAL TRADE MFE TABLE
# ============================================================================
print(f"\n{'='*90}")
print(f"  TOP 15 TRADES BY MFE LEFT ON TABLE")
print(f"{'='*90}")

top_left = mfe_df[mfe_df['is_winner']].nlargest(15, 'mfe_vs_actual')
print(f"\n  {'Strategy':25s} {'DayType':10s} {'Exit':12s} "
      f"{'Actual':>8s} {'MFE':>8s} {'Left':>8s} {'Capture':>8s}")
print(f"  {'-'*85}")

for _, row in top_left.iterrows():
    print(f"  {row['strategy']:25s} {row['day_type']:10s} {row['exit_reason']:12s} "
          f"{row['actual_pts']:>7.1f}p {row['mfe_pts']:>7.1f}p "
          f"{row['mfe_vs_actual']:>7.1f}p {row['capture_ratio']:>7.1%}")

print(f"\n{'='*90}")
print(f"  STUDY COMPLETE")
print(f"{'='*90}")
