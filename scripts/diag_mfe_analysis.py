"""MFE/MAE Analysis: How much further could winning trades have run?

For each trade:
1. Track Maximum Favorable Excursion (MFE) — how far did price go in our favor?
2. Track Maximum Adverse Excursion (MAE) — how far did price go against us?
3. Compare actual exit vs MFE — how much did we leave on the table?
4. Test different exit strategies: trail stop, partial profit, ATR-scaled targets
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from collections import defaultdict
from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from config.instruments import get_instrument
from config.constants import IB_BARS_1MIN
from engine.backtest import BacktestEngine
from engine.execution import ExecutionModel
from engine.position import PositionManager
from strategy import get_core_strategies

full_df = load_csv('NQ')
df = filter_rth(full_df)
df = compute_all_features(df)
instrument = get_instrument('MNQ')

# Run baseline to get trades
strategies = get_core_strategies()
engine = BacktestEngine(
    instrument=instrument, strategies=strategies, filters=None,
    execution=ExecutionModel(instrument),
    position_mgr=PositionManager(max_drawdown=999999),
    full_df=full_df,
)
result = engine.run(df, verbose=False)
trades = result.trades
print(f"Baseline: {len(trades)} trades, ${sum(t.net_pnl for t in trades):,.0f}")

# For each trade, compute MFE and MAE from the bar data
sessions = df.groupby('session_date')

mfe_results = []
for trade in trades:
    session_date = trade.session_date
    try:
        sdf = sessions.get_group(pd.Timestamp(session_date).date())
    except KeyError:
        try:
            sdf = sessions.get_group(session_date)
        except KeyError:
            continue

    entry_price = trade.entry_price
    direction = trade.direction
    stop = trade.stop_price
    target = trade.target_price
    risk = abs(entry_price - stop)

    # Find bars from entry to EOD
    entry_ts = trade.entry_time
    if hasattr(entry_ts, 'to_pydatetime'):
        entry_ts = entry_ts.to_pydatetime()

    # Get all bars after entry
    mask = sdf['timestamp'] >= pd.Timestamp(entry_ts)
    post_entry = sdf[mask]
    if len(post_entry) == 0:
        continue

    # Track MFE and MAE
    if direction == 'LONG':
        mfe_price = post_entry['high'].max()
        mae_price = post_entry['low'].min()
        mfe_pts = mfe_price - entry_price
        mae_pts = entry_price - mae_price
    else:  # SHORT
        mfe_price = post_entry['low'].min()
        mae_price = post_entry['high'].max()
        mfe_pts = entry_price - mfe_price
        mae_pts = mae_price - entry_price

    actual_pts = trade.net_pnl / (instrument.point_value * 1)  # approximate
    mfe_r = mfe_pts / risk if risk > 0 else 0
    mae_r = mae_pts / risk if risk > 0 else 0
    actual_r = actual_pts / risk if risk > 0 else 0
    left_on_table = mfe_pts - max(actual_pts, 0)

    # IB range for this session
    ib_df = sdf.iloc[:IB_BARS_1MIN]
    ib_range = ib_df['high'].max() - ib_df['low'].min()

    # ATR proxy: session range / 1.5
    session_range = sdf['high'].max() - sdf['low'].min()

    mfe_results.append({
        'date': session_date,
        'strategy': trade.strategy_name,
        'direction': direction,
        'entry': entry_price,
        'exit': trade.exit_price,
        'exit_reason': trade.exit_reason,
        'net_pnl': trade.net_pnl,
        'risk_pts': risk,
        'mfe_pts': mfe_pts,
        'mae_pts': mae_pts,
        'mfe_r': mfe_r,
        'mae_r': mae_r,
        'actual_r': actual_r,
        'left_on_table_pts': left_on_table,
        'ib_range': ib_range,
        'session_range': session_range,
        'bars_held': trade.bars_held,
        'winner': trade.net_pnl > 0,
    })

mdf = pd.DataFrame(mfe_results)
print(f"\nAnalyzed {len(mdf)} trades with MFE/MAE data")

# === ANALYSIS 1: Overall MFE distribution ===
print("\n" + "=" * 80)
print("MFE ANALYSIS: How much further could trades have run?")
print("=" * 80)

winners = mdf[mdf['winner']]
losers = mdf[~mdf['winner']]

print(f"\nWINNERS ({len(winners)}):")
print(f"  Avg MFE: {winners['mfe_r'].mean():.1f}R ({winners['mfe_pts'].mean():.0f} pts)")
print(f"  Avg actual: {winners['actual_r'].mean():.1f}R")
print(f"  Avg left on table: {winners['left_on_table_pts'].mean():.0f} pts")
print(f"  Median MFE: {winners['mfe_r'].median():.1f}R")
print(f"  90th pct MFE: {winners['mfe_r'].quantile(0.9):.1f}R")

print(f"\nLOSERS ({len(losers)}):")
print(f"  Avg MFE (best the trade saw): {losers['mfe_r'].mean():.1f}R ({losers['mfe_pts'].mean():.0f} pts)")
print(f"  Avg MAE: {losers['mae_r'].mean():.1f}R")
print(f"  Losers that saw 1R+ profit: {(losers['mfe_r'] >= 1.0).sum()} ({(losers['mfe_r'] >= 1.0).mean()*100:.0f}%)")
print(f"  Losers that saw 0.5R+ profit: {(losers['mfe_r'] >= 0.5).sum()} ({(losers['mfe_r'] >= 0.5).mean()*100:.0f}%)")

# === ANALYSIS 2: By strategy ===
print("\n" + "=" * 80)
print("MFE BY STRATEGY")
print("=" * 80)
print(f"{'Strategy':25s} {'Trades':>6s} {'Avg MFE':>8s} {'Avg Act':>8s} {'Left/trd':>9s} {'L>1R%':>6s}")
print("-" * 70)
for strat in sorted(mdf['strategy'].unique()):
    sdf_s = mdf[mdf['strategy'] == strat]
    avg_mfe = sdf_s['mfe_r'].mean()
    avg_act = sdf_s['actual_r'].mean()
    avg_left = sdf_s['left_on_table_pts'].mean()
    losers_saw_1r = (sdf_s[~sdf_s['winner']]['mfe_r'] >= 1.0).mean() * 100 if len(sdf_s[~sdf_s['winner']]) > 0 else 0
    print(f"{strat:25s} {len(sdf_s):>6d} {avg_mfe:>7.1f}R {avg_act:>7.1f}R {avg_left:>8.0f}pts {losers_saw_1r:>5.0f}%")

# === ANALYSIS 3: What if we used trail stop instead of fixed target? ===
print("\n" + "=" * 80)
print("TRAIL STOP SIMULATION")
print("=" * 80)

for trail_r in [0.5, 0.75, 1.0, 1.5]:
    trail_pnl = 0
    trail_wins = 0
    for _, row in mdf.iterrows():
        risk = row['risk_pts']
        mfe = row['mfe_pts']
        mae = row['mae_pts']

        # Trail stop: once price moves trail_r * risk in favor, trail at entry
        # Then trail at MFE - trail_r * risk
        if mfe >= trail_r * risk:
            # Would have been profitable (at least breakeven)
            # Estimate exit: MFE - trail_r * risk (trailing from peak)
            est_profit = max(0, mfe - trail_r * risk)
            trail_pnl += est_profit * instrument.point_value - 2.48  # commission+slippage
            trail_wins += 1
        else:
            # Stopped out at full loss
            trail_pnl += -risk * instrument.point_value - 2.48
    trail_wr = trail_wins / len(mdf) * 100
    print(f"  Trail at {trail_r}R: {trail_wins}W/{len(mdf)}t ({trail_wr:.0f}% WR), "
          f"${trail_pnl:,.0f} net, ${trail_pnl/len(mdf):.0f}/trade")

# === ANALYSIS 4: What if targets were IB-range scaled? ===
print("\n" + "=" * 80)
print("IB-SCALED TARGET SIMULATION")
print("=" * 80)

for target_mult in [0.5, 0.75, 1.0, 1.5, 2.0]:
    ib_pnl = 0
    ib_wins = 0
    for _, row in mdf.iterrows():
        ib_target = row['ib_range'] * target_mult
        risk = row['risk_pts']
        mfe = row['mfe_pts']

        if mfe >= ib_target:
            # Hit IB-scaled target
            ib_pnl += ib_target * instrument.point_value - 2.48
            ib_wins += 1
        else:
            # Didn't reach target, use actual exit
            ib_pnl += row['net_pnl']
    ib_wr = ib_wins / len(mdf) * 100
    print(f"  Target {target_mult}x IB: {ib_wins}W/{len(mdf)}t ({ib_wr:.0f}% WR), "
          f"${ib_pnl:,.0f} net, ${ib_pnl/len(mdf):.0f}/trade")

# === ANALYSIS 5: Breakeven after FVG displacement ===
print("\n" + "=" * 80)
print("BREAKEVEN AFTER DISPLACEMENT")
print("=" * 80)

for be_trigger_r in [0.3, 0.5, 0.75, 1.0]:
    be_pnl = 0
    be_wins = 0
    be_be_exits = 0
    for _, row in mdf.iterrows():
        risk = row['risk_pts']
        mfe = row['mfe_pts']
        mae = row['mae_pts']
        actual = row['net_pnl']

        if mfe >= be_trigger_r * risk:
            # Price reached trigger — move stop to BE
            # If price comes back to entry, exit at BE (0 P&L - commission)
            if mae > mfe * 0.8:  # rough: if MAE is close to entry after reaching trigger
                be_pnl += -2.48  # BE exit
                be_be_exits += 1
            else:
                # Held to original target/stop
                be_pnl += actual
                if actual > 0:
                    be_wins += 1
        else:
            # Never reached trigger, original outcome
            be_pnl += actual
            if actual > 0:
                be_wins += 1

    total_positive = be_wins + be_be_exits
    wr = total_positive / len(mdf) * 100
    print(f"  BE at {be_trigger_r}R: {be_wins}W + {be_be_exits}BE / {len(mdf)}t, "
          f"${be_pnl:,.0f} net, ${be_pnl/len(mdf):.0f}/trade")

# === ANALYSIS 6: High-vol vs low-vol periods ===
print("\n" + "=" * 80)
print("PERFORMANCE BY VOLATILITY REGIME")
print("=" * 80)
mdf['vol_regime'] = pd.cut(mdf['ib_range'], bins=[0, 100, 150, 250, 9999],
                            labels=['low(<100)', 'med(100-150)', 'norm(150-250)', 'high(>250)'])
for regime in ['low(<100)', 'med(100-150)', 'norm(150-250)', 'high(>250)']:
    rdf_r = mdf[mdf['vol_regime'] == regime]
    if len(rdf_r) == 0:
        continue
    wr = rdf_r['winner'].mean() * 100
    pnl = rdf_r['net_pnl'].sum()
    avg_mfe = rdf_r['mfe_r'].mean()
    avg_risk = rdf_r['risk_pts'].mean()
    print(f"  {regime:15s}: {len(rdf_r):>4d} trades, {wr:.0f}% WR, ${pnl:>8,.0f}, "
          f"avg_risk={avg_risk:.0f}pts, avg_MFE={avg_mfe:.1f}R")

# === ANALYSIS 7: Exit reason analysis ===
print("\n" + "=" * 80)
print("EXIT REASON BREAKDOWN")
print("=" * 80)
for reason in mdf['exit_reason'].unique():
    rdf_e = mdf[mdf['exit_reason'] == reason]
    wr = rdf_e['winner'].mean() * 100
    pnl = rdf_e['net_pnl'].sum()
    avg_mfe = rdf_e['mfe_r'].mean()
    print(f"  {reason:20s}: {len(rdf_e):>4d} trades, {wr:.0f}% WR, ${pnl:>8,.0f}, avg_MFE={avg_mfe:.1f}R")
