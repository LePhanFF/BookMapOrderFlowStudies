"""
Diagnostic 6: Analyze EOD exit trades to see if targets or trailing stops
could capture more P&L. Also check cumulative delta as a trade confirmation.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import time

from config.instruments import get_instrument
from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from config.constants import IB_BARS_1MIN

# Read the current trade log
trade_log = pd.read_csv(Path(project_root) / 'output' / 'trade_log.csv')

print(f"\n{'='*120}")
print(f"  EOD EXIT ANALYSIS: Could targets or trailing stops capture more P&L?")
print(f"{'='*120}\n")

# Load data
df = load_csv('NQ')
df = filter_rth(df)
df = compute_all_features(df)

if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
if 'session_date' not in df.columns:
    df['session_date'] = df['timestamp'].dt.date

# Analyze each EOD trade
eod_trades = trade_log[trade_log['exit_reason'] == 'EOD']

for _, trade in eod_trades.iterrows():
    session_date = trade['session_date']
    entry_price = trade['entry_price']
    setup = trade['setup_type']
    strategy = trade['strategy_name']
    net_pnl = trade['net_pnl']

    # Get session data
    session_df = df[df['session_date'].astype(str) == session_date[:10]].copy()
    if len(session_df) < IB_BARS_1MIN:
        continue

    ib_df = session_df.head(IB_BARS_1MIN)
    ib_high = ib_df['high'].max()
    ib_low = ib_df['low'].min()
    ib_range = ib_high - ib_low

    post_ib = session_df.iloc[IB_BARS_1MIN:]

    # Find the entry bar
    entry_bars = post_ib[abs(post_ib['close'] - entry_price) < 2.0]
    if len(entry_bars) == 0:
        continue

    entry_idx = entry_bars.index[0]
    remaining = session_df.loc[entry_idx:]

    # Track max favorable excursion (MFE) and max adverse excursion (MAE)
    max_price = entry_price
    min_price = entry_price
    mfe_pts = 0
    mae_pts = 0
    mfe_time = None
    mae_time = None

    for _, bar in remaining.iterrows():
        if bar['high'] > max_price:
            max_price = bar['high']
            mfe_pts = max_price - entry_price
            mfe_time = bar.get('timestamp')
        if bar['low'] < min_price:
            min_price = bar['low']
            mae_pts = entry_price - min_price
            mae_time = bar.get('timestamp')

    eod_close = remaining.iloc[-1]['close']
    eod_pnl = eod_close - entry_price

    # Could a trailing stop have captured more?
    # Trail at 0.5x IB from session high
    trail_dist = ib_range * 0.5
    trail_exit = max_price - trail_dist if max_price > entry_price + trail_dist else None
    trail_pnl = (trail_exit - entry_price) if trail_exit else None

    # Could a target at 1.0x IB have hit?
    target_1x = entry_price + ib_range * 1.0
    target_1x_hit = max_price >= target_1x

    print(f"--- {session_date[:10]} {strategy:20s} {setup:20s} ---")
    print(f"  Entry: {entry_price:.1f} | EOD Close: {eod_close:.1f} | P&L: {eod_pnl:.1f} pts (${net_pnl:.2f})")
    print(f"  IB Range: {ib_range:.1f} pts")
    print(f"  MFE: +{mfe_pts:.1f} pts ({mfe_pts/ib_range:.2f}x IB) at {mfe_time}")
    print(f"  MAE: -{mae_pts:.1f} pts")
    if trail_pnl:
        print(f"  Trail (0.5x IB): exit at {trail_exit:.1f}, P&L: +{trail_pnl:.1f} pts (vs EOD +{eod_pnl:.1f})")
    print(f"  Target 1.0x IB ({target_1x:.1f}): {'HIT' if target_1x_hit else 'NOT HIT'}")
    print()

print(f"\n{'='*120}")
print(f"  CUMULATIVE DELTA ALIGNMENT ANALYSIS")
print(f"{'='*120}\n")

# Check if cumulative delta direction aligns with trade direction at entry
for _, trade in trade_log.iterrows():
    session_date = trade['session_date']
    entry_price = trade['entry_price']
    setup = trade['setup_type']
    strategy = trade['strategy_name']
    net_pnl = trade['net_pnl']
    is_winner = trade['is_winner']

    session_df = df[df['session_date'].astype(str) == session_date[:10]].copy()
    if len(session_df) < IB_BARS_1MIN:
        continue

    post_ib = session_df.iloc[IB_BARS_1MIN:]

    # Find entry bar
    entry_bars = post_ib[abs(post_ib['close'] - entry_price) < 2.0]
    if len(entry_bars) == 0:
        continue

    entry_bar = post_ib.loc[entry_bars.index[0]]
    cum_delta = entry_bar.get('cumulative_delta', 0)

    # For LONG entries: is cumulative delta rising?
    # Check last 20 bars of cumulative delta slope
    entry_loc = list(post_ib.index).index(entry_bars.index[0])
    start_loc = max(0, entry_loc - 20)
    window = post_ib.iloc[start_loc:entry_loc + 1]

    if 'cumulative_delta' in window.columns:
        cd_values = window['cumulative_delta'].dropna()
        if len(cd_values) >= 5:
            cd_slope = (cd_values.iloc[-1] - cd_values.iloc[0]) / len(cd_values)
            cd_aligned = cd_slope > 0  # For LONG, want rising cumulative delta
        else:
            cd_slope = 0
            cd_aligned = False
    else:
        cd_slope = 0
        cd_aligned = False

    result = 'WIN' if is_winner else 'LOSS'
    align = 'ALIGNED' if cd_aligned else 'OPPOSED'

    # Only show non-obvious info
    if strategy == 'P-Day':
        continue  # Skip P-Day since they mirror Trend Day Bull

    print(f"  {session_date[:10]} {strategy:20s} | {result:4s} | ${net_pnl:>8.2f} | "
          f"CD_slope={cd_slope:>8.1f} | {align}")
