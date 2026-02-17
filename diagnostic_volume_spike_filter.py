"""
Quick test: if we add volume_spike >= 1.0 as a HARD filter (not 2-of-3),
which entries get removed?

From the deep study:
  VolumeFilter(spike>=1.0): 12/12 winners pass, 1/2 losers rejected

But the deep study matched by price from trade_log, which may not match
the actual strategy entry bar. Let's check with the actual backtest.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd

# Load the latest trade log
trade_log = pd.read_csv(Path(project_root) / 'output' / 'trade_log.csv')

from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from config.constants import IB_BARS_1MIN

df = load_csv('NQ')
df = filter_rth(df)
df = compute_all_features(df)

if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
if 'session_date' not in df.columns:
    df['session_date'] = df['timestamp'].dt.date

print(f"\n{'='*100}")
print(f"  VOLUME SPIKE AT ACTUAL ENTRY BARS (matching by entry_time)")
print(f"{'='*100}\n")

for _, trade in trade_log.iterrows():
    session_date = trade['session_date'][:10]
    entry_time_str = trade['entry_time']
    entry_price = trade['entry_price']
    strategy = trade['strategy_name']
    net_pnl = trade['net_pnl']
    is_winner = trade['is_winner']

    session_df = df[df['session_date'].astype(str) == session_date].copy()
    if len(session_df) == 0:
        continue

    # Match by entry time (more precise than price)
    entry_ts = pd.Timestamp(entry_time_str)
    time_match = session_df[session_df['timestamp'] == entry_ts]
    if len(time_match) == 0:
        # Try close price match
        time_match = session_df[abs(session_df['close'] - entry_price) < 2.0]

    if len(time_match) == 0:
        print(f"  {session_date} {strategy}: NO MATCH for entry_time={entry_time_str}")
        continue

    bar = time_match.iloc[0]
    vol_spike = bar.get('volume_spike', 1.0)
    delta_pctl = bar.get('delta_percentile', 50)
    imbalance = bar.get('imbalance_ratio', 1.0)
    delta = bar.get('delta', 0)

    result = 'WIN' if is_winner else 'LOSS'
    print(f"  {session_date} {strategy:20s} [{result}] ${net_pnl:+8.2f} | "
          f"vol_spike={vol_spike:.2f} | delta_pctl={delta_pctl:.0f} | "
          f"imb={imbalance:.3f} | delta={delta:.0f}")
