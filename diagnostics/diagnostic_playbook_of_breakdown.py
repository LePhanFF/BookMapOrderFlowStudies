"""
Diagnostic: Which OF model signals fire at each Playbook entry bar?

For every Playbook trade, check the entry bar's OF features and determine
which of the 5 standalone OF models would have also triggered there.

This answers: "Which OF models align with which Playbook strategies?"
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import time
from collections import defaultdict

from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from config.constants import IB_BARS_1MIN
from indicators.technical import calculate_ema


# ---- Load data ----
df = load_csv('NQ')
df = filter_rth(df)
df = compute_all_features(df)

if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
if 'session_date' not in df.columns:
    df['session_date'] = df['timestamp'].dt.date

# ---- Load trade log ----
trade_log = pd.read_csv(Path(project_root) / 'output' / 'trade_log.csv')

# Deduplicate: Trend Day Bull and P-Day fire on same bar for same session
# Keep one entry per unique (session_date[:10], entry_time)
seen = set()
unique_trades = []
for _, trade in trade_log.iterrows():
    key = (trade['session_date'][:10], trade['entry_time'])
    if key not in seen:
        seen.add(key)
        unique_trades.append(trade)

print(f"\n{'='*120}")
print(f"  PLAYBOOK ENTRIES x ORDER FLOW MODEL BREAKDOWN")
print(f"{'='*120}\n")
print(f"Unique entry bars: {len(unique_trades)} (from {len(trade_log)} total trades)")
print()


# ---- Build HTF bars (30-min only, best filter) ----
def build_htf_trend(df_full, period_minutes=30):
    """Build HTF EMA trend for each 1-min bar."""
    df_full = df_full.copy()
    col = f'htf_{period_minutes}m_bullish'
    df_full[col] = False

    for session_date, session_df in df_full.groupby('session_date'):
        if len(session_df) < period_minutes * 3:
            continue
        session_ts = session_df.set_index('timestamp')
        htf = session_ts.resample(f'{period_minutes}min').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum',
        }).dropna()

        if len(htf) < 22:
            continue

        ema8 = calculate_ema(htf['close'], 8)
        ema21 = calculate_ema(htf['close'], 21)
        htf_trend = ema8 > ema21

        for ts_1min in session_df['timestamp']:
            htf_idx = htf.index[htf.index <= ts_1min]
            if len(htf_idx) > 0:
                last_htf_ts = htf_idx[-1]
                if last_htf_ts in htf_trend.index:
                    df_full.loc[df_full['timestamp'] == ts_1min, col] = htf_trend[last_htf_ts]

    return df_full

print("Building 30-min HTF trend...")
df = build_htf_trend(df, 30)
print("Done.\n")


# ---- OF Model Checks ----
def check_delta_surge(bar):
    """Delta z-score > 1.5, delta > 0, vol_spike > 1.3"""
    dz = bar.get('delta_zscore', 0)
    delta = bar.get('delta', 0)
    vs = bar.get('volume_spike', 0)
    if pd.isna(dz): dz = 0
    if pd.isna(delta): delta = 0
    if pd.isna(vs): vs = 0
    return dz > 1.5 and delta > 0 and vs > 1.3

def check_absorption(bar):
    """imbalance > 1.2, vol_spike > 1.5, delta > 0, price > vwap"""
    imb = bar.get('imbalance_ratio', 0)
    vs = bar.get('volume_spike', 0)
    delta = bar.get('delta', 0)
    price = bar.get('close', 0)
    vwap = bar.get('vwap', 0)
    if pd.isna(imb): imb = 0
    if pd.isna(vs): vs = 0
    if pd.isna(delta): delta = 0
    if pd.isna(vwap): vwap = price  # fallback
    return imb > 1.2 and vs > 1.5 and delta > 0 and price > vwap

def check_cvd_divergence(bar, session_df, bar_idx_in_session):
    """Price makes local low but CVD is higher than the previous local low's CVD.
    Simplified: look back 20 bars, find lowest close, check if CVD now > CVD then."""
    if bar_idx_in_session < 20:
        return False
    lookback = session_df.iloc[max(0, bar_idx_in_session - 20):bar_idx_in_session]
    if len(lookback) == 0:
        return False
    # Find bar with lowest close in lookback
    low_idx = lookback['close'].idxmin()
    low_bar = lookback.loc[low_idx]
    current_cvd = bar.get('cumulative_delta', 0)
    prev_cvd = low_bar.get('cumulative_delta', 0)
    if pd.isna(current_cvd): current_cvd = 0
    if pd.isna(prev_cvd): prev_cvd = 0
    current_close = bar.get('close', 0)
    prev_close = low_bar.get('close', 0)
    # Price at or near the low, but CVD higher
    return current_close <= prev_close * 1.002 and current_cvd > prev_cvd and bar.get('delta', 0) > 0

def check_delta_momentum(bar, session_df, bar_idx_in_session):
    """5-bar delta sum > 300, acceleration > 0, pctl >= 70"""
    if bar_idx_in_session < 5:
        return False
    recent = session_df.iloc[max(0, bar_idx_in_session - 4):bar_idx_in_session + 1]
    deltas = recent['delta'].fillna(0).values
    delta_sum = deltas.sum()
    if len(deltas) >= 3:
        accel = deltas[-1] - deltas[-3]
    else:
        accel = 0
    pctl = bar.get('delta_percentile', 50)
    if pd.isna(pctl): pctl = 50
    return delta_sum > 300 and accel > 0 and pctl >= 70

def check_high_conviction(bar, session_df, bar_idx_in_session):
    """4+ of 5 sub-signals: delta_zs>1, imb>1.15, vol>1.3, pctl>=80, above vwap"""
    dz = bar.get('delta_zscore', 0)
    imb = bar.get('imbalance_ratio', 0)
    vs = bar.get('volume_spike', 0)
    pctl = bar.get('delta_percentile', 50)
    price = bar.get('close', 0)
    vwap = bar.get('vwap', 0)
    if pd.isna(dz): dz = 0
    if pd.isna(imb): imb = 0
    if pd.isna(vs): vs = 0
    if pd.isna(pctl): pctl = 50
    if pd.isna(vwap): vwap = price
    signals = sum([
        dz > 1.0,
        imb > 1.15,
        vs > 1.3,
        pctl >= 80,
        price > vwap,
    ])
    return signals >= 4


# ---- Analyze each Playbook entry ----
model_names = ['delta_surge', 'absorption', 'cvd_divergence', 'delta_momentum', 'high_conviction']

# Track stats
strategy_of_matrix = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'wins': 0, 'pnl': 0.0}))
entry_details = []

for trade_data in unique_trades:
    session_date_str = trade_data['session_date'][:10]
    entry_time_str = trade_data['entry_time']
    entry_price = trade_data['entry_price']
    strategy = trade_data['strategy_name']
    setup = trade_data['setup_type']
    net_pnl = trade_data['net_pnl']
    is_winner = trade_data['is_winner']

    # Get session data
    session_df = df[df['session_date'].astype(str) == session_date_str].copy().reset_index(drop=True)
    if len(session_df) == 0:
        continue

    # Match entry bar by timestamp
    entry_ts = pd.Timestamp(entry_time_str)
    match = session_df[session_df['timestamp'] == entry_ts]
    if len(match) == 0:
        # Fallback: closest price match
        match = session_df[abs(session_df['close'] - entry_price) < 2.0]
    if len(match) == 0:
        print(f"  {session_date_str} {strategy}: NO MATCH for entry_time={entry_time_str}")
        continue

    bar = match.iloc[0]
    bar_idx = match.index[0]

    # Get all OF features at entry bar
    delta = bar.get('delta', 0)
    dz = bar.get('delta_zscore', 0)
    pctl = bar.get('delta_percentile', 50)
    imb = bar.get('imbalance_ratio', 1.0)
    vs = bar.get('volume_spike', 1.0)
    cvd = bar.get('cumulative_delta', 0)
    vwap = bar.get('vwap', 0)
    htf_30m = bar.get('htf_30m_bullish', False)

    if pd.isna(delta): delta = 0
    if pd.isna(dz): dz = 0
    if pd.isna(pctl): pctl = 50
    if pd.isna(imb): imb = 1.0
    if pd.isna(vs): vs = 1.0
    if pd.isna(cvd): cvd = 0
    if pd.isna(vwap): vwap = 0

    # Check each OF model
    of_hits = {}
    of_hits['delta_surge'] = check_delta_surge(bar)
    of_hits['absorption'] = check_absorption(bar)
    of_hits['cvd_divergence'] = check_cvd_divergence(bar, session_df, bar_idx)
    of_hits['delta_momentum'] = check_delta_momentum(bar, session_df, bar_idx)
    of_hits['high_conviction'] = check_high_conviction(bar, session_df, bar_idx)

    # Also compute the Playbook OF quality gate
    of_quality = sum([
        (pctl >= 60),
        (imb > 1.0),
        (vs >= 1.0),
    ])

    # Build display
    hit_models = [m for m in model_names if of_hits[m]]
    miss_models = [m for m in model_names if not of_hits[m]]

    result_str = 'WIN' if is_winner else 'LOSS'
    htf_str = '30m' if htf_30m else '---'

    # Get all trades for this entry (Trend Bull + P-Day fire together)
    all_strategies = []
    for t in unique_trades:
        pass
    # Actually just get from trade_log
    matching_trades = trade_log[
        (trade_log['session_date'].str[:10] == session_date_str) &
        (trade_log['entry_time'] == entry_time_str)
    ]
    strat_list = matching_trades['strategy_name'].unique().tolist()

    entry_details.append({
        'date': session_date_str,
        'time': entry_time_str.split(' ')[-1] if ' ' in entry_time_str else entry_time_str[-8:],
        'strategies': strat_list,
        'setup': setup,
        'result': result_str,
        'pnl': net_pnl,
        'delta': delta,
        'dz': dz,
        'pctl': pctl,
        'imb': imb,
        'vs': vs,
        'of_quality': of_quality,
        'htf_30m': htf_30m,
        'hit_models': hit_models,
        'miss_models': miss_models,
        'of_hits': of_hits,
    })

    # Update matrix
    for strat in strat_list:
        for model in model_names:
            if of_hits[model]:
                strategy_of_matrix[strat][model]['total'] += 1
                if is_winner:
                    strategy_of_matrix[strat][model]['wins'] += 1
                strategy_of_matrix[strat][model]['pnl'] += net_pnl


# ---- Print detailed entry breakdown ----
print(f"{'='*120}")
print(f"  DETAILED ENTRY BREAKDOWN: OF Models at Each Playbook Entry")
print(f"{'='*120}\n")

for e in entry_details:
    strats = ' + '.join(e['strategies'])
    hits = ', '.join(e['hit_models']) if e['hit_models'] else 'NONE'
    misses = ', '.join(e['miss_models']) if e['miss_models'] else 'NONE'

    print(f"  {e['date']} {e['time']}  {strats:30s}  [{e['result']}] ${e['pnl']:+8.2f}")
    print(f"    OF Features: delta={e['delta']:+.0f}  dz={e['dz']:.2f}  pctl={e['pctl']:.0f}"
          f"  imb={e['imb']:.3f}  vol_spike={e['vs']:.2f}  QG={e['of_quality']}/3  HTF30m={'Y' if e['htf_30m'] else 'N'}")
    print(f"    OF Models HIT:  {hits}")
    print(f"    OF Models MISS: {misses}")
    print()


# ---- Strategy x OF Model Matrix ----
print(f"\n{'='*120}")
print(f"  STRATEGY x OF MODEL MATRIX: How many Playbook entries each OF model would have caught")
print(f"{'='*120}\n")

# Get unique strategies from trade log
all_strats = sorted(set(s for e in entry_details for s in e['strategies']))
total_entries = len(entry_details)
total_wins = sum(1 for e in entry_details if e['result'] == 'WIN')

print(f"{'Strategy':<25s}", end='')
for model in model_names:
    print(f"  {model:>16s}", end='')
print(f"  {'Playbook QG':>12s}")

print(f"{'-'*25}", end='')
for _ in model_names:
    print(f"  {'-'*16}", end='')
print(f"  {'-'*12}")

for strat in all_strats:
    strat_entries = [e for e in entry_details if strat in e['strategies']]
    strat_wins = sum(1 for e in strat_entries if e['result'] == 'WIN')
    print(f"{strat:<25s}", end='')
    for model in model_names:
        hits = sum(1 for e in strat_entries if e['of_hits'][model])
        hit_wins = sum(1 for e in strat_entries if e['of_hits'][model] and e['result'] == 'WIN')
        hit_pnl = sum(e['pnl'] for e in strat_entries if e['of_hits'][model])
        if hits > 0:
            print(f"  {hits:2d}/{len(strat_entries):2d} ({hit_wins}W ${hit_pnl:+.0f})", end='')
        else:
            print(f"  {'---':>16s}", end='')
    # Playbook quality gate hits
    qg_hits = sum(1 for e in strat_entries if e['of_quality'] >= 2)
    print(f"  {qg_hits:2d}/{len(strat_entries):2d}")

print(f"\n{'All Entries':<25s}", end='')
for model in model_names:
    hits = sum(1 for e in entry_details if e['of_hits'][model])
    hit_wins = sum(1 for e in entry_details if e['of_hits'][model] and e['result'] == 'WIN')
    hit_pnl = sum(e['pnl'] for e in entry_details if e['of_hits'][model])
    if hits > 0:
        print(f"  {hits:2d}/{total_entries:2d} ({hit_wins}W ${hit_pnl:+.0f})", end='')
    else:
        print(f"  {'---':>16s}", end='')
qg_all = sum(1 for e in entry_details if e['of_quality'] >= 2)
print(f"  {qg_all:2d}/{total_entries:2d}")


# ---- OF Model Hit Rate on Playbook Winners vs Losers ----
print(f"\n\n{'='*120}")
print(f"  OF MODEL HIT RATE: Winners vs Losers")
print(f"{'='*120}\n")

winners = [e for e in entry_details if e['result'] == 'WIN']
losers = [e for e in entry_details if e['result'] == 'LOSS']

print(f"{'OF Model':<20s}  {'Hit on Winners':>16s}  {'Hit on Losers':>16s}  {'Discrimination':>14s}")
print(f"{'-'*20}  {'-'*16}  {'-'*16}  {'-'*14}")

for model in model_names:
    w_hits = sum(1 for e in winners if e['of_hits'][model])
    l_hits = sum(1 for e in losers if e['of_hits'][model])
    w_rate = w_hits / len(winners) * 100 if winners else 0
    l_rate = l_hits / len(losers) * 100 if losers else 0
    disc = w_rate - l_rate
    disc_str = f"+{disc:.0f}pp" if disc >= 0 else f"{disc:.0f}pp"
    print(f"{model:<20s}  {w_hits:2d}/{len(winners):2d} ({w_rate:5.1f}%)  "
          f"{l_hits:2d}/{len(losers):2d} ({l_rate:5.1f}%)  {disc_str:>14s}")

# Quality gate
w_qg = sum(1 for e in winners if e['of_quality'] >= 2)
l_qg = sum(1 for e in losers if e['of_quality'] >= 2)
w_qg_rate = w_qg / len(winners) * 100 if winners else 0
l_qg_rate = l_qg / len(losers) * 100 if losers else 0
qg_disc = w_qg_rate - l_qg_rate
print(f"{'Playbook QG (2/3)':<20s}  {w_qg:2d}/{len(winners):2d} ({w_qg_rate:5.1f}%)  "
      f"{l_qg:2d}/{len(losers):2d} ({l_qg_rate:5.1f}%)  {'+' if qg_disc >= 0 else ''}{qg_disc:.0f}pp")


# ---- Per-Strategy OF Feature Averages ----
print(f"\n\n{'='*120}")
print(f"  OF FEATURE AVERAGES BY STRATEGY + OUTCOME")
print(f"{'='*120}\n")

for strat in all_strats:
    strat_entries = [e for e in entry_details if strat in e['strategies']]
    strat_winners = [e for e in strat_entries if e['result'] == 'WIN']
    strat_losers = [e for e in strat_entries if e['result'] == 'LOSS']

    print(f"  {strat} ({len(strat_entries)} entries: {len(strat_winners)}W / {len(strat_losers)}L)")

    for label, subset in [('  Winners', strat_winners), ('  Losers', strat_losers)]:
        if not subset:
            print(f"    {label}: (none)")
            continue
        avg_delta = np.mean([e['delta'] for e in subset])
        avg_dz = np.mean([e['dz'] for e in subset])
        avg_pctl = np.mean([e['pctl'] for e in subset])
        avg_imb = np.mean([e['imb'] for e in subset])
        avg_vs = np.mean([e['vs'] for e in subset])
        avg_qg = np.mean([e['of_quality'] for e in subset])
        avg_pnl = np.mean([e['pnl'] for e in subset])
        htf_pct = sum(1 for e in subset if e['htf_30m']) / len(subset) * 100
        print(f"    {label}: delta={avg_delta:+.0f}  dz={avg_dz:.2f}  pctl={avg_pctl:.0f}"
              f"  imb={avg_imb:.3f}  vol={avg_vs:.2f}  QG={avg_qg:.1f}/3"
              f"  HTF30m={htf_pct:.0f}%  avg_pnl=${avg_pnl:+.0f}")
    print()


# ---- Which OF models would have REPLACED the Playbook? ----
print(f"\n{'='*120}")
print(f"  OVERLAP ANALYSIS: Playbook entries that OF models would also catch")
print(f"{'='*120}\n")

# For each model, which playbook entries does it overlap with?
for model in model_names:
    hits = [e for e in entry_details if e['of_hits'][model]]
    misses = [e for e in entry_details if not e['of_hits'][model]]
    hit_wins = sum(1 for e in hits if e['result'] == 'WIN')
    miss_wins = sum(1 for e in misses if e['result'] == 'WIN')
    hit_pnl = sum(e['pnl'] for e in hits)
    miss_pnl = sum(e['pnl'] for e in misses)

    print(f"  {model}:")
    print(f"    Catches {len(hits)}/{total_entries} Playbook entries"
          f" ({hit_wins}W/{len(hits)-hit_wins}L, ${hit_pnl:+.0f})")
    print(f"    Misses  {len(misses)}/{total_entries} Playbook entries"
          f" ({miss_wins}W/{len(misses)-miss_wins}L, ${miss_pnl:+.0f})")

    if hits:
        hit_dates = [f"{e['date'][-5:]}({e['result'][0]})" for e in hits]
        print(f"    Caught: {', '.join(hit_dates)}")
    if misses:
        miss_dates = [f"{e['date'][-5:]}({e['result'][0]})" for e in misses]
        print(f"    Missed: {', '.join(miss_dates)}")
    print()


# ---- Summary ----
print(f"\n{'='*120}")
print(f"  SUMMARY")
print(f"{'='*120}\n")

print(f"  Total unique Playbook entries: {total_entries}")
print(f"  Winners: {total_wins}, Losers: {total_entries - total_wins}")
print()

# Count how many models fire on each entry
for e in entry_details:
    e['n_models'] = sum(1 for m in model_names if e['of_hits'][m])

for n in range(6):
    entries_n = [e for e in entry_details if e['n_models'] == n]
    if entries_n:
        wins_n = sum(1 for e in entries_n if e['result'] == 'WIN')
        pnl_n = sum(e['pnl'] for e in entries_n)
        print(f"  {n} OF models fire: {len(entries_n)} entries, {wins_n}W/{len(entries_n)-wins_n}L, ${pnl_n:+.0f}")

print()
# Best discriminator
print(f"  Best OF model overlap with Playbook winners:")
for model in model_names:
    w_hits = sum(1 for e in winners if e['of_hits'][model])
    l_hits = sum(1 for e in losers if e['of_hits'][model])
    print(f"    {model:<20s}: catches {w_hits}/{len(winners)} winners, {l_hits}/{len(losers)} losers")
