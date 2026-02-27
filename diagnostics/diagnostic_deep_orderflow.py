"""
Diagnostic 8: DEEP ORDER FLOW STUDY

Analyze EVERY order flow feature at each entry bar across all 22 trades.
Goal: Find which features systematically distinguish winners from losers,
and which features are currently unused but could improve filtering.

Features analyzed:
  - delta (raw bar delta)
  - delta_zscore (how unusual is this bar's delta vs recent history)
  - delta_percentile (rank of delta vs recent bars)
  - cumulative_delta (session CVD at entry)
  - cumulative_delta_ma (smoothed CVD)
  - CVD > CVD_MA (CVD trend alignment - the CVDFilter check)
  - imbalance_ratio (vol_ask / vol_bid)
  - volume_spike (volume vs 20-bar MA)
  - fvg_bull / fvg_bear (inside a FVG zone?)
  - ifvg_bull_entry (IFVG confluence?)
  - fvg_bull_15m (higher-TF FVG confluence?)
  - in_bpr (balanced price range?)
  - pre_delta_sum (10-bar delta momentum before entry)
  - pre_delta_trend (is delta accelerating or decelerating?)
  - vol_ask, vol_bid (raw aggressive volume)

Also computes:
  - 5-bar post-entry delta flow (did buyers follow through?)
  - Max favorable excursion (MFE) within 30 bars
  - Max adverse excursion (MAE) within 30 bars
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

from config.instruments import get_instrument
from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from config.constants import IB_BARS_1MIN, ACCEPTANCE_MIN_BARS, PM_SESSION_START, LONDON_CLOSE
from strategy.day_type import classify_trend_strength, classify_day_type
from strategy.day_confidence import DayTypeConfidenceScorer


def analyze():
    df = load_csv('NQ')
    df = filter_rth(df)
    df = compute_all_features(df)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    if 'session_date' not in df.columns:
        df['session_date'] = df['timestamp'].dt.date

    sessions = sorted(df['session_date'].unique())

    # Load trade log to find actual entry bars
    trade_log = pd.read_csv(Path(project_root) / 'output' / 'trade_log.csv')

    print(f"\n{'='*140}")
    print(f"  DEEP ORDER FLOW STUDY: ALL FEATURES AT EVERY ENTRY BAR")
    print(f"{'='*140}\n")

    all_entries = []

    for _, trade in trade_log.iterrows():
        session_date = trade['session_date']
        entry_price = trade['entry_price']
        strategy = trade['strategy_name']
        setup = trade['setup_type']
        net_pnl = trade['net_pnl']
        is_winner = trade['is_winner']
        exit_reason = trade['exit_reason']

        session_df = df[df['session_date'].astype(str) == session_date[:10]].copy()
        if len(session_df) < IB_BARS_1MIN:
            continue

        ib_df = session_df.head(IB_BARS_1MIN)
        ib_high = ib_df['high'].max()
        ib_low = ib_df['low'].min()
        ib_range = ib_high - ib_low

        post_ib = session_df.iloc[IB_BARS_1MIN:]
        if len(post_ib) == 0:
            continue

        # Find the entry bar (closest price match)
        entry_bars = post_ib[abs(post_ib['close'] - entry_price) < 3.0]
        if len(entry_bars) == 0:
            continue

        entry_idx_pos = post_ib.index.get_loc(entry_bars.index[0])
        entry_bar = post_ib.iloc[entry_idx_pos]
        entry_ts = entry_bar.get('timestamp')

        # === Collect ALL order flow features at entry ===
        features = {}
        features['session_date'] = session_date[:10]
        features['strategy'] = strategy
        features['setup'] = setup
        features['result'] = 'WIN' if is_winner else 'LOSS'
        features['net_pnl'] = net_pnl
        features['exit_reason'] = exit_reason
        features['entry_time'] = entry_ts.time() if hasattr(entry_ts, 'time') else None
        features['entry_price'] = entry_price
        features['ib_range'] = ib_range

        # Core delta features
        features['delta'] = entry_bar.get('delta', 0)
        features['delta_pct'] = entry_bar.get('delta_pct', 0)
        features['delta_zscore'] = entry_bar.get('delta_zscore', 0)
        features['delta_percentile'] = entry_bar.get('delta_percentile', 50)

        # CVD features
        features['cumulative_delta'] = entry_bar.get('cumulative_delta', 0)
        features['cumulative_delta_ma'] = entry_bar.get('cumulative_delta_ma', 0)
        cvd = entry_bar.get('cumulative_delta', 0)
        cvd_ma = entry_bar.get('cumulative_delta_ma', 0)
        features['cvd_above_ma'] = bool(cvd > cvd_ma) if not (pd.isna(cvd) or pd.isna(cvd_ma)) else None
        features['cvd_spread'] = (cvd - cvd_ma) if not (pd.isna(cvd) or pd.isna(cvd_ma)) else 0

        # Imbalance and volume
        features['imbalance_ratio'] = entry_bar.get('imbalance_ratio', 1.0)
        features['volume_spike'] = entry_bar.get('volume_spike', 1.0)
        features['volume'] = entry_bar.get('volume', 0)
        features['vol_ask'] = entry_bar.get('vol_ask', 0)
        features['vol_bid'] = entry_bar.get('vol_bid', 0)

        # FVG/IFVG/BPR features
        features['fvg_bull'] = bool(entry_bar.get('fvg_bull', False))
        features['fvg_bear'] = bool(entry_bar.get('fvg_bear', False))
        features['ifvg_bull_entry'] = bool(entry_bar.get('ifvg_bull_entry', False))
        features['ifvg_bear_entry'] = bool(entry_bar.get('ifvg_bear_entry', False))
        features['fvg_bull_15m'] = bool(entry_bar.get('fvg_bull_15m', False))
        features['fvg_bear_15m'] = bool(entry_bar.get('fvg_bear_15m', False))
        features['in_bpr'] = bool(entry_bar.get('in_bpr', False))

        # Pre-entry momentum (10 bars before)
        pre_start = max(0, entry_idx_pos - 10)
        pre_bars = post_ib.iloc[pre_start:entry_idx_pos]
        if len(pre_bars) > 0 and 'delta' in pre_bars.columns:
            pre_deltas = pre_bars['delta'].fillna(0)
            features['pre_delta_sum'] = float(pre_deltas.sum())
            features['pre_delta_avg'] = float(pre_deltas.mean())
            # Is delta accelerating? Compare first half vs second half
            half = len(pre_deltas) // 2
            if half > 0:
                first_half = pre_deltas.iloc[:half].sum()
                second_half = pre_deltas.iloc[half:].sum()
                features['delta_accel'] = float(second_half - first_half)
            else:
                features['delta_accel'] = 0.0
            # Pre-entry CVD slope
            if 'cumulative_delta' in pre_bars.columns:
                cd_vals = pre_bars['cumulative_delta'].dropna()
                if len(cd_vals) >= 3:
                    features['pre_cvd_slope'] = float((cd_vals.iloc[-1] - cd_vals.iloc[0]) / len(cd_vals))
                else:
                    features['pre_cvd_slope'] = 0.0
            else:
                features['pre_cvd_slope'] = 0.0
            # Pre-entry imbalance trend
            if 'imbalance_ratio' in pre_bars.columns:
                imb_vals = pre_bars['imbalance_ratio'].dropna()
                features['pre_imb_avg'] = float(imb_vals.mean()) if len(imb_vals) > 0 else 1.0
            else:
                features['pre_imb_avg'] = 1.0
        else:
            features['pre_delta_sum'] = 0.0
            features['pre_delta_avg'] = 0.0
            features['delta_accel'] = 0.0
            features['pre_cvd_slope'] = 0.0
            features['pre_imb_avg'] = 1.0

        # Post-entry flow (5 bars after)
        post_start = entry_idx_pos + 1
        post_end = min(len(post_ib), entry_idx_pos + 6)
        post_bars = post_ib.iloc[post_start:post_end]
        if len(post_bars) > 0 and 'delta' in post_bars.columns:
            post_deltas = post_bars['delta'].fillna(0)
            features['post_delta_sum'] = float(post_deltas.sum())
            features['post_delta_avg'] = float(post_deltas.mean())
            features['post_positive_bars'] = int((post_deltas > 0).sum())
        else:
            features['post_delta_sum'] = 0.0
            features['post_delta_avg'] = 0.0
            features['post_positive_bars'] = 0

        # MFE/MAE within 30 bars
        mfe_window = min(len(post_ib), entry_idx_pos + 31)
        mfe_bars = post_ib.iloc[entry_idx_pos + 1:mfe_window]
        if len(mfe_bars) > 0:
            features['mfe_30'] = float(mfe_bars['high'].max() - entry_price)
            features['mae_30'] = float(entry_price - mfe_bars['low'].min())
        else:
            features['mfe_30'] = 0.0
            features['mae_30'] = 0.0

        all_entries.append(features)

    # === ANALYSIS ===
    entries_df = pd.DataFrame(all_entries)

    # Skip duplicate P-Day entries (same session, same price as Trend Bull)
    # Focus on unique entries
    unique_entries = entries_df.drop_duplicates(subset=['session_date', 'entry_price'])

    winners = unique_entries[unique_entries['result'] == 'WIN']
    losers = unique_entries[unique_entries['result'] == 'LOSS']

    print(f"Total unique entries: {len(unique_entries)} ({len(winners)} W, {len(losers)} L)")
    print()

    # --- Feature Comparison Table ---
    print(f"{'='*140}")
    print(f"  FEATURE COMPARISON: WINNERS vs LOSERS")
    print(f"{'='*140}\n")

    numeric_features = [
        'delta', 'delta_zscore', 'delta_percentile',
        'cumulative_delta', 'cvd_spread',
        'imbalance_ratio', 'volume_spike', 'volume',
        'vol_ask', 'vol_bid',
        'pre_delta_sum', 'pre_delta_avg', 'delta_accel', 'pre_cvd_slope', 'pre_imb_avg',
        'post_delta_sum', 'post_delta_avg', 'post_positive_bars',
        'mfe_30', 'mae_30', 'ib_range',
    ]

    print(f"{'Feature':<25s}  {'Winners (avg)':>14s}  {'Losers (avg)':>14s}  {'W median':>10s}  {'L median':>10s}  {'Diff':>10s}  {'Signal':>8s}")
    print(f"{'-'*25}  {'-'*14}  {'-'*14}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")

    for feat in numeric_features:
        if feat not in unique_entries.columns:
            continue
        w_vals = winners[feat].dropna()
        l_vals = losers[feat].dropna()
        w_avg = w_vals.mean() if len(w_vals) > 0 else 0
        l_avg = l_vals.mean() if len(l_vals) > 0 else 0
        w_med = w_vals.median() if len(w_vals) > 0 else 0
        l_med = l_vals.median() if len(l_vals) > 0 else 0
        diff = w_avg - l_avg
        # Signal: how much they differ (rough z-score)
        combined_std = unique_entries[feat].dropna().std()
        signal = abs(diff) / combined_std if combined_std > 0 else 0

        print(f"{feat:<25s}  {w_avg:>14.2f}  {l_avg:>14.2f}  {w_med:>10.2f}  {l_med:>10.2f}  {diff:>+10.2f}  {signal:>8.2f}")

    # --- Boolean Feature Analysis ---
    print(f"\n{'='*140}")
    print(f"  BOOLEAN FEATURES: PRESENCE IN WINNERS vs LOSERS")
    print(f"{'='*140}\n")

    bool_features = ['cvd_above_ma', 'fvg_bull', 'ifvg_bull_entry', 'fvg_bull_15m', 'in_bpr']

    for feat in bool_features:
        if feat not in unique_entries.columns:
            continue
        w_true = winners[feat].sum() if feat in winners.columns else 0
        w_total = len(winners)
        l_true = losers[feat].sum() if feat in losers.columns else 0
        l_total = len(losers)
        w_pct = (w_true / w_total * 100) if w_total > 0 else 0
        l_pct = (l_true / l_total * 100) if l_total > 0 else 0
        print(f"  {feat:<25s}: Winners {w_true}/{w_total} ({w_pct:.0f}%)  |  Losers {l_true}/{l_total} ({l_pct:.0f}%)")

    # --- Per-Entry Detail ---
    print(f"\n{'='*140}")
    print(f"  INDIVIDUAL ENTRY DETAILS (UNIQUE ENTRIES)")
    print(f"{'='*140}\n")

    for _, entry in unique_entries.iterrows():
        result_tag = entry['result']
        pnl = entry['net_pnl']
        print(f"--- {entry['session_date']} [{result_tag}: ${pnl:+.2f}] {entry['strategy']} / {entry['setup']} ---")
        print(f"  Time: {entry['entry_time']}  Price: {entry['entry_price']:.1f}  IB: {entry['ib_range']:.1f}")
        print(f"  Delta: {entry['delta']:.0f}  |  zscore: {entry['delta_zscore']:.2f}  |  pctl: {entry['delta_percentile']:.0f}th")
        print(f"  CVD: {entry['cumulative_delta']:.0f}  |  CVD_MA: {entry['cumulative_delta_ma']:.0f}  |  CVD>MA: {entry['cvd_above_ma']}  |  spread: {entry['cvd_spread']:.0f}")
        print(f"  Imbalance: {entry['imbalance_ratio']:.3f}  |  VolSpike: {entry['volume_spike']:.2f}  |  Vol: {entry['volume']:.0f}")
        print(f"  Ask: {entry['vol_ask']:.0f}  |  Bid: {entry['vol_bid']:.0f}")
        print(f"  FVG_bull: {entry['fvg_bull']}  |  IFVG: {entry['ifvg_bull_entry']}  |  15m_FVG: {entry['fvg_bull_15m']}  |  BPR: {entry['in_bpr']}")
        print(f"  Pre-10bar delta: sum={entry['pre_delta_sum']:.0f}  avg={entry['pre_delta_avg']:.0f}  accel={entry['delta_accel']:.0f}  cvd_slope={entry['pre_cvd_slope']:.1f}")
        print(f"  Pre-10bar imbalance avg: {entry['pre_imb_avg']:.3f}")
        print(f"  Post-5bar delta: sum={entry['post_delta_sum']:.0f}  avg={entry['post_delta_avg']:.0f}  pos_bars={entry['post_positive_bars']}/5")
        print(f"  MFE(30): +{entry['mfe_30']:.1f}  |  MAE(30): -{entry['mae_30']:.1f}")
        print(f"  Exit: {entry['exit_reason']}")
        print()

    # --- Filter Simulation ---
    print(f"\n{'='*140}")
    print(f"  FILTER SIMULATION: WHAT IF WE APPLIED EACH FILTER?")
    print(f"{'='*140}\n")

    # Test each filter independently
    filter_tests = [
        ('DeltaFilter(pctl>=60)', lambda e: e['delta_percentile'] >= 60),
        ('DeltaFilter(pctl>=70)', lambda e: e['delta_percentile'] >= 70),
        ('DeltaFilter(pctl>=80)', lambda e: e['delta_percentile'] >= 80),
        ('CVDFilter(cvd>ma)', lambda e: e['cvd_above_ma'] == True),
        ('VolumeFilter(spike>=1.2)', lambda e: e['volume_spike'] >= 1.2),
        ('VolumeFilter(spike>=1.5)', lambda e: e['volume_spike'] >= 1.5),
        ('Imbalance(>1.1)', lambda e: e['imbalance_ratio'] > 1.1),
        ('Imbalance(>1.2)', lambda e: e['imbalance_ratio'] > 1.2),
        ('DeltaZscore(>0.5)', lambda e: e['delta_zscore'] > 0.5),
        ('DeltaZscore(>1.0)', lambda e: e['delta_zscore'] > 1.0),
        ('PreDelta(>-200)', lambda e: e['pre_delta_sum'] > -200),
        ('PreDelta(>0)', lambda e: e['pre_delta_sum'] > 0),
        ('PreCVDSlope(>0)', lambda e: e['pre_cvd_slope'] > 0),
        ('DeltaAccel(>0)', lambda e: e['delta_accel'] > 0),
        ('FVG_bull(True)', lambda e: e['fvg_bull'] == True),
        ('IFVG_bull(True)', lambda e: e['ifvg_bull_entry'] == True),
        ('FVG_15m(True)', lambda e: e['fvg_bull_15m'] == True),
        ('PostDelta(>0)', lambda e: e['post_delta_sum'] > 0),
    ]

    # Combination tests
    combo_tests = [
        ('CVD>MA + PreDelta>-200', lambda e: (e['cvd_above_ma'] == True) and (e['pre_delta_sum'] > -200)),
        ('CVD>MA + Imb>1.1', lambda e: (e['cvd_above_ma'] == True) and (e['imbalance_ratio'] > 1.1)),
        ('CVD>MA + DeltaPctl>=60', lambda e: (e['cvd_above_ma'] == True) and (e['delta_percentile'] >= 60)),
        ('PreDelta>-200 + Imb>1.1', lambda e: (e['pre_delta_sum'] > -200) and (e['imbalance_ratio'] > 1.1)),
        ('PreDelta>0 + CVD>MA', lambda e: (e['pre_delta_sum'] > 0) and (e['cvd_above_ma'] == True)),
        ('DeltaAccel>0 + CVD>MA', lambda e: (e['delta_accel'] > 0) and (e['cvd_above_ma'] == True)),
    ]

    print(f"{'Filter':<35s}  {'Pass':>5s}  {'Win':>5s}  {'Loss':>5s}  {'WR':>7s}  {'Net P&L':>10s}  {'Reject_W':>10s}  {'Reject_L':>10s}")
    print(f"{'-'*35}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*10}")

    for name, test_fn in filter_tests + combo_tests:
        passing = unique_entries[unique_entries.apply(test_fn, axis=1)]
        failing = unique_entries[~unique_entries.apply(test_fn, axis=1)]
        n_pass = len(passing)
        n_win = len(passing[passing['result'] == 'WIN'])
        n_loss = len(passing[passing['result'] == 'LOSS'])
        wr = (n_win / n_pass * 100) if n_pass > 0 else 0
        pnl = passing['net_pnl'].sum()
        rej_w = len(failing[failing['result'] == 'WIN'])
        rej_l = len(failing[failing['result'] == 'LOSS'])
        print(f"{name:<35s}  {n_pass:>5d}  {n_win:>5d}  {n_loss:>5d}  {wr:>6.1f}%  ${pnl:>9.2f}  {rej_w:>10d}  {rej_l:>10d}")

    # --- Strategy-specific analysis ---
    print(f"\n{'='*140}")
    print(f"  STRATEGY-SPECIFIC ORDER FLOW PATTERNS")
    print(f"{'='*140}\n")

    for strat in unique_entries['strategy'].unique():
        strat_entries = unique_entries[unique_entries['strategy'] == strat]
        s_winners = strat_entries[strat_entries['result'] == 'WIN']
        s_losers = strat_entries[strat_entries['result'] == 'LOSS']
        print(f"=== {strat}: {len(s_winners)}W / {len(s_losers)}L ===")
        for feat in ['delta_zscore', 'delta_percentile', 'cvd_spread', 'imbalance_ratio',
                      'volume_spike', 'pre_delta_sum', 'pre_cvd_slope']:
            w_val = s_winners[feat].mean() if len(s_winners) > 0 else 0
            l_val = s_losers[feat].mean() if len(s_losers) > 0 else 0
            print(f"  {feat:<20s}: W avg={w_val:>10.2f}  |  L avg={l_val:>10.2f}  |  diff={w_val-l_val:>+10.2f}")
        print()


if __name__ == '__main__':
    analyze()
