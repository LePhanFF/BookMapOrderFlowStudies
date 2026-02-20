"""
Diagnostic: Entry Model Optimization for Edge Fade Strategy

Tests multiple entry confirmation techniques to improve Edge Fade's capture rate.
Current EDGE_TO_MID averages $86/trade but many entries are poorly timed.

Entry confirmation models tested:
1. INVERSION_1M: 1-min bearish-to-bullish inversion candle (for longs)
2. INVERSION_5M: 5-min inversion candle confirmation
3. CVD_DIVERGENCE: Price making new low but CVD turning up (buying into weakness)
4. SMT_ES_DIVERGENCE: NQ makes new low but ES doesn't (or vice versa)
5. FVG_CONFLUENCE: Entry bar is in a bullish FVG zone
6. IFVG_CONFLUENCE: Entry bar has IFVG pullback confirmation
7. RECLAIM_CONFIRM: Price sweeps below level, then reclaims with 2+ bars above
8. FLAG_BREAK: Tight consolidation (3+ bars) then breakout
9. DELTA_SURGE: Entry on delta spike (>80th percentile) at edge

Each model is tested as an ADDITIONAL FILTER on top of existing Edge Fade logic.
We measure: trades kept, WR improvement, avg capture improvement, and net P&L.

Usage:
    python diagnostic_entry_models.py
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import time as _time

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from data.loader import load_csv
from data.features import compute_all_features
from indicators.ict_models import compute_fvg_features, compute_fvg_15m


# ============================================================================
#  LOAD & PREPARE DATA
# ============================================================================

def load_instrument_data(instrument='NQ'):
    """Load and compute all features for an instrument."""
    df = load_csv(instrument)
    df = compute_all_features(df)
    return df


def resample_to_5min(df):
    """Resample 1-min bars to 5-min bars per session."""
    results = []
    for session_date, session_df in df.groupby('session_date'):
        # Resample OHLCV
        ohlc = session_df.set_index('timestamp').resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'vol_ask': 'sum',
            'vol_bid': 'sum',
            'vol_delta': 'sum',
        }).dropna()
        ohlc['session_date'] = session_date
        # Compute delta for 5-min bars
        ohlc['delta'] = ohlc['vol_ask'] - ohlc['vol_bid']
        results.append(ohlc)
    if not results:
        return pd.DataFrame()
    return pd.concat(results).reset_index()


def load_es_data():
    """Load ES data for SMT divergence analysis."""
    try:
        es_df = load_csv('ES')
        es_df = compute_all_features(es_df)
        return es_df
    except Exception as e:
        print(f"  WARNING: Could not load ES data: {e}")
        return None


# ============================================================================
#  EDGE FADE ENTRY DETECTION (replicate strategy logic)
# ============================================================================

EDGE_ZONE_PCT = 0.25
EDGE_STOP_BUFFER = 0.15
EDGE_MIN_RR = 1.0
EDGE_FADE_COOLDOWN_BARS = 20
EDGE_FADE_MIN_IB_RANGE = 50.0
EDGE_FADE_MAX_IB_RANGE = 350.0
EDGE_FADE_LAST_ENTRY_TIME = _time(15, 20)


def find_edge_fade_entries(df):
    """
    Find all bars where Edge Fade EDGE_TO_MID would trigger.
    Returns list of entry dicts with context for confirmation testing.
    """
    entries = []

    for session_date, session_df in df.groupby('session_date'):
        session_df = session_df.sort_values('timestamp').reset_index(drop=True)

        # Get IB values
        ib_high = session_df['ib_high'].iloc[0] if 'ib_high' in session_df.columns else None
        ib_low = session_df['ib_low'].iloc[0] if 'ib_low' in session_df.columns else None
        if pd.isna(ib_high) or pd.isna(ib_low):
            continue
        ib_range = ib_high - ib_low
        if not (EDGE_FADE_MIN_IB_RANGE <= ib_range <= EDGE_FADE_MAX_IB_RANGE):
            continue
        ib_mid = (ib_high + ib_low) / 2

        # Day type filter
        day_type = session_df['day_type'].iloc[-1] if 'day_type' in session_df.columns else ''
        if isinstance(day_type, str):
            day_type = day_type.lower()
        if day_type not in ('b_day', 'neutral'):
            continue

        last_entry_bar = -EDGE_FADE_COOLDOWN_BARS
        edge_ceiling = ib_low + ib_range * EDGE_ZONE_PCT

        for idx in range(len(session_df)):
            bar = session_df.iloc[idx]
            current_price = bar['close']
            ts = bar['timestamp']

            # Time gate
            bar_time = ts.time() if hasattr(ts, 'time') else None
            if bar_time and bar_time >= EDGE_FADE_LAST_ENTRY_TIME:
                continue

            # Skip IB period
            if idx < 60:
                continue

            # Cooldown
            if idx - last_entry_bar < EDGE_FADE_COOLDOWN_BARS:
                continue

            # Price in lower 25% of IB
            if current_price <= ib_low or current_price >= edge_ceiling:
                continue

            # Delta > 0
            delta = bar.get('delta', 0)
            if pd.isna(delta):
                delta = 0
            if delta <= 0:
                continue

            # OF quality gate (2 of 3)
            delta_pctl = bar.get('delta_percentile', 50)
            imbalance = bar.get('imbalance_ratio', 1.0)
            vol_spike = bar.get('volume_spike', 1.0)
            if pd.isna(delta_pctl): delta_pctl = 50
            if pd.isna(imbalance): imbalance = 1.0
            if pd.isna(vol_spike): vol_spike = 1.0

            of_quality = sum([delta_pctl >= 60, imbalance > 1.0, vol_spike >= 1.0])
            if of_quality < 2:
                continue

            # R:R check
            stop = ib_low - ib_range * EDGE_STOP_BUFFER
            target = ib_mid
            risk = current_price - stop
            reward = target - current_price
            if risk <= 0 or reward <= 0:
                continue
            if reward / risk < EDGE_MIN_RR:
                continue

            last_entry_bar = idx

            # Compute forward results (MFE, MAE, exit price at various horizons)
            forward = compute_forward_results(session_df, idx, stop, target)

            entries.append({
                'session_date': session_date,
                'timestamp': ts,
                'bar_index': idx,
                'entry_price': current_price,
                'stop_price': stop,
                'target_price': target,
                'ib_high': ib_high,
                'ib_low': ib_low,
                'ib_mid': ib_mid,
                'ib_range': ib_range,
                'day_type': day_type,
                'delta': delta,
                'delta_percentile': delta_pctl,
                'imbalance_ratio': imbalance,
                'volume_spike': vol_spike,
                'of_quality': of_quality,
                'risk_pts': risk,
                'reward_pts': reward,
                'rr_ratio': reward / risk,
                **forward,
            })

    return entries


def compute_forward_results(session_df, entry_idx, stop, target):
    """Compute forward P&L outcomes from an entry bar."""
    entry_price = session_df.iloc[entry_idx]['close']
    n_bars = len(session_df)

    mfe = 0.0  # Max favorable excursion (highest close - entry)
    mae = 0.0  # Max adverse excursion (entry - lowest low)
    exit_price = entry_price
    exit_reason = 'EOD'
    bars_held = 0

    for i in range(entry_idx + 1, n_bars):
        bar = session_df.iloc[i]
        bars_held = i - entry_idx

        # Track MFE/MAE
        excursion_up = bar['high'] - entry_price
        excursion_down = entry_price - bar['low']
        if excursion_up > mfe:
            mfe = excursion_up
        if excursion_down > mae:
            mae = excursion_down

        # Check stop
        if bar['low'] <= stop:
            exit_price = stop
            exit_reason = 'STOP'
            break

        # Check target
        if bar['high'] >= target:
            exit_price = target
            exit_reason = 'TARGET'
            break

        # Last bar = EOD
        exit_price = bar['close']

    pnl_pts = exit_price - entry_price

    return {
        'mfe_pts': mfe,
        'mae_pts': mae,
        'pnl_pts': pnl_pts,
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'bars_held': bars_held,
        'is_winner': 1 if pnl_pts > 0 else 0,
    }


# ============================================================================
#  ENTRY CONFIRMATION MODELS
# ============================================================================

def check_inversion_1m(df, entry_idx):
    """
    1-min inversion candle: prior bar bearish (close < open), current bar bullish
    (close > open) with higher low. Classic reversal signal.
    """
    if entry_idx < 1:
        return False
    prior = df.iloc[entry_idx - 1]
    current = df.iloc[entry_idx]
    # Prior bearish, current bullish
    if prior['close'] >= prior['open']:
        return False
    if current['close'] <= current['open']:
        return False
    # Higher low shows rejection of lower prices
    if current['low'] >= prior['low']:
        return True
    return False


def check_inversion_1m_v2(df, entry_idx):
    """
    Looser 1-min inversion: just need current bar bullish after 1-2 bearish bars.
    Looks back up to 3 bars for bearish context.
    """
    if entry_idx < 1:
        return False
    current = df.iloc[entry_idx]
    if current['close'] <= current['open']:
        return False
    # Check if any of last 3 bars were bearish
    for lookback in range(1, min(4, entry_idx + 1)):
        prior = df.iloc[entry_idx - lookback]
        if prior['close'] < prior['open']:
            return True
    return False


def check_inversion_5m(df_5m, entry_ts):
    """
    5-min inversion candle: current 5-min bar is bullish after prior bearish 5-min.
    """
    if df_5m is None or df_5m.empty:
        return False
    # Find the 5-min bar containing this timestamp
    mask = df_5m['timestamp'] <= entry_ts
    if not mask.any():
        return False
    current_5m_idx = mask.sum() - 1
    if current_5m_idx < 1:
        return False

    current = df_5m.iloc[current_5m_idx]
    prior = df_5m.iloc[current_5m_idx - 1]

    # Prior bearish, current bullish
    if prior['close'] >= prior['open']:
        return False
    if current['close'] <= current['open']:
        return False
    return True


def check_cvd_divergence(df, entry_idx, lookback=10):
    """
    CVD divergence: price making new low in lookback window but CVD is turning up.
    Indicates buying into weakness (smart money accumulating).
    """
    if entry_idx < lookback:
        return False

    window = df.iloc[entry_idx - lookback:entry_idx + 1]

    # Price should be near recent low (within lower 30% of range)
    price_range = window['high'].max() - window['low'].min()
    if price_range <= 0:
        return False
    price_position = (window.iloc[-1]['close'] - window['low'].min()) / price_range
    if price_position > 0.40:
        return False  # Not near lows

    # CVD should be rising (last 3 bars trending up)
    if 'cumulative_delta' not in df.columns:
        return False
    cvd_recent = window['cumulative_delta'].tail(5)
    if len(cvd_recent) < 3:
        return False
    cvd_slope = cvd_recent.iloc[-1] - cvd_recent.iloc[-3]
    return cvd_slope > 0


def check_cvd_divergence_v2(df, entry_idx, lookback=20):
    """
    Stricter CVD divergence: price at new lookback low, CVD higher low.
    """
    if entry_idx < lookback:
        return False

    window = df.iloc[entry_idx - lookback:entry_idx + 1]
    if 'cumulative_delta' not in df.columns:
        return False

    current_price = window.iloc[-1]['close']
    current_cvd = window.iloc[-1]['cumulative_delta']

    # Find the bar with lowest price in first half of window
    first_half = window.iloc[:lookback // 2]
    if first_half.empty:
        return False
    min_price_idx = first_half['low'].idxmin()
    min_price = first_half.loc[min_price_idx, 'low']
    min_price_cvd = first_half.loc[min_price_idx, 'cumulative_delta']

    # Price should be near or below the previous low
    if current_price > min_price + 5:  # Within 5 pts of prior low
        return False

    # CVD should be HIGHER than at the prior price low
    if current_cvd > min_price_cvd:
        return True
    return False


def check_smt_divergence(nq_df, es_df, entry_idx, session_date, entry_ts, lookback=10):
    """
    SMT (Smart Money Technique) divergence:
    NQ makes new low in lookback but ES holds above its prior low.
    Divergence = one instrument failing to confirm = reversal signal.
    """
    if es_df is None:
        return False

    # Get NQ window
    nq_session = nq_df[nq_df['session_date'] == session_date]
    if nq_session.empty or entry_idx < lookback:
        return False

    nq_window = nq_session.iloc[max(0, entry_idx - lookback):entry_idx + 1]
    if len(nq_window) < lookback:
        return False

    # Get matching ES session
    es_session = es_df[es_df['session_date'] == session_date]
    if es_session.empty:
        return False

    # Match ES bars by timestamp
    es_window = es_session[es_session['timestamp'] <= entry_ts].tail(lookback + 1)
    if len(es_window) < lookback:
        return False

    # Check: NQ near recent low but ES NOT making new low
    nq_current_low = nq_window.iloc[-1]['low']
    nq_prior_low = nq_window.iloc[:-3]['low'].min()  # Exclude last 3 bars

    es_current_low = es_window.iloc[-1]['low']
    es_prior_low = es_window.iloc[:-3]['low'].min()

    # NQ making new low (or near it)
    nq_near_low = nq_current_low <= nq_prior_low + 5  # Within 5 pts

    # ES holding above its prior low
    es_holding = es_current_low > es_prior_low + 2  # Must be meaningfully above

    return nq_near_low and es_holding


def check_fvg_confluence(df, entry_idx):
    """Check if entry bar is inside a bullish FVG zone."""
    if 'fvg_bull' not in df.columns:
        return False
    bar = df.iloc[entry_idx]
    return bool(bar.get('fvg_bull', False))


def check_ifvg_confluence(df, entry_idx):
    """Check if entry bar has IFVG pullback confirmation."""
    if 'ifvg_bull_entry' not in df.columns:
        return False
    bar = df.iloc[entry_idx]
    return bool(bar.get('ifvg_bull_entry', False))


def check_reclaim_confirm(df, entry_idx, level, bars_below=2, bars_above=2):
    """
    Reclaim confirmation: price was below a level for bars_below bars,
    then reclaims above it for bars_above consecutive bars.
    """
    if entry_idx < bars_below + bars_above:
        return False

    level_price = level

    # Check if there were bars below the level recently (within last 20 bars)
    found_below = False
    below_end = -1
    for i in range(entry_idx - 1, max(entry_idx - 20, bars_below - 1), -1):
        count_below = 0
        for j in range(i, max(i - bars_below - 2, -1), -1):
            if df.iloc[j]['close'] < level_price:
                count_below += 1
            else:
                break
        if count_below >= bars_below:
            found_below = True
            below_end = i
            break

    if not found_below:
        return False

    # Check that recent bars (including entry) are above the level
    count_above = 0
    for i in range(entry_idx, max(entry_idx - bars_above, below_end), -1):
        if df.iloc[i]['close'] > level_price:
            count_above += 1
        else:
            break

    return count_above >= bars_above


def check_flag_break(df, entry_idx, consolidation_bars=5, range_pct=0.3):
    """
    Flag/consolidation break: price in tight range for N bars, then breakout.
    Consolidation = range < range_pct * ATR.
    """
    if entry_idx < consolidation_bars + 1:
        return False

    # Check if prior N bars were in tight range
    consol_window = df.iloc[entry_idx - consolidation_bars:entry_idx]
    consol_high = consol_window['high'].max()
    consol_low = consol_window['low'].min()
    consol_range = consol_high - consol_low

    # Compare to ATR
    atr = df.iloc[entry_idx].get('atr14', 20.0)
    if pd.isna(atr) or atr <= 0:
        atr = 20.0

    if consol_range > atr * range_pct:
        return False  # Not tight enough

    # Current bar breaks above consolidation high
    current_close = df.iloc[entry_idx]['close']
    return current_close > consol_high


def check_delta_surge(df, entry_idx, min_percentile=80):
    """
    Delta surge: entry bar has exceptionally high delta (buyer aggression).
    """
    bar = df.iloc[entry_idx]
    delta_pctl = bar.get('delta_percentile', 50)
    if pd.isna(delta_pctl):
        return False
    return delta_pctl >= min_percentile


def check_volume_climax(df, entry_idx, min_spike=1.5):
    """
    Volume climax: entry bar has volume spike >= 1.5x (strong participation).
    """
    bar = df.iloc[entry_idx]
    vol_spike = bar.get('volume_spike', 1.0)
    if pd.isna(vol_spike):
        return False
    return vol_spike >= min_spike


def check_strong_close(df, entry_idx, min_pct=0.7):
    """
    Strong close: bar closes in upper % of its range (bullish conviction).
    """
    bar = df.iloc[entry_idx]
    bar_range = bar['high'] - bar['low']
    if bar_range <= 0:
        return False
    close_pct = (bar['close'] - bar['low']) / bar_range
    return close_pct >= min_pct


def check_wick_rejection(df, entry_idx, min_wick_pct=0.5):
    """
    Wick rejection: long lower wick shows buyers stepping in.
    Lower wick >= min_wick_pct of total bar range.
    """
    bar = df.iloc[entry_idx]
    bar_range = bar['high'] - bar['low']
    if bar_range <= 0:
        return False
    lower_wick = min(bar['open'], bar['close']) - bar['low']
    return lower_wick / bar_range >= min_wick_pct


# ============================================================================
#  RUN DIAGNOSTIC
# ============================================================================

def run_diagnostic():
    print("=" * 100)
    print("  ENTRY MODEL OPTIMIZATION DIAGNOSTIC")
    print("  Testing confirmation filters on Edge Fade EDGE_TO_MID entries")
    print("=" * 100)

    # Load NQ data
    print("\nLoading NQ data...")
    nq_df = load_instrument_data('NQ')
    print(f"  NQ: {len(nq_df)} bars, {nq_df['session_date'].nunique()} sessions")

    # Resample to 5-min
    print("  Resampling to 5-min...")
    nq_5m = resample_to_5min(nq_df)
    print(f"  5-min: {len(nq_5m)} bars")

    # Load ES for SMT
    print("Loading ES data for SMT divergence...")
    es_df = load_es_data()
    if es_df is not None:
        print(f"  ES: {len(es_df)} bars, {es_df['session_date'].nunique()} sessions")

    # Find all Edge Fade entries
    print("\nFinding Edge Fade entry opportunities...")
    entries = find_edge_fade_entries(nq_df)
    print(f"  Found {len(entries)} EDGE_TO_MID entries")

    if not entries:
        print("  ERROR: No entries found!")
        return

    # Baseline stats
    winners = [e for e in entries if e['is_winner']]
    losers = [e for e in entries if not e['is_winner']]
    total_pnl = sum(e['pnl_pts'] for e in entries)
    avg_pnl = total_pnl / len(entries)
    wr = len(winners) / len(entries) * 100

    print(f"\n  BASELINE (no confirmation filter):")
    print(f"    Trades: {len(entries)}")
    print(f"    Winners: {len(winners)} ({wr:.1f}%)")
    print(f"    Total PnL: {total_pnl:+.1f} pts")
    print(f"    Avg PnL: {avg_pnl:+.1f} pts/trade")
    print(f"    Avg MFE: {np.mean([e['mfe_pts'] for e in entries]):.1f} pts")
    print(f"    Avg MAE: {np.mean([e['mae_pts'] for e in entries]):.1f} pts")
    print(f"    Exit distribution: "
          f"TARGET={sum(1 for e in entries if e['exit_reason']=='TARGET')}, "
          f"STOP={sum(1 for e in entries if e['exit_reason']=='STOP')}, "
          f"EOD={sum(1 for e in entries if e['exit_reason']=='EOD')}")

    # Define confirmation models to test
    models = {}

    # For each entry, run all confirmation checks
    print("\nRunning confirmation models on each entry...")

    for entry in entries:
        session_date = entry['session_date']
        session_df = nq_df[nq_df['session_date'] == session_date].sort_values('timestamp').reset_index(drop=True)

        # Find the bar index in session
        entry_ts = entry['timestamp']
        bar_mask = session_df['timestamp'] == entry_ts
        if not bar_mask.any():
            # Fuzzy match by closest timestamp
            time_diffs = abs(session_df['timestamp'] - entry_ts)
            idx = time_diffs.idxmin()
        else:
            idx = session_df[bar_mask].index[0]

        # Session 5-min data
        session_5m = nq_5m[nq_5m['session_date'] == session_date] if nq_5m is not None and not nq_5m.empty else None

        # Run all models
        entry['inversion_1m'] = check_inversion_1m(session_df, idx)
        entry['inversion_1m_v2'] = check_inversion_1m_v2(session_df, idx)
        entry['inversion_5m'] = check_inversion_5m(session_5m, entry_ts)
        entry['cvd_divergence'] = check_cvd_divergence(session_df, idx)
        entry['cvd_divergence_v2'] = check_cvd_divergence_v2(session_df, idx)
        entry['smt_divergence'] = check_smt_divergence(nq_df, es_df, idx, session_date, entry_ts)
        entry['fvg_bull'] = check_fvg_confluence(session_df, idx)
        entry['ifvg_bull'] = check_ifvg_confluence(session_df, idx)
        entry['reclaim_ibl'] = check_reclaim_confirm(session_df, idx, entry['ib_low'])
        entry['flag_break'] = check_flag_break(session_df, idx)
        entry['delta_surge'] = check_delta_surge(session_df, idx)
        entry['volume_climax'] = check_volume_climax(session_df, idx)
        entry['strong_close'] = check_strong_close(session_df, idx)
        entry['wick_rejection'] = check_wick_rejection(session_df, idx)

    # ================================================================
    #  RESULTS TABLE
    # ================================================================
    print("\n" + "=" * 100)
    print("  CONFIRMATION MODEL RESULTS")
    print("=" * 100)

    model_names = [
        'inversion_1m', 'inversion_1m_v2', 'inversion_5m',
        'cvd_divergence', 'cvd_divergence_v2',
        'smt_divergence',
        'fvg_bull', 'ifvg_bull',
        'reclaim_ibl',
        'flag_break',
        'delta_surge', 'volume_climax',
        'strong_close', 'wick_rejection',
    ]

    print(f"\n  {'Model':<25s} {'Pass':>5s} {'Win':>5s} {'WR%':>6s} {'AvgPnL':>8s} {'TotPnL':>9s} "
          f"{'AvgMFE':>7s} {'AvgMAE':>7s} {'RejW':>5s} {'RejL':>5s} {'Edge':>7s}")
    print("  " + "-" * 100)

    # Baseline row
    print(f"  {'BASELINE (none)':<25s} {len(entries):>5d} {len(winners):>5d} {wr:>5.1f}% "
          f"{avg_pnl:>+7.1f} {total_pnl:>+8.1f} "
          f"{np.mean([e['mfe_pts'] for e in entries]):>6.1f} "
          f"{np.mean([e['mae_pts'] for e in entries]):>6.1f} "
          f"{'--':>5s} {'--':>5s} {'--':>7s}")

    best_model = None
    best_edge = -999

    for model_name in model_names:
        passing = [e for e in entries if e.get(model_name, False)]
        rejected = [e for e in entries if not e.get(model_name, False)]

        if not passing:
            print(f"  {model_name:<25s} {'0':>5s} {'--':>5s} {'--':>6s} {'--':>8s} {'--':>9s} "
                  f"{'--':>7s} {'--':>7s} {'--':>5s} {'--':>5s} {'--':>7s}")
            continue

        pass_winners = [e for e in passing if e['is_winner']]
        pass_wr = len(pass_winners) / len(passing) * 100
        pass_pnl = sum(e['pnl_pts'] for e in passing)
        pass_avg = pass_pnl / len(passing)
        pass_mfe = np.mean([e['mfe_pts'] for e in passing])
        pass_mae = np.mean([e['mae_pts'] for e in passing])

        # Rejected analysis
        rej_winners = len([e for e in rejected if e['is_winner']])
        rej_losers = len([e for e in rejected if not e['is_winner']])

        # Edge = improvement in avg PnL vs baseline
        edge = pass_avg - avg_pnl

        print(f"  {model_name:<25s} {len(passing):>5d} {len(pass_winners):>5d} {pass_wr:>5.1f}% "
              f"{pass_avg:>+7.1f} {pass_pnl:>+8.1f} "
              f"{pass_mfe:>6.1f} {pass_mae:>6.1f} "
              f"{rej_winners:>5d} {rej_losers:>5d} "
              f"{edge:>+6.1f}")

        if edge > best_edge and len(passing) >= 10:
            best_edge = edge
            best_model = model_name

    # ================================================================
    #  COMBO MODELS
    # ================================================================
    print(f"\n\n{'='*100}")
    print("  COMBINATION MODELS (AND logic)")
    print(f"{'='*100}")

    combos = [
        ('inversion_1m + delta_surge', ['inversion_1m', 'delta_surge']),
        ('inversion_1m + volume_climax', ['inversion_1m', 'volume_climax']),
        ('inversion_1m_v2 + strong_close', ['inversion_1m_v2', 'strong_close']),
        ('cvd_div + inversion_1m', ['cvd_divergence', 'inversion_1m']),
        ('cvd_div + strong_close', ['cvd_divergence', 'strong_close']),
        ('cvd_div + volume_climax', ['cvd_divergence', 'volume_climax']),
        ('smt + inversion_1m', ['smt_divergence', 'inversion_1m']),
        ('smt + strong_close', ['smt_divergence', 'strong_close']),
        ('strong_close + volume_climax', ['strong_close', 'volume_climax']),
        ('strong_close + delta_surge', ['strong_close', 'delta_surge']),
        ('wick_rej + inversion_1m', ['wick_rejection', 'inversion_1m']),
        ('wick_rej + strong_close', ['wick_rejection', 'strong_close']),
        ('fvg + inversion_1m', ['fvg_bull', 'inversion_1m']),
        ('reclaim + strong_close', ['reclaim_ibl', 'strong_close']),
        ('inversion_1m_v2 + delta_surge', ['inversion_1m_v2', 'delta_surge']),
        ('cvd_div + delta_surge', ['cvd_divergence', 'delta_surge']),
    ]

    print(f"\n  {'Combo':<35s} {'Pass':>5s} {'Win':>5s} {'WR%':>6s} {'AvgPnL':>8s} {'TotPnL':>9s} "
          f"{'RejW':>5s} {'RejL':>5s} {'Edge':>7s}")
    print("  " + "-" * 95)

    for combo_name, combo_filters in combos:
        passing = [e for e in entries if all(e.get(f, False) for f in combo_filters)]
        rejected = [e for e in entries if not all(e.get(f, False) for f in combo_filters)]

        if not passing:
            print(f"  {combo_name:<35s} {'0':>5s}")
            continue

        pass_winners = [e for e in passing if e['is_winner']]
        pass_wr = len(pass_winners) / len(passing) * 100
        pass_pnl = sum(e['pnl_pts'] for e in passing)
        pass_avg = pass_pnl / len(passing)

        rej_winners = len([e for e in rejected if e['is_winner']])
        rej_losers = len([e for e in rejected if not e['is_winner']])
        edge = pass_avg - avg_pnl

        print(f"  {combo_name:<35s} {len(passing):>5d} {len(pass_winners):>5d} {pass_wr:>5.1f}% "
              f"{pass_avg:>+7.1f} {pass_pnl:>+8.1f} "
              f"{rej_winners:>5d} {rej_losers:>5d} "
              f"{edge:>+6.1f}")

    # ================================================================
    #  OR-COMBINATION MODELS (any of the filters pass)
    # ================================================================
    print(f"\n\n{'='*100}")
    print("  OR-COMBINATION MODELS (any filter passes)")
    print(f"{'='*100}")

    or_combos = [
        ('inversion_1m OR cvd_div', ['inversion_1m', 'cvd_divergence']),
        ('inversion_1m OR strong_close', ['inversion_1m', 'strong_close']),
        ('inversion_1m OR smt', ['inversion_1m', 'smt_divergence']),
        ('cvd_div OR smt', ['cvd_divergence', 'smt_divergence']),
        ('inv_1m OR cvd OR smt', ['inversion_1m', 'cvd_divergence', 'smt_divergence']),
        ('inv_1m OR delta_surge', ['inversion_1m', 'delta_surge']),
        ('strong_close OR wick_rej', ['strong_close', 'wick_rejection']),
    ]

    print(f"\n  {'Combo (OR)':<35s} {'Pass':>5s} {'Win':>5s} {'WR%':>6s} {'AvgPnL':>8s} {'TotPnL':>9s} "
          f"{'RejW':>5s} {'RejL':>5s} {'Edge':>7s}")
    print("  " + "-" * 95)

    for combo_name, combo_filters in or_combos:
        passing = [e for e in entries if any(e.get(f, False) for f in combo_filters)]
        rejected = [e for e in entries if not any(e.get(f, False) for f in combo_filters)]

        if not passing:
            print(f"  {combo_name:<35s} {'0':>5s}")
            continue

        pass_winners = [e for e in passing if e['is_winner']]
        pass_wr = len(pass_winners) / len(passing) * 100
        pass_pnl = sum(e['pnl_pts'] for e in passing)
        pass_avg = pass_pnl / len(passing)

        rej_winners = len([e for e in rejected if e['is_winner']])
        rej_losers = len([e for e in rejected if not e['is_winner']])
        edge = pass_avg - avg_pnl

        print(f"  {combo_name:<35s} {len(passing):>5d} {len(pass_winners):>5d} {pass_wr:>5.1f}% "
              f"{pass_avg:>+7.1f} {pass_pnl:>+8.1f} "
              f"{rej_winners:>5d} {rej_losers:>5d} "
              f"{edge:>+6.1f}")

    # ================================================================
    #  WINNER vs LOSER FEATURE COMPARISON
    # ================================================================
    print(f"\n\n{'='*100}")
    print("  FEATURE PRESENCE: WINNERS vs LOSERS")
    print(f"{'='*100}")

    print(f"\n  {'Feature':<25s} {'Win%':>7s} {'Loss%':>7s} {'Diff':>7s} {'Signal':>8s}")
    print("  " + "-" * 60)

    for model_name in model_names:
        win_has = sum(1 for e in winners if e.get(model_name, False))
        loss_has = sum(1 for e in losers if e.get(model_name, False))
        win_pct = win_has / len(winners) * 100 if winners else 0
        loss_pct = loss_has / len(losers) * 100 if losers else 0
        diff = win_pct - loss_pct
        signal = 'GOOD' if diff > 10 else 'WEAK' if diff > 0 else 'BAD'

        print(f"  {model_name:<25s} {win_pct:>6.1f}% {loss_pct:>6.1f}% {diff:>+6.1f}% {signal:>8s}")

    # ================================================================
    #  BEST MODEL RECOMMENDATION
    # ================================================================
    print(f"\n\n{'='*100}")
    print("  RECOMMENDATION")
    print(f"{'='*100}")

    if best_model:
        print(f"\n  Best single confirmation model: {best_model}")
        print(f"  Edge improvement: {best_edge:+.1f} pts/trade vs baseline")
    else:
        print("\n  No single model with 10+ trades improved over baseline")

    print(f"\n  NOTE: Results are in NQ POINTS per 1 contract.")
    print(f"  At $2/pt (MNQ), multiply by $2 for dollar P&L.")
    print(f"  Baseline {len(entries)} entries -> compare with trade_log_v14.csv ({len(entries)} expected)")


if __name__ == '__main__':
    run_diagnostic()
