"""
Opening Range + SMT Divergence Strategy Study
==============================================

Research question: Can we trade the 9:30 market open using pre-market data
(overnight high/low, Asia session high/low) combined with SMT divergence
between NQ, ES, and YM?

Strategy concept (from ICT methodology):
1. Pre-market: Identify overnight high/low, Asia session high/low (20:00-00:00 ET)
2. Opening range (9:30-10:00): Watch for liquidity sweep of overnight/Asia levels
3. SMT divergence: NQ vs ES — one makes new high/low while other holds
4. Entry: After sweep + SMT + FVG formation on 5-min bars, enter in direction of reversal
5. Confirmation: VWAP alignment, delta confirmation
6. Target: 2R or 3R from entry, or IB midpoint

Studies:
  A. Pre-market level identification (overnight H/L, Asia H/L, prior day H/L)
  B. Sweep frequency — how often does 9:30-10:00 sweep these levels?
  C. SMT divergence detection — NQ vs ES correlation breaks at key moments
  D. FVG formation after sweeps (5-min and 15-min bars)
  E. Combined strategy simulation with P&L
  F. Comparison to existing playbook strategies
"""

import pandas as pd
import numpy as np
from datetime import time as dt_time, timedelta
from pathlib import Path
import sys

# Add project root to path
proj_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(proj_root))

from data.loader import load_csv
from data.session import filter_rth


def load_all_instruments():
    """Load NQ, ES, YM data."""
    print('=' * 70)
    print('  LOADING DATA')
    print('=' * 70)
    nq = load_csv('NQ')
    es = load_csv('ES')
    ym = load_csv('YM')
    return nq, es, ym


def get_session_dates(df):
    """Get unique RTH session dates."""
    rth = filter_rth(df)
    return sorted(rth['session_date'].dt.date.unique())


def compute_session_levels(df, session_date):
    """
    For a given RTH session date, compute pre-market reference levels.

    Returns dict with:
      - overnight_high/low: ETH session high/low (18:00 prev day to 9:29)
      - asia_high/low: Asian session (20:00-00:00 ET prev day)
      - london_high/low: London session (02:00-05:00 ET)
      - premarket_high/low: US pre-market (06:00-09:29 ET)
      - prior_day_high/low/close: Previous RTH session
      - opening_range_high/low: First 30 min (9:30-10:00)
      - ib_high/low/mid/range: Initial Balance (9:30-10:30)
    """
    sd = pd.Timestamp(session_date)

    # Previous calendar day for overnight
    prev_day = sd - timedelta(days=1)
    # Handle weekends: if prev_day is Saturday (5), go to Friday (4)
    if prev_day.weekday() == 5:
        prev_day = prev_day - timedelta(days=1)
    elif prev_day.weekday() == 6:
        prev_day = prev_day - timedelta(days=2)

    ts = df['timestamp']

    # Asian session: 20:00-00:00 ET (previous day evening)
    asia_mask = (
        (ts >= pd.Timestamp(prev_day) + timedelta(hours=20)) &
        (ts < pd.Timestamp(sd) + timedelta(hours=0))
    )
    asia = df[asia_mask]

    # London session: 02:00-05:00 ET
    london_mask = (
        (ts >= pd.Timestamp(sd) + timedelta(hours=2)) &
        (ts < pd.Timestamp(sd) + timedelta(hours=5))
    )
    london = df[london_mask]

    # US pre-market: 06:00-09:29 ET
    premarket_mask = (
        (ts >= pd.Timestamp(sd) + timedelta(hours=6)) &
        (ts < pd.Timestamp(sd) + timedelta(hours=9, minutes=30))
    )
    premarket = df[premarket_mask]

    # Full overnight: 18:00 prev day to 09:29
    overnight_mask = (
        (ts >= pd.Timestamp(prev_day) + timedelta(hours=18)) &
        (ts < pd.Timestamp(sd) + timedelta(hours=9, minutes=30))
    )
    overnight = df[overnight_mask]

    # Opening range: 9:30-10:00
    or_mask = (
        (ts >= pd.Timestamp(sd) + timedelta(hours=9, minutes=30)) &
        (ts < pd.Timestamp(sd) + timedelta(hours=10))
    )
    opening_range = df[or_mask]

    # IB: 9:30-10:30
    ib_mask = (
        (ts >= pd.Timestamp(sd) + timedelta(hours=9, minutes=30)) &
        (ts <= pd.Timestamp(sd) + timedelta(hours=10, minutes=30))
    )
    ib = df[ib_mask]

    # Prior day RTH
    prior_rth_mask = (
        (ts >= pd.Timestamp(prev_day) + timedelta(hours=9, minutes=30)) &
        (ts <= pd.Timestamp(prev_day) + timedelta(hours=16))
    )
    prior_rth = df[prior_rth_mask]

    # RTH for this session
    rth_mask = (
        (ts >= pd.Timestamp(sd) + timedelta(hours=9, minutes=30)) &
        (ts <= pd.Timestamp(sd) + timedelta(hours=16))
    )
    rth = df[rth_mask]

    levels = {
        'session_date': session_date,
    }

    # Overnight levels
    if len(overnight) > 0:
        levels['overnight_high'] = overnight['high'].max()
        levels['overnight_low'] = overnight['low'].min()
    else:
        levels['overnight_high'] = np.nan
        levels['overnight_low'] = np.nan

    # Asia levels
    if len(asia) > 0:
        levels['asia_high'] = asia['high'].max()
        levels['asia_low'] = asia['low'].min()
    else:
        levels['asia_high'] = np.nan
        levels['asia_low'] = np.nan

    # London levels
    if len(london) > 0:
        levels['london_high'] = london['high'].max()
        levels['london_low'] = london['low'].min()
    else:
        levels['london_high'] = np.nan
        levels['london_low'] = np.nan

    # Pre-market levels
    if len(premarket) > 0:
        levels['premarket_high'] = premarket['high'].max()
        levels['premarket_low'] = premarket['low'].min()
    else:
        levels['premarket_high'] = np.nan
        levels['premarket_low'] = np.nan

    # Prior day
    if len(prior_rth) > 0:
        levels['pdh'] = prior_rth['high'].max()
        levels['pdl'] = prior_rth['low'].min()
        levels['pdc'] = prior_rth.iloc[-1]['close']
    else:
        levels['pdh'] = np.nan
        levels['pdl'] = np.nan
        levels['pdc'] = np.nan

    # Opening range
    if len(opening_range) > 0:
        levels['or_high'] = opening_range['high'].max()
        levels['or_low'] = opening_range['low'].min()
        levels['or_range'] = levels['or_high'] - levels['or_low']
    else:
        levels['or_high'] = np.nan
        levels['or_low'] = np.nan
        levels['or_range'] = np.nan

    # IB
    if len(ib) > 0:
        levels['ib_high'] = ib['high'].max()
        levels['ib_low'] = ib['low'].min()
        levels['ib_range'] = levels['ib_high'] - levels['ib_low']
        levels['ib_mid'] = (levels['ib_high'] + levels['ib_low']) / 2
    else:
        levels['ib_high'] = np.nan
        levels['ib_low'] = np.nan
        levels['ib_range'] = np.nan
        levels['ib_mid'] = np.nan

    # VWAP at 9:30
    if len(opening_range) > 0:
        levels['vwap_open'] = opening_range.iloc[0]['vwap']
    else:
        levels['vwap_open'] = np.nan

    # RTH high/low (for MFE/MAE)
    if len(rth) > 0:
        levels['rth_high'] = rth['high'].max()
        levels['rth_low'] = rth['low'].min()
        levels['rth_close'] = rth.iloc[-1]['close']
    else:
        levels['rth_high'] = np.nan
        levels['rth_low'] = np.nan
        levels['rth_close'] = np.nan

    return levels


def detect_sweep(bar_high, bar_low, level, direction, threshold_pts=2):
    """
    Detect if a bar sweeps a level.
    direction='above': bar_high > level (sweeps above)
    direction='below': bar_low < level (sweeps below)
    threshold: must exceed by at least this many points
    """
    if direction == 'above':
        return bar_high >= level + threshold_pts
    else:
        return bar_low <= level - threshold_pts


def detect_smt_divergence(nq_bars, es_bars, lookback=30):
    """
    Detect SMT divergence between NQ and ES during opening range.

    Bullish SMT: NQ makes lower low but ES holds higher low (or vice versa)
    Bearish SMT: NQ makes higher high but ES holds lower high (or vice versa)

    Args:
        nq_bars: NQ bars for the period (9:30-10:30)
        es_bars: ES bars for the same period
        lookback: bars to look back for swing comparison

    Returns:
        list of (bar_index, smt_type, details) tuples
    """
    divergences = []

    if len(nq_bars) < 5 or len(es_bars) < 5:
        return divergences

    # Align on timestamp
    nq = nq_bars.set_index('timestamp').sort_index()
    es = es_bars.set_index('timestamp').sort_index()

    # Find common timestamps
    common_ts = nq.index.intersection(es.index)
    if len(common_ts) < 5:
        return divergences

    nq = nq.loc[common_ts]
    es = es.loc[common_ts]

    # Track running high/low for each
    nq_running_high = nq['high'].expanding().max()
    nq_running_low = nq['low'].expanding().min()
    es_running_high = es['high'].expanding().max()
    es_running_low = es['low'].expanding().min()

    # Look for divergence: compare new lows/highs
    for i in range(5, len(common_ts)):
        ts = common_ts[i]

        # Check for new low in NQ
        nq_new_low = nq['low'].iloc[i] <= nq_running_low.iloc[i-1]
        es_new_low = es['low'].iloc[i] <= es_running_low.iloc[i-1]

        # Bullish SMT: NQ makes new low but ES doesn't (or vice versa)
        if nq_new_low and not es_new_low:
            divergences.append((ts, 'BULLISH_SMT', 'NQ new low, ES holding'))
        elif es_new_low and not nq_new_low:
            divergences.append((ts, 'BULLISH_SMT', 'ES new low, NQ holding'))

        # Check for new high
        nq_new_high = nq['high'].iloc[i] >= nq_running_high.iloc[i-1]
        es_new_high = es['high'].iloc[i] >= es_running_high.iloc[i-1]

        # Bearish SMT: NQ makes new high but ES doesn't (or vice versa)
        if nq_new_high and not es_new_high:
            divergences.append((ts, 'BEARISH_SMT', 'NQ new high, ES holding'))
        elif es_new_high and not nq_new_high:
            divergences.append((ts, 'BEARISH_SMT', 'ES new high, NQ holding'))

    return divergences


def detect_fvg_5min(bars_1min, direction='bull'):
    """
    Detect FVGs on synthetic 5-min bars.

    Bull FVG: bar3.low > bar1.high (gap up)
    Bear FVG: bar3.high < bar1.low (gap down)

    Returns list of (timestamp, fvg_top, fvg_bottom) tuples.
    """
    fvgs = []
    if len(bars_1min) < 15:
        return fvgs

    # Resample to 5-min bars
    bars = bars_1min.set_index('timestamp').resample('5min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        'volume': 'sum', 'vol_delta': 'sum',
    }).dropna()

    for i in range(len(bars) - 2):
        b1_high = bars['high'].iloc[i]
        b1_low = bars['low'].iloc[i]
        b3_high = bars['high'].iloc[i + 2]
        b3_low = bars['low'].iloc[i + 2]

        if direction == 'bull' and b3_low > b1_high + 1.0:
            fvgs.append((bars.index[i + 1], b3_low, b1_high))  # top, bottom
        elif direction == 'bear' and b3_high < b1_low - 1.0:
            fvgs.append((bars.index[i + 1], b1_low, b3_high))  # top, bottom

    return fvgs


def detect_fvg_15min(bars_1min, direction='bull'):
    """Detect FVGs on synthetic 15-min bars."""
    fvgs = []
    if len(bars_1min) < 45:
        return fvgs

    bars = bars_1min.set_index('timestamp').resample('15min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        'volume': 'sum', 'vol_delta': 'sum',
    }).dropna()

    for i in range(len(bars) - 2):
        b1_high = bars['high'].iloc[i]
        b3_low = bars['low'].iloc[i + 2]
        b1_low = bars['low'].iloc[i]
        b3_high = bars['high'].iloc[i + 2]

        if direction == 'bull' and b3_low > b1_high + 1.0:
            fvgs.append((bars.index[i + 1], b3_low, b1_high))
        elif direction == 'bear' and b3_high < b1_low - 1.0:
            fvgs.append((bars.index[i + 1], b1_low, b3_high))

    return fvgs


def detect_opening_reversal(bars, levels_dict):
    """
    Detect the ICT "Judas Swing" / opening range reversal pattern.

    Concept: In the first 15-30 min of RTH, price makes a false move in one
    direction (sweeping liquidity), then reverses for the real move of the day.

    Models detected:
    1. SWEEP_AND_REVERSE: Price sweeps a pre-market level, then reverses direction.
       - Within first 30 bars (9:30-10:00), price makes a high/low near a key level
       - Then within next 15 bars, price closes beyond the opening range in opposite direction
       - Entry on the reversal bar close

    2. OPENING_RANGE_FADE: Opening range (first 15 min) high/low is taken,
       then price reverses through the opposite side.
       - OR forms 9:30-9:45
       - Price breaks OR low/high, then reverses through OR high/low
       - Entry when opposite OR level is breached

    3. VWAP_RECLAIM: After sweeping a level, price reclaims VWAP in the
       reversal direction.

    Returns list of setup dicts.
    """
    setups = []
    if len(bars) < 30:
        return setups

    # Find bars that are RTH (9:30+)
    rth_start = None
    for i, bar in bars.iterrows():
        if bar['timestamp'].hour == 9 and bar['timestamp'].minute >= 30:
            rth_start = i
            break
        elif bar['timestamp'].hour >= 10:
            rth_start = i
            break
    if rth_start is None:
        return setups

    rth_bars = bars.loc[rth_start:].copy().reset_index(drop=True)
    if len(rth_bars) < 30:
        return setups

    # Opening range: first 15 bars (9:30-9:44)
    or_bars = rth_bars.iloc[:15]
    or_high = or_bars['high'].max()
    or_low = or_bars['low'].min()
    or_mid = (or_high + or_low) / 2

    # Extended opening range: first 30 bars (9:30-9:59)
    eor_bars = rth_bars.iloc[:30]
    eor_high = eor_bars['high'].max()
    eor_low = eor_bars['low'].min()

    open_price = rth_bars.iloc[0]['open']

    # Pre-market levels
    asia_h = levels_dict.get('asia_high', np.nan)
    asia_l = levels_dict.get('asia_low', np.nan)
    on_h = levels_dict.get('overnight_high', np.nan)
    on_l = levels_dict.get('overnight_low', np.nan)
    pdh = levels_dict.get('pdh', np.nan)
    pdl = levels_dict.get('pdl', np.nan)

    # === MODEL 1: Opening Range Reversal ===
    # First 30 min makes a high (Judas swing up), then reverses down
    # OR first 30 min makes a low (Judas swing down), then reverses up

    # Find the extreme bar in first 30 min
    high_bar_idx = eor_bars['high'].idxmax()
    low_bar_idx = eor_bars['low'].idxmin()
    high_bar_time = rth_bars.loc[high_bar_idx, 'timestamp'] if high_bar_idx in rth_bars.index else None
    low_bar_time = rth_bars.loc[low_bar_idx, 'timestamp'] if low_bar_idx in rth_bars.index else None

    # Check if the first move is UP (Judas swing up -> SHORT setup)
    # The high in first 30 min should be near a key level
    high_near_level = False
    high_level_name = ''
    for lname, lval in [('asia_high', asia_h), ('overnight_high', on_h), ('pdh', pdh)]:
        if not np.isnan(lval) and abs(eor_high - lval) < 30:  # Within 30 pts
            high_near_level = True
            high_level_name = lname
            break

    # Check reversal: after the high, price drops below OR mid
    if high_near_level and high_bar_idx is not None:
        post_high = rth_bars.loc[high_bar_idx:]
        for j in range(1, min(30, len(post_high))):
            bar = post_high.iloc[j]
            if bar['close'] < or_mid:
                # Reversal confirmed! SHORT setup
                setups.append({
                    'model': 'OR_REVERSAL_SHORT',
                    'entry_idx': post_high.index[j],
                    'entry_time': bar['timestamp'],
                    'entry_price': bar['close'],
                    'sweep_extreme': eor_high,
                    'sweep_level': high_level_name,
                    'or_high': or_high,
                    'or_low': or_low,
                    'or_mid': or_mid,
                    'direction': 'SHORT',
                    'stop': eor_high + 0.15 * (eor_high - eor_low),
                })
                break

    # Check if the first move is DOWN (Judas swing down -> LONG setup)
    low_near_level = False
    low_level_name = ''
    for lname, lval in [('asia_low', asia_l), ('overnight_low', on_l), ('pdl', pdl)]:
        if not np.isnan(lval) and abs(eor_low - lval) < 30:
            low_near_level = True
            low_level_name = lname
            break

    if low_near_level and low_bar_idx is not None:
        post_low = rth_bars.loc[low_bar_idx:]
        for j in range(1, min(30, len(post_low))):
            bar = post_low.iloc[j]
            if bar['close'] > or_mid:
                # Reversal confirmed! LONG setup
                setups.append({
                    'model': 'OR_REVERSAL_LONG',
                    'entry_idx': post_low.index[j],
                    'entry_time': bar['timestamp'],
                    'entry_price': bar['close'],
                    'sweep_extreme': eor_low,
                    'sweep_level': low_level_name,
                    'or_high': or_high,
                    'or_low': or_low,
                    'or_mid': or_mid,
                    'direction': 'LONG',
                    'stop': eor_low - 0.15 * (eor_high - eor_low),
                })
                break

    # === MODEL 2: VWAP Reclaim After Sweep ===
    # Price opens below VWAP, sweeps lower, then reclaims VWAP -> LONG
    # Price opens above VWAP, sweeps higher, then loses VWAP -> SHORT
    if 'vwap' in rth_bars.columns:
        vwap_open = rth_bars.iloc[0]['vwap']

        # Check for sweep low + VWAP reclaim (LONG)
        if eor_low < vwap_open - 10:  # Price went below VWAP
            for j in range(15, min(60, len(rth_bars))):
                bar = rth_bars.iloc[j]
                if bar['close'] > vwap_open and bar['vol_delta'] > 0:
                    setups.append({
                        'model': 'VWAP_RECLAIM_LONG',
                        'entry_idx': j,
                        'entry_time': bar['timestamp'],
                        'entry_price': bar['close'],
                        'sweep_extreme': eor_low,
                        'sweep_level': 'below_vwap',
                        'or_high': or_high,
                        'or_low': or_low,
                        'or_mid': or_mid,
                        'direction': 'LONG',
                        'stop': eor_low - 0.15 * (eor_high - eor_low),
                    })
                    break

        # Check for sweep high + VWAP loss (SHORT)
        if eor_high > vwap_open + 10:
            for j in range(15, min(60, len(rth_bars))):
                bar = rth_bars.iloc[j]
                if bar['close'] < vwap_open and bar['vol_delta'] < 0:
                    setups.append({
                        'model': 'VWAP_RECLAIM_SHORT',
                        'entry_idx': j,
                        'entry_time': bar['timestamp'],
                        'entry_price': bar['close'],
                        'sweep_extreme': eor_high,
                        'sweep_level': 'above_vwap',
                        'or_high': or_high,
                        'or_low': or_low,
                        'or_mid': or_mid,
                        'direction': 'SHORT',
                        'stop': eor_high + 0.15 * (eor_high - eor_low),
                    })
                    break

    return setups


def simulate_opening_range_strategy(nq, es, sessions):
    """
    Full simulation of opening range + sweep + SMT + FVG strategy.

    Entry logic:
    1. Compute pre-market levels (overnight H/L, Asia H/L, PDH/PDL)
    2. During 9:30-10:30, detect TRUE sweep (approach + pierce + reclaim) of key level
    3. Check for SMT divergence between NQ and ES at/near the sweep
    4. After sweep + reclaim, look for FVG on 5-min bars in reversal direction
    5. Enter on reclaim bar close with VWAP/delta confirmation
    6. Stop: beyond sweep extreme + buffer
    7. Targets: 2R and 3R

    Returns DataFrame of all trades.
    """
    MNQ_MULT = 2.0  # $2/point for MNQ
    trades = []

    for sd in sessions:
        sd_ts = pd.Timestamp(sd)

        # Get NQ levels
        levels = compute_session_levels(nq, sd)
        if np.isnan(levels.get('overnight_high', np.nan)):
            continue

        # Get bars: include 30 min pre-market (9:00-9:29) for approach context
        # plus opening range through 11:00 for sweep detection
        entry_window_mask = (
            (nq['timestamp'] >= sd_ts + timedelta(hours=9, minutes=0)) &
            (nq['timestamp'] <= sd_ts + timedelta(hours=11, minutes=0))
        )
        nq_entry = nq[entry_window_mask].copy().reset_index(drop=True)

        es_entry_mask = (
            (es['timestamp'] >= sd_ts + timedelta(hours=9, minutes=0)) &
            (es['timestamp'] <= sd_ts + timedelta(hours=11, minutes=0))
        )
        es_entry = es[es_entry_mask].copy().reset_index(drop=True)

        if len(nq_entry) < 15 or len(es_entry) < 15:
            continue

        # Full RTH bars for P&L tracking
        rth_mask = (
            (nq['timestamp'] >= sd_ts + timedelta(hours=9, minutes=30)) &
            (nq['timestamp'] <= sd_ts + timedelta(hours=16))
        )
        nq_rth = nq[rth_mask].copy().reset_index(drop=True)

        # Detect opening range setups
        setups = detect_opening_reversal(nq_entry, levels)

        if not setups:
            continue

        # Check for SMT divergence
        smt_divs = detect_smt_divergence(nq_entry, es_entry)

        # Process each setup (take first per model type)
        seen_models = set()
        for setup in setups:
            model = setup['model']
            if model in seen_models:
                continue
            seen_models.add(model)

            entry_price = setup['entry_price']
            entry_time = setup['entry_time']
            stop_price = setup['stop']

            # Look for FVG
            entry_idx_val = setup['entry_idx']
            post_entry_mask = nq_entry.index >= entry_idx_val
            post_entry_bars = nq_entry[post_entry_mask]

            fvg_dir = 'bull' if setup['direction'] == 'LONG' else 'bear'
            fvgs_5m = detect_fvg_5min(post_entry_bars, direction=fvg_dir)
            fvgs_15m = detect_fvg_15min(post_entry_bars, direction=fvg_dir)
            has_fvg_5m = len(fvgs_5m) > 0
            has_fvg_15m = len(fvgs_15m) > 0

            # SMT check
            matching_smt = None
            for smt_ts, smt_type, smt_detail in smt_divs:
                time_diff = abs((smt_ts - entry_time).total_seconds())
                if time_diff <= 900:  # Within 15 minutes
                    if setup['direction'] == 'LONG' and smt_type == 'BULLISH_SMT':
                        matching_smt = (smt_ts, smt_type, smt_detail)
                        break
                    elif setup['direction'] == 'SHORT' and smt_type == 'BEARISH_SMT':
                        matching_smt = (smt_ts, smt_type, smt_detail)
                        break
            has_smt = matching_smt is not None

            # VWAP alignment
            vwap_aligned = False
            if 'vwap' in nq_entry.columns and entry_idx_val < len(nq_entry):
                vwap_val = nq_entry.iloc[entry_idx_val]['vwap'] if entry_idx_val < len(nq_entry) else np.nan
                if not np.isnan(vwap_val):
                    if setup['direction'] == 'LONG' and entry_price <= vwap_val + 20:
                        vwap_aligned = True
                    elif setup['direction'] == 'SHORT' and entry_price >= vwap_val - 20:
                        vwap_aligned = True

            # Delta confirmation
            delta_confirm = False
            if entry_idx_val < len(nq_entry):
                delta_val = nq_entry.iloc[entry_idx_val]['vol_delta']
                if setup['direction'] == 'LONG' and delta_val > 0:
                    delta_confirm = True
                elif setup['direction'] == 'SHORT' and delta_val < 0:
                    delta_confirm = True

            has_fvg = has_fvg_5m or has_fvg_15m

            # Calculate risk/targets
            if setup['direction'] == 'LONG':
                risk = entry_price - stop_price
                target_2r = entry_price + 2 * risk
                target_3r = entry_price + 3 * risk
            else:
                risk = stop_price - entry_price
                target_2r = entry_price - 2 * risk
                target_3r = entry_price - 3 * risk

            if risk <= 0 or risk > 200:
                continue

            # Simulate P&L: walk forward through RTH bars
            post_entry = nq_rth[nq_rth['timestamp'] > entry_time]
            exit_price = None
            exit_reason = None
            exit_time = None
            mfe = 0
            mae = 0

            for _, bar in post_entry.iterrows():
                if setup['direction'] == 'LONG':
                    unrealized_high = bar['high'] - entry_price
                    unrealized_low = bar['low'] - entry_price
                    mfe = max(mfe, unrealized_high)
                    mae = min(mae, unrealized_low)

                    # Stop hit
                    if bar['low'] <= stop_price:
                        exit_price = stop_price
                        exit_reason = 'STOP'
                        exit_time = bar['timestamp']
                        break
                    # Target 2R hit
                    if bar['high'] >= target_2r:
                        exit_price = target_2r
                        exit_reason = 'TARGET_2R'
                        exit_time = bar['timestamp']
                        break
                else:
                    unrealized_high = entry_price - bar['low']
                    unrealized_low = entry_price - bar['high']
                    mfe = max(mfe, unrealized_high)
                    mae = min(mae, unrealized_low)

                    # Stop hit
                    if bar['high'] >= stop_price:
                        exit_price = stop_price
                        exit_reason = 'STOP'
                        exit_time = bar['timestamp']
                        break
                    # Target 2R hit
                    if bar['low'] <= target_2r:
                        exit_price = target_2r
                        exit_reason = 'TARGET_2R'
                        exit_time = bar['timestamp']
                        break

            # EOD exit if no stop/target hit
            if exit_price is None and len(post_entry) > 0:
                last_bar = post_entry.iloc[-1]
                exit_price = last_bar['close']
                exit_reason = 'EOD'
                exit_time = last_bar['timestamp']

            if exit_price is None:
                continue

            # Calculate P&L
            if setup['direction'] == 'LONG':
                pnl_pts = exit_price - entry_price
            else:
                pnl_pts = entry_price - exit_price

            pnl_dollar = pnl_pts * MNQ_MULT
            r_multiple = pnl_pts / risk if risk > 0 else 0

            or_range = setup.get('or_high', 0) - setup.get('or_low', 0)
            trade = {
                'session_date': sd,
                'entry_time': entry_time,
                'exit_time': exit_time,
                'model': model,
                'direction': setup['direction'],
                'level_swept': setup.get('sweep_level', ''),
                'entry_price': round(entry_price, 2),
                'stop_price': round(stop_price, 2),
                'exit_price': round(exit_price, 2),
                'risk_pts': round(risk, 1),
                'pnl_pts': round(pnl_pts, 1),
                'pnl_dollar': round(pnl_dollar, 2),
                'r_multiple': round(r_multiple, 2),
                'exit_reason': exit_reason,
                'mfe_pts': round(mfe, 1),
                'mae_pts': round(mae, 1),
                'has_smt': has_smt,
                'smt_detail': matching_smt[2] if matching_smt else '',
                'has_fvg_5m': has_fvg_5m,
                'has_fvg_15m': has_fvg_15m,
                'vwap_aligned': vwap_aligned,
                'delta_confirm': delta_confirm,
                'or_range': round(or_range, 1),
                'ib_range': round(levels.get('ib_range', 0), 1),
                'overnight_range': round(levels['overnight_high'] - levels['overnight_low'], 1),
            }
            trades.append(trade)

    return pd.DataFrame(trades)


def calc_metrics(trades_df, label=''):
    """Calculate standard performance metrics."""
    if len(trades_df) == 0:
        return None

    n = len(trades_df)
    wins = trades_df[trades_df['pnl_dollar'] > 0]
    losses = trades_df[trades_df['pnl_dollar'] <= 0]
    wr = len(wins) / n * 100
    avg_win = wins['pnl_dollar'].mean() if len(wins) > 0 else 0
    avg_loss = losses['pnl_dollar'].mean() if len(losses) > 0 else 0
    expectancy = (len(wins)/n * avg_win) + (len(losses)/n * avg_loss)
    net = trades_df['pnl_dollar'].sum()
    gross_win = wins['pnl_dollar'].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses['pnl_dollar'].sum()) if len(losses) > 0 else 0.01
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

    # Max DD
    cum = trades_df['pnl_dollar'].cumsum()
    runmax = cum.cummax()
    dd = cum - runmax
    max_dd = dd.min()

    # Max consecutive losses
    is_loss = (trades_df['pnl_dollar'] <= 0).astype(int).tolist()
    consec = []
    c = 0
    for v in is_loss:
        if v == 1:
            c += 1
        else:
            if c > 0:
                consec.append(c)
            c = 0
    if c > 0:
        consec.append(c)
    max_cl = max(consec) if consec else 0

    return {
        'label': label, 'trades': n, 'wr': wr, 'avg_win': avg_win,
        'avg_loss': avg_loss, 'expectancy': expectancy, 'net': net,
        'pf': pf, 'max_dd': max_dd, 'max_cl': max_cl,
    }


def print_metrics(m, indent=''):
    """Print formatted metrics."""
    if m is None:
        print(f'{indent}No trades')
        return
    pf_str = f'{m["pf"]:.2f}' if m['pf'] < 999 else 'INF'
    print(f'{indent}{m["label"]}')
    print(f'{indent}  Trades: {m["trades"]}  |  WR: {m["wr"]:.0f}%  |  Exp: ${m["expectancy"]:+.0f}  |  Net: ${m["net"]:+,.0f}  |  MaxDD: ${m["max_dd"]:+,.0f}  |  PF: {pf_str}  |  MaxCL: {m["max_cl"]}')
    print(f'{indent}  AvgWin: ${m["avg_win"]:+.0f}  |  AvgLoss: ${m["avg_loss"]:+.0f}')


def main():
    nq, es, ym = load_all_instruments()
    sessions = get_session_dates(nq)
    print(f'\nTotal RTH sessions: {len(sessions)}')
    print(f'Date range: {sessions[0]} to {sessions[-1]}')

    # ==================================================================
    #  STUDY A: Pre-Market Level Survey
    # ==================================================================
    print('\n' + '=' * 70)
    print('  STUDY A: PRE-MARKET LEVELS SURVEY')
    print('=' * 70)

    all_levels = []
    for sd in sessions:
        lv = compute_session_levels(nq, sd)
        all_levels.append(lv)

    lv_df = pd.DataFrame(all_levels)
    print(f'\n  Sessions with overnight data: {lv_df["overnight_high"].notna().sum()} / {len(sessions)}')
    print(f'  Sessions with Asia data:      {lv_df["asia_high"].notna().sum()} / {len(sessions)}')
    print(f'  Sessions with London data:    {lv_df["london_high"].notna().sum()} / {len(sessions)}')
    print(f'  Sessions with PDH/PDL:        {lv_df["pdh"].notna().sum()} / {len(sessions)}')

    # Overnight range stats
    on_range = lv_df['overnight_high'] - lv_df['overnight_low']
    asia_range = lv_df['asia_high'] - lv_df['asia_low']
    print(f'\n  Overnight range: avg {on_range.mean():.0f} pts, med {on_range.median():.0f} pts')
    print(f'  Asia range:      avg {asia_range.mean():.0f} pts, med {asia_range.median():.0f} pts')
    print(f'  OR range (30m):  avg {lv_df["or_range"].mean():.0f} pts, med {lv_df["or_range"].median():.0f} pts')
    print(f'  IB range (60m):  avg {lv_df["ib_range"].mean():.0f} pts, med {lv_df["ib_range"].median():.0f} pts')

    # ==================================================================
    #  STUDY B: How Often Does 9:30-10:00 Sweep Pre-Market Levels?
    # ==================================================================
    print('\n' + '=' * 70)
    print('  STUDY B: OPENING RANGE SWEEP FREQUENCY')
    print('=' * 70)

    sweep_counts = {
        'overnight_high': 0, 'overnight_low': 0,
        'asia_high': 0, 'asia_low': 0,
        'pdh': 0, 'pdl': 0,
        'any_sweep': 0,
    }

    for sd in sessions:
        lv = compute_session_levels(nq, sd)
        sd_ts = pd.Timestamp(sd)

        or_mask = (
            (nq['timestamp'] >= sd_ts + timedelta(hours=9, minutes=30)) &
            (nq['timestamp'] < sd_ts + timedelta(hours=10, minutes=15))
        )
        or_bars = nq[or_mask]
        if len(or_bars) == 0:
            continue

        any_sweep = False
        for _, bar in or_bars.iterrows():
            for level_name in ['overnight_high', 'overnight_low', 'asia_high', 'asia_low', 'pdh', 'pdl']:
                level_val = lv.get(level_name, np.nan)
                if np.isnan(level_val):
                    continue

                direction = 'above' if 'high' in level_name or level_name == 'pdh' else 'below'
                if detect_sweep(bar['high'], bar['low'], level_val, direction, threshold_pts=3):
                    sweep_counts[level_name] += 1
                    any_sweep = True
                    break  # Count each level once per session
            if any_sweep:
                break

        if any_sweep:
            sweep_counts['any_sweep'] += 1

    print(f'\n  Sweep frequency (9:30-10:15, {len(sessions)} sessions):')
    for level, count in sweep_counts.items():
        pct = count / len(sessions) * 100
        print(f'    {level:<20s}: {count:3d} sessions ({pct:.0f}%)')

    # ==================================================================
    #  STUDY C: SMT DIVERGENCE FREQUENCY
    # ==================================================================
    print('\n' + '=' * 70)
    print('  STUDY C: SMT DIVERGENCE FREQUENCY (NQ vs ES)')
    print('=' * 70)

    smt_sessions = {'bullish': 0, 'bearish': 0, 'any': 0}
    smt_total = {'bullish': 0, 'bearish': 0}

    for sd in sessions:
        sd_ts = pd.Timestamp(sd)
        entry_mask_nq = (
            (nq['timestamp'] >= sd_ts + timedelta(hours=9, minutes=30)) &
            (nq['timestamp'] <= sd_ts + timedelta(hours=10, minutes=30))
        )
        entry_mask_es = (
            (es['timestamp'] >= sd_ts + timedelta(hours=9, minutes=30)) &
            (es['timestamp'] <= sd_ts + timedelta(hours=10, minutes=30))
        )
        nq_bars = nq[entry_mask_nq]
        es_bars = es[entry_mask_es]

        divs = detect_smt_divergence(nq_bars, es_bars)
        has_bull = any(d[1] == 'BULLISH_SMT' for d in divs)
        has_bear = any(d[1] == 'BEARISH_SMT' for d in divs)

        if has_bull:
            smt_sessions['bullish'] += 1
        if has_bear:
            smt_sessions['bearish'] += 1
        if has_bull or has_bear:
            smt_sessions['any'] += 1

        smt_total['bullish'] += sum(1 for d in divs if d[1] == 'BULLISH_SMT')
        smt_total['bearish'] += sum(1 for d in divs if d[1] == 'BEARISH_SMT')

    print(f'\n  SMT divergences (9:30-10:30, {len(sessions)} sessions):')
    print(f'    Sessions with any SMT:     {smt_sessions["any"]} ({smt_sessions["any"]/len(sessions)*100:.0f}%)')
    print(f'    Sessions with bullish SMT:  {smt_sessions["bullish"]} ({smt_sessions["bullish"]/len(sessions)*100:.0f}%)')
    print(f'    Sessions with bearish SMT:  {smt_sessions["bearish"]} ({smt_sessions["bearish"]/len(sessions)*100:.0f}%)')
    print(f'    Total bullish SMT signals:  {smt_total["bullish"]}')
    print(f'    Total bearish SMT signals:  {smt_total["bearish"]}')

    # ==================================================================
    #  STUDY D: FVG FORMATION AFTER SWEEPS
    # ==================================================================
    print('\n' + '=' * 70)
    print('  STUDY D: FVG FORMATION FREQUENCY (5-min and 15-min)')
    print('=' * 70)

    fvg_counts = {'fvg_5m_bull': 0, 'fvg_5m_bear': 0, 'fvg_15m_bull': 0, 'fvg_15m_bear': 0}
    for sd in sessions:
        sd_ts = pd.Timestamp(sd)
        mask = (
            (nq['timestamp'] >= sd_ts + timedelta(hours=9, minutes=30)) &
            (nq['timestamp'] <= sd_ts + timedelta(hours=11))
        )
        bars = nq[mask]
        if len(bars) < 15:
            continue

        if len(detect_fvg_5min(bars, 'bull')) > 0:
            fvg_counts['fvg_5m_bull'] += 1
        if len(detect_fvg_5min(bars, 'bear')) > 0:
            fvg_counts['fvg_5m_bear'] += 1
        if len(detect_fvg_15min(bars, 'bull')) > 0:
            fvg_counts['fvg_15m_bull'] += 1
        if len(detect_fvg_15min(bars, 'bear')) > 0:
            fvg_counts['fvg_15m_bear'] += 1

    print(f'\n  FVG formation (9:30-11:00, {len(sessions)} sessions):')
    for k, v in fvg_counts.items():
        print(f'    {k:<20s}: {v:3d} sessions ({v/len(sessions)*100:.0f}%)')

    # ==================================================================
    #  STUDY E: FULL STRATEGY SIMULATION
    # ==================================================================
    print('\n' + '=' * 70)
    print('  STUDY E: FULL STRATEGY SIMULATION')
    print('=' * 70)

    trades = simulate_opening_range_strategy(nq, es, sessions)

    if len(trades) == 0:
        print('\n  No trades generated!')
        return

    print(f'\n  Total raw trades: {len(trades)}')
    print(f'  Sessions with trades: {trades["session_date"].nunique()} / {len(sessions)}')
    print(f'  Avg trades/session: {len(trades) / trades["session_date"].nunique():.1f}')

    # Direction breakdown
    print(f'\n  Direction: LONG={len(trades[trades["direction"]=="LONG"])}, SHORT={len(trades[trades["direction"]=="SHORT"])}')

    # Model breakdown
    print(f'\n  Entry models:')
    for model, count in trades['model'].value_counts().items():
        sub = trades[trades['model'] == model]
        wr = len(sub[sub['pnl_dollar'] > 0]) / len(sub) * 100 if len(sub) > 0 else 0
        print(f'    {model:<25s}: {count:3d} trades, WR={wr:.0f}%, Net=${sub["pnl_dollar"].sum():+,.0f}, Exp=${sub["pnl_dollar"].mean():+.0f}')

    # Exit reason breakdown
    print(f'\n  Exit reasons:')
    for reason, count in trades['exit_reason'].value_counts().items():
        sub = trades[trades['exit_reason'] == reason]
        print(f'    {reason:<12s}: {count:3d} trades, WR={len(sub[sub["pnl_dollar"]>0])/len(sub)*100:.0f}%, Net=${sub["pnl_dollar"].sum():+,.0f}')

    # Level swept breakdown
    print(f'\n  Levels swept:')
    for level, count in trades['level_swept'].value_counts().items():
        sub = trades[trades['level_swept'] == level]
        wr = len(sub[sub['pnl_dollar'] > 0]) / len(sub) * 100
        print(f'    {level:<20s}: {count:3d} trades, WR={wr:.0f}%, Net=${sub["pnl_dollar"].sum():+,.0f}')

    # ==================================================================
    #  STUDY E1: MODEL COMPARISON (filter combinations)
    # ==================================================================
    print('\n  --- MODEL COMPARISONS ---')

    # Model A: All raw trades (sweep only)
    m_all = calc_metrics(trades, 'Model RAW (sweep only)')
    print_metrics(m_all, '  ')

    # Model B: Sweep + SMT
    smt_trades = trades[trades['has_smt'] == True]
    m_smt = calc_metrics(smt_trades, 'Model B: Sweep + SMT')
    print_metrics(m_smt, '  ')

    # Model C: Sweep + FVG (5m or 15m)
    fvg_trades = trades[(trades['has_fvg_5m'] == True) | (trades['has_fvg_15m'] == True)]
    m_fvg = calc_metrics(fvg_trades, 'Model C: Sweep + FVG')
    print_metrics(m_fvg, '  ')

    # Model D: Sweep + VWAP aligned
    vwap_trades = trades[trades['vwap_aligned'] == True]
    m_vwap = calc_metrics(vwap_trades, 'Model D: Sweep + VWAP')
    print_metrics(m_vwap, '  ')

    # Model E: Sweep + delta confirm
    delta_trades = trades[trades['delta_confirm'] == True]
    m_delta = calc_metrics(delta_trades, 'Model E: Sweep + Delta')
    print_metrics(m_delta, '  ')

    # Model F: Sweep + SMT + FVG (full confluence)
    full_trades = trades[(trades['has_smt'] == True) & ((trades['has_fvg_5m'] == True) | (trades['has_fvg_15m'] == True))]
    m_full = calc_metrics(full_trades, 'Model F: Sweep + SMT + FVG')
    print_metrics(m_full, '  ')

    # Model G: Sweep + SMT + VWAP
    smt_vwap = trades[(trades['has_smt'] == True) & (trades['vwap_aligned'] == True)]
    m_smt_vwap = calc_metrics(smt_vwap, 'Model G: Sweep + SMT + VWAP')
    print_metrics(m_smt_vwap, '  ')

    # Model H: Sweep + FVG + Delta
    fvg_delta = trades[((trades['has_fvg_5m'] == True) | (trades['has_fvg_15m'] == True)) & (trades['delta_confirm'] == True)]
    m_fvg_delta = calc_metrics(fvg_delta, 'Model H: Sweep + FVG + Delta')
    print_metrics(m_fvg_delta, '  ')

    # LONG only
    print('\n  --- LONG ONLY ---')
    long_trades = trades[trades['direction'] == 'LONG']
    m_long = calc_metrics(long_trades, 'LONG only (all)')
    print_metrics(m_long, '  ')

    long_smt = long_trades[long_trades['has_smt'] == True]
    m_long_smt = calc_metrics(long_smt, 'LONG + SMT')
    print_metrics(m_long_smt, '  ')

    long_fvg = long_trades[(long_trades['has_fvg_5m'] == True) | (long_trades['has_fvg_15m'] == True)]
    m_long_fvg = calc_metrics(long_fvg, 'LONG + FVG')
    print_metrics(m_long_fvg, '  ')

    # SHORT only
    print('\n  --- SHORT ONLY ---')
    short_trades = trades[trades['direction'] == 'SHORT']
    m_short = calc_metrics(short_trades, 'SHORT only (all)')
    print_metrics(m_short, '  ')

    short_smt = short_trades[short_trades['has_smt'] == True]
    m_short_smt = calc_metrics(short_smt, 'SHORT + SMT')
    print_metrics(m_short_smt, '  ')

    # ==================================================================
    #  STUDY F: TRADE-BY-TRADE LOG
    # ==================================================================
    print('\n' + '=' * 70)
    print('  STUDY F: TRADE LOG (first 30)')
    print('=' * 70)

    cols = ['session_date', 'entry_time', 'direction', 'level_swept',
            'entry_price', 'exit_price', 'pnl_pts', 'pnl_dollar', 'r_multiple',
            'exit_reason', 'has_smt', 'has_fvg_5m', 'vwap_aligned', 'delta_confirm']

    trades_sorted = trades.sort_values('entry_time')
    for i, (_, t) in enumerate(trades_sorted.head(30).iterrows()):
        smt_flag = 'SMT' if t['has_smt'] else '   '
        fvg_flag = 'FVG' if t['has_fvg_5m'] or t.get('has_fvg_15m', False) else '   '
        vwap_flag = 'VW' if t['vwap_aligned'] else '  '
        delta_flag = 'D+' if t['delta_confirm'] else '  '
        win_flag = 'W' if t['pnl_dollar'] > 0 else 'L'

        et = pd.Timestamp(t['entry_time'])
        print(f'  {i+1:3d}. {str(t["session_date"])[:10]} {et.strftime("%H:%M")} '
              f'{t["direction"]:5s} swept={t["level_swept"]:<18s} '
              f'${t["pnl_dollar"]:+8.0f} ({t["r_multiple"]:+.1f}R) '
              f'{t["exit_reason"]:<10s} [{smt_flag}|{fvg_flag}|{vwap_flag}|{delta_flag}] {win_flag}')

    # ==================================================================
    #  STUDY G: COMPARISON TO EXISTING PLAYBOOK
    # ==================================================================
    print('\n' + '=' * 70)
    print('  STUDY G: COMPARISON TO EXISTING PORTFOLIO')
    print('=' * 70)

    print(f'\n  Existing Optimized Portfolio:')
    print(f'    52 trades, 83% WR, $264/trade, $13,706 net, -$351 MaxDD, PF 18.35')
    print(f'\n  Opening Range Strategy (best model):')

    # Find the best model
    best = None
    for m in [m_all, m_smt, m_fvg, m_vwap, m_delta, m_full, m_smt_vwap, m_fvg_delta,
              m_long, m_long_smt, m_long_fvg, m_short, m_short_smt]:
        if m is not None and (best is None or m['expectancy'] > best['expectancy']):
            if m['trades'] >= 5:  # minimum sample
                best = m

    if best:
        print_metrics(best, '    ')
        print(f'\n  Additive value (if non-overlapping sessions):')
        combined_net = 13706 + best['net']
        print(f'    Combined net: ${combined_net:+,.0f}')

    # R-multiple analysis
    print(f'\n  --- R-MULTIPLE DISTRIBUTION ---')
    if len(trades) > 0:
        print(f'  Avg R (all):     {trades["r_multiple"].mean():+.2f}R')
        print(f'  Avg R (winners): {trades[trades["pnl_dollar"]>0]["r_multiple"].mean():+.2f}R')
        print(f'  Avg R (losers):  {trades[trades["pnl_dollar"]<=0]["r_multiple"].mean():+.2f}R')
        print(f'  Max R:           {trades["r_multiple"].max():+.2f}R')
        print(f'  Min R:           {trades["r_multiple"].min():+.2f}R')

    print('\n' + '=' * 70)
    print('  DONE')
    print('=' * 70)


if __name__ == '__main__':
    main()
