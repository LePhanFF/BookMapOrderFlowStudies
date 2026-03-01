#!/usr/bin/env python
"""
Premarket Conditions Study for OR Acceptance

Core question: When NQ opens with acceptance (no Judas swing), does price
just continue running? What premarket conditions predict clean acceptance
vs Judas swing vs choppy consolidation?

This script:
  1. Classifies all sessions into: JUDAS, ACCEPTANCE, BOTH, CHOP
  2. Extracts premarket features (LRLR breaks, FVGs, level sweeps, momentum)
  3. Correlates premarket conditions with open type and trade outcomes
  4. Identifies which premarket patterns predict profitable acceptance trades

Data: Uses full_df (18:00 prev day → 16:00 session day) for premarket bars.
"""

import sys
from pathlib import Path
from datetime import timedelta, time as _time
from collections import defaultdict

import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from indicators.ict_models import detect_fvg_zones

# ── Constants ──────────────────────────────────────────────────────
OR_BARS = 15
EOR_BARS = 30
IB_BARS = 60
ATR_PERIOD = 14
SWEEP_THRESHOLD_RATIO = 0.17  # From or_reversal.py


# ═══════════════════════════════════════════════════════════════════
# SECTION 1: Data Loading & Overnight Level Computation
# ═══════════════════════════════════════════════════════════════════

def compute_overnight_levels(full_df, session_str):
    """Compute overnight/London/Asia/PDH levels (mirrors engine logic)."""
    sd = pd.Timestamp(session_str)
    prev_day = sd - timedelta(days=1)
    if prev_day.weekday() == 5:
        prev_day -= timedelta(days=1)
    elif prev_day.weekday() == 6:
        prev_day -= timedelta(days=2)

    ts = full_df['timestamp']
    levels = {}

    on_mask = (ts >= prev_day + timedelta(hours=18)) & (ts < sd + timedelta(hours=9, minutes=30))
    on_bars = full_df[on_mask]
    if len(on_bars) > 0:
        levels['overnight_high'] = on_bars['high'].max()
        levels['overnight_low'] = on_bars['low'].min()

    asia_mask = (ts >= prev_day + timedelta(hours=20)) & (ts < sd)
    asia = full_df[asia_mask]
    if len(asia) > 0:
        levels['asia_high'] = asia['high'].max()
        levels['asia_low'] = asia['low'].min()

    london_mask = (ts >= sd + timedelta(hours=2)) & (ts < sd + timedelta(hours=5))
    london = full_df[london_mask]
    if len(london) > 0:
        levels['london_high'] = london['high'].max()
        levels['london_low'] = london['low'].min()

    prior_rth_mask = (ts >= prev_day + timedelta(hours=9, minutes=30)) & (ts <= prev_day + timedelta(hours=16))
    prior_rth = full_df[prior_rth_mask]
    if len(prior_rth) > 0:
        levels['pdh'] = prior_rth['high'].max()
        levels['pdl'] = prior_rth['low'].min()
        levels['pdc'] = prior_rth.iloc[-1]['close']

    return levels


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: Session Classification (Judas vs Acceptance vs Chop)
# ═══════════════════════════════════════════════════════════════════

def check_judas_sweep(ib_bars, levels, eor_range):
    """Check if EOR made a Judas sweep (false break of a key level + reversal).

    Returns: (has_judas, direction, swept_level_name, swept_level_value)
    """
    if eor_range < 10:
        return False, None, None, None

    eor_bars = ib_bars.iloc[:EOR_BARS]
    or_bars = ib_bars.iloc[:OR_BARS]
    eor_high = eor_bars['high'].max()
    eor_low = eor_bars['low'].min()
    or_mid = (or_bars['high'].max() + or_bars['low'].min()) / 2
    sweep_threshold = SWEEP_THRESHOLD_RATIO * eor_range

    # Build candidate levels
    high_candidates = []
    low_candidates = []
    for name, key in [('LDN_HIGH', 'london_high'), ('ASIA_HIGH', 'asia_high'),
                       ('PDH', 'pdh'), ('ON_HIGH', 'overnight_high')]:
        v = levels.get(key)
        if v is not None:
            high_candidates.append((name, v))
    for name, key in [('LDN_LOW', 'london_low'), ('ASIA_LOW', 'asia_low'),
                       ('PDL', 'pdl'), ('ON_LOW', 'overnight_low')]:
        v = levels.get(key)
        if v is not None:
            low_candidates.append((name, v))

    # Check HIGH sweep (potential SHORT reversal = Judas up)
    best_high_sweep = None
    for name, lvl in high_candidates:
        dist = abs(eor_high - lvl)
        if dist <= sweep_threshold and eor_high >= lvl:
            if best_high_sweep is None or dist < best_high_sweep[2]:
                best_high_sweep = (name, lvl, dist)

    # Check LOW sweep (potential LONG reversal = Judas down)
    best_low_sweep = None
    for name, lvl in low_candidates:
        dist = abs(eor_low - lvl)
        if dist <= sweep_threshold and eor_low <= lvl:
            if best_low_sweep is None or dist < best_low_sweep[2]:
                best_low_sweep = (name, lvl, dist)

    # Check for reversal through OR mid
    last_eor_close = eor_bars['close'].iloc[-1]

    if best_high_sweep and last_eor_close < or_mid:
        return True, 'SHORT', best_high_sweep[0], best_high_sweep[1]
    if best_low_sweep and last_eor_close > or_mid:
        return True, 'LONG', best_low_sweep[0], best_low_sweep[1]

    return False, None, None, None


def check_acceptance(ib_bars, levels, eor_range, window='IB'):
    """Check if EOR shows acceptance (price held beyond a key level).

    Returns: (has_acceptance, direction, level_name, level_value, close_pct)
    """
    if eor_range < 10:
        return False, None, None, None, 0.0

    if window == 'OR':
        check_bars = ib_bars.iloc[:OR_BARS]
    elif window == 'IB':
        check_bars = ib_bars
    else:
        check_bars = ib_bars.iloc[:EOR_BARS]

    n = len(check_bars)
    if n == 0:
        return False, None, None, None, 0.0

    max_wicks = 5
    close_pct_thresh = 0.70
    buffer_pct = 0.10

    short_cands = []
    long_cands = []
    for name, key in [('LDN_LOW', 'london_low'), ('ASIA_LOW', 'asia_low'), ('PDL', 'pdl')]:
        v = levels.get(key)
        if v is not None:
            short_cands.append((name, v))
    for name, key in [('LDN_HIGH', 'london_high'), ('ASIA_HIGH', 'asia_high'), ('PDH', 'pdh')]:
        v = levels.get(key)
        if v is not None:
            long_cands.append((name, v))

    best = None
    for direction, candidates in [('SHORT', short_cands), ('LONG', long_cands)]:
        for name, lvl in candidates:
            if direction == 'SHORT':
                violations = int((check_bars['high'] > lvl).sum())
                closes_correct = int((check_bars['close'] < lvl).sum())
                max_spike = float(check_bars['high'].max()) - lvl
            else:
                violations = int((check_bars['low'] < lvl).sum())
                closes_correct = int((check_bars['close'] > lvl).sum())
                max_spike = lvl - float(check_bars['low'].min())

            close_pct = closes_correct / n
            buffer = buffer_pct * eor_range

            if (violations <= max_wicks and close_pct >= close_pct_thresh
                    and max_spike <= buffer):
                if best is None or close_pct > best[4]:
                    best = (True, direction, name, lvl, close_pct)

    if best:
        return best
    return False, None, None, None, 0.0


def classify_session(ib_bars, levels):
    """Classify session open type: JUDAS, ACCEPTANCE, BOTH, or CHOP."""
    eor_bars = ib_bars.iloc[:EOR_BARS]
    eor_range = eor_bars['high'].max() - eor_bars['low'].min()

    has_judas, judas_dir, judas_lvl_name, judas_lvl_val = check_judas_sweep(
        ib_bars, levels, eor_range)
    has_accept, accept_dir, accept_lvl_name, accept_lvl_val, accept_pct = check_acceptance(
        ib_bars, levels, eor_range)

    if has_judas and has_accept:
        return 'BOTH', {
            'judas_dir': judas_dir, 'judas_level': judas_lvl_name,
            'accept_dir': accept_dir, 'accept_level': accept_lvl_name,
            'accept_pct': accept_pct,
        }
    elif has_judas:
        return 'JUDAS', {
            'judas_dir': judas_dir, 'judas_level': judas_lvl_name,
        }
    elif has_accept:
        return 'ACCEPTANCE', {
            'accept_dir': accept_dir, 'accept_level': accept_lvl_name,
            'accept_pct': accept_pct,
        }
    else:
        return 'CHOP', {}


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: LRLR Swing Detection
# ═══════════════════════════════════════════════════════════════════

def resample_to_5m(bars_1m):
    """Resample 1-min bars to 5-min bars."""
    if 'timestamp' not in bars_1m.columns or len(bars_1m) < 5:
        return pd.DataFrame()

    df = bars_1m.set_index('timestamp')
    bars_5m = df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna(subset=['open'])

    # Add vol_delta if available
    if 'vol_delta' in df.columns:
        bars_5m['vol_delta'] = df['vol_delta'].resample('5min').sum()

    bars_5m = bars_5m.reset_index()
    return bars_5m


def detect_swing_points(bars, lookback=2):
    """Detect swing highs and lows using N-bar pivot detection.

    Swing High: bar[i].high > max(bar[i-lookback:i].high) AND bar[i].high > max(bar[i+1:i+lookback+1].high)
    Swing Low: bar[i].low < min(bar[i-lookback:i].low) AND bar[i].low < min(bar[i+1:i+lookback+1].low)

    Returns: list of (index, price, type) where type is 'HIGH' or 'LOW'
    """
    if len(bars) < 2 * lookback + 1:
        return []

    highs = bars['high'].values
    lows = bars['low'].values
    swings = []

    for i in range(lookback, len(bars) - lookback):
        # Swing High
        is_high = True
        for j in range(1, lookback + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_high = False
                break
        if is_high:
            swings.append((i, highs[i], 'HIGH'))

        # Swing Low
        is_low = True
        for j in range(1, lookback + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_low = False
                break
        if is_low:
            swings.append((i, lows[i], 'LOW'))

    return sorted(swings, key=lambda x: x[0])


def detect_lrlr_break(swings, bars):
    """Detect LRLR (bearish trendline) or HLHL (bullish trendline) breaks.

    Bearish LRLR: series of lower highs → price breaks above the last lower high
    Bullish HLHL: series of higher lows → price breaks below the last higher low

    Returns: dict with keys:
        has_bearish_break: bool (price broke above descending trendline = bullish)
        has_bullish_break: bool (price broke below ascending trendline = bearish)
        bearish_break_bar: int or None
        bullish_break_bar: int or None
        num_lower_highs: int
        num_higher_lows: int
    """
    result = {
        'has_bearish_break': False,  # bullish: broke above bearish LRLR trendline
        'has_bullish_break': False,  # bearish: broke below bullish HLHL trendline
        'bearish_break_bar': None,
        'bullish_break_bar': None,
        'num_lower_highs': 0,
        'num_higher_lows': 0,
    }

    if len(swings) < 3:
        return result

    highs = bars['high'].values
    closes = bars['close'].values

    # Extract swing highs and swing lows
    swing_highs = [(idx, price) for idx, price, typ in swings if typ == 'HIGH']
    swing_lows = [(idx, price) for idx, price, typ in swings if typ == 'LOW']

    # Check for LRLR (lower highs) → bearish trendline
    if len(swing_highs) >= 2:
        lower_high_count = 0
        for i in range(1, len(swing_highs)):
            if swing_highs[i][1] < swing_highs[i-1][1]:
                lower_high_count += 1
            else:
                lower_high_count = 0  # reset

        result['num_lower_highs'] = lower_high_count

        # If at least 2 consecutive lower highs → bearish trendline exists
        if lower_high_count >= 2:
            last_lower_high_price = swing_highs[-1][1]
            last_lower_high_bar = swing_highs[-1][0]

            # Check if any bar AFTER the last swing high broke above it
            for j in range(last_lower_high_bar + 1, len(bars)):
                if closes[j] > last_lower_high_price:
                    result['has_bearish_break'] = True
                    result['bearish_break_bar'] = j
                    break

    # Check for HLHL (higher lows) → bullish trendline
    if len(swing_lows) >= 2:
        higher_low_count = 0
        for i in range(1, len(swing_lows)):
            if swing_lows[i][1] > swing_lows[i-1][1]:
                higher_low_count += 1
            else:
                higher_low_count = 0

        result['num_higher_lows'] = higher_low_count

        if higher_low_count >= 2:
            last_higher_low_price = swing_lows[-1][1]
            last_higher_low_bar = swing_lows[-1][0]

            for j in range(last_higher_low_bar + 1, len(bars)):
                if closes[j] < last_higher_low_price:
                    result['has_bullish_break'] = True
                    result['bullish_break_bar'] = j
                    break

    return result


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: Premarket Feature Extraction
# ═══════════════════════════════════════════════════════════════════

def get_premarket_bars(full_df, session_str):
    """Get premarket bars segmented into windows."""
    sd = pd.Timestamp(session_str)
    prev_day = sd - timedelta(days=1)
    if prev_day.weekday() == 5:
        prev_day -= timedelta(days=1)
    elif prev_day.weekday() == 6:
        prev_day -= timedelta(days=2)

    ts = full_df['timestamp']

    # Full overnight: 18:00 prev → 9:29 session day
    full_mask = (ts >= prev_day + timedelta(hours=18)) & (ts < sd + timedelta(hours=9, minutes=30))
    full_overnight = full_df[full_mask].copy()

    # Post-London / Pre-RTH: 05:00 → 09:30 (most important window)
    post_london_mask = (ts >= sd + timedelta(hours=5)) & (ts < sd + timedelta(hours=9, minutes=30))
    post_london = full_df[post_london_mask].copy()

    # Last 60 min before open: 08:30 → 09:30
    last_60m_mask = (ts >= sd + timedelta(hours=8, minutes=30)) & (ts < sd + timedelta(hours=9, minutes=30))
    last_60m = full_df[last_60m_mask].copy()

    # London session: 02:00 → 05:00
    london_mask = (ts >= sd + timedelta(hours=2)) & (ts < sd + timedelta(hours=5))
    london = full_df[london_mask].copy()

    return {
        'full_overnight': full_overnight,
        'post_london': post_london,
        'last_60m': last_60m,
        'london': london,
    }


def extract_premarket_features(full_df, session_str, levels, ib_bars):
    """Extract all premarket features for a session.

    Returns: dict of feature name → value
    """
    features = {}
    pm = get_premarket_bars(full_df, session_str)

    full_on = pm['full_overnight']
    post_london = pm['post_london']
    last_60m = pm['last_60m']

    if len(full_on) == 0:
        return features

    # ── 1. Key Level Sweeps in Premarket ───────────────────────────
    level_names = ['london_high', 'london_low', 'asia_high', 'asia_low', 'pdh', 'pdl']
    sweep_count = 0
    clean_break_count = 0

    for lname in level_names:
        lvl = levels.get(lname)
        if lvl is None:
            features[f'pm_sweep_{lname}'] = 0
            features[f'pm_clean_break_{lname}'] = 0
            continue

        is_high_level = lname in ('london_high', 'asia_high', 'pdh')

        if is_high_level:
            # Did premarket price go above this level?
            swept = (full_on['high'] >= lvl).any()
            # Clean break = multiple closes above
            if swept:
                closes_above = (full_on['close'] > lvl).sum()
                clean = closes_above >= 3
            else:
                clean = False
        else:
            swept = (full_on['low'] <= lvl).any()
            if swept:
                closes_below = (full_on['close'] < lvl).sum()
                clean = closes_below >= 3
            else:
                clean = False

        features[f'pm_sweep_{lname}'] = int(swept)
        features[f'pm_clean_break_{lname}'] = int(clean)
        sweep_count += int(swept)
        clean_break_count += int(clean)

    features['pm_total_sweeps'] = sweep_count
    features['pm_total_clean_breaks'] = clean_break_count

    # ── 2. LRLR Trendline Break Detection ──────────────────────────
    # Use post-London bars (05:00-09:30) resampled to 5-min
    if len(post_london) >= 10:
        bars_5m = resample_to_5m(post_london)
        if len(bars_5m) >= 7:
            swings = detect_swing_points(bars_5m, lookback=2)
            lrlr = detect_lrlr_break(swings, bars_5m)
            features['pm_lrlr_bearish_break'] = int(lrlr['has_bearish_break'])
            features['pm_lrlr_bullish_break'] = int(lrlr['has_bullish_break'])
            features['pm_num_lower_highs'] = lrlr['num_lower_highs']
            features['pm_num_higher_lows'] = lrlr['num_higher_lows']
            features['pm_lrlr_any_break'] = int(
                lrlr['has_bearish_break'] or lrlr['has_bullish_break'])
        else:
            features['pm_lrlr_bearish_break'] = 0
            features['pm_lrlr_bullish_break'] = 0
            features['pm_num_lower_highs'] = 0
            features['pm_num_higher_lows'] = 0
            features['pm_lrlr_any_break'] = 0
    else:
        features['pm_lrlr_bearish_break'] = 0
        features['pm_lrlr_bullish_break'] = 0
        features['pm_num_lower_highs'] = 0
        features['pm_num_higher_lows'] = 0
        features['pm_lrlr_any_break'] = 0

    # ── 3. FVG Detection in Premarket ──────────────────────────────
    # 1-min FVGs in last 60 min before open
    if len(last_60m) >= 5:
        fvg_zones_1m = detect_fvg_zones(
            last_60m['high'].values, last_60m['low'].values,
            last_60m['close'].values, min_gap_pct=0.0001)
        bull_fvg_1m = sum(1 for z in fvg_zones_1m if z.direction == 'BULL')
        bear_fvg_1m = sum(1 for z in fvg_zones_1m if z.direction == 'BEAR')
    else:
        bull_fvg_1m = 0
        bear_fvg_1m = 0

    features['pm_fvg_bull_1m'] = bull_fvg_1m
    features['pm_fvg_bear_1m'] = bear_fvg_1m
    features['pm_fvg_total_1m'] = bull_fvg_1m + bear_fvg_1m

    # 5-min FVGs in last 60 min
    if len(last_60m) >= 15:
        bars_5m_last = resample_to_5m(last_60m)
        if len(bars_5m_last) >= 3:
            fvg_zones_5m = detect_fvg_zones(
                bars_5m_last['high'].values, bars_5m_last['low'].values,
                bars_5m_last['close'].values, min_gap_pct=0.0002)
            bull_fvg_5m = sum(1 for z in fvg_zones_5m if z.direction == 'BULL')
            bear_fvg_5m = sum(1 for z in fvg_zones_5m if z.direction == 'BEAR')
        else:
            bull_fvg_5m = 0
            bear_fvg_5m = 0
    else:
        bull_fvg_5m = 0
        bear_fvg_5m = 0

    features['pm_fvg_bull_5m'] = bull_fvg_5m
    features['pm_fvg_bear_5m'] = bear_fvg_5m
    features['pm_fvg_total_5m'] = bull_fvg_5m + bear_fvg_5m

    # 15-min FVGs in post-London
    if len(post_london) >= 45:
        bars_15m = post_london.set_index('timestamp').resample('15min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        }).dropna()
        if len(bars_15m) >= 3:
            fvg_zones_15m = detect_fvg_zones(
                bars_15m['high'].values, bars_15m['low'].values,
                bars_15m['close'].values, min_gap_pct=0.0002)
            bull_fvg_15m = sum(1 for z in fvg_zones_15m if z.direction == 'BULL')
            bear_fvg_15m = sum(1 for z in fvg_zones_15m if z.direction == 'BEAR')
        else:
            bull_fvg_15m = 0
            bear_fvg_15m = 0
    else:
        bull_fvg_15m = 0
        bear_fvg_15m = 0

    features['pm_fvg_bull_15m'] = bull_fvg_15m
    features['pm_fvg_bear_15m'] = bear_fvg_15m
    features['pm_fvg_total_15m'] = bull_fvg_15m + bear_fvg_15m

    # ── 4. Gap Analysis at Open ────────────────────────────────────
    rth_open = ib_bars['open'].iloc[0] if len(ib_bars) > 0 else None
    last_pm_close = full_on['close'].iloc[-1] if len(full_on) > 0 else None

    if rth_open is not None and last_pm_close is not None:
        gap_pts = rth_open - last_pm_close
        on_range = full_on['high'].max() - full_on['low'].min()
        features['pm_gap_pts'] = gap_pts
        features['pm_gap_pct_of_on_range'] = (abs(gap_pts) / on_range * 100) if on_range > 0 else 0
        features['pm_gap_direction'] = 1 if gap_pts > 0 else (-1 if gap_pts < 0 else 0)

        # Where does RTH open relative to overnight range?
        on_high = full_on['high'].max()
        on_low = full_on['low'].min()
        if on_range > 0:
            features['pm_open_in_on_pct'] = (rth_open - on_low) / on_range * 100
        else:
            features['pm_open_in_on_pct'] = 50.0
    else:
        features['pm_gap_pts'] = 0
        features['pm_gap_pct_of_on_range'] = 0
        features['pm_gap_direction'] = 0
        features['pm_open_in_on_pct'] = 50.0

    # ── 5. Premarket Momentum ──────────────────────────────────────
    # Slope of price in last 30 min and 60 min before open
    if len(last_60m) >= 10:
        closes = last_60m['close'].values
        x = np.arange(len(closes))
        slope_60m = np.polyfit(x, closes, 1)[0] if len(closes) >= 2 else 0.0
        features['pm_slope_60m'] = slope_60m

        # Last 30 min
        closes_30m = closes[-30:] if len(closes) >= 30 else closes
        x30 = np.arange(len(closes_30m))
        slope_30m = np.polyfit(x30, closes_30m, 1)[0] if len(closes_30m) >= 2 else 0.0
        features['pm_slope_30m'] = slope_30m
    else:
        features['pm_slope_60m'] = 0.0
        features['pm_slope_30m'] = 0.0

    # CVD in last 60 min
    if len(last_60m) >= 5 and 'vol_delta' in last_60m.columns:
        cvd = last_60m['vol_delta'].fillna(0).cumsum()
        cvd_final = cvd.iloc[-1] if len(cvd) > 0 else 0
        features['pm_cvd_60m'] = float(cvd_final)
        features['pm_cvd_direction'] = 1 if cvd_final > 0 else (-1 if cvd_final < 0 else 0)
    else:
        features['pm_cvd_60m'] = 0.0
        features['pm_cvd_direction'] = 0

    # Volume acceleration: avg volume last 15 min vs 15 min before that
    if len(last_60m) >= 30 and 'volume' in last_60m.columns:
        vol_last_15 = last_60m['volume'].iloc[-15:].mean()
        vol_prev_15 = last_60m['volume'].iloc[-30:-15].mean()
        features['pm_vol_acceleration'] = (vol_last_15 / vol_prev_15) if vol_prev_15 > 0 else 1.0
    else:
        features['pm_vol_acceleration'] = 1.0

    # Post-London directional push
    if len(post_london) >= 10:
        pl_range = post_london['high'].max() - post_london['low'].min()
        pl_move = post_london['close'].iloc[-1] - post_london['open'].iloc[0]
        features['pm_post_london_directional'] = abs(pl_move) / pl_range if pl_range > 0 else 0.0
        features['pm_post_london_direction'] = 1 if pl_move > 0 else (-1 if pl_move < 0 else 0)
    else:
        features['pm_post_london_directional'] = 0.0
        features['pm_post_london_direction'] = 0

    # ── 6. Chop Detection ──────────────────────────────────────────
    if len(last_60m) >= 10:
        on_range = full_on['high'].max() - full_on['low'].min()
        last_range = last_60m['high'].max() - last_60m['low'].min()
        features['pm_chop_range_ratio'] = last_range / on_range if on_range > 0 else 0.5

        # Direction changes in last 30 bars
        last_30 = last_60m.tail(30)
        if len(last_30) >= 3:
            close_diffs = last_30['close'].diff().dropna()
            sign_changes = ((close_diffs > 0).astype(int).diff().abs().dropna() > 0).sum()
            features['pm_direction_changes'] = int(sign_changes)
            features['pm_chop_score'] = sign_changes / len(close_diffs) if len(close_diffs) > 0 else 0.5
        else:
            features['pm_direction_changes'] = 0
            features['pm_chop_score'] = 0.5
    else:
        features['pm_chop_range_ratio'] = 0.5
        features['pm_direction_changes'] = 0
        features['pm_chop_score'] = 0.5

    return features


# ═══════════════════════════════════════════════════════════════════
# SECTION 5: Trade Outcome Matching
# ═══════════════════════════════════════════════════════════════════

def load_trade_log():
    """Load the trade log and parse it."""
    trade_log_path = project_root / 'output' / 'trade_log.csv'
    if not trade_log_path.exists():
        print(f"WARNING: Trade log not found at {trade_log_path}")
        return pd.DataFrame()
    df = pd.read_csv(trade_log_path)
    df['session_date'] = pd.to_datetime(df['session_date'])
    return df


# ═══════════════════════════════════════════════════════════════════
# SECTION 6: Analysis & Reporting
# ═══════════════════════════════════════════════════════════════════

def print_classification_summary(results):
    """Print session classification summary table."""
    counts = defaultdict(int)
    for r in results:
        counts[r['open_type']] += 1

    total = len(results)
    print("\n" + "=" * 60)
    print("SESSION CLASSIFICATION SUMMARY")
    print("=" * 60)
    print(f"{'Open Type':<15} {'Count':>6} {'% of Sessions':>15}")
    print("-" * 40)
    for otype in ['JUDAS', 'ACCEPTANCE', 'BOTH', 'CHOP']:
        c = counts.get(otype, 0)
        pct = c / total * 100 if total > 0 else 0
        print(f"{otype:<15} {c:>6} {pct:>14.1f}%")
    print(f"{'TOTAL':<15} {total:>6} {'100.0%':>15}")


def print_feature_comparison(results):
    """Print premarket feature comparison by open type."""
    feature_keys = [
        'pm_total_sweeps', 'pm_total_clean_breaks',
        'pm_lrlr_any_break', 'pm_lrlr_bearish_break', 'pm_lrlr_bullish_break',
        'pm_num_lower_highs', 'pm_num_higher_lows',
        'pm_fvg_total_1m', 'pm_fvg_total_5m', 'pm_fvg_total_15m',
        'pm_gap_pts', 'pm_gap_pct_of_on_range',
        'pm_slope_60m', 'pm_slope_30m',
        'pm_cvd_direction',
        'pm_post_london_directional',
        'pm_vol_acceleration',
        'pm_chop_range_ratio', 'pm_chop_score', 'pm_direction_changes',
        'pm_open_in_on_pct',
    ]

    by_type = defaultdict(list)
    for r in results:
        by_type[r['open_type']].append(r['features'])

    print("\n" + "=" * 90)
    print("PREMARKET FEATURES BY OPEN TYPE (mean values)")
    print("=" * 90)
    types = ['JUDAS', 'ACCEPTANCE', 'BOTH', 'CHOP']
    header = f"{'Feature':<30}"
    for t in types:
        header += f" {t:>12}"
    print(header)
    print("-" * 90)

    for fkey in feature_keys:
        row = f"{fkey:<30}"
        for t in types:
            vals = [f.get(fkey, 0) for f in by_type.get(t, [])]
            if vals:
                mean_val = np.mean(vals)
                # For binary features, show as percentage
                if fkey.startswith('pm_lrlr_') and 'break' in fkey:
                    row += f" {mean_val*100:>11.1f}%"
                else:
                    row += f" {mean_val:>12.2f}"
            else:
                row += f" {'N/A':>12}"
        print(row)


def print_acceptance_outcomes(results, trade_log):
    """Print acceptance trade outcomes segmented by premarket condition."""
    # Get acceptance sessions
    accept_sessions = [r for r in results if r['open_type'] in ('ACCEPTANCE', 'BOTH')]

    if len(accept_sessions) == 0:
        print("\nNo acceptance sessions found.")
        return

    # Match with trade log
    accept_trades = trade_log[trade_log['strategy_name'] == 'OR Acceptance'].copy()
    if len(accept_trades) == 0:
        print("\nNo OR Acceptance trades found in trade log.")
        return

    # Build session date → features lookup
    date_features = {}
    for r in accept_sessions:
        date_features[r['session_date']] = r['features']

    # Add features to trades
    for idx, trade in accept_trades.iterrows():
        tdate = str(trade['session_date']).split(' ')[0]
        feats = date_features.get(tdate, {})
        for k, v in feats.items():
            accept_trades.at[idx, k] = v

    print("\n" + "=" * 80)
    print("ACCEPTANCE TRADE OUTCOMES BY PREMARKET CONDITION")
    print("=" * 80)

    # Define condition splits
    conditions = [
        ('LRLR break present', 'pm_lrlr_any_break', lambda x: x == 1),
        ('No LRLR break', 'pm_lrlr_any_break', lambda x: x == 0),
        ('Level swept + clean break', 'pm_total_clean_breaks', lambda x: x >= 2),
        ('Few levels swept', 'pm_total_sweeps', lambda x: x <= 1),
        ('Gap aligned (>5 pts)', 'pm_gap_pts', lambda x: abs(x) > 5),
        ('Small gap (<5 pts)', 'pm_gap_pts', lambda x: abs(x) <= 5),
        ('Strong PM momentum', 'pm_post_london_directional', lambda x: x > 0.5),
        ('Weak PM momentum', 'pm_post_london_directional', lambda x: x <= 0.3),
        ('High chop score (>0.4)', 'pm_chop_score', lambda x: x > 0.4),
        ('Low chop score (<=0.4)', 'pm_chop_score', lambda x: x <= 0.4),
        ('5m FVGs present', 'pm_fvg_total_5m', lambda x: x >= 1),
        ('No 5m FVGs', 'pm_fvg_total_5m', lambda x: x == 0),
        ('CVD bullish', 'pm_cvd_direction', lambda x: x > 0),
        ('CVD bearish', 'pm_cvd_direction', lambda x: x < 0),
    ]

    print(f"{'Condition':<35} {'Trades':>7} {'WR':>7} {'Avg P&L':>10} {'Net P&L':>10} {'PF':>7}")
    print("-" * 80)

    for cond_name, feat_key, cond_fn in conditions:
        if feat_key in accept_trades.columns:
            mask = accept_trades[feat_key].apply(lambda x: cond_fn(x) if not pd.isna(x) else False)
            subset = accept_trades[mask]
        else:
            subset = pd.DataFrame()

        if len(subset) == 0:
            print(f"{cond_name:<35} {'0':>7} {'N/A':>7} {'N/A':>10} {'N/A':>10} {'N/A':>7}")
            continue

        n_trades = len(subset)
        winners = (subset['net_pnl'] > 0).sum()
        wr = winners / n_trades * 100
        avg_pnl = subset['net_pnl'].mean()
        net_pnl = subset['net_pnl'].sum()
        gross_win = subset.loc[subset['net_pnl'] > 0, 'net_pnl'].sum()
        gross_loss = abs(subset.loc[subset['net_pnl'] <= 0, 'net_pnl'].sum())
        pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

        print(f"{cond_name:<35} {n_trades:>7} {wr:>6.1f}% {avg_pnl:>9.0f}$ {net_pnl:>9.0f}$ {pf:>7.2f}")


def print_top_bottom_trades(results, trade_log, n=10):
    """Print top 10 best and worst acceptance trades with premarket features."""
    accept_trades = trade_log[trade_log['strategy_name'] == 'OR Acceptance'].copy()
    if len(accept_trades) == 0:
        return

    # Build session date → features lookup
    date_features = {}
    for r in results:
        if r['open_type'] in ('ACCEPTANCE', 'BOTH'):
            date_features[r['session_date']] = r['features']

    # Add features to trades
    for idx, trade in accept_trades.iterrows():
        tdate = str(trade['session_date']).split(' ')[0]
        feats = date_features.get(tdate, {})
        for k, v in feats.items():
            accept_trades.at[idx, k] = v

    display_feats = [
        'pm_total_sweeps', 'pm_lrlr_any_break', 'pm_fvg_total_5m',
        'pm_slope_30m', 'pm_chop_score', 'pm_gap_pts',
    ]

    for label, subset in [
        (f"TOP {n} BEST ACCEPTANCE TRADES", accept_trades.nlargest(n, 'net_pnl')),
        (f"TOP {n} WORST ACCEPTANCE TRADES", accept_trades.nsmallest(n, 'net_pnl')),
    ]:
        print(f"\n{'=' * 120}")
        print(label)
        print('=' * 120)
        header = f"{'Date':<12} {'Dir':<6} {'P&L':>8}"
        for f in display_feats:
            short = f.replace('pm_', '').replace('total_', '').replace('any_', '')[:12]
            header += f" {short:>12}"
        print(header)
        print('-' * 120)

        for _, trade in subset.iterrows():
            tdate = str(trade['session_date']).split(' ')[0]
            row = f"{tdate:<12} {trade.get('direction', '?'):<6} {trade['net_pnl']:>7.0f}$"
            for f in display_feats:
                val = trade.get(f, 'N/A')
                if isinstance(val, float):
                    row += f" {val:>12.2f}"
                else:
                    row += f" {str(val):>12}"
            print(row)


def print_first_30m_continuation_analysis(results, full_df):
    """Analyze: after acceptance in first 30 min, does price continue?

    For acceptance sessions, measure how far price moves in the acceptance
    direction in the first 30 min (OR) and first 60 min (IB) after open.
    """
    accept_sessions = [r for r in results if r['open_type'] in ('ACCEPTANCE', 'BOTH')]
    if not accept_sessions:
        print("\nNo acceptance sessions for continuation analysis.")
        return

    print("\n" + "=" * 80)
    print("FIRST 30/60 MIN CONTINUATION ANALYSIS (Acceptance Sessions)")
    print("=" * 80)
    print("Does price continue in the acceptance direction after accepting?")
    print()

    continuations = []
    for sess in accept_sessions:
        session_str = sess['session_date']
        info = sess['info']
        accept_dir = info.get('accept_dir')
        if not accept_dir:
            continue

        sd = pd.Timestamp(session_str)
        ts = full_df['timestamp']

        # Get first 30 min of RTH (9:30-10:00) and first 60 min (9:30-10:30)
        first_30m_mask = (ts >= sd + timedelta(hours=9, minutes=30)) & (ts < sd + timedelta(hours=10))
        first_60m_mask = (ts >= sd + timedelta(hours=9, minutes=30)) & (ts < sd + timedelta(hours=10, minutes=30))
        first_30m = full_df[first_30m_mask]
        first_60m = full_df[first_60m_mask]

        if len(first_30m) < 5 or len(first_60m) < 5:
            continue

        open_price = first_30m['open'].iloc[0]

        if accept_dir == 'SHORT':
            move_30m = open_price - first_30m['low'].min()
            move_60m = open_price - first_60m['low'].min()
            end_30m = first_30m['close'].iloc[-1]
            continued_30m = end_30m < open_price
            end_60m = first_60m['close'].iloc[-1]
            continued_60m = end_60m < open_price
        else:  # LONG
            move_30m = first_30m['high'].max() - open_price
            move_60m = first_60m['high'].max() - open_price
            end_30m = first_30m['close'].iloc[-1]
            continued_30m = end_30m > open_price
            end_60m = first_60m['close'].iloc[-1]
            continued_60m = end_60m > open_price

        continuations.append({
            'session_date': session_str,
            'direction': accept_dir,
            'move_30m': move_30m,
            'move_60m': move_60m,
            'continued_30m': continued_30m,
            'continued_60m': continued_60m,
            'features': sess['features'],
        })

    if not continuations:
        print("No valid acceptance sessions with continuation data.")
        return

    n = len(continuations)
    cont_30m = sum(1 for c in continuations if c['continued_30m'])
    cont_60m = sum(1 for c in continuations if c['continued_60m'])
    avg_move_30m = np.mean([c['move_30m'] for c in continuations])
    avg_move_60m = np.mean([c['move_60m'] for c in continuations])
    med_move_30m = np.median([c['move_30m'] for c in continuations])
    med_move_60m = np.median([c['move_60m'] for c in continuations])

    print(f"Total acceptance sessions analyzed: {n}")
    print()
    print(f"{'Metric':<40} {'30-min':>12} {'60-min':>12}")
    print("-" * 65)
    c30_str = f"{cont_30m}/{n} ({cont_30m/n*100:.1f}%)"
    c60_str = f"{cont_60m}/{n} ({cont_60m/n*100:.1f}%)"
    print(f"{'Continued in direction':<40} {c30_str:>12} {c60_str:>12}")
    print(f"{'Avg move (pts)':<40} {avg_move_30m:>12.1f} {avg_move_60m:>12.1f}")
    print(f"{'Median move (pts)':<40} {med_move_30m:>12.1f} {med_move_60m:>12.1f}")

    # Segment by LRLR break
    with_lrlr = [c for c in continuations if c['features'].get('pm_lrlr_any_break', 0) == 1]
    without_lrlr = [c for c in continuations if c['features'].get('pm_lrlr_any_break', 0) == 0]

    print()
    print("Segmented by LRLR Break:")
    print(f"{'Condition':<25} {'N':>5} {'Cont 30m':>12} {'Cont 60m':>12} {'Avg 60m':>10}")
    print("-" * 65)

    for label, subset in [('With LRLR break', with_lrlr), ('Without LRLR break', without_lrlr)]:
        if subset:
            n_s = len(subset)
            c30 = sum(1 for c in subset if c['continued_30m'])
            c60 = sum(1 for c in subset if c['continued_60m'])
            avg60 = np.mean([c['move_60m'] for c in subset])
            c30_s = f"{c30}/{n_s} ({c30/n_s*100:.0f}%)"
            c60_s = f"{c60}/{n_s} ({c60/n_s*100:.0f}%)"
            print(f"{label:<25} {n_s:>5} {c30_s:>14} {c60_s:>14} {avg60:>9.1f}")
        else:
            print(f"{label:<25} {'0':>5} {'N/A':>12} {'N/A':>12} {'N/A':>10}")

    # Segment by chop score
    low_chop = [c for c in continuations if c['features'].get('pm_chop_score', 0.5) <= 0.35]
    high_chop = [c for c in continuations if c['features'].get('pm_chop_score', 0.5) > 0.45]

    print()
    print("Segmented by Chop Score:")
    for label, subset in [('Low chop (<=0.35)', low_chop), ('High chop (>0.45)', high_chop)]:
        if subset:
            n_s = len(subset)
            c30 = sum(1 for c in subset if c['continued_30m'])
            c60 = sum(1 for c in subset if c['continued_60m'])
            avg60 = np.mean([c['move_60m'] for c in subset])
            c30_s = f"{c30}/{n_s} ({c30/n_s*100:.0f}%)"
            c60_s = f"{c60}/{n_s} ({c60/n_s*100:.0f}%)"
            print(f"{label:<25} {n_s:>5} {c30_s:>14} {c60_s:>14} {avg60:>9.1f}")
        else:
            print(f"{label:<25} {'0':>5} {'N/A':>12} {'N/A':>12} {'N/A':>10}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("PREMARKET CONDITIONS STUDY FOR OR ACCEPTANCE")
    print("=" * 60)
    print()

    # Load data
    full_df = load_csv('NQ')
    df_rth = filter_rth(full_df)
    df_rth = compute_all_features(df_rth)

    sessions = sorted(df_rth['session_date'].unique())
    print(f"\nTotal sessions: {len(sessions)}")

    # Load trade log
    trade_log = load_trade_log()
    if len(trade_log) > 0:
        print(f"Trade log: {len(trade_log)} trades loaded")
        accept_trades = trade_log[trade_log['strategy_name'] == 'OR Acceptance']
        rev_trades = trade_log[trade_log['strategy_name'] == 'Opening Range Rev']
        print(f"  OR Acceptance: {len(accept_trades)} trades")
        print(f"  OR Reversal:   {len(rev_trades)} trades")

    # Process each session
    results = []
    print(f"\nProcessing {len(sessions)} sessions...")

    for i, session_date in enumerate(sessions):
        session_str = str(session_date).split(' ')[0]
        session_df = df_rth[df_rth['session_date'] == session_date].copy()

        if len(session_df) < IB_BARS:
            continue

        ib_bars = session_df.head(IB_BARS)
        levels = compute_overnight_levels(full_df, session_str)

        # Classify session
        open_type, info = classify_session(ib_bars, levels)

        # Extract premarket features
        features = extract_premarket_features(full_df, session_str, levels, ib_bars)

        results.append({
            'session_date': session_str,
            'open_type': open_type,
            'info': info,
            'features': features,
        })

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(sessions)} sessions...")

    print(f"\nProcessed {len(results)} sessions total.")

    # ── Print Reports ──────────────────────────────────────────────
    print_classification_summary(results)
    print_feature_comparison(results)
    print_acceptance_outcomes(results, trade_log)
    print_top_bottom_trades(results, trade_log, n=10)
    print_first_30m_continuation_analysis(results, full_df)

    # ── Save results to CSV for further analysis ───────────────────
    rows = []
    for r in results:
        row = {
            'session_date': r['session_date'],
            'open_type': r['open_type'],
        }
        row.update(r['info'])
        row.update(r['features'])
        rows.append(row)

    results_df = pd.DataFrame(rows)
    output_path = project_root / 'output' / 'premarket_study_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    print(f"  {len(results_df)} sessions × {len(results_df.columns)} features")

    print("\n" + "=" * 60)
    print("STUDY COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
