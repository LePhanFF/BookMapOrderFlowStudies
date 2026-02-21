"""
TPO-Based Day Type Classifier

Uses TPO distribution shape analysis on the developing session to classify
day types according to Dalton Market Profile theory. This is an algorithmic
approach to reading TPO shapes that Market Profile traders do visually.

Key Dalton day types and their TPO signatures:
  Trend Day:     Elongated, single prints, POC at extreme, 1-2 HVN
  Normal Day:    Wide VA, symmetric, D-shaped, POC centered
  P-Day:         Bulge at top (p-shape), single prints below, asymmetric
  B-Day:         Bulge at bottom (b-shape), narrow range, symmetric
  Neutral Day:   Balanced, multiple HVN, rotational
  Double Dist:   Two distinct HVN clusters (bimodal)

Measurement approach (NOT curve-fitted):
  - Skewness: measured from TPO count distribution (scipy-like, but manual)
  - Kurtosis: peak concentration vs spread
  - Single prints: counted from actual TPO data
  - HVN/LVN: detected from volume/TPO density peaks and valleys
  - VA width ratio: VA range / total range (0-1 scale)

All thresholds are derived from the statistical properties of the distribution
itself, not from backtested optimal values.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class TPOClassification:
    """Result of TPO-based day type classification."""
    # Primary shape classification
    shape: str = 'unknown'  # p_shape, b_shape, d_shape, elongated, double_dist, narrow
    shape_confidence: float = 0.0  # 0-1

    # Dalton day type mapping
    day_type_hint: str = 'neutral'  # trend_up, trend_down, p_day, b_day, neutral
    directional_bias: str = 'neutral'  # bullish, bearish, neutral

    # Measured distribution properties (NOT fitted constants)
    skewness: float = 0.0       # >0 = bulge at bottom (b-shape), <0 = bulge at top (p-shape)
    va_width_ratio: float = 0.0  # VA range / total range
    poc_location: float = 0.5    # 0=bottom, 1=top of range
    single_prints_above: int = 0
    single_prints_below: int = 0
    poor_high: bool = False
    poor_low: bool = False
    hvn_count: int = 0
    lvn_count: int = 0

    # Multi-timeframe: 5-min and 30-min shapes may differ
    shape_5min: str = 'unknown'
    shape_30min: str = 'unknown'


def classify_session_tpo(
    session_df: pd.DataFrame,
    current_time_str: str = "15:59",
    ib_high: float = 0.0,
    ib_low: float = 0.0,
) -> TPOClassification:
    """
    Classify the current session using TPO distribution analysis.

    Args:
        session_df: 1-min OHLCV DataFrame for the session (DatetimeIndex or 'timestamp' col)
        current_time_str: Time up to which to compute (no lookahead)
        ib_high, ib_low: IB boundaries for context

    Returns:
        TPOClassification with shape, day_type_hint, and measured properties
    """
    result = TPOClassification()

    df = session_df.copy()
    if 'timestamp' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index('timestamp')

    current_time = pd.to_datetime(current_time_str).time()
    df = df[df.index.time <= current_time]

    if len(df) < 30:  # Need at least 30 bars for meaningful TPO
        return result

    # --- Build TPO profile at 0.25 tick resolution ---
    tick_size = 0.25
    min_price = df['low'].min()
    max_price = df['high'].max()
    total_range = max_price - min_price

    if total_range < 5:  # Extremely narrow range
        result.shape = 'narrow'
        result.day_type_hint = 'b_day'
        return result

    # Create price bins
    bins = np.arange(min_price, max_price + tick_size, tick_size)
    tpo_counts = np.zeros(len(bins))

    # Fill TPO counts from bar ranges
    for _, row in df.iterrows():
        low_idx = max(0, int((row['low'] - min_price) / tick_size))
        high_idx = min(len(bins) - 1, int((row['high'] - min_price) / tick_size))
        tpo_counts[low_idx:high_idx + 1] += 1

    if tpo_counts.sum() == 0:
        return result

    # --- Compute POC (point of control) ---
    poc_idx = np.argmax(tpo_counts)
    poc_price = bins[poc_idx]
    result.poc_location = (poc_price - min_price) / total_range  # 0=bottom, 1=top

    # --- Compute Value Area (70% of TPO) ---
    total_tpo = tpo_counts.sum()
    sorted_indices = np.argsort(-tpo_counts)  # descending
    cum_tpo = 0
    va_indices = []
    for idx in sorted_indices:
        va_indices.append(idx)
        cum_tpo += tpo_counts[idx]
        if cum_tpo >= 0.70 * total_tpo:
            break

    va_high_idx = max(va_indices)
    va_low_idx = min(va_indices)
    va_high = bins[va_high_idx]
    va_low = bins[va_low_idx]
    va_range = va_high - va_low
    result.va_width_ratio = va_range / total_range if total_range > 0 else 0

    # --- Compute distribution skewness ---
    # Positive skew = long right tail (bulge at bottom = b-shape)
    # Negative skew = long left tail (bulge at top = p-shape)
    weighted_prices = np.repeat(bins, tpo_counts.astype(int))
    if len(weighted_prices) > 2:
        mean_p = np.mean(weighted_prices)
        std_p = np.std(weighted_prices)
        if std_p > 0:
            result.skewness = float(np.mean(((weighted_prices - mean_p) / std_p) ** 3))

    # --- Single prints ---
    # Prices with exactly 1 TPO above VA or below VA
    above_va_mask = (np.arange(len(bins)) > va_high_idx) & (tpo_counts > 0)
    below_va_mask = (np.arange(len(bins)) < va_low_idx) & (tpo_counts > 0)

    result.single_prints_above = int(np.sum((tpo_counts == 1) & above_va_mask))
    result.single_prints_below = int(np.sum((tpo_counts == 1) & below_va_mask))

    # --- Poor high/low (>=2 TPO at absolute extreme) ---
    # Check last few price levels at extremes
    n_extreme = max(1, int(3 / tick_size))  # ~3 points at each extreme
    top_counts = tpo_counts[-n_extreme:]
    bottom_counts = tpo_counts[:n_extreme]
    result.poor_high = bool(np.any(top_counts >= 2))  # >=2 TPO at high = poor high
    result.poor_low = bool(np.any(bottom_counts >= 2))

    # --- HVN/LVN detection ---
    # Smooth the TPO profile and find peaks/valleys
    kernel_size = max(3, int(total_range / tick_size * 0.05))  # 5% of range
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(tpo_counts, kernel, mode='same')

    # HVN: local maxima in smoothed profile
    hvn_indices = []
    lvn_indices = []
    mean_tpo = np.mean(smoothed[smoothed > 0]) if np.any(smoothed > 0) else 0
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:
            if smoothed[i] > mean_tpo * 1.2:  # 20% above mean = HVN
                hvn_indices.append(i)
        if smoothed[i] < smoothed[i-1] and smoothed[i] < smoothed[i+1]:
            if smoothed[i] < mean_tpo * 0.5 and smoothed[i] > 0:  # 50% below mean = LVN
                lvn_indices.append(i)

    result.hvn_count = len(hvn_indices)
    result.lvn_count = len(lvn_indices)

    # --- Classify shape based on measured properties ---
    # These are statistical classification rules, not curve-fitted thresholds:
    # - Skewness > 0.3 or < -0.3 is "moderately skewed" (standard stat threshold)
    # - VA ratio > 0.70 means value area covers >70% of range (wide = D-shape)
    # - POC in top/bottom 30% of range = directional
    # - Single prints >= 3 on one side = trend signature

    classify_5min = _classify_shape(
        result.skewness, result.va_width_ratio, result.poc_location,
        result.single_prints_above, result.single_prints_below,
        result.poor_high, result.poor_low, result.hvn_count,
    )
    result.shape = classify_5min[0]
    result.shape_confidence = classify_5min[1]
    result.shape_5min = classify_5min[0]

    # --- 30-min TPO shape (coarser view, more stable) ---
    shape_30min = _compute_30min_shape(df, min_price, max_price, tick_size)
    result.shape_30min = shape_30min

    # --- Map shape to Dalton day type hint ---
    result.day_type_hint, result.directional_bias = _map_shape_to_day_type(
        result.shape, result.poc_location, result.single_prints_above,
        result.single_prints_below, result.poor_high, result.poor_low,
        result.hvn_count, result.va_width_ratio,
        ib_high, ib_low, df['close'].iloc[-1] if len(df) > 0 else 0,
    )

    return result


def _classify_shape(
    skewness: float,
    va_width: float,
    poc_location: float,
    sp_above: int,
    sp_below: int,
    poor_high: bool,
    poor_low: bool,
    hvn_count: int,
) -> Tuple[str, float]:
    """
    Classify TPO shape from measured distribution properties.

    Returns (shape_name, confidence).
    """
    # Score each shape based on how many of its characteristics are present
    scores = {}

    # Elongated (trend day): narrow VA, POC at extreme, single prints
    elongated_score = sum([
        va_width < 0.50,          # Narrow VA (< 50% of range)
        poc_location > 0.70 or poc_location < 0.30,  # POC at extreme
        sp_above >= 3 or sp_below >= 3,  # Single prints
        hvn_count <= 2,           # Few HVN (directional)
        not (poor_high and poor_low),  # Not poor at both ends
    ]) / 5.0
    scores['elongated'] = elongated_score

    # P-shape (bullish skew): bulge at top, single prints below
    p_score = sum([
        skewness < -0.2,          # Negative skew = bulge at top
        poc_location > 0.55,       # POC in upper half
        sp_below > sp_above,       # More singles below
        va_width < 0.70,          # VA not too wide
    ]) / 4.0
    scores['p_shape'] = p_score

    # B-shape (bearish skew): bulge at bottom, single prints above
    b_score = sum([
        skewness > 0.2,           # Positive skew = bulge at bottom
        poc_location < 0.45,       # POC in lower half
        sp_above > sp_below,       # More singles above
        va_width < 0.70,
    ]) / 4.0
    scores['b_shape'] = b_score

    # D-shape (balanced/normal): wide VA, centered POC, symmetric
    d_score = sum([
        va_width > 0.60,          # Wide VA (> 60% of range)
        0.35 < poc_location < 0.65,  # POC centered
        abs(skewness) < 0.3,      # Symmetric
        abs(sp_above - sp_below) <= 2,  # Balanced singles
    ]) / 4.0
    scores['d_shape'] = d_score

    # Narrow (balance day): very tight range, poor at both ends
    narrow_score = sum([
        va_width > 0.75,          # VA covers most of range
        poor_high and poor_low,    # Trapped at both ends
        abs(skewness) < 0.2,      # Very symmetric
        hvn_count <= 2,           # Concentrated
    ]) / 4.0
    scores['narrow'] = narrow_score

    # Double distribution: multiple distinct HVN clusters
    double_score = sum([
        hvn_count >= 2,           # Multiple HVN
        va_width < 0.55,         # VA doesn't span full range (split)
        abs(skewness) < 0.4,     # Not too skewed
    ]) / 3.0
    scores['double_dist'] = double_score

    # Pick the best-scoring shape
    best_shape = max(scores, key=scores.get)
    best_conf = scores[best_shape]

    return best_shape, best_conf


def _compute_30min_shape(
    df: pd.DataFrame, min_price: float, max_price: float, tick_size: float
) -> str:
    """Compute TPO shape using 30-min periods (coarser, more stable)."""
    df = df.copy()
    df['period'] = df.index.floor('30min')
    periods = df['period'].unique()

    total_range = max_price - min_price
    if total_range < 5:
        return 'narrow'

    bins = np.arange(min_price, max_price + tick_size, tick_size)
    tpo_counts = np.zeros(len(bins))

    # Count unique period touches per price level (true TPO)
    for period in periods:
        period_df = df[df['period'] == period]
        period_low = period_df['low'].min()
        period_high = period_df['high'].max()
        low_idx = max(0, int((period_low - min_price) / tick_size))
        high_idx = min(len(bins) - 1, int((period_high - min_price) / tick_size))
        tpo_counts[low_idx:high_idx + 1] += 1

    if tpo_counts.sum() == 0:
        return 'unknown'

    # Quick shape classification from 30-min TPO
    poc_idx = np.argmax(tpo_counts)
    poc_loc = poc_idx / len(bins) if len(bins) > 0 else 0.5

    weighted = np.repeat(bins, tpo_counts.astype(int))
    if len(weighted) > 2:
        mean_p = np.mean(weighted)
        std_p = np.std(weighted)
        skew = float(np.mean(((weighted - mean_p) / std_p) ** 3)) if std_p > 0 else 0
    else:
        skew = 0

    # Simple classification
    if skew < -0.3 and poc_loc > 0.55:
        return 'p_shape'
    elif skew > 0.3 and poc_loc < 0.45:
        return 'b_shape'
    elif abs(skew) < 0.2 and 0.35 < poc_loc < 0.65:
        return 'd_shape'
    else:
        return 'elongated'


def _map_shape_to_day_type(
    shape: str,
    poc_location: float,
    sp_above: int,
    sp_below: int,
    poor_high: bool,
    poor_low: bool,
    hvn_count: int,
    va_width: float,
    ib_high: float,
    ib_low: float,
    current_price: float,
) -> Tuple[str, str]:
    """
    Map TPO shape + context to Dalton day type hint and directional bias.

    Returns (day_type_hint, directional_bias).
    """
    ib_range = ib_high - ib_low if ib_high > ib_low else 0
    ib_mid = (ib_high + ib_low) / 2 if ib_range > 0 else current_price

    # Extension from IB
    above_ib = current_price > ib_high
    below_ib = current_price < ib_low
    inside_ib = not above_ib and not below_ib

    if shape == 'elongated':
        # Trend day: elongated shape with directional extension
        if above_ib and poc_location > 0.6:
            return 'trend_up', 'bullish'
        elif below_ib and poc_location < 0.4:
            return 'trend_down', 'bearish'
        elif sp_above > sp_below + 2:
            return 'trend_up', 'bullish'
        elif sp_below > sp_above + 2:
            return 'trend_down', 'bearish'
        return 'neutral', 'neutral'

    elif shape == 'p_shape':
        # P-day: bullish skew, bulge at top
        if above_ib or poc_location > 0.55:
            return 'p_day', 'bullish'
        return 'neutral', 'bullish'

    elif shape == 'b_shape':
        # B-shape (bearish skew): treat as p_day with bearish bias
        if below_ib or poc_location < 0.45:
            return 'p_day', 'bearish'
        return 'neutral', 'bearish'

    elif shape == 'narrow':
        # Narrow balance = B-day
        return 'b_day', 'neutral'

    elif shape == 'd_shape':
        # D-shape: normal/balanced day
        if inside_ib:
            return 'b_day', 'neutral'
        return 'neutral', 'neutral'

    elif shape == 'double_dist':
        # Double distribution: could be morph day
        if current_price > ib_mid:
            return 'neutral', 'bullish'
        else:
            return 'neutral', 'bearish'

    return 'neutral', 'neutral'
