"""
SMT (Smart Money Technique) Divergence Detector
=================================================

Detects divergence between correlated instruments (ES/NQ, ES/YM) at key
swing points. When one instrument makes a new extreme but the other fails
to confirm, it signals institutional distribution/accumulation.

Bullish SMT: Primary makes lower low, comparison makes higher low → buy
Bearish SMT: Primary makes higher high, comparison makes lower high → sell

Usage:
    smt = SMTDivergence(primary_df, comparison_df, lookback=5)
    signals = smt.detect()
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class SwingPoint:
    """A detected swing high or low."""
    bar_index: int
    price: float
    swing_type: str  # 'HIGH' or 'LOW'
    timestamp: object = None


@dataclass
class SMTSignal:
    """An SMT divergence signal."""
    bar_index: int
    direction: str       # 'BULLISH' or 'BEARISH'
    primary_price: float
    comparison_price: float
    confidence: str      # 'high' or 'medium'
    timestamp: object = None


def detect_swing_points(
    highs: np.ndarray,
    lows: np.ndarray,
    lookback: int = 5,
    timestamps: Optional[np.ndarray] = None,
) -> List[SwingPoint]:
    """
    Detect swing highs and lows using a simple N-bar pivot.

    A swing high at index i requires:
      highs[i] > highs[i-j] for j in 1..lookback
      highs[i] > highs[i+j] for j in 1..lookback

    A swing low at index i requires:
      lows[i] < lows[i-j] for j in 1..lookback
      lows[i] < lows[i+j] for j in 1..lookback
    """
    n = len(highs)
    swings = []

    for i in range(lookback, n - lookback):
        # Check swing high
        is_high = True
        for j in range(1, lookback + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_high = False
                break

        if is_high:
            ts = timestamps[i] if timestamps is not None else None
            swings.append(SwingPoint(
                bar_index=i,
                price=highs[i],
                swing_type='HIGH',
                timestamp=ts,
            ))

        # Check swing low
        is_low = True
        for j in range(1, lookback + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_low = False
                break

        if is_low:
            ts = timestamps[i] if timestamps is not None else None
            swings.append(SwingPoint(
                bar_index=i,
                price=lows[i],
                swing_type='LOW',
                timestamp=ts,
            ))

    return swings


def detect_smt_divergence(
    primary_highs: np.ndarray,
    primary_lows: np.ndarray,
    comparison_highs: np.ndarray,
    comparison_lows: np.ndarray,
    lookback: int = 5,
    max_bar_offset: int = 3,
    timestamps: Optional[np.ndarray] = None,
) -> List[SMTSignal]:
    """
    Detect SMT divergence between two instruments.

    Compares swing points: when primary makes a new extreme but comparison
    fails to confirm, it signals a divergence.

    Args:
        primary_highs/lows: Price data for primary instrument
        comparison_highs/lows: Price data for comparison instrument
        lookback: Bars each side for swing detection
        max_bar_offset: Max bars difference for matching swings
        timestamps: Optional timestamp array for labeling

    Returns:
        List of SMTSignal objects
    """
    primary_swings = detect_swing_points(primary_highs, primary_lows, lookback, timestamps)
    comparison_swings = detect_swing_points(comparison_highs, comparison_lows, lookback, timestamps)

    signals = []

    # Group swings by type
    primary_highs_list = [s for s in primary_swings if s.swing_type == 'HIGH']
    primary_lows_list = [s for s in primary_swings if s.swing_type == 'LOW']
    comp_highs_list = [s for s in comparison_swings if s.swing_type == 'HIGH']
    comp_lows_list = [s for s in comparison_swings if s.swing_type == 'LOW']

    # Check bearish SMT: primary makes higher high, comparison makes lower high
    for i in range(1, len(primary_highs_list)):
        p_curr = primary_highs_list[i]
        p_prev = primary_highs_list[i - 1]

        if p_curr.price <= p_prev.price:
            continue  # Primary didn't make new high

        # Find matching comparison swing
        c_match = _find_nearest_swing(comp_highs_list, p_curr.bar_index, max_bar_offset)
        c_prev = _find_nearest_swing(comp_highs_list, p_prev.bar_index, max_bar_offset)

        if c_match is None or c_prev is None:
            continue

        if c_match.price < c_prev.price:
            # Bearish divergence: primary higher high, comparison lower high
            confidence = 'high' if (p_curr.price - p_prev.price) > 5 else 'medium'
            ts = timestamps[p_curr.bar_index] if timestamps is not None else None
            signals.append(SMTSignal(
                bar_index=p_curr.bar_index,
                direction='BEARISH',
                primary_price=p_curr.price,
                comparison_price=c_match.price,
                confidence=confidence,
                timestamp=ts,
            ))

    # Check bullish SMT: primary makes lower low, comparison makes higher low
    for i in range(1, len(primary_lows_list)):
        p_curr = primary_lows_list[i]
        p_prev = primary_lows_list[i - 1]

        if p_curr.price >= p_prev.price:
            continue  # Primary didn't make new low

        # Find matching comparison swing
        c_match = _find_nearest_swing(comp_lows_list, p_curr.bar_index, max_bar_offset)
        c_prev = _find_nearest_swing(comp_lows_list, p_prev.bar_index, max_bar_offset)

        if c_match is None or c_prev is None:
            continue

        if c_match.price > c_prev.price:
            # Bullish divergence: primary lower low, comparison higher low
            confidence = 'high' if (p_prev.price - p_curr.price) > 5 else 'medium'
            ts = timestamps[p_curr.bar_index] if timestamps is not None else None
            signals.append(SMTSignal(
                bar_index=p_curr.bar_index,
                direction='BULLISH',
                primary_price=p_curr.price,
                comparison_price=c_match.price,
                confidence=confidence,
                timestamp=ts,
            ))

    return signals


def _find_nearest_swing(
    swings: List[SwingPoint],
    target_bar: int,
    max_offset: int,
) -> Optional[SwingPoint]:
    """Find the swing point closest to target_bar within max_offset."""
    best = None
    best_dist = max_offset + 1

    for s in swings:
        dist = abs(s.bar_index - target_bar)
        if dist <= max_offset and dist < best_dist:
            best = s
            best_dist = dist

    return best


def compute_smt_features(
    primary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    lookback: int = 3,
    max_bar_offset: int = 5,
) -> pd.DataFrame:
    """
    Compute SMT divergence features and merge into primary dataframe.

    Adds columns:
      - smt_bullish: True on bars where bullish SMT detected
      - smt_bearish: True on bars where bearish SMT detected
      - smt_signal: 'BULLISH', 'BEARISH', or None

    Both DataFrames must have 'session_date', 'high', 'low' columns
    and be aligned by timestamp.
    """
    primary_df = primary_df.copy()
    primary_df['smt_bullish'] = False
    primary_df['smt_bearish'] = False
    primary_df['smt_signal'] = None

    # Process per session
    for session_date in primary_df['session_date'].unique():
        p_session = primary_df[primary_df['session_date'] == session_date]
        c_session = comparison_df[comparison_df['session_date'] == session_date]

        if len(p_session) < 30 or len(c_session) < 30:
            continue

        # Align by position (both should have same number of 1-min bars)
        min_len = min(len(p_session), len(c_session))

        p_highs = p_session['high'].values[:min_len]
        p_lows = p_session['low'].values[:min_len]
        c_highs = c_session['high'].values[:min_len]
        c_lows = c_session['low'].values[:min_len]

        timestamps = p_session['timestamp'].values[:min_len] if 'timestamp' in p_session.columns else None

        signals = detect_smt_divergence(
            p_highs, p_lows, c_highs, c_lows,
            lookback=lookback,
            max_bar_offset=max_bar_offset,
            timestamps=timestamps,
        )

        # Map signals back to primary dataframe
        p_indices = p_session.index[:min_len]
        for sig in signals:
            if sig.bar_index < len(p_indices):
                idx = p_indices[sig.bar_index]
                if sig.direction == 'BULLISH':
                    primary_df.at[idx, 'smt_bullish'] = True
                    primary_df.at[idx, 'smt_signal'] = 'BULLISH'
                elif sig.direction == 'BEARISH':
                    primary_df.at[idx, 'smt_bearish'] = True
                    primary_df.at[idx, 'smt_signal'] = 'BEARISH'

    return primary_df
