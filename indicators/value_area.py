"""
Value Area Calculator — CBOT Standard POC Expansion Method
============================================================

Computes Value Area (VAH, VAL, POC) using the correct algorithm:
  1. Find POC (price with max volume/TPO count)
  2. Expand contiguously outward from POC
  3. At each step, compare volume above vs below
  4. Add the side with more volume
  5. Stop when 70% of total volume is captured

This produces a CONTIGUOUS price range, unlike the sort-descending
method used in the old rockit-framework code which could skip price levels.

Also computes prior-day VA levels for each session (for 80% Rule strategy).
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ValueAreaLevels:
    """Value Area levels for a session."""
    poc: float       # Point of Control (highest volume price)
    vah: float       # Value Area High
    val: float       # Value Area Low
    va_width: float  # VAH - VAL
    session_high: float
    session_low: float
    session_range: float


def calculate_value_area(
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    tick_size: float = 0.25,
    va_percent: float = 0.70,
) -> Optional[ValueAreaLevels]:
    """
    Calculate Value Area using the CBOT standard POC expansion method.

    Each bar's volume is distributed evenly across all price bins from its
    low to its high, producing a proper volume profile.

    Args:
        highs: Array of high prices per bar
        lows: Array of low prices per bar
        volumes: Array of volumes for each bar
        tick_size: Price bin size (0.25 for NQ/ES)
        va_percent: Fraction of volume for VA (0.70 = 70%)

    Returns:
        ValueAreaLevels or None if insufficient data
    """
    if len(highs) == 0 or len(volumes) == 0:
        return None

    price_min = float(np.min(lows))
    price_max = float(np.max(highs))

    if price_max - price_min < tick_size:
        return None

    # Build bin edges covering the full price range
    bin_start = np.floor(price_min / tick_size) * tick_size
    bin_end = np.ceil(price_max / tick_size) * tick_size + tick_size
    bins = np.arange(bin_start, bin_end, tick_size)

    if len(bins) < 2:
        return None

    n_profile = len(bins) - 1
    vol_profile = np.zeros(n_profile)

    # Distribute each bar's volume across all bins it spans (low to high)
    for bar_high, bar_low, bar_vol in zip(highs, lows, volumes):
        if bar_vol <= 0 or np.isnan(bar_vol):
            continue
        lo_idx = int(np.clip(np.searchsorted(bins, bar_low, side='right') - 1, 0, n_profile - 1))
        hi_idx = int(np.clip(np.searchsorted(bins, bar_high, side='right') - 1, 0, n_profile - 1))
        if lo_idx > hi_idx:
            lo_idx, hi_idx = hi_idx, lo_idx
        n_touched = hi_idx - lo_idx + 1
        vol_per_bin = bar_vol / n_touched
        vol_profile[lo_idx:hi_idx + 1] += vol_per_bin

    bin_centers = (bins[:-1] + bins[1:]) / 2

    total_volume = vol_profile.sum()
    if total_volume == 0:
        return None

    # Step 1: Find POC (price bin with max volume)
    poc_idx = int(np.argmax(vol_profile))
    poc_price = float(bin_centers[poc_idx])

    # Step 2: Expand outward from POC
    target_volume = total_volume * va_percent
    accumulated = vol_profile[poc_idx]
    vah_idx = poc_idx
    val_idx = poc_idx

    while accumulated < target_volume:
        above_idx = vah_idx + 1
        below_idx = val_idx - 1

        above_vol = vol_profile[above_idx] if above_idx < n_profile else 0
        below_vol = vol_profile[below_idx] if below_idx >= 0 else 0

        if above_vol == 0 and below_vol == 0:
            break

        if above_vol >= below_vol:
            accumulated += above_vol
            vah_idx = above_idx
        else:
            accumulated += below_vol
            val_idx = below_idx

    vah_price = float(bin_centers[vah_idx])
    val_price = float(bin_centers[val_idx])

    return ValueAreaLevels(
        poc=poc_price,
        vah=vah_price,
        val=val_price,
        va_width=vah_price - val_price,
        session_high=float(np.max(highs)),
        session_low=float(np.min(lows)),
        session_range=float(np.max(highs) - np.min(lows)),
    )


def compute_session_value_areas(
    df: pd.DataFrame,
    tick_size: float = 0.25,
    va_percent: float = 0.70,
) -> Dict[str, ValueAreaLevels]:
    """
    Compute Value Area for each session in the dataframe.

    Args:
        df: DataFrame with 'session_date', 'close', 'volume', 'high', 'low'
        tick_size: Price bin size
        va_percent: VA percentage

    Returns:
        Dict mapping session_date (str) -> ValueAreaLevels
    """
    va_by_session = {}

    for session_date in sorted(df['session_date'].unique()):
        session_df = df[df['session_date'] == session_date]

        if len(session_df) < 30:
            continue

        highs = session_df['high'].values
        lows = session_df['low'].values
        volumes = session_df['volume'].values

        va = calculate_value_area(highs, lows, volumes, tick_size, va_percent)
        if va is not None:
            va_by_session[str(session_date)] = va

    return va_by_session


def add_prior_va_features(df: pd.DataFrame, tick_size: float = 0.25) -> pd.DataFrame:
    """
    Compute per-session VA and add PRIOR day's VA levels to each bar.

    Adds columns:
      - prior_va_poc, prior_va_vah, prior_va_val, prior_va_width
      - prior_va_high, prior_va_low
      - open_vs_va: 'ABOVE_VAH', 'BELOW_VAL', or 'INSIDE_VA'
    """
    df = df.copy()

    # Initialize columns
    for col in ['prior_va_poc', 'prior_va_vah', 'prior_va_val',
                'prior_va_width', 'prior_va_high', 'prior_va_low']:
        df[col] = np.nan
    df['open_vs_va'] = None

    # Compute VA for each session
    va_by_session = compute_session_value_areas(df, tick_size)

    sessions = sorted(df['session_date'].unique())

    for i in range(1, len(sessions)):
        current_session = sessions[i]
        prior_session = sessions[i - 1]

        prior_key = str(prior_session)
        if prior_key not in va_by_session:
            continue

        prior_va = va_by_session[prior_key]
        session_idx = df[df['session_date'] == current_session].index

        df.loc[session_idx, 'prior_va_poc'] = prior_va.poc
        df.loc[session_idx, 'prior_va_vah'] = prior_va.vah
        df.loc[session_idx, 'prior_va_val'] = prior_va.val
        df.loc[session_idx, 'prior_va_width'] = prior_va.va_width
        df.loc[session_idx, 'prior_va_high'] = prior_va.session_high
        df.loc[session_idx, 'prior_va_low'] = prior_va.session_low

        # Classify open relative to VA
        session_df = df.loc[session_idx]
        if len(session_df) > 0:
            open_price = session_df['open'].iloc[0]
            if open_price > prior_va.vah:
                df.loc[session_idx, 'open_vs_va'] = 'ABOVE_VAH'
            elif open_price < prior_va.val:
                df.loc[session_idx, 'open_vs_va'] = 'BELOW_VAL'
            else:
                df.loc[session_idx, 'open_vs_va'] = 'INSIDE_VA'

    return df


def compute_composite_value_area(
    df: pd.DataFrame,
    session_date,
    lookback_days: int = 3,
    tick_size: float = 0.25,
    va_percent: float = 0.70,
) -> Optional[ValueAreaLevels]:
    """
    Compute a composite Value Area spanning multiple prior sessions.

    Merges volume profiles from the last N sessions to produce a more robust
    VA that reflects longer-term institutional positioning.

    Args:
        df: Full DataFrame with session_date, high, low, volume
        session_date: Current session date (composite uses prior sessions)
        lookback_days: Number of prior sessions to include (3 = 3-day composite)
        tick_size: Price bin size
        va_percent: VA percentage

    Returns:
        ValueAreaLevels from merged volume profile, or None
    """
    sessions = sorted(df['session_date'].unique())
    session_list = [s for s in sessions if s < session_date]

    if len(session_list) < lookback_days:
        return None

    # Take last N sessions
    target_sessions = session_list[-lookback_days:]
    subset = df[df['session_date'].isin(target_sessions)]

    if len(subset) < 30:
        return None

    highs = subset['high'].values
    lows = subset['low'].values
    volumes = subset['volume'].values

    return calculate_value_area(highs, lows, volumes, tick_size, va_percent)


def add_composite_va_features(
    df: pd.DataFrame,
    lookback_days: int = 3,
    tick_size: float = 0.25,
) -> pd.DataFrame:
    """
    Add composite VA (N-day) features alongside single-day VA.

    Adds columns:
      - comp_va_poc, comp_va_vah, comp_va_val, comp_va_width
      - open_vs_comp_va: 'ABOVE_VAH', 'BELOW_VAL', or 'INSIDE_VA'
    """
    df = df.copy()
    prefix = f'comp{lookback_days}'

    for col in [f'{prefix}_va_poc', f'{prefix}_va_vah', f'{prefix}_va_val',
                f'{prefix}_va_width']:
        df[col] = np.nan
    df[f'open_vs_{prefix}_va'] = None

    sessions = sorted(df['session_date'].unique())

    for session_date in sessions:
        comp_va = compute_composite_value_area(
            df, session_date, lookback_days, tick_size,
        )
        if comp_va is None:
            continue

        session_idx = df[df['session_date'] == session_date].index
        df.loc[session_idx, f'{prefix}_va_poc'] = comp_va.poc
        df.loc[session_idx, f'{prefix}_va_vah'] = comp_va.vah
        df.loc[session_idx, f'{prefix}_va_val'] = comp_va.val
        df.loc[session_idx, f'{prefix}_va_width'] = comp_va.va_width

        session_df = df.loc[session_idx]
        if len(session_df) > 0:
            open_price = session_df['open'].iloc[0]
            if open_price > comp_va.vah:
                df.loc[session_idx, f'open_vs_{prefix}_va'] = 'ABOVE_VAH'
            elif open_price < comp_va.val:
                df.loc[session_idx, f'open_vs_{prefix}_va'] = 'BELOW_VAL'
            else:
                df.loc[session_idx, f'open_vs_{prefix}_va'] = 'INSIDE_VA'

    return df
