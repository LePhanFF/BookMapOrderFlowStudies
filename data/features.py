"""
Order flow feature engineering.
Computes delta, CVD, imbalance, z-scores, and percentiles from volumetric data.
"""

import pandas as pd
import numpy as np
from config.constants import (
    DELTA_ROLLING_WINDOW,
    VOLUME_ROLLING_WINDOW,
    CVD_EMA_SPAN,
    IB_BARS_1MIN,
    DAY_TYPE_TREND_THRESHOLD,
    DAY_TYPE_P_DAY_THRESHOLD,
    DAY_TYPE_B_DAY_THRESHOLD,
)


def compute_order_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute order flow features from volumetric data.

    Requires columns: vol_ask, vol_bid, volume
    Produces: delta, delta_pct, delta_zscore, delta_percentile,
              cumulative_delta, cumulative_delta_ma, imbalance_ratio,
              volume_ma, volume_spike
    """
    df = df.copy()
    window = DELTA_ROLLING_WINDOW

    # Delta features
    df['delta'] = df['vol_ask'] - df['vol_bid']
    df['delta_pct'] = df['delta'] / df['volume'].replace(0, np.nan)

    # Rolling delta statistics
    df['delta_rolling_mean'] = df['delta'].rolling(window, min_periods=1).mean()
    df['delta_rolling_std'] = df['delta'].rolling(window, min_periods=1).std()

    # Z-score (adaptive threshold)
    df['delta_zscore'] = (
        (df['delta'] - df['delta_rolling_mean'])
        / df['delta_rolling_std'].replace(0, np.nan)
    )

    # Delta percentile (last N bars)
    df['delta_percentile'] = df['delta'].rolling(window, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )

    # Cumulative Delta
    df['cumulative_delta'] = df['delta'].cumsum()
    df['cumulative_delta_ma'] = df['cumulative_delta'].ewm(span=CVD_EMA_SPAN).mean()

    # Imbalance ratio
    df['imbalance_ratio'] = df['vol_ask'] / df['vol_bid'].replace(0, np.nan)

    # Volume spike detection
    df['volume_ma'] = df['volume'].rolling(VOLUME_ROLLING_WINDOW, min_periods=1).mean()
    df['volume_spike'] = df['volume'] / df['volume_ma']

    return df


def compute_ib_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Initial Balance features per session.

    For each session, calculates IB high/low/range from the first 60 1-min bars,
    then computes extension and direction for every bar in the session.
    """
    df = df.copy()

    # Initialize columns
    df['ib_high'] = np.nan
    df['ib_low'] = np.nan
    df['ib_range'] = np.nan
    df['ib_extension'] = 0.0
    df['ib_direction'] = 'INSIDE'

    for session, session_df in df.groupby('session_date'):
        if len(session_df) < IB_BARS_1MIN:
            continue

        ib_data = session_df.head(IB_BARS_1MIN)
        ib_high = ib_data['high'].max()
        ib_low = ib_data['low'].min()
        ib_range = ib_high - ib_low

        session_idx = session_df.index

        # Vectorized assignment for IB levels
        df.loc[session_idx, 'ib_high'] = ib_high
        df.loc[session_idx, 'ib_low'] = ib_low
        df.loc[session_idx, 'ib_range'] = ib_range

        # Compute extension and direction per bar
        closes = df.loc[session_idx, 'close']
        above_ib = closes > ib_high
        below_ib = closes < ib_low

        df.loc[session_idx[above_ib], 'ib_extension'] = closes[above_ib] - ib_high
        df.loc[session_idx[above_ib], 'ib_direction'] = 'BULL'

        df.loc[session_idx[below_ib], 'ib_extension'] = ib_low - closes[below_ib]
        df.loc[session_idx[below_ib], 'ib_direction'] = 'BEAR'

        inside = ~above_ib & ~below_ib
        df.loc[session_idx[inside], 'ib_extension'] = 0.0
        df.loc[session_idx[inside], 'ib_direction'] = 'INSIDE'

    # Forward fill for any remaining NaN
    df['ib_high'] = df['ib_high'].ffill()
    df['ib_low'] = df['ib_low'].ffill()
    df['ib_range'] = df['ib_range'].ffill()

    return df


def compute_day_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each session's day type based on IB extension.

    Uses session-end price relative to IB midpoint.
    Produces: day_type, day_direction, ib_extension (session-level columns merged).
    """
    day_types = []

    for session, session_df in df.groupby('session_date'):
        if len(session_df) < IB_BARS_1MIN:
            day_types.append({
                'session_date': session,
                'day_type': 'NEUTRAL',
                'day_direction': 'INSIDE',
            })
            continue

        ib_data = session_df.head(IB_BARS_1MIN)
        ib_high = ib_data['high'].max()
        ib_low = ib_data['low'].min()
        ib_range = ib_high - ib_low

        # End-of-session price
        current_price = session_df['close'].iloc[-1]
        ib_mid = (ib_high + ib_low) / 2

        if current_price > ib_mid:
            extension = (current_price - ib_mid) / ib_range if ib_range > 0 else 0
            direction = 'BULL'
        else:
            extension = (ib_mid - current_price) / ib_range if ib_range > 0 else 0
            direction = 'BEAR'

        if extension > DAY_TYPE_TREND_THRESHOLD:
            day_type = 'TREND'
        elif extension > DAY_TYPE_P_DAY_THRESHOLD:
            day_type = 'P_DAY'
        elif extension < DAY_TYPE_B_DAY_THRESHOLD:
            day_type = 'B_DAY'
        else:
            day_type = 'NEUTRAL'

        day_types.append({
            'session_date': session,
            'day_type': day_type,
            'day_direction': direction,
        })

    day_type_df = pd.DataFrame(day_types)
    df = df.merge(day_type_df, on='session_date', how='left', suffixes=('', '_dtype'))
    return df


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features: order flow, IB, day type, ICT models, and IB width."""
    print("Computing features...")

    df = compute_order_flow_features(df)
    print("  Order flow features computed")

    df = compute_ib_features(df)
    print("  IB features computed")

    df = compute_day_type(df)
    print("  Day types computed")

    # ICT entry model features (FVG, IFVG, BPR, 15-min FVG)
    from indicators.ict_models import add_ict_features
    df = add_ict_features(df)

    # IB Width classification + C-period rule + extension targets
    from indicators.ib_width import compute_ib_width_features
    df = compute_ib_width_features(df)
    print("  IB width + C-period features computed")

    # Prior day Value Area (for 80% Rule / 80P strategy)
    from indicators.value_area import add_prior_va_features
    df = add_prior_va_features(df)
    print("  Prior day Value Area features computed")

    return df
