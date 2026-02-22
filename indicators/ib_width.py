"""
IB Width Classification + C-Period Close Location Rule
=======================================================

Classifies Initial Balance width as NARROW, NORMAL, or WIDE based on
the ratio of IB range to ATR(14). This classification predicts day type
and guides strategy selection:

  NARROW IB (< 33% ATR): Expect trend day, trade breakouts aggressively
  NORMAL IB (33-66% ATR): Standard playbook
  WIDE IB (> 66% ATR):   Fade IB extremes, range-reversion

C-Period Close Location Rule (70-75% edge):
  - If C-period (10:30-11:00) closes OUTSIDE IB → 70-75% continuation
  - If C-period closes back INSIDE IB → 70-75% reversal to opposite extreme

Also computes:
  - A-period (9:30-10:00) and B-period (10:00-10:30) sub-ranges
  - NR4/NR7 (Narrow Range) detection for compressed IB days
  - IB extension targets (1x, 1.5x, 2x, 3x)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import time


@dataclass
class IBWidthAnalysis:
    """Results of IB width classification."""
    ib_high: float
    ib_low: float
    ib_range: float
    ib_mid: float

    # Sub-period ranges
    a_period_high: float
    a_period_low: float
    a_period_range: float
    b_period_high: float
    b_period_low: float
    b_period_range: float

    # Width classification
    ib_atr_ratio: float
    width_class: str  # 'NARROW', 'NORMAL', 'WIDE'

    # NR4/NR7
    is_nr4: bool
    is_nr7: bool

    # Extension targets
    ext_1_0_high: float
    ext_1_0_low: float
    ext_1_5_high: float
    ext_1_5_low: float
    ext_2_0_high: float
    ext_2_0_low: float
    ext_3_0_high: float
    ext_3_0_low: float


@dataclass
class CPeriodResult:
    """C-period close location analysis."""
    c_period_high: float
    c_period_low: float
    c_period_close: float
    closes_above_ib: bool
    closes_below_ib: bool
    closes_inside_ib: bool
    continuation_bias: Optional[str]  # 'BULL', 'BEAR', or None


def classify_ib_width(ib_range: float, atr: float) -> str:
    """Classify IB width relative to ATR."""
    if atr <= 0:
        return 'NORMAL'
    ratio = ib_range / atr
    if ratio < 0.33:
        return 'NARROW'
    elif ratio < 0.66:
        return 'NORMAL'
    else:
        return 'WIDE'


def analyze_ib_width(
    session_df: pd.DataFrame,
    atr: float,
    prior_ib_ranges: Optional[List[float]] = None,
) -> IBWidthAnalysis:
    """
    Perform full IB width analysis on a session's data.

    Args:
        session_df: DataFrame for a single session (1-min bars)
        atr: ATR(14) value at start of session
        prior_ib_ranges: List of prior sessions' IB ranges for NR4/NR7
    """
    # A-period: first 30 bars (9:30-10:00)
    a_period = session_df.head(30)
    a_high = a_period['high'].max()
    a_low = a_period['low'].min()
    a_range = a_high - a_low

    # B-period: bars 31-60 (10:00-10:30)
    b_period = session_df.iloc[30:60]
    if len(b_period) > 0:
        b_high = b_period['high'].max()
        b_low = b_period['low'].min()
        b_range = b_high - b_low
    else:
        b_high = a_high
        b_low = a_low
        b_range = 0

    # Full IB
    ib_data = session_df.head(60)
    ib_high = ib_data['high'].max()
    ib_low = ib_data['low'].min()
    ib_range = ib_high - ib_low
    ib_mid = (ib_high + ib_low) / 2

    # Width classification
    ib_atr_ratio = ib_range / atr if atr > 0 else 1.0
    width_class = classify_ib_width(ib_range, atr)

    # NR4/NR7 detection
    is_nr4 = False
    is_nr7 = False
    if prior_ib_ranges:
        if len(prior_ib_ranges) >= 3:
            is_nr4 = ib_range <= min(prior_ib_ranges[-3:])
        if len(prior_ib_ranges) >= 6:
            is_nr7 = ib_range <= min(prior_ib_ranges[-6:])

    # Extension targets
    return IBWidthAnalysis(
        ib_high=ib_high,
        ib_low=ib_low,
        ib_range=ib_range,
        ib_mid=ib_mid,
        a_period_high=a_high,
        a_period_low=a_low,
        a_period_range=a_range,
        b_period_high=b_high,
        b_period_low=b_low,
        b_period_range=b_range,
        ib_atr_ratio=ib_atr_ratio,
        width_class=width_class,
        is_nr4=is_nr4,
        is_nr7=is_nr7,
        ext_1_0_high=ib_high + ib_range,
        ext_1_0_low=ib_low - ib_range,
        ext_1_5_high=ib_high + 1.5 * ib_range,
        ext_1_5_low=ib_low - 1.5 * ib_range,
        ext_2_0_high=ib_high + 2.0 * ib_range,
        ext_2_0_low=ib_low - 2.0 * ib_range,
        ext_3_0_high=ib_high + 3.0 * ib_range,
        ext_3_0_low=ib_low - 3.0 * ib_range,
    )


def analyze_c_period(
    session_df: pd.DataFrame,
    ib_high: float,
    ib_low: float,
) -> Optional[CPeriodResult]:
    """
    Analyze C-period (10:30-11:00) close location relative to IB.

    The C-period close location rule:
      - C-period closes above IBH → 70-75% probability of continuation up
      - C-period closes below IBL → 70-75% probability of continuation down
      - C-period closes inside IB → 70-75% probability of reversal to opposite extreme
    """
    # C-period: bars 61-90 (10:30-11:00)
    c_period = session_df.iloc[60:90]

    if len(c_period) == 0:
        return None

    c_high = c_period['high'].max()
    c_low = c_period['low'].min()
    c_close = c_period['close'].iloc[-1]

    closes_above = c_close > ib_high
    closes_below = c_close < ib_low
    closes_inside = not closes_above and not closes_below

    if closes_above:
        continuation_bias = 'BULL'
    elif closes_below:
        continuation_bias = 'BEAR'
    else:
        continuation_bias = None  # No clear continuation signal

    return CPeriodResult(
        c_period_high=c_high,
        c_period_low=c_low,
        c_period_close=c_close,
        closes_above_ib=closes_above,
        closes_below_ib=closes_below,
        closes_inside_ib=closes_inside,
        continuation_bias=continuation_bias,
    )


def compute_ib_width_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute IB width features for all sessions.

    Adds columns:
      - ib_width_class: 'NARROW', 'NORMAL', 'WIDE'
      - ib_atr_ratio: IB range / ATR(14)
      - a_period_high/low/range
      - b_period_high/low/range
      - c_period_bias: 'BULL', 'BEAR', or None
      - is_nr4, is_nr7
      - Extension targets: ext_1_0_high/low, ext_1_5_high/low, etc.
    """
    df = df.copy()

    # Initialize columns
    new_cols = [
        'ib_width_class', 'ib_atr_ratio',
        'a_period_high', 'a_period_low', 'a_period_range',
        'b_period_high', 'b_period_low', 'b_period_range',
        'c_period_bias', 'is_nr4', 'is_nr7',
        'ext_1_0_high', 'ext_1_0_low',
        'ext_1_5_high', 'ext_1_5_low',
        'ext_2_0_high', 'ext_2_0_low',
        'ext_3_0_high', 'ext_3_0_low',
    ]
    for col in new_cols:
        df[col] = np.nan if col not in ['ib_width_class', 'c_period_bias', 'is_nr4', 'is_nr7'] else None

    df['is_nr4'] = False
    df['is_nr7'] = False

    prior_ib_ranges = []

    for session_date in sorted(df['session_date'].unique()):
        session_df = df[df['session_date'] == session_date]
        session_idx = session_df.index

        if len(session_df) < 60:
            continue

        # Get ATR
        atr = session_df.iloc[0].get('atr14', 0)
        if pd.isna(atr):
            atr = 0

        # Analyze IB width
        ib_analysis = analyze_ib_width(session_df, atr, prior_ib_ranges)

        # Assign to all bars in session
        df.loc[session_idx, 'ib_width_class'] = ib_analysis.width_class
        df.loc[session_idx, 'ib_atr_ratio'] = ib_analysis.ib_atr_ratio
        df.loc[session_idx, 'a_period_high'] = ib_analysis.a_period_high
        df.loc[session_idx, 'a_period_low'] = ib_analysis.a_period_low
        df.loc[session_idx, 'a_period_range'] = ib_analysis.a_period_range
        df.loc[session_idx, 'b_period_high'] = ib_analysis.b_period_high
        df.loc[session_idx, 'b_period_low'] = ib_analysis.b_period_low
        df.loc[session_idx, 'b_period_range'] = ib_analysis.b_period_range
        df.loc[session_idx, 'is_nr4'] = ib_analysis.is_nr4
        df.loc[session_idx, 'is_nr7'] = ib_analysis.is_nr7
        df.loc[session_idx, 'ext_1_0_high'] = ib_analysis.ext_1_0_high
        df.loc[session_idx, 'ext_1_0_low'] = ib_analysis.ext_1_0_low
        df.loc[session_idx, 'ext_1_5_high'] = ib_analysis.ext_1_5_high
        df.loc[session_idx, 'ext_1_5_low'] = ib_analysis.ext_1_5_low
        df.loc[session_idx, 'ext_2_0_high'] = ib_analysis.ext_2_0_high
        df.loc[session_idx, 'ext_2_0_low'] = ib_analysis.ext_2_0_low
        df.loc[session_idx, 'ext_3_0_high'] = ib_analysis.ext_3_0_high
        df.loc[session_idx, 'ext_3_0_low'] = ib_analysis.ext_3_0_low

        # Analyze C-period
        c_result = analyze_c_period(session_df, ib_analysis.ib_high, ib_analysis.ib_low)
        if c_result:
            df.loc[session_idx, 'c_period_bias'] = c_result.continuation_bias

        # Track for NR4/NR7
        prior_ib_ranges.append(ib_analysis.ib_range)

    return df
