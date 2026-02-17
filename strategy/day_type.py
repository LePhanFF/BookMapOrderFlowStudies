"""
Day type and trend strength classification.
Single source of truth - replaces 4 separate implementations.
"""

from enum import Enum
from config.constants import (
    IB_EXT_WEAK, IB_EXT_MODERATE, IB_EXT_STRONG,
    DAY_TYPE_TREND_THRESHOLD, DAY_TYPE_P_DAY_THRESHOLD,
    DAY_TYPE_B_DAY_THRESHOLD,
)


class DayType(Enum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    SUPER_TREND_UP = "super_trend_up"
    SUPER_TREND_DOWN = "super_trend_down"
    P_DAY = "p_day"
    B_DAY = "b_day"
    NEUTRAL = "neutral"
    PM_MORPH = "pm_morph"
    MORPH_TO_TREND = "morph_to_trend"


class TrendStrength(Enum):
    WEAK = "weak"           # < 0.5x IB extension
    MODERATE = "moderate"   # 0.5-1.0x IB extension
    STRONG = "strong"       # 1.0-2.0x IB extension
    SUPER = "super"         # > 2.0x IB extension


def classify_trend_strength(ib_extension_multiple: float) -> TrendStrength:
    """
    Classify trend strength from IB extension multiple.

    Dalton strength gates:
      Weak:     < 0.5x - hovering/poke behavior
      Moderate: 0.5-1.0x - full bracket hold
      Strong:   1.0-2.0x - stacked brackets, sustained fattening
      Super:    > 2.0x - extreme DPOC migration/compression
    """
    if ib_extension_multiple < IB_EXT_WEAK:
        return TrendStrength.WEAK
    elif ib_extension_multiple < IB_EXT_MODERATE:
        return TrendStrength.MODERATE
    elif ib_extension_multiple < IB_EXT_STRONG:
        return TrendStrength.STRONG
    else:
        return TrendStrength.SUPER


def classify_day_type(
    ib_high: float,
    ib_low: float,
    current_price: float,
    ib_direction: str = 'INSIDE',
    trend_strength: TrendStrength = TrendStrength.WEAK,
) -> DayType:
    """
    Classify current bar's day type based on IB extension and trend strength.

    Uses the Dalton framework:
      - SUPER TREND: extension > 2.0x, strength super
      - TREND (Standard): extension > 1.0x, strength moderate or strong
      - P_DAY: extension 0.5-1.0x (skewed balance)
      - B_DAY: price inside IB (ib_direction == INSIDE) with weak strength
      - NEUTRAL: everything else
    """
    ib_range = ib_high - ib_low
    if ib_range <= 0:
        return DayType.NEUTRAL

    ib_mid = (ib_high + ib_low) / 2

    if current_price > ib_mid:
        extension = (current_price - ib_mid) / ib_range
        direction = 'BULL'
    else:
        extension = (ib_mid - current_price) / ib_range
        direction = 'BEAR'

    # Super trend check first (requires super strength)
    if trend_strength == TrendStrength.SUPER:
        if direction == 'BULL':
            return DayType.SUPER_TREND_UP
        else:
            return DayType.SUPER_TREND_DOWN

    # Standard trend
    if extension > DAY_TYPE_TREND_THRESHOLD:
        if direction == 'BULL':
            return DayType.TREND_UP
        else:
            return DayType.TREND_DOWN

    # P-Day (skewed balance)
    if extension > DAY_TYPE_P_DAY_THRESHOLD:
        return DayType.P_DAY

    # B-Day: price is inside IB (narrow, no extension)
    # Dalton: "Narrow IB, price never leaves first hour, Strength: Weak"
    if ib_direction == 'INSIDE' and trend_strength == TrendStrength.WEAK:
        return DayType.B_DAY

    # Also B-Day if extension is very small (even with slight directional lean)
    if extension < DAY_TYPE_B_DAY_THRESHOLD:
        return DayType.B_DAY

    return DayType.NEUTRAL


def classify_session_day_type(
    ib_high: float,
    ib_low: float,
    session_close: float,
    ib_direction: str = 'INSIDE',
) -> tuple:
    """
    Classify a completed session's day type and trend strength.
    Used for post-session analysis (not real-time).

    Returns: (DayType, TrendStrength)
    """
    ib_range = ib_high - ib_low
    if ib_range <= 0:
        return DayType.NEUTRAL, TrendStrength.WEAK

    ib_mid = (ib_high + ib_low) / 2

    if session_close > ib_mid:
        extension = (session_close - ib_mid) / ib_range
    else:
        extension = (ib_mid - session_close) / ib_range

    strength = classify_trend_strength(extension)
    day_type = classify_day_type(ib_high, ib_low, session_close, ib_direction, strength)

    return day_type, strength
