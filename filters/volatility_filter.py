"""
Volatility filter.
Gates signals based on ATR or IB range thresholds.
"""

import pandas as pd
from filters.base import FilterBase
from strategy.signal import Signal


class VolatilityFilter(FilterBase):
    """Only trade when ATR is within acceptable range."""

    def __init__(self, min_atr: float = 5.0, max_atr: float = 100.0):
        self.min_atr = min_atr
        self.max_atr = max_atr

    @property
    def name(self) -> str:
        return f"Volatility(ATR {self.min_atr}-{self.max_atr})"

    def should_trade(self, signal: Signal, bar: pd.Series, session_context: dict) -> bool:
        atr = session_context.get('atr14', bar.get('atr14', 0))
        if atr is None or pd.isna(atr):
            return True
        return self.min_atr <= atr <= self.max_atr


class IBRangeFilter(FilterBase):
    """Only trade when IB range is within acceptable bounds."""

    def __init__(self, min_range: float = 20.0, max_range: float = 200.0):
        self.min_range = min_range
        self.max_range = max_range

    @property
    def name(self) -> str:
        return f"IBRange({self.min_range}-{self.max_range})"

    def should_trade(self, signal: Signal, bar: pd.Series, session_context: dict) -> bool:
        ib_range = session_context.get('ib_range', 0)
        return self.min_range <= ib_range <= self.max_range
