"""
Trend filter.
Requires ADX/EMA alignment for trend strategies.
"""

import pandas as pd
from filters.base import FilterBase
from strategy.signal import Signal


class TrendFilter(FilterBase):
    """Require ADX above threshold for trend strategies."""

    def __init__(self, min_adx: float = 20.0):
        self.min_adx = min_adx

    @property
    def name(self) -> str:
        return f"Trend(ADX>{self.min_adx})"

    def should_trade(self, signal: Signal, bar: pd.Series, session_context: dict) -> bool:
        adx = bar.get('adx14', session_context.get('adx14', 0))
        if adx is None or pd.isna(adx):
            return True
        return adx >= self.min_adx


class EMAAlignmentFilter(FilterBase):
    """Require price and EMA alignment with signal direction."""

    @property
    def name(self) -> str:
        return "EMAAlignment"

    def should_trade(self, signal: Signal, bar: pd.Series, session_context: dict) -> bool:
        ema20 = bar.get('ema20')
        ema50 = bar.get('ema50')
        if ema20 is None or ema50 is None:
            return True
        if pd.isna(ema20) or pd.isna(ema50):
            return True

        if signal.direction == 'LONG':
            return bar['close'] > ema20 and ema20 > ema50
        else:
            return bar['close'] < ema20 and ema20 < ema50
