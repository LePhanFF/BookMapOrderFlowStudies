"""
Time-of-day filter.
Restricts trading to specified time windows.
"""

from datetime import time
import pandas as pd
from filters.base import FilterBase
from strategy.signal import Signal
from config.constants import IB_END, EOD_CUTOFF


class TimeFilter(FilterBase):
    """Only allow signals within a time window."""

    def __init__(self, start: time = IB_END, end: time = EOD_CUTOFF):
        self.start = start
        self.end = end

    @property
    def name(self) -> str:
        return f"Time({self.start.strftime('%H:%M')}-{self.end.strftime('%H:%M')})"

    def should_trade(self, signal: Signal, bar: pd.Series, session_context: dict) -> bool:
        bar_time = session_context.get('bar_time')
        if bar_time is None:
            return True
        return self.start <= bar_time <= self.end


class LunchFadeFilter(FilterBase):
    """
    Reduce confidence during London close / lunch lull (12:15-12:45).
    From playbook: 'Be careful of lunch fade after London closes 12:15 to 12:45 PM'
    """

    def __init__(self, start: time = time(12, 15), end: time = time(12, 45)):
        self.start = start
        self.end = end

    @property
    def name(self) -> str:
        return "LunchFade"

    def should_trade(self, signal: Signal, bar: pd.Series, session_context: dict) -> bool:
        bar_time = session_context.get('bar_time')
        if bar_time is None:
            return True
        # Block new entries during lunch lull
        if self.start <= bar_time <= self.end:
            return False
        return True
