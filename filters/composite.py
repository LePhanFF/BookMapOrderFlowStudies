"""
Composite filter: chains multiple filters. Signal must pass ALL.
"""

from typing import List
import pandas as pd
from filters.base import FilterBase
from strategy.signal import Signal


class CompositeFilter(FilterBase):
    """Chains multiple filters. Signal must pass ALL filters to be executed."""

    def __init__(self, filters: List[FilterBase]):
        self._filters = filters

    @property
    def name(self) -> str:
        names = [f.name for f in self._filters]
        return f"Composite({', '.join(names)})"

    def should_trade(self, signal: Signal, bar: pd.Series, session_context: dict) -> bool:
        for f in self._filters:
            if not f.should_trade(signal, bar, session_context):
                return False
        return True
