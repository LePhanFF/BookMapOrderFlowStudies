"""
Abstract base class for signal filters.
Filters accept or reject signals before execution.
"""

from abc import ABC, abstractmethod
import pandas as pd
from strategy.signal import Signal


class FilterBase(ABC):
    """Composable filter that can accept/reject a trading signal."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Filter name for logging."""

    @abstractmethod
    def should_trade(self, signal: Signal, bar: pd.Series, session_context: dict) -> bool:
        """Return True if the signal should be executed."""
