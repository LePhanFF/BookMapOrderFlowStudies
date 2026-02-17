"""
Abstract base class for all trading strategies.

DESIGN PRINCIPLE: Strategies EMIT SIGNALS, they do NOT manage positions.
The backtest engine handles all position management, execution, and risk control.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd
from strategy.signal import Signal


class StrategyBase(ABC):
    """
    Base class for Dalton playbook strategies.

    Lifecycle:
      1. on_session_start() -- called once after IB formation
      2. on_bar() -- called for each bar after IB, return Signal or None
      3. on_session_end() -- called at session close for cleanup

    Strategies maintain per-session state (acceptance tracking, pyramid counts, etc.)
    which is reset in on_session_start().
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for reporting."""

    @property
    @abstractmethod
    def applicable_day_types(self) -> List[str]:
        """Day type values this strategy trades. Empty = all day types."""

    @abstractmethod
    def on_session_start(
        self,
        session_date,
        ib_high: float,
        ib_low: float,
        ib_range: float,
        session_context: dict,
    ) -> None:
        """
        Called once per session after IB formation.

        session_context contains:
          - ib_high, ib_low, ib_range, ib_mid
          - day_type (str), trend_strength (str)
          - vwap, ema20, ema50, atr14
          - dpoc_data (dict from rockit dpoc_migration)
          - tpo_data (dict from rockit tpo_profile)
          - vp_data (dict from rockit volume_profile)
          - confluences (dict from rockit core_confluences)
          - prior_day (dict with prior session levels)
        """

    @abstractmethod
    def on_bar(
        self,
        bar: pd.Series,
        bar_index: int,
        session_context: dict,
    ) -> Optional[Signal]:
        """
        Called for each bar after IB. Return Signal or None.

        The bar Series contains: open, high, low, close, volume,
        vol_ask, vol_bid, vol_delta, plus computed features.

        bar_index is 0-based from start of post-IB data.
        """

    def on_session_end(self, session_date) -> None:
        """Optional cleanup at session end. Override if needed."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
