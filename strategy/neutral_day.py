"""
Strategy 7: Neutral Day (Symmetric Balance - Chop Risk)

Dalton Playbook Rules:
  - Micro/flat only - rarely 3/3 Lanto
  - Symmetric fattening center, multiple HVN
  - Rotational probes repaired, flat DPOC, low ATR
  - Bias: Neutral - only edge fades
  - Explicit chop risk. No directional conviction.

Currently: No trades generated (no mechanical edge).
The playbook explicitly warns this is the most dangerous day type.
"""

from typing import Optional, List
import pandas as pd

from strategy.base import StrategyBase
from strategy.signal import Signal


class NeutralDayStrategy(StrategyBase):

    @property
    def name(self) -> str:
        return "Neutral Day"

    @property
    def applicable_day_types(self) -> List[str]:
        return ['neutral']

    def on_session_start(self, session_date, ib_high, ib_low, ib_range, session_context):
        pass  # No setup needed

    def on_bar(self, bar: pd.Series, bar_index: int, session_context: dict) -> Optional[Signal]:
        # Playbook: "No directional conviction. Explicit chop risk."
        # Stand down -- no mechanical edge on neutral days.
        return None
