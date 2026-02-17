"""
Order flow filter.
Gates signals based on delta, CVD, imbalance, and wick parade.
"""

import pandas as pd
from filters.base import FilterBase
from strategy.signal import Signal
from config.constants import DELTA_PERCENTILE_THRESHOLD


class DeltaFilter(FilterBase):
    """Require delta percentile above threshold and aligned with direction."""

    def __init__(self, min_percentile: float = DELTA_PERCENTILE_THRESHOLD):
        self.min_percentile = min_percentile

    @property
    def name(self) -> str:
        return f"Delta(>{self.min_percentile}pct)"

    def should_trade(self, signal: Signal, bar: pd.Series, session_context: dict) -> bool:
        delta_pct = bar.get('delta_percentile', 0)
        delta = bar.get('delta', 0)
        if delta_pct is None or pd.isna(delta_pct):
            return True

        if delta_pct < self.min_percentile:
            return False

        # Delta must align with signal direction
        if signal.direction == 'LONG' and delta < 0:
            return False
        if signal.direction == 'SHORT' and delta > 0:
            return False

        return True


class CVDFilter(FilterBase):
    """Require CVD trend alignment with signal direction."""

    @property
    def name(self) -> str:
        return "CVD"

    def should_trade(self, signal: Signal, bar: pd.Series, session_context: dict) -> bool:
        cvd = bar.get('cumulative_delta')
        cvd_ma = bar.get('cumulative_delta_ma')
        if cvd is None or cvd_ma is None:
            return True
        if pd.isna(cvd) or pd.isna(cvd_ma):
            return True

        if signal.direction == 'LONG':
            return cvd > cvd_ma
        else:
            return cvd < cvd_ma


class VolumeFilter(FilterBase):
    """Require volume spike on entry."""

    def __init__(self, min_spike: float = 1.2):
        self.min_spike = min_spike

    @property
    def name(self) -> str:
        return f"Volume(>{self.min_spike}x)"

    def should_trade(self, signal: Signal, bar: pd.Series, session_context: dict) -> bool:
        spike = bar.get('volume_spike', 1.0)
        if spike is None or pd.isna(spike):
            return True
        return spike >= self.min_spike
