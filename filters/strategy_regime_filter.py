"""
Per-strategy IB regime filter.

Gates signals based on the session's actual IB range and which regime
each strategy performs best in. Validated against 259-session backtest
(diag_when_to_avoid.py TEST 7).

IB Regime Buckets (NQ/MNQ-specific):
  LOW:    IB < 100 pts
  MED:    100 <= IB < 150 pts
  NORMAL: 150 <= IB < 250 pts
  HIGH:   IB >= 250 pts
"""

import pandas as pd
from filters.base import FilterBase
from strategy.signal import Signal


# Per-strategy allowed IB regimes (whitelist).
# Strategies not listed here pass through unfiltered.
# Based on 259-session analysis: skip regimes where strategy loses money.
STRATEGY_ALLOWED_REGIMES = {
    'B-Day':             ['med', 'high'],      # Loses in LOW and NORMAL
    'Edge Fade':         ['normal', 'high'],    # Loses in MED
    'Opening Range Rev': ['med', 'normal', 'high'],  # Loses in LOW
    'Bear Accept Short': ['high'],              # Only barely profitable in HIGH
    'Trend Day Bull':    [],                    # Loses in all regimes
    'P-Day':             [],                    # Loses in all regimes
    'IBH Sweep+Fail':    [],                    # Loses in all regimes (1 trade)
}

# IB range breakpoints (NQ/MNQ points)
_IB_THRESHOLDS = [100, 150, 250]
_REGIME_LABELS = ['low', 'med', 'normal', 'high']


def _classify_ib_regime(ib_range: float) -> str:
    """Classify session IB range into regime bucket."""
    if ib_range < _IB_THRESHOLDS[0]:
        return 'low'
    elif ib_range < _IB_THRESHOLDS[1]:
        return 'med'
    elif ib_range < _IB_THRESHOLDS[2]:
        return 'normal'
    else:
        return 'high'


class StrategyRegimeFilter(FilterBase):
    """
    Filter signals based on per-strategy IB regime whitelist.

    Uses the session's actual IB range (not rolling median) to classify
    the regime, then checks if the strategy is allowed to trade in it.

    Pass custom_regimes dict to override defaults.
    """

    def __init__(self, custom_regimes: dict = None):
        self._regimes = custom_regimes or STRATEGY_ALLOWED_REGIMES

    @property
    def name(self) -> str:
        return "StrategyRegime"

    def should_trade(self, signal: Signal, bar: pd.Series, session_context: dict) -> bool:
        ib_range = session_context.get('ib_range')
        if ib_range is None:
            return True  # No IB data yet (warmup), allow

        allowed = self._regimes.get(signal.strategy_name)
        if allowed is None:
            return True  # Strategy not in whitelist, no filtering

        if len(allowed) == 0:
            return False  # Strategy disabled in all regimes

        regime = _classify_ib_regime(ib_range)
        return regime in allowed
