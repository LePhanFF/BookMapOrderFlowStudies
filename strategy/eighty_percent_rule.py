"""
Strategy: Dalton 80% Rule (alias: 80P)
========================================

James Dalton's Inside-Value Logic:
  When price opens OUTSIDE the prior day's Value Area (above VAH or below VAL)
  but then trades back INTO the VA and is ACCEPTED there (2 consecutive 30-min
  TPO periods closing inside VA), there is a high probability that price will
  auction through to the OTHER side of the Value Area.

  Backtested win rate: ~60-62% (not 80% as the name implies).
  The "80%" refers to the probability of price reaching the other VA edge,
  but slippage, timing, and stops reduce effective WR.

ENTRY MODEL:
  1. Prior day's VA computed using CBOT POC expansion method
  2. Open outside VA (above VAH or below VAL)
  3. 2 consecutive 30-min bars close inside VA = acceptance confirmed
  4. Enter LONG (opened below VAL) or SHORT (opened above VAH)

EXIT MODEL (optimizable via backtest runner):
  - Stop: Beyond VA entry boundary + buffer
  - Target: Opposite VA edge (default), POC (conservative), or multi-R
  - BE trail: Move to breakeven once POC breaks
  - Aggressive trail: Trail by FVG after POC cleared
  - Time exit: 15:30 ET

FILTERS:
  - VA width >= 25 pts minimum
  - No max VA width cap (user targets wide VA days: 200-400pt)
  - No entry after 13:00 ET
"""

from datetime import time
from typing import Optional, List
import pandas as pd
import numpy as np

from strategy.base import StrategyBase
from strategy.signal import Signal


# --- Parameters ---
MIN_VA_WIDTH = 25.0             # Minimum VA width for meaningful R:R
STOP_BUFFER_PTS = 10.0          # Stop buffer beyond VA boundary
ACCEPTANCE_PERIODS = 2          # Consecutive 30-min closes inside VA
PERIOD_LENGTH_BARS = 30         # 30 bars = 30 min of 1-min data
ENTRY_CUTOFF = time(13, 0)      # No new entries after 1 PM ET
MAX_ENTRIES_PER_SESSION = 1     # Only one 80P trade per session


class EightyPercentRule(StrategyBase):
    """
    Dalton 80% Rule (80P) — mean reversion to value after failed breakout.

    Monitors for price to open outside prior day's Value Area, then
    re-enter and be accepted (2 consecutive 30-min closes inside VA).
    Targets the opposite VA boundary.
    """

    def __init__(self, target_mode: str = 'opposite_va', min_rr: float = 1.0):
        """
        Args:
            target_mode: 'opposite_va' (full traverse), 'poc' (midpoint),
                         'half_va' (halfway to opposite edge)
            min_rr: Minimum R:R ratio required to take the trade
        """
        self._target_mode = target_mode
        self._min_rr = min_rr

    @property
    def name(self) -> str:
        return "80P Rule"

    @property
    def applicable_day_types(self) -> List[str]:
        return []  # Can fire on any day type

    def on_session_start(self, session_date, ib_high, ib_low, ib_range, session_context):
        self._ib_high = ib_high
        self._ib_low = ib_low
        self._ib_range = ib_range

        # Prior day VA levels (from data pipeline)
        self._prior_vah = session_context.get('prior_va_vah', None)
        self._prior_val = session_context.get('prior_va_val', None)
        self._prior_poc = session_context.get('prior_va_poc', None)
        self._prior_va_width = session_context.get('prior_va_width', None)
        self._open_vs_va = session_context.get('open_vs_va', None)

        # State
        self._setup_armed = False
        self._setup_direction = None  # 'LONG' or 'SHORT'
        self._consecutive_inside = 0
        self._last_period_end_bar = -1
        self._triggered = False
        self._entry_count = 0

        # Validate setup
        if (self._prior_vah is not None and self._prior_val is not None
                and not pd.isna(self._prior_vah) and not pd.isna(self._prior_val)):
            va_width = self._prior_vah - self._prior_val
            if va_width >= MIN_VA_WIDTH:
                if self._open_vs_va == 'BELOW_VAL':
                    self._setup_armed = True
                    self._setup_direction = 'LONG'
                elif self._open_vs_va == 'ABOVE_VAH':
                    self._setup_armed = True
                    self._setup_direction = 'SHORT'

    def on_bar(self, bar: pd.Series, bar_index: int, session_context: dict) -> Optional[Signal]:
        if not self._setup_armed or self._triggered:
            return None

        if self._entry_count >= MAX_ENTRIES_PER_SESSION:
            return None

        bar_time = session_context.get('bar_time')
        if bar_time and bar_time >= ENTRY_CUTOFF:
            return None

        current_price = bar['close']

        if self._prior_vah is None or self._prior_val is None:
            return None

        vah = self._prior_vah
        val = self._prior_val
        poc = self._prior_poc if self._prior_poc is not None else (vah + val) / 2
        va_width = vah - val

        # --- Monitor 30-min period acceptance ---
        period_end = ((bar_index + 1) % PERIOD_LENGTH_BARS == 0)

        if period_end and bar_index > self._last_period_end_bar:
            self._last_period_end_bar = bar_index

            is_inside_va = val <= current_price <= vah

            if is_inside_va:
                self._consecutive_inside += 1
            else:
                self._consecutive_inside = 0

            if self._consecutive_inside >= ACCEPTANCE_PERIODS:
                return self._generate_signal(
                    bar, bar_index, session_context,
                    current_price, vah, val, poc, va_width,
                )

        return None

    def _compute_target(self, entry_price, vah, val, poc, direction):
        """Compute target based on target_mode."""
        if direction == 'LONG':
            if self._target_mode == 'opposite_va':
                return vah
            elif self._target_mode == 'poc':
                return poc
            elif self._target_mode == 'half_va':
                return entry_price + (vah - entry_price) * 0.5
            else:
                return vah
        else:  # SHORT
            if self._target_mode == 'opposite_va':
                return val
            elif self._target_mode == 'poc':
                return poc
            elif self._target_mode == 'half_va':
                return entry_price - (entry_price - val) * 0.5
            else:
                return val

    def _generate_signal(
        self, bar, bar_index, session_context,
        current_price, vah, val, poc, va_width,
    ) -> Optional[Signal]:
        """Generate 80P entry signal after acceptance confirmed."""

        delta = bar.get('delta', 0)
        if pd.isna(delta):
            delta = 0
        volume_spike = bar.get('volume_spike', 1.0)
        if pd.isna(volume_spike):
            volume_spike = 1.0

        if self._setup_direction == 'LONG':
            entry_price = current_price
            stop_price = val - STOP_BUFFER_PTS
            target_price = self._compute_target(entry_price, vah, val, poc, 'LONG')

            confidence = 'high' if delta > 0 else 'medium'

            risk = entry_price - stop_price
            reward = target_price - entry_price
            if risk <= 0 or reward / risk < self._min_rr:
                return None

            self._triggered = True
            self._entry_count += 1

            return Signal(
                timestamp=bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp'),
                direction='LONG',
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
                strategy_name=self.name,
                setup_type='80P_LONG',
                day_type=session_context.get('day_type', ''),
                trend_strength=session_context.get('trend_strength', ''),
                confidence=confidence,
                metadata={
                    'prior_vah': round(vah, 2),
                    'prior_val': round(val, 2),
                    'prior_poc': round(poc, 2),
                    'va_width': round(va_width, 2),
                    'open_vs_va': self._open_vs_va,
                    'consecutive_inside': self._consecutive_inside,
                    'delta': delta,
                    'volume_spike': round(volume_spike, 2),
                    'target_mode': self._target_mode,
                },
            )

        elif self._setup_direction == 'SHORT':
            entry_price = current_price
            stop_price = vah + STOP_BUFFER_PTS
            target_price = self._compute_target(entry_price, vah, val, poc, 'SHORT')

            confidence = 'high' if delta < 0 else 'medium'

            risk = stop_price - entry_price
            reward = entry_price - target_price
            if risk <= 0 or reward / risk < self._min_rr:
                return None

            self._triggered = True
            self._entry_count += 1

            return Signal(
                timestamp=bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp'),
                direction='SHORT',
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
                strategy_name=self.name,
                setup_type='80P_SHORT',
                day_type=session_context.get('day_type', ''),
                trend_strength=session_context.get('trend_strength', ''),
                confidence=confidence,
                metadata={
                    'prior_vah': round(vah, 2),
                    'prior_val': round(val, 2),
                    'prior_poc': round(poc, 2),
                    'va_width': round(va_width, 2),
                    'open_vs_va': self._open_vs_va,
                    'consecutive_inside': self._consecutive_inside,
                    'delta': delta,
                    'volume_spike': round(volume_spike, 2),
                    'target_mode': self._target_mode,
                },
            )

        return None
