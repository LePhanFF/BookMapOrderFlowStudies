"""
Strategy 6: Bear Acceptance Short

When a session extends bearish (below IBL with conviction), wait for 3+
consecutive bars to close below IBL (acceptance). Enter SHORT on the 3rd bar.

From v14 report:
  - Day type: trend_down (ext > 1.0x below) or bearish (ext > 0.5x below)
  - NEVER on b_day or neutral (14% WR on b_day, 10% on neutral)
  - 3+ consecutive bars close below IBL
  - Entry: SHORT at close of 3rd acceptance bar
  - Stop: IBL + 25% of IB range
  - Target: IBL - 0.75x IB range
  - Time: before 14:00

Performance (v14, 62 sessions):
  11 trades, 64% WR, $90/trade expectancy, PF 3.32, MaxDD -$289
"""

from datetime import time as _time
from typing import Optional, List
import pandas as pd

from strategy.base import StrategyBase
from strategy.signal import Signal

# Constants
BEAR_ACCEPT_MIN_BARS = 3          # 3 consecutive bars below IBL
BEAR_ACCEPT_MIN_EXT = 0.4         # Min extension below IBL (as fraction of IB range)
BEAR_ACCEPT_STOP_BUFFER = 0.25    # Stop: IBL + 25% of IB range
BEAR_ACCEPT_TARGET_MULT = 0.75    # Target: IBL - 0.75x IB range
BEAR_ACCEPT_LAST_ENTRY = _time(14, 0)  # No entries after 14:00
BEAR_ACCEPT_COOLDOWN_BARS = 30    # Bars between entries


class BearAcceptShort(StrategyBase):
    """
    Bear Acceptance Short: enter on structural breakdown below IBL.

    Unlike TrendDayBear (which waits for pullback to VWAP after acceptance),
    this strategy enters on the acceptance bar itself — the 3rd consecutive
    bar closing below IBL confirms the breakdown is real.
    """

    @property
    def name(self) -> str:
        return "Bear Accept Short"

    @property
    def applicable_day_types(self) -> List[str]:
        return ['trend_down', 'super_trend_down']

    def on_session_start(self, session_date, ib_high, ib_low, ib_range, session_context):
        self._ib_high = ib_high
        self._ib_low = ib_low
        self._ib_range = ib_range
        self._ib_mid = (ib_high + ib_low) / 2

        self._consecutive_below = 0
        self._entry_fired = False
        self._last_entry_bar = -BEAR_ACCEPT_COOLDOWN_BARS
        self._active = ib_range > 0

    def on_bar(self, bar: pd.Series, bar_index: int, session_context: dict) -> Optional[Signal]:
        if not self._active or self._entry_fired:
            return None

        current_price = bar['close']
        bar_time = session_context.get('bar_time')

        # Time gate
        if bar_time and bar_time >= BEAR_ACCEPT_LAST_ENTRY:
            return None

        # Track consecutive bars below IBL
        if current_price < self._ib_low:
            self._consecutive_below += 1
        else:
            self._consecutive_below = 0
            return None

        # Check if bearish extension is sufficient (>= 0.5x IB range)
        ext_below = (self._ib_low - current_price) / self._ib_range
        if ext_below < BEAR_ACCEPT_MIN_EXT:
            return None

        # Day type gate: only on bearish day types
        day_type = session_context.get('day_type', '')
        # Accept trend_down, super_trend_down only (report: 64% WR on trend_down)
        # NEVER on b_day (14% WR) or neutral (10% WR)
        if day_type not in ('trend_down', 'super_trend_down'):
            return None

        # Acceptance: 3+ consecutive bars below IBL
        if self._consecutive_below < BEAR_ACCEPT_MIN_BARS:
            return None

        # Cooldown
        if bar_index - self._last_entry_bar < BEAR_ACCEPT_COOLDOWN_BARS:
            return None

        # Compute stop and target
        stop_price = self._ib_low + self._ib_range * BEAR_ACCEPT_STOP_BUFFER
        target_price = self._ib_low - self._ib_range * BEAR_ACCEPT_TARGET_MULT

        # Validate risk/reward
        risk = stop_price - current_price
        reward = current_price - target_price
        min_dist = max(self._ib_range * 0.03, 5)  # 3% of IB range or 5 pts, whichever is larger
        if risk <= min_dist or reward <= min_dist:
            return None

        self._entry_fired = True
        self._last_entry_bar = bar_index

        return Signal(
            timestamp=bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp'),
            direction='SHORT',
            entry_price=current_price,
            stop_price=stop_price,
            target_price=target_price,
            strategy_name=self.name,
            setup_type='BEAR_ACCEPTANCE',
            day_type=session_context.get('day_type', 'trend_down'),
            trend_strength=session_context.get('trend_strength', 'moderate'),
            confidence='high' if self._consecutive_below >= 6 else 'medium',
        )
