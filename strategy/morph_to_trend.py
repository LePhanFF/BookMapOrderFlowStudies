"""
Strategy 9: Balanced Day Morph to Trend Day

Dalton Playbook Rules:
  - Full size only on morph trigger + 3/3
  - Early P/B-day -> single prints + fattening + extreme DPOC migration
    (>160 pts into VAH/VAL)
  - Must meet rigid 5/5 checklist post-morph
  - Resolution: DPOC migration/compressing into VAH/VAL + acceptance
  - Entry: Pullback to new support (VWAP, emerging dPOC)

The Failure Trigger:
  - If price trades back through the new DPOC during mid-PM attempt to extend,
    the morph has FAILED -> immediately flat

Caution:
  - Stalling DPOC or opposite wick parade >=6 = abort morph call

Acceptance Requirement:
  - Morph must show 2+ consecutive bars beyond breakout threshold for confirmation
  - Single-bar poke = fake breakout, not a morph
  - Require at least moderate trend strength (extension 0.5x+)
"""

from datetime import time
from typing import Optional, List
import pandas as pd

from strategy.base import StrategyBase
from strategy.signal import Signal
from config.constants import MORPH_TO_TREND_BREAKOUT_POINTS, MORPH_TO_TREND_TARGET_POINTS


class MorphToTrendStrategy(StrategyBase):

    @property
    def name(self) -> str:
        return "Morph to Trend"

    @property
    def applicable_day_types(self) -> List[str]:
        return ['p_day', 'b_day']

    def on_session_start(self, session_date, ib_high, ib_low, ib_range, session_context):
        self._ib_high = ib_high
        self._ib_low = ib_low
        self._ib_range = ib_range
        self._entry_taken = False
        self._breakout_bars_bull = 0
        self._breakout_bars_bear = 0

    def on_bar(self, bar: pd.Series, bar_index: int, session_context: dict) -> Optional[Signal]:
        if self._entry_taken:
            return None

        bar_time = session_context.get('bar_time')
        current_price = bar['close']
        strength = session_context.get('trend_strength', 'weak')

        # Morph requires time to develop -- only after 11:00 AM
        if bar_time and bar_time < time(11, 0):
            return None

        # Morph needs at least moderate strength to confirm real extension
        if strength == 'weak':
            return None

        breakout_threshold = MORPH_TO_TREND_BREAKOUT_POINTS

        # --- Track consecutive breakout bars ---
        if current_price > self._ib_high + breakout_threshold:
            self._breakout_bars_bull += 1
            self._breakout_bars_bear = 0  # reset opposite
        elif current_price < self._ib_low - breakout_threshold:
            self._breakout_bars_bear += 1
            self._breakout_bars_bull = 0
        else:
            # Price back inside — reset both
            self._breakout_bars_bull = 0
            self._breakout_bars_bear = 0

        # Need 2+ consecutive bars beyond threshold for morph confirmation
        if self._breakout_bars_bull >= 2:
            # Stop: meaningful distance — at least 50% of IB range below entry
            stop_distance = max(breakout_threshold, self._ib_range * 0.5)
            stop_price = current_price - stop_distance
            target_price = current_price + MORPH_TO_TREND_TARGET_POINTS

            self._entry_taken = True
            return Signal(
                timestamp=bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp'),
                direction='LONG',
                entry_price=current_price,
                stop_price=stop_price,
                target_price=target_price,
                strategy_name=self.name,
                setup_type='MORPH_TO_TREND_BULL',
                day_type=session_context.get('day_type', 'p_day'),
                trend_strength=strength,
                confidence='medium',
            )

        if self._breakout_bars_bear >= 2:
            stop_distance = max(breakout_threshold, self._ib_range * 0.5)
            stop_price = current_price + stop_distance
            target_price = current_price - MORPH_TO_TREND_TARGET_POINTS

            self._entry_taken = True
            return Signal(
                timestamp=bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp'),
                direction='SHORT',
                entry_price=current_price,
                stop_price=stop_price,
                target_price=target_price,
                strategy_name=self.name,
                setup_type='MORPH_TO_TREND_BEAR',
                day_type=session_context.get('day_type', 'p_day'),
                trend_strength=strength,
                confidence='medium',
            )

        return None
