"""
Strategy 8: Trend Day PM Morph (Early Balance -> Late Trend)

Dalton Playbook Rules:
  - Full size only on late 3/3 lock
  - Rotational early (Range D-Day), then acceptance + DPOC migration post-12:30/14:00
  - Trigger: Final-hour breakout acceptance + volume
  - Entry: Retest of breakout level (IBH/IBL or VWAP)
  - Targets: Measured late leg (common +200-400 pts), weekly extreme

No PM Morph Filter:
  - If trend strength is Weak or major HVN/LVN in the way, assume no morph

Caution:
  - Failed late breakout (rejection back into balance) = immediate flat
  - If PM extension fails and price retraces to VWAP, exit immediately
  - Use tighter stop: AM range boundary (not wide IBL/IBH stop)

VWAP Failure Rule:
  - After PM breakout, if price retraces below VWAP (bull) or above VWAP (bear),
    the extension has failed. Set stop very tight (at VWAP level) to exit quickly.

STOP FIX:
  - Stop must always be a meaningful distance from entry.
  - Minimum stop distance = 25% of AM range.
  - For LONG: stop = max(VWAP - breach, AM_low) but at least 25% of AM range below entry.
  - For SHORT: stop = min(VWAP + breach, AM_high) but at least 25% of AM range above entry.
"""

from datetime import time
from typing import Optional, List
import pandas as pd

from strategy.base import StrategyBase
from strategy.signal import Signal
from config.constants import PM_MORPH_BREAKOUT_RATIO, VWAP_BREACH_RATIO

# Minimum stop distance as fraction of AM range
PM_MORPH_MIN_STOP_FRAC = 0.25


class PMMorphStrategy(StrategyBase):

    @property
    def name(self) -> str:
        return "PM Morph"

    @property
    def applicable_day_types(self) -> List[str]:
        # PM Morph can happen on any day that starts balanced
        return ['neutral', 'b_day', 'p_day']

    def on_session_start(self, session_date, ib_high, ib_low, ib_range, session_context):
        self._ib_high = ib_high
        self._ib_low = ib_low
        self._ib_range = ib_range

        self._am_high = ib_high
        self._am_low = ib_low
        self._am_tracking = True
        self._entry_taken = False
        self._breakout_confirmed = False
        self._breakout_bars = 0
        self._breakout_dir = None  # Track which direction breakout accumulates

    def on_bar(self, bar: pd.Series, bar_index: int, session_context: dict) -> Optional[Signal]:
        if self._entry_taken:
            return None

        bar_time = session_context.get('bar_time')
        current_price = bar['close']
        strength = session_context.get('trend_strength', 'weak')

        # Track AM range (9:30-12:30) — always, regardless of day_type
        if bar_time and bar_time < time(12, 30):
            self._am_high = max(self._am_high, bar['high'])
            self._am_low = min(self._am_low, bar['low'])
            return None

        self._am_tracking = False

        # PM Morph requires strong evidence of late breakout.
        # Need at least strong strength (1.0x+ extension) to confirm morph,
        # not just moderate. Most PM "morphs" at moderate extension are noise.
        if strength in ('weak', 'moderate'):
            return None

        # --- PM breakout detection (after 12:30) ---
        am_range = self._am_high - self._am_low
        breakout_threshold = am_range * PM_MORPH_BREAKOUT_RATIO if am_range > 0 else 15

        # Track acceptance of breakout (3 bars to confirm PM morph, more conservative)
        if not self._breakout_confirmed:
            if current_price > self._am_high + breakout_threshold:
                if self._breakout_dir == 'LONG':
                    self._breakout_bars += 1
                else:
                    self._breakout_dir = 'LONG'
                    self._breakout_bars = 1
                if self._breakout_bars >= 3:
                    self._breakout_confirmed = True
                    return self._build_pm_signal(
                        bar, bar_index, session_context,
                        direction='LONG',
                    )
            elif current_price < self._am_low - breakout_threshold:
                if self._breakout_dir == 'SHORT':
                    self._breakout_bars += 1
                else:
                    self._breakout_dir = 'SHORT'
                    self._breakout_bars = 1
                if self._breakout_bars >= 3:
                    self._breakout_confirmed = True
                    return self._build_pm_signal(
                        bar, bar_index, session_context,
                        direction='SHORT',
                    )
            else:
                self._breakout_bars = 0
                self._breakout_dir = None

        return None

    def _build_pm_signal(
        self, bar: pd.Series, bar_index: int, session_context: dict,
        direction: str,
    ) -> Signal:
        current_price = bar['close']
        am_range = self._am_high - self._am_low
        min_stop_dist = am_range * PM_MORPH_MIN_STOP_FRAC  # 25% of AM range (no absolute floor)

        if direction == 'LONG':
            # Stop: AM low or VWAP-based, whichever is closer but still meaningful
            stop_price = self._am_low
            vwap = bar.get('vwap')
            if vwap is not None and not pd.isna(vwap):
                vwap_stop = vwap - self._ib_range * VWAP_BREACH_RATIO
                if vwap_stop > self._am_low:
                    stop_price = vwap_stop
            # Enforce minimum stop distance
            if current_price - stop_price < min_stop_dist:
                stop_price = current_price - min_stop_dist
            target_price = current_price + (2.0 * am_range)
            setup_type = 'PM_MORPH_BULL'
        else:
            stop_price = self._am_high
            vwap = bar.get('vwap')
            if vwap is not None and not pd.isna(vwap):
                vwap_stop = vwap + self._ib_range * VWAP_BREACH_RATIO
                if vwap_stop < self._am_high:
                    stop_price = vwap_stop
            # Enforce minimum stop distance
            if stop_price - current_price < min_stop_dist:
                stop_price = current_price + min_stop_dist
            target_price = current_price - (2.0 * am_range)
            setup_type = 'PM_MORPH_BEAR'

        self._entry_taken = True

        return Signal(
            timestamp=bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp'),
            direction=direction,
            entry_price=current_price,
            stop_price=stop_price,
            target_price=target_price,
            strategy_name=self.name,
            setup_type=setup_type,
            day_type=session_context.get('day_type', 'neutral'),
            trend_strength=session_context.get('trend_strength', ''),
            confidence='medium',
        )
