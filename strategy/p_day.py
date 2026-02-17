"""
Strategy 5: P-Day (Skewed Balance - 'p' bullish or 'b' bearish)

P-Day is NOT a trend day:
  - P-shape = short covering (bullish skew), bulge at top
  - B-shape = distribution (bearish skew), bulge at bottom

Range/Balance 80% Rule:
  - P-Day from the edge of the profile has ~80% success rate
  - Enter in the direction of the skew from the IB boundary pullback

Entry Model v2 — Pullback to VWAP or IB edge with TIGHT stop:
  OLD problem: stop at opposite IB edge = 115% IB range risk = 1 contract = tiny wins.
  NEW approach: stop below the pullback level, much tighter, better R:R.

  Entry hierarchy:
  1. VWAP pullback with delta (best R:R)
     Stop: VWAP - 0.4x IB range
  2. IB edge retest with delta
     Stop: IB edge - 0.5x IB range

  Target: 1.5x IB range from entry (conservative, matching p_day character)

Caution:
  - No shorts unless 3+ bars below IBL (short conviction)
  - Shorts require quality >= 2 (delta + volume/FVG)
"""

from typing import Optional, List
import pandas as pd

from strategy.base import StrategyBase
from strategy.signal import Signal
from config.constants import ACCEPTANCE_MIN_BARS, LONDON_CLOSE, PM_SESSION_START

PDAY_TARGET_MULT = 1.5
STOP_VWAP_BUFFER = 0.40  # Stop below VWAP - 40% IB range
STOP_IB_EDGE_BUFFER = 0.50  # Stop below IB edge - 50% IB range
STOP_MINIMUM_PTS = 15.0  # Minimum stop distance


class PDayStrategy(StrategyBase):

    @property
    def name(self) -> str:
        return "P-Day"

    @property
    def applicable_day_types(self) -> List[str]:
        return ['p_day']

    def on_session_start(self, session_date, ib_high, ib_low, ib_range, session_context):
        self._ib_high = ib_high
        self._ib_low = ib_low
        self._ib_range = ib_range
        self._ib_mid = (ib_high + ib_low) / 2

        self._direction_resolved = False
        self._resolved_direction = None
        self._consecutive_outside = 0
        self._pending_dir = None
        self._entry_taken = False

        # Order flow momentum tracking
        self._delta_history = []

    def on_bar(self, bar: pd.Series, bar_index: int, session_context: dict) -> Optional[Signal]:
        day_type = session_context.get('day_type', '')
        if day_type not in self.applicable_day_types:
            return None

        strength = session_context.get('trend_strength', 'weak')
        if strength == 'weak':
            return None

        if self._entry_taken:
            return None

        bar_time = session_context.get('bar_time')
        # Use PM_SESSION_START (13:00) as cutoff for P-Day entries.
        # P-Day VWAP pullbacks often develop between 11:30-13:00 after
        # initial directional acceptance. Diagnostic showed 6+ sessions
        # with valid signals blocked by the old 11:30 cutoff.
        if bar_time and bar_time >= PM_SESSION_START:
            return None

        current_price = bar['close']
        delta = bar.get('delta', 0)

        # Track order flow momentum (rolling 10-bar delta window)
        bar_delta = bar.get('delta', 0)
        self._delta_history.append(bar_delta if not pd.isna(bar_delta) else 0)
        if len(self._delta_history) > 10:
            self._delta_history.pop(0)

        # Determine skew direction from price relative to IB
        if not self._direction_resolved:
            if current_price > self._ib_high:
                if self._pending_dir == 'LONG':
                    self._consecutive_outside += 1
                else:
                    self._pending_dir = 'LONG'
                    self._consecutive_outside = 1
            elif current_price < self._ib_low:
                if self._pending_dir == 'SHORT':
                    self._consecutive_outside += 1
                else:
                    self._pending_dir = 'SHORT'
                    self._consecutive_outside = 1
            else:
                self._consecutive_outside = 0
                self._pending_dir = None

            if self._consecutive_outside >= ACCEPTANCE_MIN_BARS and self._pending_dir:
                self._direction_resolved = True
                self._resolved_direction = self._pending_dir
            return None

        # === LONG entries (P-shape: bullish skew) ===
        if self._resolved_direction == 'LONG':
            return self._check_long_entry(bar, bar_index, session_context)

        # === SHORT entries (B-shape: bearish skew) ===
        elif self._resolved_direction == 'SHORT':
            return self._check_short_entry(bar, bar_index, session_context)

        return None

    def _check_long_entry(self, bar, bar_index, session_context):
        current_price = bar['close']
        delta = bar.get('delta', 0)
        strength = session_context.get('trend_strength', 'weak')

        # Price must still be above IB high (p-day structure intact)
        if current_price <= self._ib_high:
            return None

        target_price = current_price + (PDAY_TARGET_MULT * self._ib_range)

        # --- Entry 1: VWAP pullback with delta (best entry) ---
        vwap = bar.get('vwap')
        if vwap is not None and not pd.isna(vwap):
            vwap_dist = abs(current_price - vwap) / self._ib_range if self._ib_range > 0 else 999
            if vwap_dist < 0.40 and current_price > vwap and delta > 0:
                # Order flow momentum check: reject entries where the pullback
                # is driven by aggressive selling (pre-entry delta strongly negative).
                pre_delta_sum = sum(self._delta_history[:-1]) if len(self._delta_history) > 1 else 0
                if pre_delta_sum < -500:
                    return None

                stop = vwap - (self._ib_range * STOP_VWAP_BUFFER)
                stop = min(stop, current_price - STOP_MINIMUM_PTS)
                if current_price - stop >= STOP_MINIMUM_PTS:
                    self._entry_taken = True
                    return Signal(
                        timestamp=bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp'),
                        direction='LONG',
                        entry_price=current_price,
                        stop_price=stop,
                        target_price=target_price,
                        strategy_name=self.name,
                        setup_type='P_DAY_VWAP_LONG',
                        day_type='p_day',
                        trend_strength=strength,
                        confidence='high',
                    )

        # IBH retest disabled — 31.2% WR, -$1,632 in testing.
        # VWAP pullback is the only reliable P-Day long entry.

        return None

    def _check_short_entry(self, bar, bar_index, session_context):
        current_price = bar['close']
        delta = bar.get('delta', 0)
        strength = session_context.get('trend_strength', 'weak')

        # Require 3+ bars for short conviction
        if self._consecutive_outside < 3:
            return None

        # Price must still be below IB low
        if current_price >= self._ib_low:
            return None

        target_price = current_price - (PDAY_TARGET_MULT * self._ib_range)

        # --- Entry 1: VWAP rejection with delta ---
        vwap = bar.get('vwap')
        if vwap is not None and not pd.isna(vwap):
            vwap_dist = abs(current_price - vwap) / self._ib_range if self._ib_range > 0 else 999
            if vwap_dist < 0.40 and current_price < vwap and delta < 0:
                has_volume = bar.get('volume_spike', 1.0) > 1.2
                if has_volume:  # Extra confirmation for shorts
                    stop = vwap + (self._ib_range * STOP_VWAP_BUFFER)
                    stop = max(stop, current_price + STOP_MINIMUM_PTS)
                    if stop - current_price >= STOP_MINIMUM_PTS:
                        self._entry_taken = True
                        return Signal(
                            timestamp=bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp'),
                            direction='SHORT',
                            entry_price=current_price,
                            stop_price=stop,
                            target_price=target_price,
                            strategy_name=self.name,
                            setup_type='P_DAY_VWAP_SHORT',
                            day_type='p_day',
                            trend_strength=strength,
                            confidence='medium',
                        )

        # IBL retest short disabled — same logic as IBH retest longs.

        return None
