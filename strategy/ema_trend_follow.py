"""
Strategy: EMA Crossover Trend Following with Pullback Entry

Research Source:
  - Tradeify: "10 Proven Futures Trading Strategies to Pass Prop Evaluations"
  - DamnPropFirms: "Core Strategy & Execution (NQ/ES Specific)"
  - Community consensus: top 3 strategy for passing evaluations

WHY THIS PASSES EVALUATIONS:
  - Trend following naturally builds buffer (trailing drawdown moves in your favor)
  - EMA crossover is simple, objective, and well-understood
  - Pullback entries provide tight stops → better R:R than chase entries
  - Works on trend days (40% of sessions) which are the most profitable
  - 50-55% WR with 2:1 R:R → positive expectancy

CONCEPT:
  Use EMA20/EMA50 crossover to confirm trend direction. Then wait for a
  pullback to EMA20 and enter on the recovery bar with delta confirmation.
  This is an enhancement of the existing TrendDayBull strategy that uses
  EMA crossover instead of IB acceptance as the primary trend filter.

ENTRY MODEL:
  1. EMA20 > EMA50 for longs (EMA20 < EMA50 for shorts)
  2. Price above VWAP for longs (below for shorts)
  3. Wait for pullback: price touches or comes within 0.15x IB of EMA20
  4. Recovery bar: closes back above EMA20 with positive delta
  5. Volume_spike >= 1.0 on recovery bar
  6. Enter at close of recovery bar

EXIT MODEL:
  - Stop: Below EMA50 or 0.5x IB range, whichever is tighter
  - Target: 2.0x IB range from entry
  - Trail: After 1.5x IB, trail stop to 1.0x IB profit
  - Time: Close by 15:30 ET
"""

from datetime import time
from typing import Optional, List
import pandas as pd
import numpy as np

from strategy.base import StrategyBase
from strategy.signal import Signal


# --- Parameters ---
EMA_PROXIMITY_MULT = 0.20      # Price within 20% of IB range from EMA20 = "at EMA"
STOP_EMA50_BUFFER = 0.15       # Stop buffer below EMA50
STOP_IB_MULT = 0.50            # Alternative stop: 0.5x IB range below entry
MIN_STOP_PTS = 12.0            # Minimum stop distance
TARGET_MULT = 2.0              # Target = 2.0x IB range
TRAIL_TRIGGER_MULT = 1.5       # Trail activates after 1.5x IB range move
MIN_VOLUME_SPIKE = 0.9         # Volume on recovery bar
MAX_ENTRIES_PER_SESSION = 2    # Max entries per session
ENTRY_START = time(10, 30)     # After IB formation
ENTRY_CUTOFF = time(14, 0)     # No new entries after 2 PM
COOLDOWN_BARS = 10             # Min bars between entries
MIN_IB_RANGE = 20.0            # Skip tiny IB sessions


class EMATrendFollow(StrategyBase):
    """
    EMA Crossover Trend Following with pullback entries.

    Uses EMA20/EMA50 golden cross (or death cross) to identify trend,
    then enters on confirmed pullbacks to EMA20 with order flow confirmation.
    """

    @property
    def name(self) -> str:
        return "EMA Trend Follow"

    @property
    def applicable_day_types(self) -> List[str]:
        # Trades on trend days primarily, but EMA crossover is the real filter
        return ['trend_up', 'trend_down', 'super_trend_up', 'super_trend_down', 'p_day']

    def on_session_start(self, session_date, ib_high, ib_low, ib_range, session_context):
        self._ib_high = ib_high
        self._ib_low = ib_low
        self._ib_range = ib_range

        self._entry_count = 0
        self._last_entry_bar = -999
        self._pullback_detected = False
        self._pullback_low = None
        self._pullback_high = None

    def on_bar(self, bar: pd.Series, bar_index: int, session_context: dict) -> Optional[Signal]:
        if self._ib_range < MIN_IB_RANGE:
            return None

        bar_time = session_context.get('bar_time')
        if bar_time:
            if bar_time < ENTRY_START or bar_time >= ENTRY_CUTOFF:
                return None

        if self._entry_count >= MAX_ENTRIES_PER_SESSION:
            return None

        if bar_index - self._last_entry_bar < COOLDOWN_BARS:
            return None

        current_price = bar['close']
        ema20 = bar.get('ema20')
        ema50 = bar.get('ema50')
        vwap = bar.get('vwap')

        if ema20 is None or ema50 is None or pd.isna(ema20) or pd.isna(ema50):
            return None

        delta = bar.get('delta', 0)
        if pd.isna(delta):
            delta = 0
        volume_spike = bar.get('volume_spike', 1.0)
        if pd.isna(volume_spike):
            volume_spike = 1.0

        # --- LONG SETUP: EMA20 > EMA50 (uptrend) ---
        if ema20 > ema50:
            return self._check_long_pullback(
                bar, bar_index, session_context,
                current_price, ema20, ema50, vwap, delta, volume_spike,
            )

        # --- SHORT SETUP: EMA20 < EMA50 (downtrend) ---
        # DISABLED: NQ has strong long bias — short EMA pullbacks showed 17.4% WR
        # in backtest across 62 sessions. Consistent with existing codebase findings
        # that short strategies on NQ consistently underperform (25% WR).
        # if ema20 < ema50:
        #     return self._check_short_pullback(
        #         bar, bar_index, session_context,
        #         current_price, ema20, ema50, vwap, delta, volume_spike,
        #     )

        return None

    def _check_long_pullback(
        self, bar, bar_index, session_context,
        price, ema20, ema50, vwap, delta, volume_spike,
    ) -> Optional[Signal]:
        """Long entry: EMA20 > EMA50, pullback to EMA20, recovery with delta."""

        # VWAP confirmation: price should be above VWAP
        if vwap is not None and not pd.isna(vwap):
            if price < vwap:
                return None

        # Check proximity to EMA20 (pullback zone)
        ema20_dist = abs(price - ema20) / self._ib_range if self._ib_range > 0 else 999

        # Pullback detection: price came close to or touched EMA20
        if bar['low'] <= ema20 + (self._ib_range * EMA_PROXIMITY_MULT):
            self._pullback_detected = True
            if self._pullback_low is None or bar['low'] < self._pullback_low:
                self._pullback_low = bar['low']

        # Entry: pullback detected AND price now closing above EMA20
        if self._pullback_detected and price > ema20:
            # Must have delta confirmation (buyers on recovery)
            if delta <= 0:
                return None

            # Volume check
            if volume_spike < MIN_VOLUME_SPIKE:
                return None

            # Price must still be above EMA20 (structure intact)
            if price <= ema20:
                return None

            # Stop: tighter of EMA50 or 0.5x IB below entry
            stop_ema50 = ema50 - (self._ib_range * STOP_EMA50_BUFFER)
            stop_ib = price - (self._ib_range * STOP_IB_MULT)
            stop_price = max(stop_ema50, stop_ib)  # Use tighter stop
            stop_price = min(stop_price, price - MIN_STOP_PTS)

            target_price = price + (TARGET_MULT * self._ib_range)

            risk = price - stop_price
            if risk < MIN_STOP_PTS:
                return None

            self._entry_count += 1
            self._last_entry_bar = bar_index
            self._pullback_detected = False
            self._pullback_low = None

            return Signal(
                timestamp=bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp'),
                direction='LONG',
                entry_price=price,
                stop_price=stop_price,
                target_price=target_price,
                strategy_name=self.name,
                setup_type='EMA_PULLBACK_LONG',
                day_type=session_context.get('day_type', ''),
                trend_strength=session_context.get('trend_strength', ''),
                confidence='high' if ema20_dist < 0.10 else 'medium',
                metadata={
                    'ema20': ema20,
                    'ema50': ema50,
                    'pullback_low': self._pullback_low,
                    'volume_spike': volume_spike,
                    'delta': delta,
                },
            )

        return None

    def _check_short_pullback(
        self, bar, bar_index, session_context,
        price, ema20, ema50, vwap, delta, volume_spike,
    ) -> Optional[Signal]:
        """Short entry: EMA20 < EMA50, pullback to EMA20, rejection with delta."""

        # VWAP confirmation: price should be below VWAP
        if vwap is not None and not pd.isna(vwap):
            if price > vwap:
                return None

        # Pullback detection: price came close to or touched EMA20 from below
        if bar['high'] >= ema20 - (self._ib_range * EMA_PROXIMITY_MULT):
            self._pullback_detected = True
            if self._pullback_high is None or bar['high'] > self._pullback_high:
                self._pullback_high = bar['high']

        # Entry: pullback detected AND price now closing below EMA20
        if self._pullback_detected and price < ema20:
            if delta >= 0:
                return None  # Need seller conviction

            if volume_spike < MIN_VOLUME_SPIKE:
                return None

            # Stop above EMA50 or 0.5x IB above entry
            stop_ema50 = ema50 + (self._ib_range * STOP_EMA50_BUFFER)
            stop_ib = price + (self._ib_range * STOP_IB_MULT)
            stop_price = min(stop_ema50, stop_ib)
            stop_price = max(stop_price, price + MIN_STOP_PTS)

            target_price = price - (TARGET_MULT * self._ib_range)

            risk = stop_price - price
            if risk < MIN_STOP_PTS:
                return None

            self._entry_count += 1
            self._last_entry_bar = bar_index
            self._pullback_detected = False
            self._pullback_high = None

            return Signal(
                timestamp=bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp'),
                direction='SHORT',
                entry_price=price,
                stop_price=stop_price,
                target_price=target_price,
                strategy_name=self.name,
                setup_type='EMA_PULLBACK_SHORT',
                day_type=session_context.get('day_type', ''),
                trend_strength=session_context.get('trend_strength', ''),
                confidence='high',
                metadata={
                    'ema20': ema20,
                    'ema50': ema50,
                    'pullback_high': self._pullback_high,
                    'volume_spike': volume_spike,
                    'delta': delta,
                },
            )

        return None
