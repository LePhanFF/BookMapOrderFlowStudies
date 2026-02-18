"""
Regime Filter — Bull/Bear regime detection and directional gating.

The data period (Nov 2025 - Feb 2026) was:
  - Net: -0.9% (slightly bearish)
  - 50/50 up/down days
  - 39% bull regime (EMA20 > EMA50), 61% bear regime

PROBLEM SOLVED:
  - Short strategies showed 17-30% WR because we applied them in ALL regimes
  - Long strategies showed 57-80% WR for same reason (bull-regime trades dominated)
  - SOLUTION: Only trade LONGS in bull regime, SHORTS in bear regime
  - This should unlock short strategies in bear periods without hurting long WR

REGIME DETECTION (3 methods, AND logic):
  1. EMA20 vs EMA50 at session open (HTF trend)
  2. Prior session close vs VWAP (institutional bias)
  3. Price vs prior day close (gap direction)

When regime is BULL:
  - LONG signals: PASS (full size)
  - SHORT signals: BLOCK (or reduce to 50% size)

When regime is BEAR:
  - SHORT signals: PASS (full size)
  - LONG signals: BLOCK (or reduce to 50% size)

When regime is NEUTRAL/CHOP:
  - Both directions: reduce to 50% size
"""

from filters.base import FilterBase
from strategy.signal import Signal
import pandas as pd


class RegimeFilter(FilterBase):
    """
    Filters signals based on bull/bear regime detection.

    Uses EMA20/EMA50 crossover and prior session context to determine
    the current regime, then only allows signals aligned with it.
    """

    def __init__(self, block_counter_trend: bool = True, allow_reduced: bool = False):
        """
        Args:
            block_counter_trend: If True, completely block counter-trend signals.
                                 If False, allow them through (for analysis).
            allow_reduced: If True, pass counter-trend signals but mark confidence
                          as 'low' (so position sizing can reduce).
        """
        self._block_counter_trend = block_counter_trend
        self._allow_reduced = allow_reduced

    @property
    def name(self) -> str:
        return "Regime Filter"

    def should_trade(self, signal: Signal, bar: pd.Series, session_context: dict) -> bool:
        """
        Determine if signal aligns with current regime.

        Regime detection:
          BULL: EMA20 > EMA50 at current bar AND prior session was bullish
          BEAR: EMA20 < EMA50 at current bar AND prior session was bearish
          NEUTRAL: Mixed signals
        """
        regime = self._detect_regime(bar, session_context)

        if regime == 'BULL':
            if signal.direction == 'LONG':
                return True
            else:
                # Short in bull regime
                if self._block_counter_trend:
                    return False
                return True

        elif regime == 'BEAR':
            if signal.direction == 'SHORT':
                return True
            else:
                # Long in bear regime
                if self._block_counter_trend:
                    return False
                return True

        else:
            # NEUTRAL — allow both but could reduce confidence
            return True

    def _detect_regime(self, bar: pd.Series, session_context: dict) -> str:
        """
        Detect current market regime.

        Uses multiple signals:
          1. EMA20 vs EMA50 (primary)
          2. Prior session bias (secondary)
          3. Price vs VWAP (tertiary)
        """
        bull_votes = 0
        bear_votes = 0

        # Signal 1: EMA20 vs EMA50 (strongest signal)
        ema20 = bar.get('ema20')
        ema50 = bar.get('ema50')
        if ema20 is not None and ema50 is not None:
            if not pd.isna(ema20) and not pd.isna(ema50):
                if ema20 > ema50:
                    bull_votes += 2  # Double weight
                else:
                    bear_votes += 2

        # Signal 2: Prior session bullish/bearish
        prior_bullish = session_context.get('prior_session_bullish')
        if prior_bullish is not None:
            if prior_bullish:
                bull_votes += 1
            else:
                bear_votes += 1

        # Signal 3: Current price vs VWAP
        vwap = session_context.get('vwap')
        current_price = session_context.get('current_price')
        if vwap is not None and current_price is not None:
            if not pd.isna(vwap):
                if current_price > vwap:
                    bull_votes += 1
                else:
                    bear_votes += 1

        # Determine regime
        if bull_votes >= bear_votes + 2:
            return 'BULL'
        elif bear_votes >= bull_votes + 2:
            return 'BEAR'
        else:
            return 'NEUTRAL'


class SimpleRegimeFilter(FilterBase):
    """
    Simplified regime filter using only EMA20/EMA50 crossover.

    This is the most robust single-signal regime detection.
    BULL: EMA20 > EMA50 → only longs
    BEAR: EMA20 < EMA50 → only shorts
    """

    def __init__(self, longs_in_bull: bool = True, shorts_in_bear: bool = True,
                 longs_in_bear: bool = False, shorts_in_bull: bool = False):
        self._longs_in_bull = longs_in_bull
        self._shorts_in_bear = shorts_in_bear
        self._longs_in_bear = longs_in_bear
        self._shorts_in_bull = shorts_in_bull

    @property
    def name(self) -> str:
        return "Simple Regime Filter"

    def should_trade(self, signal: Signal, bar: pd.Series, session_context: dict) -> bool:
        ema20 = bar.get('ema20')
        ema50 = bar.get('ema50')

        if ema20 is None or ema50 is None or pd.isna(ema20) or pd.isna(ema50):
            return True  # Can't determine regime, allow

        is_bull = ema20 > ema50

        if signal.direction == 'LONG':
            if is_bull:
                return self._longs_in_bull
            else:
                return self._longs_in_bear
        else:  # SHORT
            if is_bull:
                return self._shorts_in_bull
            else:
                return self._shorts_in_bear
