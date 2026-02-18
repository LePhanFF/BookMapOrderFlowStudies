"""
Strategy: VWAP Mean Reversion (Mid-Day Balance)

Research Source:
  - DamnPropFirms: "Core Strategy & Execution (NQ/ES Specific)"
  - Tradeify: "10 Proven Futures Trading Strategies to Pass Prop Evaluations"
  - Community consensus: 55-65% WR on ES, slightly lower on NQ

WHY THIS PASSES EVALUATIONS:
  - Capitalizes on balance/range days (40-60% of all sessions)
  - Counter-trend to existing strategies → portfolio diversification
  - Works during mid-day lull when trend strategies are inactive
  - High win rate (mean reversion has natural statistical edge)
  - Small, consistent gains → ideal for consistency rule compliance

CONCEPT:
  When price deviates significantly from VWAP on low-extension days,
  trade the return to the mean. Only engage on B-Day or Neutral sessions
  where IB extension is minimal (< 0.5x IB range).

ENTRY MODEL:
  1. Day type filter: b_day or neutral (IB extension < 0.5x)
  2. Price deviation: > 0.75x IB range away from VWAP
  3. RSI exhaustion: RSI14 > 65 (short) or RSI14 < 35 (long)
  4. Delta exhaustion then reversal
  5. Volume declining (not continuation)
  6. Enter when price starts closing back toward VWAP

EXIT MODEL:
  - Target: VWAP (the mean)
  - Stop: Beyond the deviation extreme + 0.25x IB range
  - Trail: Breakeven at 50% of distance to VWAP
  - Time: Close by 15:00 ET
"""

from datetime import time
from typing import Optional, List
import pandas as pd
import numpy as np

from strategy.base import StrategyBase
from strategy.signal import Signal


# --- Parameters ---
MIN_DEVIATION_MULT = 0.60       # Price must be >= 0.6x IB range from VWAP
RSI_OVERBOUGHT = 65.0           # RSI threshold for short entries
RSI_OVERSOLD = 35.0             # RSI threshold for long entries
MAX_VOLUME_SPIKE = 1.2          # Volume should be declining (exhaustion)
STOP_BUFFER_MULT = 0.25         # Stop beyond deviation extreme
MIN_STOP_PTS = 12.0             # Minimum stop distance
MAX_ENTRIES_PER_SESSION = 2     # Max mean reversion trades per session
ENTRY_START = time(11, 0)       # Only trade mid-day (after AM volatility)
ENTRY_CUTOFF = time(14, 30)     # No new entries after 2:30 PM
COOLDOWN_BARS = 15              # Min bars between entries
MIN_IB_RANGE = 15.0             # Skip tiny IB sessions
ALLOWED_DAY_TYPES = ['b_day', 'neutral', 'p_day']  # Range-bound day types


class MeanReversionVWAP(StrategyBase):
    """
    VWAP Mean Reversion for balance/range days.

    Enters counter-trend when price deviates significantly from VWAP
    and shows exhaustion signals (RSI extreme + declining volume).
    Targets a return to VWAP.
    """

    @property
    def name(self) -> str:
        return "Mean Reversion VWAP"

    @property
    def applicable_day_types(self) -> List[str]:
        return ALLOWED_DAY_TYPES

    def on_session_start(self, session_date, ib_high, ib_low, ib_range, session_context):
        self._ib_high = ib_high
        self._ib_low = ib_low
        self._ib_range = ib_range
        self._ib_mid = (ib_high + ib_low) / 2

        self._entry_count = 0
        self._last_entry_bar = -999
        self._session_high = ib_high
        self._session_low = ib_low

    def on_bar(self, bar: pd.Series, bar_index: int, session_context: dict) -> Optional[Signal]:
        if self._ib_range < MIN_IB_RANGE:
            return None

        # Time window
        bar_time = session_context.get('bar_time')
        if bar_time:
            if bar_time < ENTRY_START or bar_time >= ENTRY_CUTOFF:
                return None

        # Day type filter
        day_type = session_context.get('day_type', '')
        if day_type not in ALLOWED_DAY_TYPES:
            return None

        if self._entry_count >= MAX_ENTRIES_PER_SESSION:
            return None

        if bar_index - self._last_entry_bar < COOLDOWN_BARS:
            return None

        # Update session extremes
        self._session_high = max(self._session_high, bar['high'])
        self._session_low = min(self._session_low, bar['low'])

        current_price = bar['close']
        vwap = bar.get('vwap')
        if vwap is None or pd.isna(vwap):
            return None

        rsi = bar.get('rsi14')
        if rsi is None or pd.isna(rsi):
            return None

        delta = bar.get('delta', 0)
        if pd.isna(delta):
            delta = 0

        volume_spike = bar.get('volume_spike', 1.0)
        if pd.isna(volume_spike):
            volume_spike = 1.0

        # Compute deviation from VWAP
        deviation = current_price - vwap
        deviation_mult = abs(deviation) / self._ib_range if self._ib_range > 0 else 0

        if deviation_mult < MIN_DEVIATION_MULT:
            return None  # Not deviated enough

        # --- LONG: Price below VWAP, oversold, expecting mean reversion up ---
        if deviation < 0 and rsi < RSI_OVERSOLD:
            return self._check_long_reversion(
                bar, bar_index, session_context,
                current_price, vwap, deviation_mult, delta, volume_spike,
            )

        # --- SHORT: Price above VWAP, overbought, expecting mean reversion down ---
        if deviation > 0 and rsi > RSI_OVERBOUGHT:
            return self._check_short_reversion(
                bar, bar_index, session_context,
                current_price, vwap, deviation_mult, delta, volume_spike,
            )

        return None

    def _check_long_reversion(
        self, bar, bar_index, session_context,
        price, vwap, deviation_mult, delta, volume_spike,
    ) -> Optional[Signal]:
        """Long entry: price below VWAP with exhaustion, expecting return up."""
        # Exhaustion signals: volume declining (not a strong trend push)
        if volume_spike > MAX_VOLUME_SPIKE:
            return None  # Strong volume = continuation, not exhaustion

        # Delta turning positive (buyers stepping in at low)
        if delta <= 0:
            return None  # Still seller-dominated

        # Candle showing reversal: close above open (green bar)
        if price <= bar['open']:
            return None

        # Stop below session low or deviation extreme
        stop_price = min(self._session_low, bar['low']) - (self._ib_range * STOP_BUFFER_MULT)
        stop_price = min(stop_price, price - MIN_STOP_PTS)

        # Target: VWAP
        target_price = vwap

        risk = price - stop_price
        reward = target_price - price
        if risk < MIN_STOP_PTS or reward < risk * 0.8:
            return None  # Bad R:R

        self._entry_count += 1
        self._last_entry_bar = bar_index

        return Signal(
            timestamp=bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp'),
            direction='LONG',
            entry_price=price,
            stop_price=stop_price,
            target_price=target_price,
            strategy_name=self.name,
            setup_type='VWAP_MEAN_REVERT_LONG',
            day_type=session_context.get('day_type', ''),
            trend_strength=session_context.get('trend_strength', ''),
            confidence='high' if deviation_mult >= 1.0 else 'medium',
            metadata={
                'deviation_mult': round(deviation_mult, 2),
                'rsi': bar.get('rsi14', 0),
                'volume_spike': volume_spike,
                'vwap': vwap,
            },
        )

    def _check_short_reversion(
        self, bar, bar_index, session_context,
        price, vwap, deviation_mult, delta, volume_spike,
    ) -> Optional[Signal]:
        """Short entry: price above VWAP with exhaustion, expecting return down."""
        if volume_spike > MAX_VOLUME_SPIKE:
            return None

        # Delta turning negative (sellers stepping in at high)
        if delta >= 0:
            return None

        # Candle showing reversal: close below open (red bar)
        if price >= bar['open']:
            return None

        # Stop above session high or deviation extreme
        stop_price = max(self._session_high, bar['high']) + (self._ib_range * STOP_BUFFER_MULT)
        stop_price = max(stop_price, price + MIN_STOP_PTS)

        # Target: VWAP
        target_price = vwap

        risk = stop_price - price
        reward = price - target_price
        if risk < MIN_STOP_PTS or reward < risk * 0.8:
            return None

        self._entry_count += 1
        self._last_entry_bar = bar_index

        return Signal(
            timestamp=bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp'),
            direction='SHORT',
            entry_price=price,
            stop_price=stop_price,
            target_price=target_price,
            strategy_name=self.name,
            setup_type='VWAP_MEAN_REVERT_SHORT',
            day_type=session_context.get('day_type', ''),
            trend_strength=session_context.get('trend_strength', ''),
            confidence='high' if deviation_mult >= 1.0 else 'medium',
            metadata={
                'deviation_mult': round(deviation_mult, 2),
                'rsi': bar.get('rsi14', 0),
                'volume_spike': volume_spike,
                'vwap': vwap,
            },
        )
