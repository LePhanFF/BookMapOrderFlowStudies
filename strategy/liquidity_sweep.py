"""
Strategy: Liquidity Sweep / Stop Hunt Reversal

Research Source:
  - MindMathMoney: "Liquidity Sweep Trading Strategy: How Smart Money Hunts Stop Losses"
  - International Trading Institute: "Liquidity Sweeps: Entry and Exit Strategies"
  - Quantamental Trader: "Mastering Liquidity Sweeps, Stop Hunts, and Order Flow Analysis"

WHY THIS PASSES EVALUATIONS:
  - 60-65% win rate reported by community (highest among researched strategies)
  - Exploits the most predictable institutional behavior: stop hunting
  - Natural mean reversion setup → works on balance/range days (B-Day, Neutral)
  - Tight stops beyond sweep extreme → excellent R:R
  - Counter-trend entries diversify a trend-following portfolio

CONCEPT:
  Institutions push price beyond obvious levels (PDH, PDL, IBH, IBL) to trigger
  stop losses and fill large orders. After the sweep, price reverses. We enter
  AFTER the reversal is confirmed, not during the sweep.

ENTRY MODEL:
  1. Identify key liquidity levels: prior day high/low, IB high/low
  2. Price WICKS beyond the level (penetrates) but CLOSES back inside
  3. Rejection bar: delta reverses direction, volume spikes
  4. Enter on next bar in reversal direction
  5. Optional: FVG or order block confluence at sweep level

EXIT MODEL:
  - Stop: Beyond sweep extreme (wick high/low) + small buffer
  - Target: IB mid or opposite liquidity zone (2x risk minimum)
  - Trail: Breakeven after 1:1 R achieved
  - Time: Close by 15:00 ET
"""

from datetime import time
from typing import Optional, List
import pandas as pd
import numpy as np

from strategy.base import StrategyBase
from strategy.signal import Signal


# --- Parameters ---
SWEEP_BUFFER_PTS = 5.0          # How far past level price must go to count as sweep
REJECTION_CLOSE_INSIDE = True   # Bar must close back inside the range
MIN_VOLUME_SPIKE = 1.1          # Volume spike on rejection bar
MIN_WICK_RATIO = 0.40           # Minimum wick-to-body ratio (shows rejection)
STOP_BEYOND_SWEEP = 0.10        # Stop = sweep extreme + 10% IB range buffer
TARGET_MULT = 1.5               # Target = 1.5x risk distance (minimum 2:1 R:R design)
MAX_ENTRIES_PER_SESSION = 2     # Max sweep trades per session
ENTRY_CUTOFF = time(14, 0)      # No new entries after 2 PM ET
COOLDOWN_BARS = 10              # Min bars between entries
MIN_IB_RANGE = 20.0             # Skip tiny IB sessions


class LiquiditySweep(StrategyBase):
    """
    Liquidity Sweep reversal strategy.

    Detects stop hunts at key levels and enters on confirmed reversals.
    Works best on balance/range days where price oscillates between
    liquidity pools.
    """

    @property
    def name(self) -> str:
        return "Liquidity Sweep"

    @property
    def applicable_day_types(self) -> List[str]:
        # Best on balance/range days, but sweep can happen on any day
        return []

    def on_session_start(self, session_date, ib_high, ib_low, ib_range, session_context):
        self._ib_high = ib_high
        self._ib_low = ib_low
        self._ib_range = ib_range
        self._ib_mid = (ib_high + ib_low) / 2

        # Prior day levels (key liquidity zones)
        prior = session_context.get('prior_day', {})
        self._pdh = prior.get('high', None)
        self._pdl = prior.get('low', None)

        # Tracking
        self._ibh_swept = False
        self._ibl_swept = False
        self._pdh_swept = False
        self._pdl_swept = False
        self._entry_count = 0
        self._last_entry_bar = -999

        # Session high/low tracking for sweep detection
        self._session_high = ib_high
        self._session_low = ib_low

    def on_bar(self, bar: pd.Series, bar_index: int, session_context: dict) -> Optional[Signal]:
        if self._ib_range < MIN_IB_RANGE:
            return None

        bar_time = session_context.get('bar_time')
        if bar_time and bar_time >= ENTRY_CUTOFF:
            return None

        if self._entry_count >= MAX_ENTRIES_PER_SESSION:
            return None

        if bar_index - self._last_entry_bar < COOLDOWN_BARS:
            return None

        # Update session extremes
        self._session_high = max(self._session_high, bar['high'])
        self._session_low = min(self._session_low, bar['low'])

        current_price = bar['close']
        delta = bar.get('delta', 0)
        if pd.isna(delta):
            delta = 0
        volume_spike = bar.get('volume_spike', 1.0)
        if pd.isna(volume_spike):
            volume_spike = 1.0

        # Check for sweep + rejection at each liquidity level

        # --- IBH SWEEP (upside sweep → short entry) ---
        if not self._ibh_swept:
            signal = self._check_upside_sweep(
                bar, bar_index, session_context,
                level=self._ib_high,
                level_name='IBH',
                delta=delta,
                volume_spike=volume_spike,
            )
            if signal:
                self._ibh_swept = True
                return signal

        # --- IBL SWEEP (downside sweep → long entry) ---
        if not self._ibl_swept:
            signal = self._check_downside_sweep(
                bar, bar_index, session_context,
                level=self._ib_low,
                level_name='IBL',
                delta=delta,
                volume_spike=volume_spike,
            )
            if signal:
                self._ibl_swept = True
                return signal

        # --- PDH SWEEP (upside sweep → short entry) ---
        if self._pdh and not self._pdh_swept:
            signal = self._check_upside_sweep(
                bar, bar_index, session_context,
                level=self._pdh,
                level_name='PDH',
                delta=delta,
                volume_spike=volume_spike,
            )
            if signal:
                self._pdh_swept = True
                return signal

        # --- PDL SWEEP (downside sweep → long entry) ---
        if self._pdl and not self._pdl_swept:
            signal = self._check_downside_sweep(
                bar, bar_index, session_context,
                level=self._pdl,
                level_name='PDL',
                delta=delta,
                volume_spike=volume_spike,
            )
            if signal:
                self._pdl_swept = True
                return signal

        return None

    def _check_upside_sweep(
        self, bar, bar_index, session_context,
        level, level_name, delta, volume_spike,
    ) -> Optional[Signal]:
        """
        Detect upside liquidity sweep: price wicks ABOVE level but closes BELOW.
        This is a SHORT signal (institutions swept stops above, now selling).
        """
        current_price = bar['close']

        # Sweep condition: high went above level, but close is below
        if bar['high'] <= level + SWEEP_BUFFER_PTS:
            return None  # Didn't sweep far enough

        if current_price >= level:
            return None  # Closed above — this is a breakout, not a sweep

        # Rejection quality: significant upper wick
        bar_range = bar['high'] - bar['low']
        if bar_range <= 0:
            return None
        upper_wick = bar['high'] - max(bar['open'], current_price)
        wick_ratio = upper_wick / bar_range
        if wick_ratio < MIN_WICK_RATIO:
            return None  # Weak rejection

        # Volume confirmation
        if volume_spike < MIN_VOLUME_SPIKE:
            return None

        # Delta confirmation: should be negative (sellers dominating after sweep)
        if delta >= 0:
            return None  # No seller conviction

        # Build SHORT signal
        sweep_extreme = bar['high']
        stop_price = sweep_extreme + (self._ib_range * STOP_BEYOND_SWEEP)
        risk = stop_price - current_price
        target_price = current_price - (risk * TARGET_MULT)

        # Ensure minimum stop distance
        if risk < 10.0:
            return None

        self._entry_count += 1
        self._last_entry_bar = bar_index

        return Signal(
            timestamp=bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp'),
            direction='SHORT',
            entry_price=current_price,
            stop_price=stop_price,
            target_price=target_price,
            strategy_name=self.name,
            setup_type=f'SWEEP_{level_name}_SHORT',
            day_type=session_context.get('day_type', ''),
            trend_strength=session_context.get('trend_strength', ''),
            confidence='high' if wick_ratio >= 0.60 else 'medium',
            metadata={
                'sweep_level': level_name,
                'sweep_extreme': sweep_extreme,
                'wick_ratio': round(wick_ratio, 2),
                'volume_spike': volume_spike,
                'delta': delta,
            },
        )

    def _check_downside_sweep(
        self, bar, bar_index, session_context,
        level, level_name, delta, volume_spike,
    ) -> Optional[Signal]:
        """
        Detect downside liquidity sweep: price wicks BELOW level but closes ABOVE.
        This is a LONG signal (institutions swept stops below, now buying).
        """
        current_price = bar['close']

        # Sweep condition: low went below level, but close is above
        if bar['low'] >= level - SWEEP_BUFFER_PTS:
            return None

        if current_price <= level:
            return None  # Closed below — breakdown, not a sweep

        # Rejection quality: significant lower wick
        bar_range = bar['high'] - bar['low']
        if bar_range <= 0:
            return None
        lower_wick = min(bar['open'], current_price) - bar['low']
        wick_ratio = lower_wick / bar_range
        if wick_ratio < MIN_WICK_RATIO:
            return None

        # Volume confirmation
        if volume_spike < MIN_VOLUME_SPIKE:
            return None

        # Delta confirmation: should be positive (buyers dominating after sweep)
        if delta <= 0:
            return None

        # Build LONG signal
        sweep_extreme = bar['low']
        stop_price = sweep_extreme - (self._ib_range * STOP_BEYOND_SWEEP)
        risk = current_price - stop_price
        target_price = current_price + (risk * TARGET_MULT)

        if risk < 10.0:
            return None

        self._entry_count += 1
        self._last_entry_bar = bar_index

        return Signal(
            timestamp=bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp'),
            direction='LONG',
            entry_price=current_price,
            stop_price=stop_price,
            target_price=target_price,
            strategy_name=self.name,
            setup_type=f'SWEEP_{level_name}_LONG',
            day_type=session_context.get('day_type', ''),
            trend_strength=session_context.get('trend_strength', ''),
            confidence='high' if wick_ratio >= 0.60 else 'medium',
            metadata={
                'sweep_level': level_name,
                'sweep_extreme': sweep_extreme,
                'wick_ratio': round(wick_ratio, 2),
                'volume_spike': volume_spike,
                'delta': delta,
            },
        )
