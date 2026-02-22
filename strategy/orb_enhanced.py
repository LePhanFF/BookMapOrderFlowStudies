"""
Strategy: Enhanced Opening Range Breakout (ORB v2)
====================================================

Redesign of the original ORB_VWAP_BREAKOUT with research-backed improvements:

1. IB Width Classification → Adaptive targets and filters
   - NARROW IB: Aggressive breakout targets (2x IB), lower volume threshold
   - NORMAL IB: Standard targets (1.5x IB)
   - WIDE IB:   Skip breakouts, only trade sweep reversals

2. C-Period Close Location Rule (70-75% edge)
   - C-period closes outside IB → confirm breakout direction
   - C-period closes inside IB → fade the move

3. FVG Confluence at ORB Levels
   - Breakout + FVG retest = high-probability entry zone
   - Entry at FVG zone with tighter stop below FVG bottom

4. Multi-Bar Sweep Detection
   - Enhanced liquidity sweep that spans 2-3 bars
   - Sweep + rejection + delta reversal = fade entry

5. SMT Divergence Confirmation (when available)
   - Confirm breakout quality: both instruments breaking = genuine
   - Divergence against direction = skip or fade

6. LONG-ONLY by default (NQ structural long bias)
   - SHORT only on sweep reversals with multiple confirmations

ENTRY MODELS:
  A. Breakout Entry (primary):
     - Price closes above IBH with volume, delta, VWAP confirmation
     - Adaptive targets based on IB width class
     - C-period bias must align (or be neutral)

  B. FVG Retest Entry (secondary):
     - After breakout, price pulls back into FVG zone at IBH
     - Tighter stop = better R:R

  C. Sweep Reversal Entry (counter-trend):
     - Price sweeps IBH/IBL, closes back inside
     - Multi-bar detection (up to 3 bars)
     - Delta reversal + wick ratio confirmation

EXIT MODEL:
  - Adaptive targets based on IB width:
    NARROW: 2.0x IB range (trend day expectation)
    NORMAL: 1.5x IB range
    WIDE: 1.0x IB range (conservative)
  - Stop at IB midpoint (breakout) or beyond sweep extreme (sweep)
  - Trail to breakeven after 1.0x IB move
  - EOD close by 15:30
"""

from datetime import time
from typing import Optional, List
import pandas as pd
import numpy as np

from strategy.base import StrategyBase
from strategy.signal import Signal


# --- Breakout Parameters ---
MIN_VOLUME_SPIKE_NORMAL = 1.3      # Breakout volume (normal IB)
MIN_VOLUME_SPIKE_NARROW = 1.1      # Lower threshold for narrow IB (compressed → any volume = significant)
MIN_CANDLE_STRENGTH = 0.55         # Close in upper/lower 55% of bar
MIN_DELTA_THRESHOLD = 0            # Delta must align with direction
BREAKOUT_COOLDOWN_BARS = 5
MAX_ENTRIES_PER_SESSION = 3        # Increased from 2 → allow breakout + FVG retest + sweep
ENTRY_CUTOFF = time(13, 0)
MIN_IB_RANGE = 15.0                # Slightly lower than before (narrow IB may be <20)
MAX_IB_RANGE = 120.0               # Skip abnormally wide IB (news-driven)

# --- Adaptive Target Multipliers ---
TARGET_MULT_NARROW = 2.0           # Narrow IB → trend day → target 2x IB
TARGET_MULT_NORMAL = 1.5           # Standard target
TARGET_MULT_WIDE = 1.0             # Wide IB → conservative

# --- Sweep Parameters ---
SWEEP_BUFFER_PTS = 3.0             # Min points past level to qualify as sweep
SWEEP_MAX_BARS = 3                 # Multi-bar sweep window
MIN_WICK_RATIO = 0.35              # Rejection wick quality
SWEEP_MIN_VOLUME = 1.0             # Lower volume bar for sweeps (already has institutional activity)
SWEEP_TARGET_MULT = 1.5            # Sweep target = 1.5x risk

# --- FVG Retest Parameters ---
FVG_RETEST_WINDOW = 15             # Max bars after breakout to look for FVG retest
FVG_STOP_BUFFER = 2.0              # Points below FVG bottom for stop

# --- Stop Parameters ---
STOP_BUFFER_MULT = 0.50            # Stop at IB midpoint
TRAIL_TRIGGER_MULT = 1.0           # Trail to BE after 1.0x IB move


class ORBEnhanced(StrategyBase):
    """
    Enhanced Opening Range Breakout with adaptive IB width classification,
    FVG confluence, multi-bar sweep detection, and C-period confirmation.
    """

    @property
    def name(self) -> str:
        return "ORB Enhanced"

    @property
    def applicable_day_types(self) -> List[str]:
        return []  # Trades on any day type

    def on_session_start(self, session_date, ib_high, ib_low, ib_range, session_context):
        self._ib_high = ib_high
        self._ib_low = ib_low
        self._ib_range = ib_range
        self._ib_mid = (ib_high + ib_low) / 2

        # IB width classification from session context
        self._ib_width = session_context.get('ib_width_class', 'NORMAL')
        self._ib_atr_ratio = session_context.get('ib_atr_ratio', 0.5)
        self._c_period_bias = session_context.get('c_period_bias', None)

        # Extension targets
        self._ext_targets = {
            'ext_1_0_high': session_context.get('ext_1_0_high', ib_high + ib_range),
            'ext_1_5_high': session_context.get('ext_1_5_high', ib_high + 1.5 * ib_range),
            'ext_2_0_high': session_context.get('ext_2_0_high', ib_high + 2.0 * ib_range),
            'ext_1_0_low': session_context.get('ext_1_0_low', ib_low - ib_range),
            'ext_1_5_low': session_context.get('ext_1_5_low', ib_low - 1.5 * ib_range),
            'ext_2_0_low': session_context.get('ext_2_0_low', ib_low - 2.0 * ib_range),
        }

        # State
        self._breakout_up = False
        self._breakout_down = False
        self._sweep_up = False
        self._sweep_down = False
        self._fvg_retest_long = False
        self._entry_count = 0
        self._last_entry_bar = -999
        self._breakout_bar_idx = -999  # Track when breakout happened for FVG retest

        # Multi-bar sweep tracking
        self._ibh_penetration_bars = 0  # Count bars above IBH
        self._ibl_penetration_bars = 0
        self._ibh_sweep_high = 0        # Track sweep extreme
        self._ibl_sweep_low = float('inf')

    def on_bar(self, bar: pd.Series, bar_index: int, session_context: dict) -> Optional[Signal]:
        # Skip sessions with unsuitable IB
        if self._ib_range < MIN_IB_RANGE or self._ib_range > MAX_IB_RANGE:
            return None

        bar_time = session_context.get('bar_time')
        if bar_time and bar_time >= ENTRY_CUTOFF:
            return None

        if self._entry_count >= MAX_ENTRIES_PER_SESSION:
            return None

        if bar_index - self._last_entry_bar < BREAKOUT_COOLDOWN_BARS:
            return None

        current_price = bar['close']
        bar_range = bar['high'] - bar['low']
        delta = bar.get('delta', 0)
        if pd.isna(delta):
            delta = 0
        volume_spike = bar.get('volume_spike', 1.0)
        if pd.isna(volume_spike):
            volume_spike = 1.0
        vwap = bar.get('vwap')

        # Get FVG info
        has_bull_fvg = bar.get('ifvg_bull_entry', False)
        if pd.isna(has_bull_fvg):
            has_bull_fvg = False
        fvg_bull_bottom = bar.get('fvg_bull_bottom', np.nan)

        has_bear_fvg = bar.get('ifvg_bear_entry', False)
        if pd.isna(has_bear_fvg):
            has_bear_fvg = False

        # Get SMT info (if computed)
        smt_bullish = bar.get('smt_bullish', False)
        smt_bearish = bar.get('smt_bearish', False)
        if pd.isna(smt_bullish):
            smt_bullish = False
        if pd.isna(smt_bearish):
            smt_bearish = False

        # --- Track multi-bar sweep state ---
        self._track_sweep_state(bar)

        # --- ENTRY A: LONG BREAKOUT ---
        if not self._breakout_up and current_price > self._ib_high:
            signal = self._check_breakout_long(
                bar, bar_index, session_context,
                current_price, bar_range, delta, volume_spike, vwap,
                smt_bearish,
            )
            if signal:
                self._breakout_up = True
                self._breakout_bar_idx = bar_index
                return signal

        # --- ENTRY B: FVG RETEST LONG (after breakout) ---
        if (self._breakout_up and not self._fvg_retest_long
                and bar_index - self._breakout_bar_idx <= FVG_RETEST_WINDOW):
            signal = self._check_fvg_retest_long(
                bar, bar_index, session_context,
                current_price, delta, has_bull_fvg, fvg_bull_bottom,
            )
            if signal:
                self._fvg_retest_long = True
                return signal

        # --- ENTRY C: SWEEP REVERSAL LONG (IBL sweep) ---
        if not self._sweep_down:
            signal = self._check_sweep_long(
                bar, bar_index, session_context,
                current_price, bar_range, delta, volume_spike,
                smt_bullish,
            )
            if signal:
                self._sweep_down = True
                return signal

        # --- ENTRY D: SWEEP REVERSAL SHORT (IBH sweep) ---
        # Only on wide IB days or when c_period bias is bearish
        if not self._sweep_up:
            signal = self._check_sweep_short(
                bar, bar_index, session_context,
                current_price, bar_range, delta, volume_spike,
                smt_bearish,
            )
            if signal:
                self._sweep_up = True
                return signal

        return None

    def _track_sweep_state(self, bar):
        """Track multi-bar penetrations for sweep detection."""
        # IBH penetration tracking
        if bar['high'] > self._ib_high + SWEEP_BUFFER_PTS:
            self._ibh_penetration_bars += 1
            self._ibh_sweep_high = max(self._ibh_sweep_high, bar['high'])
        else:
            if bar['close'] < self._ib_high:
                # Price closed back inside — potential sweep completed
                pass  # Let sweep check handle it
            self._ibh_penetration_bars = 0
            self._ibh_sweep_high = 0

        # IBL penetration tracking
        if bar['low'] < self._ib_low - SWEEP_BUFFER_PTS:
            self._ibl_penetration_bars += 1
            self._ibl_sweep_low = min(self._ibl_sweep_low, bar['low'])
        else:
            if bar['close'] > self._ib_low:
                pass
            self._ibl_penetration_bars = 0
            self._ibl_sweep_low = float('inf')

    def _get_adaptive_target_mult(self) -> float:
        """Get target multiplier based on IB width class."""
        if self._ib_width == 'NARROW':
            return TARGET_MULT_NARROW
        elif self._ib_width == 'WIDE':
            return TARGET_MULT_WIDE
        else:
            return TARGET_MULT_NORMAL

    def _get_volume_threshold(self) -> float:
        """Get volume spike threshold based on IB width."""
        if self._ib_width == 'NARROW':
            return MIN_VOLUME_SPIKE_NARROW
        return MIN_VOLUME_SPIKE_NORMAL

    def _check_breakout_long(
        self, bar, bar_index, session_context,
        current_price, bar_range, delta, volume_spike, vwap,
        smt_bearish,
    ) -> Optional[Signal]:
        """Check for long breakout with adaptive filters."""
        # Skip breakouts on WIDE IB days (play the range instead)
        if self._ib_width == 'WIDE':
            return None

        # C-period bias check: don't go long if C-period confirmed bearish
        if self._c_period_bias == 'BEAR':
            return None

        # SMT divergence: if bearish SMT active, skip long breakout
        if smt_bearish:
            return None

        # Volume confirmation (adaptive threshold)
        vol_threshold = self._get_volume_threshold()
        if volume_spike < vol_threshold:
            return None

        # Delta confirmation
        if delta <= MIN_DELTA_THRESHOLD:
            return None

        # Candle strength
        if bar_range > 0:
            candle_strength = (current_price - bar['low']) / bar_range
            if candle_strength < MIN_CANDLE_STRENGTH:
                return None

        # VWAP confirmation
        if vwap is not None and not pd.isna(vwap):
            if current_price < vwap:
                return None

        # Adaptive target
        target_mult = self._get_adaptive_target_mult()
        target_price = current_price + (target_mult * self._ib_range)

        # Stop at IB midpoint
        stop_price = self._ib_mid
        stop_price = min(stop_price, current_price - 15.0)
        stop_price = min(stop_price, self._ib_high - 5.0)

        # Confidence
        confidence = 'high'
        if volume_spike >= 2.0:
            confidence = 'high'
        elif self._ib_width == 'NARROW' and self._c_period_bias == 'BULL':
            confidence = 'high'  # Narrow IB + C-period confirmation
        else:
            confidence = 'medium'

        self._entry_count += 1
        self._last_entry_bar = bar_index

        return Signal(
            timestamp=bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp'),
            direction='LONG',
            entry_price=current_price,
            stop_price=stop_price,
            target_price=target_price,
            strategy_name=self.name,
            setup_type=f'ORB_BREAKOUT_LONG_{self._ib_width}',
            day_type=session_context.get('day_type', ''),
            trend_strength=session_context.get('trend_strength', 'moderate'),
            confidence=confidence,
            metadata={
                'volume_spike': round(volume_spike, 2),
                'delta': delta,
                'ib_range': self._ib_range,
                'ib_width': self._ib_width,
                'ib_atr_ratio': round(self._ib_atr_ratio, 2),
                'c_period_bias': self._c_period_bias,
                'target_mult': target_mult,
                'entry_type': 'BREAKOUT',
            },
        )

    def _check_fvg_retest_long(
        self, bar, bar_index, session_context,
        current_price, delta, has_bull_fvg, fvg_bull_bottom,
    ) -> Optional[Signal]:
        """
        After a breakout, check for FVG retest entry.
        Price pulls back into a bull FVG zone near IBH → tighter entry.
        """
        if not has_bull_fvg:
            return None

        if pd.isna(fvg_bull_bottom):
            return None

        # FVG must be near the IBH area (within 1x IB range of IBH)
        if fvg_bull_bottom < self._ib_high - self._ib_range:
            return None

        # Delta must still be positive on the retest bar
        if delta <= 0:
            return None

        # Price must be holding above FVG bottom
        if current_price <= fvg_bull_bottom:
            return None

        # FVG retest entry: tighter stop below FVG bottom
        stop_price = fvg_bull_bottom - FVG_STOP_BUFFER
        stop_price = min(stop_price, current_price - 10.0)

        target_mult = self._get_adaptive_target_mult()
        target_price = current_price + (target_mult * self._ib_range)

        self._entry_count += 1
        self._last_entry_bar = bar_index

        return Signal(
            timestamp=bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp'),
            direction='LONG',
            entry_price=current_price,
            stop_price=stop_price,
            target_price=target_price,
            strategy_name=self.name,
            setup_type='ORB_FVG_RETEST_LONG',
            day_type=session_context.get('day_type', ''),
            trend_strength=session_context.get('trend_strength', 'moderate'),
            confidence='high',
            metadata={
                'ib_range': self._ib_range,
                'ib_width': self._ib_width,
                'fvg_bottom': fvg_bull_bottom,
                'target_mult': target_mult,
                'entry_type': 'FVG_RETEST',
            },
        )

    def _check_sweep_long(
        self, bar, bar_index, session_context,
        current_price, bar_range, delta, volume_spike,
        smt_bullish,
    ) -> Optional[Signal]:
        """
        Detect IBL sweep reversal → LONG entry.
        Price wicks below IBL but closes back inside. Multi-bar detection.
        """
        # Check: has price been below IBL recently?
        sweep_active = (
            bar['low'] < self._ib_low - SWEEP_BUFFER_PTS or
            self._ibl_penetration_bars > 0
        )

        if not sweep_active:
            return None

        # Must close back inside IB
        if current_price <= self._ib_low:
            return None

        # Determine sweep extreme
        sweep_low = min(bar['low'], self._ibl_sweep_low)
        if sweep_low >= self._ib_low - SWEEP_BUFFER_PTS:
            return None

        # Check penetration isn't too long (>3 bars = real breakdown, not sweep)
        if self._ibl_penetration_bars > SWEEP_MAX_BARS:
            return None

        # Rejection wick quality
        if bar_range > 0:
            lower_wick = min(bar['open'], current_price) - bar['low']
            wick_ratio = lower_wick / bar_range
            if wick_ratio < MIN_WICK_RATIO:
                return None

        # Volume confirmation
        if volume_spike < SWEEP_MIN_VOLUME:
            return None

        # Delta confirmation: must be positive (buyers after sweep)
        if delta <= 0:
            return None

        # Build LONG signal
        stop_price = sweep_low - (self._ib_range * 0.10)
        stop_price = min(stop_price, current_price - 10.0)
        risk = current_price - stop_price
        target_price = current_price + (risk * SWEEP_TARGET_MULT)

        # Boost confidence if SMT confirms
        confidence = 'high' if smt_bullish else 'medium'

        self._entry_count += 1
        self._last_entry_bar = bar_index

        return Signal(
            timestamp=bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp'),
            direction='LONG',
            entry_price=current_price,
            stop_price=stop_price,
            target_price=target_price,
            strategy_name=self.name,
            setup_type='ORB_SWEEP_IBL_LONG',
            day_type=session_context.get('day_type', ''),
            trend_strength=session_context.get('trend_strength', ''),
            confidence=confidence,
            metadata={
                'sweep_extreme': sweep_low,
                'sweep_bars': self._ibl_penetration_bars,
                'volume_spike': round(volume_spike, 2),
                'delta': delta,
                'ib_range': self._ib_range,
                'ib_width': self._ib_width,
                'smt_confirmed': smt_bullish,
                'entry_type': 'SWEEP_REVERSAL',
            },
        )

    def _check_sweep_short(
        self, bar, bar_index, session_context,
        current_price, bar_range, delta, volume_spike,
        smt_bearish,
    ) -> Optional[Signal]:
        """
        Detect IBH sweep reversal → SHORT entry.
        Only allowed on WIDE IB days or with bearish C-period bias.
        """
        # Restrict short sweeps: need WIDE IB or bearish context
        if self._ib_width != 'WIDE' and self._c_period_bias != 'BEAR':
            return None

        # Check: has price been above IBH recently?
        sweep_active = (
            bar['high'] > self._ib_high + SWEEP_BUFFER_PTS or
            self._ibh_penetration_bars > 0
        )

        if not sweep_active:
            return None

        # Must close back inside IB
        if current_price >= self._ib_high:
            return None

        # Determine sweep extreme
        sweep_high = max(bar['high'], self._ibh_sweep_high)
        if sweep_high <= self._ib_high + SWEEP_BUFFER_PTS:
            return None

        # Too many bars above = real breakout
        if self._ibh_penetration_bars > SWEEP_MAX_BARS:
            return None

        # Rejection wick quality
        if bar_range > 0:
            upper_wick = bar['high'] - max(bar['open'], current_price)
            wick_ratio = upper_wick / bar_range
            if wick_ratio < MIN_WICK_RATIO:
                return None

        # Volume confirmation
        if volume_spike < SWEEP_MIN_VOLUME:
            return None

        # Delta confirmation: must be negative (sellers after sweep)
        if delta >= 0:
            return None

        # Build SHORT signal
        stop_price = sweep_high + (self._ib_range * 0.10)
        stop_price = max(stop_price, current_price + 10.0)
        risk = stop_price - current_price
        target_price = current_price - (risk * SWEEP_TARGET_MULT)

        confidence = 'high' if smt_bearish else 'medium'

        self._entry_count += 1
        self._last_entry_bar = bar_index

        return Signal(
            timestamp=bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp'),
            direction='SHORT',
            entry_price=current_price,
            stop_price=stop_price,
            target_price=target_price,
            strategy_name=self.name,
            setup_type='ORB_SWEEP_IBH_SHORT',
            day_type=session_context.get('day_type', ''),
            trend_strength=session_context.get('trend_strength', ''),
            confidence=confidence,
            metadata={
                'sweep_extreme': sweep_high,
                'sweep_bars': self._ibh_penetration_bars,
                'volume_spike': round(volume_spike, 2),
                'delta': delta,
                'ib_range': self._ib_range,
                'ib_width': self._ib_width,
                'smt_confirmed': smt_bearish,
                'entry_type': 'SWEEP_REVERSAL',
            },
        )
