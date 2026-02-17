"""
Strategy 3: Trend Day Bear (Standard Downtrend)
Mirror of Trend Day Bull v3 with tight stops on pullback entries.

Dalton Playbook Rules:
  - Acceptance below IBL, DPOC migrates down steadily
  - Elongated lower distribution, poor high early
  - TPO Structural Signpost: Excess (Tail) >=3 TPOs at day high

Acceptance Gate:
  - 3x5-min closes below IBL (ACCEPTANCE_MIN_BARS_BEAR = 3)
  - Bear breakdowns bounce more frequently — stricter acceptance

Entry Model v3 — Pullback to Key Level with Tight Stop:
  After acceptance, wait for price to rally back to resistance, then short
  with a TIGHT stop above that level.

  Entry hierarchy:
  1. FVG BEAR pullback — price enters a bear FVG zone near/below IBL
     Stop: FVG top + small buffer
  2. VWAP rejection — price rallies to VWAP and fails
     Stop: above VWAP + 0.4x IB range
  3. EMA rejection — price touches EMA20 and rejects with negative delta
     Stop: above EMA + 0.4x IB range
  4. IBL retest fail — price retests IBL as resistance after breakdown
     Stop: above IBL + 0.5x IB range

Target: 1.5x IB (p_day), 2.0x IB (trend_down), 2.5x (compressed IB)

Caution:
  - At VAL: do NOT add short if poor low forming
  - Watch liquidation volume drying post-14:00
"""

from datetime import time
from typing import Optional, List
import pandas as pd
import numpy as np

from strategy.base import StrategyBase
from strategy.signal import Signal
from config.constants import (
    ACCEPTANCE_MIN_BARS, ACCEPTANCE_MIN_BARS_BEAR, ACCEPTANCE_IDEAL_BARS,
    PYRAMID_COOLDOWN_BARS, MAX_PYRAMIDS_STANDARD,
    LONDON_CLOSE,
)

COMPRESSED_IB_THRESHOLD = 50.0

# --- Stop and Target Multipliers (of IB range) ---
STOP_FVG_BUFFER = 0.25
STOP_VWAP_BUFFER = 0.40
STOP_EMA_BUFFER = 0.40
STOP_IBL_RETEST_BUFFER = 0.50
STOP_MINIMUM_PTS = 15.0

# Targets
TARGET_P_DAY = 1.5
TARGET_TREND = 2.0
TARGET_COMPRESSED = 2.5


class TrendDayBear(StrategyBase):

    @property
    def name(self) -> str:
        return "Trend Day Bear"

    @property
    def applicable_day_types(self) -> List[str]:
        return ['trend_down']

    def on_session_start(self, session_date, ib_high, ib_low, ib_range, session_context):
        self._ib_high = ib_high
        self._ib_low = ib_low
        self._ib_range = ib_range

        self._acceptance_confirmed = False
        self._acceptance_bar = None
        self._consecutive_below = 0
        self._pyramid_count = 0
        self._last_entry_bar = -999
        self._last_entry_price = None
        self._session_low = ib_low

        self._compressed_ib = ib_range < COMPRESSED_IB_THRESHOLD

        # Track recent swing high for stop placement
        self._recent_swing_high = ib_low

    def on_bar(self, bar: pd.Series, bar_index: int, session_context: dict) -> Optional[Signal]:
        strength = session_context.get('trend_strength', 'weak')
        current_price = bar['close']
        bar_time = session_context.get('bar_time')

        if bar['low'] < self._session_low:
            self._session_low = bar['low']

        # Track recent swing high
        if bar['high'] < self._ib_low + self._ib_range * 0.5:
            self._recent_swing_high = max(self._recent_swing_high, bar['high'])

        # -- Phase 1: Track acceptance --
        if not self._acceptance_confirmed:
            if current_price < self._ib_low:
                self._consecutive_below += 1
            else:
                self._consecutive_below = 0

            if self._consecutive_below >= ACCEPTANCE_MIN_BARS_BEAR:
                self._acceptance_confirmed = True
                self._acceptance_bar = bar_index
                self._recent_swing_high = bar['high']

            return None

        # Update swing high after acceptance
        if bar['high'] > self._recent_swing_high and bar['high'] < self._ib_high:
            self._recent_swing_high = bar['high']

        # -- Phase 2: Pullback entries after acceptance --
        day_type = session_context.get('day_type', '')
        if day_type not in ('trend_down', 'super_trend_down', 'p_day'):
            return None
        if strength == 'weak':
            return None

        trend_conf = session_context.get('trend_bear_confidence', 0.0)
        if trend_conf < 0.375:
            return None

        if bar_time and bar_time >= LONDON_CLOSE:
            return None

        if bar_index - self._last_entry_bar < PYRAMID_COOLDOWN_BARS:
            return None

        if bar.get('poor_low', False):
            return None

        # -- Initial Entry --
        if self._pyramid_count == 0:
            return self._check_initial_entry(bar, bar_index, session_context)

        # -- Pyramid entries --
        if self._pyramid_count < MAX_PYRAMIDS_STANDARD:
            if day_type in ('trend_down', 'super_trend_down'):
                return self._check_pyramid_entry(bar, bar_index, session_context)

        return None

    def _check_initial_entry(
        self, bar: pd.Series, bar_index: int, session_context: dict,
    ) -> Optional[Signal]:
        """
        Pullback entry hierarchy for bear with stops sized to pullback level.

        FVG used as CONFLUENCE only (not standalone entry — too noisy on 1-min).

        Entry priority:
          1. VWAP rejection — strongest mean reversion level
          2. EMA20 rejection — trend pullback entry
          3. IBL retest fail — breakdown level acting as resistance
        """
        current_price = bar['close']
        confidence = 'high' if self._consecutive_below >= ACCEPTANCE_IDEAL_BARS else 'medium'
        delta = bar.get('delta', 0)

        # Price must still be below IBL (bear structure intact)
        if current_price >= self._ib_low:
            return None

        day_type = session_context.get('day_type', 'trend_down')

        # Compute target
        if day_type == 'p_day':
            target_mult = TARGET_P_DAY
        elif self._compressed_ib:
            target_mult = TARGET_COMPRESSED
        else:
            target_mult = TARGET_TREND
        target_price = current_price - (target_mult * self._ib_range)

        # --- Entry 1: VWAP rejection (best proven entry) ---
        # REQUIRES: negative delta (seller confirmation)
        vwap = bar.get('vwap')
        if vwap is not None and not pd.isna(vwap):
            vwap_dist = abs(current_price - vwap) / self._ib_range if self._ib_range > 0 else 999
            if vwap_dist < 0.40 and current_price < vwap and delta < 0:
                stop = vwap + (self._ib_range * STOP_VWAP_BUFFER)
                stop = max(stop, current_price + STOP_MINIMUM_PTS)
                if stop - current_price >= STOP_MINIMUM_PTS:
                    return self._build_signal(
                        bar, bar_index, session_context,
                        entry_type='VWAP_REJECTION',
                        stop_price=stop,
                        target_price=target_price,
                        confidence='high',
                    )

        # IBL retest disabled — 37.5% WR with tight stops in testing.
        # VWAP rejection is the only reliable trend bear entry.

        return None

    def _check_pyramid_entry(
        self, bar: pd.Series, bar_index: int, session_context: dict,
    ) -> Optional[Signal]:
        """Pyramid on pullback to VWAP or EMA with confirmation."""
        current_price = bar['close']
        delta = bar.get('delta', 0)
        has_fvg = bar.get('ifvg_bear_entry', False) or bar.get('fvg_bear', False)
        has_fvg_15m = bar.get('fvg_bear_15m', False)

        if current_price >= self._ib_low:
            return None

        if self._compressed_ib:
            target_mult = TARGET_COMPRESSED
        else:
            target_mult = TARGET_TREND
        target_price = current_price - (target_mult * self._ib_range)

        # Pyramid: EMA5 rejection
        ema5 = bar.get('ema5')
        if ema5 is not None and not pd.isna(ema5) and current_price < ema5:
            ema_dist = (ema5 - current_price) / self._ib_range if self._ib_range > 0 else 999
            if ema_dist < 0.15 and (delta < 0 or has_fvg):
                stop = ema5 + (self._ib_range * 0.30)
                stop = max(stop, current_price + STOP_MINIMUM_PTS)
                if stop - current_price >= STOP_MINIMUM_PTS:
                    return self._build_signal(
                        bar, bar_index, session_context,
                        entry_type=f'EMA5_PYRAMID_P{self._pyramid_count + 1}',
                        stop_price=stop,
                        target_price=target_price,
                        confidence='high',
                        pyramid=True,
                    )

        # Pyramid: VWAP rejection
        vwap = bar.get('vwap')
        if vwap is not None and not pd.isna(vwap):
            if bar['low'] <= vwap <= bar['high'] and current_price < vwap:
                if delta < 0 or has_fvg or has_fvg_15m:
                    stop = vwap + (self._ib_range * STOP_VWAP_BUFFER)
                    stop = max(stop, current_price + STOP_MINIMUM_PTS)
                    if stop - current_price >= STOP_MINIMUM_PTS:
                        return self._build_signal(
                            bar, bar_index, session_context,
                            entry_type=f'VWAP_PYRAMID_P{self._pyramid_count + 1}',
                            stop_price=stop,
                            target_price=target_price,
                            confidence='high',
                            pyramid=True,
                        )

        return None

    def _build_signal(
        self, bar: pd.Series, bar_index: int, session_context: dict,
        entry_type: str, stop_price: float, target_price: float,
        confidence: str, pyramid: bool = False,
    ) -> Signal:
        """Build a signal with the given stop and target."""
        current_price = bar['close']
        strength = session_context.get('trend_strength', 'weak')

        if pyramid:
            self._pyramid_count += 1
        else:
            self._pyramid_count = 1

        self._last_entry_bar = bar_index
        self._last_entry_price = current_price

        return Signal(
            timestamp=bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp'),
            direction='SHORT',
            entry_price=current_price,
            stop_price=stop_price,
            target_price=target_price,
            strategy_name=self.name,
            setup_type=entry_type,
            day_type=session_context.get('day_type', 'trend_down'),
            trend_strength=strength,
            confidence=confidence,
            pyramid_level=self._pyramid_count - 1,
        )
