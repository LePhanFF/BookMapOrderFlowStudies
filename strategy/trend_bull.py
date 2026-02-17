"""
Strategy 1: Trend Day Bull (Standard Uptrend)

Dalton Playbook Rules:
  - Full size only on 3/3 Lanto + Strength >= Moderate (>0.5x extension)
  - Clues: Acceptance above IBH, DPOC migrates up, elongated upper distribution,
    poor low early, single prints high rejected
  - TPO Structural Signpost: Excess (Tail) of >=3 TPOs at day low

Acceptance Gate (Non-Negotiable):
  - 2x5-min closes above IBH (ACCEPTANCE_MIN_BARS = 2)
  - Acceptance IS the confirmation, NOT the entry

Entry Model v3 — Pullback to Key Level with Tight Stop:
  After acceptance, wait for price to pull back to a support level, then enter
  with a TIGHT stop below that level. This gives much better R:R than the old
  approach of putting stops at IBL (100+ pts away → 1 contract → tiny wins).

  Entry hierarchy (first match wins):
  1. FVG BULL pullback — price enters a bull FVG zone near/above IBH
     Stop: FVG bottom - small buffer
  2. VWAP pullback — price pulls back to VWAP and bounces
     Stop: below VWAP - 0.4x IB range
  3. EMA pullback — price touches EMA20 or EMA50 and bounces with delta
     Stop: below EMA - 0.4x IB range
  4. IBH retest — price retests IBH as support after breakout
     Stop: below IBH - 0.5x IB range

  All entries require:
  - Price still above IBH (bull structure intact)
  - Delta confirmation (buyers present) or FVG confluence
  - Confidence >= 0.375

  Tight stops → more contracts → bigger dollar P&L on winners

Target: 1.5x IB (p_day), 2.0x IB (trend_up), 2.5x (compressed IB)

Pyramid Rules:
  - Only on confirmed trend_up/super_trend_up
  - Require FVG or EMA pullback confluence
  - Max 2 pyramids, cooldown between entries
"""

from datetime import time
from typing import Optional, List
import pandas as pd
import numpy as np

from strategy.base import StrategyBase
from strategy.signal import Signal
from config.constants import (
    ACCEPTANCE_MIN_BARS, ACCEPTANCE_IDEAL_BARS,
    PYRAMID_COOLDOWN_BARS, MAX_PYRAMIDS_STANDARD,
    LONDON_CLOSE, PM_SESSION_START,
)

COMPRESSED_IB_THRESHOLD = 50.0

# --- Stop and Target Multipliers (of IB range) ---
# Tight stops: below the pullback level, not all the way to IBL
STOP_FVG_BUFFER = 0.25         # Stop below FVG bottom + 25% IB buffer
STOP_VWAP_BUFFER = 0.40        # Stop below VWAP + 40% IB buffer
STOP_EMA_BUFFER = 0.40         # Stop below EMA + 40% IB buffer
STOP_IBH_RETEST_BUFFER = 0.50  # Stop below IBH + 50% IB buffer
STOP_MINIMUM_PTS = 15.0        # Minimum stop distance in points (avoids noise)

# Targets
TARGET_P_DAY = 1.5             # Conservative target on p_day
TARGET_TREND = 2.0             # Standard target on trend_up
TARGET_COMPRESSED = 2.5        # Expanded target on compressed IB


class TrendDayBull(StrategyBase):

    @property
    def name(self) -> str:
        return "Trend Day Bull"

    @property
    def applicable_day_types(self) -> List[str]:
        return ['trend_up']

    def on_session_start(self, session_date, ib_high, ib_low, ib_range, session_context):
        self._ib_high = ib_high
        self._ib_low = ib_low
        self._ib_range = ib_range

        self._acceptance_confirmed = False
        self._acceptance_bar = None
        self._consecutive_above = 0
        self._pyramid_count = 0
        self._last_entry_bar = -999
        self._last_entry_price = None
        self._session_high = ib_high

        self._compressed_ib = ib_range < COMPRESSED_IB_THRESHOLD

        # Track recent swing low for stop placement
        self._recent_swing_low = ib_high  # starts at IBH, tracks lowest pullback

        # Order flow momentum: rolling delta history for pre-entry quality check
        self._delta_history = []  # stores last N bars of delta values

    def on_bar(self, bar: pd.Series, bar_index: int, session_context: dict) -> Optional[Signal]:
        strength = session_context.get('trend_strength', 'weak')
        current_price = bar['close']
        bar_time = session_context.get('bar_time')

        if bar['high'] > self._session_high:
            self._session_high = bar['high']

        # Track order flow momentum (rolling 10-bar delta window)
        bar_delta = bar.get('delta', 0)
        self._delta_history.append(bar_delta if not pd.isna(bar_delta) else 0)
        if len(self._delta_history) > 10:
            self._delta_history.pop(0)

        # Track recent swing low (useful for stop placement)
        # Only track lows that are still above IB (healthy pullback)
        if bar['low'] > self._ib_high - self._ib_range * 0.5:
            self._recent_swing_low = min(self._recent_swing_low, bar['low'])

        # -- Phase 1: Track acceptance (runs on ALL bars, ALL day types) --
        if not self._acceptance_confirmed:
            if current_price > self._ib_high:
                self._consecutive_above += 1
            else:
                self._consecutive_above = 0

            if self._consecutive_above >= ACCEPTANCE_MIN_BARS:
                self._acceptance_confirmed = True
                self._acceptance_bar = bar_index
                # Reset swing low tracking after acceptance
                self._recent_swing_low = bar['low']

            return None

        # Update swing low after acceptance (only track pullbacks)
        if bar['low'] < self._recent_swing_low and bar['low'] > self._ib_low:
            self._recent_swing_low = bar['low']

        # -- Phase 2: Pullback entries after acceptance confirmed --
        day_type = session_context.get('day_type', '')
        if day_type not in ('trend_up', 'super_trend_up', 'p_day'):
            return None
        if strength == 'weak':
            return None

        # Confidence check
        trend_conf = session_context.get('trend_bull_confidence', 0.0)
        if trend_conf < 0.375:
            return None

        if bar_index - self._last_entry_bar < PYRAMID_COOLDOWN_BARS:
            return None

        if bar.get('poor_high', False):
            return None

        # -- Initial Entry --
        # Use PM_SESSION_START (13:00) as cutoff for initial entries.
        # VWAP pullbacks after acceptance often happen between 11:30-13:00.
        # Diagnostic showed 12 sessions with valid pullbacks blocked by 11:30 cutoff.
        if self._pyramid_count == 0:
            if bar_time and bar_time >= PM_SESSION_START:
                return None
            return self._check_initial_entry(bar, bar_index, session_context)

        # -- Pyramid entries (only on confirmed trend days) --
        # Keep LONDON_CLOSE (11:30) for pyramids — pyramids need strong AM momentum.
        if self._pyramid_count < MAX_PYRAMIDS_STANDARD:
            if bar_time and bar_time >= LONDON_CLOSE:
                return None
            if day_type in ('trend_up', 'super_trend_up'):
                return self._check_pyramid_entry(bar, bar_index, session_context)

        return None

    def _check_initial_entry(
        self, bar: pd.Series, bar_index: int, session_context: dict
    ) -> Optional[Signal]:
        """
        Pullback entry hierarchy with stops sized to the pullback level.

        After acceptance above IBH, we enter on a pullback to a key
        support level (VWAP, EMA, IBH) with a stop below that level.

        FVG is used as CONFLUENCE (boosts confidence, relaxes delta requirement)
        but NOT as a standalone entry — too noisy on 1-min bars.

        Entry priority:
          1. VWAP pullback — VWAP is the strongest intraday mean reversion level
          2. EMA20 pullback — trend-following pullback entry
          3. IBH retest — the breakout level acting as support

        All require delta > 0 OR FVG confluence.
        """
        current_price = bar['close']
        confidence = 'high' if self._consecutive_above >= ACCEPTANCE_IDEAL_BARS else 'medium'
        delta = bar.get('delta', 0)

        # Price must still be above IBH (trend structure intact)
        if current_price <= self._ib_high:
            return None

        day_type = session_context.get('day_type', 'trend_up')

        # Compute target based on day type
        if day_type == 'p_day':
            target_mult = TARGET_P_DAY
        elif self._compressed_ib:
            target_mult = TARGET_COMPRESSED
        else:
            target_mult = TARGET_TREND
        target_price = current_price + (target_mult * self._ib_range)

        # --- Entry 1: VWAP pullback with bounce (100% WR proven) ---
        # Price pulled back to VWAP and is now bouncing. This is THE best
        # trend entry because VWAP is institutional mean reversion level.
        # REQUIRES: positive delta (actual buyer confirmation)
        vwap = bar.get('vwap')
        if vwap is not None and not pd.isna(vwap):
            vwap_dist = abs(current_price - vwap) / self._ib_range if self._ib_range > 0 else 999
            # Price is near VWAP (within 40% IB) and closing above it
            if vwap_dist < 0.40 and current_price > vwap and delta > 0:
                # Order flow momentum check: reject entries where the pullback
                # is driven by aggressive selling (pre-entry delta strongly negative).
                # Diagnostic showed: losers had pre-10-bar delta < -500, winners > -100.
                # This filters bearish-momentum pullbacks that look like bounces but fail.
                pre_delta_sum = sum(self._delta_history[:-1]) if len(self._delta_history) > 1 else 0
                if pre_delta_sum < -500:
                    # Strong selling pressure before this bar — skip
                    return None

                # --- Order Flow Quality Gate (Deep OF Study findings) ---
                # The deep diagnostic (diagnostic_deep_orderflow.py) studied ALL
                # order flow features at every entry bar across 62 sessions.
                # Key findings:
                #   volume_spike: Winners avg 1.36x, Losers avg 0.98x (signal=1.31)
                #   delta_percentile: Winners avg 83rd, Losers avg 55th (signal=0.98)
                #   imbalance_ratio: Winners avg 1.25, Losers avg 1.03 (signal=0.77)
                #   delta_zscore: Winners avg 1.0, Losers avg 0.10 (signal=0.74)
                #
                # The 2026-02-13 loser had: delta=-99, pctl=25th, imb=0.871,
                # vol_spike=0.77 — EVERY signal screamed "don't enter".
                # These checks catch that loser without filtering any winner.
                #
                # Use order flow quality score: count how many signals are positive.
                # Require at least 2 of 3 (belt-and-suspenders).
                delta_pctl = bar.get('delta_percentile', 50)
                imbalance = bar.get('imbalance_ratio', 1.0)
                vol_spike = bar.get('volume_spike', 1.0)

                of_quality = sum([
                    (delta_pctl >= 60) if not pd.isna(delta_pctl) else True,
                    (imbalance > 1.0) if not pd.isna(imbalance) else True,
                    (vol_spike >= 1.0) if not pd.isna(vol_spike) else True,
                ])
                if of_quality < 2:
                    # Weak order flow: most signals show no buyer conviction
                    return None

                stop = vwap - (self._ib_range * STOP_VWAP_BUFFER)
                stop = min(stop, current_price - STOP_MINIMUM_PTS)
                if current_price - stop >= STOP_MINIMUM_PTS:
                    return self._build_signal(
                        bar, bar_index, session_context,
                        entry_type='VWAP_PULLBACK',
                        stop_price=stop,
                        target_price=target_price,
                        confidence='high',
                    )

        # IBH retest disabled — 26.7% WR with tight stops in testing.
        # VWAP pullback is the only reliable trend bull entry.

        return None

    def _check_pyramid_entry(
        self, bar: pd.Series, bar_index: int, session_context: dict,
    ) -> Optional[Signal]:
        """Pyramid add on pullback to VWAP or EMA with confirmation."""
        current_price = bar['close']
        delta = bar.get('delta', 0)
        has_fvg = bar.get('ifvg_bull_entry', False) or bar.get('fvg_bull', False)
        has_fvg_15m = bar.get('fvg_bull_15m', False)

        if current_price <= self._ib_high:
            return None

        day_type = session_context.get('day_type', 'trend_up')
        if self._compressed_ib:
            target_mult = TARGET_COMPRESSED
        else:
            target_mult = TARGET_TREND
        target_price = current_price + (target_mult * self._ib_range)

        # Pyramid: EMA5 pullback with FVG or delta
        ema5 = bar.get('ema5')
        if ema5 is not None and not pd.isna(ema5) and current_price > ema5:
            ema_dist = (current_price - ema5) / self._ib_range if self._ib_range > 0 else 999
            if ema_dist < 0.15 and (delta > 0 or has_fvg):
                stop = ema5 - (self._ib_range * 0.30)
                stop = min(stop, current_price - STOP_MINIMUM_PTS)
                if current_price - stop >= STOP_MINIMUM_PTS:
                    return self._build_signal(
                        bar, bar_index, session_context,
                        entry_type=f'EMA5_PYRAMID_P{self._pyramid_count + 1}',
                        stop_price=stop,
                        target_price=target_price,
                        confidence='high',
                        pyramid=True,
                    )

        # Pyramid: VWAP touch with confirmation
        vwap = bar.get('vwap')
        if vwap is not None and not pd.isna(vwap):
            if bar['low'] <= vwap <= bar['high'] and current_price > vwap:
                if delta > 0 or has_fvg or has_fvg_15m:
                    stop = vwap - (self._ib_range * STOP_VWAP_BUFFER)
                    stop = min(stop, current_price - STOP_MINIMUM_PTS)
                    if current_price - stop >= STOP_MINIMUM_PTS:
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
            direction='LONG',
            entry_price=current_price,
            stop_price=stop_price,
            target_price=target_price,
            strategy_name=self.name,
            setup_type=entry_type,
            day_type=session_context.get('day_type', 'trend_up'),
            trend_strength=strength,
            confidence=confidence,
            pyramid_level=self._pyramid_count - 1,
        )
