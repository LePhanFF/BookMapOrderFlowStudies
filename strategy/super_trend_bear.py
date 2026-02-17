"""
Strategy 4: Super Trend Day Bear (Extreme Liquidation)
Mirror of Super Trend Bull.

Dalton Playbook Rules:
  - DPOC < IBL early, extreme migration (>400 pts down)
  - Multiple LVN seam rejections (gift zone permanent after 2)
  - High-volume breakdown
  - TPO "Gift Seam" Rule: Single-print low rejections = permanent gift adds
  - Velocity seams must remain unfilled

Acceptance Gate:
  - 2x5-min closes below IBL = acceptance confirmed
  - Immediate entry on acceptance bar

Entry Model:
  - Early: IBL breakdown acceptance (2x5-min + volume)
  - Add-ons: LVN seam retests, VWAP rejections (every rejection = add)
  - MAX 3 pyramids (aggressive mode)

Stops:
  - Initial: Above IBH (wide on liquidation days)
  - Trail: 30-min trail after +150 pts
"""

from datetime import time
from typing import Optional, List
import pandas as pd

from strategy.base import StrategyBase
from strategy.signal import Signal
from config.constants import (
    ACCEPTANCE_MIN_BARS, PYRAMID_COOLDOWN_BARS,
    MAX_PYRAMIDS_AGGRESSIVE, TREND_STOP_IB_BUFFER,
    EMA_PULLBACK_THRESHOLD, LONDON_CLOSE,
)


class SuperTrendBear(StrategyBase):

    @property
    def name(self) -> str:
        return "Super Trend Bear"

    @property
    def applicable_day_types(self) -> List[str]:
        return ['super_trend_down']

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

    def on_bar(self, bar: pd.Series, bar_index: int, session_context: dict) -> Optional[Signal]:
        strength = session_context.get('trend_strength', 'weak')
        current_price = bar['close']
        bar_time = session_context.get('bar_time')

        # ── Phase 1: Track acceptance (ALL bars, ALL day types) ──
        if not self._acceptance_confirmed:
            if current_price < self._ib_low:
                self._consecutive_below += 1
            else:
                self._consecutive_below = 0

            if self._consecutive_below >= ACCEPTANCE_MIN_BARS:
                self._acceptance_confirmed = True
                self._acceptance_bar = bar_index

                # Immediate entry on super trend acceptance
                day_type = session_context.get('day_type', '')
                if day_type in self.applicable_day_types and strength == 'super':
                    return self._build_signal(
                        bar, bar_index, session_context,
                        entry_type='IBL_BREAKDOWN_SUPER',
                        is_initial=True,
                    )

            return None

        # ── Phase 2: Only emit on super_trend_down + super strength ──
        day_type = session_context.get('day_type', '')
        if day_type not in self.applicable_day_types:
            return None
        if strength != 'super':
            return None

        if bar_time and bar_time >= LONDON_CLOSE:
            return None

        if bar_index - self._last_entry_bar < PYRAMID_COOLDOWN_BARS:
            return None

        # ── Initial entry if not taken on acceptance bar ──
        if self._pyramid_count == 0:
            if current_price < self._ib_low:
                bars_since = bar_index - self._acceptance_bar if self._acceptance_bar else 0
                if bars_since <= 8:
                    return self._build_signal(
                        bar, bar_index, session_context,
                        entry_type='IBL_BREAKDOWN_SUPER',
                        is_initial=True,
                    )
            return None

        # ── Aggressive pyramid adds (max 3) ──
        if self._pyramid_count < MAX_PYRAMIDS_AGGRESSIVE:
            return self._check_pyramid(bar, bar_index, session_context)

        return None

    def _check_pyramid(
        self, bar: pd.Series, bar_index: int, session_context: dict,
    ) -> Optional[Signal]:
        """Check for aggressive pyramid opportunities."""
        current_price = bar['close']

        # 1. VWAP rejection gift add
        vwap = bar.get('vwap')
        if vwap is not None and not pd.isna(vwap):
            if bar['low'] <= vwap <= bar['high'] and current_price < vwap:
                return self._build_signal(
                    bar, bar_index, session_context,
                    entry_type=f'VWAP_REJECTION_GIFT_P{self._pyramid_count + 1}',
                    is_initial=False,
                )

        # 2. EMA pullback (shallow on super days)
        ema20 = bar.get('ema20')
        if ema20 is not None and not pd.isna(ema20) and current_price < ema20:
            ema_dist = (ema20 - current_price) / self._ib_range if self._ib_range > 0 else 999
            if ema_dist < EMA_PULLBACK_THRESHOLD:
                return self._build_signal(
                    bar, bar_index, session_context,
                    entry_type=f'EMA_PULLBACK_P{self._pyramid_count + 1}',
                    is_initial=False,
                )

        # 3. IBL retest add (deep pullback = gift on super days)
        if bar['low'] <= self._ib_low <= bar['high'] and current_price < self._ib_low:
            return self._build_signal(
                bar, bar_index, session_context,
                entry_type=f'IBL_RETEST_GIFT_P{self._pyramid_count + 1}',
                is_initial=False,
            )

        return None

    def _build_signal(
        self, bar: pd.Series, bar_index: int, session_context: dict,
        entry_type: str, is_initial: bool,
    ) -> Signal:
        current_price = bar['close']

        if is_initial:
            stop_price = self._ib_high + (self._ib_range * TREND_STOP_IB_BUFFER)
            self._pyramid_count = 1
        else:
            if self._last_entry_price is not None:
                stop_price = min(
                    self._ib_high + (self._ib_range * TREND_STOP_IB_BUFFER),
                    self._last_entry_price + (self._ib_range * 0.25),
                )
            else:
                stop_price = self._ib_high + (self._ib_range * TREND_STOP_IB_BUFFER)
            self._pyramid_count += 1

        # Aggressive target: 3x IB range for super trend
        target_price = current_price - (3.0 * self._ib_range)

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
            day_type='super_trend_down',
            trend_strength='super',
            confidence='high',
            pyramid_level=self._pyramid_count - 1,
        )
