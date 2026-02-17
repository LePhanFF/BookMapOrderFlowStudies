"""
Strategy 2: Super Trend Day Bull (Explosive >300-400 pts)

Dalton Playbook Rules:
  - Full size + pyramid only on 3/3 + Strength = Super (>2.0x extension)
  - DPOC extreme migration (>300 pts), super-trending call early
  - DPOC above IBH by 10:30
  - Fattening upper, single prints rejected instantly, volume builds on breakout
  - Meets rigid 5/5 checklist PLUS extreme additives (migration >300 pts, wick parade >=8)

Acceptance Gate:
  - 2x5-min closes above IBH = acceptance confirmed
  - Immediate entry on acceptance bar

Entry Model:
  - Early: IBH breakout acceptance (2x5-min + volume)
  - Add-ons: Every VWAP/dPOC retest, consecutive 30-min DPOC jumps >=30-40 pts
  - Single-print high rejections = permanent gift adds
  - MAX 3 pyramids (aggressive mode)

TPO "Never Fill" Rule:
  - Single prints (velocity seams) above IBH must remain unfilled
  - If price fills mid-day prints, momentum has died

Stops:
  - Initial: Below IBL (wide buffer on explosive days)
  - Trail aggressively after +150 pts
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


class SuperTrendBull(StrategyBase):

    @property
    def name(self) -> str:
        return "Super Trend Bull"

    @property
    def applicable_day_types(self) -> List[str]:
        return ['super_trend_up']

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

    def on_bar(self, bar: pd.Series, bar_index: int, session_context: dict) -> Optional[Signal]:
        strength = session_context.get('trend_strength', 'weak')
        current_price = bar['close']
        bar_time = session_context.get('bar_time')

        # ── Phase 1: Track acceptance (ALL bars, ALL day types) ──
        if not self._acceptance_confirmed:
            if current_price > self._ib_high:
                self._consecutive_above += 1
            else:
                self._consecutive_above = 0

            if self._consecutive_above >= ACCEPTANCE_MIN_BARS:
                self._acceptance_confirmed = True
                self._acceptance_bar = bar_index

                # Immediate entry on super trend acceptance
                day_type = session_context.get('day_type', '')
                if day_type in self.applicable_day_types and strength == 'super':
                    return self._build_signal(
                        bar, bar_index, session_context,
                        entry_type='IBH_BREAKOUT_SUPER',
                        is_initial=True,
                    )

            return None

        # ── Phase 2: Only emit on super_trend_up + super strength ──
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
            if current_price > self._ib_high:
                bars_since = bar_index - self._acceptance_bar if self._acceptance_bar else 0
                if bars_since <= 8:
                    return self._build_signal(
                        bar, bar_index, session_context,
                        entry_type='IBH_BREAKOUT_SUPER',
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

        # 1. VWAP/dPOC retest gift add
        vwap = bar.get('vwap')
        if vwap is not None and not pd.isna(vwap):
            if bar['low'] <= vwap <= bar['high'] and current_price > vwap:
                return self._build_signal(
                    bar, bar_index, session_context,
                    entry_type=f'VWAP_GIFT_ADD_P{self._pyramid_count + 1}',
                    is_initial=False,
                )

        # 2. EMA pullback (shallow on super days)
        ema20 = bar.get('ema20')
        if ema20 is not None and not pd.isna(ema20) and current_price > ema20:
            ema_dist = (current_price - ema20) / self._ib_range if self._ib_range > 0 else 999
            if ema_dist < EMA_PULLBACK_THRESHOLD:
                return self._build_signal(
                    bar, bar_index, session_context,
                    entry_type=f'EMA_PULLBACK_P{self._pyramid_count + 1}',
                    is_initial=False,
                )

        # 3. IBH retest add (deep pullback = gift on super days)
        if bar['low'] <= self._ib_high <= bar['high'] and current_price > self._ib_high:
            return self._build_signal(
                bar, bar_index, session_context,
                entry_type=f'IBH_RETEST_GIFT_P{self._pyramid_count + 1}',
                is_initial=False,
            )

        return None

    def _build_signal(
        self, bar: pd.Series, bar_index: int, session_context: dict,
        entry_type: str, is_initial: bool,
    ) -> Signal:
        current_price = bar['close']

        if is_initial:
            # Wide stop below IBL on explosive days
            stop_price = self._ib_low - (self._ib_range * TREND_STOP_IB_BUFFER)
            self._pyramid_count = 1
        else:
            # Trailing stop for pyramids
            if self._last_entry_price is not None:
                stop_price = max(
                    self._ib_low - (self._ib_range * TREND_STOP_IB_BUFFER),
                    self._last_entry_price - (self._ib_range * 0.25),
                )
            else:
                stop_price = self._ib_low - (self._ib_range * TREND_STOP_IB_BUFFER)
            self._pyramid_count += 1

        # Aggressive target: 3x IB range for super trend
        target_price = current_price + (3.0 * self._ib_range)

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
            day_type='super_trend_up',
            trend_strength='super',
            confidence='high',
            pyramid_level=self._pyramid_count - 1,
        )
