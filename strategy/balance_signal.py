"""
Strategy: Balance Signal (VA Edge Fade + Wide IB Reclaim)

Purpose: Quality-filtered mean-reversion signals on balance/inside/neutral days.
Two entry modes with a shared 6-factor scoring system that separates high-quality
setups from mechanical noise.

Mode 1: VA Edge Fade
  - Price touches prior VAH/VAL and fades back inside VA
  - Dalton 2-bar acceptance confirms the fade
  - Target: prior POC (VA mode) or VWAP, adjusted by developing profile shape

Mode 2: Wide IB Reclaim
  - IB range 350-500pt (wide but not trend)
  - Price breaks IB edge, then reclaims back inside
  - Requires HVN at the reclaimed edge (F2 >= 1)
  - Target: IB midpoint, adjusted by developing profile shape

6-Factor Quality Scoring (0-10 pts):
  F1: Prior session profile shape (0-2)  — P/b/D shape directional lean
  F2: HVN/LVN at VA edge (0-2)          — Volume structure at the faded level
  F3: Volume distribution skew (0-1)     — Exhaustion signal from volume imbalance
  F4: VWAP confirmation (0-1)            — VWAP position + slope alignment
  F5: IB containment within prior VA (0-2) — Balance day structural confirmation
  F6: Delta/CVD divergence (0-2)         — Absorption + smart money divergence

Minimum score to fire: 5 (configurable via BALANCE_MIN_QUALITY_SCORE).

Key design decisions:
- All day types allowed (scoring handles quality gating, not day_type filter)
- LONG and SHORT supported (unlike Edge Fade which is LONG-only)
- BPR zones tracked as alternative targets when VWAP fails
- Dynamic targets adjust based on developing POC migration
"""

from datetime import time as _time
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np

from strategy.base import StrategyBase
from strategy.signal import Signal
from config.constants import (
    BALANCE_MIN_QUALITY_SCORE,
    BALANCE_COOLDOWN_BARS,
    BALANCE_LAST_ENTRY_TIME,
    BALANCE_VA_STOP_BUFFER_PCT,
    BALANCE_VA_MIN_RR,
    BALANCE_IB_RECLAIM_MIN,
    BALANCE_IB_RECLAIM_MAX,
    BALANCE_IB_STOP_BUFFER_PCT,
    BALANCE_ACCEPT_EARLY,
    BALANCE_ACCEPT_INSIDE,
    BALANCE_HVN_PERCENTILE,
    BALANCE_LVN_PERCENTILE,
    BALANCE_EDGE_PROXIMITY_PCT,
)


class BalanceSignal(StrategyBase):
    """
    Quality-scored mean-reversion on balance/inside/neutral days.

    Combines VA Edge Fade and Wide IB Reclaim modes with a 6-factor
    scoring system. Each factor is a separate method for unit testing.
    """

    @property
    def name(self) -> str:
        return "Balance Signal"

    @property
    def applicable_day_types(self) -> List[str]:
        return []  # All day types — scoring handles filtering

    # ================================================================
    #  Session lifecycle
    # ================================================================

    def on_session_start(self, session_date, ib_high, ib_low, ib_range, session_context):
        self._session_date = session_date
        self._ib_high = ib_high
        self._ib_low = ib_low
        self._ib_range = ib_range
        self._ib_mid = (ib_high + ib_low) / 2

        # Prior VA levels
        self._prior_vah = session_context.get('prior_va_high') or session_context.get('prior_vp_vah')
        self._prior_val = session_context.get('prior_va_low') or session_context.get('prior_vp_val')
        self._prior_poc = session_context.get('prior_va_poc') or session_context.get('prior_vp_poc')

        # Prior TPO shape
        self._prior_tpo_shape = session_context.get('prior_tpo_shape') or session_context.get('tpo_shape', '')
        self._tpo_poc_location = session_context.get('tpo_poc_location', 0.5)

        # VA range for buffer calculations
        if self._prior_vah and self._prior_val and self._prior_vah > self._prior_val:
            self._va_range = self._prior_vah - self._prior_val
        else:
            self._va_range = 0.0

        # ATR for volatility-aware calculations
        self._atr14 = session_context.get('atr14', 0.0)

        # IB regime check for Wide IB Reclaim eligibility
        self._wide_ib_eligible = BALANCE_IB_RECLAIM_MIN <= ib_range <= BALANCE_IB_RECLAIM_MAX

        # Dalton acceptance counters (consecutive closes above/below levels)
        self._accept_below_vah = 0   # Closes below VAH (for SHORT fade)
        self._accept_above_val = 0   # Closes above VAL (for LONG fade)
        self._accept_above_ibl = 0   # Closes above IBL (for LONG reclaim)
        self._accept_below_ibh = 0   # Closes below IBH (for SHORT reclaim)

        # Touch tracking (did price reach the VA edge?)
        self._touched_vah = False
        self._touched_val = False
        self._broke_ibh = False
        self._broke_ibl = False

        # BPR state (from bar columns)
        self._bpr_high = None
        self._bpr_low = None
        self._bpr_active = False

        # DPOC tracking (developing POC of today)
        self._volume_by_price = {}  # price_bin -> cumulative volume

        # VWAP tracking
        self._vwap_history = []     # Last N VWAP values for slope

        # CVD tracking
        self._cvd_history = []      # Last N CVD values for divergence
        self._price_history = []    # Last N close prices for divergence

        # Cooldown per mode
        self._last_va_fade_bar = -BALANCE_COOLDOWN_BARS
        self._last_ib_reclaim_bar = -BALANCE_COOLDOWN_BARS

        # One signal per mode per session
        self._va_fade_fired = False
        self._ib_reclaim_fired = False

        # Prior session volume profile bins (for HVN/LVN)
        self._prior_vol_bins = self._build_prior_vol_bins(session_context)

        # Store previous bar close for reclaim detection
        self._prev_close = None

    # ================================================================
    #  Main on_bar
    # ================================================================

    def on_bar(self, bar: pd.Series, bar_index: int, session_context: dict) -> Optional[Signal]:
        # Skip if no prior VA (first session or missing data)
        if not self._prior_vah or not self._prior_val or self._va_range <= 0:
            self._prev_close = bar['close']
            return None

        current_price = bar['close']

        # Time gate
        bar_time = session_context.get('bar_time')
        if bar_time and bar_time >= BALANCE_LAST_ENTRY_TIME:
            self._prev_close = current_price
            return None

        # Update tracking state
        self._update_vwap_tracking(session_context)
        self._update_dpoc(bar)
        self._update_bpr_state(bar)
        self._update_cvd_tracking(bar)
        self._update_acceptance(bar)

        # Check Mode 1: VA Edge Fade
        if not self._va_fade_fired:
            signal = self._check_va_edge_fade(bar, bar_index, session_context)
            if signal:
                self._prev_close = current_price
                return signal

        # Check Mode 2: Wide IB Reclaim
        if not self._ib_reclaim_fired and self._wide_ib_eligible:
            signal = self._check_wide_ib_reclaim(bar, bar_index, session_context)
            if signal:
                self._prev_close = current_price
                return signal

        self._prev_close = current_price
        return None

    # ================================================================
    #  6-Factor Scoring
    # ================================================================

    def score_profile_shape(self, direction: str) -> int:
        """F1: Prior session profile shape (0-2 pts).

        P-shape (POC in upper third) → bullish lean → +2 LONG, 0 SHORT
        b-shape (POC in lower third) → bearish lean → +2 SHORT, 0 LONG
        D-shape (POC centered)       → neutral      → +1 either direction
        """
        shape = str(self._prior_tpo_shape).upper().strip() if self._prior_tpo_shape else ''
        poc_loc = self._tpo_poc_location  # 0=bottom, 1=top of range

        # Explicit shape labels
        if shape in ('P', 'P-SHAPE'):
            return 2 if direction == 'LONG' else 0
        if shape in ('B', 'B-SHAPE'):
            return 2 if direction == 'SHORT' else 0
        if shape in ('D', 'D-SHAPE'):
            return 1

        # Fallback: use POC location ratio
        if poc_loc >= 0.67:
            return 2 if direction == 'LONG' else 0
        if poc_loc <= 0.33:
            return 2 if direction == 'SHORT' else 0
        return 1  # centered POC = neutral

    def score_hvn_lvn_at_edge(self, direction: str) -> int:
        """F2: HVN/LVN near the VA edge being faded (0-2 pts).

        HVN at edge → strong support/resistance → +2
        Neutral volume → +1
        LVN at edge → weak level, likely to break → +0
        """
        if not self._prior_vol_bins:
            return 1  # No data, give neutral score

        # Determine which edge we're fading
        if direction == 'LONG':
            edge_price = self._prior_val
        else:
            edge_price = self._prior_vah

        if edge_price is None:
            return 1

        # Check volume bins near the edge
        proximity = self._va_range * BALANCE_EDGE_PROXIMITY_PCT
        nearby_volumes = []
        for price_bin, vol in self._prior_vol_bins.items():
            if abs(price_bin - edge_price) <= proximity:
                nearby_volumes.append(vol)

        if not nearby_volumes:
            return 1

        all_volumes = list(self._prior_vol_bins.values())
        if not all_volumes:
            return 1

        avg_nearby = np.mean(nearby_volumes)
        hvn_threshold = np.percentile(all_volumes, BALANCE_HVN_PERCENTILE)
        lvn_threshold = np.percentile(all_volumes, BALANCE_LVN_PERCENTILE)

        if avg_nearby >= hvn_threshold:
            return 2  # HVN — strong level
        if avg_nearby <= lvn_threshold:
            return 0  # LVN — weak level
        return 1      # Neutral

    def score_volume_skew(self, direction: str) -> int:
        """F3: Volume distribution skew (0-1 pt).

        More volume below POC + fading VAL → sellers exhausted → +1 LONG
        More volume above POC + fading VAH → buyers exhausted → +1 SHORT
        """
        if not self._prior_vol_bins or not self._prior_poc:
            return 0

        vol_below = sum(v for p, v in self._prior_vol_bins.items() if p < self._prior_poc)
        vol_above = sum(v for p, v in self._prior_vol_bins.items() if p >= self._prior_poc)
        total = vol_below + vol_above

        if total <= 0:
            return 0

        below_ratio = vol_below / total

        if direction == 'LONG' and below_ratio > 0.55:
            return 1  # Heavy selling exhausted below POC
        if direction == 'SHORT' and below_ratio < 0.45:
            return 1  # Heavy buying exhausted above POC
        return 0

    def score_vwap_confirmation(self, direction: str, bar: pd.Series) -> int:
        """F4: VWAP position + slope (0-1 pt).

        LONG: price <= VWAP and VWAP slope flat/rising → +1
        SHORT: price >= VWAP and VWAP slope flat/falling → +1
        """
        if len(self._vwap_history) < 3:
            return 0

        current_vwap = self._vwap_history[-1]
        if current_vwap is None or current_vwap <= 0:
            return 0

        price = bar['close']

        # VWAP slope over last 5 readings
        slope_window = self._vwap_history[-5:] if len(self._vwap_history) >= 5 else self._vwap_history
        vwap_slope = slope_window[-1] - slope_window[0]

        if direction == 'LONG':
            if price <= current_vwap and vwap_slope >= -2:  # flat or rising
                return 1
        elif direction == 'SHORT':
            if price >= current_vwap and vwap_slope <= 2:   # flat or falling
                return 1
        return 0

    def score_ib_context(self) -> int:
        """F5: IB containment within prior VA (0-2 pts).

        IB fully inside prior VA → +1 (balance confirmation)
        IB narrow (< 50% of VA range) → +1 (low-energy, MR favorable)
        """
        score = 0

        if self._prior_vah and self._prior_val:
            # IB inside prior VA
            if self._ib_high <= self._prior_vah and self._ib_low >= self._prior_val:
                score += 1

            # IB narrow relative to VA
            if self._va_range > 0 and self._ib_range < self._va_range * 0.50:
                score += 1

        return score

    def score_delta_cvd_divergence(self, direction: str, bar: pd.Series) -> int:
        """F6: Delta absorption + CVD divergence (0-2 pts).

        Absorption: delta opposes price direction but price stable → +1
        CVD divergence: CVD trend disagrees with recent price trend → +1
        """
        score = 0
        delta = bar.get('delta', 0)
        if pd.isna(delta):
            delta = 0

        # --- Absorption check ---
        # LONG: negative delta (sellers) but price not dropping = absorption
        # SHORT: positive delta (buyers) but price not rising = absorption
        if len(self._price_history) >= 3:
            price_change = bar['close'] - self._price_history[-3]
            if direction == 'LONG' and delta < 0 and price_change >= -5:
                score += 1
            elif direction == 'SHORT' and delta > 0 and price_change <= 5:
                score += 1

        # --- CVD divergence check ---
        if len(self._cvd_history) >= 5 and len(self._price_history) >= 5:
            cvd_change = self._cvd_history[-1] - self._cvd_history[-5]
            price_trend = bar['close'] - self._price_history[-5]

            # LONG: price falling but CVD rising = bullish divergence
            if direction == 'LONG' and price_trend < -3 and cvd_change > 0:
                score += 1
            # SHORT: price rising but CVD falling = bearish divergence
            elif direction == 'SHORT' and price_trend > 3 and cvd_change < 0:
                score += 1

        return score

    def compute_total_score(self, direction: str, bar: pd.Series) -> Tuple[int, dict]:
        """Aggregate all 6 factors. Returns (total, {factor: score} breakdown)."""
        breakdown = {
            'F1_profile_shape': self.score_profile_shape(direction),
            'F2_hvn_lvn': self.score_hvn_lvn_at_edge(direction),
            'F3_volume_skew': self.score_volume_skew(direction),
            'F4_vwap': self.score_vwap_confirmation(direction, bar),
            'F5_ib_context': self.score_ib_context(),
            'F6_delta_cvd': self.score_delta_cvd_divergence(direction, bar),
        }
        total = sum(breakdown.values())
        return total, breakdown

    # ================================================================
    #  Signal Modes
    # ================================================================

    def _check_va_edge_fade(self, bar: pd.Series, bar_index: int,
                            session_context: dict) -> Optional[Signal]:
        """Mode 1: VA Edge Fade — fade from prior VAH/VAL back inside VA."""
        # Cooldown
        if bar_index - self._last_va_fade_bar < BALANCE_COOLDOWN_BARS:
            return None

        current_price = bar['close']

        # --- SHORT fade: price touched VAH and faded back below ---
        if bar['high'] >= self._prior_vah:
            self._touched_vah = True

        if self._touched_vah and current_price < self._prior_vah:
            if self._accept_below_vah >= BALANCE_ACCEPT_EARLY:
                signal = self._try_emit_va_fade(
                    'SHORT', bar, bar_index, session_context,
                    entry=current_price,
                    stop=self._prior_vah + self._va_range * BALANCE_VA_STOP_BUFFER_PCT,
                )
                if signal:
                    return signal

        # --- LONG fade: price touched VAL and faded back above ---
        if bar['low'] <= self._prior_val:
            self._touched_val = True

        if self._touched_val and current_price > self._prior_val:
            if self._accept_above_val >= BALANCE_ACCEPT_EARLY:
                signal = self._try_emit_va_fade(
                    'LONG', bar, bar_index, session_context,
                    entry=current_price,
                    stop=self._prior_val - self._va_range * BALANCE_VA_STOP_BUFFER_PCT,
                )
                if signal:
                    return signal

        return None

    def _try_emit_va_fade(self, direction: str, bar: pd.Series, bar_index: int,
                          session_context: dict, entry: float,
                          stop: float) -> Optional[Signal]:
        """Score and emit a VA Edge Fade signal if quality is sufficient."""
        score, breakdown = self.compute_total_score(direction, bar)

        if score < BALANCE_MIN_QUALITY_SCORE:
            return None

        target = self.compute_dynamic_target('VA', direction, bar)
        if target is None:
            return None

        # R:R check
        risk = abs(entry - stop)
        reward = abs(target - entry)
        if risk <= 0 or reward <= 0:
            return None
        if reward / risk < BALANCE_VA_MIN_RR:
            return None

        # Sanity: target must be in correct direction
        if direction == 'LONG' and target <= entry:
            return None
        if direction == 'SHORT' and target >= entry:
            return None

        self._va_fade_fired = True
        self._last_va_fade_bar = bar_index

        confidence = 'high' if score >= 7 else 'medium'
        ts = bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp')

        return Signal(
            timestamp=ts,
            direction=direction,
            entry_price=entry,
            stop_price=stop,
            target_price=target,
            strategy_name=self.name,
            setup_type='VA_EDGE_FADE',
            day_type=session_context.get('day_type', 'neutral'),
            trend_strength=session_context.get('trend_strength', 'weak'),
            confidence=confidence,
            metadata={
                'score': score,
                'breakdown': breakdown,
                'mode': 'VA_EDGE_FADE',
                'bpr_active': self._bpr_active,
            },
        )

    def _check_wide_ib_reclaim(self, bar: pd.Series, bar_index: int,
                               session_context: dict) -> Optional[Signal]:
        """Mode 2: Wide IB Reclaim — IB break + reclaim on 350-500pt IB."""
        # Cooldown
        if bar_index - self._last_ib_reclaim_bar < BALANCE_COOLDOWN_BARS:
            return None

        current_price = bar['close']
        prev = self._prev_close
        if prev is None:
            return None

        # --- LONG reclaim: was below IBL, now back above ---
        if prev < self._ib_low and current_price > self._ib_low:
            self._broke_ibl = True

        if self._broke_ibl and current_price > self._ib_low:
            if self._accept_above_ibl >= BALANCE_ACCEPT_EARLY:
                signal = self._try_emit_ib_reclaim(
                    'LONG', bar, bar_index, session_context,
                    entry=current_price,
                    stop=self._ib_low - self._ib_range * BALANCE_IB_STOP_BUFFER_PCT,
                )
                if signal:
                    return signal

        # --- SHORT reclaim: was above IBH, now back below ---
        if prev > self._ib_high and current_price < self._ib_high:
            self._broke_ibh = True

        if self._broke_ibh and current_price < self._ib_high:
            if self._accept_below_ibh >= BALANCE_ACCEPT_EARLY:
                signal = self._try_emit_ib_reclaim(
                    'SHORT', bar, bar_index, session_context,
                    entry=current_price,
                    stop=self._ib_high + self._ib_range * BALANCE_IB_STOP_BUFFER_PCT,
                )
                if signal:
                    return signal

        return None

    def _try_emit_ib_reclaim(self, direction: str, bar: pd.Series, bar_index: int,
                             session_context: dict, entry: float,
                             stop: float) -> Optional[Signal]:
        """Score and emit a Wide IB Reclaim signal if quality is sufficient."""
        score, breakdown = self.compute_total_score(direction, bar)

        if score < BALANCE_MIN_QUALITY_SCORE:
            return None

        # Require HVN at edge (F2 >= 1) for reclaim trades
        if breakdown['F2_hvn_lvn'] < 1:
            return None

        target = self.compute_dynamic_target('IB', direction, bar)
        if target is None:
            return None

        # R:R check
        risk = abs(entry - stop)
        reward = abs(target - entry)
        if risk <= 0 or reward <= 0:
            return None
        if reward / risk < BALANCE_VA_MIN_RR:
            return None

        # Sanity
        if direction == 'LONG' and target <= entry:
            return None
        if direction == 'SHORT' and target >= entry:
            return None

        self._ib_reclaim_fired = True
        self._last_ib_reclaim_bar = bar_index

        confidence = 'high' if score >= 7 else 'medium'
        ts = bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp')

        return Signal(
            timestamp=ts,
            direction=direction,
            entry_price=entry,
            stop_price=stop,
            target_price=target,
            strategy_name=self.name,
            setup_type='WIDE_IB_RECLAIM',
            day_type=session_context.get('day_type', 'neutral'),
            trend_strength=session_context.get('trend_strength', 'weak'),
            confidence=confidence,
            metadata={
                'score': score,
                'breakdown': breakdown,
                'mode': 'WIDE_IB_RECLAIM',
                'ib_range': self._ib_range,
                'bpr_active': self._bpr_active,
            },
        )

    # ================================================================
    #  Dynamic Target
    # ================================================================

    def compute_dynamic_target(self, mode: str, direction: str,
                               bar: pd.Series) -> Optional[float]:
        """Compute target adjusted by developing profile shape.

        VA mode default: prior POC
        IB mode default: IB midpoint
        P-shape developing → LONG extends to P-mid, SHORT tightens to VWAP
        b-shape developing → SHORT extends to b-mid, LONG tightens to VWAP
        BPR zone: if VWAP fails, BPR midpoint becomes fallback target
        """
        # Base target
        if mode == 'VA':
            base_target = self._prior_poc if self._prior_poc else self._ib_mid
        else:  # IB
            base_target = self._ib_mid

        if base_target is None:
            return None

        # Developing POC migration
        dpoc = self._get_developing_poc()
        vwap = self._vwap_history[-1] if self._vwap_history else None

        if dpoc and self._prior_poc:
            # P-shape developing (DPOC migrating upward)
            if dpoc > self._prior_poc + self._va_range * 0.1:
                if direction == 'LONG':
                    # Extend target: average of base and DPOC
                    base_target = (base_target + dpoc) / 2
                elif direction == 'SHORT' and vwap:
                    # Tighten to VWAP (closer target, higher fill rate)
                    base_target = vwap

            # b-shape developing (DPOC migrating downward)
            elif dpoc < self._prior_poc - self._va_range * 0.1:
                if direction == 'SHORT':
                    base_target = (base_target + dpoc) / 2
                elif direction == 'LONG' and vwap:
                    base_target = vwap

        # BPR fallback: if main target is beyond BPR, use BPR midpoint
        if self._bpr_active and self._bpr_high and self._bpr_low:
            bpr_mid = (self._bpr_high + self._bpr_low) / 2
            if direction == 'LONG' and base_target > bpr_mid:
                base_target = bpr_mid
            elif direction == 'SHORT' and base_target < bpr_mid:
                base_target = bpr_mid

        return base_target

    # ================================================================
    #  State Tracking Helpers
    # ================================================================

    def _update_acceptance(self, bar: pd.Series):
        """Track consecutive closes above/below key levels (Dalton acceptance)."""
        close = bar['close']

        # Below VAH (for SHORT fade)
        if close < self._prior_vah:
            self._accept_below_vah += 1
        else:
            self._accept_below_vah = 0

        # Above VAL (for LONG fade)
        if close > self._prior_val:
            self._accept_above_val += 1
        else:
            self._accept_above_val = 0

        # Above IBL (for LONG reclaim)
        if close > self._ib_low:
            self._accept_above_ibl += 1
        else:
            self._accept_above_ibl = 0

        # Below IBH (for SHORT reclaim)
        if close < self._ib_high:
            self._accept_below_ibh += 1
        else:
            self._accept_below_ibh = 0

    def _update_vwap_tracking(self, session_context: dict):
        """Track VWAP values for slope calculation."""
        vwap = session_context.get('vwap')
        if vwap and not pd.isna(vwap) and vwap > 0:
            self._vwap_history.append(vwap)
            if len(self._vwap_history) > 30:
                self._vwap_history.pop(0)

    def _update_dpoc(self, bar: pd.Series):
        """Track developing POC (highest-volume price bin today)."""
        price = bar['close']
        volume = bar.get('volume', 0)
        if pd.isna(volume) or volume <= 0:
            return

        # Bin to nearest 5 points
        bin_price = round(price / 5) * 5
        self._volume_by_price[bin_price] = self._volume_by_price.get(bin_price, 0) + volume

    def _get_developing_poc(self) -> Optional[float]:
        """Return the current DPOC (price bin with highest cumulative volume)."""
        if not self._volume_by_price:
            return None
        return max(self._volume_by_price, key=self._volume_by_price.get)

    def _update_bpr_state(self, bar: pd.Series):
        """Track BPR from existing bar columns (computed by ict_models.py)."""
        in_bpr = bar.get('in_bpr', False)
        if pd.isna(in_bpr):
            in_bpr = False

        if in_bpr:
            bull_bottom = bar.get('fvg_bull_bottom')
            bull_top = bar.get('fvg_bull_top')
            bear_bottom = bar.get('fvg_bear_bottom')
            bear_top = bar.get('fvg_bear_top')

            # BPR overlap zone
            if (bull_bottom is not None and bear_bottom is not None
                    and not pd.isna(bull_bottom) and not pd.isna(bear_bottom)
                    and bull_top is not None and bear_top is not None
                    and not pd.isna(bull_top) and not pd.isna(bear_top)):
                self._bpr_low = max(bull_bottom, bear_bottom)
                self._bpr_high = min(bull_top, bear_top)
                self._bpr_active = True
        else:
            # BPR can deactivate if price moves away
            if self._bpr_active:
                # Keep BPR levels but mark inactive if price moved significantly
                if self._bpr_high and self._bpr_low:
                    bpr_range = self._bpr_high - self._bpr_low
                    if bpr_range > 0 and (bar['close'] > self._bpr_high + bpr_range
                                          or bar['close'] < self._bpr_low - bpr_range):
                        self._bpr_active = False

    def _update_cvd_tracking(self, bar: pd.Series):
        """Track CVD and price for divergence detection."""
        cvd = bar.get('cumulative_delta', 0)
        if pd.isna(cvd):
            cvd = 0
        self._cvd_history.append(cvd)
        if len(self._cvd_history) > 30:
            self._cvd_history.pop(0)

        self._price_history.append(bar['close'])
        if len(self._price_history) > 30:
            self._price_history.pop(0)

    def _build_prior_vol_bins(self, session_context: dict) -> dict:
        """Build volume-by-price bins from prior session data for HVN/LVN scoring.

        Uses vp_data from session_context if available, otherwise returns empty.
        """
        vp_data = session_context.get('vp_data')
        if vp_data and isinstance(vp_data, dict):
            bins = vp_data.get('volume_bins') or vp_data.get('bins')
            if bins and isinstance(bins, dict):
                return bins

        # Fallback: construct approximate bins from prior VA + POC
        # This gives us enough structure for HVN/LVN scoring even without
        # full volume profile data
        if self._prior_vah and self._prior_val and self._prior_poc:
            bins = {}
            va_range = self._prior_vah - self._prior_val
            if va_range > 0:
                step = max(va_range / 20, 1)
                price = self._prior_val
                while price <= self._prior_vah:
                    # Approximate: volume peaks at POC, drops toward edges
                    dist_from_poc = abs(price - self._prior_poc) / va_range
                    vol = max(100 - dist_from_poc * 200, 10)
                    bins[round(price / 5) * 5] = vol
                    price += step
                return bins

        return {}
