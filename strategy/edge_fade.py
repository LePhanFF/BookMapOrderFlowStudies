"""
Strategy: Edge Fade (Edge-to-Mid Mean Reversion)

Purpose: Increase trade frequency on balance/neutral days by fading from
IB edges back to IB midpoint. This strategy fills the massive gap in the
existing Playbook which only trades 12/62 sessions.

Two entry models, both LONG-ONLY (NQ long bias):

1. EDGE_TO_MID: Price near IBL on b_day/neutral, fade to IB mid
   - 90 trades, 53.3% WR, +$7,714 net in v14 backtest (VWAP breach exempt)
   - Condition: Price in lower 25% of IB range
   - Requires: delta > 0, OF quality >= 2
   - Entry confirmation boost: CVD divergence, FVG confluence, 5-min inversion

2. FAILED_BREAKDOWN: Price broke below IBL, then reclaims above
   - DISABLED: Engine's VWAP breach PM exit kills this entry (-$987 net)
   - TODO: Re-enable if position management is adjusted

Entry Confirmation Models (from diagnostic_entry_models.py):
- CVD_DIVERGENCE: Price near lows but CVD rising = smart money accumulating
  * 57.1% WR vs 33.3% baseline, +$14/trade edge, 20.8% winner discrimination
- FVG_CONFLUENCE: Entry inside bullish Fair Value Gap zone
  * 57.1% WR, +$13.2/trade edge, 20.8% winner discrimination
- INVERSION_5M: 5-min bearish-to-bullish candle reversal at edge
  * 45.5% WR with 11 trades, +$5.5/trade edge, 16.7% discrimination
- When any confirmation is present: confidence='high' -> larger position via sizer

Key design decisions:
- LONG ONLY: All shorts on NQ are negative expectancy (confirmed by data)
- No acceptance required: Works on sessions that never break above IBH
- Applicable on b_day AND neutral: Catches sessions misclassified between
  b_day and neutral (extension 0.2-0.5x threshold boundary)
- OF quality gate (2-of-3): Maintains entry quality while allowing volume
- VWAP breach PM exempt: Mean reversion trades naturally oscillate near VWAP
"""

from datetime import time as _time
from typing import Optional, List
import pandas as pd
import numpy as np

from strategy.base import StrategyBase
from strategy.signal import Signal

# Edge fade constants
EDGE_FADE_COOLDOWN_BARS = 20        # Bars between entries per model
EDGE_FADE_IB_EXPANSION_RATIO = 1.2  # Only trade when IB >= 1.2x rolling avg (expansion = overextension = good mean reversion)
EDGE_FADE_IB_LOOKBACK = 5           # Rolling window for IB average (responsive to recent regime)
EDGE_FADE_LAST_ENTRY_TIME = _time(13, 30)  # No entries after 13:30 (PM morph kills mean reversion: 0% WR)
EDGE_FADE_MAX_BEARISH_EXT = 0.3    # Max ext_down as fraction of IB (bearish days = 37% WR)

# EDGE_TO_MID parameters
EDGE_ZONE_PCT = 0.25               # Lower 25% of IB = entry zone
EDGE_STOP_BUFFER = 0.15            # Stop: IBL - 15% IB range
EDGE_MIN_RR = 1.0                  # Minimum reward/risk ratio

# FAILED_BREAKDOWN parameters
BREAKDOWN_MIN_BARS_BELOW = 2       # Must be below IBL for 2+ bars
BREAKDOWN_STOP_BUFFER = 0.05       # Stop: session low - 5% IB range
BREAKDOWN_MIN_RR = 0.5             # Accept wider stop (higher WR compensates)

# Entry confirmation parameters
CVD_LOOKBACK = 10                  # Bars to check for CVD divergence
CVD_PRICE_POSITION_MAX = 0.40      # Price must be in lower 40% of lookback range
CVD_SLOPE_BARS = 5                 # CVD rising over last N bars


class EdgeFadeStrategy(StrategyBase):
    """
    Edge-to-mid mean reversion on balance/neutral days.

    Fills the trade frequency gap: existing Playbook trades 10/62 sessions,
    this strategy trades 47/62 sessions (76% coverage).

    Entry confirmation models boost confidence when present:
    - CVD divergence (smart money accumulating into weakness)
    - FVG confluence (institutional order block supporting entry)
    - 5-min inversion candle (structural reversal on meaningful timeframe)
    """

    @property
    def name(self) -> str:
        return "Edge Fade"

    @property
    def applicable_day_types(self) -> List[str]:
        return ['b_day', 'neutral']

    def on_session_start(self, session_date, ib_high, ib_low, ib_range, session_context):
        self._ib_high = ib_high
        self._ib_low = ib_low
        self._ib_range = ib_range
        self._ib_mid = (ib_high + ib_low) / 2

        # Expansion filter: only trade when IB is wider than recent average
        # Wider IB = overextension = better mean reversion opportunity
        # Narrow IB = low vol / choppy = poor mean reversion (Aug-Sep regime)
        ib_history = session_context.get('ib_range_history', [])
        if len(ib_history) >= EDGE_FADE_IB_LOOKBACK:
            recent = ib_history[-EDGE_FADE_IB_LOOKBACK:]
            avg_ib = sum(recent) / len(recent)
            ib_ratio = ib_range / avg_ib if avg_ib > 0 else 1.0
            self._active = ib_ratio >= EDGE_FADE_IB_EXPANSION_RATIO
        else:
            # Not enough history — allow trading (first few sessions)
            self._active = True

        # Track max extension below IBL for bearish day filter
        self._max_ext_down = 0.0

        # Per-model cooldown tracking
        self._last_edge_to_mid_bar = -EDGE_FADE_COOLDOWN_BARS
        self._last_failed_breakdown_bar = -EDGE_FADE_COOLDOWN_BARS

        # Failed breakdown state tracking
        self._ever_below_ibl = False
        self._below_ibl_start = -1
        self._bars_below_ibl_count = 0
        self._session_low = ib_low

        # Delta momentum tracking
        self._delta_history = []

        # Entry confirmation state tracking
        self._price_history = []      # (close, low, high) tuples for lookback
        self._cvd_history = []        # CVD values for divergence detection
        self._bar_candles = []        # (open, close) for inversion detection
        self._5m_candles = []         # 5-min OHLC for 5-min inversion
        self._5m_bar_count = 0        # Counter for 5-min aggregation
        self._5m_open = None
        self._5m_high = None
        self._5m_low = None
        self._5m_close = None

    def on_bar(self, bar: pd.Series, bar_index: int, session_context: dict) -> Optional[Signal]:
        if not self._active:
            return None

        current_price = bar['close']
        delta = bar.get('delta', 0)
        if pd.isna(delta):
            delta = 0

        # === ALWAYS track state regardless of day_type ===
        # (breakdown can start on trend_down bars, but reclaim entry fires on b_day/neutral)

        # Track session low and bearish extension
        if bar['low'] < self._session_low:
            self._session_low = bar['low']
        # Use bar low for extension tracking (catches wicks below IBL)
        if bar['low'] < self._ib_low and self._ib_range > 0:
            ext_down = (self._ib_low - bar['low']) / self._ib_range
            if ext_down > self._max_ext_down:
                self._max_ext_down = ext_down

        # Track delta momentum
        self._delta_history.append(delta)
        if len(self._delta_history) > 10:
            self._delta_history.pop(0)

        # Track price/CVD history for entry confirmation
        self._price_history.append((current_price, bar['low'], bar['high']))
        if len(self._price_history) > 30:
            self._price_history.pop(0)

        cvd = bar.get('cumulative_delta', 0)
        if pd.isna(cvd):
            cvd = 0
        self._cvd_history.append(cvd)
        if len(self._cvd_history) > 30:
            self._cvd_history.pop(0)

        # Track 1-min candle data for inversion detection
        bar_open = bar['open'] if 'open' in bar.index else current_price
        self._bar_candles.append((bar_open, current_price))
        if len(self._bar_candles) > 5:
            self._bar_candles.pop(0)

        # Aggregate 5-min candles
        self._5m_bar_count += 1
        if self._5m_open is None:
            self._5m_open = bar_open
            self._5m_high = bar['high']
            self._5m_low = bar['low']
        else:
            self._5m_high = max(self._5m_high, bar['high'])
            self._5m_low = min(self._5m_low, bar['low'])
        self._5m_close = current_price

        if self._5m_bar_count >= 5:
            self._5m_candles.append((self._5m_open, self._5m_high, self._5m_low, self._5m_close))
            if len(self._5m_candles) > 10:
                self._5m_candles.pop(0)
            # Reset 5-min aggregation
            self._5m_bar_count = 0
            self._5m_open = None
            self._5m_high = None
            self._5m_low = None
            self._5m_close = None

        # Track bars below IBL (for failed breakdown)
        if current_price < self._ib_low:
            self._bars_below_ibl_count += 1
            if not self._ever_below_ibl:
                self._ever_below_ibl = True
                self._below_ibl_start = bar_index
        else:
            # Reset count when back above (but ever_below_ibl stays True)
            if self._bars_below_ibl_count > 0:
                self._bars_below_ibl_count = 0

        # Compute OF quality score (bullish signals for LONG entries)
        delta_pctl = bar.get('delta_percentile', 50)
        imbalance = bar.get('imbalance_ratio', 1.0)
        vol_spike = bar.get('volume_spike', 1.0)

        # Handle NaN safely
        if pd.isna(delta_pctl):
            delta_pctl = 50
        if pd.isna(imbalance):
            imbalance = 1.0
        if pd.isna(vol_spike):
            vol_spike = 1.0

        of_quality = sum([
            delta_pctl >= 60,
            imbalance > 1.0,
            vol_spike >= 1.0,
        ])

        # Time gate
        bar_time = session_context.get('bar_time')
        if bar_time and bar_time >= EDGE_FADE_LAST_ENTRY_TIME:
            return None

        # ── FAILED_BREAKDOWN entry DISABLED ──
        # Diagnostic showed 62% WR in simple sim but engine's VWAP breach PM exit
        # cuts too many trades early, resulting in -$987 net (45% WR) in real backtest.
        # The entry reclaims from below IBL into a zone that's still VWAP-vulnerable.
        # TODO: Re-enable if VWAP breach PM is tuned or if stop/target adjusted.

        # Bearish day filter: skip if ext_down > 0.3x IB (bearish days = 37% WR)
        if self._max_ext_down >= EDGE_FADE_MAX_BEARISH_EXT:
            return None

        # Day type gate for EDGE_TO_MID (only b_day/neutral)
        day_type = session_context.get('day_type', '')
        if day_type not in self.applicable_day_types:
            return None

        # ── Entry: EDGE_TO_MID (mean reversion from lower IB edge) ──
        signal = self._check_edge_to_mid(
            bar, bar_index, session_context, current_price, delta, of_quality
        )
        if signal:
            return signal

        return None

    # ================================================================
    #  Entry confirmation checks
    # ================================================================

    def _check_cvd_divergence(self) -> bool:
        """
        CVD divergence: price near lookback lows but CVD is rising.
        Indicates smart money accumulating into weakness.

        Diagnostic result: 57.1% WR (vs 33.3% baseline), +20.8% discrimination.
        Present in 33% of winners, 12.5% of losers.
        """
        if len(self._price_history) < CVD_LOOKBACK or len(self._cvd_history) < CVD_LOOKBACK:
            return False

        # Use last CVD_LOOKBACK entries
        prices = self._price_history[-CVD_LOOKBACK:]
        cvds = self._cvd_history[-CVD_LOOKBACK:]

        # Price must be in lower portion of lookback range
        price_highs = [p[2] for p in prices]  # high values
        price_lows = [p[1] for p in prices]   # low values
        price_range = max(price_highs) - min(price_lows)
        if price_range <= 0:
            return False

        current_price = prices[-1][0]  # close
        price_position = (current_price - min(price_lows)) / price_range
        if price_position > CVD_PRICE_POSITION_MAX:
            return False  # Not near lows

        # CVD must be rising over recent bars
        recent_cvds = cvds[-CVD_SLOPE_BARS:]
        if len(recent_cvds) < 3:
            return False
        cvd_slope = recent_cvds[-1] - recent_cvds[-3]
        return cvd_slope > 0

    def _check_fvg_confluence(self, bar: pd.Series) -> bool:
        """
        FVG confluence: entry bar is inside a bullish Fair Value Gap zone.
        Institutional order block providing support.

        Diagnostic result: 57.1% WR (vs 33.3% baseline), +20.8% discrimination.
        Present in 33% of winners, 12.5% of losers.
        """
        fvg_bull = bar.get('fvg_bull', False)
        if pd.isna(fvg_bull):
            return False
        return bool(fvg_bull)

    def _check_ifvg_confluence(self, bar: pd.Series) -> bool:
        """
        IFVG confluence: Inverse Fair Value Gap pullback confirmation.
        Same detection rate as FVG in our data.
        """
        ifvg = bar.get('ifvg_bull_entry', False)
        if pd.isna(ifvg):
            return False
        return bool(ifvg)

    def _check_inversion_5m(self) -> bool:
        """
        5-min inversion candle: prior 5-min bar bearish, current 5-min bar bullish.
        Structural reversal confirmation on meaningful timeframe.

        Diagnostic result: 45.5% WR (vs 33.3% baseline), +16.7% discrimination.
        5-min filters noise that 1-min can't; bearish-to-bullish at edge = reversal.
        """
        if len(self._5m_candles) < 2:
            return False

        prior_5m = self._5m_candles[-2]  # (open, high, low, close)
        current_5m = self._5m_candles[-1]

        prior_bearish = prior_5m[3] < prior_5m[0]   # close < open
        current_bullish = current_5m[3] > current_5m[0]  # close > open

        return prior_bearish and current_bullish

    def _compute_entry_confirmation(self, bar: pd.Series) -> dict:
        """
        Run all entry confirmation models and return results.

        Returns dict with:
        - has_confirmation: bool (any model passed)
        - confirmations: list of model names that passed
        - confidence: 'high' if confirmed, 'medium' otherwise
        """
        confirmations = []

        if self._check_cvd_divergence():
            confirmations.append('cvd_divergence')

        if self._check_fvg_confluence(bar):
            confirmations.append('fvg_bull')

        if self._check_ifvg_confluence(bar):
            confirmations.append('ifvg_bull')

        if self._check_inversion_5m():
            confirmations.append('inversion_5m')

        has_confirmation = len(confirmations) > 0
        confidence = 'high' if has_confirmation else 'medium'

        return {
            'has_confirmation': has_confirmation,
            'confirmations': confirmations,
            'confidence': confidence,
        }

    # ================================================================
    #  Entry models
    # ================================================================

    def _check_failed_breakdown(self, bar, bar_index, session_context,
                                 current_price, delta, of_quality) -> Optional[Signal]:
        """
        FAILED_BREAKDOWN: Price broke below IBL (2+ bars), now reclaiming above.

        62.1% WR, $57/trade in diagnostic testing (29 trades).
        Structural logic: shorts trapped below IBL are forced to cover,
        creating strong buying pressure as price reclaims IB range.
        """
        # Cooldown check
        if bar_index - self._last_failed_breakdown_bar < EDGE_FADE_COOLDOWN_BARS:
            return None

        # Must have been below IBL at some point
        if not self._ever_below_ibl:
            return None

        # Must now be back above IBL but still in lower half of IB
        if current_price <= self._ib_low:
            return None
        if current_price >= self._ib_mid:
            return None

        # Must have been below IBL for at least 2 bars
        # Check lookback: were there bars below IBL recently?
        if bar_index - self._below_ibl_start < BREAKDOWN_MIN_BARS_BELOW:
            return None

        # Buyer confirmation
        if delta <= 0:
            return None

        # OF quality gate
        if of_quality < 2:
            return None

        # Pre-delta momentum check (IB-scaled)
        pre_delta = sum(self._delta_history[:-1]) if len(self._delta_history) > 1 else 0
        if pre_delta < -self._ib_range * 0.80:
            return None  # Too much selling pressure

        # Compute stop and target
        stop = self._session_low - self._ib_range * BREAKDOWN_STOP_BUFFER
        target = self._ib_mid

        # R:R check
        risk = current_price - stop
        reward = target - current_price
        if risk <= 0 or reward <= 0:
            return None
        if reward / risk < BREAKDOWN_MIN_RR:
            return None

        self._last_failed_breakdown_bar = bar_index
        # Reset so we don't keep firing on same breakdown
        self._ever_below_ibl = False

        strength = session_context.get('trend_strength', 'weak')
        confidence = 'high'  # 62% WR justifies high confidence

        return Signal(
            timestamp=bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp'),
            direction='LONG',
            entry_price=current_price,
            stop_price=stop,
            target_price=target,
            strategy_name=self.name,
            setup_type='FAILED_BREAKDOWN',
            day_type=session_context.get('day_type', 'neutral'),
            trend_strength=strength,
            confidence=confidence,
        )

    def _check_edge_to_mid(self, bar, bar_index, session_context,
                            current_price, delta, of_quality) -> Optional[Signal]:
        """
        EDGE_TO_MID: Price in lower 25% of IB, fade toward midpoint.

        v14 results: 90 trades, 53.3% WR, $86/trade, $7,714 net.
        Mean reversion: when price visits the lower edge of the IB range
        with buyer confirmation, fade back toward the institutional VWAP/midpoint.

        Entry confirmation models boost confidence when present:
        - CVD divergence: 57.1% WR, +$14/trade edge
        - FVG confluence: 57.1% WR, +$13/trade edge
        - 5-min inversion: 45.5% WR, +$5.5/trade edge
        """
        # Cooldown check
        if bar_index - self._last_edge_to_mid_bar < EDGE_FADE_COOLDOWN_BARS:
            return None

        # Price must be in lower 25% of IB (the edge zone)
        edge_ceiling = self._ib_low + self._ib_range * EDGE_ZONE_PCT
        if current_price <= self._ib_low or current_price >= edge_ceiling:
            return None

        # Buyer confirmation
        if delta <= 0:
            return None

        # OF quality gate
        if of_quality < 2:
            return None

        # Compute stop and target
        stop = self._ib_low - self._ib_range * EDGE_STOP_BUFFER
        target = self._ib_mid

        # R:R check
        risk = current_price - stop
        reward = target - current_price
        if risk <= 0 or reward <= 0:
            return None
        if reward / risk < EDGE_MIN_RR:
            return None

        self._last_edge_to_mid_bar = bar_index

        strength = session_context.get('trend_strength', 'weak')

        # Run entry confirmation models
        confirm = self._compute_entry_confirmation(bar)

        return Signal(
            timestamp=bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp'),
            direction='LONG',
            entry_price=current_price,
            stop_price=stop,
            target_price=target,
            strategy_name=self.name,
            setup_type='EDGE_TO_MID',
            day_type=session_context.get('day_type', 'neutral'),
            trend_strength=strength,
            confidence=confirm['confidence'],
            metadata={
                'confirmations': confirm['confirmations'],
                'has_confirmation': confirm['has_confirmation'],
            },
        )
