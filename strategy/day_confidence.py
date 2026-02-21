"""
Day Type Confidence Scorer — Dalton Checklist Matching

Instead of classifying day type from a single metric (IB extension),
evaluate how well the developing session matches each day type's
characteristic checklist. Only trade when confidence is high enough.

Dalton Day Type Checklists:

Trend Day (Bull/Bear):
  [x] Skinny TPO profile (narrow price distribution)
  [x] Single prints / velocity seams present
  [x] Fattening at directional extreme, not center
  [x] Accelerating DPOC migration (monotonic, >50pts cumulative)
  [x] One-timeframing (consecutive bars in same direction)
  [x] Excess tail at origin (>=3 bars at session start extreme)
  [x] IB extension >= 0.5x (moderate+ strength)
  [x] 80%+ of time outside IB after acceptance
  Confidence = count_of_matching_checks / total_checks

P-Day (Skewed Balance):
  [x] Moderate extension (0.5-1.0x IB range)
  [x] Skewed volume profile (bulge at one side)
  [x] DPOC migration present but not extreme
  [x] Single prints on one side only
  [x] No excess tail at both ends
  [x] Price spends >60% of post-IB time above/below IB mid

B-Day (Narrow Balance):
  [x] Weak extension (<0.5x)
  [x] Price stays inside IB (or barely outside)
  [x] Symmetric/flat DPOC
  [x] Poor highs AND poor lows (rejections at both extremes)
  [x] Low range relative to ATR

Neutral Day:
  [x] Symmetric fattening in center
  [x] Multiple HVN (high volume nodes)
  [x] Rotational probes repaired
  [x] Flat DPOC, low ATR

TPO/Volume Profile/DPOC data is passed through session_context by the engine
(computed by profile adapters). This data is available for strategies to use
directly but does NOT modify the base confidence scores, which are calibrated
to work with the existing strategy thresholds.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class DayTypeConfidence:
    """Confidence scores for each day type, updated bar-by-bar."""
    trend_bull: float = 0.0
    trend_bear: float = 0.0
    p_day_bull: float = 0.0
    p_day_bear: float = 0.0
    b_day: float = 0.0
    neutral: float = 0.0

    # Checklist details for debugging
    checklist: Dict[str, Dict[str, bool]] = field(default_factory=dict)

    @property
    def best_type(self) -> str:
        scores = {
            'trend_up': self.trend_bull,
            'trend_down': self.trend_bear,
            'p_day_bull': self.p_day_bull,
            'p_day_bear': self.p_day_bear,
            'b_day': self.b_day,
            'neutral': self.neutral,
        }
        return max(scores, key=scores.get)

    @property
    def best_confidence(self) -> float:
        return max(
            self.trend_bull, self.trend_bear,
            self.p_day_bull, self.p_day_bear,
            self.b_day, self.neutral,
        )


class DayTypeConfidenceScorer:
    """
    Bar-by-bar day type confidence scoring using Dalton checklists.

    Usage:
        scorer = DayTypeConfidenceScorer()
        scorer.on_session_start(ib_high, ib_low, ib_range, atr)
        for bar in post_ib_bars:
            confidence = scorer.update(bar, bar_index)
            # confidence.trend_bull = 0.0-1.0, etc.
    """

    def __init__(self):
        self._reset()

    def _reset(self):
        self._ib_high = 0.0
        self._ib_low = 0.0
        self._ib_range = 0.0
        self._ib_mid = 0.0
        self._atr = 0.0

        # Tracking variables
        self._bars_above_ib = 0
        self._bars_below_ib = 0
        self._bars_inside_ib = 0
        self._total_bars = 0

        self._bars_above_mid = 0
        self._bars_below_mid = 0

        self._session_high = 0.0
        self._session_low = 999999.0
        self._session_close = 0.0

        # One-timeframing: consecutive bars in same direction
        self._otf_up_streak = 0
        self._otf_down_streak = 0
        self._max_otf_up = 0
        self._max_otf_down = 0
        self._prev_close = None

        # DPOC migration tracking (simplified: use close as proxy)
        self._close_history: List[float] = []
        self._dpoc_estimates: List[float] = []

        # Extension tracking
        self._max_extension_up = 0.0
        self._max_extension_down = 0.0

        # Price range tracking (for skinny profile detection)
        self._price_bins: Dict[int, int] = {}  # bin -> bar count

    def on_session_start(self, ib_high: float, ib_low: float, ib_range: float,
                         atr: float = 0.0, tpo_data: dict = None):
        """Initialize for a new session."""
        self._reset()
        self._ib_high = ib_high
        self._ib_low = ib_low
        self._ib_range = ib_range
        self._ib_mid = (ib_high + ib_low) / 2
        self._atr = atr if atr > 0 else ib_range * 2  # fallback
        self._session_high = ib_high
        self._session_low = ib_low
        # tpo_data accepted for API compatibility but not used in scoring

    def update(self, bar: pd.Series, bar_index: int,
               tpo_data: dict = None) -> DayTypeConfidence:
        """
        Update confidence scores with a new bar.
        Returns the current confidence assessment.
        """
        close = bar['close']
        high = bar['high']
        low = bar['low']

        self._total_bars += 1
        self._session_close = close
        self._close_history.append(close)

        # Track session extremes
        if high > self._session_high:
            self._session_high = high
        if low < self._session_low:
            self._session_low = low

        # Track IB position
        if close > self._ib_high:
            self._bars_above_ib += 1
        elif close < self._ib_low:
            self._bars_below_ib += 1
        else:
            self._bars_inside_ib += 1

        if close > self._ib_mid:
            self._bars_above_mid += 1
        else:
            self._bars_below_mid += 1

        # Track extension
        if close > self._ib_mid:
            ext = (close - self._ib_mid) / self._ib_range if self._ib_range > 0 else 0
            self._max_extension_up = max(self._max_extension_up, ext)
        else:
            ext = (self._ib_mid - close) / self._ib_range if self._ib_range > 0 else 0
            self._max_extension_down = max(self._max_extension_down, ext)

        # Track one-timeframing
        if self._prev_close is not None:
            if close > self._prev_close:
                self._otf_up_streak += 1
                self._otf_down_streak = 0
            elif close < self._prev_close:
                self._otf_down_streak += 1
                self._otf_up_streak = 0
            # else: no change
        self._max_otf_up = max(self._max_otf_up, self._otf_up_streak)
        self._max_otf_down = max(self._max_otf_down, self._otf_down_streak)
        self._prev_close = close

        # Track price distribution (for profile shape)
        bin_size = max(self._ib_range * 0.1, 1.0)
        price_bin = int(close / bin_size)
        self._price_bins[price_bin] = self._price_bins.get(price_bin, 0) + 1

        # Track DPOC migration (estimate: mode of close prices so far)
        if len(self._close_history) >= 5:
            # Simple DPOC estimate: most common price bin
            recent_bins = [int(c / bin_size) for c in self._close_history[-20:]]
            if recent_bins:
                mode_bin = max(set(recent_bins), key=recent_bins.count)
                self._dpoc_estimates.append(mode_bin * bin_size)

        # Compute confidence scores
        return self._compute_confidence()

    def _compute_confidence(self) -> DayTypeConfidence:
        """Compute confidence for each day type based on accumulated evidence."""
        confidence = DayTypeConfidence()

        if self._total_bars < 5:
            # Not enough data yet
            confidence.neutral = 0.5
            return confidence

        # === Trend Day Bull Checklist ===
        trend_bull_checks = {}

        # 1. Extension >= 0.5x (moderate+)
        trend_bull_checks['extension_moderate'] = self._max_extension_up >= 0.5

        # 2. Extension >= 1.0x (strong)
        trend_bull_checks['extension_strong'] = self._max_extension_up >= 1.0

        # 3. Bars above IB > 60% of total
        pct_above = self._bars_above_ib / self._total_bars if self._total_bars > 0 else 0
        trend_bull_checks['time_above_ib'] = pct_above > 0.6

        # 4. One-timeframing (>=4 consecutive bars up)
        trend_bull_checks['one_timeframing'] = self._max_otf_up >= 4

        # 5. DPOC migration upward (estimate)
        if len(self._dpoc_estimates) >= 3:
            dpoc_diff = self._dpoc_estimates[-1] - self._dpoc_estimates[0]
            trend_bull_checks['dpoc_migration'] = dpoc_diff > self._ib_range * 0.3
        else:
            trend_bull_checks['dpoc_migration'] = False

        # 6. Narrow profile (few unique price bins relative to range)
        session_range = self._session_high - self._session_low
        bin_size = max(self._ib_range * 0.1, 1.0)
        expected_bins = session_range / bin_size if bin_size > 0 else 1
        actual_bins = len(self._price_bins)
        # Skinny profile: actual bins used vs expected - lower ratio = skinnier
        bin_usage = actual_bins / max(expected_bins, 1)
        trend_bull_checks['skinny_profile'] = bin_usage < 0.7

        # 7. No bear extension (directional, not rotational)
        trend_bull_checks['no_bear_ext'] = self._max_extension_down < 0.3

        # 8. Bars above IB midpoint > 70%
        pct_above_mid = self._bars_above_mid / self._total_bars if self._total_bars > 0 else 0
        trend_bull_checks['directional_bias'] = pct_above_mid > 0.7

        confidence.trend_bull = sum(trend_bull_checks.values()) / len(trend_bull_checks)
        confidence.checklist['trend_bull'] = trend_bull_checks

        # === Trend Day Bear Checklist ===
        trend_bear_checks = {}
        trend_bear_checks['extension_moderate'] = self._max_extension_down >= 0.5
        trend_bear_checks['extension_strong'] = self._max_extension_down >= 1.0
        pct_below = self._bars_below_ib / self._total_bars if self._total_bars > 0 else 0
        trend_bear_checks['time_below_ib'] = pct_below > 0.6
        trend_bear_checks['one_timeframing'] = self._max_otf_down >= 4
        if len(self._dpoc_estimates) >= 3:
            dpoc_diff = self._dpoc_estimates[0] - self._dpoc_estimates[-1]
            trend_bear_checks['dpoc_migration'] = dpoc_diff > self._ib_range * 0.3
        else:
            trend_bear_checks['dpoc_migration'] = False
        trend_bear_checks['skinny_profile'] = bin_usage < 0.7
        trend_bear_checks['no_bull_ext'] = self._max_extension_up < 0.3
        pct_below_mid = self._bars_below_mid / self._total_bars if self._total_bars > 0 else 0
        trend_bear_checks['directional_bias'] = pct_below_mid > 0.7

        confidence.trend_bear = sum(trend_bear_checks.values()) / len(trend_bear_checks)
        confidence.checklist['trend_bear'] = trend_bear_checks

        # === P-Day Bull (bullish skew) ===
        p_bull_checks = {}
        p_bull_checks['moderate_ext'] = 0.3 <= self._max_extension_up <= 1.5
        p_bull_checks['skew_bull'] = pct_above_mid > 0.55
        pct_inside = self._bars_inside_ib / self._total_bars if self._total_bars > 0 else 0
        p_bull_checks['some_inside_time'] = pct_inside > 0.1
        p_bull_checks['not_extreme_otf'] = self._max_otf_up < 8  # not a full trend
        p_bull_checks['measured_range'] = session_range < self._atr * 1.5

        confidence.p_day_bull = sum(p_bull_checks.values()) / len(p_bull_checks)
        confidence.checklist['p_day_bull'] = p_bull_checks

        # === P-Day Bear (bearish skew) ===
        p_bear_checks = {}
        p_bear_checks['moderate_ext'] = 0.3 <= self._max_extension_down <= 1.5
        p_bear_checks['skew_bear'] = pct_below_mid > 0.55
        p_bear_checks['some_inside_time'] = pct_inside > 0.1
        p_bear_checks['not_extreme_otf'] = self._max_otf_down < 8
        p_bear_checks['measured_range'] = session_range < self._atr * 1.5

        confidence.p_day_bear = sum(p_bear_checks.values()) / len(p_bear_checks)
        confidence.checklist['p_day_bear'] = p_bear_checks

        # === B-Day ===
        b_day_checks = {}
        b_day_checks['weak_ext'] = max(self._max_extension_up, self._max_extension_down) < 0.5
        b_day_checks['mostly_inside'] = pct_inside > 0.4
        b_day_checks['narrow_range'] = session_range < self._ib_range * 1.8
        b_day_checks['no_otf'] = max(self._max_otf_up, self._max_otf_down) < 4
        b_day_checks['balanced'] = abs(pct_above_mid - 0.5) < 0.15

        confidence.b_day = sum(b_day_checks.values()) / len(b_day_checks)
        confidence.checklist['b_day'] = b_day_checks

        # === Neutral ===
        neutral_checks = {}
        neutral_checks['small_ext'] = max(self._max_extension_up, self._max_extension_down) < 0.8
        neutral_checks['balanced_time'] = abs(pct_above_mid - 0.5) < 0.2
        neutral_checks['wide_profile'] = bin_usage > 0.5
        neutral_checks['moderate_range'] = self._ib_range * 0.5 < session_range < self._atr * 1.5

        confidence.neutral = sum(neutral_checks.values()) / len(neutral_checks)
        confidence.checklist['neutral'] = neutral_checks

        return confidence
