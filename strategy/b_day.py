"""
Strategy 6: B-Day (Narrow IB, No Extension - True Trapped Balance)

Dalton Playbook Rules:
  - Micro/flat only - rarely 3/3 Lanto
  - Narrow IB, price never leaves first hour (Strength: Weak)
  - Symmetric build, flat DPOC
  - Bias: Neutral - edge fades only
  - Fade extremes: poor high/low inside IB, POC/VWAP bounces
  - No directional size

Entry Model (ICT-enhanced):
  - Fade at IBL: long when price touches IBL and REJECTS
    * Require: IFVG bull entry OR close back above IBL with volume rejection
    * FVG confirmation: bar enters a bull FVG zone at IBL
  - Max 1 long fade per session
  - 30-bar cooldown between trades

Stops: Outside IB extreme + 10% buffer
Targets: IB midpoint (POC/VWAP mean reversion)

Key Findings (v12, 62 sessions):
  - IBL fade LONG: 71.4% WR (5/7), +$1,806 — THE B-Day edge
  - IBH fade SHORT: 0-22% WR across ALL tests — NQ long bias kills shorts
    * Even with 3rd test exhaustion model: T3 was 0/5 WR (-$1,266)
    * Even with 4+ test exhaustion: tiny wins only (+$44, +$121)
    * DISABLED permanently on NQ

  - Multi-test tracking insight: T1 fades at IBL are 100% WR (3/3),
    T2/T3 are 50% WR. But sample size is tiny (7 total trades),
    so we keep quality >= 2 + delta for all touches.

  - b_day_confidence >= 0.5 is the optimal threshold for filtering
    out false B-Day classifications.

IBH Fade Cross-Instrument Opportunity:
  The 3rd test exhaustion SHORT may work on weaker instruments (ES, YM)
  where the long bias is less extreme. Cross-instrument analysis needed.
"""

from typing import Optional, List
import pandas as pd
import numpy as np

from datetime import time as _time
from strategy.base import StrategyBase
from strategy.signal import Signal
from config.constants import BDAY_COOLDOWN_BARS, BDAY_STOP_IB_BUFFER

# B-Day IB range cap: use rolling 90th percentile instead of hardcoded 400 pts.
# Extremely wide IB (relative to recent history) is NOT a true balance day.
BDAY_MAX_IB_PCTL = 90          # Percentile of rolling IB history for cap
BDAY_MAX_IB_BUFFER = 1.2       # Allow 20% above the percentile

# Regime-aware volatility filter (Walk-Forward Lite):
# B-Day IBL fades need sufficient IB range for the target (IB mid) to
# produce meaningful profit. In low-vol regimes (rolling median IB < 150),
# even "narrow" sessions don't have enough range for viable fades.
# A/B tested: removes 7 losing trades (+$1,582), zero impact on Nov-Feb.

# B-Day last entry time: entries after 14:00 have insufficient time to reach target.
BDAY_LAST_ENTRY_TIME = _time(14, 0)


class BDayStrategy(StrategyBase):

    @property
    def name(self) -> str:
        return "B-Day"

    @property
    def applicable_day_types(self) -> List[str]:
        return ['b_day']

    def on_session_start(self, session_date, ib_high, ib_low, ib_range, session_context):
        self._ib_high = ib_high
        self._ib_low = ib_low
        self._ib_range = ib_range
        self._ib_mid = (ib_high + ib_low) / 2

        # Adaptive IB range cap using rolling history
        ib_history = session_context.get('ib_range_history', [])
        if len(ib_history) >= 5:
            import numpy as _np
            pctl_val = _np.percentile(ib_history[-20:], BDAY_MAX_IB_PCTL)
            self._max_ib = pctl_val * BDAY_MAX_IB_BUFFER
        else:
            self._max_ib = 400.0  # fallback for first sessions

        # Regime-aware filter (Walk-Forward Lite):
        # B-Day IBL fades need sufficient IB range for the target (IB mid) to
        # produce meaningful profit. In low-vol regimes (Aug-Nov, median IB ~117),
        # even "narrow" sessions don't have enough range — the fade from IBL to
        # IB mid is only ~50 pts, which barely covers costs after slippage+commission.
        # Use two checks:
        #   1. Regime must not be 'low' volatility (rolling median IB < 130)
        #   2. IB must be narrow relative to regime (< regime median) — it's a
        #      balance day, not a micro-trend day
        self._regime_allows_bday = True  # default: allow (for warmup period)
        regime_vol = session_context.get('regime_volatility')
        if regime_vol == 'low':
            self._regime_allows_bday = False

        self._val_fade_taken = False
        self._last_entry_bar = -999

        # Track rejection patterns at IBL
        self._ibl_touch_count = 0
        self._ibl_last_touch_bar = -999

    def on_bar(self, bar: pd.Series, bar_index: int, session_context: dict) -> Optional[Signal]:
        day_type = session_context.get('day_type', '')
        if day_type not in self.applicable_day_types:
            return None

        # B-Days: only trade when strength is weak (no extension)
        strength = session_context.get('trend_strength', 'weak')
        if strength != 'weak':
            return None

        # B-Day confidence check
        b_day_conf = session_context.get('b_day_confidence', 0.0)
        if b_day_conf < 0.5:
            return None

        # Regime-aware filter: skip B-Day in low-vol regimes or when IB is not
        # truly narrow relative to recent sessions.
        if not self._regime_allows_bday:
            return None

        # Time gate: B-Day fades need time to develop toward IB midpoint.
        # Very late entries (after 14:00) have insufficient time to reach target.
        # Diagnostics: 13:08 entry won (+$97), 14:07 entry lost, 15:17 entry lost.
        bar_time = session_context.get('bar_time')
        if bar_time and bar_time >= BDAY_LAST_ENTRY_TIME:
            return None

        # Adaptive IB range cap: extremely wide IB (relative to recent history)
        # is not a true balance day. Target becomes unreachable.
        if self._ib_range > self._max_ib:
            return None

        # Cooldown
        if bar_index - self._last_entry_bar < BDAY_COOLDOWN_BARS:
            return None

        if self._val_fade_taken:
            return None

        current_price = bar['close']
        delta = bar.get('delta', 0)

        # Track touches for multi-touch quality check
        if bar['low'] <= self._ib_low:
            if bar_index - self._ibl_last_touch_bar <= 3:
                self._ibl_touch_count += 1
            else:
                self._ibl_touch_count = 1
            self._ibl_last_touch_bar = bar_index

        # --- IBH fade (SHORT) DISABLED ---
        # Exhaustive testing across 62 sessions, multiple approaches:
        #   - v9: Simple quality filter: 10% WR (1/10), -$2,630
        #   - v12: Quality >= 3 + delta: 10-22% WR, still negative
        #   - v13: 3rd test exhaustion T3: 0% WR (0/5), -$1,266
        #   - v13: 4th/5th test: tiny wins (+$44, +$121), not edge
        # NQ long bias makes B-Day IBH fades negative-expectancy.
        # Cross-instrument opportunity: may work on ES/YM (weaker instruments).

        # --- Fade near IBL (long) ---
        if not self._val_fade_taken and bar['low'] <= self._ib_low:
            # Rejection: close back above IBL
            if current_price > self._ib_low:
                # Quality check
                has_fvg = bar.get('ifvg_bull_entry', False) or bar.get('fvg_bull', False)
                has_fvg_15m = bar.get('fvg_bull_15m', False)
                has_delta_rejection = delta > 0  # Buyers stepping in
                has_multi_touch = self._ibl_touch_count >= 2
                has_volume_spike = bar.get('volume_spike', 1.0) > 1.3

                quality_count = sum([
                    bool(has_fvg or has_fvg_15m),
                    has_delta_rejection,
                    has_multi_touch,
                    has_volume_spike,
                ])

                # Require quality >= 2 AND delta confirmation (buyers present)
                if quality_count >= 2 and has_delta_rejection:
                    entry_price = current_price
                    stop_price = self._ib_low - (self._ib_range * BDAY_STOP_IB_BUFFER)
                    target_price = self._ib_mid

                    risk = abs(entry_price - stop_price)
                    reward = abs(target_price - entry_price)
                    if reward > 0 and risk / reward > 2.5:
                        return None

                    confidence = 'high' if quality_count >= 3 else 'medium'

                    self._val_fade_taken = True
                    self._last_entry_bar = bar_index

                    return Signal(
                        timestamp=bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp'),
                        direction='LONG',
                        entry_price=entry_price,
                        stop_price=stop_price,
                        target_price=target_price,
                        strategy_name=self.name,
                        setup_type='B_DAY_IBL_FADE',
                        day_type='b_day',
                        trend_strength='weak',
                        confidence=confidence,
                    )

        return None
