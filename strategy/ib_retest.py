"""
Strategy: IB Retest (Post-OR IB Level Retest with CVD Confirmation)

Purpose: Trade the 10:30-11:30 window after OR trade completes and before
Edge Fade kicks in. Captures the IB level retest when price pulls back to
VWAP then re-engages the IB boundary with order flow confirmation.

Evidence (from diag_post_or_pullback.py, diag_htf_trend_setup.py):
- BEAR/SHORT retest of IBH after IBH established: $884, PF 1.38
- Wide IB BEAR (250+): 52% WR, +$432, PF 1.27
- Wide IB BULL (250+): 24% WR, -$1,166 (structural failure)
- CVD divergence at 2nd test = failure signal (Edge Fade catches that)

Two setups:
1. IB_RETEST_BULL: Price pulled back from IBH, CVD still bullish, re-enters
   near IBH -> long into PDH / weekly high extension
2. IB_RETEST_BEAR: Price pulled back from IBL, CVD still bearish, re-enters
   near IBL -> short into PDL / weekly low extension

Key design decisions:
- IB range 100-350 only (LOW = no clear IB, WIDE bull fails)
- CVD baseline at IB end filters out divergence setups (Edge Fade's territory)
- One signal per session per direction
- HTF targets: nearest prior-day or weekly level beyond 0.5x IB extension
"""

from datetime import time as _time
from typing import Optional, List
import pandas as pd
import numpy as np

from strategy.base import StrategyBase
from strategy.signal import Signal

# Time window
IB_RETEST_EARLIEST = _time(10, 30)
IB_RETEST_LATEST = _time(11, 30)

# IB range bounds (points)
IB_RETEST_MIN_RANGE = 100
IB_RETEST_MAX_RANGE = 350

# Proximity: how close to IB level for entry (fraction of IB range)
IB_RETEST_NEAR_PCT = 0.20  # Within 20% of IB range from the level
IB_RETEST_TOO_EXTENDED_PCT = 0.50  # Beyond 50% = too extended, skip

# Minimum extension for HTF target
IB_RETEST_MIN_TARGET_EXT = 0.50  # Target must be at least 0.5x IB beyond level


class IBRetestStrategy(StrategyBase):
    """
    Post-OR IB level retest strategy.

    After IB formation, if price confirms direction (broke IBH or IBL),
    wait for pullback to VWAP, then re-entry near the IB level with
    CVD confirmation. Targets HTF levels (PDH/PDL, weekly H/L).
    """

    @property
    def name(self) -> str:
        return "IB Retest"

    @property
    def applicable_day_types(self) -> List[str]:
        return ['neutral', 'b_day', 'trend_bull', 'trend_bear', 'p_day']

    def on_session_start(self, session_date, ib_high, ib_low, ib_range, session_context):
        self._ib_high = ib_high
        self._ib_low = ib_low
        self._ib_range = ib_range
        self._ib_mid = (ib_high + ib_low) / 2

        # CVD baseline at IB end
        ib_bars = session_context.get('ib_bars')
        if ib_bars is not None and 'cumulative_delta' in ib_bars.columns:
            cvd_val = ib_bars.iloc[-1]['cumulative_delta']
            self._cvd_baseline = cvd_val if not pd.isna(cvd_val) else None
        else:
            self._cvd_baseline = None

        # HTF targets from session context
        pdh = session_context.get('pdh')
        pdl = session_context.get('pdl')
        weekly_high = session_context.get('weekly_high')
        weekly_low = session_context.get('weekly_low')

        # BULL target: nearest resistance above IBH
        bull_candidates = []
        if pdh is not None and not pd.isna(pdh) and pdh > ib_high:
            bull_candidates.append(pdh)
        if weekly_high is not None and not pd.isna(weekly_high) and weekly_high > ib_high:
            bull_candidates.append(weekly_high)
        self._bull_htf_target = min(bull_candidates) if bull_candidates else None

        # BEAR target: nearest support below IBL
        bear_candidates = []
        if pdl is not None and not pd.isna(pdl) and pdl < ib_low:
            bear_candidates.append(pdl)
        if weekly_low is not None and not pd.isna(weekly_low) and weekly_low < ib_low:
            bear_candidates.append(weekly_low)
        self._bear_htf_target = max(bear_candidates) if bear_candidates else None

        # One signal per direction per session
        self._bull_fired = False
        self._bear_fired = False

        # Skip if IB range outside bounds
        self._active = IB_RETEST_MIN_RANGE <= ib_range <= IB_RETEST_MAX_RANGE

    def on_bar(self, bar: pd.Series, bar_index: int, session_context: dict) -> Optional[Signal]:
        if not self._active:
            return None

        # Time window gate
        bar_time = session_context.get('bar_time')
        if bar_time is None:
            return None
        if bar_time < IB_RETEST_EARLIEST or bar_time > IB_RETEST_LATEST:
            return None

        close = bar['close']

        # CVD check
        cvd = bar.get('cumulative_delta', None)
        if cvd is not None and pd.isna(cvd):
            cvd = None
        if self._cvd_baseline is None or cvd is None:
            return None

        # Try BEAR setup first (stronger evidence)
        if not self._bear_fired:
            signal = self._check_bear(bar, bar_index, session_context, close, cvd)
            if signal:
                return signal

        # BULL side disabled: 44.3% WR, -$2,362 in 259-session backtest
        # Wide IB bull = 24% WR structural failure; even MED/NORMAL loses
        # if not self._bull_fired:
        #     signal = self._check_bull(bar, bar_index, session_context, close, cvd)
        #     if signal:
        #         return signal

        return None

    def _check_bear(self, bar, bar_index, session_context, close, cvd) -> Optional[Signal]:
        """BEAR: price just below IBL, CVD bearish, short into PDL/weekly low."""
        ib_low = self._ib_low
        ib_range = self._ib_range

        # Price must be below IBL (confirmed bear IB)
        if close >= ib_low:
            return None

        # Near IBL: within 20% of IB range below it
        lower_bound = ib_low - ib_range * IB_RETEST_NEAR_PCT
        if close < lower_bound:
            return None

        # Not too extended
        if close < ib_low - ib_range * IB_RETEST_TOO_EXTENDED_PCT:
            return None

        # CVD must be bearish (below baseline at IB end)
        if cvd >= self._cvd_baseline:
            return None

        # Compute target
        min_target = ib_low - ib_range * IB_RETEST_MIN_TARGET_EXT
        if self._bear_htf_target is not None and self._bear_htf_target < min_target:
            target = self._bear_htf_target
        else:
            target = min_target

        # Stop at IBH (full IB range)
        stop = self._ib_high

        # Sanity: risk/reward
        risk = stop - close
        reward = close - target
        if risk <= 0 or reward <= 0:
            return None

        self._bear_fired = True

        return Signal(
            timestamp=bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp'),
            direction='SHORT',
            entry_price=close,
            stop_price=stop,
            target_price=target,
            strategy_name=self.name,
            setup_type='IB_RETEST_BEAR',
            day_type=session_context.get('day_type', 'neutral'),
            trend_strength=session_context.get('trend_strength', 'weak'),
            confidence='medium',
        )

    def _check_bull(self, bar, bar_index, session_context, close, cvd) -> Optional[Signal]:
        """BULL: price just above IBH, CVD bullish, long into PDH/weekly high."""
        ib_high = self._ib_high
        ib_range = self._ib_range

        # Price must be above IBH (confirmed bull IB)
        if close <= ib_high:
            return None

        # Near IBH: within 20% of IB range above it
        upper_bound = ib_high + ib_range * IB_RETEST_NEAR_PCT
        if close > upper_bound:
            return None

        # Not too extended
        if close > ib_high + ib_range * IB_RETEST_TOO_EXTENDED_PCT:
            return None

        # CVD must be bullish (above baseline at IB end)
        if cvd <= self._cvd_baseline:
            return None

        # Compute target
        min_target = ib_high + ib_range * IB_RETEST_MIN_TARGET_EXT
        if self._bull_htf_target is not None and self._bull_htf_target > min_target:
            target = self._bull_htf_target
        else:
            target = min_target

        # Stop at IBL (full IB range)
        stop = self._ib_low

        # Sanity: risk/reward
        risk = close - stop
        reward = target - close
        if risk <= 0 or reward <= 0:
            return None

        self._bull_fired = True

        return Signal(
            timestamp=bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp'),
            direction='LONG',
            entry_price=close,
            stop_price=stop,
            target_price=target,
            strategy_name=self.name,
            setup_type='IB_RETEST_BULL',
            day_type=session_context.get('day_type', 'neutral'),
            trend_strength=session_context.get('trend_strength', 'weak'),
            confidence='medium',
        )
