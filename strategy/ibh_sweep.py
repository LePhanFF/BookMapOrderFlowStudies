"""
Strategy 5: IBH Sweep + Failure Fade (SHORT)

On balance days, price sweeps above IBH into overhead liquidity (PDH),
then fails to hold. Trapped longs create selling pressure.

From v14 report:
  - Day type: b_day ONLY (23% WR on p_day)
  - Price sweeps above IBH, reaching near PDH or significant overhead level
  - Failure: bar closes below IBH despite the sweep
  - Delta < 0 on failure bar
  - Stop: sweep high + 15% of IB range
  - Target: IB midpoint

Performance (v14, 62 sessions):
  4 trades, 100% WR, $146/trade, PF INF, MaxDD $0
"""

from datetime import time as _time
from typing import Optional, List
import pandas as pd

from strategy.base import StrategyBase
from strategy.signal import Signal

IBH_SWEEP_MIN_IB_RATIO = 0.03     # Min sweep above IBH as fraction of IB range
IBH_SWEEP_PDH_PROXIMITY_ATR = 0.05  # Sweep must reach within 5% of ATR of PDH
IBH_SWEEP_LARGE_SWEEP_ATR = 0.075   # Large sweep threshold (no PDH required)
IBH_SWEEP_STOP_BUFFER = 0.15
IBH_SWEEP_LAST_ENTRY = _time(14, 0)
IBH_SWEEP_MAX_IB_RANGE_ATR = 1.5    # Max IB range as ATR multiple


class IBHSweepFail(StrategyBase):

    @property
    def name(self) -> str:
        return "IBH Sweep+Fail"

    @property
    def applicable_day_types(self) -> List[str]:
        return ['b_day']

    def on_session_start(self, session_date, ib_high, ib_low, ib_range, session_context):
        self._ib_high = ib_high
        self._ib_low = ib_low
        self._ib_range = ib_range
        self._ib_mid = (ib_high + ib_low) / 2
        self._atr = session_context.get('atr14', 200.0)
        # Use actual PDH from overnight computation, fallback to prior session
        self._pdh = session_context.get('pdh') or session_context.get('prior_session_high')
        self._london_high = session_context.get('london_high')
        self._entry_fired = False
        max_ib = self._atr * IBH_SWEEP_MAX_IB_RANGE_ATR
        self._active = ib_range > 0 and ib_range <= max_ib

    def on_bar(self, bar: pd.Series, bar_index: int, session_context: dict) -> Optional[Signal]:
        if not self._active or self._entry_fired:
            return None

        current_price = bar['close']
        bar_time = session_context.get('bar_time')

        if bar_time and bar_time >= IBH_SWEEP_LAST_ENTRY:
            return None

        # Day type gate: b_day ONLY
        day_type = session_context.get('day_type', '')
        if day_type != 'b_day':
            return None

        # b_day confidence must be established
        b_day_conf = session_context.get('b_day_confidence', 0.0)
        if b_day_conf < 0.3:
            return None

        # Sweep + failure on THIS bar
        sweep_pts = bar['high'] - self._ib_high
        min_sweep = self._ib_range * IBH_SWEEP_MIN_IB_RATIO
        if sweep_pts < min_sweep:
            return None

        # Sweep must reach near PDH (overhead liquidity target)
        pdh_proximity = self._atr * IBH_SWEEP_PDH_PROXIMITY_ATR
        large_sweep = self._atr * IBH_SWEEP_LARGE_SWEEP_ATR
        if self._pdh is not None:
            near_pdh = bar['high'] >= self._pdh - pdh_proximity
            if not near_pdh:
                # Without PDH proximity, require a very large sweep
                if sweep_pts < large_sweep:
                    return None

        # Failure: close must be below IBH with rejection wick
        if current_price >= self._ib_high:
            return None

        # Rejection wick: upper wick must be significant (> 50% of bar range)
        bar_range = bar['high'] - bar['low']
        upper_wick = bar['high'] - max(current_price, bar['open'])
        if bar_range > 0 and upper_wick / bar_range < 0.3:
            return None  # Not a convincing rejection

        # Delta confirmation
        delta = bar.get('delta', 0)
        if pd.isna(delta):
            delta = 0
        if delta >= 0:
            return None

        # Stop and target
        stop_price = bar['high'] + self._ib_range * IBH_SWEEP_STOP_BUFFER
        target_price = self._ib_mid

        risk = stop_price - current_price
        reward = current_price - target_price
        if risk <= 0 or reward <= 0:
            return None

        self._entry_fired = True

        return Signal(
            timestamp=bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp'),
            direction='SHORT',
            entry_price=current_price,
            stop_price=stop_price,
            target_price=target_price,
            strategy_name=self.name,
            setup_type='IBH_SWEEP_FAIL',
            day_type=day_type,
            trend_strength=session_context.get('trend_strength', 'weak'),
            confidence='high',
            metadata={'sweep_pts': sweep_pts},
        )
