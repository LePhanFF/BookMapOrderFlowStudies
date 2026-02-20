"""
Strategy 7: Opening Range Reversal (LONG and SHORT)

Trades the ICT "Judas Swing" at market open. In the first 30 minutes
(9:30-10:00), price makes a false move to sweep pre-market liquidity
(overnight H/L, PDH/PDL), then reverses. Enter on the reversal.

From v14 report:
  - Time: 9:30-10:30 ET (opening range period)
  - OR extreme sweeps overnight H/L (within 30 pts)
  - Price reverses through OR midpoint
  - VWAP aligned (entry within 20 pts of VWAP)
  - Stop: sweep extreme + 15% of EOR range
  - Target: 2R
  - NOT fading the opening drive (fades = 40% WR)

Performance (v14, 62 sessions):
  20 trades (optimized), 80% WR, $190/trade, PF 6.3, MaxDD -$407

Implementation note: This strategy scans the IB bars (passed via
session_context['ib_bars']) during on_session_start() to detect
OR reversal setups that occur 9:30-10:30. Cached signals are
emitted on the first post-IB on_bar() call.

Detection logic (matches diagnostic_opening_range_smt.py):
  1. OR = first 15 bars (9:30-9:44), EOR = first 30 bars (9:30-9:59)
  2. Sweep: EOR high/low near a key level (within 30 pts)
  3. Reversal: after the extreme bar, price closes beyond OR mid
  4. NOT fading: if opening drive is strong, only trade with it
     - DRIVE_UP + SHORT = FADE (excluded, 40% WR)
     - DRIVE_DOWN + LONG = FADE (excluded)
     - DRIVE_UP + LONG = CONTINUATION (allowed)
     - DRIVE_DOWN + SHORT = CONTINUATION (allowed)
     - ROTATION + any = allowed
"""

from datetime import time as _time
from typing import Optional, List
import pandas as pd
import numpy as np

from strategy.base import StrategyBase
from strategy.signal import Signal

# Constants - matching diagnostic_opening_range_smt.py
OR_BARS = 15                      # First 15 bars = Opening Range (9:30-9:44)
EOR_BARS = 30                     # First 30 bars = Extended OR (9:30-9:59)
SWEEP_THRESHOLD_ATR = 0.10        # Level must be within 10% of ATR to count as sweep
VWAP_ALIGNED_ATR = 0.10           # Entry must be within 10% of ATR of VWAP
OR_STOP_BUFFER = 0.15             # Stop: sweep extreme + 15% of EOR range (ratio-based)
MIN_RISK_PTS = 5.0                # Minimum risk in points (microstructure floor, not vol-dependent)
MAX_RISK_ATR = 1.0                # Maximum risk as ATR multiple
MIN_EOR_RANGE_ATR = 0.05          # Minimum EOR range as ATR multiple
DRIVE_THRESHOLD = 0.4             # Opening drive classification threshold (ratio-based)


class OpeningRangeReversal(StrategyBase):
    """
    Opening Range Reversal: fade the Judas Swing at the open.

    Scans the IB bars for OR reversal setups during on_session_start().
    A valid setup requires:
      1. EOR extreme sweeps an overnight/prior-day level
      2. Price reverses through OR midpoint after the sweep extreme
      3. Entry is VWAP-aligned
      4. Not fading the opening drive
    """

    @property
    def name(self) -> str:
        return "Opening Range Rev"

    @property
    def applicable_day_types(self) -> List[str]:
        return []  # All day types (this strategy runs before day type is established)

    @property
    def trades_pre_ib(self) -> bool:
        return True

    def on_session_start(self, session_date, ib_high, ib_low, ib_range, session_context):
        self._cached_signal = None
        self._signal_emitted = False
        self._ib_range = ib_range

        ib_bars = session_context.get('ib_bars')
        if ib_bars is None or len(ib_bars) < EOR_BARS:
            return

        # ATR for adaptive thresholds
        atr = session_context.get('atr14', 200.0)
        self._atr = atr
        vwap_threshold = atr * VWAP_ALIGNED_ATR
        max_risk = atr * MAX_RISK_ATR

        # OR = first 15 bars (9:30-9:44)
        or_bars = ib_bars.iloc[:OR_BARS]
        or_high = or_bars['high'].max()
        or_low = or_bars['low'].min()
        or_mid = (or_high + or_low) / 2

        # EOR = first 30 bars (9:30-9:59) for sweep detection
        eor_bars = ib_bars.iloc[:EOR_BARS]
        eor_high = eor_bars['high'].max()
        eor_low = eor_bars['low'].min()
        eor_range = eor_high - eor_low

        if eor_range < atr * MIN_EOR_RANGE_ATR:
            return

        # Classify opening drive from first 5 bars
        first_5 = ib_bars.iloc[:5]
        open_price = first_5.iloc[0]['open']
        close_5th = first_5.iloc[4]['close']
        drive_range = first_5['high'].max() - first_5['low'].min()
        if drive_range > 0:
            drive_pct = (close_5th - open_price) / drive_range
        else:
            drive_pct = 0

        # Classify drive direction
        if drive_pct > DRIVE_THRESHOLD:
            opening_drive = 'DRIVE_UP'
        elif drive_pct < -DRIVE_THRESHOLD:
            opening_drive = 'DRIVE_DOWN'
        else:
            opening_drive = 'ROTATION'

        # Get overnight levels from session context (computed from ETH data)
        overnight_high = session_context.get('overnight_high') or session_context.get('prior_session_high')
        overnight_low = session_context.get('overnight_low') or session_context.get('prior_session_low')

        if overnight_high is None or overnight_low is None:
            return

        pdh = session_context.get('pdh') or session_context.get('prior_session_high')
        pdl = session_context.get('pdl') or session_context.get('prior_session_low')
        asia_high = session_context.get('asia_high')
        asia_low = session_context.get('asia_low')

        # Check for sweeps of key levels (matching diagnostic logic)
        high_levels = [overnight_high]
        if pdh: high_levels.append(pdh)
        if asia_high: high_levels.append(asia_high)
        low_levels = [overnight_low]
        if pdl: low_levels.append(pdl)
        if asia_low: low_levels.append(asia_low)

        # Sweep: EOR extreme is near a key level (ATR-adaptive threshold)
        sweep_threshold = atr * SWEEP_THRESHOLD_ATR

        swept_high_level = None
        for lvl in high_levels:
            if lvl is not None and abs(eor_high - lvl) < sweep_threshold:
                swept_high_level = lvl
                break

        swept_low_level = None
        for lvl in low_levels:
            if lvl is not None and abs(eor_low - lvl) < sweep_threshold:
                swept_low_level = lvl
                break

        if swept_high_level is None and swept_low_level is None:
            return

        # Find the extreme bars in the EOR
        high_bar_idx = eor_bars['high'].idxmax()
        low_bar_idx = eor_bars['low'].idxmin()

        # === SHORT SETUP: Judas swing UP, then reversal DOWN ===
        if swept_high_level is not None:
            # NOT fading filter: if drive is UP and we SHORT, that's a FADE → skip
            if opening_drive == 'DRIVE_UP':
                pass  # FADE - skip this setup
            else:
                # Scan bars AFTER the high extreme for reversal below OR mid
                post_high = ib_bars.loc[high_bar_idx:]
                for j in range(1, min(30, len(post_high))):
                    bar = post_high.iloc[j]
                    price = bar['close']

                    if price >= or_mid:
                        continue

                    # VWAP alignment
                    vwap = bar.get('vwap', np.nan)
                    if pd.isna(vwap) or abs(price - vwap) > vwap_threshold:
                        continue

                    # Delta confirmation
                    delta = bar.get('delta', bar.get('vol_delta', 0))
                    if pd.isna(delta):
                        delta = 0
                    if delta >= 0:
                        continue

                    stop = eor_high + eor_range * OR_STOP_BUFFER
                    risk = stop - price
                    if risk < MIN_RISK_PTS or risk > max_risk:
                        continue
                    target = price - 2 * risk

                    bar_ts = bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp')
                    self._cached_signal = Signal(
                        timestamp=bar_ts,
                        direction='SHORT',
                        entry_price=price,
                        stop_price=stop,
                        target_price=target,
                        strategy_name=self.name,
                        setup_type='OR_REVERSAL_SHORT',
                        day_type='neutral',
                        trend_strength='moderate',
                        confidence='high',
                        metadata={
                            'level_swept': 'high',
                            'sweep_pts': eor_high - or_mid,
                            'vwap_aligned': True,
                            'opening_drive': opening_drive,
                        },
                    )
                    return

        # === LONG SETUP: Judas swing DOWN, then reversal UP ===
        if swept_low_level is not None:
            # NOT fading filter: if drive is DOWN and we LONG, that's a FADE → skip
            if opening_drive == 'DRIVE_DOWN':
                pass  # FADE - skip this setup
            else:
                post_low = ib_bars.loc[low_bar_idx:]
                for j in range(1, min(30, len(post_low))):
                    bar = post_low.iloc[j]
                    price = bar['close']

                    if price <= or_mid:
                        continue

                    # VWAP alignment
                    vwap = bar.get('vwap', np.nan)
                    if pd.isna(vwap) or abs(price - vwap) > vwap_threshold:
                        continue

                    # Delta confirmation
                    delta = bar.get('delta', bar.get('vol_delta', 0))
                    if pd.isna(delta):
                        delta = 0
                    if delta <= 0:
                        continue

                    stop = eor_low - eor_range * OR_STOP_BUFFER
                    risk = price - stop
                    if risk < MIN_RISK_PTS or risk > max_risk:
                        continue
                    target = price + 2 * risk

                    bar_ts = bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp')
                    self._cached_signal = Signal(
                        timestamp=bar_ts,
                        direction='LONG',
                        entry_price=price,
                        stop_price=stop,
                        target_price=target,
                        strategy_name=self.name,
                        setup_type='OR_REVERSAL_LONG',
                        day_type='neutral',
                        trend_strength='moderate',
                        confidence='high',
                        metadata={
                            'level_swept': 'low',
                            'sweep_pts': or_mid - eor_low,
                            'vwap_aligned': True,
                            'opening_drive': opening_drive,
                        },
                    )
                    return

    def on_bar(self, bar: pd.Series, bar_index: int, session_context: dict) -> Optional[Signal]:
        # Emit cached signal on first bar after IB
        if self._cached_signal is not None and not self._signal_emitted:
            self._signal_emitted = True
            signal = self._cached_signal
            self._cached_signal = None
            return signal

        return None
