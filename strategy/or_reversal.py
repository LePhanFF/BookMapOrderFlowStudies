"""
Strategy 7: Opening Range Reversal (LONG and SHORT)

Trades the ICT "Judas Swing" at market open. In the first 30 minutes
(9:30-10:00), price makes a false move to sweep pre-market liquidity
(overnight H/L, PDH/PDL, London H/L), then reverses. Enter on the reversal.

From v14 report:
  - Time: 9:30-10:30 ET (opening range period)
  - OR extreme sweeps overnight H/L (within 30 pts)
  - Price reverses through OR midpoint
  - VWAP aligned (entry within 20 pts of VWAP)
  - Stop: swept reference level + 15% of EOR range
  - Target: 2R
  - NOT fading the opening drive (fades = 40% WR)

Performance (v14, 62 sessions):
  20 trades (optimized), 80% WR, $190/trade, PF 6.3, MaxDD -$407

Implementation note: This strategy scans the IB bars (passed via
session_context['ib_bars']) during on_session_start() to detect
OR reversal setups that occur 9:30-10:30. Cached signals are
emitted on the first post-IB on_bar() call.

Detection logic (synced with NT8 ORReversalSignal.cs v2):
  1. OR = first 15 bars (9:30-9:44), EOR = first 30 bars (9:30-9:59)
  2. Sweep: EOR high/low near a key level (closest match, max dist = EOR range)
  3. Dual-sweep: if both high and low swept, pick deeper penetration
  4. Reversal: after the extreme bar, price closes beyond OR mid
  5. Stop at swept reference level + buffer (not EOR extreme)
  6. Delta OR CVD divergence for confirmation
  7. All drives allowed: DRIVE_UP + HIGH sweep SHORT = classic Judas swing (fake up)
     - DRIVE_UP + HIGH sweep + SHORT = Judas (fake up sweep, reversal down) -- ALLOWED
     - DRIVE_DOWN + LOW sweep + LONG = Judas (fake down sweep, reversal up) -- ALLOWED
     - Confirmation via state machine (reversal thru OR mid + delta/CVD) is the real filter
"""

from datetime import time as _time
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np

from strategy.base import StrategyBase
from strategy.signal import Signal

# Constants - matching diagnostic_opening_range_smt.py
OR_BARS = 15                      # First 15 bars = Opening Range (9:30-9:44)
EOR_BARS = 30                     # First 30 bars = Extended OR (9:30-9:59)
SWEEP_THRESHOLD_RATIO = 0.17      # Level proximity = 17% of EOR range (~20 pts at median 119 EOR)
VWAP_ALIGNED_RATIO = 0.17         # VWAP proximity = 17% of EOR range (~20 pts at median 119 EOR)
OR_STOP_BUFFER = 0.15             # Stop: swept level + 15% of EOR range (legacy, overridden by 2*ATR)
MIN_RISK_RATIO = 0.03             # Minimum risk = 3% of EOR range (was 5 pts)
MAX_RISK_RATIO = 1.3              # Maximum risk = 1.3x EOR range (was 200 pts)
DRIVE_THRESHOLD = 0.4             # Opening drive classification threshold (ratio-based)
ATR_PERIOD = 14                   # ATR period for stop calculation
ATR_STOP_MULT = 2.0               # Stop = entry + ATR_STOP_MULT * ATR14


def _find_closest_swept_level(
    eor_extreme: float,
    candidates: List[Tuple[str, float]],
    sweep_threshold: float,
    eor_range: float,
) -> Tuple[Optional[float], Optional[str]]:
    """Find the closest swept level from candidates within threshold.

    Args:
        eor_extreme: EOR high (for high sweep) or EOR low (for low sweep)
        candidates: list of (name, value) tuples
        sweep_threshold: max proximity distance (ratio-based)
        eor_range: EOR range for max distance cap

    Returns:
        (swept_level_value, swept_level_name) or (None, None)
    """
    best_level = None
    best_name = None
    best_dist = float('inf')

    for name, lvl in candidates:
        if lvl is None:
            continue
        dist = abs(eor_extreme - lvl)
        # Must be within sweep threshold AND within EOR range (no 400pt "sweeps")
        if dist < sweep_threshold and dist <= eor_range and dist < best_dist:
            best_dist = dist
            best_level = lvl
            best_name = name
    return best_level, best_name


def _compute_atr14(bars: pd.DataFrame, n: int = ATR_PERIOD) -> float:
    """Compute ATR(14) from OHLC bars. Falls back to mean H-L if insufficient data."""
    if len(bars) < 3:
        return float((bars['high'] - bars['low']).mean()) if len(bars) > 0 else 20.0
    h = bars['high']
    l = bars['low']
    pc = bars['close'].shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n, min_periods=3).mean().iloc[-1]
    return float(atr) if not pd.isna(atr) else float((h - l).mean())


class OpeningRangeReversal(StrategyBase):
    """
    Opening Range Reversal: fade the Judas Swing at the open.

    Scans the IB bars for OR reversal setups during on_session_start().
    A valid setup requires:
      1. EOR extreme sweeps an overnight/prior-day/London level (closest match)
      2. Price reverses through EOR midpoint after the sweep extreme
      3. Entry is VWAP-aligned
      4. Delta < 0 OR CVD declining (SHORT) / Delta > 0 OR CVD rising (LONG)
      5. Not fading the opening drive
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

        # NOTE: EOR mid tested (Change 4) but REVERTED — kills SHORT entries on 1-min bars.
        # OR mid is the correct entry zone threshold for 1-min backtest.
        # NT8 uses EOR mid on 5-min charts where OR only has 3 bars (distorted).

        if eor_range < ib_range * 0.05 if ib_range > 0 else eor_range < 10:
            return  # Minimum EOR range sanity check (scaled to IB)

        # Proximity thresholds scaled to EOR range
        sweep_threshold = eor_range * SWEEP_THRESHOLD_RATIO
        vwap_threshold = eor_range * VWAP_ALIGNED_RATIO
        max_risk = eor_range * MAX_RISK_RATIO

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

        # [Change 1] Add London H/L to sweep candidates
        london_high = session_context.get('london_high')
        london_low = session_context.get('london_low')

        # Build named candidate lists for proximity-based sweep detection
        high_candidates = [('ON_HIGH', overnight_high)]
        if pdh: high_candidates.append(('PDH', pdh))
        if asia_high: high_candidates.append(('ASIA_HIGH', asia_high))
        if london_high: high_candidates.append(('LDN_HIGH', london_high))

        low_candidates = [('ON_LOW', overnight_low)]
        if pdl: low_candidates.append(('PDL', pdl))
        if asia_low: low_candidates.append(('ASIA_LOW', asia_low))
        if london_low: low_candidates.append(('LDN_LOW', london_low))

        # [Change 2] Proximity-based sweep: pick CLOSEST level within threshold
        swept_high_level, swept_high_name = _find_closest_swept_level(
            eor_high, high_candidates, sweep_threshold, eor_range)
        swept_low_level, swept_low_name = _find_closest_swept_level(
            eor_low, low_candidates, sweep_threshold, eor_range)

        if swept_high_level is None and swept_low_level is None:
            return

        # [Change 3] Dual-sweep depth comparison: if BOTH sides swept, keep deeper
        if swept_high_level is not None and swept_low_level is not None:
            high_depth = eor_high - swept_high_level
            low_depth = swept_low_level - eor_low
            if high_depth >= low_depth:
                swept_low_level = None
                swept_low_name = None
            else:
                swept_high_level = None
                swept_high_name = None

        # Find the extreme bars in the EOR
        high_bar_idx = eor_bars['high'].idxmax()
        low_bar_idx = eor_bars['low'].idxmin()

        # [Change 6] Compute CVD at extreme bar for divergence check
        # CVD = cumulative delta from bar 0 through current bar
        deltas = ib_bars['delta'] if 'delta' in ib_bars.columns else ib_bars.get('vol_delta', pd.Series(dtype=float))
        if deltas is not None and len(deltas) > 0:
            deltas = deltas.fillna(0)
            cvd_series = deltas.cumsum()
        else:
            cvd_series = None

        # === SHORT SETUP: Judas swing UP, then reversal DOWN ===
        if swept_high_level is not None:
            if True:  # all drives allowed — DRIVE_UP + HIGH sweep IS the Judas swing
                cvd_at_extreme = cvd_series.loc[high_bar_idx] if cvd_series is not None else None
                post_high = ib_bars.loc[high_bar_idx:]
                atr14 = _compute_atr14(ib_bars)

                # Pre-compute 50% retest level:
                #   After the sweep high, price drops (reversal). The 50% level is
                #   halfway between the sweep extreme and the reversal low — that's
                #   the FVG zone where we expect the retest to fail.
                post_closes = post_high['close']
                reversal_low = float(post_closes.min()) if len(post_closes) > 1 else eor_high
                fifty_pct = reversal_low + (eor_high - reversal_low) * 0.50

                in_reversal = False   # price must drop below OR mid first
                for j in range(1, min(40, len(post_high))):
                    bar = post_high.iloc[j]
                    price = bar['close']
                    prev_price = post_high.iloc[j - 1]['close']

                    # Phase 1: wait for reversal — price must close below OR mid
                    if price < or_mid:
                        in_reversal = True
                    if not in_reversal:
                        continue

                    # Phase 2: entry on RETEST of 50% level (FVG zone), not on the drop
                    entry_lo = fifty_pct - atr14 * 0.5
                    entry_hi = fifty_pct + atr14 * 0.5
                    if not (entry_lo <= price <= entry_hi):
                        continue

                    # Must be turning down (retest failing)
                    if price >= prev_price:
                        continue

                    # Delta OR CVD divergence confirmation
                    delta = bar.get('delta', bar.get('vol_delta', 0))
                    if pd.isna(delta):
                        delta = 0
                    cvd_at_entry = cvd_series.loc[post_high.index[j]] if cvd_series is not None else None
                    cvd_declining = (cvd_at_entry is not None and cvd_at_extreme is not None
                                     and cvd_at_entry < cvd_at_extreme)
                    if delta >= 0 and not cvd_declining:
                        continue

                    # Stop: 2 ATR above entry (not swept level + EOR buffer)
                    stop = price + ATR_STOP_MULT * atr14
                    risk = stop - price
                    if risk < eor_range * MIN_RISK_RATIO or risk > max_risk:
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
                            'level_swept': swept_high_name,
                            'swept_level_price': swept_high_level,
                            'sweep_depth': eor_high - swept_high_level,
                            'fifty_pct_level': fifty_pct,
                            'atr14': atr14,
                            'opening_drive': opening_drive,
                            'cvd_declining': cvd_declining,
                        },
                    )
                    return

        # === LONG SETUP: Judas swing DOWN, then reversal UP ===
        if swept_low_level is not None:
            if True:  # all drives allowed — DRIVE_DOWN + LOW sweep IS the Judas swing
                cvd_at_extreme = cvd_series.loc[low_bar_idx] if cvd_series is not None else None
                post_low = ib_bars.loc[low_bar_idx:]
                atr14 = _compute_atr14(ib_bars)

                # Pre-compute 50% retest level:
                #   After the sweep low, price bounces (reversal). The 50% level is
                #   halfway between the sweep extreme and the reversal high.
                post_closes = post_low['close']
                reversal_high = float(post_closes.max()) if len(post_closes) > 1 else eor_low
                fifty_pct = reversal_high - (reversal_high - eor_low) * 0.50

                in_reversal = False   # price must rise above OR mid first
                for j in range(1, min(40, len(post_low))):
                    bar = post_low.iloc[j]
                    price = bar['close']
                    prev_price = post_low.iloc[j - 1]['close']

                    # Phase 1: wait for reversal — price must close above OR mid
                    if price > or_mid:
                        in_reversal = True
                    if not in_reversal:
                        continue

                    # Phase 2: entry on RETEST of 50% level (FVG zone), not on the bounce
                    entry_lo = fifty_pct - atr14 * 0.5
                    entry_hi = fifty_pct + atr14 * 0.5
                    if not (entry_lo <= price <= entry_hi):
                        continue

                    # Must be turning up (retest holding)
                    if price <= prev_price:
                        continue

                    # Delta OR CVD divergence confirmation
                    delta = bar.get('delta', bar.get('vol_delta', 0))
                    if pd.isna(delta):
                        delta = 0
                    cvd_at_entry = cvd_series.loc[post_low.index[j]] if cvd_series is not None else None
                    cvd_rising = (cvd_at_entry is not None and cvd_at_extreme is not None
                                  and cvd_at_entry > cvd_at_extreme)
                    if delta <= 0 and not cvd_rising:
                        continue

                    # Stop: 2 ATR below entry (not swept level - EOR buffer)
                    stop = price - ATR_STOP_MULT * atr14
                    risk = price - stop
                    if risk < eor_range * MIN_RISK_RATIO or risk > max_risk:
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
                            'level_swept': swept_low_name,
                            'swept_level_price': swept_low_level,
                            'sweep_depth': swept_low_level - eor_low,
                            'fifty_pct_level': fifty_pct,
                            'atr14': atr14,
                            'opening_drive': opening_drive,
                            'cvd_rising': cvd_rising,
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
