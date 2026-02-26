"""
Strategy: Opening Range Acceptance (LONG and SHORT)

Trades directional continuation when price BREAKS a key level at the open
and HOLDS it — NOT a fake sweep. This is the opposite of the Judas swing:
instead of entering on the reversal, we enter on the 50% retest of the
continuation move after acceptance.

Pattern (SHORT):
  1. Price opens/drives below London Low in the first 15-30 min
  2. EOR High stays BELOW London Low (no spike above = true acceptance)
  3. Price bounces 50% back toward London Low (the accepted level)
  4. At the 50% level, price turns back DOWN with neg delta → ENTER SHORT
  5. Stop: above London Low + 0.5 ATR buffer
  6. Target: 2R

Pattern (LONG): Mirror for London High acceptance.

Examples:
  2/02: IBH acceptance at open → LONG
  2/04: Closed below London Low first 15 min → SHORT at 50% of range
  2/03: Acceptance below London Low (5-min FVG retest → never hit)

Key distinction from Judas swing:
  - Judas:      EOR HIGH > London Low (fake spike), price REVERSES back through OR mid
  - Acceptance: EOR HIGH < London Low (full break), price holds below and we SHORT
"""

from datetime import time as _time
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np

from strategy.base import StrategyBase
from strategy.signal import Signal

# Matching constants with or_reversal.py
OR_BARS = 15
EOR_BARS = 30
MIN_RISK_RATIO = 0.03
MAX_RISK_RATIO = 1.3
ATR_PERIOD = 14
ATR_STOP_MULT = 0.5    # Stop buffer beyond acceptance level (in ATR units)


def _compute_atr14(bars: pd.DataFrame, n: int = ATR_PERIOD) -> float:
    if len(bars) < 3:
        return float((bars['high'] - bars['low']).mean()) if len(bars) > 0 else 20.0
    h = bars['high']
    l = bars['low']
    pc = bars['close'].shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n, min_periods=3).mean().iloc[-1]
    return float(atr) if not pd.isna(atr) else float((h - l).mean())


class ORAcceptanceStrategy(StrategyBase):
    """
    London level acceptance: break and hold above/below London H/L at the open.
    Enter on 50% retest of the acceptance move with delta confirmation.
    """

    @property
    def name(self) -> str:
        return "OR Acceptance"

    @property
    def applicable_day_types(self) -> List[str]:
        return []  # All day types — runs pre-IB like OR Reversal

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

        # OR = first 15 bars, EOR = first 30 bars
        or_bars = ib_bars.iloc[:OR_BARS]
        or_high = or_bars['high'].max()
        or_low = or_bars['low'].min()

        eor_bars = ib_bars.iloc[:EOR_BARS]
        eor_high = eor_bars['high'].max()
        eor_low = eor_bars['low'].min()
        eor_range = eor_high - eor_low

        if eor_range < 10:
            return

        max_risk = eor_range * MAX_RISK_RATIO

        # London levels and key reference prices
        london_high = session_context.get('london_high')
        london_low = session_context.get('london_low')
        asia_high = session_context.get('asia_high')
        asia_low = session_context.get('asia_low')
        pdh = session_context.get('pdh') or session_context.get('prior_session_high')
        pdl = session_context.get('pdl') or session_context.get('prior_session_low')
        prior_va_high = session_context.get('prior_va_high') or session_context.get('prior_vp_vah')
        prior_va_low = session_context.get('prior_va_low') or session_context.get('prior_vp_val')
        prior_va_poc = session_context.get('prior_va_poc') or session_context.get('prior_vp_poc')

        # ── Directional bias score ─────────────────────────────────────────────
        # Resolves direction when both London levels are in play.
        # +1 = bullish signal, -1 = bearish signal.
        # Rule: use the bias to confirm which acceptance side to trade.
        bias = 0

        if asia_low and eor_low < asia_low:       bias += 1  # Asia Low swept → Judas DOWN → reversal UP
        if london_low and eor_low < london_low:   bias += 1  # London Low swept → bullish
        if pdl and eor_low > pdl:                 bias += 1  # PDL held (buyers defended it)
        if prior_va_low and eor_low > prior_va_low: bias += 1  # Prior VAL not broken → above VA
        if prior_va_poc and eor_bars['close'].iloc[0] > prior_va_poc: bias += 1  # opened above POC

        if asia_high and eor_high > asia_high:    bias -= 1  # Asia High swept → Judas UP → reversal DOWN
        if london_high and eor_high > london_high: bias -= 1  # London High swept → bearish
        if pdh and eor_high < pdh:                bias -= 1  # PDH held (sellers defended it)
        if prior_va_high and eor_high < prior_va_high: bias -= 1  # Prior VAH not broken → below VA
        if prior_va_poc and eor_bars['close'].iloc[0] < prior_va_poc: bias -= 1  # opened below POC

        # CVD for order flow confirmation
        deltas = ib_bars['delta'] if 'delta' in ib_bars.columns else ib_bars.get('vol_delta', pd.Series(dtype=float))
        if deltas is not None and len(deltas) > 0:
            deltas = deltas.fillna(0)
            cvd_series = deltas.cumsum()
        else:
            cvd_series = None

        atr14 = _compute_atr14(ib_bars)
        cvd_at_start = cvd_series.iloc[0] if cvd_series is not None else None

        # ==========================================
        # ACCEPTANCE SHORT: price held below London Low
        # ==========================================
        # Condition: EOR HIGH is below London Low (no spike above = true acceptance)
        # Bias filter: when both London levels touched, only SHORT if bias <= 0 (bearish or neutral)
        # When bias > 0 (bullish), the market is more likely heading UP to LDN High — skip SHORT.
        if (london_low
                and eor_high < london_low          # entire EOR below London Low
                and or_low < london_low             # OR itself confirmed below
                and bias <= 0):                     # bias not pointing strongly bullish
            acceptance_level = london_low
            acceptance_name = 'LDN_LOW'

            ib_low_price = ib_bars['low'].min()
            # 50% of the range from IB low back up to the broken level
            fifty_pct = ib_low_price + (acceptance_level - ib_low_price) * 0.50

            in_below = False
            for j in range(1, min(60, len(ib_bars))):
                bar = ib_bars.iloc[j]
                price = bar['close']
                prev_price = ib_bars.iloc[j - 1]['close']

                # Must be below acceptance level (confirmed acceptance)
                if price < acceptance_level:
                    in_below = True
                if not in_below:
                    continue
                # Invalidate: if price reclaims the level, pattern is over
                if price >= acceptance_level:
                    break

                # Entry zone: 50% bounce back toward acceptance level
                entry_lo = fifty_pct - atr14 * 0.5
                entry_hi = fifty_pct + atr14 * 0.5
                if not (entry_lo <= price <= entry_hi):
                    continue

                # Retest must be FAILING: price turning back down
                if price >= prev_price:
                    continue

                # Order flow: neg delta or CVD declining
                delta = bar.get('delta', bar.get('vol_delta', 0))
                if pd.isna(delta):
                    delta = 0
                cvd_at_entry = cvd_series.loc[ib_bars.index[j]] if cvd_series is not None else None
                cvd_declining = (cvd_at_entry is not None and cvd_at_start is not None
                                 and cvd_at_entry < cvd_at_start)
                if delta >= 0 and not cvd_declining:
                    continue

                # Stop: above acceptance level + 0.5 ATR
                stop = acceptance_level + ATR_STOP_MULT * atr14
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
                    setup_type='OR_ACCEPTANCE_SHORT',
                    day_type='neutral',
                    trend_strength='moderate',
                    confidence='high',
                    metadata={
                        'acceptance_level': acceptance_level,
                        'acceptance_name': acceptance_name,
                        'fifty_pct_level': fifty_pct,
                        'ib_low': ib_low_price,
                        'atr14': atr14,
                        'bias_score': bias,
                        'cvd_declining': cvd_declining,
                    },
                )
                break

        # ==========================================
        # ACCEPTANCE LONG: price held above London High
        # ==========================================
        # Bias filter: when both levels touched, only LONG if bias >= 0 (bullish or neutral).
        # Key case: Asia Low swept + London Low swept → bias >= +2 → strongly bullish → LONG.
        if (self._cached_signal is None
                and london_high
                and eor_low > london_high          # entire EOR above London High
                and or_high > london_high           # OR itself confirmed above
                and bias >= 0):                     # bias not pointing strongly bearish
            acceptance_level = london_high
            acceptance_name = 'LDN_HIGH'

            ib_high_price = ib_bars['high'].max()
            # 50% of the range from IB high back down toward the broken level
            fifty_pct = ib_high_price - (ib_high_price - acceptance_level) * 0.50

            in_above = False
            for j in range(1, min(60, len(ib_bars))):
                bar = ib_bars.iloc[j]
                price = bar['close']
                prev_price = ib_bars.iloc[j - 1]['close']

                if price > acceptance_level:
                    in_above = True
                if not in_above:
                    continue
                # Invalidate: if price loses the level
                if price <= acceptance_level:
                    break

                entry_lo = fifty_pct - atr14 * 0.5
                entry_hi = fifty_pct + atr14 * 0.5
                if not (entry_lo <= price <= entry_hi):
                    continue

                # Pullback must be HOLDING: price turning back up
                if price <= prev_price:
                    continue

                delta = bar.get('delta', bar.get('vol_delta', 0))
                if pd.isna(delta):
                    delta = 0
                cvd_at_entry = cvd_series.loc[ib_bars.index[j]] if cvd_series is not None else None
                cvd_rising = (cvd_at_entry is not None and cvd_at_start is not None
                              and cvd_at_entry > cvd_at_start)
                if delta <= 0 and not cvd_rising:
                    continue

                stop = acceptance_level - ATR_STOP_MULT * atr14
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
                    setup_type='OR_ACCEPTANCE_LONG',
                    day_type='neutral',
                    trend_strength='moderate',
                    confidence='high',
                    metadata={
                        'acceptance_level': acceptance_level,
                        'acceptance_name': acceptance_name,
                        'fifty_pct_level': fifty_pct,
                        'ib_high': ib_high_price,
                        'atr14': atr14,
                        'bias_score': bias,
                        'cvd_rising': cvd_rising,
                    },
                )
                break

    def on_bar(self, bar: pd.Series, bar_index: int, session_context: dict) -> Optional[Signal]:
        if self._cached_signal is not None and not self._signal_emitted:
            self._signal_emitted = True
            signal = self._cached_signal
            self._cached_signal = None
            return signal
        return None
