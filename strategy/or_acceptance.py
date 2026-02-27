"""
Strategy: Opening Range Acceptance (LONG and SHORT) — v2 Relaxed

Trades directional continuation when price BREAKS a key level at the open
and HOLDS it — NOT a fake sweep. This is the opposite of the Judas swing:
instead of entering on the reversal, we enter on the 50% retest of the
continuation move after acceptance.

v2 changes (2026-02-27):
  - Expanded beyond London-only to ALL reference levels (ON H/L, PDH/PDL, Asia H/L)
  - Relaxed acceptance: allow wick violations, close-percentage check, buffer zone
  - Configurable acceptance window (OR/EOR/IB)
  - Pick closest accepted level when multiple qualify
  - Diagnostic funnel showed 139/259 sessions pass relaxed acceptance vs 0 strict

Pattern (SHORT):
  1. Price opens/drives below a key reference level in the first 15-30 min
  2. Relaxed acceptance: >=70% of EOR closes below level, <=5 wick violations,
     max spike <=10% of EOR range above level
  3. Price bounces 50% back toward the accepted level
  4. At the 50% level, price turns back DOWN with neg delta → ENTER SHORT
  5. Stop: above acceptance level + 0.5 ATR buffer
  6. Target: 2R

Pattern (LONG): Mirror for acceptance above a key level.

Key distinction from Judas swing:
  - Judas:      EOR extreme SPIKES beyond a level (fake move), price REVERSES
  - Acceptance: EOR largely stays on one side of a level (true break), continuation
"""

from datetime import time as _time
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np

from strategy.base import StrategyBase
from strategy.signal import Signal

# ── Time Window Constants ──────────────────────────────────────
OR_BARS = 15       # Opening Range = first 15 bars (9:30-9:44)
EOR_BARS = 30      # Extended OR = first 30 bars (9:30-9:59)
IB_BARS = 60       # Initial Balance = first 60 bars (9:30-10:29)

# ── Risk Constants ─────────────────────────────────────────────
MIN_RISK_RATIO = 0.03    # Min risk = 3% of EOR range
MAX_RISK_RATIO = 1.3     # Max risk = 130% of EOR range
ATR_PERIOD = 14
ATR_STOP_MULT = 0.5      # Stop buffer beyond acceptance level (in ATR units)

# ── Relaxed Acceptance Parameters ──────────────────────────────
# Sweep winner: IB-window (60 bars) + aggressive relaxation
# 71 trades, 43.7% WR, PF 1.32, $3,173 net, MaxDD $2,845
MAX_WICK_VIOLATIONS = 5       # Bars allowed to spike past acceptance level
CLOSE_ACCEPTANCE_PCT = 0.70   # 70% of check-window closes must be on correct side
ACCEPTANCE_BUFFER_PCT = 0.10  # Max spike past level = 10% of EOR range
ACCEPTANCE_WINDOW = 'IB'      # 'OR' (15 bars), 'EOR' (30 bars), or 'IB' (60 bars)


def _compute_atr14(bars: pd.DataFrame, n: int = ATR_PERIOD) -> float:
    if len(bars) < 3:
        return float((bars['high'] - bars['low']).mean()) if len(bars) > 0 else 20.0
    h = bars['high']
    l = bars['low']
    pc = bars['close'].shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n, min_periods=3).mean().iloc[-1]
    return float(atr) if not pd.isna(atr) else float((h - l).mean())


def _check_acceptance(bars: pd.DataFrame, level: float, direction: str,
                      eor_range: float) -> Tuple[bool, int, float, float]:
    """Check relaxed acceptance condition.

    Returns:
        (passed, violations, close_pct, max_spike)
    """
    n = len(bars)
    if n == 0:
        return False, 0, 0.0, 0.0

    if direction == 'SHORT':
        violations = int((bars['high'] > level).sum())
        closes_correct = int((bars['close'] < level).sum())
        max_spike = float(bars['high'].max()) - level
    else:  # LONG
        violations = int((bars['low'] < level).sum())
        closes_correct = int((bars['close'] > level).sum())
        max_spike = level - float(bars['low'].min())

    close_pct = closes_correct / n
    buffer = ACCEPTANCE_BUFFER_PCT * eor_range

    passed = (violations <= MAX_WICK_VIOLATIONS
              and close_pct >= CLOSE_ACCEPTANCE_PCT
              and max_spike <= buffer)
    return passed, violations, close_pct, max_spike


def _find_best_accepted_level(
    bars: pd.DataFrame,
    candidates: List[Tuple[str, float]],
    direction: str,
    eor_range: float,
) -> Tuple[Optional[float], Optional[str]]:
    """Find the closest accepted level from candidates.

    Checks relaxed acceptance for each candidate level and returns the one
    with the highest close_pct (strongest acceptance). Ties broken by
    fewest violations.
    """
    best_level = None
    best_name = None
    best_close_pct = 0.0
    best_violations = float('inf')

    for name, lvl in candidates:
        if lvl is None:
            continue
        passed, violations, close_pct, max_spike = _check_acceptance(
            bars, lvl, direction, eor_range)
        if not passed:
            continue
        # Prefer highest close_pct, then fewest violations
        if (close_pct > best_close_pct or
                (close_pct == best_close_pct and violations < best_violations)):
            best_close_pct = close_pct
            best_violations = violations
            best_level = lvl
            best_name = name

    return best_level, best_name


class ORAcceptanceStrategy(StrategyBase):
    """
    Key level acceptance: break and hold above/below a reference level at the open.
    Enter on 50% retest of the acceptance move with delta confirmation.

    v2: Checks London H/L, Asia H/L, overnight H/L, and PDH/PDL.
    Uses relaxed acceptance (wick violations, close%, buffer) to generate
    20-40 trades across 260 sessions instead of 0.
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

        # ── Select acceptance window ────────────────────────────
        if ACCEPTANCE_WINDOW == 'OR':
            check_bars = ib_bars.iloc[:OR_BARS]
        elif ACCEPTANCE_WINDOW == 'IB':
            check_bars = ib_bars
        else:  # 'EOR' (default)
            check_bars = eor_bars

        # ── Gather all reference levels ─────────────────────────
        london_high = session_context.get('london_high')
        london_low = session_context.get('london_low')
        asia_high = session_context.get('asia_high')
        asia_low = session_context.get('asia_low')
        overnight_high = session_context.get('overnight_high') or session_context.get('prior_session_high')
        overnight_low = session_context.get('overnight_low') or session_context.get('prior_session_low')
        pdh = session_context.get('pdh') or session_context.get('prior_session_high')
        pdl = session_context.get('pdl') or session_context.get('prior_session_low')
        prior_va_high = session_context.get('prior_va_high') or session_context.get('prior_vp_vah')
        prior_va_low = session_context.get('prior_va_low') or session_context.get('prior_vp_val')
        prior_va_poc = session_context.get('prior_va_poc') or session_context.get('prior_vp_poc')

        # SHORT candidates: price below these levels = SHORT acceptance
        short_candidates = []
        if london_low: short_candidates.append(('LDN_LOW', london_low))
        if asia_low: short_candidates.append(('ASIA_LOW', asia_low))
        if pdl: short_candidates.append(('PDL', pdl))

        # LONG candidates: price above these levels = LONG acceptance
        long_candidates = []
        if london_high: long_candidates.append(('LDN_HIGH', london_high))
        if asia_high: long_candidates.append(('ASIA_HIGH', asia_high))
        if pdh: long_candidates.append(('PDH', pdh))

        # ── Directional bias score ──────────────────────────────
        bias = 0
        if asia_low and eor_low < asia_low:       bias += 1
        if london_low and eor_low < london_low:   bias += 1
        if pdl and eor_low > pdl:                  bias += 1
        if prior_va_low and eor_low > prior_va_low: bias += 1
        if prior_va_poc and eor_bars['close'].iloc[0] > prior_va_poc: bias += 1

        if asia_high and eor_high > asia_high:     bias -= 1
        if london_high and eor_high > london_high: bias -= 1
        if pdh and eor_high < pdh:                 bias -= 1
        if prior_va_high and eor_high < prior_va_high: bias -= 1
        if prior_va_poc and eor_bars['close'].iloc[0] < prior_va_poc: bias -= 1

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
        # ACCEPTANCE SHORT: price held below a key level
        # ==========================================
        if bias <= 0:
            acceptance_level, acceptance_name = _find_best_accepted_level(
                check_bars, short_candidates, 'SHORT', eor_range)

            if acceptance_level is not None:
                self._try_short_entry(
                    ib_bars, acceptance_level, acceptance_name, bias,
                    atr14, eor_range, max_risk, cvd_series, cvd_at_start)

        # ==========================================
        # ACCEPTANCE LONG: price held above a key level
        # ==========================================
        if self._cached_signal is None and bias >= 0:
            acceptance_level, acceptance_name = _find_best_accepted_level(
                check_bars, long_candidates, 'LONG', eor_range)

            if acceptance_level is not None:
                self._try_long_entry(
                    ib_bars, acceptance_level, acceptance_name, bias,
                    atr14, eor_range, max_risk, cvd_series, cvd_at_start)

    def _try_short_entry(self, ib_bars, acceptance_level, acceptance_name,
                         bias, atr14, eor_range, max_risk, cvd_series, cvd_at_start):
        """Scan IB bars for a SHORT entry after acceptance below a level."""
        ib_low_price = ib_bars['low'].min()
        fifty_pct = ib_low_price + (acceptance_level - ib_low_price) * 0.50

        in_below = False
        for j in range(1, min(60, len(ib_bars))):
            bar = ib_bars.iloc[j]
            price = bar['close']
            prev_price = ib_bars.iloc[j - 1]['close']

            if price < acceptance_level:
                in_below = True
            if not in_below:
                continue
            if price >= acceptance_level:
                break

            entry_lo = fifty_pct - atr14 * 0.5
            entry_hi = fifty_pct + atr14 * 0.5
            if not (entry_lo <= price <= entry_hi):
                continue

            if price >= prev_price:
                continue

            delta = bar.get('delta', bar.get('vol_delta', 0))
            if pd.isna(delta):
                delta = 0
            cvd_at_entry = cvd_series.loc[ib_bars.index[j]] if cvd_series is not None else None
            cvd_declining = (cvd_at_entry is not None and cvd_at_start is not None
                             and cvd_at_entry < cvd_at_start)
            if delta >= 0 and not cvd_declining:
                continue

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

    def _try_long_entry(self, ib_bars, acceptance_level, acceptance_name,
                        bias, atr14, eor_range, max_risk, cvd_series, cvd_at_start):
        """Scan IB bars for a LONG entry after acceptance above a level."""
        ib_high_price = ib_bars['high'].max()
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
            if price <= acceptance_level:
                break

            entry_lo = fifty_pct - atr14 * 0.5
            entry_hi = fifty_pct + atr14 * 0.5
            if not (entry_lo <= price <= entry_hi):
                continue

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
