"""
Strategy: Opening Range Acceptance (LONG and SHORT) — v3 Level Retest

Trades directional continuation when price BREAKS a key level at the open
and HOLDS it — NOT a fake sweep. This is the opposite of the Judas swing:
instead of entering on the reversal, we enter AT the acceptance level via
a limit order on the retest pullback.

v3 changes (2026-03-01):
  - Entry: limit order AT the acceptance level on retest (not 50% retrace)
  - Acceptance: 2x consecutive 5-min closes beyond the level
  - Stop: acceptance level ± 0.5 ATR buffer (~15 pts risk vs ~70 pts in v2)
  - Target: 2R (~30 pts)
  - Filter: Skip BOTH sessions (Judas sweep + acceptance on different levels)
  - Risk reduced ~5x: avg 15 pts vs avg 70 pts in v2

v2 was broken — root cause (2026-03-01 study):
  - 50% retrace entry put entries 50-100 pts from the acceptance level
  - Entries <=20 pts from level = 96% WR. Entries >60 pts = 15% WR.
  - v3 systematically targets the tight-entry profile.

Pattern (LONG):
  1. 9:30-10:30: Monitor 5-min bars against key levels
  2. Acceptance: 2 consecutive 5-min closes ABOVE the level
  3. Entry: Limit BUY at the acceptance level on retest pullback
  4. Stop: Acceptance level - 0.5 ATR buffer (~15 pts)
  5. Target: 2R (~30 pts)
  6. Filter: Skip if both sides of EOR swept different levels (BOTH session)

Pattern (SHORT): Mirror for acceptance below a key level.

Key distinction from Judas swing:
  - Judas:      EOR extreme SPIKES beyond a level (fake move), price REVERSES
  - Acceptance: Price breaks and HOLDS beyond a level, continuation
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
ATR_PERIOD = 14
ATR_STOP_MULT = 0.5      # Stop buffer beyond acceptance level (in ATR units)
MIN_RISK_PTS = 3          # Minimum risk in points (noise floor)
MAX_RISK_PTS = 40         # Maximum risk in points (cap for ATR spikes)

# ── v3 Acceptance Parameters ──────────────────────────────────
ACCEPT_5M_BARS = 2        # 2 consecutive 5-min closes for acceptance confirmation
RETEST_WINDOW = 30        # Max 1-min bars after acceptance to wait for limit fill

# ── BOTH Session Filter ───────────────────────────────────────
SWEEP_THRESHOLD_RATIO = 0.17  # Matching OR Reversal sweep detection threshold


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


def _resample_to_5m(bars_1m: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-min bars to 5-min OHLCV bars."""
    groups = []
    for i in range(0, len(bars_1m), 5):
        chunk = bars_1m.iloc[i:i + 5]
        if len(chunk) == 0:
            continue
        bar_5m = {
            'open': chunk.iloc[0]['open'],
            'high': chunk['high'].max(),
            'low': chunk['low'].min(),
            'close': chunk.iloc[-1]['close'],
        }
        for col in ('delta', 'vol_delta'):
            if col in chunk.columns:
                bar_5m[col] = chunk[col].fillna(0).sum()
        if 'timestamp' in chunk.columns:
            bar_5m['timestamp'] = chunk.iloc[-1]['timestamp']
        groups.append(bar_5m)
    return pd.DataFrame(groups)


class ORAcceptanceStrategy(StrategyBase):
    """
    v3: 2x 5-min acceptance + limit retest at acceptance level.

    Scans IB bars for level acceptance on 5-min timeframe, then checks
    if price retests the level (limit fill simulation) on 1-min bars.
    Skip BOTH sessions where EOR swept levels on both sides.
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

        # EOR for range measurement
        eor_bars = ib_bars.iloc[:EOR_BARS]
        eor_high = eor_bars['high'].max()
        eor_low = eor_bars['low'].min()
        eor_range = eor_high - eor_low

        if eor_range < 10:
            return

        atr14 = _compute_atr14(ib_bars)

        # ── Gather all reference levels ─────────────────────────
        london_high = session_context.get('london_high')
        london_low = session_context.get('london_low')
        asia_high = session_context.get('asia_high')
        asia_low = session_context.get('asia_low')
        pdh = session_context.get('pdh') or session_context.get('prior_session_high')
        pdl = session_context.get('pdl') or session_context.get('prior_session_low')
        overnight_high = session_context.get('overnight_high') or session_context.get('prior_session_high')
        overnight_low = session_context.get('overnight_low') or session_context.get('prior_session_low')

        # LONG candidates (acceptance above these levels)
        long_candidates = []
        if london_high:
            long_candidates.append(('LDN_HIGH', london_high))
        if asia_high:
            long_candidates.append(('ASIA_HIGH', asia_high))
        if pdh:
            long_candidates.append(('PDH', pdh))

        # SHORT candidates (acceptance below these levels)
        short_candidates = []
        if london_low:
            short_candidates.append(('LDN_LOW', london_low))
        if asia_low:
            short_candidates.append(('ASIA_LOW', asia_low))
        if pdl:
            short_candidates.append(('PDL', pdl))

        # ── BOTH filter: skip if EOR swept levels on both sides ──
        sweep_threshold = eor_range * SWEEP_THRESHOLD_RATIO
        high_candidates_all = []
        if overnight_high:
            high_candidates_all.append(overnight_high)
        if pdh:
            high_candidates_all.append(pdh)
        if asia_high:
            high_candidates_all.append(asia_high)
        if london_high:
            high_candidates_all.append(london_high)

        low_candidates_all = []
        if overnight_low:
            low_candidates_all.append(overnight_low)
        if pdl:
            low_candidates_all.append(pdl)
        if asia_low:
            low_candidates_all.append(asia_low)
        if london_low:
            low_candidates_all.append(london_low)

        high_swept = any(
            abs(eor_high - lvl) <= sweep_threshold
            for lvl in high_candidates_all
        )
        low_swept = any(
            abs(eor_low - lvl) <= sweep_threshold
            for lvl in low_candidates_all
        )

        if high_swept and low_swept:
            return  # BOTH session — skip

        # ── Resample IB bars to 5-min ─────────────────────────────
        bars_5m = _resample_to_5m(ib_bars)

        # ── Try LONG acceptance first ─────────────────────────────
        entry = self._find_v3_entry(
            ib_bars, bars_5m, long_candidates, 'LONG', atr14)
        if entry is not None:
            self._cached_signal = Signal(
                timestamp=entry['bar_ts'],
                direction='LONG',
                entry_price=entry['entry_price'],
                stop_price=entry['stop'],
                target_price=entry['target'],
                strategy_name=self.name,
                setup_type='OR_ACCEPTANCE_LONG',
                day_type='neutral',
                trend_strength='moderate',
                confidence='high',
                metadata={
                    'acceptance_level': entry['level'],
                    'acceptance_name': entry['level_name'],
                    'atr14': atr14,
                    'risk_pts': entry['risk'],
                    'version': 'v3',
                },
            )
            return

        # ── Try SHORT acceptance ──────────────────────────────────
        entry = self._find_v3_entry(
            ib_bars, bars_5m, short_candidates, 'SHORT', atr14)
        if entry is not None:
            self._cached_signal = Signal(
                timestamp=entry['bar_ts'],
                direction='SHORT',
                entry_price=entry['entry_price'],
                stop_price=entry['stop'],
                target_price=entry['target'],
                strategy_name=self.name,
                setup_type='OR_ACCEPTANCE_SHORT',
                day_type='neutral',
                trend_strength='moderate',
                confidence='high',
                metadata={
                    'acceptance_level': entry['level'],
                    'acceptance_name': entry['level_name'],
                    'atr14': atr14,
                    'risk_pts': entry['risk'],
                    'version': 'v3',
                },
            )

    def _find_v3_entry(self, ib_bars, bars_5m, candidates, direction, atr14):
        """Find a v3 entry: 2x 5-min acceptance + limit fill at level.

        Checks each candidate level for:
          1. 2 consecutive 5-min closes beyond the level (acceptance)
          2. Price retest at the level on 1-min bars (limit fill)

        Returns dict with entry details, or None.
        """
        best_entry = None
        best_fill_idx = float('inf')

        for name, level in candidates:
            if level is None:
                continue

            # Check 2x consecutive 5-min closes beyond the level
            accept_idx = self._check_5m_acceptance(bars_5m, level, direction)
            if accept_idx < 0:
                continue

            # Convert 5-min index to 1-min start index (first bar after acceptance)
            accept_1m_start = (accept_idx + 1) * 5
            if accept_1m_start >= len(ib_bars):
                continue

            # Check limit fill: does price retest the level?
            fill_idx = self._check_limit_fill(
                ib_bars, level, direction, accept_1m_start)
            if fill_idx < 0:
                continue

            # Compute risk
            risk = ATR_STOP_MULT * atr14
            if risk < MIN_RISK_PTS or risk > MAX_RISK_PTS:
                continue

            # Pick earliest fill across all candidate levels
            if fill_idx < best_fill_idx:
                best_fill_idx = fill_idx

                entry_price = level
                if direction == 'LONG':
                    stop = level - risk
                    target = level + 2 * risk
                else:
                    stop = level + risk
                    target = level - 2 * risk

                bar = ib_bars.iloc[fill_idx]
                bar_ts = (bar.get('timestamp', bar.name)
                          if hasattr(bar, 'name')
                          else bar.get('timestamp'))

                best_entry = {
                    'entry_price': entry_price,
                    'stop': stop,
                    'target': target,
                    'risk': risk,
                    'level': level,
                    'level_name': name,
                    'bar_ts': bar_ts,
                }

        return best_entry

    def _check_5m_acceptance(self, bars_5m, level, direction,
                              n_bars=ACCEPT_5M_BARS):
        """Check for n consecutive 5-min closes beyond level.

        Returns: index of the last accepted 5-min bar, or -1.
        """
        consecutive = 0
        for i in range(len(bars_5m)):
            close = bars_5m.iloc[i]['close']
            if direction == 'LONG' and close > level:
                consecutive += 1
            elif direction == 'SHORT' and close < level:
                consecutive += 1
            else:
                consecutive = 0

            if consecutive >= n_bars:
                return i
        return -1

    def _check_limit_fill(self, ib_bars, level, direction, start_idx,
                           max_bars=RETEST_WINDOW):
        """Check if price retests the acceptance level (limit fill simulation).

        LONG: limit buy at level -> need bar low <= level (pullback to level)
        SHORT: limit sell at level -> need bar high >= level (bounce to level)

        Returns: index of the fill bar, or -1.
        """
        end_idx = min(start_idx + max_bars, len(ib_bars))
        for i in range(start_idx, end_idx):
            bar = ib_bars.iloc[i]
            if direction == 'LONG' and bar['low'] <= level:
                return i
            elif direction == 'SHORT' and bar['high'] >= level:
                return i
        return -1

    def on_bar(self, bar: pd.Series, bar_index: int, session_context: dict) -> Optional[Signal]:
        if self._cached_signal is not None and not self._signal_emitted:
            self._signal_emitted = True
            signal = self._cached_signal
            self._cached_signal = None
            return signal
        return None
