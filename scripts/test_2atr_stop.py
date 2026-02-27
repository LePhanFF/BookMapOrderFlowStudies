"""Test 2 ATR stop for OR Reversal vs current stop logic."""
import warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, '.')
import pandas as pd, numpy as np
from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from engine.backtest import BacktestEngine, IB_BARS_1MIN
from config.instruments import MNQ
from engine.position import PositionManager
from strategy.signal import Signal
from strategy.base import StrategyBase
from strategy.or_reversal import (
    OpeningRangeReversal, _find_closest_swept_level,
    OR_BARS, EOR_BARS, SWEEP_THRESHOLD_RATIO, VWAP_ALIGNED_RATIO,
    OR_STOP_BUFFER, MIN_RISK_RATIO, MAX_RISK_RATIO, DRIVE_THRESHOLD,
)
from typing import Optional, List, Tuple


class ORRevATRStop(StrategyBase):
    """OR Reversal with ATR-based stop instead of swept level + buffer."""

    def __init__(self, atr_multiplier=2.0):
        self.atr_mult = atr_multiplier

    @property
    def name(self): return "Opening Range Rev"

    @property
    def applicable_day_types(self): return []

    @property
    def trades_pre_ib(self): return True

    def on_session_start(self, session_date, ib_high, ib_low, ib_range, session_context):
        self._cached_signal = None
        self._signal_emitted = False
        self._ib_range = ib_range

        ib_bars = session_context.get('ib_bars')
        if ib_bars is None or len(ib_bars) < EOR_BARS:
            return

        # Get ATR from session context
        atr = session_context.get('atr14', 0)
        if atr <= 0:
            # Fallback: compute from IB bars
            tr = ib_bars['high'] - ib_bars['low']
            atr = tr.mean() if len(tr) > 0 else 0
        if atr <= 0:
            return

        or_bars = ib_bars.iloc[:OR_BARS]
        or_high = or_bars['high'].max()
        or_low = or_bars['low'].min()
        or_mid = (or_high + or_low) / 2

        eor_bars = ib_bars.iloc[:EOR_BARS]
        eor_high = eor_bars['high'].max()
        eor_low = eor_bars['low'].min()
        eor_range = eor_high - eor_low

        if eor_range < ib_range * 0.05 if ib_range > 0 else eor_range < 10:
            return

        sweep_threshold = eor_range * SWEEP_THRESHOLD_RATIO
        vwap_threshold = eor_range * VWAP_ALIGNED_RATIO
        max_risk = eor_range * MAX_RISK_RATIO

        first_5 = ib_bars.iloc[:5]
        open_price = first_5.iloc[0]['open']
        close_5th = first_5.iloc[4]['close']
        drive_range = first_5['high'].max() - first_5['low'].min()
        drive_pct = (close_5th - open_price) / drive_range if drive_range > 0 else 0
        if drive_pct > DRIVE_THRESHOLD: opening_drive = 'DRIVE_UP'
        elif drive_pct < -DRIVE_THRESHOLD: opening_drive = 'DRIVE_DOWN'
        else: opening_drive = 'ROTATION'

        overnight_high = session_context.get('overnight_high') or session_context.get('prior_session_high')
        overnight_low = session_context.get('overnight_low') or session_context.get('prior_session_low')
        if overnight_high is None or overnight_low is None:
            return

        pdh = session_context.get('pdh') or session_context.get('prior_session_high')
        pdl = session_context.get('pdl') or session_context.get('prior_session_low')
        asia_high = session_context.get('asia_high')
        asia_low = session_context.get('asia_low')
        london_high = session_context.get('london_high')
        london_low = session_context.get('london_low')

        high_candidates = [('ON_HIGH', overnight_high)]
        if pdh: high_candidates.append(('PDH', pdh))
        if asia_high: high_candidates.append(('ASIA_HIGH', asia_high))
        if london_high: high_candidates.append(('LDN_HIGH', london_high))

        low_candidates = [('ON_LOW', overnight_low)]
        if pdl: low_candidates.append(('PDL', pdl))
        if asia_low: low_candidates.append(('ASIA_LOW', asia_low))
        if london_low: low_candidates.append(('LDN_LOW', london_low))

        swept_high_level, swept_high_name = _find_closest_swept_level(
            eor_high, high_candidates, sweep_threshold, eor_range)
        swept_low_level, swept_low_name = _find_closest_swept_level(
            eor_low, low_candidates, sweep_threshold, eor_range)

        if swept_high_level is None and swept_low_level is None:
            return

        # Dual-sweep depth comparison
        if swept_high_level is not None and swept_low_level is not None:
            high_depth = eor_high - swept_high_level
            low_depth = swept_low_level - eor_low
            if high_depth >= low_depth:
                swept_low_level = swept_low_name = None
            else:
                swept_high_level = swept_high_name = None

        high_bar_idx = eor_bars['high'].idxmax()
        low_bar_idx = eor_bars['low'].idxmin()

        # CVD
        deltas = ib_bars['delta'] if 'delta' in ib_bars.columns else None
        cvd_series = deltas.fillna(0).cumsum() if deltas is not None else None

        # ATR-based stop distance
        atr_stop = atr * self.atr_mult

        # SHORT SETUP
        if swept_high_level is not None and opening_drive != 'DRIVE_UP':
            cvd_at_extreme = cvd_series.loc[high_bar_idx] if cvd_series is not None else None
            post_high = ib_bars.loc[high_bar_idx:]
            for j in range(1, min(30, len(post_high))):
                bar = post_high.iloc[j]
                price = bar['close']
                if price >= or_mid:
                    continue
                vwap = bar.get('vwap', np.nan)
                if pd.isna(vwap) or abs(price - vwap) > vwap_threshold:
                    continue
                delta = bar.get('delta', bar.get('vol_delta', 0))
                if pd.isna(delta): delta = 0
                cvd_at_entry = cvd_series.loc[post_high.index[j]] if cvd_series is not None else None
                cvd_declining = (cvd_at_entry is not None and cvd_at_extreme is not None
                                 and cvd_at_entry < cvd_at_extreme)
                if delta >= 0 and not cvd_declining:
                    continue

                # ATR-based stop
                stop = price + atr_stop
                risk = atr_stop
                if risk < eor_range * MIN_RISK_RATIO:
                    continue
                target = price - 2 * risk

                bar_ts = bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp')
                self._cached_signal = Signal(
                    timestamp=bar_ts, direction='SHORT', entry_price=price,
                    stop_price=stop, target_price=target, strategy_name=self.name,
                    setup_type='OR_REVERSAL_SHORT', day_type='neutral',
                    trend_strength='moderate', confidence='high', metadata={})
                return

        # LONG SETUP
        if swept_low_level is not None and opening_drive != 'DRIVE_DOWN':
            cvd_at_extreme = cvd_series.loc[low_bar_idx] if cvd_series is not None else None
            post_low = ib_bars.loc[low_bar_idx:]
            for j in range(1, min(30, len(post_low))):
                bar = post_low.iloc[j]
                price = bar['close']
                if price <= or_mid:
                    continue
                vwap = bar.get('vwap', np.nan)
                if pd.isna(vwap) or abs(price - vwap) > vwap_threshold:
                    continue
                delta = bar.get('delta', bar.get('vol_delta', 0))
                if pd.isna(delta): delta = 0
                cvd_at_entry = cvd_series.loc[post_low.index[j]] if cvd_series is not None else None
                cvd_rising = (cvd_at_entry is not None and cvd_at_extreme is not None
                              and cvd_at_entry > cvd_at_extreme)
                if delta <= 0 and not cvd_rising:
                    continue

                # ATR-based stop
                stop = price - atr_stop
                risk = atr_stop
                if risk < eor_range * MIN_RISK_RATIO:
                    continue
                target = price + 2 * risk

                bar_ts = bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp')
                self._cached_signal = Signal(
                    timestamp=bar_ts, direction='LONG', entry_price=price,
                    stop_price=stop, target_price=target, strategy_name=self.name,
                    setup_type='OR_REVERSAL_LONG', day_type='neutral',
                    trend_strength='moderate', confidence='high', metadata={})
                return

    def on_bar(self, bar, bar_index, session_context):
        if self._cached_signal is not None and not self._signal_emitted:
            self._signal_emitted = True
            signal = self._cached_signal
            self._cached_signal = None
            return signal
        return None


# Load data
df = load_csv('NQ')
full_df = df.copy()
rth = filter_rth(df)
rth = compute_all_features(rth)


def run_test(label, strat):
    pm = PositionManager(max_drawdown=999999)
    eng = BacktestEngine(instrument=MNQ, strategies=[strat], filters=None,
                         position_mgr=pm, full_df=full_df)
    result = eng.run(rth, verbose=False)
    trades = result.trades
    wins = sum(1 for t in trades if t.net_pnl > 0)
    total = len(trades)
    wr = wins/total*100 if total > 0 else 0
    net = sum(t.net_pnl for t in trades)
    shorts = sum(1 for t in trades if t.direction == 'SHORT')
    longs = total - shorts
    maxdd = 0; eq = 0; peak = 0
    for t in trades:
        eq += t.net_pnl
        if eq > peak: peak = eq
        dd = peak - eq
        if dd > maxdd: maxdd = dd
    gross_win = sum(t.net_pnl for t in trades if t.net_pnl > 0)
    gross_loss = abs(sum(t.net_pnl for t in trades if t.net_pnl < 0))
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
    avg_win = gross_win / wins if wins > 0 else 0
    avg_loss = gross_loss / (total - wins) if (total - wins) > 0 else 0
    print(f'{label:35s}: {total:3d}t ({shorts:2d}S/{longs:2d}L) WR {wr:5.1f}% Net ${net:>9,.0f} PF {pf:.2f} DD ${maxdd:>7,.0f} AvgW ${avg_win:>6,.0f} AvgL ${avg_loss:>6,.0f}')


print("=== Stop Comparison: Swept Level + Buffer vs ATR ===\n")
run_test("Current (swept lvl + 15% buffer)", OpeningRangeReversal())
for mult in [1.0, 1.5, 2.0, 2.5, 3.0]:
    run_test(f"ATR x{mult:.1f} stop", ORRevATRStop(atr_multiplier=mult))
