"""Test each OR Reversal NT8 improvement independently."""
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
from typing import Optional, List, Tuple

OR_BARS = 15
EOR_BARS = 30
SWEEP_THRESHOLD_RATIO = 0.17
VWAP_ALIGNED_RATIO = 0.17
OR_STOP_BUFFER = 0.15
MIN_RISK_RATIO = 0.03
MAX_RISK_RATIO = 1.3
DRIVE_THRESHOLD = 0.4


def _find_closest(eor_extreme, candidates, sweep_threshold, eor_range):
    best_level = None
    best_name = None
    best_dist = float('inf')
    for name, lvl in candidates:
        if lvl is None:
            continue
        dist = abs(eor_extreme - lvl)
        if dist < sweep_threshold and dist <= eor_range and dist < best_dist:
            best_dist = dist
            best_level = lvl
            best_name = name
    return best_level, best_name


class ORRevTest(StrategyBase):
    """Testable OR Reversal with toggle flags for each change."""

    def __init__(self, use_london=False, use_closest=False, use_dual_depth=False,
                 use_eor_mid=False, use_swept_stop=False, use_cvd=False):
        self.use_london = use_london
        self.use_closest = use_closest
        self.use_dual_depth = use_dual_depth
        self.use_eor_mid = use_eor_mid
        self.use_swept_stop = use_swept_stop
        self.use_cvd = use_cvd

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

        or_bars = ib_bars.iloc[:OR_BARS]
        or_high = or_bars['high'].max()
        or_low = or_bars['low'].min()
        or_mid = (or_high + or_low) / 2

        eor_bars = ib_bars.iloc[:EOR_BARS]
        eor_high = eor_bars['high'].max()
        eor_low = eor_bars['low'].min()
        eor_range = eor_high - eor_low
        eor_mid = (eor_high + eor_low) / 2

        # Which midpoint to use for entry zone
        entry_mid = eor_mid if self.use_eor_mid else or_mid

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

        # Build candidates
        high_candidates = [('ON_HIGH', overnight_high)]
        if pdh: high_candidates.append(('PDH', pdh))
        if asia_high: high_candidates.append(('ASIA_HIGH', asia_high))
        if self.use_london and london_high:
            high_candidates.append(('LDN_HIGH', london_high))

        low_candidates = [('ON_LOW', overnight_low)]
        if pdl: low_candidates.append(('PDL', pdl))
        if asia_low: low_candidates.append(('ASIA_LOW', asia_low))
        if self.use_london and london_low:
            low_candidates.append(('LDN_LOW', london_low))

        # Sweep detection
        if self.use_closest:
            swept_high_level, swept_high_name = _find_closest(
                eor_high, high_candidates, sweep_threshold, eor_range)
            swept_low_level, swept_low_name = _find_closest(
                eor_low, low_candidates, sweep_threshold, eor_range)
        else:
            # Original: first match in fixed order
            swept_high_level = swept_high_name = None
            for name, lvl in high_candidates:
                if lvl is not None and abs(eor_high - lvl) < sweep_threshold:
                    swept_high_level = lvl
                    swept_high_name = name
                    break
            swept_low_level = swept_low_name = None
            for name, lvl in low_candidates:
                if lvl is not None and abs(eor_low - lvl) < sweep_threshold:
                    swept_low_level = lvl
                    swept_low_name = name
                    break

        if swept_high_level is None and swept_low_level is None:
            return

        # Dual-sweep depth comparison
        if self.use_dual_depth and swept_high_level is not None and swept_low_level is not None:
            high_depth = eor_high - swept_high_level
            low_depth = swept_low_level - eor_low
            if high_depth >= low_depth:
                swept_low_level = swept_low_name = None
            else:
                swept_high_level = swept_high_name = None

        high_bar_idx = eor_bars['high'].idxmax()
        low_bar_idx = eor_bars['low'].idxmin()

        # CVD series
        cvd_series = None
        if self.use_cvd:
            deltas = ib_bars['delta'] if 'delta' in ib_bars.columns else None
            if deltas is not None:
                cvd_series = deltas.fillna(0).cumsum()

        # SHORT SETUP
        if swept_high_level is not None and opening_drive != 'DRIVE_UP':
            cvd_at_extreme = cvd_series.loc[high_bar_idx] if cvd_series is not None else None
            post_high = ib_bars.loc[high_bar_idx:]
            for j in range(1, min(30, len(post_high))):
                bar = post_high.iloc[j]
                price = bar['close']
                if price >= entry_mid:
                    continue
                vwap = bar.get('vwap', np.nan)
                if pd.isna(vwap) or abs(price - vwap) > vwap_threshold:
                    continue

                delta = bar.get('delta', bar.get('vol_delta', 0))
                if pd.isna(delta): delta = 0
                if self.use_cvd:
                    cvd_at_entry = cvd_series.loc[post_high.index[j]] if cvd_series is not None else None
                    cvd_declining = (cvd_at_entry is not None and cvd_at_extreme is not None
                                     and cvd_at_entry < cvd_at_extreme)
                    if delta >= 0 and not cvd_declining:
                        continue
                else:
                    if delta >= 0:
                        continue

                stop_base = swept_high_level if self.use_swept_stop else eor_high
                stop = stop_base + eor_range * OR_STOP_BUFFER
                risk = stop - price
                if risk < eor_range * MIN_RISK_RATIO or risk > max_risk:
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
                if price <= entry_mid:
                    continue
                vwap = bar.get('vwap', np.nan)
                if pd.isna(vwap) or abs(price - vwap) > vwap_threshold:
                    continue

                delta = bar.get('delta', bar.get('vol_delta', 0))
                if pd.isna(delta): delta = 0
                if self.use_cvd:
                    cvd_at_entry = cvd_series.loc[post_low.index[j]] if cvd_series is not None else None
                    cvd_rising = (cvd_at_entry is not None and cvd_at_extreme is not None
                                  and cvd_at_entry > cvd_at_extreme)
                    if delta <= 0 and not cvd_rising:
                        continue
                else:
                    if delta <= 0:
                        continue

                stop_base = swept_low_level if self.use_swept_stop else eor_low
                stop = stop_base - eor_range * OR_STOP_BUFFER
                risk = price - stop
                if risk < eor_range * MIN_RISK_RATIO or risk > max_risk:
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


def run_test(label, **kwargs):
    strat = ORRevTest(**kwargs)
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
    maxdd = 0
    eq = 0
    peak = 0
    for t in trades:
        eq += t.net_pnl
        if eq > peak: peak = eq
        dd = peak - eq
        if dd > maxdd: maxdd = dd
    pf = sum(t.net_pnl for t in trades if t.net_pnl > 0) / max(1, abs(sum(t.net_pnl for t in trades if t.net_pnl < 0)))
    print(f'{label:40s}: {total:3d}t ({shorts:2d}S/{longs:2d}L) WR {wr:5.1f}% Net ${net:>9,.0f} PF {pf:.2f} DD ${maxdd:>7,.0f}')


print("=== Individual Change Impact ===\n")
run_test("BASELINE (no changes)")
run_test("1. +London H/L", use_london=True)
run_test("2. +Closest sweep", use_closest=True)
run_test("3. +Dual-depth", use_dual_depth=True)
run_test("4. +EOR mid", use_eor_mid=True)
run_test("5. +Swept stop", use_swept_stop=True)
run_test("6. +CVD divergence", use_cvd=True)

print("\n=== Combined Tests ===\n")
run_test("ALL 6 changes", use_london=True, use_closest=True, use_dual_depth=True,
         use_eor_mid=True, use_swept_stop=True, use_cvd=True)
run_test("1+2+5+6 (no EOR mid, no dual)", use_london=True, use_closest=True,
         use_swept_stop=True, use_cvd=True)
run_test("1+2+6 (London+closest+CVD)", use_london=True, use_closest=True, use_cvd=True)
run_test("5+6 (swept stop + CVD)", use_swept_stop=True, use_cvd=True)
run_test("1+2+3+5+6 (no EOR mid)", use_london=True, use_closest=True, use_dual_depth=True,
         use_swept_stop=True, use_cvd=True)
run_test("1+5+6 (London+swept+CVD)", use_london=True, use_swept_stop=True, use_cvd=True)
