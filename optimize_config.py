"""
STRATEGY OPTIMIZATION: Approaching Industry Standard ($2K+/month per account)

Problem:
  Current config: 0.47 trades/session, $1,370/month → way below industry ($2K+)

Root causes identified:
  1. Trade volume too low (quality gates, time windows, acceptance bars)
  2. Risk sizing conservative ($400/trade → 1-2 contracts)
  3. Too few strategies active (5 of 12 available)

Approach:
  Phase 1: Test strategy combinations at current risk ($400)
  Phase 2: Test relaxed strategy variants (subclasses with looser params)
  Phase 3: Test risk scaling ($400-$1500) on best combos
  Phase 4: Grid search: best combo × best risk × regime filter
  Phase 5: Final verdict with full trade log
"""

import sys
from pathlib import Path
from copy import deepcopy
from datetime import time
from itertools import combinations
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.instruments import get_instrument
from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from engine.execution import ExecutionModel
from engine.backtest import BacktestEngine
from engine.position import PositionManager
from strategy import (
    TrendDayBull, SuperTrendBull, PDayStrategy, BDayStrategy, MeanReversionVWAP,
    LiquiditySweep, EMATrendFollow, ORBVwapBreakout,
)
from filters.regime_filter import SimpleRegimeFilter
from strategy.signal import Signal

# ============================================================================
# DATA LOADING
# ============================================================================
instrument = get_instrument('MNQ')
df_raw = load_csv('NQ')
df = filter_rth(df_raw)
df = compute_all_features(df)

if 'session_date' not in df.columns:
    df['session_date'] = df['timestamp'].dt.date

sessions = sorted(df['session_date'].unique())
n_sessions = len(sessions)
months = n_sessions / 22

print("=" * 130)
print("  STRATEGY OPTIMIZATION: APPROACHING INDUSTRY STANDARD")
print(f"  Data: {n_sessions} sessions ({months:.1f} months)")
print("  Industry target: $2,000+/month per account, 1-3 trades/day, 60%+ WR")
print("=" * 130)


# ============================================================================
# HELPER: Run a config and return metrics
# ============================================================================
def run_config(strategies, regime_filter, risk_per_trade=400, max_contracts=5,
               label="", verbose=False):
    """Run a backtest config and return a metrics dict."""
    exec_m = ExecutionModel(instrument, slippage_ticks=1)
    pos_m = PositionManager(account_size=150000)

    engine = BacktestEngine(
        instrument=instrument,
        strategies=strategies,
        filters=regime_filter,
        execution=exec_m,
        position_mgr=pos_m,
        risk_per_trade=risk_per_trade,
        max_contracts=max_contracts,
    )

    result = engine.run(df, verbose=False)
    trades = result.trades

    n = len(trades)
    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl <= 0]
    total_pnl = sum(t.net_pnl for t in trades)
    wr = len(wins) / max(n, 1) * 100
    per_trade = total_pnl / max(n, 1)
    per_month = total_pnl / months
    trades_per_session = n / n_sessions

    avg_win = np.mean([t.net_pnl for t in wins]) if wins else 0
    avg_loss = np.mean([t.net_pnl for t in losses]) if losses else 0
    avg_ctrs = np.mean([t.contracts for t in trades]) if trades else 0

    # Daily P&L tracking
    daily_pnl = {}
    for t in trades:
        d = str(t.session_date)
        daily_pnl[d] = daily_pnl.get(d, 0) + t.net_pnl
    max_dd = min(daily_pnl.values()) if daily_pnl else 0
    active_days = len(daily_pnl)
    dd_buffer = 4500 / abs(max_dd) if max_dd < 0 else 999

    # Consistency check: max day profit should not exceed 40% of total
    # (Tradeify consistency rule)
    max_day_profit = max(daily_pnl.values()) if daily_pnl else 0
    consistency_ok = max_day_profit <= total_pnl * 0.40 if total_pnl > 0 else True

    # Profit factor
    gross_wins = sum(t.net_pnl for t in wins) if wins else 0
    gross_losses = abs(sum(t.net_pnl for t in losses)) if losses else 1
    pf = gross_wins / gross_losses if gross_losses > 0 else 999

    # Per-strategy breakdown
    by_strat = {}
    for t in trades:
        by_strat.setdefault(t.setup_type, []).append(t)

    return {
        'label': label,
        'trades': trades,
        'n': n,
        'wins': len(wins),
        'losses': len(losses),
        'wr': wr,
        'total_pnl': total_pnl,
        'per_month': per_month,
        'per_trade': per_trade,
        'trades_per_session': trades_per_session,
        'active_days': active_days,
        'active_pct': active_days / n_sessions * 100,
        'max_dd': max_dd,
        'dd_buffer': dd_buffer,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_ctrs': avg_ctrs,
        'profit_factor': pf,
        'consistency_ok': consistency_ok,
        'max_day_profit': max_day_profit,
        'by_strat': by_strat,
        'risk': risk_per_trade,
    }


def print_config_row(m):
    """Print a single config result row."""
    consist = "OK" if m['consistency_ok'] else "FAIL"
    print(f"  {m['label']:<52s} {m['n']:>4d} {m['trades_per_session']:>5.2f} "
          f"{m['wr']:>5.1f}% {m['profit_factor']:>5.2f} ${m['per_month']:>7,.0f} "
          f"${m['max_dd']:>6,.0f} {m['dd_buffer']:>5.1f}x {m['active_pct']:>4.0f}% "
          f"{m['avg_ctrs']:>4.1f}c {consist}")


def print_header():
    print(f"\n  {'Config':<52s} {'Trd':>4s} {'T/S':>5s} "
          f"{'WR':>6s} {'PF':>5s} {'$/Mo':>8s} "
          f"{'MaxDD':>7s} {'Buf':>5s} {'Act%':>5s} "
          f"{'Ctrs':>5s} {'Con':>4s}")
    print("  " + "-" * 120)


# ============================================================================
# RELAXED STRATEGY VARIANTS (subclasses with looser parameters)
# ============================================================================

class TrendDayBullRelaxed(TrendDayBull):
    """Relaxed Trend Bull: lower confidence, 1/3 quality gate, later cutoff."""

    def on_bar(self, bar, bar_index, session_context):
        # Override confidence check threshold
        orig_conf = session_context.get('trend_bull_confidence', 0.0)
        # Temporarily lower to 0.25 (was 0.375)
        session_context['_orig_trend_bull_confidence'] = orig_conf
        if orig_conf >= 0.25:
            session_context['trend_bull_confidence'] = max(orig_conf, 0.375)
        result = super().on_bar(bar, bar_index, session_context)
        # Restore
        session_context['trend_bull_confidence'] = orig_conf
        return result


class PDayRelaxed(PDayStrategy):
    """Relaxed P-Day: 1/3 quality gate, later cutoff."""

    def _check_long_entry(self, bar, bar_index, session_context):
        current_price = bar['close']
        delta = bar.get('delta', 0)
        strength = session_context.get('trend_strength', 'weak')

        if current_price <= self._ib_high:
            return None

        target_price = current_price + (1.5 * self._ib_range)

        vwap = bar.get('vwap')
        if vwap is not None and not pd.isna(vwap):
            vwap_dist = abs(current_price - vwap) / self._ib_range if self._ib_range > 0 else 999
            if vwap_dist < 0.40 and current_price > vwap and delta > 0:
                pre_delta_sum = sum(self._delta_history[:-1]) if len(self._delta_history) > 1 else 0
                if pre_delta_sum < -500:
                    return None

                # RELAXED: 1 of 3 quality gate (was 2 of 3)
                delta_pctl = bar.get('delta_percentile', 50)
                imbalance = bar.get('imbalance_ratio', 1.0)
                vol_spike = bar.get('volume_spike', 1.0)
                of_quality = sum([
                    (delta_pctl >= 60) if not pd.isna(delta_pctl) else True,
                    (imbalance > 1.0) if not pd.isna(imbalance) else True,
                    (vol_spike >= 1.0) if not pd.isna(vol_spike) else True,
                ])
                if of_quality < 1:  # Relaxed from 2
                    return None

                stop = vwap - (self._ib_range * 0.40)
                stop = min(stop, current_price - 15.0)
                if current_price - stop >= 15.0:
                    self._entry_taken = True
                    return Signal(
                        timestamp=bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp'),
                        direction='LONG',
                        entry_price=current_price,
                        stop_price=stop,
                        target_price=target_price,
                        strategy_name=self.name,
                        setup_type='P_DAY_VWAP_LONG',
                        day_type='p_day',
                        trend_strength=strength,
                        confidence='medium',
                    )
        return None


class BDayRelaxed(BDayStrategy):
    """Relaxed B-Day: lower confidence (0.4), 1/4 quality gate, allow 2 trades."""

    def on_bar(self, bar, bar_index, session_context):
        day_type = session_context.get('day_type', '')
        if day_type not in self.applicable_day_types:
            return None

        strength = session_context.get('trend_strength', 'weak')
        if strength != 'weak':
            return None

        # RELAXED: confidence 0.4 (was 0.5)
        b_day_conf = session_context.get('b_day_confidence', 0.0)
        if b_day_conf < 0.4:
            return None

        bar_time = session_context.get('bar_time')
        # RELAXED: time cutoff 15:00 (was 14:00)
        if bar_time and bar_time >= time(15, 0):
            return None

        if self._ib_range > 400:
            return None

        # RELAXED: cooldown 15 bars (was 30)
        if bar_index - self._last_entry_bar < 15:
            return None

        if self._val_fade_taken:
            return None

        current_price = bar['close']
        delta = bar.get('delta', 0)

        if bar['low'] <= self._ib_low:
            if bar_index - self._ibl_last_touch_bar <= 3:
                self._ibl_touch_count += 1
            else:
                self._ibl_touch_count = 1
            self._ibl_last_touch_bar = bar_index

        if not self._val_fade_taken and bar['low'] <= self._ib_low:
            if current_price > self._ib_low:
                has_fvg = bar.get('ifvg_bull_entry', False) or bar.get('fvg_bull', False)
                has_fvg_15m = bar.get('fvg_bull_15m', False)
                has_delta_rejection = delta > 0
                has_multi_touch = self._ibl_touch_count >= 2
                has_volume_spike = bar.get('volume_spike', 1.0) > 1.3

                quality_count = sum([
                    bool(has_fvg or has_fvg_15m),
                    has_delta_rejection,
                    has_multi_touch,
                    has_volume_spike,
                ])

                # RELAXED: quality >= 1 + delta (was quality >= 2 + delta)
                if quality_count >= 1 and has_delta_rejection:
                    entry_price = current_price
                    stop_price = self._ib_low - (self._ib_range * 0.10)
                    target_price = self._ib_mid

                    risk_val = abs(entry_price - stop_price)
                    reward = abs(target_price - entry_price)
                    if reward > 0 and risk_val / reward > 2.5:
                        return None

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
                        confidence='medium',
                    )

        return None


class MeanRevRelaxed(MeanReversionVWAP):
    """Relaxed MeanRev: lower deviation (0.50), relaxed RSI (40/60), 3 trades, later cutoff."""

    def on_bar(self, bar, bar_index, session_context):
        if self._ib_range < 15:
            return None

        bar_time = session_context.get('bar_time')
        if bar_time:
            # RELAXED: start 10:30, end 15:00 (was 11:00-14:30)
            if bar_time < time(10, 30) or bar_time >= time(15, 0):
                return None

        day_type = session_context.get('day_type', '')
        if day_type not in ['b_day', 'neutral', 'p_day']:
            return None

        # RELAXED: 3 entries (was 2)
        if self._entry_count >= 3:
            return None

        # RELAXED: cooldown 10 bars (was 15)
        if bar_index - self._last_entry_bar < 10:
            return None

        self._session_high = max(self._session_high, bar['high'])
        self._session_low = min(self._session_low, bar['low'])

        current_price = bar['close']
        vwap = bar.get('vwap')
        if vwap is None or pd.isna(vwap):
            return None

        rsi = bar.get('rsi14')
        if rsi is None or pd.isna(rsi):
            return None

        delta = bar.get('delta', 0)
        if pd.isna(delta):
            delta = 0

        volume_spike = bar.get('volume_spike', 1.0)
        if pd.isna(volume_spike):
            volume_spike = 1.0

        deviation = current_price - vwap
        deviation_mult = abs(deviation) / self._ib_range if self._ib_range > 0 else 0

        # RELAXED: deviation 0.50 (was 0.60)
        if deviation_mult < 0.50:
            return None

        # RELAXED: RSI 40/60 (was 35/65)
        if deviation < 0 and rsi < 40:
            return self._check_long_reversion(
                bar, bar_index, session_context,
                current_price, vwap, deviation_mult, delta, volume_spike,
            )
        if deviation > 0 and rsi > 60:
            return self._check_short_reversion(
                bar, bar_index, session_context,
                current_price, vwap, deviation_mult, delta, volume_spike,
            )

        return None


class ORBRelaxed(ORBVwapBreakout):
    """Relaxed ORB: lower volume (1.1), lower candle strength (0.50), later cutoff."""

    def on_bar(self, bar, bar_index, session_context):
        if self._ib_range < 20:
            return None

        bar_time = session_context.get('bar_time')
        # RELAXED: cutoff 14:00 (was 13:00)
        if bar_time and bar_time >= time(14, 0):
            return None

        if self._entry_count >= 2:
            return None

        if bar_index - self._last_entry_bar < 5:
            return None

        current_price = bar['close']
        bar_range = bar['high'] - bar['low']
        delta = bar.get('delta', 0)
        if pd.isna(delta):
            delta = 0
        volume_spike = bar.get('volume_spike', 1.0)
        if pd.isna(volume_spike):
            volume_spike = 1.0
        vwap = bar.get('vwap')

        # LONG BREAKOUT
        if not self._breakout_up and current_price > self._ib_high:
            # RELAXED: volume 1.1 (was 1.3), candle strength 0.50 (was 0.55)
            if volume_spike >= 1.1 and delta > 0:
                if bar_range > 0:
                    candle_strength = (current_price - bar['low']) / bar_range
                    if candle_strength < 0.50:
                        return None
                if vwap is not None and not pd.isna(vwap) and current_price < vwap:
                    return None

                stop_price = self._ib_mid
                stop_price = min(stop_price, current_price - 15.0)
                stop_price = min(stop_price, self._ib_high - 5.0)
                target_price = current_price + (1.5 * self._ib_range)

                self._breakout_up = True
                self._entry_count += 1
                self._last_entry_bar = bar_index

                return Signal(
                    timestamp=bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp'),
                    direction='LONG',
                    entry_price=current_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    strategy_name=self.name,
                    setup_type='ORB_BREAKOUT_LONG',
                    day_type=session_context.get('day_type', ''),
                    trend_strength=session_context.get('trend_strength', 'moderate'),
                    confidence='medium',
                )

        # SHORT BREAKOUT
        if not self._breakout_down and current_price < self._ib_low:
            if volume_spike >= 1.1 and delta < 0:
                if bar_range > 0:
                    candle_strength = (bar['high'] - current_price) / bar_range
                    if candle_strength < 0.50:
                        return None
                if vwap is not None and not pd.isna(vwap) and current_price > vwap:
                    return None

                stop_price = self._ib_high - (self._ib_range * 0.50)
                stop_price = max(stop_price, current_price + 15.0)
                target_price = current_price - (1.5 * self._ib_range)

                self._breakout_down = True
                self._entry_count += 1
                self._last_entry_bar = bar_index

                return Signal(
                    timestamp=bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp'),
                    direction='SHORT',
                    entry_price=current_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    strategy_name=self.name,
                    setup_type='ORB_BREAKOUT_SHORT',
                    day_type=session_context.get('day_type', ''),
                    trend_strength=session_context.get('trend_strength', 'moderate'),
                    confidence='medium',
                )

        return None


class LiqSweepRelaxed(LiquiditySweep):
    """Relaxed LiqSweep: lower wick (0.30), lower volume (1.0), 3 max trades, later cutoff."""

    def on_bar(self, bar, bar_index, session_context):
        if self._ib_range < 20:
            return None

        bar_time = session_context.get('bar_time')
        # RELAXED: cutoff 15:00 (was 14:00)
        if bar_time and bar_time >= time(15, 0):
            return None

        # RELAXED: 3 entries (was 2)
        if self._entry_count >= 3:
            return None

        if bar_index - self._last_entry_bar < 10:
            return None

        self._session_high = max(self._session_high, bar['high'])
        self._session_low = min(self._session_low, bar['low'])

        current_price = bar['close']
        delta = bar.get('delta', 0)
        if pd.isna(delta):
            delta = 0
        volume_spike = bar.get('volume_spike', 1.0)
        if pd.isna(volume_spike):
            volume_spike = 1.0

        # RELAXED parameters passed to check methods
        for level, name, swept_attr in [
            (self._ib_high, 'IBH', '_ibh_swept'),
            (self._ib_low, 'IBL', '_ibl_swept'),
        ]:
            if not getattr(self, swept_attr):
                if name in ('IBH', 'PDH'):
                    signal = self._check_upside_sweep_relaxed(
                        bar, bar_index, session_context, level, name, delta, volume_spike,
                    )
                else:
                    signal = self._check_downside_sweep_relaxed(
                        bar, bar_index, session_context, level, name, delta, volume_spike,
                    )
                if signal:
                    setattr(self, swept_attr, True)
                    return signal

        # PDH/PDL
        if self._pdh and not self._pdh_swept:
            signal = self._check_upside_sweep_relaxed(
                bar, bar_index, session_context, self._pdh, 'PDH', delta, volume_spike,
            )
            if signal:
                self._pdh_swept = True
                return signal

        if self._pdl and not self._pdl_swept:
            signal = self._check_downside_sweep_relaxed(
                bar, bar_index, session_context, self._pdl, 'PDL', delta, volume_spike,
            )
            if signal:
                self._pdl_swept = True
                return signal

        return None

    def _check_upside_sweep_relaxed(self, bar, bar_index, session_context,
                                     level, level_name, delta, volume_spike):
        current_price = bar['close']
        if bar['high'] <= level + 5.0:
            return None
        if current_price >= level:
            return None
        bar_range = bar['high'] - bar['low']
        if bar_range <= 0:
            return None
        upper_wick = bar['high'] - max(bar['open'], current_price)
        wick_ratio = upper_wick / bar_range
        # RELAXED: 0.30 (was 0.40)
        if wick_ratio < 0.30:
            return None
        # RELAXED: volume 1.0 (was 1.1)
        if volume_spike < 1.0:
            return None
        if delta >= 0:
            return None
        sweep_extreme = bar['high']
        stop_price = sweep_extreme + (self._ib_range * 0.10)
        risk = stop_price - current_price
        target_price = current_price - (risk * 1.5)
        if risk < 10.0:
            return None
        self._entry_count += 1
        self._last_entry_bar = bar_index
        return Signal(
            timestamp=bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp'),
            direction='SHORT',
            entry_price=current_price,
            stop_price=stop_price,
            target_price=target_price,
            strategy_name=self.name,
            setup_type=f'SWEEP_{level_name}_SHORT',
            day_type=session_context.get('day_type', ''),
            trend_strength=session_context.get('trend_strength', ''),
            confidence='medium',
        )

    def _check_downside_sweep_relaxed(self, bar, bar_index, session_context,
                                       level, level_name, delta, volume_spike):
        current_price = bar['close']
        if bar['low'] >= level - 5.0:
            return None
        if current_price <= level:
            return None
        bar_range = bar['high'] - bar['low']
        if bar_range <= 0:
            return None
        lower_wick = min(bar['open'], current_price) - bar['low']
        wick_ratio = lower_wick / bar_range
        if wick_ratio < 0.30:
            return None
        if volume_spike < 1.0:
            return None
        if delta <= 0:
            return None
        sweep_extreme = bar['low']
        stop_price = sweep_extreme - (self._ib_range * 0.10)
        risk = current_price - stop_price
        target_price = current_price + (risk * 1.5)
        if risk < 10.0:
            return None
        self._entry_count += 1
        self._last_entry_bar = bar_index
        return Signal(
            timestamp=bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp'),
            direction='LONG',
            entry_price=current_price,
            stop_price=stop_price,
            target_price=target_price,
            strategy_name=self.name,
            setup_type=f'SWEEP_{level_name}_LONG',
            day_type=session_context.get('day_type', ''),
            trend_strength=session_context.get('trend_strength', ''),
            confidence='medium',
        )


class EMATrendRelaxed(EMATrendFollow):
    """Relaxed EMA: wider proximity (0.30), lower volume (0.8), later cutoff, 3 trades."""

    def on_bar(self, bar, bar_index, session_context):
        if self._ib_range < 20:
            return None

        bar_time = session_context.get('bar_time')
        if bar_time:
            # RELAXED: start 10:30, end 15:00 (was 14:00)
            if bar_time < time(10, 30) or bar_time >= time(15, 0):
                return None

        # RELAXED: 3 entries (was 2)
        if self._entry_count >= 3:
            return None

        # RELAXED: cooldown 8 bars (was 10)
        if bar_index - self._last_entry_bar < 8:
            return None

        current_price = bar['close']
        ema20 = bar.get('ema20')
        ema50 = bar.get('ema50')
        vwap = bar.get('vwap')

        if ema20 is None or ema50 is None or pd.isna(ema20) or pd.isna(ema50):
            return None

        delta = bar.get('delta', 0)
        if pd.isna(delta):
            delta = 0
        volume_spike = bar.get('volume_spike', 1.0)
        if pd.isna(volume_spike):
            volume_spike = 1.0

        if ema20 > ema50:
            return self._check_long_relaxed(
                bar, bar_index, session_context,
                current_price, ema20, ema50, vwap, delta, volume_spike,
            )
        if ema20 < ema50:
            return self._check_short_relaxed(
                bar, bar_index, session_context,
                current_price, ema20, ema50, vwap, delta, volume_spike,
            )
        return None

    def _check_long_relaxed(self, bar, bar_index, session_context,
                             price, ema20, ema50, vwap, delta, volume_spike):
        if vwap is not None and not pd.isna(vwap):
            if price < vwap:
                return None

        ema20_dist = abs(price - ema20) / self._ib_range if self._ib_range > 0 else 999

        # RELAXED: proximity 0.30 (was 0.20)
        if bar['low'] <= ema20 + (self._ib_range * 0.30):
            self._pullback_detected = True
            if self._pullback_low is None or bar['low'] < self._pullback_low:
                self._pullback_low = bar['low']

        if self._pullback_detected and price > ema20:
            if delta <= 0:
                return None
            # RELAXED: volume 0.8 (was 0.9)
            if volume_spike < 0.8:
                return None

            stop_ema50 = ema50 - (self._ib_range * 0.15)
            stop_ib = price - (self._ib_range * 0.50)
            stop_price = max(stop_ema50, stop_ib)
            stop_price = min(stop_price, price - 12.0)
            target_price = price + (2.0 * self._ib_range)
            risk = price - stop_price
            if risk < 12.0:
                return None

            self._entry_count += 1
            self._last_entry_bar = bar_index
            self._pullback_detected = False
            self._pullback_low = None

            return Signal(
                timestamp=bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp'),
                direction='LONG',
                entry_price=price,
                stop_price=stop_price,
                target_price=target_price,
                strategy_name=self.name,
                setup_type='EMA_PULLBACK_LONG',
                day_type=session_context.get('day_type', ''),
                trend_strength=session_context.get('trend_strength', ''),
                confidence='medium',
            )
        return None

    def _check_short_relaxed(self, bar, bar_index, session_context,
                              price, ema20, ema50, vwap, delta, volume_spike):
        if vwap is not None and not pd.isna(vwap):
            if price > vwap:
                return None

        if bar['high'] >= ema20 - (self._ib_range * 0.30):
            self._pullback_detected = True
            if self._pullback_high is None or bar['high'] > self._pullback_high:
                self._pullback_high = bar['high']

        if self._pullback_detected and price < ema20:
            if delta >= 0:
                return None
            if volume_spike < 0.8:
                return None

            stop_ema50 = ema50 + (self._ib_range * 0.15)
            stop_ib = price + (self._ib_range * 0.50)
            stop_price = min(stop_ema50, stop_ib)
            stop_price = max(stop_price, price + 12.0)
            target_price = price - (2.0 * self._ib_range)
            risk = stop_price - price
            if risk < 12.0:
                return None

            self._entry_count += 1
            self._last_entry_bar = bar_index
            self._pullback_detected = False
            self._pullback_high = None

            return Signal(
                timestamp=bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp'),
                direction='SHORT',
                entry_price=price,
                stop_price=stop_price,
                target_price=target_price,
                strategy_name=self.name,
                setup_type='EMA_PULLBACK_SHORT',
                day_type=session_context.get('day_type', ''),
                trend_strength=session_context.get('trend_strength', ''),
                confidence='medium',
            )
        return None


# ============================================================================
# REGIME FILTERS
# ============================================================================
rf_long_only = SimpleRegimeFilter(
    longs_in_bull=True, longs_in_bear=True,
    shorts_in_bull=False, shorts_in_bear=False,
)

rf_selective_shorts = SimpleRegimeFilter(
    longs_in_bull=True, longs_in_bear=True,
    shorts_in_bull=False, shorts_in_bear=True,  # Shorts only in bear regime
)


# ============================================================================
# PHASE 1: STRATEGY COMBINATIONS AT $400 RISK (LONG ONLY)
# ============================================================================
print("\n\n" + "=" * 130)
print("  PHASE 1: STRATEGY COMBINATIONS (LONG ONLY, $400 risk)")
print("=" * 130)

# Define strategy pools
core = [TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(), MeanReversionVWAP()]
extras = [LiquiditySweep(), EMATrendFollow(), ORBVwapBreakout()]

phase1_results = []

# Test core first
m = run_config(core, rf_long_only, risk_per_trade=400, label="A: Core (5 strategies)")
phase1_results.append(m)

# Core + each extra individually
for extra_cls, name in [(LiquiditySweep, "LiqSweep"), (EMATrendFollow, "EMA"), (ORBVwapBreakout, "ORB")]:
    strats = [TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(), MeanReversionVWAP(), extra_cls()]
    m = run_config(strats, rf_long_only, risk_per_trade=400, label=f"  + {name}")
    phase1_results.append(m)

# Core + 2 extras
for combo in combinations([(LiquiditySweep, "LiqSweep"), (EMATrendFollow, "EMA"), (ORBVwapBreakout, "ORB")], 2):
    strats = [TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(), MeanReversionVWAP()]
    names = []
    for cls, n in combo:
        strats.append(cls())
        names.append(n)
    m = run_config(strats, rf_long_only, risk_per_trade=400, label=f"  + {' + '.join(names)}")
    phase1_results.append(m)

# All 8
strats_all = [TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(),
              MeanReversionVWAP(), LiquiditySweep(), EMATrendFollow(), ORBVwapBreakout()]
m = run_config(strats_all, rf_long_only, risk_per_trade=400, label="  + ALL 3 extras")
phase1_results.append(m)

print_header()
for m in phase1_results:
    print_config_row(m)


# ============================================================================
# PHASE 2: RELAXED VARIANTS AT $400 RISK (LONG ONLY)
# ============================================================================
print("\n\n" + "=" * 130)
print("  PHASE 2: RELAXED STRATEGY VARIANTS (LONG ONLY, $400 risk)")
print("=" * 130)

phase2_results = []

# Relaxed core
relaxed_core = [TrendDayBull(), SuperTrendBull(), PDayRelaxed(), BDayRelaxed(), MeanRevRelaxed()]
m = run_config(relaxed_core, rf_long_only, risk_per_trade=400, label="B: Relaxed Core (PDay+BDay+MR)")
phase2_results.append(m)

# Relaxed core + relaxed extras
relaxed_all = [TrendDayBull(), SuperTrendBull(), PDayRelaxed(), BDayRelaxed(),
               MeanRevRelaxed(), LiqSweepRelaxed(), EMATrendRelaxed(), ORBRelaxed()]
m = run_config(relaxed_all, rf_long_only, risk_per_trade=400, label="C: ALL Relaxed (LONG only)")
phase2_results.append(m)

# Relaxed with selective shorts
m = run_config(relaxed_all, rf_selective_shorts, risk_per_trade=400,
               label="D: ALL Relaxed + Selective Shorts")
phase2_results.append(m)

# No regime filter at all
m = run_config(relaxed_all, None, risk_per_trade=400,
               label="E: ALL Relaxed + NO filter")
phase2_results.append(m)

# Standard core + relaxed extras only
mixed = [TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(),
         MeanReversionVWAP(), LiqSweepRelaxed(), EMATrendRelaxed(), ORBRelaxed()]
m = run_config(mixed, rf_long_only, risk_per_trade=400,
               label="F: Std Core + Relaxed Extras (LONG only)")
phase2_results.append(m)

m = run_config(mixed, rf_selective_shorts, risk_per_trade=400,
               label="G: Std Core + Relaxed Extras + Sel. Shorts")
phase2_results.append(m)

print_header()
for m in phase2_results:
    print_config_row(m)


# ============================================================================
# PHASE 3: RISK SCALING ON TOP CONFIGS
# ============================================================================
print("\n\n" + "=" * 130)
print("  PHASE 3: RISK SCALING ON TOP CONFIGS ($400-$1500)")
print("=" * 130)

# Collect the best configs from phase 1 and 2
all_configs = phase1_results + phase2_results
# Filter to positive PnL configs and sort by per_month
viable = [c for c in all_configs if c['per_month'] > 0 and c['wr'] >= 45]
viable.sort(key=lambda x: x['per_month'], reverse=True)
top_configs = viable[:5]  # Top 5 by monthly income

print(f"\n  Testing risk scaling on top {len(top_configs)} configs:")
for c in top_configs:
    print(f"    {c['label']}: ${c['per_month']:,.0f}/mo, {c['wr']:.1f}% WR, {c['trades_per_session']:.2f} tr/s")

phase3_results = []

for risk in [400, 600, 800, 1000, 1500]:
    print(f"\n  --- Risk ${risk} ---")
    print_header()
    for base_config in top_configs:
        label = base_config['label']

        # Rebuild strategies based on label
        if 'ALL Relaxed + Selective' in label:
            strats = [TrendDayBull(), SuperTrendBull(), PDayRelaxed(), BDayRelaxed(),
                      MeanRevRelaxed(), LiqSweepRelaxed(), EMATrendRelaxed(), ORBRelaxed()]
            rf = rf_selective_shorts
        elif 'ALL Relaxed + NO' in label:
            strats = [TrendDayBull(), SuperTrendBull(), PDayRelaxed(), BDayRelaxed(),
                      MeanRevRelaxed(), LiqSweepRelaxed(), EMATrendRelaxed(), ORBRelaxed()]
            rf = None
        elif 'ALL Relaxed' in label:
            strats = [TrendDayBull(), SuperTrendBull(), PDayRelaxed(), BDayRelaxed(),
                      MeanRevRelaxed(), LiqSweepRelaxed(), EMATrendRelaxed(), ORBRelaxed()]
            rf = rf_long_only
        elif 'Std Core + Relaxed Extras + Sel' in label:
            strats = [TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(),
                      MeanReversionVWAP(), LiqSweepRelaxed(), EMATrendRelaxed(), ORBRelaxed()]
            rf = rf_selective_shorts
        elif 'Std Core + Relaxed Extras' in label:
            strats = [TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(),
                      MeanReversionVWAP(), LiqSweepRelaxed(), EMATrendRelaxed(), ORBRelaxed()]
            rf = rf_long_only
        elif 'Relaxed Core' in label:
            strats = [TrendDayBull(), SuperTrendBull(), PDayRelaxed(), BDayRelaxed(), MeanRevRelaxed()]
            rf = rf_long_only
        elif 'ALL 3 extras' in label:
            strats = [TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(),
                      MeanReversionVWAP(), LiquiditySweep(), EMATrendFollow(), ORBVwapBreakout()]
            rf = rf_long_only
        elif 'LiqSweep + EMA + ORB' in label or 'LiqSweep + EMA' in label:
            strats = [TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(),
                      MeanReversionVWAP(), LiquiditySweep(), EMATrendFollow()]
            rf = rf_long_only
        else:
            strats = [TrendDayBull(), SuperTrendBull(), PDayStrategy(), BDayStrategy(), MeanReversionVWAP()]
            rf = rf_long_only

        m = run_config(strats, rf, risk_per_trade=risk, max_contracts=5,
                       label=f"${risk} | {label[:40]}")
        phase3_results.append(m)
        print_config_row(m)


# ============================================================================
# PHASE 4: FIND THE OPTIMAL CONFIG
# ============================================================================
print("\n\n" + "=" * 130)
print("  PHASE 4: BEST CONFIGS RANKED BY MONTHLY INCOME (SAFE)")
print("=" * 130)

# Filter for safe configs: DD buffer >= 4x (safe for $4,500 trailing DD)
all_tested = phase1_results + phase2_results + phase3_results
safe = [c for c in all_tested if c['dd_buffer'] >= 4.0 and c['per_month'] > 0]
safe.sort(key=lambda x: x['per_month'], reverse=True)

print(f"\n  Total configs tested: {len(all_tested)}")
print(f"  Safe configs (DD buffer >= 4x): {len(safe)}")

print_header()
for i, m in enumerate(safe[:20]):
    print_config_row(m)


# ============================================================================
# PHASE 5: BEST CONFIG — DETAILED TRADE LOG
# ============================================================================
if safe:
    best = safe[0]
    print("\n\n" + "=" * 130)
    print(f"  PHASE 5: BEST CONFIG DETAILED ANALYSIS")
    print(f"  Config: {best['label']}")
    print(f"  $/month: ${best['per_month']:,.0f} | WR: {best['wr']:.1f}% | PF: {best['profit_factor']:.2f}")
    print(f"  Trades: {best['n']} ({best['trades_per_session']:.2f}/session)")
    print(f"  Max DD: ${best['max_dd']:,.0f} | DD Buffer: {best['dd_buffer']:.1f}x")
    print(f"  Risk/trade: ${best['risk']}")
    print("=" * 130)

    # Per-strategy breakdown
    print(f"\n  PER-STRATEGY BREAKDOWN:")
    print(f"  {'Setup Type':<35s} {'Trades':>6s} {'WR':>6s} {'Total PnL':>10s} {'Avg PnL':>9s}")
    print("  " + "-" * 75)

    for setup, trades in sorted(best['by_strat'].items(), key=lambda x: sum(t.net_pnl for t in x[1]), reverse=True):
        n = len(trades)
        w = sum(1 for t in trades if t.net_pnl > 0)
        wr = w / n * 100
        pnl = sum(t.net_pnl for t in trades)
        avg = pnl / n
        print(f"  {setup:<35s} {n:>6d} {wr:>5.1f}% ${pnl:>9,.0f} ${avg:>8,.0f}")

    # Daily P&L histogram
    daily_pnl = {}
    for t in best['trades']:
        d = str(t.session_date)
        daily_pnl[d] = daily_pnl.get(d, 0) + t.net_pnl

    all_daily = []
    for s in sessions:
        d = str(s)
        all_daily.append(daily_pnl.get(d, 0))

    pos_days = [p for p in all_daily if p > 0]
    neg_days = [p for p in all_daily if p < 0]
    zero_days = [p for p in all_daily if p == 0]

    print(f"\n  DAILY P&L:")
    print(f"    Win days:     {len(pos_days)} ({len(pos_days)/len(all_daily)*100:.0f}%) avg ${np.mean(pos_days):,.0f}" if pos_days else "    Win days: 0")
    print(f"    Loss days:    {len(neg_days)} ({len(neg_days)/len(all_daily)*100:.0f}%) avg ${np.mean(neg_days):,.0f}" if neg_days else "    Loss days: 0")
    print(f"    No-trade days: {len(zero_days)} ({len(zero_days)/len(all_daily)*100:.0f}%)")
    print(f"    Daily avg (all):    ${np.mean(all_daily):,.0f}")
    print(f"    Daily avg (active): ${np.mean([p for p in all_daily if p != 0]):,.0f}" if any(p != 0 for p in all_daily) else "")

    # Win/Loss economics
    wins_list = [t for t in best['trades'] if t.net_pnl > 0]
    losses_list = [t for t in best['trades'] if t.net_pnl <= 0]
    print(f"\n  TRADE ECONOMICS:")
    print(f"    Avg winner: ${np.mean([t.net_pnl for t in wins_list]):,.0f}" if wins_list else "")
    print(f"    Avg loser:  ${np.mean([t.net_pnl for t in losses_list]):,.0f}" if losses_list else "")
    print(f"    Avg contracts: {best['avg_ctrs']:.1f}")
    print(f"    Max single win:  ${max(t.net_pnl for t in best['trades']):,.0f}")
    print(f"    Max single loss: ${min(t.net_pnl for t in best['trades']):,.0f}")

    # Sample trades
    print(f"\n  SAMPLE TRADES (first 20):")
    print(f"  {'Date':<14s} {'Strategy':<28s} {'Dir':<6s} {'Entry':>10s} {'Exit':>10s} "
          f"{'Pts':>8s} {'Ctrs':>4s} {'Net':>9s} {'Reason':<8s}")
    print("  " + "-" * 105)

    sorted_trades = sorted(best['trades'], key=lambda x: str(x.session_date))
    for t in sorted_trades[:20]:
        pts = t.exit_price - t.entry_price if t.direction == 'LONG' else t.entry_price - t.exit_price
        print(f"  {str(t.session_date):<14s} {t.setup_type:<28s} {t.direction:<6s} "
              f"{t.entry_price:>10.2f} {t.exit_price:>10.2f} "
              f"{pts:>+7.1f} {t.contracts:>4d} ${t.net_pnl:>8,.0f} {t.exit_reason:<8s}")

    # Projection
    print(f"\n  MONTHLY PROJECTION (per account):")
    print(f"    Backtest monthly income: ${best['per_month']:,.0f}")
    print(f"    Conservative (80% of backtest): ${best['per_month'] * 0.80:,.0f}")
    print(f"    Pessimistic (60% of backtest): ${best['per_month'] * 0.60:,.0f}")
    print(f"\n  ACROSS 5 ACCOUNTS:")
    print(f"    Backtest: ${best['per_month'] * 5:,.0f}/month")
    print(f"    Conservative: ${best['per_month'] * 5 * 0.80:,.0f}/month")
    print(f"    Pessimistic: ${best['per_month'] * 5 * 0.60:,.0f}/month")


# ============================================================================
# SUMMARY
# ============================================================================
print("\n\n" + "=" * 130)
print("  OPTIMIZATION SUMMARY")
print("=" * 130)

if safe:
    best = safe[0]
    current = phase1_results[0]  # Config A

    print(f"""
  CURRENT CONFIG:
    Strategies: Core 5 (TrendBull, SuperTrendBull, PDay, BDay, MeanRev)
    Risk: $400/trade | Filter: LONG only
    Result: {current['n']} trades, {current['wr']:.1f}% WR, ${current['per_month']:,.0f}/month
    Trades/session: {current['trades_per_session']:.2f} | Active days: {current['active_pct']:.0f}%

  OPTIMIZED CONFIG:
    Config: {best['label']}
    Risk: ${best['risk']}/trade | Max contracts: 5 MNQ
    Result: {best['n']} trades, {best['wr']:.1f}% WR, ${best['per_month']:,.0f}/month
    Trades/session: {best['trades_per_session']:.2f} | Active days: {best['active_pct']:.0f}%
    Max DD: ${best['max_dd']:,.0f} | DD Buffer: {best['dd_buffer']:.1f}x
    Profit Factor: {best['profit_factor']:.2f}

  IMPROVEMENT:
    Monthly income: ${current['per_month']:,.0f} → ${best['per_month']:,.0f} ({(best['per_month']/max(current['per_month'],1)-1)*100:+.0f}%)
    Trade volume: {current['trades_per_session']:.2f} → {best['trades_per_session']:.2f} tr/session
    Active days: {current['active_pct']:.0f}% → {best['active_pct']:.0f}%

  ACROSS 5 ACCOUNTS: ${best['per_month'] * 5:,.0f}/month (conservative: ${best['per_month'] * 5 * 0.80:,.0f})
""")
else:
    print("\n  No safe configs found! All tested configs had excessive drawdown.")
