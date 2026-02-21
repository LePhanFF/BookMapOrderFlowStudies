"""
Unified Backtest Engine.

Replaces: backtest_engine.py, dalton_playbook_engine.py,
          trend_following_engine.py, and the _simulate_trade()
          methods embedded in playbook_strategies.py.

Pipeline: data -> session grouping -> IB detection -> day type classification
       -> strategy signals -> filter chain -> execution -> position management
       -> equity tracking -> results
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import List, Optional, Dict
from dataclasses import dataclass, field

from config.constants import (
    IB_BARS_1MIN, RTH_START, RTH_END, PM_SESSION_START, EOD_CUTOFF,
    DEFAULT_MAX_RISK_PER_TRADE, DEFAULT_MAX_CONTRACTS,
    VWAP_BREACH_RATIO, VWAP_BREACH_POINTS,
)
from config.instruments import InstrumentSpec, get_instrument
from engine.trade import Trade
from engine.execution import ExecutionModel
from engine.position import PositionManager, OpenPosition
from engine.equity import EquityCurve
from strategy.base import StrategyBase
from strategy.signal import Signal
from strategy.day_type import (
    classify_trend_strength, classify_day_type, DayType, TrendStrength,
)
from strategy.day_confidence import DayTypeConfidenceScorer

# Profile adapters (TPO, Volume, DPOC) — optional, graceful fallback
try:
    from profile.tpo_profile import TPOProfileAdapter
    from profile.volume_profile import VolumeProfileAdapter
    from profile.dpoc_migration import DPOCMigrationAdapter
    _PROFILE_AVAILABLE = True
except ImportError:
    _PROFILE_AVAILABLE = False

# How often to update TPO/DPOC during bar loop (every 30 bars = ~30 min)
_PROFILE_UPDATE_INTERVAL = 30

# TPO-based day type classifier
try:
    from engine.tpo_classifier import classify_session_tpo
    _TPO_CLASSIFIER_AVAILABLE = True
except ImportError:
    _TPO_CLASSIFIER_AVAILABLE = False


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    trades: List[Trade] = field(default_factory=list)
    equity_curve: Optional[EquityCurve] = None
    sessions_processed: int = 0
    signals_generated: int = 0
    signals_filtered: int = 0
    signals_executed: int = 0


class BacktestEngine:
    """
    Unified bar-by-bar backtest engine.

    For each session:
      1. Compute IB from first 60 1-min bars
      2. Classify day type + trend strength
      3. Build session_context dict
      4. Notify strategies via on_session_start()
      5. For each bar after IB:
         a. Manage open positions (check stop/target hit)
         b. PM management (trail to BE after 1PM, VWAP breach exit)
         c. Check PositionManager limits
         d. If daily loss exceeded: force close all open positions
         e. Ask each active strategy: on_bar() -> Signal?
         f. Apply filter chain to signal
         g. If passes: execute via ExecutionModel
      6. Force close any open positions at EOD
      7. Notify strategies via on_session_end()
    """

    def __init__(
        self,
        instrument: InstrumentSpec,
        strategies: List[StrategyBase],
        filters=None,
        execution: Optional[ExecutionModel] = None,
        position_mgr: Optional[PositionManager] = None,
        risk_per_trade: float = DEFAULT_MAX_RISK_PER_TRADE,
        max_contracts: int = DEFAULT_MAX_CONTRACTS,
        full_df: Optional[pd.DataFrame] = None,
    ):
        self.instrument = instrument
        self.strategies = strategies
        self.filters = filters
        self.execution = execution or ExecutionModel(instrument)
        self.position_mgr = position_mgr or PositionManager()
        self.risk_per_trade = risk_per_trade
        self.max_contracts = max_contracts
        self.full_df = full_df  # Full (unfiltered) data for overnight level computation

        # Initialize profile adapters if available
        self._tpo_adapter = None
        self._volume_adapter = None
        self._dpoc_adapter = None
        if _PROFILE_AVAILABLE:
            self._tpo_adapter = TPOProfileAdapter()
            self._dpoc_adapter = DPOCMigrationAdapter()

    def run(self, df: pd.DataFrame, verbose: bool = True) -> BacktestResult:
        """Run the backtest on the provided DataFrame."""
        result = BacktestResult()
        result.equity_curve = EquityCurve(self.position_mgr.equity)

        # Ensure we have a timestamp-based index or column
        if 'timestamp' in df.columns:
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df['timestamp'] = df.index

        # Get session grouping
        if 'session_date' not in df.columns:
            df['session_date'] = df['timestamp'].dt.date

        sessions = sorted(df['session_date'].unique())

        if verbose:
            print(f"Backtest: {len(sessions)} sessions, {len(self.strategies)} strategies")
            print(f"  Instrument: {self.instrument.symbol}")
            print(f"  Risk/trade: ${self.risk_per_trade}")
            print(f"  Slippage: {self.execution.slippage_ticks} ticks/side")
            print(f"  Commission: ${self.execution.commission}/side")
            print()

        prior_session_context = {}  # Tracks prior session info for context
        ib_range_history = []  # Rolling IB ranges for regime detection
        strategy_trade_history = []  # Rolling trade outcomes for WR lookback

        # Initialize VolumeProfileAdapter with full RTH dataset
        if _PROFILE_AVAILABLE and self._volume_adapter is None:
            try:
                df_for_vp = df.copy()
                if 'timestamp' in df_for_vp.columns and not isinstance(df_for_vp.index, pd.DatetimeIndex):
                    df_for_vp = df_for_vp.set_index('timestamp')
                self._volume_adapter = VolumeProfileAdapter(df_for_vp)
            except Exception:
                self._volume_adapter = None

        for session_date in sessions:
            session_df = df[df['session_date'] == session_date].copy()
            session_str = str(session_date)

            if len(session_df) < IB_BARS_1MIN:
                continue

            self._process_session(
                session_df, session_str, result, verbose,
                prior_session_context=prior_session_context,
            )
            result.sessions_processed += 1

            # Store prior session data for next session
            last_bar = session_df.iloc[-1]
            ib_df_tmp = session_df.head(IB_BARS_1MIN)
            session_ib_range = ib_df_tmp['high'].max() - ib_df_tmp['low'].min()
            ib_range_history.append(session_ib_range)

            # Track trade outcomes for per-strategy rolling WR
            session_trades = [t for t in result.trades if t.session_date == session_str]
            for t in session_trades:
                strategy_trade_history.append({
                    'strategy': t.strategy_name,
                    'net_pnl': t.net_pnl,
                    'session': session_str,
                })

            # Compute rolling regime stats from IB history
            regime_stats = self._compute_regime_stats(
                ib_range_history, strategy_trade_history)

            prior_session_context = {
                'prior_close': last_bar['close'],
                'prior_vwap': last_bar.get('vwap', None),
                'prior_session_high': session_df['high'].max(),
                'prior_session_low': session_df['low'].min(),
                'ib_range_history': list(ib_range_history[-20:]),  # Last 20 sessions
            }
            prior_session_context.update(regime_stats)

            # Compute prior-day TPO/VP levels for next session's inside-day detection
            if self._volume_adapter is not None:
                try:
                    vp_levels = self._volume_adapter.get_session_levels(session_df, "15:59")
                    prior_session_context['prior_vp_poc'] = vp_levels.get('vp_poc')
                    prior_session_context['prior_vp_vah'] = vp_levels.get('vp_vah')
                    prior_session_context['prior_vp_val'] = vp_levels.get('vp_val')
                except Exception:
                    pass
            if self._tpo_adapter is not None:
                try:
                    tpo_signals = self._tpo_adapter.get_session_signals(session_df, "15:59")
                    prior_session_context['prior_tpo_poc'] = tpo_signals.get('tpo_poc')
                    prior_session_context['prior_tpo_vah'] = tpo_signals.get('tpo_vah')
                    prior_session_context['prior_tpo_val'] = tpo_signals.get('tpo_val')
                    prior_session_context['prior_tpo_shape'] = tpo_signals.get('tpo_shape')
                except Exception:
                    pass

            # Record equity snapshot at session end
            result.equity_curve.record(
                timestamp=session_df['timestamp'].iloc[-1],
                equity=self.position_mgr.equity,
                daily_pnl=self.position_mgr.daily_pnl.get(session_str, 0),
                trade_count=self.position_mgr.daily_trade_count.get(session_str, 0),
                session_date=session_str,
            )

        if verbose:
            self._print_summary(result)

        return result

    def _process_session(
        self, session_df: pd.DataFrame, session_str: str,
        result: BacktestResult, verbose: bool,
        prior_session_context: Optional[Dict] = None,
    ) -> None:
        """Process a single trading session."""
        # --- Phase 1: Compute IB ---
        ib_df = session_df.head(IB_BARS_1MIN)
        ib_high = ib_df['high'].max()
        ib_low = ib_df['low'].min()
        ib_range = ib_high - ib_low
        ib_mid = (ib_high + ib_low) / 2

        if ib_range <= 0:
            return

        # --- Phase 2: Classify day type ---
        # Use end-of-IB price for initial classification
        post_ib_df = session_df.iloc[IB_BARS_1MIN:]
        if len(post_ib_df) == 0:
            return

        last_ib_close = ib_df['close'].iloc[-1]
        ext_mult = 0.0
        if last_ib_close > ib_high:
            ext_mult = (last_ib_close - ib_mid) / ib_range
        elif last_ib_close < ib_low:
            ext_mult = (ib_mid - last_ib_close) / ib_range

        trend_strength = classify_trend_strength(ext_mult)

        # We'll reclassify dynamically as session progresses
        ib_direction = 'INSIDE'

        # --- Phase 3: Build session context ---
        session_context = {
            'ib_high': ib_high,
            'ib_low': ib_low,
            'ib_range': ib_range,
            'ib_mid': ib_mid,
            'day_type': DayType.NEUTRAL.value,
            'trend_strength': trend_strength.value,
            'session_date': session_str,
            'ib_bars': ib_df,  # IB bars for OR Reversal strategy
        }

        # Add latest indicator values from IB end
        last_ib = ib_df.iloc[-1]
        for col in ['vwap', 'ema20', 'ema50', 'atr14', 'rsi14', 'adx14']:
            if col in last_ib.index:
                session_context[col] = last_ib[col]

        # Add prior session context
        if prior_session_context:
            session_context.update(prior_session_context)
            # Compute prior session bias: close above/below VWAP
            pc = prior_session_context.get('prior_close')
            pv = prior_session_context.get('prior_vwap')
            if pc is not None and pv is not None and not pd.isna(pv):
                session_context['prior_session_bullish'] = pc > pv
            else:
                session_context['prior_session_bullish'] = None

        # Add overnight levels from full (unfiltered) data if available
        if self.full_df is not None:
            overnight_levels = self._compute_overnight_levels(session_str)
            session_context.update(overnight_levels)

        # --- Phase 3b: Inside-day detection (current IB within prior day's value area) ---
        prior_vah = session_context.get('prior_vp_vah') or session_context.get('prior_tpo_vah')
        prior_val = session_context.get('prior_vp_val') or session_context.get('prior_tpo_val')
        prior_vp_poc = session_context.get('prior_vp_poc') or session_context.get('prior_tpo_poc')
        if prior_vah is not None and prior_val is not None:
            # Inside day: IB fits within prior day's value area
            session_context['inside_day'] = (ib_high <= prior_vah and ib_low >= prior_val)
            # VA relationship: where did we open relative to prior VA?
            if ib_mid > prior_vah:
                session_context['va_relationship'] = 'above_va'
            elif ib_mid < prior_val:
                session_context['va_relationship'] = 'below_va'
            else:
                session_context['va_relationship'] = 'inside_va'
            session_context['prior_va_high'] = prior_vah
            session_context['prior_va_low'] = prior_val
            if prior_vp_poc is not None:
                session_context['prior_va_poc'] = prior_vp_poc
        else:
            session_context['inside_day'] = None
            session_context['va_relationship'] = None

        # --- Phase 3c: Compute TPO profile at IB end (initial shape + structure) ---
        if self._tpo_adapter is not None:
            try:
                prior_day_dict = None
                if prior_vp_poc is not None:
                    prior_day_dict = {
                        'poc': prior_vp_poc,
                        'vah': prior_vah, 'val': prior_val,
                        'high': session_context.get('prior_session_high'),
                        'low': session_context.get('prior_session_low'),
                    }
                tpo_signals = self._tpo_adapter.get_session_signals(
                    ib_df, "10:29", prior_day=prior_day_dict)
                session_context.update(tpo_signals)
            except Exception:
                pass

        # --- Phase 3c2: TPO-based day type classification at IB end ---
        if _TPO_CLASSIFIER_AVAILABLE:
            try:
                tpo_class = classify_session_tpo(ib_df, "10:29", ib_high, ib_low)
                session_context['tpo_shape'] = tpo_class.shape
                session_context['tpo_shape_30min'] = tpo_class.shape_30min
                session_context['tpo_day_type_hint'] = tpo_class.day_type_hint
                session_context['tpo_directional_bias'] = tpo_class.directional_bias
                session_context['tpo_skewness'] = tpo_class.skewness
                session_context['tpo_va_width_ratio'] = tpo_class.va_width_ratio
                session_context['tpo_poc_location'] = tpo_class.poc_location
                session_context['tpo_hvn_count'] = tpo_class.hvn_count
            except Exception:
                pass

        # --- Phase 3d: Initialize day type confidence scorer ---
        confidence_scorer = DayTypeConfidenceScorer()
        atr = session_context.get('atr14', 0.0)
        confidence_scorer.on_session_start(ib_high, ib_low, ib_range, atr,
                                           tpo_data=session_context)

        # --- Phase 4: Notify strategies ---
        active_strategies = []
        for strategy in self.strategies:
            strategy.on_session_start(
                session_date=session_str,
                ib_high=ib_high,
                ib_low=ib_low,
                ib_range=ib_range,
                session_context=session_context,
            )
            active_strategies.append(strategy)

        # Track session high/low for trailing stops
        session_high = ib_high
        session_low = ib_low

        # --- Phase 5: Bar-by-bar after IB ---
        for bar_idx in range(len(post_ib_df)):
            bar = post_ib_df.iloc[bar_idx]
            timestamp = bar['timestamp'] if 'timestamp' in bar.index else post_ib_df.index[bar_idx]
            bar_time = timestamp.time() if hasattr(timestamp, 'time') else None

            # Track session extremes
            if bar['high'] > session_high:
                session_high = bar['high']
            if bar['low'] < session_low:
                session_low = bar['low']

            # 5a: Update dynamic session context
            current_price = bar['close']
            if current_price > ib_high:
                ib_direction = 'BULL'
                ext = (current_price - ib_mid) / ib_range
            elif current_price < ib_low:
                ib_direction = 'BEAR'
                ext = (ib_mid - current_price) / ib_range
            else:
                ib_direction = 'INSIDE'
                ext = 0.0

            trend_strength = classify_trend_strength(ext)
            day_type = classify_day_type(ib_high, ib_low, current_price, ib_direction, trend_strength)

            session_context['day_type'] = day_type.value
            session_context['trend_strength'] = trend_strength.value
            session_context['ib_direction'] = ib_direction
            session_context['current_price'] = current_price
            session_context['bar_time'] = bar_time

            # Update VWAP if available
            if 'vwap' in bar.index:
                session_context['vwap'] = bar['vwap']

            # 5a2: Periodic TPO/DPOC update (every 30 bars ≈ 30 min)
            if bar_idx > 0 and bar_idx % _PROFILE_UPDATE_INTERVAL == 0:
                bars_so_far = session_df.iloc[:IB_BARS_1MIN + bar_idx + 1]
                bar_time_str = bar_time.strftime('%H:%M') if bar_time else '15:59'
                if self._tpo_adapter is not None:
                    try:
                        tpo_signals = self._tpo_adapter.get_session_signals(
                            bars_so_far, bar_time_str)
                        session_context.update(tpo_signals)
                    except Exception:
                        pass
                if self._dpoc_adapter is not None:
                    try:
                        dpoc_signals = self._dpoc_adapter.get_migration_signals(
                            bars_so_far, bar_time_str,
                            atr14=session_context.get('atr14'),
                            current_close=current_price)
                        session_context.update(dpoc_signals)
                    except Exception:
                        pass
                if _TPO_CLASSIFIER_AVAILABLE:
                    try:
                        tpo_class = classify_session_tpo(
                            bars_so_far, bar_time_str, ib_high, ib_low)
                        session_context['tpo_shape'] = tpo_class.shape
                        session_context['tpo_shape_30min'] = tpo_class.shape_30min
                        session_context['tpo_day_type_hint'] = tpo_class.day_type_hint
                        session_context['tpo_directional_bias'] = tpo_class.directional_bias
                        session_context['tpo_skewness'] = tpo_class.skewness
                        session_context['tpo_va_width_ratio'] = tpo_class.va_width_ratio
                        session_context['tpo_poc_location'] = tpo_class.poc_location
                        session_context['tpo_hvn_count'] = tpo_class.hvn_count
                    except Exception:
                        pass

            # 5a3: Update day type confidence scorer
            day_confidence = confidence_scorer.update(bar, bar_idx, tpo_data=session_context)
            session_context['day_confidence'] = day_confidence
            session_context['trend_bull_confidence'] = day_confidence.trend_bull
            session_context['trend_bear_confidence'] = day_confidence.trend_bear
            session_context['p_day_bull_confidence'] = day_confidence.p_day_bull
            session_context['p_day_bear_confidence'] = day_confidence.p_day_bear
            session_context['b_day_confidence'] = day_confidence.b_day

            # 5b: Manage open positions
            closed_trades = self._manage_positions(
                bar, timestamp, session_str, bar_time,
                session_high=session_high, session_low=session_low, ib_range=ib_range,
            )
            for trade in closed_trades:
                result.trades.append(trade)

            # 5c: Check if daily loss exceeded -> force close
            if self.position_mgr.daily_loss_exceeded(session_str):
                force_closed = self._force_close_all(bar, timestamp, session_str, 'DAILY_LOSS')
                result.trades.extend(force_closed)
                continue

            # 5d: Skip if can't open new trades
            if not self.position_mgr.can_open_trade(session_str):
                continue

            # 5e: Check contract limits
            open_contracts = self.position_mgr.get_open_contracts()
            if open_contracts >= self.max_contracts:
                continue

            # 5f: Get signals from active strategies
            # Note: ALL strategies receive on_bar() calls so they can track
            # state (acceptance, wick counts, etc.) across day-type transitions.
            # Each strategy checks day_type internally and only emits signals
            # when conditions match.
            for strategy in active_strategies:
                signal = strategy.on_bar(bar, bar_idx, session_context)
                if signal is None:
                    continue

                result.signals_generated += 1

                # 5g: Apply filters
                if self.filters is not None:
                    if not self.filters.should_trade(signal, bar, session_context):
                        result.signals_filtered += 1
                        continue

                # 5h: Execute
                trade = self._execute_signal(signal, bar, timestamp, session_str)
                if trade is not None:
                    result.signals_executed += 1

        # --- Phase 6: Force close EOD ---
        if self.position_mgr.has_open_positions():
            last_bar = post_ib_df.iloc[-1]
            last_ts = last_bar['timestamp'] if 'timestamp' in last_bar.index else post_ib_df.index[-1]
            eod_trades = self._force_close_all(last_bar, last_ts, session_str, 'EOD')
            result.trades.extend(eod_trades)

        # --- Phase 7: Notify strategies ---
        for strategy in active_strategies:
            strategy.on_session_end(session_str)

    def _manage_positions(
        self, bar: pd.Series, timestamp, session_str: str, bar_time,
        session_high: float = 0, session_low: float = 0, ib_range: float = 0,
    ) -> List[Trade]:
        """Check open positions for stop/target hit and PM management."""
        closed_trades = []
        positions_to_close = []

        trend_strategies = ('Trend Day Bull', 'Trend Day Bear',
                           'Super Trend Bull', 'Super Trend Bear',
                           'Morph to Trend')
        # Edge Fade trades are mean reversion from IB edge to midpoint.
        # They naturally oscillate near VWAP, so VWAP breach PM exit would
        # prematurely kill valid trades. Their stop/target handles risk.
        vwap_breach_exempt = trend_strategies + ('Edge Fade',)

        for pos in self.position_mgr.open_positions:
            pos.bars_held += 1

            # Update peak MFE for R-based management
            pos.update_peak_favorable(bar['high'], bar['low'])

            # Trend strategy trailing: trail by 1.0x IB range from session extreme
            # This locks in profit as the trend extends while giving enough room
            if pos.strategy_name in trend_strategies and ib_range > 0:
                if pos.bars_held >= 20 and pos.unrealized_pnl_points(bar['close']) > ib_range * 0.5:
                    trail_dist = ib_range * 1.0  # trail 1 IB range behind extreme
                    pos.trail_by_session_extreme(session_high, session_low, trail_dist)

            # NOTE: Per-strategy trail/BE management tested (2026-02-21) but REVERTED.
            # Bar-by-bar study showed Trail@0.75R helps OR Rev and BE@0.75R helps Edge Fade
            # in 1-contract simulation, but with risk-based contract sizing in the engine,
            # both changes HURT total P&L ($13,986 vs $19,462 baseline).
            # Root cause: trail cuts the right tail of big winners, and risk-based sizing
            # means fewer contracts on high-risk trades (where MFE is highest).
            # Keep infrastructure (update_peak_favorable, trail_by_r_multiple, breakeven_at_r)
            # for future use with partial-exit or adaptive management.

            # PM Management: trail to breakeven after 1 PM
            # Only trail to BE when profit exceeds a meaningful threshold.
            # Trailing at any tiny profit caused 23 premature BE stops (-$2.74).
            # Require at least 0.3x IB range profit before activating BE trail.
            if bar_time and bar_time >= PM_SESSION_START:
                min_be_profit = ib_range * 0.3 if ib_range > 0 else 20.0
                if pos.unrealized_pnl_points(bar['close']) > min_be_profit:
                    pos.trail_to_breakeven()

                # VWAP breach in PM = trend failure
                # Only apply to non-trend strategies (PM Morph, P-Day, B-Day).
                # Trend day entries with acceptance hold through PM; their stop handles exits.
                if pos.strategy_name not in vwap_breach_exempt:
                    if 'vwap' in bar.index:
                        vwap = bar['vwap']
                        # IB-scaled VWAP breach threshold
                        vwap_breach = ib_range * VWAP_BREACH_RATIO if ib_range > 0 else VWAP_BREACH_POINTS
                        if pos.direction == 'LONG' and bar['close'] < vwap - vwap_breach:
                            positions_to_close.append((pos, 'VWAP_BREACH_PM', bar['close']))
                            continue
                        elif pos.direction == 'SHORT' and bar['close'] > vwap + vwap_breach:
                            positions_to_close.append((pos, 'VWAP_BREACH_PM', bar['close']))
                            continue

            # EOD cutoff
            if bar_time and bar_time >= EOD_CUTOFF:
                positions_to_close.append((pos, 'EOD', bar['close']))
                continue

            # Check stop
            if pos.check_stop_hit(bar['low'], bar['high']):
                exit_price = pos.trailing_stop
                positions_to_close.append((pos, 'STOP', exit_price))
                continue

            # Check target
            if pos.check_target_hit(bar['low'], bar['high']):
                exit_price = pos.target_price
                positions_to_close.append((pos, 'TARGET', exit_price))
                continue

        # Close positions
        for pos, reason, exit_price in positions_to_close:
            trade = self._close_position(pos, exit_price, timestamp, session_str, reason)
            closed_trades.append(trade)
            self.position_mgr.remove_position(pos)

        return closed_trades

    def _force_close_all(
        self, bar: pd.Series, timestamp, session_str: str, reason: str,
    ) -> List[Trade]:
        """Force close all open positions."""
        trades = []
        positions = self.position_mgr.close_all_positions()
        for pos in positions:
            trade = self._close_position(pos, bar['close'], timestamp, session_str, reason)
            trades.append(trade)
        return trades

    def _close_position(
        self, pos: OpenPosition, raw_exit_price: float,
        timestamp, session_str: str, reason: str,
    ) -> Trade:
        """Close a position and create a Trade record."""
        exit_fill = self.execution.fill_exit(pos.direction, raw_exit_price)

        gross, comm, slip, net = self.execution.calculate_net_pnl(
            pos.direction, pos.entry_price, exit_fill, pos.contracts,
        )

        self.position_mgr.record_trade_pnl(session_str, net)

        return Trade(
            strategy_name=pos.strategy_name,
            setup_type=pos.setup_type,
            day_type=pos.day_type,
            trend_strength=pos.trend_strength,
            session_date=session_str,
            entry_time=pos.entry_time,
            exit_time=timestamp,
            bars_held=pos.bars_held,
            direction=pos.direction,
            contracts=pos.contracts,
            signal_price=pos.entry_price,
            entry_price=pos.entry_price,
            exit_price=exit_fill,
            stop_price=pos.trailing_stop,
            target_price=pos.target_price,
            gross_pnl=gross,
            commission=comm,
            slippage_cost=slip,
            net_pnl=net,
            exit_reason=reason,
        )

    def _execute_signal(
        self, signal: Signal, bar: pd.Series, timestamp, session_str: str,
    ) -> Optional[Trade]:
        """Execute a signal: create a position."""
        # Calculate contracts
        stop_distance = signal.risk_points
        if stop_distance <= 0:
            return None

        contracts = self.execution.calculate_contracts(self.risk_per_trade, stop_distance)
        contracts = min(contracts, self.max_contracts - self.position_mgr.get_open_contracts())
        if contracts <= 0:
            return None

        # Apply entry slippage
        entry_fill = self.execution.fill_entry(signal.direction, signal.entry_price)

        # Create open position
        pos = OpenPosition(
            direction=signal.direction,
            entry_price=entry_fill,
            stop_price=signal.stop_price,
            target_price=signal.target_price,
            contracts=contracts,
            entry_time=timestamp,
            strategy_name=signal.strategy_name,
            setup_type=signal.setup_type,
            day_type=signal.day_type,
            trend_strength=signal.trend_strength,
            session_date=session_str,
            pyramid_level=signal.pyramid_level,
        )

        self.position_mgr.add_position(pos)
        return None  # Trade is not complete yet -- will be closed later

    def _compute_regime_stats(
        self, ib_range_history: list, strategy_trade_history: list,
    ) -> dict:
        """
        Compute rolling regime statistics from recent session history.

        Walk-Forward Lite: use the last 20 sessions to characterize the
        current volatility regime and per-strategy recent performance.
        Strategies can use these stats to adapt thresholds dynamically.
        """
        stats = {}
        lookback = 20

        if len(ib_range_history) >= 5:
            recent_ib = ib_range_history[-lookback:]
            stats['regime_ib_median'] = float(np.median(recent_ib))
            stats['regime_ib_mean'] = float(np.mean(recent_ib))
            stats['regime_ib_std'] = float(np.std(recent_ib))
            stats['regime_ib_p25'] = float(np.percentile(recent_ib, 25))
            stats['regime_ib_p75'] = float(np.percentile(recent_ib, 75))

            # Classify regime: low_vol if median IB < 150, high_vol if > 250.
            # These are NQ/MNQ-specific thresholds -- directional strategies
            # need ~80+ pts of IB for viable targets (IB mid). 150 is the
            # minimum viable IB median. Validated: 140-150 produce identical
            # results (plateau), confirming robustness.
            # NOTE: For other instruments, scale by tick_value ratio or recalibrate.
            median_ib = stats['regime_ib_median']
            if median_ib < 150:
                stats['regime_volatility'] = 'low'
            elif median_ib < 250:
                stats['regime_volatility'] = 'normal'
            else:
                stats['regime_volatility'] = 'high'
        else:
            stats['regime_ib_median'] = None
            stats['regime_ib_mean'] = None
            stats['regime_ib_std'] = None
            stats['regime_ib_p25'] = None
            stats['regime_ib_p75'] = None
            stats['regime_volatility'] = None

        # Per-strategy rolling WR (last 20 trades per strategy)
        if strategy_trade_history:
            strat_wr = {}
            # Group by strategy, take last 20 trades each
            by_strat = {}
            for t in strategy_trade_history:
                by_strat.setdefault(t['strategy'], []).append(t)
            for sname, trades in by_strat.items():
                recent = trades[-20:]
                wins = sum(1 for t in recent if t['net_pnl'] > 0)
                strat_wr[sname] = wins / len(recent) if recent else 0.0
            stats['regime_strategy_wr'] = strat_wr
        else:
            stats['regime_strategy_wr'] = {}

        return stats

    def _compute_overnight_levels(self, session_str: str) -> dict:
        """Compute overnight/ETH levels for a session from full (unfiltered) data."""
        from datetime import timedelta
        sd = pd.Timestamp(session_str)
        prev_day = sd - timedelta(days=1)
        if prev_day.weekday() == 5:
            prev_day -= timedelta(days=1)
        elif prev_day.weekday() == 6:
            prev_day -= timedelta(days=2)

        ts = self.full_df['timestamp']
        levels = {}

        # Overnight: 18:00 prev day to 9:29 session day
        on_mask = (ts >= prev_day + timedelta(hours=18)) & (ts < sd + timedelta(hours=9, minutes=30))
        on_bars = self.full_df[on_mask]
        if len(on_bars) > 0:
            levels['overnight_high'] = on_bars['high'].max()
            levels['overnight_low'] = on_bars['low'].min()
        else:
            levels['overnight_high'] = None
            levels['overnight_low'] = None

        # Asia session: 20:00-00:00 prev day evening
        asia_mask = (ts >= prev_day + timedelta(hours=20)) & (ts < sd)
        asia = self.full_df[asia_mask]
        if len(asia) > 0:
            levels['asia_high'] = asia['high'].max()
            levels['asia_low'] = asia['low'].min()

        # London session: 02:00-05:00 session day
        london_mask = (ts >= sd + timedelta(hours=2)) & (ts < sd + timedelta(hours=5))
        london = self.full_df[london_mask]
        if len(london) > 0:
            levels['london_high'] = london['high'].max()
            levels['london_low'] = london['low'].min()

        # Prior day RTH
        prior_rth_mask = (ts >= prev_day + timedelta(hours=9, minutes=30)) & (ts <= prev_day + timedelta(hours=16))
        prior_rth = self.full_df[prior_rth_mask]
        if len(prior_rth) > 0:
            levels['pdh'] = prior_rth['high'].max()
            levels['pdl'] = prior_rth['low'].min()
            levels['pdc'] = prior_rth.iloc[-1]['close']

        return levels

    def _print_summary(self, result: BacktestResult) -> None:
        """Print backtest summary."""
        trades = result.trades
        print(f"\n{'='*70}")
        print(f"BACKTEST RESULTS")
        print(f"{'='*70}")
        print(f"Sessions processed: {result.sessions_processed}")
        print(f"Signals generated:  {result.signals_generated}")
        print(f"Signals filtered:   {result.signals_filtered}")
        print(f"Trades executed:    {len(trades)}")

        if not trades:
            print("  No trades to report.")
            return

        wins = [t for t in trades if t.net_pnl > 0]
        losses = [t for t in trades if t.net_pnl <= 0]

        total_pnl = sum(t.net_pnl for t in trades)
        total_gross = sum(t.gross_pnl for t in trades)
        total_comm = sum(t.commission for t in trades)
        total_slip = sum(t.slippage_cost for t in trades)

        win_rate = len(wins) / len(trades) * 100 if trades else 0
        avg_win = np.mean([t.net_pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.net_pnl for t in losses]) if losses else 0

        gross_profit = sum(t.net_pnl for t in wins)
        gross_loss = abs(sum(t.net_pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        expectancy = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * abs(avg_loss))

        print(f"\n--- Performance ---")
        print(f"Win Rate:       {win_rate:.1f}% ({len(wins)}W / {len(losses)}L)")
        print(f"Profit Factor:  {profit_factor:.2f}")
        print(f"Expectancy:     ${expectancy:,.2f} per trade")
        print(f"Avg Win:        ${avg_win:,.2f}")
        print(f"Avg Loss:       ${avg_loss:,.2f}")
        print(f"R:R Actual:     {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "")

        print(f"\n--- P&L ---")
        print(f"Gross P&L:      ${total_gross:,.2f}")
        print(f"Commissions:    ${total_comm:,.2f}")
        print(f"Slippage:       ${total_slip:,.2f}")
        print(f"Net P&L:        ${total_pnl:,.2f}")

        print(f"\n--- Equity ---")
        print(f"Starting:       ${self.position_mgr.account_size:,.2f}")
        print(f"Ending:         ${self.position_mgr.equity:,.2f}")
        print(f"Max Drawdown:   ${self.position_mgr.max_drawdown_seen:,.2f}")
        if result.equity_curve:
            print(f"Max DD %:       {result.equity_curve.max_drawdown_pct:.2f}%")

        # Per-strategy breakdown
        strat_trades: Dict[str, List[Trade]] = {}
        for t in trades:
            strat_trades.setdefault(t.strategy_name, []).append(t)

        if len(strat_trades) > 1:
            print(f"\n--- By Strategy ---")
            for sname, strades in sorted(strat_trades.items()):
                s_wins = sum(1 for t in strades if t.net_pnl > 0)
                s_wr = s_wins / len(strades) * 100 if strades else 0
                s_pnl = sum(t.net_pnl for t in strades)
                print(f"  {sname:25s}: {len(strades):3d} trades, {s_wr:5.1f}% WR, ${s_pnl:>10,.2f}")
