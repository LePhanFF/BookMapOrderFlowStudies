"""
Live Trading Wrapper
====================
Handles paper trading and live trading execution with:
- Risk management (daily limits, consecutive losses)
- Trade execution simulation
- P&L tracking
- Position management
- Emergency stops

Based on FINAL_STRATEGY.md and DUAL_STRATEGY_SYSTEM.md
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from pathlib import Path

from dual_strategy import DualStrategy, TradeSignal, SignalType


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradeStatus(Enum):
    """Trade status"""
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class ExitReason(Enum):
    """Reason for trade exit"""
    STOP_LOSS = "stop_loss"
    PROFIT_TARGET = "profit_target"
    TIME_EXIT = "time_exit"
    MANUAL = "manual"
    EMERGENCY = "emergency"


@dataclass
class Position:
    """Represents an open position"""
    signal: TradeSignal
    entry_time: datetime
    status: TradeStatus = TradeStatus.OPEN
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[ExitReason] = None
    pnl: float = 0.0
    bars_held: int = 0
    
    def calculate_pnl(self, exit_price: float, point_value: float = 2.0) -> float:
        """Calculate P&L for the position"""
        if self.signal.direction == 'LONG':
            price_diff = exit_price - self.signal.entry_price
        else:  # SHORT
            price_diff = self.signal.entry_price - exit_price
        
        return price_diff * point_value * self.signal.position_size
    
    def check_exit_conditions(self, current_bar: pd.Series, 
                             bar_count: int) -> Tuple[bool, ExitReason, float]:
        """
        Check if position should be exited
        
        Returns: (should_exit, reason, exit_price)
        """
        current_price = current_bar['close']
        high = current_bar['high']
        low = current_bar['low']
        
        # Check stop loss
        if self.signal.direction == 'LONG':
            if low <= self.signal.stop_price:
                return True, ExitReason.STOP_LOSS, self.signal.stop_price
        else:  # SHORT
            if high >= self.signal.stop_price:
                return True, ExitReason.STOP_LOSS, self.signal.stop_price
        
        # Check profit target
        if self.signal.direction == 'LONG':
            if high >= self.signal.target_price:
                return True, ExitReason.PROFIT_TARGET, self.signal.target_price
        else:  # SHORT
            if low <= self.signal.target_price:
                return True, ExitReason.PROFIT_TARGET, self.signal.target_price
        
        # Check time exit
        if bar_count >= self.signal.position_size:  # Using position_size as max_hold_bars temporarily
            return True, ExitReason.TIME_EXIT, current_price
        
        return False, None, current_price


@dataclass
class TradingState:
    """Current trading state"""
    date: datetime
    daily_pnl: float = 0.0
    daily_trades: int = 0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    current_position: Optional[Position] = None
    positions_closed: List[Position] = field(default_factory=list)
    trading_enabled: bool = True
    
    def reset_daily(self, new_date: datetime):
        """Reset daily counters"""
        self.date = new_date
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.current_position = None
        self.positions_closed = []
        self.trading_enabled = True


class LiveTradingWrapper:
    """
    Live Trading Wrapper for Dual Strategy
    
    Handles:
    - Paper trading simulation
    - Live trade execution
    - Risk management
    - Trade logging
    - Performance tracking
    """
    
    # Risk management defaults
    DEFAULT_RISK_CONFIG = {
        'max_daily_loss': 2000.0,
        'max_drawdown': 4000.0,
        'max_consecutive_losses': 5,
        'max_trades_per_day': 15,
        'emergency_stop_loss': 1500.0,
        'point_value': 2.0,  # MNQ
        'commission_per_contract': 0.25,  # per side
        'slippage_ticks': 0,  # Limit orders = no slippage
        'tick_size': 0.25,  # MNQ tick size
        'mode': 'paper',  # 'paper' or 'live'
    }
    
    def __init__(self, strategy: DualStrategy, 
                 risk_config: Optional[Dict] = None,
                 log_dir: str = './trade_logs'):
        """
        Initialize trading wrapper
        
        Args:
            strategy: DualStrategy instance
            risk_config: Risk management configuration
            log_dir: Directory for trade logs
        """
        self.strategy = strategy
        self.risk_config = {**self.DEFAULT_RISK_CONFIG, **(risk_config or {})}
        self.state = None
        self.trade_history: List[Position] = []
        self.daily_stats: Dict[datetime, Dict] = {}
        
        # Setup logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Position size scaling
        self.current_tier_1_size = strategy.config['tier_1_size']
        self.current_tier_2_size = strategy.config['tier_2_size']
        
        logger.info(f"LiveTradingWrapper initialized (mode: {self.risk_config['mode']})")
    
    def start_new_day(self, date: datetime):
        """Start trading for a new day"""
        if self.state:
            # Save previous day's stats
            self._save_daily_stats()
        
        self.state = TradingState(date=date)
        logger.info(f"Started new trading day: {date.date()}")
        
        # Reset position sizes
        self.current_tier_1_size = self.strategy.config['tier_1_size']
        self.current_tier_2_size = self.strategy.config['tier_2_size']
    
    def can_trade(self) -> Tuple[bool, str]:
        """
        Check if trading is allowed
        
        Returns: (can_trade, reason)
        """
        if not self.state:
            return False, "No trading state initialized"
        
        if not self.state.trading_enabled:
            return False, "Trading disabled"
        
        # Check daily loss limit
        if self.state.daily_pnl <= -self.risk_config['max_daily_loss']:
            return False, f"Daily loss limit reached: ${self.state.daily_pnl:.2f}"
        
        # Check max trades per day
        if self.state.daily_trades >= self.risk_config['max_trades_per_day']:
            return False, f"Max trades per day reached: {self.state.daily_trades}"
        
        # Check consecutive losses
        if self.state.consecutive_losses >= self.risk_config['max_consecutive_losses']:
            return False, f"Max consecutive losses reached: {self.state.consecutive_losses}"
        
        # Check if already in position
        if self.state.current_position:
            return False, "Already in position"
        
        return True, "OK"
    
    def validate_signal_risk(self, signal: TradeSignal) -> Tuple[bool, str]:
        """
        Validate signal against current risk parameters
        
        Returns: (is_valid, reason)
        """
        # Calculate actual position size based on current scaling
        if signal.signal_type == SignalType.TIER_1:
            signal.position_size = self.current_tier_1_size
        else:
            signal.position_size = self.current_tier_2_size
        
        # Validate with strategy
        is_valid, reason = self.strategy.validate_signal(
            signal, 
            max_risk_per_trade=400.0
        )
        
        if not is_valid:
            return False, reason
        
        return True, "OK"
    
    def enter_position(self, signal: TradeSignal) -> Optional[Position]:
        """
        Enter a new position
        
        Returns Position object or None if entry failed
        """
        can_trade, reason = self.can_trade()
        if not can_trade:
            logger.warning(f"Cannot enter position: {reason}")
            return None
        
        # Validate signal
        is_valid, reason = self.validate_signal_risk(signal)
        if not is_valid:
            logger.warning(f"Signal validation failed: {reason}")
            return None
        
        # Create position
        position = Position(
            signal=signal,
            entry_time=signal.timestamp,
            status=TradeStatus.OPEN
        )
        
        self.state.current_position = position
        self.state.daily_trades += 1
        
        logger.info(
            f"ENTER: {signal.direction} {signal.position_size} MNQ @ {signal.entry_price:.2f} "
            f"(Type: {signal.signal_type.name}, Stop: {signal.stop_price:.2f}, "
            f"Target: {signal.target_price:.2f})"
        )
        
        return position
    
    def exit_position(self, position: Position, exit_price: float, 
                     reason: ExitReason, exit_time: datetime):
        """
        Close a position
        """
        position.status = TradeStatus.CLOSED
        position.exit_time = exit_time
        position.exit_price = exit_price
        position.exit_reason = reason
        
        # Calculate P&L
        position.pnl = position.calculate_pnl(
            exit_price, 
            self.risk_config['point_value']
        )
        
        # Subtract commission (entry + exit)
        commission = (position.signal.position_size * 
                     self.risk_config['commission_per_contract'] * 2)
        position.pnl -= commission
        
        # Update state
        self.state.daily_pnl += position.pnl
        self.state.positions_closed.append(position)
        self.state.current_position = None
        
        # Update consecutive counters
        if position.pnl > 0:
            self.state.consecutive_wins += 1
            self.state.consecutive_losses = 0
        else:
            self.state.consecutive_losses += 1
            self.state.consecutive_wins = 0
            
            # Apply position size reduction after consecutive losses
            if self.state.consecutive_losses >= 3:
                self._reduce_position_size()
        
        # Check if we should stop trading
        if self.state.daily_pnl <= -self.risk_config['max_daily_loss']:
            self._stop_trading("Daily loss limit reached")
        
        logger.info(
            f"EXIT: {position.signal.direction} {position.signal.position_size} MNQ @ {exit_price:.2f} "
            f"P&L: ${position.pnl:.2f} (Reason: {reason.value})"
        )
        
        return position
    
    def _reduce_position_size(self):
        """Reduce position size after consecutive losses"""
        self.current_tier_1_size = max(10, self.current_tier_1_size - 5)
        self.current_tier_2_size = max(5, self.current_tier_2_size - 5)
        
        logger.warning(
            f"Position size reduced due to consecutive losses. "
            f"Tier 1: {self.current_tier_1_size}, Tier 2: {self.current_tier_2_size}"
        )
    
    def _stop_trading(self, reason: str):
        """Stop trading for the day"""
        self.state.trading_enabled = False
        
        # Close any open position
        if self.state.current_position:
            logger.warning("Closing open position due to stop trading")
            current_price = self.state.current_position.signal.entry_price
            self.exit_position(
                self.state.current_position,
                current_price,
                ExitReason.EMERGENCY,
                datetime.now()
            )
        
        logger.error(f"Trading stopped: {reason}")
    
    def process_bar(self, bar: pd.Series, bar_time: datetime) -> Optional[Position]:
        """
        Process a single price bar
        
        Args:
            bar: Price bar data
            bar_time: Timestamp of the bar
        
        Returns:
            Closed position if position was exited, None otherwise
        """
        if not self.state:
            logger.error("No trading state. Call start_new_day() first.")
            return None
        
        closed_position = None
        
        # Check if we have an open position
        if self.state.current_position:
            position = self.state.current_position
            position.bars_held += 1
            
            # Check exit conditions
            should_exit, reason, exit_price = position.check_exit_conditions(
                bar, position.bars_held
            )
            
            if should_exit:
                closed_position = self.exit_position(
                    position, exit_price, reason, bar_time
                )
        
        return closed_position
    
    def generate_and_enter(self, df: pd.DataFrame, bar_idx: int) -> Optional[Position]:
        """
        Generate signals and enter position if valid
        
        Args:
            df: DataFrame with price data
            bar_idx: Current bar index
        
        Returns:
            Position if entered, None otherwise
        """
        if not self.can_trade()[0]:
            return None
        
        # Get data up to current bar
        df_slice = df.iloc[:bar_idx+1]
        
        # Generate signals
        signals = self.strategy.get_all_signals(df_slice)
        
        # Check if there's a signal on the current bar
        if not signals:
            return None
        
        # Get the most recent signal
        latest_signal = signals[-1]
        current_time = df.iloc[bar_idx]['timestamp']
        
        # Check if signal is for current bar
        if latest_signal.timestamp == current_time:
            return self.enter_position(latest_signal)
        
        return None
    
    def run_backtest_simulation(self, df: pd.DataFrame) -> Dict:
        """
        Run paper trading simulation on historical data
        
        Args:
            df: DataFrame with historical data
        
        Returns:
            Dictionary with performance metrics
        """
        logger.info("Starting paper trading simulation...")
        
        # Pre-compute all signals
        df = self.strategy.generate_signals(df)
        
        # Process each bar
        for idx, row in df.iterrows():
            bar_time = row['timestamp']
            
            # Check if new day
            if self.state is None or bar_time.date() != self.state.date.date():
                self.start_new_day(bar_time)
            
            # Process any open position
            self.process_bar(row, bar_time)
            
            # Check for new entry (if no position)
            if not self.state.current_position:
                # Check for signals
                if row['has_signal']:
                    # Determine direction
                    if row['signal_A_long'] or row['signal_B_long']:
                        direction = 'LONG'
                        signal_type = SignalType.TIER_1 if row['signal_A_long'] else SignalType.TIER_2
                    else:
                        direction = 'SHORT'
                        signal_type = SignalType.TIER_1 if row['signal_A_short'] else SignalType.TIER_2
                    
                    # Create signal
                    signal = self.strategy.create_trade_signal(row, direction, signal_type)
                    if signal:
                        self.enter_position(signal)
        
        # Save final stats
        self._save_daily_stats()
        
        # Calculate performance metrics
        metrics = self._calculate_metrics()
        
        logger.info("Paper trading simulation complete")
        return metrics
    
    def _save_daily_stats(self):
        """Save daily trading statistics"""
        if not self.state:
            return
        
        self.daily_stats[self.state.date] = {
            'date': self.state.date.strftime('%Y-%m-%d'),
            'daily_pnl': self.state.daily_pnl,
            'daily_trades': self.state.daily_trades,
            'positions': [self._position_to_dict(p) for p in self.state.positions_closed]
        }
    
    def _position_to_dict(self, position: Position) -> Dict:
        """Convert position to dictionary"""
        return {
            'entry_time': position.entry_time.isoformat(),
            'exit_time': position.exit_time.isoformat() if position.exit_time else None,
            'direction': position.signal.direction,
            'signal_type': position.signal.signal_type.name,
            'position_size': position.signal.position_size,
            'entry_price': position.signal.entry_price,
            'exit_price': position.exit_price,
            'stop_price': position.signal.stop_price,
            'target_price': position.signal.target_price,
            'exit_reason': position.exit_reason.value if position.exit_reason else None,
            'pnl': position.pnl,
            'bars_held': position.bars_held
        }
    
    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics"""
        all_positions = []
        for stats in self.daily_stats.values():
            all_positions.extend(stats['positions'])
        
        if not all_positions:
            return {'error': 'No trades executed'}
        
        # Calculate metrics
        total_trades = len(all_positions)
        winning_trades = sum(1 for p in all_positions if p['pnl'] > 0)
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(p['pnl'] for p in all_positions)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        winning_pnls = [p['pnl'] for p in all_positions if p['pnl'] > 0]
        losing_pnls = [p['pnl'] for p in all_positions if p['pnl'] <= 0]
        
        avg_win = np.mean(winning_pnls) if winning_pnls else 0
        avg_loss = np.mean(losing_pnls) if losing_pnls else 0
        
        # Profit factor
        gross_profit = sum(winning_pnls)
        gross_loss = abs(sum(losing_pnls))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Consecutive losses
        max_consecutive_losses = 0
        current_streak = 0
        for p in all_positions:
            if p['pnl'] <= 0:
                current_streak += 1
                max_consecutive_losses = max(max_consecutive_losses, current_streak)
            else:
                current_streak = 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_consecutive_losses': max_consecutive_losses,
            'daily_stats': self.daily_stats
        }
    
    def save_trade_log(self, filename: Optional[str] = None):
        """Save trade log to file"""
        if filename is None:
            filename = f"trade_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.log_dir / filename
        
        # Compile all trades
        all_trades = []
        for stats in self.daily_stats.values():
            all_trades.extend(stats['positions'])
        
        with open(filepath, 'w') as f:
            json.dump(all_trades, f, indent=2)
        
        logger.info(f"Trade log saved to {filepath}")
        return filepath


# Convenience functions
def run_paper_trading(df: pd.DataFrame, 
                     strategy_config: Optional[Dict] = None,
                     risk_config: Optional[Dict] = None) -> Dict:
    """
    Quick function to run paper trading simulation
    
    Usage:
        results = run_paper_trading(df)
        print(f"Total P&L: ${results['total_pnl']:.2f}")
    """
    strategy = DualStrategy(strategy_config)
    wrapper = LiveTradingWrapper(strategy, risk_config)
    return wrapper.run_backtest_simulation(df)


if __name__ == '__main__':
    print("Live Trading Wrapper")
    print("=" * 50)
    print("\nConfiguration:")
    for key, value in LiveTradingWrapper.DEFAULT_RISK_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\nUsage:")
    print("  from live_trading_wrapper import LiveTradingWrapper")
    print("  from dual_strategy import DualStrategy")
    print("")
    print("  strategy = DualStrategy()")
    print("  wrapper = LiveTradingWrapper(strategy)")
    print("  wrapper.start_new_day(datetime.now())")
    print("  # ... process bars ...")
