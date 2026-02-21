"""
Dual Strategy System
====================
Implements the dual-tier strategy with:
- Strategy A (Tier 1): Imbalance + Volume + CVD (31 contracts)
- Strategy B (Tier 2): Delta > 85 + CVD (15-20 contracts)

Based on FINAL_STRATEGY.md and DUAL_STRATEGY_SYSTEM.md
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class SignalType(Enum):
    """Signal strength classification"""
    NONE = 0
    TIER_2 = 1  # Strategy B only
    TIER_1 = 2  # Strategy A only
    BOTH = 3    # Both strategies agree


@dataclass
class TradeSignal:
    """Represents a trading signal"""
    timestamp: datetime
    direction: str  # 'LONG' or 'SHORT'
    signal_type: SignalType
    entry_price: float
    stop_price: float
    target_price: float
    position_size: int
    atr: float
    delta_percentile: float
    cvd_rising: bool
    imbalance_pct: Optional[float] = None
    volume_spike: Optional[bool] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging"""
        return {
            'timestamp': self.timestamp,
            'direction': self.direction,
            'signal_type': self.signal_type.name,
            'entry_price': self.entry_price,
            'stop_price': self.stop_price,
            'target_price': self.target_price,
            'position_size': self.position_size,
            'atr': self.atr,
            'delta_percentile': self.delta_percentile,
            'cvd_rising': self.cvd_rising,
            'imbalance_pct': self.imbalance_pct,
            'volume_spike': self.volume_spike
        }


class DualStrategy:
    """
    Dual Strategy System
    
    Strategy A (Tier 1): Imbalance > 85 + Volume > 1.5x + CVD trend
    Strategy B (Tier 2): Delta > 85 + CVD trend
    
    Position Sizing:
    - Tier 1 signals: 31 contracts (full size)
    - Tier 2 signals: 15 contracts (half size)
    - Both signals: 31 contracts (full size, Strategy A takes priority)
    """
    
    # Default parameters
    DEFAULT_CONFIG = {
        'instrument': 'MNQ',
        'timeframe': '1min',
        'session_start': time(10, 0),    # 10:00 AM ET
        'session_end': time(13, 0),      # 1:00 PM ET
        'delta_period': 20,
        'delta_threshold': 85,
        'imbalance_period': 20,
        'imbalance_threshold': 85,
        'volume_spike_threshold': 1.5,
        'atr_period': 14,
        'stop_multiplier': 0.4,
        'reward_multiplier': 2.0,
        'max_hold_bars': 8,
        'tier_1_size': 31,   # Full size for Strategy A
        'tier_2_size': 15,   # Half size for Strategy B
        'point_value': 2.0,  # MNQ is $2 per point
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize strategy with configuration"""
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.signals: List[TradeSignal] = []
        
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features needed for dual strategy
        
        Features:
        - Delta and delta percentile
        - CVD (Cumulative Delta) and trend
        - Imbalance ratio and percentile
        - Volume spike detection
        - ATR
        """
        df = df.copy()
        
        # Ensure we have required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 
                        'volume', 'vol_ask', 'vol_bid']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert timestamp if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Add time column for session filtering
        df['time'] = df['timestamp'].dt.time
        
        # Delta features
        df['delta'] = df['vol_ask'] - df['vol_bid']
        
        # Delta percentile (rolling 20-bar)
        delta_period = self.config['delta_period']
        df['delta_percentile'] = df['delta'].rolling(delta_period, min_periods=1).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
        )
        
        # CVD (Cumulative Delta) and trend
        df['cvd'] = df['delta'].cumsum()
        df['cvd_ma'] = df['cvd'].rolling(20, min_periods=1).mean()
        df['cvd_rising'] = df['cvd'] > df['cvd_ma']
        df['cvd_falling'] = df['cvd'] < df['cvd_ma']
        
        # Imbalance features (Strategy A)
        df['imbalance_raw'] = df['vol_ask'] / (df['vol_bid'].replace(0, 1))
        df['imbalance_smooth'] = df['imbalance_raw'].rolling(5, min_periods=1).mean()
        
        # Imbalance percentile
        imb_period = self.config['imbalance_period']
        df['imbalance_pct'] = df['imbalance_smooth'].rolling(imb_period, min_periods=1).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
        )
        
        # Volume spike detection
        df['volume_ma'] = df['volume'].rolling(20, min_periods=1).mean()
        vol_threshold = self.config['volume_spike_threshold']
        df['volume_spike'] = df['volume'] > (df['volume_ma'] * vol_threshold)
        
        # ATR (if not already present)
        if 'atr14' not in df.columns:
            df['atr14'] = self._calculate_atr(df, period=14)
        
        # Session filter
        session_start = self.config['session_start']
        session_end = self.config['session_end']
        df['in_session'] = (df['time'] >= session_start) & (df['time'] <= session_end)
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        atr = true_range.rolling(period, min_periods=1).mean()
        return atr
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals for both strategies
        
        Returns dataframe with signal columns added
        """
        df = self.compute_features(df)
        
        delta_threshold = self.config['delta_threshold']
        imb_threshold = self.config['imbalance_threshold']
        
        # Strategy A signals (Tier 1 - High Quality)
        # Long: Imbalance > 85 + Volume spike + CVD rising + Delta > 0
        df['signal_A_long'] = (
            df['in_session'] &
            (df['imbalance_pct'] > imb_threshold) &
            df['volume_spike'] &
            df['cvd_rising'] &
            (df['delta'] > 0)
        )
        
        # Short: Imbalance > 85 + Volume spike + CVD falling + Delta < 0
        df['signal_A_short'] = (
            df['in_session'] &
            (df['imbalance_pct'] > imb_threshold) &
            df['volume_spike'] &
            df['cvd_falling'] &
            (df['delta'] < 0)
        )
        
        # Strategy B signals (Tier 2 - Standard Quality)
        # Long: Delta > 85 + CVD rising + Delta > 0
        df['signal_B_long'] = (
            df['in_session'] &
            (df['delta_percentile'] > delta_threshold) &
            df['cvd_rising'] &
            (df['delta'] > 0) &
            ~df['signal_A_long']  # Don't double count
        )
        
        # Short: Delta > 85 + CVD falling + Delta < 0
        df['signal_B_short'] = (
            df['in_session'] &
            (df['delta_percentile'] > delta_threshold) &
            df['cvd_falling'] &
            (df['delta'] < 0) &
            ~df['signal_A_short']  # Don't double count
        )
        
        # Combined signal type
        df['signal_type'] = SignalType.NONE.value
        
        # Both strategies agree
        df.loc[df['signal_A_long'] | df['signal_A_short'], 'signal_type'] = SignalType.TIER_1.value
        df.loc[df['signal_B_long'] | df['signal_B_short'], 'signal_type'] = SignalType.TIER_2.value
        
        # Any signal
        df['has_signal'] = (
            df['signal_A_long'] | df['signal_A_short'] |
            df['signal_B_long'] | df['signal_B_short']
        )
        
        return df
    
    def calculate_position_size(self, signal_type: SignalType) -> int:
        """
        Calculate position size based on signal type
        
        Tier 1 (Strategy A): 31 contracts
        Tier 2 (Strategy B): 15 contracts
        """
        if signal_type == SignalType.TIER_1 or signal_type == SignalType.BOTH:
            return self.config['tier_1_size']
        elif signal_type == SignalType.TIER_2:
            return self.config['tier_2_size']
        else:
            return 0
    
    def calculate_stops(self, entry_price: float, atr: float) -> Tuple[float, float]:
        """
        Calculate stop loss and profit target
        
        Returns: (stop_price, target_price)
        """
        stop_dist = atr * self.config['stop_multiplier']
        target_dist = stop_dist * self.config['reward_multiplier']
        
        return stop_dist, target_dist
    
    def create_trade_signal(self, row: pd.Series, direction: str, 
                           signal_type: SignalType) -> Optional[TradeSignal]:
        """
        Create a trade signal from a dataframe row
        
        Args:
            row: DataFrame row with all features
            direction: 'LONG' or 'SHORT'
            signal_type: SignalType enum
        
        Returns:
            TradeSignal object or None
        """
        if not signal_type or signal_type == SignalType.NONE:
            return None
        
        # Get position size
        position_size = self.calculate_position_size(signal_type)
        
        if position_size == 0:
            return None
        
        # Entry price (use close of signal bar)
        entry_price = row['close']
        
        # Calculate stops
        atr = row.get('atr14', row.get('atr', 10))  # Default 10 if no ATR
        stop_dist, target_dist = self.calculate_stops(entry_price, atr)
        
        if direction == 'LONG':
            stop_price = entry_price - stop_dist
            target_price = entry_price + target_dist
        else:  # SHORT
            stop_price = entry_price + stop_dist
            target_price = entry_price - target_dist
        
        # Create signal
        signal = TradeSignal(
            timestamp=row['timestamp'],
            direction=direction,
            signal_type=signal_type,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            position_size=position_size,
            atr=atr,
            delta_percentile=row['delta_percentile'],
            cvd_rising=row['cvd_rising'],
            imbalance_pct=row.get('imbalance_pct'),
            volume_spike=row.get('volume_spike')
        )
        
        return signal
    
    def get_all_signals(self, df: pd.DataFrame) -> List[TradeSignal]:
        """
        Get all trade signals from dataframe
        
        Returns list of TradeSignal objects
        """
        df = self.generate_signals(df)
        signals = []
        
        for idx, row in df.iterrows():
            signal = None
            
            # Check Strategy A first (priority)
            if row['signal_A_long']:
                signal = self.create_trade_signal(
                    row, 'LONG', SignalType.TIER_1
                )
            elif row['signal_A_short']:
                signal = self.create_trade_signal(
                    row, 'SHORT', SignalType.TIER_1
                )
            # Then Strategy B
            elif row['signal_B_long']:
                signal = self.create_trade_signal(
                    row, 'LONG', SignalType.TIER_2
                )
            elif row['signal_B_short']:
                signal = self.create_trade_signal(
                    row, 'SHORT', SignalType.TIER_2
                )
            
            if signal:
                signals.append(signal)
        
        self.signals = signals
        return signals
    
    def get_signal_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for signals
        
        Returns dictionary with counts and metrics
        """
        df = self.generate_signals(df)
        
        summary = {
            'total_bars': len(df),
            'session_bars': df['in_session'].sum(),
            'strategy_A_long': df['signal_A_long'].sum(),
            'strategy_A_short': df['signal_A_short'].sum(),
            'strategy_B_long': df['signal_B_long'].sum(),
            'strategy_B_short': df['signal_B_short'].sum(),
            'total_signals': df['has_signal'].sum(),
        }
        
        # Signal distribution by hour
        df['hour'] = df['timestamp'].dt.hour
        summary['signals_by_hour'] = df[df['has_signal']].groupby('hour').size().to_dict()
        
        return summary
    
    def validate_signal(self, signal: TradeSignal, 
                       max_risk_per_trade: float = 400.0) -> Tuple[bool, str]:
        """
        Validate a trade signal against risk parameters
        
        Returns: (is_valid, reason)
        """
        point_value = self.config['point_value']
        
        # Calculate risk
        stop_distance = abs(signal.entry_price - signal.stop_price)
        risk_per_contract = stop_distance * point_value
        total_risk = risk_per_contract * signal.position_size
        
        if total_risk > max_risk_per_trade:
            return False, f"Risk ${total_risk:.2f} exceeds max ${max_risk_per_trade}"
        
        # Check ATR is reasonable
        if signal.atr <= 0 or np.isnan(signal.atr):
            return False, "Invalid ATR value"
        
        # Check entry price is valid
        if signal.entry_price <= 0 or np.isnan(signal.entry_price):
            return False, "Invalid entry price"
        
        return True, "Valid"


# Convenience functions for quick usage
def create_dual_strategy(config: Optional[Dict] = None) -> DualStrategy:
    """Factory function to create dual strategy"""
    return DualStrategy(config)


def get_signals_for_dataframe(df: pd.DataFrame, 
                              config: Optional[Dict] = None) -> List[TradeSignal]:
    """
    Quick function to get signals from dataframe
    
    Usage:
        signals = get_signals_for_dataframe(df)
        for signal in signals:
            print(f"{signal.direction} at {signal.entry_price}")
    """
    strategy = DualStrategy(config)
    return strategy.get_all_signals(df)


if __name__ == '__main__':
    # Example usage
    print("Dual Strategy System")
    print("=" * 50)
    print("\nConfiguration:")
    for key, value in DualStrategy.DEFAULT_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\nStrategy Logic:")
    print("  Strategy A (Tier 1): Imbalance > 85% + Volume > 1.5x + CVD trend")
    print("  Strategy B (Tier 2): Delta > 85% + CVD trend")
    print("\nPosition Sizing:")
    print("  Tier 1 signals: 31 contracts (full size)")
    print("  Tier 2 signals: 15 contracts (half size)")
