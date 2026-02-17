"""
Trend Following Backtest Engine with Dalton TPO Integration
===========================================================

Combines:
1. Multi-timeframe trend detection (EMA)
2. Dalton TPO concepts (Value Area, IB range, Trend Strength)
3. Breakout entries (20-period high/low, prior day levels, VAH/VAL)
4. Mechanical rules with no discretion

Entry Types:
- Value Area Breakout (VAH/VAL breach)
- IB Range Extension (trend confirmation)
- Prior Day High/Low breakout
- 20-period high/low breakout

Trend Strength Filters:
- Weak: <0.5x IB extension - AVOID or micro size
- Moderate: 0.5-1.0x IB extension - half size
- Strong: 1.0-2.0x IB extension - full size
- Super: >2.0x IB extension - full size + runners
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sys
sys.path.append('/home/lphan/jupyterlab/BookMapOrderFlowStudies')

from src.indicators.technical import (
    calculate_ema, calculate_atr, calculate_adx,
    add_all_indicators
)


@dataclass
class Trade:
    """Represents a completed trade"""
    entry_time: datetime
    exit_time: datetime
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    contracts: int
    pnl: float
    exit_reason: str
    setup_type: str  # 'VA_BREAK', 'IB_EXTENSION', 'PRIOR_DAY', '20PERIOD'
    trend_strength: str  # 'WEAK', 'MODERATE', 'STRONG', 'SUPER'


class TrendFollowingEngine:
    """
    Trend Following Strategy with Dalton TPO Integration
    
    Strategy Components:
    1. HTF Trend Filter (15-min): EMA alignment + IB trend
    2. Entry Signals: Breakouts of key levels (VAH/VAL, IBH/IBL, Prior Day)
    3. Trend Strength: Size positions based on IB extension
    4. Risk Management: Dynamic stops based on structure
    """
    
    def __init__(self, 
                 symbol: str = 'MNQ',
                 risk_per_trade: float = 400.0,
                 point_value: float = 2.0,
                 htf_timeframe: str = '15min',
                 entry_timeframe: str = '5min'):
        """
        Initialize engine
        
        Args:
            symbol: Trading symbol
            risk_per_trade: Maximum risk per trade ($)
            point_value: Value per point ($)
            htf_timeframe: Higher timeframe for trend (15min or 30min)
            entry_timeframe: Entry timeframe (5min or 1min)
        """
        self.symbol = symbol
        self.risk_per_trade = risk_per_trade
        self.point_value = point_value
        self.htf_timeframe = htf_timeframe
        self.entry_timeframe = entry_timeframe
        
        # Results tracking
        self.trades: List[Trade] = []
        self.daily_stats: Dict = {}
        
    def prepare_data(self, df_1min: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare multi-timeframe data with TPO calculations
        
        Args:
            df_1min: 1-minute OHLCV dataframe
            
        Returns:
            (htf_df, entry_df) - Higher timeframe and entry timeframe dataframes
        """
        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df_1min['timestamp']):
            df_1min['timestamp'] = pd.to_datetime(df_1min['timestamp'])
        
        df_1min.set_index('timestamp', inplace=True)
        
        # Resample to higher timeframe (15-min)
        htf_df = df_1min.resample(self.htf_timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Resample to entry timeframe (5-min)
        entry_df = df_1min.resample(self.entry_timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Add indicators to both
        htf_df = add_all_indicators(htf_df)
        entry_df = add_all_indicators(entry_df)
        
        # Calculate TPO/Value Area metrics on HTF
        htf_df = self._calculate_tpo_metrics(htf_df)
        
        return htf_df, entry_df
    
    def _calculate_tpo_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate TPO and Value Area metrics
        
        For simplicity, we calculate:
        - IB Range (first 60 min of session)
        - Value Area (70% of volume around POC)
        - Trend Strength (IB extension multiple)
        """
        df = df.copy()
        
        # Session identification (assuming RTH 9:30-16:00)
        df['time'] = df.index.time
        df['session_date'] = df.index.date
        
        # Identify IB period (first 4 bars of 15-min = 60 min)
        df['is_ib'] = df.groupby('session_date').cumcount() < 4
        
        # Calculate IB High/Low for each session
        session_groups = df.groupby('session_date')
        
        ib_highs = []
        ib_lows = []
        
        for session_date, group in session_groups:
            ib_bars = group[group['is_ib']]
            if len(ib_bars) > 0:
                ib_high = ib_bars['high'].max()
                ib_low = ib_bars['low'].min()
                ib_highs.extend([ib_high] * len(group))
                ib_lows.extend([ib_low] * len(group))
            else:
                ib_highs.extend([np.nan] * len(group))
                ib_lows.extend([np.nan] * len(group))
        
        df['ib_high'] = ib_highs
        df['ib_low'] = ib_lows
        df['ib_range'] = df['ib_high'] - df['ib_low']
        
        # Calculate IB extension (how far beyond IB)
        df['ib_extension_up'] = (df['high'] - df['ib_high']) / df['ib_range']
        df['ib_extension_down'] = (df['ib_low'] - df['low']) / df['ib_range']
        
        # Value Area (simplified: use VWAP ± ATR as proxy)
        df['vah'] = df['vwap'] + (0.5 * df['atr14'])
        df['val'] = df['vwap'] - (0.5 * df['atr14'])
        
        # Prior day high/low (simplified)
        df['prior_day_high'] = df.groupby('session_date')['high'].transform('max').shift(4)
        df['prior_day_low'] = df.groupby('session_date')['low'].transform('min').shift(4)
        
        # 20-period high/low
        df['20_period_high'] = df['high'].rolling(20, min_periods=1).max()
        df['20_period_low'] = df['low'].rolling(20, min_periods=1).min()
        
        return df
    
    def detect_trend_direction(self, htf_row: pd.Series) -> str:
        """
        Detect trend direction using HTF indicators
        
        Returns: 'UP', 'DOWN', or 'NEUTRAL'
        """
        # EMA trend
        ema_bullish = htf_row['close'] > htf_row['ema20'] and htf_row['ema20'] > htf_row['ema50']
        ema_bearish = htf_row['close'] < htf_row['ema20'] and htf_row['ema20'] < htf_row['ema50']
        
        # ADX filter (strong trend > 25)
        strong_trend = htf_row['adx14'] > 25
        
        if ema_bullish and strong_trend:
            return 'UP'
        elif ema_bearish and strong_trend:
            return 'DOWN'
        else:
            return 'NEUTRAL'
    
    def classify_trend_strength(self, htf_row: pd.Series, direction: str) -> str:
        """
        Classify trend strength based on Dalton playbook
        
        Returns: 'WEAK', 'MODERATE', 'STRONG', 'SUPER'
        """
        if direction == 'UP':
            extension = htf_row['ib_extension_up']
        elif direction == 'DOWN':
            extension = htf_row['ib_extension_down']
        else:
            return 'NEUTRAL'
        
        # Dalton trend strength classification
        if extension < 0.5:
            return 'WEAK'
        elif 0.5 <= extension < 1.0:
            return 'MODERATE'
        elif 1.0 <= extension < 2.0:
            return 'STRONG'
        else:  # > 2.0
            return 'SUPER'
    
    def calculate_position_size(self, stop_distance: float, trend_strength: str) -> int:
        """
        Calculate position size based on risk and trend strength
        
        Trend Strength Sizing:
        - WEAK: Micro size (25% of normal) or skip
        - MODERATE: Half size (50%)
        - STRONG: Full size (100%)
        - SUPER: Full size (100%)
        """
        # Base size
        base_size = int(self.risk_per_trade / (stop_distance * self.point_value))
        
        # Adjust for trend strength
        multipliers = {
            'WEAK': 0.0,      # Skip weak trends
            'MODERATE': 0.5,  # Half size
            'STRONG': 1.0,    # Full size
            'SUPER': 1.0      # Full size
        }
        
        multiplier = multipliers.get(trend_strength, 0.0)
        contracts = int(base_size * multiplier)
        
        # Min/max limits
        return max(0, min(contracts, 30))  # Max 30 contracts
    
    def check_entry_signals(self, 
                           htf_row: pd.Series, 
                           entry_row: pd.Series,
                           prior_entry: pd.Series) -> Optional[Dict]:
        """
        Check for entry signals
        
        Returns: Signal dict or None
        """
        # Get trend direction
        trend = self.detect_trend_direction(htf_row)
        
        if trend == 'NEUTRAL':
            return None
        
        # Get trend strength
        trend_strength = self.classify_trend_strength(htf_row, trend)
        
        # Skip weak trends
        if trend_strength == 'WEAK':
            return None
        
        signal = None
        
        # Check session time (avoid first 30 min and last 30 min)
        current_time = entry_row.name.time() if hasattr(entry_row.name, 'time') else entry_row['time']
        if isinstance(current_time, str):
            current_time = datetime.strptime(current_time, '%H:%M:%S').time()
        
        if current_time < time(10, 0) or current_time > time(15, 30):
            return None  # Too early or too late
        
        # Signal 1: Value Area Breakout
        if trend == 'UP' and entry_row['close'] > htf_row['vah']:
            if prior_entry['close'] <= htf_row['vah']:  # Cross above
                signal = {
                    'direction': 'LONG',
                    'setup': 'VAH_BREAK',
                    'entry_price': entry_row['close'],
                    'stop_price': htf_row['val'],  # Stop below VAL
                    'trend_strength': trend_strength
                }
        
        elif trend == 'DOWN' and entry_row['close'] < htf_row['val']:
            if prior_entry['close'] >= htf_row['val']:  # Cross below
                signal = {
                    'direction': 'SHORT',
                    'setup': 'VAL_BREAK',
                    'entry_price': entry_row['close'],
                    'stop_price': htf_row['vah'],  # Stop above VAH
                    'trend_strength': trend_strength
                }
        
        # Signal 2: IB Extension (if no VA signal)
        if signal is None and trend_strength in ['STRONG', 'SUPER']:
            if trend == 'UP' and htf_row['ib_extension_up'] > 1.0:
                signal = {
                    'direction': 'LONG',
                    'setup': 'IB_EXTENSION',
                    'entry_price': entry_row['close'],
                    'stop_price': htf_row['ib_high'],
                    'trend_strength': trend_strength
                }
            elif trend == 'DOWN' and htf_row['ib_extension_down'] > 1.0:
                signal = {
                    'direction': 'SHORT',
                    'setup': 'IB_EXTENSION',
                    'entry_price': entry_row['close'],
                    'stop_price': htf_row['ib_low'],
                    'trend_strength': trend_strength
                }
        
        # Signal 3: 20-Period High/Low Breakout
        if signal is None:
            if trend == 'UP' and entry_row['close'] > htf_row['20_period_high']:
                signal = {
                    'direction': 'LONG',
                    'setup': '20PERIOD_HIGH',
                    'entry_price': entry_row['close'],
                    'stop_price': entry_row['low'],
                    'trend_strength': trend_strength
                }
            elif trend == 'DOWN' and entry_row['close'] < htf_row['20_period_low']:
                signal = {
                    'direction': 'SHORT',
                    'setup': '20PERIOD_LOW',
                    'entry_price': entry_row['close'],
                    'stop_price': entry_row['high'],
                    'trend_strength': trend_strength
                }
        
        return signal
    
    def run_backtest(self, df_1min: pd.DataFrame) -> Dict:
        """
        Run complete backtest
        
        Returns:
            Dictionary with results and statistics
        """
        print(f"Running Trend Following backtest for {self.symbol}...")
        
        # Prepare data
        htf_df, entry_df = self.prepare_data(df_1min)
        
        print(f"  HTF data: {len(htf_df)} bars ({self.htf_timeframe})")
        print(f"  Entry data: {len(entry_df)} bars ({self.entry_timeframe})")
        
        # Align dataframes
        # For each entry bar, find the corresponding HTF bar
        in_position = False
        current_trade = None
        
        for i in range(1, len(entry_df)):
            if in_position:
                # Manage existing position
                pass  # Will implement
            else:
                # Look for new entry
                entry_row = entry_df.iloc[i]
                prior_entry = entry_df.iloc[i-1]
                
                # Get corresponding HTF bar
                entry_time = entry_row.name
                htf_idx = htf_df.index.get_indexer([entry_time], method='pad')[0]
                
                if htf_idx < 0:
                    continue
                
                htf_row = htf_df.iloc[htf_idx]
                
                # Check for signal
                signal = self.check_entry_signals(htf_row, entry_row, prior_entry)
                
                if signal:
                    print(f"  Signal at {entry_time}: {signal['direction']} {signal['setup']}")
        
        # Calculate results
        results = self._calculate_statistics()
        
        return results
    
    def _calculate_statistics(self) -> Dict:
        """Calculate backtest statistics"""
        if not self.trades:
            return {'error': 'No trades executed'}
        
        # Basic stats
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t.pnl > 0)
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(t.pnl for t in self.trades)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        winning_pnls = [t.pnl for t in self.trades if t.pnl > 0]
        losing_pnls = [t.pnl for t in self.trades if t.pnl <= 0]
        
        avg_win = np.mean(winning_pnls) if winning_pnls else 0
        avg_loss = np.mean(losing_pnls) if losing_pnls else 0
        
        # Profit factor
        gross_profit = sum(winning_pnls)
        gross_loss = abs(sum(losing_pnls))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
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
            'trades': self.trades
        }


if __name__ == '__main__':
    print("Trend Following Engine with Dalton TPO")
    print("=" * 50)
    print("\nThis engine combines:")
    print("  - Multi-timeframe trend detection")
    print("  - Dalton TPO concepts (Value Area, IB Range)")
    print("  - Trend strength classification")
    print("  - Dynamic position sizing")
    print("\nUsage:")
    print("  engine = TrendFollowingEngine(symbol='MNQ')")
    print("  results = engine.run_backtest(df)")
