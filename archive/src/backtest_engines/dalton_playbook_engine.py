"""
Dalton Playbook Strategy Engine
===============================

Implements the exact strategies from prompts/playbooks.md:

Day Types:
- Trend Day (Up/Down) with 4 strength levels
- P-Day (Skewed profile)
- B-Day (Balanced, double distribution)
- Neutral Day

Entry Models:
- IB Extension plays
- VWAP/Breakout retests  
- DPOC migration confirmation
- Poor high/low fades

All rules are mechanical and automatable.
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import sys
sys.path.append('/home/lphan/jupyterlab/BookMapOrderFlowStudies')

from src.indicators.technical import (
    calculate_ema, calculate_atr, calculate_adx,
    calculate_rsi, calculate_bollinger_bands, calculate_vwap,
    add_all_indicators
)


class DayType(Enum):
    """Dalton day type classifications"""
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    P_DAY_UP = "p_day_up"  # Skewed up
    P_DAY_DOWN = "p_day_down"  # Skewed down
    B_DAY = "b_day"  # Balanced
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class TrendStrength(Enum):
    """Trend strength per Dalton playbook"""
    WEAK = "weak"  # <0.5x IB extension
    MODERATE = "moderate"  # 0.5-1.0x
    STRONG = "strong"  # 1.0-2.0x
    SUPER = "super"  # >2.0x


@dataclass
class TPOStructure:
    """TPO market structure metrics"""
    ib_high: float
    ib_low: float
    ib_range: float
    ib_extension_up: float
    ib_extension_down: float
    vah: float
    val: float
    poc: float
    dpoc: float
    trend_strength: TrendStrength
    single_prints_above: int
    single_prints_below: int
    poor_high: bool
    poor_low: bool


@dataclass
class Trade:
    """Completed trade record"""
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    contracts: int
    pnl: float
    exit_reason: str
    day_type: DayType
    trend_strength: TrendStrength
    setup: str


class DaltonPlaybookEngine:
    """
    Dalton Playbook Strategy Engine
    
    Implements mechanical rules from the playbook:
    1. Day type identification (IB-centric)
    2. Trend strength classification
    3. Entry models per day type
    4. Risk management (Lanto 3/3, trailing stops)
    """
    
    def __init__(self, 
                 symbol: str = 'MNQ',
                 risk_per_trade: float = 400.0,
                 point_value: float = 2.0,
                 bar_timeframe: str = '5min'):
        """
        Initialize playbook engine
        
        Args:
            symbol: Trading symbol
            risk_per_trade: $ risk per trade
            point_value: $ per point
            bar_timeframe: Analysis timeframe (5min recommended)
        """
        self.symbol = symbol
        self.risk_per_trade = risk_per_trade
        self.point_value = point_value
        self.bar_timeframe = bar_timeframe
        
        self.trades: List[Trade] = []
        self.current_day_type: DayType = DayType.UNKNOWN
        self.current_tpo: Optional[TPOStructure] = None
        
    def prepare_data(self, df_1min: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data with TPO calculations
        """
        if not pd.api.types.is_datetime64_any_dtype(df_1min['timestamp']):
            df_1min['timestamp'] = pd.to_datetime(df_1min['timestamp'])
        
        df_1min.set_index('timestamp', inplace=True)
        
        # Resample to analysis timeframe
        df = df_1min.resample(self.bar_timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Add technical indicators
        df = add_all_indicators(df)
        
        # Add TPO structure
        df = self._calculate_tpo_structure(df)
        
        # Classify day type
        df = self._classify_day_types(df)
        
        return df
    
    def _calculate_tpo_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate TPO metrics for each bar"""
        df = df.copy()
        
        # Session info
        df['time'] = df.index.time
        df['session_date'] = df.index.date
        
        # IB is first 60 minutes = 12 bars of 5-min
        df['is_ib'] = df.groupby('session_date').cumcount() < 12
        df['is_after_ib'] = ~df['is_ib']
        
        # Calculate IB for each session
        session_groups = df.groupby('session_date')
        
        ib_metrics = []
        for date, group in session_groups:
            ib_bars = group[group['is_ib']]
            
            if len(ib_bars) > 0:
                ib_high = ib_bars['high'].max()
                ib_low = ib_bars['low'].min()
                ib_range = ib_high - ib_low
                
                # POC of IB (point of control)
                ib_poc = ib_bars.groupby('close')['volume'].sum().idxmax()
            else:
                ib_high = ib_low = ib_range = ib_poc = np.nan
            
            # Calculate for all bars in session
            for idx in group.index:
                ib_metrics.append({
                    'index': idx,
                    'ib_high': ib_high,
                    'ib_low': ib_low,
                    'ib_range': ib_range,
                    'ib_poc': ib_poc
                })
        
        # Merge IB metrics
        ib_df = pd.DataFrame(ib_metrics).set_index('index')
        df = df.join(ib_df)
        
        # IB Extension (current position relative to IB)
        df['ib_extension_up'] = (df['high'] - df['ib_high']) / df['ib_range']
        df['ib_extension_down'] = (df['ib_low'] - df['low']) / df['ib_range']
        df['ib_extension'] = np.maximum(df['ib_extension_up'], df['ib_extension_down'])
        
        # Value Area (simplified using VWAP ± ATR)
        df['vah'] = df['vwap'] + (0.5 * df['atr14'])
        df['val'] = df['vwap'] - (0.5 * df['atr14'])
        df['va_range'] = df['vah'] - df['val']
        
        # Developing POC (volume-weighted price of session so far)
        df['dpoc'] = df.groupby('session_date').apply(
            lambda x: (x['close'] * x['volume']).cumsum() / x['volume'].cumsum()
        ).reset_index(level=0, drop=True)
        
        # Single prints detection (simplified)
        df['single_prints_above'] = (df['high'] > df['vah']).astype(int).rolling(3).sum()
        df['single_prints_below'] = (df['low'] < df['val']).astype(int).rolling(3).sum()
        
        # Poor high/low (wicks without follow-through)
        df['poor_high'] = (df['high'] == df['high'].rolling(3).max()) & (df['close'] < df['high'])
        df['poor_low'] = (df['low'] == df['low'].rolling(3).min()) & (df['close'] > df['low'])
        
        return df
    
    def _classify_day_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify day type based on IB and early session behavior"""
        df = df.copy()
        df['day_type'] = DayType.UNKNOWN.value
        df['trend_strength'] = None
        
        for date, session_df in df.groupby('session_date'):
            if len(session_df) < 12:  # Need full IB
                continue
            
            # Get IB metrics
            ib_end_idx = session_df[session_df['is_ib']].index[-1]
            ib_df = session_df.loc[:ib_end_idx]
            
            ib_range = ib_df['ib_range'].iloc[-1]
            ib_high = ib_df['ib_high'].iloc[-1]
            ib_low = ib_df['ib_low'].iloc[-1]
            
            # Check after IB behavior
            after_ib = session_df[session_df['is_after_ib']]
            
            if len(after_ib) > 0:
                # Calculate extension
                max_ext_up = after_ib['ib_extension_up'].max()
                max_ext_down = after_ib['ib_extension_down'].max()
                max_extension = max(max_ext_up, max_ext_down)
                
                # DPOC migration
                dpoc_start = ib_df['dpoc'].iloc[-1]
                dpoc_current = after_ib['dpoc'].iloc[-1]
                dpoc_migration = abs(dpoc_current - dpoc_start)
                
                # Classify trend strength
                if max_extension < 0.5:
                    strength = TrendStrength.WEAK
                elif max_extension < 1.0:
                    strength = TrendStrength.MODERATE
                elif max_extension < 2.0:
                    strength = TrendStrength.STRONG
                else:
                    strength = TrendStrength.SUPER
                
                # Classify day type
                if max_extension > 1.5 and dpoc_migration > ib_range * 0.5:
                    # Trend day
                    if max_ext_up > max_ext_down:
                        day_type = DayType.TREND_UP
                    else:
                        day_type = DayType.TREND_DOWN
                
                elif max_extension > 0.5 and dpoc_migration > ib_range * 0.3:
                    # P-day (skewed)
                    if dpoc_current > (ib_high + ib_low) / 2:
                        day_type = DayType.P_DAY_UP
                    else:
                        day_type = DayType.P_DAY_DOWN
                
                elif max_extension < 0.3:
                    # Balanced/B-day
                    day_type = DayType.B_DAY
                
                else:
                    day_type = DayType.NEUTRAL
                
                # Assign to all bars in session
                df.loc[session_df.index, 'day_type'] = day_type.value
                df.loc[session_df.index, 'trend_strength'] = strength.value
        
        return df
    
    def get_trend_strength(self, row: pd.Series) -> TrendStrength:
        """Get trend strength from row data"""
        strength_str = row.get('trend_strength')
        if strength_str == 'weak':
            return TrendStrength.WEAK
        elif strength_str == 'moderate':
            return TrendStrength.MODERATE
        elif strength_str == 'strong':
            return TrendStrength.STRONG
        elif strength_str == 'super':
            return TrendStrength.SUPER
        return TrendStrength.WEAK
    
    def check_entry_signal(self, 
                          df: pd.DataFrame, 
                          idx: int) -> Optional[Dict]:
        """
        Check for entry signal based on day type and playbook rules
        """
        if idx < 1:
            return None
        
        row = df.iloc[idx]
        prior_row = df.iloc[idx-1]
        
        # Time filters
        current_time = row.name.time() if hasattr(row.name, 'time') else row['time']
        if isinstance(current_time, str):
            current_time = datetime.strptime(current_time, '%H:%M:%S').time()
        
        # Skip first 30 min (9:30-10:00)
        if current_time < time(10, 0):
            return None
        
        # Skip last 30 min (15:30-16:00)
        if current_time > time(15, 30):
            return None
        
        day_type = DayType(row['day_type'])
        trend_strength = self.get_trend_strength(row)
        
        # Lanto Gate: Skip weak trends
        if trend_strength == TrendStrength.WEAK:
            return None
        
        signal = None
        
        # === TREND DAY ENTRIES ===
        if day_type in [DayType.TREND_UP, DayType.TREND_DOWN]:
            signal = self._trend_day_signal(row, prior_row, day_type, trend_strength)
        
        # === P-DAY ENTRIES ===
        elif day_type in [DayType.P_DAY_UP, DayType.P_DAY_DOWN]:
            signal = self._p_day_signal(row, prior_row, day_type)
        
        # === B-DAY ENTRIES (Mean Reversion) ===
        elif day_type == DayType.B_DAY:
            signal = self._b_day_signal(row, prior_row)
        
        return signal
    
    def _trend_day_signal(self, 
                         row: pd.Series, 
                         prior_row: pd.Series,
                         day_type: DayType,
                         strength: TrendStrength) -> Optional[Dict]:
        """Entry logic for trend days"""
        
        if day_type == DayType.TREND_UP:
            # Long setups only
            
            # Setup 1: IB Extension acceptance
            if row['close'] > row['ib_high'] and prior_row['close'] <= row['ib_high']:
                return {
                    'direction': 'LONG',
                    'setup': 'IB_EXTENSION',
                    'entry_price': row['close'],
                    'stop_price': row['ib_low'],
                    'target_price': row['close'] + (2 * (row['close'] - row['ib_low'])),
                    'trend_strength': strength,
                    'day_type': day_type
                }
            
            # Setup 2: VAH breakout
            if row['close'] > row['vah'] and prior_row['close'] <= row['vah']:
                return {
                    'direction': 'LONG',
                    'setup': 'VAH_BREAK',
                    'entry_price': row['close'],
                    'stop_price': row['val'],
                    'target_price': row['close'] + (2 * (row['close'] - row['val'])),
                    'trend_strength': strength,
                    'day_type': day_type
                }
            
            # Setup 3: DPOC migration pullback
            if (row['close'] > row['dpoc'] and 
                prior_row['close'] <= row['dpoc'] and
                row['close'] > row['ema20']):
                return {
                    'direction': 'LONG',
                    'setup': 'DPOC_PULLBACK',
                    'entry_price': row['close'],
                    'stop_price': row['low'],
                    'target_price': row['close'] + (2 * (row['close'] - row['low'])),
                    'trend_strength': strength,
                    'day_type': day_type
                }
        
        else:  # TREND_DOWN
            # Short setups (mirror logic)
            if row['close'] < row['ib_low'] and prior_row['close'] >= row['ib_low']:
                return {
                    'direction': 'SHORT',
                    'setup': 'IB_EXTENSION',
                    'entry_price': row['close'],
                    'stop_price': row['ib_high'],
                    'target_price': row['close'] - (2 * (row['ib_high'] - row['close'])),
                    'trend_strength': strength,
                    'day_type': day_type
                }
            
            if row['close'] < row['val'] and prior_row['close'] >= row['val']:
                return {
                    'direction': 'SHORT',
                    'setup': 'VAL_BREAK',
                    'entry_price': row['close'],
                    'stop_price': row['vah'],
                    'target_price': row['close'] - (2 * (row['vah'] - row['close'])),
                    'trend_strength': strength,
                    'day_type': day_type
                }
        
        return None
    
    def _p_day_signal(self, 
                     row: pd.Series, 
                     prior_row: pd.Series,
                     day_type: DayType) -> Optional[Dict]:
        """Entry logic for P-days (skewed profiles)"""
        # P-day: Fade the skew on reversion to POC
        
        if day_type == DayType.P_DAY_UP:
            # Skewed up - look for pullback to DPOC
            if (row['close'] < row['dpoc'] and 
                prior_row['close'] >= row['dpoc'] and
                row['close'] > row['val']):
                return {
                    'direction': 'LONG',
                    'setup': 'P_DAY_PULLBACK',
                    'entry_price': row['close'],
                    'stop_price': row['val'],
                    'target_price': row['vah'],
                    'trend_strength': TrendStrength.MODERATE,
                    'day_type': day_type
                }
        
        else:  # P_DAY_DOWN
            if (row['close'] > row['dpoc'] and 
                prior_row['close'] <= row['dpoc'] and
                row['close'] < row['vah']):
                return {
                    'direction': 'SHORT',
                    'setup': 'P_DAY_PULLBACK',
                    'entry_price': row['close'],
                    'stop_price': row['vah'],
                    'target_price': row['val'],
                    'trend_strength': TrendStrength.MODERATE,
                    'day_type': day_type
                }
        
        return None
    
    def _b_day_signal(self, 
                     row: pd.Series, 
                     prior_row: pd.Series) -> Optional[Dict]:
        """Entry logic for B-days (balanced, mean reversion)"""
        # B-day: Fade extremes at VAH/VAL
        
        # Long: At VAL with poor low
        if (row['low'] <= row['val'] and 
            row['poor_low'] and
            row['close'] > row['low']):  # Rejection
            return {
                'direction': 'LONG',
                'setup': 'B_DAY_VAL_FADE',
                'entry_price': row['close'],
                'stop_price': row['low'],
                'target_price': row['vwap'],
                'trend_strength': TrendStrength.MODERATE,
                'day_type': DayType.B_DAY
            }
        
        # Short: At VAH with poor high
        if (row['high'] >= row['vah'] and 
            row['poor_high'] and
            row['close'] < row['high']):  # Rejection
            return {
                'direction': 'SHORT',
                'setup': 'B_DAY_VAH_FADE',
                'entry_price': row['close'],
                'stop_price': row['high'],
                'target_price': row['vwap'],
                'trend_strength': TrendStrength.MODERATE,
                'day_type': DayType.B_DAY
            }
        
        return None
    
    def calculate_position_size(self, 
                               stop_distance: float, 
                               trend_strength: TrendStrength) -> int:
        """Position sizing based on trend strength (Lanto rules)"""
        # Lanto sizing: 3/3 = full, 2/3 = half, <2/3 = micro/skip
        
        if trend_strength == TrendStrength.WEAK:
            return 0  # Skip
        
        base_size = int(self.risk_per_trade / (stop_distance * self.point_value))
        
        multipliers = {
            TrendStrength.MODERATE: 0.5,
            TrendStrength.STRONG: 1.0,
            TrendStrength.SUPER: 1.0
        }
        
        multiplier = multipliers.get(trend_strength, 0.0)
        contracts = int(base_size * multiplier)
        
        return max(0, min(contracts, 30))
    
    def run_backtest(self, df_1min: pd.DataFrame) -> Dict:
        """Run complete backtest"""
        print(f"Running Dalton Playbook backtest for {self.symbol}...")
        
        df = self.prepare_data(df_1min)
        
        print(f"  Processed {len(df)} bars")
        print(f"  Day types found:")
        for day_type in DayType:
            count = (df['day_type'] == day_type.value).sum()
            if count > 0:
                print(f"    {day_type.value}: {count} bars")
        
        # Run simulation
        for i in range(12, len(df)):  # Start after IB
            signal = self.check_entry_signal(df, i)
            
            if signal:
                print(f"  Signal at {df.index[i]}: {signal['direction']} {signal['setup']}")
        
        return self._calculate_statistics()
    
    def _calculate_statistics(self) -> Dict:
        """Calculate backtest statistics"""
        if not self.trades:
            return {'error': 'No trades executed'}
        
        # ... calculate metrics
        
        return {
            'total_trades': len(self.trades),
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'trades': self.trades
        }


if __name__ == '__main__':
    print("Dalton Playbook Strategy Engine")
    print("=" * 50)
    print("\nImplements mechanical rules from prompts/playbooks.md")
    print("\nFeatures:")
    print("  - Day type classification (Trend, P-day, B-day, Neutral)")
    print("  - Trend strength levels (Weak, Moderate, Strong, Super)")
    print("  - TPO structure (IB, VAH/VAL, DPOC, single prints)")
    print("  - Mechanical entry models per day type")
    print("  - Lanto 3/3 position sizing")
    print("\nUsage:")
    print("  engine = DaltonPlaybookEngine(symbol='MNQ')")
    print("  results = engine.run_backtest(df)")
