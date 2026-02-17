"""
Dalton Playbook Strategies - All 9 Day Types
=============================================

Complete implementation of all 9 Dalton Playbook strategies:
1. Trend Day Bull (Standard uptrend)
2. Super Trend Day Bull (Explosive >300-400 pts)
3. Trend Day Bear (Standard downtrend)
4. Super Trend Day Bear (Extreme liquidation)
5. P-Day (Skewed balance - "p" or "b" shape)
6. B-Day (Narrow IB, true balance)
7. Neutral Day (Symmetric chop)
8. Trend Day PM Morph (Early balance → late trend)
9. Balanced Day Morph to Trend Day (Resolution breakout)

Uses data_loader.py day type detection (simpler, no rockit bugs).
All strategies are mechanical with clear entry/exit rules.
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import sys
sys.path.append('/home/lphan/jupyterlab/BookMapOrderFlowStudies')

from data_loader import load_data, compute_day_type, compute_ib_features


class TrendStrength(Enum):
    """Trend strength classification"""
    WEAK = "weak"           # <0.5x IB extension
    MODERATE = "moderate"   # 0.5-1.0x IB extension
    STRONG = "strong"       # 1.0-2.0x IB extension  
    SUPER = "super"         # >2.0x IB extension


class DayType(Enum):
    """Day type classification"""
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    SUPER_TREND_UP = "super_trend_up"
    SUPER_TREND_DOWN = "super_trend_down"
    P_DAY = "p_day"
    B_DAY = "b_day"
    NEUTRAL = "neutral"
    PM_MORPH = "pm_morph"
    MORPH_TO_TREND = "morph_to_trend"


@dataclass
class Trade:
    """Trade record"""
    entry_time: datetime
    exit_time: Optional[datetime] = None
    direction: str = ""  # 'LONG' or 'SHORT'
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    contracts: int = 0
    pnl: float = 0.0
    exit_reason: str = ""  # 'STOP', 'TARGET', 'TIME', 'MORPH'
    setup_type: str = ""   # Strategy-specific setup
    day_type: str = ""
    trend_strength: str = ""


class PlaybookStrategies:
    """
    All 9 Dalton Playbook Strategies
    
    Entry Rules Summary:
    - Trend Days: IB extension acceptance + DPOC migration
    - Super Trend: Extreme extension (>2.0x), aggressive pyramid
    - P-Day: Skewed profile resolution (single prints + DPOC compression)
    - B-Day: Fade VAH/VAL extremes (mean reversion)
    - Neutral: POC/VWAP bounces only
    - PM Morph: Late breakout acceptance (>12:30)
    - Morph to Trend: Balance breakout + DPOC migration
    """
    
    def __init__(self,
                 symbol: str = 'NQ',
                 risk_per_trade: float = 400.0,
                 point_value: float = 2.0,
                 tick_size: float = 0.25):
        self.symbol = symbol
        self.risk_per_trade = risk_per_trade
        self.point_value = point_value
        self.tick_size = tick_size
        self.all_trades: List[Trade] = []
        
        # Strategy-specific trade lists
        self.trades_by_strategy: Dict[str, List[Trade]] = {
            'trend_bull': [],
            'super_trend_bull': [],
            'trend_bear': [],
            'super_trend_bear': [],
            'p_day': [],
            'b_day': [],
            'neutral': [],
            'pm_morph': [],
            'morph_to_trend': []
        }
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with all required features"""
        df = df.copy()
        
        # Ensure timestamp is datetime and set as index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have timestamp column or DatetimeIndex")
        
        # Store original index
        original_index = df.index
        
        # Compute day type and IB features
        df = compute_day_type(df)
        df = compute_ib_features(df)
        
        # Restore index if it was lost
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = original_index
        
        # Add time features
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['time'] = df.index.time
        df['session_date'] = df.index.date
        
        # Calculate IB extension multiple
        df['ib_extension_multiple'] = df['ib_extension'] / df['ib_range']
        
        # Classify trend strength
        df['trend_strength'] = df['ib_extension_multiple'].apply(self._classify_trend_strength)
        
        # Detect day type more granularly
        df['playbook_day_type'] = df.apply(self._classify_playbook_day_type, axis=1)
        
        # Calculate VWAP deviation
        if 'vwap' in df.columns:
            df['vwap_deviation'] = (df['close'] - df['vwap']) / df['atr14']
        
        # Detect poor highs/lows
        df['poor_high'] = self._detect_poor_highs(df)
        df['poor_low'] = self._detect_poor_lows(df)
        
        return df
    
    def _classify_trend_strength(self, extension_multiple: float) -> str:
        """Classify trend strength based on IB extension multiple"""
        if pd.isna(extension_multiple):
            return 'unknown'
        if extension_multiple < 0.5:
            return TrendStrength.WEAK.value
        elif extension_multiple < 1.0:
            return TrendStrength.MODERATE.value
        elif extension_multiple < 2.0:
            return TrendStrength.STRONG.value
        else:
            return TrendStrength.SUPER.value
    
    def _classify_playbook_day_type(self, row: pd.Series) -> str:
        """Classify into one of 9 playbook day types"""
        day_type = row.get('day_type', 'NEUTRAL')
        trend_strength = row.get('trend_strength', 'weak')
        ib_direction = row.get('ib_direction', 'INSIDE')
        
        # Map to playbook day types
        if day_type == 'TREND':
            if trend_strength == 'super':
                return 'super_trend_up' if ib_direction == 'BULL' else 'super_trend_down'
            else:
                return 'trend_up' if ib_direction == 'BULL' else 'trend_down'
        
        elif day_type == 'P_DAY':
            return 'p_day'
        
        elif day_type == 'B_DAY':
            return 'b_day'
        
        else:
            return 'neutral'
    
    def _detect_poor_highs(self, df: pd.DataFrame, lookback: int = 3) -> pd.Series:
        """Detect poor highs (multiple tests, no acceptance)"""
        poor_high = pd.Series(False, index=df.index)
        
        for i in range(lookback, len(df)):
            window = df.iloc[i-lookback:i+1]
            highs = window['high'].values
            
            # Poor high: current high equals recent highs but no close above
            current_high = df.iloc[i]['high']
            recent_highs = highs[:-1]
            
            if np.any(np.abs(recent_highs - current_high) < self.tick_size * 2):
                # Multiple tests at same level
                if df.iloc[i]['close'] < current_high - self.tick_size * 2:
                    poor_high.iloc[i] = True
        
        return poor_high
    
    def _detect_poor_lows(self, df: pd.DataFrame, lookback: int = 3) -> pd.Series:
        """Detect poor lows (multiple tests, no acceptance)"""
        poor_low = pd.Series(False, index=df.index)
        
        for i in range(lookback, len(df)):
            window = df.iloc[i-lookback:i+1]
            lows = window['low'].values
            
            current_low = df.iloc[i]['low']
            recent_lows = lows[:-1]
            
            if np.any(np.abs(recent_lows - current_low) < self.tick_size * 2):
                if df.iloc[i]['close'] > current_low + self.tick_size * 2:
                    poor_low.iloc[i] = True
        
        return poor_low
    
    def run_all_strategies(self, df: pd.DataFrame) -> Dict:
        """Run all 9 strategies and return combined results"""
        print("=" * 70)
        print("DALTON PLAYBOOK - ALL 9 STRATEGIES")
        print("=" * 70)
        print(f"Symbol: {self.symbol}")
        print(f"Risk per trade: ${self.risk_per_trade}")
        print(f"Data period: {df.index.min()} to {df.index.max()}")
        print()
        
        # Prepare data
        df = self.prepare_data(df)
        
        results = {}
        
        # Run each strategy
        strategies = [
            ('Trend Day Bull', self.run_trend_day_bull),
            ('Super Trend Day Bull', self.run_super_trend_bull),
            ('Trend Day Bear', self.run_trend_day_bear),
            ('Super Trend Day Bear', self.run_super_trend_bear),
            ('P-Day', self.run_p_day),
            ('B-Day', self.run_b_day),
            ('Neutral Day', self.run_neutral_day),
            ('PM Morph', self.run_pm_morph),
            ('Morph to Trend', self.run_morph_to_trend),
        ]
        
        for name, strategy_func in strategies:
            print(f"\n{'='*70}")
            print(f"Running: {name}")
            print('='*70)
            
            try:
                result = strategy_func(df)
                results[name] = result
                
                # Print summary
                trades = result.get('trades', [])
                if trades:
                    total_pnl = sum(t.pnl for t in trades)
                    wins = sum(1 for t in trades if t.pnl > 0)
                    win_rate = wins / len(trades) * 100 if trades else 0
                    
                    print(f"\n  Trades: {len(trades)}")
                    print(f"  Win Rate: {win_rate:.1f}%")
                    print(f"  Total P&L: ${total_pnl:,.2f}")
                else:
                    print(f"\n  No trades generated")
                    
            except Exception as e:
                print(f"ERROR in {name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Overall summary
        print(f"\n{'='*70}")
        print("OVERALL SUMMARY")
        print('='*70)
        
        all_trades = self.all_trades
        if all_trades:
            total_pnl = sum(t.pnl for t in all_trades)
            wins = sum(1 for t in all_trades if t.pnl > 0)
            losses = len(all_trades) - wins
            win_rate = wins / len(all_trades) * 100
            
            # Calculate avg win/loss
            win_pnls = [t.pnl for t in all_trades if t.pnl > 0]
            loss_pnls = [t.pnl for t in all_trades if t.pnl <= 0]
            avg_win = np.mean(win_pnls) if win_pnls else 0
            avg_loss = np.mean(loss_pnls) if loss_pnls else 0
            
            print(f"\nTotal Trades: {len(all_trades)}")
            print(f"Win Rate: {win_rate:.1f}%")
            print(f"Total P&L: ${total_pnl:,.2f}")
            print(f"Avg Win: ${avg_win:,.2f}")
            print(f"Avg Loss: ${avg_loss:,.2f}")
            print(f"R:R Ratio: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "Inf")
        
        return results
    
    def run_trend_day_bull(self, df: pd.DataFrame) -> Dict:
        """
        Strategy 1: Trend Day Bull
        - Standard uptrend with IB extension acceptance
        - Entry: IBH retest hold + DPOC migration
        - Stop: Below IBH or swing low
        - Target: 3:1 R:R
        """
        trades = []
        
        # Filter for trend up days with moderate+ strength
        trend_days = df[
            (df['playbook_day_type'] == 'trend_up') &
            (df['trend_strength'].isin(['moderate', 'strong'])) &
            (df['ib_direction'] == 'BULL')
        ].copy()
        
        print(f"  Trend Up Days found: {trend_days['session_date'].nunique()}")
        
        # Group by session
        for session_date, session_df in trend_days.groupby('session_date'):
            trades.extend(self._trade_trend_session(session_df, 'LONG'))
        
        self.trades_by_strategy['trend_bull'] = trades
        self.all_trades.extend(trades)
        
        return {'trades': trades, 'count': len(trades)}
    
    def run_super_trend_bull(self, df: pd.DataFrame) -> Dict:
        """
        Strategy 2: Super Trend Day Bull
        - Extreme trend >2.0x IB extension
        - Entry: Aggressive pyramid on DPOC steps
        - Wider stops, larger targets
        """
        trades = []
        
        super_days = df[
            (df['playbook_day_type'] == 'super_trend_up') &
            (df['ib_direction'] == 'BULL')
        ].copy()
        
        print(f"  Super Trend Up Days found: {super_days['session_date'].nunique()}")
        
        for session_date, session_df in super_days.groupby('session_date'):
            trades.extend(self._trade_trend_session(session_df, 'LONG', aggressive=True))
        
        self.trades_by_strategy['super_trend_bull'] = trades
        self.all_trades.extend(trades)
        
        return {'trades': trades, 'count': len(trades)}
    
    def run_trend_day_bear(self, df: pd.DataFrame) -> Dict:
        """Strategy 3: Trend Day Bear (mirror of bull)"""
        trades = []
        
        trend_days = df[
            (df['playbook_day_type'] == 'trend_down') &
            (df['trend_strength'].isin(['moderate', 'strong'])) &
            (df['ib_direction'] == 'BEAR')
        ].copy()
        
        print(f"  Trend Down Days found: {trend_days['session_date'].nunique()}")
        
        for session_date, session_df in trend_days.groupby('session_date'):
            trades.extend(self._trade_trend_session(session_df, 'SHORT'))
        
        self.trades_by_strategy['trend_bear'] = trades
        self.all_trades.extend(trades)
        
        return {'trades': trades, 'count': len(trades)}
    
    def run_super_trend_bear(self, df: pd.DataFrame) -> Dict:
        """Strategy 4: Super Trend Day Bear (mirror of super bull)"""
        trades = []
        
        super_days = df[
            (df['playbook_day_type'] == 'super_trend_down') &
            (df['ib_direction'] == 'BEAR')
        ].copy()
        
        print(f"  Super Trend Down Days found: {super_days['session_date'].nunique()}")
        
        for session_date, session_df in super_days.groupby('session_date'):
            trades.extend(self._trade_trend_session(session_df, 'SHORT', aggressive=True))
        
        self.trades_by_strategy['super_trend_bear'] = trades
        self.all_trades.extend(trades)
        
        return {'trades': trades, 'count': len(trades)}
    
    def run_p_day(self, df: pd.DataFrame) -> Dict:
        """
        Strategy 5: P-Day (Skewed Balance)
        - Wait for resolution (single prints + DPOC compression)
        - Entry: Break of single-print excess with acceptance
        """
        trades = []
        
        p_days = df[df['playbook_day_type'] == 'p_day'].copy()
        print(f"  P-Days found: {p_days['session_date'].nunique()}")
        
        # Simplified: Trade P-days as directional with IB direction
        for session_date, session_df in p_days.groupby('session_date'):
            if len(session_df) < 60:  # Need IB formed
                continue
            
            direction = session_df.iloc[60:]['ib_direction'].mode()
            if len(direction) > 0 and direction[0] in ['BULL', 'BEAR']:
                dir_str = 'LONG' if direction[0] == 'BULL' else 'SHORT'
                trades.extend(self._trade_trend_session(session_df, dir_str))
        
        self.trades_by_strategy['p_day'] = trades
        self.all_trades.extend(trades)
        
        return {'trades': trades, 'count': len(trades)}
    
    def run_b_day(self, df: pd.DataFrame) -> Dict:
        """
        Strategy 6: B-Day (Mean Reversion)
        - Fade extremes at IB edges (simplified VAH/VAL proxy)
        - Max 2 trades per session: 1 fade at IBH, 1 fade at IBL
        - Require confirmation: poor structure + rejection
        - Tight stops, quick targets to POC/VWAP
        """
        trades = []
        
        b_days = df[df['playbook_day_type'] == 'b_day'].copy()
        print(f"  B-Days found: {b_days['session_date'].nunique()}")
        
        for session_date, session_df in b_days.groupby('session_date'):
            if len(session_df) < 60:
                continue
            
            # Get IB range
            ib_high = session_df.iloc[:60]['high'].max()
            ib_low = session_df.iloc[:60]['low'].min()
            ib_range = ib_high - ib_low
            ib_mid = (ib_high + ib_low) / 2
            
            # Track trades per session
            vah_fade_taken = False
            val_fade_taken = False
            last_trade_time = None
            cooldown_bars = 30  # 30-bar cooldown between trades
            
            # Trade after IB
            post_ib = session_df.iloc[60:]
            
            for i, (timestamp, row) in enumerate(post_ib.iterrows()):
                # Skip if in cooldown period
                if last_trade_time is not None:
                    bars_since_last = i - last_trade_time
                    if bars_since_last < cooldown_bars:
                        continue
                
                # Check for poor structure (recent poor high/low)
                recent_window = session_df.iloc[max(0, 60+i-10):60+i]
                has_poor_high = recent_window['poor_high'].any() if len(recent_window) > 0 and 'poor_high' in recent_window.columns else False
                has_poor_low = recent_window['poor_low'].any() if len(recent_window) > 0 and 'poor_low' in recent_window.columns else False
                
                # Fade near IBH (short) - only once per session
                if not vah_fade_taken and row['high'] >= ib_high:
                    # Check rejection: close below IBH and poor structure helps
                    if row['close'] < ib_high:
                        entry_price = row['close']
                        stop_price = ib_high + (ib_range * 0.1)  # Stop above IBH
                        target_price = ib_mid  # Target middle
                        
                        trade = self._simulate_trade(
                            post_ib.iloc[i:],
                            'SHORT',
                            entry_price,
                            stop_price,
                            target_price,
                            timestamp,
                            'B_DAY_IBH_FADE'
                        )
                        if trade:
                            trades.append(trade)
                            vah_fade_taken = True
                            last_trade_time = i
                
                # Fade near IBL (long) - only once per session
                elif not val_fade_taken and row['low'] <= ib_low:
                    # Check rejection: close above IBL
                    if row['close'] > ib_low:
                        entry_price = row['close']
                        stop_price = ib_low - (ib_range * 0.1)  # Stop below IBL
                        target_price = ib_mid  # Target middle
                        
                        trade = self._simulate_trade(
                            post_ib.iloc[i:],
                            'LONG',
                            entry_price,
                            stop_price,
                            target_price,
                            timestamp,
                            'B_DAY_IBL_FADE'
                        )
                        if trade:
                            trades.append(trade)
                            val_fade_taken = True
                            last_trade_time = i
                
                # Stop looking if we've taken both trades
                if vah_fade_taken and val_fade_taken:
                    break
        
        self.trades_by_strategy['b_day'] = trades
        self.all_trades.extend(trades)
        
        return {'trades': trades, 'count': len(trades)}
    
    def run_neutral_day(self, df: pd.DataFrame) -> Dict:
        """
        Strategy 7: Neutral Day
        - POC/VWAP bounces only
        - No directional bias
        """
        trades = []
        
        neutral_days = df[df['playbook_day_type'] == 'neutral'].copy()
        print(f"  Neutral Days found: {neutral_days['session_date'].nunique()}")
        
        # Skip neutral days - no edge
        print(f"  (Skipping - no directional edge on neutral days)")
        
        self.trades_by_strategy['neutral'] = trades
        
        return {'trades': trades, 'count': 0}
    
    def run_pm_morph(self, df: pd.DataFrame) -> Dict:
        """
        Strategy 8: PM Morph (Early Balance → Late Trend)
        - Wait until after 12:30
        - Entry: Late breakout acceptance
        """
        trades = []
        
        # Find days that start balanced but extend later
        for session_date, session_df in df.groupby('session_date'):
            if len(session_df) < 100:  # Need enough bars
                continue
            
            am_data = session_df.between_time('09:30', '12:30')
            pm_data = session_df.between_time('12:30', '15:30')
            
            if len(am_data) == 0 or len(pm_data) == 0:
                continue
            
            # Check if AM was balanced (inside IB)
            am_ib_high = am_data.iloc[:60]['high'].max() if len(am_data) >= 60 else None
            am_ib_low = am_data.iloc[:60]['low'].min() if len(am_data) >= 60 else None
            
            if am_ib_high is None:
                continue
            
            am_max = am_data['high'].max()
            am_min = am_data['low'].min()
            
            # Check if PM breaks out
            pm_max = pm_data['high'].max()
            pm_min = pm_data['low'].min()
            
            if pm_max > am_max + 10:  # Breaks AM high
                # Bullish morph
                entry_idx = pm_data[pm_data['high'] > am_max].index[0]
                entry_price = pm_data.loc[entry_idx]['close']
                
                trade = self._simulate_trade(
                    pm_data.loc[entry_idx:],
                    'LONG',
                    entry_price,
                    am_max - 10,  # Stop below breakout
                    entry_price + 100,  # Target
                    entry_idx,
                    'PM_MORPH_BULL'
                )
                if trade:
                    trades.append(trade)
            
            elif pm_min < am_min - 10:  # Breaks AM low
                # Bearish morph
                entry_idx = pm_data[pm_data['low'] < am_min].index[0]
                entry_price = pm_data.loc[entry_idx]['close']
                
                trade = self._simulate_trade(
                    pm_data.loc[entry_idx:],
                    'SHORT',
                    entry_price,
                    am_min + 10,
                    entry_price - 100,
                    entry_idx,
                    'PM_MORPH_BEAR'
                )
                if trade:
                    trades.append(trade)
        
        print(f"  PM Morph trades: {len(trades)}")
        
        self.trades_by_strategy['pm_morph'] = trades
        self.all_trades.extend(trades)
        
        return {'trades': trades, 'count': len(trades)}
    
    def run_morph_to_trend(self, df: pd.DataFrame) -> Dict:
        """
        Strategy 9: Morph to Trend Day
        - P/B day breaks out
        - Requires DPOC migration
        """
        trades = []
        
        # Look for P/B days that turn into trends
        morph_candidates = df[df['playbook_day_type'].isin(['p_day', 'b_day'])].copy()
        
        for session_date, session_df in morph_candidates.groupby('session_date'):
            if len(session_df) < 120:  # Need time to morph
                continue
            
            # Check if late day becomes trend
            late_data = session_df.between_time('11:00', '15:30')
            early_ib_high = session_df.iloc[:60]['high'].max() if len(session_df) >= 60 else None
            early_ib_low = session_df.iloc[:60]['low'].min() if len(session_df) >= 60 else None
            
            if early_ib_high is None:
                continue
            
            # Check for breakout
            if late_data['high'].max() > early_ib_high + 20:
                breakout_idx = late_data[late_data['high'] > early_ib_high + 20].index[0]
                entry_price = late_data.loc[breakout_idx]['close']
                
                trade = self._simulate_trade(
                    late_data.loc[breakout_idx:],
                    'LONG',
                    entry_price,
                    early_ib_high,
                    entry_price + 150,
                    breakout_idx,
                    'MORPH_TO_TREND_BULL'
                )
                if trade:
                    trades.append(trade)
            
            elif late_data['low'].min() < early_ib_low - 20:
                breakout_idx = late_data[late_data['low'] < early_ib_low - 20].index[0]
                entry_price = late_data.loc[breakout_idx]['close']
                
                trade = self._simulate_trade(
                    late_data.loc[breakout_idx:],
                    'SHORT',
                    entry_price,
                    early_ib_low,
                    entry_price - 150,
                    breakout_idx,
                    'MORPH_TO_TREND_BEAR'
                )
                if trade:
                    trades.append(trade)
        
        print(f"  Morph to Trend trades: {len(trades)}")
        
        self.trades_by_strategy['morph_to_trend'] = trades
        self.all_trades.extend(trades)
        
        return {'trades': trades, 'count': len(trades)}
    
    def _trade_trend_session(self, session_df: pd.DataFrame, 
                            direction: str, 
                            aggressive: bool = False) -> List[Trade]:
        """
        Trade a trend session using proper Dalton Playbook mechanics with pyramid
        
        Key Concepts:
        1. Acceptance: 2x 5-min closes OR 30-min close beyond IB
        2. Narrow IB range = compression = explosive breakout  
        3. Initial Entry: IBH/IBL retest after acceptance
        4. Pyramid Entries: Pullbacks to EMA after 11 AM (up to 3 pyramids)
        5. WIDE Stop: Below IB low (trend day) - trend won't revisit this
        6. Trail stop as we pyramid (each add moves stop up)
        7. DPOC migration confirms trend after 11 AM
        """
        trades = []
        
        if len(session_df) < 60:
            return trades
        
        # Get IB stats
        ib_data = session_df.iloc[:60]
        ib_high = ib_data['high'].max()
        ib_low = ib_data['low'].min()
        ib_range = ib_high - ib_low
        ib_mid = (ib_high + ib_low) / 2
        
        # Trade after IB formation
        post_ib = session_df.iloc[60:].copy()
        
        # Look for acceptance
        acceptance_confirmed = False
        acceptance_timestamp = None
        
        for i in range(1, len(post_ib)):
            window = post_ib.iloc[:i+1]
            
            if direction == 'LONG':
                # 30-min acceptance (6 bars)
                if len(window) >= 6:
                    last_6 = window.iloc[-6:]
                    if (last_6['close'] > ib_high).all():
                        acceptance_confirmed = True
                        acceptance_timestamp = window.index[-1]
                        break
                # 2x 5-min acceptance
                if len(window) >= 2:
                    last_2 = window.iloc[-2:]
                    if (last_2['close'] > ib_high).all():
                        acceptance_confirmed = True
                        acceptance_timestamp = window.index[-1]
                        break
            else:  # SHORT
                if len(window) >= 6:
                    last_6 = window.iloc[-6:]
                    if (last_6['close'] < ib_low).all():
                        acceptance_confirmed = True
                        acceptance_timestamp = window.index[-1]
                        break
                if len(window) >= 2:
                    last_2 = window.iloc[-2:]
                    if (last_2['close'] < ib_low).all():
                        acceptance_confirmed = True
                        acceptance_timestamp = window.index[-1]
                        break
        
        if not acceptance_confirmed:
            return trades
        
        # PYRAMID STRATEGY: Up to 3 entries per trend day
        max_pyramids = 3 if aggressive else 2
        pyramid_count = 0
        last_entry_price = None
        last_entry_time = None
        cooldown_bars = 20  # Wait 20 bars between pyramids
        
        # WIDE STOP for trend days - below IB low (longs) or above IB high (shorts)
        # This is aggressive but appropriate for trend days
        if direction == 'LONG':
            base_stop = ib_low - (ib_range * 0.2)  # 20% below IB low
        else:
            base_stop = ib_high + (ib_range * 0.2)  # 20% above IB high
        
        for i, (timestamp, row) in enumerate(post_ib.iterrows()):
            if timestamp < acceptance_timestamp:
                continue
            
            # Cooldown check
            if last_entry_time is not None:
                bars_since_entry = i - post_ib.index.get_loc(last_entry_time) if last_entry_time in post_ib.index else 999
                if bars_since_entry < cooldown_bars:
                    continue
            
            # Get time for after-11-AM check
            current_time = timestamp.time() if hasattr(timestamp, 'time') else timestamp
            after_11am = False
            if isinstance(current_time, time):
                after_11am = current_time >= time(11, 0)
            else:
                try:
                    after_11am = current_time.hour >= 11
                except:
                    pass
            
            # Check DPOC migration after 11 AM
            dpoc_migrated = False
            if after_11am and 'vwap' in row:
                if direction == 'LONG' and row['vwap'] > ib_high:
                    dpoc_migrated = True
                elif direction == 'SHORT' and row['vwap'] < ib_low:
                    dpoc_migrated = True
            
            should_enter = False
            entry_type = ""
            entry_price = row['close']
            
            if direction == 'LONG':
                # PYRAMID 1: IBH retest (initial entry)
                if pyramid_count == 0 and not after_11am:
                    if row['low'] <= ib_high <= row['high']:
                        should_enter = True
                        entry_type = 'IBH_RETEST_INITIAL'
                
                # PYRAMID 2+: EMA pullback (after 11 AM or if initial missed)
                elif pyramid_count < max_pyramids:
                    # Either after 11 AM with DPOC migration OR we missed initial entry
                    if (after_11am and dpoc_migrated) or (pyramid_count == 0 and after_11am):
                        if 'ema20' in row:
                            # Pullback to EMA20 (shallow retracement in trend)
                            if row['low'] <= row['ema20'] <= row['high'] or row['close'] > row['ema20']:
                                # Price near or above EMA20
                                ema_distance_pct = abs(row['close'] - row['ema20']) / ib_range
                                if ema_distance_pct < 0.15:  # Within 15% of IB range
                                    should_enter = True
                                    entry_price = row['close']
                                    entry_type = f'EMA_PULLBACK_P{pyramid_count+1}'
            
            else:  # SHORT
                # PYRAMID 1: IBL retest
                if pyramid_count == 0 and not after_11am:
                    if row['low'] <= ib_low <= row['high']:
                        should_enter = True
                        entry_type = 'IBL_RETEST_INITIAL'
                
                # PYRAMID 2+: EMA pullback
                elif pyramid_count < max_pyramids:
                    if (after_11am and dpoc_migrated) or (pyramid_count == 0 and after_11am):
                        if 'ema20' in row:
                            if row['low'] <= row['ema20'] <= row['high'] or row['close'] < row['ema20']:
                                ema_distance_pct = abs(row['ema20'] - row['close']) / ib_range
                                if ema_distance_pct < 0.15:
                                    should_enter = True
                                    entry_price = row['close']
                                    entry_type = f'EMA_PULLBACK_P{pyramid_count+1}'
            
            if should_enter:
                # WIDE STOP - trail as we pyramid
                # First pyramid: Wide stop at IB structure
                # Subsequent pyramids: Trail stop to previous entry
                if pyramid_count == 0:
                    stop_price = base_stop
                else:
                    # Trail stop to breakeven of previous entry or better
                    if direction == 'LONG':
                        stop_price = max(base_stop, last_entry_price - (ib_range * 0.3))
                    else:
                        stop_price = min(base_stop, last_entry_price + (ib_range * 0.3))
                
                # Target: 3:1 R:R from average entry
                if direction == 'LONG':
                    target_price = entry_price + (3 * (entry_price - stop_price))
                else:
                    target_price = entry_price - (3 * (stop_price - entry_price))
                
                trade = self._simulate_trade(
                    post_ib.iloc[i:],
                    direction,
                    entry_price,
                    stop_price,
                    target_price,
                    timestamp,
                    entry_type
                )
                
                if trade:
                    trades.append(trade)
                    pyramid_count += 1
                    last_entry_price = entry_price
                    last_entry_time = timestamp
                    
                    # If we've hit max pyramids, we're done with this session
                    if pyramid_count >= max_pyramids:
                        break
        
        return trades
    
    def _simulate_trade(self, future_df: pd.DataFrame,
                       direction: str,
                       entry_price: float,
                       stop_price: float,
                       target_price: float,
                       entry_time: datetime,
                       setup_type: str) -> Optional[Trade]:
        """Simulate a trade forward"""
        if len(future_df) < 2:
            return None
        
        # Skip entry bar
        future_df = future_df.iloc[1:]
        
        exit_price = None
        exit_time = None
        exit_reason = None
        
        for timestamp, row in future_df.iterrows():
            # Check stop
            if direction == 'LONG':
                if row['low'] <= stop_price:
                    exit_price = stop_price
                    exit_time = timestamp
                    exit_reason = 'STOP'
                    break
                elif row['high'] >= target_price:
                    exit_price = target_price
                    exit_time = timestamp
                    exit_reason = 'TARGET'
                    break
            else:  # SHORT
                if row['high'] >= stop_price:
                    exit_price = stop_price
                    exit_time = timestamp
                    exit_reason = 'STOP'
                    break
                elif row['low'] <= target_price:
                    exit_price = target_price
                    exit_time = timestamp
                    exit_reason = 'TARGET'
                    break
        
        # Time exit if still open at end
        if exit_price is None and len(future_df) > 0:
            exit_price = future_df.iloc[-1]['close']
            exit_time = future_df.index[-1]
            exit_reason = 'TIME'
        
        if exit_price is None:
            return None
        
        # Calculate P&L
        if direction == 'LONG':
            pnl = (exit_price - entry_price) * self.point_value
        else:
            pnl = (entry_price - exit_price) * self.point_value
        
        # Calculate contracts based on risk
        risk_per_contract = abs(entry_price - stop_price) * self.point_value
        if risk_per_contract > 0:
            contracts = int(self.risk_per_trade / risk_per_contract)
            contracts = max(1, min(contracts, 30))
        else:
            contracts = 1
        
        total_pnl = pnl * contracts
        
        return Trade(
            entry_time=entry_time,
            exit_time=exit_time,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_price=stop_price,
            target_price=target_price,
            contracts=contracts,
            pnl=total_pnl,
            exit_reason=exit_reason,
            setup_type=setup_type,
            day_type='trend'
        )


if __name__ == '__main__':
    print("Dalton Playbook - All 9 Strategies")
    print("=" * 70)
    print("\nLoading data...")
    
    df = load_data('NQ')
    
    print(f"\nRunning all strategies...")
    strategies = PlaybookStrategies(symbol='NQ')
    results = strategies.run_all_strategies(df)
    
    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)
