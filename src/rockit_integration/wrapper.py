"""
Rockit Framework Integration
============================
Wrapper to use existing rockit-framework for backtesting

This module integrates with the user's existing rockit-framework
to calculate TPO, IB, and all Dalton metrics for backtesting.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
import json

# Add rockit-framework to path
sys.path.insert(0, '/home/lphan/jupyterlab/BookMapOrderFlowStudies/rockit-framework')

from orchestrator import generate_snapshot, clean_for_json
from modules.loader import load_nq_csv
from modules.ib_location import get_ib_location
from modules.tpo_profile import get_tpo_profile
from modules.volume_profile import get_volume_profile
from modules.dpoc_migration import get_dpoc_migration
from modules.wick_parade import get_wick_parade
from modules.core_confluences import get_core_confluences


class RockitBacktestWrapper:
    """
    Wrapper to use rockit-framework for historical backtesting
    
    Iterates through each bar in historical data, generates
    rockit snapshot, and extracts signals for strategy testing.
    """
    
    def __init__(self, symbol: str = 'NQ'):
        self.symbol = symbol
        self.current_index = 0
        self.snapshots = []
        
    def prepare_data_for_rockit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare dataframe to match rockit-framework expected format
        """
        df = df.copy()
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            else:
                raise ValueError("Dataframe needs timestamp column or DatetimeIndex")
        
        # Add required columns if not present
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Ensure all indicator columns exist
        indicator_cols = ['ema20', 'ema50', 'ema200', 'rsi14', 'atr14', 'vwap']
        for col in indicator_cols:
            if col not in df.columns:
                print(f"Warning: {col} not in data, rockit may fail")
        
        return df
    
    def generate_snapshot_at_time(self, 
                                  df: pd.DataFrame, 
                                  current_time: datetime) -> Optional[Dict]:
        """
        Generate rockit snapshot for a specific point in time
        
        Args:
            df: Full historical dataframe
            current_time: Current timestamp (no lookahead)
        
        Returns:
            Snapshot dict with all rockit metrics
        """
        # Filter data up to current time (no lookahead!)
        df_up_to_now = df[df.index <= current_time].copy()
        
        if len(df_up_to_now) < 12:  # Need at least IB
            return None
        
        # Get session date
        session_date = current_time.strftime('%Y-%m-%d')
        current_time_str = current_time.strftime('%H:%M')
        
        # Build config for rockit
        config = {
            'csv_paths': {'nq': None},  # We'll pass df directly
            'session_date': session_date,
            'current_time': current_time_str,
            'output_dir': '/tmp/rockit_backtest'
        }
        
        try:
            # Manually call modules (since we have df already)
            ib_data = get_ib_location(df_up_to_now, current_time_str)
            volume_profile = get_volume_profile(df_up_to_now, df_up_to_now, current_time_str)
            prior_day = volume_profile.get('previous_day', {})
            
            tpo_profile = get_tpo_profile(df_up_to_now, current_time_str, prior_day=prior_day)
            dpoc_migration = get_dpoc_migration(df_up_to_now, current_time_str)
            wick_parade = get_wick_parade(df_up_to_now, current_time_str)
            
            # Build intraday data
            intraday_data = {
                'ib': ib_data,
                'volume_profile': volume_profile,
                'tpo_profile': tpo_profile,
                'dpoc_migration': dpoc_migration,
                'wick_parade': wick_parade
            }
            
            # Get core confluences (the signals!)
            core_confluences = get_core_confluences(intraday_data, current_time_str)
            
            # Build snapshot
            snapshot = {
                'session_date': session_date,
                'current_et_time': current_time_str,
                'ib': ib_data,
                'tpo_profile': tpo_profile,
                'volume_profile': volume_profile,
                'dpoc_migration': dpoc_migration,
                'core_confluences': core_confluences,
                'current_price': ib_data.get('current_close'),
                'day_type': self._classify_day_type(ib_data, tpo_profile, core_confluences)
            }
            
            return clean_for_json(snapshot)
            
        except Exception as e:
            print(f"Error generating snapshot at {current_time}: {e}")
            return None
    
    def _classify_day_type(self, ib_data: Dict, tpo_profile: Dict, confluences: Dict) -> str:
        """
        Classify day type based on rockit data
        
        Returns: 'trend_up', 'trend_down', 'p_day_up', 'p_day_down', 'b_day', 'neutral'
        """
        # Get IB extension
        ib_high = ib_data.get('ib_high')
        ib_low = ib_data.get('ib_low')
        current_price = ib_data.get('current_close')
        
        if not all([ib_high, ib_low, current_price]):
            return 'unknown'
        
        ib_range = ib_high - ib_low
        
        # Check price location
        price_vs_ib = ib_data.get('price_vs_ib', 'middle')
        
        # Get DPOC
        dpoc = tpo_profile.get('current_poc')
        vah = tpo_profile.get('current_vah') or tpo_profile.get('vah')
        val = tpo_profile.get('current_val') or tpo_profile.get('val')
        
        # Get confluences
        ib_acceptance = confluences.get('ib_acceptance', {})
        migration = confluences.get('migration', {})
        
        # Calculate extension
        if current_price > ib_high:
            extension = (current_price - ib_high) / ib_range if ib_range > 0 else 0
        elif current_price < ib_low:
            extension = (ib_low - current_price) / ib_range if ib_range > 0 else 0
        else:
            extension = 0
        
        # Classify
        if extension > 1.5:
            # Trend day
            if current_price > ib_high:
                return 'trend_up'
            else:
                return 'trend_down'
        
        elif extension > 0.5:
            # P-day (skewed)
            if dpoc and dpoc > (ib_high + ib_low) / 2:
                return 'p_day_up'
            else:
                return 'p_day_down'
        
        elif extension < 0.3:
            # Balanced
            return 'b_day'
        
        else:
            return 'neutral'
    
    def iterate_snapshots(self, df: pd.DataFrame, interval_minutes: int = 5):
        """
        Iterate through historical data and generate snapshots
        
        Args:
            df: Historical dataframe
            interval_minutes: How often to generate snapshots
        
        Yields:
            (timestamp, snapshot) tuples
        """
        df = self.prepare_data_for_rockit(df)
        
        # Get unique sessions
        df['date'] = df.index.date
        sessions = df['date'].unique()
        
        print(f"Processing {len(sessions)} sessions...")
        
        for session_date in sessions:
            session_df = df[df['date'] == session_date]
            
            # Generate times every interval_minutes from 10:00 to 15:30
            start_time = datetime.combine(session_date, time(10, 0))
            end_time = datetime.combine(session_date, time(15, 30))
            
            current_time = start_time
            while current_time <= end_time:
                if current_time in session_df.index:
                    snapshot = self.generate_snapshot_at_time(df, current_time)
                    if snapshot:
                        yield current_time, snapshot
                
                current_time += timedelta(minutes=interval_minutes)
    
    def get_entry_signals(self, snapshot: Dict) -> List[Dict]:
        """
        Extract actionable entry signals from snapshot
        
        Returns list of signals with:
        - direction: 'LONG' or 'SHORT'
        - setup: signal type
        - confidence: signal strength
        - stop_price: suggested stop
        - target_price: suggested target
        """
        signals = []
        
        day_type = snapshot.get('day_type', 'unknown')
        confluences = snapshot.get('core_confluences', {})
        ib_data = snapshot.get('ib', {})
        tpo = snapshot.get('tpo_profile', {})
        
        current_price = ib_data.get('current_close')
        if not current_price:
            return signals
        
        # Extract confluence data
        ib_acc = confluences.get('ib_acceptance', {})
        dpoc_vs_ib = confluences.get('dpoc_vs_ib', {})
        compression = confluences.get('dpoc_compression', {})
        
        # === TREND DAY SIGNALS ===
        if day_type in ['trend_up', 'trend_down']:
            
            if day_type == 'trend_up':
                # Long signals
                if ib_acc.get('close_above_ibh') or ib_acc.get('price_accepted_higher') == 'Yes':
                    signals.append({
                        'direction': 'LONG',
                        'setup': 'IB_EXTENSION_ACCEPTANCE',
                        'confidence': 'high' if ib_acc.get('close_above_ibh') else 'medium',
                        'stop_price': ib_data.get('ib_low'),
                        'target_price': current_price + (2 * (current_price - ib_data.get('ib_low', current_price))),
                        'day_type': day_type
                    })
                
                if dpoc_vs_ib.get('dpoc_above_ibh'):
                    signals.append({
                        'direction': 'LONG',
                        'setup': 'DPOC_MIGRATION_CONFIRMATION',
                        'confidence': 'high',
                        'stop_price': ib_data.get('ib_high'),
                        'target_price': current_price + (2 * (current_price - ib_data.get('ib_high', current_price))),
                        'day_type': day_type
                    })
            
            else:  # trend_down
                # Short signals (mirror)
                if ib_acc.get('close_below_ibl') or ib_acc.get('price_accepted_lower') == 'Yes':
                    signals.append({
                        'direction': 'SHORT',
                        'setup': 'IB_EXTENSION_ACCEPTANCE',
                        'confidence': 'high',
                        'stop_price': ib_data.get('ib_high'),
                        'target_price': current_price - (2 * (ib_data.get('ib_high', current_price) - current_price)),
                        'day_type': day_type
                    })
        
        # === B-DAY SIGNALS (Mean Reversion) ===
        elif day_type == 'b_day':
            vah = tpo.get('current_vah') or tpo.get('vah')
            val = tpo.get('current_val') or tpo.get('val')
            
            if vah and current_price >= vah:
                # Fade VAH
                signals.append({
                    'direction': 'SHORT',
                    'setup': 'B_DAY_VAH_FADE',
                    'confidence': 'medium',
                    'stop_price': current_price + 5,  # Tight stop
                    'target_price': tpo.get('current_poc', current_price - 10),
                    'day_type': day_type
                })
            
            if val and current_price <= val:
                # Fade VAL
                signals.append({
                    'direction': 'LONG',
                    'setup': 'B_DAY_VAL_FADE',
                    'confidence': 'medium',
                    'stop_price': current_price - 5,
                    'target_price': tpo.get('current_poc', current_price + 10),
                    'day_type': day_type
                })
        
        # === P-DAY SIGNALS ===
        elif day_type in ['p_day_up', 'p_day_down']:
            dpoc = tpo.get('current_poc')
            
            if day_type == 'p_day_up' and dpoc and current_price < dpoc:
                # Pullback to DPOC in up skew
                signals.append({
                    'direction': 'LONG',
                    'setup': 'P_DAY_DPOC_PULLBACK',
                    'confidence': 'medium',
                    'stop_price': tpo.get('current_val', current_price - 5),
                    'target_price': tpo.get('current_vah', current_price + 10),
                    'day_type': day_type
                })
        
        return signals


if __name__ == '__main__':
    print("Rockit Framework Integration Wrapper")
    print("=" * 50)
    print("\nThis wrapper integrates with rockit-framework to:")
    print("  1. Calculate TPO/IB/Value Area for historical data")
    print("  2. Classify day types (Trend, P-Day, B-Day)")
    print("  3. Extract mechanical entry signals")
    print("  4. Run backtests on each strategy separately")
    print("\nUsage:")
    print("  wrapper = RockitBacktestWrapper()")
    print("  for timestamp, snapshot in wrapper.iterate_snapshots(df):")
    print("      signals = wrapper.get_entry_signals(snapshot)")
    print("      # Execute trades based on signals")
