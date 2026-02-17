"""
Playbook Strategy: B-Day (Balanced Day)
========================================

Mean Reversion Strategy for Balanced Days
Based on Dalton playbook mechanical rules.

Entry Signals:
1. Fade VAH with poor high (short)
2. Fade VAL with poor low (long)
3. Double distribution fade (extremes)

Filters:
- Day type must be 'b_day' (balanced)
- ADX < 25 (no strong trend)
- Price extends >1 ATR from VWAP
- Poor high/low structure present
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import sys
sys.path.append('/home/lphan/jupyterlab/BookMapOrderFlowStudies')

from src.rockit_integration.wrapper import RockitBacktestWrapper


@dataclass
class BDayTrade:
    """Trade record for B-Day strategy"""
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
    setup_type: str  # 'VAH_FADE', 'VAL_FADE'


class BDayStrategy:
    """
    Mechanical B-Day (Mean Reversion) Strategy
    
    Rules:
    1. Day type must be 'b_day'
    2. Entry: Fade VAH/VAL with poor structure
    3. Stop: 2-5 points (tight)
    4. Target: VWAP or middle of range (1.5:1 R:R)
    5. Time exit: 5 bars if no profit
    """
    
    def __init__(self,
                 symbol: str = 'MNQ',
                 risk_per_trade: float = 400.0,
                 point_value: float = 2.0):
        self.symbol = symbol
        self.risk_per_trade = risk_per_trade
        self.point_value = point_value
        self.trades: List[BDayTrade] = []
        self.wrapper = RockitBacktestWrapper(symbol)
        
    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """
        Run backtest on B-Days only
        
        Returns performance metrics
        """
        print(f"Running B-Day Strategy backtest...")
        print(f"Symbol: {self.symbol}")
        print(f"Risk per trade: ${self.risk_per_trade}")
        print()
        
        b_day_count = 0
        signal_count = 0
        
        for timestamp, snapshot in self.wrapper.iterate_snapshots(df):
            day_type = snapshot.get('day_type', '')
            
            # Only trade B-Days
            if day_type != 'b_day':
                continue
            
            b_day_count += 1
            
            # Get signals
            signals = self.wrapper.get_entry_signals(snapshot)
            
            for signal in signals:
                if 'B_DAY' in signal['setup']:
                    signal_count += 1
                    
                    # Calculate position size (tight stops = larger size)
                    stop_distance = abs(signal['entry_price'] - signal['stop_price'])
                    if stop_distance > 0:
                        contracts = int(self.risk_per_trade / (stop_distance * self.point_value))
                        contracts = max(0, min(contracts, 30))
                        
                        print(f"  {timestamp}: {signal['direction']} {signal['setup']} "
                              f"@ {signal['entry_price']:.2f}, "
                              f"stop {signal['stop_price']:.2f}, "
                              f"{contracts} contracts")
        
        print(f"\nB-Days found: {b_day_count}")
        print(f"Signals generated: {signal_count}")
        
        return {
            'strategy': 'B-Day (Mean Reversion)',
            'b_days_found': b_day_count,
            'signals_generated': signal_count,
            'trades': self.trades
        }


if __name__ == '__main__':
    print("B-Day Strategy (Mean Reversion)")
    print("=" * 50)
    print("\nStrategy Rules:")
    print("  - Trade only on B-Days (balanced)")
    print("  - Entry: Fade VAH with poor high OR fade VAL with poor low")
    print("  - Stop: 2-5 points (tight)")
    print("  - Target: VWAP (1.5:1 R:R)")
    print("  - Counter-trend, quick exits")
    print("\nUsage:")
    print("  strategy = BDayStrategy(symbol='MNQ')")
    print("  results = strategy.run_backtest(df)")
