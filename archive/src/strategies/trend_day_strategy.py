"""
Playbook Strategy: Trend Day
============================

Strategy for Trend Days (Up or Down)
Based on Dalton playbook mechanical rules.

Entry Signals:
1. IB Extension Acceptance (breakout above IBH/below IBL)
2. DPOC Migration Confirmation (POC follows breakout)
3. Value Area Breakout (break above VAH/below VAL)

Filters:
- Trend strength: Strong/Super only (skip Weak, half size Moderate)
- Time: 10:00-15:30 ET
- IB acceptance: Require 2 closes beyond IB
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sys
sys.path.append('/home/lphan/jupyterlab/BookMapOrderFlowStudies')

from src.rockit_integration.wrapper import RockitBacktestWrapper


@dataclass
class TrendDayTrade:
    """Trade record for Trend Day strategy"""
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
    setup_type: str  # 'IB_EXTENSION', 'DPOC_MIGRATION', 'VA_BREAK'


class TrendDayStrategy:
    """
    Mechanical Trend Day Strategy
    
    Rules:
    1. Day type must be 'trend_up' or 'trend_down'
    2. Entry on IB extension acceptance OR DPOC migration
    3. Stop: Below IB low (longs) / Above IB high (shorts)
    4. Target: 3:1 R:R (trend following)
    5. Time exit: 15:30 if still open
    """
    
    def __init__(self,
                 symbol: str = 'MNQ',
                 risk_per_trade: float = 400.0,
                 point_value: float = 2.0):
        self.symbol = symbol
        self.risk_per_trade = risk_per_trade
        self.point_value = point_value
        self.trades: List[TrendDayTrade] = []
        self.wrapper = RockitBacktestWrapper(symbol)
        
    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """
        Run backtest on Trend Days only
        
        Returns performance metrics
        """
        print(f"Running Trend Day Strategy backtest...")
        print(f"Symbol: {self.symbol}")
        print(f"Risk per trade: ${self.risk_per_trade}")
        print()
        
        trend_day_count = 0
        signal_count = 0
        
        for timestamp, snapshot in self.wrapper.iterate_snapshots(df):
            day_type = snapshot.get('day_type', '')
            
            # Only trade Trend Days
            if day_type not in ['trend_up', 'trend_down']:
                continue
            
            trend_day_count += 1
            
            # Get signals
            signals = self.wrapper.get_entry_signals(snapshot)
            
            for signal in signals:
                if 'IB_EXTENSION' in signal['setup'] or 'DPOC_MIGRATION' in signal['setup']:
                    signal_count += 1
                    
                    # Calculate position size
                    stop_distance = abs(signal['entry_price'] - signal['stop_price'])
                    if stop_distance > 0:
                        contracts = int(self.risk_per_trade / (stop_distance * self.point_value))
                        contracts = max(0, min(contracts, 30))
                        
                        print(f"  {timestamp}: {signal['direction']} {signal['setup']} "
                              f"@ {signal['entry_price']:.2f}, "
                              f"stop {signal['stop_price']:.2f}, "
                              f"{contracts} contracts")
        
        print(f"\nTrend Days found: {trend_day_count}")
        print(f"Signals generated: {signal_count}")
        
        return {
            'strategy': 'Trend Day',
            'trend_days_found': trend_day_count,
            'signals_generated': signal_count,
            'trades': self.trades
        }


if __name__ == '__main__':
    print("Trend Day Strategy")
    print("=" * 50)
    print("\nStrategy Rules:")
    print("  - Trade only on Trend Days (up/down)")
    print("  - Entry: IB extension acceptance OR DPOC migration")
    print("  - Stop: IB low/high")
    print("  - Target: 3:1 R:R")
    print("  - Skip Weak trends")
    print("\nUsage:")
    print("  strategy = TrendDayStrategy(symbol='MNQ')")
    print("  results = strategy.run_backtest(df)")
