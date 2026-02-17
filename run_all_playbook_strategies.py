#!/usr/bin/env python3
"""
Run All 9 Dalton Playbook Strategies
=====================================

Simple runner that executes all 9 strategies from the playbook
without relying on the buggy rockit framework.
"""

import sys
sys.path.append('/home/lphan/jupyterlab/BookMapOrderFlowStudies')

from src.strategies.playbook_strategies import PlaybookStrategies
from data_loader import load_data


def main():
    print("=" * 70)
    print("DALTON PLAYBOOK - ALL 9 STRATEGIES BACKTEST")
    print("=" * 70)
    print()
    
    # Load data
    print("Loading NQ data...")
    df = load_data('NQ')
    print()
    
    # Initialize strategies
    strategies = PlaybookStrategies(
        symbol='NQ',
        risk_per_trade=400.0,
        point_value=2.0
    )
    
    # Run all strategies
    results = strategies.run_all_strategies(df)
    
    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)
    
    # Summary by strategy
    print("\nStrategy Performance Summary:")
    print("-" * 70)
    
    for strategy_name, result in results.items():
        trades = result.get('trades', [])
        count = len(trades)
        
        if count > 0:
            total_pnl = sum(t.pnl for t in trades)
            wins = sum(1 for t in trades if t.pnl > 0)
            win_rate = wins / count * 100
            
            print(f"\n{strategy_name}:")
            print(f"  Trades: {count}")
            print(f"  Win Rate: {win_rate:.1f}%")
            print(f"  Total P&L: ${total_pnl:,.2f}")
        else:
            print(f"\n{strategy_name}: No trades")


if __name__ == '__main__':
    main()
