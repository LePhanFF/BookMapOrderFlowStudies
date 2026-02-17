"""
Playbook Strategies Backtest Runner
===================================

Runs each playbook strategy independently on historical data:
1. Trend Day Strategy
2. B-Day (Mean Reversion) Strategy  
3. P-Day Strategy

Uses rockit-framework to calculate day types and signals.
"""

import pandas as pd
import numpy as np
import sys
from datetime import datetime
import json

sys.path.append('/home/lphan/jupyterlab/BookMapOrderFlowStudies')

from src.strategies.trend_day_strategy import TrendDayStrategy
from src.strategies.b_day_strategy import BDayStrategy
from src.data.yahoo_loader import get_futures_data


def load_or_download_data(symbol: str = 'MNQ=F', 
                          period: str = "90d",
                          use_existing: bool = True) -> pd.DataFrame:
    """
    Load data from CSV or download from Yahoo Finance
    """
    csv_path = f'csv/{symbol.replace("=", "")}_volumetric.csv'
    
    if use_existing:
        try:
            print(f"Loading existing data from {csv_path}...")
            df = pd.read_csv(csv_path)
            print(f"  Loaded {len(df)} rows")
            return df
        except FileNotFoundError:
            print(f"  CSV not found, downloading from Yahoo...")
    
    # Download from Yahoo
    print(f"Downloading {symbol} data from Yahoo Finance...")
    df = get_futures_data(symbol, period=period, interval="5m")
    
    if not df.empty:
        # Save for future use
        df.to_csv(csv_path, index=False)
        print(f"  Saved to {csv_path}")
    
    return df


def run_all_strategies(symbol: str = 'MNQ'):
    """
    Run all playbook strategies separately and compare results
    """
    print("=" * 70)
    print("PLAYBOOK STRATEGIES BACKTEST")
    print("=" * 70)
    print(f"Symbol: {symbol}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()
    
    # Load data
    futures_symbol = 'MNQ=F' if symbol == 'MNQ' else 'MES=F'
    df = load_or_download_data(futures_symbol, period="90d")
    
    if df.empty:
        print("ERROR: No data loaded!")
        return
    
    print(f"\nData loaded: {len(df)} rows")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print()
    
    # Initialize strategies
    strategies = {
        'Trend Day': TrendDayStrategy(symbol=symbol),
        'B-Day (Mean Reversion)': BDayStrategy(symbol=symbol),
        # 'P-Day': PDAYStrategy(symbol=symbol),  # Add later
    }
    
    results = {}
    
    # Run each strategy
    for name, strategy in strategies.items():
        print(f"\n{'='*70}")
        print(f"Running: {name}")
        print('='*70)
        
        try:
            result = strategy.run_backtest(df)
            results[name] = result
            
            print(f"\n{name} Results:")
            print(f"  Strategy: {result['strategy']}")
            if 'trend_days_found' in result:
                print(f"  Trend Days: {result['trend_days_found']}")
            if 'b_days_found' in result:
                print(f"  B-Days: {result['b_days_found']}")
            print(f"  Signals: {result['signals_generated']}")
            
        except Exception as e:
            print(f"ERROR running {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON")
    print('='*70)
    
    for name, result in results.items():
        print(f"\n{name}:")
        for key, value in result.items():
            if key != 'trades':
                print(f"  {key}: {value}")
    
    # Save results
    output_file = f"results/playbook_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        # Convert to serializable format
        save_results = {}
        for name, result in results.items():
            save_results[name] = {k: v for k, v in result.items() if k != 'trades'}
        json.dump(save_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("DALTON PLAYBOOK STRATEGIES - SEPARATE BACKTESTS")
    print("=" * 70)
    print("\nThis will run each playbook strategy independently:")
    print("  1. Trend Day Strategy")
    print("  2. B-Day (Mean Reversion) Strategy")
    print("  3. P-Day Strategy (future)")
    print("\nEach strategy only trades on its specific day type.")
    print("=" * 70)
    print()
    
    # Run backtests
    results = run_all_strategies(symbol='MNQ')
    
    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)
