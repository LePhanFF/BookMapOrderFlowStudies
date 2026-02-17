"""
Backtest Engine
==============
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from data_loader import load_data, filter_rth_session, compute_all_features


# Configuration
ACCOUNT_SIZE = 150000
MAX_DRAWDOWN = 4000
MAX_RISK_PER_TRADE = 400
MAX_DAY_LOSS = 2000
MAX_TRADES_PER_DAY = 10


class BacktestEngine:
    """Vectorized backtest engine"""
    
    def __init__(self, df, config=None):
        self.df = df.copy()
        self.config = config or {}
        
        # Risk parameters
        self.account = self.config.get('account', ACCOUNT_SIZE)
        self.max_drawdown = self.config.get('max_dd', MAX_DRAWDOWN)
        self.max_risk = self.config.get('max_risk', MAX_RISK_PER_TRADE)
        self.max_day_loss = self.config.get('max_day_loss', MAX_DAY_LOSS)
        
        # Results
        self.trades = []
        self.equity_curve = []
        self.daily_pnl = {}
        
    def generate_signals_layer_0(self):
        """
        Layer 0: Delta only (Baseline)
        Entry: Delta percentile > 80%
        """
        df = self.df.copy()
        
        # Signal: Long
        df['signal_long'] = (df['delta_percentile'] > 80) & (df['delta'] > 0)
        
        # Signal: Short
        df['signal_short'] = (df['delta_percentile'] > 80) & (df['delta'] < 0)
        
        return df
    
    def generate_signals_layer_1(self):
        """
        Layer 1: + IB Direction Filter
        Trade AFTER IB is established (after 10:30)
        Trade in direction of IB break only
        """
        df = self.generate_signals_layer_0()
        
        # Time filter: Only trade AFTER IB is established (10:30)
        df['time'] = df['timestamp'].dt.time
        df['after_ib'] = df['time'] >= time(10, 30)
        
        # Long: IB bullish + delta long + after IB
        df['signal_long'] = df['signal_long'] & (df['ib_direction'] == 'BULL') & df['after_ib']
        
        # Short: IB bearish + delta short + after IB
        df['signal_short'] = df['signal_short'] & (df['ib_direction'] == 'BEAR') & df['after_ib']
        
        return df
    
    def generate_signals_layer_2(self):
        """
        Layer 2: + Day Type Filter
        Skip B-Day and Neutral days
        """
        df = self.generate_signals_layer_1()
        
        # Only trade on Trend and P-Day
        df['signal_long'] = df['signal_long'] & df['day_type'].isin(['SUPER_TREND', 'TREND', 'P_DAY'])
        df['signal_short'] = df['signal_short'] & df['day_type'].isin(['SUPER_TREND', 'TREND', 'P_DAY'])
        
        return df
    
    def generate_signals_layer_3(self):
        """
        Layer 3: + VWAP Context
        Long: Price above VWAP + pullback to VAL
        Short: Price below VWAP + rally to VAH
        """
        df = self.generate_signals_layer_2()
        
        # Long: Price above VWAP
        df['signal_long'] = df['signal_long'] & (df['close'] > df['vwap'])
        
        # Short: Price below VWAP
        df['signal_short'] = df['signal_short'] & (df['close'] < df['vwap'])
        
        return df
    
    def set_custom_signals(self, df):
        """Set custom signals for testing"""
        self.df = df
        return df
    
    def run_with_custom_signals(self, df, layer_name='Custom', verbose=True):
        """Run backtest with custom signals in df"""
        self.df = df.copy()
        
        # Initialize tracking
        trades = []
        equity = self.account
        position = None
        entry_price = 0
        entry_time = None
        entry_bar = None
        daily_pnl = {}
        bars_held = 0
        max_bars_held = 5
        
        for idx, row in df.iterrows():
            date = row['session_date']
            current_price = row['close']
            current_time = row['timestamp']
            
            if date not in daily_pnl:
                daily_pnl[date] = 0
            
            if daily_pnl[date] <= -self.max_day_loss:
                continue
            
            # Entry
            if position is None:
                if row.get('signal_long', False):
                    position = 'long'
                    entry_price = current_price
                    entry_time = current_time
                    entry_bar = idx
                    bars_held = 0
                elif row.get('signal_short', False):
                    position = 'short'
                    entry_price = current_price
                    entry_time = current_time
                    entry_bar = idx
                    bars_held = 0
            
            # Exit
            elif position is not None:
                bars_held += 1
                
                if position == 'long':
                    pnl = current_price - entry_price
                else:
                    pnl = entry_price - current_price
                
                should_exit = False
                exit_reason = ''
                stop_distance = self.calculate_stop(row, 'atr_1x')
                
                if bars_held >= max_bars_held:
                    should_exit = True
                    exit_reason = 'time'
                if pnl <= -stop_distance:
                    should_exit = True
                    exit_reason = 'stop'
                if pnl >= stop_distance * 2:
                    should_exit = True
                    exit_reason = 'tp'
                
                if should_exit:
                    trade = {
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': position,
                        'pnl': pnl,
                        'exit_reason': exit_reason,
                        'bars_held': bars_held,
                        'date': date
                    }
                    trades.append(trade)
                    equity += pnl
                    daily_pnl[date] = daily_pnl.get(date, 0) + pnl
                    position = None
        
        if verbose:
            print(f'{layer_name}: {len(trades)} trades, Equity: \${equity:.2f}')
        
        return {
            'trades': trades,
            'equity': equity,
            'daily_pnl': daily_pnl,
            'layer': layer_name,
        }
    
    def generate_signals(self, layer=0):
        """Generate signals based on layer"""
        if layer == 0:
            return self.generate_signals_layer_0()
        elif layer == 1:
            return self.generate_signals_layer_1()
        elif layer == 2:
            return self.generate_signals_layer_2()
        elif layer == 3:
            return self.generate_signals_layer_3()
        else:
            return self.generate_signals_layer_3()
    
    def calculate_stop(self, row, method='atr'):
        """Calculate stop loss in points"""
        atr = row.get('atr14', 10)
        
        if method == 'atr_1x':
            return atr
        elif method == 'atr_2x':
            return atr * 2
        elif method == 'atr_05x':
            return atr * 0.5
        elif method == 'ib':
            # Use IB range as stop
            return row.get('ib_range', 20) * 0.5
        else:
            return atr  # Default 1x ATR
    
    def run_backtest(self, layer=0, stop_method='atr_1x', verbose=True):
        """Run backtest with given parameters"""
        
        # Get signals
        df = self.generate_signals(layer)
        
        # Initialize tracking
        trades = []
        equity = self.account
        position = None
        entry_price = 0
        entry_time = None
        entry_bar = None
        daily_pnl = {}
        bars_held = 0
        max_bars_held = 5  # Exit after 5 bars
        
        # Iterate through bars
        for idx, row in df.iterrows():
            date = row['session_date']
            current_price = row['close']
            current_time = row['timestamp']
            
            # Track daily P&L
            if date not in daily_pnl:
                daily_pnl[date] = 0
            
            # Skip if daily loss limit hit
            if daily_pnl[date] <= -self.max_day_loss:
                continue
            
            # === ENTRY ===
            if position is None:
                # Check for long signal
                if row.get('signal_long', False):
                    position = 'long'
                    entry_price = current_price
                    entry_time = current_time
                    entry_bar = idx
                    bars_held = 0
                    
                # Check for short signal
                elif row.get('signal_short', False):
                    position = 'short'
                    entry_price = current_price
                    entry_time = current_time
                    entry_bar = idx
                    bars_held = 0
            
            # === EXIT ===
            elif position is not None:
                bars_held += 1
                
                # Calculate current P&L
                if position == 'long':
                    pnl = current_price - entry_price
                else:  # short
                    pnl = entry_price - current_price
                
                # Exit conditions
                should_exit = False
                exit_reason = ''
                
                # 1. Time-based exit (after N bars)
                if bars_held >= max_bars_held:
                    should_exit = True
                    exit_reason = 'time'
                
                # 2. Stop loss hit
                stop_distance = self.calculate_stop(row, stop_method)
                if pnl <= -stop_distance:
                    should_exit = True
                    exit_reason = 'stop'
                
                # 3. Take profit (2x risk)
                if pnl >= stop_distance * 2:
                    should_exit = True
                    exit_reason = 'tp'
                
                # Execute exit
                if should_exit:
                    trade = {
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': position,
                        'pnl': pnl,
                        'exit_reason': exit_reason,
                        'bars_held': bars_held,
                        'date': date
                    }
                    trades.append(trade)
                    
                    # Update equity and daily P&L
                    equity += pnl
                    daily_pnl[date] = daily_pnl.get(date, 0) + pnl
                    
                    # Reset position
                    position = None
                    entry_price = 0
                    entry_time = None
        
        if verbose:
            print(f"Layer {layer}: {len(trades)} trades, Equity: ${equity:.2f}")
        
        return {
            'trades': trades,
            'equity': equity,
            'daily_pnl': daily_pnl,
            'layer': layer,
            'stop_method': stop_method
        }
    
    def calculate_metrics(self, results):
        """Calculate performance metrics"""
        trades = results.get('trades', [])
        
        if len(trades) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'expectancy': 0,
                'profit_factor': 0,
                'max_drawdown': 0
            }
        
        wins = [t['pnl'] for t in trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in trades if t['pnl'] < 0]
        
        win_rate = len(wins) / len(trades) if trades else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
        
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'expectancy': expectancy,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }


def run_layer_comparison(instrument='NQ', layers=[0, 1, 2, 3]):
    """Run comparison across different layers"""
    
    print(f"\n{'='*60}")
    print(f"LAYER COMPARISON - {instrument}")
    print(f"{'='*60}")
    
    # Load data
    df = load_data(instrument)
    df = filter_rth_session(df)
    df = compute_all_features(df)
    
    results = []
    
    for layer in layers:
        print(f"\nTesting Layer {layer}...")
        
        engine = BacktestEngine(df)
        result = engine.run_backtest(layer=layer, verbose=False)
        metrics = engine.calculate_metrics(result)
        
        results.append({
            'layer': layer,
            **metrics
        })
        
        print(f"  Trades: {metrics['total_trades']}")
        print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
        print(f"  Expectancy: ${metrics['expectancy']:.2f}")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    
    return pd.DataFrame(results)


if __name__ == '__main__':
    # Quick test
    results = run_layer_comparison('NQ', [0, 1, 2, 3])
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(results)
