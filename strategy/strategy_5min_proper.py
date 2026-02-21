"""
Proper 5-Minute Order Flow Strategy Implementation
===================================================
Correctly calculates 5-min ATR and aggregates order flow metrics
"""

import pandas as pd
import numpy as np
from data_loader import load_data, filter_rth_session, compute_all_features

def aggregate_to_5min(df_1min):
    """
    Aggregate 1-minute data to 5-minute bars with proper order flow metrics
    """
    df = df_1min.copy()
    df['time_group'] = df['timestamp'].dt.floor('5min')
    df['hour'] = df['timestamp'].dt.hour  # Add hour before grouping
    
    # Aggregate OHLCV
    df_5min = df.groupby('time_group').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'vol_ask': 'sum',
        'vol_bid': 'sum',
        'vol_delta': 'sum',
        'session_date': 'first',
        'hour': 'first'
    }).reset_index()
    
    df_5min['timestamp'] = df_5min['time_group']
    
    # Calculate TRUE 5-minute ATR
    df_5min['tr1'] = df_5min['high'] - df_5min['low']
    df_5min['tr2'] = abs(df_5min['high'] - df_5min['close'].shift(1))
    df_5min['tr3'] = abs(df_5min['low'] - df_5min['close'].shift(1))
    df_5min['true_range'] = df_5min[['tr1', 'tr2', 'tr3']].max(axis=1)
    df_5min['atr14'] = df_5min['true_range'].rolling(window=14, min_periods=1).mean()
    
    # Calculate Order Flow metrics on 5-min bars
    df_5min['delta'] = df_5min['vol_ask'] - df_5min['vol_bid']
    df_5min['delta_pct'] = df_5min['delta'] / df_5min['volume'].replace(0, np.nan)
    
    # Delta percentile (rank in last 20 5-min bars)
    df_5min['delta_rolling_mean'] = df_5min['delta'].rolling(window=20, min_periods=1).mean()
    df_5min['delta_rolling_std'] = df_5min['delta'].rolling(window=20, min_periods=1).std()
    df_5min['delta_zscore'] = (df_5min['delta'] - df_5min['delta_rolling_mean']) / df_5min['delta_rolling_std'].replace(0, np.nan)
    df_5min['delta_percentile'] = df_5min['delta'].rolling(window=20, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50, 
        raw=False
    )
    
    # Cumulative Delta (CVD) - running sum of 5-min deltas
    df_5min['cvd'] = df_5min['delta'].cumsum()
    df_5min['cvd_ma'] = df_5min['cvd'].rolling(window=20, min_periods=1).mean()
    df_5min['cvd_rising'] = df_5min['cvd'] > df_5min['cvd_ma']
    
    # Imbalance ratio
    df_5min['imbalance_ratio'] = df_5min['vol_ask'] / df_5min['vol_bid'].replace(0, np.nan)
    
    # Volume spike
    df_5min['volume_ma'] = df_5min['volume'].rolling(window=20, min_periods=1).mean()
    df_5min['volume_spike'] = df_5min['volume'] / df_5min['volume_ma']
    
    return df_5min


def run_5min_backtest(df_5min, stop_mult=0.5, rr=1.5, max_bars=8, 
                      signal_type='delta_cvd', delta_threshold=85):
    """
    Run backtest on 5-minute data
    """
    df = df_5min.copy()
    
    # Generate signals based on type
    if signal_type == 'delta_cvd':
        df['signal_long'] = (df['delta_percentile'] > delta_threshold) & (df['delta'] > 0) & df['cvd_rising']
        df['signal_short'] = (df['delta_percentile'] > delta_threshold) & (df['delta'] < 0) & ~df['cvd_rising']
    elif signal_type == 'delta_only':
        df['signal_long'] = df['delta'] > 0
        df['signal_short'] = df['delta'] < 0
    elif signal_type == 'imbalance':
        df['signal_long'] = (df['imbalance_ratio'] > 2) & (df['delta'] > 0)
        df['signal_short'] = (df['imbalance_ratio'] < 0.5) & (df['delta'] < 0)
    
    trades = []
    in_pos = False
    position_type = None
    entry_price = 0
    stop_price = 0
    target_price = 0
    entry_bar = 0
    
    for idx, row in df.iterrows():
        if not in_pos:
            # Entry logic
            if row['signal_long']:
                in_pos = True
                position_type = 'long'
                entry_price = row['close']
                atr = row.get('atr14', df['atr14'].mean())
                stop_dist = atr * stop_mult
                stop_price = entry_price - stop_dist
                target_price = entry_price + (stop_dist * rr)
                entry_bar = idx
                
            elif row['signal_short']:
                in_pos = True
                position_type = 'short'
                entry_price = row['close']
                atr = row.get('atr14', df['atr14'].mean())
                stop_dist = atr * stop_mult
                stop_price = entry_price + stop_dist
                target_price = entry_price - (stop_dist * rr)
                entry_bar = idx
        
        else:
            # Exit logic
            bars_held = idx - entry_bar
            current_price = row['close']
            
            if position_type == 'long':
                pnl = current_price - entry_price
                
                # Check exits
                if current_price <= stop_price:
                    trades.append({'pnl': stop_price - entry_price, 'exit': 'stop', 'bars': bars_held})
                    in_pos = False
                elif current_price >= target_price:
                    trades.append({'pnl': target_price - entry_price, 'exit': 'target', 'bars': bars_held})
                    in_pos = False
                elif bars_held >= max_bars:
                    trades.append({'pnl': pnl, 'exit': 'time', 'bars': bars_held})
                    in_pos = False
                    
            else:  # short
                pnl = entry_price - current_price
                
                # Check exits
                if current_price >= stop_price:
                    trades.append({'pnl': entry_price - stop_price, 'exit': 'stop', 'bars': bars_held})
                    in_pos = False
                elif current_price <= target_price:
                    trades.append({'pnl': entry_price - target_price, 'exit': 'target', 'bars': bars_held})
                    in_pos = False
                elif bars_held >= max_bars:
                    trades.append({'pnl': pnl, 'exit': 'time', 'bars': bars_held})
                    in_pos = False
    
    return trades


def calculate_metrics(trades):
    """Calculate performance metrics"""
    if not trades:
        return None
    
    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    
    total_trades = len(trades)
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0
    expectancy = (win_rate/100 * avg_win) - ((1-win_rate/100) * avg_loss)
    profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 0
    
    # Consecutive streaks
    consecutive = []
    current = 0
    for p in pnls:
        if p > 0:
            current = current + 1 if current > 0 else 1
        else:
            current = current - 1 if current < 0 else -1
        consecutive.append(current)
    
    max_win_streak = max([c for c in consecutive if c > 0] or [0])
    max_loss_streak = abs(min([c for c in consecutive if c < 0] or [0]))
    
    # Exit reasons
    exits = [t['exit'] for t in trades]
    stop_exits = exits.count('stop')
    target_exits = exits.count('target')
    time_exits = exits.count('time')
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'expectancy': expectancy,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_win_streak': max_win_streak,
        'max_loss_streak': max_loss_streak,
        'stop_exits': stop_exits,
        'target_exits': target_exits,
        'time_exits': time_exits,
        'largest_win': max(wins) if wins else 0,
        'largest_loss': min(losses) if losses else 0
    }


if __name__ == '__main__':
    print("="*70)
    print("5-MINUTE ORDER FLOW STRATEGY - PROPER IMPLEMENTATION")
    print("="*70)
    
    # Load data
    df_1min = load_data('NQ')
    df_1min = filter_rth_session(df_1min)
    df_1min = compute_all_features(df_1min)
    
    print(f"\n1. Data Loaded:")
    print(f"   1-minute bars: {len(df_1min):,}")
    
    # Aggregate to 5-min
    df_5min = aggregate_to_5min(df_1min)
    print(f"   5-minute bars: {len(df_5min):,}")
    
    print(f"\n2. ATR Comparison:")
    print(f"   1-min ATR(14): {df_1min['atr14'].mean():.1f} points")
    print(f"   5-min ATR(14): {df_5min['atr14'].mean():.1f} points")
    print(f"   Ratio: {df_5min['atr14'].mean() / df_1min['atr14'].mean():.2f}x")
    
    print(f"\n3. Order Flow Metrics (5-min):")
    print(f"   Delta avg: {df_5min['delta'].mean():.1f}")
    print(f"   Delta percentile avg: {df_5min['delta_percentile'].mean():.1f}")
    print(f"   CVD range: {df_5min['cvd'].max() - df_5min['cvd'].min():.0f}")
    
    # Test different configurations
    print("\n" + "="*70)
    print("STRATEGY TESTING")
    print("="*70)
    
    time_windows = [
        ("11:00-13:00", 11, 13),
        ("10:00-12:00", 10, 12),
        ("13:00-15:00", 13, 15),
    ]
    
    results = []
    
    for time_name, hour_start, hour_end in time_windows:
        df_window = df_5min[(df_5min['hour'] >= hour_start) & (df_5min['hour'] < hour_end)].copy()
        df_window = df_window.dropna()
        
        if len(df_window) < 50:
            continue
        
        for stop_mult in [0.5, 0.75, 1.0]:
            for rr in [1.5, 2.0]:
                for signal_type in ['delta_cvd', 'delta_only']:
                    for delta_threshold in [80, 85]:
                        
                        trades = run_5min_backtest(
                            df_window, 
                            stop_mult=stop_mult, 
                            rr=rr, 
                            max_bars=8,
                            signal_type=signal_type,
                            delta_threshold=delta_threshold
                        )
                        
                        if len(trades) > 20:
                            metrics = calculate_metrics(trades)
                            
                            if metrics and metrics['expectancy'] > 0:
                                results.append({
                                    'time': time_name,
                                    'signal': signal_type,
                                    'delta': delta_threshold,
                                    'stop': stop_mult,
                                    'rr': rr,
                                    'trades': metrics['total_trades'],
                                    'wr': metrics['win_rate'],
                                    'exp': metrics['expectancy'],
                                    'pf': metrics['profit_factor'],
                                    'avg_win': metrics['avg_win'],
                                    'avg_loss': metrics['avg_loss'],
                                    'largest_win': metrics['largest_win'],
                                    'largest_loss': metrics['largest_loss']
                                })
    
    # Sort by expectancy
    results = sorted(results, key=lambda x: x['exp'], reverse=True)
    
    print(f"\nTOP 15 CONFIGURATIONS (sorted by expectancy):")
    print("-" * 100)
    print(f"{'Time':<12} {'Signal':<12} {'Delta':<6} {'Stop':<6} {'RR':<4} {'n':<5} {'WR%':<6} {'Exp':<7} {'PF':<5} {'AvgW':<7} {'AvgL':<7}")
    print("-" * 100)
    
    for r in results[:15]:
        print(f"{r['time']:<12} {r['signal']:<12} {r['delta']:<6} {r['stop']:.1f}x  {r['rr']:.1f}  {r['trades']:<5} {r['wr']:.1f}  {r['exp']:.2f}  {r['pf']:.2f}  {r['avg_win']:.1f}  {r['avg_loss']:.1f}")
    
    print("\n" + "="*70)
    print("BEST CONFIGURATION FOUND:")
    print("="*70)
    if results:
        best = results[0]
        print(f"\nTime: {best['time']}")
        print(f"Signal: {best['signal']}, Delta > {best['delta']}")
        print(f"Stop: {best['stop']}x ATR, Target: {best['rr']}:1 R:R")
        print(f"\nResults:")
        print(f"  Trades: {best['trades']}")
        print(f"  Win Rate: {best['wr']:.1f}%")
        print(f"  Expectancy: {best['exp']:.2f} points/trade")
        print(f"  Profit Factor: {best['pf']:.2f}")
        print(f"  Avg Win: {best['avg_win']:.1f} pts")
        print(f"  Avg Loss: {best['avg_loss']:.1f} pts")
        
        # Calculate MNQ sizing
        print(f"\nMNQ Sizing (22 contracts):")
        daily_trades = best['trades'] / 62  # per day
        daily_pnl = best['exp'] * 2 * 22 * daily_trades
        print(f"  Per trade: ${best['exp'] * 2 * 22:.2f}")
        print(f"  Daily (~{daily_trades:.1f} trades): ${daily_pnl:.2f}")
        print(f"  Days to $9K: {9000/daily_pnl:.0f}")
