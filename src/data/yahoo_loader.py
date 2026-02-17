"""
Yahoo Finance Data Loader
=========================
Pulls options and futures data from Yahoo Finance for backtesting

Supports:
- Futures: NQ=F, ES=F, YM=F
- Options: SPY, QQQ
- Underlying: ^NDX, ^GSPC (indices)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')


def get_futures_data(symbol: str, period: str = "90d", interval: str = "1m") -> pd.DataFrame:
    """
    Get futures data from Yahoo Finance
    
    Args:
        symbol: Futures symbol (e.g., 'NQ=F', 'ES=F', 'MNQ=F')
        period: Data period ('60d', '90d', '1y')
        interval: Bar interval ('1m', '5m', '15m', '1h', '1d')
    
    Returns:
        DataFrame with OHLCV data
    """
    print(f"Downloading {symbol} futures data...")
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            print(f"No data returned for {symbol}")
            return pd.DataFrame()
        
        # Reset index to make datetime a column
        df.reset_index(inplace=True)
        
        # Rename columns to match our standard
        df.rename(columns={
            'Datetime': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)
        
        # Add symbol column
        df['symbol'] = symbol
        
        print(f"Downloaded {len(df)} bars for {symbol}")
        return df
        
    except Exception as e:
        print(f"Error downloading {symbol}: {e}")
        return pd.DataFrame()


def get_options_chain(symbol: str, expiration: str = None) -> dict:
    """
    Get options chain for a symbol
    
    Args:
        symbol: Underlying symbol (e.g., 'SPY', 'QQQ')
        expiration: Expiration date (YYYY-MM-DD). If None, gets nearest
    
    Returns:
        Dictionary with calls and puts DataFrames
    """
    print(f"Getting options chain for {symbol}...")
    
    try:
        ticker = yf.Ticker(symbol)
        
        # Get available expirations
        expirations = ticker.options
        
        if not expirations:
            print(f"No options available for {symbol}")
            return None
        
        # Use specified expiration or nearest
        if expiration is None:
            expiration = expirations[0]
        
        if expiration not in expirations:
            print(f"Expiration {expiration} not available. Available: {expirations[:5]}")
            return None
        
        # Get options chain
        opt_chain = ticker.option_chain(expiration)
        
        calls = opt_chain.calls
        puts = opt_chain.puts
        
        print(f"Retrieved {len(calls)} calls and {len(puts)} puts for {symbol} {expiration}")
        
        return {
            'calls': calls,
            'puts': puts,
            'expiration': expiration,
            'underlying': ticker.info.get('regularMarketPrice', None)
        }
        
    except Exception as e:
        print(f"Error getting options for {symbol}: {e}")
        return None


def get_underlying_data(symbol: str, period: str = "90d", interval: str = "1m") -> pd.DataFrame:
    """
    Get underlying stock/ETF data for options backtesting
    
    Args:
        symbol: Symbol (e.g., 'SPY', 'QQQ', 'SPX')
        period: Data period
        interval: Bar interval
    
    Returns:
        DataFrame with OHLCV data
    """
    print(f"Downloading {symbol} underlying data...")
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            print(f"No data returned for {symbol}")
            return pd.DataFrame()
        
        df.reset_index(inplace=True)
        
        df.rename(columns={
            'Datetime': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)
        
        df['symbol'] = symbol
        
        print(f"Downloaded {len(df)} bars for {symbol}")
        return df
        
    except Exception as e:
        print(f"Error downloading {symbol}: {e}")
        return pd.DataFrame()


def get_available_expirations(symbol: str) -> list:
    """Get list of available option expiration dates"""
    try:
        ticker = yf.Ticker(symbol)
        return ticker.options
    except:
        return []


def download_all_data(futures_symbols: list = None, 
                     underlying_symbols: list = None,
                     period: str = "90d") -> dict:
    """
    Download all necessary data for backtesting
    
    Args:
        futures_symbols: List of futures symbols
        underlying_symbols: List of underlying symbols for options
        period: Data period
    
    Returns:
        Dictionary with all data
    """
    data = {
        'futures': {},
        'underlying': {},
        'options_expirations': {}
    }
    
    # Download futures data
    if futures_symbols:
        for symbol in futures_symbols:
            df = get_futures_data(symbol, period=period)
            if not df.empty:
                data['futures'][symbol] = df
    
    # Download underlying data
    if underlying_symbols:
        for symbol in underlying_symbols:
            df = get_underlying_data(symbol, period=period)
            if not df.empty:
                data['underlying'][symbol] = df
                # Get available expirations
                data['options_expirations'][symbol] = get_available_expirations(symbol)
    
    return data


if __name__ == '__main__':
    print("Yahoo Finance Data Loader")
    print("=" * 50)
    
    # Test downloading futures data
    print("\n1. Testing Futures Data Download:")
    futures_symbols = ['NQ=F', 'MNQ=F', 'ES=F', 'MES=F']
    
    for symbol in futures_symbols:
        df = get_futures_data(symbol, period="5d", interval="5m")  # Short period for testing
        if not df.empty:
            print(f"  ✓ {symbol}: {len(df)} bars")
            print(f"    Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        else:
            print(f"  ✗ {symbol}: Failed to download")
    
    # Test downloading underlying data
    print("\n2. Testing Underlying Data Download:")
    underlying_symbols = ['SPY', 'QQQ']
    
    for symbol in underlying_symbols:
        df = get_underlying_data(symbol, period="5d", interval="5m")
        if not df.empty:
            print(f"  ✓ {symbol}: {len(df)} bars")
            # Get options expirations
            expirations = get_available_expirations(symbol)
            print(f"    Available expirations: {len(expirations)}")
            if expirations:
                print(f"    Next 3: {expirations[:3]}")
        else:
            print(f"  ✗ {symbol}: Failed to download")
    
    print("\n" + "=" * 50)
    print("Data loader ready for backtesting!")
    print("\nUsage:")
    print("  from src.data.yahoo_loader import get_futures_data, get_underlying_data")
    print("  df = get_futures_data('MNQ=F', period='90d', interval='1m')")
