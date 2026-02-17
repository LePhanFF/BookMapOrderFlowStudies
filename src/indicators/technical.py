"""
Technical Indicators Module
===========================
Calculates indicators needed for trend following and mean reversion strategies

Indicators:
- EMA (Exponential Moving Average)
- ADX (Average Directional Index)
- Bollinger Bands
- RSI (Relative Strength Index)
- ATR (Average True Range)
- VWAP (Volume Weighted Average Price)
"""

import pandas as pd
import numpy as np


def calculate_ema(series: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    
    atr = true_range.rolling(window=period, min_periods=1).mean()
    return atr


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    """Calculate Bollinger Bands
    Returns: (middle_band, upper_band, lower_band)
    """
    middle = series.rolling(window=period, min_periods=1).mean()
    std = series.rolling(window=period, min_periods=1).std()
    
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    return middle, upper, lower


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index (ADX)
    
    ADX measures trend strength (not direction)
    ADX > 25: Strong trend
    ADX 20-25: Moderate trend
    ADX < 20: Weak trend/no trend
    """
    # Calculate +DM and -DM
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Calculate TR
    tr1 = df['high'] - df['low']
    tr2 = np.abs(df['high'] - df['close'].shift())
    tr3 = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smooth TR, +DM, -DM
    atr = tr.rolling(window=period, min_periods=1).mean()
    plus_di = 100 * (plus_dm.rolling(window=period, min_periods=1).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period, min_periods=1).mean() / atr)
    
    # Calculate DX and ADX
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=period, min_periods=1).mean()
    
    return adx


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate Volume Weighted Average Price"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    return vwap


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to dataframe"""
    df = df.copy()
    
    # Trend indicators
    df['ema20'] = calculate_ema(df['close'], 20)
    df['ema50'] = calculate_ema(df['close'], 50)
    df['atr14'] = calculate_atr(df, 14)
    df['adx14'] = calculate_adx(df, 14)
    
    # Mean reversion indicators
    df['rsi14'] = calculate_rsi(df['close'], 14)
    df['bb_middle'], df['bb_upper'], df['bb_lower'] = calculate_bollinger_bands(df['close'], 20, 2.0)
    
    # Volume/price
    df['vwap'] = calculate_vwap(df)
    
    return df


if __name__ == '__main__':
    # Test with sample data
    print("Technical Indicators Module Loaded")
    print("\nAvailable indicators:")
    print("  - EMA (20, 50)")
    print("  - ATR (14)")
    print("  - ADX (14)")
    print("  - RSI (14)")
    print("  - Bollinger Bands (20, 2.0)")
    print("  - VWAP")
    print("\nUse: add_all_indicators(df) to add all indicators to your dataframe")
