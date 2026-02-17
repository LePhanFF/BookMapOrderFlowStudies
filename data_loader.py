"""
Data Loader and Feature Engineering
===================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, time


def load_data(instrument='NQ'):
    """Load and prepare data for analysis"""
    filepath = f'/home/lphan/jupyterlab/BookMapOrderFlowStudies/csv/{instrument}_Volumetric_1.csv'
    
    # Read CSV, skip schema line (first line starts with #)
    df = pd.read_csv(filepath, low_memory=False, comment='#')
    
    # Convert columns
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'ema20', 'ema50', 'ema200', 
                    'rsi14', 'atr14', 'vwap', 'vwap_upper1', 'vwap_upper2', 'vwap_upper3',
                    'vwap_lower1', 'vwap_lower2', 'vwap_lower3', 'vol_ask', 'vol_bid', 'vol_delta']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Parse timestamps - handle bad rows
    df = df[df['timestamp'].astype(str).str.len() > 10].copy()  # Filter bad rows
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['session_date'] = pd.to_datetime(df['session_date'], errors='coerce')
    df['time'] = df['timestamp'].dt.time
    df['session_date'] = pd.to_datetime(df['session_date'])
    df['time'] = df['timestamp'].dt.time
    
    print(f"Loaded {instrument}: {len(df):,} rows")
    print(f"Date range: {df['session_date'].min()} to {df['session_date'].max()}")
    print(f"Sessions: {df['session_date'].nunique()}")
    
    return df


def filter_rth_session(df):
    """Filter to US Regular Trading Hours only: 9:30 - 15:00"""
    rth_start = time(9, 30)
    rth_end = time(15, 0)
    
    df_rth = df[(df['time'] >= rth_start) & (df['time'] <= rth_end)].copy()
    print(f"RTH rows: {len(df_rth):,} ({len(df_rth)/len(df)*100:.1f}%)")
    
    return df_rth


def compute_order_flow_features(df):
    """Compute order flow features"""
    # Delta features
    df['delta'] = df['vol_ask'] - df['vol_bid']
    df['delta_pct'] = df['delta'] / df['volume'].replace(0, np.nan)
    
    # Rolling delta statistics (20-bar window)
    df['delta_rolling_mean'] = df['delta'].rolling(20, min_periods=1).mean()
    df['delta_rolling_std'] = df['delta'].rolling(20, min_periods=1).std()
    
    # Z-score (adaptive threshold)
    df['delta_zscore'] = (df['delta'] - df['delta_rolling_mean']) / df['delta_rolling_std'].replace(0, np.nan)
    
    # Delta percentile (last 20 bars)
    df['delta_percentile'] = df['delta'].rolling(20, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )
    
    # Cumulative Delta
    df['cumulative_delta'] = df['delta'].cumsum()
    df['cumulative_delta_ma'] = df['cumulative_delta'].ewm(span=20).mean()
    
    # Imbalance ratio
    df['imbalance_ratio'] = df['vol_ask'] / df['vol_bid'].replace(0, np.nan)
    
    # Volume spike
    df['volume_ma'] = df['volume'].rolling(20, min_periods=1).mean()
    df['volume_spike'] = df['volume'] / df['volume_ma']
    
    return df


def compute_volume_profile(df, window=390):
    """Compute volume profile features (rolling window)"""
    # This is a simplified version - computes rolling POC, VAH, VAL
    # For proper VP, we'd need anchored profiles
    
    # Rolling volume profile (per session)
    def rolling_vp(grp):
        if len(grp) < 10:
            return grp
        try:
            # Create price bins
            price_range = grp['close'].max() - grp['close'].min()
            if price_range == 0:
                return grp
            bins = 20
            grp['price_bin'] = pd.cut(grp['close'], bins=bins, labels=False)
            
            # Volume by price
            vol_by_price = grp.groupby('price_bin')['volume'].sum()
            
            if len(vol_by_price) > 0:
                # POC (bin with max volume)
                poc_bin = vol_by_price.idxmax()
                total_vol = vol_by_price.sum()
                
                # Value Area (70%)
                cumsum = vol_by_price.sort_values(ascending=False).cumsum()
                va_vol = total_vol * 0.70
                va_bins = cumsum[cumsum <= va_vol].index
                
                grp['poc_bin'] = poc_bin
                grp['vah_bin'] = va_bins.max() if len(va_bins) > 0 else poc_bin
                grp['val_bin'] = va_bins.min() if len(va_bins) > 0 else poc_bin
                
        except:
            pass
        return grp
    
    # Apply rolling VP (expensive, so use session-level)
    return df


def compute_day_type(df):
    """Compute day type based on IB and trend"""
    # Group by session
    day_types = []
    
    for session, session_df in df.groupby('session_date'):
        if len(session_df) < 60:
            day_types.append({'session_date': session, 'day_type': 'NEUTRAL', 'ib_range': 100})
            continue
        
        # Opening Range (first 60 min = 60 bars)
        ib_data = session_df.head(60)
        ib_high = ib_data['high'].max()
        ib_low = ib_data['low'].min()
        ib_range = ib_high - ib_low
        
        # Current price position at end of day
        current_price = session_df['close'].iloc[-1]
        
        # IB Extension - measured from mid-IB
        ib_mid = (ib_high + ib_low) / 2
        if current_price > ib_mid:
            # Price above midpoint
            extension = (current_price - ib_mid) / ib_range if ib_range > 0 else 0
            direction = 'BULL'
        else:
            # Price below midpoint
            extension = (ib_mid - current_price) / ib_range if ib_range > 0 else 0
            direction = 'BEAR'
        
        # Classify day type based on extension from midpoint
        if extension > 1.0:
            day_type = 'TREND'
        elif extension > 0.5:
            day_type = 'P_DAY'
        elif extension < 0.2:
            day_type = 'B_DAY'
        else:
            day_type = 'NEUTRAL'
        
        day_types.append({
            'session_date': session,
            'day_type': day_type,
            'day_direction': direction,
            'ib_range': ib_range,
            'ib_extension': abs(current_price - ib_mid),
            'ib_high': ib_high,
            'ib_low': ib_low
        })
    
    day_type_df = pd.DataFrame(day_types)
    df = df.merge(day_type_df, on='session_date', how='left')
    
    return df


def compute_ib_features(df):
    """Compute Opening Range (IB) features per session"""
    # Initialize columns
    df['ib_high'] = np.nan
    df['ib_low'] = np.nan
    df['ib_range'] = np.nan
    df['ib_extension'] = np.nan
    df['ib_direction'] = 'INSIDE'
    
    # Group by session and compute IB
    for session, session_df in df.groupby('session_date'):
        if len(session_df) < 60:
            continue
        
        # First 60 minutes IB
        ib_data = session_df.head(60)
        ib_high = ib_data['high'].max()
        ib_low = ib_data['low'].min()
        
        # Get indices for this session
        session_indices = session_df.index
        
        # Calculate extension and direction for each bar
        for idx in session_indices:
            row = df.loc[idx]
            current_price = row['close']
            
            # IB extension
            if current_price > ib_high:
                extension = current_price - ib_high
                direction = 'BULL'
            elif current_price < ib_low:
                extension = ib_low - current_price
                direction = 'BEAR'
            else:
                extension = 0
                direction = 'INSIDE'
            
            df.loc[idx, 'ib_high'] = ib_high
            df.loc[idx, 'ib_low'] = ib_low
            df.loc[idx, 'ib_range'] = ib_high - ib_low
            df.loc[idx, 'ib_extension'] = extension
            df.loc[idx, 'ib_direction'] = direction
    
    # Forward fill for missing values
    df['ib_high'] = df['ib_high'].ffill()
    df['ib_low'] = df['ib_low'].ffill()
    df['ib_range'] = df['ib_range'].ffill()
    df['ib_extension'] = df['ib_extension'].ffill()
    df['ib_direction'] = df['ib_direction'].ffill()
    
    return df


def compute_all_features(df):
    """Compute all features for the strategy"""
    print("\nComputing features...")
    
    # Order flow features
    df = compute_order_flow_features(df)
    print("  - Order flow features computed")
    
    # Day type
    df = compute_day_type(df)
    print("  - Day type computed")
    
    # IB features
    df = compute_ib_features(df)
    print("  - IB features computed")
    
    return df


if __name__ == '__main__':
    # Test data loading
    df = load_data('NQ')
    df_rth = filter_rth_session(df)
    df_rth = compute_all_features(df_rth)
    
    print(f"\nTotal features: {len(df_rth.columns)}")
    print(f"Sample data:\n{df_rth.head()}")
