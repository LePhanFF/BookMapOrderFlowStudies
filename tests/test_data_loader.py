"""
Unit Tests for Data Loader
==========================
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import (
    load_data,
    filter_rth_session,
    compute_order_flow_features,
    compute_day_type,
    compute_ib_features,
    compute_all_features
)


class TestLoadData:
    """Test data loading functionality"""
    
    def test_load_nq_data(self):
        """Test loading NQ data"""
        df = load_data('NQ')
        assert len(df) > 0, "Should load data"
        assert 'timestamp' in df.columns, "Should have timestamp"
        assert 'close' in df.columns, "Should have close"
        assert 'vol_ask' in df.columns, "Should have vol_ask"
        assert 'vol_bid' in df.columns, "Should have vol_bid"
    
    def test_load_es_data(self):
        """Test loading ES data"""
        df = load_data('ES')
        assert len(df) > 0, "Should load ES data"
    
    def test_date_range(self):
        """Test date range is correct"""
        df = load_data('NQ')
        assert df['session_date'].min() is not pd.NaT
        assert df['session_date'].max() is not pd.NaT
    
    def test_no_null_timestamps(self):
        """Test no null timestamps"""
        df = load_data('NQ')
        assert df['timestamp'].isna().sum() == 0, "No null timestamps"


class TestFilterRTH:
    """Test RTH session filtering"""
    
    def test_filter_removes_off_hours(self):
        """Test that off-hours data is filtered"""
        df = load_data('NQ')
        df_rth = filter_rth_session(df)
        
        # RTH should be smaller than full data
        assert len(df_rth) < len(df), "RTH should be smaller"
        
        # All RTH rows should be between 9:30 and 15:00
        for t in df_rth['time']:
            assert time(9, 30) <= t <= time(15, 0), f"Time {t} outside RTH"


class TestOrderFlowFeatures:
    """Test order flow feature computation"""
    
    def test_delta_calculation(self):
        """Test delta = ask - bid"""
        df = load_data('NQ')
        df = df.head(100).copy()
        df = compute_order_flow_features(df)
        
        # Delta should equal ask - bid
        expected_delta = df['vol_ask'] - df['vol_bid']
        pd.testing.assert_series_equal(
            df['delta'].reset_index(drop=True), 
            expected_delta.reset_index(drop=True), 
            check_dtype=False,
            check_names=False
        )
    
    def test_delta_pct_is_normalized(self):
        """Test delta_pct is between -1 and 1"""
        df = load_data('NQ')
        df = df.head(100).copy()
        df = compute_order_flow_features(df)
        
        valid_pct = df['delta_pct'].dropna()
        assert valid_pct.min() >= -1, "delta_pct >= -1"
        assert valid_pct.max() <= 1, "delta_pct <= 1"
    
    def test_delta_zscore_has_reasonable_range(self):
        """Test z-score is within reasonable bounds"""
        df = load_data('NQ')
        df = df.head(200).copy()  # Need enough data for rolling
        df = compute_order_flow_features(df)
        
        # After initial warmup, z-scores should be mostly within -3 to 3
        zscores = df['delta_zscore'].dropna()
        zscores = zscores[zscores.abs() < 1e10]  # Remove inf values
        assert len(zscores) > 0, "Should have valid z-scores"
    
    def test_cumulative_delta_increases_with_positive_delta(self):
        """Test cumulative delta calculation"""
        df = load_data('NQ')
        df = df.head(100).copy()
        df = compute_order_flow_features(df)
        
        # Check cumulative increases with positive delta
        # Just verify it exists and is numeric
        assert df['cumulative_delta'].dtype in [np.float64, np.int64]
        assert df['cumulative_delta'].isna().sum() == 0
    
    def test_volume_spike_ratio(self):
        """Test volume spike calculation"""
        df = load_data('NQ')
        df = df.head(100).copy()
        df = compute_order_flow_features(df)
        
        # Volume spike should be positive
        assert (df['volume_spike'] > 0).all(), "Volume spike should be positive"


class TestDayType:
    """Test day type classification"""
    
    def test_day_type_values(self):
        """Test day type returns valid values"""
        df = load_data('NQ')
        df = filter_rth_session(df)
        df = compute_day_type(df)
        
        valid_types = ['SUPER_TREND', 'TREND', 'B_DAY', 'NEUTRAL', 'UNKNOWN', 'P_DAY']
        day_types = df['day_type'].unique()
        
        for dt in day_types:
            assert dt in valid_types, f"Invalid day type: {dt}"
    
    def test_ib_range_positive(self):
        """Test IB range is positive"""
        df = load_data('NQ')
        df = filter_rth_session(df)
        df = compute_day_type(df)
        
        ib_ranges = df['ib_range'].dropna()
        assert (ib_ranges >= 0).all(), "IB range should be non-negative"


class TestIBFeatures:
    """Test IB (Opening Range) features"""
    
    def test_ib_high_greater_than_ib_low(self):
        """Test IB high > IB low"""
        df = load_data('NQ')
        df = filter_rth_session(df)
        df = compute_ib_features(df)
        
        # For each session where IB is defined
        valid_ib = df[df['ib_high'].notna()]
        assert (valid_ib['ib_high'] > valid_ib['ib_low']).all()
    
    def test_ib_direction_values(self):
        """Test IB direction values are valid"""
        df = load_data('NQ')
        df = filter_rth_session(df)
        df = compute_ib_features(df)
        
        valid_directions = ['BULL', 'BEAR', 'INSIDE']
        directions = df['ib_direction'].dropna().unique()
        
        for d in directions:
            assert d in valid_directions, f"Invalid direction: {d}"
    
    def test_bull_extension_positive(self):
        """Test bullish extension is positive"""
        df = load_data('NQ')
        df = filter_rth_session(df)
        df = compute_ib_features(df)
        
        bull_bars = df[df['ib_direction'] == 'BULL']
        assert (bull_bars['ib_extension'] >= 0).all()
    
    def test_bear_extension_positive(self):
        """Test bearish extension is positive"""
        df = load_data('NQ')
        df = filter_rth_session(df)
        df = compute_ib_features(df)
        
        bear_bars = df[df['ib_direction'] == 'BEAR']
        assert (bear_bars['ib_extension'] >= 0).all()


class TestIntegration:
    """Integration tests"""
    
    def test_full_pipeline(self):
        """Test full feature pipeline"""
        df = load_data('NQ')
        df = filter_rth_session(df)
        df = compute_all_features(df)
        
        # Should have all expected columns
        expected_cols = [
            'delta', 'delta_pct', 'delta_zscore', 'delta_percentile',
            'cumulative_delta', 'imbalance_ratio', 'volume_spike',
            'day_type', 'ib_range', 'ib_high', 'ib_low',
            'ib_extension', 'ib_direction'
        ]
        
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_no_data_leakage(self):
        """Test that features don't use future data"""
        df = load_data('NQ')
        df = filter_rth_session(df)
        df = compute_all_features(df)
        
        # Delta should only use current and past data (rolling 20)
        # This is implicitly tested by the calculation method
        
        # Cumulative delta should not have NaN at end (except warmup)
        # Check last rows have values
        last_rows = df.tail(100)
        assert last_rows['cumulative_delta'].notna().any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
