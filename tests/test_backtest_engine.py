"""
Unit Tests for Backtest Engine
==============================
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_data, filter_rth_session, compute_all_features
from backtest_engine import BacktestEngine, run_layer_comparison


class TestBacktestEngine:
    """Test backtest engine"""
    
    @pytest.fixture
    def sample_data(self):
        """Load sample data for testing"""
        df = load_data('NQ')
        df = filter_rth_session(df)
        df = compute_all_features(df)
        return df.head(1000)  # Use subset for speed
    
    def test_engine_initialization(self, sample_data):
        """Test engine initializes correctly"""
        engine = BacktestEngine(sample_data)
        assert engine.account == 150000
        assert engine.max_risk == 400
    
    def test_layer_0_signals(self, sample_data):
        """Test Layer 0 signal generation"""
        engine = BacktestEngine(sample_data)
        df = engine.generate_signals_layer_0()
        
        assert 'signal_long' in df.columns
        assert 'signal_short' in df.columns
        assert df['signal_long'].dtype == bool
        assert df['signal_short'].dtype == bool
    
    def test_layer_1_ib_filter(self, sample_data):
        """Test Layer 1 adds IB filter"""
        engine = BacktestEngine(sample_data)
        df0 = engine.generate_signals_layer_0()
        df1 = engine.generate_signals_layer_1()
        
        # Layer 1 should have fewer signals (more restrictive)
        assert df1['signal_long'].sum() <= df0['signal_long'].sum()
        assert df1['signal_short'].sum() <= df0['signal_short'].sum()
    
    def test_layer_2_daytype_filter(self, sample_data):
        """Test Layer 2 adds day type filter"""
        engine = BacktestEngine(sample_data)
        df1 = engine.generate_signals_layer_1()
        df2 = engine.generate_signals_layer_2()
        
        # Layer 2 should have fewer signals
        assert df2['signal_long'].sum() <= df1['signal_long'].sum()
        assert df2['signal_short'].sum() <= df1['signal_short'].sum()
    
    def test_layer_3_vwap_filter(self, sample_data):
        """Test Layer 3 adds VWAP filter"""
        engine = BacktestEngine(sample_data)
        df2 = engine.generate_signals_layer_2()
        df3 = engine.generate_signals_layer_3()
        
        # Layer 3 should have fewer signals
        assert df3['signal_long'].sum() <= df2['signal_long'].sum()
        assert df3['signal_short'].sum() <= df2['signal_short'].sum()
    
    def test_stop_calculation(self, sample_data):
        """Test stop loss calculation"""
        engine = BacktestEngine(sample_data)
        row = sample_data.iloc[0]
        row['atr14'] = 20
        row['ib_range'] = 100
        
        # Test different methods
        assert engine.calculate_stop(row, 'atr_1x') == 20
        assert engine.calculate_stop(row, 'atr_2x') == 40
        assert engine.calculate_stop(row, 'atr_05x') == 10
    
    def test_metrics_calculation(self):
        """Test metrics calculation"""
        engine = BacktestEngine(pd.DataFrame())
        
        # Empty trades
        results = {'trades': []}
        metrics = engine.calculate_metrics(results)
        assert metrics['total_trades'] == 0
        assert metrics['win_rate'] == 0
        
        # With trades
        results = {
            'trades': [
                {'pnl': 100},
                {'pnl': -50},
                {'pnl': 150},
                {'pnl': -25}
            ]
        }
        metrics = engine.calculate_metrics(results)
        
        assert metrics['total_trades'] == 4
        assert metrics['win_rate'] == 0.5  # 2/4
        assert metrics['profit_factor'] > 0


class TestLayerComparison:
    """Test layer comparison"""
    
    def test_comparison_runs(self):
        """Test that comparison runs without error"""
        # Use small subset for testing
        from data_loader import load_data, filter_rth_session, compute_all_features
        
        df = load_data('NQ')
        df = filter_rth_session(df)
        df = compute_all_features(df)
        df = df.head(500)  # Very small for quick test
        
        engine = BacktestEngine(df)
        
        # Test each layer
        for layer in [0, 1, 2, 3]:
            result = engine.run_backtest(layer=layer, verbose=False)
            assert 'trades' in result
            assert 'equity' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
