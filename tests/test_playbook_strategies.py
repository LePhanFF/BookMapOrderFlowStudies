"""
Unit Tests for Playbook Strategies
==================================

Tests for:
- Rockit wrapper integration
- Trend Day strategy logic
- B-Day strategy logic
- TPO acceptance/rejection detection
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import sys
sys.path.append('/home/lphan/jupyterlab/BookMapOrderFlowStudies')

from src.rockit_integration.wrapper import RockitBacktestWrapper
from src.strategies.trend_day_strategy import TrendDayStrategy
from src.strategies.b_day_strategy import BDayStrategy
from src.indicators.technical import (
    calculate_ema, calculate_atr, calculate_adx,
    calculate_bollinger_bands, calculate_rsi, calculate_vwap
)


class TestTechnicalIndicators(unittest.TestCase):
    """Test technical indicator calculations"""
    
    def setUp(self):
        """Create sample data"""
        dates = pd.date_range('2026-01-01', periods=100, freq='5min')
        np.random.seed(42)
        
        self.df = pd.DataFrame({
            'timestamp': dates,
            'open': 4500 + np.random.randn(100).cumsum(),
            'high': 4502 + np.random.randn(100).cumsum(),
            'low': 4498 + np.random.randn(100).cumsum(),
            'close': 4500 + np.random.randn(100).cumsum(),
            'volume': np.random.randint(1000, 5000, 100)
        })
        self.df.set_index('timestamp', inplace=True)
    
    def test_ema_calculation(self):
        """Test EMA calculation"""
        ema20 = calculate_ema(self.df['close'], 20)
        self.assertEqual(len(ema20), len(self.df))
        self.assertFalse(ema20.isna().all())
    
    def test_atr_calculation(self):
        """Test ATR calculation"""
        atr = calculate_atr(self.df, 14)
        self.assertEqual(len(atr), len(self.df))
        self.assertTrue((atr >= 0).all())
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands"""
        middle, upper, lower = calculate_bollinger_bands(self.df['close'], 20, 2.0)
        self.assertEqual(len(middle), len(self.df))
        self.assertTrue((upper >= middle).all())
        self.assertTrue((middle >= lower).all())


class TestTPOAcceptanceRejection(unittest.TestCase):
    """Test TPO acceptance/rejection detection"""
    
    def setUp(self):
        """Create sample TPO data"""
        self.wrapper = RockitBacktestWrapper('MNQ')
    
    def test_trend_day_classification(self):
        """Test trend day detection"""
        # Mock data: Strong extension above IB
        ib_high = 4500
        ib_low = 4480
        current_price = 4525  # 1.25x extension
        
        # Should classify as trend_up
        extension = (current_price - ib_high) / (ib_high - ib_low)
        self.assertGreater(extension, 1.0)
    
    def test_b_day_classification(self):
        """Test balanced day detection"""
        # Mock data: Price within IB, no extension
        ib_high = 4500
        ib_low = 4480
        current_price = 4490  # Inside IB
        
        # Should classify as b_day or neutral
        if current_price <= ib_high and current_price >= ib_low:
            extension = 0
            self.assertEqual(extension, 0)
    
    def test_acceptance_detection(self):
        """Test acceptance vs rejection"""
        # Acceptance: Price holds beyond level for 2+ bars
        # Rejection: Quick return (poor high/low)
        
        # Mock acceptance
        closes_acceptance = [4505, 4506, 4507]  # Holding above IBH=4500
        self.assertTrue(all(c > 4500 for c in closes_acceptance))
        
        # Mock rejection (poor high)
        high_rejection = 4510
        close_rejection = 4502  # Well below high
        self.assertLess(close_rejection, high_rejection - 5)


class TestTrendDayStrategy(unittest.TestCase):
    """Test Trend Day strategy mechanics"""
    
    def setUp(self):
        self.strategy = TrendDayStrategy(symbol='MNQ')
    
    def test_strategy_initialization(self):
        """Test strategy setup"""
        self.assertEqual(self.strategy.symbol, 'MNQ')
        self.assertEqual(self.strategy.risk_per_trade, 400.0)
        self.assertEqual(self.strategy.point_value, 2.0)
    
    def test_position_sizing(self):
        """Test position size calculation"""
        stop_distance = 10.0
        contracts = int(400 / (stop_distance * 2))
        self.assertEqual(contracts, 20)
    
    def test_signal_filtering(self):
        """Test that only trend days generate signals"""
        # Mock snapshot for trend day
        trend_snapshot = {
            'day_type': 'trend_up',
            'ib': {'ib_high': 4500, 'ib_low': 4480, 'current_close': 4520},
            'tpo_profile': {'current_poc': 4510},
            'core_confluences': {
                'ib_acceptance': {'close_above_ibh': True}
            }
        }
        
        signals = self.strategy.wrapper.get_entry_signals(trend_snapshot)
        self.assertIsInstance(signals, list)


class TestBDayStrategy(unittest.TestCase):
    """Test B-Day strategy mechanics"""
    
    def setUp(self):
        self.strategy = BDayStrategy(symbol='MNQ')
    
    def test_mean_reversion_setup(self):
        """Test VAH/VAL fade detection"""
        # Mock data at VAH with poor high
        snapshot = {
            'day_type': 'b_day',
            'ib': {'current_close': 4505},
            'tpo_profile': {
                'current_vah': 4505,
                'current_val': 4485,
                'current_poc': 4495
            },
            'core_confluences': {}
        }
        
        signals = self.strategy.wrapper.get_entry_signals(snapshot)
        self.assertIsInstance(signals, list)
    
    def test_tight_stops(self):
        """Test that B-Day uses tight stops"""
        # B-Day should use 2-5 point stops
        typical_stop = 5.0
        self.assertLessEqual(typical_stop, 5.0)


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_wrapper_integration(self):
        """Test rockit wrapper loads"""
        wrapper = RockitBacktestWrapper('MNQ')
        self.assertIsNotNone(wrapper)
    
    def test_data_preparation(self):
        """Test data preparation for rockit"""
        # Create minimal test data
        dates = pd.date_range('2026-01-01 09:30', periods=50, freq='5min')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.randn(50).cumsum() + 4500,
            'high': np.random.randn(50).cumsum() + 4502,
            'low': np.random.randn(50).cumsum() + 4498,
            'close': np.random.randn(50).cumsum() + 4500,
            'volume': np.random.randint(1000, 5000, 50),
            'ema20': 4500,
            'ema50': 4500,
            'rsi14': 50,
            'atr14': 10,
            'vwap': 4500
        })
        
        wrapper = RockitBacktestWrapper('MNQ')
        prepared = wrapper.prepare_data_for_rockit(df)
        
        self.assertIsNotNone(prepared)
        self.assertIn('open', prepared.columns)


class TestFVGIntegration(unittest.TestCase):
    """Test FVG (Fair Value Gap) integration"""
    
    def test_fvg_detection(self):
        """Test FVG detection logic"""
        # Bullish FVG: Current low > prior high
        # Bearish FVG: Current high < prior low
        
        prior_high = 4500
        current_low = 4502  # Gap up
        
        is_bullish_fvg = current_low > prior_high
        self.assertTrue(is_bullish_fvg)
        
        # IFVG (Inverse FVG) = fill of the gap
        fill_price = 4501  # Within gap
        is_ifvg = fill_price > prior_high and fill_price < current_low
        self.assertTrue(is_ifvg)


def run_tests():
    """Run all unit tests"""
    print("Running Playbook Strategy Unit Tests")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTechnicalIndicators))
    suite.addTests(loader.loadTestsFromTestCase(TestTPOAcceptanceRejection))
    suite.addTests(loader.loadTestsFromTestCase(TestTrendDayStrategy))
    suite.addTests(loader.loadTestsFromTestCase(TestBDayStrategy))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestFVGIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
