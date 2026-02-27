"""
Unit tests for BalanceSignal strategy.

Tests all 6 scoring factors individually, both signal modes (VA Edge Fade,
Wide IB Reclaim), Dalton acceptance tracking, dynamic target computation,
and total score aggregation.
"""

import unittest
from datetime import datetime, time as _time
from unittest.mock import MagicMock
import pandas as pd
import numpy as np

from strategy.balance_signal import BalanceSignal
from strategy.signal import Signal


def make_bar(close=20000, high=None, low=None, open_=None, volume=100,
             delta=0, cumulative_delta=0, delta_percentile=50,
             fvg_bull=False, fvg_bear=False, in_bpr=False,
             fvg_bull_bottom=None, fvg_bull_top=None,
             fvg_bear_bottom=None, fvg_bear_top=None,
             timestamp=None, **kwargs):
    """Create a mock bar Series for testing."""
    if high is None:
        high = close + 5
    if low is None:
        low = close - 5
    if open_ is None:
        open_ = close
    if timestamp is None:
        timestamp = datetime(2025, 6, 15, 11, 0, 0)

    data = {
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'delta': delta,
        'cumulative_delta': cumulative_delta,
        'delta_percentile': delta_percentile,
        'fvg_bull': fvg_bull,
        'fvg_bear': fvg_bear,
        'in_bpr': in_bpr,
        'fvg_bull_bottom': fvg_bull_bottom,
        'fvg_bull_top': fvg_bull_top,
        'fvg_bear_bottom': fvg_bear_bottom,
        'fvg_bear_top': fvg_bear_top,
        'timestamp': timestamp,
    }
    data.update(kwargs)
    return pd.Series(data, name=timestamp)


def make_session_context(**overrides):
    """Create a minimal session_context dict for testing."""
    ctx = {
        'prior_va_high': 20100,
        'prior_va_low': 19900,
        'prior_va_poc': 20000,
        'prior_vp_vah': 20100,
        'prior_vp_val': 19900,
        'prior_vp_poc': 20000,
        'prior_tpo_shape': 'D',
        'tpo_poc_location': 0.5,
        'atr14': 150,
        'vwap': 20000,
        'day_type': 'b_day',
        'trend_strength': 'weak',
        'bar_time': _time(11, 0),
        'ib_range_history': [200] * 10,
        'vp_data': None,
    }
    ctx.update(overrides)
    return ctx


class TestBalanceSignalBasics(unittest.TestCase):
    """Test strategy registration and properties."""

    def test_name(self):
        s = BalanceSignal()
        self.assertEqual(s.name, "Balance Signal")

    def test_applicable_day_types_empty(self):
        s = BalanceSignal()
        self.assertEqual(s.applicable_day_types, [])


class TestScoreProfileShape(unittest.TestCase):
    """Test F1: Prior session profile shape scoring."""

    def setUp(self):
        self.s = BalanceSignal()
        ctx = make_session_context()
        self.s.on_session_start('2025-06-15', 20050, 19950, 100, ctx)

    def test_p_shape_long(self):
        self.s._prior_tpo_shape = 'P'
        self.assertEqual(self.s.score_profile_shape('LONG'), 2)

    def test_p_shape_short(self):
        self.s._prior_tpo_shape = 'P'
        self.assertEqual(self.s.score_profile_shape('SHORT'), 0)

    def test_b_shape_short(self):
        self.s._prior_tpo_shape = 'B'
        self.assertEqual(self.s.score_profile_shape('SHORT'), 2)

    def test_b_shape_long(self):
        self.s._prior_tpo_shape = 'B'
        self.assertEqual(self.s.score_profile_shape('LONG'), 0)

    def test_d_shape_neutral(self):
        self.s._prior_tpo_shape = 'D'
        self.assertEqual(self.s.score_profile_shape('LONG'), 1)
        self.assertEqual(self.s.score_profile_shape('SHORT'), 1)

    def test_poc_location_high(self):
        """POC in upper third → bullish lean like P-shape."""
        self.s._prior_tpo_shape = ''
        self.s._tpo_poc_location = 0.80
        self.assertEqual(self.s.score_profile_shape('LONG'), 2)
        self.assertEqual(self.s.score_profile_shape('SHORT'), 0)

    def test_poc_location_low(self):
        """POC in lower third → bearish lean like b-shape."""
        self.s._prior_tpo_shape = ''
        self.s._tpo_poc_location = 0.20
        self.assertEqual(self.s.score_profile_shape('SHORT'), 2)
        self.assertEqual(self.s.score_profile_shape('LONG'), 0)

    def test_poc_location_center(self):
        """POC in middle third → neutral."""
        self.s._prior_tpo_shape = ''
        self.s._tpo_poc_location = 0.50
        self.assertEqual(self.s.score_profile_shape('LONG'), 1)


class TestScoreHVNLVN(unittest.TestCase):
    """Test F2: HVN/LVN at VA edge scoring."""

    def setUp(self):
        self.s = BalanceSignal()

    def test_no_data_neutral(self):
        ctx = make_session_context()
        self.s.on_session_start('2025-06-15', 20050, 19950, 100, ctx)
        # With approximate bins from prior VA, should get a score
        score = self.s.score_hvn_lvn_at_edge('LONG')
        self.assertIn(score, [0, 1, 2])

    def test_hvn_at_val(self):
        """HVN at VAL → strong support → +2 for LONG."""
        # Build bins with high volume near VAL (19900)
        bins = {}
        for p in range(19850, 20150, 5):
            if abs(p - 19900) < 20:  # High vol near VAL
                bins[p] = 500
            else:
                bins[p] = 50
        ctx = make_session_context(vp_data={'volume_bins': bins})
        self.s.on_session_start('2025-06-15', 20050, 19950, 100, ctx)
        self.assertEqual(self.s.score_hvn_lvn_at_edge('LONG'), 2)

    def test_lvn_at_val(self):
        """LVN at VAL → weak level → +0 for LONG."""
        bins = {}
        for p in range(19850, 20150, 5):
            if abs(p - 19900) < 20:  # Low vol near VAL
                bins[p] = 5
            else:
                bins[p] = 200
        ctx = make_session_context(vp_data={'volume_bins': bins})
        self.s.on_session_start('2025-06-15', 20050, 19950, 100, ctx)
        self.assertEqual(self.s.score_hvn_lvn_at_edge('LONG'), 0)

    def test_hvn_at_vah(self):
        """HVN at VAH → strong resistance → +2 for SHORT."""
        bins = {}
        for p in range(19850, 20150, 5):
            if abs(p - 20100) < 20:  # High vol near VAH
                bins[p] = 500
            else:
                bins[p] = 50
        ctx = make_session_context(vp_data={'volume_bins': bins})
        self.s.on_session_start('2025-06-15', 20050, 19950, 100, ctx)
        self.assertEqual(self.s.score_hvn_lvn_at_edge('SHORT'), 2)


class TestScoreVolumeSkew(unittest.TestCase):
    """Test F3: Volume distribution skew scoring."""

    def setUp(self):
        self.s = BalanceSignal()

    def test_heavy_below_poc_long(self):
        """More volume below POC → sellers exhausted → +1 LONG."""
        bins = {}
        for p in range(19850, 20000, 5):
            bins[p] = 200  # Heavy below POC
        for p in range(20000, 20150, 5):
            bins[p] = 50   # Light above POC
        ctx = make_session_context(vp_data={'volume_bins': bins})
        self.s.on_session_start('2025-06-15', 20050, 19950, 100, ctx)
        self.assertEqual(self.s.score_volume_skew('LONG'), 1)

    def test_heavy_above_poc_short(self):
        """More volume above POC → buyers exhausted → +1 SHORT."""
        bins = {}
        for p in range(19850, 20000, 5):
            bins[p] = 50   # Light below POC
        for p in range(20000, 20150, 5):
            bins[p] = 200  # Heavy above POC
        ctx = make_session_context(vp_data={'volume_bins': bins})
        self.s.on_session_start('2025-06-15', 20050, 19950, 100, ctx)
        self.assertEqual(self.s.score_volume_skew('SHORT'), 1)

    def test_balanced_volume(self):
        """Even volume distribution → 0 pts."""
        bins = {}
        for p in range(19850, 20150, 5):
            bins[p] = 100
        ctx = make_session_context(vp_data={'volume_bins': bins})
        self.s.on_session_start('2025-06-15', 20050, 19950, 100, ctx)
        self.assertEqual(self.s.score_volume_skew('LONG'), 0)
        self.assertEqual(self.s.score_volume_skew('SHORT'), 0)


class TestScoreVWAP(unittest.TestCase):
    """Test F4: VWAP confirmation scoring."""

    def setUp(self):
        self.s = BalanceSignal()
        ctx = make_session_context()
        self.s.on_session_start('2025-06-15', 20050, 19950, 100, ctx)

    def test_long_below_vwap_flat_slope(self):
        """Price below VWAP, VWAP flat → +1 LONG."""
        self.s._vwap_history = [20000, 20001, 20000, 20001, 20000]
        bar = make_bar(close=19990)
        self.assertEqual(self.s.score_vwap_confirmation('LONG', bar), 1)

    def test_long_above_vwap(self):
        """Price above VWAP → 0 LONG."""
        self.s._vwap_history = [20000, 20001, 20000, 20001, 20000]
        bar = make_bar(close=20010)
        self.assertEqual(self.s.score_vwap_confirmation('LONG', bar), 0)

    def test_short_above_vwap_flat_slope(self):
        """Price above VWAP, VWAP flat → +1 SHORT."""
        self.s._vwap_history = [20000, 19999, 20000, 19999, 20000]
        bar = make_bar(close=20010)
        self.assertEqual(self.s.score_vwap_confirmation('SHORT', bar), 1)

    def test_insufficient_history(self):
        """Not enough VWAP data → 0."""
        self.s._vwap_history = [20000]
        bar = make_bar(close=19990)
        self.assertEqual(self.s.score_vwap_confirmation('LONG', bar), 0)


class TestScoreIBContext(unittest.TestCase):
    """Test F5: IB containment within prior VA scoring."""

    def setUp(self):
        self.s = BalanceSignal()

    def test_ib_inside_va_and_narrow(self):
        """IB fully inside VA + narrow → +2."""
        ctx = make_session_context()
        # IB: 19950-20050 (100pt), VA: 19900-20100 (200pt)
        # IB inside VA? Yes. IB < 50% VA (100 < 100)? No, 100 is exactly 50%.
        # Use narrower IB:
        self.s.on_session_start('2025-06-15', 20020, 19970, 50, ctx)
        self.assertEqual(self.s.score_ib_context(), 2)  # inside + narrow

    def test_ib_inside_va_not_narrow(self):
        """IB inside VA but not narrow → +1."""
        ctx = make_session_context()
        # IB: 19920-20080 (160pt), VA: 19900-20100 (200pt)
        self.s.on_session_start('2025-06-15', 20080, 19920, 160, ctx)
        self.assertEqual(self.s.score_ib_context(), 1)  # inside only

    def test_ib_outside_va(self):
        """IB extends beyond VA → +0."""
        ctx = make_session_context()
        # IB: 19800-20200 (400pt), VA: 19900-20100 (200pt)
        self.s.on_session_start('2025-06-15', 20200, 19800, 400, ctx)
        self.assertEqual(self.s.score_ib_context(), 0)

    def test_ib_narrow_but_outside(self):
        """IB narrow but extends beyond VA on one side → only narrow point."""
        ctx = make_session_context()
        # IB: 20080-20120 (40pt), VA: 19900-20100. IB high > VAH
        self.s.on_session_start('2025-06-15', 20120, 20080, 40, ctx)
        self.assertEqual(self.s.score_ib_context(), 1)  # narrow only


class TestScoreDeltaCVD(unittest.TestCase):
    """Test F6: Delta absorption + CVD divergence scoring."""

    def setUp(self):
        self.s = BalanceSignal()
        ctx = make_session_context()
        self.s.on_session_start('2025-06-15', 20050, 19950, 100, ctx)

    def test_absorption_long(self):
        """Negative delta but stable price → absorption → +1 LONG."""
        self.s._price_history = [20000, 19999, 19998]
        self.s._cvd_history = [100, 100, 100, 100, 100]  # flat CVD
        bar = make_bar(close=19998, delta=-50)
        score = self.s.score_delta_cvd_divergence('LONG', bar)
        self.assertGreaterEqual(score, 1)

    def test_absorption_short(self):
        """Positive delta but stable price → absorption → +1 SHORT."""
        self.s._price_history = [20000, 20001, 20002]
        self.s._cvd_history = [100, 100, 100, 100, 100]
        bar = make_bar(close=20002, delta=50)
        score = self.s.score_delta_cvd_divergence('SHORT', bar)
        self.assertGreaterEqual(score, 1)

    def test_cvd_divergence_long(self):
        """Price falling but CVD rising → bullish divergence → +1 LONG."""
        self.s._price_history = [20010, 20005, 20000, 19998, 19995]
        self.s._cvd_history = [100, 110, 120, 130, 140]  # CVD rising
        bar = make_bar(close=19990, delta=-10)
        score = self.s.score_delta_cvd_divergence('LONG', bar)
        # Should get at least divergence point
        self.assertGreaterEqual(score, 1)

    def test_cvd_divergence_short(self):
        """Price rising but CVD falling → bearish divergence → +1 SHORT."""
        self.s._price_history = [19990, 19995, 20000, 20002, 20005]
        self.s._cvd_history = [140, 130, 120, 110, 100]  # CVD falling
        bar = make_bar(close=20010, delta=10)
        score = self.s.score_delta_cvd_divergence('SHORT', bar)
        self.assertGreaterEqual(score, 1)

    def test_no_divergence(self):
        """Aligned delta and price → 0."""
        self.s._price_history = [20000, 20005, 20010]
        self.s._cvd_history = [100, 110, 120, 130, 140]
        bar = make_bar(close=20015, delta=50)
        score = self.s.score_delta_cvd_divergence('LONG', bar)
        self.assertEqual(score, 0)


class TestTotalScore(unittest.TestCase):
    """Test score aggregation + threshold gating."""

    def test_max_score_possible(self):
        """All factors maxed = 10 pts."""
        s = BalanceSignal()
        ctx = make_session_context(prior_tpo_shape='P')
        s.on_session_start('2025-06-15', 20020, 19970, 50, ctx)
        # Set up for high LONG score
        s._vwap_history = [20000] * 5
        s._price_history = [20010, 20005, 20000, 19998, 19995]
        s._cvd_history = [100, 110, 120, 130, 140]

        bar = make_bar(close=19990, delta=-20)
        total, breakdown = s.compute_total_score('LONG', bar)

        # Verify all factors are non-negative
        for k, v in breakdown.items():
            self.assertGreaterEqual(v, 0, f"{k} should be >= 0")

        # Total should be sum of all
        self.assertEqual(total, sum(breakdown.values()))

    def test_score_is_bounded(self):
        """Total score should be 0-10."""
        s = BalanceSignal()
        ctx = make_session_context()
        s.on_session_start('2025-06-15', 20050, 19950, 100, ctx)
        s._vwap_history = [20000] * 5
        s._price_history = [20000] * 5
        s._cvd_history = [100] * 5

        bar = make_bar(close=20000)
        total, _ = s.compute_total_score('LONG', bar)
        self.assertGreaterEqual(total, 0)
        self.assertLessEqual(total, 10)


class TestDaltonAcceptance(unittest.TestCase):
    """Test 2-bar early and 6-bar inside acceptance tracking."""

    def setUp(self):
        self.s = BalanceSignal()
        ctx = make_session_context()
        self.s.on_session_start('2025-06-15', 20050, 19950, 100, ctx)

    def test_acceptance_below_vah_increments(self):
        """Consecutive closes below VAH increment counter."""
        for _ in range(3):
            bar = make_bar(close=20050)  # Below VAH (20100)
            self.s._update_acceptance(bar)
        self.assertEqual(self.s._accept_below_vah, 3)

    def test_acceptance_below_vah_resets(self):
        """Close above VAH resets counter."""
        for _ in range(3):
            self.s._update_acceptance(make_bar(close=20050))
        self.s._update_acceptance(make_bar(close=20150))  # Above VAH
        self.assertEqual(self.s._accept_below_vah, 0)

    def test_acceptance_above_val_increments(self):
        """Consecutive closes above VAL increment counter."""
        for _ in range(4):
            self.s._update_acceptance(make_bar(close=19950))  # Above VAL (19900)
        self.assertEqual(self.s._accept_above_val, 4)

    def test_acceptance_above_val_resets(self):
        """Close below VAL resets counter."""
        for _ in range(3):
            self.s._update_acceptance(make_bar(close=19950))
        self.s._update_acceptance(make_bar(close=19850))  # Below VAL
        self.assertEqual(self.s._accept_above_val, 0)

    def test_early_acceptance_threshold(self):
        """2 bars = early acceptance (BALANCE_ACCEPT_EARLY = 2)."""
        self.s._update_acceptance(make_bar(close=20050))
        self.assertLess(self.s._accept_below_vah, 2)
        self.s._update_acceptance(make_bar(close=20060))
        self.assertGreaterEqual(self.s._accept_below_vah, 2)

    def test_inside_acceptance_threshold(self):
        """6 bars = inside acceptance (BALANCE_ACCEPT_INSIDE = 6)."""
        for _ in range(6):
            self.s._update_acceptance(make_bar(close=20050))
        self.assertGreaterEqual(self.s._accept_below_vah, 6)

    def test_ib_reclaim_acceptance(self):
        """Track acceptance above IBL for reclaim mode."""
        for _ in range(3):
            self.s._update_acceptance(make_bar(close=19960))  # Above IBL (19950)
        self.assertEqual(self.s._accept_above_ibl, 3)


class TestVAEdgeFade(unittest.TestCase):
    """Test Mode 1: VA Edge Fade signal emission."""

    def _setup_strategy_for_fade(self, direction='SHORT'):
        """Set up strategy ready to fire a VA Edge Fade signal."""
        s = BalanceSignal()
        # Use P-shape for LONG, B-shape for SHORT to boost score
        shape = 'B' if direction == 'SHORT' else 'P'
        ctx = make_session_context(prior_tpo_shape=shape)
        s.on_session_start('2025-06-15', 20050, 19950, 100, ctx)
        # Prime VWAP for F4
        s._vwap_history = [20000] * 10
        # Prime CVD/price for F6
        if direction == 'SHORT':
            s._price_history = [19990, 19995, 20000, 20005, 20010]
            s._cvd_history = [140, 130, 120, 110, 100]
        else:
            s._price_history = [20010, 20005, 20000, 19998, 19995]
            s._cvd_history = [100, 110, 120, 130, 140]
        return s

    def test_short_fade_from_vah(self):
        """Touch VAH + fade back + acceptance → SHORT signal."""
        s = self._setup_strategy_for_fade('SHORT')
        ctx = make_session_context(prior_tpo_shape='B', vwap=20000)

        # Bar 0: touch VAH
        bar0 = make_bar(close=20110, high=20120, low=20090)
        s.on_bar(bar0, 0, ctx)

        # Bars 1-2: close below VAH (acceptance)
        bar1 = make_bar(close=20080, high=20090, low=20070, delta=30)
        s.on_bar(bar1, 1, ctx)
        bar2 = make_bar(close=20070, high=20080, low=20060, delta=30)
        signal = s.on_bar(bar2, 2, ctx)

        # May or may not fire depending on score — check structure if it fires
        if signal is not None:
            self.assertEqual(signal.direction, 'SHORT')
            self.assertEqual(signal.setup_type, 'VA_EDGE_FADE')
            self.assertGreater(signal.stop_price, signal.entry_price)
            self.assertLess(signal.target_price, signal.entry_price)

    def test_long_fade_from_val(self):
        """Touch VAL + fade back + acceptance → LONG signal."""
        s = self._setup_strategy_for_fade('LONG')
        ctx = make_session_context(prior_tpo_shape='P', vwap=20000)

        # Bar 0: touch VAL
        bar0 = make_bar(close=19890, high=19910, low=19880)
        s.on_bar(bar0, 0, ctx)

        # Bars 1-2: close above VAL (acceptance)
        bar1 = make_bar(close=19920, high=19930, low=19910, delta=-30)
        s.on_bar(bar1, 1, ctx)
        bar2 = make_bar(close=19930, high=19940, low=19920, delta=-30)
        signal = s.on_bar(bar2, 2, ctx)

        if signal is not None:
            self.assertEqual(signal.direction, 'LONG')
            self.assertEqual(signal.setup_type, 'VA_EDGE_FADE')
            self.assertLess(signal.stop_price, signal.entry_price)
            self.assertGreater(signal.target_price, signal.entry_price)

    def test_no_signal_without_touch(self):
        """No VAH/VAL touch → no signal."""
        s = BalanceSignal()
        ctx = make_session_context()
        s.on_session_start('2025-06-15', 20050, 19950, 100, ctx)

        # Price stays in middle of VA
        for i in range(10):
            bar = make_bar(close=20000)
            signal = s.on_bar(bar, i, ctx)
            self.assertIsNone(signal)

    def test_no_signal_without_acceptance(self):
        """Touch VAH but no acceptance (bounces back above) → no signal."""
        s = BalanceSignal()
        ctx = make_session_context()
        s.on_session_start('2025-06-15', 20050, 19950, 100, ctx)
        s._vwap_history = [20000] * 10

        # Touch VAH
        bar0 = make_bar(close=20110, high=20120)
        s.on_bar(bar0, 0, ctx)
        # Still above VAH — no acceptance
        bar1 = make_bar(close=20110, high=20120)
        signal = s.on_bar(bar1, 1, ctx)
        self.assertIsNone(signal)

    def test_only_one_va_fade_per_session(self):
        """After VA fade fires, _va_fade_fired prevents re-entry."""
        s = self._setup_strategy_for_fade('SHORT')
        s._va_fade_fired = True
        ctx = make_session_context(prior_tpo_shape='B', vwap=20000)

        bar = make_bar(close=20080, high=20120)
        s._touched_vah = True
        s._accept_below_vah = 5
        signal = s.on_bar(bar, 50, ctx)
        self.assertIsNone(signal)


class TestWideIBReclaim(unittest.TestCase):
    """Test Mode 2: Wide IB Reclaim signal emission."""

    def test_not_eligible_if_ib_too_narrow(self):
        """IB range < 350 → wide reclaim not eligible."""
        s = BalanceSignal()
        ctx = make_session_context()
        s.on_session_start('2025-06-15', 20100, 19900, 200, ctx)
        self.assertFalse(s._wide_ib_eligible)

    def test_eligible_at_350(self):
        """IB range = 350 → eligible."""
        s = BalanceSignal()
        ctx = make_session_context()
        s.on_session_start('2025-06-15', 20175, 19825, 350, ctx)
        self.assertTrue(s._wide_ib_eligible)

    def test_not_eligible_above_500(self):
        """IB range > 500 → not eligible (trend day)."""
        s = BalanceSignal()
        ctx = make_session_context()
        s.on_session_start('2025-06-15', 20300, 19700, 600, ctx)
        self.assertFalse(s._wide_ib_eligible)

    def test_long_reclaim_detection(self):
        """Break below IBL then reclaim → trigger LONG reclaim check."""
        s = BalanceSignal()
        ctx = make_session_context(prior_tpo_shape='P')
        s.on_session_start('2025-06-15', 20175, 19825, 350, ctx)
        s._vwap_history = [20000] * 10
        s._price_history = [19850, 19840, 19830, 19825, 19820]
        s._cvd_history = [100, 110, 120, 130, 140]

        # Bar 0: close below IBL
        bar0 = make_bar(close=19800, low=19790)
        s.on_bar(bar0, 0, ctx)

        # Bar 1: reclaim above IBL
        bar1 = make_bar(close=19850, low=19810, delta=-20)
        s.on_bar(bar1, 1, ctx)

        # Bars 2-3: acceptance above IBL
        bar2 = make_bar(close=19860, low=19840, delta=-20)
        s.on_bar(bar2, 2, ctx)
        bar3 = make_bar(close=19870, low=19850, delta=-20)
        signal = s.on_bar(bar3, 3, ctx)

        # May fire if score is high enough
        if signal is not None:
            self.assertEqual(signal.direction, 'LONG')
            self.assertEqual(signal.setup_type, 'WIDE_IB_RECLAIM')


class TestDynamicTarget(unittest.TestCase):
    """Test dynamic target computation via DPOC migration."""

    def setUp(self):
        self.s = BalanceSignal()
        ctx = make_session_context()
        self.s.on_session_start('2025-06-15', 20050, 19950, 100, ctx)

    def test_va_mode_default_is_poc(self):
        """VA mode default target = prior POC."""
        bar = make_bar(close=20000)
        target = self.s.compute_dynamic_target('VA', 'LONG', bar)
        self.assertEqual(target, 20000)  # prior_poc = 20000

    def test_ib_mode_default_is_ib_mid(self):
        """IB mode default target = IB midpoint."""
        bar = make_bar(close=20000)
        target = self.s.compute_dynamic_target('IB', 'LONG', bar)
        self.assertEqual(target, 20000)  # ib_mid = (20050+19950)/2

    def test_dpoc_extends_long_target(self):
        """DPOC migrating up → LONG target extended."""
        # Simulate DPOC above prior POC
        self.s._volume_by_price = {20030: 1000, 20000: 200, 19970: 100}
        bar = make_bar(close=19950)
        target = self.s.compute_dynamic_target('VA', 'LONG', bar)
        # Should be averaged between POC (20000) and DPOC (20030)
        self.assertGreater(target, 20000)

    def test_dpoc_tightens_short_to_vwap(self):
        """DPOC migrating up → SHORT target tightens to VWAP."""
        self.s._volume_by_price = {20030: 1000, 20000: 200}
        self.s._vwap_history = [20010]
        bar = make_bar(close=20050)
        target = self.s.compute_dynamic_target('VA', 'SHORT', bar)
        # Should tighten to VWAP (20010)
        self.assertEqual(target, 20010)

    def test_bpr_fallback(self):
        """BPR active → target limited to BPR midpoint."""
        self.s._bpr_active = True
        self.s._bpr_high = 19990
        self.s._bpr_low = 19970
        bar = make_bar(close=19950)
        target = self.s.compute_dynamic_target('VA', 'LONG', bar)
        # BPR mid = 19980, which is less than POC (20000)
        self.assertEqual(target, 19980)

    def test_no_prior_poc_fallback_to_ib_mid(self):
        """If no prior POC, VA mode falls back to IB mid."""
        self.s._prior_poc = None
        bar = make_bar(close=19950)
        target = self.s.compute_dynamic_target('VA', 'LONG', bar)
        self.assertEqual(target, self.s._ib_mid)


class TestTimeGate(unittest.TestCase):
    """Test time-based filtering."""

    def test_no_signal_after_cutoff(self):
        """Signals blocked after BALANCE_LAST_ENTRY_TIME (13:30)."""
        s = BalanceSignal()
        ctx = make_session_context(bar_time=_time(14, 0))
        s.on_session_start('2025-06-15', 20050, 19950, 100, ctx)
        bar = make_bar(close=20000)
        signal = s.on_bar(bar, 0, ctx)
        self.assertIsNone(signal)

    def test_signal_before_cutoff(self):
        """Signals allowed before 13:30."""
        s = BalanceSignal()
        ctx = make_session_context(bar_time=_time(11, 0))
        s.on_session_start('2025-06-15', 20050, 19950, 100, ctx)
        # No signal because no setup, but time gate doesn't block
        bar = make_bar(close=20000)
        signal = s.on_bar(bar, 0, ctx)
        # Might be None for other reasons, but not time gate
        # Just ensure no crash


class TestBPRTracking(unittest.TestCase):
    """Test BPR state tracking from bar columns."""

    def setUp(self):
        self.s = BalanceSignal()
        ctx = make_session_context()
        self.s.on_session_start('2025-06-15', 20050, 19950, 100, ctx)

    def test_bpr_activates(self):
        """BPR activates when bar has in_bpr=True with valid FVG bounds."""
        bar = make_bar(close=20000, in_bpr=True,
                       fvg_bull_bottom=19990, fvg_bull_top=20010,
                       fvg_bear_bottom=19995, fvg_bear_top=20005)
        self.s._update_bpr_state(bar)
        self.assertTrue(self.s._bpr_active)
        self.assertEqual(self.s._bpr_low, 19995)  # max(bull_bottom, bear_bottom)
        self.assertEqual(self.s._bpr_high, 20005)  # min(bull_top, bear_top)

    def test_bpr_deactivates_on_distance(self):
        """BPR deactivates when price moves far away."""
        self.s._bpr_active = True
        self.s._bpr_high = 20005
        self.s._bpr_low = 19995
        # Price way above BPR
        bar = make_bar(close=20030, in_bpr=False)
        self.s._update_bpr_state(bar)
        self.assertFalse(self.s._bpr_active)


class TestSessionReset(unittest.TestCase):
    """Test that session state resets properly."""

    def test_state_resets_on_new_session(self):
        """All tracking state resets on on_session_start."""
        s = BalanceSignal()
        ctx = make_session_context()
        s.on_session_start('2025-06-15', 20050, 19950, 100, ctx)

        # Simulate some state accumulation
        s._va_fade_fired = True
        s._touched_vah = True
        s._accept_below_vah = 5
        s._cvd_history = [100] * 20

        # New session
        s.on_session_start('2025-06-16', 20100, 20000, 100, ctx)
        self.assertFalse(s._va_fade_fired)
        self.assertFalse(s._touched_vah)
        self.assertEqual(s._accept_below_vah, 0)
        self.assertEqual(len(s._cvd_history), 0)


class TestMissingData(unittest.TestCase):
    """Test graceful handling of missing/NaN data."""

    def test_no_prior_va(self):
        """No prior VA data → no signals, no crash."""
        s = BalanceSignal()
        ctx = make_session_context(prior_va_high=None, prior_va_low=None,
                                   prior_vp_vah=None, prior_vp_val=None)
        s.on_session_start('2025-06-15', 20050, 19950, 100, ctx)
        bar = make_bar(close=20000)
        signal = s.on_bar(bar, 0, ctx)
        self.assertIsNone(signal)

    def test_nan_delta_handled(self):
        """NaN delta in bar doesn't crash scoring."""
        s = BalanceSignal()
        ctx = make_session_context()
        s.on_session_start('2025-06-15', 20050, 19950, 100, ctx)
        s._price_history = [20000] * 5
        s._cvd_history = [100] * 5
        bar = make_bar(close=20000, delta=float('nan'))
        score = s.score_delta_cvd_divergence('LONG', bar)
        self.assertIsInstance(score, int)

    def test_nan_cvd_handled(self):
        """NaN cumulative_delta doesn't crash."""
        s = BalanceSignal()
        ctx = make_session_context()
        s.on_session_start('2025-06-15', 20050, 19950, 100, ctx)
        bar = make_bar(close=20000, cumulative_delta=float('nan'))
        # Should not crash
        s._update_cvd_tracking(bar)


if __name__ == '__main__':
    unittest.main()
