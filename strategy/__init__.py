"""Strategy modules - one file per Dalton day-type strategy."""

from strategy.trend_bull import TrendDayBull
from strategy.trend_bear import TrendDayBear
from strategy.super_trend_bull import SuperTrendBull
from strategy.super_trend_bear import SuperTrendBear
from strategy.p_day import PDayStrategy
from strategy.b_day import BDayStrategy
from strategy.neutral_day import NeutralDayStrategy
from strategy.pm_morph import PMMorphStrategy
from strategy.morph_to_trend import MorphToTrendStrategy
from strategy.edge_fade import EdgeFadeStrategy
from strategy.bear_accept import BearAcceptShort
from strategy.ibh_sweep import IBHSweepFail
from strategy.or_reversal import OpeningRangeReversal

ALL_STRATEGIES = [
    TrendDayBull,
    TrendDayBear,
    SuperTrendBull,
    SuperTrendBear,
    PDayStrategy,
    BDayStrategy,
    NeutralDayStrategy,
    PMMorphStrategy,
    MorphToTrendStrategy,
    EdgeFadeStrategy,
    BearAcceptShort,
    IBHSweepFail,
    OpeningRangeReversal,
]

# v14 Report 7-Strategy Portfolio (matches 2026.02.19-63days_final_report.md)
# 1. Trend Day Bull     - VWAP pullback on trend_up/p_day (8 trades, 75% WR)
# 2. P-Day              - VWAP pullback on p_day (8 trades, 75% WR)
# 3. B-Day IBL Fade     - IBL fade on b_day (4 trades, 100% WR)
# 4. Edge Fade OPTIMIZED - Mean reversion with 3 filters (17 trades, 94% WR)
# 5. IBH Sweep+Fail     - IBH sweep fade on b_day (4 trades, 100% WR)
# 6. Bear Accept Short  - Acceptance short on trend_down (11 trades, 64% WR)
# 7. Opening Range Rev  - Judas Swing reversal (20 trades, 80% WR)
CORE_STRATEGIES = [
    TrendDayBull,
    PDayStrategy,
    BDayStrategy,
    EdgeFadeStrategy,
    IBHSweepFail,
    BearAcceptShort,
    OpeningRangeReversal,
]


def get_all_strategies():
    """Instantiate all Dalton playbook strategies."""
    return [cls() for cls in ALL_STRATEGIES]


def get_core_strategies():
    """Instantiate the v14 7-strategy portfolio."""
    return [cls() for cls in CORE_STRATEGIES]


def get_strategies_by_name(*names):
    """Instantiate strategies by name (case-insensitive partial match)."""
    name_map = {cls().name.lower(): cls for cls in ALL_STRATEGIES}
    result = []
    for name in names:
        key = name.lower()
        for sname, cls in name_map.items():
            if key in sname:
                result.append(cls())
                break
    return result
