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
]

# Core strategies with demonstrated positive edge
# Trend Bear and SuperTrend Bear removed: 25% WR, negative expectancy
# NQ has strong long bias — short strategies consistently underperform
CORE_STRATEGIES = [
    TrendDayBull,
    SuperTrendBull,
    PDayStrategy,
    BDayStrategy,
]


def get_all_strategies():
    """Instantiate all 9 Dalton playbook strategies."""
    return [cls() for cls in ALL_STRATEGIES]


def get_core_strategies():
    """Instantiate only strategies with demonstrated positive edge."""
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
