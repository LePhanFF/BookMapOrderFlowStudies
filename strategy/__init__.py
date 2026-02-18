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

# New research-based strategies (Feb 2026 evaluation research)
from strategy.orb_vwap_breakout import ORBVwapBreakout
from strategy.liquidity_sweep import LiquiditySweep
from strategy.mean_reversion_vwap import MeanReversionVWAP
from strategy.ema_trend_follow import EMATrendFollow

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
    # Research-based strategies
    ORBVwapBreakout,
    LiquiditySweep,
    MeanReversionVWAP,
    EMATrendFollow,
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

# Research-based strategies for evaluation passing (Feb 2026)
# These complement the core Dalton strategies with community-validated approaches
RESEARCH_STRATEGIES = [
    ORBVwapBreakout,
    LiquiditySweep,
    MeanReversionVWAP,
    EMATrendFollow,
]


def get_all_strategies():
    """Instantiate all 9 Dalton playbook strategies."""
    return [cls() for cls in ALL_STRATEGIES]


def get_core_strategies():
    """Instantiate only strategies with demonstrated positive edge."""
    return [cls() for cls in CORE_STRATEGIES]


def get_research_strategies():
    """Instantiate research-based evaluation strategies."""
    return [cls() for cls in RESEARCH_STRATEGIES]


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
