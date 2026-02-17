"""
Signal dataclass - what strategies emit.
Signals are NOT trades. The engine decides whether to execute.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Signal:
    """
    A trading signal emitted by a strategy.

    The backtest engine receives signals and decides whether to execute
    based on risk limits, filters, and position state.
    """
    timestamp: datetime
    direction: str              # 'LONG' or 'SHORT'
    entry_price: float          # Desired entry price (before slippage)
    stop_price: float
    target_price: float
    strategy_name: str
    setup_type: str             # e.g., 'IBH_RETEST', 'EMA_PULLBACK', 'VAH_FADE'
    day_type: str
    trend_strength: str = ""
    confidence: str = 'medium'  # 'low', 'medium', 'high'
    pyramid_level: int = 0      # 0 = initial entry, 1+ = pyramid add
    metadata: dict = field(default_factory=dict)

    @property
    def risk_points(self) -> float:
        """Risk in points from entry to stop."""
        return abs(self.entry_price - self.stop_price)

    @property
    def reward_points(self) -> float:
        """Reward in points from entry to target."""
        return abs(self.target_price - self.entry_price)

    @property
    def risk_reward_ratio(self) -> float:
        """Planned R:R ratio."""
        risk = self.risk_points
        return self.reward_points / risk if risk > 0 else 0.0
