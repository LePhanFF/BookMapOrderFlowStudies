"""
Unified Trade dataclass.
Replaces the separate Trade definitions scattered across the codebase.
All P&L fields are in DOLLARS (not points).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Trade:
    """A completed trade with full cost accounting."""

    # Identification
    strategy_name: str = ""
    setup_type: str = ""
    day_type: str = ""
    trend_strength: str = ""
    session_date: str = ""

    # Timing
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    bars_held: int = 0

    # Direction and size
    direction: str = ""         # 'LONG' or 'SHORT'
    contracts: int = 0

    # Prices
    signal_price: float = 0.0   # Price at signal generation
    entry_price: float = 0.0    # Actual fill price after slippage
    exit_price: float = 0.0     # Actual exit fill price after slippage
    stop_price: float = 0.0
    target_price: float = 0.0

    # P&L (all in dollars)
    gross_pnl: float = 0.0      # Before costs
    commission: float = 0.0     # Round-trip commission
    slippage_cost: float = 0.0  # Round-trip slippage cost
    net_pnl: float = 0.0       # gross_pnl - commission - slippage_cost

    # Exit
    exit_reason: str = ""       # 'STOP', 'TARGET', 'TIME', 'EOD', 'DAILY_LOSS', 'VWAP_BREACH_PM'

    # Metadata
    metadata: dict = field(default_factory=dict)

    @property
    def is_winner(self) -> bool:
        return self.net_pnl > 0

    @property
    def risk_points(self) -> float:
        """Distance from entry to stop in points."""
        if self.direction == 'LONG':
            return abs(self.entry_price - self.stop_price)
        else:
            return abs(self.stop_price - self.entry_price)

    @property
    def reward_points(self) -> float:
        """Distance from entry to target in points."""
        if self.direction == 'LONG':
            return abs(self.target_price - self.entry_price)
        else:
            return abs(self.entry_price - self.target_price)

    @property
    def r_multiple(self) -> float:
        """Actual R-multiple achieved (net P&L / risk)."""
        risk = self.risk_points
        if risk == 0:
            return 0.0
        if self.direction == 'LONG':
            actual_points = self.exit_price - self.entry_price
        else:
            actual_points = self.entry_price - self.exit_price
        return actual_points / risk
