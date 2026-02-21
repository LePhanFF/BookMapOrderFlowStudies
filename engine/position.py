"""
Position manager: tracks open positions, enforces daily loss limits,
max drawdown, and trade count limits.

FIXED: Old code skipped bars when daily loss hit but didn't close open positions.
This implementation force-closes everything when limits are breached.
"""

from typing import List, Dict, Optional
from datetime import datetime
from engine.trade import Trade
from config.constants import (
    DEFAULT_ACCOUNT_SIZE, DEFAULT_MAX_DRAWDOWN,
    DEFAULT_MAX_DAY_LOSS, DEFAULT_MAX_TRADES_PER_DAY,
    DEFAULT_MAX_CONTRACTS,
)


class OpenPosition:
    """An open (not yet closed) position."""

    def __init__(
        self,
        direction: str,
        entry_price: float,
        stop_price: float,
        target_price: float,
        contracts: int,
        entry_time: datetime,
        strategy_name: str,
        setup_type: str,
        day_type: str,
        trend_strength: str,
        session_date: str,
        pyramid_level: int = 0,
    ):
        self.direction = direction
        self.entry_price = entry_price
        self.stop_price = stop_price
        self.target_price = target_price
        self.contracts = contracts
        self.entry_time = entry_time
        self.strategy_name = strategy_name
        self.setup_type = setup_type
        self.day_type = day_type
        self.trend_strength = trend_strength
        self.session_date = session_date
        self.pyramid_level = pyramid_level
        self.bars_held = 0
        self.trailing_stop = stop_price
        self.breakeven_activated = False
        self.peak_favorable = 0.0  # Track peak MFE in points for R-based trailing
        self.risk_points = abs(entry_price - stop_price)  # Initial risk for R calculation

    def unrealized_pnl_points(self, current_price: float) -> float:
        """Current unrealized P&L in points."""
        if self.direction == 'LONG':
            return current_price - self.entry_price
        else:
            return self.entry_price - current_price

    def check_stop_hit(self, bar_low: float, bar_high: float) -> bool:
        """Check if stop was hit on this bar."""
        if self.direction == 'LONG':
            return bar_low <= self.trailing_stop
        else:
            return bar_high >= self.trailing_stop

    def check_target_hit(self, bar_low: float, bar_high: float) -> bool:
        """Check if target was hit on this bar."""
        if self.direction == 'LONG':
            return bar_high >= self.target_price
        else:
            return bar_low <= self.target_price

    def trail_to_breakeven(self):
        """Move stop to breakeven (entry price)."""
        if not self.breakeven_activated:
            if self.direction == 'LONG':
                self.trailing_stop = max(self.trailing_stop, self.entry_price)
            else:
                self.trailing_stop = min(self.trailing_stop, self.entry_price)
            self.breakeven_activated = True

    def update_peak_favorable(self, bar_high: float, bar_low: float):
        """Update peak MFE from current bar."""
        if self.direction == 'LONG':
            favorable = bar_high - self.entry_price
        else:
            favorable = self.entry_price - bar_low
        self.peak_favorable = max(self.peak_favorable, favorable)

    def trail_by_r_multiple(self, trigger_r: float, trail_r: float):
        """Trail stop once peak MFE reaches trigger_r * risk, trailing at trail_r * risk from peak.

        For OR Reversal: trigger_r=0.75, trail_r=0.75
        Once price moves 0.75R in favor, trail stop at peak - 0.75R.
        """
        if self.risk_points <= 0:
            return
        if self.peak_favorable < trigger_r * self.risk_points:
            return  # Haven't reached trigger yet

        if self.direction == 'LONG':
            trail_level = (self.entry_price + self.peak_favorable) - trail_r * self.risk_points
            self.trailing_stop = max(self.trailing_stop, trail_level)
        else:
            trail_level = (self.entry_price - self.peak_favorable) + trail_r * self.risk_points
            self.trailing_stop = min(self.trailing_stop, trail_level)

    def breakeven_at_r(self, trigger_r: float):
        """Move to breakeven once peak MFE reaches trigger_r * risk.

        For Edge Fade: trigger_r=0.75
        """
        if self.breakeven_activated:
            return
        if self.risk_points <= 0:
            return
        if self.peak_favorable >= trigger_r * self.risk_points:
            self.trail_to_breakeven()

    def trail_by_session_extreme(self, session_high: float, session_low: float, trail_distance: float):
        """
        Trail stop based on session extreme minus a fixed trailing distance.
        For LONG: trail_stop = session_high - trail_distance (ratchets up only)
        For SHORT: trail_stop = session_low + trail_distance (ratchets down only)
        """
        if self.direction == 'LONG':
            new_stop = session_high - trail_distance
            if new_stop > self.trailing_stop:
                self.trailing_stop = new_stop
        else:
            new_stop = session_low + trail_distance
            if new_stop < self.trailing_stop:
                self.trailing_stop = new_stop


class PositionManager:
    """
    Manages open positions and enforces risk limits.

    Tracks:
      - Open positions per session
      - Daily P&L
      - Peak equity and max drawdown
      - Trade counts per day
    """

    def __init__(
        self,
        account_size: float = DEFAULT_ACCOUNT_SIZE,
        max_drawdown: float = DEFAULT_MAX_DRAWDOWN,
        max_day_loss: float = DEFAULT_MAX_DAY_LOSS,
        max_trades_per_day: int = DEFAULT_MAX_TRADES_PER_DAY,
        max_contracts: int = DEFAULT_MAX_CONTRACTS,
    ):
        self.account_size = account_size
        self.equity = account_size
        self.peak_equity = account_size
        self.max_drawdown = max_drawdown
        self.max_day_loss = max_day_loss
        self.max_trades_per_day = max_trades_per_day
        self.max_contracts = max_contracts

        self.open_positions: List[OpenPosition] = []
        self.daily_pnl: Dict[str, float] = {}
        self.daily_trade_count: Dict[str, int] = {}
        self.max_drawdown_seen: float = 0.0

    def can_open_trade(self, session_date: str) -> bool:
        """Check all risk limits before opening a new trade."""
        # Daily loss limit
        if self.daily_pnl.get(session_date, 0) <= -self.max_day_loss:
            return False

        # Max drawdown from peak
        current_dd = self.peak_equity - self.equity
        if current_dd >= self.max_drawdown:
            return False

        # Max trades per day
        if self.daily_trade_count.get(session_date, 0) >= self.max_trades_per_day:
            return False

        return True

    def daily_loss_exceeded(self, session_date: str) -> bool:
        """Check if daily loss limit was hit."""
        return self.daily_pnl.get(session_date, 0) <= -self.max_day_loss

    def record_trade_pnl(self, session_date: str, net_pnl: float) -> None:
        """Record a completed trade's P&L."""
        self.daily_pnl[session_date] = self.daily_pnl.get(session_date, 0) + net_pnl
        self.daily_trade_count[session_date] = self.daily_trade_count.get(session_date, 0) + 1
        self.equity += net_pnl
        self.peak_equity = max(self.peak_equity, self.equity)
        self.max_drawdown_seen = max(self.max_drawdown_seen, self.peak_equity - self.equity)

    def has_open_positions(self) -> bool:
        return len(self.open_positions) > 0

    def get_open_contracts(self) -> int:
        """Total contracts currently open."""
        return sum(p.contracts for p in self.open_positions)

    def add_position(self, position: OpenPosition) -> None:
        self.open_positions.append(position)

    def remove_position(self, position: OpenPosition) -> None:
        if position in self.open_positions:
            self.open_positions.remove(position)

    def close_all_positions(self) -> List[OpenPosition]:
        """Remove and return all open positions (for force-close)."""
        positions = self.open_positions.copy()
        self.open_positions.clear()
        return positions
