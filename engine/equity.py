"""
Equity curve tracking.
Records equity snapshots for drawdown analysis and visualization.
"""

from dataclasses import dataclass, field
from typing import List
from datetime import datetime


@dataclass
class EquitySnapshot:
    timestamp: datetime
    equity: float
    daily_pnl: float
    trade_count: int
    drawdown: float         # Current drawdown from peak
    session_date: str = ""


class EquityCurve:
    """Tracks equity over time."""

    def __init__(self, initial_equity: float):
        self.initial_equity = initial_equity
        self.snapshots: List[EquitySnapshot] = []
        self._peak = initial_equity
        self._current = initial_equity

    def record(
        self,
        timestamp: datetime,
        equity: float,
        daily_pnl: float,
        trade_count: int,
        session_date: str = "",
    ) -> None:
        self._current = equity
        self._peak = max(self._peak, equity)
        dd = self._peak - equity

        self.snapshots.append(EquitySnapshot(
            timestamp=timestamp,
            equity=equity,
            daily_pnl=daily_pnl,
            trade_count=trade_count,
            drawdown=dd,
            session_date=session_date,
        ))

    @property
    def max_drawdown(self) -> float:
        if not self.snapshots:
            return 0.0
        return max(s.drawdown for s in self.snapshots)

    @property
    def max_drawdown_pct(self) -> float:
        if not self.snapshots or self._peak == 0:
            return 0.0
        return self.max_drawdown / self._peak * 100

    @property
    def final_equity(self) -> float:
        return self._current

    @property
    def total_return(self) -> float:
        return self._current - self.initial_equity

    @property
    def total_return_pct(self) -> float:
        if self.initial_equity == 0:
            return 0.0
        return (self._current - self.initial_equity) / self.initial_equity * 100

    def to_series(self):
        """Convert to pandas Series for plotting. Import lazily."""
        import pandas as pd
        return pd.Series(
            {s.timestamp: s.equity for s in self.snapshots},
            name='equity',
        )
