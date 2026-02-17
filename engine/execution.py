"""
Execution model: slippage, commissions, and realistic fill simulation.
The old system had ZERO execution cost modeling -- this fixes that.
"""

from config.instruments import InstrumentSpec
from config.constants import DEFAULT_SLIPPAGE_TICKS


class ExecutionModel:
    """
    Simulates realistic trade execution with slippage and commissions.

    All costs are in DOLLARS.
    """

    def __init__(
        self,
        instrument: InstrumentSpec,
        slippage_ticks: int = None,
        commission_per_side: float = None,
    ):
        self.instrument = instrument
        self.slippage_ticks = slippage_ticks if slippage_ticks is not None else instrument.slippage_ticks
        self.commission = commission_per_side if commission_per_side is not None else instrument.commission

    def fill_entry(self, direction: str, price: float) -> float:
        """Apply slippage to entry. Buys fill higher, sells fill lower."""
        slip = self.slippage_ticks * self.instrument.tick_size
        if direction == 'LONG':
            return price + slip
        else:
            return price - slip

    def fill_exit(self, direction: str, price: float) -> float:
        """Apply slippage to exit. Sells fill lower, covers fill higher."""
        slip = self.slippage_ticks * self.instrument.tick_size
        if direction == 'LONG':
            return price - slip  # Selling to close long fills lower
        else:
            return price + slip  # Buying to cover short fills higher

    def calculate_gross_pnl(
        self, direction: str, entry_fill: float, exit_fill: float, contracts: int,
    ) -> float:
        """Calculate gross P&L in dollars (before costs)."""
        if direction == 'LONG':
            points = exit_fill - entry_fill
        else:
            points = entry_fill - exit_fill
        return points * self.instrument.point_value * contracts

    def calculate_commission(self, contracts: int) -> float:
        """Round-trip commission in dollars."""
        return self.commission * 2 * contracts

    def calculate_slippage_cost(self, contracts: int) -> float:
        """Round-trip slippage cost in dollars."""
        return (self.slippage_ticks * self.instrument.tick_value * 2 * contracts)

    def calculate_net_pnl(
        self, direction: str, entry_fill: float, exit_fill: float, contracts: int,
    ) -> tuple:
        """
        Full P&L breakdown.

        Returns: (gross_pnl, commission, slippage_cost, net_pnl) -- all in dollars.
        """
        gross = self.calculate_gross_pnl(direction, entry_fill, exit_fill, contracts)
        comm = self.calculate_commission(contracts)
        slip = self.calculate_slippage_cost(contracts)
        net = gross - comm - slip
        return gross, comm, slip, net

    def calculate_contracts(
        self, risk_per_trade: float, stop_distance_points: float,
    ) -> int:
        """
        Position size based on fixed dollar risk.

        contracts = risk / (stop_distance * point_value)
        Returns 0 if stop distance is zero or risk per contract exceeds budget.
        Minimum 1 contract only when risk fits within budget.
        """
        if stop_distance_points <= 0:
            return 0
        risk_per_contract = stop_distance_points * self.instrument.point_value
        if risk_per_contract > risk_per_trade * 2:
            # Don't take trades where a single contract risks >2x the per-trade budget
            return 0
        contracts = int(risk_per_trade / risk_per_contract)
        return max(1, contracts)
