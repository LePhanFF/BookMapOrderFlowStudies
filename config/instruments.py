"""
Instrument specifications for futures contracts.
Used for P&L calculation, position sizing, and execution cost modeling.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    point_value: float      # Dollars per full point move
    tick_size: float         # Minimum price increment
    tick_value: float        # Dollars per tick
    commission: float        # Per contract per side (typical)
    slippage_ticks: int      # Default slippage assumption per side


NQ = InstrumentSpec('NQ', 20.0, 0.25, 5.0, 2.05, 1)
MNQ = InstrumentSpec('MNQ', 2.0, 0.25, 0.50, 0.62, 1)
ES = InstrumentSpec('ES', 50.0, 0.25, 12.50, 2.05, 1)
MES = InstrumentSpec('MES', 5.0, 0.25, 1.25, 0.62, 1)
YM = InstrumentSpec('YM', 5.0, 1.0, 5.0, 2.05, 1)
MYM = InstrumentSpec('MYM', 0.50, 1.0, 0.50, 0.62, 1)

INSTRUMENTS = {
    'NQ': NQ,
    'MNQ': MNQ,
    'ES': ES,
    'MES': MES,
    'YM': YM,
    'MYM': MYM,
}


def get_instrument(symbol: str) -> InstrumentSpec:
    """Get instrument spec by symbol. Raises KeyError if not found."""
    symbol = symbol.upper()
    if symbol not in INSTRUMENTS:
        raise KeyError(f"Unknown instrument '{symbol}'. Available: {list(INSTRUMENTS.keys())}")
    return INSTRUMENTS[symbol]
