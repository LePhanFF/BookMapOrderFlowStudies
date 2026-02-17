"""
Adapter for rockit-framework/modules/ib_location.py.
Wraps get_ib_location() and normalizes output for session_context.
"""

import sys
from pathlib import Path
import pandas as pd


def _ensure_rockit_on_path():
    rockit_root = Path(__file__).resolve().parent.parent / 'rockit-framework'
    if str(rockit_root) not in sys.path:
        sys.path.insert(0, str(rockit_root))


class IBAnalysisAdapter:
    """Compute IB location analysis using rockit-framework."""

    def compute(
        self,
        session_df: pd.DataFrame,
        current_time_str: str = "15:59",
    ) -> dict:
        """
        Compute IB location for a session.

        Args:
            session_df: Single-session DataFrame with DatetimeIndex.
            current_time_str: Time snapshot (no lookahead).

        Returns:
            Full IB location dict from rockit module.
        """
        _ensure_rockit_on_path()
        from modules.ib_location import get_ib_location

        df = session_df
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index('timestamp')

        try:
            return get_ib_location(df, current_time_str)
        except Exception:
            return {'note': 'ib_computation_failed'}

    def get_ib_levels(
        self,
        session_df: pd.DataFrame,
        current_time_str: str = "15:59",
    ) -> dict:
        """
        Extract IB levels for engine use.

        Returns:
            Dict with ib_high, ib_low, ib_range, ib_mid, price_vs_ib, price_vs_vwap.
        """
        ib = self.compute(session_df, current_time_str)

        return {
            'ib_status': ib.get('ib_status', 'unknown'),
            'ib_high': ib.get('ib_high'),
            'ib_low': ib.get('ib_low'),
            'ib_range': ib.get('ib_range'),
            'ib_mid': ib.get('ib_mid'),
            'price_vs_ib': ib.get('price_vs_ib', 'unknown'),
            'price_vs_vwap': ib.get('price_vs_vwap', 'unknown'),
        }
