"""
Adapter for rockit-framework/modules/wick_parade.py.
Wraps get_wick_parade() and normalizes output for session_context.
"""

import sys
from pathlib import Path
import pandas as pd


def _ensure_rockit_on_path():
    rockit_root = Path(__file__).resolve().parent.parent / 'rockit-framework'
    if str(rockit_root) not in sys.path:
        sys.path.insert(0, str(rockit_root))


class WickParadeAdapter:
    """Compute wick parade counts using rockit-framework."""

    def compute(
        self,
        session_df: pd.DataFrame,
        current_time_str: str = "15:59",
        window_minutes: int = 60,
    ) -> dict:
        """
        Compute wick parade for a session.

        Args:
            session_df: Single-session DataFrame with DatetimeIndex.
            current_time_str: Time snapshot (no lookahead).
            window_minutes: Rolling window for wick counting.

        Returns:
            Dict with bullish_wick_parade_count, bearish_wick_parade_count.
        """
        _ensure_rockit_on_path()
        from modules.wick_parade import get_wick_parade

        df = session_df
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index('timestamp')

        try:
            return get_wick_parade(df, current_time_str, window_minutes)
        except Exception:
            return {
                'bullish_wick_parade_count': 0,
                'bearish_wick_parade_count': 0,
            }

    def get_wick_signals(
        self,
        session_df: pd.DataFrame,
        current_time_str: str = "15:59",
        window_minutes: int = 60,
    ) -> dict:
        """
        Extract wick parade signals for engine use.

        Dalton Rule #14: >=6 bullish wicks = long override,
                         >=6 bearish wicks = short override (abort trend).

        Returns:
            Dict with wick counts and override flags.
        """
        wp = self.compute(session_df, current_time_str, window_minutes)

        bull_count = wp.get('bullish_wick_parade_count', 0)
        bear_count = wp.get('bearish_wick_parade_count', 0)

        return {
            'wick_bull_count': bull_count,
            'wick_bear_count': bear_count,
            'wick_bull_override': bull_count >= 6,
            'wick_bear_override': bear_count >= 6,
        }
