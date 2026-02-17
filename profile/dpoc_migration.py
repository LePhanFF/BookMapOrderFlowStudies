"""
Adapter for rockit-framework/modules/dpoc_migration.py.
Wraps get_dpoc_migration() and normalizes output for session_context.
"""

import sys
from pathlib import Path
import pandas as pd


def _ensure_rockit_on_path():
    rockit_root = Path(__file__).resolve().parent.parent / 'rockit-framework'
    if str(rockit_root) not in sys.path:
        sys.path.insert(0, str(rockit_root))


class DPOCMigrationAdapter:
    """Compute DPOC migration analysis using rockit-framework."""

    def compute(
        self,
        session_df: pd.DataFrame,
        current_time_str: str = "15:59",
        atr14: float = None,
        current_close: float = None,
    ) -> dict:
        """
        Compute DPOC migration for a session.

        Args:
            session_df: Single-session DataFrame with DatetimeIndex.
            current_time_str: Time snapshot (no lookahead).
            atr14: Current ATR14 value (for adaptive thresholds).
            current_close: Current close price.

        Returns:
            Full DPOC migration dict from rockit module.
        """
        _ensure_rockit_on_path()
        from modules.dpoc_migration import get_dpoc_migration

        df = session_df
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index('timestamp')

        try:
            return get_dpoc_migration(df, current_time_str, atr14, current_close)
        except Exception:
            return {'note': 'dpoc_computation_failed'}

    def get_migration_signals(
        self,
        session_df: pd.DataFrame,
        current_time_str: str = "15:59",
        atr14: float = None,
        current_close: float = None,
    ) -> dict:
        """
        Extract DPOC migration signals for engine use.

        Returns:
            Dict with dpoc_regime, direction, net_migration, velocity, etc.
        """
        dpoc = self.compute(session_df, current_time_str, atr14, current_close)

        return {
            'dpoc_regime': dpoc.get('dpoc_regime', 'unknown'),
            'dpoc_direction': dpoc.get('direction', 'flat'),
            'dpoc_net_migration': dpoc.get('net_migration_pts', 0.0),
            'dpoc_velocity': dpoc.get('avg_velocity_per_30min', 0.0),
            'dpoc_accelerating': dpoc.get('accelerating', False),
            'dpoc_decelerating': dpoc.get('decelerating', False),
            'dpoc_stabilizing': dpoc.get('is_stabilizing', False),
            'dpoc_reclaiming': dpoc.get('reclaiming_opposite', False),
            'dpoc_retain_pct': dpoc.get('relative_retain_percent', 0.0),
        }
