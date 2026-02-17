"""
Adapter for rockit-framework/modules/volume_profile.py.
Wraps get_volume_profile() to produce a normalized dict for session_context.
"""

import sys
from pathlib import Path
import pandas as pd


def _ensure_rockit_on_path():
    """Add rockit-framework to sys.path if needed."""
    rockit_root = Path(__file__).resolve().parent.parent / 'rockit-framework'
    if str(rockit_root) not in sys.path:
        sys.path.insert(0, str(rockit_root))


class VolumeProfileAdapter:
    """Compute volume profile for a session using rockit-framework."""

    def __init__(self, df_all: pd.DataFrame):
        """
        Args:
            df_all: Full dataset (multi-session) with DatetimeIndex.
        """
        _ensure_rockit_on_path()
        self._df_all = df_all

    def compute(
        self, session_df: pd.DataFrame, current_time_str: str = "15:59"
    ) -> dict:
        """
        Compute volume profile for a session.

        Args:
            session_df: Single-session DataFrame with DatetimeIndex.
            current_time_str: Time up to which to compute (no lookahead).

        Returns:
            Dict with current_session, previous_day, previous_3_days profiles.
        """
        from modules.volume_profile import get_volume_profile

        # Ensure DatetimeIndex
        df_ext = self._df_all
        if not isinstance(df_ext.index, pd.DatetimeIndex):
            df_ext = df_ext.set_index('timestamp')

        df_cur = session_df
        if not isinstance(df_cur.index, pd.DatetimeIndex):
            df_cur = df_cur.set_index('timestamp')

        try:
            result = get_volume_profile(df_ext, df_cur, current_time_str)
        except Exception:
            result = {
                'current_session': {'poc': None, 'vah': None, 'val': None,
                                    'high': None, 'low': None,
                                    'hvn_nodes': [], 'lvn_nodes': []},
                'previous_day': {'poc': None, 'vah': None, 'val': None,
                                 'high': None, 'low': None,
                                 'hvn_nodes': [], 'lvn_nodes': []},
                'previous_3_days': {'poc': None, 'vah': None, 'val': None,
                                    'high': None, 'low': None,
                                    'hvn_nodes': [], 'lvn_nodes': []},
            }

        return result

    def get_session_levels(
        self, session_df: pd.DataFrame, current_time_str: str = "15:59"
    ) -> dict:
        """
        Extract key volume profile levels for engine use.

        Returns:
            Dict with poc, vah, val, hvn_nodes, lvn_nodes for current session.
        """
        vp = self.compute(session_df, current_time_str)
        current = vp.get('current_session', {})

        def safe_float(v):
            if v is None or v == 'not_available':
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        return {
            'vp_poc': safe_float(current.get('poc')),
            'vp_vah': safe_float(current.get('vah')),
            'vp_val': safe_float(current.get('val')),
            'vp_hvn_nodes': current.get('hvn_nodes', []),
            'vp_lvn_nodes': current.get('lvn_nodes', []),
            'prev_day_poc': safe_float(vp.get('previous_day', {}).get('poc')),
            'prev_day_vah': safe_float(vp.get('previous_day', {}).get('vah')),
            'prev_day_val': safe_float(vp.get('previous_day', {}).get('val')),
        }
