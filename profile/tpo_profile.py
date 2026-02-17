"""
Adapter for rockit-framework/modules/tpo_profile.py.
Wraps get_tpo_profile() and normalizes output for session_context.
"""

import sys
from pathlib import Path
import pandas as pd


def _ensure_rockit_on_path():
    rockit_root = Path(__file__).resolve().parent.parent / 'rockit-framework'
    if str(rockit_root) not in sys.path:
        sys.path.insert(0, str(rockit_root))


class TPOProfileAdapter:
    """Compute TPO profile for a session using rockit-framework."""

    def compute(
        self,
        session_df: pd.DataFrame,
        current_time_str: str = "15:59",
        prior_day: dict = None,
    ) -> dict:
        """
        Compute TPO profile for a session.

        Args:
            session_df: Single-session DataFrame with DatetimeIndex.
            current_time_str: Time up to which to compute (no lookahead).
            prior_day: Dict with prior day poc/vah/val/high/low.

        Returns:
            Dict with TPO profile details.
        """
        _ensure_rockit_on_path()
        from modules.tpo_profile import get_tpo_profile

        df = session_df
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index('timestamp')

        try:
            return get_tpo_profile(df, current_time_str, prior_day)
        except Exception:
            return {'note': 'tpo_computation_failed'}

    def get_session_signals(
        self,
        session_df: pd.DataFrame,
        current_time_str: str = "15:59",
        prior_day: dict = None,
    ) -> dict:
        """
        Extract TPO signals for engine use.

        Returns:
            Dict with poor_high, poor_low, single_prints, fattening, tpo_shape.
        """
        tpo = self.compute(session_df, current_time_str, prior_day)

        return {
            'tpo_poc': tpo.get('current_poc'),
            'tpo_vah': tpo.get('current_vah'),
            'tpo_val': tpo.get('current_val'),
            'single_prints_above': tpo.get('single_prints_above_vah', 0),
            'single_prints_below': tpo.get('single_prints_below_val', 0),
            'poor_high': bool(tpo.get('poor_high', 0)),
            'poor_low': bool(tpo.get('poor_low', 0)),
            'effective_poor_high': bool(tpo.get('effective_poor_high', 0)),
            'effective_poor_low': bool(tpo.get('effective_poor_low', 0)),
            'fattening_zone': tpo.get('fattening_zone', 'inside_va'),
            'tpo_shape': tpo.get('tpo_shape', 'normal'),
            'rejection_high': tpo.get('rejection_at_high', 'none'),
            'rejection_low': tpo.get('rejection_at_low', 'none'),
        }
