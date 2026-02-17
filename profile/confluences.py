"""
Adapter for rockit-framework/modules/core_confluences.py.
Wraps get_core_confluences() and normalizes output for session_context.
"""

import sys
from pathlib import Path


def _ensure_rockit_on_path():
    rockit_root = Path(__file__).resolve().parent.parent / 'rockit-framework'
    if str(rockit_root) not in sys.path:
        sys.path.insert(0, str(rockit_root))


class ConfluenceAdapter:
    """Compute core confluences using rockit-framework."""

    def compute(
        self,
        intraday_data: dict,
        current_time_str: str = "15:59",
    ) -> dict:
        """
        Compute confluences from assembled intraday data.

        Args:
            intraday_data: Dict with keys 'ib', 'volume_profile', 'tpo_profile',
                           'dpoc_migration' from other adapters.
            current_time_str: Time snapshot.

        Returns:
            Full confluences dict from rockit module.
        """
        _ensure_rockit_on_path()
        from modules.core_confluences import get_core_confluences

        try:
            return get_core_confluences(intraday_data, current_time_str)
        except Exception:
            return {'note': 'confluence_computation_failed'}

    def get_confluence_signals(
        self,
        intraday_data: dict,
        current_time_str: str = "15:59",
    ) -> dict:
        """
        Extract confluence signals for engine use.

        Returns:
            Dict with ib_acceptance, dpoc_compression, tpo_signals, etc.
        """
        conf = self.compute(intraday_data, current_time_str)

        ib_acc = conf.get('ib_acceptance', {})
        dpoc_comp = conf.get('dpoc_compression', {})
        tpo_sig = conf.get('tpo_signals', {})
        migration = conf.get('migration', {})

        return {
            'conf_close_above_ibh': ib_acc.get('close_above_ibh', False),
            'conf_close_below_ibl': ib_acc.get('close_below_ibl', False),
            'conf_dpoc_compression_bias': dpoc_comp.get('compression_bias', 'none'),
            'conf_single_above': tpo_sig.get('single_prints_above', False),
            'conf_single_below': tpo_sig.get('single_prints_below', False),
            'conf_fattening_upper': tpo_sig.get('fattening_upper', False),
            'conf_fattening_lower': tpo_sig.get('fattening_lower', False),
            'conf_migration_up': migration.get('significant_up', False),
            'conf_migration_down': migration.get('significant_down', False),
        }
