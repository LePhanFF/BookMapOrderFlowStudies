"""
Profile adapters wrapping rockit-framework modules.
Normalizes interfaces for use in the backtest engine session_context.
"""

from profile.volume_profile import VolumeProfileAdapter
from profile.tpo_profile import TPOProfileAdapter
from profile.ib_analysis import IBAnalysisAdapter
from profile.dpoc_migration import DPOCMigrationAdapter
from profile.confluences import ConfluenceAdapter
from profile.wick_parade import WickParadeAdapter

__all__ = [
    'VolumeProfileAdapter',
    'TPOProfileAdapter',
    'IBAnalysisAdapter',
    'DPOCMigrationAdapter',
    'ConfluenceAdapter',
    'WickParadeAdapter',
]
