"""
Session filtering utilities.
Handles RTH, ETH, and IB period filtering for US futures.
"""

import pandas as pd
from datetime import time
from config.constants import RTH_START, RTH_END, ETH_START, IB_END


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to US Regular Trading Hours: 9:30 - 16:00 ET.

    FIXED: Old code used 15:00 as RTH end, missing the last hour.
    """
    if 'time' not in df.columns:
        df = df.copy()
        df['time'] = df['timestamp'].dt.time

    mask = (df['time'] >= RTH_START) & (df['time'] <= RTH_END)
    df_rth = df[mask].copy()

    print(f"  RTH filter: {len(df_rth):,} / {len(df):,} rows ({len(df_rth)/len(df)*100:.1f}%)")
    return df_rth


def filter_eth(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to Extended Trading Hours: 18:00 - 9:29 ET (overnight session)."""
    if 'time' not in df.columns:
        df = df.copy()
        df['time'] = df['timestamp'].dt.time

    mask = (df['time'] >= ETH_START) | (df['time'] < RTH_START)
    return df[mask].copy()


def filter_ib_period(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to Initial Balance period: 9:30 - 10:30 ET."""
    if 'time' not in df.columns:
        df = df.copy()
        df['time'] = df['timestamp'].dt.time

    mask = (df['time'] >= RTH_START) & (df['time'] <= IB_END)
    return df[mask].copy()
