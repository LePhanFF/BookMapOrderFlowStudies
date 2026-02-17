"""
Cross-platform CSV data loader for NinjaTrader volumetric exports.
Replaces the old data_loader.py with proper path handling and bug fixes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def find_project_root() -> Path:
    """Walk up from this file to find the repo root (contains csv/ directory)."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / 'csv').is_dir():
            return current
        current = current.parent
    raise FileNotFoundError("Could not locate project root with csv/ directory")


def load_csv(instrument: str = 'NQ', csv_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load NinjaTrader volumetric CSV data.

    Args:
        instrument: 'NQ', 'ES', or 'YM'
        csv_dir: Optional path to csv directory. Auto-discovered if None.

    Returns:
        DataFrame with parsed timestamps and numeric columns.
    """
    if csv_dir is None:
        csv_dir = find_project_root() / 'csv'
    else:
        csv_dir = Path(csv_dir)

    filepath = csv_dir / f'{instrument}_Volumetric_1.csv'

    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    # Read CSV, skip schema line (first line starts with #)
    df = pd.read_csv(filepath, low_memory=False, comment='#')

    # Convert numeric columns
    numeric_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'ema20', 'ema50', 'ema200', 'rsi14', 'atr14',
        'vwap', 'vwap_upper1', 'vwap_upper2', 'vwap_upper3',
        'vwap_lower1', 'vwap_lower2', 'vwap_lower3',
        'vol_ask', 'vol_bid', 'vol_delta',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Parse timestamps
    df = df[df['timestamp'].astype(str).str.len() >= 19].copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
    df = df.dropna(subset=['timestamp'])

    # Parse session date
    df['session_date'] = pd.to_datetime(df['session_date'], errors='coerce')

    # Derived time columns
    df['time'] = df['timestamp'].dt.time

    print(f"Loaded {instrument}: {len(df):,} rows")
    print(f"  Date range: {df['session_date'].min().date()} to {df['session_date'].max().date()}")
    print(f"  Sessions: {df['session_date'].nunique()}")

    return df
