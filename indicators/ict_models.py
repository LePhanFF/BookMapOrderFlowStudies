"""
ICT Entry Model Detectors
=========================

Fair Value Gap (FVG):
  3-candle pattern where candle 2 gaps beyond candle 1 & 3.
  Bull FVG: candle3.low > candle1.high (gap up, price left inefficiency below)
  Bear FVG: candle3.high < candle1.low (gap down, price left inefficiency above)

Inverse FVG (IFVG):
  FVG that gets revisited and holds as support/resistance.
  A bull FVG that price returns to and bounces from = confirmed IFVG entry.
  A bear FVG that price returns to and rejects = confirmed IFVG entry.

Balanced Price Range (BPR):
  Overlapping bull FVG and bear FVG at the same price level.
  When price breaks the BPR boundary, it signals trend continuation.
  BPR failure: price tests BPR and reverses = reversal signal.

15-Min FVG:
  FVG detected on 15-minute resampled bars (5 x 1-min bars aggregated into
  synthetic 15-min bars). These are higher-timeframe inefficiencies.

Usage:
  These are computed as bar-level boolean columns that strategies can check:
    - fvg_bull: True if current bar is inside a bull FVG zone
    - fvg_bear: True if current bar is inside a bear FVG zone
    - ifvg_bull_entry: True if price revisits a bull FVG and holds
    - ifvg_bear_entry: True if price revisits a bear FVG and rejects
    - bpr_zone: True if current bar is inside a BPR overlap
    - fvg_bull_15m: True if in a 15-min bull FVG zone
    - fvg_bear_15m: True if in a 15-min bear FVG zone
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FVGZone:
    """A Fair Value Gap zone."""
    direction: str       # 'BULL' or 'BEAR'
    top: float           # Upper boundary
    bottom: float        # Lower boundary
    bar_index: int       # When it was created
    filled: bool = False # Whether price has completely filled it
    tested: bool = False # Whether price has revisited the zone


def detect_fvg_zones(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    min_gap_pct: float = 0.0001,  # Minimum gap as fraction of price
) -> List[FVGZone]:
    """
    Detect all FVG zones in a price series.

    Bull FVG: bar[i+2].low > bar[i].high  (gap to the upside)
    Bear FVG: bar[i+2].high < bar[i].low  (gap to the downside)

    Returns list of FVGZone objects.
    """
    zones = []
    n = len(highs)

    for i in range(n - 2):
        bar1_high = highs[i]
        bar1_low = lows[i]
        bar3_high = highs[i + 2]
        bar3_low = lows[i + 2]
        mid_price = closes[i + 1]

        min_gap = mid_price * min_gap_pct if mid_price > 0 else 0.5

        # Bull FVG: gap up — candle 3 low > candle 1 high
        if bar3_low > bar1_high + min_gap:
            zones.append(FVGZone(
                direction='BULL',
                top=bar3_low,
                bottom=bar1_high,
                bar_index=i + 1,  # middle candle
            ))

        # Bear FVG: gap down — candle 3 high < candle 1 low
        if bar3_high < bar1_low - min_gap:
            zones.append(FVGZone(
                direction='BEAR',
                top=bar1_low,
                bottom=bar3_high,
                bar_index=i + 1,
            ))

    return zones


def compute_fvg_features(df: pd.DataFrame, lookback: int = 30) -> pd.DataFrame:
    """
    Compute FVG-based features for each bar.

    For each bar, checks if price is currently inside any active (unfilled)
    FVG zone from the last `lookback` bars.

    Adds columns:
      - fvg_bull: True if current bar's low touches an active bull FVG
      - fvg_bear: True if current bar's high touches an active bear FVG
      - ifvg_bull_entry: True if price revisits bull FVG and holds (close > zone bottom)
      - ifvg_bear_entry: True if price revisits bear FVG and rejects (close < zone top)
      - fvg_bull_bottom: Bottom of nearest active bull FVG (for stop placement)
      - fvg_bear_top: Top of nearest active bear FVG (for stop placement)
      - in_bpr: True if current price is in a BPR (overlapping bull+bear FVG)
    """
    df = df.copy()

    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    opens = df['open'].values
    n = len(df)

    # Initialize columns
    df['fvg_bull'] = False
    df['fvg_bear'] = False
    df['ifvg_bull_entry'] = False
    df['ifvg_bear_entry'] = False
    df['fvg_bull_bottom'] = np.nan
    df['fvg_bull_top'] = np.nan
    df['fvg_bear_top'] = np.nan
    df['fvg_bear_bottom'] = np.nan
    df['in_bpr'] = False

    # Track active zones
    active_bull_zones: List[FVGZone] = []
    active_bear_zones: List[FVGZone] = []

    for i in range(2, n):
        # Detect new FVGs
        bar1_high = highs[i - 2]
        bar1_low = lows[i - 2]
        bar3_high = highs[i]
        bar3_low = lows[i]
        mid_price = closes[i - 1]
        min_gap = mid_price * 0.0001 if mid_price > 0 else 0.5

        # Bull FVG
        if bar3_low > bar1_high + min_gap:
            active_bull_zones.append(FVGZone(
                direction='BULL',
                top=bar3_low,
                bottom=bar1_high,
                bar_index=i - 1,
            ))

        # Bear FVG
        if bar3_high < bar1_low - min_gap:
            active_bear_zones.append(FVGZone(
                direction='BEAR',
                top=bar1_low,
                bottom=bar3_high,
                bar_index=i - 1,
            ))

        # Expire old zones
        active_bull_zones = [
            z for z in active_bull_zones
            if not z.filled and (i - z.bar_index) <= lookback
        ]
        active_bear_zones = [
            z for z in active_bear_zones
            if not z.filled and (i - z.bar_index) <= lookback
        ]

        current_close = closes[i]
        current_low = lows[i]
        current_high = highs[i]

        # Check bull FVG zones
        nearest_bull = None
        for z in active_bull_zones:
            # Price has entered the FVG zone from above (pullback into gap)
            if current_low <= z.top:
                z.tested = True

                # Check if zone is completely filled (price went through)
                if current_close < z.bottom:
                    z.filled = True
                    continue

                # IFVG entry: price tested the zone and is holding above bottom
                if current_close > z.bottom:
                    df.iat[i, df.columns.get_loc('fvg_bull')] = True
                    df.iat[i, df.columns.get_loc('ifvg_bull_entry')] = True

                    if nearest_bull is None or z.bottom > nearest_bull.bottom:
                        nearest_bull = z

        if nearest_bull:
            df.iat[i, df.columns.get_loc('fvg_bull_bottom')] = nearest_bull.bottom
            df.iat[i, df.columns.get_loc('fvg_bull_top')] = nearest_bull.top

        # Check bear FVG zones
        nearest_bear = None
        for z in active_bear_zones:
            if current_high >= z.bottom:
                z.tested = True

                if current_close > z.top:
                    z.filled = True
                    continue

                if current_close < z.top:
                    df.iat[i, df.columns.get_loc('fvg_bear')] = True
                    df.iat[i, df.columns.get_loc('ifvg_bear_entry')] = True

                    if nearest_bear is None or z.top < nearest_bear.top:
                        nearest_bear = z

        if nearest_bear:
            df.iat[i, df.columns.get_loc('fvg_bear_top')] = nearest_bear.top
            df.iat[i, df.columns.get_loc('fvg_bear_bottom')] = nearest_bear.bottom

        # BPR detection: overlapping bull and bear FVGs at same price
        for bz in active_bull_zones:
            for sz in active_bear_zones:
                # Overlap: bull zone and bear zone share price range
                overlap_bottom = max(bz.bottom, sz.bottom)
                overlap_top = min(bz.top, sz.top)
                if overlap_bottom < overlap_top:
                    if overlap_bottom <= current_close <= overlap_top:
                        df.iat[i, df.columns.get_loc('in_bpr')] = True
                        break

    return df


def compute_fvg_15m(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """
    Compute 15-minute Fair Value Gaps by resampling 1-min bars.

    Detects FVGs on the higher timeframe (15-min) and maps them back
    to 1-minute bars for precise entry timing.

    Adds columns:
      - fvg_bull_15m: True if current 1-min bar is inside an active 15-min bull FVG
      - fvg_bear_15m: True if current 1-min bar is inside an active 15-min bear FVG
      - fvg_bull_15m_bottom: Bottom of nearest 15-min bull FVG
      - fvg_bear_15m_top: Top of nearest 15-min bear FVG
    """
    df = df.copy()
    df['fvg_bull_15m'] = False
    df['fvg_bear_15m'] = False
    df['fvg_bull_15m_bottom'] = np.nan
    df['fvg_bear_15m_top'] = np.nan

    if 'timestamp' not in df.columns:
        return df

    # Resample to 15-minute bars per session
    for session_date, session_df in df.groupby('session_date'):
        if len(session_df) < 45:  # Need at least 3 x 15-min bars
            continue

        # Create 15-min OHLC
        session_df_ts = session_df.set_index('timestamp')
        bars_15m = session_df_ts.resample('15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
        }).dropna()

        if len(bars_15m) < 3:
            continue

        # Detect FVG zones on 15-min bars
        zones = detect_fvg_zones(
            bars_15m['high'].values,
            bars_15m['low'].values,
            bars_15m['close'].values,
            min_gap_pct=0.0002,  # Slightly larger threshold for 15-min
        )

        # Map zones back to 1-min bars
        active_zones = zones.copy()
        session_idx = session_df.index

        for idx in session_idx:
            row = df.loc[idx]
            current_close = row['close']
            current_low = row['low']
            current_high = row['high']

            # Check bull zones
            for z in active_zones:
                if z.direction == 'BULL' and not z.filled:
                    if current_low <= z.top and current_close > z.bottom:
                        df.at[idx, 'fvg_bull_15m'] = True
                        df.at[idx, 'fvg_bull_15m_bottom'] = z.bottom
                    elif current_close < z.bottom:
                        z.filled = True

                elif z.direction == 'BEAR' and not z.filled:
                    if current_high >= z.bottom and current_close < z.top:
                        df.at[idx, 'fvg_bear_15m'] = True
                        df.at[idx, 'fvg_bear_15m_top'] = z.top
                    elif current_close > z.top:
                        z.filled = True

    return df


def add_ict_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all ICT entry model features to the dataframe."""
    print("  Computing FVG/IFVG/BPR features...")
    df = compute_fvg_features(df, lookback=30)

    print("  Computing 15-min FVG features...")
    df = compute_fvg_15m(df, lookback=10)

    return df
