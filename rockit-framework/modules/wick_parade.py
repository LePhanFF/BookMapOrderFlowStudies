# modules/wick_parade.py
import pandas as pd
import numpy as np

def get_wick_parade(df_nq, current_time_str="11:45", window_minutes=60):
    """
    Calculate lower-wick (bullish) and upper-wick (bearish) parade counts on 5-min data.
    - Uses only bars up to current_time (no lookahead).
    - Counts in recent window for responsive buying/selling (Rule #14 override).
    - Threshold 0.6 – tune if needed.
    """
    # Parse current time
    current_time = pd.to_datetime(current_time_str).time()

    # Available data up to current time
    available_df = df_nq[df_nq.index.time <= current_time].copy()

    if len(available_df) == 0:
        return {"bullish_wick_parade_count": 0, "bearish_wick_parade_count": 0, "note": "no_data_yet"}

    # Recent window: last window_minutes min
    start_time = available_df.index[-1] - pd.Timedelta(minutes=window_minutes)
    recent_df = available_df[available_df.index >= start_time]

    if len(recent_df) == 0:
        return {"bullish_wick_parade_count": 0, "bearish_wick_parade_count": 0, "note": "pre_open"}

    # Bullish (lower wicks)
    lower_wick_ratio = (recent_df['close'] - recent_df['low']) / (recent_df['high'] - recent_df['low'])
    lower_wick_ratio = lower_wick_ratio.replace([np.inf, -np.inf], 0)
    bullish_count = (lower_wick_ratio > 0.6).sum()

    # Bearish (upper wicks)
    upper_wick_ratio = (recent_df['high'] - recent_df['close']) / (recent_df['high'] - recent_df['low'])
    upper_wick_ratio = upper_wick_ratio.replace([np.inf, -np.inf], 0)
    bearish_count = (upper_wick_ratio > 0.6).sum()

    return {
        "bullish_wick_parade_count": int(bullish_count),
        "bearish_wick_parade_count": int(bearish_count),
        "window_minutes": window_minutes,
        "note": f"Counts in last {window_minutes} min – ≥6 bullish = long override (Rule #14), ≥6 bearish = short override"
    }