# modules/premarket.py
import pandas as pd
from datetime import time, timedelta

def get_premarket(df_nq, df_es=None, df_ym=None, session_date=None):
    """
    Calculate Asia high/low, London high/low, Overnight high/low (London to 9:30), previous day high/low, previous week high/low.
    SMT preopen: 08:45–09:15 divergence check.
    Compression: London/ON ratio.
    Uses the full multi-day df to calculate previous day/week.
    """
    if session_date is None:
        raise ValueError("session_date must be provided")

    current_date = pd.to_datetime(session_date)

    # Asia: 19:00 previous to 3:00 current
    previous_day = current_date - timedelta(days=1)
    asia_start = pd.to_datetime(previous_day).replace(hour=19, minute=0)
    asia_end = pd.to_datetime(current_date).replace(hour=3, minute=0)
    asia_df = df_nq[(df_nq.index >= asia_start) & (df_nq.index < asia_end)]
    asia_high = asia_df['high'].max() if not asia_df.empty else float('nan')
    asia_low = asia_df['low'].min() if not asia_df.empty else float('nan')

    # London: 3:00–5:00 current
    london_start = pd.to_datetime(current_date).replace(hour=3, minute=0)
    london_end = pd.to_datetime(current_date).replace(hour=5, minute=0)
    london_df = df_nq[(df_nq.index >= london_start) & (df_nq.index < london_end)]
    london_high = london_df['high'].max() if not london_df.empty else float('nan')
    london_low = london_df['low'].min() if not london_df.empty else float('nan')
    london_range = london_high - london_low if not pd.isna(london_high) else float('nan')

    # Overnight: 5:00–9:30 current
    on_start = london_end
    on_end = pd.to_datetime(current_date).replace(hour=9, minute=30)
    on_df = df_nq[(df_nq.index >= on_start) & (df_nq.index < on_end)]
    on_high = on_df['high'].max() if not on_df.empty else float('nan')
    on_low = on_df['low'].min() if not on_df.empty else float('nan')
    on_range = on_high - on_low if not pd.isna(on_high) else float('nan')

    # Compression: London/ON ratio
    compression_flag = bool((london_range / on_range) <= 0.35 if on_range > 0 else False)
    compression_ratio = round(london_range / on_range, 3) if on_range > 0 else 0.0

    # Previous day high/low: find the last trading day before current
    all_dates_prev = pd.Series(pd.to_datetime(df_nq['session_date'].unique())).sort_values()
    previous_dates = all_dates_prev[all_dates_prev < current_date]
    if previous_dates.empty:
        prev_day_high = float('nan')
        prev_day_low = float('nan')
    else:
        prev_day = previous_dates.iloc[-1]
        prev_day_str = prev_day.strftime('%Y-%m-%d')
        prev_day_df = df_nq[df_nq['session_date'] == prev_day_str]
        prev_day_high = prev_day_df['high'].max() if not prev_day_df.empty else float('nan')
        prev_day_low = prev_day_df['low'].min() if not prev_day_df.empty else float('nan')

    # Previous week high/low: last 5 unique session_dates before current
    all_dates = pd.Series(pd.to_datetime(df_nq['session_date'].unique())).sort_values()
    prev_dates = all_dates[all_dates < current_date].iloc[-5:]  # last 5 before today
    prev_week_df = df_nq[df_nq['session_date'].isin(prev_dates.dt.strftime('%Y-%m-%d'))]
    prev_week_high = prev_week_df['high'].max() if not prev_week_df.empty else float('nan')
    prev_week_low = prev_week_df['low'].min() if not prev_week_df.empty else float('nan')

    # SMT preopen: 08:45–09:15 divergence check
    smt_preopen = "neutral"
    if df_es is not None and df_ym is not None:
        preopen_start = pd.to_datetime(current_date).replace(hour=8, minute=45)
        preopen_end = pd.to_datetime(current_date).replace(hour=9, minute=15)
        preopen_df_nq = df_nq[(df_nq.index >= preopen_start) & (df_nq.index < preopen_end)]
        preopen_df_es = df_es[(df_es.index >= preopen_start) & (df_es.index < preopen_end)]
        preopen_df_ym = df_ym[(df_ym.index >= preopen_start) & (df_ym.index < preopen_end)]

        # Improved divergence logic (refined for accuracy)
        nq_swept_low = preopen_df_nq['low'].min() < on_low
        es_held_low = preopen_df_es['low'].min() >= on_low
        ym_held_low = preopen_df_ym['low'].min() >= on_low

        nq_swept_high = preopen_df_nq['high'].max() > on_high
        es_held_high = preopen_df_es['high'].max() <= on_high
        ym_held_high = preopen_df_ym['high'].max() <= on_high

        if nq_swept_low and (es_held_low or ym_held_low):
            smt_preopen = "bullish_divergence (NQ swept low, ES/YM held)"
        elif nq_swept_high and (es_held_high or ym_held_high):
            smt_preopen = "bearish_divergence (NQ swept high, ES/YM held)"

    # Return with NaN handling (use "not_available" string for JSON)
    return {
        "asia_high": round(asia_high, 2) if not pd.isna(asia_high) else "not_available",
        "asia_low": round(asia_low, 2) if not pd.isna(asia_low) else "not_available",
        "london_high": round(london_high, 2) if not pd.isna(london_high) else "not_available",
        "london_low": round(london_low, 2) if not pd.isna(london_low) else "not_available",
        "london_range": round(london_range, 2) if not pd.isna(london_range) else "not_available",
        "overnight_high": round(on_high, 2) if not pd.isna(on_high) else "not_available",
        "overnight_low": round(on_low, 2) if not pd.isna(on_low) else "not_available",
        "overnight_range": round(on_range, 2) if not pd.isna(on_range) else "not_available",
        "previous_day_high": round(prev_day_high, 2) if not pd.isna(prev_day_high) else "not_available",
        "previous_day_low": round(prev_day_low, 2) if not pd.isna(prev_day_low) else "not_available",
        "previous_week_high": round(prev_week_high, 2) if not pd.isna(prev_week_high) else "not_available",
        "previous_week_low": round(prev_week_low, 2) if not pd.isna(prev_week_low) else "not_available",
        "compression_flag": compression_flag,
        "compression_ratio": compression_ratio,
        "smt_preopen": smt_preopen
    }