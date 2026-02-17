# modules/loader.py
import pandas as pd
from datetime import datetime, timedelta

def load_nq_csv(file_path, session_date):
    """
    Loads extended data for premarket + current day for intraday.
    - Extended: current + previous 14 days (safe for previous week)
    - Current: only the session_date
    """
    df = pd.read_csv(
        file_path,
        skiprows=1,
        parse_dates=['timestamp'],
        date_format='%Y-%m-%dT%H:%M:%S.%f'
    )
    df.set_index('timestamp', inplace=True)

    current_date = pd.to_datetime(session_date).date()
    start_date = (pd.to_datetime(session_date) - timedelta(days=14)).date()  # buffer for week/Asia
    end_date = (pd.to_datetime(session_date) + timedelta(days=1)).date()     # include current full day

    # Extended df: all data in date range
    df_extended = df[
        (df.index.date >= start_date) &
        (df.index.date <= end_date)
    ].copy()

    if df_extended.empty:
        raise ValueError(f"No extended data for {session_date}")

    # Current day only
    df_current = df_extended[df_extended['session_date'] == session_date].copy()

    print(f"Loaded extended: {len(df_extended)} rows, current day: {len(df_current)} rows")
    return df_extended, df_current