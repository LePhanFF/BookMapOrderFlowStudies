# modules/volume_profile.py
import pandas as pd
import numpy as np

def get_volume_profile(df_extended, df_current, current_time_str="11:45"):
    """
    Calculate current session volume profile (up to current_time) and previous day + previous 3 days.
    - Current: POC, VAH, VAL (70% value area) + top HVN/LVN nodes
    - Previous day & previous 3 days: same
    No lookahead – uses only available data up to current_time for current session.
    """
    # Current session up to current_time
    current_time = pd.to_datetime(current_time_str).time()
    available_current = df_current[df_current.index.time <= current_time].copy()
    current_session = available_current.between_time('09:30', current_time_str)

    # Previous day from extended df
    current_date = pd.to_datetime(df_extended['session_date'].max())
    all_dates = pd.Series(pd.to_datetime(df_extended['session_date'].unique())).sort_values()
    previous_dates = all_dates[all_dates < current_date]
    if previous_dates.empty:
        prev_day_df = pd.DataFrame()
    else:
        prev_day = previous_dates.iloc[-1]
        prev_day_str = prev_day.strftime('%Y-%m-%d')
        prev_day_df = df_extended[df_extended['session_date'] == prev_day_str]

    # Previous 3 days from extended df
    all_dates = pd.Series(pd.to_datetime(df_extended['session_date'].unique())).sort_values()
    prev_3_dates = all_dates[all_dates < current_date].iloc[-3:]
    prev_3_df = df_extended[df_extended['session_date'].isin(prev_3_dates.dt.strftime('%Y-%m-%d'))]

    def calculate_profile(df):
        if df.empty:
            return {
                "poc": "not_available",
                "vah": "not_available",
                "val": "not_available",
                "high": "not_available",
                "low": "not_available",
                "hvn_nodes": [],
                "lvn_nodes": []
            }

        high = df['high'].max()
        low = df['low'].min()

        # Volume profile binning (NQ tick = 0.25)
        df = df.copy()
        price_bins = np.arange(df['low'].min() - 0.25, df['high'].max() + 0.5, 0.25)
        df['price_bin'] = pd.cut(df['close'], bins=price_bins, labels=price_bins[:-1])

        vol_profile = df.groupby('price_bin', observed=False)['volume'].sum()
        total_volume = vol_profile.sum()

        if total_volume == 0:
            return {
                "poc": "not_available",
                "vah": "not_available",
                "val": "not_available",
                "high": round(high, 2),
                "low": round(low, 2),
                "hvn_nodes": [],
                "lvn_nodes": []
            }

        poc = vol_profile.idxmax()

        # 70% value area
        sorted_vol = vol_profile.sort_values(ascending=False)
        cum_vol = 0
        va_prices = []
        for price, vol in sorted_vol.items():
            va_prices.append(price)
            cum_vol += vol
            if cum_vol >= 0.7 * total_volume:
                break

        vah = max(va_prices)
        val = min(va_prices)

        # HVN/LVN: top/bottom 10% volume prices (top 3 each)
        cum_vol_hvn = 0
        cum_vol_lvn = total_volume
        hvn_nodes = []
        lvn_nodes = []
        for price, vol in sorted_vol.items():
            if cum_vol_hvn < 0.1 * total_volume:
                hvn_nodes.append(price)
                cum_vol_hvn += vol
            if cum_vol_lvn > 0.9 * total_volume:
                lvn_nodes.append(price)
                cum_vol_lvn -= vol

        hvn_nodes = [round(float(p), 2) for p in hvn_nodes[:3]]
        lvn_nodes = [round(float(p), 2) for p in sorted(vol_profile.nsmallest(3).index)]

        return {
            "poc": round(float(poc), 2),
            "vah": round(float(vah), 2),
            "val": round(float(val), 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "hvn_nodes": hvn_nodes,
            "lvn_nodes": lvn_nodes
        }

    current_profile = calculate_profile(current_session)
    prev_day_profile = calculate_profile(prev_day_df)
    prev_3_profile = calculate_profile(prev_3_df)

    return {
        "current_session": current_profile,
        "previous_day": prev_day_profile,
        "previous_3_days": prev_3_profile,
        "note": "Volume profile (70% value area) + top HVN/LVN nodes. Current = up to snapshot time (no lookahead)"
    }