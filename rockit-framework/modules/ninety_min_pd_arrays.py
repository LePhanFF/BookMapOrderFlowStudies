# modules/ninety_min_pd_arrays.py
import pandas as pd
from datetime import time

def get_ninety_min_pd_arrays(df_nq, current_time_str="11:45"):
    """
    90-min range + Premium/Discount arrays + displacement/expansion.
    - 90-min period: 09:30–11:00 ET
    - Equilibrium = 50% midpoint
    - Discount (below) = bullish zone
    - Premium (above) = bearish zone
    - Expansion = price beyond range = trend confirmation
    No lookahead.
    """
    current_time = pd.to_datetime(current_time_str).time()
    available_df = df_nq[df_nq.index.time <= current_time].copy()

    if len(available_df) == 0:
        return {"note": "no_data_yet"}

    # 90-min range: 09:30–11:00
    ninety_df = available_df.between_time('09:30', '11:00')

    if len(ninety_df) == 0:
        return {"note": "90_min_not_started"}

    ninety_high = ninety_df['high'].max()
    ninety_low = ninety_df['low'].min()
    ninety_range = ninety_high - ninety_low
    equilibrium = (ninety_high + ninety_low) / 2  # 50%

    current_price = available_df['close'].iloc[-1]

    in_discount = current_price < equilibrium
    in_premium = current_price > equilibrium

    expansion = "up" if current_price > ninety_high else "down" if current_price < ninety_low else "inside"

    bias_potential = "bullish" if in_discount else "bearish" if in_premium else "neutral"

    return {
        "ninety_min_high": round(ninety_high, 2),
        "ninety_min_low": round(ninety_low, 2),
        "ninety_min_range": round(ninety_range, 2),
        "equilibrium_50": round(equilibrium, 2),
        "current_in_discount": int(in_discount),
        "current_in_premium": int(in_premium),
        "expansion_status": expansion,
        "bias_potential": bias_potential,
        "note": "90-min PD arrays: discount = bullish zone, premium = bearish. Expansion = trend confirmation"
    }