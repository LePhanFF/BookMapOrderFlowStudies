# modules/tpo_profile.py
import pandas as pd
import numpy as np
from datetime import time

def get_tpo_profile(df_nq, current_time_str="11:45", prior_day=None):
    """
    Compact TPO profile facts for ROCKIT prompts.
    - Current session only, up to current_time (no lookahead)
    - 30-min TPO letters
    - Outputs only actionable signals
    - prior_day: dict from volume_profile.previous_day (poc, vah, val, high, low)
    """
    current_time = pd.to_datetime(current_time_str).time()
    available_df = df_nq[df_nq.index.time <= current_time].copy()

    if len(available_df) == 0:
        return {"note": "no_data_yet"}

    # Current session from 09:30
    session_df = available_df.between_time('09:30', current_time_str)

    if len(session_df) == 0:
        return {"note": "pre_open"}

    # Resample to 30-min periods, assign letter (A, B, C...)
    session_df = session_df.copy()
    session_df['period'] = session_df.index.floor('30min')
    periods = session_df['period'].unique()
    letter_map = {p: chr(65 + i) for i, p in enumerate(sorted(periods))}  # A, B, C...

    # Price bins (0.25 tick)
    min_price = session_df['low'].min()
    max_price = session_df['high'].max()
    bins = np.arange(min_price - 0.25, max_price + 0.5, 0.25)

    # Build TPO count per price
    tpo_counts = pd.Series(0, index=bins)
    for _, row in session_df.iterrows():
        period_letter = letter_map[row['period']]
        price_range = np.arange(row['low'], row['high'] + 0.25, 0.25)
        tpo_counts.loc[price_range] += 1

    tpo_counts = tpo_counts[tpo_counts > 0]  # drop zero

    if tpo_counts.empty:
        return {"note": "no_tpo_yet"}

    # Current POC (price with max TPO)
    current_poc = tpo_counts.idxmax()

    # Value Area 70%
    total_tpo = tpo_counts.sum()
    sorted_tpo = tpo_counts.sort_values(ascending=False)
    cum_tpo = 0
    va_prices = []
    for price, count in sorted_tpo.items():
        va_prices.append(price)
        cum_tpo += count
        if cum_tpo >= 0.7 * total_tpo:
            break
    current_vah = max(va_prices)
    current_val = min(va_prices)

    # Single prints
    single_above = (tpo_counts[tpo_counts.index > current_vah] == 1).sum()
    single_below = (tpo_counts[tpo_counts.index < current_val] == 1).sum()

    # Rejection strength flags
    rejection_high = "none"
    if single_above >= 5:
        rejection_high = "strong"
    elif single_above >= 3:
        rejection_high = "moderate"
    elif single_above >= 1:
        rejection_high = "weak"

    rejection_low = "none"
    if single_below >= 5:
        rejection_low = "strong"
    elif single_below >= 3:
        rejection_low = "moderate"
    elif single_below >= 1:
        rejection_low = "weak"

    # Poor high/low (classic Dalton — ≥2 TPO at absolute extreme)
    max_price_level = tpo_counts.index.max()
    min_price_level = tpo_counts.index.min()
    poor_high = (tpo_counts[max_price_level] >= 2)
    poor_low = (tpo_counts[min_price_level] >= 2)

    # Effective poor high/low (includes stacked singles for exhaustion reads)
    effective_poor_high = 1 if (poor_high or single_above >= 4) else 0
    effective_poor_low = 1 if (poor_low or single_below >= 4) else 0

    # Fattening zone
    current_price = session_df['close'].iloc[-1]
    current_high = session_df['high'].max()
    current_low = session_df['low'].min()

    if current_price >= current_vah:
        fattening = "above_vah"
    elif current_price <= current_val:
        fattening = "below_val"
    elif abs(current_price - current_vah) < abs(current_price - current_val):
        fattening = "at_vah"
    elif abs(current_price - current_val) < abs(current_price - current_vah):
        fattening = "at_val"
    else:
        fattening = "inside_va"

    # Improved shape classification with rejection overlay
    profile_range = max_price - min_price
    va_range = current_vah - current_val

    base_shape = "b_shape" if va_range / profile_range > 0.7 else "normal"

    tpo_shape = base_shape
    if rejection_high != "none":
        tpo_shape = f"{base_shape}_with_{rejection_high}_upper_rejection"
    if rejection_low != "none":
        tpo_shape = f"{tpo_shape}_with_{rejection_low}_lower_rejection"  # append if both

    # Naked prior levels (high-edge confluence — uses existing prior_day data)
    naked_levels = {
        "prior_poc": "NA",
        "prior_vah": "NA",
        "prior_val": "NA",
        "prior_high": "NA",
        "prior_low": "NA"
    }

    if prior_day:
        prior_poc = prior_day.get('poc')
        prior_vah = prior_day.get('vah')
        prior_val = prior_day.get('val')
        prior_high = prior_day.get('high')
        prior_low = prior_day.get('low')

        if prior_poc:
            naked_levels["prior_poc"] = "naked" if abs(current_price - prior_poc) > 10 else "tested"  # simple buffer
        if prior_vah:
            naked_levels["prior_vah"] = "naked_resistance" if current_high < prior_vah else "tested"
        if prior_val:
            naked_levels["prior_val"] = "naked_support" if current_low > prior_val else "tested"
        if prior_high:
            naked_levels["prior_high"] = "naked_resistance" if current_high < prior_high else "tested"
        if prior_low:
            naked_levels["prior_low"] = "naked_support" if current_low > prior_low else "tested"

    return {
        "current_poc": round(current_poc, 2),
        "current_vah": round(current_vah, 2),
        "current_val": round(current_val, 2),
        "single_prints_above_vah": int(single_above),
        "single_prints_below_val": int(single_below),
        "poor_high": int(poor_high),
        "poor_low": int(poor_low),
        "effective_poor_high": int(effective_poor_high),
        "effective_poor_low": int(effective_poor_low),
        "rejection_at_high": rejection_high,
        "rejection_at_low": rejection_low,
        "fattening_zone": fattening,
        "tpo_shape": tpo_shape,
        "naked_levels": naked_levels,
        "note": "Compact TPO facts for ROCKIT – single prints = rejection, fattening = acceptance. Rejection flags + effective poor + naked priors prioritize exhaustion/fade reads."
    }