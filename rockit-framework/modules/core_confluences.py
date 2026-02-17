# modules/core_confluences.py
# Dedicated module for precomputing all critical comparisons.
# Takes merged intraday data from orchestrator and outputs booleans/strings.
# Thresholds are tunable – start with these based on your oath (e.g., 20pts for significance).

from datetime import time

def to_float(value):
    """
    Convert value to float, handling 'not_available' strings and None.
    Returns None if conversion fails.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        if value.lower() == 'not_available' or value.strip() == '':
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    return None

def get_core_confluences(intraday_data, current_time_str="11:45"):
    """
    Computes explicit confluences from raw intraday values.
    Handles partial data gracefully (returns False/None/"none" if values missing).
    """
    # Extract raw values from sub-sections
    ib = intraday_data.get('ib', {})  # from ib_location.py
    volume_profile = intraday_data.get('volume_profile', {})  # from volume_profile.py
    tpo_profile = intraday_data.get('tpo_profile', {})  # from tpo_profile.py
    migration = intraday_data.get('dpoc_migration', {})  # from dpoc_migration.py

    # Flexible DPoC extraction (check tpo first, then volume)
    current_dpoc = to_float(tpo_profile.get('current_poc'))
    if current_dpoc is None:
        current_dpoc = to_float(volume_profile.get('current_session', {}).get('poc'))

    # Other extracts - convert to float where needed
    ib_high = to_float(ib.get('ib_high'))
    ib_low = to_float(ib.get('ib_low'))
    current_close = to_float(ib.get('current_close'))
    vah = to_float(tpo_profile.get('current_vah')) or to_float(volume_profile.get('current_session', {}).get('vah'))
    val = to_float(tpo_profile.get('current_val')) or to_float(volume_profile.get('current_session', {}).get('val'))
    migration_steps = migration.get('steps_since_1030', 0)
    single_above = tpo_profile.get('single_prints_above_vah', 0)
    single_below = tpo_profile.get('single_prints_below_val', 0)
    fattening_zone = tpo_profile.get('fattening_zone', 'inside_va')

    # DEBUG: Log types and values to diagnose the issue
    print(f"[DEBUG] current_dpoc: value={current_dpoc}, type={type(current_dpoc)}")
    print(f"[DEBUG] vah: value={vah}, type={type(vah)}")
    print(f"[DEBUG] val: value={val}, type={type(val)}")
    print(f"[DEBUG] ib_high: value={ib_high}, type={type(ib_high)}")
    print(f"[DEBUG] ib_low: value={ib_low}, type={type(ib_low)}")
    print(f"[DEBUG] current_close: value={current_close}, type={type(current_close)}")
    print(f"[DEBUG] tpo_profile keys: {list(tpo_profile.keys())}")
    print(f"[DEBUG] volume_profile current_session keys: {list(volume_profile.get('current_session', {}).keys())}")

    # Early return if no data available
    if current_dpoc is None and vah is None and val is None:
        print("[WARNING] No numeric data available for core_confluences - returning empty result")
        return {
            "note": "no_data_available",
            "ib_acceptance": {"close_above_ibh": False, "close_below_ibl": False, "price_accepted_higher": "No", "price_accepted_lower": "No"},
            "dpoc_vs_ib": {"dpoc_above_ibh": False, "dpoc_below_ibl": False, "dpoc_extreme_shift": "none"},
            "dpoc_compression": {"compressing_against_vah": False, "compressing_against_val": False, "compression_bias": "none"},
            "price_location": {"in_upper_third": False, "in_lower_third": False, "in_middle": False, "location_label": "unknown"},
            "tpo_signals": {"single_prints_above": False, "single_prints_below": False, "fattening_upper": False, "fattening_lower": False},
            "migration": {"significant_up": False, "significant_down": False, "net_direction": "flat", "pts_since_1030": 0}
        }

    # Time check
    current_time = time(*map(int, current_time_str.split(':')))
    is_post_1030 = current_time >= time(10, 30)

    # Thresholds
    COMPRESSION_THRESHOLD_PTS = 20.0
    SIGNIFICANT_MIGRATION_PTS = 20.0

    confluences = {
        "ib_acceptance": {
            "close_above_ibh": bool(ib_high is not None and current_close > ib_high),
            "close_below_ibl": bool(ib_low is not None and current_close < ib_low),
            "price_accepted_higher": "Yes" if ib_high is not None and current_close > ib_high else "No",
            "price_accepted_lower": "Yes" if ib_low is not None and current_close < ib_low else "No"
        },
        "dpoc_vs_ib": {
            "dpoc_above_ibh": bool(current_dpoc is not None and ib_high is not None and current_dpoc > ib_high),
            "dpoc_below_ibl": bool(current_dpoc is not None and ib_low is not None and current_dpoc < ib_low),
            "dpoc_extreme_shift": (
                "bullish" if current_dpoc is not None and ib_high is not None and current_dpoc > ib_high else
                "bearish" if current_dpoc is not None and ib_low is not None and current_dpoc < ib_low else
                "none"
            )
        },
        "dpoc_compression": {
            "compressing_against_vah": bool(
                current_dpoc is not None and vah is not None and val is not None and
                abs(current_dpoc - vah) <= COMPRESSION_THRESHOLD_PTS and
                current_dpoc >= val  # Pushing upper
            ),
            "compressing_against_val": bool(
                current_dpoc is not None and val is not None and vah is not None and
                abs(current_dpoc - val) <= COMPRESSION_THRESHOLD_PTS and
                current_dpoc <= vah  # Pushing lower
            ),
            "compression_bias": (
                "aggressive_bullish" if (
                    current_dpoc is not None and vah is not None and val is not None and
                    abs(current_dpoc - vah) <= COMPRESSION_THRESHOLD_PTS and
                    current_dpoc >= val
                ) else
                "aggressive_bearish" if (
                    current_dpoc is not None and val is not None and vah is not None and
                    abs(current_dpoc - val) <= COMPRESSION_THRESHOLD_PTS and
                    current_dpoc <= vah
                ) else
                "none"
            )
        },
        "price_location": {
            "in_upper_third": ib.get('price_vs_ib') == "upper_third_hug",
            "in_lower_third": ib.get('price_vs_ib') == "lower_third_hug",
            "in_middle": ib.get('price_vs_ib') == "middle",
            "location_label": ib.get('price_vs_ib', "unknown")
        },
        "tpo_signals": {
            "single_prints_above": bool(single_above > 0),
            "single_prints_below": bool(single_below > 0),
            "fattening_upper": fattening_zone.lower() in ["at_vah", "above_vah"],  # Case-insensitive
            "fattening_lower": fattening_zone.lower() in ["at_val", "below_val"]
        },
        "migration": {
            "significant_up": bool(is_post_1030 and migration_steps > SIGNIFICANT_MIGRATION_PTS),
            "significant_down": bool(is_post_1030 and migration_steps < -SIGNIFICANT_MIGRATION_PTS),
            "net_direction": migration.get('migration_direction', 'flat'),
            "pts_since_1030": migration_steps
        }
      }

    confluences["note"] = "Precomputed confluences – use these for bias logic, quote raw values from intraday"

    return confluences