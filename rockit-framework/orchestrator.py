# orchestrator.py
# Purpose: Merges JSON fragments from all modules into one final compact snapshot JSON.
# Keeps everything deterministic and modular – add new modules here as needed.

import json
import os
import numpy as np
from modules.loader import load_nq_csv
from modules.premarket import get_premarket
from modules.ib_location import get_ib_location
from modules.wick_parade import get_wick_parade
from modules.dpoc_migration import get_dpoc_migration
from modules.volume_profile import get_volume_profile
from modules.tpo_profile import get_tpo_profile
from modules.ninety_min_pd_arrays import get_ninety_min_pd_arrays
from modules.fvg_detection import get_fvg_detection
from modules.core_confluences import get_core_confluences  # NEW: For precomputed signals

# Future: add more e.g., cross_market, vix_regime, intraday_sampling

def clean_for_json(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

def generate_snapshot(config):
    """
    Generates a snapshot JSON by calling modules and merging their outputs.
    Loads NQ primary, ES/YM optional for SMT/cross-market.
    """
    # Load NQ (required) – extended for premarket, current for intraday
    df_extended, df_current = load_nq_csv(config['csv_paths']['nq'], config['session_date'])
    
    # Load ES/YM if paths provided (use extended/current as needed)
    df_es_extended = df_es_current = None
    if 'es' in config['csv_paths'] and config['csv_paths']['es']:
        df_es_extended, df_es_current = load_nq_csv(config['csv_paths']['es'], config['session_date'])
    
    df_ym_extended = df_ym_current = None
    if 'ym' in config['csv_paths'] and config['csv_paths']['ym']:
        df_ym_extended, df_ym_current = load_nq_csv(config['csv_paths']['ym'], config['session_date'])
    
    # IB data (used by several modules)
    ib_data = get_ib_location(df_current, config['current_time'])
    
    # Volume profile first — needed for prior_day in tpo_profile
    volume_profile_result = get_volume_profile(df_extended, df_current, config['current_time'])
    prior_day = volume_profile_result.get('previous_day', {})

    # Collect intraday fragments
    intraday_data = {
        "ib": ib_data,
        "wick_parade": get_wick_parade(df_current, config['current_time']),
        "dpoc_migration": get_dpoc_migration(
            df_current, 
            config['current_time'], 
            atr14_current=ib_data.get('atr14'),
            current_close=ib_data.get('current_close')
        ),
        "volume_profile": volume_profile_result,
        "tpo_profile": get_tpo_profile(
            df_current, 
            config['current_time'], 
            prior_day=prior_day
        ),
        "ninety_min_pd_arrays": get_ninety_min_pd_arrays(df_current, config['current_time']),
        "fvg_detection": get_fvg_detection(df_extended, df_current, config['current_time']),
        # Add other intraday modules here (e.g., "cross_market": get_cross_market(...))
    }
    
    # Compute core confluences from the collected intraday data
    core_confluences_data = get_core_confluences(intraday_data, config['current_time'])
    
    # Assemble full snapshot
    snapshot = {
        "session_date": config['session_date'],
        "current_et_time": config['current_time'],
        "premarket": get_premarket(df_extended, df_es_extended, df_ym_extended, session_date=config['session_date']),
        "intraday": intraday_data,
        "core_confluences": core_confluences_data
    }
    
    # Clean snapshot for JSON serialization
    snapshot = clean_for_json(snapshot)

    # Save to output dir
    os.makedirs(config['output_dir'], exist_ok=True)
    filename = f"{config['session_date']}_{config['current_time'].replace(':', '')}.json"
    path = os.path.join(config['output_dir'], filename)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=4, ensure_ascii=False)
    
    print(f"Snapshot saved: {path}")
    return snapshot