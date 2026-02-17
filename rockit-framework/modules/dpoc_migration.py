# modules/dpoc_migration.py
import pandas as pd
from datetime import time
from collections import Counter
import numpy as np

def get_dpoc_migration(df_nq, current_time_str="11:45", atr14_current=None, current_close=None):
    """
    Further enhanced DPOC migration analysis (post-10:30 only).
    New focus:
    - Detect "on the move" (trending with momentum)
    - "stabilizing" (cluster forming → potential floor/ceiling hold)
    - "potential BPR/reversal" (exhausted trend → stabilization + price reclaiming opposite)
    Handles classic scenarios:
      • Step down repeatedly → trending strong/fading
      • Holds floor, small bounce, then new low → fading → continues trend
      • Holds floor + sustained reclaim → potential reversal
    All deterministic, no lookahead.
    """
    current_time = pd.to_datetime(current_time_str).time()
    available_df = df_nq[df_nq.index.time <= current_time].copy()

    if len(available_df) == 0:
        return {"note": "no_data_yet"}

    session_df = available_df.between_time('09:30', current_time_str)
    post_ib_df = session_df[session_df.index.time >= time(10, 30)]

    if len(post_ib_df) == 0:
        return {"migration_status": "pre_1030", "note": "No post-IB data yet"}

    post_ib_df = post_ib_df.copy()
    post_ib_df['slice_start'] = post_ib_df.index.floor('30min')
    slices = sorted(post_ib_df['slice_start'].unique())

    # Thresholds (ATR-aware)
    atr = atr14_current if atr14_current else 25
    jump_threshold = max(20, atr * 1.0)          # Big move in slice
    cluster_threshold = max(15, atr * 0.6)       # Tight cluster = stabilizing
    exhausted_threshold_pts = max(50, atr * 3)  # Big prior migration to consider "exhausted"
    velocity_strong = max(25, atr * 0.8)
    velocity_low = max(8, atr * 0.3)

    dpoc_history = []
    completed_dpocs = []
    prev_dpoc = None

    for slice_start in slices:
        slice_end = slice_start + pd.Timedelta(minutes=30)
        slice_df = post_ib_df[(post_ib_df.index >= slice_start) & (post_ib_df.index < slice_end)]
        developing = len(slice_df) < 4

        if not developing:
            price_vol = Counter()
            for _, row in slice_df.iterrows():
                price = round(row['vwap'] * 4) / 4.0
                price_vol[price] += int(row['volume'])

            poc = max(price_vol, key=price_vol.get) if price_vol else slice_df['close'].mean()
            poc = round(poc, 2)
            completed_dpocs.append(poc)
            prev_dpoc = poc
        else:
            poc = prev_dpoc if prev_dpoc is not None else None

        delta = round(poc - prev_dpoc, 2) if prev_dpoc is not None and poc is not None else 0.0
        jump = abs(delta) >= jump_threshold

        entry = {
            "slice": slice_start.strftime('%H:%M'),
            "dpoc": poc,
            "delta_pts": delta,
            "jump": jump,
            "developing": developing,
            "bar_count": len(slice_df)
        }
        dpoc_history.append(entry)

    if len(completed_dpocs) < 2:
        return {
            "dpoc_history": dpoc_history,
            "migration_status": "insufficient_completed_slices",
            "note": "Need ≥2 completed slices for full analysis"
        }

    # Core metrics (same as before)
    first_dpoc = completed_dpocs[0]
    current_dpoc = completed_dpocs[-1]
    direction = "up" if current_dpoc > first_dpoc else "down" if current_dpoc < first_dpoc else "flat"
    net_migration = round(current_dpoc - first_dpoc, 2)

    peak = max(completed_dpocs) if direction != "down" else min(completed_dpocs)
    nadir = min(completed_dpocs) if direction == "down" else max(completed_dpocs)
    excursion = peak - first_dpoc if direction == "up" else first_dpoc - nadir
    relative_retain = (current_dpoc - first_dpoc) / excursion if direction == "up" and excursion > 0 else \
                      (first_dpoc - current_dpoc) / excursion if direction == "down" and excursion > 0 else 1.0
    relative_retain_pct = round(relative_retain * 100, 1)

    # Velocity (last ≤3 completed deltas)
    num_recent = min(3, len(completed_dpocs) - 1)
    deltas = [completed_dpocs[i+1] - completed_dpocs[i] for i in range(len(completed_dpocs)-1)]
    recent_deltas = deltas[-num_recent:] if deltas else []
    avg_velocity = round(np.mean(recent_deltas), 2) if recent_deltas else 0.0
    abs_velocity = abs(avg_velocity)

    # Acceleration/deceleration
    accelerating = decelerating = False
    if len(recent_deltas) >= 2:
        accel = recent_deltas[-1] - recent_deltas[-2]
        if abs(accel) >= 8:
            if np.sign(avg_velocity) == np.sign(accel):
                accelerating = True
            elif np.sign(avg_velocity) == -np.sign(accel):
                decelerating = True

    # === NEW: Stabilizing / Cluster Detection ===
    recent_cluster_dpocs = completed_dpocs[-min(4, len(completed_dpocs)):]
    cluster_range = round(max(recent_cluster_dpocs) - min(recent_cluster_dpocs), 2) if len(recent_cluster_dpocs) > 1 else 0.0
    is_stabilizing = cluster_range <= cluster_threshold and abs_velocity <= velocity_low

    # Direction changes in recent deltas (oscillation)
    direction_changes = sum(1 for i in range(1, len(recent_deltas)) if np.sign(recent_deltas[i]) != np.sign(recent_deltas[i-1]) and recent_deltas[i-1] != 0)
    oscillating = direction_changes >= 2

    # === NEW: Price vs DPOC Cluster (reclaim detection) ===
    cluster_mid = round(np.mean(recent_cluster_dpocs), 2)
    price_vs_cluster = "above" if current_close and current_close > cluster_mid + cluster_threshold/2 else \
                       "below" if current_close and current_close < cluster_mid - cluster_threshold/2 else "inside"
    reclaiming_opposite = False
    if direction == "down" and price_vs_cluster == "above":
        reclaiming_opposite = True
    elif direction == "up" and price_vs_cluster == "below":
        reclaiming_opposite = True

    # === NEW: Exhaustion Detection ===
    prior_exhausted = abs(net_migration) >= exhausted_threshold_pts and relative_retain_pct < 50

    # === NEW: DPOC Regime Classification ===
    if abs_velocity >= velocity_strong and not decelerating and relative_retain_pct >= 70:
        dpoc_regime = "trending_on_the_move"
    elif abs_velocity >= velocity_low and (decelerating or relative_retain_pct < 60):
        dpoc_regime = "trending_fading_momentum"
    elif is_stabilizing and not prior_exhausted:
        dpoc_regime = "stabilizing_hold forming_floor" if direction == "down" else "stabilizing_hold forming_ceiling"
    elif is_stabilizing and prior_exhausted and reclaiming_opposite:
        dpoc_regime = "potential_bpr_reversal"
    elif is_stabilizing and oscillating:
        dpoc_regime = "balancing_choppy"
    else:
        dpoc_regime = "transitional_unclear"

    return {
        "dpoc_history": dpoc_history,
        "direction": direction,
        "net_migration_pts": float(net_migration),
        "avg_velocity_per_30min": float(avg_velocity),
        "abs_velocity": float(abs_velocity),
        "relative_retain_percent": float(relative_retain_pct),
        "accelerating": bool(accelerating),
        "decelerating": bool(decelerating),
        "cluster_range_last_4": float(cluster_range),
        "is_stabilizing": bool(is_stabilizing),
        "price_vs_dpoc_cluster": price_vs_cluster,
        "reclaiming_opposite": bool(reclaiming_opposite),
        "prior_exhausted": bool(prior_exhausted),
        "dpoc_regime": dpoc_regime,
        "note": "DPOC regime classifies: trending_on_the_move (strong continuation), trending_fading_momentum (weakening), stabilizing_hold (floor/ceiling forming), potential_bpr_reversal (exhausted + reclaim), balancing_choppy. Covers step-down → hold → continue vs reverse scenarios."
    }