"""
Diagnostic: Failed Breakout / IB Sweep Reclaim Study

Scans 259 sessions for scenarios where price breaks outside IB then fails
and reclaims back inside. These are the "inverse" of Trend Bull/Bear --
every failed breakout is a mean reversion trade.

Scenarios studied:
1. FAILED BULL: Price accepts above IBH (2+ closes), then reclaims below IBH
   -> SHORT to VWAP, IB mid, or IBL
2. FAILED BEAR: Price accepts below IBL (2+ closes), then reclaims above IBL
   -> LONG to VWAP, IB mid, or IBH
3. IBH SWEEP RECLAIM: Price sweeps IBH (wick or 1 close), immediately reclaims
   -> SHORT to VWAP or IB mid (faster version, no acceptance needed)
4. IBL SWEEP RECLAIM: Price sweeps IBL (wick or 1 close), immediately reclaims
   -> LONG to VWAP or IB mid

For each scenario, we measure:
- How often it occurs (frequency)
- MFE (max favorable excursion) -- how far price goes in our favor
- MAE (max adverse excursion) -- how far against before hitting target
- Simulated P&L with fixed stop/target combos
"""

import sys
import warnings
import io
import numpy as np

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from config.constants import IB_BARS_1MIN
from config.instruments import get_instrument

# ================================================================
# CONFIG
# ================================================================
ACCEPTANCE_BARS = 2          # Bars outside IB to count as "accepted"
SWEEP_MAX_BARS = 3           # Max bars outside IB for a "sweep" (quick poke)
RECLAIM_CONFIRM_BARS = 2     # Bars back inside IB to confirm reclaim
ENTRY_DEADLINE_BAR = 240     # No entries after bar 240 (~14:00 ET from 9:30)
MIN_IB_RANGE = 30            # Skip sessions with IB < 30 pts

instrument = get_instrument('MNQ')
point_value = instrument.point_value  # $2 for MNQ
slippage = 2 * instrument.tick_size   # 2 ticks round trip
commission = instrument.commission * 2  # round trip

# Target levels to test (as fraction of IB range from entry)
TARGET_LEVELS = {
    'VWAP': 'vwap',       # Dynamic -- actual VWAP distance
    'IB_MID': 0.50,       # Half of IB range
    'OPPOSING_IB': 1.00,  # Full IB range (IBH->IBL or IBL->IBH)
}

# Stop levels to test (as fraction of IB range beyond entry)
STOP_LEVELS = [0.15, 0.25, 0.40]


# ================================================================
# LOAD DATA
# ================================================================
print("Loading data...")
full_df = load_csv('NQ')
df = filter_rth(full_df)
df = compute_all_features(df)
sessions = sorted(df['session_date'].unique())
print(f"Loaded {len(sessions)} sessions\n")


# ================================================================
# SCAN SESSIONS
# ================================================================

scenarios = {
    'FAILED_BULL': [],    # Accepted above IBH, then reclaimed below
    'FAILED_BEAR': [],    # Accepted below IBL, then reclaimed above
    'SWEEP_IBH': [],      # Quick sweep of IBH, immediate reclaim
    'SWEEP_IBL': [],      # Quick sweep of IBL, immediate reclaim
}

for session_date in sessions:
    session_df = df[df['session_date'] == session_date].copy()
    session_str = str(session_date)

    if len(session_df) < IB_BARS_1MIN:
        continue

    ib_df = session_df.head(IB_BARS_1MIN)
    ib_high = ib_df['high'].max()
    ib_low = ib_df['low'].min()
    ib_range = ib_high - ib_low
    ib_mid = (ib_high + ib_low) / 2

    if ib_range < MIN_IB_RANGE:
        continue

    post_ib = session_df.iloc[IB_BARS_1MIN:].reset_index(drop=True)
    if len(post_ib) < 20:
        continue

    # === Track state for acceptance and sweep detection ===
    consec_above_ibh = 0
    consec_below_ibl = 0
    max_consec_above = 0
    max_consec_below = 0

    # Bull side: accepted above IBH then reclaimed
    bull_accepted = False
    bull_accept_bar = -1
    bull_accept_high = ib_high  # track how far above IBH it went
    bull_reclaimed = False
    bull_reclaim_bar = -1

    # Bear side: accepted below IBL then reclaimed
    bear_accepted = False
    bear_accept_bar = -1
    bear_accept_low = ib_low
    bear_reclaimed = False
    bear_reclaim_bar = -1

    # Sweep tracking (quick poke outside then back)
    sweep_ibh_events = []  # (bar_idx, high_reached)
    sweep_ibl_events = []  # (bar_idx, low_reached)

    # First pass: identify all events
    bars_above_ibh = 0  # consecutive closes above
    bars_below_ibl = 0  # consecutive closes below
    first_above_bar = -1
    first_below_bar = -1
    peak_above = ib_high
    trough_below = ib_low

    for bar_idx in range(len(post_ib)):
        bar = post_ib.iloc[bar_idx]
        close = bar['close']
        high = bar['high']
        low = bar['low']

        # --- BULL SIDE (above IBH) ---
        if close > ib_high:
            bars_above_ibh += 1
            if bars_above_ibh == 1:
                first_above_bar = bar_idx
            if high > peak_above:
                peak_above = high

            # Check for acceptance
            if bars_above_ibh >= ACCEPTANCE_BARS and not bull_accepted:
                bull_accepted = True
                bull_accept_bar = bar_idx
                bull_accept_high = peak_above
        else:
            # Close back inside IB (or below)
            if bars_above_ibh > 0:
                if bull_accepted and not bull_reclaimed and close <= ib_high:
                    # FAILED BULL: accepted above, now back inside
                    bull_reclaimed = True
                    bull_reclaim_bar = bar_idx

                elif 0 < bars_above_ibh <= SWEEP_MAX_BARS and not bull_accepted:
                    # SWEEP IBH: quick poke above, back inside
                    if close <= ib_high:
                        sweep_ibh_events.append({
                            'bar_idx': bar_idx,
                            'peak': peak_above,
                            'bars_outside': bars_above_ibh,
                        })

            bars_above_ibh = 0
            peak_above = ib_high

        # --- BEAR SIDE (below IBL) ---
        if close < ib_low:
            bars_below_ibl += 1
            if bars_below_ibl == 1:
                first_below_bar = bar_idx
            if low < trough_below:
                trough_below = low

            if bars_below_ibl >= ACCEPTANCE_BARS and not bear_accepted:
                bear_accepted = True
                bear_accept_bar = bar_idx
                bear_accept_low = trough_below
        else:
            if bars_below_ibl > 0:
                if bear_accepted and not bear_reclaimed and close >= ib_low:
                    bear_reclaimed = True
                    bear_reclaim_bar = bar_idx

                elif 0 < bars_below_ibl <= SWEEP_MAX_BARS and not bear_accepted:
                    if close >= ib_low:
                        sweep_ibl_events.append({
                            'bar_idx': bar_idx,
                            'trough': trough_below,
                            'bars_outside': bars_below_ibl,
                        })

            bars_below_ibl = 0
            trough_below = ib_low

    # === Second pass: simulate trades for identified events ===

    def simulate_trade(post_ib_df, entry_bar, direction, entry_price,
                       ib_high, ib_low, ib_range, ib_mid):
        """
        Simulate a trade from entry_bar forward. Track MFE, MAE,
        and whether various target levels are hit before stops.

        Returns dict with results for each stop/target combo.
        """
        remaining = post_ib_df.iloc[entry_bar + 1:]
        if len(remaining) == 0:
            return None

        vwap_at_entry = post_ib_df.iloc[entry_bar].get('vwap', ib_mid)
        if vwap_at_entry is None or np.isnan(vwap_at_entry):
            vwap_at_entry = ib_mid

        results = {
            'session': session_str,
            'entry_bar': entry_bar,
            'direction': direction,
            'entry_price': entry_price,
            'ib_range': ib_range,
            'ib_high': ib_high,
            'ib_low': ib_low,
            'vwap_at_entry': vwap_at_entry,
        }

        # Track MFE and MAE
        mfe = 0.0  # max favorable
        mae = 0.0  # max adverse

        for _, future_bar in remaining.iterrows():
            if direction == 'SHORT':
                favorable = entry_price - future_bar['low']
                adverse = future_bar['high'] - entry_price
            else:  # LONG
                favorable = future_bar['high'] - entry_price
                adverse = entry_price - future_bar['low']

            mfe = max(mfe, favorable)
            mae = max(mae, adverse)

        results['mfe_pts'] = mfe
        results['mae_pts'] = mae
        results['mfe_ib_pct'] = mfe / ib_range if ib_range > 0 else 0
        results['mae_ib_pct'] = mae / ib_range if ib_range > 0 else 0

        # Simulate stop/target combos
        for stop_mult in STOP_LEVELS:
            stop_dist = ib_range * stop_mult

            for tgt_name, tgt_val in TARGET_LEVELS.items():
                if tgt_name == 'VWAP':
                    if direction == 'SHORT':
                        tgt_dist = entry_price - vwap_at_entry
                    else:
                        tgt_dist = vwap_at_entry - entry_price
                    if tgt_dist <= 0:
                        # VWAP is wrong side, skip
                        continue
                else:
                    tgt_dist = ib_range * tgt_val

                # Walk forward bar by bar
                hit_target = False
                hit_stop = False
                exit_price = None
                exit_type = 'EOD'

                for _, fb in remaining.iterrows():
                    if direction == 'SHORT':
                        # Check stop first (high hits stop)
                        if fb['high'] >= entry_price + stop_dist:
                            hit_stop = True
                            exit_price = entry_price + stop_dist
                            exit_type = 'STOP'
                            break
                        # Check target (low hits target)
                        if fb['low'] <= entry_price - tgt_dist:
                            hit_target = True
                            exit_price = entry_price - tgt_dist
                            exit_type = 'TARGET'
                            break
                    else:  # LONG
                        if fb['low'] <= entry_price - stop_dist:
                            hit_stop = True
                            exit_price = entry_price - stop_dist
                            exit_type = 'STOP'
                            break
                        if fb['high'] >= entry_price + tgt_dist:
                            hit_target = True
                            exit_price = entry_price + tgt_dist
                            exit_type = 'TARGET'
                            break

                if exit_price is None:
                    # EOD exit
                    exit_price = remaining.iloc[-1]['close']

                if direction == 'SHORT':
                    pnl_pts = entry_price - exit_price
                else:
                    pnl_pts = exit_price - entry_price

                pnl_dollar = pnl_pts * point_value - slippage * point_value - commission
                rr = pnl_pts / stop_dist if stop_dist > 0 else 0

                key = f"stop{int(stop_mult*100)}_{tgt_name}"
                results[f'{key}_pnl'] = pnl_dollar
                results[f'{key}_exit'] = exit_type
                results[f'{key}_rr'] = rr
                results[f'{key}_win'] = 1 if pnl_dollar > 0 else 0

        return results

    # --- FAILED BULL: SHORT from reclaim ---
    if bull_reclaimed and bull_reclaim_bar < ENTRY_DEADLINE_BAR:
        entry_price = post_ib.iloc[bull_reclaim_bar]['close']
        result = simulate_trade(
            post_ib, bull_reclaim_bar, 'SHORT', entry_price,
            ib_high, ib_low, ib_range, ib_mid)
        if result:
            result['accept_bar'] = bull_accept_bar
            result['peak_extension'] = bull_accept_high - ib_high
            result['peak_ext_pct'] = (bull_accept_high - ib_high) / ib_range
            scenarios['FAILED_BULL'].append(result)

    # --- FAILED BEAR: LONG from reclaim ---
    if bear_reclaimed and bear_reclaim_bar < ENTRY_DEADLINE_BAR:
        entry_price = post_ib.iloc[bear_reclaim_bar]['close']
        result = simulate_trade(
            post_ib, bear_reclaim_bar, 'LONG', entry_price,
            ib_high, ib_low, ib_range, ib_mid)
        if result:
            result['accept_bar'] = bear_accept_bar
            result['trough_extension'] = ib_low - bear_accept_low
            result['trough_ext_pct'] = (ib_low - bear_accept_low) / ib_range
            scenarios['FAILED_BEAR'].append(result)

    # --- SWEEP IBH: SHORT from first reclaim ---
    for sweep in sweep_ibh_events[:2]:  # max 2 per session
        if sweep['bar_idx'] < ENTRY_DEADLINE_BAR:
            entry_price = post_ib.iloc[sweep['bar_idx']]['close']
            result = simulate_trade(
                post_ib, sweep['bar_idx'], 'SHORT', entry_price,
                ib_high, ib_low, ib_range, ib_mid)
            if result:
                result['peak'] = sweep['peak']
                result['bars_outside'] = sweep['bars_outside']
                result['sweep_depth'] = sweep['peak'] - ib_high
                scenarios['SWEEP_IBH'].append(result)

    # --- SWEEP IBL: LONG from first reclaim ---
    for sweep in sweep_ibl_events[:2]:
        if sweep['bar_idx'] < ENTRY_DEADLINE_BAR:
            entry_price = post_ib.iloc[sweep['bar_idx']]['close']
            result = simulate_trade(
                post_ib, sweep['bar_idx'], 'LONG', entry_price,
                ib_high, ib_low, ib_range, ib_mid)
            if result:
                result['trough'] = sweep['trough']
                result['bars_outside'] = sweep['bars_outside']
                result['sweep_depth'] = ib_low - sweep['trough']
                scenarios['SWEEP_IBL'].append(result)


# ================================================================
# REPORT
# ================================================================
print("=" * 120)
print("  FAILED BREAKOUT / IB SWEEP RECLAIM STUDY -- 259 Sessions")
print("=" * 120)

for scenario_name, trades in scenarios.items():
    n = len(trades)
    if n == 0:
        print(f"\n{'─'*80}")
        print(f"  {scenario_name}: 0 occurrences")
        continue

    print(f"\n{'─'*120}")
    direction = trades[0]['direction']
    print(f"  {scenario_name} ({direction}) -- {n} occurrences")
    print(f"{'─'*120}")

    # MFE/MAE analysis
    mfes = [t['mfe_pts'] for t in trades]
    maes = [t['mae_pts'] for t in trades]
    mfe_pcts = [t['mfe_ib_pct'] for t in trades]
    mae_pcts = [t['mae_ib_pct'] for t in trades]
    ib_ranges = [t['ib_range'] for t in trades]

    print(f"\n  MFE (max favorable excursion):")
    print(f"    Mean: {np.mean(mfes):>7.1f} pts ({np.mean(mfe_pcts):>.2f}x IB)")
    print(f"    Med:  {np.median(mfes):>7.1f} pts ({np.median(mfe_pcts):>.2f}x IB)")
    print(f"    P25:  {np.percentile(mfes, 25):>7.1f} pts")
    print(f"    P75:  {np.percentile(mfes, 75):>7.1f} pts")

    print(f"\n  MAE (max adverse excursion):")
    print(f"    Mean: {np.mean(maes):>7.1f} pts ({np.mean(mae_pcts):>.2f}x IB)")
    print(f"    Med:  {np.median(maes):>7.1f} pts ({np.median(mae_pcts):>.2f}x IB)")
    print(f"    P25:  {np.percentile(maes, 25):>7.1f} pts")
    print(f"    P75:  {np.percentile(maes, 75):>7.1f} pts")

    print(f"\n  IB Range: mean={np.mean(ib_ranges):.0f}, med={np.median(ib_ranges):.0f}")

    # Extra context for each scenario
    if scenario_name == 'FAILED_BULL' and 'peak_ext_pct' in trades[0]:
        exts = [t['peak_ext_pct'] for t in trades]
        print(f"  Peak extension above IBH: mean={np.mean(exts):.2f}x IB, med={np.median(exts):.2f}x IB")

    if scenario_name == 'FAILED_BEAR' and 'trough_ext_pct' in trades[0]:
        exts = [t['trough_ext_pct'] for t in trades]
        print(f"  Trough extension below IBL: mean={np.mean(exts):.2f}x IB, med={np.median(exts):.2f}x IB")

    if scenario_name.startswith('SWEEP') and 'sweep_depth' in trades[0]:
        depths = [t['sweep_depth'] for t in trades]
        bars_out = [t['bars_outside'] for t in trades]
        print(f"  Sweep depth: mean={np.mean(depths):.1f} pts, med={np.median(depths):.1f} pts")
        print(f"  Bars outside IB: mean={np.mean(bars_out):.1f}")

    # Stop/Target combo results
    print(f"\n  {'Stop':>6s} | {'Target':>12s} | {'Trades':>6s} | {'WR':>6s} | {'Net P&L':>10s} | {'Avg P&L':>9s} | {'PF':>5s} | {'Avg RR':>6s}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*9}-+-{'-'*5}-+-{'-'*6}")

    best_pnl = -999999
    best_combo = ""

    for stop_mult in STOP_LEVELS:
        for tgt_name in TARGET_LEVELS:
            key = f"stop{int(stop_mult*100)}_{tgt_name}"
            pnl_key = f'{key}_pnl'

            # Some trades may not have VWAP combo
            valid = [t for t in trades if pnl_key in t]
            if not valid:
                continue

            wins = sum(t[f'{key}_win'] for t in valid)
            total_pnl = sum(t[pnl_key] for t in valid)
            avg_pnl = total_pnl / len(valid)
            wr = wins / len(valid) * 100
            rrs = [t[f'{key}_rr'] for t in valid]
            avg_rr = np.mean(rrs)

            win_pnls = [t[pnl_key] for t in valid if t[pnl_key] > 0]
            loss_pnls = [t[pnl_key] for t in valid if t[pnl_key] <= 0]
            gross_win = sum(win_pnls) if win_pnls else 0
            gross_loss = abs(sum(loss_pnls)) if loss_pnls else 0.01
            pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

            marker = ""
            if total_pnl > best_pnl:
                best_pnl = total_pnl
                best_combo = f"stop={stop_mult:.0%} tgt={tgt_name}"

            if total_pnl > 500:
                marker = " <-- GOOD"
            elif total_pnl < -500:
                marker = " <-- BAD"

            print(f"  {stop_mult:>5.0%}  | {tgt_name:>12s} | {len(valid):>6d} | {wr:>5.1f}% | ${total_pnl:>9,.0f} | ${avg_pnl:>8,.0f} | {pf:>5.2f} | {avg_rr:>+5.2f}{marker}")

    print(f"\n  >> Best combo: {best_combo} (${best_pnl:,.0f})")

    # Print individual trades for best combo
    if best_pnl > 0 and n <= 50:
        print(f"\n  Individual trades ({best_combo}):")
        # Find the best key
        for stop_mult in STOP_LEVELS:
            for tgt_name in TARGET_LEVELS:
                key = f"stop{int(stop_mult*100)}_{tgt_name}"
                valid = [t for t in trades if f'{key}_pnl' in t]
                if valid and sum(t[f'{key}_pnl'] for t in valid) == best_pnl:
                    for t in valid:
                        pnl = t[f'{key}_pnl']
                        exit_type = t[f'{key}_exit']
                        w = "W" if pnl > 0 else "L"
                        print(f"    {t['session']:>12s}  {t['direction']:>5s}  "
                              f"entry={t['entry_price']:>9.2f}  IB={t['ib_range']:>6.0f}  "
                              f"${pnl:>+8,.0f}  {exit_type:>6s}  [{w}]")
                    break
            else:
                continue
            break


# ================================================================
# SUMMARY
# ================================================================
print(f"\n{'='*120}")
print(f"  SUMMARY")
print(f"{'='*120}")
print(f"\n  {'Scenario':<20s} | {'N':>5s} | {'Direction':>9s} | {'MFE med':>8s} | {'MAE med':>8s} | {'MFE/MAE':>7s}")
print(f"  {'-'*20}-+-{'-'*5}-+-{'-'*9}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}")

for name, trades in scenarios.items():
    if not trades:
        print(f"  {name:<20s} | {'0':>5s} | {'--':>9s} | {'--':>8s} | {'--':>8s} | {'--':>7s}")
        continue
    direction = trades[0]['direction']
    med_mfe = np.median([t['mfe_pts'] for t in trades])
    med_mae = np.median([t['mae_pts'] for t in trades])
    ratio = med_mfe / med_mae if med_mae > 0 else float('inf')
    print(f"  {name:<20s} | {len(trades):>5d} | {direction:>9s} | {med_mfe:>7.0f}p | {med_mae:>7.0f}p | {ratio:>6.2f}x")

total = sum(len(t) for t in scenarios.values())
print(f"\n  Total events: {total} across {len(sessions)} sessions")
print(f"  Coverage: {total/len(sessions)*100:.0f}% of sessions have at least one failed breakout")

# Count sessions with at least one event
sessions_with_events = set()
for trades in scenarios.values():
    for t in trades:
        sessions_with_events.add(t['session'])
print(f"  Unique sessions with events: {len(sessions_with_events)} ({len(sessions_with_events)/len(sessions)*100:.0f}%)")
