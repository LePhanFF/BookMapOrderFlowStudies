#!/usr/bin/env python
"""
OR Acceptance Funnel Diagnostic

Instruments every filter stage of the OR Acceptance strategy to understand
why 0 trades fired in 260 NQ sessions. Measures how many sessions pass
each gate, from London levels existing through to final signal emission.

Also tests relaxed acceptance conditions to find the sweet spot that
produces 20-40 trades while maintaining edge quality.
"""

import sys
from pathlib import Path
from datetime import timedelta
from collections import defaultdict

import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features

# Constants matching or_acceptance.py
OR_BARS = 15
EOR_BARS = 30
IB_BARS = 60
ATR_PERIOD = 14
ATR_STOP_MULT = 0.5
MIN_RISK_RATIO = 0.03
MAX_RISK_RATIO = 1.3


def compute_atr14(bars, n=ATR_PERIOD):
    if len(bars) < 3:
        return float((bars['high'] - bars['low']).mean()) if len(bars) > 0 else 20.0
    h, l, pc = bars['high'], bars['low'], bars['close'].shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n, min_periods=3).mean().iloc[-1]
    return float(atr) if not pd.isna(atr) else float((h - l).mean())


def compute_overnight_levels(full_df, session_str):
    """Compute overnight/London/Asia/PDH levels for a session (mirrors engine logic)."""
    sd = pd.Timestamp(session_str)
    prev_day = sd - timedelta(days=1)
    if prev_day.weekday() == 5:
        prev_day -= timedelta(days=1)
    elif prev_day.weekday() == 6:
        prev_day -= timedelta(days=2)

    ts = full_df['timestamp']
    levels = {}

    # Overnight: 18:00 prev day to 9:29 session day
    on_mask = (ts >= prev_day + timedelta(hours=18)) & (ts < sd + timedelta(hours=9, minutes=30))
    on_bars = full_df[on_mask]
    if len(on_bars) > 0:
        levels['overnight_high'] = on_bars['high'].max()
        levels['overnight_low'] = on_bars['low'].min()

    # Asia session: 20:00-00:00 prev day evening
    asia_mask = (ts >= prev_day + timedelta(hours=20)) & (ts < sd)
    asia = full_df[asia_mask]
    if len(asia) > 0:
        levels['asia_high'] = asia['high'].max()
        levels['asia_low'] = asia['low'].min()

    # London session: 02:00-05:00 session day
    london_mask = (ts >= sd + timedelta(hours=2)) & (ts < sd + timedelta(hours=5))
    london = full_df[london_mask]
    if len(london) > 0:
        levels['london_high'] = london['high'].max()
        levels['london_low'] = london['low'].min()

    # Prior day RTH
    prior_rth_mask = (ts >= prev_day + timedelta(hours=9, minutes=30)) & (ts <= prev_day + timedelta(hours=16))
    prior_rth = full_df[prior_rth_mask]
    if len(prior_rth) > 0:
        levels['pdh'] = prior_rth['high'].max()
        levels['pdl'] = prior_rth['low'].min()

    return levels


def check_acceptance_relaxed(bars, level, direction, eor_range,
                              max_wicks=5, close_pct_thresh=0.70, buffer_pct=0.10):
    """Check relaxed acceptance condition.

    Args:
        bars: DataFrame of bars to check
        level: The reference level price
        direction: 'SHORT' (price below level) or 'LONG' (price above level)
        eor_range: EOR range for buffer scaling
        max_wicks: Max bars allowed to violate the level
        close_pct_thresh: Min fraction of closes that must be on correct side
        buffer_pct: Max allowed spike past level as fraction of eor_range

    Returns:
        (passed, violations, close_pct, max_spike) tuple
    """
    n = len(bars)
    if n == 0:
        return False, 0, 0.0, 0.0

    if direction == 'SHORT':
        violations = int((bars['high'] > level).sum())
        closes_correct = int((bars['close'] < level).sum())
        max_spike = float(bars['high'].max()) - level
    else:
        violations = int((bars['low'] < level).sum())
        closes_correct = int((bars['close'] > level).sum())
        max_spike = level - float(bars['low'].min())

    close_pct = closes_correct / n
    buffer = buffer_pct * eor_range

    passed = (violations <= max_wicks
              and close_pct >= close_pct_thresh
              and max_spike <= buffer)
    return passed, violations, close_pct, max_spike


def run_funnel():
    print("=" * 70)
    print("  OR ACCEPTANCE FUNNEL DIAGNOSTIC")
    print("=" * 70)

    # Load data
    full_df = load_csv('NQ')
    df = filter_rth(full_df)
    df = compute_all_features(df)

    sessions = sorted(df['session_date'].dropna().unique())
    n_sessions = len(sessions)
    print(f"\nTotal sessions: {n_sessions}")

    # ═══════════════════════════════════════════════════════════
    # FUNNEL COUNTERS
    # ═══════════════════════════════════════════════════════════
    counters = defaultdict(lambda: {'SHORT': 0, 'LONG': 0})

    # Detailed tracking for sessions that pass relaxed conditions
    relaxed_details = []

    # Per-level acceptance counts (expanded levels)
    level_counters = defaultdict(lambda: {'SHORT': 0, 'LONG': 0})

    # Track the "near miss" distribution
    miss_distances = {'SHORT': [], 'LONG': []}

    for session_date in sessions:
        session_str = str(session_date)[:10]
        session_df = df[df['session_date'] == session_date].copy()

        if len(session_df) < IB_BARS:
            continue

        ib_bars = session_df.head(IB_BARS)
        or_bars = ib_bars.iloc[:OR_BARS]
        eor_bars = ib_bars.iloc[:EOR_BARS]

        or_high = or_bars['high'].max()
        or_low = or_bars['low'].min()
        eor_high = eor_bars['high'].max()
        eor_low = eor_bars['low'].min()
        eor_range = eor_high - eor_low

        # Gate 1: EOR range valid
        if eor_range < 10:
            continue
        counters['eor_valid']['SHORT'] += 1
        counters['eor_valid']['LONG'] += 1

        # Get overnight levels
        levels = compute_overnight_levels(full_df, session_str)
        london_high = levels.get('london_high')
        london_low = levels.get('london_low')
        overnight_high = levels.get('overnight_high')
        overnight_low = levels.get('overnight_low')
        asia_high = levels.get('asia_high')
        asia_low = levels.get('asia_low')
        pdh = levels.get('pdh')
        pdl = levels.get('pdl')

        # Gate 2: London levels exist
        if london_high:
            counters['london_exists']['LONG'] += 1
        if london_low:
            counters['london_exists']['SHORT'] += 1

        atr14 = compute_atr14(ib_bars)

        # ── Build all candidate levels ──────────────────────────
        short_levels = []  # price below these = SHORT acceptance
        if london_low: short_levels.append(('LDN_LOW', london_low))
        if overnight_low: short_levels.append(('ON_LOW', overnight_low))
        if pdl: short_levels.append(('PDL', pdl))
        if asia_low: short_levels.append(('ASIA_LOW', asia_low))

        long_levels = []  # price above these = LONG acceptance
        if london_high: long_levels.append(('LDN_HIGH', london_high))
        if overnight_high: long_levels.append(('ON_HIGH', overnight_high))
        if pdh: long_levels.append(('PDH', pdh))
        if asia_high: long_levels.append(('ASIA_HIGH', asia_high))

        # ── Check acceptance for each window and level ──────────
        windows = {
            'OR': ib_bars.iloc[:OR_BARS],
            'EOR': ib_bars.iloc[:EOR_BARS],
            'IB': ib_bars,
        }

        # Config variants to test
        configs = [
            {'name': 'strict',       'wicks': 0, 'close_pct': 1.00, 'buffer': 0.00},
            {'name': '1_wick',       'wicks': 1, 'close_pct': 0.90, 'buffer': 0.05},
            {'name': '2_wicks',      'wicks': 2, 'close_pct': 0.85, 'buffer': 0.05},
            {'name': '3_wicks',      'wicks': 3, 'close_pct': 0.80, 'buffer': 0.05},
            {'name': '5_wicks',      'wicks': 5, 'close_pct': 0.70, 'buffer': 0.10},
            {'name': '8_wicks',      'wicks': 8, 'close_pct': 0.60, 'buffer': 0.15},
            {'name': 'close_80pct',  'wicks': 99, 'close_pct': 0.80, 'buffer': 0.20},
            {'name': 'close_70pct',  'wicks': 99, 'close_pct': 0.70, 'buffer': 0.20},
        ]

        for direction in ['SHORT', 'LONG']:
            candidate_levels = short_levels if direction == 'SHORT' else long_levels

            for level_name, level_price in candidate_levels:
                # Check strict acceptance (current code)
                if direction == 'SHORT':
                    strict_pass = eor_high < level_price
                    miss_dist = eor_high - level_price  # positive = eor_high is above level (miss)
                else:
                    strict_pass = eor_low > level_price
                    miss_dist = level_price - eor_low  # positive = eor_low is below level (miss)

                miss_distances[direction].append({
                    'date': session_str,
                    'level_name': level_name,
                    'level_price': level_price,
                    'miss_dist': miss_dist,
                    'miss_pct': miss_dist / eor_range if eor_range > 0 else 0,
                    'eor_range': eor_range,
                })

                if strict_pass:
                    key = f'strict_{level_name}'
                    counters[key][direction] += 1

                # Test each window × config combination
                for win_name, win_bars in windows.items():
                    for cfg in configs:
                        passed, violations, close_pct, max_spike = check_acceptance_relaxed(
                            win_bars, level_price, direction, eor_range,
                            max_wicks=cfg['wicks'],
                            close_pct_thresh=cfg['close_pct'],
                            buffer_pct=cfg['buffer'],
                        )

                        key = f"{win_name}_{cfg['name']}_{level_name}"
                        if passed:
                            counters[key][direction] += 1

                        # Track the aggressive EOR config with all levels
                        if passed and win_name == 'EOR' and cfg['name'] == '5_wicks':
                            relaxed_details.append({
                                'date': session_str,
                                'direction': direction,
                                'level_name': level_name,
                                'level_price': level_price,
                                'eor_high': eor_high,
                                'eor_low': eor_low,
                                'eor_range': eor_range,
                                'violations': violations,
                                'close_pct': close_pct,
                                'max_spike': max_spike,
                                'atr14': atr14,
                            })

                        if passed and win_name == 'EOR' and cfg['name'] == '5_wicks':
                            level_counters[f"{win_name}_{cfg['name']}"][direction] += 1

    # ═══════════════════════════════════════════════════════════
    # PRINT RESULTS
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"  FUNNEL ANALYSIS ({n_sessions} sessions)")
    print(f"{'=' * 70}")

    # Summary table: strict acceptance per level
    print(f"\n--- Strict Acceptance (current code: eor_extreme fully past level) ---")
    print(f"{'Gate':<40} | {'SHORT':>6} | {'LONG':>6} | {'Total':>6}")
    print("-" * 65)
    print(f"{'Sessions with valid EOR':<40} | {counters['eor_valid']['SHORT']:>6} | {counters['eor_valid']['LONG']:>6} | {counters['eor_valid']['SHORT']:>6}")
    print(f"{'London levels exist':<40} | {counters['london_exists']['SHORT']:>6} | {counters['london_exists']['LONG']:>6} | {max(counters['london_exists']['SHORT'], counters['london_exists']['LONG']):>6}")
    for level_name in ['LDN_LOW', 'LDN_HIGH', 'ON_LOW', 'ON_HIGH', 'PDL', 'PDH', 'ASIA_LOW', 'ASIA_HIGH']:
        key = f'strict_{level_name}'
        s, l = counters[key]['SHORT'], counters[key]['LONG']
        if s + l > 0:
            print(f"{'  Strict: ' + level_name:<40} | {s:>6} | {l:>6} | {s+l:>6}")

    # Near-miss analysis
    print(f"\n--- Near Miss Analysis ---")
    print("How close does EOR extreme get to each level? (miss_dist > 0 = missed)")
    for direction in ['SHORT', 'LONG']:
        misses = miss_distances[direction]
        if not misses:
            continue
        miss_df = pd.DataFrame(misses)
        for level_name in miss_df['level_name'].unique():
            ldf = miss_df[miss_df['level_name'] == level_name]
            dist = ldf['miss_dist']
            pct = ldf['miss_pct']
            n_strict = (dist < 0).sum()  # negative dist = passed strict
            n_close = ((dist >= 0) & (dist < 20)).sum()
            n_medium = ((dist >= 20) & (dist < 50)).sum()
            print(f"  {direction} {level_name}: passed={n_strict}, <20pts miss={n_close}, "
                  f"<50pts miss={n_medium}, median miss={dist.median():.1f} pts ({pct.median()*100:.1f}% of EOR)")

    # Relaxed acceptance results
    print(f"\n--- Relaxed Acceptance by Config × Window × Level ---")
    print(f"{'Config':<45} | {'SHORT':>6} | {'LONG':>6} | {'Total':>6}")
    print("-" * 70)

    # Group by window × config, sum across all levels
    agg = defaultdict(lambda: {'SHORT': 0, 'LONG': 0})
    for key, counts in counters.items():
        if '_' not in key or key.startswith('strict_') or key in ('eor_valid', 'london_exists'):
            continue
        parts = key.split('_')
        # key format: WINDOW_CONFIG_LEVEL
        # We want to aggregate by WINDOW_CONFIG across all levels
        # But also show per-level for the best configs
        win = parts[0]
        # Find where level name starts (it's the last 1-2 parts)
        for lvl_name in ['LDN_LOW', 'LDN_HIGH', 'ON_LOW', 'ON_HIGH', 'PDL', 'PDH', 'ASIA_LOW', 'ASIA_HIGH']:
            if key.endswith(lvl_name):
                cfg_name = key[len(win)+1:-(len(lvl_name)+1)]
                agg_key = f"{win} / {cfg_name} / ALL"
                agg[agg_key]['SHORT'] += counts['SHORT']
                agg[agg_key]['LONG'] += counts['LONG']

                agg_key2 = f"{win} / {cfg_name} / {lvl_name}"
                agg[agg_key2]['SHORT'] += counts['SHORT']
                agg[agg_key2]['LONG'] += counts['LONG']
                break

    # Print aggregated (ALL levels) first
    for key in sorted(agg.keys()):
        if '/ ALL' in key:
            s, l = agg[key]['SHORT'], agg[key]['LONG']
            if s + l > 0:
                print(f"  {key:<43} | {s:>6} | {l:>6} | {s+l:>6}")

    # Print per-level detail for the best config
    print(f"\n--- Per-Level Detail (EOR / 5_wicks / each level) ---")
    for key in sorted(agg.keys()):
        if 'EOR / 5_wicks' in key and '/ ALL' not in key:
            s, l = agg[key]['SHORT'], agg[key]['LONG']
            if s + l > 0:
                print(f"  {key:<43} | {s:>6} | {l:>6} | {s+l:>6}")

    # Deduplicated session count (a session may match multiple levels)
    print(f"\n--- Unique Sessions with Signal (EOR / 5_wicks, any level) ---")
    if relaxed_details:
        rd_df = pd.DataFrame(relaxed_details)
        unique_short = rd_df[rd_df['direction'] == 'SHORT']['date'].nunique()
        unique_long = rd_df[rd_df['direction'] == 'LONG']['date'].nunique()
        unique_total = rd_df['date'].nunique()
        print(f"  SHORT sessions: {unique_short}")
        print(f"  LONG sessions:  {unique_long}")
        print(f"  Total unique:   {unique_total} ({unique_total/n_sessions*100:.1f}% of {n_sessions} sessions)")

        # Print detailed sessions
        print(f"\n--- Session Details (EOR / 5_wicks, all matching levels) ---")
        print(f"{'Date':<12} {'Dir':<6} {'Level':<10} {'LvlPrice':>10} {'EOR_H':>10} {'EOR_L':>10} "
              f"{'EOR_R':>7} {'Viols':>5} {'Cl%':>5} {'Spike':>7} {'ATR14':>7}")
        print("-" * 105)
        for _, row in rd_df.sort_values('date').iterrows():
            print(f"{row['date']:<12} {row['direction']:<6} {row['level_name']:<10} "
                  f"{row['level_price']:>10.2f} {row['eor_high']:>10.2f} {row['eor_low']:>10.2f} "
                  f"{row['eor_range']:>7.1f} {row['violations']:>5} {row['close_pct']:>5.1%} "
                  f"{row['max_spike']:>7.1f} {row['atr14']:>7.1f}")

        # Can we actually produce entries from these?
        print(f"\n--- Entry Feasibility Check ---")
        print("For each accepted session, check if price reaches 50% retest zone:")
        feasible_count = 0
        for _, row in rd_df.drop_duplicates('date').sort_values('date').iterrows():
            session_df = df[df['session_date'].astype(str).str[:10] == row['date']]
            if len(session_df) < IB_BARS:
                continue
            ib = session_df.head(IB_BARS)
            post_ib = session_df.iloc[IB_BARS:]
            if len(post_ib) == 0:
                continue

            atr14 = row['atr14']
            if row['direction'] == 'SHORT':
                ib_low = ib['low'].min()
                fifty_pct = ib_low + (row['level_price'] - ib_low) * 0.50
                entry_lo = fifty_pct - atr14 * 0.5
                entry_hi = fifty_pct + atr14 * 0.5
                reached = ((post_ib['high'] >= entry_lo) & (post_ib['low'] <= entry_hi)).any()
            else:
                ib_high = ib['high'].max()
                fifty_pct = ib_high - (ib_high - row['level_price']) * 0.50
                entry_lo = fifty_pct - atr14 * 0.5
                entry_hi = fifty_pct + atr14 * 0.5
                reached = ((post_ib['high'] >= entry_lo) & (post_ib['low'] <= entry_hi)).any()

            status = "RETEST" if reached else "no retest"
            if reached:
                feasible_count += 1
            print(f"  {row['date']} {row['direction']:<6} level={row['level_name']:<10} "
                  f"50%={fifty_pct:.2f} zone=[{entry_lo:.2f}, {entry_hi:.2f}] → {status}")

        print(f"\n  Feasible entries: {feasible_count} / {rd_df['date'].nunique()} unique sessions")
    else:
        print("  No sessions passed relaxed acceptance!")

    # Summary recommendation
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY & RECOMMENDATIONS")
    print(f"{'=' * 70}")

    best_count = 0
    best_config = ''
    for key in sorted(agg.keys()):
        if '/ ALL' in key:
            total = agg[key]['SHORT'] + agg[key]['LONG']
            if total > best_count:
                best_count = total
                best_config = key

    print(f"\n  Most permissive config: {best_config} = {best_count} signals")
    if relaxed_details:
        rd_df = pd.DataFrame(relaxed_details)
        print(f"  Unique sessions (EOR/5_wicks/all levels): {rd_df['date'].nunique()}")
        print(f"  Signal-to-session ratio: {rd_df['date'].nunique()}/{n_sessions} = {rd_df['date'].nunique()/n_sessions*100:.1f}%")

    print(f"\n  Next steps:")
    print(f"  1. Update strategy/or_acceptance.py with relaxed acceptance logic")
    print(f"  2. Run parameter sweep to find WR/PF-optimal configuration")
    print(f"  3. Backtest with: python scripts/run_backtest.py -s 'OR Acceptance' --max-drawdown 999999")


if __name__ == '__main__':
    run_funnel()
