"""
80% Rule (80P) — Comprehensive Exit Optimization Study
========================================================

Replay-based analysis of 80P setups under multiple configurations:

VA SOURCE:
  1. RTH-only 1-day VA (standard Dalton)
  2. Full ETH+RTH 1-day VA (overnight included)
  3. 3-day composite VA
  4. 5-day composite VA

ACCEPTANCE CRITERIA:
  - 1 consecutive 30-min close inside VA (faster entry)
  - 2 consecutive 30-min closes (standard Dalton)

EXIT MODES:
  A. Fixed target: opposite VA edge
  B. Fixed target: POC (conservative)
  C. Fixed target: half-VA (midpoint between entry and opposite edge)
  D. Multi-R: 1R, 1.5R, 2R, 3R, 4R
  E. BE at POC: move stop to breakeven when POC is reached, target remains opposite VA
  F. Aggressive trail: after POC breaks, trail by FVG or fixed offset

DIRECTION:
  - LONG only (strongest edge)
  - LONG + SHORT

For each config, outputs: trades, WR, avg R, PF, $/month
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.instruments import get_instrument
from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from indicators.value_area import (
    calculate_value_area, compute_session_value_areas,
    compute_composite_value_area, ValueAreaLevels,
)

# ============================================================================
# CONFIGURATION
# ============================================================================
INSTRUMENT = get_instrument('MNQ')
TICK_VALUE = INSTRUMENT.tick_value / INSTRUMENT.tick_size  # $/point
SLIPPAGE_PTS = 0.50  # 2 ticks slippage round-trip
COMMISSION_PER_CONTRACT = 1.24  # round-trip per MNQ contract
CONTRACTS = 5
MIN_VA_WIDTH = 25.0
STOP_BUFFER = 10.0
ENTRY_CUTOFF = time(13, 0)
PERIOD_BARS = 30  # 30-min TPO period

# Exit modes to test
EXIT_MODES = [
    'opposite_va',   # Full traverse to opposite VA edge
    'poc',           # Take profit at POC
    'half_va',       # Halfway to opposite edge
    '1R',            # 1x risk
    '1.5R',          # 1.5x risk
    '2R',            # 2x risk
    '3R',            # 3x risk
    '4R',            # 4x risk
    'be_at_poc',     # Target=opposite VA, but move stop to BE when POC reached
    'trail_after_poc',  # Target=opposite VA, trail aggressively after POC
]


@dataclass
class SetupResult:
    """Result of replaying one 80P setup."""
    session_date: str
    direction: str
    entry_price: float
    stop_price: float
    exit_price: float
    exit_reason: str  # TARGET, STOP, EOD, VWAP_BREACH
    pnl_pts: float
    pnl_dollars: float
    risk_pts: float
    va_width: float
    entry_bar_idx: int
    bars_held: int
    mfe_pts: float  # max favorable excursion
    mae_pts: float  # max adverse excursion


def compute_va_sets(df_rth: pd.DataFrame, df_full: pd.DataFrame) -> Dict:
    """Compute all VA variants for each session."""
    va_sets = {}

    # 1. RTH 1-day VA
    rth_va = compute_session_value_areas(df_rth, tick_size=0.25, va_percent=0.70)
    va_sets['rth_1d'] = rth_va

    # 2. Full (ETH+RTH) 1-day VA
    full_va = compute_session_value_areas(df_full, tick_size=0.25, va_percent=0.70)
    va_sets['eth_1d'] = full_va

    # 3. 3-day composite (RTH)
    sessions = sorted(df_rth['session_date'].unique())
    comp3 = {}
    for s in sessions:
        va = compute_composite_value_area(df_rth, s, lookback_days=3)
        if va:
            comp3[str(s)] = va
    va_sets['rth_3d'] = comp3

    # 4. 5-day composite (RTH)
    comp5 = {}
    for s in sessions:
        va = compute_composite_value_area(df_rth, s, lookback_days=5)
        if va:
            comp5[str(s)] = va
    va_sets['rth_5d'] = comp5

    return va_sets


def find_80p_setups(
    df_rth: pd.DataFrame,
    va_by_session: Dict[str, ValueAreaLevels],
    acceptance_periods: int = 2,
    directions: str = 'LONG_ONLY',  # or 'BOTH'
) -> List[Dict]:
    """
    Scan for 80P setups: open outside prior VA, price re-enters and is accepted.

    Returns list of setup dicts with entry info and the session bars for replay.
    """
    setups = []
    sessions = sorted(df_rth['session_date'].unique())

    for i in range(1, len(sessions)):
        current = sessions[i]
        prior = sessions[i - 1]
        prior_key = str(prior)

        if prior_key not in va_by_session:
            continue

        prior_va = va_by_session[prior_key]
        if prior_va.va_width < MIN_VA_WIDTH:
            continue

        session_df = df_rth[df_rth['session_date'] == current].reset_index(drop=True)
        if len(session_df) < 90:  # Need at least IB + some bars
            continue

        open_price = session_df['open'].iloc[0]
        vah = prior_va.vah
        val = prior_va.val
        poc = prior_va.poc

        # Determine direction
        if open_price < val:
            direction = 'LONG'
        elif open_price > vah:
            direction = 'SHORT'
        else:
            continue  # Inside VA, no setup

        if directions == 'LONG_ONLY' and direction == 'SHORT':
            continue

        # Post-IB bars (after first 60 1-min bars)
        if len(session_df) <= 60:
            continue
        post_ib = session_df.iloc[60:].reset_index(drop=True)

        # Check acceptance: N consecutive 30-min closes inside VA
        consecutive = 0
        entry_bar_idx = None

        for bar_idx in range(len(post_ib)):
            period_end = ((bar_idx + 1) % PERIOD_BARS == 0)
            if not period_end:
                continue

            close = post_ib.iloc[bar_idx]['close']
            bar_time = post_ib.iloc[bar_idx]['timestamp']
            if hasattr(bar_time, 'time'):
                bt = bar_time.time()
            else:
                bt = None

            if bt and bt >= ENTRY_CUTOFF:
                break

            is_inside = val <= close <= vah
            if is_inside:
                consecutive += 1
            else:
                consecutive = 0

            if consecutive >= acceptance_periods:
                entry_bar_idx = bar_idx
                break

        if entry_bar_idx is None:
            continue

        entry_price = post_ib.iloc[entry_bar_idx]['close']

        # Compute stop
        if direction == 'LONG':
            stop_price = val - STOP_BUFFER
        else:
            stop_price = vah + STOP_BUFFER

        risk_pts = abs(entry_price - stop_price)
        if risk_pts <= 0:
            continue

        setups.append({
            'session_date': str(current),
            'direction': direction,
            'entry_price': entry_price,
            'stop_price': stop_price,
            'risk_pts': risk_pts,
            'vah': vah,
            'val': val,
            'poc': poc,
            'va_width': prior_va.va_width,
            'entry_bar_idx': entry_bar_idx,
            'remaining_bars': post_ib.iloc[entry_bar_idx:],
        })

    return setups


def replay_setup(setup: Dict, exit_mode: str) -> Optional[SetupResult]:
    """
    Replay a setup forward from entry bar using a specific exit mode.
    Returns the result of the trade.
    """
    direction = setup['direction']
    entry_price = setup['entry_price']
    stop_price = setup['stop_price']
    risk_pts = setup['risk_pts']
    vah = setup['vah']
    val = setup['val']
    poc = setup['poc']
    bars = setup['remaining_bars']

    if risk_pts <= 0:
        return None

    # Compute target based on exit mode
    if direction == 'LONG':
        if exit_mode == 'opposite_va':
            target = vah
        elif exit_mode == 'poc':
            target = poc
        elif exit_mode == 'half_va':
            target = entry_price + (vah - entry_price) * 0.5
        elif exit_mode.endswith('R'):
            r_mult = float(exit_mode[:-1])
            target = entry_price + risk_pts * r_mult
        elif exit_mode in ('be_at_poc', 'trail_after_poc'):
            target = vah
        else:
            target = vah
    else:  # SHORT
        if exit_mode == 'opposite_va':
            target = val
        elif exit_mode == 'poc':
            target = poc
        elif exit_mode == 'half_va':
            target = entry_price - (entry_price - val) * 0.5
        elif exit_mode.endswith('R'):
            r_mult = float(exit_mode[:-1])
            target = entry_price - risk_pts * r_mult
        elif exit_mode in ('be_at_poc', 'trail_after_poc'):
            target = val
        else:
            target = val

    # Check R:R viability
    if direction == 'LONG':
        reward = target - entry_price
    else:
        reward = entry_price - target
    if reward <= 0:
        return None

    # Replay bars
    current_stop = stop_price
    mfe = 0.0
    mae = 0.0
    poc_reached = False
    exit_price = None
    exit_reason = None
    bars_held = 0

    for i in range(len(bars)):
        bar = bars.iloc[i]
        bars_held += 1
        bar_high = bar['high']
        bar_low = bar['low']
        bar_close = bar['close']

        # Track bar time for EOD
        bar_time = bar['timestamp'] if 'timestamp' in bar.index else None
        if bar_time and hasattr(bar_time, 'time'):
            bt = bar_time.time()
            if bt >= time(15, 30):
                exit_price = bar_close
                exit_reason = 'EOD'
                break

        # MFE/MAE tracking
        if direction == 'LONG':
            fav = bar_high - entry_price
            adv = entry_price - bar_low
        else:
            fav = entry_price - bar_low
            adv = bar_high - entry_price
        mfe = max(mfe, fav)
        mae = max(mae, adv)

        # BE at POC logic
        if exit_mode == 'be_at_poc' and not poc_reached:
            if direction == 'LONG' and bar_high >= poc:
                poc_reached = True
                current_stop = max(current_stop, entry_price)
            elif direction == 'SHORT' and bar_low <= poc:
                poc_reached = True
                current_stop = min(current_stop, entry_price)

        # Trail after POC logic
        if exit_mode == 'trail_after_poc' and not poc_reached:
            if direction == 'LONG' and bar_high >= poc:
                poc_reached = True
                current_stop = max(current_stop, entry_price)
            elif direction == 'SHORT' and bar_low <= poc:
                poc_reached = True
                current_stop = min(current_stop, entry_price)

        if exit_mode == 'trail_after_poc' and poc_reached:
            # Aggressive trail: trail by 15 pts from session high/low after POC
            trail_dist = 15.0
            if direction == 'LONG':
                new_stop = bar_high - trail_dist
                current_stop = max(current_stop, new_stop)
            else:
                new_stop = bar_low + trail_dist
                current_stop = min(current_stop, new_stop)

        # VWAP breach check (PM session)
        if bar_time and hasattr(bar_time, 'time') and bar_time.time() >= time(13, 0):
            vwap = bar.get('vwap', None) if hasattr(bar, 'get') else bar.get('vwap')
            if vwap is not None and not pd.isna(vwap):
                if direction == 'LONG' and bar_close < vwap - 10:
                    exit_price = bar_close
                    exit_reason = 'VWAP_BREACH'
                    break
                elif direction == 'SHORT' and bar_close > vwap + 10:
                    exit_price = bar_close
                    exit_reason = 'VWAP_BREACH'
                    break

        # Check stop
        if direction == 'LONG' and bar_low <= current_stop:
            exit_price = current_stop
            exit_reason = 'STOP'
            break
        elif direction == 'SHORT' and bar_high >= current_stop:
            exit_price = current_stop
            exit_reason = 'STOP'
            break

        # Check target
        if direction == 'LONG' and bar_high >= target:
            exit_price = target
            exit_reason = 'TARGET'
            break
        elif direction == 'SHORT' and bar_low <= target:
            exit_price = target
            exit_reason = 'TARGET'
            break

    # If we didn't exit, close at last bar
    if exit_price is None:
        exit_price = bars.iloc[-1]['close']
        exit_reason = 'EOD'

    # Compute P&L
    if direction == 'LONG':
        pnl_pts = exit_price - entry_price - SLIPPAGE_PTS
    else:
        pnl_pts = entry_price - exit_price - SLIPPAGE_PTS

    pnl_dollars = pnl_pts * TICK_VALUE * CONTRACTS - COMMISSION_PER_CONTRACT * CONTRACTS

    return SetupResult(
        session_date=setup['session_date'],
        direction=direction,
        entry_price=entry_price,
        stop_price=stop_price,
        exit_price=exit_price,
        exit_reason=exit_reason,
        pnl_pts=pnl_pts,
        pnl_dollars=pnl_dollars,
        risk_pts=risk_pts,
        va_width=setup['va_width'],
        entry_bar_idx=setup['entry_bar_idx'],
        bars_held=bars_held,
        mfe_pts=mfe,
        mae_pts=mae,
    )


def summarize_results(results: List[SetupResult], label: str, months: float) -> Dict:
    """Summarize a set of trade results."""
    if not results:
        return {'label': label, 'n': 0, 'wr': 0, 'pf': 0, 'per_month': 0,
                'avg_win': 0, 'avg_loss': 0, 'rr': 0, 'avg_mfe': 0, 'avg_mae': 0}

    wins = [r for r in results if r.pnl_dollars > 0]
    losses = [r for r in results if r.pnl_dollars <= 0]
    n = len(results)
    wr = len(wins) / n * 100
    total_pnl = sum(r.pnl_dollars for r in results)
    per_month = total_pnl / months if months > 0 else 0

    avg_win = np.mean([r.pnl_dollars for r in wins]) if wins else 0
    avg_loss = np.mean([r.pnl_dollars for r in losses]) if losses else 0

    gw = sum(r.pnl_dollars for r in wins)
    gl = abs(sum(r.pnl_dollars for r in losses))
    pf = gw / gl if gl > 0 else float('inf')
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    avg_mfe = np.mean([r.mfe_pts for r in results])
    avg_mae = np.mean([r.mae_pts for r in results])
    avg_risk = np.mean([r.risk_pts for r in results])
    avg_va_w = np.mean([r.va_width for r in results])

    return {
        'label': label, 'n': n, 'wr': wr, 'pf': pf,
        'per_month': per_month, 'total_pnl': total_pnl,
        'avg_win': avg_win, 'avg_loss': avg_loss, 'rr': rr,
        'avg_mfe': avg_mfe, 'avg_mae': avg_mae,
        'avg_risk': avg_risk, 'avg_va_w': avg_va_w,
        'trades_per_month': n / months if months > 0 else 0,
    }


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("Loading data...")
    instrument = get_instrument('MNQ')
    df_raw = load_csv('NQ')

    # Prepare both RTH and full datasets
    df_rth = filter_rth(df_raw)
    df_full = df_raw.copy()  # includes overnight

    # Add session_date to both
    if 'session_date' not in df_rth.columns:
        df_rth['session_date'] = df_rth['timestamp'].dt.date
    if 'session_date' not in df_full.columns:
        # For full data, assign session based on 18:00 cutoff
        df_full['session_date'] = df_full['timestamp'].dt.date
        # Bars between 18:00-23:59 belong to next trading day
        evening_mask = df_full['timestamp'].dt.time >= time(18, 0)
        df_full.loc[evening_mask, 'session_date'] = (
            pd.to_datetime(df_full.loc[evening_mask, 'session_date']) + pd.Timedelta(days=1)
        ).dt.date

    # Compute features on RTH (for VWAP etc.)
    df_rth = compute_all_features(df_rth)

    sessions = sorted(df_rth['session_date'].unique())
    n_sessions = len(sessions)
    months = n_sessions / 22

    print(f"\nSessions: {n_sessions}, Months: {months:.1f}")

    # ========================================================================
    # COMPUTE ALL VA VARIANTS
    # ========================================================================
    print("\nComputing Value Area variants...")
    va_sets = compute_va_sets(df_rth, df_full)

    for va_name, va_dict in va_sets.items():
        widths = [v.va_width for v in va_dict.values()]
        if widths:
            print(f"  {va_name:10s}: {len(va_dict)} sessions, "
                  f"mean width={np.mean(widths):.0f}, "
                  f"median={np.median(widths):.0f}, "
                  f"min={np.min(widths):.0f}, max={np.max(widths):.0f}")

    # ========================================================================
    # RUN ALL CONFIGURATIONS
    # ========================================================================
    all_summaries = []

    # Test matrix
    va_sources = ['rth_1d', 'eth_1d', 'rth_3d', 'rth_5d']
    acceptance_options = [1, 2]
    direction_options = ['LONG_ONLY', 'BOTH']

    print(f"\n{'='*120}")
    print(f"  RUNNING 80P OPTIMIZATION MATRIX")
    print(f"{'='*120}")

    for va_source in va_sources:
        va_dict = va_sets[va_source]
        for acc_periods in acceptance_options:
            for dir_opt in direction_options:
                # Find setups
                setups = find_80p_setups(
                    df_rth, va_dict,
                    acceptance_periods=acc_periods,
                    directions=dir_opt,
                )

                if not setups:
                    continue

                # Test each exit mode
                for exit_mode in EXIT_MODES:
                    results = []
                    for setup in setups:
                        result = replay_setup(setup, exit_mode)
                        if result:
                            results.append(result)

                    if results:
                        label = f"{va_source}|acc{acc_periods}|{dir_opt[:4]}|{exit_mode}"
                        summary = summarize_results(results, label, months)
                        all_summaries.append(summary)

    # ========================================================================
    # RESULTS TABLE
    # ========================================================================
    print(f"\n{'='*160}")
    print(f"  80P OPTIMIZATION RESULTS — ALL CONFIGURATIONS ({len(all_summaries)} tested)")
    print(f"{'='*160}")

    # Sort by $/month descending
    all_summaries.sort(key=lambda x: x['per_month'], reverse=True)

    print(f"\n  {'Config':<50s} {'N':>4s} {'Trd/Mo':>7s} {'WR':>6s} {'PF':>6s} "
          f"{'$/Mo':>9s} {'AvgWin':>8s} {'AvgLoss':>8s} {'R:R':>5s} "
          f"{'MFE':>6s} {'MAE':>6s} {'AvgRsk':>7s} {'VAW':>6s}")
    print(f"  {'-'*145}")

    for s in all_summaries:
        if s['n'] == 0:
            continue
        print(f"  {s['label']:<50s} {s['n']:>4d} {s['trades_per_month']:>6.1f} "
              f"{s['wr']:>5.1f}% {s['pf']:>5.2f} "
              f"${s['per_month']:>7,.0f} ${s['avg_win']:>6,.0f} ${s['avg_loss']:>6,.0f} "
              f"{s['rr']:>4.2f} "
              f"{s['avg_mfe']:>5.0f}p {s['avg_mae']:>5.0f}p "
              f"{s['avg_risk']:>5.0f}p {s['avg_va_w']:>5.0f}p")

    # ========================================================================
    # TOP 10 ANALYSIS
    # ========================================================================
    top10 = [s for s in all_summaries if s['n'] >= 3][:10]
    print(f"\n{'='*120}")
    print(f"  TOP 10 CONFIGS (min 3 trades)")
    print(f"{'='*120}")

    for rank, s in enumerate(top10, 1):
        print(f"\n  #{rank}: {s['label']}")
        print(f"      Trades: {s['n']}, {s['trades_per_month']:.1f}/mo")
        print(f"      WR: {s['wr']:.1f}%, PF: {s['pf']:.2f}, R:R: {s['rr']:.2f}")
        print(f"      $/month: ${s['per_month']:,.0f}")
        print(f"      Avg MFE: {s['avg_mfe']:.0f}pt, Avg MAE: {s['avg_mae']:.0f}pt")
        print(f"      Avg risk: {s['avg_risk']:.0f}pt, Avg VA width: {s['avg_va_w']:.0f}pt")

    # ========================================================================
    # MFE ANALYSIS — How far do winners run?
    # ========================================================================
    print(f"\n{'='*120}")
    print(f"  MFE ANALYSIS — Optimal Target Discovery")
    print(f"{'='*120}")

    # Use the best VA source from top results
    best_va = all_summaries[0]['label'].split('|')[0] if all_summaries else 'rth_1d'
    best_acc = int(all_summaries[0]['label'].split('|')[1].replace('acc', '')) if all_summaries else 2

    setups = find_80p_setups(
        df_rth, va_sets.get(best_va, va_sets['rth_1d']),
        acceptance_periods=best_acc,
        directions='BOTH',
    )

    if setups:
        # Replay all with opposite_va target to get MFE data
        results = []
        for setup in setups:
            result = replay_setup(setup, 'opposite_va')
            if result:
                results.append(result)

        if results:
            print(f"\n  Using {best_va}, acc={best_acc}, BOTH directions, {len(results)} trades:")
            mfes = [r.mfe_pts for r in results]
            maes = [r.mae_pts for r in results]
            risks = [r.risk_pts for r in results]
            va_widths = [r.va_width for r in results]

            print(f"\n  MFE (Max Favorable Excursion):")
            for pct in [25, 50, 75, 90]:
                print(f"    {pct}th percentile: {np.percentile(mfes, pct):.0f} pts")
            print(f"    Mean: {np.mean(mfes):.0f} pts")

            print(f"\n  MAE (Max Adverse Excursion):")
            for pct in [25, 50, 75, 90]:
                print(f"    {pct}th percentile: {np.percentile(maes, pct):.0f} pts")
            print(f"    Mean: {np.mean(maes):.0f} pts")

            print(f"\n  Risk (Entry to Stop):")
            print(f"    Mean: {np.mean(risks):.0f} pts")
            print(f"    Median: {np.median(risks):.0f} pts")

            print(f"\n  VA Width of triggered setups:")
            print(f"    Mean: {np.mean(va_widths):.0f} pts")
            print(f"    Median: {np.median(va_widths):.0f} pts")

            # MFE in R multiples
            r_mfes = [mfe / risk if risk > 0 else 0 for mfe, risk in zip(mfes, risks)]
            print(f"\n  MFE in R-multiples:")
            for pct in [25, 50, 75, 90]:
                print(f"    {pct}th percentile: {np.percentile(r_mfes, pct):.1f}R")
            print(f"    Mean: {np.mean(r_mfes):.1f}R")

    # ========================================================================
    # EXIT REASON BREAKDOWN
    # ========================================================================
    print(f"\n{'='*120}")
    print(f"  EXIT REASON BREAKDOWN (opposite_va target)")
    print(f"{'='*120}")

    for va_source in va_sources:
        setups = find_80p_setups(
            df_rth, va_sets[va_source],
            acceptance_periods=2, directions='BOTH',
        )
        if not setups:
            continue
        results = [replay_setup(s, 'opposite_va') for s in setups]
        results = [r for r in results if r is not None]
        if not results:
            continue

        print(f"\n  {va_source} (acc=2, BOTH):")
        reasons = {}
        for r in results:
            reasons[r.exit_reason] = reasons.get(r.exit_reason, 0) + 1
        for reason, count in sorted(reasons.items()):
            avg_pnl = np.mean([r.pnl_dollars for r in results if r.exit_reason == reason])
            wr = sum(1 for r in results if r.exit_reason == reason and r.pnl_dollars > 0) / count * 100
            print(f"    {reason:15s}: {count:3d} ({count/len(results)*100:5.1f}%), "
                  f"WR={wr:5.1f}%, avg P&L=${avg_pnl:,.0f}")

    # ========================================================================
    # INDIVIDUAL TRADE LOG — Best config
    # ========================================================================
    if all_summaries:
        best = all_summaries[0]
        parts = best['label'].split('|')
        va_src, acc, dir_s, exit_m = parts[0], int(parts[1].replace('acc', '')), parts[2], parts[3]
        dir_opt = 'LONG_ONLY' if dir_s == 'LONG' else 'BOTH'

        setups = find_80p_setups(df_rth, va_sets[va_src], acceptance_periods=acc, directions=dir_opt)
        results = [replay_setup(s, exit_m) for s in setups]
        results = [r for r in results if r is not None]

        print(f"\n{'='*120}")
        print(f"  TRADE LOG — Best Config: {best['label']}")
        print(f"{'='*120}")

        print(f"\n  {'Date':12s} {'Dir':6s} {'Entry':>10s} {'Stop':>10s} {'Exit':>10s} "
              f"{'P&L':>8s} {'Reason':12s} {'Risk':>6s} {'MFE':>6s} {'MAE':>6s} {'VA_W':>6s}")
        print(f"  {'-'*105}")

        for r in sorted(results, key=lambda x: x.session_date):
            print(f"  {r.session_date:12s} {r.direction:6s} "
                  f"{r.entry_price:>10.2f} {r.stop_price:>10.2f} {r.exit_price:>10.2f} "
                  f"${r.pnl_dollars:>7,.0f} {r.exit_reason:12s} "
                  f"{r.risk_pts:>5.0f}p {r.mfe_pts:>5.0f}p {r.mae_pts:>5.0f}p "
                  f"{r.va_width:>5.0f}p")

    # ========================================================================
    # USER SCENARIO CHECK
    # ========================================================================
    print(f"\n{'='*120}")
    print(f"  USER SCENARIO: 5 trades/month, 80% WR, 1:4 R:R")
    print(f"{'='*120}")

    # Find configs closest to user's target
    viable = [s for s in all_summaries if s['n'] >= 3 and s['wr'] >= 60]
    if viable:
        print(f"\n  Configs with WR >= 60% and 3+ trades:")
        for s in viable[:15]:
            print(f"    {s['label']:<50s} "
                  f"WR={s['wr']:5.1f}% R:R={s['rr']:.2f} "
                  f"trd/mo={s['trades_per_month']:.1f} "
                  f"$/mo=${s['per_month']:,.0f}")
    else:
        print("  No configs met WR >= 60% with 3+ trades.")
        if all_summaries:
            print("  Best available:")
            for s in all_summaries[:5]:
                print(f"    {s['label']:<50s} "
                      f"WR={s['wr']:5.1f}% trd/mo={s['trades_per_month']:.1f} "
                      f"$/mo=${s['per_month']:,.0f}")

    print(f"\n{'='*120}")
    print(f"  OPTIMIZATION COMPLETE")
    print(f"{'='*120}")
