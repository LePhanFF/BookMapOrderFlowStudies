"""
OR Reversal Deep Analysis - Last 90 Days (Dec 2025 - Feb 2026)

Runs the OR Rev strategy on the full dataset, captures detailed diagnostics
for every session, and analyzes winners vs losers for optimization opportunities.
"""

import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
from datetime import time, timedelta, date
import pandas as pd
import numpy as np
from collections import defaultdict

# Add project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.instruments import get_instrument, MNQ
from config.constants import (
    IB_BARS_1MIN, DEFAULT_ACCOUNT_SIZE, DEFAULT_MAX_RISK_PER_TRADE,
)
from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from engine.backtest import BacktestEngine
from engine.execution import ExecutionModel
from engine.position import PositionManager
from strategy.or_reversal import (
    OpeningRangeReversal, OR_BARS, EOR_BARS,
    SWEEP_THRESHOLD_RATIO, VWAP_ALIGNED_RATIO, OR_STOP_BUFFER,
    MIN_RISK_RATIO, MAX_RISK_RATIO, DRIVE_THRESHOLD,
    _find_closest_swept_level,
)
from strategy.signal import Signal
from filters.strategy_regime_filter import StrategyRegimeFilter, _classify_ib_regime

# ────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────
LOOKBACK_DAYS = 90
INSTRUMENT = 'MNQ'
CSV_INSTRUMENT = 'NQ'  # CSV stored as NQ even for MNQ


def run_backtest_or_rev_only():
    """Run the full backtest with OR Rev only, capture trades."""
    instrument = get_instrument(INSTRUMENT)
    full_df = load_csv(CSV_INSTRUMENT)
    df = filter_rth(full_df)
    df = compute_all_features(df)

    strategies = [OpeningRangeReversal()]
    filters = StrategyRegimeFilter()  # Only regime filter (no time filter for OR Rev)

    from filters.composite import CompositeFilter
    composite = CompositeFilter([filters])

    execution = ExecutionModel(instrument, slippage_ticks=1)
    position_mgr = PositionManager(account_size=DEFAULT_ACCOUNT_SIZE, max_drawdown=999999)

    engine = BacktestEngine(
        instrument=instrument,
        strategies=strategies,
        filters=composite,
        execution=execution,
        position_mgr=position_mgr,
        risk_per_trade=DEFAULT_MAX_RISK_PER_TRADE,
        max_contracts=30,
        full_df=full_df,
    )

    result = engine.run(df, verbose=False)
    return result, df, full_df


def compute_overnight_levels(full_df, session_str):
    """Replicate BacktestEngine._compute_overnight_levels for diagnostics."""
    sd = pd.Timestamp(session_str)
    prev_day = sd - timedelta(days=1)
    if prev_day.weekday() == 5:
        prev_day -= timedelta(days=1)
    elif prev_day.weekday() == 6:
        prev_day -= timedelta(days=2)

    ts = full_df['timestamp']
    levels = {}

    on_mask = (ts >= prev_day + timedelta(hours=18)) & (ts < sd + timedelta(hours=9, minutes=30))
    on_bars = full_df[on_mask]
    if len(on_bars) > 0:
        levels['overnight_high'] = on_bars['high'].max()
        levels['overnight_low'] = on_bars['low'].min()
    else:
        levels['overnight_high'] = None
        levels['overnight_low'] = None

    london_mask = (ts >= sd + timedelta(hours=2)) & (ts < sd + timedelta(hours=5))
    london = full_df[london_mask]
    if len(london) > 0:
        levels['london_high'] = london['high'].max()
        levels['london_low'] = london['low'].min()

    asia_mask = (ts >= prev_day + timedelta(hours=20)) & (ts < sd)
    asia = full_df[asia_mask]
    if len(asia) > 0:
        levels['asia_high'] = asia['high'].max()
        levels['asia_low'] = asia['low'].min()

    prior_rth_mask = (ts >= prev_day + timedelta(hours=9, minutes=30)) & (ts <= prev_day + timedelta(hours=16))
    prior_rth = full_df[prior_rth_mask]
    if len(prior_rth) > 0:
        levels['pdh'] = prior_rth['high'].max()
        levels['pdl'] = prior_rth['low'].min()

    return levels


def diagnose_session(session_df, full_df, session_str):
    """
    Run the OR Rev detection logic step-by-step on a single session,
    returning detailed diagnostics about why it fired or didn't fire.
    """
    diag = {
        'session_date': session_str,
        'fired': False,
        'skip_reason': None,
        'direction': None,
        'entry_price': None,
        'stop_price': None,
        'target_price': None,
        'risk_pts': None,
        'swept_level_name': None,
        'swept_level_price': None,
        'sweep_depth': None,
        'drive_type': None,
        'drive_pct': None,
        'delta_at_entry': None,
        'cvd_at_entry': None,
        'cvd_at_extreme': None,
        'cvd_divergence': None,
        'eor_range': None,
        'ib_range': None,
        'ib_regime': None,
        'or_high': None,
        'or_low': None,
        'or_mid': None,
        'eor_high': None,
        'eor_low': None,
        'entry_bar_idx': None,
        'entry_time': None,
        'vwap_at_entry': None,
        # Extra diagnostics for skip analysis
        'high_swept': False,
        'low_swept': False,
        'both_swept': False,
        'high_sweep_name': None,
        'low_sweep_name': None,
        'high_sweep_depth': None,
        'low_sweep_depth': None,
        'fade_blocked': False,
        'no_reversal': False,
        'no_vwap': False,
        'no_delta_cvd': False,
        'risk_invalid': False,
        'too_few_bars': False,
    }

    ib_bars = session_df.head(IB_BARS_1MIN)
    if len(ib_bars) < EOR_BARS:
        diag['skip_reason'] = 'TOO_FEW_BARS'
        diag['too_few_bars'] = True
        return diag

    # IB range
    ib_high = ib_bars['high'].max()
    ib_low = ib_bars['low'].min()
    ib_range = ib_high - ib_low
    diag['ib_range'] = ib_range
    diag['ib_regime'] = _classify_ib_regime(ib_range)

    # OR = first 15 bars
    or_bars = ib_bars.iloc[:OR_BARS]
    or_high = or_bars['high'].max()
    or_low = or_bars['low'].min()
    or_mid = (or_high + or_low) / 2
    diag['or_high'] = or_high
    diag['or_low'] = or_low
    diag['or_mid'] = or_mid

    # EOR = first 30 bars
    eor_bars = ib_bars.iloc[:EOR_BARS]
    eor_high = eor_bars['high'].max()
    eor_low = eor_bars['low'].min()
    eor_range = eor_high - eor_low
    diag['eor_high'] = eor_high
    diag['eor_low'] = eor_low
    diag['eor_range'] = eor_range

    if eor_range < ib_range * 0.05 if ib_range > 0 else eor_range < 10:
        diag['skip_reason'] = 'EOR_RANGE_TOO_SMALL'
        return diag

    sweep_threshold = eor_range * SWEEP_THRESHOLD_RATIO
    vwap_threshold = eor_range * VWAP_ALIGNED_RATIO
    max_risk = eor_range * MAX_RISK_RATIO

    # Opening drive
    first_5 = ib_bars.iloc[:5]
    open_price = first_5.iloc[0]['open']
    close_5th = first_5.iloc[4]['close']
    drive_range = first_5['high'].max() - first_5['low'].min()
    drive_pct = (close_5th - open_price) / drive_range if drive_range > 0 else 0
    diag['drive_pct'] = drive_pct

    if drive_pct > DRIVE_THRESHOLD:
        opening_drive = 'DRIVE_UP'
    elif drive_pct < -DRIVE_THRESHOLD:
        opening_drive = 'DRIVE_DOWN'
    else:
        opening_drive = 'ROTATION'
    diag['drive_type'] = opening_drive

    # Overnight levels
    levels = compute_overnight_levels(full_df, session_str)
    overnight_high = levels.get('overnight_high')
    overnight_low = levels.get('overnight_low')

    if overnight_high is None or overnight_low is None:
        diag['skip_reason'] = 'NO_OVERNIGHT_LEVELS'
        return diag

    pdh = levels.get('pdh')
    pdl = levels.get('pdl')
    asia_high = levels.get('asia_high')
    asia_low = levels.get('asia_low')
    london_high = levels.get('london_high')
    london_low = levels.get('london_low')

    # Build candidates
    high_candidates = [('ON_HIGH', overnight_high)]
    if pdh: high_candidates.append(('PDH', pdh))
    if asia_high: high_candidates.append(('ASIA_HIGH', asia_high))
    if london_high: high_candidates.append(('LDN_HIGH', london_high))

    low_candidates = [('ON_LOW', overnight_low)]
    if pdl: low_candidates.append(('PDL', pdl))
    if asia_low: low_candidates.append(('ASIA_LOW', asia_low))
    if london_low: low_candidates.append(('LDN_LOW', london_low))

    # Sweep detection
    swept_high_level, swept_high_name = _find_closest_swept_level(
        eor_high, high_candidates, sweep_threshold, eor_range)
    swept_low_level, swept_low_name = _find_closest_swept_level(
        eor_low, low_candidates, sweep_threshold, eor_range)

    diag['high_swept'] = swept_high_level is not None
    diag['low_swept'] = swept_low_level is not None
    diag['high_sweep_name'] = swept_high_name
    diag['low_sweep_name'] = swept_low_name

    if swept_high_level is not None:
        diag['high_sweep_depth'] = eor_high - swept_high_level
    if swept_low_level is not None:
        diag['low_sweep_depth'] = swept_low_level - eor_low

    if swept_high_level is None and swept_low_level is None:
        diag['skip_reason'] = 'NO_SWEEP'
        return diag

    # Dual-sweep resolution
    if swept_high_level is not None and swept_low_level is not None:
        diag['both_swept'] = True
        high_depth = eor_high - swept_high_level
        low_depth = swept_low_level - eor_low
        if high_depth >= low_depth:
            swept_low_level = None
            swept_low_name = None
        else:
            swept_high_level = None
            swept_high_name = None

    # CVD series
    deltas = ib_bars['delta'] if 'delta' in ib_bars.columns else ib_bars.get('vol_delta', pd.Series(dtype=float))
    if deltas is not None and len(deltas) > 0:
        deltas = deltas.fillna(0)
        cvd_series = deltas.cumsum()
    else:
        cvd_series = None

    high_bar_idx = eor_bars['high'].idxmax()
    low_bar_idx = eor_bars['low'].idxmin()

    # ── SHORT SETUP ──
    if swept_high_level is not None:
        diag['swept_level_name'] = swept_high_name
        diag['swept_level_price'] = swept_high_level
        diag['sweep_depth'] = eor_high - swept_high_level
        diag['direction'] = 'SHORT'

        if opening_drive == 'DRIVE_UP':
            diag['skip_reason'] = 'FADE_BLOCKED'
            diag['fade_blocked'] = True
            return diag

        cvd_at_extreme = cvd_series.loc[high_bar_idx] if cvd_series is not None else None
        diag['cvd_at_extreme'] = cvd_at_extreme

        post_high = ib_bars.loc[high_bar_idx:]
        found_reversal = False
        found_vwap = False
        found_delta_cvd = False
        found_valid_risk = False

        for j in range(1, min(30, len(post_high))):
            bar = post_high.iloc[j]
            price = bar['close']

            if price >= or_mid:
                continue
            found_reversal = True

            vwap = bar.get('vwap', np.nan)
            if pd.isna(vwap) or abs(price - vwap) > vwap_threshold:
                continue
            found_vwap = True

            delta = bar.get('delta', bar.get('vol_delta', 0))
            if pd.isna(delta):
                delta = 0
            cvd_at_entry = cvd_series.loc[post_high.index[j]] if cvd_series is not None else None
            cvd_declining = (cvd_at_entry is not None and cvd_at_extreme is not None
                             and cvd_at_entry < cvd_at_extreme)
            if delta >= 0 and not cvd_declining:
                continue
            found_delta_cvd = True

            stop = swept_high_level + eor_range * OR_STOP_BUFFER
            risk = stop - price
            if risk < eor_range * MIN_RISK_RATIO or risk > max_risk:
                continue
            found_valid_risk = True
            target = price - 2 * risk

            diag['fired'] = True
            diag['entry_price'] = price
            diag['stop_price'] = stop
            diag['target_price'] = target
            diag['risk_pts'] = risk
            diag['delta_at_entry'] = delta
            diag['cvd_at_entry'] = cvd_at_entry
            diag['cvd_divergence'] = cvd_declining
            diag['vwap_at_entry'] = vwap
            diag['entry_bar_idx'] = j
            bar_ts = bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp')
            diag['entry_time'] = str(bar_ts)
            return diag

        # Didn't fire - determine why
        if not found_reversal:
            diag['skip_reason'] = 'NO_REVERSAL_BELOW_OR_MID'
            diag['no_reversal'] = True
        elif not found_vwap:
            diag['skip_reason'] = 'NO_VWAP_ALIGNMENT'
            diag['no_vwap'] = True
        elif not found_delta_cvd:
            diag['skip_reason'] = 'NO_DELTA_CVD_CONFIRMATION'
            diag['no_delta_cvd'] = True
        elif not found_valid_risk:
            diag['skip_reason'] = 'RISK_INVALID'
            diag['risk_invalid'] = True
        else:
            diag['skip_reason'] = 'UNKNOWN_SHORT'
        return diag

    # ── LONG SETUP ──
    if swept_low_level is not None:
        diag['swept_level_name'] = swept_low_name
        diag['swept_level_price'] = swept_low_level
        diag['sweep_depth'] = swept_low_level - eor_low
        diag['direction'] = 'LONG'

        if opening_drive == 'DRIVE_DOWN':
            diag['skip_reason'] = 'FADE_BLOCKED'
            diag['fade_blocked'] = True
            return diag

        cvd_at_extreme = cvd_series.loc[low_bar_idx] if cvd_series is not None else None
        diag['cvd_at_extreme'] = cvd_at_extreme

        post_low = ib_bars.loc[low_bar_idx:]
        found_reversal = False
        found_vwap = False
        found_delta_cvd = False
        found_valid_risk = False

        for j in range(1, min(30, len(post_low))):
            bar = post_low.iloc[j]
            price = bar['close']

            if price <= or_mid:
                continue
            found_reversal = True

            vwap = bar.get('vwap', np.nan)
            if pd.isna(vwap) or abs(price - vwap) > vwap_threshold:
                continue
            found_vwap = True

            delta = bar.get('delta', bar.get('vol_delta', 0))
            if pd.isna(delta):
                delta = 0
            cvd_at_entry = cvd_series.loc[post_low.index[j]] if cvd_series is not None else None
            cvd_rising = (cvd_at_entry is not None and cvd_at_extreme is not None
                          and cvd_at_entry > cvd_at_extreme)
            if delta <= 0 and not cvd_rising:
                continue
            found_delta_cvd = True

            stop = swept_low_level - eor_range * OR_STOP_BUFFER
            risk = price - stop
            if risk < eor_range * MIN_RISK_RATIO or risk > max_risk:
                continue
            found_valid_risk = True
            target = price + 2 * risk

            diag['fired'] = True
            diag['entry_price'] = price
            diag['stop_price'] = stop
            diag['target_price'] = target
            diag['risk_pts'] = risk
            diag['delta_at_entry'] = delta
            diag['cvd_at_entry'] = cvd_at_entry
            diag['cvd_divergence'] = cvd_rising
            diag['vwap_at_entry'] = vwap
            diag['entry_bar_idx'] = j
            bar_ts = bar.get('timestamp', bar.name) if hasattr(bar, 'name') else bar.get('timestamp')
            diag['entry_time'] = str(bar_ts)
            return diag

        if not found_reversal:
            diag['skip_reason'] = 'NO_REVERSAL_ABOVE_OR_MID'
            diag['no_reversal'] = True
        elif not found_vwap:
            diag['skip_reason'] = 'NO_VWAP_ALIGNMENT'
            diag['no_vwap'] = True
        elif not found_delta_cvd:
            diag['skip_reason'] = 'NO_DELTA_CVD_CONFIRMATION'
            diag['no_delta_cvd'] = True
        elif not found_valid_risk:
            diag['skip_reason'] = 'RISK_INVALID'
            diag['risk_invalid'] = True
        else:
            diag['skip_reason'] = 'UNKNOWN_LONG'
        return diag

    diag['skip_reason'] = 'NO_VALID_SETUP'
    return diag


def main():
    print("=" * 80)
    print("  OR REVERSAL DEEP ANALYSIS - LAST 90 DAYS")
    print("=" * 80)

    # ── 1. Run backtest ──
    print("\n[1] Running OR Rev backtest on full dataset...")
    result, df, full_df = run_backtest_or_rev_only()
    trades = result.trades

    # Determine last 90 days cutoff
    all_sessions = sorted(df['session_date'].unique())
    last_date = all_sessions[-1]
    cutoff_date = pd.Timestamp(last_date) - pd.Timedelta(days=LOOKBACK_DAYS)
    cutoff_str = str(cutoff_date.date())
    recent_sessions = [s for s in all_sessions if str(s) >= cutoff_str]
    recent_session_strs = set(str(s) for s in recent_sessions)

    print(f"  Total trades: {len(trades)}")
    print(f"  Cutoff date: {cutoff_str}")
    print(f"  Recent sessions (last {LOOKBACK_DAYS} days): {len(recent_sessions)}")

    # Filter trades to last 90 days
    recent_trades = [t for t in trades if t.session_date in recent_session_strs]
    print(f"  Recent OR Rev trades: {len(recent_trades)}")

    # ── 2. Detailed trade diagnostics ──
    print("\n[2] Running per-session diagnostics...")
    diagnostics = []
    for session_date in recent_sessions:
        session_str = str(session_date)
        session_df = df[df['session_date'] == session_date].copy()
        if len(session_df) < IB_BARS_1MIN:
            continue
        diag = diagnose_session(session_df, full_df, session_str)
        diagnostics.append(diag)

    diag_df = pd.DataFrame(diagnostics)
    fired_df = diag_df[diag_df['fired'] == True].copy()
    skipped_df = diag_df[diag_df['fired'] == False].copy()

    # ── 3. Match diagnostics to actual trade results ──
    trade_map = {}
    for t in recent_trades:
        trade_map[t.session_date] = t

    # ── SECTION A: Trade Detail Table ──
    print("\n" + "=" * 80)
    print("  SECTION A: ALL OR REV TRADES (LAST 90 DAYS)")
    print("=" * 80)

    print(f"\n{'Date':>12} {'Dir':>5} {'Entry':>10} {'Stop':>10} {'Target':>10} "
          f"{'W/L':>4} {'P&L':>10} {'Exit':>6} {'Swept':>10} {'Drive':>10} "
          f"{'Delta':>7} {'CVD_E':>8} {'CVD_X':>8} {'EOR':>6} {'IB':>6} {'Risk':>6}")
    print("-" * 155)

    for d in sorted(fired_df.to_dict('records'), key=lambda x: x['session_date']):
        t = trade_map.get(d['session_date'])
        if t:
            wl = 'W' if t.net_pnl > 0 else 'L'
            pnl_str = f"${t.net_pnl:>8,.2f}"
            exit_reason = t.exit_reason[:6]
        else:
            wl = '?'
            pnl_str = '     N/A'
            exit_reason = 'FILT'  # Filtered by regime

        cvd_e = f"{d['cvd_at_entry']:.0f}" if d['cvd_at_entry'] is not None else 'N/A'
        cvd_x = f"{d['cvd_at_extreme']:.0f}" if d['cvd_at_extreme'] is not None else 'N/A'
        delta_str = f"{d['delta_at_entry']:.0f}" if d['delta_at_entry'] is not None else 'N/A'

        print(f"{d['session_date']:>12} {d['direction']:>5} {d['entry_price']:>10.2f} "
              f"{d['stop_price']:>10.2f} {d['target_price']:>10.2f} "
              f"{wl:>4} {pnl_str:>10} {exit_reason:>6} {str(d['swept_level_name']):>10} "
              f"{d['drive_type']:>10} {delta_str:>7} {cvd_e:>8} {cvd_x:>8} "
              f"{d['eor_range']:>6.1f} {d['ib_range']:>6.1f} {d['risk_pts']:>6.1f}")

    # ── SECTION B: Skipped Sessions ──
    print("\n" + "=" * 80)
    print("  SECTION B: SESSIONS WHERE OR REV DID NOT FIRE (LAST 90 DAYS)")
    print("=" * 80)

    skip_counts = skipped_df['skip_reason'].value_counts()
    print(f"\nSkip Reason Summary:")
    for reason, count in skip_counts.items():
        print(f"  {reason:>35}: {count:>3} sessions ({count/len(skipped_df)*100:.1f}%)")

    print(f"\n{'Date':>12} {'Skip Reason':>35} {'Dir':>6} {'Drive':>10} "
          f"{'Hi Swept':>10} {'Lo Swept':>10} {'IB':>6} {'EOR':>6} {'Regime':>7}")
    print("-" * 120)

    for _, d in skipped_df.sort_values('session_date').iterrows():
        hi_sw = d.get('high_sweep_name', '-') or '-'
        lo_sw = d.get('low_sweep_name', '-') or '-'
        dir_str = str(d.get('direction', '-') or '-')
        drive_str = str(d.get('drive_type', '-') or '-')
        ib_str = f"{d['ib_range']:.0f}" if pd.notna(d.get('ib_range')) else '-'
        eor_str = f"{d['eor_range']:.0f}" if pd.notna(d.get('eor_range')) else '-'
        regime_str = str(d.get('ib_regime', '-') or '-')

        print(f"{d['session_date']:>12} {d['skip_reason']:>35} {dir_str:>6} "
              f"{drive_str:>10} {hi_sw:>10} {lo_sw:>10} {ib_str:>6} {eor_str:>6} {regime_str:>7}")

    # ── SECTION C: Loser Analysis ──
    print("\n" + "=" * 80)
    print("  SECTION C: LOSER DEEP DIVE")
    print("=" * 80)

    # Get actual losers from trades
    losers = [t for t in recent_trades if t.net_pnl <= 0]
    winners = [t for t in recent_trades if t.net_pnl > 0]

    print(f"\nRecent trades: {len(recent_trades)} ({len(winners)}W / {len(losers)}L)")
    if recent_trades:
        print(f"Win rate: {len(winners)/len(recent_trades)*100:.1f}%")
        print(f"Total P&L: ${sum(t.net_pnl for t in recent_trades):,.2f}")

    if losers:
        print(f"\n--- Loser Details ---")
        for t in losers:
            d = next((x for x in diagnostics if x['session_date'] == t.session_date), None)
            if d is None:
                continue
            print(f"\n  {t.session_date} | {t.direction} | P&L: ${t.net_pnl:,.2f} | Exit: {t.exit_reason}")
            print(f"    Entry: {t.entry_price:.2f} | Stop: {t.stop_price:.2f} | Target: {t.target_price:.2f}")
            print(f"    Risk: {d['risk_pts']:.1f} pts | EOR range: {d['eor_range']:.1f} | IB range: {d['ib_range']:.1f}")
            print(f"    Swept: {d['swept_level_name']} at {d['swept_level_price']:.2f} | Depth: {d['sweep_depth']:.1f} pts")
            print(f"    Drive: {d['drive_type']} (pct: {d['drive_pct']:.3f})")
            print(f"    Delta: {d['delta_at_entry']} | CVD entry: {d['cvd_at_entry']} | CVD extreme: {d['cvd_at_extreme']}")
            print(f"    CVD divergence: {d['cvd_divergence']}")

            # Analyze why it lost
            reasons = []
            if d['risk_pts'] and d['eor_range']:
                risk_pct = d['risk_pts'] / d['eor_range']
                if risk_pct < 0.15:
                    reasons.append(f"TIGHT_STOP (risk={risk_pct:.0%} of EOR)")
            if d['sweep_depth'] is not None and d['eor_range']:
                depth_pct = d['sweep_depth'] / d['eor_range']
                if depth_pct < 0.05:
                    reasons.append(f"SHALLOW_SWEEP ({depth_pct:.1%} of EOR, {d['sweep_depth']:.1f} pts)")
            if d['delta_at_entry'] is not None:
                if d['direction'] == 'SHORT' and d['delta_at_entry'] > -50:
                    reasons.append(f"WEAK_DELTA ({d['delta_at_entry']:.0f})")
                if d['direction'] == 'LONG' and d['delta_at_entry'] < 50:
                    reasons.append(f"WEAK_DELTA ({d['delta_at_entry']:.0f})")
            if d['cvd_divergence'] is True and d['delta_at_entry'] is not None:
                if d['direction'] == 'SHORT' and d['delta_at_entry'] >= 0:
                    reasons.append("CVD_ONLY_NO_DELTA (weaker signal)")
                if d['direction'] == 'LONG' and d['delta_at_entry'] <= 0:
                    reasons.append("CVD_ONLY_NO_DELTA (weaker signal)")
            if t.exit_reason == 'EOD' or t.exit_reason == 'VWAP_BREACH_PM':
                reasons.append(f"LATE_EXIT ({t.exit_reason})")

            if reasons:
                print(f"    LOSS FACTORS: {' | '.join(reasons)}")
            else:
                print(f"    LOSS FACTORS: None identified (legitimate stop-out)")

    # ── SECTION D: Winner vs Loser Pattern Analysis ──
    print("\n" + "=" * 80)
    print("  SECTION D: WINNER vs LOSER PATTERN ANALYSIS")
    print("=" * 80)

    # Build analysis dataframe merging diagnostics with trade results
    analysis_rows = []
    for t in recent_trades:
        d = next((x for x in diagnostics if x['session_date'] == t.session_date), None)
        if d is None:
            continue
        row = dict(d)
        row['net_pnl'] = t.net_pnl
        row['is_winner'] = t.net_pnl > 0
        row['exit_reason'] = t.exit_reason
        row['r_multiple'] = t.r_multiple
        row['bars_held'] = t.bars_held
        row['contracts'] = t.contracts
        # Day of week
        try:
            row['day_of_week'] = pd.Timestamp(t.session_date).day_name()
            row['dow_num'] = pd.Timestamp(t.session_date).dayofweek
        except:
            row['day_of_week'] = 'Unknown'
            row['dow_num'] = -1
        analysis_rows.append(row)

    if not analysis_rows:
        print("\nNo trades to analyze in the last 90 days.")
        return

    adf = pd.DataFrame(analysis_rows)
    win_df = adf[adf['is_winner'] == True]
    lose_df = adf[adf['is_winner'] == False]

    # ── D1: IB Range Distribution ──
    print("\n--- D1: IB Range Distribution ---")
    for label, subset in [('WINNERS', win_df), ('LOSERS', lose_df), ('ALL', adf)]:
        if len(subset) == 0:
            continue
        ib_vals = subset['ib_range'].dropna()
        print(f"  {label:>8}: n={len(subset):>3} | IB mean={ib_vals.mean():>6.1f} "
              f"med={ib_vals.median():>6.1f} min={ib_vals.min():>6.1f} max={ib_vals.max():>6.1f}")

    # IB regime breakdown
    print("\n  IB Regime Breakdown:")
    for regime in ['low', 'med', 'normal', 'high']:
        r_trades = adf[adf['ib_regime'] == regime]
        if len(r_trades) == 0:
            continue
        r_wins = r_trades[r_trades['is_winner'] == True]
        r_pnl = r_trades['net_pnl'].sum()
        wr = len(r_wins) / len(r_trades) * 100 if len(r_trades) > 0 else 0
        print(f"    {regime:>7}: {len(r_trades):>3} trades, {wr:>5.1f}% WR, ${r_pnl:>8,.2f} net")

    # ── D2: Drive Strength ──
    print("\n--- D2: Drive Strength (drive_pct) ---")
    for label, subset in [('WINNERS', win_df), ('LOSERS', lose_df)]:
        if len(subset) == 0:
            continue
        dp = subset['drive_pct'].dropna()
        print(f"  {label:>8}: mean={dp.mean():>6.3f} med={dp.median():>6.3f} "
              f"min={dp.min():>6.3f} max={dp.max():>6.3f}")

    # Drive type breakdown
    print("\n  Drive Type Breakdown:")
    for drive in ['DRIVE_UP', 'DRIVE_DOWN', 'ROTATION']:
        d_trades = adf[adf['drive_type'] == drive]
        if len(d_trades) == 0:
            continue
        d_wins = d_trades[d_trades['is_winner'] == True]
        d_pnl = d_trades['net_pnl'].sum()
        wr = len(d_wins) / len(d_trades) * 100 if len(d_trades) > 0 else 0
        print(f"    {drive:>12}: {len(d_trades):>3} trades, {wr:>5.1f}% WR, ${d_pnl:>8,.2f} net")

    # ── D3: Sweep Depth ──
    print("\n--- D3: Sweep Depth ---")
    for label, subset in [('WINNERS', win_df), ('LOSERS', lose_df)]:
        if len(subset) == 0:
            continue
        sd = subset['sweep_depth'].dropna()
        if len(sd) == 0:
            continue
        print(f"  {label:>8}: mean={sd.mean():>6.1f} med={sd.median():>6.1f} "
              f"min={sd.min():>6.1f} max={sd.max():>6.1f}")

    # Sweep depth as % of EOR range
    print("\n  Sweep Depth as % of EOR Range:")
    adf['sweep_depth_pct'] = adf['sweep_depth'] / adf['eor_range']
    for label, subset in [('WINNERS', adf[adf['is_winner']]), ('LOSERS', adf[~adf['is_winner']])]:
        if len(subset) == 0:
            continue
        sp = subset['sweep_depth_pct'].dropna()
        if len(sp) == 0:
            continue
        print(f"  {label:>8}: mean={sp.mean():>6.1%} med={sp.median():>6.1%} "
              f"min={sp.min():>6.1%} max={sp.max():>6.1%}")

    # ── D4: Swept Level Name ──
    print("\n--- D4: Swept Level Name ---")
    for lvl in adf['swept_level_name'].dropna().unique():
        l_trades = adf[adf['swept_level_name'] == lvl]
        l_wins = l_trades[l_trades['is_winner'] == True]
        l_pnl = l_trades['net_pnl'].sum()
        wr = len(l_wins) / len(l_trades) * 100 if len(l_trades) > 0 else 0
        print(f"  {lvl:>12}: {len(l_trades):>3} trades, {wr:>5.1f}% WR, ${l_pnl:>8,.2f} net")

    # ── D5: Delta Magnitude ──
    print("\n--- D5: Delta Magnitude at Entry ---")
    for label, subset in [('WINNERS', win_df), ('LOSERS', lose_df)]:
        if len(subset) == 0:
            continue
        da = subset['delta_at_entry'].dropna()
        if len(da) == 0:
            continue
        print(f"  {label:>8}: mean={da.mean():>8.1f} med={da.median():>8.1f} "
              f"min={da.min():>8.1f} max={da.max():>8.1f}")

    # Delta magnitude (absolute)
    adf['abs_delta'] = adf['delta_at_entry'].abs()
    print("\n  Abs Delta Buckets:")
    for lo, hi, label in [(0, 100, '<100'), (100, 300, '100-300'),
                          (300, 600, '300-600'), (600, 99999, '600+')]:
        bucket = adf[(adf['abs_delta'] >= lo) & (adf['abs_delta'] < hi)]
        if len(bucket) == 0:
            continue
        b_wins = bucket[bucket['is_winner'] == True]
        b_pnl = bucket['net_pnl'].sum()
        wr = len(b_wins) / len(bucket) * 100 if len(bucket) > 0 else 0
        print(f"    {label:>8}: {len(bucket):>3} trades, {wr:>5.1f}% WR, ${b_pnl:>8,.2f} net")

    # ── D6: CVD Divergence Strength ──
    print("\n--- D6: CVD Divergence ---")
    adf['cvd_delta'] = adf.apply(
        lambda r: (r['cvd_at_entry'] - r['cvd_at_extreme'])
        if r['cvd_at_entry'] is not None and r['cvd_at_extreme'] is not None else np.nan,
        axis=1
    )
    for label, subset in [('WINNERS', adf[adf['is_winner']]), ('LOSERS', adf[~adf['is_winner']])]:
        if len(subset) == 0:
            continue
        cd = subset['cvd_delta'].dropna()
        if len(cd) == 0:
            continue
        print(f"  {label:>8}: CVD change mean={cd.mean():>8.1f} med={cd.median():>8.1f}")

    # ── D7: Entry Time ──
    print("\n--- D7: Entry Bar Index (earlier = better?) ---")
    for label, subset in [('WINNERS', win_df), ('LOSERS', lose_df)]:
        if len(subset) == 0:
            continue
        bi = subset['entry_bar_idx'].dropna()
        if len(bi) == 0:
            continue
        print(f"  {label:>8}: mean={bi.mean():>5.1f} med={bi.median():>5.1f} "
              f"min={bi.min():>5.0f} max={bi.max():>5.0f}")

    # ── D8: Day of Week ──
    print("\n--- D8: Day of Week ---")
    for dow in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        d_trades = adf[adf['day_of_week'] == dow]
        if len(d_trades) == 0:
            continue
        d_wins = d_trades[d_trades['is_winner'] == True]
        d_pnl = d_trades['net_pnl'].sum()
        wr = len(d_wins) / len(d_trades) * 100 if len(d_trades) > 0 else 0
        print(f"  {dow:>12}: {len(d_trades):>3} trades, {wr:>5.1f}% WR, ${d_pnl:>8,.2f} net")

    # ── D9: Direction (LONG vs SHORT) ──
    print("\n--- D9: Direction Breakdown ---")
    for direction in ['SHORT', 'LONG']:
        d_trades = adf[adf['direction'] == direction]
        if len(d_trades) == 0:
            continue
        d_wins = d_trades[d_trades['is_winner'] == True]
        d_pnl = d_trades['net_pnl'].sum()
        wr = len(d_wins) / len(d_trades) * 100 if len(d_trades) > 0 else 0
        avg_r = d_trades['r_multiple'].mean()
        print(f"  {direction:>6}: {len(d_trades):>3} trades, {wr:>5.1f}% WR, "
              f"${d_pnl:>8,.2f} net, avg R={avg_r:>5.2f}")

        # Sub-breakdown: direction + drive
        for drive in ['DRIVE_UP', 'DRIVE_DOWN', 'ROTATION']:
            dd = d_trades[d_trades['drive_type'] == drive]
            if len(dd) == 0:
                continue
            dd_wins = dd[dd['is_winner'] == True]
            dd_pnl = dd['net_pnl'].sum()
            wr2 = len(dd_wins) / len(dd) * 100 if len(dd) > 0 else 0
            print(f"    + {drive:>12}: {len(dd):>3} trades, {wr2:>5.1f}% WR, ${dd_pnl:>8,.2f}")

    # ── D10: Risk in Points ──
    print("\n--- D10: Risk in Points ---")
    for label, subset in [('WINNERS', win_df), ('LOSERS', lose_df)]:
        if len(subset) == 0:
            continue
        rp = subset['risk_pts'].dropna()
        if len(rp) == 0:
            continue
        print(f"  {label:>8}: mean={rp.mean():>6.1f} med={rp.median():>6.1f} "
              f"min={rp.min():>6.1f} max={rp.max():>6.1f}")

    # ── D11: EOR Range ──
    print("\n--- D11: EOR Range ---")
    for label, subset in [('WINNERS', win_df), ('LOSERS', lose_df)]:
        if len(subset) == 0:
            continue
        er = subset['eor_range'].dropna()
        print(f"  {label:>8}: mean={er.mean():>6.1f} med={er.median():>6.1f} "
              f"min={er.min():>6.1f} max={er.max():>6.1f}")

    # ── D12: Exit Reason Breakdown ──
    print("\n--- D12: Exit Reason ---")
    for reason in adf['exit_reason'].unique():
        r_trades = adf[adf['exit_reason'] == reason]
        r_wins = r_trades[r_trades['is_winner'] == True]
        r_pnl = r_trades['net_pnl'].sum()
        wr = len(r_wins) / len(r_trades) * 100 if len(r_trades) > 0 else 0
        print(f"  {reason:>15}: {len(r_trades):>3} trades, {wr:>5.1f}% WR, ${r_pnl:>8,.2f} net")

    # ── D13: Monthly Breakdown ──
    print("\n--- D13: Monthly Breakdown ---")
    adf['month'] = adf['session_date'].apply(lambda x: x[:7])
    for month in sorted(adf['month'].unique()):
        m_trades = adf[adf['month'] == month]
        m_wins = m_trades[m_trades['is_winner'] == True]
        m_pnl = m_trades['net_pnl'].sum()
        wr = len(m_wins) / len(m_trades) * 100 if len(m_trades) > 0 else 0
        print(f"  {month}: {len(m_trades):>3} trades, {wr:>5.1f}% WR, ${m_pnl:>8,.2f} net")

    # ── D14: Bars Held ──
    print("\n--- D14: Bars Held ---")
    for label, subset in [('WINNERS', win_df), ('LOSERS', lose_df)]:
        if len(subset) == 0:
            continue
        bh = subset['bars_held']
        print(f"  {label:>8}: mean={bh.mean():>6.1f} med={bh.median():>6.1f} "
              f"min={bh.min():>5.0f} max={bh.max():>5.0f}")

    # Recompute win/lose from adf which now has sweep_depth_pct, abs_delta, cvd_delta
    win_df = adf[adf['is_winner'] == True]
    lose_df = adf[adf['is_winner'] == False]

    # ── SECTION E: FILTER SUGGESTIONS ──
    print("\n" + "=" * 80)
    print("  SECTION E: ACTIONABLE FILTER SUGGESTIONS")
    print("=" * 80)

    suggestions = []

    # E1: Sweep depth filter
    if len(lose_df) > 0 and len(win_df) > 0:
        w_sd = win_df['sweep_depth_pct'].dropna()
        l_sd = lose_df['sweep_depth_pct'].dropna()
        if len(w_sd) > 0 and len(l_sd) > 0:
            if l_sd.median() < w_sd.median():
                threshold = l_sd.quantile(0.75)
                # How many losers would this filter catch?
                caught = (l_sd < threshold).sum()
                saved_wins = (w_sd < threshold).sum()
                suggestions.append(
                    f"SWEEP_DEPTH: Losers have shallower sweeps (med {l_sd.median():.1%}) vs "
                    f"winners ({w_sd.median():.1%}). "
                    f"Filter sweep_depth_pct < {threshold:.1%} would remove {caught}/{len(l_sd)} losers "
                    f"but also {saved_wins}/{len(w_sd)} winners."
                )

    # E2: Delta magnitude filter
    if len(lose_df) > 0 and len(win_df) > 0:
        w_ad = win_df['abs_delta'].dropna() if 'abs_delta' in win_df.columns else pd.Series()
        l_ad = lose_df['abs_delta'].dropna() if 'abs_delta' in lose_df.columns else pd.Series()
        if len(w_ad) > 0 and len(l_ad) > 0:
            if l_ad.median() < w_ad.median():
                threshold = l_ad.quantile(0.5)
                caught = (l_ad < threshold).sum()
                saved_wins = (w_ad < threshold).sum()
                suggestions.append(
                    f"DELTA_MAGNITUDE: Losers have weaker |delta| (med {l_ad.median():.0f}) vs "
                    f"winners ({w_ad.median():.0f}). "
                    f"Filter |delta| < {threshold:.0f} would remove {caught}/{len(l_ad)} losers "
                    f"and {saved_wins}/{len(w_ad)} winners."
                )

    # E3: Entry bar timing
    if len(lose_df) > 0 and len(win_df) > 0:
        w_bi = win_df['entry_bar_idx'].dropna()
        l_bi = lose_df['entry_bar_idx'].dropna()
        if len(w_bi) > 0 and len(l_bi) > 0:
            if l_bi.median() > w_bi.median():
                suggestions.append(
                    f"ENTRY_TIMING: Losers enter later (med bar {l_bi.median():.0f}) vs "
                    f"winners (med bar {w_bi.median():.0f}). "
                    f"Consider max entry bar threshold."
                )

    # E4: Drive strength interaction
    if len(adf) > 0:
        rotation_trades = adf[adf['drive_type'] == 'ROTATION']
        non_rotation = adf[adf['drive_type'] != 'ROTATION']
        if len(rotation_trades) > 2 and len(non_rotation) > 2:
            rot_wr = rotation_trades['is_winner'].mean() * 100
            non_rot_wr = non_rotation['is_winner'].mean() * 100
            suggestions.append(
                f"DRIVE_TYPE: ROTATION={rot_wr:.0f}% WR vs directional drives={non_rot_wr:.0f}% WR. "
                f"{'ROTATION favored.' if rot_wr > non_rot_wr else 'Directional drives favored.'}"
            )

    # E5: LONG vs SHORT filter
    if len(adf) > 0:
        longs = adf[adf['direction'] == 'LONG']
        shorts = adf[adf['direction'] == 'SHORT']
        if len(longs) > 2 and len(shorts) > 2:
            long_wr = longs['is_winner'].mean() * 100
            short_wr = shorts['is_winner'].mean() * 100
            long_pnl = longs['net_pnl'].sum()
            short_pnl = shorts['net_pnl'].sum()
            suggestions.append(
                f"DIRECTION: SHORT={short_wr:.0f}% WR (${short_pnl:,.0f}) vs "
                f"LONG={long_wr:.0f}% WR (${long_pnl:,.0f}). "
                f"{'Consider disabling LONG.' if long_wr < 50 and short_wr > 60 else ''}"
            )

    # E6: EOR range filter
    if len(lose_df) > 0 and len(win_df) > 0:
        w_er = win_df['eor_range'].dropna()
        l_er = lose_df['eor_range'].dropna()
        if len(w_er) > 0 and len(l_er) > 0:
            suggestions.append(
                f"EOR_RANGE: Winners EOR med={w_er.median():.0f} vs Losers EOR med={l_er.median():.0f}. "
                f"{'Small EOR = tighter risk, more stop-outs.' if l_er.median() < w_er.median() else 'EOR range similar between W/L.'}"
            )

    # E7: CVD divergence strength
    if 'cvd_delta' in adf.columns:
        w_cd = adf[adf['is_winner']]['cvd_delta'].dropna()
        l_cd = adf[~adf['is_winner']]['cvd_delta'].dropna()
        if len(w_cd) > 0 and len(l_cd) > 0:
            suggestions.append(
                f"CVD_STRENGTH: Winners CVD shift med={w_cd.median():.0f} vs "
                f"Losers CVD shift med={l_cd.median():.0f}. "
                f"{'Stronger CVD divergence = better signal.' if abs(w_cd.median()) > abs(l_cd.median()) else ''}"
            )

    # E8: Risk points distribution
    if len(lose_df) > 0 and len(win_df) > 0:
        w_rp = win_df['risk_pts'].dropna()
        l_rp = lose_df['risk_pts'].dropna()
        if len(w_rp) > 0 and len(l_rp) > 0:
            suggestions.append(
                f"RISK_PTS: Winners risk med={w_rp.median():.1f} pts vs "
                f"Losers risk med={l_rp.median():.1f} pts. "
                f"{'Wide risk = more room for price to work.' if w_rp.median() > l_rp.median() else ''}"
            )

    # E9: "Both swept" (dual sweep) analysis
    both_swept_trades = adf[adf['both_swept'] == True]
    single_swept_trades = adf[adf['both_swept'] == False]
    if len(both_swept_trades) > 0:
        bs_wr = both_swept_trades['is_winner'].mean() * 100
        ss_wr = single_swept_trades['is_winner'].mean() * 100 if len(single_swept_trades) > 0 else 0
        suggestions.append(
            f"DUAL_SWEEP: Both-sides-swept sessions: {len(both_swept_trades)} trades, "
            f"{bs_wr:.0f}% WR vs single-sweep {ss_wr:.0f}% WR. "
            f"{'Dual sweeps = choppy session, consider skipping.' if bs_wr < ss_wr - 10 else ''}"
        )

    # Print suggestions
    for i, s in enumerate(suggestions, 1):
        print(f"\n  [{i}] {s}")

    # ── SECTION F: HYPOTHETICAL FILTER IMPACT ──
    print("\n" + "=" * 80)
    print("  SECTION F: HYPOTHETICAL FILTER BACKTESTS")
    print("=" * 80)

    if len(adf) > 0:
        baseline_trades = len(adf)
        baseline_pnl = adf['net_pnl'].sum()
        baseline_wr = adf['is_winner'].mean() * 100

        print(f"\n  BASELINE: {baseline_trades} trades, {baseline_wr:.1f}% WR, ${baseline_pnl:,.2f}")

        # Test various filters
        filters_to_test = []

        # Min sweep depth %
        for thresh in [0.02, 0.03, 0.05, 0.08]:
            mask = adf['sweep_depth_pct'] >= thresh
            if mask.sum() > 0 and mask.sum() < len(adf):
                filtered = adf[mask]
                f_wr = filtered['is_winner'].mean() * 100
                f_pnl = filtered['net_pnl'].sum()
                filters_to_test.append((
                    f"sweep_depth >= {thresh:.0%} EOR",
                    len(filtered), f_wr, f_pnl
                ))

        # Min abs delta
        if 'abs_delta' in adf.columns:
            for thresh in [50, 100, 200, 300]:
                mask = adf['abs_delta'] >= thresh
                if mask.sum() > 0 and mask.sum() < len(adf):
                    filtered = adf[mask]
                    f_wr = filtered['is_winner'].mean() * 100
                    f_pnl = filtered['net_pnl'].sum()
                    filters_to_test.append((
                        f"|delta| >= {thresh}",
                        len(filtered), f_wr, f_pnl
                    ))

        # Max entry bar
        for thresh in [5, 10, 15, 20]:
            mask = adf['entry_bar_idx'] <= thresh
            if mask.sum() > 0 and mask.sum() < len(adf):
                filtered = adf[mask]
                f_wr = filtered['is_winner'].mean() * 100
                f_pnl = filtered['net_pnl'].sum()
                filters_to_test.append((
                    f"entry_bar <= {thresh}",
                    len(filtered), f_wr, f_pnl
                ))

        # SHORT only
        mask = adf['direction'] == 'SHORT'
        if mask.sum() > 0:
            filtered = adf[mask]
            f_wr = filtered['is_winner'].mean() * 100
            f_pnl = filtered['net_pnl'].sum()
            filters_to_test.append((
                "SHORT only",
                len(filtered), f_wr, f_pnl
            ))

        # ROTATION only
        mask = adf['drive_type'] == 'ROTATION'
        if mask.sum() > 0:
            filtered = adf[mask]
            f_wr = filtered['is_winner'].mean() * 100
            f_pnl = filtered['net_pnl'].sum()
            filters_to_test.append((
                "ROTATION drive only",
                len(filtered), f_wr, f_pnl
            ))

        # Non-rotation only (continuation trades)
        mask = adf['drive_type'] != 'ROTATION'
        if mask.sum() > 0:
            filtered = adf[mask]
            f_wr = filtered['is_winner'].mean() * 100
            f_pnl = filtered['net_pnl'].sum()
            filters_to_test.append((
                "Directional drive only",
                len(filtered), f_wr, f_pnl
            ))

        # Min IB range
        for thresh in [100, 120, 150]:
            mask = adf['ib_range'] >= thresh
            if mask.sum() > 0 and mask.sum() < len(adf):
                filtered = adf[mask]
                f_wr = filtered['is_winner'].mean() * 100
                f_pnl = filtered['net_pnl'].sum()
                filters_to_test.append((
                    f"IB >= {thresh}",
                    len(filtered), f_wr, f_pnl
                ))

        # Min EOR range
        for thresh in [50, 75, 100]:
            mask = adf['eor_range'] >= thresh
            if mask.sum() > 0 and mask.sum() < len(adf):
                filtered = adf[mask]
                f_wr = filtered['is_winner'].mean() * 100
                f_pnl = filtered['net_pnl'].sum()
                filters_to_test.append((
                    f"EOR >= {thresh}",
                    len(filtered), f_wr, f_pnl
                ))

        # Combined: SHORT + sweep_depth >= 3%
        mask = (adf['direction'] == 'SHORT') & (adf['sweep_depth_pct'] >= 0.03)
        if mask.sum() > 0:
            filtered = adf[mask]
            f_wr = filtered['is_winner'].mean() * 100
            f_pnl = filtered['net_pnl'].sum()
            filters_to_test.append((
                "SHORT + sweep >= 3%",
                len(filtered), f_wr, f_pnl
            ))

        # Combined: SHORT + |delta| >= 100
        if 'abs_delta' in adf.columns:
            mask = (adf['direction'] == 'SHORT') & (adf['abs_delta'] >= 100)
            if mask.sum() > 0:
                filtered = adf[mask]
                f_wr = filtered['is_winner'].mean() * 100
                f_pnl = filtered['net_pnl'].sum()
                filters_to_test.append((
                    "SHORT + |delta| >= 100",
                    len(filtered), f_wr, f_pnl
                ))

        print(f"\n  {'Filter':>35} {'Trades':>7} {'WR%':>6} {'Net P&L':>12} {'vs Base':>10}")
        print("  " + "-" * 75)
        for name, count, wr, pnl in filters_to_test:
            delta_pnl = pnl - baseline_pnl
            print(f"  {name:>35} {count:>7} {wr:>5.1f}% ${pnl:>10,.2f} "
                  f"{'+'if delta_pnl>=0 else ''}{delta_pnl:>8,.2f}")

    print("\n" + "=" * 80)
    print("  ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
