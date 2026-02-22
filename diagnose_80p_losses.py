"""
80P Loss Diagnosis — Why is WR low when VA traversal is high?
==============================================================

The paradox: 78% of accepted 80P setups traverse the full VA, yet win rate
is only ~37% with 4R target and ~50% with opposite_va target. This script
diagnoses where the edge leaks:

1. STOP ANALYSIS: Are stops too tight relative to price action?
   - Risk (entry→stop) vs VA width ratio
   - MAE distribution: how deep do trades go before recovering?
   - How many trades hit stop BEFORE reaching target?

2. ENTRY TIMING: Are we entering too early or at the wrong price?
   - Entry location within VA (near VAL, mid, near VAH)
   - How many bars after IB does acceptance happen?
   - Price movement between acceptance and actual move

3. EXIT REASON FORENSICS: What kills the trades?
   - Stop vs EOD vs VWAP breach breakdown
   - For stopped trades: did price eventually reach target AFTER stop?
   - For EOD trades: was price heading toward target at close?

4. RISK MODEL: Is the stop placement fundamentally flawed?
   - Stop beyond VAL/VAH + 10pt buffer — is this enough?
   - What if we used ATR-based stops? Wider fixed buffer?
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import time
from typing import Dict, List, Optional, Tuple

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.instruments import get_instrument
from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from indicators.value_area import compute_session_value_areas, ValueAreaLevels

INSTRUMENT = get_instrument('MNQ')
TICK_VALUE = INSTRUMENT.tick_value / INSTRUMENT.tick_size
SLIPPAGE_PTS = 0.50
COMMISSION = 1.24
CONTRACTS = 5
MIN_VA_WIDTH = 25.0
STOP_BUFFER = 10.0
ENTRY_CUTOFF = time(13, 0)
PERIOD_BARS = 30

# ============================================================================
# DATA LOADING
# ============================================================================
print("Loading data...")
df_raw = load_csv('NQ')
df_rth = filter_rth(df_raw)
df_rth = compute_all_features(df_rth)

if 'session_date' not in df_rth.columns:
    df_rth['session_date'] = df_rth['timestamp'].dt.date

# Full data for ETH VA
df_full = df_raw.copy()
if 'session_date' not in df_full.columns:
    df_full['session_date'] = df_full['timestamp'].dt.date
    evening_mask = df_full['timestamp'].dt.time >= time(18, 0)
    df_full.loc[evening_mask, 'session_date'] = (
        pd.to_datetime(df_full.loc[evening_mask, 'session_date']) + pd.Timedelta(days=1)
    ).dt.date

sessions = sorted(df_rth['session_date'].unique())
n_sessions = len(sessions)
months = n_sessions / 22
print(f"Sessions: {n_sessions}, Months: {months:.1f}")

# Compute ETH VA (best performing VA source)
print("Computing ETH Value Areas...")
eth_va = compute_session_value_areas(df_full, tick_size=0.25, va_percent=0.70)
rth_va = compute_session_value_areas(df_rth, tick_size=0.25, va_percent=0.70)


# ============================================================================
# DETAILED TRADE REPLAY — captures everything
# ============================================================================
def detailed_replay(setup: Dict, exit_mode: str = 'opposite_va') -> Optional[Dict]:
    """
    Full forensic replay of an 80P trade.
    Captures stop hit timing, MFE timing, target proximity, and post-exit behavior.
    """
    direction = setup['direction']
    entry_price = setup['entry_price']
    vah = setup['vah']
    val = setup['val']
    poc = setup['poc']
    va_width = vah - val
    risk_pts = setup['risk_pts']
    stop_price = setup['stop_price']
    bars = setup['remaining_bars']
    session_bars = setup['session_bars']  # ALL session bars for post-exit analysis

    # Compute target
    if exit_mode == 'opposite_va':
        target = vah if direction == 'LONG' else val
    elif exit_mode.endswith('R'):
        r_mult = float(exit_mode[:-1])
        if direction == 'LONG':
            target = entry_price + risk_pts * r_mult
        else:
            target = entry_price - risk_pts * r_mult
    else:
        target = vah if direction == 'LONG' else val

    if direction == 'LONG':
        reward = target - entry_price
    else:
        reward = entry_price - target
    if reward <= 0:
        return None

    # Replay
    current_stop = stop_price
    mfe = 0.0
    mae = 0.0
    mfe_bar = 0
    mae_bar = 0
    exit_price = None
    exit_reason = None
    bars_held = 0
    exit_bar_idx = None

    # Track price path
    price_path = []

    for i in range(len(bars)):
        bar = bars.iloc[i]
        bars_held += 1
        bar_high = bar['high']
        bar_low = bar['low']
        bar_close = bar['close']

        bar_time = bar.get('timestamp')
        bt = bar_time.time() if bar_time and hasattr(bar_time, 'time') else None

        if bt and bt >= time(15, 30):
            exit_price = bar_close
            exit_reason = 'EOD'
            exit_bar_idx = i
            break

        # MFE / MAE
        if direction == 'LONG':
            fav = bar_high - entry_price
            adv = entry_price - bar_low
        else:
            fav = entry_price - bar_low
            adv = bar_high - entry_price

        if fav > mfe:
            mfe = fav
            mfe_bar = i
        if adv > mae:
            mae = adv
            mae_bar = i

        price_path.append({
            'bar': i, 'close': bar_close, 'high': bar_high, 'low': bar_low,
            'fav': fav, 'adv': adv, 'cum_mfe': mfe, 'cum_mae': mae,
        })

        # Stop check
        if direction == 'LONG' and bar_low <= current_stop:
            exit_price = current_stop
            exit_reason = 'STOP'
            exit_bar_idx = i
            break
        elif direction == 'SHORT' and bar_high >= current_stop:
            exit_price = current_stop
            exit_reason = 'STOP'
            exit_bar_idx = i
            break

        # Target check
        if direction == 'LONG' and bar_high >= target:
            exit_price = target
            exit_reason = 'TARGET'
            exit_bar_idx = i
            break
        elif direction == 'SHORT' and bar_low <= target:
            exit_price = target
            exit_reason = 'TARGET'
            exit_bar_idx = i
            break

    if exit_price is None:
        exit_price = bars.iloc[-1]['close']
        exit_reason = 'EOD'
        exit_bar_idx = len(bars) - 1

    # P&L
    if direction == 'LONG':
        pnl_pts = exit_price - entry_price - SLIPPAGE_PTS
    else:
        pnl_pts = entry_price - exit_price - SLIPPAGE_PTS
    pnl_dollars = pnl_pts * TICK_VALUE * CONTRACTS - COMMISSION * CONTRACTS

    # POST-EXIT analysis: did price eventually reach target after the stop?
    post_exit_reached_target = False
    post_exit_reached_opposite_va = False
    post_exit_mfe = 0.0

    if exit_bar_idx is not None and exit_bar_idx < len(bars) - 1:
        for j in range(exit_bar_idx + 1, len(bars)):
            bar = bars.iloc[j]
            if direction == 'LONG':
                post_fav = bar['high'] - entry_price
                if bar['high'] >= target:
                    post_exit_reached_target = True
                if bar['high'] >= vah:
                    post_exit_reached_opposite_va = True
            else:
                post_fav = entry_price - bar['low']
                if bar['low'] <= target:
                    post_exit_reached_target = True
                if bar['low'] <= val:
                    post_exit_reached_opposite_va = True
            post_exit_mfe = max(post_exit_mfe, post_fav)

    # Entry location within VA
    if direction == 'LONG':
        entry_pct_in_va = (entry_price - val) / va_width if va_width > 0 else 0.5
        distance_to_target = vah - entry_price
    else:
        entry_pct_in_va = (vah - entry_price) / va_width if va_width > 0 else 0.5
        distance_to_target = entry_price - val

    return {
        'session_date': setup['session_date'],
        'direction': direction,
        'entry_price': entry_price,
        'stop_price': stop_price,
        'target': target,
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'pnl_pts': pnl_pts,
        'pnl_dollars': pnl_dollars,
        'risk_pts': risk_pts,
        'va_width': va_width,
        'vah': vah,
        'val': val,
        'poc': poc,

        # Timing
        'entry_bar_idx': setup['entry_bar_idx'],
        'bars_held': bars_held,
        'mfe_bar': mfe_bar,
        'mae_bar': mae_bar,

        # MFE / MAE
        'mfe_pts': mfe,
        'mae_pts': mae,
        'mfe_r': mfe / risk_pts if risk_pts > 0 else 0,
        'mae_r': mae / risk_pts if risk_pts > 0 else 0,

        # Stop analysis
        'stop_hit_before_mfe': (exit_reason == 'STOP' and mae_bar <= mfe_bar),
        'risk_to_va_width_ratio': risk_pts / va_width if va_width > 0 else 0,
        'entry_pct_in_va': entry_pct_in_va,
        'distance_to_target': distance_to_target,
        'reward_pts': reward,
        'rr_ratio': reward / risk_pts if risk_pts > 0 else 0,

        # Post-exit
        'post_exit_reached_target': post_exit_reached_target,
        'post_exit_reached_opposite_va': post_exit_reached_opposite_va,
        'post_exit_mfe': post_exit_mfe,

        'is_winner': pnl_dollars > 0,
    }


def find_setups_with_full_session(
    df_rth: pd.DataFrame,
    va_by_session: Dict[str, ValueAreaLevels],
    acceptance_periods: int = 1,
    directions: str = 'BOTH',
) -> List[Dict]:
    """Find 80P setups, keeping full session bars for post-exit analysis."""
    setups = []
    sessions_list = sorted(df_rth['session_date'].unique())

    for i in range(1, len(sessions_list)):
        current = sessions_list[i]
        prior = sessions_list[i - 1]
        prior_key = str(prior)

        if prior_key not in va_by_session:
            continue

        prior_va = va_by_session[prior_key]
        if prior_va.va_width < MIN_VA_WIDTH:
            continue

        session_df = df_rth[df_rth['session_date'] == current].reset_index(drop=True)
        if len(session_df) < 90:
            continue

        open_price = session_df['open'].iloc[0]
        vah = prior_va.vah
        val = prior_va.val
        poc = prior_va.poc

        if open_price < val:
            direction = 'LONG'
        elif open_price > vah:
            direction = 'SHORT'
        else:
            continue

        if directions == 'LONG_ONLY' and direction == 'SHORT':
            continue

        post_ib = session_df.iloc[60:].reset_index(drop=True)
        if len(post_ib) < PERIOD_BARS:
            continue

        consecutive = 0
        entry_bar_idx = None

        for bar_idx in range(len(post_ib)):
            period_end = ((bar_idx + 1) % PERIOD_BARS == 0)
            if not period_end:
                continue

            close = post_ib.iloc[bar_idx]['close']
            bar_time = post_ib.iloc[bar_idx]['timestamp']
            bt = bar_time.time() if hasattr(bar_time, 'time') else None

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
            'session_bars': session_df,
        })

    return setups


# ============================================================================
# MAIN ANALYSIS
# ============================================================================
print("\nFinding 80P setups (ETH VA, acc=1, BOTH)...")
setups = find_setups_with_full_session(df_rth, eth_va, acceptance_periods=1, directions='BOTH')
print(f"Found {len(setups)} setups")

# Test with multiple exit modes
for exit_mode in ['opposite_va', '4R', '2R']:
    print(f"\n{'='*120}")
    print(f"  DIAGNOSIS: exit_mode = {exit_mode}")
    print(f"{'='*120}")

    trades = []
    for setup in setups:
        result = detailed_replay(setup, exit_mode)
        if result:
            trades.append(result)

    if not trades:
        print("  No trades.")
        continue

    df_t = pd.DataFrame(trades)
    winners = df_t[df_t['is_winner']]
    losers = df_t[~df_t['is_winner']]

    n = len(df_t)
    n_w = len(winners)
    n_l = len(losers)
    wr = n_w / n * 100

    print(f"\n  OVERVIEW: {n} trades, {n_w}W / {n_l}L, WR={wr:.1f}%")

    # ------------------------------------------------------------------
    # 1. EXIT REASON BREAKDOWN
    # ------------------------------------------------------------------
    print(f"\n  {'─'*80}")
    print(f"  1. EXIT REASON BREAKDOWN")
    print(f"  {'─'*80}")

    for reason in ['TARGET', 'STOP', 'EOD', 'VWAP_BREACH']:
        subset = df_t[df_t['exit_reason'] == reason]
        if len(subset) == 0:
            continue
        sub_wr = (subset['pnl_dollars'] > 0).mean() * 100
        avg_pnl = subset['pnl_dollars'].mean()
        avg_mfe = subset['mfe_pts'].mean()
        avg_mae = subset['mae_pts'].mean()
        print(f"    {reason:15s}: {len(subset):3d} ({len(subset)/n*100:5.1f}%)  "
              f"WR={sub_wr:5.1f}%  avg P&L=${avg_pnl:>7,.0f}  "
              f"MFE={avg_mfe:>5.0f}p  MAE={avg_mae:>5.0f}p")

    # ------------------------------------------------------------------
    # 2. STOP ANALYSIS — ARE STOPS TOO TIGHT?
    # ------------------------------------------------------------------
    print(f"\n  {'─'*80}")
    print(f"  2. STOP ANALYSIS")
    print(f"  {'─'*80}")

    stopped = df_t[df_t['exit_reason'] == 'STOP']
    if len(stopped) > 0:
        # How many stopped trades would have eventually reached target?
        n_would_have_won = stopped['post_exit_reached_target'].sum()
        n_would_have_reached_va = stopped['post_exit_reached_opposite_va'].sum()
        print(f"\n    Stopped trades: {len(stopped)}")
        print(f"    Of those, price LATER reached target: {n_would_have_won} ({n_would_have_won/len(stopped)*100:.0f}%)")
        print(f"    Of those, price LATER reached opposite VA: {n_would_have_reached_va} ({n_would_have_reached_va/len(stopped)*100:.0f}%)")

        print(f"\n    Stop loss depth analysis:")
        print(f"      Mean risk (entry→stop): {stopped['risk_pts'].mean():.0f} pts")
        print(f"      Mean VA width:          {stopped['va_width'].mean():.0f} pts")
        print(f"      Risk / VA width:        {stopped['risk_to_va_width_ratio'].mean():.2f}")
        print(f"      Mean MAE at stop:       {stopped['mae_pts'].mean():.0f} pts")
        print(f"      Mean MFE before stop:   {stopped['mfe_pts'].mean():.0f} pts")

        # Did they get any favorable move before stop?
        mfe_before_stop = stopped['mfe_pts']
        print(f"\n    MFE BEFORE getting stopped:")
        for pct in [25, 50, 75, 90]:
            print(f"      {pct}th pctile: {np.percentile(mfe_before_stop, pct):.0f} pts")
        zero_mfe = (mfe_before_stop < 5).sum()
        print(f"      Trades with <5pt MFE before stop: {zero_mfe} ({zero_mfe/len(stopped)*100:.0f}%)")

        # Time analysis: how quickly did stop get hit?
        print(f"\n    Bars held before stop:")
        print(f"      Mean: {stopped['bars_held'].mean():.0f} bars ({stopped['bars_held'].mean():.0f} min)")
        print(f"      Median: {stopped['bars_held'].median():.0f} bars")
        for pct in [25, 50, 75]:
            print(f"      {pct}th pctile: {np.percentile(stopped['bars_held'], pct):.0f} bars")

    # ------------------------------------------------------------------
    # 3. RISK MODEL: Entry→Stop vs VA width
    # ------------------------------------------------------------------
    print(f"\n  {'─'*80}")
    print(f"  3. RISK MODEL — Entry Location & Risk Sizing")
    print(f"  {'─'*80}")

    print(f"\n    Risk (entry→stop) statistics:")
    print(f"      Mean:   {df_t['risk_pts'].mean():.0f} pts")
    print(f"      Median: {df_t['risk_pts'].median():.0f} pts")
    print(f"      Min:    {df_t['risk_pts'].min():.0f} pts")
    print(f"      Max:    {df_t['risk_pts'].max():.0f} pts")

    print(f"\n    VA width statistics:")
    print(f"      Mean:   {df_t['va_width'].mean():.0f} pts")
    print(f"      Median: {df_t['va_width'].median():.0f} pts")

    print(f"\n    Risk / VA width ratio:")
    print(f"      Mean:   {df_t['risk_to_va_width_ratio'].mean():.2f}")
    print(f"      This means stop is ~{df_t['risk_to_va_width_ratio'].mean()*100:.0f}% of VA width away")

    # Entry location within VA
    print(f"\n    Entry location within VA (0=VAL, 1=VAH):")
    for label, subset in [('LONG trades', df_t[df_t['direction'] == 'LONG']),
                           ('SHORT trades', df_t[df_t['direction'] == 'SHORT'])]:
        if len(subset) == 0:
            continue
        pct_in = subset['entry_pct_in_va']
        print(f"      {label}: mean={pct_in.mean():.2f}, median={pct_in.median():.2f}")
        w = subset[subset['is_winner']]['entry_pct_in_va']
        l = subset[~subset['is_winner']]['entry_pct_in_va']
        if len(w) > 0:
            print(f"        Winners enter at: {w.mean():.2f}")
        if len(l) > 0:
            print(f"        Losers enter at:  {l.mean():.2f}")

    # Distance to target vs distance to stop
    print(f"\n    Reward/Risk ratio:")
    print(f"      Mean R:R:  {df_t['rr_ratio'].mean():.2f}")
    print(f"      Mean reward (pts): {df_t['reward_pts'].mean():.0f}")
    print(f"      Mean risk (pts):   {df_t['risk_pts'].mean():.0f}")

    # ------------------------------------------------------------------
    # 4. MFE/MAE ANALYSIS — What happens during the trade?
    # ------------------------------------------------------------------
    print(f"\n  {'─'*80}")
    print(f"  4. MFE / MAE ANALYSIS")
    print(f"  {'─'*80}")

    print(f"\n    ALL TRADES:")
    print(f"      MFE: mean={df_t['mfe_pts'].mean():.0f}p, median={np.median(df_t['mfe_pts']):.0f}p")
    print(f"      MAE: mean={df_t['mae_pts'].mean():.0f}p, median={np.median(df_t['mae_pts']):.0f}p")
    print(f"      MFE(R): mean={df_t['mfe_r'].mean():.1f}R, median={np.median(df_t['mfe_r']):.1f}R")
    print(f"      MAE(R): mean={df_t['mae_r'].mean():.1f}R, median={np.median(df_t['mae_r']):.1f}R")

    print(f"\n    WINNERS ({n_w}):")
    if n_w > 0:
        print(f"      MFE: mean={winners['mfe_pts'].mean():.0f}p")
        print(f"      MAE: mean={winners['mae_pts'].mean():.0f}p  (drawdown before recovery)")
        print(f"      Bars held: mean={winners['bars_held'].mean():.0f}")

    print(f"\n    LOSERS ({n_l}):")
    if n_l > 0:
        print(f"      MFE: mean={losers['mfe_pts'].mean():.0f}p  (favorable move before failing)")
        print(f"      MAE: mean={losers['mae_pts'].mean():.0f}p")
        print(f"      Bars held: mean={losers['bars_held'].mean():.0f}")

    # Fraction of trades where MFE >= various R levels
    print(f"\n    MFE DISTRIBUTION (what R-levels do trades reach?):")
    for r_level in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
        count = (df_t['mfe_r'] >= r_level).sum()
        print(f"      MFE >= {r_level:.1f}R: {count:3d}/{n} ({count/n*100:5.1f}%)")

    # ------------------------------------------------------------------
    # 5. THE KEY QUESTION: Stopped trades that would have won
    # ------------------------------------------------------------------
    print(f"\n  {'─'*80}")
    print(f"  5. STOPPED TRADES — Post-Exit Price Action")
    print(f"  {'─'*80}")

    if len(stopped) > 0:
        reached_tgt = stopped[stopped['post_exit_reached_target']]
        reached_va = stopped[stopped['post_exit_reached_opposite_va']]
        not_reached = stopped[~stopped['post_exit_reached_target']]

        print(f"\n    {len(stopped)} stopped trades:")
        print(f"      Price LATER reached exact target:  {len(reached_tgt)} ({len(reached_tgt)/len(stopped)*100:.0f}%)")
        print(f"      Price LATER reached opposite VA:   {len(reached_va)} ({len(reached_va)/len(stopped)*100:.0f}%)")
        print(f"      Price NEVER reached target:        {len(not_reached)} ({len(not_reached)/len(stopped)*100:.0f}%)")

        if len(reached_tgt) > 0:
            print(f"\n    RECOVERABLE LOSSES (stopped but price later hit target):")
            print(f"      These {len(reached_tgt)} trades WOULD have won with wider stops.")
            print(f"      Their avg risk: {reached_tgt['risk_pts'].mean():.0f} pts")
            print(f"      Their avg MAE:  {reached_tgt['mae_pts'].mean():.0f} pts")
            print(f"      Needed buffer:  {reached_tgt['mae_pts'].mean() - reached_tgt['risk_pts'].mean():.0f} pts MORE")
            extra_needed = reached_tgt['mae_pts'] - reached_tgt['risk_pts']
            for pct in [50, 75, 90]:
                print(f"        {pct}th pctile extra stop needed: {np.percentile(extra_needed, pct):.0f} pts")

    # ------------------------------------------------------------------
    # 6. WHAT IF WE USED WIDER STOPS?
    # ------------------------------------------------------------------
    print(f"\n  {'─'*80}")
    print(f"  6. WHAT-IF: WIDER STOPS")
    print(f"  {'─'*80}")

    for extra_buffer in [0, 10, 20, 30, 50, 75, 100]:
        total_buffer = STOP_BUFFER + extra_buffer
        n_win = 0
        n_total = 0
        total_pnl = 0.0

        for setup in setups:
            direction = setup['direction']
            entry_price = setup['entry_price']
            vah = setup['vah']
            val = setup['val']
            risk = abs(entry_price - setup['stop_price']) + extra_buffer

            if direction == 'LONG':
                wide_stop = val - total_buffer
                if exit_mode == 'opposite_va':
                    target = vah
                elif exit_mode.endswith('R'):
                    r_mult = float(exit_mode[:-1])
                    target = entry_price + risk * r_mult
                else:
                    target = vah
                reward = target - entry_price
            else:
                wide_stop = vah + total_buffer
                if exit_mode == 'opposite_va':
                    target = val
                elif exit_mode.endswith('R'):
                    r_mult = float(exit_mode[:-1])
                    target = entry_price - risk * r_mult
                else:
                    target = val
                reward = entry_price - target

            if reward <= 0:
                continue

            bars = setup['remaining_bars']
            hit_target = False
            hit_stop = False

            for j in range(len(bars)):
                bar = bars.iloc[j]
                bt_time = bar.get('timestamp')
                bt = bt_time.time() if bt_time and hasattr(bt_time, 'time') else None
                if bt and bt >= time(15, 30):
                    break

                if direction == 'LONG':
                    if bar['low'] <= wide_stop:
                        hit_stop = True
                        break
                    if bar['high'] >= target:
                        hit_target = True
                        break
                else:
                    if bar['high'] >= wide_stop:
                        hit_stop = True
                        break
                    if bar['low'] <= target:
                        hit_target = True
                        break

            n_total += 1
            if hit_target:
                pnl = reward - SLIPPAGE_PTS
                n_win += 1
            elif hit_stop:
                pnl = -risk - SLIPPAGE_PTS
            else:
                # EOD — use approximate
                pnl = -SLIPPAGE_PTS  # roughly flat

            total_pnl += pnl * TICK_VALUE * CONTRACTS - COMMISSION * CONTRACTS

        if n_total > 0:
            wr_w = n_win / n_total * 100
            monthly = total_pnl / months
            print(f"    Buffer={total_buffer:3.0f}pt: {n_total} trades, "
                  f"WR={wr_w:5.1f}%, $/mo=${monthly:>7,.0f}")

    # ------------------------------------------------------------------
    # 7. ENTRY DELAY ANALYSIS
    # ------------------------------------------------------------------
    print(f"\n  {'─'*80}")
    print(f"  7. ENTRY TIMING — How late do we enter?")
    print(f"  {'─'*80}")

    entry_minutes = df_t['entry_bar_idx'].values  # bars after IB start = minutes after IB
    print(f"\n    Bars after IB open until entry:")
    print(f"      Mean:   {entry_minutes.mean():.0f} bars")
    print(f"      Median: {np.median(entry_minutes):.0f} bars")
    for pct in [25, 50, 75, 90]:
        print(f"      {pct}th pctile: {np.percentile(entry_minutes, pct):.0f} bars")

    # Winners vs losers entry timing
    if n_w > 0 and n_l > 0:
        print(f"\n    Winners entry after IB: mean={winners['entry_bar_idx'].mean():.0f} bars")
        print(f"    Losers entry after IB:  mean={losers['entry_bar_idx'].mean():.0f} bars")

    # ------------------------------------------------------------------
    # 8. TRADE-BY-TRADE LOG
    # ------------------------------------------------------------------
    print(f"\n  {'─'*80}")
    print(f"  8. TRADE LOG (sorted by date)")
    print(f"  {'─'*80}")

    print(f"\n  {'Date':12s} {'Dir':5s} {'Entry':>8s} {'Stop':>8s} {'Target':>8s} "
          f"{'Exit':>8s} {'Reason':8s} {'P&L':>8s} {'Risk':>5s} {'MFE':>5s} {'MAE':>5s} "
          f"{'MFE_R':>5s} {'MAE_R':>5s} {'Post?':>5s}")
    print(f"  {'-'*120}")

    for _, t in df_t.sort_values('session_date').iterrows():
        post_marker = '*' if t['exit_reason'] == 'STOP' and t['post_exit_reached_target'] else ''
        print(f"  {t['session_date']:12s} {t['direction']:5s} "
              f"{t['entry_price']:>8.1f} {t['stop_price']:>8.1f} {t['target']:>8.1f} "
              f"{t['exit_price']:>8.1f} {t['exit_reason']:8s} "
              f"${t['pnl_dollars']:>7,.0f} {t['risk_pts']:>4.0f}p "
              f"{t['mfe_pts']:>4.0f}p {t['mae_pts']:>4.0f}p "
              f"{t['mfe_r']:>4.1f}R {t['mae_r']:>4.1f}R {post_marker:>5s}")

    print(f"\n  * = stopped out but price later reached target (recoverable loss)")


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print(f"\n{'='*120}")
print(f"  DIAGNOSIS SUMMARY")
print(f"{'='*120}")

# Run one more time with opposite_va to get the summary stats
trades_va = []
for setup in setups:
    result = detailed_replay(setup, 'opposite_va')
    if result:
        trades_va.append(result)

df_va = pd.DataFrame(trades_va)
stopped = df_va[df_va['exit_reason'] == 'STOP']
eod = df_va[df_va['exit_reason'] == 'EOD']
target_hit = df_va[df_va['exit_reason'] == 'TARGET']

print(f"""
  ENTRY MODEL ISSUES:
    - Entry is at 30-min close INSIDE VA (acceptance bar)
    - For LONG: entry location within VA: {df_va[df_va['direction']=='LONG']['entry_pct_in_va'].mean():.0%} up from VAL
    - For SHORT: entry location within VA: {df_va[df_va['direction']=='SHORT']['entry_pct_in_va'].mean():.0%} up from VAH
    - Mean entry delay after IB: {df_va['entry_bar_idx'].mean():.0f} bars ({df_va['entry_bar_idx'].mean():.0f} min)

  RISK MODEL ISSUES:
    - Stop = VA boundary + {STOP_BUFFER}pt buffer
    - Mean risk (entry→stop): {df_va['risk_pts'].mean():.0f} pts
    - Mean VA width: {df_va['va_width'].mean():.0f} pts
    - Risk is {df_va['risk_to_va_width_ratio'].mean():.0%} of VA width

  STOP-OUT ANALYSIS (opposite_va target):
    - {len(stopped)} of {len(df_va)} trades ({len(stopped)/len(df_va)*100:.0f}%) stopped out
    - Of stopped trades, {stopped['post_exit_reached_target'].sum()} ({stopped['post_exit_reached_target'].mean()*100:.0f}%) would have reached target with wider stop
    - Mean MAE of stopped trades: {stopped['mae_pts'].mean():.0f} pts
    - Mean risk of stopped trades: {stopped['risk_pts'].mean():.0f} pts
    - Stopped trades needed ~{(stopped['mae_pts'].mean() - stopped['risk_pts'].mean()):.0f} pts more room

  MFE REALITY:
    - Median trade MFE: {np.median(df_va['mfe_pts']):.0f} pts ({np.median(df_va['mfe_r']):.1f}R)
    - Trades reaching 1R: {(df_va['mfe_r'] >= 1.0).sum()}/{len(df_va)} ({(df_va['mfe_r'] >= 1.0).mean()*100:.0f}%)
    - Trades reaching 2R: {(df_va['mfe_r'] >= 2.0).sum()}/{len(df_va)} ({(df_va['mfe_r'] >= 2.0).mean()*100:.0f}%)

  EOD EXITS:
    - {len(eod)} trades ({len(eod)/len(df_va)*100:.0f}%) exit at end of day
    - These trades ran out of time, not direction — avg MFE: {eod['mfe_pts'].mean():.0f}p
""")

print(f"{'='*120}")
print(f"  DIAGNOSIS COMPLETE")
print(f"{'='*120}")
