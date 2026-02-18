"""
IBH Fade Strategy Test — Balance/Neutral/P-Day Structures

Tests the concept: SHORT at IBH rejection on non-trend days,
targeting VAH → POC → VWAP → VAL (tiered value area exits).

On true balance days:
  - VAH ≈ upper IB zone (roughly IBH - 0.15x IB range)
  - POC ≈ IB midpoint / VWAP
  - VAL ≈ lower IB zone (roughly IBL + 0.15x IB range)

Tests multiple variations:
  A) IBH fade SHORT → target VWAP (what existing B-Day tested)
  B) IBH fade SHORT → tiered targets (VAH/POC/VWAP/VAL with partials)
  C) IBH fade SHORT with strict rejection criteria + order flow
  D) IBH fade SHORT → IB mid only (most conservative target)
  E) IBH fade on any non-trend day (b_day + neutral + p_day)
  F) IBH fade with prior day context (only when prior day was trend/extension)
  G) IBH fade LONG only (flip: price REJECTS at IBH from below, fails to break)
"""

import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.instruments import get_instrument
from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from config.constants import IB_BARS_1MIN
from strategy.day_type import classify_trend_strength, classify_day_type
from engine.execution import ExecutionModel

# Load data once
instrument = get_instrument('MNQ')
df = load_csv('NQ')
df = filter_rth(df)
df = compute_all_features(df)

# Setup
if 'session_date' not in df.columns:
    df['session_date'] = df['timestamp'].dt.date

sessions = sorted(df['session_date'].unique())
execution = ExecutionModel(instrument, slippage_ticks=1)

MAX_CONTRACTS = 5
RISK_PER_TRADE = 400

print("=" * 130)
print("  IBH FADE STRATEGY TEST — BALANCE/NEUTRAL/P-DAY STRUCTURES")
print(f"  Data: {len(sessions)} sessions, Max {MAX_CONTRACTS} MNQ")
print("=" * 130)


# ============================================================================
# PHASE 1: RAW DATA ANALYSIS — What happens at IBH on balance days?
# ============================================================================
print("\n\n" + "=" * 130)
print("  PHASE 1: RAW ANALYSIS — Price behavior at IBH on balance days")
print("=" * 130)

ibh_rejections = []  # price touches IBH and retreats
ibh_extensions = []  # price breaks through IBH

for session_date in sessions:
    sdf = df[df['session_date'] == session_date].copy()
    if len(sdf) < IB_BARS_1MIN + 10:
        continue

    ib_df = sdf.head(IB_BARS_1MIN)
    post_ib = sdf.iloc[IB_BARS_1MIN:]

    ib_high = ib_df['high'].max()
    ib_low = ib_df['low'].min()
    ib_range = ib_high - ib_low
    ib_mid = (ib_high + ib_low) / 2

    if ib_range <= 0:
        continue

    # Track day development
    session_high = ib_high
    session_low = ib_low
    ibh_touched = False
    ibh_touch_bar = None
    ibh_touch_price = None
    first_extension = None

    for i, (idx, bar) in enumerate(post_ib.iterrows()):
        if bar['high'] > session_high:
            session_high = bar['high']
        if bar['low'] < session_low:
            session_low = bar['low']

        price = bar['close']

        # Classify day type dynamically
        if price > ib_high:
            ib_dir = 'BULL'
            ext = (price - ib_mid) / ib_range
        elif price < ib_low:
            ib_dir = 'BEAR'
            ext = (ib_mid - price) / ib_range
        else:
            ib_dir = 'INSIDE'
            ext = 0.0

        strength = classify_trend_strength(ext)
        day_type = classify_day_type(ib_high, ib_low, price, ib_dir, strength)

        # Track IBH touch (price gets within 0.1x IB of IBH)
        if not ibh_touched and bar['high'] >= ib_high - (ib_range * 0.10):
            ibh_touched = True
            ibh_touch_bar = i
            ibh_touch_price = bar['high']

        # Track first extension beyond IBH
        if first_extension is None and price > ib_high:
            first_extension = i

    if ibh_touched:
        # Outcome: what happened after touching IBH?
        last_bar = post_ib.iloc[-1]
        session_close = last_bar['close']

        # Did price ultimately extend beyond IBH significantly?
        max_extension_above_ibh = (session_high - ib_high) / ib_range if ib_range > 0 else 0
        final_day_type = day_type.value if hasattr(day_type, 'value') else str(day_type)

        # Price relative to value area at close
        close_vs_vwap = session_close - last_bar.get('vwap', ib_mid)

        record = {
            'date': str(session_date),
            'ib_range': ib_range,
            'ibh': ib_high,
            'ibl': ib_low,
            'ib_mid': ib_mid,
            'touch_bar': ibh_touch_bar,
            'max_ext_above_ibh': max_extension_above_ibh,
            'session_close': session_close,
            'final_day_type': final_day_type,
            'close_vs_ibh': session_close - ib_high,
            'close_vs_ib_mid': session_close - ib_mid,
            'extended': max_extension_above_ibh > 0.5,  # Extended significantly
        }

        if max_extension_above_ibh > 0.5:
            ibh_extensions.append(record)
        else:
            ibh_rejections.append(record)

total_ibh = len(ibh_rejections) + len(ibh_extensions)
print(f"\nSessions touching IBH: {total_ibh} / {len(sessions)}")
print(f"  IBH Rejections (ext < 0.5x IB): {len(ibh_rejections)} ({len(ibh_rejections)/total_ibh*100:.0f}%)")
print(f"  IBH Extensions (ext >= 0.5x IB): {len(ibh_extensions)} ({len(ibh_extensions)/total_ibh*100:.0f}%)")

# Where did rejected sessions close?
if ibh_rejections:
    closes_vs_mid = [r['close_vs_ib_mid'] for r in ibh_rejections]
    closes_vs_ibh = [r['close_vs_ibh'] for r in ibh_rejections]
    below_ibh = sum(1 for r in ibh_rejections if r['session_close'] < r['ibh'])
    below_mid = sum(1 for r in ibh_rejections if r['session_close'] < r['ib_mid'])

    print(f"\n  Rejected sessions close analysis:")
    print(f"    Closed below IBH:   {below_ibh}/{len(ibh_rejections)} ({below_ibh/len(ibh_rejections)*100:.0f}%)")
    print(f"    Closed below IB mid:{below_mid}/{len(ibh_rejections)} ({below_mid/len(ibh_rejections)*100:.0f}%)")
    print(f"    Avg close vs IBH:   {np.mean(closes_vs_ibh):+.1f} pts")
    print(f"    Avg close vs IB mid:{np.mean(closes_vs_mid):+.1f} pts")

    # By final day type
    dt_counts = defaultdict(int)
    for r in ibh_rejections:
        dt_counts[r['final_day_type']] += 1
    print(f"    Day types: {dict(dt_counts)}")


# ============================================================================
# PHASE 2: SIMULATE IBH FADE TRADES
# ============================================================================
print("\n\n" + "=" * 130)
print("  PHASE 2: SIMULATED IBH FADE TRADES")
print("=" * 130)

# We'll simulate multiple configurations of the IBH fade strategy
# by iterating bar-by-bar through each session

def simulate_ibh_fade(config_name, allowed_day_types, target_mode,
                       require_rejection_bar=True, require_delta_neg=True,
                       require_volume=False, max_extension=0.3):
    """
    Simulate IBH fade SHORT strategy.

    target_mode: 'ib_mid', 'vwap', 'val', 'tiered'
    """
    trades = []

    for session_date in sessions:
        sdf = df[df['session_date'] == session_date].copy()
        if len(sdf) < IB_BARS_1MIN + 10:
            continue

        ib_df = sdf.head(IB_BARS_1MIN)
        post_ib = sdf.iloc[IB_BARS_1MIN:]

        ib_high = ib_df['high'].max()
        ib_low = ib_df['low'].min()
        ib_range = ib_high - ib_low
        ib_mid = (ib_high + ib_low) / 2

        if ib_range <= 0 or ib_range > 400:
            continue

        # Approximate value area levels for balance days
        vah = ib_high - (ib_range * 0.15)  # ~85th percentile
        poc = ib_mid                         # POC ≈ IB midpoint
        val = ib_low + (ib_range * 0.15)    # ~15th percentile

        entry_taken = False
        entry_price = None
        stop_price = None
        target_price = None
        entry_bar_idx = None
        entry_time = None

        session_high = ib_high

        for i, (idx, bar) in enumerate(post_ib.iterrows()):
            price = bar['close']
            bar_time = bar['timestamp'].time() if 'timestamp' in bar.index else None

            # No entries after 2:00 PM
            from datetime import time as _time
            if bar_time and bar_time >= _time(14, 0):
                if entry_taken:
                    # Close at EOD
                    exit_price = price
                    exit_fill = execution.fill_exit('SHORT', exit_price)
                    gross, comm, slip, net = execution.calculate_net_pnl(
                        'SHORT', entry_price, exit_fill, MAX_CONTRACTS)
                    trades.append({
                        'date': str(session_date), 'entry_price': entry_price,
                        'exit_price': exit_fill, 'net_pnl': net,
                        'exit_reason': 'EOD', 'bars_held': i - entry_bar_idx,
                        'config': config_name,
                    })
                    entry_taken = False
                break  # Stop processing after 2 PM for this strategy

            if bar['high'] > session_high:
                session_high = bar['high']

            # Day type check
            if price > ib_high:
                ib_dir = 'BULL'
                ext = (price - ib_mid) / ib_range
            elif price < ib_low:
                ib_dir = 'BEAR'
                ext = (ib_mid - price) / ib_range
            else:
                ib_dir = 'INSIDE'
                ext = 0.0

            strength = classify_trend_strength(ext)
            day_type = classify_day_type(ib_high, ib_low, price, ib_dir, strength)
            dt_val = day_type.value if hasattr(day_type, 'value') else str(day_type)

            # Manage open position
            if entry_taken:
                # Check stop
                if bar['high'] >= stop_price:
                    exit_fill = execution.fill_exit('SHORT', stop_price)
                    gross, comm, slip, net = execution.calculate_net_pnl(
                        'SHORT', entry_price, exit_fill, MAX_CONTRACTS)
                    trades.append({
                        'date': str(session_date), 'entry_price': entry_price,
                        'exit_price': exit_fill, 'net_pnl': net,
                        'exit_reason': 'STOP', 'bars_held': i - entry_bar_idx,
                        'config': config_name,
                    })
                    entry_taken = False
                    continue

                # Check target
                if bar['low'] <= target_price:
                    exit_fill = execution.fill_exit('SHORT', target_price)
                    gross, comm, slip, net = execution.calculate_net_pnl(
                        'SHORT', entry_price, exit_fill, MAX_CONTRACTS)
                    trades.append({
                        'date': str(session_date), 'entry_price': entry_price,
                        'exit_price': exit_fill, 'net_pnl': net,
                        'exit_reason': 'TARGET', 'bars_held': i - entry_bar_idx,
                        'config': config_name,
                    })
                    entry_taken = False
                    continue

                # Check VWAP breach (for tiered, if price drops below VWAP take profit)
                if target_mode == 'tiered':
                    vwap_now = bar.get('vwap', ib_mid)
                    if not pd.isna(vwap_now) and bar['low'] <= vwap_now:
                        exit_fill = execution.fill_exit('SHORT', vwap_now)
                        gross, comm, slip, net = execution.calculate_net_pnl(
                            'SHORT', entry_price, exit_fill, MAX_CONTRACTS)
                        trades.append({
                            'date': str(session_date), 'entry_price': entry_price,
                            'exit_price': exit_fill, 'net_pnl': net,
                            'exit_reason': 'VWAP_HIT', 'bars_held': i - entry_bar_idx,
                            'config': config_name,
                        })
                        entry_taken = False
                        continue

                continue

            # === ENTRY LOGIC ===
            if entry_taken:
                continue

            # Day type must be in allowed list
            if dt_val not in allowed_day_types:
                continue

            # Price must be near IBH (within 0.15x IB range)
            if bar['high'] < ib_high - (ib_range * 0.15):
                continue

            # Max extension check: don't fade if already extended far above IBH
            if session_high > ib_high + (ib_range * max_extension):
                continue

            # Rejection bar: wick above IBH but close below
            if require_rejection_bar:
                if not (bar['high'] >= ib_high - (ib_range * 0.05) and price < ib_high):
                    continue
                # Wick ratio check (seller rejection)
                bar_range = bar['high'] - bar['low']
                if bar_range > 0:
                    upper_wick = bar['high'] - max(bar['open'], price)
                    wick_ratio = upper_wick / bar_range
                    if wick_ratio < 0.25:
                        continue  # Not enough rejection wick

            # Delta check
            delta = bar.get('delta', 0)
            if pd.isna(delta):
                delta = 0
            if require_delta_neg and delta >= 0:
                continue

            # Volume check
            if require_volume:
                vol_spike = bar.get('volume_spike', 1.0)
                if pd.isna(vol_spike) or vol_spike < 1.0:
                    continue

            # ENTRY SHORT
            entry_price_raw = price
            entry_fill = execution.fill_entry('SHORT', entry_price_raw)
            entry_price = entry_fill

            # Stop: above IBH + buffer
            stop_price = ib_high + (ib_range * 0.15)
            stop_price = max(stop_price, entry_price + 15.0)

            # Target based on mode
            if target_mode == 'ib_mid':
                target_price = ib_mid
            elif target_mode == 'vwap':
                vwap_val = bar.get('vwap', ib_mid)
                target_price = vwap_val if not pd.isna(vwap_val) else ib_mid
            elif target_mode == 'val':
                target_price = val
            elif target_mode == 'tiered':
                # First target: POC/IB mid (let VWAP check handle deeper targets)
                target_price = val  # Ultimate target is VAL, but VWAP check exits earlier
            elif target_mode == 'vah':
                target_price = vah
            else:
                target_price = ib_mid

            # R:R check
            risk = stop_price - entry_price
            reward = entry_price - target_price
            if reward <= 0 or risk <= 0:
                entry_taken = False
                entry_price = None
                continue

            entry_taken = True
            entry_bar_idx = i
            entry_time = bar.get('timestamp')

        # Force close at EOD if still open
        if entry_taken and len(post_ib) > 0:
            last_bar = post_ib.iloc[-1]
            exit_fill = execution.fill_exit('SHORT', last_bar['close'])
            gross, comm, slip, net = execution.calculate_net_pnl(
                'SHORT', entry_price, exit_fill, MAX_CONTRACTS)
            trades.append({
                'date': str(session_date), 'entry_price': entry_price,
                'exit_price': exit_fill, 'net_pnl': net,
                'exit_reason': 'EOD', 'bars_held': len(post_ib) - entry_bar_idx,
                'config': config_name,
            })

    return trades


# ============================================================================
# RUN ALL CONFIGURATIONS
# ============================================================================

configs = [
    # (name, allowed_day_types, target_mode, rejection_bar, delta_neg, volume, max_ext)
    ('A: B-Day IBH→IB mid (strict)',
     ['b_day'], 'ib_mid', True, True, False, 0.3),
    ('B: B-Day IBH→VWAP',
     ['b_day'], 'vwap', True, True, False, 0.3),
    ('C: B-Day IBH→VAL',
     ['b_day'], 'val', True, True, False, 0.3),
    ('D: B-Day IBH→VAH (quick scalp)',
     ['b_day'], 'vah', True, True, False, 0.3),
    ('E: Balance+Neutral IBH→IB mid',
     ['b_day', 'neutral'], 'ib_mid', True, True, False, 0.3),
    ('F: Balance+Neutral+PDay IBH→IB mid',
     ['b_day', 'neutral', 'p_day'], 'ib_mid', True, True, False, 0.3),
    ('G: Balance+Neutral IBH→VWAP',
     ['b_day', 'neutral'], 'vwap', True, True, False, 0.3),
    ('H: Balance+Neutral IBH→Tiered',
     ['b_day', 'neutral'], 'tiered', True, True, False, 0.3),
    ('I: Relaxed (no delta req) IBH→VWAP',
     ['b_day', 'neutral'], 'vwap', True, False, False, 0.3),
    ('J: Strict (delta+vol) IBH→IB mid',
     ['b_day', 'neutral'], 'ib_mid', True, True, True, 0.3),
    ('K: Wide ext allowed (0.5x) IBH→IB mid',
     ['b_day', 'neutral'], 'ib_mid', True, True, False, 0.5),
    ('L: No rejection bar IBH→VWAP',
     ['b_day', 'neutral'], 'vwap', False, True, False, 0.3),
]

all_results = []

for name, day_types, target, rej, delta_n, vol, max_ext in configs:
    trades = simulate_ibh_fade(name, day_types, target, rej, delta_n, vol, max_ext)
    wins = [t for t in trades if t['net_pnl'] > 0]
    losses = [t for t in trades if t['net_pnl'] <= 0]
    total_pnl = sum(t['net_pnl'] for t in trades)
    wr = len(wins) / len(trades) * 100 if trades else 0
    avg_win = np.mean([t['net_pnl'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['net_pnl'] for t in losses]) if losses else 0

    all_results.append({
        'name': name,
        'trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'wr': wr,
        'pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'trade_list': trades,
    })

# ============================================================================
# PRINT RESULTS TABLE
# ============================================================================
print(f"\n{'Config':<48s} {'Trades':>6s} {'W':>3s} {'L':>3s} {'WR%':>6s} {'Net PnL':>10s} {'AvgWin':>8s} {'AvgLoss':>8s}")
print("-" * 100)

for r in all_results:
    print(f"{r['name']:<48s} {r['trades']:>6d} {r['wins']:>3d} {r['losses']:>3d} "
          f"{r['wr']:>5.1f}% ${r['pnl']:>8,.0f} ${r['avg_win']:>6,.0f} ${r['avg_loss']:>6,.0f}")

print("-" * 100)

# ============================================================================
# DETAILED TRADE LOG FOR BEST CONFIG
# ============================================================================
best = max(all_results, key=lambda x: x['pnl'] if x['trades'] > 0 else -9999)
print(f"\n\n{'=' * 130}")
print(f"  BEST CONFIG: {best['name']}")
print(f"  {best['trades']} trades | {best['wr']:.1f}% WR | ${best['pnl']:,.0f} net")
print(f"{'=' * 130}")

if best['trade_list']:
    print(f"\n{'Date':<14s} {'Entry':>10s} {'Exit':>10s} {'Net PnL':>10s} {'Exit Reason':<12s} {'Bars':<6s}")
    print("-" * 70)
    for t in sorted(best['trade_list'], key=lambda x: x['date']):
        print(f"{t['date']:<14s} {t['entry_price']:>10.2f} {t['exit_price']:>10.2f} "
              f"${t['net_pnl']:>8,.2f} {t['exit_reason']:<12s} {t['bars_held']:<6d}")

    # Exit reason summary
    exit_summary = defaultdict(list)
    for t in best['trade_list']:
        exit_summary[t['exit_reason']].append(t['net_pnl'])

    print(f"\n  Exit reasons:")
    for reason, pnls in sorted(exit_summary.items()):
        w = sum(1 for p in pnls if p > 0)
        print(f"    {reason:<12s}: {len(pnls)} trades, {w}/{len(pnls)} wins, ${sum(pnls):>8,.0f}")

# ============================================================================
# COMPARISON: IBH FADE SHORT vs EXISTING LONG-ONLY STRATEGIES
# ============================================================================
print(f"\n\n{'=' * 130}")
print("  COMPARISON: IBH FADE SHORT vs CURRENT LONG-ONLY PLAYBOOK")
print(f"{'=' * 130}")
print(f"""
  CURRENT PLAYBOOK (Core + MeanRev, LONG ONLY):
    75.9% WR | 29 trades | $3,861 net | $133/trade

  BEST IBH FADE SHORT:
    {best['wr']:.1f}% WR | {best['trades']} trades | ${best['pnl']:,.0f} net | ${best['pnl']/best['trades'] if best['trades'] else 0:,.0f}/trade

  VERDICT:""")

if best['wr'] >= 50 and best['pnl'] > 0:
    print(f"    IBH Fade SHORT shows SOME edge ({best['wr']:.1f}% WR, ${best['pnl']:,.0f}).")
    print(f"    But adding it to the playbook would lower portfolio WR from 75.9%.")
    combined_trades = 29 + best['trades']
    combined_wins = 22 + best['wins']
    combined_wr = combined_wins / combined_trades * 100 if combined_trades else 0
    combined_pnl = 3861 + best['pnl']
    print(f"    Combined: {combined_wr:.1f}% WR, {combined_trades} trades, ${combined_pnl:,.0f}")
    if combined_wr >= 70:
        print(f"    COMBINED WR STILL >= 70% — could be added if you want more trades.")
    else:
        print(f"    COMBINED WR DROPS BELOW 70% — NOT recommended for Lightning 70% target.")
elif best['trades'] == 0:
    print(f"    No trades triggered. The strict criteria (rejection + delta) + balance day")
    print(f"    classification doesn't produce enough IBH touches on short side.")
else:
    print(f"    IBH Fade SHORT is NEGATIVE or sub-50% WR on NQ.")
    print(f"    Confirms NQ long bias kills shorts even on balance/neutral days.")
    print(f"    DO NOT ADD to playbook.")
