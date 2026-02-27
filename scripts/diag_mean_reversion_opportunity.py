"""Quantify mean reversion opportunity across 259 sessions.

Key hypothesis: 75% of sessions fail to break out of IB.
For those sessions, mean reversion from IB edge back to mid/opposite edge
is a high-probability trade.

Studies:
1. IB breakout failure rate (extension < 0.5x then reverts)
2. Mean reversion from IB edge: entry, target, stop, R:R
3. "Weak extension" fade: when price pokes beyond IB but comes back
4. Time-of-day analysis: when do fakeouts peak?
5. What % of post-IB extensions revert within N bars?
6. Profit potential if we faded every weak extension
7. Compare to what our current strategies actually capture
"""
import sys, warnings, io
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import time
from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from config.instruments import get_instrument
from config.constants import IB_BARS_1MIN
from strategy.day_type import classify_day_type, classify_trend_strength

full_df = load_csv('NQ')
df = filter_rth(full_df)
df = compute_all_features(df)
instrument = get_instrument('MNQ')
point_value = instrument.point_value

sessions = sorted(df['session_date'].unique())
print(f"Sessions: {len(sessions)}")
print(f"Date range: {sessions[0]} to {sessions[-1]}")

# ================================================================
# STUDY 1: IB Breakout Success/Failure Rate
# ================================================================
print(f"\n{'='*120}")
print(f"  STUDY 1: IB BREAKOUT SUCCESS vs FAILURE (bar-by-bar)")
print(f"{'='*120}")

breakout_data = []

for session_date in sessions:
    sdf = df[df['session_date'] == session_date].copy()
    if len(sdf) < IB_BARS_1MIN + 30:
        continue

    ib_df = sdf.head(IB_BARS_1MIN)
    ib_high = ib_df['high'].max()
    ib_low = ib_df['low'].min()
    ib_range = ib_high - ib_low
    ib_mid = (ib_high + ib_low) / 2

    if ib_range <= 0:
        continue

    post_ib = sdf.iloc[IB_BARS_1MIN:]
    if len(post_ib) == 0:
        continue

    # Track breakout attempts
    bull_breakout = False
    bear_breakout = False
    bull_max_ext = 0.0
    bear_max_ext = 0.0
    bull_reverted = False
    bear_reverted = False
    first_bull_break_bar = None
    first_bear_break_bar = None

    # Track bar-by-bar extensions
    bar_extensions = []

    for i in range(len(post_ib)):
        bar = post_ib.iloc[i]
        h = bar['high']
        l = bar['low']
        c = bar['close']

        # Bull breakout
        if h > ib_high:
            if not bull_breakout:
                bull_breakout = True
                first_bull_break_bar = i
            ext = (h - ib_high) / ib_range
            bull_max_ext = max(bull_max_ext, ext)

        # Bear breakout
        if l < ib_low:
            if not bear_breakout:
                bear_breakout = True
                first_bear_break_bar = i
            ext = (ib_low - l) / ib_range
            bear_max_ext = max(bear_max_ext, ext)

        # Track close position relative to IB
        if c > ib_high:
            bar_ext = (c - ib_mid) / ib_range
        elif c < ib_low:
            bar_ext = -(ib_mid - c) / ib_range  # negative for bear
        else:
            bar_ext = 0.0
        bar_extensions.append(bar_ext)

    # EOD close position
    eod_close = post_ib.iloc[-1]['close']
    eod_inside_ib = ib_low <= eod_close <= ib_high

    # Bull reversion: broke above IB but closed back inside
    if bull_breakout and eod_close <= ib_high:
        bull_reverted = True
    # Bear reversion: broke below IB but closed back inside
    if bear_breakout and eod_close >= ib_low:
        bear_reverted = True

    # Final extension
    if eod_close > ib_mid:
        eod_ext = (eod_close - ib_mid) / ib_range
    else:
        eod_ext = (ib_mid - eod_close) / ib_range

    month = str(session_date)[:7]

    breakout_data.append({
        'date': session_date,
        'month': month,
        'ib_range': ib_range,
        'bull_breakout': bull_breakout,
        'bear_breakout': bear_breakout,
        'bull_max_ext': bull_max_ext,
        'bear_max_ext': bear_max_ext,
        'bull_reverted': bull_reverted,
        'bear_reverted': bear_reverted,
        'eod_inside_ib': eod_inside_ib,
        'eod_ext': eod_ext,
        'eod_close': eod_close,
        'ib_high': ib_high,
        'ib_low': ib_low,
        'ib_mid': ib_mid,
        'max_ext': max(bull_max_ext, bear_max_ext),
        'session_range': sdf['high'].max() - sdf['low'].min(),
    })

bdf = pd.DataFrame(breakout_data)
n = len(bdf)

# Overall breakout stats
any_breakout = bdf['bull_breakout'] | bdf['bear_breakout']
no_breakout = ~any_breakout
bull_only = bdf['bull_breakout'] & ~bdf['bear_breakout']
bear_only = ~bdf['bull_breakout'] & bdf['bear_breakout']
both_break = bdf['bull_breakout'] & bdf['bear_breakout']

print(f"\n  IB Breakout Summary ({n} sessions):")
print(f"    No breakout (inside IB all day):   {no_breakout.sum():>4d} ({no_breakout.sum()/n*100:>5.1f}%)")
print(f"    Bull breakout only:                {bull_only.sum():>4d} ({bull_only.sum()/n*100:>5.1f}%)")
print(f"    Bear breakout only:                {bear_only.sum():>4d} ({bear_only.sum()/n*100:>5.1f}%)")
print(f"    BOTH sides broken:                 {both_break.sum():>4d} ({both_break.sum()/n*100:>5.1f}%)")
print(f"    Any breakout:                      {any_breakout.sum():>4d} ({any_breakout.sum()/n*100:>5.1f}%)")

# Reversion rates
bull_bo = bdf[bdf['bull_breakout']]
bear_bo = bdf[bdf['bear_breakout']]
print(f"\n  Reversion after breakout:")
print(f"    Bull breakouts: {len(bull_bo)}, reverted back inside: {bdf['bull_reverted'].sum()} ({bdf['bull_reverted'].sum()/len(bull_bo)*100:.1f}%)" if len(bull_bo) > 0 else "")
print(f"    Bear breakouts: {len(bear_bo)}, reverted back inside: {bdf['bear_reverted'].sum()} ({bdf['bear_reverted'].sum()/len(bear_bo)*100:.1f}%)" if len(bear_bo) > 0 else "")

# EOD inside IB
print(f"\n  End-of-day close inside IB: {bdf['eod_inside_ib'].sum()}/{n} ({bdf['eod_inside_ib'].sum()/n*100:.1f}%)")

# ================================================================
# STUDY 2: Breakout Extension Depth vs Reversion
# ================================================================
print(f"\n\n{'='*120}")
print(f"  STUDY 2: BREAKOUT EXTENSION DEPTH vs REVERSION RATE")
print(f"{'='*120}")

# How far does price extend beyond IB, and how often does it come back?
ext_bins = [0, 0.2, 0.5, 1.0, 1.5, 2.0, 5.0]
ext_labels = ['0-0.2x', '0.2-0.5x', '0.5-1.0x', '1.0-1.5x', '1.5-2.0x', '>2.0x']

# Use max extension (whichever side extended more)
breakout_sessions = bdf[any_breakout].copy()
breakout_sessions['ext_bucket'] = pd.cut(breakout_sessions['max_ext'], bins=ext_bins, labels=ext_labels, right=False)

print(f"\n  Max extension beyond IB edge (as multiple of IB range):")
print(f"  {'Extension':12s} {'Sessions':>8s} {'Reverted':>8s} {'Rev%':>8s} {'EOD Inside':>10s} {'Avg IB':>8s}")
print(f"  {'-'*65}")

for bucket in ext_labels:
    sub = breakout_sessions[breakout_sessions['ext_bucket'] == bucket]
    if len(sub) == 0:
        continue
    # "Reverted" = either bull or bear reverted
    reverted = ((sub['bull_breakout'] & sub['bull_reverted']) | (sub['bear_breakout'] & sub['bear_reverted'])).sum()
    eod_in = sub['eod_inside_ib'].sum()
    print(f"  {bucket:12s} {len(sub):>8d} {reverted:>8d} {reverted/len(sub)*100:>7.1f}% {eod_in:>10d} {sub['ib_range'].mean():>7.0f}pt")

# ================================================================
# STUDY 3: Bar-by-Bar Mean Reversion Simulation
# ================================================================
print(f"\n\n{'='*120}")
print(f"  STUDY 3: MEAN REVERSION TRADE SIMULATION")
print(f"{'='*120}")
print(f"  Strategy: When price touches IB edge from inside, fade back to IB mid")
print(f"  Stop: 0.3x IB range beyond edge | Target: IB mid (0.5x IB range)")

reversion_trades = []

for session_date in sessions:
    sdf = df[df['session_date'] == session_date].copy()
    if len(sdf) < IB_BARS_1MIN + 30:
        continue

    ib_df = sdf.head(IB_BARS_1MIN)
    ib_high = ib_df['high'].max()
    ib_low = ib_df['low'].min()
    ib_range = ib_high - ib_low
    ib_mid = (ib_high + ib_low) / 2

    if ib_range < 30:  # skip ultra-narrow IB
        continue

    post_ib = sdf.iloc[IB_BARS_1MIN:]
    month = str(session_date)[:7]

    # Simulate: fade touches of IB high/low
    # Max 1 trade per direction per session
    took_short = False
    took_long = False

    for i in range(len(post_ib)):
        bar = post_ib.iloc[i]
        bar_time = bar['timestamp'].time() if hasattr(bar['timestamp'], 'time') else None

        # No new trades after 2:30 PM
        if bar_time and bar_time >= time(14, 30):
            break

        # SHORT: price touches IB high from inside (bar high >= ib_high, prev close < ib_high)
        if not took_short and bar['high'] >= ib_high and bar['close'] <= ib_high + 0.1 * ib_range:
            # Entry at IB high, stop at IB high + 0.3x IB range, target at IB mid
            entry = ib_high
            stop = ib_high + 0.3 * ib_range
            target = ib_mid
            risk = stop - entry
            reward = entry - target

            # Simulate forward
            result = None
            exit_price = None
            bars_held = 0
            for j in range(i + 1, len(post_ib)):
                fbar = post_ib.iloc[j]
                bars_held += 1

                # Check stop (high goes above stop)
                if fbar['high'] >= stop:
                    result = 'STOP'
                    exit_price = stop
                    break
                # Check target (low goes below target)
                if fbar['low'] <= target:
                    result = 'TARGET'
                    exit_price = target
                    break
                # EOD exit
                fbar_time = fbar['timestamp'].time() if hasattr(fbar['timestamp'], 'time') else None
                if fbar_time and fbar_time >= time(15, 30):
                    result = 'EOD'
                    exit_price = fbar['close']
                    break

            if result is None:
                result = 'EOD'
                exit_price = post_ib.iloc[-1]['close']

            pnl_pts = entry - exit_price  # short trade
            pnl_dollar = pnl_pts * point_value

            reversion_trades.append({
                'date': session_date, 'month': month,
                'direction': 'SHORT', 'entry': entry, 'stop': stop,
                'target': target, 'exit': exit_price, 'result': result,
                'pnl_pts': pnl_pts, 'pnl_dollar': pnl_dollar,
                'risk_pts': risk, 'reward_pts': reward,
                'bars_held': bars_held, 'ib_range': ib_range,
                'r_multiple': pnl_pts / risk if risk > 0 else 0,
            })
            took_short = True

        # LONG: price touches IB low from inside
        if not took_long and bar['low'] <= ib_low and bar['close'] >= ib_low - 0.1 * ib_range:
            entry = ib_low
            stop = ib_low - 0.3 * ib_range
            target = ib_mid
            risk = entry - stop
            reward = target - entry

            result = None
            exit_price = None
            bars_held = 0
            for j in range(i + 1, len(post_ib)):
                fbar = post_ib.iloc[j]
                bars_held += 1

                if fbar['low'] <= stop:
                    result = 'STOP'
                    exit_price = stop
                    break
                if fbar['high'] >= target:
                    result = 'TARGET'
                    exit_price = target
                    break
                fbar_time = fbar['timestamp'].time() if hasattr(fbar['timestamp'], 'time') else None
                if fbar_time and fbar_time >= time(15, 30):
                    result = 'EOD'
                    exit_price = fbar['close']
                    break

            if result is None:
                result = 'EOD'
                exit_price = post_ib.iloc[-1]['close']

            pnl_pts = exit_price - entry  # long trade
            pnl_dollar = pnl_pts * point_value

            reversion_trades.append({
                'date': session_date, 'month': month,
                'direction': 'LONG', 'entry': entry, 'stop': stop,
                'target': target, 'exit': exit_price, 'result': result,
                'pnl_pts': pnl_pts, 'pnl_dollar': pnl_dollar,
                'risk_pts': risk, 'reward_pts': reward,
                'bars_held': bars_held, 'ib_range': ib_range,
                'r_multiple': pnl_pts / risk if risk > 0 else 0,
            })
            took_long = True

rtdf = pd.DataFrame(reversion_trades)
total_trades = len(rtdf)
winners = rtdf[rtdf['pnl_dollar'] > 0]
losers = rtdf[rtdf['pnl_dollar'] <= 0]
net_pnl = rtdf['pnl_dollar'].sum()
wr = len(winners) / total_trades * 100 if total_trades > 0 else 0
avg_win = winners['pnl_dollar'].mean() if len(winners) > 0 else 0
avg_loss = losers['pnl_dollar'].mean() if len(losers) > 0 else 0

print(f"\n  BASIC MEAN REVERSION (fade IB edge -> target IB mid, stop 0.3x IB beyond edge)")
print(f"  Trades: {total_trades} | Winners: {len(winners)} | Losers: {len(losers)}")
print(f"  Win Rate: {wr:.1f}%")
print(f"  Net P&L: ${net_pnl:,.0f} (MNQ ${point_value}/pt)")
print(f"  Avg Winner: ${avg_win:,.0f} | Avg Loser: ${avg_loss:,.0f}")
print(f"  Avg R-multiple: {rtdf['r_multiple'].mean():.2f}R")
print(f"  Expectancy: ${net_pnl/total_trades:,.0f}/trade")

# By exit type
print(f"\n  By Exit Type:")
for exit_type in ['TARGET', 'STOP', 'EOD']:
    sub = rtdf[rtdf['result'] == exit_type]
    if len(sub) > 0:
        print(f"    {exit_type:8s}: {len(sub):>4d} trades, ${sub['pnl_dollar'].sum():>8,.0f}, "
              f"avg ${sub['pnl_dollar'].mean():>6,.0f}, avg {sub['bars_held'].mean():.0f} bars")

# By direction
print(f"\n  By Direction:")
for direction in ['LONG', 'SHORT']:
    sub = rtdf[rtdf['direction'] == direction]
    if len(sub) > 0:
        sw = len(sub[sub['pnl_dollar'] > 0])
        print(f"    {direction:6s}: {len(sub):>4d} trades, WR {sw/len(sub)*100:.1f}%, "
              f"${sub['pnl_dollar'].sum():>8,.0f}, avg ${sub['pnl_dollar'].mean():>6,.0f}")

# ================================================================
# STUDY 4: Refined Mean Reversion (with filters)
# ================================================================
print(f"\n\n{'='*120}")
print(f"  STUDY 4: FILTERED MEAN REVERSION STRATEGIES")
print(f"{'='*120}")

# Test different filter combinations
filters = {
    'Baseline (no filter)': rtdf,
    'IB > 100 pts': rtdf[rtdf['ib_range'] > 100],
    'IB > 150 pts': rtdf[rtdf['ib_range'] > 150],
    'IB < 300 pts': rtdf[rtdf['ib_range'] < 300],
    'IB 100-300 pts': rtdf[(rtdf['ib_range'] >= 100) & (rtdf['ib_range'] <= 300)],
    'IB 100-250 pts': rtdf[(rtdf['ib_range'] >= 100) & (rtdf['ib_range'] <= 250)],
}

print(f"\n  {'Filter':25s} {'Trades':>6s} {'WR':>6s} {'Net P&L':>10s} {'$/trade':>8s} {'AvgW':>8s} {'AvgL':>8s}")
print(f"  {'-'*80}")

for label, sub in filters.items():
    if len(sub) == 0:
        continue
    n = len(sub)
    w = len(sub[sub['pnl_dollar'] > 0])
    net = sub['pnl_dollar'].sum()
    wr = w / n * 100
    aw = sub[sub['pnl_dollar'] > 0]['pnl_dollar'].mean() if w > 0 else 0
    al = sub[sub['pnl_dollar'] <= 0]['pnl_dollar'].mean() if n - w > 0 else 0
    print(f"  {label:25s} {n:>6d} {wr:>5.1f}% ${net:>8,.0f} ${net/n:>6,.0f} ${aw:>6,.0f} ${al:>6,.0f}")

# ================================================================
# STUDY 5: Alternative R:R Configurations
# ================================================================
print(f"\n\n{'='*120}")
print(f"  STUDY 5: MEAN REVERSION WITH DIFFERENT STOP/TARGET CONFIGS")
print(f"{'='*120}")

configs = [
    ('Stop 0.2x, Tgt Mid',   0.2, 0.5),
    ('Stop 0.3x, Tgt Mid',   0.3, 0.5),   # baseline
    ('Stop 0.4x, Tgt Mid',   0.4, 0.5),
    ('Stop 0.3x, Tgt 0.3x',  0.3, 0.3),   # tighter target
    ('Stop 0.3x, Tgt 0.7x',  0.3, 0.7),   # wider target (opposite IB edge)
    ('Stop 0.5x, Tgt Mid',   0.5, 0.5),    # wider stop
    ('Stop 0.5x, Tgt 0.7x',  0.5, 0.7),   # wide stop, far target
]

print(f"\n  Simulating each config from scratch (bar-by-bar)...")
print(f"  {'Config':25s} {'Trades':>6s} {'WR':>6s} {'Net P&L':>10s} {'$/trade':>8s} {'PF':>6s} {'R:R':>6s}")
print(f"  {'-'*80}")

for config_name, stop_mult, target_mult in configs:
    config_trades = []

    for session_date in sessions:
        sdf = df[df['session_date'] == session_date].copy()
        if len(sdf) < IB_BARS_1MIN + 30:
            continue

        ib_df = sdf.head(IB_BARS_1MIN)
        ib_high = ib_df['high'].max()
        ib_low = ib_df['low'].min()
        ib_range = ib_high - ib_low
        ib_mid = (ib_high + ib_low) / 2

        if ib_range < 50 or ib_range > 350:  # reasonable IB filter
            continue

        post_ib = sdf.iloc[IB_BARS_1MIN:]
        took_short = False
        took_long = False

        for i in range(len(post_ib)):
            bar = post_ib.iloc[i]
            bar_time = bar['timestamp'].time() if hasattr(bar['timestamp'], 'time') else None
            if bar_time and bar_time >= time(14, 30):
                break

            # SHORT at IB high
            if not took_short and bar['high'] >= ib_high and bar['close'] <= ib_high + 0.1 * ib_range:
                entry = ib_high
                stop = ib_high + stop_mult * ib_range
                target = ib_high - target_mult * ib_range
                risk = stop - entry

                result = None
                exit_price = None
                for j in range(i + 1, len(post_ib)):
                    fbar = post_ib.iloc[j]
                    if fbar['high'] >= stop:
                        result = 'STOP'; exit_price = stop; break
                    if fbar['low'] <= target:
                        result = 'TARGET'; exit_price = target; break
                    fbt = fbar['timestamp'].time() if hasattr(fbar['timestamp'], 'time') else None
                    if fbt and fbt >= time(15, 30):
                        result = 'EOD'; exit_price = fbar['close']; break
                if result is None:
                    result = 'EOD'; exit_price = post_ib.iloc[-1]['close']

                pnl = (entry - exit_price) * point_value
                config_trades.append({'pnl': pnl, 'risk': risk * point_value})
                took_short = True

            # LONG at IB low
            if not took_long and bar['low'] <= ib_low and bar['close'] >= ib_low - 0.1 * ib_range:
                entry = ib_low
                stop = ib_low - stop_mult * ib_range
                target = ib_low + target_mult * ib_range
                risk = entry - stop

                result = None
                exit_price = None
                for j in range(i + 1, len(post_ib)):
                    fbar = post_ib.iloc[j]
                    if fbar['low'] <= stop:
                        result = 'STOP'; exit_price = stop; break
                    if fbar['high'] >= target:
                        result = 'TARGET'; exit_price = target; break
                    fbt = fbar['timestamp'].time() if hasattr(fbar['timestamp'], 'time') else None
                    if fbt and fbt >= time(15, 30):
                        result = 'EOD'; exit_price = fbar['close']; break
                if result is None:
                    result = 'EOD'; exit_price = post_ib.iloc[-1]['close']

                pnl = (exit_price - entry) * point_value
                config_trades.append({'pnl': pnl, 'risk': risk * point_value})
                took_long = True

    if not config_trades:
        continue
    cdf = pd.DataFrame(config_trades)
    n = len(cdf)
    w = len(cdf[cdf['pnl'] > 0])
    net = cdf['pnl'].sum()
    gw = cdf[cdf['pnl'] > 0]['pnl'].sum()
    gl = cdf[cdf['pnl'] <= 0]['pnl'].sum()
    pf = abs(gw / gl) if gl != 0 else float('inf')
    rr = (target_mult / stop_mult)
    print(f"  {config_name:25s} {n:>6d} {w/n*100:>5.1f}% ${net:>8,.0f} ${net/n:>6,.0f} {pf:>5.2f} {rr:>5.2f}")

# ================================================================
# STUDY 6: "Weak Extension Fade" -- fade pokes beyond IB that haven't committed
# ================================================================
print(f"\n\n{'='*120}")
print(f"  STUDY 6: WEAK EXTENSION FADE (price pokes 0.1-0.3x beyond IB edge)")
print(f"{'='*120}")
print(f"  Entry: when close is 0.1-0.3x beyond IB edge (uncommitted poke)")
print(f"  Target: IB mid | Stop: 0.5x beyond IB edge")

weak_fade_trades = []

for session_date in sessions:
    sdf = df[df['session_date'] == session_date].copy()
    if len(sdf) < IB_BARS_1MIN + 30:
        continue

    ib_df = sdf.head(IB_BARS_1MIN)
    ib_high = ib_df['high'].max()
    ib_low = ib_df['low'].min()
    ib_range = ib_high - ib_low
    ib_mid = (ib_high + ib_low) / 2

    if ib_range < 50 or ib_range > 350:
        continue

    post_ib = sdf.iloc[IB_BARS_1MIN:]
    took_short = False
    took_long = False
    month = str(session_date)[:7]

    for i in range(1, len(post_ib)):
        bar = post_ib.iloc[i]
        prev = post_ib.iloc[i - 1]
        bar_time = bar['timestamp'].time() if hasattr(bar['timestamp'], 'time') else None
        if bar_time and bar_time >= time(14, 0):
            break

        c = bar['close']

        # Weak bull poke: close is 0.1-0.3x above IB high
        bull_ext = (c - ib_high) / ib_range if c > ib_high else 0
        bear_ext = (ib_low - c) / ib_range if c < ib_low else 0

        # SHORT: weak bull extension that might fail
        if not took_short and 0.05 <= bull_ext <= 0.3:
            # Extra filter: previous bar was inside IB (just poked out)
            if prev['close'] <= ib_high + 0.1 * ib_range:
                entry = c
                stop = ib_high + 0.5 * ib_range
                target = ib_mid
                risk = stop - entry

                if risk <= 0:
                    continue

                result = None
                exit_price = None
                bars_held = 0
                for j in range(i + 1, len(post_ib)):
                    fbar = post_ib.iloc[j]
                    bars_held += 1
                    if fbar['high'] >= stop:
                        result = 'STOP'; exit_price = stop; break
                    if fbar['low'] <= target:
                        result = 'TARGET'; exit_price = target; break
                    fbt = fbar['timestamp'].time() if hasattr(fbar['timestamp'], 'time') else None
                    if fbt and fbt >= time(15, 30):
                        result = 'EOD'; exit_price = fbar['close']; break
                if result is None:
                    result = 'EOD'; exit_price = post_ib.iloc[-1]['close']

                pnl_pts = entry - exit_price
                pnl_dollar = pnl_pts * point_value
                weak_fade_trades.append({
                    'date': session_date, 'month': month,
                    'direction': 'SHORT', 'entry': entry, 'result': result,
                    'pnl_pts': pnl_pts, 'pnl_dollar': pnl_dollar,
                    'risk_pts': risk, 'bars_held': bars_held,
                    'ib_range': ib_range, 'ext_at_entry': bull_ext,
                    'r_multiple': pnl_pts / risk if risk > 0 else 0,
                })
                took_short = True

        # LONG: weak bear extension that might fail
        if not took_long and 0.05 <= bear_ext <= 0.3:
            if prev['close'] >= ib_low - 0.1 * ib_range:
                entry = c
                stop = ib_low - 0.5 * ib_range
                target = ib_mid
                risk = entry - stop

                if risk <= 0:
                    continue

                result = None
                exit_price = None
                bars_held = 0
                for j in range(i + 1, len(post_ib)):
                    fbar = post_ib.iloc[j]
                    bars_held += 1
                    if fbar['low'] <= stop:
                        result = 'STOP'; exit_price = stop; break
                    if fbar['high'] >= target:
                        result = 'TARGET'; exit_price = target; break
                    fbt = fbar['timestamp'].time() if hasattr(fbar['timestamp'], 'time') else None
                    if fbt and fbt >= time(15, 30):
                        result = 'EOD'; exit_price = fbar['close']; break
                if result is None:
                    result = 'EOD'; exit_price = post_ib.iloc[-1]['close']

                pnl_pts = exit_price - entry
                pnl_dollar = pnl_pts * point_value
                weak_fade_trades.append({
                    'date': session_date, 'month': month,
                    'direction': 'LONG', 'entry': entry, 'result': result,
                    'pnl_pts': pnl_pts, 'pnl_dollar': pnl_dollar,
                    'risk_pts': risk, 'bars_held': bars_held,
                    'ib_range': ib_range, 'ext_at_entry': bear_ext,
                    'r_multiple': pnl_pts / risk if risk > 0 else 0,
                })
                took_long = True

if weak_fade_trades:
    wfdf = pd.DataFrame(weak_fade_trades)
    n = len(wfdf)
    w = len(wfdf[wfdf['pnl_dollar'] > 0])
    net = wfdf['pnl_dollar'].sum()
    gw = wfdf[wfdf['pnl_dollar'] > 0]['pnl_dollar'].sum()
    gl = wfdf[wfdf['pnl_dollar'] <= 0]['pnl_dollar'].sum()
    pf = abs(gw / gl) if gl != 0 else float('inf')

    print(f"\n  Weak Extension Fade Results:")
    print(f"  Trades: {n} | Winners: {w} ({w/n*100:.1f}%) | Net: ${net:,.0f}")
    print(f"  PF: {pf:.2f} | Avg R: {wfdf['r_multiple'].mean():.2f}")
    print(f"  Avg Winner: ${wfdf[wfdf['pnl_dollar']>0]['pnl_dollar'].mean():,.0f} | Avg Loser: ${wfdf[wfdf['pnl_dollar']<=0]['pnl_dollar'].mean():,.0f}")

    # By exit type
    print(f"\n  By Exit:")
    for ex in ['TARGET', 'STOP', 'EOD']:
        sub = wfdf[wfdf['result'] == ex]
        if len(sub) > 0:
            print(f"    {ex:8s}: {len(sub):>4d}, ${sub['pnl_dollar'].sum():>8,.0f}, avg R {sub['r_multiple'].mean():.2f}")

    # Monthly
    print(f"\n  Monthly P&L:")
    for month in sorted(wfdf['month'].unique()):
        sub = wfdf[wfdf['month'] == month]
        mw = len(sub[sub['pnl_dollar'] > 0])
        print(f"    {month}: {len(sub):>3d}t, WR {mw/len(sub)*100:>4.0f}%, ${sub['pnl_dollar'].sum():>7,.0f}")

# ================================================================
# STUDY 7: Compare to Our Current Strategy Capture
# ================================================================
print(f"\n\n{'='*120}")
print(f"  STUDY 7: CURRENT STRATEGIES vs MEAN REVERSION OPPORTUNITY")
print(f"{'='*120}")

# How many of our current trades are effectively mean reversion?
from engine.backtest import BacktestEngine
from engine.execution import ExecutionModel
from engine.position import PositionManager
from strategy import get_core_strategies

strategies = get_core_strategies()
engine = BacktestEngine(
    instrument=instrument, strategies=strategies, filters=None,
    execution=ExecutionModel(instrument),
    position_mgr=PositionManager(max_drawdown=999999),
    full_df=full_df,
)
result = engine.run(df, verbose=False)

# Classify each existing trade as trend-following or mean-reversion
trend_following = 0
mean_revert = 0
other = 0

for t in result.trades:
    # If entry is beyond IB edge and target is toward IB mid = mean reversion
    # If entry is beyond IB edge and target is further from IB = trend following
    # We'd need session context but can approximate from strategy name
    sname = t.strategy_name
    if sname in ['B-Day IBL Fade', 'B-Day', 'Edge Fade']:
        mean_revert += 1
    elif sname in ['Trend Day Bull', 'Trend Up', 'P-Day', 'Opening Range Reversal']:
        # OR Rev is reversal (mean reversion from extreme)
        mean_revert += 1
    else:
        trend_following += 1

total_current = len(result.trades)
print(f"\n  Current portfolio: {total_current} trades across 259 sessions")
print(f"  Mean-reversion-like strategies: {mean_revert} trades")
print(f"  Trend-following strategies: {trend_following} trades")
print(f"  Other: {other} trades")

# Sessions with NO current trades vs sessions with reversion opportunity
current_dates = set(str(t.session_date)[:10] for t in result.trades)
all_dates = set(str(d)[:10] for d in sessions)
no_trade_dates = all_dates - current_dates
print(f"\n  Sessions with current trades: {len(current_dates)}")
print(f"  Sessions with ZERO trades: {len(no_trade_dates)}")

# How many zero-trade sessions had IB edge touches?
if weak_fade_trades:
    reversion_dates = set(str(t['date'])[:10] for t in weak_fade_trades)
    overlap = no_trade_dates & reversion_dates
    print(f"  Zero-trade sessions with reversion opportunity: {len(overlap)}")
    print(f"  -> These are MISSED opportunities our system doesn't capture")

# ================================================================
# STUDY 8: Per-Month Comparison -- Current vs Mean Reversion Add
# ================================================================
print(f"\n\n{'='*120}")
print(f"  STUDY 8: MONTHLY COMPARISON -- Current Portfolio + Mean Reversion")
print(f"{'='*120}")

# Current portfolio monthly
current_monthly = defaultdict(float)
for t in result.trades:
    m = str(t.session_date)[:7]
    current_monthly[m] += t.net_pnl

# Mean reversion monthly (weak fade)
mr_monthly = defaultdict(float)
if weak_fade_trades:
    for t in weak_fade_trades:
        mr_monthly[t['month']] += t['pnl_dollar']

all_months = sorted(set(list(current_monthly.keys()) + list(mr_monthly.keys())))
print(f"\n  {'Month':10s} {'Current':>10s} {'MR Add':>10s} {'Combined':>10s} {'MR Helps?':>10s}")
print(f"  {'-'*55}")

total_current_pnl = 0
total_mr_pnl = 0
for month in all_months:
    cp = current_monthly.get(month, 0)
    mr = mr_monthly.get(month, 0)
    combined = cp + mr
    total_current_pnl += cp
    total_mr_pnl += mr
    helps = "YES" if mr > 0 else ("FLAT" if mr == 0 else "no")
    # Highlight months where current is negative but MR helps
    if cp < 0 and combined > cp:
        helps = "** SAVES **"
    print(f"  {month:10s} ${cp:>8,.0f} ${mr:>8,.0f} ${combined:>8,.0f} {helps:>10s}")

print(f"  {'-'*55}")
print(f"  {'TOTAL':10s} ${total_current_pnl:>8,.0f} ${total_mr_pnl:>8,.0f} ${total_current_pnl+total_mr_pnl:>8,.0f}")

# Drawdown of combined
all_trades_combined = []
for t in result.trades:
    all_trades_combined.append({'date': str(t.session_date), 'pnl': t.net_pnl, 'source': 'current'})
if weak_fade_trades:
    for t in weak_fade_trades:
        all_trades_combined.append({'date': str(t['date']), 'pnl': t['pnl_dollar'], 'source': 'MR'})
all_trades_combined.sort(key=lambda x: x['date'])

eq = 0; pk = 0; mdd = 0
for t in all_trades_combined:
    eq += t['pnl']
    pk = max(pk, eq)
    mdd = max(mdd, pk - eq)

eq_curr = 0; pk_curr = 0; mdd_curr = 0
for t in sorted([{'date': str(t.session_date), 'pnl': t.net_pnl} for t in result.trades], key=lambda x: x['date']):
    eq_curr += t['pnl']
    pk_curr = max(pk_curr, eq_curr)
    mdd_curr = max(mdd_curr, pk_curr - eq_curr)

print(f"\n  Current portfolio DD: ${mdd_curr:,.0f}")
print(f"  Combined (Current + MR) DD: ${mdd:,.0f}")
print(f"  Prop firm DD limit: $4,500")

print(f"\n{'='*120}")
print(f"  END OF MEAN REVERSION STUDY")
print(f"{'='*120}")
