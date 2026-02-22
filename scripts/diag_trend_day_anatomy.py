"""
Diagnostic: Trend Day Anatomy

Questions to answer:
1. How often does price extend 0.5x+ beyond IB? 1.0x? 1.5x? 2.0x?
2. When Trend Bull trades, what are the actual stop distances?
3. If we used tighter stops (IBH retest, not IBL), what happens?
4. What % of OR Rev trades land on days that end up as trend days?
5. After acceptance above IBH, what's the MFE from various entry points?
6. What's the REAL problem -- stop width, target distance, or frequency?
"""

import sys, warnings, io
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
from strategy.day_type import classify_trend_strength, classify_day_type

instrument = get_instrument('MNQ')
point_value = instrument.point_value
slippage_cost = 2 * instrument.tick_size * point_value + instrument.commission * 2

print("Loading data...")
full_df = load_csv('NQ')
df = filter_rth(full_df)
df = compute_all_features(df)
sessions = sorted(df['session_date'].unique())
print(f"Loaded {len(sessions)} sessions\n")

# ================================================================
# TEST 1: How often does price extend beyond IB?
# ================================================================
print("=" * 100)
print("  TEST 1: IB EXTENSION FREQUENCY -- How often does price extend beyond IB?")
print("=" * 100)

extensions = []  # (session, max_bull_ext, max_bear_ext, ib_range, final_day_type)

for session_date in sessions:
    sdf = df[df['session_date'] == session_date]
    if len(sdf) < IB_BARS_1MIN:
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

    session_high = post_ib['high'].max()
    session_low = post_ib['low'].min()
    session_close = post_ib['close'].iloc[-1]

    bull_ext = max(0, (session_high - ib_high) / ib_range)
    bear_ext = max(0, (ib_low - session_low) / ib_range)

    # Final day type (based on close)
    if session_close > ib_high:
        ib_dir = 'BULL'
        ext = (session_close - ib_mid) / ib_range
    elif session_close < ib_low:
        ib_dir = 'BEAR'
        ext = (ib_mid - session_close) / ib_range
    else:
        ib_dir = 'INSIDE'
        ext = 0.0
    strength = classify_trend_strength(ext)
    day_type = classify_day_type(ib_high, ib_low, session_close, ib_dir, strength).value

    extensions.append({
        'session': str(session_date),
        'bull_ext': bull_ext,
        'bear_ext': bear_ext,
        'max_ext': max(bull_ext, bear_ext),
        'ib_range': ib_range,
        'day_type': day_type,
        'close_vs_ib': 'above' if session_close > ib_high else ('below' if session_close < ib_low else 'inside'),
    })

n = len(extensions)
print(f"\n  Total sessions analyzed: {n}")

# Extension frequency
thresholds = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
print(f"\n  BULL extension (above IBH):")
for t in thresholds:
    count = sum(1 for e in extensions if e['bull_ext'] >= t)
    print(f"    >= {t:.1f}x IB: {count:>4d} sessions ({count/n*100:>5.1f}%)")

print(f"\n  BEAR extension (below IBL):")
for t in thresholds:
    count = sum(1 for e in extensions if e['bear_ext'] >= t)
    print(f"    >= {t:.1f}x IB: {count:>4d} sessions ({count/n*100:>5.1f}%)")

print(f"\n  EITHER direction:")
for t in thresholds:
    count = sum(1 for e in extensions if e['max_ext'] >= t)
    print(f"    >= {t:.1f}x IB: {count:>4d} sessions ({count/n*100:>5.1f}%)")

# Final day type distribution
print(f"\n  Final day type (at session close):")
from collections import Counter
dt_counts = Counter(e['day_type'] for e in extensions)
for dt, cnt in sorted(dt_counts.items(), key=lambda x: -x[1]):
    print(f"    {dt:<20s}: {cnt:>4d} ({cnt/n*100:>5.1f}%)")

# Close position vs IB
print(f"\n  Session close position:")
for pos in ['above', 'inside', 'below']:
    cnt = sum(1 for e in extensions if e['close_vs_ib'] == pos)
    print(f"    {pos:>7s} IB: {cnt:>4d} ({cnt/n*100:>5.1f}%)")


# ================================================================
# TEST 2: Acceptance then what? (After 2+ closes above IBH)
# ================================================================
print(f"\n{'='*100}")
print("  TEST 2: AFTER ACCEPTANCE ABOVE IBH -- What actually happens?")
print("=" * 100)

accept_bull_events = []

for session_date in sessions:
    sdf = df[df['session_date'] == session_date]
    if len(sdf) < IB_BARS_1MIN:
        continue

    ib_df = sdf.head(IB_BARS_1MIN)
    ib_high = ib_df['high'].max()
    ib_low = ib_df['low'].min()
    ib_range = ib_high - ib_low
    ib_mid = (ib_high + ib_low) / 2
    if ib_range <= 0:
        continue

    post_ib = sdf.iloc[IB_BARS_1MIN:].reset_index(drop=True)
    if len(post_ib) < 20:
        continue

    consec = 0
    accepted = False
    accept_bar = -1
    accept_price = 0

    for i in range(len(post_ib)):
        bar = post_ib.iloc[i]
        if bar['close'] > ib_high:
            consec += 1
            if consec >= 2 and not accepted:
                accepted = True
                accept_bar = i
                accept_price = bar['close']
        else:
            consec = 0

    if not accepted:
        continue

    # From acceptance bar forward, track what happens
    remaining = post_ib.iloc[accept_bar:]
    if len(remaining) < 5:
        continue

    vwap_at_accept = remaining.iloc[0].get('vwap', ib_mid)
    if vwap_at_accept is None or np.isnan(vwap_at_accept):
        vwap_at_accept = ib_mid

    post_accept_high = remaining['high'].max()
    post_accept_low = remaining['low'].min()
    session_close = remaining['close'].iloc[-1]

    # Max extension from IBH after acceptance
    further_ext = (post_accept_high - ib_high) / ib_range

    # Did it come back inside IB?
    came_back_inside = post_accept_low <= ib_high
    came_back_to_mid = post_accept_low <= ib_mid
    came_back_to_ibl = post_accept_low <= ib_low

    # Where did it close?
    close_ext = (session_close - ib_high) / ib_range if session_close > ib_high else 0

    # MFE from different entry points
    # Entry A: at acceptance price (accept_price, the close of accept bar)
    mfe_from_accept = post_accept_high - accept_price
    mae_from_accept = accept_price - post_accept_low

    # Entry B: at IBH retest (if price comes back to IBH after accept)
    ibh_retest = False
    ibh_retest_bar = -1
    for i in range(1, len(remaining)):
        if remaining.iloc[i]['low'] <= ib_high + ib_range * 0.05:  # within 5% above IBH
            ibh_retest = True
            ibh_retest_bar = accept_bar + i
            break

    if ibh_retest:
        retest_remaining = post_ib.iloc[ibh_retest_bar:]
        mfe_from_retest = retest_remaining['high'].max() - ib_high
        mae_from_retest = ib_high - retest_remaining['low'].min()
    else:
        mfe_from_retest = 0
        mae_from_retest = 0

    # Simulate: LONG from IBH retest, stop just below IBH
    if ibh_retest:
        entry = ib_high + ib_range * 0.02  # enter just above IBH
        for stop_pct in [0.10, 0.15, 0.25]:
            stop = entry - ib_range * stop_pct
            for tgt_name, tgt_pct in [('0.5x', 0.5), ('1.0x', 1.0), ('1.5x', 1.5)]:
                target = entry + ib_range * tgt_pct
                hit_stop = False
                hit_target = False
                exit_price = None

                for j in range(ibh_retest_bar - accept_bar, len(remaining)):
                    fb = remaining.iloc[j]
                    if fb['low'] <= stop:
                        hit_stop = True
                        exit_price = stop
                        break
                    if fb['high'] >= target:
                        hit_target = True
                        exit_price = target
                        break

                if exit_price is None:
                    exit_price = remaining.iloc[-1]['close']

                pnl = (exit_price - entry) * point_value - slippage_cost
                accept_bull_events.append({
                    'session': str(session_date),
                    'entry': 'IBH_RETEST',
                    'stop_pct': stop_pct,
                    'tgt_name': tgt_name,
                    'pnl': pnl,
                    'win': 1 if pnl > 0 else 0,
                    'ib_range': ib_range,
                    'further_ext': further_ext,
                    'came_back_inside': came_back_inside,
                    'came_back_to_mid': came_back_to_mid,
                })

    # Also simulate LONG from VWAP pullback after acceptance
    # Find first bar where price is near VWAP and above IBH
    vwap_entry = False
    for i in range(1, len(remaining)):
        bar = remaining.iloc[i]
        vwap = bar.get('vwap', None)
        if vwap is None or np.isnan(vwap):
            continue
        if bar['close'] > vwap and abs(bar['close'] - vwap) < ib_range * 0.3:
            if bar['close'] > ib_high:
                vwap_entry = True
                vwap_entry_bar = accept_bar + i
                vwap_entry_price = bar['close']
                vwap_price = vwap
                break

    if vwap_entry:
        vwap_remaining = post_ib.iloc[vwap_entry_bar:]
        for stop_pct in [0.10, 0.15, 0.25]:
            stop = vwap_entry_price - ib_range * stop_pct
            for tgt_name, tgt_pct in [('0.5x', 0.5), ('1.0x', 1.0), ('1.5x', 1.5)]:
                target = vwap_entry_price + ib_range * tgt_pct
                hit_stop = False
                hit_target = False
                exit_price = None

                for _, fb in vwap_remaining.iterrows():
                    if fb['low'] <= stop:
                        exit_price = stop
                        break
                    if fb['high'] >= target:
                        exit_price = target
                        break

                if exit_price is None:
                    exit_price = vwap_remaining.iloc[-1]['close']

                pnl = (exit_price - vwap_entry_price) * point_value - slippage_cost
                accept_bull_events.append({
                    'session': str(session_date),
                    'entry': 'VWAP_PULLBACK',
                    'stop_pct': stop_pct,
                    'tgt_name': tgt_name,
                    'pnl': pnl,
                    'win': 1 if pnl > 0 else 0,
                    'ib_range': ib_range,
                    'further_ext': further_ext,
                    'came_back_inside': came_back_inside,
                    'came_back_to_mid': came_back_to_mid,
                })

# What happens after acceptance?
accept_sessions = []
for session_date in sessions:
    sdf = df[df['session_date'] == session_date]
    if len(sdf) < IB_BARS_1MIN:
        continue
    ib_df = sdf.head(IB_BARS_1MIN)
    ib_high = ib_df['high'].max()
    ib_low = ib_df['low'].min()
    ib_range = ib_high - ib_low
    if ib_range <= 0:
        continue
    post_ib = sdf.iloc[IB_BARS_1MIN:].reset_index(drop=True)
    if len(post_ib) < 20:
        continue

    consec = 0
    accepted = False
    for i in range(len(post_ib)):
        if post_ib.iloc[i]['close'] > ib_high:
            consec += 1
            if consec >= 2:
                accepted = True
                break
        else:
            consec = 0

    if accepted:
        session_close = post_ib['close'].iloc[-1]
        session_high = post_ib['high'].max()
        session_low = post_ib['low'].min()

        max_above = (session_high - ib_high) / ib_range
        came_back = session_low <= ib_high
        came_mid = session_low <= (ib_high + ib_low) / 2
        came_ibl = session_low <= ib_low
        closed_above = session_close > ib_high

        accept_sessions.append({
            'max_ext': max_above,
            'came_back_inside': came_back,
            'came_back_to_mid': came_mid,
            'came_back_to_ibl': came_ibl,
            'closed_above_ibh': closed_above,
            'ib_range': ib_range,
        })

na = len(accept_sessions)
print(f"\n  Sessions with bull acceptance (2+ closes above IBH): {na} / {len(extensions)} ({na/len(extensions)*100:.0f}%)")

if na > 0:
    print(f"\n  After acceptance above IBH, what happens?")
    print(f"    Max extension above IBH:")
    for t in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        cnt = sum(1 for s in accept_sessions if s['max_ext'] >= t)
        print(f"      >= {t:.1f}x IB: {cnt:>4d} ({cnt/na*100:>5.1f}%)")

    cb_inside = sum(1 for s in accept_sessions if s['came_back_inside'])
    cb_mid = sum(1 for s in accept_sessions if s['came_back_to_mid'])
    cb_ibl = sum(1 for s in accept_sessions if s['came_back_to_ibl'])
    closed_above = sum(1 for s in accept_sessions if s['closed_above_ibh'])

    print(f"\n    Came back inside IB (below IBH): {cb_inside:>4d} ({cb_inside/na*100:.0f}%)")
    print(f"    Came back to IB mid:             {cb_mid:>4d} ({cb_mid/na*100:.0f}%)")
    print(f"    Came back to IBL:                {cb_ibl:>4d} ({cb_ibl/na*100:.0f}%)")
    print(f"    Closed above IBH at EOD:         {closed_above:>4d} ({closed_above/na*100:.0f}%)")


# ================================================================
# TEST 3: Trend continuation from IBH retest -- Can it work with tight stops?
# ================================================================
print(f"\n{'='*100}")
print("  TEST 3: TREND CONTINUATION -- IBH retest & VWAP pullback with tight stops")
print("=" * 100)

for entry_type in ['IBH_RETEST', 'VWAP_PULLBACK']:
    subset = [e for e in accept_bull_events if e['entry'] == entry_type]
    if not subset:
        print(f"\n  {entry_type}: no events found")
        continue

    print(f"\n  {entry_type}:")
    print(f"  {'Stop':>6s} | {'Target':>8s} | {'Trades':>6s} | {'WR':>6s} | {'Net P&L':>10s} | {'Avg':>8s} | {'PF':>5s}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*5}")

    for stop_pct in [0.10, 0.15, 0.25]:
        for tgt_name in ['0.5x', '1.0x', '1.5x']:
            trades = [e for e in subset if e['stop_pct'] == stop_pct and e['tgt_name'] == tgt_name]
            if not trades:
                continue
            wins = sum(t['win'] for t in trades)
            total = sum(t['pnl'] for t in trades)
            avg = total / len(trades)
            wr = wins / len(trades) * 100

            win_pnls = [t['pnl'] for t in trades if t['pnl'] > 0]
            loss_pnls = [t['pnl'] for t in trades if t['pnl'] <= 0]
            gw = sum(win_pnls) if win_pnls else 0
            gl = abs(sum(loss_pnls)) if loss_pnls else 0.01
            pf = gw / gl

            marker = " <-- GOOD" if total > 500 else (" <-- BAD" if total < -500 else "")
            print(f"  {stop_pct:>5.0%}  | {tgt_name:>8s} | {len(trades):>6d} | {wr:>5.1f}% | ${total:>9,.0f} | ${avg:>7,.0f} | {pf:>5.2f}{marker}")


# ================================================================
# TEST 4: Same analysis for BEAR side
# ================================================================
print(f"\n{'='*100}")
print("  TEST 4: BEAR ACCEPTANCE -- Sessions with 3+ closes below IBL")
print("=" * 100)

accept_bear_sessions = []
for session_date in sessions:
    sdf = df[df['session_date'] == session_date]
    if len(sdf) < IB_BARS_1MIN:
        continue
    ib_df = sdf.head(IB_BARS_1MIN)
    ib_high = ib_df['high'].max()
    ib_low = ib_df['low'].min()
    ib_range = ib_high - ib_low
    if ib_range <= 0:
        continue
    post_ib = sdf.iloc[IB_BARS_1MIN:].reset_index(drop=True)
    if len(post_ib) < 20:
        continue

    consec = 0
    accepted = False
    for i in range(len(post_ib)):
        if post_ib.iloc[i]['close'] < ib_low:
            consec += 1
            if consec >= 3:
                accepted = True
                break
        else:
            consec = 0

    if accepted:
        session_close = post_ib['close'].iloc[-1]
        session_high = post_ib['high'].max()
        session_low = post_ib['low'].min()

        max_below = (ib_low - session_low) / ib_range
        came_back = session_high >= ib_low
        came_mid = session_high >= (ib_high + ib_low) / 2
        came_ibh = session_high >= ib_high
        closed_below = session_close < ib_low

        accept_bear_sessions.append({
            'max_ext': max_below,
            'came_back_inside': came_back,
            'came_back_to_mid': came_mid,
            'came_back_to_ibh': came_ibh,
            'closed_below_ibl': closed_below,
            'ib_range': ib_range,
        })

nb = len(accept_bear_sessions)
print(f"\n  Sessions with bear acceptance (3+ closes below IBL): {nb} / {len(extensions)} ({nb/len(extensions)*100:.0f}%)")

if nb > 0:
    print(f"\n  After acceptance below IBL, what happens?")
    print(f"    Max extension below IBL:")
    for t in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        cnt = sum(1 for s in accept_bear_sessions if s['max_ext'] >= t)
        print(f"      >= {t:.1f}x IB: {cnt:>4d} ({cnt/nb*100:>5.1f}%)")

    cb_inside = sum(1 for s in accept_bear_sessions if s['came_back_inside'])
    cb_mid = sum(1 for s in accept_bear_sessions if s['came_back_to_mid'])
    cb_ibh = sum(1 for s in accept_bear_sessions if s['came_back_to_ibh'])
    closed_below = sum(1 for s in accept_bear_sessions if s['closed_below_ibl'])

    print(f"\n    Came back inside IB (above IBL): {cb_inside:>4d} ({cb_inside/nb*100:.0f}%)")
    print(f"    Came back to IB mid:             {cb_mid:>4d} ({cb_mid/nb*100:.0f}%)")
    print(f"    Came back to IBH:                {cb_ibh:>4d} ({cb_ibh/nb*100:.0f}%)")
    print(f"    Closed below IBL at EOD:         {closed_below:>4d} ({closed_below/nb*100:.0f}%)")


# ================================================================
# TEST 5: Extension distribution -- how far do real trend days go?
# ================================================================
print(f"\n{'='*100}")
print("  TEST 5: EXTENSION DISTRIBUTION -- How far do trend days actually extend?")
print("=" * 100)

# Only look at sessions that extended 0.5x+ in either direction
trend_sessions = [e for e in extensions if e['max_ext'] >= 0.5]
print(f"\n  Sessions with 0.5x+ extension: {len(trend_sessions)} ({len(trend_sessions)/n*100:.0f}%)")

if trend_sessions:
    bull_trends = [e for e in trend_sessions if e['bull_ext'] >= 0.5]
    bear_trends = [e for e in trend_sessions if e['bear_ext'] >= 0.5]

    print(f"\n  BULL trends (0.5x+ above IBH): {len(bull_trends)}")
    if bull_trends:
        exts = [e['bull_ext'] for e in bull_trends]
        ibs = [e['ib_range'] for e in bull_trends]
        print(f"    Extension: mean={np.mean(exts):.2f}x, med={np.median(exts):.2f}x, max={max(exts):.2f}x")
        print(f"    IB range:  mean={np.mean(ibs):.0f}, med={np.median(ibs):.0f}")
        print(f"    Day types: {Counter(e['day_type'] for e in bull_trends)}")

    print(f"\n  BEAR trends (0.5x+ below IBL): {len(bear_trends)}")
    if bear_trends:
        exts = [e['bear_ext'] for e in bear_trends]
        ibs = [e['ib_range'] for e in bear_trends]
        print(f"    Extension: mean={np.mean(exts):.2f}x, med={np.median(exts):.2f}x, max={max(exts):.2f}x")
        print(f"    IB range:  mean={np.mean(ibs):.0f}, med={np.median(ibs):.0f}")
        print(f"    Day types: {Counter(e['day_type'] for e in bear_trends)}")

    # IB range on trend days vs non-trend days
    non_trend = [e for e in extensions if e['max_ext'] < 0.5]
    print(f"\n  IB range comparison:")
    print(f"    Trend days (0.5x+ ext):  mean={np.mean([e['ib_range'] for e in trend_sessions]):.0f}, med={np.median([e['ib_range'] for e in trend_sessions]):.0f}")
    print(f"    Non-trend days (<0.5x):  mean={np.mean([e['ib_range'] for e in non_trend]):.0f}, med={np.median([e['ib_range'] for e in non_trend]):.0f}")
