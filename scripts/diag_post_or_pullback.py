"""
Diagnostic: Post-Opening-Range Pullback to IB Edge

The idea:
1. Opening range (9:30-10:30) establishes direction with a strong move
2. We already took OR Rev profits (or the OR move was directional, not a reversal)
3. After IB forms, price pulls back
4. Enter on pullback, targeting IBH (longs) or IBL (shorts)
5. NOT trying to continue beyond IB -- just retesting the level

This is the "second trade" after OR profits. Target is modest (IBH/IBL),
stop is tight (below pullback structure), and we avoid the continuation
trap that kills Trend Bull/Bear.

Also tests: does the pullback-to-IBH/IBL trade work better when coming
from an oversold/overbought condition (strong OR move that overextended)?
"""

import sys, warnings, io
import numpy as np
from collections import defaultdict

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from config.constants import IB_BARS_1MIN
from config.instruments import get_instrument

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
# CONFIG
# ================================================================
# OR period = first 60 bars (IB). We look at the OR move direction.
# "Strong OR" = price moved significantly during IB formation

# Pullback entry window: bar 60 (IB end) to bar 180 (~12:30 ET)
ENTRY_START_BAR = 0    # relative to post-IB (bar 0 = 10:30)
ENTRY_END_BAR = 180    # ~13:30 ET
MIN_IB_RANGE = 50      # Skip tiny IB days

# Pullback detection: price retraces toward VWAP or mid-IB from the OR direction
# For BULL OR (IB close near IBH): pullback = price drops toward VWAP/IBmid, then bounces
# For BEAR OR (IB close near IBL): pullback = price rises toward VWAP/IBmid, then drops

# Stop options (fraction of IB range below entry)
STOP_PCTS = [0.10, 0.15, 0.20, 0.25]

# ================================================================
# SCAN
# ================================================================

trades = []

for session_date in sessions:
    sdf = df[df['session_date'] == session_date]
    if len(sdf) < IB_BARS_1MIN:
        continue

    ib_df = sdf.head(IB_BARS_1MIN)
    ib_high = ib_df['high'].max()
    ib_low = ib_df['low'].min()
    ib_range = ib_high - ib_low
    ib_mid = (ib_high + ib_low) / 2
    if ib_range < MIN_IB_RANGE:
        continue

    # OR direction: where did price close at end of IB?
    ib_close = ib_df['close'].iloc[-1]
    ib_open = ib_df['open'].iloc[0]
    or_move = ib_close - ib_open  # positive = bullish OR
    or_move_pct = or_move / ib_range  # normalized

    # Classify OR direction
    # BULL OR: IB close in upper third of IB
    # BEAR OR: IB close in lower third of IB
    ib_position = (ib_close - ib_low) / ib_range  # 0 = at IBL, 1 = at IBH

    post_ib = sdf.iloc[IB_BARS_1MIN:].reset_index(drop=True)
    if len(post_ib) < 30:
        continue

    # ================================================================
    # SCENARIO A: BULL OR -> Pullback -> LONG targeting IBH
    # ================================================================
    # Conditions: IB close in upper 40% of IB range
    # Pullback: price drops to lower half of IB or near VWAP
    # Entry: when price bounces (close > prior close, delta > 0)
    # Target: IBH
    # Stop: below pullback low

    if ib_position >= 0.60:  # Bullish OR
        pullback_found = False
        pullback_low = ib_high
        pullback_low_bar = -1

        for i in range(min(ENTRY_END_BAR, len(post_ib))):
            bar = post_ib.iloc[i]
            close = bar['close']

            # Track pullback low (must be inside IB, not below IBL)
            if close < pullback_low and close >= ib_low:
                pullback_low = close
                pullback_low_bar = i

            # Detect bounce: price is pulling back toward lower IB zone
            # then starts bouncing (close > open, close > prior bar close)
            if pullback_low_bar >= 0 and i > pullback_low_bar:
                pb_depth = (ib_high - pullback_low) / ib_range  # how deep the pullback was

                # Pullback must be meaningful (at least 30% of IB from IBH)
                if pb_depth < 0.30:
                    continue

                # Bounce confirmation: current close > pullback low and > prior close
                prior_close = post_ib.iloc[i-1]['close'] if i > 0 else close
                delta = bar.get('delta', 0)
                if np.isnan(delta):
                    delta = 0

                if close > prior_close and close > pullback_low and delta > 0:
                    # Check VWAP position
                    vwap = bar.get('vwap', ib_mid)
                    if vwap is None or np.isnan(vwap):
                        vwap = ib_mid

                    # Entry is the bounce bar close
                    entry_price = close
                    target = ib_high

                    # Skip if already above target
                    if entry_price >= target:
                        continue

                    reward = target - entry_price

                    # Calculate distance from VWAP
                    vwap_dist_pct = (entry_price - vwap) / ib_range if ib_range > 0 else 0

                    # Near VWAP or below = better quality
                    near_vwap = abs(entry_price - vwap) < ib_range * 0.25

                    # Simulate with different stops
                    remaining = post_ib.iloc[i+1:]
                    if len(remaining) == 0:
                        continue

                    for stop_pct in STOP_PCTS:
                        stop = entry_price - ib_range * stop_pct
                        risk = entry_price - stop

                        if risk <= 0 or reward <= 0:
                            continue
                        rr = reward / risk

                        exit_price = None
                        exit_type = 'EOD'
                        for _, fb in remaining.iterrows():
                            if fb['low'] <= stop:
                                exit_price = stop
                                exit_type = 'STOP'
                                break
                            if fb['high'] >= target:
                                exit_price = target
                                exit_type = 'TARGET'
                                break
                        if exit_price is None:
                            exit_price = remaining.iloc[-1]['close']

                        pnl = (exit_price - entry_price) * point_value - slippage_cost

                        trades.append({
                            'session': str(session_date),
                            'scenario': 'BULL_OR_PB_LONG',
                            'direction': 'LONG',
                            'entry_price': entry_price,
                            'target': target,
                            'stop_pct': stop_pct,
                            'pnl': pnl,
                            'win': 1 if pnl > 0 else 0,
                            'exit_type': exit_type,
                            'rr_planned': rr,
                            'ib_range': ib_range,
                            'pb_depth': pb_depth,
                            'or_strength': or_move_pct,
                            'near_vwap': near_vwap,
                            'vwap_dist': vwap_dist_pct,
                            'entry_bar': i,
                            'ib_position': ib_position,
                        })

                    pullback_found = True
                    break  # Only take first pullback entry per session

    # ================================================================
    # SCENARIO B: BEAR OR -> Pullback -> SHORT targeting IBL
    # ================================================================
    if ib_position <= 0.40:  # Bearish OR
        pullback_found = False
        pullback_high = ib_low
        pullback_high_bar = -1

        for i in range(min(ENTRY_END_BAR, len(post_ib))):
            bar = post_ib.iloc[i]
            close = bar['close']

            if close > pullback_high and close <= ib_high:
                pullback_high = close
                pullback_high_bar = i

            if pullback_high_bar >= 0 and i > pullback_high_bar:
                pb_depth = (pullback_high - ib_low) / ib_range

                if pb_depth < 0.30:
                    continue

                prior_close = post_ib.iloc[i-1]['close'] if i > 0 else close
                delta = bar.get('delta', 0)
                if np.isnan(delta):
                    delta = 0

                if close < prior_close and close < pullback_high and delta < 0:
                    vwap = bar.get('vwap', ib_mid)
                    if vwap is None or np.isnan(vwap):
                        vwap = ib_mid

                    entry_price = close
                    target = ib_low

                    if entry_price <= target:
                        continue

                    reward = entry_price - target
                    near_vwap = abs(entry_price - vwap) < ib_range * 0.25
                    vwap_dist_pct = (vwap - entry_price) / ib_range if ib_range > 0 else 0

                    remaining = post_ib.iloc[i+1:]
                    if len(remaining) == 0:
                        continue

                    for stop_pct in STOP_PCTS:
                        stop = entry_price + ib_range * stop_pct
                        risk = stop - entry_price

                        if risk <= 0 or reward <= 0:
                            continue
                        rr = reward / risk

                        exit_price = None
                        exit_type = 'EOD'
                        for _, fb in remaining.iterrows():
                            if fb['high'] >= stop:
                                exit_price = stop
                                exit_type = 'STOP'
                                break
                            if fb['low'] <= target:
                                exit_price = target
                                exit_type = 'TARGET'
                                break
                        if exit_price is None:
                            exit_price = remaining.iloc[-1]['close']

                        pnl = (entry_price - exit_price) * point_value - slippage_cost

                        trades.append({
                            'session': str(session_date),
                            'scenario': 'BEAR_OR_PB_SHORT',
                            'direction': 'SHORT',
                            'entry_price': entry_price,
                            'target': target,
                            'stop_pct': stop_pct,
                            'pnl': pnl,
                            'win': 1 if pnl > 0 else 0,
                            'exit_type': exit_type,
                            'rr_planned': rr,
                            'ib_range': ib_range,
                            'pb_depth': pb_depth,
                            'or_strength': abs(or_move_pct),
                            'near_vwap': near_vwap,
                            'vwap_dist': vwap_dist_pct,
                            'entry_bar': i,
                            'ib_position': ib_position,
                        })

                    pullback_found = True
                    break


# ================================================================
# REPORT
# ================================================================
print("=" * 120)
print("  POST-OR PULLBACK TO IB EDGE -- Results")
print("=" * 120)

for scenario in ['BULL_OR_PB_LONG', 'BEAR_OR_PB_SHORT']:
    subset = [t for t in trades if t['scenario'] == scenario]
    if not subset:
        print(f"\n  {scenario}: 0 trades")
        continue

    direction = subset[0]['direction']
    print(f"\n{'─'*120}")
    print(f"  {scenario} ({direction}) -- {len(subset)//len(STOP_PCTS)} unique sessions")
    print(f"{'─'*120}")

    # Overall results by stop level
    print(f"\n  {'Stop':>6s} | {'Trades':>6s} | {'WR':>6s} | {'Net P&L':>10s} | {'Avg':>8s} | {'PF':>5s} | {'Avg RR':>7s} | {'Tgt Hits':>8s}")
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*5}-+-{'-'*7}-+-{'-'*8}")

    for stop_pct in STOP_PCTS:
        st = [t for t in subset if t['stop_pct'] == stop_pct]
        wins = sum(t['win'] for t in st)
        total = sum(t['pnl'] for t in st)
        avg = total / len(st) if st else 0
        wr = wins / len(st) * 100 if st else 0
        avg_rr = np.mean([t['rr_planned'] for t in st])
        tgt_hits = sum(1 for t in st if t['exit_type'] == 'TARGET')

        win_pnls = [t['pnl'] for t in st if t['pnl'] > 0]
        loss_pnls = [t['pnl'] for t in st if t['pnl'] <= 0]
        gw = sum(win_pnls) if win_pnls else 0
        gl = abs(sum(loss_pnls)) if loss_pnls else 0.01
        pf = gw / gl

        marker = " <-- BEST" if total > 1000 else (" <-- GOOD" if total > 500 else (" <-- BAD" if total < -500 else ""))
        print(f"  {stop_pct:>5.0%}  | {len(st):>6d} | {wr:>5.1f}% | ${total:>9,.0f} | ${avg:>7,.0f} | {pf:>5.2f} | {avg_rr:>6.1f}R |  {tgt_hits:>4d}{marker}")

    # ================================================================
    # Filter: NEAR VWAP entries only
    # ================================================================
    print(f"\n  --- FILTER: Near VWAP entries only (within 25% IB of VWAP) ---")
    print(f"  {'Stop':>6s} | {'Trades':>6s} | {'WR':>6s} | {'Net P&L':>10s} | {'Avg':>8s} | {'PF':>5s}")
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*5}")

    for stop_pct in STOP_PCTS:
        st = [t for t in subset if t['stop_pct'] == stop_pct and t['near_vwap']]
        if not st:
            continue
        wins = sum(t['win'] for t in st)
        total = sum(t['pnl'] for t in st)
        avg = total / len(st) if st else 0
        wr = wins / len(st) * 100 if st else 0
        gw = sum(t['pnl'] for t in st if t['pnl'] > 0) or 0
        gl = abs(sum(t['pnl'] for t in st if t['pnl'] <= 0)) or 0.01
        pf = gw / gl
        marker = " <-- BEST" if total > 1000 else (" <-- GOOD" if total > 500 else (" <-- BAD" if total < -500 else ""))
        print(f"  {stop_pct:>5.0%}  | {len(st):>6d} | {wr:>5.1f}% | ${total:>9,.0f} | ${avg:>7,.0f} | {pf:>5.2f}{marker}")

    # ================================================================
    # Filter: Strong OR only (top 50% by OR move strength)
    # ================================================================
    or_strengths = sorted([t['or_strength'] for t in subset if t['stop_pct'] == STOP_PCTS[0]])
    if or_strengths:
        med_strength = np.median(or_strengths)
        print(f"\n  --- FILTER: Strong OR only (OR move >= {med_strength:.2f}x IB, top 50%) ---")
        print(f"  {'Stop':>6s} | {'Trades':>6s} | {'WR':>6s} | {'Net P&L':>10s} | {'Avg':>8s} | {'PF':>5s}")
        print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*5}")

        for stop_pct in STOP_PCTS:
            st = [t for t in subset if t['stop_pct'] == stop_pct and t['or_strength'] >= med_strength]
            if not st:
                continue
            wins = sum(t['win'] for t in st)
            total = sum(t['pnl'] for t in st)
            avg = total / len(st) if st else 0
            wr = wins / len(st) * 100 if st else 0
            gw = sum(t['pnl'] for t in st if t['pnl'] > 0) or 0
            gl = abs(sum(t['pnl'] for t in st if t['pnl'] <= 0)) or 0.01
            pf = gw / gl
            marker = " <-- BEST" if total > 1000 else (" <-- GOOD" if total > 500 else (" <-- BAD" if total < -500 else ""))
            print(f"  {stop_pct:>5.0%}  | {len(st):>6d} | {wr:>5.1f}% | ${total:>9,.0f} | ${avg:>7,.0f} | {pf:>5.2f}{marker}")

    # ================================================================
    # Filter: Deep pullback only (>= 50% IB retrace)
    # ================================================================
    print(f"\n  --- FILTER: Deep pullback only (>= 50% IB retrace) ---")
    print(f"  {'Stop':>6s} | {'Trades':>6s} | {'WR':>6s} | {'Net P&L':>10s} | {'Avg':>8s} | {'PF':>5s}")
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*5}")

    for stop_pct in STOP_PCTS:
        st = [t for t in subset if t['stop_pct'] == stop_pct and t['pb_depth'] >= 0.50]
        if not st:
            continue
        wins = sum(t['win'] for t in st)
        total = sum(t['pnl'] for t in st)
        avg = total / len(st) if st else 0
        wr = wins / len(st) * 100 if st else 0
        gw = sum(t['pnl'] for t in st if t['pnl'] > 0) or 0
        gl = abs(sum(t['pnl'] for t in st if t['pnl'] <= 0)) or 0.01
        pf = gw / gl
        marker = " <-- BEST" if total > 1000 else (" <-- GOOD" if total > 500 else (" <-- BAD" if total < -500 else ""))
        print(f"  {stop_pct:>5.0%}  | {len(st):>6d} | {wr:>5.1f}% | ${total:>9,.0f} | ${avg:>7,.0f} | {pf:>5.2f}{marker}")

    # ================================================================
    # Filter: COMBO -- strong OR + near VWAP + deep pullback
    # ================================================================
    if or_strengths:
        print(f"\n  --- FILTER: COMBO (strong OR + near VWAP + deep pullback >= 50%) ---")
        print(f"  {'Stop':>6s} | {'Trades':>6s} | {'WR':>6s} | {'Net P&L':>10s} | {'Avg':>8s} | {'PF':>5s}")
        print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*5}")

        for stop_pct in STOP_PCTS:
            st = [t for t in subset if t['stop_pct'] == stop_pct
                  and t['or_strength'] >= med_strength
                  and t['near_vwap']
                  and t['pb_depth'] >= 0.50]
            if not st:
                print(f"  {stop_pct:>5.0%}  | {0:>6d} | {'--':>6s} |        -- |      -- |    --")
                continue
            wins = sum(t['win'] for t in st)
            total = sum(t['pnl'] for t in st)
            avg = total / len(st) if st else 0
            wr = wins / len(st) * 100 if st else 0
            gw = sum(t['pnl'] for t in st if t['pnl'] > 0) or 0
            gl = abs(sum(t['pnl'] for t in st if t['pnl'] <= 0)) or 0.01
            pf = gw / gl
            marker = " <-- BEST" if total > 1000 else (" <-- GOOD" if total > 500 else (" <-- BAD" if total < -500 else ""))
            print(f"  {stop_pct:>5.0%}  | {len(st):>6d} | {wr:>5.1f}% | ${total:>9,.0f} | ${avg:>7,.0f} | {pf:>5.2f}{marker}")


# ================================================================
# PULLBACK DEPTH DISTRIBUTION
# ================================================================
print(f"\n{'='*120}")
print("  PULLBACK DEPTH DISTRIBUTION")
print("=" * 120)

for scenario in ['BULL_OR_PB_LONG', 'BEAR_OR_PB_SHORT']:
    subset = [t for t in trades if t['scenario'] == scenario and t['stop_pct'] == STOP_PCTS[0]]
    if not subset:
        continue

    depths = [t['pb_depth'] for t in subset]
    print(f"\n  {scenario} ({len(subset)} trades):")
    print(f"    Pullback depth: mean={np.mean(depths):.2f}x IB, med={np.median(depths):.2f}x IB")
    print(f"    P25={np.percentile(depths, 25):.2f}, P75={np.percentile(depths, 75):.2f}")

    for threshold in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        cnt = sum(1 for d in depths if d >= threshold)
        print(f"    >= {threshold:.0%} retrace: {cnt} trades ({cnt/len(depths)*100:.0f}%)")

    # Entry bar distribution (when does the pullback entry fire?)
    bars = [t['entry_bar'] for t in subset]
    print(f"\n    Entry timing (bars post-IB):")
    print(f"    Mean: bar {np.mean(bars):.0f} (~{10.5 + np.mean(bars)/60:.1f}h ET)")
    print(f"    Med:  bar {np.median(bars):.0f} (~{10.5 + np.median(bars)/60:.1f}h ET)")
    print(f"    P25:  bar {np.percentile(bars, 25):.0f}")
    print(f"    P75:  bar {np.percentile(bars, 75):.0f}")
