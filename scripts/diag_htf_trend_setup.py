"""
Diagnostic: HTF Confluence Trend Day Setup

The user's thesis for REAL trend days:
1. Oversold on HTF (multi-day selloff, RSI low, price below prior VA)
2. Sweeps London low or overnight low (liquidity grab)
3. Strong bullish IB (OR establishes direction)
4. Pullback to VWAP / 15m FVG zone after IB
5. Target: IBH (realistic), NOT 2x extension
6. Narrow IB = possible extension; Wide IB = poke & fail = P-day

Inverse for bear:
1. Overbought, price above prior VA
2. Sweeps London high
3. Bearish IB
4. Pullback to VWAP from below
5. Target: IBL

We have all the HTF data in session_context:
- overnight_high/low, london_high/low, asia_high/low
- pdh, pdl, pdc (prior day RTH)
- prior_vp_vah, prior_vp_val, prior_vp_poc
- rsi14 (intraday, but we can check multi-day context)
"""

import sys, warnings, io
import numpy as np
from datetime import time as dtime, timedelta

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from config.constants import IB_BARS_1MIN
from config.instruments import get_instrument
import pandas as pd

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
# COMPUTE SESSION-LEVEL HTF CONTEXT
# ================================================================

session_data = []

# Track rolling closes for multi-day trend detection
close_history = []  # (session_date, close)

for session_date in sessions:
    sdf = df[df['session_date'] == session_date]
    if len(sdf) < IB_BARS_1MIN:
        continue

    session_str = str(session_date)

    ib_df = sdf.head(IB_BARS_1MIN)
    ib_high = ib_df['high'].max()
    ib_low = ib_df['low'].min()
    ib_range = ib_high - ib_low
    ib_mid = (ib_high + ib_low) / 2
    if ib_range <= 0:
        continue

    ib_open = ib_df['open'].iloc[0]
    ib_close = ib_df['close'].iloc[-1]
    ib_position = (ib_close - ib_low) / ib_range

    post_ib = sdf.iloc[IB_BARS_1MIN:].reset_index(drop=True)
    if len(post_ib) < 20:
        continue

    session_high = sdf['high'].max()
    session_low = sdf['low'].min()
    session_close = sdf['close'].iloc[-1]

    bull_ext = max(0, (session_high - ib_high) / ib_range)
    bear_ext = max(0, (ib_low - session_low) / ib_range)

    # --- HTF CONTEXT ---
    # Overnight / London / Asia levels
    sd = pd.Timestamp(session_str)
    prev_day = sd - timedelta(days=1)
    if prev_day.weekday() == 5:
        prev_day -= timedelta(days=1)
    elif prev_day.weekday() == 6:
        prev_day -= timedelta(days=2)

    ts = full_df['timestamp']

    # Overnight
    on_mask = (ts >= prev_day + timedelta(hours=18)) & (ts < sd + timedelta(hours=9, minutes=30))
    on_bars = full_df[on_mask]
    on_high = on_bars['high'].max() if len(on_bars) > 0 else None
    on_low = on_bars['low'].min() if len(on_bars) > 0 else None

    # London
    london_mask = (ts >= sd + timedelta(hours=2)) & (ts < sd + timedelta(hours=5))
    london = full_df[london_mask]
    london_high = london['high'].max() if len(london) > 0 else None
    london_low = london['low'].min() if len(london) > 0 else None

    # Prior day RTH
    prior_mask = (ts >= prev_day + timedelta(hours=9, minutes=30)) & (ts <= prev_day + timedelta(hours=16))
    prior_rth = full_df[prior_mask]
    pdh = prior_rth['high'].max() if len(prior_rth) > 0 else None
    pdl = prior_rth['low'].min() if len(prior_rth) > 0 else None
    pdc = prior_rth.iloc[-1]['close'] if len(prior_rth) > 0 else None

    # Multi-day context: is this session coming from oversold/overbought?
    # Use: did prior 2-3 sessions close consecutively lower (oversold) or higher (overbought)?
    consec_down = 0
    consec_up = 0
    if len(close_history) >= 2:
        for i in range(len(close_history)-1, 0, -1):
            if close_history[i][1] < close_history[i-1][1]:
                consec_down += 1
            else:
                break
        for i in range(len(close_history)-1, 0, -1):
            if close_history[i][1] > close_history[i-1][1]:
                consec_up += 1
            else:
                break

    # Sweep detection: did IB low sweep London/overnight low?
    swept_london_low = london_low is not None and ib_low <= london_low
    swept_london_high = london_high is not None and ib_high >= london_high
    swept_on_low = on_low is not None and ib_low <= on_low
    swept_on_high = on_high is not None and ib_high >= on_high
    swept_pdl = pdl is not None and ib_low <= pdl
    swept_pdh = pdh is not None and ib_high >= pdh

    # Open relative to prior day levels
    gap_from_pdc = (ib_open - pdc) if pdc else 0

    # RSI from IB end
    rsi = ib_df['rsi14'].iloc[-1] if 'rsi14' in ib_df.columns else None
    if rsi is not None and np.isnan(rsi):
        rsi = None

    close_history.append((session_str, session_close))

    rec = {
        'session': session_str,
        'ib_high': ib_high, 'ib_low': ib_low, 'ib_range': ib_range,
        'ib_mid': ib_mid, 'ib_position': ib_position,
        'ib_open': ib_open, 'ib_close': ib_close,
        'session_high': session_high, 'session_low': session_low,
        'session_close': session_close,
        'bull_ext': bull_ext, 'bear_ext': bear_ext,
        'on_high': on_high, 'on_low': on_low,
        'london_high': london_high, 'london_low': london_low,
        'pdh': pdh, 'pdl': pdl, 'pdc': pdc,
        'swept_london_low': swept_london_low,
        'swept_london_high': swept_london_high,
        'swept_on_low': swept_on_low,
        'swept_on_high': swept_on_high,
        'swept_pdl': swept_pdl,
        'swept_pdh': swept_pdh,
        'consec_down': consec_down,
        'consec_up': consec_up,
        'rsi': rsi,
        'gap_from_pdc': gap_from_pdc,
    }

    # --- SIMULATE PULLBACK ENTRY ---
    # For bullish setup: after strong bullish IB, pullback, long to IBH
    # For bearish setup: after strong bearish IB, pullback, short to IBL

    for setup_dir in ['BULL', 'BEAR']:
        if setup_dir == 'BULL' and ib_position < 0.55:
            continue
        if setup_dir == 'BEAR' and ib_position > 0.45:
            continue

        target = ib_high if setup_dir == 'BULL' else ib_low

        # Find pullback and bounce
        best_pullback = None
        if setup_dir == 'BULL':
            pb_extreme = ib_high
            for i in range(min(180, len(post_ib))):
                bar = post_ib.iloc[i]
                close = bar['close']
                if close < pb_extreme:
                    pb_extreme = close
                pb_depth = (ib_high - pb_extreme) / ib_range if pb_extreme < ib_high else 0

                if pb_depth >= 0.25 and i > 0:
                    prior_c = post_ib.iloc[i-1]['close']
                    delta = bar.get('delta', 0)
                    if np.isnan(delta): delta = 0
                    vwap = bar.get('vwap', ib_mid)
                    if vwap is None or np.isnan(vwap): vwap = ib_mid

                    # Check for 15m FVG at entry
                    has_fvg = bar.get('fvg_bull_15m', False)
                    if has_fvg is None or (isinstance(has_fvg, float) and np.isnan(has_fvg)):
                        has_fvg = False

                    if close > prior_c and delta > 0 and close < target:
                        near_vwap = abs(close - vwap) < ib_range * 0.25
                        best_pullback = {
                            'bar_idx': i,
                            'entry_price': close,
                            'pb_depth': pb_depth,
                            'near_vwap': near_vwap,
                            'has_fvg': bool(has_fvg),
                            'vwap': vwap,
                        }
                        break
        else:  # BEAR
            pb_extreme = ib_low
            for i in range(min(180, len(post_ib))):
                bar = post_ib.iloc[i]
                close = bar['close']
                if close > pb_extreme:
                    pb_extreme = close
                pb_depth = (pb_extreme - ib_low) / ib_range if pb_extreme > ib_low else 0

                if pb_depth >= 0.25 and i > 0:
                    prior_c = post_ib.iloc[i-1]['close']
                    delta = bar.get('delta', 0)
                    if np.isnan(delta): delta = 0
                    vwap = bar.get('vwap', ib_mid)
                    if vwap is None or np.isnan(vwap): vwap = ib_mid

                    has_fvg = bar.get('fvg_bear_15m', False)
                    if has_fvg is None or (isinstance(has_fvg, float) and np.isnan(has_fvg)):
                        has_fvg = False

                    if close < prior_c and delta < 0 and close > target:
                        near_vwap = abs(close - vwap) < ib_range * 0.25
                        best_pullback = {
                            'bar_idx': i,
                            'entry_price': close,
                            'pb_depth': pb_depth,
                            'near_vwap': near_vwap,
                            'has_fvg': bool(has_fvg),
                            'vwap': vwap,
                        }
                        break

        if best_pullback is None:
            continue

        entry = best_pullback['entry_price']
        remaining = post_ib.iloc[best_pullback['bar_idx']+1:]
        if len(remaining) == 0:
            continue

        reward = abs(target - entry)
        if reward <= 0:
            continue

        for stop_pct in [0.15, 0.20, 0.25]:
            if setup_dir == 'BULL':
                stop = entry - ib_range * stop_pct
            else:
                stop = entry + ib_range * stop_pct
            risk = ib_range * stop_pct
            rr = reward / risk if risk > 0 else 0

            exit_price = None
            exit_type = 'EOD'
            for _, fb in remaining.iterrows():
                if setup_dir == 'BULL':
                    if fb['low'] <= stop:
                        exit_price = stop; exit_type = 'STOP'; break
                    if fb['high'] >= target:
                        exit_price = target; exit_type = 'TARGET'; break
                else:
                    if fb['high'] >= stop:
                        exit_price = stop; exit_type = 'STOP'; break
                    if fb['low'] <= target:
                        exit_price = target; exit_type = 'TARGET'; break

            if exit_price is None:
                exit_price = remaining.iloc[-1]['close']

            if setup_dir == 'BULL':
                pnl = (exit_price - entry) * point_value - slippage_cost
            else:
                pnl = (entry - exit_price) * point_value - slippage_cost

            trade = dict(rec)
            trade.update({
                'setup_dir': setup_dir,
                'stop_pct': stop_pct,
                'entry_price': entry,
                'pnl': pnl,
                'win': 1 if pnl > 0 else 0,
                'exit_type': exit_type,
                'rr': rr,
                'pb_depth': best_pullback['pb_depth'],
                'near_vwap': best_pullback['near_vwap'],
                'has_fvg': best_pullback['has_fvg'],
            })
            session_data.append(trade)


# ================================================================
# REPORT
# ================================================================
print("=" * 120)
print("  HTF CONFLUENCE TREND SETUP -- Results")
print("=" * 120)

for setup_dir in ['BULL', 'BEAR']:
    all_trades = [t for t in session_data if t['setup_dir'] == setup_dir]
    if not all_trades:
        print(f"\n  {setup_dir}: 0 trades")
        continue

    direction = 'LONG' if setup_dir == 'BULL' else 'SHORT'

    print(f"\n{'─'*120}")
    print(f"  {setup_dir} SETUP ({direction}) -- {len(all_trades)//3} sessions with pullback entry")
    print(f"{'─'*120}")

    # Baseline (no HTF filter)
    print(f"\n  BASELINE (no HTF filter):")
    print(f"  {'Stop':>6s} | {'Trades':>6s} | {'WR':>6s} | {'Net P&L':>10s} | {'Avg':>8s} | {'PF':>5s}")
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*5}")
    for sp in [0.15, 0.20, 0.25]:
        st = [t for t in all_trades if t['stop_pct'] == sp]
        w = sum(t['win'] for t in st); total = sum(t['pnl'] for t in st)
        wr = w/len(st)*100; avg = total/len(st)
        gw = sum(t['pnl'] for t in st if t['pnl']>0) or 0
        gl = abs(sum(t['pnl'] for t in st if t['pnl']<=0)) or 0.01
        print(f"  {sp:>5.0%}  | {len(st):>6d} | {wr:>5.1f}% | ${total:>9,.0f} | ${avg:>7,.0f} | {gw/gl:>5.2f}")

    # ================================================================
    # HTF FILTERS
    # ================================================================
    filters = {}

    if setup_dir == 'BULL':
        filters['Swept London/ON Low'] = lambda t: t['swept_london_low'] or t['swept_on_low']
        filters['Swept PDL'] = lambda t: t['swept_pdl']
        filters['Consec Down >= 2'] = lambda t: t['consec_down'] >= 2
        filters['RSI < 45 at IB'] = lambda t: t['rsi'] is not None and t['rsi'] < 45
        filters['Near VWAP entry'] = lambda t: t['near_vwap']
        filters['Has 15m FVG'] = lambda t: t['has_fvg']
        filters['Any Sweep (London/ON/PD Low)'] = lambda t: t['swept_london_low'] or t['swept_on_low'] or t['swept_pdl']
        filters['Sweep + Consec Down'] = lambda t: (t['swept_london_low'] or t['swept_on_low'] or t['swept_pdl']) and t['consec_down'] >= 1
        filters['Narrow IB (<150)'] = lambda t: t['ib_range'] < 150
        filters['Wide IB (>=200)'] = lambda t: t['ib_range'] >= 200
    else:
        filters['Swept London/ON High'] = lambda t: t['swept_london_high'] or t['swept_on_high']
        filters['Swept PDH'] = lambda t: t['swept_pdh']
        filters['Consec Up >= 2'] = lambda t: t['consec_up'] >= 2
        filters['RSI > 55 at IB'] = lambda t: t['rsi'] is not None and t['rsi'] > 55
        filters['Near VWAP entry'] = lambda t: t['near_vwap']
        filters['Has 15m FVG'] = lambda t: t['has_fvg']
        filters['Any Sweep (London/ON/PD High)'] = lambda t: t['swept_london_high'] or t['swept_on_high'] or t['swept_pdh']
        filters['Sweep + Consec Up'] = lambda t: (t['swept_london_high'] or t['swept_on_high'] or t['swept_pdh']) and t['consec_up'] >= 1
        filters['Narrow IB (<150)'] = lambda t: t['ib_range'] < 150
        filters['Wide IB (>=200)'] = lambda t: t['ib_range'] >= 200

    # Best stop level for comparison
    best_stop = 0.20

    print(f"\n  HTF FILTER IMPACT (stop={best_stop:.0%}):")
    print(f"  {'Filter':<35s} | {'Trades':>6s} | {'WR':>6s} | {'Net P&L':>10s} | {'Avg':>8s} | {'PF':>5s}")
    print(f"  {'-'*35}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*5}")

    for fname, ffunc in filters.items():
        st = [t for t in all_trades if t['stop_pct'] == best_stop and ffunc(t)]
        if len(st) < 3:
            print(f"  {fname:<35s} | {len(st):>6d} | {'--':>6s} | {'--':>10s} | {'--':>8s} | {'--':>5s}")
            continue
        w = sum(t['win'] for t in st); total = sum(t['pnl'] for t in st)
        wr = w/len(st)*100; avg = total/len(st)
        gw = sum(t['pnl'] for t in st if t['pnl']>0) or 0
        gl = abs(sum(t['pnl'] for t in st if t['pnl']<=0)) or 0.01
        marker = " <-- BEST" if total > 1000 else (" <-- GOOD" if total > 500 else (" <-- BAD" if total < -500 else ""))
        print(f"  {fname:<35s} | {len(st):>6d} | {wr:>5.1f}% | ${total:>9,.0f} | ${avg:>7,.0f} | {gw/gl:>5.2f}{marker}")

    # ================================================================
    # COMBO FILTERS
    # ================================================================
    print(f"\n  COMBO FILTERS (stop={best_stop:.0%}):")
    print(f"  {'Combo':<45s} | {'Trades':>6s} | {'WR':>6s} | {'Net P&L':>10s} | {'Avg':>8s} | {'PF':>5s}")
    print(f"  {'-'*45}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*5}")

    if setup_dir == 'BULL':
        combos = {
            'Sweep Low + Near VWAP': lambda t: (t['swept_london_low'] or t['swept_on_low'] or t['swept_pdl']) and t['near_vwap'],
            'Sweep Low + Narrow IB': lambda t: (t['swept_london_low'] or t['swept_on_low'] or t['swept_pdl']) and t['ib_range'] < 150,
            'Consec Down + Near VWAP': lambda t: t['consec_down'] >= 1 and t['near_vwap'],
            'Consec Down + Sweep + VWAP': lambda t: t['consec_down'] >= 1 and (t['swept_london_low'] or t['swept_on_low'] or t['swept_pdl']) and t['near_vwap'],
            'Sweep Low + FVG': lambda t: (t['swept_london_low'] or t['swept_on_low'] or t['swept_pdl']) and t['has_fvg'],
            'Any Sweep + Narrow IB + VWAP': lambda t: (t['swept_london_low'] or t['swept_on_low'] or t['swept_pdl']) and t['ib_range'] < 150 and t['near_vwap'],
        }
    else:
        combos = {
            'Sweep High + Near VWAP': lambda t: (t['swept_london_high'] or t['swept_on_high'] or t['swept_pdh']) and t['near_vwap'],
            'Sweep High + Narrow IB': lambda t: (t['swept_london_high'] or t['swept_on_high'] or t['swept_pdh']) and t['ib_range'] < 150,
            'Consec Up + Near VWAP': lambda t: t['consec_up'] >= 1 and t['near_vwap'],
            'Consec Up + Sweep + VWAP': lambda t: t['consec_up'] >= 1 and (t['swept_london_high'] or t['swept_on_high'] or t['swept_pdh']) and t['near_vwap'],
            'Sweep High + FVG': lambda t: (t['swept_london_high'] or t['swept_on_high'] or t['swept_pdh']) and t['has_fvg'],
            'Any Sweep + Narrow IB + VWAP': lambda t: (t['swept_london_high'] or t['swept_on_high'] or t['swept_pdh']) and t['ib_range'] < 150 and t['near_vwap'],
        }

    for cname, cfunc in combos.items():
        st = [t for t in all_trades if t['stop_pct'] == best_stop and cfunc(t)]
        if len(st) < 3:
            print(f"  {cname:<45s} | {len(st):>6d} | {'--':>6s} | {'--':>10s} | {'--':>8s} | {'--':>5s}")
            continue
        w = sum(t['win'] for t in st); total = sum(t['pnl'] for t in st)
        wr = w/len(st)*100; avg = total/len(st)
        gw = sum(t['pnl'] for t in st if t['pnl']>0) or 0
        gl = abs(sum(t['pnl'] for t in st if t['pnl']<=0)) or 0.01
        marker = " <-- BEST" if total > 1000 else (" <-- GOOD" if total > 500 else (" <-- BAD" if total < -500 else ""))
        print(f"  {cname:<45s} | {len(st):>6d} | {wr:>5.1f}% | ${total:>9,.0f} | ${avg:>7,.0f} | {gw/gl:>5.2f}{marker}")

    # IB range impact
    print(f"\n  IB RANGE IMPACT (stop={best_stop:.0%}):")
    for lo, hi, label in [(0, 120, '<120 (narrow)'), (120, 180, '120-180 (med)'), (180, 250, '180-250 (wide)'), (250, 999, '250+ (very wide)')]:
        st = [t for t in all_trades if t['stop_pct'] == best_stop and lo <= t['ib_range'] < hi]
        if len(st) < 3:
            print(f"    {label:<25s}: {len(st)} trades (too few)")
            continue
        w = sum(t['win'] for t in st); total = sum(t['pnl'] for t in st)
        wr = w/len(st)*100
        gw = sum(t['pnl'] for t in st if t['pnl']>0) or 0
        gl = abs(sum(t['pnl'] for t in st if t['pnl']<=0)) or 0.01
        print(f"    {label:<25s}: {len(st):>3d}t  {wr:>5.1f}% WR  ${total:>8,.0f}  PF={gw/gl:.2f}")
