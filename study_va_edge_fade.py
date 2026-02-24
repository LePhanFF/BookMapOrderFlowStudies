"""
VA Edge Fade Study — Fading Failed Breakouts at Value Area Boundaries
=====================================================================

SETUP: Price approaches VA edge (VAH or VAL), pokes outside, FAILS → fade it.
This is NOT the 80P rule (opens outside VA). This is:
  - Price is near/inside VA, pushes to test the boundary
  - Pokes outside (sweep), then shows rejection (inversion candle)
  - Fade the failed breakout back into VA

TRADE 1 — First Touch (Sweep & Fail):
  - First time price pokes outside VA edge
  - Entry: 5-min inversion candle OR 2x 5-min close back inside VA
  - Stop: 2 ATR
  - Target: Shallow trail (0.2 to 0.5 ATR), POC, mid-VA, opposite VA

TRADE 2 — Second Test (Retest Failure):
  - Second rejection at the same VA edge after an initial poke
  - Entry models tested:
    A) 5-min inversion candle (single candle reversal)
    B) 2x 5-min candles closing back inside VA
    C) Limit at sweep extreme (double top/bottom pattern)
    D) Limit at VA edge (exact edge entry on retest)
  - Stop: 2 ATR
  - Targets: 0.2 ATR trail, 0.5 ATR trail, 1R, 2R, POC, mid-VA

STOP TYPES TESTED:
  - 2 ATR (primary focus)
  - 80-point fixed
  - Swing point
  - VA edge + 10pt buffer

DETECTION LOGIC:
  Session opens INSIDE prior VA (or opens outside but trades back in early).
  We watch for price to approach and TEST the VA edge, poke outside, then fail.
  The "poke outside" must be a wick/sweep, not a sustained breakout.
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
from indicators.value_area import compute_session_value_areas

INSTRUMENT = get_instrument('MNQ')
TICK_VALUE = INSTRUMENT.tick_value / INSTRUMENT.tick_size  # $2/pt for MNQ
SLIPPAGE_PTS = 0.50
COMMISSION = 1.24
CONTRACTS = 5
MIN_VA_WIDTH = 25.0
ENTRY_CUTOFF = time(13, 0)
EOD_EXIT = time(15, 30)

# ============================================================================
# DATA
# ============================================================================
print("Loading data...")
df_raw = load_csv('NQ')
df_rth = filter_rth(df_raw)
df_rth = compute_all_features(df_rth)

if 'session_date' not in df_rth.columns:
    df_rth['session_date'] = df_rth['timestamp'].dt.date

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

print("Computing ETH Value Areas...")
eth_va = compute_session_value_areas(df_full, tick_size=0.25, va_percent=0.70)


# ============================================================================
# HELPERS
# ============================================================================
def aggregate_bars(bars_df, period_min):
    """Aggregate 1-min bars into N-min OHLCV bars."""
    agg_bars = []
    for start in range(0, len(bars_df), period_min):
        end = min(start + period_min, len(bars_df))
        chunk = bars_df.iloc[start:end]
        if len(chunk) == 0:
            continue
        agg_bars.append({
            'bar_start': start,
            'bar_end': end - 1,
            'open': chunk.iloc[0]['open'],
            'high': chunk['high'].max(),
            'low': chunk['low'].min(),
            'close': chunk.iloc[-1]['close'],
            'volume': chunk['volume'].sum(),
            'vol_delta': chunk['vol_delta'].sum() if 'vol_delta' in chunk.columns else 0,
            'timestamp': chunk.iloc[-1]['timestamp'],
        })
    return agg_bars


def is_inversion_candle_short(bar, vah):
    """
    5-min inversion candle for SHORT:
    - High pokes above VAH (sweep)
    - Close below VAH (rejection — failed to hold above)
    - Close below open (bearish candle body)
    """
    return (bar['high'] > vah and
            bar['close'] < vah and
            bar['close'] < bar['open'])


def is_inversion_candle_long(bar, val):
    """
    5-min inversion candle for LONG:
    - Low pokes below VAL (sweep)
    - Close above VAL (rejection — failed to hold below)
    - Close above open (bullish candle body)
    """
    return (bar['low'] < val and
            bar['close'] > val and
            bar['close'] > bar['open'])


# ============================================================================
# FIND VA EDGE POKE EVENTS
# ============================================================================
def find_va_edge_fades(df_rth, va_by_session):
    """
    Find sessions where price approaches VA edge, pokes outside, and fails.
    Returns list of "poke events" with all context needed for entry models.

    A poke event is: price was inside VA (or near edge), pushes outside
    (5-min bar high > VAH or low < VAL), then shows rejection.

    We track FIRST poke and SECOND poke separately.
    """
    all_events = []

    for i in range(1, len(sessions)):
        current = sessions[i]
        prior = sessions[i - 1]
        prior_key = str(prior)

        if prior_key not in va_by_session:
            continue
        prior_va = va_by_session[prior_key]
        if prior_va.va_width < MIN_VA_WIDTH:
            continue

        vah = prior_va.vah
        val = prior_va.val
        poc = prior_va.poc
        va_width = vah - val
        mid_va = (vah + val) / 2.0

        session_df = df_rth[df_rth['session_date'] == current].reset_index(drop=True)
        if len(session_df) < 90:
            continue

        # Get ATR from current session data (use 14-period ATR if available)
        atr_val = session_df['atr_14'].iloc[60] if 'atr_14' in session_df.columns else None
        if atr_val is None or pd.isna(atr_val) or atr_val <= 0:
            # Fallback: compute from recent 1-min bar ranges
            recent = session_df.iloc[:60]
            atr_val = (recent['high'] - recent['low']).mean()
        if atr_val <= 0:
            atr_val = 15.0

        # Build 5-min bars for the full session
        bars_5m = aggregate_bars(session_df, 5)

        # Track poke events at VAH and VAL
        vah_pokes = []  # Each poke above VAH → SHORT opportunity
        val_pokes = []  # Each poke below VAL → LONG opportunity

        for bi, bar5 in enumerate(bars_5m):
            bt = bar5['timestamp']
            bt_time = bt.time() if hasattr(bt, 'time') else None
            if bt_time and bt_time >= ENTRY_CUTOFF:
                break

            # --- VAH poke (SHORT opportunity) ---
            if bar5['high'] > vah:
                sweep_pts = bar5['high'] - vah
                # Check for inversion candle (bearish rejection)
                is_inversion = is_inversion_candle_short(bar5, vah)

                # Check for 2x5min close back inside: this bar + next bar
                is_2x5m = False
                if bi + 1 < len(bars_5m):
                    next_bar = bars_5m[bi + 1]
                    nt = next_bar['timestamp']
                    nt_time = nt.time() if hasattr(nt, 'time') else None
                    if (bar5['close'] < vah and next_bar['close'] < vah and
                        (not nt_time or nt_time < ENTRY_CUTOFF)):
                        is_2x5m = True

                if is_inversion or is_2x5m:
                    vah_pokes.append({
                        'bar_idx_5m': bi,
                        'bar_end_1m': bar5['bar_end'],
                        'direction': 'SHORT',
                        'edge': 'VAH',
                        'sweep_high': bar5['high'],
                        'sweep_pts': sweep_pts,
                        'bar_open': bar5['open'],
                        'bar_high': bar5['high'],
                        'bar_low': bar5['low'],
                        'bar_close': bar5['close'],
                        'is_inversion': is_inversion,
                        'is_2x5m': is_2x5m,
                        'poke_number': len(vah_pokes) + 1,
                    })

            # --- VAL poke (LONG opportunity) ---
            if bar5['low'] < val:
                sweep_pts = val - bar5['low']
                is_inversion = is_inversion_candle_long(bar5, val)

                is_2x5m = False
                if bi + 1 < len(bars_5m):
                    next_bar = bars_5m[bi + 1]
                    nt = next_bar['timestamp']
                    nt_time = nt.time() if hasattr(nt, 'time') else None
                    if (bar5['close'] > val and next_bar['close'] > val and
                        (not nt_time or nt_time < ENTRY_CUTOFF)):
                        is_2x5m = True

                if is_inversion or is_2x5m:
                    val_pokes.append({
                        'bar_idx_5m': bi,
                        'bar_end_1m': bar5['bar_end'],
                        'direction': 'LONG',
                        'edge': 'VAL',
                        'sweep_low': bar5['low'],
                        'sweep_pts': sweep_pts,
                        'bar_open': bar5['open'],
                        'bar_high': bar5['high'],
                        'bar_low': bar5['low'],
                        'bar_close': bar5['close'],
                        'is_inversion': is_inversion,
                        'is_2x5m': is_2x5m,
                        'poke_number': len(val_pokes) + 1,
                    })

        # Combine and tag first / second poke
        for pokes, edge in [(vah_pokes, 'VAH'), (val_pokes, 'VAL')]:
            for poke in pokes:
                poke.update({
                    'session_date': str(current),
                    'vah': vah,
                    'val': val,
                    'poc': poc,
                    'va_width': va_width,
                    'mid_va': mid_va,
                    'atr': atr_val,
                    'session_df': session_df,
                    'bars_5m': bars_5m,
                })
                all_events.append(poke)

    return all_events


# ============================================================================
# REPLAY TRADE — VA EDGE FADE
# ============================================================================
def replay_va_edge_fade(event, entry_model, stop_model, target_model):
    """
    Replay a VA edge fade trade.

    entry_model: 'inversion' | '2x5m' | 'limit_sweep' | 'limit_edge'
    stop_model:  '2atr' | '80pt' | 'swing' | 'va_edge_10'
    target_model: '0.2atr_trail' | '0.5atr_trail' | '1R' | '2R' | '4R' |
                  'poc' | 'mid_va' | 'opposite_va'
    """
    direction = event['direction']
    vah = event['vah']
    val = event['val']
    poc = event['poc']
    mid_va = event['mid_va']
    va_width = event['va_width']
    atr = event['atr']
    session_df = event['session_df']
    bars_5m = event['bars_5m']
    bi = event['bar_idx_5m']
    bar = event  # the poke bar itself

    # --- Entry Model ---
    if entry_model == 'inversion' and not event['is_inversion']:
        return None
    if entry_model == '2x5m' and not event['is_2x5m']:
        return None

    if entry_model in ('inversion', '2x5m'):
        # Market entry at close of signal bar (or next bar for 2x5m)
        if entry_model == '2x5m' and bi + 1 < len(bars_5m):
            signal_bar = bars_5m[bi + 1]
        else:
            signal_bar = bars_5m[bi]
        entry_price = signal_bar['close']
        entry_bar_1m = signal_bar['bar_end']

    elif entry_model == 'limit_sweep':
        # Limit order at the sweep extreme (double top/bottom)
        # Only makes sense on 2nd+ poke — set limit at 1st poke extreme
        if event['poke_number'] < 2:
            return None
        if direction == 'SHORT':
            entry_price = event['sweep_high']
        else:
            entry_price = event['sweep_low'] if 'sweep_low' in event else event['bar_low']
        # Check if limit gets filled in subsequent bars
        entry_bar_1m = _find_limit_fill(session_df, event['bar_end_1m'],
                                         entry_price, direction)
        if entry_bar_1m is None:
            return None

    elif entry_model == 'limit_edge':
        # Limit at exact VA edge on retest
        if direction == 'SHORT':
            entry_price = vah
        else:
            entry_price = val
        entry_bar_1m = _find_limit_fill(session_df, event['bar_end_1m'],
                                         entry_price, direction)
        if entry_bar_1m is None:
            return None
    else:
        return None

    # --- Stop Model ---
    if stop_model == '2atr':
        stop_distance = 2.0 * atr
        if direction == 'SHORT':
            stop_price = entry_price + stop_distance
        else:
            stop_price = entry_price - stop_distance
    elif stop_model == '80pt':
        if direction == 'SHORT':
            stop_price = entry_price + 80
        else:
            stop_price = entry_price - 80
    elif stop_model == 'swing':
        # Use 5-min candle extreme as swing stop
        if direction == 'SHORT':
            stop_price = event['bar_high'] + 5  # 5pt buffer above sweep high
        else:
            stop_price = event['bar_low'] - 5   # 5pt buffer below sweep low
    elif stop_model == 'va_edge_10':
        if direction == 'SHORT':
            stop_price = vah + 10
        else:
            stop_price = val - 10
    else:
        return None

    risk_pts = abs(entry_price - stop_price)
    if risk_pts <= 0 or risk_pts > 200:
        return None

    # --- Target Model ---
    is_trail = target_model.endswith('atr_trail')
    if target_model == 'poc':
        target_price = poc
    elif target_model == 'mid_va':
        target_price = mid_va
    elif target_model == 'opposite_va':
        target_price = val if direction == 'SHORT' else vah
    elif target_model.endswith('R'):
        r_mult = float(target_model[:-1])
        if direction == 'SHORT':
            target_price = entry_price - risk_pts * r_mult
        else:
            target_price = entry_price + risk_pts * r_mult
    elif is_trail:
        # Trailing stop mode — no fixed target, trail at N * ATR
        trail_factor = float(target_model.split('atr')[0])
        trail_distance = trail_factor * atr
        target_price = None  # No fixed target, use trailing logic
    else:
        return None

    # Validate target makes sense
    if target_price is not None:
        if direction == 'SHORT':
            reward = entry_price - target_price
        else:
            reward = target_price - entry_price
        if reward <= 0:
            return None

    # --- Replay from entry bar ---
    remaining = session_df.iloc[entry_bar_1m:]
    if len(remaining) == 0:
        return None

    mfe = 0.0
    mae = 0.0
    exit_price = None
    exit_reason = None
    bars_held = 0
    trail_stop = stop_price  # Initialize trailing stop

    for idx in range(len(remaining)):
        bar_row = remaining.iloc[idx]
        bars_held += 1
        bt_time = bar_row.get('timestamp')
        bt = bt_time.time() if bt_time and hasattr(bt_time, 'time') else None
        if bt and bt >= EOD_EXIT:
            exit_price = bar_row['close']
            exit_reason = 'EOD'
            break

        if direction == 'SHORT':
            fav = entry_price - bar_row['low']
            adv = bar_row['high'] - entry_price
        else:
            fav = bar_row['high'] - entry_price
            adv = entry_price - bar_row['low']
        mfe = max(mfe, fav)
        mae = max(mae, adv)

        # Trailing stop update (if trail mode)
        if is_trail and idx > 0:
            trail_factor = float(target_model.split('atr')[0])
            trail_distance = trail_factor * atr
            if direction == 'SHORT':
                # Trail the stop down as price moves in our favor
                new_trail = bar_row['low'] + trail_distance
                if new_trail < trail_stop:
                    trail_stop = new_trail
            else:
                new_trail = bar_row['high'] - trail_distance
                if new_trail > trail_stop:
                    trail_stop = new_trail

        # Check stop (use trail_stop if trailing, otherwise static stop)
        active_stop = trail_stop if is_trail else stop_price
        if direction == 'SHORT' and bar_row['high'] >= active_stop:
            exit_price = active_stop
            exit_reason = 'STOP' if active_stop == stop_price else 'TRAIL_STOP'
            # If trailing stop was hit ABOVE entry, it's a loss; below entry it's a win
            if is_trail and exit_price < entry_price:
                exit_reason = 'TRAIL_STOP'
            break
        elif direction == 'LONG' and bar_row['low'] <= active_stop:
            exit_price = active_stop
            exit_reason = 'STOP' if active_stop == stop_price else 'TRAIL_STOP'
            if is_trail and exit_price > entry_price:
                exit_reason = 'TRAIL_STOP'
            break

        # Check fixed target (if not trailing)
        if target_price is not None:
            if direction == 'SHORT' and bar_row['low'] <= target_price:
                exit_price = target_price
                exit_reason = 'TARGET'
                break
            elif direction == 'LONG' and bar_row['high'] >= target_price:
                exit_price = target_price
                exit_reason = 'TARGET'
                break

    if exit_price is None:
        exit_price = remaining.iloc[-1]['close']
        exit_reason = 'EOD'

    if direction == 'SHORT':
        pnl_pts = entry_price - exit_price - SLIPPAGE_PTS
    else:
        pnl_pts = exit_price - entry_price - SLIPPAGE_PTS
    pnl_dollars = pnl_pts * TICK_VALUE * CONTRACTS - COMMISSION * CONTRACTS

    return {
        'session_date': event['session_date'],
        'direction': direction,
        'edge': event['edge'],
        'poke_number': event['poke_number'],
        'sweep_pts': event['sweep_pts'],
        'entry_model': entry_model,
        'entry_price': entry_price,
        'stop_price': stop_price,
        'target_price': target_price,
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'pnl_pts': pnl_pts,
        'pnl_dollars': pnl_dollars,
        'risk_pts': risk_pts,
        'mfe_pts': mfe,
        'mae_pts': mae,
        'bars_held': bars_held,
        'atr': atr,
        'va_width': va_width,
        'is_winner': pnl_dollars > 0,
        'is_inversion': event['is_inversion'],
        'is_2x5m': event['is_2x5m'],
    }


def _find_limit_fill(session_df, start_bar_1m, limit_price, direction):
    """Find the 1-min bar where a limit order gets filled."""
    for bi in range(start_bar_1m + 1, len(session_df)):
        bar = session_df.iloc[bi]
        bt_time = bar.get('timestamp')
        bt = bt_time.time() if bt_time and hasattr(bt_time, 'time') else None
        if bt and bt >= EOD_EXIT:
            return None
        # SHORT limit: filled when price rises to limit_price
        if direction == 'SHORT' and bar['high'] >= limit_price:
            return bi
        # LONG limit: filled when price drops to limit_price
        elif direction == 'LONG' and bar['low'] <= limit_price:
            return bi
    return None


# ============================================================================
# PRINT HELPERS
# ============================================================================
def print_results(results, label, months_val):
    """Print a summary row for a set of trade results."""
    if not results:
        print(f"    {label:<60s}    0")
        return {}

    df_r = pd.DataFrame(results)
    n = len(df_r)
    wr = df_r['is_winner'].mean() * 100
    gw = df_r[df_r['is_winner']]['pnl_dollars'].sum()
    gl = abs(df_r[~df_r['is_winner']]['pnl_dollars'].sum())
    pf = gw / gl if gl > 0 else float('inf')
    pm = df_r['pnl_dollars'].sum() / months_val
    avg_w = df_r[df_r['is_winner']]['pnl_dollars'].mean() if df_r['is_winner'].any() else 0
    avg_l = df_r[~df_r['is_winner']]['pnl_dollars'].mean() if (~df_r['is_winner']).any() else 0
    avg_risk = df_r['risk_pts'].mean()
    stopped = (df_r['exit_reason'] == 'STOP').sum()
    target_hit = (df_r['exit_reason'] == 'TARGET').sum()
    trail_stopped = (df_r['exit_reason'] == 'TRAIL_STOP').sum()
    eod = (df_r['exit_reason'] == 'EOD').sum()
    avg_mfe = df_r['mfe_pts'].mean()
    avg_mae = df_r['mae_pts'].mean()

    print(f"    {label:<60s} {n:>4d} {n/months_val:>5.1f} "
          f"{wr:>5.1f}% {pf:>5.2f} ${pm:>8,.0f} ${avg_w:>7,.0f} ${avg_l:>7,.0f} "
          f"{avg_risk:>5.0f}p {avg_mfe:>5.0f}/{avg_mae:>4.0f} "
          f"{stopped:>3d}/{target_hit:>3d}/{trail_stopped:>3d}/{eod:>3d}")

    return {'label': label, 'n': n, 'wr': wr, 'pf': pf, 'pm': pm,
            'avg_risk': avg_risk, 'stopped': stopped}


def print_header():
    print(f"    {'Config':<60s} {'N':>4s} {'T/Mo':>5s} "
          f"{'WR':>6s} {'PF':>6s} {'$/Mo':>9s} {'AvgWin':>8s} {'AvgLoss':>8s} "
          f"{'Risk':>6s} {'MFE/MAE':>9s} {'S/T/Tr/E':>13s}")
    print(f"    {'-'*155}")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================
print("\n" + "=" * 120)
print("  VA EDGE FADE STUDY — Fading Failed Breakouts at Value Area Boundaries")
print("=" * 120)

print("\nScanning for VA edge poke events...")
events = find_va_edge_fades(df_rth, eth_va)
print(f"Found {len(events)} VA edge poke/rejection events")

# Summary stats
first_pokes = [e for e in events if e['poke_number'] == 1]
second_pokes = [e for e in events if e['poke_number'] == 2]
third_plus = [e for e in events if e['poke_number'] >= 3]
vah_events = [e for e in events if e['edge'] == 'VAH']
val_events = [e for e in events if e['edge'] == 'VAL']
inversion_events = [e for e in events if e['is_inversion']]
two_x_5m_events = [e for e in events if e['is_2x5m']]

print(f"\n  Breakdown:")
print(f"    1st poke (first touch):  {len(first_pokes)}")
print(f"    2nd poke (retest):       {len(second_pokes)}")
print(f"    3rd+ poke:               {len(third_plus)}")
print(f"    VAH fades (SHORT):       {len(vah_events)}")
print(f"    VAL fades (LONG):        {len(val_events)}")
print(f"    Inversion candle:        {len(inversion_events)}")
print(f"    2x5min close:            {len(two_x_5m_events)}")

sweep_pts = [e['sweep_pts'] for e in events]
print(f"\n  Sweep depth (pts beyond VA edge):")
print(f"    Mean:   {np.mean(sweep_pts):.1f}")
print(f"    Median: {np.median(sweep_pts):.1f}")
for pct in [25, 50, 75, 90]:
    print(f"    {pct}th:   {np.percentile(sweep_pts, pct):.1f}")

atrs = [e['atr'] for e in events]
print(f"\n  ATR at entry:")
print(f"    Mean:   {np.mean(atrs):.1f}")
print(f"    Median: {np.median(atrs):.1f}")

# ============================================================================
# SECTION 1: FIRST TOUCH — 2 ATR Stop, Various Targets
# ============================================================================
print(f"\n\n{'='*120}")
print(f"  SECTION 1: FIRST TOUCH (Poke #1) — Entry on Inversion Candle or 2x5min")
print(f"{'='*120}")

entry_models = ['inversion', '2x5m']
stop_models = ['2atr', '80pt', 'va_edge_10', 'swing']
target_models = ['0.2atr_trail', '0.5atr_trail', '1R', '2R', '4R', 'poc', 'mid_va', 'opposite_va']

all_configs = []

for entry_m in entry_models:
    print(f"\n  ━━━ Entry: {entry_m.upper()} ━━━")

    for stop_m in stop_models:
        print(f"\n    Stop: {stop_m}")
        print_header()

        for target_m in target_models:
            results = []
            for event in first_pokes:
                r = replay_va_edge_fade(event, entry_m, stop_m, target_m)
                if r:
                    results.append(r)

            label = f"1st_touch | {entry_m} | {stop_m} | {target_m}"
            stats = print_results(results, label, months)
            if stats:
                stats['entry'] = entry_m
                stats['stop'] = stop_m
                stats['target'] = target_m
                stats['poke'] = '1st'
                all_configs.append(stats)


# ============================================================================
# SECTION 2: SECOND TEST (Poke #2) — All Entry Models
# ============================================================================
print(f"\n\n{'='*120}")
print(f"  SECTION 2: SECOND TEST (Poke #2) — Multiple Entry Models")
print(f"{'='*120}")

entry_models_2 = ['inversion', '2x5m', 'limit_sweep', 'limit_edge']

for entry_m in entry_models_2:
    print(f"\n  ━━━ Entry: {entry_m.upper()} ━━━")

    for stop_m in stop_models:
        print(f"\n    Stop: {stop_m}")
        print_header()

        for target_m in target_models:
            results = []
            for event in second_pokes:
                r = replay_va_edge_fade(event, entry_m, stop_m, target_m)
                if r:
                    results.append(r)

            label = f"2nd_test | {entry_m} | {stop_m} | {target_m}"
            stats = print_results(results, label, months)
            if stats:
                stats['entry'] = entry_m
                stats['stop'] = stop_m
                stats['target'] = target_m
                stats['poke'] = '2nd'
                all_configs.append(stats)


# ============================================================================
# SECTION 3: COMBINED 1st + 2nd (Take Both Trades)
# ============================================================================
print(f"\n\n{'='*120}")
print(f"  SECTION 3: COMBINED — Take 1st Touch AND 2nd Test (both trades)")
print(f"{'='*120}")

combined_events = [e for e in events if e['poke_number'] <= 2]

for entry_m in ['inversion', '2x5m']:
    print(f"\n  ━━━ Entry: {entry_m.upper()} ━━━")

    for stop_m in ['2atr', 'va_edge_10']:
        print(f"\n    Stop: {stop_m}")
        print_header()

        for target_m in target_models:
            results = []
            for event in combined_events:
                r = replay_va_edge_fade(event, entry_m, stop_m, target_m)
                if r:
                    results.append(r)

            label = f"combined | {entry_m} | {stop_m} | {target_m}"
            stats = print_results(results, label, months)
            if stats:
                stats['entry'] = entry_m
                stats['stop'] = stop_m
                stats['target'] = target_m
                stats['poke'] = 'combined'
                all_configs.append(stats)


# ============================================================================
# SECTION 4: SHORTS ONLY (Fade VAH Poke)
# ============================================================================
print(f"\n\n{'='*120}")
print(f"  SECTION 4: SHORTS ONLY — Fade VAH Poke (price fails above VAH)")
print(f"{'='*120}")

short_events = [e for e in events if e['direction'] == 'SHORT']
short_first = [e for e in short_events if e['poke_number'] == 1]
short_second = [e for e in short_events if e['poke_number'] == 2]

print(f"\n  SHORT events: {len(short_events)} total, {len(short_first)} 1st poke, {len(short_second)} 2nd poke")

for poke_label, poke_set in [('1st_SHORT', short_first), ('2nd_SHORT', short_second)]:
    if not poke_set:
        print(f"\n  No {poke_label} events found.")
        continue

    print(f"\n  ━━━ {poke_label} ━━━")
    for entry_m in ['inversion', '2x5m']:
        for stop_m in ['2atr', 'va_edge_10']:
            print(f"\n    {entry_m} + {stop_m}")
            print_header()
            for target_m in target_models:
                results = []
                for event in poke_set:
                    r = replay_va_edge_fade(event, entry_m, stop_m, target_m)
                    if r:
                        results.append(r)
                label = f"{poke_label} | {entry_m} | {stop_m} | {target_m}"
                stats = print_results(results, label, months)
                if stats:
                    stats['entry'] = entry_m
                    stats['stop'] = stop_m
                    stats['target'] = target_m
                    stats['poke'] = poke_label
                    all_configs.append(stats)


# ============================================================================
# SECTION 5: 80-POINT STOP with 2 ATR — Limit Retest Strategies
# ============================================================================
print(f"\n\n{'='*120}")
print(f"  SECTION 5: 80-POINT STOP with 2 ATR — Limit Retest Entry Models")
print(f"{'='*120}")
print(f"  Testing 80pt stop and 2 ATR stop with limit entries on VA edge retest")

for stop_m in ['80pt', '2atr']:
    print(f"\n  ━━━ Stop: {stop_m} ━━━")
    for entry_m in ['limit_edge', 'limit_sweep']:
        print(f"\n    Entry: {entry_m}")
        print_header()

        # Test on second pokes (retest scenarios)
        for target_m in target_models:
            results = []
            for event in second_pokes:
                r = replay_va_edge_fade(event, entry_m, stop_m, target_m)
                if r:
                    results.append(r)
            label = f"retest_limit | {entry_m} | {stop_m} | {target_m}"
            stats = print_results(results, label, months)
            if stats:
                stats['entry'] = entry_m
                stats['stop'] = stop_m
                stats['target'] = target_m
                stats['poke'] = 'retest_limit'
                all_configs.append(stats)


# ============================================================================
# SECTION 6: TOP CONFIGURATIONS RANKED
# ============================================================================
print(f"\n\n{'='*120}")
print(f"  TOP CONFIGURATIONS — Sorted by $/Month (min 3 trades)")
print(f"{'='*120}")

# Filter configs with meaningful trade count
viable = [c for c in all_configs if c.get('n', 0) >= 3]
viable.sort(key=lambda x: x.get('pm', 0), reverse=True)

print(f"\n  {'Rank':>4s}  {'Config':<70s} {'N':>4s} {'WR':>6s} {'PF':>6s} {'$/Mo':>9s} {'Risk':>6s}")
print(f"  {'-'*115}")

for rank, c in enumerate(viable[:40], 1):
    print(f"  {rank:>4d}  {c['label']:<70s} {c['n']:>4d} "
          f"{c['wr']:>5.1f}% {c['pf']:>5.2f} ${c['pm']:>8,.0f} {c.get('avg_risk', 0):>5.0f}p")


# ============================================================================
# SECTION 7: TOP BY WIN RATE (min 5 trades, PF > 1.0)
# ============================================================================
print(f"\n\n{'='*120}")
print(f"  TOP BY WIN RATE (min 5 trades, PF > 1.0)")
print(f"{'='*120}")

wr_viable = [c for c in all_configs if c.get('n', 0) >= 5 and c.get('pf', 0) > 1.0]
wr_viable.sort(key=lambda x: x.get('wr', 0), reverse=True)

print(f"\n  {'Rank':>4s}  {'Config':<70s} {'N':>4s} {'WR':>6s} {'PF':>6s} {'$/Mo':>9s}")
print(f"  {'-'*105}")

for rank, c in enumerate(wr_viable[:20], 1):
    print(f"  {rank:>4d}  {c['label']:<70s} {c['n']:>4d} "
          f"{c['wr']:>5.1f}% {c['pf']:>5.2f} ${c['pm']:>8,.0f}")


# ============================================================================
# SECTION 8: TRADE LOG — Best Configuration
# ============================================================================
if viable:
    best = viable[0]
    print(f"\n\n{'='*120}")
    print(f"  TRADE LOG — {best['label']}")
    print(f"  WR={best['wr']:.1f}%, PF={best['pf']:.2f}, $/Mo=${best['pm']:,.0f}")
    print(f"{'='*120}")

    # Determine which events to use based on poke type
    poke_type = best.get('poke', '')
    if poke_type == '1st':
        replay_events = first_pokes
    elif poke_type == '2nd':
        replay_events = second_pokes
    elif poke_type == 'combined':
        replay_events = combined_events
    elif '1st_SHORT' in poke_type:
        replay_events = short_first
    elif '2nd_SHORT' in poke_type:
        replay_events = short_second
    elif 'retest_limit' in poke_type:
        replay_events = second_pokes
    else:
        replay_events = events

    best_results = []
    for event in replay_events:
        r = replay_va_edge_fade(event, best['entry'], best['stop'], best['target'])
        if r:
            best_results.append(r)

    if best_results:
        df_best = pd.DataFrame(best_results)
        print(f"\n  {'Date':12s} {'Dir':5s} {'Edge':4s} {'Pk#':>3s} "
              f"{'Entry':>8s} {'Stop':>8s} {'Target':>8s} {'Exit':>8s} "
              f"{'Reason':10s} {'P&L':>8s} {'Risk':>5s} {'MFE':>5s} {'MAE':>5s} "
              f"{'Sweep':>5s}")
        print(f"  {'-'*120}")

        for _, t in df_best.sort_values('session_date').iterrows():
            tgt_str = f"{t['target_price']:>8.1f}" if t['target_price'] is not None else "   trail"
            print(f"  {t['session_date']:12s} {t['direction']:5s} {t['edge']:4s} "
                  f"{t['poke_number']:>3d} "
                  f"{t['entry_price']:>8.1f} {t['stop_price']:>8.1f} {tgt_str} "
                  f"{t['exit_price']:>8.1f} {t['exit_reason']:10s} "
                  f"${t['pnl_dollars']:>7,.0f} {t['risk_pts']:>4.0f}p "
                  f"{t['mfe_pts']:>4.0f}p {t['mae_pts']:>4.0f}p "
                  f"{t['sweep_pts']:>4.1f}p")


# ============================================================================
# SECTION 9: SUMMARY & RECOMMENDATIONS
# ============================================================================
print(f"\n\n{'='*120}")
print(f"  SUMMARY & RECOMMENDATIONS")
print(f"{'='*120}")

print(f"""
  STUDY PARAMETERS:
    - {n_sessions} sessions ({months:.1f} months)
    - {len(events)} total VA edge poke/rejection events detected
    - {len(first_pokes)} first-touch events, {len(second_pokes)} second-test events
    - {len(vah_events)} VAH fades (SHORT), {len(val_events)} VAL fades (LONG)
    - Median ATR: {np.median(atrs):.1f} pts
    - Median sweep depth: {np.median(sweep_pts):.1f} pts beyond VA edge

  ENTRY MODELS TESTED:
    1. Inversion candle (5-min bearish/bullish rejection candle at VA edge)
    2. 2x 5-min close back inside VA (consecutive closes confirming rejection)
    3. Limit at sweep extreme (double top/bottom on 2nd test)
    4. Limit at VA edge (exact edge entry on retest)

  STOP MODELS TESTED:
    1. 2 ATR (primary)
    2. 80-point fixed
    3. VA edge + 10pt buffer
    4. Swing point (5-min candle extreme + 5pt)

  TARGET MODELS TESTED:
    1. 0.2 ATR trailing stop (shallow scalp)
    2. 0.5 ATR trailing stop (moderate trail)
    3. 1R, 2R, 4R fixed risk-multiple targets
    4. POC, Mid-VA, Opposite VA
""")

if viable:
    print(f"  TOP 5 CONFIGURATIONS:")
    for rank, c in enumerate(viable[:5], 1):
        print(f"    {rank}. {c['label']}")
        print(f"       {c['n']} trades, WR={c['wr']:.1f}%, PF={c['pf']:.2f}, $/Mo=${c['pm']:,.0f}")

if wr_viable:
    print(f"\n  HIGHEST WIN RATE CONFIGS (min 5 trades):")
    for rank, c in enumerate(wr_viable[:3], 1):
        print(f"    {rank}. {c['label']}")
        print(f"       {c['n']} trades, WR={c['wr']:.1f}%, PF={c['pf']:.2f}, $/Mo=${c['pm']:,.0f}")

print(f"\n{'='*120}")
print(f"  ANALYSIS COMPLETE")
print(f"{'='*120}")
