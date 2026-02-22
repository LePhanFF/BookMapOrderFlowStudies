"""
80P Entry Model Diagnosis — Is the acceptance entry too shallow?
================================================================

Questions:
1. After acceptance, does price continue INTO the VA or reverse back OUT?
   → If 50% reverse back out, acceptance is NOT confirming mean-reversion
2. Would a limit order at 50% VA depth improve WR?
3. What if we skip acceptance and just use a limit at 50% VA?
4. How many "accepted" setups are actually failed continuations?
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import time
from typing import Dict, List, Optional

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
# FIND ALL 80P SETUPS with full session data
# ============================================================================
def find_all_setups(df_rth, va_by_session, directions='BOTH'):
    """Find every session where open is outside prior VA. Track everything."""
    setups = []

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
        if len(session_df) < 90:
            continue

        open_price = session_df['open'].iloc[0]
        vah = prior_va.vah
        val = prior_va.val
        poc = prior_va.poc
        va_width = vah - val

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

        # Track acceptance (1x 30-min close inside VA)
        consecutive = 0
        acceptance_bar_idx = None
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
            if consecutive >= 1 and acceptance_bar_idx is None:
                acceptance_bar_idx = bar_idx
                break

        # Track when price reaches various VA depths
        va_50pct = val + va_width * 0.50 if direction == 'LONG' else vah - va_width * 0.50
        va_35pct = val + va_width * 0.35 if direction == 'LONG' else vah - va_width * 0.35
        va_25pct = val + va_width * 0.25 if direction == 'LONG' else vah - va_width * 0.25

        depth_50_bar = None
        depth_35_bar = None
        depth_25_bar = None
        first_inside_bar = None

        for bar_idx in range(len(post_ib)):
            bar = post_ib.iloc[bar_idx]
            bar_time = bar['timestamp']
            bt = bar_time.time() if hasattr(bar_time, 'time') else None
            if bt and bt >= ENTRY_CUTOFF:
                break

            if direction == 'LONG':
                if bar['high'] >= val and first_inside_bar is None:
                    first_inside_bar = bar_idx
                if bar['high'] >= va_25pct and depth_25_bar is None:
                    depth_25_bar = bar_idx
                if bar['high'] >= va_35pct and depth_35_bar is None:
                    depth_35_bar = bar_idx
                if bar['high'] >= va_50pct and depth_50_bar is None:
                    depth_50_bar = bar_idx
            else:
                if bar['low'] <= vah and first_inside_bar is None:
                    first_inside_bar = bar_idx
                if bar['low'] <= vah - va_width * 0.25 and depth_25_bar is None:
                    depth_25_bar = bar_idx
                if bar['low'] <= vah - va_width * 0.35 and depth_35_bar is None:
                    depth_35_bar = bar_idx
                if bar['low'] <= va_50pct and depth_50_bar is None:
                    depth_50_bar = bar_idx

        setups.append({
            'session_date': str(current),
            'direction': direction,
            'open_price': open_price,
            'vah': vah,
            'val': val,
            'poc': poc,
            'va_width': va_width,
            'acceptance_bar': acceptance_bar_idx,
            'first_inside_bar': first_inside_bar,
            'depth_25_bar': depth_25_bar,
            'depth_35_bar': depth_35_bar,
            'depth_50_bar': depth_50_bar,
            'post_ib': post_ib,
            'session_df': session_df,
        })

    return setups


def replay_with_entry(setup, entry_mode, target_mode='opposite_va'):
    """
    Replay a trade with different entry strategies.

    entry_mode:
      'acceptance' — enter at 30-min acceptance bar close (current model)
      'limit_25'   — limit order at 25% VA depth
      'limit_35'   — limit order at 35% VA depth
      'limit_50'   — limit order at 50% VA depth (POC area)
      'immediate'  — enter as soon as price touches VA boundary (no acceptance)
    """
    direction = setup['direction']
    vah = setup['vah']
    val = setup['val']
    poc = setup['poc']
    va_width = setup['va_width']
    post_ib = setup['post_ib']

    # Determine entry bar and price based on entry_mode
    if entry_mode == 'acceptance':
        if setup['acceptance_bar'] is None:
            return None
        entry_bar = setup['acceptance_bar']
        entry_price = post_ib.iloc[entry_bar]['close']
    elif entry_mode == 'immediate':
        if setup['first_inside_bar'] is None:
            return None
        entry_bar = setup['first_inside_bar']
        if direction == 'LONG':
            entry_price = val  # limit at VA edge
        else:
            entry_price = vah
    elif entry_mode == 'limit_25':
        if setup['depth_25_bar'] is None:
            return None
        entry_bar = setup['depth_25_bar']
        if direction == 'LONG':
            entry_price = val + va_width * 0.25
        else:
            entry_price = vah - va_width * 0.25
    elif entry_mode == 'limit_35':
        if setup['depth_35_bar'] is None:
            return None
        entry_bar = setup['depth_35_bar']
        if direction == 'LONG':
            entry_price = val + va_width * 0.35
        else:
            entry_price = vah - va_width * 0.35
    elif entry_mode == 'limit_50':
        if setup['depth_50_bar'] is None:
            return None
        entry_bar = setup['depth_50_bar']
        if direction == 'LONG':
            entry_price = val + va_width * 0.50
        else:
            entry_price = vah - va_width * 0.50
    else:
        return None

    # Stop: always VA boundary + buffer
    if direction == 'LONG':
        stop_price = val - STOP_BUFFER
    else:
        stop_price = vah + STOP_BUFFER

    risk_pts = abs(entry_price - stop_price)
    if risk_pts <= 0:
        return None

    # Target
    if target_mode == 'opposite_va':
        target = vah if direction == 'LONG' else val
    elif target_mode.endswith('R'):
        r_mult = float(target_mode[:-1])
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

    # Replay from entry bar forward
    remaining = post_ib.iloc[entry_bar:]
    mfe = 0.0
    mae = 0.0
    exit_price = None
    exit_reason = None
    bars_held = 0

    for i in range(len(remaining)):
        bar = remaining.iloc[i]
        bars_held += 1

        bt_time = bar.get('timestamp')
        bt = bt_time.time() if bt_time and hasattr(bt_time, 'time') else None
        if bt and bt >= time(15, 30):
            exit_price = bar['close']
            exit_reason = 'EOD'
            break

        if direction == 'LONG':
            fav = bar['high'] - entry_price
            adv = entry_price - bar['low']
        else:
            fav = entry_price - bar['low']
            adv = bar['high'] - entry_price
        mfe = max(mfe, fav)
        mae = max(mae, adv)

        # Stop
        if direction == 'LONG' and bar['low'] <= stop_price:
            exit_price = stop_price
            exit_reason = 'STOP'
            break
        elif direction == 'SHORT' and bar['high'] >= stop_price:
            exit_price = stop_price
            exit_reason = 'STOP'
            break

        # Target
        if direction == 'LONG' and bar['high'] >= target:
            exit_price = target
            exit_reason = 'TARGET'
            break
        elif direction == 'SHORT' and bar['low'] <= target:
            exit_price = target
            exit_reason = 'TARGET'
            break

    if exit_price is None:
        exit_price = remaining.iloc[-1]['close']
        exit_reason = 'EOD'

    if direction == 'LONG':
        pnl_pts = exit_price - entry_price - SLIPPAGE_PTS
    else:
        pnl_pts = entry_price - exit_price - SLIPPAGE_PTS

    pnl_dollars = pnl_pts * TICK_VALUE * CONTRACTS - COMMISSION * CONTRACTS

    # Entry depth within VA
    if direction == 'LONG':
        entry_depth = (entry_price - val) / va_width if va_width > 0 else 0
    else:
        entry_depth = (vah - entry_price) / va_width if va_width > 0 else 0

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
        'reward_pts': reward,
        'rr_ratio': reward / risk_pts if risk_pts > 0 else 0,
        'va_width': va_width,
        'mfe_pts': mfe,
        'mae_pts': mae,
        'mfe_r': mfe / risk_pts if risk_pts > 0 else 0,
        'bars_held': bars_held,
        'entry_bar': entry_bar,
        'entry_depth': entry_depth,
        'is_winner': pnl_dollars > 0,
    }


# ============================================================================
# MAIN
# ============================================================================
print("\nFinding all 80P setups (ETH VA, BOTH directions)...")
setups = find_all_setups(df_rth, eth_va, directions='BOTH')
print(f"Total setups (open outside prior VA): {len(setups)}")

# Count how many reach various depths
n_accepted = sum(1 for s in setups if s['acceptance_bar'] is not None)
n_inside = sum(1 for s in setups if s['first_inside_bar'] is not None)
n_d25 = sum(1 for s in setups if s['depth_25_bar'] is not None)
n_d35 = sum(1 for s in setups if s['depth_35_bar'] is not None)
n_d50 = sum(1 for s in setups if s['depth_50_bar'] is not None)

print(f"\n{'='*100}")
print(f"  HOW DEEP DOES PRICE PENETRATE INTO THE VA?")
print(f"{'='*100}")
print(f"\n  Of {len(setups)} setups (open outside VA, before 13:00):")
print(f"    Price touches VA boundary:  {n_inside:3d} ({n_inside/len(setups)*100:.0f}%)")
print(f"    1x30min acceptance (acc=1):  {n_accepted:3d} ({n_accepted/len(setups)*100:.0f}%)")
print(f"    Price reaches 25% into VA:  {n_d25:3d} ({n_d25/len(setups)*100:.0f}%)")
print(f"    Price reaches 35% into VA:  {n_d35:3d} ({n_d35/len(setups)*100:.0f}%)")
print(f"    Price reaches 50% into VA:  {n_d50:3d} ({n_d50/len(setups)*100:.0f}%)")

# ============================================================================
# COMPARE ENTRY MODES
# ============================================================================
print(f"\n{'='*100}")
print(f"  ENTRY MODE COMPARISON")
print(f"{'='*100}")

entry_modes = ['acceptance', 'immediate', 'limit_25', 'limit_35', 'limit_50']
target_modes = ['opposite_va', '2R', '4R']

for target_mode in target_modes:
    print(f"\n  ── Target: {target_mode} ──")
    print(f"  {'Entry Mode':<18s} {'N':>4s} {'Trd/Mo':>7s} {'WR':>6s} {'PF':>6s} "
          f"{'$/Mo':>9s} {'AvgWin':>8s} {'AvgLoss':>8s} {'R:R':>5s} "
          f"{'MFE':>5s} {'MAE':>5s} {'Risk':>5s} {'Depth':>5s}")
    print(f"  {'-'*110}")

    for entry_mode in entry_modes:
        results = []
        for setup in setups:
            r = replay_with_entry(setup, entry_mode, target_mode)
            if r:
                results.append(r)

        if not results:
            print(f"  {entry_mode:<18s} {'0':>4s}")
            continue

        df_r = pd.DataFrame(results)
        n = len(df_r)
        wins = df_r[df_r['is_winner']]
        losses = df_r[~df_r['is_winner']]
        wr = len(wins) / n * 100
        total_pnl = df_r['pnl_dollars'].sum()
        per_month = total_pnl / months
        avg_win = wins['pnl_dollars'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl_dollars'].mean() if len(losses) > 0 else 0
        gw = wins['pnl_dollars'].sum() if len(wins) > 0 else 0
        gl = abs(losses['pnl_dollars'].sum()) if len(losses) > 0 else 0.01
        pf = gw / gl if gl > 0 else float('inf')
        rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        print(f"  {entry_mode:<18s} {n:>4d} {n/months:>6.1f} "
              f"{wr:>5.1f}% {pf:>5.2f} "
              f"${per_month:>7,.0f} ${avg_win:>6,.0f} ${avg_loss:>6,.0f} "
              f"{rr:>4.2f} "
              f"{df_r['mfe_pts'].mean():>4.0f}p {df_r['mae_pts'].mean():>4.0f}p "
              f"{df_r['risk_pts'].mean():>4.0f}p {df_r['entry_depth'].mean():>4.2f}")


# ============================================================================
# ACCEPTANCE ANALYSIS: What happens AFTER acceptance?
# ============================================================================
print(f"\n{'='*100}")
print(f"  WHAT HAPPENS AFTER ACCEPTANCE? (Does price continue into VA or reverse out?)")
print(f"{'='*100}")

accepted_setups = [s for s in setups if s['acceptance_bar'] is not None]
print(f"\n  {len(accepted_setups)} accepted setups analyzed:")

continues_deeper = 0
reverses_out = 0
chops = 0

for setup in accepted_setups:
    direction = setup['direction']
    acc_bar = setup['acceptance_bar']
    post_ib = setup['post_ib']
    vah = setup['vah']
    val = setup['val']
    va_width = setup['va_width']

    entry_price = post_ib.iloc[acc_bar]['close']

    # Track next 30 bars (30 min) after acceptance
    next_30 = post_ib.iloc[acc_bar + 1: acc_bar + 31]
    if len(next_30) == 0:
        continue

    if direction == 'LONG':
        entry_depth = (entry_price - val) / va_width
        # Did it go deeper into VA (higher)?
        max_depth_after = (next_30['high'].max() - val) / va_width
        # Did it reverse back below VAL?
        min_after = next_30['low'].min()
        reversed_below = min_after < val
        went_deeper = max_depth_after > entry_depth + 0.10  # at least 10% deeper
    else:
        entry_depth = (vah - entry_price) / va_width
        max_depth_after = (vah - next_30['low'].min()) / va_width
        max_after = next_30['high'].max()
        reversed_below = max_after > vah
        went_deeper = max_depth_after > entry_depth + 0.10

    if reversed_below:
        reverses_out += 1
    elif went_deeper:
        continues_deeper += 1
    else:
        chops += 1

total = continues_deeper + reverses_out + chops
print(f"\n  In the 30 minutes AFTER acceptance:")
print(f"    Continues deeper into VA:  {continues_deeper:3d} ({continues_deeper/total*100:.0f}%)")
print(f"    Reverses back out of VA:   {reverses_out:3d} ({reverses_out/total*100:.0f}%)")
print(f"    Chops (no clear direction): {chops:3d} ({chops/total*100:.0f}%)")


# ============================================================================
# THE REAL QUESTION: Is acceptance just a coin flip?
# ============================================================================
print(f"\n{'='*100}")
print(f"  ACCEPTANCE vs NO ACCEPTANCE — Does the filter add edge?")
print(f"{'='*100}")

# Compare: all setups where price touches VA (no acceptance needed) vs accepted only
for target_mode in ['opposite_va', '4R']:
    print(f"\n  ── Target: {target_mode} ──")

    # Immediate entry (any VA touch)
    results_imm = [replay_with_entry(s, 'immediate', target_mode) for s in setups]
    results_imm = [r for r in results_imm if r]

    # Acceptance entry
    results_acc = [replay_with_entry(s, 'acceptance', target_mode) for s in setups]
    results_acc = [r for r in results_acc if r]

    # Limit 50% entry
    results_50 = [replay_with_entry(s, 'limit_50', target_mode) for s in setups]
    results_50 = [r for r in results_50 if r]

    for label, results in [('VA touch (no filter)', results_imm),
                            ('Acceptance (current)', results_acc),
                            ('Limit 50% VA depth', results_50)]:
        if not results:
            continue
        df_r = pd.DataFrame(results)
        n = len(df_r)
        wr = (df_r['is_winner'].sum() / n) * 100
        total = df_r['pnl_dollars'].sum()
        pm = total / months
        stopped = (df_r['exit_reason'] == 'STOP').sum()
        target_hit = (df_r['exit_reason'] == 'TARGET').sum()
        eod = (df_r['exit_reason'] == 'EOD').sum()
        print(f"\n    {label}:")
        print(f"      Trades: {n}, {n/months:.1f}/mo")
        print(f"      WR: {wr:.1f}%, $/mo: ${pm:,.0f}")
        print(f"      Exits: TARGET={target_hit}, STOP={stopped}, EOD={eod}")
        print(f"      Avg risk: {df_r['risk_pts'].mean():.0f}p, "
              f"Avg R:R: {df_r['rr_ratio'].mean():.1f}")
        print(f"      Avg entry depth: {df_r['entry_depth'].mean():.2f}")


# ============================================================================
# LIMIT ENTRY + ACCEPTANCE COMBO
# ============================================================================
print(f"\n{'='*100}")
print(f"  COMBO: Acceptance filter + Limit entry at 35-50% VA depth")
print(f"{'='*100}")

for target_mode in ['opposite_va', '2R', '4R']:
    print(f"\n  ── Target: {target_mode} ──")

    for depth_pct in [0.25, 0.35, 0.50]:
        depth_key = f'depth_{int(depth_pct*100)}_bar'
        results = []

        for setup in setups:
            # Require acceptance AND depth
            if setup['acceptance_bar'] is None:
                continue
            if setup[depth_key] is None:
                continue

            # Use the later of acceptance bar or depth bar as entry
            entry_bar = max(setup['acceptance_bar'], setup[depth_key])

            direction = setup['direction']
            vah = setup['vah']
            val = setup['val']
            va_width = setup['va_width']
            post_ib = setup['post_ib']

            # Entry price at depth level
            if direction == 'LONG':
                entry_price = val + va_width * depth_pct
                stop_price = val - STOP_BUFFER
            else:
                entry_price = vah - va_width * depth_pct
                stop_price = vah + STOP_BUFFER

            risk_pts = abs(entry_price - stop_price)
            if risk_pts <= 0:
                continue

            if target_mode == 'opposite_va':
                target = vah if direction == 'LONG' else val
            elif target_mode.endswith('R'):
                r_mult = float(target_mode[:-1])
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
                continue

            # Replay from entry bar
            remaining = post_ib.iloc[entry_bar:]
            mfe = 0.0
            mae = 0.0
            exit_price = None
            exit_reason = None
            bars_held = 0

            for i in range(len(remaining)):
                bar = remaining.iloc[i]
                bars_held += 1
                bt_time = bar.get('timestamp')
                bt = bt_time.time() if bt_time and hasattr(bt_time, 'time') else None
                if bt and bt >= time(15, 30):
                    exit_price = bar['close']
                    exit_reason = 'EOD'
                    break

                if direction == 'LONG':
                    fav = bar['high'] - entry_price
                    adv = entry_price - bar['low']
                else:
                    fav = entry_price - bar['low']
                    adv = bar['high'] - entry_price
                mfe = max(mfe, fav)
                mae = max(mae, adv)

                if direction == 'LONG' and bar['low'] <= stop_price:
                    exit_price = stop_price
                    exit_reason = 'STOP'
                    break
                elif direction == 'SHORT' and bar['high'] >= stop_price:
                    exit_price = stop_price
                    exit_reason = 'STOP'
                    break

                if direction == 'LONG' and bar['high'] >= target:
                    exit_price = target
                    exit_reason = 'TARGET'
                    break
                elif direction == 'SHORT' and bar['low'] <= target:
                    exit_price = target
                    exit_reason = 'TARGET'
                    break

            if exit_price is None:
                exit_price = remaining.iloc[-1]['close']
                exit_reason = 'EOD'

            if direction == 'LONG':
                pnl_pts = exit_price - entry_price - SLIPPAGE_PTS
            else:
                pnl_pts = entry_price - exit_price - SLIPPAGE_PTS
            pnl_dollars = pnl_pts * TICK_VALUE * CONTRACTS - COMMISSION * CONTRACTS

            results.append({
                'pnl_dollars': pnl_dollars,
                'is_winner': pnl_dollars > 0,
                'exit_reason': exit_reason,
                'risk_pts': risk_pts,
                'mfe_pts': mfe,
                'mae_pts': mae,
                'entry_depth': depth_pct,
            })

        if not results:
            print(f"    Acc + limit_{int(depth_pct*100)}%: 0 trades")
            continue

        df_r = pd.DataFrame(results)
        n = len(df_r)
        wr = df_r['is_winner'].mean() * 100
        pm = df_r['pnl_dollars'].sum() / months
        stopped = (df_r['exit_reason'] == 'STOP').sum()
        target_hit = (df_r['exit_reason'] == 'TARGET').sum()
        eod = (df_r['exit_reason'] == 'EOD').sum()
        print(f"    Acc + limit_{int(depth_pct*100)}%: {n} trades, {n/months:.1f}/mo, "
              f"WR={wr:.1f}%, $/mo=${pm:,.0f}, "
              f"STOP={stopped} TARGET={target_hit} EOD={eod}, "
              f"risk={df_r['risk_pts'].mean():.0f}p")


# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*100}")
print(f"  DIAGNOSIS SUMMARY")
print(f"{'='*100}")

print(f"""
  THE ENTRY PROBLEM EXPLAINED:

  Current model: Wait for 1x 30-min close inside VA → enter at that close price

  What happens after acceptance:
    - {continues_deeper} of {total} ({continues_deeper/total*100:.0f}%) continue deeper into VA (good)
    - {reverses_out} of {total} ({reverses_out/total*100:.0f}%) reverse back OUT of VA within 30 min (bad)
    - {chops} of {total} ({chops/total*100:.0f}%) chop sideways near the edge

  This means acceptance is NOT a strong directional filter.
  ~{reverses_out/total*100:.0f}% of "accepted" trades are just temporary VA touches
  that immediately reverse — they're failed mean-reversion setups
  (or rather, continuation trades that briefly pulled back to VA).

  The acceptance bar fires when price is barely inside VA (entry depth ~0.32).
  A limit order deeper into the VA would:
    1. Skip the trades that only touch the edge and reverse
    2. Get a better entry price (closer to target, further from stop)
    3. Reduce trade count but improve WR and expectancy
""")


# ============================================================================
# ATR-BASED STOP ANALYSIS
# ============================================================================
print(f"\n{'='*100}")
print(f"  ATR-BASED STOP ANALYSIS — Should we use 1 ATR or 2 ATR instead of fixed buffer?")
print(f"{'='*100}")

# Compute prior session ATR for each setup
for setup in setups:
    session_date = setup['session_date']
    # Find prior session
    idx = None
    for j, s in enumerate(sessions):
        if str(s) == session_date:
            idx = j
            break
    if idx is None or idx < 1:
        setup['prior_atr'] = None
        continue

    prior = sessions[idx - 1]
    prior_df = df_rth[df_rth['session_date'] == prior]
    if len(prior_df) < 30:
        setup['prior_atr'] = None
        continue

    # ATR = average of bar (high - low) over prior session
    bar_ranges = prior_df['high'] - prior_df['low']
    setup['prior_atr'] = bar_ranges.mean()

    # Also compute session-level true range (session high - session low)
    setup['prior_session_range'] = prior_df['high'].max() - prior_df['low'].min()

# Show ATR stats
atrs = [s['prior_atr'] for s in setups if s['prior_atr'] is not None]
session_ranges = [s['prior_session_range'] for s in setups if s.get('prior_session_range') is not None]
print(f"\n  Prior session 1-min bar ATR:")
print(f"    Mean: {np.mean(atrs):.1f} pts")
print(f"    Median: {np.median(atrs):.1f} pts")
for pct in [25, 50, 75, 90]:
    print(f"    {pct}th pctile: {np.percentile(atrs, pct):.1f} pts")

print(f"\n  Prior session range (high - low):")
print(f"    Mean: {np.mean(session_ranges):.0f} pts")
print(f"    Median: {np.median(session_ranges):.0f} pts")

# Test different stop models with acceptance entry + opposite_va target
print(f"\n  Stop Model Comparison (acceptance entry, BOTH directions):")
print(f"  {'Stop Model':<35s} {'N':>4s} {'WR':>6s} {'PF':>6s} {'$/Mo':>9s} "
      f"{'AvgRisk':>7s} {'Stopped':>8s} {'Target':>7s}")
print(f"  {'-'*90}")

stop_models = [
    ('VA_edge + 10pt (current)', 'fixed_10'),
    ('VA_edge + 20pt', 'fixed_20'),
    ('VA_edge + 1 ATR', 'atr_1'),
    ('VA_edge + 1.5 ATR', 'atr_1.5'),
    ('VA_edge + 2 ATR', 'atr_2'),
    ('VA_edge + 0.5 session range', 'range_0.5'),
    ('Entry - 1 ATR', 'entry_atr_1'),
    ('Entry - 2 ATR', 'entry_atr_2'),
]

for label, stop_model in stop_models:
    results = []
    for setup in setups:
        if setup['acceptance_bar'] is None:
            continue
        if setup.get('prior_atr') is None:
            continue

        direction = setup['direction']
        post_ib = setup['post_ib']
        acc_bar = setup['acceptance_bar']
        entry_price = post_ib.iloc[acc_bar]['close']
        vah = setup['vah']
        val = setup['val']
        atr = setup['prior_atr']
        sess_range = setup.get('prior_session_range', atr * 10)

        # Compute stop based on model
        if stop_model == 'fixed_10':
            stop_price = val - 10 if direction == 'LONG' else vah + 10
        elif stop_model == 'fixed_20':
            stop_price = val - 20 if direction == 'LONG' else vah + 20
        elif stop_model == 'atr_1':
            stop_price = val - atr if direction == 'LONG' else vah + atr
        elif stop_model == 'atr_1.5':
            stop_price = val - atr * 1.5 if direction == 'LONG' else vah + atr * 1.5
        elif stop_model == 'atr_2':
            stop_price = val - atr * 2 if direction == 'LONG' else vah + atr * 2
        elif stop_model == 'range_0.5':
            stop_price = val - sess_range * 0.5 if direction == 'LONG' else vah + sess_range * 0.5
        elif stop_model == 'entry_atr_1':
            stop_price = entry_price - atr if direction == 'LONG' else entry_price + atr
        elif stop_model == 'entry_atr_2':
            stop_price = entry_price - atr * 2 if direction == 'LONG' else entry_price + atr * 2
        else:
            continue

        risk_pts = abs(entry_price - stop_price)
        if risk_pts <= 0:
            continue

        target = vah if direction == 'LONG' else val
        reward = abs(target - entry_price)
        if reward <= 0:
            continue

        remaining = post_ib.iloc[acc_bar:]
        exit_price = None
        exit_reason = None

        for i in range(len(remaining)):
            bar = remaining.iloc[i]
            bt_time = bar.get('timestamp')
            bt = bt_time.time() if bt_time and hasattr(bt_time, 'time') else None
            if bt and bt >= time(15, 30):
                exit_price = bar['close']
                exit_reason = 'EOD'
                break

            if direction == 'LONG' and bar['low'] <= stop_price:
                exit_price = stop_price
                exit_reason = 'STOP'
                break
            elif direction == 'SHORT' and bar['high'] >= stop_price:
                exit_price = stop_price
                exit_reason = 'STOP'
                break

            if direction == 'LONG' and bar['high'] >= target:
                exit_price = target
                exit_reason = 'TARGET'
                break
            elif direction == 'SHORT' and bar['low'] <= target:
                exit_price = target
                exit_reason = 'TARGET'
                break

        if exit_price is None:
            exit_price = remaining.iloc[-1]['close']
            exit_reason = 'EOD'

        if direction == 'LONG':
            pnl_pts = exit_price - entry_price - SLIPPAGE_PTS
        else:
            pnl_pts = entry_price - exit_price - SLIPPAGE_PTS
        pnl_dollars = pnl_pts * TICK_VALUE * CONTRACTS - COMMISSION * CONTRACTS

        results.append({
            'pnl_dollars': pnl_dollars,
            'is_winner': pnl_dollars > 0,
            'exit_reason': exit_reason,
            'risk_pts': risk_pts,
        })

    if not results:
        print(f"  {label:<35s} 0")
        continue

    df_r = pd.DataFrame(results)
    n = len(df_r)
    wr = df_r['is_winner'].mean() * 100
    total_pnl = df_r['pnl_dollars'].sum()
    pm = total_pnl / months
    gw = df_r[df_r['is_winner']]['pnl_dollars'].sum()
    gl = abs(df_r[~df_r['is_winner']]['pnl_dollars'].sum())
    pf = gw / gl if gl > 0 else float('inf')
    stopped = (df_r['exit_reason'] == 'STOP').sum()
    target_hit = (df_r['exit_reason'] == 'TARGET').sum()
    avg_risk = df_r['risk_pts'].mean()

    print(f"  {label:<35s} {n:>4d} {wr:>5.1f}% {pf:>5.2f} "
          f"${pm:>7,.0f} {avg_risk:>5.0f}p "
          f"{stopped:>6d} {target_hit:>6d}")


# Also test ATR stops with limit_35 entry (best entry improvement candidate)
print(f"\n  ATR Stops + Limit 35% Entry (opposite_va target):")
print(f"  {'Stop Model':<35s} {'N':>4s} {'WR':>6s} {'PF':>6s} {'$/Mo':>9s} "
      f"{'AvgRisk':>7s} {'Stopped':>8s} {'Target':>7s}")
print(f"  {'-'*90}")

for label, stop_model in stop_models:
    results = []
    for setup in setups:
        if setup['depth_35_bar'] is None:
            continue
        if setup.get('prior_atr') is None:
            continue

        direction = setup['direction']
        post_ib = setup['post_ib']
        entry_bar = setup['depth_35_bar']
        vah = setup['vah']
        val = setup['val']
        va_width = setup['va_width']
        atr = setup['prior_atr']
        sess_range = setup.get('prior_session_range', atr * 10)

        if direction == 'LONG':
            entry_price = val + va_width * 0.35
        else:
            entry_price = vah - va_width * 0.35

        # Compute stop
        if stop_model == 'fixed_10':
            stop_price = val - 10 if direction == 'LONG' else vah + 10
        elif stop_model == 'fixed_20':
            stop_price = val - 20 if direction == 'LONG' else vah + 20
        elif stop_model == 'atr_1':
            stop_price = val - atr if direction == 'LONG' else vah + atr
        elif stop_model == 'atr_1.5':
            stop_price = val - atr * 1.5 if direction == 'LONG' else vah + atr * 1.5
        elif stop_model == 'atr_2':
            stop_price = val - atr * 2 if direction == 'LONG' else vah + atr * 2
        elif stop_model == 'range_0.5':
            stop_price = val - sess_range * 0.5 if direction == 'LONG' else vah + sess_range * 0.5
        elif stop_model == 'entry_atr_1':
            stop_price = entry_price - atr if direction == 'LONG' else entry_price + atr
        elif stop_model == 'entry_atr_2':
            stop_price = entry_price - atr * 2 if direction == 'LONG' else entry_price + atr * 2
        else:
            continue

        risk_pts = abs(entry_price - stop_price)
        if risk_pts <= 0:
            continue

        target = vah if direction == 'LONG' else val
        reward = abs(target - entry_price)
        if reward <= 0:
            continue

        remaining = post_ib.iloc[entry_bar:]
        exit_price = None
        exit_reason = None

        for i in range(len(remaining)):
            bar = remaining.iloc[i]
            bt_time = bar.get('timestamp')
            bt = bt_time.time() if bt_time and hasattr(bt_time, 'time') else None
            if bt and bt >= time(15, 30):
                exit_price = bar['close']
                exit_reason = 'EOD'
                break
            if direction == 'LONG' and bar['low'] <= stop_price:
                exit_price = stop_price
                exit_reason = 'STOP'
                break
            elif direction == 'SHORT' and bar['high'] >= stop_price:
                exit_price = stop_price
                exit_reason = 'STOP'
                break
            if direction == 'LONG' and bar['high'] >= target:
                exit_price = target
                exit_reason = 'TARGET'
                break
            elif direction == 'SHORT' and bar['low'] <= target:
                exit_price = target
                exit_reason = 'TARGET'
                break

        if exit_price is None:
            exit_price = remaining.iloc[-1]['close']
            exit_reason = 'EOD'

        if direction == 'LONG':
            pnl_pts = exit_price - entry_price - SLIPPAGE_PTS
        else:
            pnl_pts = entry_price - exit_price - SLIPPAGE_PTS
        pnl_dollars = pnl_pts * TICK_VALUE * CONTRACTS - COMMISSION * CONTRACTS

        results.append({
            'pnl_dollars': pnl_dollars,
            'is_winner': pnl_dollars > 0,
            'exit_reason': exit_reason,
            'risk_pts': risk_pts,
        })

    if not results:
        print(f"  {label:<35s} 0")
        continue

    df_r = pd.DataFrame(results)
    n = len(df_r)
    wr = df_r['is_winner'].mean() * 100
    total_pnl = df_r['pnl_dollars'].sum()
    pm = total_pnl / months
    gw = df_r[df_r['is_winner']]['pnl_dollars'].sum()
    gl = abs(df_r[~df_r['is_winner']]['pnl_dollars'].sum())
    pf = gw / gl if gl > 0 else float('inf')
    stopped = (df_r['exit_reason'] == 'STOP').sum()
    target_hit = (df_r['exit_reason'] == 'TARGET').sum()
    avg_risk = df_r['risk_pts'].mean()

    print(f"  {label:<35s} {n:>4d} {wr:>5.1f}% {pf:>5.2f} "
          f"${pm:>7,.0f} {avg_risk:>5.0f}p "
          f"{stopped:>6d} {target_hit:>6d}")


print(f"\n{'='*100}")
print(f"  ANALYSIS COMPLETE")
print(f"{'='*100}")
