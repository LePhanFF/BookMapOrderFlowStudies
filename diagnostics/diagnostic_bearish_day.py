"""
Bearish Trend Day Study

Two angles:
1. SHORTING the bearish move (AM breakdown below IBL)
   - Fast, furious selloff days
   - Entry: IBL acceptance (2+ bars below), VWAP rejection short

2. PM RECOVERY LONG after bearish exhaustion
   - After a big down move, price overshoots and retraces in PM
   - Entry: VWAP reclaim, VAL test, IB extension retest from below
   - The thesis: fast selloffs overshoot, then mean revert

Also study: weak trend_up days that fade in PM back to VWAP/VAL.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import time as dt_time, timedelta
from collections import Counter

project_root = Path(__file__).resolve().parent

import sys
sys.path.insert(0, str(project_root))
from data.loader import load_csv
from data.session import filter_rth


def load_data():
    nq = load_csv('NQ')
    es = load_csv('ES')
    return nq, es


def compute_sessions(nq_rth):
    """Build comprehensive session context for every session."""
    sessions = {}

    for session_date, group in nq_rth.groupby('session_date'):
        group = group.sort_values('timestamp')
        date_str = str(session_date)[:10]

        # IB period
        ib = group[group['time'] <= dt_time(10, 30)]
        if len(ib) < 10:
            continue

        ib_high = ib['high'].max()
        ib_low = ib['low'].min()
        ib_range = ib_high - ib_low
        ib_mid = (ib_high + ib_low) / 2

        if ib_range < 15:
            continue

        # Post-IB
        post_ib = group[group['time'] > dt_time(10, 30)]
        if len(post_ib) == 0:
            continue

        session_high = group['high'].max()
        session_low = group['low'].min()
        session_close = group.iloc[-1]['close']
        session_open = group.iloc[0]['open']

        # Extension calculations
        ext_up = (session_high - ib_high) / ib_range if ib_range > 0 else 0
        ext_down = (ib_low - session_low) / ib_range if ib_range > 0 else 0

        # Net direction
        net_move = session_close - session_open

        # VWAP at various points
        vwap_cols = group[group['vwap'].notna()]
        vwap_at_noon = None
        vwap_at_pm = None
        noon_bars = vwap_cols[vwap_cols['time'] == dt_time(12, 0)]
        if len(noon_bars) > 0:
            vwap_at_noon = noon_bars.iloc[0]['vwap']
        pm_bars = vwap_cols[vwap_cols['time'] == dt_time(13, 0)]
        if len(pm_bars) > 0:
            vwap_at_pm = pm_bars.iloc[0]['vwap']

        # Day type classification (both directions)
        if ext_up > 1.0:
            day_type = 'trend_up'
        elif ext_down > 1.0:
            day_type = 'trend_down'
        elif ext_up > 0.5:
            day_type = 'p_day'
        elif ext_down > 0.5:
            day_type = 'b_day_bear'  # bearish skew
        elif ext_up > 0.2 or ext_down > 0.2:
            day_type = 'b_day'
        else:
            day_type = 'neutral'

        # AM vs PM analysis
        am = group[group['time'] <= dt_time(12, 0)]
        pm = group[group['time'] > dt_time(12, 0)]

        am_low = am['low'].min() if len(am) > 0 else ib_low
        am_high = am['high'].max() if len(am) > 0 else ib_high
        pm_low = pm['low'].min() if len(pm) > 0 else session_low
        pm_high = pm['high'].max() if len(pm) > 0 else session_high
        pm_close = pm.iloc[-1]['close'] if len(pm) > 0 else session_close

        # Bearish move size
        max_drop_from_ib = ib_low - session_low  # positive = how far below IBL

        # PM recovery (for bearish days)
        if max_drop_from_ib > 0:
            pm_recovery = pm_close - session_low  # how much price recovered from the low
            pm_recovery_pct = pm_recovery / max_drop_from_ib if max_drop_from_ib > 10 else 0
        else:
            pm_recovery = 0
            pm_recovery_pct = 0

        # Check: did price accept below IBL? (2+ bars close below)
        post_ib_bars = post_ib.sort_values('timestamp')
        consec_below = 0
        max_consec_below = 0
        acceptance_below_time = None
        for _, bar in post_ib_bars.iterrows():
            if bar['close'] < ib_low:
                consec_below += 1
                if consec_below >= 2 and acceptance_below_time is None:
                    acceptance_below_time = bar['timestamp']
                max_consec_below = max(max_consec_below, consec_below)
            else:
                consec_below = 0

        accepted_below = max_consec_below >= 2

        # Check: did price touch VWAP in PM after going below IBL?
        vwap_reclaim_pm = False
        vwap_touch_pm_time = None
        if accepted_below and len(pm) > 0:
            for _, bar in pm.sort_values('timestamp').iterrows():
                vwap = bar.get('vwap')
                if vwap and not pd.isna(vwap):
                    if bar['low'] <= vwap <= bar['high'] or bar['close'] > vwap:
                        vwap_reclaim_pm = True
                        vwap_touch_pm_time = bar['timestamp']
                        break

        sessions[date_str] = {
            'date': date_str,
            'ib_high': ib_high,
            'ib_low': ib_low,
            'ib_range': ib_range,
            'ib_mid': ib_mid,
            'day_type': day_type,
            'ext_up': ext_up,
            'ext_down': ext_down,
            'session_high': session_high,
            'session_low': session_low,
            'session_close': session_close,
            'net_move': net_move,
            'max_drop_from_ib': max_drop_from_ib,
            'pm_recovery': pm_recovery,
            'pm_recovery_pct': pm_recovery_pct,
            'accepted_below': accepted_below,
            'acceptance_below_time': acceptance_below_time,
            'vwap_reclaim_pm': vwap_reclaim_pm,
            'vwap_touch_pm_time': vwap_touch_pm_time,
            'am_low': am_low,
            'am_high': am_high,
            'pm_low': pm_low,
            'pm_high': pm_high,
            'pm_close': pm_close,
            'vwap_at_noon': vwap_at_noon,
            'vwap_at_pm': vwap_at_pm,
            'bars': group,
            'post_ib': post_ib,
        }

    return sessions


def study_bearish_days(sessions, nq_rth):
    """Study all sessions with downside extension."""

    print('\n' + '=' * 90)
    print('  PHASE 1: IDENTIFYING BEARISH SESSIONS')
    print('=' * 90)

    # All sessions with meaningful downside extension (>0.3x IB below IBL)
    bearish = {k: v for k, v in sessions.items() if v['ext_down'] > 0.3}
    strong_bear = {k: v for k, v in sessions.items() if v['ext_down'] > 0.5}
    trend_down = {k: v for k, v in sessions.items() if v['ext_down'] > 1.0}

    print(f'All sessions: {len(sessions)}')
    print(f'Bearish (ext_down > 0.3x): {len(bearish)}')
    print(f'Strong bearish (ext_down > 0.5x): {len(strong_bear)}')
    print(f'Trend down (ext_down > 1.0x): {len(trend_down)}')

    # Day type distribution including bearish
    all_types = Counter(s['day_type'] for s in sessions.values())
    print(f'\nDay type distribution:')
    for dt, count in all_types.most_common():
        print(f'  {dt}: {count} ({count/len(sessions)*100:.0f}%)')

    print(f'\n--- BEARISH SESSION DETAILS (ext_down > 0.3x) ---')
    print(f'{"Date":<12s} {"Type":<12s} {"Ext Down":>8s} {"Drop pts":>9s} {"PM Recov":>9s} {"Recov%":>7s} {"Accept?":>8s} {"VWAP PM?":>9s} {"Net Move":>9s}')
    print('-' * 100)

    for date_str in sorted(bearish.keys()):
        s = bearish[date_str]
        print(f'{date_str:<12s} {s["day_type"]:<12s} {s["ext_down"]:>7.2f}x {s["max_drop_from_ib"]:>8.0f}pts '
              f'{s["pm_recovery"]:>8.0f}pts {s["pm_recovery_pct"]:>6.0%} '
              f'{"YES" if s["accepted_below"] else "no":>8s} '
              f'{"YES" if s["vwap_reclaim_pm"] else "no":>9s} '
              f'{s["net_move"]:>+8.0f}pts')

    return bearish, strong_bear, trend_down


def simulate_bear_short(sessions, nq_rth):
    """Study 1: Shorting the bearish move - IBL acceptance short."""

    print('\n' + '=' * 90)
    print('  STUDY 1: SHORTING THE BEARISH MOVE (IBL ACCEPTANCE SHORT)')
    print('=' * 90)

    bearish_sessions = {k: v for k, v in sessions.items() if v['accepted_below']}
    print(f'Sessions with IBL acceptance: {len(bearish_sessions)}')

    results = []

    for date_str, s in sorted(bearish_sessions.items()):
        bars = s['bars'].sort_values('timestamp')
        ib_low = s['ib_low']
        ib_high = s['ib_high']
        ib_range = s['ib_range']
        ib_mid = s['ib_mid']
        accept_time = s['acceptance_below_time']

        if accept_time is None:
            continue

        # === MODEL A: VWAP Rejection Short ===
        # After acceptance below IBL, wait for price to pull back to VWAP and reject
        post_accept = bars[bars['timestamp'] > accept_time]

        for _, bar in post_accept.iterrows():
            if bar['time'] >= dt_time(14, 0):  # Time cutoff
                break

            vwap = bar.get('vwap')
            if vwap is None or pd.isna(vwap):
                continue

            # Price near VWAP from below and rejecting (close < VWAP, delta < 0)
            delta = bar.get('vol_delta', 0)
            if pd.isna(delta):
                delta = 0

            vwap_dist = abs(bar['close'] - vwap) / ib_range if ib_range > 0 else 999

            if vwap_dist < 0.30 and bar['close'] < vwap and delta < 0:
                entry_price = bar['close']
                stop_price = vwap + ib_range * 0.30  # Stop above VWAP
                target_price = s['session_low'] + ib_range * 0.10  # Near session low

                risk = stop_price - entry_price
                reward = entry_price - target_price
                if risk <= 5 or reward <= 5:
                    continue

                # Simulate
                remaining = bars[bars['timestamp'] > bar['timestamp']]
                session_end_bars = remaining[remaining['time'] <= dt_time(15, 30)]

                exit_price = entry_price
                exit_reason = 'NO_BARS'
                mfe = 0

                for _, rb in session_end_bars.iterrows():
                    mfe = max(mfe, entry_price - rb['low'])
                    if rb['high'] >= stop_price:
                        exit_price = stop_price
                        exit_reason = 'STOP'
                        break
                    if rb['low'] <= target_price:
                        exit_price = target_price
                        exit_reason = 'TARGET'
                        break
                else:
                    if len(session_end_bars) > 0:
                        exit_price = session_end_bars.iloc[-1]['close']
                        exit_reason = 'EOD'

                pnl_pts = entry_price - exit_price  # SHORT

                results.append({
                    'date': date_str,
                    'model': 'VWAP_REJECTION_SHORT',
                    'entry_time': str(bar['timestamp']),
                    'bar_time': bar['time'],
                    'entry_price': entry_price,
                    'stop_price': stop_price,
                    'target_price': target_price,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'pnl_pts': pnl_pts,
                    'risk_pts': risk,
                    'reward_pts': reward,
                    'rr': reward / risk if risk > 0 else 0,
                    'mfe_pts': mfe,
                    'day_type': s['day_type'],
                    'ext_down': s['ext_down'],
                })
                break  # One trade per session per model

        # === MODEL B: IBL Break + Retest Short ===
        # Price breaks below IBL, retests IBL from below, short the rejection
        for _, bar in post_accept.iterrows():
            if bar['time'] >= dt_time(13, 0):
                break

            # Price approaching IBL from below (close within 0.2x IB of IBL)
            dist_to_ibl = (ib_low - bar['close']) / ib_range if ib_range > 0 else 999

            delta = bar.get('vol_delta', 0)
            if pd.isna(delta):
                delta = 0

            if 0 < dist_to_ibl < 0.20 and bar['high'] >= ib_low - ib_range * 0.05 and delta < 0:
                entry_price = bar['close']
                stop_price = ib_low + ib_range * 0.20  # Stop above IBL
                target_price = entry_price - ib_range * 0.50  # 0.5x IB range target

                risk = stop_price - entry_price
                reward = entry_price - target_price
                if risk <= 5 or reward <= 5:
                    continue

                remaining = bars[bars['timestamp'] > bar['timestamp']]
                session_end_bars = remaining[remaining['time'] <= dt_time(15, 30)]

                exit_price = entry_price
                exit_reason = 'NO_BARS'
                mfe = 0

                for _, rb in session_end_bars.iterrows():
                    mfe = max(mfe, entry_price - rb['low'])
                    if rb['high'] >= stop_price:
                        exit_price = stop_price
                        exit_reason = 'STOP'
                        break
                    if rb['low'] <= target_price:
                        exit_price = target_price
                        exit_reason = 'TARGET'
                        break
                else:
                    if len(session_end_bars) > 0:
                        exit_price = session_end_bars.iloc[-1]['close']
                        exit_reason = 'EOD'

                pnl_pts = entry_price - exit_price

                results.append({
                    'date': date_str,
                    'model': 'IBL_RETEST_SHORT',
                    'entry_time': str(bar['timestamp']),
                    'bar_time': bar['time'],
                    'entry_price': entry_price,
                    'stop_price': stop_price,
                    'target_price': target_price,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'pnl_pts': pnl_pts,
                    'risk_pts': risk,
                    'reward_pts': reward,
                    'rr': reward / risk if risk > 0 else 0,
                    'mfe_pts': mfe,
                    'day_type': s['day_type'],
                    'ext_down': s['ext_down'],
                })
                break

    if results:
        rdf = pd.DataFrame(results)
        for model in rdf['model'].unique():
            sub = rdf[rdf['model'] == model]
            wins = sub[sub['pnl_pts'] > 0]
            print(f'\n--- {model} ---')
            print(f'Trades: {len(sub)}, Wins: {len(wins)}, WR: {len(wins)/len(sub)*100:.0f}%')
            print(f'Net: {sub["pnl_pts"].sum():+.1f} pts, Avg: {sub["pnl_pts"].mean():+.1f} pts/trade')
            print(f'Avg MFE: {sub["mfe_pts"].mean():.1f} pts')
            for _, t in sub.sort_values('date').iterrows():
                status = 'WIN' if t['pnl_pts'] > 0 else 'LOSS'
                print(f'  {t["date"]} {str(t["bar_time"])[:5]} {t["day_type"]:>11s} ext={t["ext_down"]:.1f}x | '
                      f'{t["exit_reason"]:>6s} {t["pnl_pts"]:+7.1f}pts (MFE {t["mfe_pts"]:.0f}) [{status}]')
    else:
        print('No bear short trades found')

    return results


def simulate_pm_recovery_long(sessions, nq_rth):
    """Study 2: PM recovery LONG after bearish exhaustion."""

    print('\n' + '=' * 90)
    print('  STUDY 2: PM RECOVERY LONG (AFTER BEARISH EXHAUSTION)')
    print('=' * 90)

    # Sessions with meaningful down move (ext_down > 0.3x)
    bearish = {k: v for k, v in sessions.items() if v['ext_down'] > 0.3}
    print(f'Bearish sessions (ext_down > 0.3x): {len(bearish)}')

    # How many of these recover in PM?
    recovers = {k: v for k, v in bearish.items() if v['pm_recovery_pct'] > 0.30}
    print(f'Sessions with 30%+ PM recovery: {len(recovers)} ({len(recovers)/len(bearish)*100:.0f}%)')

    results = []

    for date_str, s in sorted(bearish.items()):
        bars = s['bars'].sort_values('timestamp')
        ib_low = s['ib_low']
        ib_high = s['ib_high']
        ib_range = s['ib_range']
        ib_mid = s['ib_mid']

        # PM bars (after 12:00)
        pm_bars = bars[(bars['time'] >= dt_time(12, 0)) & (bars['time'] <= dt_time(15, 20))]

        if len(pm_bars) == 0:
            continue

        # Find session low before PM
        am_bars = bars[bars['time'] < dt_time(12, 0)]
        if len(am_bars) == 0:
            continue

        am_session_low = am_bars['low'].min()

        # === MODEL A: VWAP Reclaim Long ===
        # Price was below VWAP all morning, now reclaims above VWAP in PM
        for _, bar in pm_bars.iterrows():
            if bar['time'] >= dt_time(14, 30):
                break

            vwap = bar.get('vwap')
            if vwap is None or pd.isna(vwap):
                continue

            delta = bar.get('vol_delta', 0)
            if pd.isna(delta):
                delta = 0

            # Price crossing above VWAP with positive delta
            if bar['close'] > vwap and bar['open'] < vwap and delta > 0:
                entry_price = bar['close']
                stop_price = am_session_low - ib_range * 0.10  # Stop below session low
                # Target: VWAP + partial recovery toward IB
                target_price = vwap + ib_range * 0.50  # Half IB range above VWAP

                risk = entry_price - stop_price
                reward = target_price - entry_price
                if risk <= 5 or reward <= 5 or reward / risk < 0.8:
                    continue

                remaining = bars[bars['timestamp'] > bar['timestamp']]
                session_end_bars = remaining[remaining['time'] <= dt_time(15, 30)]

                exit_price = entry_price
                exit_reason = 'NO_BARS'
                mfe = 0

                for _, rb in session_end_bars.iterrows():
                    mfe = max(mfe, rb['high'] - entry_price)
                    if rb['low'] <= stop_price:
                        exit_price = stop_price
                        exit_reason = 'STOP'
                        break
                    if rb['high'] >= target_price:
                        exit_price = target_price
                        exit_reason = 'TARGET'
                        break
                else:
                    if len(session_end_bars) > 0:
                        exit_price = session_end_bars.iloc[-1]['close']
                        exit_reason = 'EOD'

                pnl_pts = exit_price - entry_price  # LONG

                results.append({
                    'date': date_str,
                    'model': 'PM_VWAP_RECLAIM_LONG',
                    'entry_time': str(bar['timestamp']),
                    'bar_time': bar['time'],
                    'entry_price': entry_price,
                    'stop_price': stop_price,
                    'target_price': target_price,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'pnl_pts': pnl_pts,
                    'risk_pts': risk,
                    'reward_pts': reward,
                    'rr': reward / risk if risk > 0 else 0,
                    'mfe_pts': mfe,
                    'day_type': s['day_type'],
                    'ext_down': s['ext_down'],
                    'pm_recovery_pct': s['pm_recovery_pct'],
                })
                break

        # === MODEL B: Session Low Bounce Long ===
        # Price reaches session low area, bounces with delta, long for recovery
        for _, bar in pm_bars.iterrows():
            if bar['time'] >= dt_time(14, 0):
                break

            delta = bar.get('vol_delta', 0)
            if pd.isna(delta):
                delta = 0

            # Price near session low (within 0.15x IB) and bouncing
            dist_from_low = bar['close'] - am_session_low
            if 0 < dist_from_low < ib_range * 0.25 and delta > 0 and bar['close'] > bar['open']:
                entry_price = bar['close']
                stop_price = am_session_low - ib_range * 0.10
                # Target: IBL or VWAP (whichever is closer)
                vwap = bar.get('vwap', ib_low)
                if pd.isna(vwap):
                    vwap = ib_low
                target_price = min(ib_low, vwap)  # Closer target

                risk = entry_price - stop_price
                reward = target_price - entry_price
                if risk <= 5 or reward <= 5 or reward / risk < 0.5:
                    continue

                remaining = bars[bars['timestamp'] > bar['timestamp']]
                session_end_bars = remaining[remaining['time'] <= dt_time(15, 30)]

                exit_price = entry_price
                exit_reason = 'NO_BARS'
                mfe = 0

                for _, rb in session_end_bars.iterrows():
                    mfe = max(mfe, rb['high'] - entry_price)
                    if rb['low'] <= stop_price:
                        exit_price = stop_price
                        exit_reason = 'STOP'
                        break
                    if rb['high'] >= target_price:
                        exit_price = target_price
                        exit_reason = 'TARGET'
                        break
                else:
                    if len(session_end_bars) > 0:
                        exit_price = session_end_bars.iloc[-1]['close']
                        exit_reason = 'EOD'

                pnl_pts = exit_price - entry_price

                results.append({
                    'date': date_str,
                    'model': 'SESSION_LOW_BOUNCE_LONG',
                    'entry_time': str(bar['timestamp']),
                    'bar_time': bar['time'],
                    'entry_price': entry_price,
                    'stop_price': stop_price,
                    'target_price': target_price,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'pnl_pts': pnl_pts,
                    'risk_pts': risk,
                    'reward_pts': reward,
                    'rr': reward / risk if risk > 0 else 0,
                    'mfe_pts': mfe,
                    'day_type': s['day_type'],
                    'ext_down': s['ext_down'],
                    'pm_recovery_pct': s['pm_recovery_pct'],
                })
                break

        # === MODEL C: Breakdown Area Reclaim Long ===
        # Price broke below IBL in AM, now in PM reclaims back above IBL
        if s['accepted_below']:
            for _, bar in pm_bars.iterrows():
                if bar['time'] >= dt_time(14, 30):
                    break

                delta = bar.get('vol_delta', 0)
                if pd.isna(delta):
                    delta = 0

                # Close above IBL after having been below
                if bar['close'] > ib_low and bar['open'] < ib_low and delta > 0:
                    entry_price = bar['close']
                    stop_price = am_session_low - ib_range * 0.05
                    target_price = ib_mid  # Target IB midpoint

                    risk = entry_price - stop_price
                    reward = target_price - entry_price
                    if risk <= 5 or reward <= 5:
                        continue

                    remaining = bars[bars['timestamp'] > bar['timestamp']]
                    session_end_bars = remaining[remaining['time'] <= dt_time(15, 30)]

                    exit_price = entry_price
                    exit_reason = 'NO_BARS'
                    mfe = 0

                    for _, rb in session_end_bars.iterrows():
                        mfe = max(mfe, rb['high'] - entry_price)
                        if rb['low'] <= stop_price:
                            exit_price = stop_price
                            exit_reason = 'STOP'
                            break
                        if rb['high'] >= target_price:
                            exit_price = target_price
                            exit_reason = 'TARGET'
                            break
                    else:
                        if len(session_end_bars) > 0:
                            exit_price = session_end_bars.iloc[-1]['close']
                            exit_reason = 'EOD'

                    pnl_pts = exit_price - entry_price

                    results.append({
                        'date': date_str,
                        'model': 'IBL_RECLAIM_LONG',
                        'entry_time': str(bar['timestamp']),
                        'bar_time': bar['time'],
                        'entry_price': entry_price,
                        'stop_price': stop_price,
                        'target_price': target_price,
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'pnl_pts': pnl_pts,
                        'risk_pts': risk,
                        'reward_pts': reward,
                        'rr': reward / risk if risk > 0 else 0,
                        'mfe_pts': mfe,
                        'day_type': s['day_type'],
                        'ext_down': s['ext_down'],
                        'pm_recovery_pct': s['pm_recovery_pct'],
                    })
                    break

    if results:
        rdf = pd.DataFrame(results)
        print(f'\nTotal PM recovery trades: {len(rdf)}')

        for model in rdf['model'].unique():
            sub = rdf[rdf['model'] == model]
            wins = sub[sub['pnl_pts'] > 0]
            print(f'\n--- {model} ---')
            print(f'Trades: {len(sub)}, Wins: {len(wins)}, WR: {len(wins)/len(sub)*100:.0f}%')
            print(f'Net: {sub["pnl_pts"].sum():+.1f} pts, Avg: {sub["pnl_pts"].mean():+.1f} pts/trade')
            print(f'Avg MFE: {sub["mfe_pts"].mean():.1f} pts, Avg R:R planned: {sub["rr"].mean():.2f}')
            for _, t in sub.sort_values('date').iterrows():
                status = 'WIN' if t['pnl_pts'] > 0 else 'LOSS'
                print(f'  {t["date"]} {str(t["bar_time"])[:5]} {t["day_type"]:>11s} ext_down={t["ext_down"]:.1f}x pm_recov={t["pm_recovery_pct"]:.0%} | '
                      f'{t["exit_reason"]:>6s} {t["pnl_pts"]:+7.1f}pts (MFE {t["mfe_pts"]:.0f}) [{status}]')
    else:
        print('No PM recovery trades found')

    return results


def study_weak_trend_up_fade(sessions, nq_rth):
    """Study 3: Weak trend_up days that fade in PM."""

    print('\n' + '=' * 90)
    print('  STUDY 3: WEAK TREND UP DAYS THAT FADE IN PM')
    print('=' * 90)

    # p_day or weak trend_up sessions
    weak_up = {k: v for k, v in sessions.items()
               if v['day_type'] in ('p_day', 'trend_up') and v['ext_up'] < 1.5}

    print(f'Weak uptrend sessions (p_day + trend_up ext < 1.5x): {len(weak_up)}')

    for date_str in sorted(weak_up.keys()):
        s = weak_up[date_str]
        # Did price return to VWAP in PM?
        bars = s['bars']
        pm_bars = bars[(bars['time'] >= dt_time(13, 0)) & (bars['time'] <= dt_time(15, 30))]

        if len(pm_bars) == 0:
            continue

        # Check if PM low went below VWAP
        pm_below_vwap = False
        for _, bar in pm_bars.iterrows():
            vwap = bar.get('vwap')
            if vwap and not pd.isna(vwap):
                if bar['low'] < vwap:
                    pm_below_vwap = True
                    break

        # How far did PM retrace from the high?
        session_high = s['session_high']
        pm_low = s['pm_low']
        retrace = session_high - pm_low
        retrace_pct = retrace / s['ib_range'] if s['ib_range'] > 0 else 0

        pm_close_vs_vwap = ''
        if s['vwap_at_pm']:
            if s['pm_close'] > s['vwap_at_pm']:
                pm_close_vs_vwap = 'above VWAP'
            else:
                pm_close_vs_vwap = 'BELOW VWAP'

        print(f'  {date_str} {s["day_type"]:>10s} ext_up={s["ext_up"]:.1f}x | '
              f'PM retrace: {retrace:.0f}pts ({retrace_pct:.1f}x IB) | '
              f'PM below VWAP: {"YES" if pm_below_vwap else "no":>3s} | {pm_close_vs_vwap}')


def study_mfe_on_bearish(sessions, nq_rth):
    """Study raw MFE: on bearish days, how much does price actually move?"""

    print('\n' + '=' * 90)
    print('  STUDY 4: RAW OPPORTUNITY ON BEARISH DAYS')
    print('=' * 90)

    bearish = {k: v for k, v in sessions.items() if v['ext_down'] > 0.3}

    print(f'\n{"Date":<12s} {"Type":<12s} {"IB Rng":>7s} {"Drop":>7s} {"PM Recov":>9s} {"Short MFE":>10s} {"Long MFE":>10s} {"Opp Short":>10s} {"Opp Long":>10s}')
    print('-' * 100)

    total_short_opp = 0
    total_long_opp = 0

    for date_str in sorted(bearish.keys()):
        s = bearish[date_str]
        bars = s['bars']

        # Short opportunity: max drop from IBL to session low (points)
        short_mfe = s['max_drop_from_ib']

        # Long opportunity in PM: recovery from session low
        long_mfe = s['pm_recovery']

        # Realistic capture (50% of MFE)
        short_opp = short_mfe * 0.50
        long_opp = long_mfe * 0.50

        total_short_opp += short_opp
        total_long_opp += long_opp

        print(f'{date_str:<12s} {s["day_type"]:<12s} {s["ib_range"]:>6.0f}pts {s["max_drop_from_ib"]:>6.0f}pts '
              f'{s["pm_recovery"]:>8.0f}pts '
              f'{short_mfe:>9.0f}pts {long_mfe:>9.0f}pts '
              f'{short_opp:>9.0f}pts {long_opp:>9.0f}pts')

    print(f'\n  Total short opportunity (50% MFE): {total_short_opp:.0f} pts = ${total_short_opp * 2:,.0f} (1 MNQ)')
    print(f'  Total long opportunity (50% MFE):  {total_long_opp:.0f} pts = ${total_long_opp * 2:,.0f} (1 MNQ)')
    print(f'  Combined opportunity: {total_short_opp + total_long_opp:.0f} pts = ${(total_short_opp + total_long_opp) * 2:,.0f}')


def study_htf_context(sessions, nq):
    """Study 5: HTF (Daily/4H) context for bearish setups.

    ERL -> IRL framework:
    - External Range Liquidity (ERL): Previous day high/low, swing high/low
    - Internal Range Liquidity (IRL): FVG zones, order blocks, VWAP

    The thesis: after price sweeps an ERL level on the daily/4H chart,
    it should target IRL (FVG fill, order block) on the other side.

    If daily/4H shows bearish structure (sweep of PDH then rejection,
    or price inside a 4H bearish FVG), the intraday bearish move has
    HTF backing and is more likely to follow through.
    """
    print('\n' + '=' * 90)
    print('  STUDY 5: HTF (DAILY/4H) ERL -> IRL CONTEXT')
    print('=' * 90)

    # Build daily OHLC
    rth_mask = (nq['time'] >= dt_time(9, 30)) & (nq['time'] <= dt_time(16, 0))
    nq_rth_all = nq[rth_mask].copy()

    daily = nq_rth_all.groupby('session_date').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        'volume': 'sum',
    }).sort_index()

    print(f'Daily bars: {len(daily)}')

    # Build 4H OHLC from RTH data
    nq_rth_indexed = nq_rth_all.set_index('timestamp').sort_index()
    h4_high = nq_rth_indexed['high'].resample('4h').max()
    h4_low = nq_rth_indexed['low'].resample('4h').min()
    h4_open = nq_rth_indexed['open'].resample('4h').first()
    h4_close = nq_rth_indexed['close'].resample('4h').last()

    h4 = pd.DataFrame({
        'open': h4_open, 'high': h4_high,
        'low': h4_low, 'close': h4_close,
    }).dropna()
    print(f'4H bars: {len(h4)}')

    # Find daily bearish FVGs
    daily_bear_fvgs = []
    for i in range(2, len(daily)):
        d2 = daily.iloc[i - 2]
        d0 = daily.iloc[i]
        # Bearish FVG: candle 2-ago low > current candle high (gap down)
        if d2['low'] > d0['high']:
            daily_bear_fvgs.append({
                'top': d2['low'], 'bottom': d0['high'],
                'mid': (d2['low'] + d0['high']) / 2,
                'date': daily.index[i],
                'size': d2['low'] - d0['high'],
            })
    # Bullish FVGs (for IRL target on shorts)
    daily_bull_fvgs = []
    for i in range(2, len(daily)):
        d2 = daily.iloc[i - 2]
        d0 = daily.iloc[i]
        if d0['low'] > d2['high']:
            daily_bull_fvgs.append({
                'top': d0['low'], 'bottom': d2['high'],
                'mid': (d0['low'] + d2['high']) / 2,
                'date': daily.index[i],
            })

    # Find 4H bearish FVGs
    h4_bear_fvgs = []
    for i in range(2, len(h4)):
        c2 = h4.iloc[i - 2]
        c0 = h4.iloc[i]
        if c2['low'] > c0['high']:
            h4_bear_fvgs.append({
                'top': c2['low'], 'bottom': c0['high'],
                'mid': (c2['low'] + c0['high']) / 2,
                'time': h4.index[i],
                'size': c2['low'] - c0['high'],
            })
    # 4H bullish FVGs
    h4_bull_fvgs = []
    for i in range(2, len(h4)):
        c2 = h4.iloc[i - 2]
        c0 = h4.iloc[i]
        if c0['low'] > c2['high']:
            h4_bull_fvgs.append({
                'top': c0['low'], 'bottom': c2['high'],
                'mid': (c0['low'] + c2['high']) / 2,
                'time': h4.index[i],
            })

    print(f'\nDaily bearish FVGs found: {len(daily_bear_fvgs)}')
    for f in daily_bear_fvgs:
        print(f'  {str(f["date"])[:10]}: {f["bottom"]:.0f} - {f["top"]:.0f} (size={f["size"]:.0f}pts)')

    print(f'\nDaily bullish FVGs found: {len(daily_bull_fvgs)}')
    for f in daily_bull_fvgs:
        print(f'  {str(f["date"])[:10]}: {f["bottom"]:.0f} - {f["top"]:.0f}')

    print(f'\n4H bearish FVGs found: {len(h4_bear_fvgs)}')
    for f in h4_bear_fvgs[-10:]:  # Last 10 only
        print(f'  {str(f["time"])[:16]}: {f["bottom"]:.0f} - {f["top"]:.0f} (size={f["size"]:.0f}pts)')

    # For each session, check HTF bearish context
    print(f'\n--- PER-SESSION HTF BEARISH CONTEXT ---')
    print(f'{"Date":<12s} {"Type":<12s} {"Ext Dn":>6s} {"PDH Sweep?":>10s} {"In 4H Bear FVG?":>16s} {"In Daily Bear FVG?":>18s} {"Below Daily Bull FVG?":>22s} {"Drop":>6s}')
    print('-' * 115)

    htf_bearish_sessions = []

    for date_str in sorted(sessions.keys()):
        s = sessions[date_str]
        session_date = pd.Timestamp(date_str)

        # Get previous day data
        prev_days = daily[daily.index < session_date].tail(3)
        if len(prev_days) == 0:
            continue

        prev_day = prev_days.iloc[-1]
        pdh = prev_day['high']
        pdl = prev_day['low']

        session_high = s['session_high']
        session_low = s['session_low']

        # Did price sweep PDH then reverse? (ERL sweep)
        swept_pdh = session_high >= pdh - 5  # Reached or exceeded PDH
        closed_below_pdh = s['session_close'] < pdh  # But closed below

        # Is price in a 4H bearish FVG zone?
        in_4h_bear_fvg = False
        for fvg in h4_bear_fvgs:
            fvg_time = fvg['time']
            # FVG must be BEFORE this session
            if fvg_time.date() < session_date.date():
                if fvg['bottom'] <= session_high <= fvg['top'] + 30:
                    in_4h_bear_fvg = True
                    break

        # Is price in a daily bearish FVG zone?
        in_daily_bear_fvg = False
        for fvg in daily_bear_fvgs:
            fvg_date = fvg['date']
            if fvg_date < session_date:
                if fvg['bottom'] <= session_high <= fvg['top'] + 30:
                    in_daily_bear_fvg = True
                    break

        # Is price approaching from above a daily bullish FVG? (IRL target)
        above_daily_bull_fvg = False
        nearest_bull_fvg = None
        for fvg in reversed(daily_bull_fvgs):
            if fvg['date'] < session_date and fvg['top'] < session_low + 200:
                above_daily_bull_fvg = True
                nearest_bull_fvg = fvg
                break

        has_htf_bear = (swept_pdh and closed_below_pdh) or in_4h_bear_fvg or in_daily_bear_fvg

        if s['ext_down'] > 0.2 or has_htf_bear:
            pdh_label = 'YES->rev' if (swept_pdh and closed_below_pdh) else ('touched' if swept_pdh else 'no')
            print(f'{date_str:<12s} {s["day_type"]:<12s} {s["ext_down"]:>5.1f}x '
                  f'{pdh_label:>10s} '
                  f'{"YES" if in_4h_bear_fvg else "no":>16s} '
                  f'{"YES" if in_daily_bear_fvg else "no":>18s} '
                  f'{"YES" if above_daily_bull_fvg else "no":>22s} '
                  f'{s["max_drop_from_ib"]:>5.0f}pts')

            if has_htf_bear and s['ext_down'] > 0.3:
                htf_bearish_sessions.append({
                    'date': date_str,
                    'day_type': s['day_type'],
                    'ext_down': s['ext_down'],
                    'drop': s['max_drop_from_ib'],
                    'pm_recovery': s['pm_recovery'],
                    'pm_recovery_pct': s['pm_recovery_pct'],
                    'swept_pdh': swept_pdh and closed_below_pdh,
                    'in_4h_fvg': in_4h_bear_fvg,
                    'in_daily_fvg': in_daily_bear_fvg,
                    'has_bull_fvg_target': above_daily_bull_fvg,
                })

    print(f'\n--- HTF-BACKED BEARISH SESSIONS (ext_down > 0.3x + HTF context) ---')
    print(f'Found: {len(htf_bearish_sessions)}')

    for s in htf_bearish_sessions:
        htf_reasons = []
        if s['swept_pdh']:
            htf_reasons.append('PDH sweep+fail')
        if s['in_4h_fvg']:
            htf_reasons.append('in 4H bear FVG')
        if s['in_daily_fvg']:
            htf_reasons.append('in daily bear FVG')

        print(f'  {s["date"]} {s["day_type"]:>11s} drop={s["drop"]:.0f}pts pm_recov={s["pm_recovery_pct"]:.0%} | HTF: {", ".join(htf_reasons)}')

    # Summary
    print(f'\n--- SUMMARY ---')
    all_bearish = [s for s in sessions.values() if s['ext_down'] > 0.3]
    htf_dates = {s['date'] for s in htf_bearish_sessions}
    with_htf = [s for s in all_bearish if s['date'] in htf_dates]
    without_htf = [s for s in all_bearish if s['date'] not in htf_dates]

    if with_htf:
        avg_drop_with = np.mean([s['max_drop_from_ib'] for s in with_htf])
        avg_recov_with = np.mean([s['pm_recovery_pct'] for s in with_htf])
        print(f'With HTF bearish context: {len(with_htf)} sessions, avg drop={avg_drop_with:.0f}pts, avg PM recovery={avg_recov_with:.0%}')

    if without_htf:
        avg_drop_without = np.mean([s['max_drop_from_ib'] for s in without_htf])
        avg_recov_without = np.mean([s['pm_recovery_pct'] for s in without_htf])
        print(f'Without HTF context:      {len(without_htf)} sessions, avg drop={avg_drop_without:.0f}pts, avg PM recovery={avg_recov_without:.0%}')


def main():
    print('Loading data...')
    nq, es = load_data()
    nq_rth = filter_rth(nq)

    print('\nComputing session context...')
    sessions = compute_sessions(nq_rth)
    print(f'Total sessions: {len(sessions)}')

    # Phase 1: Identify bearish sessions
    bearish, strong_bear, trend_down = study_bearish_days(sessions, nq_rth)

    # Phase 2: Raw opportunity analysis
    study_mfe_on_bearish(sessions, nq_rth)

    # Phase 3: Short the bearish move
    bear_short_results = simulate_bear_short(sessions, nq_rth)

    # Phase 4: PM recovery longs
    pm_long_results = simulate_pm_recovery_long(sessions, nq_rth)

    # Phase 5: Weak trend up fades
    study_weak_trend_up_fade(sessions, nq_rth)

    # Phase 6: HTF context
    study_htf_context(sessions, nq)


if __name__ == '__main__':
    main()
