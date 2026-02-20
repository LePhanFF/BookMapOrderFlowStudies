"""
Bearish Day Study v2 - Deep Analysis with HTF Context Filtering

Builds on v1 results. Key questions:
1. Do HTF-backed bearish sessions produce better SHORT entries?
2. What's the best PM recovery LONG model?
3. Can we filter by day_type to separate winners from losers?
4. VWAP rejection short with HTF confluence: viable?
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import time as dt_time
from collections import Counter

project_root = Path(__file__).resolve().parent

import sys
sys.path.insert(0, str(project_root))
from data.loader import load_csv
from data.session import filter_rth


def load_data():
    nq = load_csv('NQ')
    return nq


def build_htf(nq):
    """Build daily OHLC and 4H OHLC, find FVGs."""
    rth_mask = (nq['time'] >= dt_time(9, 30)) & (nq['time'] <= dt_time(16, 0))
    nq_rth_all = nq[rth_mask].copy()

    daily = nq_rth_all.groupby('session_date').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        'volume': 'sum',
    }).sort_index()

    # 4H bars
    nq_rth_indexed = nq_rth_all.set_index('timestamp').sort_index()
    h4 = pd.DataFrame({
        'open': nq_rth_indexed['open'].resample('4h').first(),
        'high': nq_rth_indexed['high'].resample('4h').max(),
        'low': nq_rth_indexed['low'].resample('4h').min(),
        'close': nq_rth_indexed['close'].resample('4h').last(),
    }).dropna()

    # Daily bearish FVGs
    daily_bear_fvgs = []
    for i in range(2, len(daily)):
        d2 = daily.iloc[i - 2]
        d0 = daily.iloc[i]
        if d2['low'] > d0['high']:
            daily_bear_fvgs.append({
                'top': d2['low'], 'bottom': d0['high'],
                'date': daily.index[i],
            })

    # 4H bearish FVGs
    h4_bear_fvgs = []
    for i in range(2, len(h4)):
        c2 = h4.iloc[i - 2]
        c0 = h4.iloc[i]
        if c2['low'] > c0['high']:
            h4_bear_fvgs.append({
                'top': c2['low'], 'bottom': c0['high'],
                'time': h4.index[i],
            })

    return daily, h4, daily_bear_fvgs, h4_bear_fvgs


def classify_session(group, daily, h4_bear_fvgs, daily_bear_fvgs):
    """Classify one session with full HTF context."""
    group = group.sort_values('timestamp')
    session_date = group.iloc[0]['session_date']

    ib = group[group['time'] <= dt_time(10, 30)]
    if len(ib) < 10:
        return None

    ib_high = ib['high'].max()
    ib_low = ib['low'].min()
    ib_range = ib_high - ib_low
    ib_mid = (ib_high + ib_low) / 2

    if ib_range < 15:
        return None

    post_ib = group[group['time'] > dt_time(10, 30)]
    if len(post_ib) == 0:
        return None

    session_high = group['high'].max()
    session_low = group['low'].min()
    session_close = group.iloc[-1]['close']
    session_open = group.iloc[0]['open']

    ext_up = (session_high - ib_high) / ib_range
    ext_down = (ib_low - session_low) / ib_range

    if ext_up > 1.0:
        day_type = 'trend_up'
    elif ext_down > 1.0:
        day_type = 'trend_down'
    elif ext_up > 0.5:
        day_type = 'p_day'
    elif ext_down > 0.5:
        day_type = 'b_day_bear'
    elif ext_up > 0.2 or ext_down > 0.2:
        day_type = 'b_day'
    else:
        day_type = 'neutral'

    # Acceptance below IBL
    consec_below = 0
    max_consec_below = 0
    accept_below_time = None
    for _, bar in post_ib.sort_values('timestamp').iterrows():
        if bar['close'] < ib_low:
            consec_below += 1
            if consec_below >= 3 and accept_below_time is None:
                accept_below_time = bar['timestamp']
            max_consec_below = max(max_consec_below, consec_below)
        else:
            consec_below = 0

    accepted_below = max_consec_below >= 3  # Stricter: 3 bars (like bear acceptance)

    # HTF context
    prev_days = daily[daily.index < session_date].tail(3)
    pdh = prev_days.iloc[-1]['high'] if len(prev_days) > 0 else 0
    pdl = prev_days.iloc[-1]['low'] if len(prev_days) > 0 else 0

    # PDH sweep + fail
    swept_pdh_failed = session_high >= pdh - 5 and session_close < pdh

    # In 4H bear FVG
    in_4h_fvg = False
    active_4h_fvg = None
    for fvg in h4_bear_fvgs:
        if fvg['time'].date() < session_date.date():
            if fvg['bottom'] <= session_high <= fvg['top'] + 50:
                in_4h_fvg = True
                active_4h_fvg = fvg
                break

    # In daily bear FVG
    in_daily_fvg = False
    for fvg in daily_bear_fvgs:
        if fvg['date'] < session_date:
            if fvg['bottom'] <= session_high <= fvg['top'] + 50:
                in_daily_fvg = True
                break

    htf_bearish = swept_pdh_failed or in_4h_fvg or in_daily_fvg

    # AM/PM analysis
    am = group[group['time'] <= dt_time(12, 0)]
    pm = group[group['time'] > dt_time(12, 0)]
    am_low = am['low'].min() if len(am) > 0 else ib_low
    pm_close = pm.iloc[-1]['close'] if len(pm) > 0 else session_close
    pm_low = pm['low'].min() if len(pm) > 0 else session_low

    max_drop = ib_low - session_low
    pm_recovery = pm_close - session_low if max_drop > 0 else 0
    pm_recovery_pct = pm_recovery / max_drop if max_drop > 10 else 0

    return {
        'session_date': session_date,
        'ib_high': ib_high, 'ib_low': ib_low, 'ib_range': ib_range, 'ib_mid': ib_mid,
        'day_type': day_type,
        'ext_up': ext_up, 'ext_down': ext_down,
        'session_high': session_high, 'session_low': session_low,
        'session_close': session_close, 'session_open': session_open,
        'accepted_below': accepted_below, 'accept_below_time': accept_below_time,
        'htf_bearish': htf_bearish,
        'swept_pdh_failed': swept_pdh_failed,
        'in_4h_fvg': in_4h_fvg, 'in_daily_fvg': in_daily_fvg,
        'pdh': pdh, 'pdl': pdl,
        'max_drop': max_drop,
        'pm_recovery': pm_recovery, 'pm_recovery_pct': pm_recovery_pct,
        'am_low': am_low, 'pm_low': pm_low, 'pm_close': pm_close,
        'bars': group, 'post_ib': post_ib,
        'active_4h_fvg': active_4h_fvg,
    }


def simulate_shorts_with_htf(sessions_list, nq_rth, label="ALL"):
    """Simulate bear shorts with multiple entry models, optionally filtered."""
    results = []

    for s in sessions_list:
        if not s['accepted_below'] or s['accept_below_time'] is None:
            continue

        bars = s['bars'].sort_values('timestamp')
        ib_low = s['ib_low']
        ib_high = s['ib_high']
        ib_range = s['ib_range']
        ib_mid = s['ib_mid']
        accept_time = s['accept_below_time']
        date_str = str(s['session_date'])[:10]

        post_accept = bars[bars['timestamp'] >= accept_time]

        # --- MODEL: ACCEPTANCE SHORT ---
        # Enter short IMMEDIATELY on IBL acceptance (3 bars below)
        # Stop: IBL + 20% IB range (if price reclaims IBL, we're wrong)
        # Target: IB range below IBL (1x extension)
        accept_bar = bars[bars['timestamp'] == accept_time]
        if len(accept_bar) > 0:
            bar = accept_bar.iloc[0]
            entry_price = bar['close']
            stop_price = ib_low + ib_range * 0.25
            target_price = ib_low - ib_range * 0.75  # 0.75x extension target

            risk = stop_price - entry_price
            reward = entry_price - target_price

            if risk > 5 and reward > 5:
                remaining = bars[bars['timestamp'] > bar['timestamp']]
                remaining = remaining[remaining['time'] <= dt_time(15, 30)]

                exit_price = entry_price
                exit_reason = 'NO_BARS'
                mfe = 0

                for _, rb in remaining.iterrows():
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
                    if len(remaining) > 0:
                        exit_price = remaining.iloc[-1]['close']
                        exit_reason = 'EOD'

                pnl_pts = entry_price - exit_price

                results.append({
                    'date': date_str,
                    'model': 'ACCEPTANCE_SHORT',
                    'bar_time': bar['time'],
                    'entry_price': entry_price,
                    'stop_price': stop_price,
                    'target_price': target_price,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'pnl_pts': pnl_pts,
                    'risk_pts': risk,
                    'mfe_pts': mfe,
                    'day_type': s['day_type'],
                    'ext_down': s['ext_down'],
                    'htf': s['htf_bearish'],
                    'pdh_sweep': s['swept_pdh_failed'],
                    'in_4h_fvg': s['in_4h_fvg'],
                    'in_daily_fvg': s['in_daily_fvg'],
                })

        # --- MODEL: VWAP REJECTION SHORT (post-acceptance) ---
        # After acceptance, wait for pullback toward VWAP, short on rejection
        for _, bar in post_accept.iterrows():
            if bar['time'] >= dt_time(14, 0):
                break

            vwap = bar.get('vwap')
            if vwap is None or pd.isna(vwap):
                continue

            delta = bar.get('vol_delta', 0)
            if pd.isna(delta):
                delta = 0

            vwap_dist = abs(bar['close'] - vwap) / ib_range
            # Close below VWAP but near it, with selling delta
            if vwap_dist < 0.25 and bar['close'] < vwap and delta < 0:
                entry_price = bar['close']
                stop_price = vwap + ib_range * 0.20
                target_price = s['am_low'] if s['am_low'] < entry_price - 20 else entry_price - ib_range * 0.50

                risk = stop_price - entry_price
                reward = entry_price - target_price

                if risk <= 5 or reward <= 5:
                    continue

                remaining = bars[bars['timestamp'] > bar['timestamp']]
                remaining = remaining[remaining['time'] <= dt_time(15, 30)]

                exit_price = entry_price
                exit_reason = 'NO_BARS'
                mfe = 0

                for _, rb in remaining.iterrows():
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
                    if len(remaining) > 0:
                        exit_price = remaining.iloc[-1]['close']
                        exit_reason = 'EOD'

                pnl_pts = entry_price - exit_price

                results.append({
                    'date': date_str,
                    'model': 'VWAP_REJECT_SHORT',
                    'bar_time': bar['time'],
                    'entry_price': entry_price,
                    'stop_price': stop_price,
                    'target_price': target_price,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'pnl_pts': pnl_pts,
                    'risk_pts': risk,
                    'mfe_pts': mfe,
                    'day_type': s['day_type'],
                    'ext_down': s['ext_down'],
                    'htf': s['htf_bearish'],
                    'pdh_sweep': s['swept_pdh_failed'],
                    'in_4h_fvg': s['in_4h_fvg'],
                    'in_daily_fvg': s['in_daily_fvg'],
                })
                break

        # --- MODEL: IBL RETEST SHORT ---
        # After acceptance, price bounces back toward IBL, short the rejection
        for _, bar in post_accept.iterrows():
            if bar['time'] >= dt_time(13, 30):
                break

            delta = bar.get('vol_delta', 0)
            if pd.isna(delta):
                delta = 0

            # Price approaching IBL from below with negative delta
            if bar['high'] >= ib_low - ib_range * 0.10 and bar['close'] < ib_low and delta < 0:
                entry_price = bar['close']
                stop_price = ib_low + ib_range * 0.15
                target_price = entry_price - ib_range * 0.50

                risk = stop_price - entry_price
                reward = entry_price - target_price

                if risk <= 5 or reward <= 5:
                    continue

                remaining = bars[bars['timestamp'] > bar['timestamp']]
                remaining = remaining[remaining['time'] <= dt_time(15, 30)]

                exit_price = entry_price
                exit_reason = 'NO_BARS'
                mfe = 0

                for _, rb in remaining.iterrows():
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
                    if len(remaining) > 0:
                        exit_price = remaining.iloc[-1]['close']
                        exit_reason = 'EOD'

                pnl_pts = entry_price - exit_price

                results.append({
                    'date': date_str,
                    'model': 'IBL_RETEST_SHORT',
                    'bar_time': bar['time'],
                    'entry_price': entry_price,
                    'stop_price': stop_price,
                    'target_price': target_price,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'pnl_pts': pnl_pts,
                    'risk_pts': risk,
                    'mfe_pts': mfe,
                    'day_type': s['day_type'],
                    'ext_down': s['ext_down'],
                    'htf': s['htf_bearish'],
                    'pdh_sweep': s['swept_pdh_failed'],
                    'in_4h_fvg': s['in_4h_fvg'],
                    'in_daily_fvg': s['in_daily_fvg'],
                })
                break

    return results


def simulate_pm_recovery_v2(sessions_list, label="ALL"):
    """PM Recovery LONG with multiple models and exit strategies."""
    results = []

    for s in sessions_list:
        if s['ext_down'] < 0.3:
            continue

        bars = s['bars'].sort_values('timestamp')
        ib_low = s['ib_low']
        ib_high = s['ib_high']
        ib_range = s['ib_range']
        ib_mid = s['ib_mid']
        date_str = str(s['session_date'])[:10]

        pm_bars = bars[(bars['time'] >= dt_time(12, 0)) & (bars['time'] <= dt_time(15, 20))]
        if len(pm_bars) == 0:
            continue

        am_bars = bars[bars['time'] < dt_time(12, 0)]
        if len(am_bars) == 0:
            continue
        am_low = am_bars['low'].min()

        # Track delta momentum: rolling 10-bar sum
        bars_sorted = bars.sort_values('timestamp').reset_index(drop=True)
        bars_sorted['delta_10'] = bars_sorted['vol_delta'].rolling(10, min_periods=5).sum()

        # --- MODEL: DELTA DIVERGENCE LONG ---
        # Price makes new low but delta diverges (delta_10 turning positive)
        # This signals selling exhaustion / short covering
        for idx, bar in pm_bars.iterrows():
            if bar['time'] >= dt_time(14, 30):
                break

            # Match by timestamp in bars_sorted
            match = bars_sorted[bars_sorted['timestamp'] == bar['timestamp']]
            if len(match) == 0:
                continue

            delta_10 = match.iloc[0].get('delta_10', 0)
            if pd.isna(delta_10):
                continue

            delta = bar.get('vol_delta', 0)
            if pd.isna(delta):
                delta = 0

            # Divergence: price near session low but delta turning positive
            dist_from_low = bar['close'] - am_low
            if dist_from_low < ib_range * 0.30 and delta_10 > 0 and delta > 0 and bar['close'] > bar['open']:
                entry_price = bar['close']
                stop_price = am_low - ib_range * 0.10
                # Target: VWAP or IBL (whichever is closer/lower)
                vwap = bar.get('vwap', ib_low)
                if pd.isna(vwap):
                    vwap = ib_low
                target_price = min(ib_low, vwap) if min(ib_low, vwap) > entry_price + 15 else entry_price + ib_range * 0.40

                risk = entry_price - stop_price
                reward = target_price - entry_price
                if risk <= 5 or reward <= 5:
                    continue

                remaining = bars[bars['timestamp'] > bar['timestamp']]
                remaining = remaining[remaining['time'] <= dt_time(15, 30)]

                exit_price = entry_price
                exit_reason = 'NO_BARS'
                mfe = 0

                for _, rb in remaining.iterrows():
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
                    if len(remaining) > 0:
                        exit_price = remaining.iloc[-1]['close']
                        exit_reason = 'EOD'

                pnl_pts = exit_price - entry_price

                results.append({
                    'date': date_str,
                    'model': 'DELTA_DIVERGENCE_LONG',
                    'bar_time': bar['time'],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'pnl_pts': pnl_pts,
                    'risk_pts': risk,
                    'mfe_pts': mfe,
                    'day_type': s['day_type'],
                    'ext_down': s['ext_down'],
                    'pm_recovery_pct': s['pm_recovery_pct'],
                    'htf': s['htf_bearish'],
                })
                break

        # --- MODEL: FIRST GREEN CANDLE NEAR LOW ---
        # After sustained selling, first bullish candle with volume near session low
        found_green = False
        consec_red = 0
        for idx, bar in pm_bars.iterrows():
            if bar['time'] >= dt_time(14, 0):
                break

            if bar['close'] < bar['open']:
                consec_red += 1
                continue

            # Need at least 3 consecutive red candles before
            if consec_red < 3:
                consec_red = 0
                continue

            delta = bar.get('vol_delta', 0)
            if pd.isna(delta):
                delta = 0

            vol = bar.get('volume', 0)
            if pd.isna(vol):
                vol = 0

            dist_from_low = bar['close'] - am_low
            if dist_from_low < ib_range * 0.35 and delta > 0:
                entry_price = bar['close']
                stop_price = am_low - ib_range * 0.08
                target_price = ib_low if ib_low > entry_price + 20 else entry_price + ib_range * 0.35

                risk = entry_price - stop_price
                reward = target_price - entry_price
                if risk <= 5 or reward <= 5:
                    consec_red = 0
                    continue

                remaining = bars[bars['timestamp'] > bar['timestamp']]
                remaining = remaining[remaining['time'] <= dt_time(15, 30)]

                exit_price = entry_price
                exit_reason = 'NO_BARS'
                mfe = 0

                for _, rb in remaining.iterrows():
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
                    if len(remaining) > 0:
                        exit_price = remaining.iloc[-1]['close']
                        exit_reason = 'EOD'

                pnl_pts = exit_price - entry_price

                results.append({
                    'date': date_str,
                    'model': 'FIRST_GREEN_NEAR_LOW',
                    'bar_time': bar['time'],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'pnl_pts': pnl_pts,
                    'risk_pts': risk,
                    'mfe_pts': mfe,
                    'day_type': s['day_type'],
                    'ext_down': s['ext_down'],
                    'pm_recovery_pct': s['pm_recovery_pct'],
                    'htf': s['htf_bearish'],
                })
                found_green = True
                break

            consec_red = 0

        # --- MODEL: IBL RECLAIM LONG (PM) ---
        # Price was accepted below IBL, now reclaims above IBL in PM
        if s['accepted_below']:
            for idx, bar in pm_bars.iterrows():
                if bar['time'] >= dt_time(14, 30):
                    break

                delta = bar.get('vol_delta', 0)
                if pd.isna(delta):
                    delta = 0

                # Cross above IBL with positive delta
                if bar['close'] > ib_low and bar['open'] < ib_low and delta > 0:
                    entry_price = bar['close']
                    stop_price = bar['low'] - ib_range * 0.05  # Tight stop below entry bar low
                    target_price = ib_mid

                    risk = entry_price - stop_price
                    reward = target_price - entry_price
                    if risk <= 5 or reward <= 5:
                        continue

                    remaining = bars[bars['timestamp'] > bar['timestamp']]
                    remaining = remaining[remaining['time'] <= dt_time(15, 30)]

                    exit_price = entry_price
                    exit_reason = 'NO_BARS'
                    mfe = 0

                    for _, rb in remaining.iterrows():
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
                        if len(remaining) > 0:
                            exit_price = remaining.iloc[-1]['close']
                            exit_reason = 'EOD'

                    pnl_pts = exit_price - entry_price

                    results.append({
                        'date': date_str,
                        'model': 'IBL_RECLAIM_LONG',
                        'bar_time': bar['time'],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'pnl_pts': pnl_pts,
                        'risk_pts': risk,
                        'mfe_pts': mfe,
                        'day_type': s['day_type'],
                        'ext_down': s['ext_down'],
                        'pm_recovery_pct': s['pm_recovery_pct'],
                        'htf': s['htf_bearish'],
                    })
                    break

    return results


def print_results(results, title):
    """Print detailed results for a set of trades."""
    if not results:
        print(f'\n  {title}: No trades found')
        return

    rdf = pd.DataFrame(results)
    print(f'\n{"=" * 90}')
    print(f'  {title}')
    print(f'{"=" * 90}')

    for model in sorted(rdf['model'].unique()):
        sub = rdf[rdf['model'] == model]
        wins = sub[sub['pnl_pts'] > 0]
        losses = sub[sub['pnl_pts'] <= 0]

        print(f'\n--- {model} ---')
        print(f'Trades: {len(sub)}, Wins: {len(wins)}, WR: {len(wins)/len(sub)*100:.0f}%')
        print(f'Net: {sub["pnl_pts"].sum():+.1f} pts = ${sub["pnl_pts"].sum() * 2:+,.0f} (1 MNQ)')
        print(f'Avg: {sub["pnl_pts"].mean():+.1f} pts/trade')
        print(f'Avg Win: {wins["pnl_pts"].mean():+.1f} pts' if len(wins) > 0 else 'No wins')
        print(f'Avg Loss: {losses["pnl_pts"].mean():+.1f} pts' if len(losses) > 0 else 'No losses')
        print(f'Avg MFE: {sub["mfe_pts"].mean():.1f} pts')

        # Exit breakdown
        for reason in ['TARGET', 'STOP', 'EOD']:
            r_sub = sub[sub['exit_reason'] == reason]
            if len(r_sub) > 0:
                print(f'  {reason}: {len(r_sub)} trades, avg P&L: {r_sub["pnl_pts"].mean():+.1f} pts')

        print(f'\n  {"Date":<12s} {"Time":>5s} {"DayType":>11s} {"Ext":>5s} {"HTF":>4s} {"Exit":>6s} {"PnL":>8s} {"MFE":>6s} {"W/L"}')
        for _, t in sub.sort_values('date').iterrows():
            status = 'W' if t['pnl_pts'] > 0 else 'L'
            htf = 'YES' if t.get('htf', False) else 'no'
            ext_val = t.get('ext_down', t.get('ext', 0))
            print(f'  {t["date"]:<12s} {str(t["bar_time"])[:5]:>5s} {t["day_type"]:>11s} {ext_val:>4.1f}x {htf:>4s} '
                  f'{t["exit_reason"]:>6s} {t["pnl_pts"]:+7.1f}pts {t["mfe_pts"]:>5.0f} [{status}]')

    # HTF filter breakdown
    if 'htf' in rdf.columns:
        print(f'\n--- HTF FILTER IMPACT ---')
        for htf_val in [True, False]:
            sub = rdf[rdf['htf'] == htf_val]
            if len(sub) > 0:
                wins = sub[sub['pnl_pts'] > 0]
                label_str = 'WITH HTF' if htf_val else 'NO HTF'
                print(f'  {label_str}: {len(sub)} trades, {len(wins)/len(sub)*100:.0f}% WR, '
                      f'{sub["pnl_pts"].sum():+.1f} pts net, {sub["pnl_pts"].mean():+.1f} avg')

    # Day type breakdown
    print(f'\n--- DAY TYPE BREAKDOWN ---')
    for dt in sorted(rdf['day_type'].unique()):
        sub = rdf[rdf['day_type'] == dt]
        wins = sub[sub['pnl_pts'] > 0]
        print(f'  {dt:<12s}: {len(sub)} trades, {len(wins)/len(sub)*100:.0f}% WR, '
              f'{sub["pnl_pts"].sum():+.1f} pts net')


def main():
    print('Loading data...')
    nq = load_data()
    nq_rth = filter_rth(nq)

    print('\nBuilding HTF context...')
    daily, h4, daily_bear_fvgs, h4_bear_fvgs = build_htf(nq)

    print('\nClassifying sessions...')
    sessions = []
    for session_date, group in nq_rth.groupby('session_date'):
        s = classify_session(group, daily, h4_bear_fvgs, daily_bear_fvgs)
        if s is not None:
            sessions.append(s)

    print(f'Total sessions: {len(sessions)}')

    # Bearish sessions
    bearish = [s for s in sessions if s['ext_down'] > 0.3]
    htf_bearish = [s for s in bearish if s['htf_bearish']]
    no_htf_bearish = [s for s in bearish if not s['htf_bearish']]

    print(f'\nBearish sessions (ext_down > 0.3x): {len(bearish)}')
    print(f'  With HTF bearish context: {len(htf_bearish)}')
    print(f'  Without HTF context: {len(no_htf_bearish)}')

    # ===== SHORTS =====
    print('\n\n' + '#' * 90)
    print('#  SECTION A: BEAR SHORTS')
    print('#' * 90)

    # All sessions (including non-bearish that have IBL acceptance)
    accepted_below_sessions = [s for s in sessions if s['accepted_below']]
    print(f'\nSessions with IBL acceptance (3+ bars below): {len(accepted_below_sessions)}')

    # A1: All shorts (no filter)
    all_shorts = simulate_shorts_with_htf(sessions, nq_rth, "ALL")
    print_results(all_shorts, "A1: ALL SHORTS (unfiltered)")

    # A2: HTF-backed shorts only
    htf_sessions = [s for s in sessions if s['htf_bearish']]
    htf_shorts = simulate_shorts_with_htf(htf_sessions, nq_rth, "HTF")
    print_results(htf_shorts, "A2: HTF-BACKED SHORTS ONLY")

    # A3: Trend down + b_day_bear only (structural filter)
    bear_type_sessions = [s for s in sessions if s['day_type'] in ('trend_down', 'b_day_bear')]
    type_shorts = simulate_shorts_with_htf(bear_type_sessions, nq_rth, "BEAR_TYPES")
    print_results(type_shorts, "A3: TREND_DOWN + B_DAY_BEAR ONLY")

    # A4: HTF + bear day types (double filter)
    htf_bear_type = [s for s in sessions if s['htf_bearish'] and s['day_type'] in ('trend_down', 'b_day_bear')]
    htf_type_shorts = simulate_shorts_with_htf(htf_bear_type, nq_rth, "HTF+BEAR")
    print_results(htf_type_shorts, "A4: HTF + BEAR DAY TYPE (strictest)")

    # ===== PM RECOVERY LONGS =====
    print('\n\n' + '#' * 90)
    print('#  SECTION B: PM RECOVERY LONGS')
    print('#' * 90)

    # B1: All bearish sessions
    all_pm_longs = simulate_pm_recovery_v2(sessions, "ALL")
    print_results(all_pm_longs, "B1: PM RECOVERY LONGS (all bearish)")

    # B2: HTF-backed bearish (bigger drops, less recovery expected)
    htf_pm_longs = simulate_pm_recovery_v2(htf_bearish, "HTF")
    print_results(htf_pm_longs, "B2: PM RECOVERY LONGS (HTF bearish only)")

    # B3: Non-HTF bearish (these tend to recover more)
    no_htf_pm_longs = simulate_pm_recovery_v2(no_htf_bearish, "NO_HTF")
    print_results(no_htf_pm_longs, "B3: PM RECOVERY LONGS (no HTF context)")

    # ===== COMBINED BEST-OF =====
    print('\n\n' + '#' * 90)
    print('#  SECTION C: BEST COMBINATIONS')
    print('#' * 90)

    # Find best short model + filter
    all_short_results = []
    for label, res in [("ALL", all_shorts), ("HTF", htf_shorts),
                        ("BEAR_TYPE", type_shorts), ("HTF+BEAR", htf_type_shorts)]:
        if res:
            rdf = pd.DataFrame(res)
            for model in rdf['model'].unique():
                sub = rdf[rdf['model'] == model]
                wins = sub[sub['pnl_pts'] > 0]
                all_short_results.append({
                    'combo': f'{label} + {model}',
                    'trades': len(sub),
                    'wr': len(wins)/len(sub)*100,
                    'net_pts': sub['pnl_pts'].sum(),
                    'avg_pts': sub['pnl_pts'].mean(),
                    'avg_mfe': sub['mfe_pts'].mean(),
                })

    print(f'\n--- SHORT MODEL COMPARISON ---')
    print(f'{"Combo":<40s} {"Trades":>6s} {"WR":>5s} {"Net Pts":>8s} {"Avg":>7s} {"MFE":>5s}')
    print('-' * 75)
    for r in sorted(all_short_results, key=lambda x: x['net_pts'], reverse=True):
        print(f'{r["combo"]:<40s} {r["trades"]:>6d} {r["wr"]:>4.0f}% {r["net_pts"]:>+7.0f} {r["avg_pts"]:>+6.1f} {r["avg_mfe"]:>5.0f}')

    # Find best long model
    all_long_results = []
    for label, res in [("ALL", all_pm_longs), ("HTF", htf_pm_longs), ("NO_HTF", no_htf_pm_longs)]:
        if res:
            rdf = pd.DataFrame(res)
            for model in rdf['model'].unique():
                sub = rdf[rdf['model'] == model]
                wins = sub[sub['pnl_pts'] > 0]
                all_long_results.append({
                    'combo': f'{label} + {model}',
                    'trades': len(sub),
                    'wr': len(wins)/len(sub)*100 if len(sub) > 0 else 0,
                    'net_pts': sub['pnl_pts'].sum(),
                    'avg_pts': sub['pnl_pts'].mean(),
                    'avg_mfe': sub['mfe_pts'].mean(),
                })

    print(f'\n--- LONG MODEL COMPARISON ---')
    print(f'{"Combo":<40s} {"Trades":>6s} {"WR":>5s} {"Net Pts":>8s} {"Avg":>7s} {"MFE":>5s}')
    print('-' * 75)
    for r in sorted(all_long_results, key=lambda x: x['net_pts'], reverse=True):
        print(f'{r["combo"]:<40s} {r["trades"]:>6d} {r["wr"]:>4.0f}% {r["net_pts"]:>+7.0f} {r["avg_pts"]:>+6.1f} {r["avg_mfe"]:>5.0f}')

    # === OVERALL SUMMARY ===
    print(f'\n{"=" * 90}')
    print(f'  FINAL SUMMARY: BEARISH DAY STRATEGIES')
    print(f'{"=" * 90}')

    print(f'\nBearish sessions: {len(bearish)} of {len(sessions)} total ({len(bearish)/len(sessions)*100:.0f}%)')
    print(f'HTF-backed bearish: {len(htf_bearish)} ({len(htf_bearish)/len(bearish)*100:.0f}% of bearish)')
    print(f'Avg drop WITH HTF: {np.mean([s["max_drop"] for s in htf_bearish]):.0f} pts')
    print(f'Avg drop WITHOUT HTF: {np.mean([s["max_drop"] for s in no_htf_bearish]):.0f} pts' if no_htf_bearish else '')
    print(f'PM recovery > 30%: {sum(1 for s in bearish if s["pm_recovery_pct"] > 0.30)} sessions ({sum(1 for s in bearish if s["pm_recovery_pct"] > 0.30)/len(bearish)*100:.0f}%)')

    # Best short
    if all_short_results:
        best_short = max(all_short_results, key=lambda x: x['net_pts'])
        print(f'\nBest SHORT: {best_short["combo"]}')
        print(f'  {best_short["trades"]} trades, {best_short["wr"]:.0f}% WR, {best_short["net_pts"]:+.0f} pts net')

    # Best long
    if all_long_results:
        best_long = max(all_long_results, key=lambda x: x['net_pts'])
        print(f'\nBest LONG: {best_long["combo"]}')
        print(f'  {best_long["trades"]} trades, {best_long["wr"]:.0f}% WR, {best_long["net_pts"]:+.0f} pts net')


if __name__ == '__main__':
    main()
