"""
IBH Sweep + Failure SHORT Study

The hypothesis: On balance/neutral days, if price sweeps ABOVE IBH into a
liquidity zone (PDH, London high, 4H FVG, daily FVG) and FAILS to hold,
this is a high-probability short entry fading back to IB mid.

Previous attempts at IBH shorts (blind fades at every touch) failed 0-22% WR.
This study tests whether SELECTIVE shorting with HTF confluence changes the outcome.

Key confluence levels to test:
1. Previous Day High (PDH) - sweep of PDH above IBH
2. London session high (ETH overnight high)
3. 4H bearish FVG zones (computed from 4H OHLC)
4. Daily bearish FVG zones (computed from daily OHLC)
5. VWAP upper bands (institutional rejection levels)
6. SMT divergence with ES (NQ makes new high, ES doesn't)
7. Delta divergence (price makes new high with negative delta)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import time as dt_time, timedelta

project_root = Path(__file__).resolve().parent.parent

import sys
sys.path.insert(0, str(project_root))
from data.loader import load_csv
from data.session import filter_rth


def load_data():
    """Load NQ and ES data."""
    nq = load_csv('NQ')
    es = load_csv('ES')
    return nq, es


def compute_session_context(nq_rth):
    """Compute IB high/low/mid/range and day type for each session."""
    sessions = {}

    for session_date, group in nq_rth.groupby('session_date'):
        group = group.sort_values('timestamp')
        date_str = str(session_date)[:10]

        # IB period: first 60 minutes (9:30 - 10:30)
        ib = group[group['time'] <= dt_time(10, 30)]
        if len(ib) < 10:
            continue

        ib_high = ib['high'].max()
        ib_low = ib['low'].min()
        ib_range = ib_high - ib_low
        ib_mid = (ib_high + ib_low) / 2

        if ib_range < 20:  # Too narrow
            continue

        # Post-IB bars
        post_ib = group[group['time'] > dt_time(10, 30)]

        # Extension above IBH
        if len(post_ib) > 0:
            session_high = group['high'].max()
            extension_up = (session_high - ib_high) / ib_range if ib_range > 0 else 0
        else:
            extension_up = 0

        # Day type classification (simplified)
        if extension_up > 1.0:
            day_type = 'trend_up'
        elif extension_up > 0.5:
            day_type = 'p_day'
        elif extension_up > 0.2:
            day_type = 'b_day'
        else:
            day_type = 'neutral'

        sessions[date_str] = {
            'date': date_str,
            'ib_high': ib_high,
            'ib_low': ib_low,
            'ib_range': ib_range,
            'ib_mid': ib_mid,
            'day_type': day_type,
            'extension_up': extension_up,
            'session_high': group['high'].max(),
            'session_low': group['low'].min(),
            'bars': group,
            'post_ib': post_ib,
        }

    return sessions


def compute_htf_levels(nq_full, session_date_str):
    """Compute HTF confluence levels for a given session.

    Returns dict of level_name -> price.
    """
    session_date = pd.Timestamp(session_date_str)
    levels = {}

    # --- Previous Day High (PDH) ---
    # Get prior session's RTH data
    prev_date = session_date - timedelta(days=1)
    # Try up to 5 days back for weekends/holidays
    for i in range(1, 6):
        check_date = session_date - timedelta(days=i)
        prev_bars = nq_full[
            (nq_full['session_date'] == check_date) &
            (nq_full['time'] >= dt_time(9, 30)) &
            (nq_full['time'] <= dt_time(16, 0))
        ]
        if len(prev_bars) > 0:
            levels['PDH'] = prev_bars['high'].max()
            levels['PDL'] = prev_bars['low'].min()
            levels['prev_close'] = prev_bars.iloc[-1]['close']
            break

    # --- London/Overnight High ---
    # ETH session: 18:00 previous day to 09:29 current day
    eth_bars = nq_full[
        (nq_full['session_date'] == session_date) &
        (nq_full['time'] < dt_time(9, 30))
    ]
    if len(eth_bars) > 0:
        levels['London_High'] = eth_bars['high'].max()
        levels['London_Low'] = eth_bars['low'].min()
        levels['Overnight_High'] = eth_bars['high'].max()

    # --- VWAP Upper Bands (from data) ---
    # These reset daily, so get the value during the session
    session_bars = nq_full[
        (nq_full['session_date'] == session_date) &
        (nq_full['time'] >= dt_time(9, 30))
    ]
    if len(session_bars) > 0:
        # VWAP bands stabilize after ~30 bars
        mid_session = session_bars[session_bars['time'] >= dt_time(11, 0)]
        if len(mid_session) > 0 and 'vwap_upper1' in mid_session.columns:
            vwap_u1 = mid_session['vwap_upper1'].median()
            if not pd.isna(vwap_u1):
                levels['VWAP_Upper1'] = vwap_u1

    # --- 4H Bearish FVG ---
    # Resample to 4H bars and look for bearish FVGs
    # A bearish FVG: bar[i-2].low > bar[i].high (gap between candle i-2 low and candle i high)
    # We need several days of 4H data
    lookback_start = session_date - timedelta(days=5)
    recent = nq_full[
        (nq_full['timestamp'] >= lookback_start) &
        (nq_full['timestamp'] < session_date + timedelta(days=1)) &
        (nq_full['time'] >= dt_time(9, 30)) &
        (nq_full['time'] <= dt_time(16, 0))
    ].copy()

    if len(recent) > 60:
        recent = recent.set_index('timestamp')
        h4 = recent['close'].resample('4h').ohlc()
        h4.columns = ['open', 'high', 'low', 'close']
        h4 = h4.dropna()

        # Also get high/low from original data
        h4_high = recent['high'].resample('4h').max()
        h4_low = recent['low'].resample('4h').min()
        h4_open = recent['open'].resample('4h').first()
        h4_close = recent['close'].resample('4h').last()

        h4_df = pd.DataFrame({
            'open': h4_open, 'high': h4_high,
            'low': h4_low, 'close': h4_close
        }).dropna()

        # Find bearish FVGs: candle[i-2].low > candle[i].high
        fvg_zones = []
        for i in range(2, len(h4_df)):
            candle_2_ago = h4_df.iloc[i - 2]
            candle_now = h4_df.iloc[i]
            # Bearish FVG: gap between candle 2 bars ago LOW and current bar HIGH
            if candle_2_ago['low'] > candle_now['high']:
                fvg_top = candle_2_ago['low']
                fvg_bottom = candle_now['high']
                fvg_zones.append({
                    'top': fvg_top, 'bottom': fvg_bottom,
                    'mid': (fvg_top + fvg_bottom) / 2,
                    'time': h4_df.index[i],
                })

        # Keep FVGs that are near/above the session's price range
        session_high = session_bars['high'].max() if len(session_bars) > 0 else 0
        relevant_fvgs = [f for f in fvg_zones if f['bottom'] <= session_high + 200]
        if relevant_fvgs:
            # Take the closest bearish FVG above current price area
            levels['FVG_4H_Bear'] = relevant_fvgs[-1]  # Most recent

    # --- Daily Bearish FVG ---
    daily_lookback = session_date - timedelta(days=20)
    daily_data = nq_full[
        (nq_full['session_date'] >= daily_lookback) &
        (nq_full['session_date'] < session_date) &
        (nq_full['time'] >= dt_time(9, 30)) &
        (nq_full['time'] <= dt_time(16, 0))
    ]

    if len(daily_data) > 0:
        daily_ohlc = daily_data.groupby('session_date').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        })

        daily_fvgs = []
        for i in range(2, len(daily_ohlc)):
            d2 = daily_ohlc.iloc[i - 2]
            d0 = daily_ohlc.iloc[i]
            if d2['low'] > d0['high']:
                daily_fvgs.append({
                    'top': d2['low'], 'bottom': d0['high'],
                    'mid': (d2['low'] + d0['high']) / 2,
                    'date': daily_ohlc.index[i],
                })

        if daily_fvgs:
            levels['FVG_Daily_Bear'] = daily_fvgs[-1]

    return levels


def find_ibh_sweeps(session, nq_full, es_full):
    """Find all IBH sweep + failure events in a session.

    A sweep = price goes ABOVE IBH (high > IBH) then CLOSES back below IBH.
    A failure = after the sweep, price fails to reclaim IBH on the next attempt.
    """
    bars = session['post_ib'].sort_values('timestamp')
    ib_high = session['ib_high']
    ib_low = session['ib_low']
    ib_mid = session['ib_mid']
    ib_range = session['ib_range']

    sweeps = []
    sweep_active = False
    sweep_high = 0
    sweep_start_idx = None

    for i, (idx, bar) in enumerate(bars.iterrows()):
        # Detect sweep above IBH
        if bar['high'] > ib_high:
            if not sweep_active:
                sweep_active = True
                sweep_start_idx = i
                sweep_high = bar['high']
            else:
                sweep_high = max(sweep_high, bar['high'])

        # Detect failure: was above IBH, now closing below
        if sweep_active and bar['close'] < ib_high:
            # This is a sweep + failure candle
            sweep_pts_above = sweep_high - ib_high

            # Only count if it went meaningfully above (not just noise)
            if sweep_pts_above >= 5:  # At least 5 pts above IBH

                # Get ES data for SMT check
                es_session = es_full[
                    (es_full['session_date'] == pd.Timestamp(session['date'])) &
                    (es_full['time'] >= dt_time(9, 30))
                ].sort_values('timestamp')

                # Check SMT: did ES also make a new high at the same time?
                smt_divergence = False
                if len(es_session) > 0:
                    # ES IB high
                    es_ib = es_session[es_session['time'] <= dt_time(10, 30)]
                    if len(es_ib) > 0:
                        es_ib_high = es_ib['high'].max()
                        # ES bars around the sweep time
                        sweep_time = bar['timestamp']
                        es_around = es_session[
                            (es_session['timestamp'] >= sweep_time - timedelta(minutes=5)) &
                            (es_session['timestamp'] <= sweep_time + timedelta(minutes=5))
                        ]
                        if len(es_around) > 0:
                            es_high_at_sweep = es_around['high'].max()
                            # SMT: NQ swept above IBH but ES did NOT make new high
                            if es_high_at_sweep <= es_ib_high:
                                smt_divergence = True

                # Delta at the sweep bar
                delta = bar.get('vol_delta', 0)
                if pd.isna(delta):
                    delta = 0

                # Volume spike
                vol = bar.get('volume', 0)
                avg_vol = bars.iloc[max(0, i-20):i]['volume'].mean() if i > 5 else vol
                vol_spike = vol / avg_vol if avg_vol > 0 else 1.0

                sweeps.append({
                    'session': session['date'],
                    'sweep_time': bar['timestamp'],
                    'bar_time': bar['time'],
                    'ib_high': ib_high,
                    'ib_low': ib_low,
                    'ib_mid': ib_mid,
                    'ib_range': ib_range,
                    'sweep_high': sweep_high,
                    'sweep_pts': sweep_pts_above,
                    'close_after_sweep': bar['close'],
                    'entry_price': bar['close'],  # SHORT at close of failure bar
                    'delta_at_sweep': delta,
                    'vol_spike': vol_spike,
                    'smt_divergence': smt_divergence,
                    'day_type': session['day_type'],
                    'extension_up': session['extension_up'],
                    'bar_idx': idx,
                })

            sweep_active = False
            sweep_high = 0

    return sweeps


def simulate_short_trade(sweep, session_bars, target_price, stop_price):
    """Simulate a SHORT trade from the sweep failure point.

    Returns dict with outcome.
    """
    entry_price = sweep['entry_price']
    entry_time = sweep['sweep_time']

    # Get bars after entry to session end
    bars_after = session_bars[session_bars['timestamp'] > entry_time].sort_values('timestamp')

    if len(bars_after) == 0:
        return {
            'exit_price': entry_price,
            'exit_reason': 'NO_BARS',
            'pnl_pts': 0,
            'bars_held': 0,
            'mfe_pts': 0,
            'mae_pts': 0,
        }

    # Track MFE/MAE for SHORT
    mfe_price = bars_after['low'].min()  # Best case for short
    mae_price = bars_after['high'].max()  # Worst case for short
    mfe_pts = entry_price - mfe_price  # Positive = favorable
    mae_pts = mae_price - entry_price  # Positive = adverse

    # Simulate bar-by-bar
    for i, (_, bar) in enumerate(bars_after.iterrows()):
        # Stop hit (price goes above stop)
        if bar['high'] >= stop_price:
            return {
                'exit_price': stop_price,
                'exit_reason': 'STOP',
                'pnl_pts': entry_price - stop_price,  # Negative
                'bars_held': i + 1,
                'mfe_pts': mfe_pts,
                'mae_pts': mae_pts,
            }

        # Target hit (price goes below target)
        if bar['low'] <= target_price:
            return {
                'exit_price': target_price,
                'exit_reason': 'TARGET',
                'pnl_pts': entry_price - target_price,  # Positive
                'bars_held': i + 1,
                'mfe_pts': mfe_pts,
                'mae_pts': mae_pts,
            }

    # EOD exit
    eod_price = bars_after.iloc[-1]['close']
    return {
        'exit_price': eod_price,
        'exit_reason': 'EOD',
        'pnl_pts': entry_price - eod_price,
        'bars_held': len(bars_after),
        'mfe_pts': mfe_pts,
        'mae_pts': mae_pts,
    }


def main():
    print('Loading data...')
    nq, es = load_data()

    # Filter to RTH
    nq_rth = filter_rth(nq)

    # Compute session context
    print('\nComputing session context...')
    sessions = compute_session_context(nq_rth)
    print(f'Sessions: {len(sessions)}')

    # Day type distribution
    day_types = [s['day_type'] for s in sessions.values()]
    from collections import Counter
    print(f'Day types: {Counter(day_types)}')

    # Only look at b_day and neutral sessions for fade shorts
    fade_sessions = {k: v for k, v in sessions.items()
                     if v['day_type'] in ('b_day', 'neutral', 'p_day')}
    print(f'Fade-eligible sessions (b_day/neutral/p_day): {len(fade_sessions)}')

    # ============================================
    # FIND ALL IBH SWEEPS
    # ============================================
    print('\n' + '=' * 90)
    print('  PHASE 1: FINDING ALL IBH SWEEP + FAILURE EVENTS')
    print('=' * 90)

    all_sweeps = []
    for date_str, session in fade_sessions.items():
        sweeps = find_ibh_sweeps(session, nq, es)
        all_sweeps.extend(sweeps)

    print(f'\nTotal IBH sweep + failure events: {len(all_sweeps)}')
    if not all_sweeps:
        print('No sweeps found! Exiting.')
        return

    sweep_df = pd.DataFrame(all_sweeps)
    print(f'Sessions with sweeps: {sweep_df["session"].nunique()}')
    print(f'Avg sweep above IBH: {sweep_df["sweep_pts"].mean():.1f} pts')
    print(f'Day type distribution: {sweep_df["day_type"].value_counts().to_dict()}')
    print(f'SMT divergence present: {sweep_df["smt_divergence"].sum()}/{len(sweep_df)} ({sweep_df["smt_divergence"].mean()*100:.0f}%)')
    print(f'Negative delta at sweep: {(sweep_df["delta_at_sweep"] < 0).sum()}/{len(sweep_df)} ({(sweep_df["delta_at_sweep"] < 0).mean()*100:.0f}%)')

    # ============================================
    # COMPUTE HTF CONFLUENCE LEVELS
    # ============================================
    print('\n' + '=' * 90)
    print('  PHASE 2: HTF CONFLUENCE LEVELS')
    print('=' * 90)

    # For each session with a sweep, compute HTF levels
    confluence_results = []
    for _, sweep in sweep_df.iterrows():
        session_date = sweep['session']
        htf_levels = compute_htf_levels(nq, session_date)

        # Check which confluence levels the sweep touched
        sweep_high = sweep['sweep_high']
        ib_high = sweep['ib_high']

        near_pdh = False
        near_london_high = False
        in_4h_fvg = False
        in_daily_fvg = False
        near_vwap_upper = False

        # PDH: sweep within 20 pts of PDH
        if 'PDH' in htf_levels:
            pdh = htf_levels['PDH']
            near_pdh = abs(sweep_high - pdh) <= 30 or (sweep_high >= pdh - 10)

        # London high: sweep within 20 pts
        if 'London_High' in htf_levels:
            lh = htf_levels['London_High']
            near_london_high = abs(sweep_high - lh) <= 30 or (sweep_high >= lh - 10)

        # 4H FVG: sweep enters the FVG zone
        if 'FVG_4H_Bear' in htf_levels:
            fvg = htf_levels['FVG_4H_Bear']
            in_4h_fvg = fvg['bottom'] <= sweep_high <= fvg['top'] + 20

        # Daily FVG
        if 'FVG_Daily_Bear' in htf_levels:
            fvg = htf_levels['FVG_Daily_Bear']
            in_daily_fvg = fvg['bottom'] <= sweep_high <= fvg['top'] + 20

        # VWAP upper band
        if 'VWAP_Upper1' in htf_levels:
            vu1 = htf_levels['VWAP_Upper1']
            near_vwap_upper = abs(sweep_high - vu1) <= 20

        any_confluence = near_pdh or near_london_high or in_4h_fvg or in_daily_fvg or near_vwap_upper
        confluence_count = sum([near_pdh, near_london_high, in_4h_fvg, in_daily_fvg, near_vwap_upper])

        confluence_results.append({
            **sweep.to_dict(),
            'near_pdh': near_pdh,
            'near_london_high': near_london_high,
            'in_4h_fvg': in_4h_fvg,
            'in_daily_fvg': in_daily_fvg,
            'near_vwap_upper': near_vwap_upper,
            'any_confluence': any_confluence,
            'confluence_count': confluence_count,
            'htf_levels': htf_levels,
        })

    cdf = pd.DataFrame(confluence_results)

    print(f'\nConfluence level hit rates at IBH sweep:')
    print(f'  PDH:          {cdf["near_pdh"].sum()}/{len(cdf)} ({cdf["near_pdh"].mean()*100:.0f}%)')
    print(f'  London High:  {cdf["near_london_high"].sum()}/{len(cdf)} ({cdf["near_london_high"].mean()*100:.0f}%)')
    print(f'  4H Bear FVG:  {cdf["in_4h_fvg"].sum()}/{len(cdf)} ({cdf["in_4h_fvg"].mean()*100:.0f}%)')
    print(f'  Daily Bear FVG: {cdf["in_daily_fvg"].sum()}/{len(cdf)} ({cdf["in_daily_fvg"].mean()*100:.0f}%)')
    print(f'  VWAP Upper:   {cdf["near_vwap_upper"].sum()}/{len(cdf)} ({cdf["near_vwap_upper"].mean()*100:.0f}%)')
    print(f'  ANY confluece: {cdf["any_confluence"].sum()}/{len(cdf)} ({cdf["any_confluence"].mean()*100:.0f}%)')
    print(f'  2+ confluence: {(cdf["confluence_count"] >= 2).sum()}/{len(cdf)} ({(cdf["confluence_count"] >= 2).mean()*100:.0f}%)')

    # ============================================
    # SIMULATE SHORT TRADES
    # ============================================
    print('\n' + '=' * 90)
    print('  PHASE 3: SHORT TRADE SIMULATION')
    print('=' * 90)

    # Test different configurations
    configs = [
        ('Blind IBH fade (no filter)', lambda row: True),
        ('Negative delta', lambda row: row['delta_at_sweep'] < 0),
        ('SMT divergence', lambda row: row['smt_divergence']),
        ('Any HTF confluence', lambda row: row['any_confluence']),
        ('2+ HTF confluence', lambda row: row['confluence_count'] >= 2),
        ('PDH confluence', lambda row: row['near_pdh']),
        ('London High confluence', lambda row: row['near_london_high']),
        ('4H Bear FVG', lambda row: row['in_4h_fvg']),
        ('Neg delta + any HTF', lambda row: row['delta_at_sweep'] < 0 and row['any_confluence']),
        ('SMT + any HTF', lambda row: row['smt_divergence'] and row['any_confluence']),
        ('Neg delta + SMT', lambda row: row['delta_at_sweep'] < 0 and row['smt_divergence']),
        ('Neg delta + SMT + HTF', lambda row: row['delta_at_sweep'] < 0 and row['smt_divergence'] and row['any_confluence']),
        ('b_day only + any filter', lambda row: row['day_type'] == 'b_day' and (row['any_confluence'] or row['smt_divergence'] or row['delta_at_sweep'] < 0)),
    ]

    # Target: IB mid. Stop: sweep high + buffer
    print(f'\nTarget: IB midpoint | Stop: sweep high + 15% IB range')
    print(f'\n{"Config":<35s} {"Trades":>6s} {"WR":>6s} {"Net pts":>8s} {"Avg pts":>8s} {"MFE":>6s} {"MAE":>6s} {"Win$":>8s} {"Loss$":>8s}')
    print('-' * 100)

    for config_name, filter_fn in configs:
        filtered = cdf[cdf.apply(filter_fn, axis=1)]

        if len(filtered) == 0:
            print(f'{config_name:<35s} {"0":>6s} {"N/A":>6s}')
            continue

        # Take first sweep per session only (avoid over-trading)
        first_per_session = filtered.groupby('session').first().reset_index()

        outcomes = []
        for _, sweep in first_per_session.iterrows():
            session_key = sweep['session']
            if session_key not in sessions:
                continue

            session = sessions[session_key]
            session_bars = session['bars']

            # Stop: sweep high + 15% of IB range
            stop_price = sweep['sweep_high'] + session['ib_range'] * 0.15
            # Target: IB midpoint
            target_price = session['ib_mid']

            result = simulate_short_trade(sweep, session_bars, target_price, stop_price)
            result['entry_price'] = sweep['entry_price']
            result['session'] = session_key
            result['stop_price'] = stop_price
            result['target_price'] = target_price
            result['risk_pts'] = stop_price - sweep['entry_price']
            result['reward_pts'] = sweep['entry_price'] - target_price
            outcomes.append(result)

        if not outcomes:
            print(f'{config_name:<35s} {"0":>6s} {"N/A":>6s}')
            continue

        odf = pd.DataFrame(outcomes)
        wins = odf[odf['pnl_pts'] > 0]
        losses = odf[odf['pnl_pts'] <= 0]
        wr = len(wins) / len(odf) * 100
        net_pts = odf['pnl_pts'].sum()
        avg_pts = odf['pnl_pts'].mean()
        mfe_avg = odf['mfe_pts'].mean()
        mae_avg = odf['mae_pts'].mean()
        win_avg = wins['pnl_pts'].mean() * 2 if len(wins) > 0 else 0  # $2/pt MNQ
        loss_avg = losses['pnl_pts'].mean() * 2 if len(losses) > 0 else 0

        print(f'{config_name:<35s} {len(odf):>6d} {wr:>5.1f}% {net_pts:>+8.1f} {avg_pts:>+8.1f} {mfe_avg:>6.1f} {mae_avg:>6.1f} ${win_avg:>+7.1f} ${loss_avg:>+7.1f}')

    # ============================================
    # DETAILED TRADE-BY-TRADE FOR BEST CONFIG
    # ============================================
    print('\n' + '=' * 90)
    print('  PHASE 4: TRADE-BY-TRADE (ANY HTF CONFLUENCE, FIRST PER SESSION)')
    print('=' * 90)

    # Use "Any HTF confluence" as the main filter
    filtered = cdf[cdf['any_confluence'] == True]
    if len(filtered) > 0:
        first_per_session = filtered.groupby('session').first().reset_index()

        for _, sweep in first_per_session.iterrows():
            session_key = sweep['session']
            if session_key not in sessions:
                continue

            session = sessions[session_key]
            stop_price = sweep['sweep_high'] + session['ib_range'] * 0.15
            target_price = session['ib_mid']
            result = simulate_short_trade(sweep, session['bars'], target_price, stop_price)

            risk = stop_price - sweep['entry_price']
            reward = sweep['entry_price'] - target_price
            rr = reward / risk if risk > 0 else 0

            confluences = []
            if sweep['near_pdh']:
                confluences.append('PDH')
            if sweep['near_london_high']:
                confluences.append('LDN')
            if sweep['in_4h_fvg']:
                confluences.append('4H-FVG')
            if sweep['in_daily_fvg']:
                confluences.append('D-FVG')
            if sweep['near_vwap_upper']:
                confluences.append('VWAP')
            if sweep['smt_divergence']:
                confluences.append('SMT')

            status = 'WIN' if result['pnl_pts'] > 0 else 'LOSS'
            print(f'  {session_key} | {str(sweep["bar_time"])[:5]} | '
                  f'sweep +{sweep["sweep_pts"]:.0f}pts | '
                  f'entry={sweep["entry_price"]:.0f} stop={stop_price:.0f} tgt={target_price:.0f} | '
                  f'R:R={rr:.1f} | {result["exit_reason"]} {result["pnl_pts"]:+.1f}pts | '
                  f'[{status}] | {",".join(confluences)}')

    # Also test wider target (VWAP or IBL)
    print('\n' + '=' * 90)
    print('  PHASE 5: ALTERNATIVE TARGETS (IBL, VWAP, 2R)')
    print('=' * 90)

    for target_name, target_fn in [
        ('IB mid', lambda s: s['ib_mid']),
        ('IB low', lambda s: s['ib_low']),
        ('IB mid - 25% range', lambda s: s['ib_mid'] - s['ib_range'] * 0.25),
        ('2R from entry', None),  # Special: 2x risk
    ]:
        filtered = cdf[cdf['any_confluence'] == True]
        if len(filtered) == 0:
            continue

        first_per_session = filtered.groupby('session').first().reset_index()
        outcomes = []

        for _, sweep in first_per_session.iterrows():
            session_key = sweep['session']
            if session_key not in sessions:
                continue
            session = sessions[session_key]
            stop_price = sweep['sweep_high'] + session['ib_range'] * 0.15

            if target_fn is not None:
                target_price = target_fn(session)
            else:
                # 2R target
                risk = stop_price - sweep['entry_price']
                target_price = sweep['entry_price'] - 2 * risk

            result = simulate_short_trade(sweep, session['bars'], target_price, stop_price)
            outcomes.append(result)

        if outcomes:
            odf = pd.DataFrame(outcomes)
            wins = odf[odf['pnl_pts'] > 0]
            wr = len(wins) / len(odf) * 100
            net = odf['pnl_pts'].sum()
            avg = odf['pnl_pts'].mean()
            print(f'  {target_name:<25s}: {len(odf)} trades, {wr:.0f}% WR, net={net:+.1f} pts, avg={avg:+.1f} pts/trade')


if __name__ == '__main__':
    main()
