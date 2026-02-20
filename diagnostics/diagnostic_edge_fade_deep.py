"""
Deep Dive: Edge Fade Strategy Diagnostic

Questions to answer:
1. AM vs PM: Does the 12:30-13:30 transition zone kill trades?
2. Day type context: b_day vs neutral, and sub-classifications
3. Sweep & reverse: Do entries after IBL sweep + reclaim work better?
4. Double top / retest at IBH before fading: confluence?
5. SMT divergence with ES: NQ weak but ES strong = don't fade
6. Volume profile: HVN/LVN context at entry
7. Time distribution: when do winners vs losers cluster?
8. IB range impact: compressed vs wide IB
9. Session coverage: which day types produce trades, which don't?
10. Multi-trade sessions: are 2nd/3rd trades worse than 1st?
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import time as dt_time
from collections import Counter, defaultdict

project_root = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(project_root))
from data.loader import load_csv
from data.session import filter_rth


def load_trades():
    """Load trade log and filter Edge Fade only."""
    df = pd.read_csv(project_root / 'output' / 'trade_log.csv')
    ef = df[df['strategy_name'] == 'Edge Fade'].copy()
    ef['entry_time'] = pd.to_datetime(ef['entry_time'])
    ef['exit_time'] = pd.to_datetime(ef['exit_time'])
    ef['entry_hour'] = ef['entry_time'].dt.hour
    ef['entry_minute'] = ef['entry_time'].dt.minute
    ef['entry_hhmm'] = ef['entry_time'].dt.strftime('%H:%M')
    ef['entry_date'] = ef['entry_time'].dt.date
    ef['time_obj'] = ef['entry_time'].apply(lambda x: x.time())
    return ef


def load_all_data():
    nq = load_csv('NQ')
    es = load_csv('ES')
    nq_rth = filter_rth(nq)
    es_rth = filter_rth(es)
    return nq, es, nq_rth, es_rth


def compute_session_context(nq_rth, es_rth):
    """Build per-session context for NQ and ES."""
    sessions = {}

    for session_date, group in nq_rth.groupby('session_date'):
        group = group.sort_values('timestamp')
        date_str = str(session_date)[:10]

        ib = group[group['time'] <= dt_time(10, 30)]
        if len(ib) < 10:
            continue

        ib_high = ib['high'].max()
        ib_low = ib['low'].min()
        ib_range = ib_high - ib_low
        ib_mid = (ib_high + ib_low) / 2

        if ib_range < 15:
            continue

        post_ib = group[group['time'] > dt_time(10, 30)]
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

        # VWAP at noon and PM
        vwap_at_noon = None
        noon_bars = group[group['time'] == dt_time(12, 0)]
        if len(noon_bars) > 0:
            vwap_at_noon = noon_bars.iloc[0].get('vwap')

        # How many times did price touch IBL post-IB?
        ibl_touches = 0
        ibh_touches = 0
        for _, bar in post_ib.iterrows():
            if bar['low'] <= ib_low + ib_range * 0.05:
                ibl_touches += 1
            if bar['high'] >= ib_high - ib_range * 0.05:
                ibh_touches += 1

        # Did price sweep below IBL (wick below then close above)?
        ibl_sweep = False
        for _, bar in post_ib.iterrows():
            if bar['low'] < ib_low and bar['close'] > ib_low:
                ibl_sweep = True
                break

        # Session high/low timing
        am_bars = group[group['time'] <= dt_time(12, 0)]
        pm_bars = group[group['time'] > dt_time(12, 0)]

        # Get VWAP position at key times
        vwap_series = group[['timestamp', 'time', 'vwap', 'close']].dropna(subset=['vwap'])

        # Volume profile: approximate HVN/LVN using volume distribution
        # Split IB range into 10 zones, count volume in each
        n_zones = 10
        zone_size = ib_range / n_zones
        volume_profile = [0] * n_zones
        for _, bar in group.iterrows():
            zone_idx = int((bar['close'] - ib_low) / zone_size) if zone_size > 0 else 0
            zone_idx = max(0, min(n_zones - 1, zone_idx))
            vol = bar.get('volume', 0)
            if pd.isna(vol):
                vol = 0
            volume_profile[zone_idx] += vol

        # Find HVN (high volume node) and LVN (low volume node) within IB
        if sum(volume_profile) > 0:
            hvn_zone = np.argmax(volume_profile)
            lvn_zone = np.argmin(volume_profile[:5])  # LVN in lower half only
            hvn_price = ib_low + (hvn_zone + 0.5) * zone_size
            lvn_price = ib_low + (lvn_zone + 0.5) * zone_size
        else:
            hvn_price = ib_mid
            lvn_price = ib_low

        sessions[date_str] = {
            'ib_high': ib_high, 'ib_low': ib_low, 'ib_range': ib_range,
            'ib_mid': ib_mid, 'day_type': day_type,
            'ext_up': ext_up, 'ext_down': ext_down,
            'session_high': session_high, 'session_low': session_low,
            'session_close': session_close,
            'ibl_touches': ibl_touches, 'ibh_touches': ibh_touches,
            'ibl_sweep': ibl_sweep,
            'hvn_price': hvn_price, 'lvn_price': lvn_price,
            'volume_profile': volume_profile,
            'bars': group,
        }

    # ES context for SMT
    es_sessions = {}
    for session_date, group in es_rth.groupby('session_date'):
        group = group.sort_values('timestamp')
        date_str = str(session_date)[:10]

        ib = group[group['time'] <= dt_time(10, 30)]
        if len(ib) < 5:
            continue

        es_sessions[date_str] = {
            'ib_high': ib['high'].max(),
            'ib_low': ib['low'].min(),
            'session_high': group['high'].max(),
            'session_low': group['low'].min(),
            'session_close': group.iloc[-1]['close'],
            'bars': group,
        }

    return sessions, es_sessions


def main():
    print('Loading trades and data...')
    ef = load_trades()
    nq, es, nq_rth, es_rth = load_all_data()
    sessions, es_sessions = compute_session_context(nq_rth, es_rth)

    print(f'Edge Fade trades: {len(ef)}')
    print(f'Sessions: {len(sessions)}')
    print(f'ES sessions: {len(es_sessions)}')

    # ================================================================
    #  1. SESSION COVERAGE: What day types produce trades?
    # ================================================================
    print('\n' + '=' * 90)
    print('  1. SESSION COVERAGE BY DAY TYPE')
    print('=' * 90)

    day_type_counts = Counter(s['day_type'] for s in sessions.values())
    trade_dates = set(ef['entry_date'].astype(str).unique())

    print(f'\n{"Day Type":<15s} {"Sessions":>8s} {"With Trade":>10s} {"No Trade":>10s} {"Coverage":>10s}')
    print('-' * 60)

    for dt in ['trend_up', 'p_day', 'b_day', 'b_day_bear', 'neutral', 'trend_down']:
        dt_sessions = [d for d, s in sessions.items() if s['day_type'] == dt]
        dt_with_trade = [d for d in dt_sessions if d in trade_dates]
        n_total = len(dt_sessions)
        n_trade = len(dt_with_trade)
        coverage = n_trade / n_total * 100 if n_total > 0 else 0
        print(f'{dt:<15s} {n_total:>8d} {n_trade:>10d} {n_total - n_trade:>10d} {coverage:>9.0f}%')

    total = len(sessions)
    total_trade = len(trade_dates)
    print(f'{"TOTAL":<15s} {total:>8d} {total_trade:>10d} {total - total_trade:>10d} {total_trade/total*100:>9.0f}%')

    # ================================================================
    #  2. TIME-OF-DAY ANALYSIS
    # ================================================================
    print('\n' + '=' * 90)
    print('  2. TIME-OF-DAY ANALYSIS')
    print('=' * 90)

    # Create time bins
    def time_bucket(t):
        h = t.hour
        m = t.minute
        if h < 11:
            return '10:30-11:00'
        elif h == 11 and m < 30:
            return '11:00-11:30'
        elif h == 11:
            return '11:30-12:00'
        elif h == 12 and m < 30:
            return '12:00-12:30'
        elif h == 12:
            return '12:30-13:00'
        elif h == 13 and m < 30:
            return '13:00-13:30'
        elif h == 13:
            return '13:30-14:00'
        elif h == 14 and m < 30:
            return '14:00-14:30'
        elif h == 14:
            return '14:30-15:00'
        else:
            return '15:00+'

    ef['time_bucket'] = ef['time_obj'].apply(time_bucket)

    print(f'\n{"Time Bucket":<16s} {"Trades":>7s} {"Wins":>5s} {"WR":>5s} {"Net $":>9s} {"Avg $":>8s} {"AvgWin":>8s} {"AvgLoss":>8s}')
    print('-' * 80)

    for bucket in ['10:30-11:00', '11:00-11:30', '11:30-12:00', '12:00-12:30',
                    '12:30-13:00', '13:00-13:30', '13:30-14:00', '14:00-14:30',
                    '14:30-15:00', '15:00+']:
        sub = ef[ef['time_bucket'] == bucket]
        if len(sub) == 0:
            print(f'{bucket:<16s} {"0":>7s}')
            continue
        wins = sub[sub['is_winner'] == 1]
        losses = sub[sub['is_winner'] == 0]
        wr = len(wins) / len(sub) * 100
        avg_win = wins['net_pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['net_pnl'].mean() if len(losses) > 0 else 0
        print(f'{bucket:<16s} {len(sub):>7d} {len(wins):>5d} {wr:>4.0f}% ${sub["net_pnl"].sum():>+8,.0f} '
              f'${sub["net_pnl"].mean():>+7,.0f} ${avg_win:>+7,.0f} ${avg_loss:>+7,.0f}')

    # AM vs PM split
    print('\n  --- AM vs PM ---')
    am_trades = ef[ef['time_obj'] < dt_time(12, 30)]
    pm_early = ef[(ef['time_obj'] >= dt_time(12, 30)) & (ef['time_obj'] < dt_time(13, 30))]
    pm_late = ef[ef['time_obj'] >= dt_time(13, 30)]

    for label, sub in [('AM (10:30-12:30)', am_trades),
                        ('TRANSITION (12:30-13:30)', pm_early),
                        ('PM (13:30+)', pm_late)]:
        if len(sub) == 0:
            print(f'  {label}: 0 trades')
            continue
        wins = sub[sub['is_winner'] == 1]
        losses = sub[sub['is_winner'] == 0]
        wr = len(wins) / len(sub) * 100
        exp = sub['net_pnl'].mean()
        avg_win = wins['net_pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['net_pnl'].mean() if len(losses) > 0 else 0

        # Consecutive loss analysis
        wl = sub['is_winner'].tolist()
        consec = []
        c = 0
        for v in wl:
            if v == 0:
                c += 1
            else:
                if c > 0: consec.append(c)
                c = 0
        if c > 0: consec.append(c)
        max_cl = max(consec) if consec else 0

        # Max drawdown
        cum = sub['net_pnl'].cumsum()
        dd = cum - cum.cummax()
        max_dd = dd.min()

        print(f'  {label}: {len(sub)} trades, {wr:.0f}% WR, ${sub["net_pnl"].sum():+,.0f} net, '
              f'${exp:+,.0f}/trade, MaxDD ${max_dd:+,.0f}, MaxConsecL {max_cl}')

    # ================================================================
    #  3. MULTI-TRADE SESSION ANALYSIS
    # ================================================================
    print('\n' + '=' * 90)
    print('  3. MULTI-TRADE SESSION ANALYSIS')
    print('=' * 90)

    trades_per_session = ef.groupby('entry_date').size()
    print(f'\n  Trades per session distribution:')
    for n_trades, count in trades_per_session.value_counts().sort_index().items():
        print(f'    {n_trades} trade(s): {count} sessions')

    # Assign trade number within session
    ef_sorted = ef.sort_values('entry_time')
    ef_sorted['trade_num'] = ef_sorted.groupby('entry_date').cumcount() + 1

    print(f'\n{"Trade #":<10s} {"Trades":>7s} {"Wins":>5s} {"WR":>5s} {"Net $":>9s} {"Avg $":>8s}')
    print('-' * 50)
    for tn in sorted(ef_sorted['trade_num'].unique()):
        sub = ef_sorted[ef_sorted['trade_num'] == tn]
        wins = sub[sub['is_winner'] == 1]
        wr = len(wins) / len(sub) * 100
        print(f'Trade #{tn:<5d} {len(sub):>7d} {len(wins):>5d} {wr:>4.0f}% ${sub["net_pnl"].sum():>+8,.0f} ${sub["net_pnl"].mean():>+7,.0f}')

    # ================================================================
    #  4. IB RANGE IMPACT
    # ================================================================
    print('\n' + '=' * 90)
    print('  4. IB RANGE IMPACT')
    print('=' * 90)

    # Merge session context into trades
    ef_ctx = ef_sorted.copy()
    ef_ctx['date_key'] = ef_ctx['entry_date'].astype(str)

    ib_ranges = []
    for _, row in ef_ctx.iterrows():
        dk = row['date_key']
        if dk in sessions:
            ib_ranges.append(sessions[dk]['ib_range'])
        else:
            ib_ranges.append(None)
    ef_ctx['ib_range_val'] = ib_ranges

    ef_ctx = ef_ctx.dropna(subset=['ib_range_val'])

    # IB range buckets
    def ib_bucket(r):
        if r < 100:
            return '<100 (tight)'
        elif r < 150:
            return '100-150'
        elif r < 200:
            return '150-200'
        elif r < 250:
            return '200-250'
        elif r < 300:
            return '250-300'
        else:
            return '300+ (wide)'

    ef_ctx['ib_bucket'] = ef_ctx['ib_range_val'].apply(ib_bucket)

    print(f'\n{"IB Range":<16s} {"Trades":>7s} {"Wins":>5s} {"WR":>5s} {"Net $":>9s} {"Avg $":>8s} {"Stop Dist":>10s}')
    print('-' * 70)
    for bucket in ['<100 (tight)', '100-150', '150-200', '200-250', '250-300', '300+ (wide)']:
        sub = ef_ctx[ef_ctx['ib_bucket'] == bucket]
        if len(sub) == 0:
            continue
        wins = sub[sub['is_winner'] == 1]
        wr = len(wins) / len(sub) * 100
        avg_stop_dist = (sub['entry_price'] - sub['stop_price']).mean()
        print(f'{bucket:<16s} {len(sub):>7d} {len(wins):>5d} {wr:>4.0f}% ${sub["net_pnl"].sum():>+8,.0f} '
              f'${sub["net_pnl"].mean():>+7,.0f} {avg_stop_dist:>9.0f}pts')

    # ================================================================
    #  5. IBL SWEEP CONTEXT
    # ================================================================
    print('\n' + '=' * 90)
    print('  5. IBL SWEEP + RECLAIM CONTEXT')
    print('=' * 90)

    # For each trade, check if IBL was swept before entry
    sweep_labels = []
    for _, row in ef_ctx.iterrows():
        dk = row['date_key']
        if dk in sessions:
            s = sessions[dk]
            entry_time = row['entry_time']
            bars = s['bars']
            pre_entry = bars[bars['timestamp'] < entry_time]
            post_ib_pre_entry = pre_entry[pre_entry['time'] > dt_time(10, 30)]

            # Did any bar sweep below IBL before this entry?
            swept = False
            for _, bar in post_ib_pre_entry.iterrows():
                if bar['low'] < s['ib_low'] and bar['close'] > s['ib_low']:
                    swept = True
                    break
                if bar['close'] < s['ib_low']:
                    swept = True
                    break

            sweep_labels.append('AFTER_SWEEP' if swept else 'NO_SWEEP')
        else:
            sweep_labels.append('UNKNOWN')

    ef_ctx['sweep_context'] = sweep_labels

    for ctx in ['NO_SWEEP', 'AFTER_SWEEP']:
        sub = ef_ctx[ef_ctx['sweep_context'] == ctx]
        if len(sub) == 0:
            continue
        wins = sub[sub['is_winner'] == 1]
        wr = len(wins) / len(sub) * 100
        exp = sub['net_pnl'].mean()
        print(f'  {ctx}: {len(sub)} trades, {wr:.0f}% WR, ${sub["net_pnl"].sum():+,.0f} net, ${exp:+,.0f}/trade')

    # ================================================================
    #  6. SMT DIVERGENCE WITH ES
    # ================================================================
    print('\n' + '=' * 90)
    print('  6. SMT DIVERGENCE WITH ES')
    print('=' * 90)

    smt_labels = []
    for _, row in ef_ctx.iterrows():
        dk = row['date_key']
        if dk in sessions and dk in es_sessions:
            nq_s = sessions[dk]
            es_s = es_sessions[dk]

            entry_time = row['entry_time']

            # NQ making new session low near entry, but ES NOT making new low
            nq_bars = nq_s['bars']
            es_bars = es_s['bars']

            nq_pre = nq_bars[nq_bars['timestamp'] <= entry_time]
            es_pre = es_bars[es_bars['timestamp'] <= entry_time]

            if len(nq_pre) < 5 or len(es_pre) < 5:
                smt_labels.append('NO_DATA')
                continue

            # Check last 10 bars: is NQ making lower lows while ES holds?
            nq_recent = nq_pre.tail(10)
            es_recent = es_pre.tail(10)

            nq_low_10 = nq_recent['low'].min()
            nq_low_prior = nq_pre.iloc[:-10]['low'].min() if len(nq_pre) > 10 else nq_pre['low'].min()

            es_low_10 = es_recent['low'].min()
            es_low_prior = es_pre.iloc[:-10]['low'].min() if len(es_pre) > 10 else es_pre['low'].min()

            # SMT bullish divergence: NQ new low, ES higher low
            nq_new_low = nq_low_10 <= nq_low_prior
            es_higher_low = es_low_10 > es_low_prior

            if nq_new_low and es_higher_low:
                smt_labels.append('SMT_BULLISH')  # Good for long fade
            elif not nq_new_low and not es_higher_low:
                smt_labels.append('ALIGNED_DOWN')  # Both weak, don't fade
            else:
                smt_labels.append('NEUTRAL')
        else:
            smt_labels.append('NO_DATA')

    ef_ctx['smt'] = smt_labels

    for smt in ['SMT_BULLISH', 'NEUTRAL', 'ALIGNED_DOWN', 'NO_DATA']:
        sub = ef_ctx[ef_ctx['smt'] == smt]
        if len(sub) == 0:
            continue
        wins = sub[sub['is_winner'] == 1]
        wr = len(wins) / len(sub) * 100
        print(f'  {smt}: {len(sub)} trades, {wr:.0f}% WR, ${sub["net_pnl"].sum():+,.0f} net, ${sub["net_pnl"].mean():+,.0f}/trade')

    # ================================================================
    #  7. IBH RETEST / DOUBLE TOP BEFORE FADE
    # ================================================================
    print('\n' + '=' * 90)
    print('  7. IBH DOUBLE TOP / RETEST BEFORE ENTRY')
    print('=' * 90)

    ibh_context = []
    for _, row in ef_ctx.iterrows():
        dk = row['date_key']
        if dk in sessions:
            s = sessions[dk]
            entry_time = row['entry_time']
            bars = s['bars']
            pre_entry = bars[(bars['timestamp'] < entry_time) & (bars['time'] > dt_time(10, 30))]

            # Count IBH touches before this trade
            ibh = s['ib_high']
            ib_range = s['ib_range']
            ibh_touch_count = 0
            for _, bar in pre_entry.iterrows():
                if bar['high'] >= ibh - ib_range * 0.05:
                    ibh_touch_count += 1

            if ibh_touch_count >= 3:
                ibh_context.append('IBH_3+_TOUCHES')
            elif ibh_touch_count >= 1:
                ibh_context.append('IBH_1-2_TOUCHES')
            else:
                ibh_context.append('NO_IBH_TOUCH')
        else:
            ibh_context.append('UNKNOWN')

    ef_ctx['ibh_context'] = ibh_context

    for ctx in ['NO_IBH_TOUCH', 'IBH_1-2_TOUCHES', 'IBH_3+_TOUCHES']:
        sub = ef_ctx[ef_ctx['ibh_context'] == ctx]
        if len(sub) == 0:
            continue
        wins = sub[sub['is_winner'] == 1]
        wr = len(wins) / len(sub) * 100
        print(f'  {ctx}: {len(sub)} trades, {wr:.0f}% WR, ${sub["net_pnl"].sum():+,.0f} net, ${sub["net_pnl"].mean():+,.0f}/trade')

    # ================================================================
    #  8. VOLUME PROFILE CONTEXT (HVN/LVN)
    # ================================================================
    print('\n' + '=' * 90)
    print('  8. VOLUME PROFILE: ENTRY NEAR HVN vs LVN')
    print('=' * 90)

    vp_context = []
    for _, row in ef_ctx.iterrows():
        dk = row['date_key']
        if dk in sessions:
            s = sessions[dk]
            entry_price = row['entry_price']
            hvn = s['hvn_price']
            lvn = s['lvn_price']
            ib_range = s['ib_range']

            dist_to_hvn = abs(entry_price - hvn) / ib_range if ib_range > 0 else 999
            dist_to_lvn = abs(entry_price - lvn) / ib_range if ib_range > 0 else 999

            if dist_to_lvn < 0.15:
                vp_context.append('NEAR_LVN')  # Low volume = price rejects quickly
            elif dist_to_hvn < 0.15:
                vp_context.append('NEAR_HVN')  # High volume = price accepted here
            else:
                vp_context.append('BETWEEN')
        else:
            vp_context.append('UNKNOWN')

    ef_ctx['vp_context'] = vp_context

    for ctx in ['NEAR_LVN', 'BETWEEN', 'NEAR_HVN']:
        sub = ef_ctx[ef_ctx['vp_context'] == ctx]
        if len(sub) == 0:
            continue
        wins = sub[sub['is_winner'] == 1]
        wr = len(wins) / len(sub) * 100
        print(f'  {ctx}: {len(sub)} trades, {wr:.0f}% WR, ${sub["net_pnl"].sum():+,.0f} net, ${sub["net_pnl"].mean():+,.0f}/trade')

    # ================================================================
    #  9. EXIT REASON DEEP DIVE
    # ================================================================
    print('\n' + '=' * 90)
    print('  9. EXIT REASON ANALYSIS')
    print('=' * 90)

    for reason in ef['exit_reason'].unique():
        sub = ef[ef['exit_reason'] == reason]
        wins = sub[sub['is_winner'] == 1]
        wr = len(wins) / len(sub) * 100 if len(sub) > 0 else 0
        print(f'  {reason}: {len(sub)} trades, {wr:.0f}% WR, ${sub["net_pnl"].sum():+,.0f} net, '
              f'avg ${sub["net_pnl"].mean():+,.0f}, avg bars {sub["bars_held"].mean():.0f}')

    # ================================================================
    #  10. BARS HELD ANALYSIS
    # ================================================================
    print('\n' + '=' * 90)
    print('  10. BARS HELD ANALYSIS')
    print('=' * 90)

    def bars_bucket(b):
        if b <= 10:
            return '0-10 (quick)'
        elif b <= 30:
            return '11-30'
        elif b <= 60:
            return '31-60'
        elif b <= 120:
            return '61-120'
        else:
            return '120+ (long hold)'

    ef['bars_bucket'] = ef['bars_held'].apply(bars_bucket)

    for bucket in ['0-10 (quick)', '11-30', '31-60', '61-120', '120+ (long hold)']:
        sub = ef[ef['bars_bucket'] == bucket]
        if len(sub) == 0:
            continue
        wins = sub[sub['is_winner'] == 1]
        wr = len(wins) / len(sub) * 100
        print(f'  {bucket}: {len(sub)} trades, {wr:.0f}% WR, ${sub["net_pnl"].sum():+,.0f} net, ${sub["net_pnl"].mean():+,.0f}/trade')

    # ================================================================
    #  11. COMBINED FILTER ANALYSIS
    # ================================================================
    print('\n' + '=' * 90)
    print('  11. COMBINED FILTER ANALYSIS')
    print('=' * 90)

    # Test combinations of filters
    filters = {
        'AM_ONLY': ef_ctx['time_obj'] < dt_time(12, 30),
        'AFTER_SWEEP': ef_ctx['sweep_context'] == 'AFTER_SWEEP',
        'NO_PM_TRANSITION': ~((ef_ctx['time_obj'] >= dt_time(12, 30)) & (ef_ctx['time_obj'] < dt_time(13, 30))),
        'SMT_BULLISH': ef_ctx['smt'] == 'SMT_BULLISH',
        'IBH_TOUCHED': ef_ctx['ibh_context'].isin(['IBH_1-2_TOUCHES', 'IBH_3+_TOUCHES']),
        'NEAR_LVN': ef_ctx['vp_context'] == 'NEAR_LVN',
        'TIGHT_IB(<200)': ef_ctx['ib_range_val'] < 200,
        'TRADE_1_ONLY': ef_ctx['trade_num'] == 1,
        'TRADE_1_OR_2': ef_ctx['trade_num'] <= 2,
    }

    print(f'\n{"Filter":<25s} {"Trades":>7s} {"Wins":>5s} {"WR":>5s} {"Net $":>9s} {"Exp/Trade":>10s} {"MaxCL":>6s}')
    print('-' * 75)

    # Baseline (no filter)
    sub = ef_ctx
    wins = sub[sub['is_winner'] == 1]
    wr = len(wins) / len(sub) * 100
    print(f'{"BASELINE (no filter)":<25s} {len(sub):>7d} {len(wins):>5d} {wr:>4.0f}% ${sub["net_pnl"].sum():>+8,.0f} ${sub["net_pnl"].mean():>+9,.0f}')

    for name, mask in filters.items():
        sub = ef_ctx[mask]
        if len(sub) < 3:
            continue
        wins = sub[sub['is_winner'] == 1]
        wr = len(wins) / len(sub) * 100
        # Max consec loss
        wl = sub.sort_values('entry_time')['is_winner'].tolist()
        consec = []
        c = 0
        for v in wl:
            if v == 0:
                c += 1
            else:
                if c > 0: consec.append(c)
                c = 0
        if c > 0: consec.append(c)
        max_cl = max(consec) if consec else 0

        print(f'{name:<25s} {len(sub):>7d} {len(wins):>5d} {wr:>4.0f}% ${sub["net_pnl"].sum():>+8,.0f} ${sub["net_pnl"].mean():>+9,.0f} {max_cl:>6d}')

    # Test combined filters
    print(f'\n  --- COMBINED FILTERS ---')

    combos = {
        'AM + SWEEP': (ef_ctx['time_obj'] < dt_time(12, 30)) & (ef_ctx['sweep_context'] == 'AFTER_SWEEP'),
        'AM + NO_SWEEP': (ef_ctx['time_obj'] < dt_time(12, 30)) & (ef_ctx['sweep_context'] == 'NO_SWEEP'),
        'AM + Trade1': (ef_ctx['time_obj'] < dt_time(12, 30)) & (ef_ctx['trade_num'] == 1),
        'NO_TRANS + Trade<=2': ~((ef_ctx['time_obj'] >= dt_time(12, 30)) & (ef_ctx['time_obj'] < dt_time(13, 30))) & (ef_ctx['trade_num'] <= 2),
        'AM + IBH_TOUCHED': (ef_ctx['time_obj'] < dt_time(12, 30)) & ef_ctx['ibh_context'].isin(['IBH_1-2_TOUCHES', 'IBH_3+_TOUCHES']),
        'SWEEP + TIGHT_IB': (ef_ctx['sweep_context'] == 'AFTER_SWEEP') & (ef_ctx['ib_range_val'] < 200),
        'AM + SWEEP + Trade1': (ef_ctx['time_obj'] < dt_time(12, 30)) & (ef_ctx['sweep_context'] == 'AFTER_SWEEP') & (ef_ctx['trade_num'] == 1),
        'AM + LVN': (ef_ctx['time_obj'] < dt_time(12, 30)) & (ef_ctx['vp_context'] == 'NEAR_LVN'),
    }

    for name, mask in combos.items():
        sub = ef_ctx[mask]
        if len(sub) < 3:
            print(f'  {name}: {len(sub)} trades (too few)')
            continue
        wins = sub[sub['is_winner'] == 1]
        wr = len(wins) / len(sub) * 100
        wl = sub.sort_values('entry_time')['is_winner'].tolist()
        consec = []
        c = 0
        for v in wl:
            if v == 0:
                c += 1
            else:
                if c > 0: consec.append(c)
                c = 0
        if c > 0: consec.append(c)
        max_cl = max(consec) if consec else 0

        cum = sub.sort_values('entry_time')['net_pnl'].cumsum()
        dd = cum - cum.cummax()
        max_dd = dd.min()

        print(f'  {name:<25s} {len(sub):>5d} trades, {wr:>4.0f}% WR, ${sub["net_pnl"].sum():>+8,.0f} net, '
              f'${sub["net_pnl"].mean():>+7,.0f}/trade, MaxDD ${max_dd:+,.0f}, MaxCL {max_cl}')

    # ================================================================
    #  12. WORST LOSING STREAKS - TRADE BY TRADE
    # ================================================================
    print('\n' + '=' * 90)
    print('  12. WORST LOSING STREAKS')
    print('=' * 90)

    ef_sorted = ef.sort_values('entry_time').reset_index(drop=True)
    streak_trades = []
    current_streak = []

    for _, row in ef_sorted.iterrows():
        if row['is_winner'] == 0:
            current_streak.append(row)
        else:
            if len(current_streak) >= 4:
                streak_trades.append(current_streak.copy())
            current_streak = []
    if len(current_streak) >= 4:
        streak_trades.append(current_streak)

    for i, streak in enumerate(streak_trades):
        total_loss = sum(t['net_pnl'] for t in streak)
        print(f'\n  Streak {i+1}: {len(streak)} consecutive losses, ${total_loss:+,.0f}')
        for t in streak:
            print(f'    {str(t["entry_time"])[:16]} {t["exit_reason"]:>6s} ${t["net_pnl"]:+,.0f} '
                  f'bars={t["bars_held"]} day={t["day_type"]}')


if __name__ == '__main__':
    main()
