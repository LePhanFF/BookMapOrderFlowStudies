"""
Full Portfolio Report v14.2
- Edge Fade 90 trades broken out by every filter
- All strategies combined with optimized Edge Fade
- Expectancy, drawdown, and recovery metrics for everything
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import time as dt_time
from collections import Counter

project_root = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(project_root))
from data.loader import load_csv
from data.session import filter_rth


def load_all():
    df = pd.read_csv(project_root / 'output' / 'trade_log.csv')
    nq = load_csv('NQ')
    es = load_csv('ES')
    nq_rth = filter_rth(nq)
    es_rth = filter_rth(es)
    return df, nq, es, nq_rth, es_rth


def build_sessions(nq_rth, es_rth):
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

        # IBL sweep detection
        ibl_sweep = False
        for _, bar in post_ib.iterrows():
            if bar['low'] < ib_low and bar['close'] > ib_low:
                ibl_sweep = True
                break
            if bar['close'] < ib_low:
                ibl_sweep = True
                break

        # Volume profile: 10 zones within IB
        n_zones = 10
        zone_size = ib_range / n_zones if ib_range > 0 else 1
        volume_profile = [0.0] * n_zones
        for _, bar in group.iterrows():
            zone_idx = int((bar['close'] - ib_low) / zone_size)
            zone_idx = max(0, min(n_zones - 1, zone_idx))
            vol = bar.get('volume', 0)
            if pd.isna(vol):
                vol = 0
            volume_profile[zone_idx] += vol

        if sum(volume_profile) > 0:
            hvn_zone = np.argmax(volume_profile)
            lvn_zone = np.argmin(volume_profile[:5])
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
            'ibl_sweep': ibl_sweep,
            'hvn_price': hvn_price, 'lvn_price': lvn_price,
            'bars': group,
        }

    # ES for SMT
    es_sessions = {}
    for session_date, group in es_rth.groupby('session_date'):
        group = group.sort_values('timestamp')
        date_str = str(session_date)[:10]
        ib = group[group['time'] <= dt_time(10, 30)]
        if len(ib) < 5:
            continue
        es_sessions[date_str] = {
            'ib_high': ib['high'].max(), 'ib_low': ib['low'].min(),
            'session_high': group['high'].max(), 'session_low': group['low'].min(),
            'bars': group,
        }

    return sessions, es_sessions


def enrich_edge_fade(ef, sessions, es_sessions):
    """Add all contextual columns to Edge Fade trades."""
    ef = ef.copy()
    ef['entry_time'] = pd.to_datetime(ef['entry_time'])
    ef['exit_time'] = pd.to_datetime(ef['exit_time'])
    ef['entry_date'] = ef['entry_time'].dt.date
    ef['date_key'] = ef['entry_date'].astype(str)
    ef['time_obj'] = ef['entry_time'].apply(lambda x: x.time())

    # Trade number within session
    ef = ef.sort_values('entry_time').reset_index(drop=True)
    ef['trade_num'] = ef.groupby('entry_date').cumcount() + 1

    # Time bucket
    def time_bucket(t):
        if t < dt_time(11, 0): return 'AM1 10:30-11:00'
        elif t < dt_time(11, 30): return 'AM2 11:00-11:30'
        elif t < dt_time(12, 0): return 'AM3 11:30-12:00'
        elif t < dt_time(12, 30): return 'AM4 12:00-12:30'
        elif t < dt_time(13, 0): return 'TR1 12:30-13:00'
        elif t < dt_time(13, 30): return 'TR2 13:00-13:30'
        elif t < dt_time(14, 0): return 'PM1 13:30-14:00'
        elif t < dt_time(14, 30): return 'PM2 14:00-14:30'
        elif t < dt_time(15, 0): return 'PM3 14:30-15:00'
        else: return 'PM4 15:00+'
    ef['time_bucket'] = ef['time_obj'].apply(time_bucket)

    # Session context
    ib_ranges, day_types_actual, sweeps, vp_ctx, ext_downs = [], [], [], [], []

    for _, row in ef.iterrows():
        dk = row['date_key']
        if dk not in sessions:
            ib_ranges.append(None)
            day_types_actual.append('unknown')
            sweeps.append('UNKNOWN')
            vp_ctx.append('UNKNOWN')
            ext_downs.append(0)
            continue

        s = sessions[dk]
        ib_ranges.append(s['ib_range'])
        day_types_actual.append(s['day_type'])
        ext_downs.append(s['ext_down'])

        # Sweep before entry
        entry_time = row['entry_time']
        bars = s['bars']
        pre_entry = bars[(bars['timestamp'] < entry_time) & (bars['time'] > dt_time(10, 30))]
        swept = False
        for _, bar in pre_entry.iterrows():
            if bar['low'] < s['ib_low'] and bar['close'] > s['ib_low']:
                swept = True
                break
            if bar['close'] < s['ib_low']:
                swept = True
                break
        sweeps.append('SWEEP' if swept else 'NO_SWEEP')

        # Volume profile context
        entry_price = row['entry_price']
        dist_hvn = abs(entry_price - s['hvn_price']) / s['ib_range'] if s['ib_range'] > 0 else 999
        dist_lvn = abs(entry_price - s['lvn_price']) / s['ib_range'] if s['ib_range'] > 0 else 999
        if dist_lvn < 0.15:
            vp_ctx.append('LVN')
        elif dist_hvn < 0.15:
            vp_ctx.append('HVN')
        else:
            vp_ctx.append('MID')

    ef['ib_range_val'] = ib_ranges
    ef['actual_day_type'] = day_types_actual
    ef['sweep'] = sweeps
    ef['vp'] = vp_ctx
    ef['ext_down'] = ext_downs

    # IB range bucket
    def ib_bkt(r):
        if r is None: return 'unknown'
        if r < 100: return '<100'
        elif r < 150: return '100-150'
        elif r < 200: return '150-200'
        elif r < 250: return '200-250'
        elif r < 300: return '250-300'
        else: return '300+'
    ef['ib_bucket'] = ef['ib_range_val'].apply(ib_bkt)

    # AM/PM classification
    def session_period(t):
        if t < dt_time(12, 30): return 'AM'
        elif t < dt_time(13, 30): return 'TRANSITION'
        else: return 'PM'
    ef['period'] = ef['time_obj'].apply(session_period)

    # Is this a bearish day? (ext_down > 0.3)
    ef['bearish_day'] = ef['ext_down'] > 0.3

    return ef


def calc_metrics(sub, name=''):
    """Calculate comprehensive metrics for a trade set."""
    if len(sub) == 0:
        return None

    sub = sub.sort_values('entry_time').reset_index(drop=True)
    n = len(sub)
    wins = sub[sub['is_winner'] == 1]
    losses = sub[sub['is_winner'] == 0]

    wr = len(wins) / n * 100
    avg_win = wins['net_pnl'].mean() if len(wins) > 0 else 0
    avg_loss = losses['net_pnl'].mean() if len(losses) > 0 else 0
    expectancy = sub['net_pnl'].mean()
    net = sub['net_pnl'].sum()
    pf = wins['net_pnl'].sum() / abs(losses['net_pnl'].sum()) if len(losses) > 0 and losses['net_pnl'].sum() != 0 else 999

    # Drawdown
    cum = sub['net_pnl'].cumsum()
    dd = cum - cum.cummax()
    max_dd = dd.min()

    # Recovery
    dd_to_exp = abs(max_dd / expectancy) if expectancy > 0 else 999

    # Max consecutive losses
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

    # W:L ratio
    wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 999

    return {
        'name': name, 'n': n, 'wins': len(wins), 'losses': len(losses),
        'wr': wr, 'avg_win': avg_win, 'avg_loss': avg_loss,
        'wl_ratio': wl_ratio, 'expectancy': expectancy, 'net': net,
        'pf': pf, 'max_dd': max_dd, 'dd_to_exp': dd_to_exp,
        'max_cl': max_cl,
    }


def print_metrics(m, indent=''):
    if m is None:
        return
    pf_s = f'{m["pf"]:.2f}' if m['pf'] < 900 else 'INF'
    wl_s = f'{m["wl_ratio"]:.1f}' if m['wl_ratio'] < 900 else 'INF'
    dd_s = f'{m["dd_to_exp"]:.1f}' if m['dd_to_exp'] < 900 else 'N/A'
    print(f'{indent}{m["name"]:<35s} {m["n"]:>4d} {m["wins"]:>3d}W/{m["losses"]:>2d}L '
          f'{m["wr"]:>5.0f}%  ${m["avg_win"]:>+7,.0f} ${m["avg_loss"]:>+7,.0f}  {wl_s:>5s}  '
          f'${m["expectancy"]:>+7,.0f}  ${m["net"]:>+9,.0f}  ${m["max_dd"]:>+8,.0f}  {dd_s:>5s}  {pf_s:>6s}  {m["max_cl"]:>3d}')


def print_table_header(indent=''):
    print(f'{indent}{"Strategy":<35s} {"N":>4s} {"W/L":>7s} '
          f'{"WR":>5s}  {"AvgWin":>8s} {"AvgLoss":>8s}  {"W:L":>5s}  '
          f'{"Exp/$":>8s}  {"Net $":>10s}  {"MaxDD":>9s}  {"DD/E":>5s}  {"PF":>6s}  {"MCL":>3s}')
    print(f'{indent}{"-"*130}')


def main():
    print('Loading data...')
    df, nq, es, nq_rth, es_rth = load_all()
    sessions, es_sessions = build_sessions(nq_rth, es_rth)

    # Separate strategies
    ef_raw = df[df['strategy_name'] == 'Edge Fade'].copy()
    playbook = df[df['strategy_name'] != 'Edge Fade'].copy()

    # Enrich Edge Fade
    ef = enrich_edge_fade(ef_raw, sessions, es_sessions)

    print(f'\nTotal trades: {len(df)}')
    print(f'Edge Fade: {len(ef)}, Playbook: {len(playbook)}')
    print(f'Sessions: {len(sessions)}')

    # ==================================================================
    #  SECTION A: EDGE FADE 90 TRADES - FULL BREAKOUT
    # ==================================================================
    print('\n')
    print('#' * 130)
    print('#  SECTION A: EDGE FADE - 90 TRADES BROKEN OUT BY EVERY FILTER')
    print('#' * 130)

    # A1: BASELINE
    print('\n  A1. BASELINE (all 90)')
    print_table_header('  ')
    print_metrics(calc_metrics(ef, 'EDGE FADE (all 90)'), '  ')

    # A2: BY TIME PERIOD
    print('\n  A2. BY TIME PERIOD')
    print_table_header('  ')
    for period in ['AM', 'TRANSITION', 'PM']:
        sub = ef[ef['period'] == period]
        print_metrics(calc_metrics(sub, f'  {period}'), '  ')

    # A3: BY TIME BUCKET (30-min)
    print('\n  A3. BY 30-MIN TIME BUCKET')
    print_table_header('  ')
    for bucket in sorted(ef['time_bucket'].unique()):
        sub = ef[ef['time_bucket'] == bucket]
        if len(sub) > 0:
            print_metrics(calc_metrics(sub, f'  {bucket}'), '  ')

    # A4: BY IB RANGE
    print('\n  A4. BY IB RANGE')
    print_table_header('  ')
    for bkt in ['<100', '100-150', '150-200', '200-250', '250-300', '300+']:
        sub = ef[ef['ib_bucket'] == bkt]
        if len(sub) > 0:
            print_metrics(calc_metrics(sub, f'  IB {bkt}'), '  ')

    # A5: BY SWEEP CONTEXT
    print('\n  A5. BY IBL SWEEP CONTEXT')
    print_table_header('  ')
    for ctx in ['SWEEP', 'NO_SWEEP']:
        sub = ef[ef['sweep'] == ctx]
        if len(sub) > 0:
            print_metrics(calc_metrics(sub, f'  {ctx}'), '  ')

    # A6: BY VOLUME PROFILE
    print('\n  A6. BY VOLUME PROFILE (HVN/LVN)')
    print_table_header('  ')
    for ctx in ['LVN', 'MID', 'HVN']:
        sub = ef[ef['vp'] == ctx]
        if len(sub) > 0:
            print_metrics(calc_metrics(sub, f'  Near {ctx}'), '  ')

    # A7: BY ACTUAL DAY TYPE (the real day type, not what Edge Fade sees)
    print('\n  A7. BY ACTUAL DAY TYPE')
    print_table_header('  ')
    for dt in ['neutral', 'b_day', 'b_day_bear', 'trend_down', 'trend_up', 'p_day']:
        sub = ef[ef['actual_day_type'] == dt]
        if len(sub) > 0:
            print_metrics(calc_metrics(sub, f'  {dt}'), '  ')

    # A8: BY TRADE NUMBER
    print('\n  A8. BY TRADE NUMBER WITHIN SESSION')
    print_table_header('  ')
    for tn in sorted(ef['trade_num'].unique()):
        sub = ef[ef['trade_num'] == tn]
        if len(sub) >= 2:
            print_metrics(calc_metrics(sub, f'  Trade #{tn}'), '  ')

    # A9: BY EXIT REASON
    print('\n  A9. BY EXIT REASON')
    print_table_header('  ')
    for reason in ['TARGET', 'EOD', 'STOP']:
        sub = ef[ef['exit_reason'] == reason]
        if len(sub) > 0:
            print_metrics(calc_metrics(sub, f'  {reason}'), '  ')

    # A10: BEARISH DAY FILTER
    print('\n  A10. BEARISH DAY (ext_down > 0.3x) FILTER')
    print_table_header('  ')
    sub_bear = ef[ef['bearish_day'] == True]
    sub_nobear = ef[ef['bearish_day'] == False]
    if len(sub_bear) > 0:
        print_metrics(calc_metrics(sub_bear, '  ON BEARISH DAY (bad)'), '  ')
    if len(sub_nobear) > 0:
        print_metrics(calc_metrics(sub_nobear, '  NOT BEARISH (good)'), '  ')

    # ==================================================================
    #  SECTION B: EDGE FADE FILTER COMBINATIONS
    # ==================================================================
    print('\n')
    print('#' * 130)
    print('#  SECTION B: EDGE FADE - FILTER COMBINATIONS (ranked by expectancy)')
    print('#' * 130)

    filter_results = []

    # Define filter combinations
    combos = {
        'BASELINE (no filter)': pd.Series([True] * len(ef), index=ef.index),
        'CUT PM (before 13:30)': ef['time_obj'] < dt_time(13, 30),
        'CUT LATE PM (before 14:00)': ef['time_obj'] < dt_time(14, 0),
        'AM ONLY (before 12:30)': ef['time_obj'] < dt_time(12, 30),
        'TIGHT IB (<200)': ef['ib_range_val'] < 200,
        'TIGHT IB (<250)': ef['ib_range_val'] < 250,
        'AFTER SWEEP': ef['sweep'] == 'SWEEP',
        'NEAR LVN': ef['vp'] == 'LVN',
        'NOT BEARISH DAY': ef['bearish_day'] == False,
        'TRADE 1-2 ONLY': ef['trade_num'] <= 2,
        'TRADE 1-3 ONLY': ef['trade_num'] <= 3,
        # Doubles
        'SWEEP + IB<200': (ef['sweep'] == 'SWEEP') & (ef['ib_range_val'] < 200),
        'SWEEP + IB<250': (ef['sweep'] == 'SWEEP') & (ef['ib_range_val'] < 250),
        'AM + SWEEP': (ef['time_obj'] < dt_time(12, 30)) & (ef['sweep'] == 'SWEEP'),
        'AM + LVN': (ef['time_obj'] < dt_time(12, 30)) & (ef['vp'] == 'LVN'),
        'AM + IB<200': (ef['time_obj'] < dt_time(12, 30)) & (ef['ib_range_val'] < 200),
        'CUT PM + IB<200': (ef['time_obj'] < dt_time(13, 30)) & (ef['ib_range_val'] < 200),
        'CUT PM + IB<250': (ef['time_obj'] < dt_time(13, 30)) & (ef['ib_range_val'] < 250),
        'CUT PM + SWEEP': (ef['time_obj'] < dt_time(13, 30)) & (ef['sweep'] == 'SWEEP'),
        'CUT PM + NOT BEAR': (ef['time_obj'] < dt_time(13, 30)) & (ef['bearish_day'] == False),
        'NOT BEAR + IB<200': (ef['bearish_day'] == False) & (ef['ib_range_val'] < 200),
        'NOT BEAR + SWEEP': (ef['bearish_day'] == False) & (ef['sweep'] == 'SWEEP'),
        'LVN + IB<200': (ef['vp'] == 'LVN') & (ef['ib_range_val'] < 200),
        # Triples
        'SWEEP + IB<200 + CUT PM': (ef['sweep'] == 'SWEEP') & (ef['ib_range_val'] < 200) & (ef['time_obj'] < dt_time(13, 30)),
        'SWEEP + IB<200 + NOT BEAR': (ef['sweep'] == 'SWEEP') & (ef['ib_range_val'] < 200) & (ef['bearish_day'] == False),
        'CUT PM + IB<200 + NOT BEAR': (ef['time_obj'] < dt_time(13, 30)) & (ef['ib_range_val'] < 200) & (ef['bearish_day'] == False),
        'CUT PM + SWEEP + NOT BEAR': (ef['time_obj'] < dt_time(13, 30)) & (ef['sweep'] == 'SWEEP') & (ef['bearish_day'] == False),
        'AM + SWEEP + IB<200': (ef['time_obj'] < dt_time(12, 30)) & (ef['sweep'] == 'SWEEP') & (ef['ib_range_val'] < 200),
        'AM + LVN + IB<200': (ef['time_obj'] < dt_time(12, 30)) & (ef['vp'] == 'LVN') & (ef['ib_range_val'] < 200),
        'LVN + IB<200 + CUT PM': (ef['vp'] == 'LVN') & (ef['ib_range_val'] < 200) & (ef['time_obj'] < dt_time(13, 30)),
        # Quads
        'SWEEP+IB<200+CUT PM+!BEAR': (ef['sweep'] == 'SWEEP') & (ef['ib_range_val'] < 200) & (ef['time_obj'] < dt_time(13, 30)) & (ef['bearish_day'] == False),
    }

    for name, mask in combos.items():
        sub = ef[mask]
        if len(sub) < 3:
            continue
        m = calc_metrics(sub, name)
        if m:
            filter_results.append(m)

    # Sort by expectancy
    filter_results.sort(key=lambda x: x['expectancy'], reverse=True)

    print('\n')
    print_table_header('')
    for m in filter_results:
        print_metrics(m, '')

    # ==================================================================
    #  SECTION C: ALL STRATEGIES COMPARISON
    # ==================================================================
    print('\n')
    print('#' * 130)
    print('#  SECTION C: ALL STRATEGIES - CURRENT vs OPTIMIZED')
    print('#' * 130)

    # Playbook strategies
    print('\n  C1. PLAYBOOK STRATEGIES (unchanged)')
    print_table_header('  ')
    for strat in ['Trend Day Bull', 'P-Day', 'B-Day']:
        sub = playbook[playbook['strategy_name'] == strat]
        if len(sub) > 0:
            print_metrics(calc_metrics(sub, f'  {strat}'), '  ')

    # Research strategies (manual data)
    print('\n  C2. RESEARCH STRATEGIES (from diagnostics)')
    print(f'  {"Strategy":<35s} {"N":>4s} {"W/L":>7s} {"WR":>5s}  {"Exp/$":>8s}  {"Net $":>10s}  {"MaxDD":>9s}  {"PF":>6s}  {"MCL":>3s}')
    print(f'  {"-"*100}')
    print(f'  {"IBH Sweep+Fail SHORT (b_day)":<35s} {"4":>4s} {"4W/ 0L":>7s} {"100":>4s}%  {"$+146":>8s}  {"$+582":>10s}  {"$0":>9s}  {"INF":>6s}  {"0":>3s}')
    print(f'  {"Bear Accept SHORT (td+bdb)":<35s} {"11":>4s} {"7W/ 4L":>7s} {" 64":>4s}%  {"$+90":>8s}  {"$+995":>10s}  {"$-289":>9s}  {"3.32":>6s}  {"3":>3s}')

    # Edge Fade variants
    print('\n  C3. EDGE FADE VARIANTS')
    print_table_header('  ')

    # Current (all 90)
    m_curr = calc_metrics(ef, '  CURRENT (all 90)')
    print_metrics(m_curr, '  ')

    # Best single filters
    best_singles = [
        ('  + CUT PM (before 13:30)', ef['time_obj'] < dt_time(13, 30)),
        ('  + TIGHT IB (<200)', ef['ib_range_val'] < 200),
        ('  + NOT BEARISH DAY', ef['bearish_day'] == False),
        ('  + AFTER SWEEP', ef['sweep'] == 'SWEEP'),
        ('  + NEAR LVN', ef['vp'] == 'LVN'),
    ]
    for name, mask in best_singles:
        sub = ef[mask]
        print_metrics(calc_metrics(sub, name), '  ')

    # Best combos
    print(f'\n  {"--- TOP COMBINATIONS ---":}')
    print_table_header('  ')
    top_combos = [
        ('  SWEEP + IB<200', (ef['sweep'] == 'SWEEP') & (ef['ib_range_val'] < 200)),
        ('  CUT PM + IB<200', (ef['time_obj'] < dt_time(13, 30)) & (ef['ib_range_val'] < 200)),
        ('  CUT PM + NOT BEAR', (ef['time_obj'] < dt_time(13, 30)) & (ef['bearish_day'] == False)),
        ('  NOT BEAR + IB<200', (ef['bearish_day'] == False) & (ef['ib_range_val'] < 200)),
        ('  CUT PM + IB<200 + NOT BEAR', (ef['time_obj'] < dt_time(13, 30)) & (ef['ib_range_val'] < 200) & (ef['bearish_day'] == False)),
        ('  SWEEP+IB<200+CUT PM+!BEAR', (ef['sweep'] == 'SWEEP') & (ef['ib_range_val'] < 200) & (ef['time_obj'] < dt_time(13, 30)) & (ef['bearish_day'] == False)),
    ]
    for name, mask in top_combos:
        sub = ef[mask]
        if len(sub) >= 3:
            print_metrics(calc_metrics(sub, name), '  ')

    # ==================================================================
    #  SECTION D: COMBINED PORTFOLIOS
    # ==================================================================
    print('\n')
    print('#' * 130)
    print('#  SECTION D: COMBINED PORTFOLIOS')
    print('#' * 130)

    # Portfolio 1: Current (all 110 trades)
    print('\n  D1. PORTFOLIO COMPARISONS')
    print_table_header('  ')

    m_all = calc_metrics(df.copy(), '  CURRENT (110 trades)')
    print_metrics(m_all, '  ')

    # Ensure consistent datetime types
    playbook_c = playbook.copy()
    playbook_c['entry_time'] = pd.to_datetime(playbook_c['entry_time'])

    # Portfolio 2: Playbook + Edge Fade CUT PM + IB<200
    mask_opt1 = (ef['time_obj'] < dt_time(13, 30)) & (ef['ib_range_val'] < 200)
    ef_opt1 = ef[mask_opt1][list(playbook_c.columns)].copy()
    ef_opt1['entry_time'] = pd.to_datetime(ef_opt1['entry_time'])
    port2 = pd.concat([playbook_c, ef_opt1], ignore_index=True).sort_values('entry_time')
    m_port2 = calc_metrics(port2, '  OPTIMIZED 1: PB + EF(CutPM+IB<200)')
    print_metrics(m_port2, '  ')

    # Portfolio 3: Playbook + Edge Fade SWEEP + IB<200
    mask_opt2 = (ef['sweep'] == 'SWEEP') & (ef['ib_range_val'] < 200)
    ef_opt2 = ef[mask_opt2][list(playbook_c.columns)].copy()
    ef_opt2['entry_time'] = pd.to_datetime(ef_opt2['entry_time'])
    port3 = pd.concat([playbook_c, ef_opt2], ignore_index=True).sort_values('entry_time')
    m_port3 = calc_metrics(port3, '  OPTIMIZED 2: PB + EF(Sweep+IB<200)')
    print_metrics(m_port3, '  ')

    # Portfolio 4: Playbook + Edge Fade NOT BEAR + IB<200
    mask_opt3 = (ef['bearish_day'] == False) & (ef['ib_range_val'] < 200)
    ef_opt3 = ef[mask_opt3][list(playbook_c.columns)].copy()
    ef_opt3['entry_time'] = pd.to_datetime(ef_opt3['entry_time'])
    port4 = pd.concat([playbook_c, ef_opt3], ignore_index=True).sort_values('entry_time')
    m_port4 = calc_metrics(port4, '  OPTIMIZED 3: PB + EF(!Bear+IB<200)')
    print_metrics(m_port4, '  ')

    # Portfolio 5: Playbook + Edge Fade CUT PM + IB<200 + NOT BEAR
    mask_opt4 = (ef['time_obj'] < dt_time(13, 30)) & (ef['ib_range_val'] < 200) & (ef['bearish_day'] == False)
    ef_opt4 = ef[mask_opt4][list(playbook_c.columns)].copy()
    ef_opt4['entry_time'] = pd.to_datetime(ef_opt4['entry_time'])
    port5 = pd.concat([playbook_c, ef_opt4], ignore_index=True).sort_values('entry_time')
    m_port5 = calc_metrics(port5, '  OPTIMIZED 4: PB+EF(CutPM+IB<200+!Bear)')
    print_metrics(m_port5, '  ')

    # Portfolio 6: FULL (Playbook + Best EF + Research shorts)
    # Add Bear Acceptance Short manually
    bear_short_trades = pd.DataFrame([
        {'entry_time': '2025-11-20 11:03', 'net_pnl': 189.6, 'is_winner': 1},
        {'entry_time': '2025-12-12 11:01', 'net_pnl': 111.6, 'is_winner': 1},
        {'entry_time': '2025-12-31 10:57', 'net_pnl': -106.0, 'is_winner': 0},
        {'entry_time': '2026-01-02 10:59', 'net_pnl': -93.6, 'is_winner': 0},
        {'entry_time': '2026-01-15 10:51', 'net_pnl': -89.6, 'is_winner': 0},
        {'entry_time': '2026-01-20 13:39', 'net_pnl': 206.2, 'is_winner': 1},
        {'entry_time': '2026-01-28 10:44', 'net_pnl': 121.0, 'is_winner': 1},
        {'entry_time': '2026-01-30 11:22', 'net_pnl': -139.2, 'is_winner': 0},
        {'entry_time': '2026-02-03 12:15', 'net_pnl': 177.0, 'is_winner': 1},
        {'entry_time': '2026-02-04 11:05', 'net_pnl': 352.4, 'is_winner': 1},
        {'entry_time': '2026-02-12 10:39', 'net_pnl': 265.6, 'is_winner': 1},
    ])
    bear_short_trades['entry_time'] = pd.to_datetime(bear_short_trades['entry_time'])

    # IBH Sweep+Fail
    ibh_fade_trades = pd.DataFrame([
        {'entry_time': '2025-12-08 12:18', 'net_pnl': 112.0, 'is_winner': 1},
        {'entry_time': '2025-12-22 11:45', 'net_pnl': 88.0, 'is_winner': 1},
        {'entry_time': '2026-01-14 11:30', 'net_pnl': 140.0, 'is_winner': 1},
        {'entry_time': '2026-01-29 12:05', 'net_pnl': 242.0, 'is_winner': 1},
    ])
    ibh_fade_trades['entry_time'] = pd.to_datetime(ibh_fade_trades['entry_time'])

    # Combine: Playbook + Best EF + Bear Short + IBH Fade
    pb_slim = playbook_c[['entry_time', 'net_pnl', 'is_winner']].copy()
    pb_slim['entry_time'] = pd.to_datetime(pb_slim['entry_time'])
    ef_slim = ef_opt4[['entry_time', 'net_pnl', 'is_winner']].copy()
    ef_slim['entry_time'] = pd.to_datetime(ef_slim['entry_time'])
    full_port_pnl = pd.concat([
        pb_slim, ef_slim, bear_short_trades, ibh_fade_trades,
    ], ignore_index=True).sort_values('entry_time')

    m_full = calc_metrics(full_port_pnl, '  FULL OPTIMIZED PORTFOLIO')
    print_metrics(m_full, '  ')

    # ==================================================================
    #  SECTION E: SESSION COVERAGE
    # ==================================================================
    print('\n')
    print('#' * 130)
    print('#  SECTION E: SESSION COVERAGE BY DAY TYPE')
    print('#' * 130)

    print(f'\n  {"Day Type":<15s} {"Sessions":>8s} | {"Playbook":>10s} {"Edge Fade":>10s} {"EF Optimized":>12s} {"Bear Short":>10s} {"IBH Fade":>10s} | {"Total Opt":>10s}')
    print(f'  {"-"*105}')

    # Playbook dates
    pb_dates = set(playbook['entry_time'].apply(lambda x: pd.to_datetime(x).strftime('%Y-%m-%d')).unique())
    ef_dates = set(ef['date_key'].unique())
    ef_opt_dates = set(ef[mask_opt4]['date_key'].unique())
    bear_dates = set(['2025-11-20', '2025-12-12', '2025-12-31', '2026-01-02', '2026-01-15',
                      '2026-01-20', '2026-01-28', '2026-01-30', '2026-02-03', '2026-02-04', '2026-02-12'])
    ibh_dates = set(['2025-12-08', '2025-12-22', '2026-01-14', '2026-01-29'])

    for dt in ['trend_up', 'p_day', 'b_day', 'b_day_bear', 'neutral', 'trend_down']:
        dt_sessions = set(d for d, s in sessions.items() if s['day_type'] == dt)
        n_total = len(dt_sessions)
        n_pb = len(dt_sessions & pb_dates)
        n_ef = len(dt_sessions & ef_dates)
        n_ef_opt = len(dt_sessions & ef_opt_dates)
        n_bear = len(dt_sessions & bear_dates)
        n_ibh = len(dt_sessions & ibh_dates)
        n_any = len(dt_sessions & (pb_dates | ef_opt_dates | bear_dates | ibh_dates))

        print(f'  {dt:<15s} {n_total:>8d} | {n_pb:>10d} {n_ef:>10d} {n_ef_opt:>12d} {n_bear:>10d} {n_ibh:>10d} | {n_any:>10d}')

    total_sessions = len(sessions)
    all_opt_dates = pb_dates | ef_opt_dates | bear_dates | ibh_dates
    print(f'  {"TOTAL":<15s} {total_sessions:>8d} | {len(pb_dates):>10d} {len(ef_dates):>10d} {len(ef_opt_dates):>12d} {len(bear_dates):>10d} {len(ibh_dates):>10d} | {len(all_opt_dates):>10d}')
    print(f'\n  Coverage: {len(all_opt_dates)}/{total_sessions} = {len(all_opt_dates)/total_sessions*100:.0f}% of sessions have at least one trade')

    # ==================================================================
    #  SECTION F: TRADE-BY-TRADE EDGE FADE (with filter flags)
    # ==================================================================
    print('\n')
    print('#' * 130)
    print('#  SECTION F: EDGE FADE TRADE-BY-TRADE (90 trades with filter flags)')
    print('#' * 130)

    print(f'\n  {"Date":<12s} {"Time":>5s} {"#":>2s} {"DayType":>11s} {"IB":>5s} {"Sweep":>6s} {"VP":>4s} {"Bear":>5s} {"Exit":>6s} '
          f'{"Net $":>8s} {"Bars":>4s} {"KEEP?":>5s}')
    print(f'  {"-"*100}')

    for _, row in ef.sort_values('entry_time').iterrows():
        keep = 'YES' if (row['time_obj'] < dt_time(13, 30) and
                         row['ib_range_val'] is not None and row['ib_range_val'] < 200 and
                         not row['bearish_day']) else 'no'

        bear_flag = 'BEAR' if row['bearish_day'] else ''
        print(f'  {row["date_key"]:<12s} {str(row["time_obj"])[:5]:>5s} {row["trade_num"]:>2d} '
              f'{row["actual_day_type"]:>11s} {row["ib_range_val"]:>5.0f} {row["sweep"]:>6s} {row["vp"]:>4s} '
              f'{bear_flag:>5s} {row["exit_reason"]:>6s} ${row["net_pnl"]:>+7,.0f} {row["bars_held"]:>4d} '
              f'{keep:>5s}')

    # Count kept vs removed
    kept = ef[(ef['time_obj'] < dt_time(13, 30)) & (ef['ib_range_val'] < 200) & (ef['bearish_day'] == False)]
    removed = ef[~((ef['time_obj'] < dt_time(13, 30)) & (ef['ib_range_val'] < 200) & (ef['bearish_day'] == False))]
    print(f'\n  KEPT: {len(kept)} trades, ${kept["net_pnl"].sum():+,.0f} net')
    print(f'  REMOVED: {len(removed)} trades, ${removed["net_pnl"].sum():+,.0f} net')
    print(f'  Removed trades are COSTING you ${abs(removed[removed["net_pnl"] < 0]["net_pnl"].sum()):,.0f} in losses')


if __name__ == '__main__':
    main()
