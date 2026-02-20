"""
Opening Range Strategy v2 - Deep Filter Analysis
=================================================

Takes the 97 raw trades from v1 and enriches each with:
1. HVN/LVN rotation context (is price near HVN = chop, or LVN = fast move?)
2. Opening drive vs fade classification (first 5-min bar direction)
3. Pre-market liquidity already taken (did overnight already sweep PDH/PDL?)
4. Day type at entry time
5. IB range at entry
6. Pre-market range compression (tight overnight = more explosive open)
7. Gap direction (gap up/down vs prior close)
8. Multi-bar momentum (first 5 bars direction)
9. SMT with YM as third instrument
10. OR range size (narrow vs wide)

Then tests filter combinations to find the optimized subset.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path
import sys

proj_root = Path(__file__).resolve().parent
sys.path.insert(0, str(proj_root))

from data.loader import load_csv
from data.session import filter_rth

# Import the v1 functions we need
from diagnostic_opening_range_smt import (
    compute_session_levels,
    detect_smt_divergence,
    detect_fvg_5min,
    detect_fvg_15min,
    detect_opening_reversal,
    calc_metrics,
    print_metrics,
)


def compute_volume_profile_simple(bars, n_bins=50):
    """
    Simple volume profile computation without rockit dependency.
    Returns POC, VAH, VAL, HVN nodes, LVN nodes.
    """
    if len(bars) < 10:
        return {'poc': None, 'vah': None, 'val': None, 'hvn': [], 'lvn': []}

    price_range = bars['high'].max() - bars['low'].min()
    if price_range < 1:
        return {'poc': None, 'vah': None, 'val': None, 'hvn': [], 'lvn': []}

    bin_size = price_range / n_bins
    low_base = bars['low'].min()

    # Build volume at price
    vol_at_price = np.zeros(n_bins)
    for _, bar in bars.iterrows():
        bar_low_bin = max(0, int((bar['low'] - low_base) / bin_size))
        bar_high_bin = min(n_bins - 1, int((bar['high'] - low_base) / bin_size))
        bar_vol = bar['volume'] / max(1, bar_high_bin - bar_low_bin + 1)
        for b in range(bar_low_bin, bar_high_bin + 1):
            vol_at_price[b] += bar_vol

    # POC = highest volume bin
    poc_bin = np.argmax(vol_at_price)
    poc = low_base + (poc_bin + 0.5) * bin_size

    # Value area (70% of volume)
    total_vol = vol_at_price.sum()
    target_vol = total_vol * 0.70
    accumulated = vol_at_price[poc_bin]
    low_idx = poc_bin
    high_idx = poc_bin

    while accumulated < target_vol:
        expand_low = vol_at_price[low_idx - 1] if low_idx > 0 else 0
        expand_high = vol_at_price[high_idx + 1] if high_idx < n_bins - 1 else 0

        if expand_low >= expand_high and low_idx > 0:
            low_idx -= 1
            accumulated += vol_at_price[low_idx]
        elif high_idx < n_bins - 1:
            high_idx += 1
            accumulated += vol_at_price[high_idx]
        else:
            break

    val = low_base + low_idx * bin_size
    vah = low_base + (high_idx + 1) * bin_size

    # HVN/LVN detection (peaks and valleys in volume profile)
    avg_vol = vol_at_price.mean()
    hvn = []
    lvn = []
    for b in range(1, n_bins - 1):
        price = low_base + (b + 0.5) * bin_size
        if vol_at_price[b] > avg_vol * 1.5:
            hvn.append(price)
        elif vol_at_price[b] < avg_vol * 0.5 and vol_at_price[b] > 0:
            lvn.append(price)

    return {'poc': poc, 'vah': vah, 'val': val, 'hvn': hvn, 'lvn': lvn}


def classify_opening_drive(rth_bars):
    """
    Classify the opening as:
    - DRIVE_UP: Strong directional move up in first 5 bars (close[4] > open[0] + threshold)
    - DRIVE_DOWN: Strong directional move down
    - ROTATION: Choppy, no clear direction
    """
    if len(rth_bars) < 5:
        return 'UNKNOWN'

    first_5 = rth_bars.iloc[:5]
    open_price = first_5.iloc[0]['open']
    close_5th = first_5.iloc[4]['close']
    range_5 = first_5['high'].max() - first_5['low'].min()

    # Net move as fraction of range
    net_move = close_5th - open_price
    if range_5 < 1:
        return 'ROTATION'

    move_pct = net_move / range_5

    if move_pct > 0.4:
        return 'DRIVE_UP'
    elif move_pct < -0.4:
        return 'DRIVE_DOWN'
    else:
        return 'ROTATION'


def check_premarket_liquidity_taken(levels):
    """
    Check if overnight session already took key liquidity (PDH/PDL).
    If overnight high > PDH, sell-side liquidity above PDH was already taken.
    If overnight low < PDL, buy-side liquidity below PDL was already taken.
    """
    on_h = levels.get('overnight_high', np.nan)
    on_l = levels.get('overnight_low', np.nan)
    pdh = levels.get('pdh', np.nan)
    pdl = levels.get('pdl', np.nan)

    result = {
        'pdh_taken_overnight': False,
        'pdl_taken_overnight': False,
        'both_taken': False,
    }

    if not np.isnan(on_h) and not np.isnan(pdh):
        if on_h >= pdh - 5:  # Within 5 pts counts
            result['pdh_taken_overnight'] = True

    if not np.isnan(on_l) and not np.isnan(pdl):
        if on_l <= pdl + 5:
            result['pdl_taken_overnight'] = True

    result['both_taken'] = result['pdh_taken_overnight'] and result['pdl_taken_overnight']

    return result


def check_near_hvn_lvn(price, vp_data, threshold_pts=20):
    """
    Check if entry price is near an HVN or LVN.
    Near HVN = choppy, price accepted (bad for reversal)
    Near LVN = price rejected, fast move expected (good for reversal)
    """
    if vp_data is None or vp_data.get('poc') is None:
        return 'UNKNOWN', None

    # Check LVN first (more bullish for reversal trades)
    for lvn_price in vp_data.get('lvn', []):
        if abs(price - lvn_price) < threshold_pts:
            return 'NEAR_LVN', lvn_price

    # Check HVN
    for hvn_price in vp_data.get('hvn', []):
        if abs(price - hvn_price) < threshold_pts:
            return 'NEAR_HVN', hvn_price

    # Check POC
    if abs(price - vp_data['poc']) < threshold_pts:
        return 'NEAR_POC', vp_data['poc']

    return 'NONE', None


def compute_gap(open_price, pdc):
    """Classify the gap at open."""
    if np.isnan(pdc):
        return 'NONE', 0

    gap = open_price - pdc
    if gap > 30:
        return 'GAP_UP_LARGE', gap
    elif gap > 10:
        return 'GAP_UP_SMALL', gap
    elif gap < -30:
        return 'GAP_DOWN_LARGE', gap
    elif gap < -10:
        return 'GAP_DOWN_SMALL', gap
    else:
        return 'FLAT', gap


def detect_smt_3way(nq_bars, es_bars, ym_bars):
    """
    Detect SMT divergence across all 3 instruments.
    Stronger signal when 2 of 3 diverge vs the third.
    """
    result = {'nq_es': False, 'nq_ym': False, 'es_ym': False, 'triple': False}

    nq_es_divs = detect_smt_divergence(nq_bars, es_bars)
    result['nq_es'] = len(nq_es_divs) > 0
    result['nq_es_type'] = nq_es_divs[0][1] if nq_es_divs else None

    # NQ vs YM
    nq_ym_divs = detect_smt_divergence(nq_bars, ym_bars)
    result['nq_ym'] = len(nq_ym_divs) > 0

    # ES vs YM
    es_ym_divs = detect_smt_divergence(es_bars, ym_bars)
    result['es_ym'] = len(es_ym_divs) > 0

    # Triple = at least 2 pairs show divergence
    count = sum([result['nq_es'], result['nq_ym'], result['es_ym']])
    result['triple'] = count >= 2

    return result


def enrich_and_simulate(nq, es, ym, sessions):
    """
    Run the full simulation with enrichment for every trade.
    """
    MNQ_MULT = 2.0
    trades = []

    for sd in sessions:
        sd_ts = pd.Timestamp(sd)

        levels = compute_session_levels(nq, sd)
        if np.isnan(levels.get('overnight_high', np.nan)):
            continue

        # Get bars: 9:00-11:00 for detection
        mask_nq = (
            (nq['timestamp'] >= sd_ts + timedelta(hours=9)) &
            (nq['timestamp'] <= sd_ts + timedelta(hours=11))
        )
        nq_entry = nq[mask_nq].copy().reset_index(drop=True)

        mask_es = (
            (es['timestamp'] >= sd_ts + timedelta(hours=9)) &
            (es['timestamp'] <= sd_ts + timedelta(hours=11))
        )
        es_entry = es[mask_es].copy().reset_index(drop=True)

        mask_ym = (
            (ym['timestamp'] >= sd_ts + timedelta(hours=9)) &
            (ym['timestamp'] <= sd_ts + timedelta(hours=11))
        )
        ym_entry = ym[mask_ym].copy().reset_index(drop=True)

        if len(nq_entry) < 15 or len(es_entry) < 15:
            continue

        # RTH bars for P&L
        rth_mask = (
            (nq['timestamp'] >= sd_ts + timedelta(hours=9, minutes=30)) &
            (nq['timestamp'] <= sd_ts + timedelta(hours=16))
        )
        nq_rth = nq[rth_mask].copy().reset_index(drop=True)

        if len(nq_rth) < 30:
            continue

        # Detect setups
        setups = detect_opening_reversal(nq_entry, levels)
        if not setups:
            continue

        # Pre-compute enrichment data for this session
        # Volume profile from overnight + first 30 min
        vp_mask = (
            (nq['timestamp'] >= sd_ts + timedelta(hours=9, minutes=30)) &
            (nq['timestamp'] <= sd_ts + timedelta(hours=10))
        )
        vp_bars = nq[vp_mask]
        # Use prior day RTH for volume profile
        prev_day = sd_ts - timedelta(days=1)
        if prev_day.weekday() == 5: prev_day -= timedelta(days=1)
        elif prev_day.weekday() == 6: prev_day -= timedelta(days=2)
        prev_rth_mask = (
            (nq['timestamp'] >= pd.Timestamp(prev_day) + timedelta(hours=9, minutes=30)) &
            (nq['timestamp'] <= pd.Timestamp(prev_day) + timedelta(hours=16))
        )
        prev_rth = nq[prev_rth_mask]
        vp_data = compute_volume_profile_simple(prev_rth) if len(prev_rth) > 30 else None

        # Opening drive classification
        opening_drive = classify_opening_drive(nq_rth)

        # Pre-market liquidity
        pm_liq = check_premarket_liquidity_taken(levels)

        # Gap
        open_price = nq_rth.iloc[0]['open']
        gap_type, gap_pts = compute_gap(open_price, levels.get('pdc', np.nan))

        # OR range
        or_range = levels.get('or_range', np.nan)

        # Overnight range compression
        on_range = levels['overnight_high'] - levels['overnight_low']

        # IB range
        ib_range = levels.get('ib_range', np.nan)

        # SMT 3-way
        smt_3way = detect_smt_3way(nq_entry, es_entry, ym_entry)

        # Day type (compute from extension at time of OR)
        # Simple: if OR high > overnight high = trending up, etc.
        or_high = levels.get('or_high', np.nan)
        or_low = levels.get('or_low', np.nan)

        # Process setups
        seen_models = set()
        for setup in setups:
            model = setup['model']
            if model in seen_models:
                continue
            seen_models.add(model)

            entry_price = setup['entry_price']
            entry_time = setup['entry_time']
            stop_price = setup['stop']

            # Risk/targets
            if setup['direction'] == 'LONG':
                risk = entry_price - stop_price
                target_2r = entry_price + 2 * risk
                target_3r = entry_price + 3 * risk
            else:
                risk = stop_price - entry_price
                target_2r = entry_price - 2 * risk
                target_3r = entry_price - 3 * risk

            if risk <= 0 or risk > 200:
                continue

            # Enrichment
            vp_zone, vp_level = check_near_hvn_lvn(entry_price, vp_data)

            # Is this a fade or a drive trade?
            # Fade: direction OPPOSES the opening drive
            # Drive continuation: direction ALIGNS with opening drive
            trade_type = 'UNKNOWN'
            if opening_drive == 'DRIVE_UP':
                trade_type = 'FADE' if setup['direction'] == 'SHORT' else 'CONTINUATION'
            elif opening_drive == 'DRIVE_DOWN':
                trade_type = 'FADE' if setup['direction'] == 'LONG' else 'CONTINUATION'
            else:
                trade_type = 'ROTATION'

            # Pre-market liquidity relevance
            # If we're going LONG and PDL was already taken overnight = bullish (stops cleared)
            # If we're going SHORT and PDH was already taken overnight = bearish
            pm_liq_aligned = False
            if setup['direction'] == 'LONG' and pm_liq['pdl_taken_overnight']:
                pm_liq_aligned = True
            elif setup['direction'] == 'SHORT' and pm_liq['pdh_taken_overnight']:
                pm_liq_aligned = True

            # Gap alignment
            gap_aligned = False
            if setup['direction'] == 'LONG' and gap_pts < -10:  # Gap down -> sweep low -> LONG
                gap_aligned = True
            elif setup['direction'] == 'SHORT' and gap_pts > 10:  # Gap up -> sweep high -> SHORT
                gap_aligned = True

            # FVG check
            entry_idx = setup['entry_idx']
            post_mask = nq_entry.index >= entry_idx
            post_bars = nq_entry[post_mask]
            fvg_dir = 'bull' if setup['direction'] == 'LONG' else 'bear'
            has_fvg_5m = len(detect_fvg_5min(post_bars, fvg_dir)) > 0
            has_fvg_15m = len(detect_fvg_15min(post_bars, fvg_dir)) > 0

            # Delta
            delta_confirm = False
            if entry_idx < len(nq_entry):
                dv = nq_entry.iloc[entry_idx]['vol_delta']
                if setup['direction'] == 'LONG' and dv > 0:
                    delta_confirm = True
                elif setup['direction'] == 'SHORT' and dv < 0:
                    delta_confirm = True

            # VWAP
            vwap_aligned = False
            if 'vwap' in nq_entry.columns and entry_idx < len(nq_entry):
                vw = nq_entry.iloc[entry_idx]['vwap']
                if not np.isnan(vw):
                    if setup['direction'] == 'LONG' and entry_price <= vw + 20:
                        vwap_aligned = True
                    elif setup['direction'] == 'SHORT' and entry_price >= vw - 20:
                        vwap_aligned = True

            # Simulate P&L
            post_entry = nq_rth[nq_rth['timestamp'] > entry_time]
            exit_price = None
            exit_reason = None
            exit_time = None
            mfe = 0
            mae = 0

            for _, bar in post_entry.iterrows():
                if setup['direction'] == 'LONG':
                    mfe = max(mfe, bar['high'] - entry_price)
                    mae = min(mae, bar['low'] - entry_price)
                    if bar['low'] <= stop_price:
                        exit_price, exit_reason, exit_time = stop_price, 'STOP', bar['timestamp']
                        break
                    if bar['high'] >= target_2r:
                        exit_price, exit_reason, exit_time = target_2r, 'TARGET_2R', bar['timestamp']
                        break
                else:
                    mfe = max(mfe, entry_price - bar['low'])
                    mae = min(mae, entry_price - bar['high'])
                    if bar['high'] >= stop_price:
                        exit_price, exit_reason, exit_time = stop_price, 'STOP', bar['timestamp']
                        break
                    if bar['low'] <= target_2r:
                        exit_price, exit_reason, exit_time = target_2r, 'TARGET_2R', bar['timestamp']
                        break

            if exit_price is None and len(post_entry) > 0:
                last = post_entry.iloc[-1]
                exit_price, exit_reason, exit_time = last['close'], 'EOD', last['timestamp']

            if exit_price is None:
                continue

            pnl_pts = (exit_price - entry_price) if setup['direction'] == 'LONG' else (entry_price - exit_price)
            pnl_dollar = pnl_pts * MNQ_MULT
            r_mult = pnl_pts / risk if risk > 0 else 0

            trades.append({
                'session_date': sd,
                'entry_time': entry_time,
                'exit_time': exit_time,
                'model': model,
                'direction': setup['direction'],
                'level_swept': setup.get('sweep_level', ''),
                'entry_price': round(entry_price, 2),
                'stop_price': round(stop_price, 2),
                'exit_price': round(exit_price, 2),
                'risk_pts': round(risk, 1),
                'pnl_pts': round(pnl_pts, 1),
                'pnl_dollar': round(pnl_dollar, 2),
                'r_multiple': round(r_mult, 2),
                'exit_reason': exit_reason,
                'mfe_pts': round(mfe, 1),
                'mae_pts': round(mae, 1),
                # Enrichment columns
                'vp_zone': vp_zone,
                'trade_type': trade_type,        # FADE / CONTINUATION / ROTATION
                'opening_drive': opening_drive,
                'pm_liq_aligned': pm_liq_aligned,
                'pdh_taken': pm_liq['pdh_taken_overnight'],
                'pdl_taken': pm_liq['pdl_taken_overnight'],
                'gap_type': gap_type,
                'gap_pts': round(gap_pts, 1),
                'gap_aligned': gap_aligned,
                'or_range': round(or_range, 1) if not np.isnan(or_range) else 0,
                'ib_range': round(ib_range, 1) if not np.isnan(ib_range) else 0,
                'on_range': round(on_range, 1),
                'has_smt_nq_es': smt_3way['nq_es'],
                'has_smt_triple': smt_3way['triple'],
                'has_fvg_5m': has_fvg_5m,
                'has_fvg_15m': has_fvg_15m,
                'vwap_aligned': vwap_aligned,
                'delta_confirm': delta_confirm,
                'on_compressed': on_range < 180,  # Below median
            })

    return pd.DataFrame(trades)


def main():
    print('=' * 70)
    print('  LOADING DATA')
    print('=' * 70)
    nq = load_csv('NQ')
    es = load_csv('ES')
    ym = load_csv('YM')

    rth = filter_rth(nq)
    sessions = sorted(rth['session_date'].dt.date.unique())
    print(f'\nSessions: {len(sessions)}')

    # ================================================================
    print('\n' + '=' * 70)
    print('  ENRICHED SIMULATION')
    print('=' * 70)

    trades = enrich_and_simulate(nq, es, ym, sessions)
    if len(trades) == 0:
        print('No trades!')
        return

    n = len(trades)
    print(f'\n  Total trades: {n}')
    print(f'  Sessions with trades: {trades["session_date"].nunique()} / {len(sessions)}')

    # Baseline
    print('\n' + '-' * 50)
    print('  BASELINE (all trades)')
    print('-' * 50)
    print_metrics(calc_metrics(trades, 'ALL'), '  ')

    # ================================================================
    print('\n' + '=' * 70)
    print('  DIMENSION ANALYSIS: WINNERS vs LOSERS')
    print('=' * 70)

    wins = trades[trades['pnl_dollar'] > 0]
    losses = trades[trades['pnl_dollar'] <= 0]

    # --- By Model ---
    print('\n  --- BY MODEL ---')
    for model in trades['model'].unique():
        sub = trades[trades['model'] == model]
        m = calc_metrics(sub, f'  {model}')
        if m:
            print_metrics(m, '  ')

    # --- Only OR_REVERSAL (drop VWAP_RECLAIM) ---
    print('\n  --- OR_REVERSAL ONLY (drop VWAP_RECLAIM) ---')
    or_only = trades[trades['model'].str.startswith('OR_REVERSAL')]
    print_metrics(calc_metrics(or_only, 'OR_REVERSAL only'), '  ')

    # --- By Direction ---
    print('\n  --- BY DIRECTION ---')
    for d in ['LONG', 'SHORT']:
        sub = trades[trades['direction'] == d]
        print_metrics(calc_metrics(sub, f'{d}'), '  ')

    # --- By Trade Type (FADE vs CONTINUATION vs ROTATION) ---
    print('\n  --- BY TRADE TYPE (vs opening drive) ---')
    for tt in trades['trade_type'].unique():
        sub = trades[trades['trade_type'] == tt]
        m = calc_metrics(sub, f'{tt}')
        if m: print_metrics(m, '  ')

    # --- By VP Zone ---
    print('\n  --- BY VOLUME PROFILE ZONE ---')
    for zone in trades['vp_zone'].unique():
        sub = trades[trades['vp_zone'] == zone]
        m = calc_metrics(sub, f'{zone}')
        if m: print_metrics(m, '  ')

    # --- By Pre-Market Liquidity ---
    print('\n  --- BY PRE-MARKET LIQUIDITY ---')
    for aligned in [True, False]:
        sub = trades[trades['pm_liq_aligned'] == aligned]
        label = 'PM Liq ALIGNED' if aligned else 'PM Liq NOT aligned'
        m = calc_metrics(sub, label)
        if m: print_metrics(m, '  ')

    # --- By Gap ---
    print('\n  --- BY GAP ALIGNMENT ---')
    for aligned in [True, False]:
        sub = trades[trades['gap_aligned'] == aligned]
        label = 'Gap ALIGNED' if aligned else 'Gap NOT aligned'
        m = calc_metrics(sub, label)
        if m: print_metrics(m, '  ')

    # --- By Opening Drive ---
    print('\n  --- BY OPENING DRIVE ---')
    for drive in trades['opening_drive'].unique():
        sub = trades[trades['opening_drive'] == drive]
        m = calc_metrics(sub, f'{drive}')
        if m: print_metrics(m, '  ')

    # --- By OR Range Size ---
    print('\n  --- BY OR RANGE SIZE ---')
    med_or = trades['or_range'].median()
    for label, mask in [('Narrow OR (< median)', trades['or_range'] < med_or),
                        ('Wide OR (>= median)', trades['or_range'] >= med_or)]:
        sub = trades[mask]
        m = calc_metrics(sub, label)
        if m: print_metrics(m, '  ')

    # --- By Overnight Range (compressed vs wide) ---
    print('\n  --- BY OVERNIGHT RANGE ---')
    for label, mask in [('ON Compressed (< 180)', trades['on_compressed'] == True),
                        ('ON Wide (>= 180)', trades['on_compressed'] == False)]:
        sub = trades[mask]
        m = calc_metrics(sub, label)
        if m: print_metrics(m, '  ')

    # --- By IB Range ---
    print('\n  --- BY IB RANGE ---')
    med_ib = trades['ib_range'].median()
    for label, mask in [('IB < 200', trades['ib_range'] < 200),
                        ('IB >= 200', trades['ib_range'] >= 200)]:
        sub = trades[mask]
        m = calc_metrics(sub, label)
        if m: print_metrics(m, '  ')

    # --- By SMT ---
    print('\n  --- BY SMT (NQ vs ES) ---')
    for smt in [True, False]:
        sub = trades[trades['has_smt_nq_es'] == smt]
        label = 'SMT present' if smt else 'No SMT'
        m = calc_metrics(sub, label)
        if m: print_metrics(m, '  ')

    print('\n  --- BY TRIPLE SMT (2+ pairs diverge) ---')
    for smt in [True, False]:
        sub = trades[trades['has_smt_triple'] == smt]
        label = 'Triple SMT' if smt else 'No triple SMT'
        m = calc_metrics(sub, label)
        if m: print_metrics(m, '  ')

    # --- By FVG ---
    print('\n  --- BY FVG ---')
    for fvg in [True, False]:
        sub = trades[(trades['has_fvg_5m'] == fvg) | (trades['has_fvg_15m'] == fvg)]
        label = 'Has FVG' if fvg else 'No FVG'
        m = calc_metrics(sub, label)
        if m: print_metrics(m, '  ')

    # --- Level quality ---
    print('\n  --- BY LEVEL SWEPT ---')
    for level in trades['level_swept'].unique():
        sub = trades[trades['level_swept'] == level]
        m = calc_metrics(sub, f'{level}')
        if m and m['trades'] >= 3:
            print_metrics(m, '  ')

    # ================================================================
    print('\n' + '=' * 70)
    print('  FILTER COMBINATIONS (ranked by expectancy)')
    print('=' * 70)

    combos = []

    def test_filter(label, mask):
        sub = trades[mask]
        m = calc_metrics(sub, label)
        if m and m['trades'] >= 5:
            combos.append(m)

    # Single filters
    test_filter('OR_REVERSAL only', trades['model'].str.startswith('OR_REVERSAL'))
    test_filter('FADE only', trades['trade_type'] == 'FADE')
    test_filter('CONTINUATION only', trades['trade_type'] == 'CONTINUATION')
    test_filter('Near LVN', trades['vp_zone'] == 'NEAR_LVN')
    test_filter('Gap aligned', trades['gap_aligned'] == True)
    test_filter('PM Liq aligned', trades['pm_liq_aligned'] == True)
    test_filter('ON compressed', trades['on_compressed'] == True)
    test_filter('Has FVG 5m', trades['has_fvg_5m'] == True)
    test_filter('VWAP aligned', trades['vwap_aligned'] == True)
    test_filter('Triple SMT', trades['has_smt_triple'] == True)
    test_filter('OR Reversal SHORT', (trades['model'] == 'OR_REVERSAL_SHORT'))
    test_filter('OR Reversal LONG', (trades['model'] == 'OR_REVERSAL_LONG'))
    test_filter('IB < 200', trades['ib_range'] < 200)
    test_filter('Overnight H/L sweep', trades['level_swept'].isin(['overnight_high', 'overnight_low']))

    # Combined filters
    test_filter('OR_REV + FVG',
        trades['model'].str.startswith('OR_REVERSAL') & (trades['has_fvg_5m'] | trades['has_fvg_15m']))
    test_filter('OR_REV + VWAP',
        trades['model'].str.startswith('OR_REVERSAL') & trades['vwap_aligned'])
    test_filter('OR_REV + Gap aligned',
        trades['model'].str.startswith('OR_REVERSAL') & trades['gap_aligned'])
    test_filter('OR_REV + PM Liq aligned',
        trades['model'].str.startswith('OR_REVERSAL') & trades['pm_liq_aligned'])
    test_filter('FADE + FVG',
        (trades['trade_type'] == 'FADE') & (trades['has_fvg_5m'] | trades['has_fvg_15m']))
    test_filter('FADE + VWAP',
        (trades['trade_type'] == 'FADE') & trades['vwap_aligned'])
    test_filter('FADE + Gap',
        (trades['trade_type'] == 'FADE') & trades['gap_aligned'])
    test_filter('Overnight sweep + VWAP',
        trades['level_swept'].isin(['overnight_high', 'overnight_low']) & trades['vwap_aligned'])
    test_filter('Overnight sweep + FVG',
        trades['level_swept'].isin(['overnight_high', 'overnight_low']) & (trades['has_fvg_5m'] | trades['has_fvg_15m']))
    test_filter('OR_REV + ON_compressed',
        trades['model'].str.startswith('OR_REVERSAL') & trades['on_compressed'])
    test_filter('OR_REV + IB<200',
        trades['model'].str.startswith('OR_REVERSAL') & (trades['ib_range'] < 200))

    # Triple combos
    test_filter('OR_REV + FVG + VWAP',
        trades['model'].str.startswith('OR_REVERSAL') & (trades['has_fvg_5m'] | trades['has_fvg_15m']) & trades['vwap_aligned'])
    test_filter('OR_REV + FVG + Gap',
        trades['model'].str.startswith('OR_REVERSAL') & (trades['has_fvg_5m'] | trades['has_fvg_15m']) & trades['gap_aligned'])
    test_filter('FADE + FVG + VWAP',
        (trades['trade_type'] == 'FADE') & (trades['has_fvg_5m'] | trades['has_fvg_15m']) & trades['vwap_aligned'])
    test_filter('OR_REV + FVG + PM Liq',
        trades['model'].str.startswith('OR_REVERSAL') & (trades['has_fvg_5m'] | trades['has_fvg_15m']) & trades['pm_liq_aligned'])
    test_filter('ON sweep + FVG + VWAP',
        trades['level_swept'].isin(['overnight_high', 'overnight_low']) & (trades['has_fvg_5m'] | trades['has_fvg_15m']) & trades['vwap_aligned'])
    test_filter('FADE + ON_compressed + FVG',
        (trades['trade_type'] == 'FADE') & trades['on_compressed'] & (trades['has_fvg_5m'] | trades['has_fvg_15m']))
    test_filter('OR_REV SHORT + FVG + VWAP',
        (trades['model'] == 'OR_REVERSAL_SHORT') & (trades['has_fvg_5m'] | trades['has_fvg_15m']) & trades['vwap_aligned'])

    # Sort by expectancy
    combos.sort(key=lambda x: x['expectancy'], reverse=True)

    print(f'\n  {"Filter":<40s} {"Trd":>4s} {"WR":>5s} {"Exp":>7s} {"Net":>10s} {"MaxDD":>8s} {"PF":>6s} {"MCL":>4s}')
    print(f'  {"-"*40} {"---":>4s} {"---":>5s} {"---":>7s} {"---":>10s} {"---":>8s} {"---":>6s} {"---":>4s}')
    for m in combos:
        pf_str = f'{m["pf"]:.1f}' if m['pf'] < 999 else 'INF'
        print(f'  {m["label"]:<40s} {m["trades"]:4d} {m["wr"]:4.0f}% ${m["expectancy"]:+6.0f} ${m["net"]:+9,.0f}  ${m["max_dd"]:+7,.0f} {pf_str:>6s} {m["max_cl"]:4d}')

    # ================================================================
    print('\n' + '=' * 70)
    print('  TRADE LOG: BEST MODEL')
    print('=' * 70)

    # Show the best combo trades
    if combos:
        best_label = combos[0]['label']
        print(f'\n  Best model: {best_label}')
        print(f'  Showing all trades:\n')

        # Reconstruct the best filter
        # For now, show OR_REVERSAL trades with enrichment
        best_trades = trades[trades['model'].str.startswith('OR_REVERSAL')].sort_values('entry_time')

        for i, (_, t) in enumerate(best_trades.iterrows()):
            win = 'W' if t['pnl_dollar'] > 0 else 'L'
            et = pd.Timestamp(t['entry_time'])
            fvg = 'FVG' if t['has_fvg_5m'] or t['has_fvg_15m'] else '   '
            vw = 'VW' if t['vwap_aligned'] else '  '
            gp = 'GP' if t['gap_aligned'] else '  '
            pm = 'PM' if t['pm_liq_aligned'] else '  '
            print(f'  {i+1:3d}. {str(t["session_date"])[:10]} {et.strftime("%H:%M")} '
                  f'{t["direction"]:5s} {t["level_swept"]:<18s} '
                  f'${t["pnl_dollar"]:+8.0f} ({t["r_multiple"]:+.1f}R) '
                  f'{t["exit_reason"]:<10s} '
                  f'{t["trade_type"]:<12s} {t["vp_zone"]:<10s} '
                  f'[{fvg}|{vw}|{gp}|{pm}] {win}')

    # ================================================================
    print('\n' + '=' * 70)
    print('  COMPARISON TO EXISTING PORTFOLIO')
    print('=' * 70)

    print(f'\n  Existing Optimized: 52 trades, 83% WR, $264/trade, $13,706 net, -$351 MaxDD')
    if combos:
        best = combos[0]
        print(f'  Best OR model:      {best["trades"]} trades, {best["wr"]:.0f}% WR, ${best["expectancy"]:+.0f}/trade, ${best["net"]:+,.0f} net, ${best["max_dd"]:+,.0f} MaxDD')
        print(f'\n  Combined (additive): ${13706 + best["net"]:+,.0f} estimated net')

    print('\n  DONE')


if __name__ == '__main__':
    main()
