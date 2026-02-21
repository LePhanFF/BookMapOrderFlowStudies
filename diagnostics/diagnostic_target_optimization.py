"""
Edge Fade Stop/Target Optimization Research

Analyzes MFE (max favorable excursion), MAE, and simulates alternative
exit strategies to find what leaves least money on the table.
"""

import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

def load_nq_data():
    """Load NQ 1-min bar data from csv/ directory using the project data loader."""
    import sys
    sys.path.insert(0, str(project_root))
    from data.loader import load_csv

    try:
        nq = load_csv('NQ')
        print(f'Loaded {len(nq)} NQ bars via data.loader')
        return nq
    except Exception as e:
        print(f'Error loading via data.loader: {e}')
        # Fallback: direct CSV load
        csv_dir = project_root / 'csv'
        nq_file = csv_dir / 'NQ_Volumetric_1.csv'
        if nq_file.exists():
            nq = pd.read_csv(nq_file, parse_dates=['timestamp'])
            print(f'Loaded {len(nq)} NQ bars from {nq_file}')
            return nq
        print('ERROR: Cannot find NQ data')
        return None


def compute_mfe_mae(ef, nq):
    """Compute MFE/MAE for each Edge Fade trade."""
    results = []

    for idx, trade in ef.iterrows():
        entry_time = pd.Timestamp(trade['entry_time'])
        entry_price = trade['entry_price']
        target_price = trade['target_price']
        stop_price = trade['stop_price']
        session_date = str(trade['session_date'])[:10]
        session_end = pd.Timestamp(session_date + ' 15:30:00')

        # Bars after entry to session end
        mask = (nq['timestamp'] >= entry_time) & (nq['timestamp'] <= session_end)
        session_bars = nq[mask]

        if len(session_bars) == 0:
            continue

        # MFE/MAE
        mfe_price = session_bars['high'].max()
        mfe_pts = mfe_price - entry_price
        mae_price = session_bars['low'].min()
        mae_pts = entry_price - mae_price

        # Actual capture
        actual_pts = trade['exit_price'] - entry_price
        ib_mid_dist = target_price - entry_price
        stop_dist = entry_price - stop_price

        # VWAP at entry
        entry_mask = (nq['timestamp'] >= entry_time - pd.Timedelta(minutes=1)) & \
                     (nq['timestamp'] <= entry_time + pd.Timedelta(minutes=1))
        entry_bars = nq[entry_mask]
        vwap_at_entry = None
        if len(entry_bars) > 0 and 'vwap' in entry_bars.columns:
            vwap_at_entry = entry_bars['vwap'].iloc[0]

        # Check different target levels hit
        bars_after = nq[(nq['timestamp'] > entry_time) & (nq['timestamp'] <= session_end)]
        hit_50 = (bars_after['high'] >= entry_price + 50).any() if len(bars_after) > 0 else False
        hit_75 = (bars_after['high'] >= entry_price + 75).any() if len(bars_after) > 0 else False
        hit_100 = (bars_after['high'] >= entry_price + 100).any() if len(bars_after) > 0 else False
        hit_ib_mid = (bars_after['high'] >= target_price).any() if len(bars_after) > 0 else False

        # Time to MFE
        if len(bars_after) > 0:
            mfe_idx = bars_after['high'].idxmax()
            mfe_time = nq.loc[mfe_idx, 'timestamp']
            time_to_mfe = (mfe_time - entry_time).total_seconds() / 60
        else:
            time_to_mfe = 0

        results.append({
            'session': session_date,
            'entry_time': str(entry_time),
            'entry_price': entry_price,
            'exit_price': trade['exit_price'],
            'exit_reason': trade['exit_reason'],
            'is_winner': trade['is_winner'],
            'actual_pts': actual_pts,
            'net_pnl': trade['net_pnl'],
            'contracts': trade['contracts'],
            'mfe_pts': mfe_pts,
            'mae_pts': mae_pts,
            'ib_mid_dist': ib_mid_dist,
            'target_dist': ib_mid_dist,
            'stop_dist': stop_dist,
            'vwap_at_entry': vwap_at_entry,
            'hit_50pt': hit_50,
            'hit_75pt': hit_75,
            'hit_100pt': hit_100,
            'hit_ib_mid': hit_ib_mid,
            'time_to_mfe_min': time_to_mfe,
        })

    return pd.DataFrame(results)


def simulate_exits(ef, nq):
    """Simulate alternative exit strategies bar-by-bar."""
    strategies = {
        'Current (IB mid)': [],
        '50pt fixed': [],
        '75pt fixed': [],
        '2R target': [],
        '1.5R target': [],
        'Trail 25pt': [],
        'Trail 35pt': [],
        'Trail 50pt': [],
        'IB mid + trail 25pt': [],
        'Partial 50pt + trail': [],
    }

    for _, trade in ef.iterrows():
        entry_time = pd.Timestamp(trade['entry_time'])
        entry_price = trade['entry_price']
        stop_price_orig = trade['stop_price']
        target_price = trade['target_price']
        session_date = str(trade['session_date'])[:10]
        session_end = pd.Timestamp(session_date + ' 15:30:00')
        contracts = trade['contracts']

        risk = entry_price - stop_price_orig
        if risk <= 0:
            risk = 40  # default for breakeven stops

        bars_after = nq[(nq['timestamp'] > entry_time) & (nq['timestamp'] <= session_end)]
        if len(bars_after) == 0:
            for k in strategies:
                strategies[k].append(0)
            continue

        # Current result
        strategies['Current (IB mid)'].append(trade['net_pnl'])

        # Fixed target strategies
        for strat_name, target_pts in [('50pt fixed', 50), ('75pt fixed', 75)]:
            target = entry_price + target_pts
            exited = False
            for _, b in bars_after.iterrows():
                if b['low'] <= stop_price_orig:
                    pnl = (stop_price_orig - entry_price) * 2 * contracts - contracts * 1.24
                    strategies[strat_name].append(pnl)
                    exited = True
                    break
                if b['high'] >= target:
                    pnl = (target - entry_price) * 2 * contracts - contracts * 1.24
                    strategies[strat_name].append(pnl)
                    exited = True
                    break
            if not exited:
                eod_price = bars_after.iloc[-1]['close']
                pnl = (eod_price - entry_price) * 2 * contracts - contracts * 1.24
                strategies[strat_name].append(pnl)

        # R-multiple targets
        for label, mult in [('2R target', 2.0), ('1.5R target', 1.5)]:
            target = entry_price + risk * mult
            exited = False
            for _, b in bars_after.iterrows():
                if b['low'] <= stop_price_orig:
                    pnl = (stop_price_orig - entry_price) * 2 * contracts - contracts * 1.24
                    strategies[label].append(pnl)
                    exited = True
                    break
                if b['high'] >= target:
                    pnl = (target - entry_price) * 2 * contracts - contracts * 1.24
                    strategies[label].append(pnl)
                    exited = True
                    break
            if not exited:
                eod_price = bars_after.iloc[-1]['close']
                pnl = (eod_price - entry_price) * 2 * contracts - contracts * 1.24
                strategies[label].append(pnl)

        # Trailing stop strategies
        for label, trail_dist in [('Trail 25pt', 25), ('Trail 35pt', 35), ('Trail 50pt', 50)]:
            peak = entry_price
            exited = False
            for _, b in bars_after.iterrows():
                if b['low'] <= stop_price_orig:
                    pnl = (stop_price_orig - entry_price) * 2 * contracts - contracts * 1.24
                    strategies[label].append(pnl)
                    exited = True
                    break
                if b['high'] > peak:
                    peak = b['high']
                trail_stop = peak - trail_dist
                if trail_stop > entry_price and b['low'] <= trail_stop:
                    pnl = (trail_stop - entry_price) * 2 * contracts - contracts * 1.24
                    strategies[label].append(pnl)
                    exited = True
                    break
            if not exited:
                eod_price = bars_after.iloc[-1]['close']
                pnl = (eod_price - entry_price) * 2 * contracts - contracts * 1.24
                strategies[label].append(pnl)

        # Hybrid: IB mid target OR trail 25pt from high (whichever first)
        peak = entry_price
        exited = False
        for _, b in bars_after.iterrows():
            if b['low'] <= stop_price_orig:
                pnl = (stop_price_orig - entry_price) * 2 * contracts - contracts * 1.24
                strategies['IB mid + trail 25pt'].append(pnl)
                exited = True
                break
            if b['high'] >= target_price:
                pnl = (target_price - entry_price) * 2 * contracts - contracts * 1.24
                strategies['IB mid + trail 25pt'].append(pnl)
                exited = True
                break
            if b['high'] > peak:
                peak = b['high']
            trail_stop = peak - 25
            if trail_stop > entry_price and b['low'] <= trail_stop:
                pnl = (trail_stop - entry_price) * 2 * contracts - contracts * 1.24
                strategies['IB mid + trail 25pt'].append(pnl)
                exited = True
                break
        if not exited:
            eod_price = bars_after.iloc[-1]['close']
            pnl = (eod_price - entry_price) * 2 * contracts - contracts * 1.24
            strategies['IB mid + trail 25pt'].append(pnl)

        # Partial: take half at 50pts, trail rest with 25pt trail
        half = max(1, contracts // 2)
        other_half = contracts - half
        exited_first = False
        exited_second = False
        pnl_total = 0
        peak = entry_price

        for _, b in bars_after.iterrows():
            # Check stop first
            if b['low'] <= stop_price_orig:
                if not exited_first:
                    pnl_total += (stop_price_orig - entry_price) * 2 * half - half * 1.24
                if not exited_second:
                    pnl_total += (stop_price_orig - entry_price) * 2 * other_half - other_half * 1.24
                exited_first = True
                exited_second = True
                break

            # First half: 50pt target
            if not exited_first and b['high'] >= entry_price + 50:
                pnl_total += 50 * 2 * half - half * 1.24
                exited_first = True

            # Track peak for trail
            if b['high'] > peak:
                peak = b['high']

            # Second half: trail 25pt from peak (only after first half exits)
            if exited_first and not exited_second:
                trail_stop = peak - 25
                if trail_stop > entry_price and b['low'] <= trail_stop:
                    pnl_total += (trail_stop - entry_price) * 2 * other_half - other_half * 1.24
                    exited_second = True
                    break

        # EOD exit for anything not exited
        if not exited_first or not exited_second:
            eod_price = bars_after.iloc[-1]['close']
            if not exited_first:
                pnl_total += (eod_price - entry_price) * 2 * half - half * 1.24
            if not exited_second:
                pnl_total += (eod_price - entry_price) * 2 * other_half - other_half * 1.24

        strategies['Partial 50pt + trail'].append(pnl_total)

    return strategies


def analyze_session_loss_limits(ef):
    """Research per-session loss limits."""
    print('\n' + '=' * 80)
    print('  RESEARCH 2: PER-SESSION LOSS LIMITS')
    print('=' * 80)

    # Group by session
    ef_copy = ef.copy()
    ef_copy['session'] = ef_copy['session_date'].str[:10]

    session_stats = []
    for session, group in ef_copy.groupby('session'):
        trades = group.sort_values('entry_time')
        n_trades = len(trades)
        n_stops = len(trades[trades['exit_reason'] == 'STOP'])
        session_net = trades['net_pnl'].sum()

        # Cumulative P&L within session
        cum_pnl = trades['net_pnl'].cumsum()
        max_dd = cum_pnl.min()

        # Which trade turned session profitable?
        first_win_idx = None
        for i, (_, t) in enumerate(trades.iterrows()):
            if cum_pnl.iloc[i] > 0:
                first_win_idx = i
                break

        session_stats.append({
            'session': session,
            'n_trades': n_trades,
            'n_stops': n_stops,
            'session_net': session_net,
            'max_session_dd': max_dd,
            'first_profitable_trade': first_win_idx,
        })

    sdf = pd.DataFrame(session_stats)

    # Simulate different max-stop-per-session rules
    print('\nMax stops per session analysis:')
    for max_stops in [1, 2, 3, 4, None]:
        label = f'Max {max_stops} stops' if max_stops else 'No limit (current)'

        total_net = 0
        total_trades = 0
        sessions_with_trades = 0

        for _, session_row in sdf.iterrows():
            session = session_row['session']
            trades = ef_copy[ef_copy['session'] == session].sort_values('entry_time')

            stop_count = 0
            session_pnl = 0
            session_trade_count = 0

            for _, trade in trades.iterrows():
                if max_stops is not None and stop_count >= max_stops:
                    break  # Stop trading this session

                session_pnl += trade['net_pnl']
                session_trade_count += 1

                if trade['exit_reason'] == 'STOP':
                    stop_count += 1

            total_net += session_pnl
            total_trades += session_trade_count
            if session_trade_count > 0:
                sessions_with_trades += 1

        avg = total_net / total_trades if total_trades > 0 else 0
        print(f'  {label:<22s}: {total_trades:>3d} trades, net=${total_net:>+9,.0f}, avg=${avg:>+7,.1f}/trade')

    # Max daily loss limit
    print('\nMax daily loss limit:')
    for max_loss in [-200, -400, -600, -800, None]:
        label = f'${max_loss} daily loss limit' if max_loss else 'No limit (current)'

        total_net = 0
        total_trades = 0

        for _, session_row in sdf.iterrows():
            session = session_row['session']
            trades = ef_copy[ef_copy['session'] == session].sort_values('entry_time')

            session_pnl = 0
            for _, trade in trades.iterrows():
                if max_loss is not None and session_pnl <= max_loss:
                    break  # Stop trading
                session_pnl += trade['net_pnl']
                total_trades += 1

            total_net += session_pnl

        avg = total_net / total_trades if total_trades > 0 else 0
        print(f'  {label:<25s}: {total_trades:>3d} trades, net=${total_net:>+9,.0f}, avg=${avg:>+7,.1f}/trade')

    # Worst sessions
    print('\nWorst Edge Fade sessions:')
    for _, row in sdf.nsmallest(5, 'session_net').iterrows():
        print(f"  {row['session']}: {row['n_trades']} trades, {row['n_stops']} stops, net=${row['session_net']:+,.0f}")


def analyze_trend_pm_failure(nq, trades):
    """Research trend day PM failure combined with fade strategies."""
    print('\n' + '=' * 80)
    print('  RESEARCH 3: TREND DAY PM FAILURE -> FADE OPPORTUNITY')
    print('=' * 80)

    # Find sessions with Trend Bull / P-Day entries
    trend_sessions = trades[trades['strategy_name'].isin(['Trend Day Bull', 'P-Day'])]['session_date'].str[:10].unique()

    # Find sessions with Edge Fade entries
    ef_sessions = trades[trades['strategy_name'] == 'Edge Fade']['session_date'].str[:10].unique()

    # Overlap: sessions with BOTH trend AND edge fade
    overlap = set(trend_sessions) & set(ef_sessions)

    print(f'Trend/P-Day sessions: {len(trend_sessions)}')
    print(f'Edge Fade sessions: {len(ef_sessions)}')
    print(f'Sessions with BOTH: {len(overlap)}')

    if overlap:
        print(f'\nOverlapping sessions:')
        for s in sorted(overlap):
            trend_trades = trades[(trades['session_date'].str[:10] == s) &
                                  (trades['strategy_name'].isin(['Trend Day Bull', 'P-Day']))]
            ef_trades = trades[(trades['session_date'].str[:10] == s) &
                                (trades['strategy_name'] == 'Edge Fade')]

            trend_pnl = trend_trades['net_pnl'].sum()
            ef_pnl = ef_trades['net_pnl'].sum()
            trend_exit = trend_trades['exit_reason'].iloc[0] if len(trend_trades) > 0 else 'N/A'

            print(f'  {s}: Trend ${trend_pnl:+,.0f} ({trend_exit}) | Edge Fade ${ef_pnl:+,.0f} ({len(ef_trades)} trades)')

    # Sessions where trend entered but failed (stop/breakeven)
    trend_failures = trades[(trades['strategy_name'].isin(['Trend Day Bull', 'P-Day'])) &
                            (trades['is_winner'] == 0)]
    print(f'\nTrend/P-Day FAILURES: {len(trend_failures)} trades')
    for _, t in trend_failures.iterrows():
        s = str(t['session_date'])[:10]
        # Did Edge Fade also fire?
        ef_on_same_day = trades[(trades['session_date'].str[:10] == s) &
                                 (trades['strategy_name'] == 'Edge Fade')]
        ef_status = f'{len(ef_on_same_day)} EF trades, ${ef_on_same_day["net_pnl"].sum():+,.0f}' if len(ef_on_same_day) > 0 else 'No EF'
        print(f'  {s}: {t["strategy_name"]} {t["exit_reason"]} ${t["net_pnl"]:+,.0f} | {ef_status}')

    # Key question: on sessions where trend fails in PM,
    # does price fade back to IB range (Edge Fade opportunity)?
    print('\nSessions where trend WORKS + edge fade WORKS (complementary):')
    for s in sorted(trend_sessions):
        trend_t = trades[(trades['session_date'].str[:10] == s) &
                          (trades['strategy_name'].isin(['Trend Day Bull', 'P-Day']))]
        ef_t = trades[(trades['session_date'].str[:10] == s) &
                       (trades['strategy_name'] == 'Edge Fade')]

        if len(trend_t) > 0 and len(ef_t) > 0:
            trend_pnl = trend_t['net_pnl'].sum()
            ef_pnl = ef_t['net_pnl'].sum()
            combined = trend_pnl + ef_pnl
            print(f'  {s}: Trend ${trend_pnl:+,.0f} + EF ${ef_pnl:+,.0f} = ${combined:+,.0f}')

    # Day type distribution
    print('\nDay type distribution across all 62 sessions:')
    # Group all trades by session
    all_sessions = trades.groupby(trades['session_date'].str[:10]).agg({
        'day_type': 'first',
        'strategy_name': lambda x: list(x.unique()),
        'net_pnl': 'sum',
    })
    print(all_sessions['day_type'].value_counts())


def main():
    print('Loading data...')
    nq = load_nq_data()
    trades = pd.read_csv(project_root / 'output' / 'trade_log.csv')
    ef = trades[trades['strategy_name'] == 'Edge Fade'].copy()
    print(f'Edge Fade trades: {len(ef)}')

    if nq is None:
        print('ERROR: Could not load NQ data')
        return

    # ---- RESEARCH 1: MFE/MAE + Alternative Exits ----
    print('\n' + '=' * 80)
    print('  RESEARCH 1: MFE/MAE + ALTERNATIVE EXIT STRATEGIES')
    print('=' * 80)

    rdf = compute_mfe_mae(ef, nq)

    print(f'\nTrades analyzed: {len(rdf)}')
    print(f'MFE (pts): mean={rdf["mfe_pts"].mean():.1f}, median={rdf["mfe_pts"].median():.1f}, '
          f'p25={rdf["mfe_pts"].quantile(0.25):.1f}, p75={rdf["mfe_pts"].quantile(0.75):.1f}')
    print(f'MAE (pts): mean={rdf["mae_pts"].mean():.1f}, median={rdf["mae_pts"].median():.1f}')
    print(f'Actual capture: mean={rdf["actual_pts"].mean():.1f}, median={rdf["actual_pts"].median():.1f}')
    print(f'IB mid distance: mean={rdf["ib_mid_dist"].mean():.1f}, median={rdf["ib_mid_dist"].median():.1f}')
    print(f'Time to MFE: mean={rdf["time_to_mfe_min"].mean():.0f} min, median={rdf["time_to_mfe_min"].median():.0f} min')

    rdf['capture_eff'] = rdf['actual_pts'] / rdf['mfe_pts'].replace(0, np.nan) * 100
    print(f'Capture efficiency: mean={rdf["capture_eff"].mean():.1f}%, median={rdf["capture_eff"].median():.1f}%')

    rdf['mfe_vs_target'] = rdf['mfe_pts'] / rdf['target_dist'].replace(0, np.nan)
    print(f'MFE / target distance: mean={rdf["mfe_vs_target"].mean():.2f}x, median={rdf["mfe_vs_target"].median():.2f}x')

    print(f'\nTarget hit rates (ignoring stop):')
    print(f'  50pt fixed:  {rdf["hit_50pt"].mean()*100:.1f}%')
    print(f'  75pt fixed:  {rdf["hit_75pt"].mean()*100:.1f}%')
    print(f'  100pt fixed: {rdf["hit_100pt"].mean()*100:.1f}%')
    print(f'  IB midpoint: {rdf["hit_ib_mid"].mean()*100:.1f}%')

    # Winners vs losers
    winners = rdf[rdf['is_winner'] == 1]
    losers = rdf[rdf['is_winner'] == 0]
    print(f'\nWinners ({len(winners)}): MFE={winners["mfe_pts"].mean():.1f}, actual={winners["actual_pts"].mean():.1f}, eff={winners["capture_eff"].mean():.1f}%')
    print(f'Losers ({len(losers)}):  MFE={losers["mfe_pts"].mean():.1f}, actual={losers["actual_pts"].mean():.1f}, MAE={losers["mae_pts"].mean():.1f}')

    losers_had_profit = losers[losers['mfe_pts'] > 10]
    print(f'\nLosers with MFE>10pts before stop: {len(losers_had_profit)}/{len(losers)} ({len(losers_had_profit)/len(losers)*100:.0f}%)')
    if len(losers_had_profit) > 0:
        print(f'  Avg MFE: {losers_had_profit["mfe_pts"].mean():.1f}, Avg MAE: {losers_had_profit["mae_pts"].mean():.1f}')

    # EOD analysis
    eod = rdf[rdf['exit_reason'] == 'EOD']
    print(f'\nEOD exits ({len(eod)}): MFE={eod["mfe_pts"].mean():.1f}, actual={eod["actual_pts"].mean():.1f}')
    print(f'  Wasted MFE: {(eod["mfe_pts"] - eod["actual_pts"]).mean():.1f} pts left on table')
    eod_hit = eod[eod['hit_ib_mid'] == True]
    print(f'  EOD trades that DID hit IB mid during session: {len(eod_hit)}/{len(eod)}')

    # By exit reason
    print('\n--- BY EXIT REASON ---')
    for reason in ['TARGET', 'STOP', 'EOD']:
        sub = rdf[rdf['exit_reason'] == reason]
        if len(sub) > 0:
            print(f'{reason} ({len(sub)}): MFE={sub["mfe_pts"].mean():.1f}, actual={sub["actual_pts"].mean():.1f}, '
                  f'eff={sub["capture_eff"].mean():.1f}%, '
                  f'50pt={sub["hit_50pt"].mean()*100:.0f}%, 75pt={sub["hit_75pt"].mean()*100:.0f}%, 100pt={sub["hit_100pt"].mean()*100:.0f}%')

    # ---- SIMULATE ALTERNATIVE EXITS ----
    print('\n' + '=' * 80)
    print('  ALTERNATIVE EXIT STRATEGY SIMULATION')
    print('=' * 80)

    strategies = simulate_exits(ef, nq)

    print(f'\n{"Strategy":<28s} {"Trades":>6s} {"Net PnL":>10s} {"Avg/Tr":>10s} {"WR":>6s} {"WinAvg":>9s} {"LossAvg":>9s}')
    print('-' * 80)
    for name, pnls in strategies.items():
        if not pnls:
            continue
        arr = np.array(pnls)
        wins_arr = arr[arr > 0]
        losses_arr = arr[arr <= 0]
        wr = len(wins_arr) / len(arr) * 100 if len(arr) > 0 else 0
        win_avg = wins_arr.mean() if len(wins_arr) > 0 else 0
        loss_avg = losses_arr.mean() if len(losses_arr) > 0 else 0
        print(f'{name:<28s} {len(arr):>6d} ${arr.sum():>9,.0f} ${arr.mean():>9,.1f} {wr:>5.1f}% ${win_avg:>8,.1f} ${loss_avg:>8,.1f}')

    # ---- RESEARCH 2: Session Loss Limits ----
    analyze_session_loss_limits(ef)

    # ---- RESEARCH 3: Trend PM Failure ----
    analyze_trend_pm_failure(nq, trades)


if __name__ == '__main__':
    main()
