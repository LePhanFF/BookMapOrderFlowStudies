"""
Diagnostic 10: STANDALONE ORDER FLOW STRATEGY + HTF FILTER COMPARISON

Test pure order flow entries (no Dalton day type framework) and compare:
  A) Order Flow standalone (pure OF signals)
  B) Order Flow + 5-min HTF trend filter
  C) Order Flow + 15-min HTF trend filter
  D) Order Flow + 30-min HTF trend filter
  E) Playbook alone (current system, from trade_log.csv)
  F) Playbook + Order Flow (current system with OF quality gate)

Order Flow Entry Models tested:
  1. Delta Surge LONG: delta_zscore > 1.5, delta > 0, volume_spike > 1.3
  2. Absorption LONG: large vol_bid but price holds/rises (buyers absorbing sellers)
     imbalance_ratio > 1.2, volume_spike > 1.5, delta > 0, price > vwap
  3. CVD Divergence LONG: price makes new low but CVD makes higher low
  4. Delta Momentum LONG: 5-bar delta sum strongly positive, acceleration > 0
  5. Combined High-Conviction: 3+ of above signals fire simultaneously

HTF Trend Filters:
  - 5-min: EMA8 > EMA21 on 5-min bars (short-term trend aligned bullish)
  - 15-min: EMA8 > EMA21 on 15-min bars (medium-term trend aligned)
  - 30-min: EMA8 > EMA21 on 30-min bars (longer-term trend aligned)

All entries:
  - LONG only (NQ long bias)
  - Stop: entry - 0.4x IB range (or ATR-based if no IB context)
  - Target: none (EOD exit for fair comparison with playbook)
  - Max 1 entry per session per model
  - Entry window: 10:30 - 14:00
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import time, timedelta
from collections import defaultdict

from data.loader import load_csv
from data.session import filter_rth
from data.features import compute_all_features
from config.constants import IB_BARS_1MIN
from indicators.technical import calculate_ema


# ─── HTF Bar Construction ───────────────────────────────────────────────

def build_htf_bars(df, period_minutes):
    """Resample 1-min bars to HTF bars per session and compute EMA trend."""
    df = df.copy()
    if 'timestamp' not in df.columns:
        return df

    htf_ema8_col = f'htf_{period_minutes}m_ema8'
    htf_ema21_col = f'htf_{period_minutes}m_ema21'
    htf_trend_col = f'htf_{period_minutes}m_bullish'

    df[htf_ema8_col] = np.nan
    df[htf_ema21_col] = np.nan
    df[htf_trend_col] = False

    for session_date, session_df in df.groupby('session_date'):
        if len(session_df) < period_minutes * 3:
            continue

        session_ts = session_df.set_index('timestamp')
        htf = session_ts.resample(f'{period_minutes}min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }).dropna()

        if len(htf) < 5:
            continue

        htf['ema8'] = calculate_ema(htf['close'], 8)
        htf['ema21'] = calculate_ema(htf['close'], 21)
        htf['bullish'] = htf['ema8'] > htf['ema21']

        # Map back to 1-min bars: each 1-min bar gets the HTF values
        # from the COMPLETED HTF bar (not the current one being built)
        for idx in session_df.index:
            ts = df.loc[idx, 'timestamp']
            # Find the last completed HTF bar before this timestamp
            completed = htf[htf.index <= ts]
            if len(completed) > 0:
                last_htf = completed.iloc[-1]
                df.at[idx, htf_ema8_col] = last_htf['ema8']
                df.at[idx, htf_ema21_col] = last_htf['ema21']
                df.at[idx, htf_trend_col] = last_htf['bullish']

    return df


# ─── Order Flow Entry Signal Detection ──────────────────────────────────

def detect_of_entries(session_df, ib_high, ib_low, ib_range, post_ib_start_idx):
    """
    Scan post-IB bars for standalone order flow entry signals.
    Returns list of entry dicts with signal type, bar info, etc.
    """
    entries = []
    post_ib = session_df.iloc[post_ib_start_idx:]

    if len(post_ib) == 0:
        return entries

    # Track per-model: only 1 entry per model per session
    taken = set()

    # Rolling windows for momentum/divergence
    delta_window = []
    cvd_lows = []  # (bar_idx, cvd_value, price_low)
    recent_price_low = float('inf')
    recent_cvd_at_low = None

    for i in range(len(post_ib)):
        bar = post_ib.iloc[i]
        ts = bar.get('timestamp')
        bar_time = ts.time() if hasattr(ts, 'time') else None

        if bar_time is None:
            continue
        if bar_time < time(10, 30) or bar_time >= time(14, 0):
            continue

        price = bar['close']
        delta = bar.get('delta', 0)
        if pd.isna(delta):
            delta = 0
        delta_zs = bar.get('delta_zscore', 0)
        if pd.isna(delta_zs):
            delta_zs = 0
        delta_pctl = bar.get('delta_percentile', 50)
        if pd.isna(delta_pctl):
            delta_pctl = 50
        imb = bar.get('imbalance_ratio', 1.0)
        if pd.isna(imb):
            imb = 1.0
        vol_spike = bar.get('volume_spike', 1.0)
        if pd.isna(vol_spike):
            vol_spike = 1.0
        cvd = bar.get('cumulative_delta', 0)
        if pd.isna(cvd):
            cvd = 0
        vwap = bar.get('vwap')

        # Rolling delta for momentum
        delta_window.append(delta)
        if len(delta_window) > 5:
            delta_window.pop(0)

        # Track price lows and CVD at those lows (for divergence)
        if bar['low'] < recent_price_low:
            recent_price_low = bar['low']
            recent_cvd_at_low = cvd

        # Stop and EOD prices for P&L calc
        stop_dist = ib_range * 0.4 if ib_range > 0 else 40.0
        stop_price = price - stop_dist
        eod_price = post_ib.iloc[-1]['close']

        # Check if stop gets hit
        remaining = post_ib.iloc[i + 1:]
        hit_stop = False
        exit_price = eod_price
        for _, future_bar in remaining.iterrows():
            if future_bar['low'] <= stop_price:
                hit_stop = True
                exit_price = stop_price
                break

        pnl_pts = exit_price - price

        base_entry = {
            'bar_time': bar_time,
            'price': price,
            'delta': delta,
            'delta_zscore': delta_zs,
            'delta_pctl': delta_pctl,
            'imbalance': imb,
            'vol_spike': vol_spike,
            'cvd': cvd,
            'pnl_pts': pnl_pts,
            'hit_stop': hit_stop,
            'exit_price': exit_price,
            'ib_range': ib_range,
            'htf_5m': bool(bar.get('htf_5m_bullish', False)),
            'htf_15m': bool(bar.get('htf_15m_bullish', False)),
            'htf_30m': bool(bar.get('htf_30m_bullish', False)),
        }

        # ── Model 1: Delta Surge ──
        if 'delta_surge' not in taken:
            if delta_zs > 1.5 and delta > 0 and vol_spike > 1.3:
                entry = {**base_entry, 'model': 'delta_surge'}
                entries.append(entry)
                taken.add('delta_surge')

        # ── Model 2: Absorption (buyers absorbing sellers) ──
        if 'absorption' not in taken:
            if (imb > 1.2 and vol_spike > 1.5 and delta > 0
                    and vwap is not None and not pd.isna(vwap) and price > vwap):
                entry = {**base_entry, 'model': 'absorption'}
                entries.append(entry)
                taken.add('absorption')

        # ── Model 3: CVD Divergence ──
        # Price at/near recent low but CVD is higher than previous low's CVD
        if 'cvd_divergence' not in taken:
            if (recent_cvd_at_low is not None and i > 20
                    and bar['low'] <= recent_price_low * 1.001  # within 0.1%
                    and cvd > recent_cvd_at_low + 500  # CVD higher by significant amount
                    and delta > 0):
                entry = {**base_entry, 'model': 'cvd_divergence'}
                entries.append(entry)
                taken.add('cvd_divergence')

        # ── Model 4: Delta Momentum ──
        if 'delta_momentum' not in taken:
            if len(delta_window) >= 5:
                delta_sum_5 = sum(delta_window)
                first_half = sum(delta_window[:2])
                second_half = sum(delta_window[3:])
                accel = second_half - first_half
                if (delta_sum_5 > 300 and accel > 0 and delta > 0
                        and delta_pctl >= 70):
                    entry = {**base_entry, 'model': 'delta_momentum'}
                    entries.append(entry)
                    taken.add('delta_momentum')

        # ── Model 5: High Conviction Composite ──
        if 'high_conviction' not in taken:
            signals = 0
            if delta_zs > 1.0 and delta > 0:
                signals += 1
            if imb > 1.15:
                signals += 1
            if vol_spike > 1.3:
                signals += 1
            if delta_pctl >= 80:
                signals += 1
            if vwap is not None and not pd.isna(vwap) and price > vwap:
                signals += 1

            if signals >= 4:
                entry = {**base_entry, 'model': 'high_conviction'}
                entries.append(entry)
                taken.add('high_conviction')

    return entries


# ─── Main Analysis ──────────────────────────────────────────────────────

def analyze():
    print(f"\n{'='*140}")
    print(f"  STANDALONE ORDER FLOW vs PLAYBOOK: COMPREHENSIVE COMPARISON")
    print(f"{'='*140}\n")

    # Load and prepare data
    df = load_csv('NQ')
    df = filter_rth(df)
    df = compute_all_features(df)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    if 'session_date' not in df.columns:
        df['session_date'] = df['timestamp'].dt.date

    # Build HTF bars
    print("Building HTF trend filters...")
    df = build_htf_bars(df, 5)
    print("  5-min HTF done")
    df = build_htf_bars(df, 15)
    print("  15-min HTF done")
    df = build_htf_bars(df, 30)
    print("  30-min HTF done")

    sessions = sorted(df['session_date'].unique())
    print(f"\n{len(sessions)} sessions loaded\n")

    # ─── Collect all OF entries across all sessions ───
    all_entries = []

    for session_date in sessions:
        session_df = df[df['session_date'] == session_date].copy()
        if len(session_df) < IB_BARS_1MIN:
            continue

        ib_df = session_df.head(IB_BARS_1MIN)
        ib_high = ib_df['high'].max()
        ib_low = ib_df['low'].min()
        ib_range = ib_high - ib_low

        if ib_range <= 0:
            continue

        entries = detect_of_entries(session_df, ib_high, ib_low, ib_range,
                                    post_ib_start_idx=IB_BARS_1MIN)

        for e in entries:
            e['session_date'] = str(session_date)[:10]
            all_entries.append(e)

    entries_df = pd.DataFrame(all_entries)
    if len(entries_df) == 0:
        print("No OF entries found!")
        return

    # ─── Load playbook results for comparison ───
    trade_log = pd.read_csv(Path(project_root) / 'output' / 'trade_log.csv')
    # Deduplicate playbook (Trend Bull and P-Day fire on same bar)
    playbook_unique = trade_log.drop_duplicates(subset=['session_date', 'entry_price'])
    playbook_unique = playbook_unique.copy()

    # ─── Results by Model ───
    print(f"{'='*140}")
    print(f"  ORDER FLOW STANDALONE: RESULTS BY MODEL")
    print(f"{'='*140}\n")

    # Use MNQ point value ($2) for P&L
    point_value = 2.0

    models = entries_df['model'].unique()

    print(f"{'Model':<20s}  {'Trades':>6s}  {'Wins':>5s}  {'Losses':>6s}  {'WR':>7s}  "
          f"{'Avg Win':>10s}  {'Avg Loss':>10s}  {'Net pts':>10s}  {'Net $':>10s}")
    print(f"{'-'*20}  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    model_results = {}

    for model in sorted(models):
        m_df = entries_df[entries_df['model'] == model]
        wins = m_df[m_df['pnl_pts'] > 0]
        losses = m_df[m_df['pnl_pts'] <= 0]
        wr = len(wins) / len(m_df) * 100 if len(m_df) > 0 else 0
        avg_win = wins['pnl_pts'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl_pts'].mean() if len(losses) > 0 else 0
        net_pts = m_df['pnl_pts'].sum()
        net_dollar = net_pts * point_value

        model_results[model] = {
            'trades': len(m_df), 'wins': len(wins), 'losses': len(losses),
            'wr': wr, 'net_pts': net_pts, 'net_dollar': net_dollar,
        }

        print(f"{model:<20s}  {len(m_df):>6d}  {len(wins):>5d}  {len(losses):>6d}  {wr:>6.1f}%  "
              f"{avg_win:>+10.1f}  {avg_loss:>+10.1f}  {net_pts:>+10.1f}  ${net_dollar:>+9.2f}")

    # ─── HTF Filter Impact ───
    print(f"\n{'='*140}")
    print(f"  ORDER FLOW + HTF TREND FILTER: IMPACT ON EACH MODEL")
    print(f"{'='*140}\n")

    htf_filters = [
        ('No Filter', lambda e: True),
        ('5-min EMA8>21', lambda e: e['htf_5m']),
        ('15-min EMA8>21', lambda e: e['htf_15m']),
        ('30-min EMA8>21', lambda e: e['htf_30m']),
    ]

    for model in sorted(models):
        m_df = entries_df[entries_df['model'] == model]
        print(f"\n--- {model} ({len(m_df)} total) ---")
        print(f"  {'HTF Filter':<20s}  {'Pass':>5s}  {'Win':>4s}  {'Loss':>5s}  {'WR':>7s}  {'Net pts':>10s}  {'Net $':>10s}")
        print(f"  {'-'*20}  {'-'*5}  {'-'*4}  {'-'*5}  {'-'*7}  {'-'*10}  {'-'*10}")

        for filter_name, filter_fn in htf_filters:
            filtered = m_df[m_df.apply(filter_fn, axis=1)]
            wins = filtered[filtered['pnl_pts'] > 0]
            losses = filtered[filtered['pnl_pts'] <= 0]
            wr = len(wins) / len(filtered) * 100 if len(filtered) > 0 else 0
            net_pts = filtered['pnl_pts'].sum()
            net_dollar = net_pts * point_value

            print(f"  {filter_name:<20s}  {len(filtered):>5d}  {len(wins):>4d}  {len(losses):>5d}  "
                  f"{wr:>6.1f}%  {net_pts:>+10.1f}  ${net_dollar:>+9.2f}")

    # ─── Best OF Model + Best HTF ───
    print(f"\n{'='*140}")
    print(f"  BEST COMBINATIONS: OF MODEL + HTF FILTER")
    print(f"{'='*140}\n")

    best_combos = []
    for model in sorted(models):
        m_df = entries_df[entries_df['model'] == model]
        for filter_name, filter_fn in htf_filters:
            filtered = m_df[m_df.apply(filter_fn, axis=1)]
            if len(filtered) == 0:
                continue
            wins = filtered[filtered['pnl_pts'] > 0]
            wr = len(wins) / len(filtered) * 100
            net_pts = filtered['pnl_pts'].sum()
            net_dollar = net_pts * point_value
            best_combos.append({
                'combo': f"{model} + {filter_name}",
                'trades': len(filtered),
                'wins': len(wins),
                'wr': wr,
                'net_pts': net_pts,
                'net_dollar': net_dollar,
                'expectancy': net_dollar / len(filtered) if len(filtered) > 0 else 0,
            })

    best_combos.sort(key=lambda x: x['net_dollar'], reverse=True)

    print(f"{'Combination':<45s}  {'Trades':>6s}  {'Win':>4s}  {'WR':>7s}  {'Net $':>10s}  {'Expect':>10s}")
    print(f"{'-'*45}  {'-'*6}  {'-'*4}  {'-'*7}  {'-'*10}  {'-'*10}")

    for c in best_combos[:15]:
        print(f"{c['combo']:<45s}  {c['trades']:>6d}  {c['wins']:>4d}  {c['wr']:>6.1f}%  "
              f"${c['net_dollar']:>+9.2f}  ${c['expectancy']:>+9.2f}")

    # ─── 4-WAY COMPARISON ───
    print(f"\n{'='*140}")
    print(f"  FINAL 4-WAY COMPARISON")
    print(f"{'='*140}\n")

    # A) Best standalone OF
    best_of_alone = max(best_combos, key=lambda x: x['net_dollar'] if 'No Filter' in x['combo'] else -99999)
    # Find best no-filter OF
    no_filter_combos = [c for c in best_combos if 'No Filter' in c['combo']]
    best_of_alone = max(no_filter_combos, key=lambda x: x['net_dollar']) if no_filter_combos else None

    # B) Best OF + HTF
    htf_combos = [c for c in best_combos if 'No Filter' not in c['combo']]
    best_of_htf = max(htf_combos, key=lambda x: x['net_dollar']) if htf_combos else None

    # C) Playbook alone
    pb_wins = len(playbook_unique[playbook_unique['is_winner'] == 1])
    pb_total = len(playbook_unique)
    pb_wr = pb_wins / pb_total * 100 if pb_total > 0 else 0
    pb_net = playbook_unique['net_pnl'].sum()
    pb_expect = pb_net / pb_total if pb_total > 0 else 0

    # D) Playbook + OF = already implemented (same as playbook since OF is built in)
    # Show all four
    approaches = []

    if best_of_alone:
        approaches.append({
            'name': f"OF Standalone ({best_of_alone['combo']})",
            'trades': best_of_alone['trades'],
            'wr': best_of_alone['wr'],
            'net': best_of_alone['net_dollar'],
            'expect': best_of_alone['expectancy'],
        })

    if best_of_htf:
        approaches.append({
            'name': f"OF + HTF ({best_of_htf['combo']})",
            'trades': best_of_htf['trades'],
            'wr': best_of_htf['wr'],
            'net': best_of_htf['net_dollar'],
            'expect': best_of_htf['expectancy'],
        })

    approaches.append({
        'name': 'Playbook (Dalton + OF Quality Gate)',
        'trades': pb_total,
        'wr': pb_wr,
        'net': pb_net,
        'expect': pb_expect,
    })

    # Also show ALL OF models combined (take best entry per session)
    # Group by session, take best entry
    if len(entries_df) > 0:
        session_best = entries_df.sort_values('pnl_pts', ascending=False).drop_duplicates('session_date', keep='first')
        sb_wins = len(session_best[session_best['pnl_pts'] > 0])
        sb_wr = sb_wins / len(session_best) * 100
        sb_net = session_best['pnl_pts'].sum() * point_value
        sb_expect = sb_net / len(session_best)
        approaches.append({
            'name': 'OF All Models (best per session)',
            'trades': len(session_best),
            'wr': sb_wr,
            'net': sb_net,
            'expect': sb_expect,
        })

        # Best entry per session with 15m HTF filter
        htf15_entries = entries_df[entries_df['htf_15m'] == True]
        if len(htf15_entries) > 0:
            htf15_best = htf15_entries.sort_values('pnl_pts', ascending=False).drop_duplicates('session_date', keep='first')
            h15_wins = len(htf15_best[htf15_best['pnl_pts'] > 0])
            h15_wr = h15_wins / len(htf15_best) * 100
            h15_net = htf15_best['pnl_pts'].sum() * point_value
            h15_expect = h15_net / len(htf15_best)
            approaches.append({
                'name': 'OF All + 15m HTF (best per session)',
                'trades': len(htf15_best),
                'wr': h15_wr,
                'net': h15_net,
                'expect': h15_expect,
            })

    approaches.sort(key=lambda x: x['net'], reverse=True)

    print(f"{'Approach':<50s}  {'Trades':>6s}  {'WR':>7s}  {'Net P&L':>12s}  {'Expectancy':>12s}")
    print(f"{'-'*50}  {'-'*6}  {'-'*7}  {'-'*12}  {'-'*12}")

    for a in approaches:
        print(f"{a['name']:<50s}  {a['trades']:>6d}  {a['wr']:>6.1f}%  ${a['net']:>+11.2f}  ${a['expect']:>+11.2f}")

    # ─── Per-entry detail for best OF model ───
    print(f"\n{'='*140}")
    print(f"  INDIVIDUAL OF ENTRIES (ALL MODELS)")
    print(f"{'='*140}\n")

    for _, e in entries_df.iterrows():
        result = 'WIN' if e['pnl_pts'] > 0 else 'LOSS'
        pnl_dollar = e['pnl_pts'] * point_value
        htf_flags = []
        if e['htf_5m']: htf_flags.append('5m')
        if e['htf_15m']: htf_flags.append('15m')
        if e['htf_30m']: htf_flags.append('30m')
        htf_str = '+'.join(htf_flags) if htf_flags else 'NONE'

        print(f"  {e['session_date']} {e['bar_time']} | {e['model']:<18s} | {result:4s} "
              f"${pnl_dollar:>+8.2f} ({e['pnl_pts']:>+6.1f} pts) | "
              f"dz={e['delta_zscore']:>5.2f} pctl={e['delta_pctl']:>3.0f} imb={e['imbalance']:>5.3f} "
              f"vol={e['vol_spike']:>5.2f} | HTF: {htf_str}")


if __name__ == '__main__':
    analyze()
