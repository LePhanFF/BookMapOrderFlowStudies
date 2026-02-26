"""
Analyze Balance Signal backtest results.

Measures per-factor efficacy, score threshold optimization, mode breakdown,
BPR accuracy, acceptance filter value, dynamic vs fixed targets, day type
distribution, and monthly P&L.

Usage:
    python scripts/analyze_balance_signal.py [--csv output/trade_log.csv]
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_trades(csv_path: str) -> pd.DataFrame:
    """Load trade log and filter to Balance Signal trades."""
    df = pd.read_csv(csv_path, parse_dates=['entry_time', 'exit_time'])
    strat_col = 'strategy_name' if 'strategy_name' in df.columns else 'strategy'
    bs = df[df[strat_col] == 'Balance Signal'].copy()
    if bs.empty:
        print("ERROR: No 'Balance Signal' trades found in trade log.")
        print(f"  Available strategies: {df[strat_col].unique().tolist()}")
        sys.exit(1)
    print(f"Loaded {len(bs)} Balance Signal trades from {csv_path}")
    return bs


def parse_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Extract score breakdown from metadata column."""
    import ast

    scores = []
    for _, row in df.iterrows():
        meta = row.get('metadata', '{}')
        if isinstance(meta, str):
            try:
                meta = ast.literal_eval(meta)
            except (ValueError, SyntaxError):
                meta = {}
        if not isinstance(meta, dict):
            meta = {}

        breakdown = meta.get('breakdown', {})
        scores.append({
            'total_score': meta.get('score', 0),
            'F1_profile_shape': breakdown.get('F1_profile_shape', 0),
            'F2_hvn_lvn': breakdown.get('F2_hvn_lvn', 0),
            'F3_volume_skew': breakdown.get('F3_volume_skew', 0),
            'F4_vwap': breakdown.get('F4_vwap', 0),
            'F5_ib_context': breakdown.get('F5_ib_context', 0),
            'F6_delta_cvd': breakdown.get('F6_delta_cvd', 0),
            'mode': meta.get('mode', 'UNKNOWN'),
            'bpr_active': meta.get('bpr_active', False),
        })

    score_df = pd.DataFrame(scores)
    return pd.concat([df.reset_index(drop=True), score_df], axis=1)


def analyze_per_factor(df: pd.DataFrame):
    """1. Per-factor contribution: WR when factor is max vs 0."""
    print("\n" + "=" * 70)
    print("1. PER-FACTOR CONTRIBUTION ANALYSIS")
    print("=" * 70)

    factor_cols = [c for c in df.columns if c.startswith('F')]
    is_winner = df['net_pnl'] > 0

    for col in factor_cols:
        max_val = df[col].max()
        if max_val == 0:
            continue

        at_max = df[col] == max_val
        at_zero = df[col] == 0

        max_n = at_max.sum()
        zero_n = at_zero.sum()

        max_wr = is_winner[at_max].mean() * 100 if max_n > 0 else 0
        zero_wr = is_winner[at_zero].mean() * 100 if zero_n > 0 else 0

        max_avg_pnl = df.loc[at_max, 'net_pnl'].mean() if max_n > 0 else 0
        zero_avg_pnl = df.loc[at_zero, 'net_pnl'].mean() if zero_n > 0 else 0

        print(f"\n  {col} (max={max_val}):")
        print(f"    At max:  {max_n:3d} trades, {max_wr:5.1f}% WR, ${max_avg_pnl:+.0f}/trade")
        print(f"    At zero: {zero_n:3d} trades, {zero_wr:5.1f}% WR, ${zero_avg_pnl:+.0f}/trade")
        edge = max_wr - zero_wr
        print(f"    Edge:    {edge:+.1f}% WR improvement")


def analyze_score_threshold(df: pd.DataFrame):
    """2. Score threshold analysis: WR at various cutoffs."""
    print("\n" + "=" * 70)
    print("2. SCORE THRESHOLD ANALYSIS")
    print("=" * 70)

    is_winner = df['net_pnl'] > 0

    print(f"\n  {'Score':>5} | {'Trades':>6} | {'WR':>6} | {'Avg P&L':>9} | {'Net P&L':>9} | {'PF':>5}")
    print("  " + "-" * 55)

    for threshold in range(3, 11):
        mask = df['total_score'] >= threshold
        n = mask.sum()
        if n == 0:
            continue
        wr = is_winner[mask].mean() * 100
        avg_pnl = df.loc[mask, 'net_pnl'].mean()
        net = df.loc[mask, 'net_pnl'].sum()
        wins = df.loc[mask & is_winner, 'net_pnl'].sum()
        losses = abs(df.loc[mask & ~is_winner, 'net_pnl'].sum())
        pf = wins / losses if losses > 0 else float('inf')
        print(f"  >= {threshold:2d} | {n:6d} | {wr:5.1f}% | ${avg_pnl:+8.0f} | ${net:+8.0f} | {pf:5.2f}")


def analyze_mode_breakdown(df: pd.DataFrame):
    """3. Mode breakdown: VA Edge Fade vs Wide IB Reclaim."""
    print("\n" + "=" * 70)
    print("3. MODE BREAKDOWN")
    print("=" * 70)

    is_winner = df['net_pnl'] > 0

    for mode in df['mode'].unique():
        mask = df['mode'] == mode
        n = mask.sum()
        if n == 0:
            continue
        wr = is_winner[mask].mean() * 100
        net = df.loc[mask, 'net_pnl'].sum()
        avg = df.loc[mask, 'net_pnl'].mean()
        wins = df.loc[mask & is_winner, 'net_pnl'].sum()
        losses = abs(df.loc[mask & ~is_winner, 'net_pnl'].sum())
        pf = wins / losses if losses > 0 else float('inf')

        print(f"\n  {mode}:")
        print(f"    Trades: {n}, WR: {wr:.1f}%, Net: ${net:+,.0f}, Avg: ${avg:+,.0f}/trade, PF: {pf:.2f}")

        # Direction breakdown
        for d in ['LONG', 'SHORT']:
            d_mask = mask & (df['direction'] == d)
            dn = d_mask.sum()
            if dn > 0:
                dwr = is_winner[d_mask].mean() * 100
                dnet = df.loc[d_mask, 'net_pnl'].sum()
                print(f"    {d}: {dn} trades, {dwr:.1f}% WR, ${dnet:+,.0f}")


def analyze_bpr_accuracy(df: pd.DataFrame):
    """4. BPR accuracy: when BPR is active, does it help?"""
    print("\n" + "=" * 70)
    print("4. BPR ACCURACY")
    print("=" * 70)

    is_winner = df['net_pnl'] > 0
    bpr_mask = df['bpr_active'] == True

    bpr_n = bpr_mask.sum()
    no_bpr_n = (~bpr_mask).sum()

    if bpr_n > 0:
        bpr_wr = is_winner[bpr_mask].mean() * 100
        bpr_avg = df.loc[bpr_mask, 'net_pnl'].mean()
        print(f"  With BPR:    {bpr_n} trades, {bpr_wr:.1f}% WR, ${bpr_avg:+,.0f}/trade")
    else:
        print("  With BPR:    0 trades")

    if no_bpr_n > 0:
        no_bpr_wr = is_winner[~bpr_mask].mean() * 100
        no_bpr_avg = df.loc[~bpr_mask, 'net_pnl'].mean()
        print(f"  Without BPR: {no_bpr_n} trades, {no_bpr_wr:.1f}% WR, ${no_bpr_avg:+,.0f}/trade")


def analyze_acceptance_filter(df: pd.DataFrame):
    """5. Acceptance filter value: compare with vs without Dalton acceptance."""
    print("\n" + "=" * 70)
    print("5. ACCEPTANCE FILTER VALUE")
    print("=" * 70)
    print("  (Dalton acceptance is required for all signals — no 'without' comparison)")
    print("  This filter is structural. All trades pass the 2-bar acceptance gate.")
    print("  To measure, would need to re-run backtest without acceptance requirement.")


def analyze_day_type_distribution(df: pd.DataFrame):
    """7. Day type distribution: which day types produce signals?"""
    print("\n" + "=" * 70)
    print("6. DAY TYPE DISTRIBUTION")
    print("=" * 70)

    is_winner = df['net_pnl'] > 0

    print(f"\n  {'Day Type':>15} | {'Trades':>6} | {'WR':>6} | {'Net P&L':>9} | {'Avg P&L':>9}")
    print("  " + "-" * 55)

    for dt in sorted(df['day_type'].unique()):
        mask = df['day_type'] == dt
        n = mask.sum()
        wr = is_winner[mask].mean() * 100
        net = df.loc[mask, 'net_pnl'].sum()
        avg = df.loc[mask, 'net_pnl'].mean()
        print(f"  {dt:>15} | {n:6d} | {wr:5.1f}% | ${net:+8.0f} | ${avg:+8.0f}")


def analyze_monthly_pnl(df: pd.DataFrame):
    """8. Monthly P&L grid."""
    print("\n" + "=" * 70)
    print("7. MONTHLY P&L GRID")
    print("=" * 70)

    df['month'] = df['entry_time'].dt.to_period('M')
    is_winner = df['net_pnl'] > 0

    print(f"\n  {'Month':>10} | {'Trades':>6} | {'WR':>6} | {'Net P&L':>9} | {'MaxDD':>8}")
    print("  " + "-" * 55)

    for month in sorted(df['month'].unique()):
        mask = df['month'] == month
        n = mask.sum()
        wr = is_winner[mask].mean() * 100
        net = df.loc[mask, 'net_pnl'].sum()

        # Approximate max drawdown within month
        cum = df.loc[mask, 'net_pnl'].cumsum()
        peak = cum.cummax()
        dd = (cum - peak).min()

        print(f"  {str(month):>10} | {n:6d} | {wr:5.1f}% | ${net:+8.0f} | ${dd:+7.0f}")

    total_net = df['net_pnl'].sum()
    total_wr = is_winner.mean() * 100
    cum_all = df['net_pnl'].cumsum()
    max_dd = (cum_all - cum_all.cummax()).min()
    print("  " + "-" * 55)
    print(f"  {'TOTAL':>10} | {len(df):6d} | {total_wr:5.1f}% | ${total_net:+8.0f} | ${max_dd:+7.0f}")


def print_summary(df: pd.DataFrame):
    """Print overall summary stats."""
    print("\n" + "=" * 70)
    print("BALANCE SIGNAL — BACKTEST SUMMARY")
    print("=" * 70)

    is_winner = df['net_pnl'] > 0
    n = len(df)
    wr = is_winner.mean() * 100
    net = df['net_pnl'].sum()
    avg = df['net_pnl'].mean()
    wins_total = df.loc[is_winner, 'net_pnl'].sum()
    losses_total = abs(df.loc[~is_winner, 'net_pnl'].sum())
    pf = wins_total / losses_total if losses_total > 0 else float('inf')
    cum = df['net_pnl'].cumsum()
    max_dd = (cum - cum.cummax()).min()

    print(f"  Trades:     {n}")
    print(f"  Win Rate:   {wr:.1f}%")
    print(f"  Net P&L:    ${net:+,.0f}")
    print(f"  Avg/Trade:  ${avg:+,.0f}")
    print(f"  PF:         {pf:.2f}")
    print(f"  Max DD:     ${max_dd:+,.0f}")
    print(f"  Avg Score:  {df['total_score'].mean():.1f}")


def main():
    parser = argparse.ArgumentParser(description='Analyze Balance Signal backtest results')
    parser.add_argument('--csv', default='output/trade_log.csv',
                        help='Path to trade log CSV')
    args = parser.parse_args()

    df = load_trades(args.csv)
    df = parse_metadata(df)

    print_summary(df)
    analyze_per_factor(df)
    analyze_score_threshold(df)
    analyze_mode_breakdown(df)
    analyze_bpr_accuracy(df)
    analyze_acceptance_filter(df)
    analyze_day_type_distribution(df)
    analyze_monthly_pnl(df)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
