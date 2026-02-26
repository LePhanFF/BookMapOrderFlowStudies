"""Prop firm scaling analysis: copy trade across multiple firms and account sizes."""
import pandas as pd
import numpy as np

df = pd.read_csv('output/trade_log.csv')
core = df[df['strategy_name'].isin(['Opening Range Rev', 'Edge Fade'])].copy()

# Baseline stats (1 MNQ)
core['date'] = pd.to_datetime(core['entry_time']).dt.date
daily_pnl = core.groupby('date')['net_pnl'].sum()
equity = core['net_pnl'].cumsum()
peak = equity.cummax()
dd = equity - peak

print("=" * 80)
print("PROP FIRM SCALING ANALYSIS -- OR Rev + Edge Fade (149 trades/year)")
print("=" * 80)
print()

# Prop firm rules (as of early 2026, approximate)
firms = [
    {
        'name': 'Apex',
        'size': '$50K',
        'target': 3000,
        'trail_dd': 2500,  # EOD trailing
        'max_contracts_mnq': 10,  # ~2 NQ
        'max_contracts_nq': 2,
        'monthly_fee': 147,
        'activation_fee': 85,
        'payout_split': 0.90,  # 90% after first $25K
        'notes': 'EOD trailing DD, popular for scaling',
    },
    {
        'name': 'Tradeify',
        'size': '$50K',
        'target': 3000,
        'trail_dd': 2000,
        'max_contracts_mnq': 5,
        'max_contracts_nq': 1,
        'monthly_fee': 0,  # one-time eval
        'activation_fee': 0,
        'payout_split': 0.80,
        'notes': 'Select Flex, EOD trailing DD',
    },
    {
        'name': 'Tradeify',
        'size': '$150K',
        'target': 9000,
        'trail_dd': 4500,
        'max_contracts_mnq': 15,
        'max_contracts_nq': 3,
        'monthly_fee': 0,
        'activation_fee': 0,
        'payout_split': 0.80,
        'notes': 'Select Flex, EOD trailing DD',
    },
    {
        'name': 'TopStep',
        'size': '$50K',
        'target': 3000,
        'trail_dd': 2000,
        'max_contracts_mnq': 5,
        'max_contracts_nq': 1,
        'monthly_fee': 165,
        'activation_fee': 149,
        'payout_split': 0.90,
        'notes': 'EOD trailing, Trading Combine',
    },
    {
        'name': 'TopStep',
        'size': '$150K',
        'target': 9000,
        'trail_dd': 4500,
        'max_contracts_mnq': 15,
        'max_contracts_nq': 3,
        'monthly_fee': 375,
        'activation_fee': 149,
        'payout_split': 0.90,
        'notes': 'EOD trailing, Trading Combine',
    },
]

print("BASELINE (1 MNQ, OR Rev + Edge Fade)")
print(f"  Trades: {len(core)}")
print(f"  Win Rate: {core['is_winner'].mean()*100:.1f}%")
print(f"  Net P&L: ${core['net_pnl'].sum():,.0f}")
print(f"  Max DD: ${abs(dd.min()):,.0f}")
print(f"  Worst Day: ${daily_pnl.min():,.0f}")
print(f"  Max Consec Losses: 6")
print(f"  Worst Streak (6L): ${core[core['is_winner']==0]['net_pnl'].mean()*6:,.0f}")
print()

print("=" * 80)
print("SCENARIO ANALYSIS BY ACCOUNT SIZE")
print("=" * 80)

for contracts_mnq, label in [(1, '1 MNQ'), (2, '2 MNQ'), (3, '3 MNQ'), (5, '5 MNQ (1 NQ)'), (10, '10 MNQ (2 NQ)')]:
    scale = contracts_mnq
    annual_pnl = core['net_pnl'].sum() * scale
    max_dd = abs(dd.min()) * scale
    worst_day = daily_pnl.min() * scale
    avg_risk = abs(core['signal_price'] - core['stop_price']).mean() * 2 * scale  # $2/pt * contracts

    print(f"\n--- {label} per trade ---")
    print(f"  Risk/trade: ${avg_risk:,.0f}")
    print(f"  Annual P&L: ${annual_pnl:,.0f} (backtest) | ${annual_pnl*0.7:,.0f} (70% live)")
    print(f"  Max DD: ${max_dd:,.0f}")
    print(f"  Worst Day: ${worst_day:,.0f}")

    # Check against each prop firm
    for firm in firms:
        max_mnq = firm['max_contracts_mnq']
        if contracts_mnq > max_mnq:
            continue
        fits = max_dd < firm['trail_dd']
        safe = max_dd < firm['trail_dd'] * 0.75  # 75% safety margin
        months_to_target = firm['target'] / (annual_pnl / 12) if annual_pnl > 0 else 999

        status = "SAFE" if safe else ("TIGHT" if fits else "BUST")
        print(f"    {firm['name']} {firm['size']}: DD ${max_dd:,.0f} vs ${firm['trail_dd']:,.0f} limit = {status} | "
              f"Target ${firm['target']:,.0f} in ~{months_to_target:.1f} months | "
              f"Payout {firm['payout_split']*100:.0f}%")

print()
print("=" * 80)
print("MULTI-ACCOUNT COPY TRADE SCENARIOS")
print("=" * 80)

scenarios = [
    ('5x Apex $50K @ 2 MNQ', 5, 'Apex', 2, 2500, 3000, 0.90, 147),
    ('5x Tradeify $50K @ 1 MNQ', 5, 'Tradeify', 1, 2000, 3000, 0.80, 0),
    ('5x Tradeify $150K @ 2 MNQ', 5, 'Tradeify', 2, 4500, 9000, 0.80, 0),
    ('10x Apex $50K @ 2 MNQ', 10, 'Apex', 2, 2500, 3000, 0.90, 147),
    ('20x Apex $50K @ 1 MNQ', 20, 'Apex', 1, 2500, 3000, 0.90, 147),
    ('20x Apex $50K @ 2 MNQ', 20, 'Apex', 2, 2500, 3000, 0.90, 147),
    ('5x TopStep $150K @ 2 MNQ', 5, 'TopStep', 2, 4500, 9000, 0.90, 375),
    ('Mixed: 5 Apex + 5 Tradeify $50K @ 1 MNQ', 10, 'Mixed', 1, 2000, 3000, 0.85, 74),
]

print()
for name, n_accts, firm, mnq, trail_dd, target, split, monthly_fee in scenarios:
    scale = mnq
    annual_pnl_per = core['net_pnl'].sum() * scale
    max_dd_per = abs(dd.min()) * scale
    worst_day_per = abs(daily_pnl.min()) * scale

    # Live discount
    live_pnl_per = annual_pnl_per * 0.7

    # After payout split
    payout_per = live_pnl_per * split
    total_payout = payout_per * n_accts

    # Monthly fees
    annual_fees = monthly_fee * 12 * n_accts

    # Net after fees and split
    net_total = total_payout - annual_fees

    # Safety
    dd_pct = max_dd_per / trail_dd * 100
    safe = "SAFE" if dd_pct < 75 else ("TIGHT" if dd_pct < 100 else "BUST")

    # Months to pass target per account
    monthly_pnl = live_pnl_per / 12
    months_to_pass = target / monthly_pnl if monthly_pnl > 0 else 999

    print(f"  {name}")
    print(f"    Per account: ${live_pnl_per:,.0f}/yr (70% live) | DD ${max_dd_per:,.0f} vs ${trail_dd:,.0f} = {safe} ({dd_pct:.0f}%)")
    print(f"    Months to target: {months_to_pass:.1f} months to hit ${target:,.0f}")
    print(f"    After {split*100:.0f}% split: ${payout_per:,.0f}/acct/yr")
    print(f"    Annual fees: ${annual_fees:,.0f} ({n_accts} accts x ${monthly_fee}/mo)")
    print(f"    TOTAL NET: ${net_total:,.0f}/year across {n_accts} accounts")
    print(f"    Worst day (all accts): ${worst_day_per:,.0f} per account (simultaneous)")
    print()

print("=" * 80)
print("$50K ACCOUNT RISK WARNING")
print("=" * 80)
print()
print("At 1 MNQ on a $50K account:")
print(f"  Max DD: ${abs(dd.min()):,.0f} vs typical $2,000-$2,500 trailing DD")
print(f"  DD uses {abs(dd.min())/2000*100:.0f}%-{abs(dd.min())/2500*100:.0f}% of the limit")
print(f"  Worst day: ${abs(daily_pnl.min()):,.0f}")
print()
print("At 2 MNQ on a $50K account:")
print(f"  Max DD: ${abs(dd.min())*2:,.0f} vs typical $2,000-$2,500 trailing DD")
print(f"  DD uses {abs(dd.min())*2/2000*100:.0f}%-{abs(dd.min())*2/2500*100:.0f}% of the limit")
print(f"  *** THIS WILL BLOW MOST $50K ACCOUNTS ***")
print()
print("VERDICT: $50K accounts = 1 MNQ ONLY. For 2+ MNQ, use $150K accounts.")
