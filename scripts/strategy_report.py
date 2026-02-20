"""Strategy-by-strategy expectancy and drawdown report."""
import pandas as pd
import numpy as np

# Load the backtest trade log
df = pd.read_csv('output/trade_log.csv')
print(f'Total trades: {len(df)}')
print(f'Strategies: {df["strategy_name"].value_counts().to_dict()}')
print(f'Setup types: {df["setup_type"].value_counts().to_dict()}')
print()

def analyze_strategy(sub, name):
    sub = sub.sort_values('entry_time').reset_index(drop=True)

    wins = sub[sub['is_winner'] == 1]
    losses = sub[sub['is_winner'] == 0]

    n = len(sub)
    wr = len(wins) / n * 100

    avg_win = wins['net_pnl'].mean() if len(wins) > 0 else 0
    avg_loss = losses['net_pnl'].mean() if len(losses) > 0 else 0

    # Expectancy = (WR * Avg Win) + ((1-WR) * Avg Loss)
    expectancy = (len(wins)/n * avg_win) + (len(losses)/n * avg_loss)

    # Profit Factor = Gross Wins / |Gross Losses|
    gross_win = wins['net_pnl'].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses['net_pnl'].sum()) if len(losses) > 0 else 0.01
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

    # Win/Loss ratio
    wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

    # Max drawdown (equity curve)
    cumulative = sub['net_pnl'].cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_dd = drawdown.min()

    # Recovery: how many trades to recover from max DD
    dd_idx = drawdown.idxmin()
    recovery_trades = 0
    if max_dd < 0:
        for i in range(dd_idx, len(cumulative)):
            if cumulative.iloc[i] >= running_max.iloc[dd_idx]:
                recovery_trades = i - dd_idx
                break
        else:
            recovery_trades = len(cumulative) - dd_idx  # still recovering

    # Max consecutive losses
    is_loss = (sub['is_winner'] == 0).astype(int).tolist()
    consec = []
    count = 0
    for v in is_loss:
        if v == 1:
            count += 1
        else:
            if count > 0:
                consec.append(count)
            count = 0
    if count > 0:
        consec.append(count)
    max_consec_loss = max(consec) if consec else 0

    # Max consecutive wins
    is_win = (sub['is_winner'] == 1).astype(int).tolist()
    consec_w = []
    count = 0
    for v in is_win:
        if v == 1:
            count += 1
        else:
            if count > 0:
                consec_w.append(count)
            count = 0
    if count > 0:
        consec_w.append(count)
    max_consec_win = max(consec_w) if consec_w else 0

    # R-multiple stats
    avg_r_win = wins['r_multiple'].mean() if len(wins) > 0 else 0
    avg_r_loss = losses['r_multiple'].mean() if len(losses) > 0 else 0

    # Largest single win and loss
    largest_win = wins['net_pnl'].max() if len(wins) > 0 else 0
    largest_loss = losses['net_pnl'].min() if len(losses) > 0 else 0

    # Exit reason breakdown
    exit_counts = sub['exit_reason'].value_counts().to_dict()

    # Day type breakdown
    day_types = sub['day_type'].value_counts().to_dict()

    # Expectancy in R-terms
    r_expectancy = (len(wins)/n * avg_r_win) + (len(losses)/n * avg_r_loss)

    # Drawdown-to-expectancy ratio (how many trades to recover from max DD)
    dd_to_exp = abs(max_dd / expectancy) if expectancy > 0 else float('inf')

    sep = '=' * 70
    print(sep)
    print(f'  {name}')
    print(sep)
    print(f'  Trades:             {n}')
    print(f'  Wins / Losses:      {len(wins)}W / {len(losses)}L')
    print(f'  Win Rate:           {wr:.1f}%')
    print()
    print(f'  --- P&L ---')
    print(f'  Avg Win:            ${avg_win:+,.2f}')
    print(f'  Avg Loss:           ${avg_loss:+,.2f}')
    print(f'  Win:Loss Ratio:     {wl_ratio:.2f}:1')
    print(f'  Largest Win:        ${largest_win:+,.2f}')
    print(f'  Largest Loss:       ${largest_loss:+,.2f}')
    print(f'  Net P&L:            ${sub["net_pnl"].sum():+,.2f}')
    print(f'  Profit Factor:      {pf:.2f}')
    print()
    print(f'  --- EXPECTANCY ---')
    print(f'  $ Expectancy:       ${expectancy:+,.2f} per trade')
    print(f'  R Expectancy:       {r_expectancy:+.2f}R per trade')
    print(f'  Avg R (winners):    {avg_r_win:+.2f}R')
    print(f'  Avg R (losers):     {avg_r_loss:+.2f}R')
    print()
    print(f'  --- DRAWDOWN ---')
    print(f'  Max Drawdown:       ${max_dd:+,.2f}')
    print(f'  DD / Expectancy:    {dd_to_exp:.1f} trades to recover')
    print(f'  Max Consec Losses:  {max_consec_loss}')
    print(f'  Max Consec Wins:    {max_consec_win}')
    print(f'  Recovery Trades:    {recovery_trades}')
    print()
    print(f'  --- CONTEXT ---')
    print(f'  Day Types:          {day_types}')
    print(f'  Exit Reasons:       {exit_counts}')
    print()

    return {
        'name': name, 'trades': n, 'wr': wr, 'expectancy': expectancy,
        'max_dd': max_dd, 'dd_to_exp': dd_to_exp, 'pf': pf,
        'net_pnl': sub['net_pnl'].sum(), 'avg_win': avg_win, 'avg_loss': avg_loss,
        'wl_ratio': wl_ratio, 'max_consec_loss': max_consec_loss,
        'r_expectancy': r_expectancy,
    }


# ===== BACKTESTED STRATEGIES (from trade_log.csv) =====
results = []

# Strategy-level breakdown
for strat in sorted(df['strategy_name'].unique()):
    sub = df[df['strategy_name'] == strat].copy()
    r = analyze_strategy(sub, strat)
    results.append(r)

# Also break Edge Fade by setup type
for setup in df[df['strategy_name'] == 'Edge Fade']['setup_type'].unique():
    sub = df[(df['strategy_name'] == 'Edge Fade') & (df['setup_type'] == setup)].copy()
    if len(sub) >= 3:
        r = analyze_strategy(sub, f'Edge Fade ({setup})')
        results.append(r)

# Combined portfolio
print()
r_all = analyze_strategy(df.copy(), 'COMBINED PORTFOLIO')
results.append(r_all)

# ===== RESEARCH STRATEGIES (from diagnostic studies) =====
print()
print('#' * 70)
print('#  RESEARCH STRATEGIES (from diagnostic studies, not in trade_log)')
print('#' * 70)
print()

# IBH Sweep + Failure Fade (from diagnostic_ibh_fade_short.py results)
print('=' * 70)
print('  IBH Sweep + Failure Fade SHORT (b_day only)')
print('=' * 70)
print('  Source: diagnostic_ibh_fade_short.py')
print('  Trades:             4')
print('  Win Rate:           100.0%')
print('  Net:                +291 pts ($582)')
print()
print('  Avg Win:            +72.8 pts ($145.50)')
print('  Avg Loss:           N/A (no losses)')
print()
print('  EXPECTANCY:         $145.50 per trade')
print('  MAX DRAWDOWN:       $0.00 (no losing trades)')
print('  Max Consec Losses:  0')
print()
print('  CAUTION: Only 4 trades. 100% WR is likely sample bias.')
print('  Need more data before deploying live.')
print()

# Bear Acceptance Short (from diagnostic_bearish_day_v2.py)
# trend_down + b_day_bear, ACCEPTANCE_SHORT model
# 11 trades: 7W 4L
bear_wins = [94.8, 55.8, 103.1, 88.5, 176.2, 132.8, 60.5]  # pts
bear_losses = [-53.0, -46.8, -44.8, -69.6]  # pts
# All in points, convert to $ at $2/pt MNQ
bear_wins_d = [x*2 for x in bear_wins]
bear_losses_d = [x*2 for x in bear_losses]
bear_all_d = bear_wins_d + bear_losses_d

n_bear = len(bear_all_d)
wr_bear = len(bear_wins_d) / n_bear * 100
avg_win_bear = np.mean(bear_wins_d)
avg_loss_bear = np.mean(bear_losses_d)
exp_bear = (len(bear_wins_d)/n_bear * avg_win_bear) + (len(bear_losses_d)/n_bear * avg_loss_bear)
pf_bear = sum(bear_wins_d) / abs(sum(bear_losses_d))
wl_bear = abs(avg_win_bear / avg_loss_bear)

# Simulate equity curve in order: chronological from diagnostic output
# dates: 11/20 W, 12/12 W, 12/31 L, 1/2 L, 1/15 L, 1/20 W, 1/28 W, 1/30 L, 2/3 W, 2/4 W, 2/12 W
bear_pnl_seq = [94.8, 55.8, -53.0, -46.8, -44.8, 103.1, 60.5, -69.6, 88.5, 176.2, 132.8]
bear_pnl_seq_d = [x*2 for x in bear_pnl_seq]
bear_cum = np.cumsum(bear_pnl_seq_d)
bear_runmax = np.maximum.accumulate(bear_cum)
bear_dd = bear_cum - bear_runmax
bear_max_dd = bear_dd.min()

# Max consec loss
bear_wl = [1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1]
consec_l = []
c = 0
for v in bear_wl:
    if v == 0:
        c += 1
    else:
        if c > 0: consec_l.append(c)
        c = 0
if c > 0: consec_l.append(c)
max_cl_bear = max(consec_l) if consec_l else 0

# Recovery
dd_idx_bear = np.argmin(bear_dd)
recov_bear = 0
for i in range(dd_idx_bear, len(bear_cum)):
    if bear_cum[i] >= bear_runmax[dd_idx_bear]:
        recov_bear = i - dd_idx_bear
        break

dd_to_exp_bear = abs(bear_max_dd / exp_bear) if exp_bear > 0 else float('inf')

print('=' * 70)
print('  Bear Acceptance Short (trend_down + b_day_bear)')
print('=' * 70)
print('  Source: diagnostic_bearish_day_v2.py')
print(f'  Trades:             {n_bear}')
print(f'  Wins / Losses:      {len(bear_wins_d)}W / {len(bear_losses_d)}L')
print(f'  Win Rate:           {wr_bear:.1f}%')
print()
print(f'  --- P&L ---')
print(f'  Avg Win:            ${avg_win_bear:+,.2f}')
print(f'  Avg Loss:           ${avg_loss_bear:+,.2f}')
print(f'  Win:Loss Ratio:     {wl_bear:.2f}:1')
print(f'  Net P&L:            ${sum(bear_pnl_seq_d):+,.2f}')
print(f'  Profit Factor:      {pf_bear:.2f}')
print()
print(f'  --- EXPECTANCY ---')
print(f'  $ Expectancy:       ${exp_bear:+,.2f} per trade')
print()
print(f'  --- DRAWDOWN ---')
print(f'  Max Drawdown:       ${bear_max_dd:+,.2f}')
print(f'  DD / Expectancy:    {dd_to_exp_bear:.1f} trades to recover')
print(f'  Max Consec Losses:  {max_cl_bear}')
print(f'  Recovery Trades:    {recov_bear}')
print()
print(f'  Equity curve (cumulative $):')
for i, (pnl, cum) in enumerate(zip(bear_pnl_seq_d, bear_cum)):
    dd_val = bear_dd[i]
    print(f'    Trade {i+1:2d}: {pnl:+8.2f} -> cum ${cum:+9.2f}  dd: ${dd_val:+.2f}')
print()


# ===== MASTER COMPARISON TABLE =====
print()
print('#' * 70)
print('#  MASTER STRATEGY COMPARISON')
print('#' * 70)
print()

# Add research strategies to results
results.append({
    'name': 'IBH Sweep+Fail SHORT*', 'trades': 4, 'wr': 100.0,
    'expectancy': 145.50, 'max_dd': 0, 'dd_to_exp': 0, 'pf': float('inf'),
    'net_pnl': 582, 'avg_win': 145.50, 'avg_loss': 0,
    'wl_ratio': float('inf'), 'max_consec_loss': 0, 'r_expectancy': 0,
})
results.append({
    'name': 'Bear Accept SHORT*', 'trades': n_bear, 'wr': wr_bear,
    'expectancy': exp_bear, 'max_dd': bear_max_dd, 'dd_to_exp': dd_to_exp_bear,
    'pf': pf_bear, 'net_pnl': sum(bear_pnl_seq_d), 'avg_win': avg_win_bear,
    'avg_loss': avg_loss_bear, 'wl_ratio': wl_bear, 'max_consec_loss': max_cl_bear,
    'r_expectancy': 0,
})

# Sort by expectancy descending
sorted_results = sorted(results, key=lambda x: x['expectancy'], reverse=True)

header = f'{"Strategy":<30s} {"Trades":>6s} {"WR":>5s} {"Exp/Trade":>10s} {"Net P&L":>10s} {"MaxDD":>10s} {"DD/Exp":>6s} {"PF":>6s} {"W:L":>5s} {"MaxCL":>5s}'
print(header)
print('-' * len(header))

for r in sorted_results:
    # Skip sub-breakdowns and combined for the main table
    if 'COMBINED' in r['name'] or '(' in r['name']:
        continue
    pf_str = f'{r["pf"]:.2f}' if r['pf'] < 999 else 'INF'
    wl_str = f'{r["wl_ratio"]:.1f}' if r['wl_ratio'] < 999 else 'INF'
    dd_str = f'{r["dd_to_exp"]:.1f}' if r['dd_to_exp'] < 999 else 'N/A'
    print(f'{r["name"]:<30s} {r["trades"]:>6d} {r["wr"]:>4.0f}% ${r["expectancy"]:>+8.2f} ${r["net_pnl"]:>+9,.0f} ${r["max_dd"]:>+9,.0f} {dd_str:>6s} {pf_str:>6s} {wl_str:>5s} {r["max_consec_loss"]:>5d}')

# Print combined
print('-' * len(header))
r = r_all
pf_str = f'{r["pf"]:.2f}'
wl_str = f'{r["wl_ratio"]:.1f}'
dd_str = f'{r["dd_to_exp"]:.1f}'
print(f'{"COMBINED (backtested)":<30s} {r["trades"]:>6d} {r["wr"]:>4.0f}% ${r["expectancy"]:>+8.2f} ${r["net_pnl"]:>+9,.0f} ${r["max_dd"]:>+9,.0f} {dd_str:>6s} {pf_str:>6s} {wl_str:>5s} {r["max_consec_loss"]:>5d}')

print()
print('* = Research study (not yet in backtest engine)')
print()
print('KEY METRICS EXPLAINED:')
print('  Exp/Trade = Expected $ per trade (WR * AvgWin + LR * AvgLoss)')
print('  DD/Exp    = Trades needed to recover from max drawdown')
print('  PF        = Profit Factor (Gross Wins / Gross Losses)')
print('  W:L       = Win:Loss ratio (Avg Win / |Avg Loss|)')
print('  MaxCL     = Max consecutive losses')
