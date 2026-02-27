"""Quick calc: 4 NQ contract projections from 1 MNQ backtest."""
import pandas as pd
import numpy as np

df = pd.read_csv('output/trade_log.csv')
core = df[df['strategy_name'].isin(['Opening Range Rev', 'Edge Fade'])].copy()

# MNQ = $2/pt, NQ = $20/pt. So 4 NQ = 40 MNQ equivalent
scale = 40

print('=== 4 NQ Contracts (= 40 MNQ equivalent) ===')
print(f'Trades: {len(core)} per year')
print()

# Per-trade stats
print('--- Per Trade ---')
avg_win = core[core['is_winner']==1]['net_pnl'].mean() * scale
avg_loss = core[core['is_winner']==0]['net_pnl'].mean() * scale
avg_risk_pts = abs(core['signal_price'] - core['stop_price']).mean()
print(f'Avg risk: {avg_risk_pts:.0f} pts x $20/pt x 4 = ${avg_risk_pts*20*4:,.0f} per trade')
print(f'Avg win: ${avg_win:,.0f}')
print(f'Avg loss: ${avg_loss:,.0f}')
print(f'Expectancy: ${core["net_pnl"].mean()*scale:,.0f}/trade')
print()

# Annual
annual = core['net_pnl'].sum() * scale
print('--- Annual P&L ---')
print(f'Backtest (100%): ${annual:,.0f}')
print(f'70% live factor:  ${annual*0.7:,.0f}')
print(f'50% live factor:  ${annual*0.5:,.0f}')
print()

# Monthly
core['month'] = pd.to_datetime(core['entry_time']).dt.to_period('M')
monthly = core.groupby('month')['net_pnl'].sum() * scale
print('--- Monthly P&L (4 NQ) ---')
cumul = 0
for m, pnl in monthly.items():
    cumul += pnl
    print(f'  {m} | ${pnl:>10,.0f} | cum ${cumul:>10,.0f}')
print(f'  Avg month: ${monthly.mean():,.0f}')
print(f'  Best month: ${monthly.max():,.0f} ({monthly.idxmax()})')
print(f'  Worst month: ${monthly.min():,.0f} ({monthly.idxmin()})')
print()

# Drawdown
equity = core['net_pnl'].cumsum() * scale
peak = equity.cummax()
dd = equity - peak
print('--- Risk (4 NQ) ---')
print(f'Max Drawdown: ${dd.min():,.0f}')
print(f'Max DD as % of $1M: {abs(dd.min())/1000000*100:.1f}%')

# Worst day
core['date'] = pd.to_datetime(core['entry_time']).dt.date
daily = core.groupby('date')['net_pnl'].sum() * scale
print(f'Worst day: ${daily.min():,.0f} ({daily.idxmin()})')
print(f'Best day: ${daily.max():,.0f} ({daily.idxmax()})')
print()

# Risk per trade as % of $1M
print('--- Risk Context ---')
print(f'Risk per trade: ${avg_risk_pts*20*4:,.0f} = {avg_risk_pts*20*4/1000000*100:.2f}% of $1M')
print(f'Max consecutive losses: 6')
print(f'Worst streak cost: ${avg_loss*6:,.0f} (6 consecutive losers)')
