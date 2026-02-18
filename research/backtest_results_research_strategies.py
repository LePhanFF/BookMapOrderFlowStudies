"""
Backtest Results: Research Strategies vs Core Strategies
========================================================

Run Date: Feb 2026
Data Period: Nov 18, 2025 - Feb 16, 2026 (62 sessions)
Instrument: MNQ ($2/pt, $0.62/side commission, 1 tick slippage)
Account: $150,000 | Risk: $400/trade | Max: 30 contracts
Filters: NONE (raw strategy performance)

RESULTS SUMMARY
===============

┌─────────────────────────────────────────────────────────────────────────────────┐
│                         STRATEGY COMPARISON TABLE                              │
├────────────────────┬───────┬───────┬────────┬──────────┬────────┬──────┬───────┤
│ Strategy           │Trades │ WR%   │  PF    │  Net P&L │ Expect │MaxDD │Sharpe │
├────────────────────┼───────┼───────┼────────┼──────────┼────────┼──────┼───────┤
│ CORE ONLY          │   20  │ 80.0% │ 405.5  │ $4,434   │ $222   │ 0.0% │ 18.81 │
│ RESEARCH ONLY      │   51  │ 39.2% │  1.29  │ $2,460   │ $48    │ 2.7% │  2.02 │
│ COMBINED (8 strat) │   63  │ 47.6% │  1.57  │ $4,865   │ $77    │ 2.6% │  3.60 │
├────────────────────┴───────┴───────┴────────┴──────────┴────────┴──────┴───────┤
│ COMBINED > CORE: +$432 more P&L, 3x more trades, more diversification         │
│ COMBINED > RESEARCH: +$2,405 more P&L, higher win rate, better Sharpe          │
└───────────────────────────────────────────────────────────────────────────────────┘

INDIVIDUAL STRATEGY BREAKDOWN (Combined Run)
=============================================

Strategy              Trades   WR%      Net P&L    Expectancy   Verdict
───────────────────── ──────  ──────  ──────────  ──────────   ────────────
Trend Day Bull            5   80.0%   $  792.82   $158.56/t    KEEP ✓
P-Day                     5   80.0%   $  792.82   $158.56/t    KEEP ✓
B-Day                     2  100.0%   $  819.12   $409.56/t    KEEP ✓
EMA Trend Follow         36   36.1%   $2,945.71   $ 81.83/t    KEEP ✓ (volume driver)
Liquidity Sweep           2   50.0%   $  137.99   $ 69.00/t    KEEP ✓ (needs more data)
Mean Reversion VWAP       1  100.0%   $  109.76   $109.76/t    KEEP ✓ (needs more data)
ORB VWAP Breakout        12   41.7%  $  -733.02  -$ 61.09/t    REVIEW ✗ (negative)


SETUP TYPE ANALYSIS
===================

Setup                  Trades   WR%      Net P&L    Expect      Verdict
────────────────────── ──────  ──────  ──────────  ──────────   ────────────
EMA_PULLBACK_LONG          36  36.1%   $2,945.71   $ 81.83      BEST RESEARCH SETUP
B_DAY_IBL_FADE              2 100.0%   $  819.12   $409.56      BEST OVERALL (low N)
P_DAY_VWAP_LONG             5  80.0%   $  792.82   $158.56      STRONG
VWAP_PULLBACK               5  80.0%   $  792.82   $158.56      STRONG
SWEEP_IBL_LONG              1 100.0%   $  529.44   $529.44      PROMISING (low N)
VWAP_MEAN_REVERT_LONG       1 100.0%   $  109.76   $109.76      PROMISING (low N)
ORB_BREAKOUT_LONG          12  41.7%  $  -733.02  -$ 61.09      NEEDS WORK
SWEEP_PDL_LONG              1    0%   $  -391.45  -$391.45      SKIP PDL SWEEPS


KEY FINDINGS
============

1. SHORTS ARE POISON ON NQ
   - Before disabling shorts: -$1,581 total P&L
   - After disabling shorts:  +$2,460 total P&L
   - Delta: +$4,041 improvement
   - EMA Short: 17.4% WR → disabled
   - ORB Short: 30.0% WR → disabled
   - Mean Reversion Short: 37.5% WR → disabled
   - This is consistent with existing codebase findings (NQ long bias)

2. EMA TREND FOLLOW IS THE VOLUME DRIVER
   - 36 trades (57% of all research trades)
   - 36.1% WR but 1.99 R:R → positive expectancy
   - $2,946 profit (60% of combined total)
   - Winners average 244 bars held (4+ hours) → not scalps
   - Fills the gap between core strategy trades (only 20 trades)

3. COMBINED PORTFOLIO IS SUPERIOR
   - 63 trades vs 20 (core only) — 3x more opportunities
   - $4,865 vs $4,434 — +$431 more profit
   - Better diversification across day types
   - More data points for Monte Carlo simulation
   - 47.6% WR with 1.72 R:R = solid positive expectancy

4. ORB VWAP NEEDS REFINEMENT
   - 41.7% WR but negative P&L (-$733)
   - Stop too wide (IB midpoint still too far)
   - Only 1 of 12 trades hit target — most exit at EOD or stop
   - Consider: tighter stop at VWAP, or require volume_spike >= 2.0

5. EXIT DISTRIBUTION SHOWS HOLDING ISSUE
   - 46% STOP, 41% EOD, 11% VWAP_BREACH, 2% TARGET
   - Very few trades hit target → targets may be too ambitious
   - Consider: tighter targets (1.0x IB instead of 1.5-2.0x)
   - Or: time-based exits (close after 2 hours if in profit)

6. DAY TYPE EDGE
   - B-Day: 42.9% WR, $5,057 profit → best day type
   - P-Day: 53.6% WR, -$192 → breakeven
   - Research strategies add value on B-Days specifically


RECOMMENDATIONS FOR LIVE TRADING
=================================

EVALUATION PHASE (Pass in ~9 days):
  Use: Trend Day Bull + P-Day + B-Day + EMA Trend Follow
  Skip: ORB VWAP (negative expectancy), Shorts (all)
  Size: 31 MNQ, $400 risk/trade
  Window: 10:00-13:00 ET
  Expected: ~47% WR, $77/trade, ~12 trades/week

FUNDED PHASE (Protect drawdown):
  Use: Trend Day Bull + P-Day + B-Day + Liquidity Sweep
  Skip: EMA Trend Follow (too many trades, risk of overtrading)
  Size: 20 MNQ, $300 risk/trade
  Window: 10:00-12:00 ET
  Add: Micro buffer builder (1-5 micros first 5 days)

PARAMETER TUNING NEEDED:
  1. ORB VWAP: tighten stop or require volume_spike >= 2.0
  2. EMA Trend Follow: consider 10:30-12:00 window (vs current all-day)
  3. All strategies: add time-based profit-taking (close winners after 2hrs)
  4. Consistency pacer: cap daily profit at $500 during evaluation
"""

if __name__ == '__main__':
    print(__doc__)
