"""
=====================================================================================
  EXECUTION PLAYBOOK: 70% Win Rate / 5 MNQ / Lightning x5
  Generated: Feb 18, 2026
  Backtest: 62 sessions (Nov 18, 2025 - Feb 16, 2026)
=====================================================================================

  TARGET: 70%+ Win Rate at 5 MNQ for maximum Lightning account survival
  RESULT: 75.9% WR | 29 trades | $3,861 net | PF 8.86 | Max DD $333

  ACCOUNTS: 5x Tradeify Lightning $150K ($729 each = $3,645 total)
  PROJECTED INCOME (Realistic 70% WR): $3,497/month ($41,965/year)
  SURVIVAL RATE: 100% (all 5 accounts) at Realistic scenario

=====================================================================================
  STRATEGY PORTFOLIO (4 strategies, LONG ONLY)
=====================================================================================

STRATEGY 1: TREND DAY BULL (Dalton Playbook)
─────────────────────────────────────────────
  Backtest: 8 trades, 75% WR, $1,074 net, $134 expectancy

  DAY TYPE: trend_up, super_trend_up, p_day
  SETUP: VWAP_PULLBACK after acceptance above IB High

  ENTRY RULES:
    1. Wait for Initial Balance (first 60 min) to form
    2. Price must accept above IBH (2+ consecutive 5-min closes above IBH)
    3. Wait for pullback to VWAP (price within 0.4x IB range of VWAP)
    4. Entry bar: close above VWAP
    5. Delta > 0 on entry bar (buyers present)
    6. Pre-entry momentum: sum of last 10 bars delta > -500
    7. Order flow quality gate: 2 of 3 must be true:
       - delta_percentile >= 60th
       - imbalance_ratio > 1.0
       - volume_spike >= 1.0
    8. Trend bull confidence >= 0.375
    9. Entry window: 10:30 AM - 1:00 PM ET

  STOP: VWAP - (0.4 x IB range), minimum 15 pts
  TARGET: entry + (2.0 x IB range) for trend days, 1.5x for p_day
  TRAIL: After 1 PM, trail to breakeven if profit > 0.3x IB range

  PYRAMID (optional, on confirmed trend days only):
    - Max 2 adds, cooldown between entries
    - Pyramid on EMA5 pullback or VWAP retouch
    - Entry before 11:30 AM only for pyramids

─────────────────────────────────────────────

STRATEGY 2: P-DAY (Skewed Balance, LONG ONLY)
─────────────────────────────────────────────
  Backtest: 8 trades, 75% WR, $992 net, $124 expectancy

  DAY TYPE: p_day (bullish skew, short covering)
  SETUP: P_DAY_VWAP_LONG after directional acceptance

  ENTRY RULES:
    1. Wait for directional acceptance: 2+ consecutive closes above IBH
    2. This confirms P-shape (bullish skew)
    3. Wait for pullback to VWAP (within 0.4x IB range)
    4. Entry bar: close above VWAP with delta > 0
    5. Pre-entry momentum: sum of last 10 bars delta > -500
    6. Order flow quality: 2 of 3 (delta_pctl>=60, imb>1.0, vol>=1.0)
    7. Entry window: 10:30 AM - 1:00 PM ET
    8. One entry per session maximum

  STOP: VWAP - (0.4 x IB range), minimum 15 pts
  TARGET: entry + (1.5 x IB range)
  TRAIL: Breakeven after 1 PM if profit > 0.3x IB range

  NOTE: SHORTS DISABLED — NQ long bias makes P-Day shorts 0-22% WR

─────────────────────────────────────────────

STRATEGY 3: B-DAY IBL FADE (Balance Day, LONG ONLY)
─────────────────────────────────────────────
  Backtest: 4 trades, 100% WR, $1,486 net, $372 expectancy

  DAY TYPE: b_day (narrow IB, no extension, true trapped balance)
  SETUP: B_DAY_IBL_FADE — fade at IB Low

  ENTRY RULES:
    1. Day classified as b_day with confidence >= 0.5
    2. Trend strength must be "weak" (no IB extension)
    3. IB range < 400 pts (extremely wide IB = not true balance)
    4. Price touches or wicks below IBL
    5. Entry bar: close back above IBL (rejection confirmed)
    6. Delta > 0 (buyers stepping in at low)
    7. Quality score >= 2 from:
       - FVG/IFVG bull zone present
       - Delta rejection (delta > 0)
       - Multi-touch (2+ tests of IBL)
       - Volume spike > 1.3x
    8. Risk:Reward check: risk/reward must be < 2.5
    9. Entry window: before 2:00 PM ET
    10. One entry per session maximum

  STOP: IBL - (0.10 x IB range)
  TARGET: IB midpoint (VWAP/POC mean reversion)
  TRAIL: Breakeven after 1 PM if profit > 0.3x IB range

  NOTE: SHORTS (IBH FADE) DISABLED — 0-22% WR across all tests on NQ

─────────────────────────────────────────────

STRATEGY 4: MEAN REVERSION VWAP (LONG ONLY)
─────────────────────────────────────────────
  Backtest: 9 trades, 66.7% WR, $309 net, $34 expectancy

  DAY TYPE: b_day, neutral, p_day (balance/range days)
  SETUP: VWAP_MEAN_REVERT_LONG — fade deviation below VWAP

  ENTRY RULES:
    1. Day type must be b_day, neutral, or p_day
    2. IB range >= 15 pts (skip tiny IB sessions)
    3. Price deviation from VWAP >= 0.6x IB range (meaningful deviation)
    4. Price BELOW VWAP (for long entry)
    5. RSI14 < 35 (oversold condition)
    6. Volume_spike < 1.2 (exhaustion, not continuation)
    7. Delta > 0 (buyers stepping in at low)
    8. Candle reversal: close > open (green/bullish bar)
    9. Risk:Reward check: reward >= 0.8x risk
    10. Entry window: 11:00 AM - 2:30 PM ET
    11. Max 2 entries per session, 15-bar cooldown

  STOP: Below session low or bar low - (0.25 x IB range), min 12 pts
  TARGET: VWAP (the mean — expect full reversion)
  TRAIL: Breakeven at 50% distance to VWAP

  NOTE: SHORTS DISABLED via regime filter (LONG only)
  Short mean reversion was 37.5% WR on NQ

─────────────────────────────────────────────

=====================================================================================
  REGIME FILTER (applied to all strategies)
=====================================================================================

  FILTER: LONG ONLY — Block ALL short signals
  IMPLEMENTATION: SimpleRegimeFilter(
      longs_in_bull=True,   longs_in_bear=True,
      shorts_in_bull=False,  shorts_in_bear=False,
  )

  RATIONALE: NQ has extreme long bias. Over 62 sessions:
    - Long strategies: 57-100% WR
    - Short strategies: 0-30% WR
    - Even in 61% bear regime (EMA20 < EMA50), longs outperform shorts

=====================================================================================
  POSITION SIZING & RISK MANAGEMENT
=====================================================================================

  MAX CONTRACTS: 5 MNQ per trade (HARD LIMIT)
  RISK PER TRADE: $400 max
  CONTRACT SIZING: risk_per_trade / (stop_distance_pts x $2/pt/MNQ)
    Example: $400 / (30 pts x $2) = 6.67 → capped at 5 MNQ

  DAILY LOSS LIMIT: -$750 (self-imposed, well under Lightning's $3,750)
  MAX TRADES PER DAY: 2 (from backtest: avg 1.5/active day)

  ACCOUNT PROTECTION:
    - Lightning drawdown: $4,500 EOD trailing
    - Max daily loss from backtest: -$327 (at 5 MNQ)
    - Drawdown buffer: $4,500 / $327 = 13.7x daily max loss
    - Need 13+ consecutive worst days to blow account

=====================================================================================
  DAILY EXECUTION ROUTINE
=====================================================================================

  PRE-MARKET (8:30-9:30 AM ET):
    1. Check prior session: was it bullish or bearish?
    2. Check EMA20/EMA50: bull or bear regime?
    3. Note any overnight levels (PDH/PDL)
    4. Decision: trade or sit today based on regime + confidence

  INITIAL BALANCE (9:30-10:30 AM ET):
    1. Record IB High and IB Low at 10:30
    2. Calculate IB range
    3. Classify day type from extension behavior
    4. DO NOT TRADE during IB formation

  POST-IB TRADING (10:30 AM - 2:30 PM ET):
    1. Watch for acceptance: 2+ closes above IBH (bullish) or below IBL
    2. If accepted above IBH → watch for VWAP pullback (Trend Bull / P-Day)
    3. If IB holds (b_day) → watch for IBL fade (B-Day)
    4. If balance day → watch for VWAP deviation (Mean Reversion)
    5. Max 2 trades per day

  PM SESSION (1:00-3:00 PM ET):
    1. Trail winning positions to breakeven
    2. NO new entries after 2:30 PM
    3. Watch for VWAP breach exit on non-trend positions

  CLOSE (3:00-4:00 PM ET):
    1. Close all positions by EOD
    2. EOD exits have 100% WR historically (winners drift all day)

=====================================================================================
  CONSISTENCY RULE COMPLIANCE (Lightning)
=====================================================================================

  RULE: No single day can exceed 20% of total profit (1st payout)

  FROM BACKTEST:
    - Best single day: $633 (14.5% of total $4,352 profit)
    - PASSES 20% consistency rule

  STRATEGY:
    - At 5 MNQ, max possible daily profit ≈ $600-800
    - Need 5+ winning days to spread profit (backtest had 14 win days)
    - Win day rate: 73.7% → consistency naturally maintained
    - If close to threshold: skip remaining trades for the day

  PAYOUT CADENCE:
    - Lightning requires 7 trading days between payouts
    - Expect ~5-7 payouts per year per account
    - With 5 accounts: 25-35 payouts per year

=====================================================================================
  MONTE CARLO PROJECTIONS (10,000 simulations each)
=====================================================================================

  BACKTEST-EXACT (75.9% WR):
    Single account: $13,624/year, 99.7% survival
    5-account portfolio: $64,475/year ($5,373/month), 100% profitable

  REALISTIC LIVE (70% WR, degraded fills):
    Single account: $9,122/year, 100% survival
    5-account portfolio: $41,965/year ($3,497/month), 100% profitable

  CONSERVATIVE (65% WR):
    Single account: $6,737/year, 100% survival
    5-account portfolio: $30,041/year ($2,503/month), 100% profitable

  STRESS TEST (60% WR):
    Single account: $4,055/year, 99.9% survival
    5-account portfolio: $16,632/year ($1,386/month), 100% profitable

  BREAKEVEN: System remains profitable down to ~55% WR

=====================================================================================
  KEY INSIGHTS FROM BACKTEST
=====================================================================================

  1. LONG ONLY IS NON-NEGOTIABLE ON NQ
     Every short strategy tested showed 0-30% WR. Even with regime
     filter (shorts only in bear regime), shorts underperform.
     The data period was -0.9% net, 61% bear regime — and longs STILL won.

  2. EOD EXITS ARE YOUR BEST FRIEND
     100% WR on EOD exits. Winners drift in your favor all day.
     Let positions run to close. Don't take profit early.

  3. STOP EXITS ARE YOUR ENEMY
     0% WR on stop exits. But stops are tiny ($5-$80 avg).
     The key is getting stopped out CHEAPLY, not avoiding stops entirely.

  4. VWAP PULLBACK IS THE HIGHEST-EDGE ENTRY
     Both Trend Bull and P-Day use VWAP pullback as primary entry.
     VWAP is the institutional mean reversion level.
     Order flow confirmation (delta > 0, quality >= 2) filters false pullbacks.

  5. MEAN REVERSION ADDS TRADE VOLUME WITHOUT KILLING WR
     MeanRevVWAP adds 9 trades at 66.7% WR — lower than core but still
     profitable. This is critical for consistency rule compliance (more win days).

  6. 5 MNQ IS THE SWEET SPOT
     - Small enough: max daily loss only -$327 (13.7x buffer)
     - Large enough: meaningful profits ($133/trade avg, $3,861 over 62 sessions)
     - Copy-tradeable: same signals on all 5 accounts simultaneously

  7. CONSISTENCY RULE PASSES NATURALLY
     With 14 winning days and best day at 14.5% of total,
     the 20% consistency rule is not binding.

=====================================================================================
  WHAT NOT TO DO
=====================================================================================

  - DO NOT short NQ. Period. Even in bear regime.
  - DO NOT increase beyond 5 MNQ (diminishing returns, increased risk)
  - DO NOT trade during IB formation (9:30-10:30 AM)
  - DO NOT take profit early — let positions run to EOD
  - DO NOT add EMA Trend Follow or ORB strategies (drops WR below 70%)
  - DO NOT chase entries without order flow confirmation
  - DO NOT force trades on low-confidence days
  - DO NOT skip the pre-entry momentum check (10-bar delta > -500)

=====================================================================================
  COPY TRADING SETUP
=====================================================================================

  All 5 Lightning accounts trade the same signals simultaneously:
    1. Configure NinjaTrader or TradingView with identical strategy parameters
    2. Use order flow data from primary account for signal generation
    3. Copy entry/stop/target to all 5 accounts
    4. All accounts use 5 MNQ per trade
    5. Total exposure: 25 MNQ across 5 accounts
    6. Payout staggered across accounts (7-day minimum between payouts)
    7. If one account approaches drawdown limit, reduce to 3 MNQ on that account

  Monthly Payout Target per Account:
    - Realistic: $760/month ($9,122/year)
    - Conservative: $561/month ($6,737/year)

  Total Monthly Target (5 accounts):
    - Realistic: $3,497/month after costs
    - Conservative: $2,503/month after costs

=====================================================================================
"""

# Print the playbook summary when run
if __name__ == '__main__':
    print(__doc__)
