"""
Evaluation & Payout Strategy Research — Backtestable Implementations

Source: Web research (Feb 2026) on strategies people consistently use to:
  1. Pass prop firm evaluations (50%+ pass rate or better)
  2. Consistently receive payouts from funded accounts

These strategies are designed to plug into the existing BacktestEngine via
the StrategyBase interface using available data columns:
  - OHLCV, delta, CVD, imbalance_ratio, volume_spike, delta_percentile
  - VWAP, EMA20, EMA50, ATR14, RSI14
  - IB high/low/range, day type, trend strength
  - FVG/IFVG/BPR detection from indicators/ict_models.py

RESEARCH SUMMARY
================

Industry Context (2025-2026):
  - Prop firm pass rates: 5-15% industry average
  - Only ~7% of those who pass ever receive a payout
  - Firms with highest pass rates: MyFundedFutures (~25%), Apex
  - Key differentiator: EOD drawdown >> intraday trailing drawdown
  - 90/10 profit split is the new baseline

What Separates Passing Traders From Failing Ones:
  1. Strategy aligned to firm rules (not just "good" strategy)
  2. Documented plan with specific entry/exit rules
  3. Consistent position sizing (losers vary size by 300%+)
  4. Distributed profits across days (consistency rules)
  5. Always use stop losses (3x longer funded retention)
  6. Risk 0.5-1.0% per trade max
  7. Start with micros, scale to minis after building buffer

STRATEGIES CAPTURED (12 Total)
==============================

Category A: Evaluation Passing Strategies (high pass rate)
  1. ORB_VWAP — Opening Range Breakout with VWAP + Volume Confirmation
  2. LIQUIDITY_SWEEP — Stop Hunt Reversal (ICT-style)
  3. MOMENTUM_SCALP — Controlled Momentum Scalping (delta + imbalance)
  4. MEAN_REVERT_VWAP — VWAP Mean Reversion (mid-day)
  5. TREND_FOLLOW_EMA — Trend Following with EMA Crossover
  6. ICT_FVG_OB — ICT Fair Value Gap + Order Block Confluence

Category B: Funded Payout Consistency Strategies (capital preservation)
  7. MICRO_BUFFER — Micro Contract Buffer Builder
  8. PAYOUT_PROTECT — Trailing Drawdown Protection Mode
  9. CONSISTENCY_PACER — Daily Profit Cap / Consistency Rule Compliance
  10. SESSION_WINDOW — Session-Restricted Trading (peak hours only)

Category C: Hybrid Evaluation+Funded Strategies
  11. DUAL_REGIME — Aggressive Eval → Conservative Funded Auto-Switch
  12. MULTI_ACCOUNT — Sequential Account Factory Optimization

Sources:
  - https://tradeify.co/post/futures-trading-strategies-pass-prop-evaluations
  - https://damnpropfirms.com/prop-firms/core-strategy-execution-nq-es-specific/
  - https://www.quantvps.com/blog/prop-firm-statistics
  - https://www.fxempire.com/news/article/how-to-pass-any-prop-firm-evaluation-in-2026
  - https://myfundedfutures.com/blog/prop-firm-evaluation-process-the-ultimate-guide
  - https://phidiaspropfirm.com/education/fair-value-gap
  - https://phidiaspropfirm.com/education/ict-trading-guide
  - https://masterytraderacademy.com/prop-firm-stay-funded-withdrawal-strategy/
  - https://holaprime.com/blogs/trading-tips/futures-strategies-prop-firm-challenge/
  - https://traders.mba/support/order-flow-imbalance-scalping/
  - https://www.mindmathmoney.com/articles/liquidity-sweep-trading-strategy
"""

# This module defines the research strategies as data structures that can be
# iterated by the backtest system or Monte Carlo simulator.

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class StrategyResearch:
    """Captured strategy research entry."""
    id: str
    name: str
    category: str           # 'evaluation', 'payout', 'hybrid'
    source: str             # Primary source URL or community reference
    description: str
    instruments: List[str]  # ['NQ', 'MNQ', 'ES', 'MES']
    timeframe: str          # '1min', '5min', etc.
    session_window: str     # e.g., '09:30-11:00 ET'

    # Entry rules
    entry_rules: List[str]
    entry_indicators: List[str]

    # Exit rules
    exit_rules: List[str]
    stop_method: str
    target_method: str
    risk_reward_ratio: float

    # Risk management
    max_risk_pct: float          # % of account per trade
    max_daily_loss_pct: float    # % of account per day
    max_trades_per_day: int
    position_sizing: str         # Description of sizing approach

    # Performance expectations
    expected_win_rate: float     # 0.0-1.0
    expected_rr: float           # Risk:Reward
    expected_expectancy: float   # Avg R per trade
    pass_rate_estimate: str      # 'high', 'medium', 'low'

    # Prop firm compatibility
    consistency_compliant: bool  # Meets 35-40% consistency rules
    microscalp_compliant: bool   # Meets >10sec hold requirement
    eod_flat: bool               # Closes by 4:59 PM ET

    # Implementation notes
    backtestable: bool           # Can be backtested with current data
    implementation_notes: str
    metadata: Dict = field(default_factory=dict)


# ============================================================================
#  CATEGORY A: EVALUATION PASSING STRATEGIES
# ============================================================================

STRATEGY_01_ORB_VWAP = StrategyResearch(
    id='ORB_VWAP',
    name='Opening Range Breakout + VWAP Confirmation',
    category='evaluation',
    source='https://tradeify.co/post/futures-trading-strategies-pass-prop-evaluations',
    description=(
        'Trades breakouts of the Initial Balance (first 30-60 min range) '
        'confirmed by VWAP direction and volume surge. The most commonly cited '
        'strategy for passing evaluations quickly. Backtested ORB on NQ showed '
        '74.56% win rate and 2.51 profit factor over 114 trades in one study. '
        'Key insight: wait for candle CLOSE beyond IB, not just a wick.'
    ),
    instruments=['NQ', 'MNQ', 'ES', 'MES'],
    timeframe='1min',
    session_window='09:30-12:00 ET',

    entry_rules=[
        '1. Compute IB high/low from first 60 1-min bars (09:30-10:30)',
        '2. Wait for 1-min candle to CLOSE above IBH (long) or below IBL (short)',
        '3. Confirm: price must be above session VWAP for longs, below for shorts',
        '4. Confirm: breakout bar volume >= 1.5x 20-bar average (volume_spike >= 1.5)',
        '5. Confirm: breakout bar close in upper 60% of range (candle strength)',
        '6. Confirm: delta aligns with direction (positive for long, negative for short)',
        '7. Enter at close of confirmation bar',
    ],
    entry_indicators=['IB_HIGH', 'IB_LOW', 'VWAP', 'VOLUME_SPIKE', 'DELTA', 'ATR14'],

    exit_rules=[
        'Target: 1.5x IB range from breakout level',
        'Stop: Just inside opposite IB boundary (IBL for longs, IBH for shorts)',
        'Trail: Move stop to breakeven after 1.0x IB range move',
        'Time exit: Close by 15:30 ET regardless',
    ],
    stop_method='IB boundary + 0.25x IB range buffer',
    target_method='1.5x IB range from entry',
    risk_reward_ratio=2.0,

    max_risk_pct=0.75,
    max_daily_loss_pct=1.5,
    max_trades_per_day=3,
    position_sizing='risk / (stop_distance * point_value)',

    expected_win_rate=0.55,
    expected_rr=2.0,
    expected_expectancy=0.65,
    pass_rate_estimate='high',

    consistency_compliant=True,
    microscalp_compliant=True,
    eod_flat=True,
    backtestable=True,

    implementation_notes=(
        'Directly compatible with existing IB detection in data/features.py. '
        'Volume confirmation uses existing volume_spike column. VWAP filter '
        'uses existing vwap column. This is an enhancement of existing Trend Day '
        'Bull strategy with explicit breakout bar quality checks.'
    ),
)

STRATEGY_02_LIQUIDITY_SWEEP = StrategyResearch(
    id='LIQUIDITY_SWEEP',
    name='Liquidity Sweep / Stop Hunt Reversal',
    category='evaluation',
    source='https://www.mindmathmoney.com/articles/liquidity-sweep-trading-strategy',
    description=(
        'Exploits institutional stop hunting at obvious levels (prior day high/low, '
        'IB extremes, round numbers). Price sweeps beyond a key level to trigger '
        'stops, then reverses. Reported 60-65% win rate with 2:1 R:R. Works '
        'because institutions need to fill large orders at liquidity pools. '
        'The key is waiting for REJECTION after the sweep, not entering during it.'
    ),
    instruments=['NQ', 'MNQ'],
    timeframe='1min',
    session_window='09:30-11:30 ET',

    entry_rules=[
        '1. Identify liquidity zones: prior day high/low, IB high/low, round numbers',
        '2. Wait for price to SWEEP beyond the level (wick past, but closes inside)',
        '3. Confirm rejection: candle closes back inside the prior range',
        '4. Confirm: delta reverses (delta_percentile shifts to opposing direction)',
        '5. Confirm: volume spike on rejection bar (volume_spike >= 1.2)',
        '6. Enter on next bar after rejection candle closes',
        '7. Optional: FVG or order block confluence at sweep level adds confidence',
    ],
    entry_indicators=[
        'PRIOR_DAY_HIGH', 'PRIOR_DAY_LOW', 'IB_HIGH', 'IB_LOW',
        'DELTA', 'DELTA_PERCENTILE', 'VOLUME_SPIKE', 'FVG', 'IFVG',
    ],

    exit_rules=[
        'Target: Opposite liquidity zone (e.g., sweep PDH → target PDL midpoint)',
        'Stop: Beyond the sweep extreme (wick high/low + small buffer)',
        'Trail: Move to breakeven after 1:1 R achieved',
        'Time exit: Close by 15:00 ET',
    ],
    stop_method='Beyond sweep extreme + 0.1x IB range',
    target_method='Next liquidity zone or 2.0x risk distance',
    risk_reward_ratio=2.0,

    max_risk_pct=0.75,
    max_daily_loss_pct=1.5,
    max_trades_per_day=2,
    position_sizing='risk / (stop_distance * point_value)',

    expected_win_rate=0.60,
    expected_rr=2.0,
    expected_expectancy=0.80,
    pass_rate_estimate='high',

    consistency_compliant=True,
    microscalp_compliant=True,
    eod_flat=True,
    backtestable=True,

    implementation_notes=(
        'Requires tracking prior day high/low from session_context["prior_day"]. '
        'Sweep detection: bar high > level but close < level (for upside sweep). '
        'Delta reversal detectable from existing delta columns. FVG confluence '
        'from indicators/ict_models.py. This is a NEW strategy type not in the '
        'current codebase — a reversal/fade setup vs the existing trend-following.'
    ),
)

STRATEGY_03_MOMENTUM_SCALP = StrategyResearch(
    id='MOMENTUM_SCALP',
    name='Controlled Momentum Scalping (Order Flow)',
    category='evaluation',
    source='https://traders.mba/support/order-flow-imbalance-scalping/',
    description=(
        'Scalps 5-20 point moves on NQ using order flow imbalance detection. '
        'Enters when delta imbalance exceeds 85th percentile with volume surge, '
        'exits quickly at predefined targets. This is the EXISTING dual strategy '
        'approach already implemented in the codebase (DualOrderFlow_Evaluation.cs). '
        'Key insight from community: mix scalps with longer holds to meet >10sec rule. '
        'Most successful scalpers on NQ target 10-20 clean points per session.'
    ),
    instruments=['NQ', 'MNQ'],
    timeframe='1min',
    session_window='10:00-13:00 ET',

    entry_rules=[
        '1. Delta percentile >= 85th (strong directional conviction)',
        '2. Volume spike >= 1.5x 20-bar average',
        '3. CVD aligns with direction (cumulative_delta > cumulative_delta_ma for long)',
        '4. VWAP confirmation: price above VWAP for long, below for short',
        '5. Imbalance ratio > 1.2 for longs, < 0.8 for shorts',
        '6. Enter at market on confirmation bar close',
        'Alternative trigger: Imbalance > 85% + Volume > 1.5x + CVD alignment',
    ],
    entry_indicators=[
        'DELTA_PERCENTILE', 'VOLUME_SPIKE', 'CUMULATIVE_DELTA',
        'CUMULATIVE_DELTA_MA', 'VWAP', 'IMBALANCE_RATIO',
    ],

    exit_rules=[
        'Target: 10-20 points (0.5x-1.0x IB range)',
        'Stop: 8-12 points (0.3x-0.5x IB range)',
        'Trail: Move stop to breakeven after 10pt move',
        'Time exit: Close within 15-30 minutes max hold',
    ],
    stop_method='Fixed points or 0.4x IB range',
    target_method='Fixed 10-20 points or 0.5-1.0x IB range',
    risk_reward_ratio=1.5,

    max_risk_pct=0.50,
    max_daily_loss_pct=1.0,
    max_trades_per_day=5,
    position_sizing='31 MNQ (evaluation) / 20 MNQ (funded)',

    expected_win_rate=0.44,
    expected_rr=1.5,
    expected_expectancy=0.22,
    pass_rate_estimate='high',

    consistency_compliant=True,
    microscalp_compliant=True,  # Must hold > 10sec per trade
    eod_flat=True,
    backtestable=True,

    implementation_notes=(
        'This IS the existing DualOrderFlow strategy. Included here for '
        'completeness and to document it as the community-validated approach. '
        'The 44.2% WR with 1.52 pts/trade matches community expectations. '
        'Key optimization: the 10:00-13:00 window is validated by community data.'
    ),
)

STRATEGY_04_MEAN_REVERT_VWAP = StrategyResearch(
    id='MEAN_REVERT_VWAP',
    name='VWAP Mean Reversion (Mid-Day)',
    category='evaluation',
    source='https://damnpropfirms.com/prop-firms/core-strategy-execution-nq-es-specific/',
    description=(
        'Trades mean reversion to VWAP during low-volatility mid-day sessions '
        '(11:00-14:00 ET). When price deviates significantly from VWAP, enter '
        'counter-trend expecting a return to the mean. Works best on balance/B-Day '
        'sessions. Community reports 55-65% win rate on ES, slightly lower on NQ. '
        'Key: avoid trending days — check IB extension < 0.5x before engaging.'
    ),
    instruments=['NQ', 'MNQ', 'ES', 'MES'],
    timeframe='1min',
    session_window='11:00-14:00 ET',

    entry_rules=[
        '1. Day type filter: only trade on b_day or neutral days (IB ext < 0.5x)',
        '2. Price deviation: price is > 0.75x IB range away from VWAP',
        '3. RSI confirmation: RSI14 > 70 (short) or RSI14 < 30 (long)',
        '4. Delta exhaustion: delta_percentile extreme then reversing',
        '5. Volume declining: volume_spike < 1.0 (exhaustion, not continuation)',
        '6. Enter when price starts closing back toward VWAP',
    ],
    entry_indicators=[
        'VWAP', 'RSI14', 'DELTA_PERCENTILE', 'VOLUME_SPIKE',
        'IB_RANGE', 'DAY_TYPE', 'ATR14',
    ],

    exit_rules=[
        'Target: VWAP (or VWAP + small overshoot for momentum)',
        'Stop: Beyond the deviation extreme + 0.25x IB range',
        'Trail: Move to breakeven at 50% of distance to VWAP',
        'Time exit: Close by 15:00 ET',
    ],
    stop_method='Beyond deviation extreme + buffer',
    target_method='Return to VWAP',
    risk_reward_ratio=1.5,

    max_risk_pct=0.50,
    max_daily_loss_pct=1.0,
    max_trades_per_day=3,
    position_sizing='risk / (stop_distance * point_value)',

    expected_win_rate=0.58,
    expected_rr=1.5,
    expected_expectancy=0.45,
    pass_rate_estimate='medium',

    consistency_compliant=True,
    microscalp_compliant=True,
    eod_flat=True,
    backtestable=True,

    implementation_notes=(
        'Enhancement of existing B-Day strategy. Uses existing VWAP column and '
        'RSI14. The day_type filter prevents this from firing on trend days '
        'where mean reversion fails. Mid-day window avoids the high-volatility '
        'open and close. New: RSI exhaustion + declining volume as entry filter.'
    ),
)

STRATEGY_05_TREND_FOLLOW_EMA = StrategyResearch(
    id='TREND_FOLLOW_EMA',
    name='Trend Following with EMA Crossover + Pullback',
    category='evaluation',
    source='https://tradeify.co/post/futures-trading-strategies-pass-prop-evaluations',
    description=(
        'Classic trend-following using EMA20/EMA50 crossover to identify direction, '
        'then entering on pullbacks to the moving average. Works on trend days. '
        'Community consensus: trend following is one of the top 3 strategies for '
        'passing evaluations because it aligns with prop firm trailing drawdown '
        '(you build a buffer early and protect it). Expected 50-55% WR with 2:1 R:R.'
    ),
    instruments=['NQ', 'MNQ', 'ES', 'MES'],
    timeframe='1min',
    session_window='10:00-15:00 ET',

    entry_rules=[
        '1. EMA20 > EMA50 for longs (EMA20 < EMA50 for shorts)',
        '2. Price above VWAP for longs (below for shorts)',
        '3. Wait for pullback: price touches or comes within 0.15x IB of EMA20',
        '4. Pullback confirmation: bar closes back above EMA20',
        '5. Delta positive on pullback recovery bar (buyers stepping in)',
        '6. Volume_spike >= 1.0 on recovery bar',
        '7. Enter at close of recovery bar',
    ],
    entry_indicators=[
        'EMA20', 'EMA50', 'VWAP', 'DELTA', 'VOLUME_SPIKE', 'ATR14',
    ],

    exit_rules=[
        'Target: 2.0x IB range from entry',
        'Stop: Below EMA50 or 0.5x IB range below entry, whichever is tighter',
        'Trail: Once 1.5x IB is reached, trail stop to 1.0x IB profit',
        'Time exit: Close by 15:30 ET',
    ],
    stop_method='Below EMA50 or 0.5x IB range',
    target_method='2.0x IB range',
    risk_reward_ratio=2.0,

    max_risk_pct=0.75,
    max_daily_loss_pct=1.5,
    max_trades_per_day=3,
    position_sizing='risk / (stop_distance * point_value)',

    expected_win_rate=0.50,
    expected_rr=2.0,
    expected_expectancy=0.50,
    pass_rate_estimate='high',

    consistency_compliant=True,
    microscalp_compliant=True,
    eod_flat=True,
    backtestable=True,

    implementation_notes=(
        'Very similar to existing TrendDayBull but uses EMA crossover as the '
        'primary trend filter instead of IB acceptance. The EMA20 pullback entry '
        'is already in the existing strategy. Key addition: EMA20/50 crossover '
        'as the regime filter, and the explicit pullback-to-EMA20 requirement. '
        'Uses all existing data columns.'
    ),
)

STRATEGY_06_ICT_FVG_OB = StrategyResearch(
    id='ICT_FVG_OB',
    name='ICT Fair Value Gap + Order Block Confluence',
    category='evaluation',
    source='https://phidiaspropfirm.com/education/fair-value-gap',
    description=(
        'Combines ICT fair value gap entries with order block confluence zones. '
        'When a FVG aligns with an order block, the probability of a reversal or '
        'continuation is significantly higher. Community reports that ICT-trained '
        'traders show superior market structure awareness but struggle with '
        'execution — adding order flow confirmation (delta) fixes this gap. '
        'Works best during London-NY overlap (09:30-12:00 ET for futures).'
    ),
    instruments=['NQ', 'MNQ'],
    timeframe='1min',
    session_window='09:30-12:00 ET',

    entry_rules=[
        '1. Identify market structure: higher highs/lows (bullish) or lower/lower (bearish)',
        '2. Detect FVG: 3-candle pattern with gap between candle 1 high and candle 3 low',
        '3. Wait for price to retrace INTO the FVG zone',
        '4. Check for order block at same zone (last opposing candle before the move)',
        '5. Delta confirmation: delta aligns with intended direction at FVG fill',
        '6. Enter when price touches FVG zone AND shows rejection (wick, not close through)',
        '7. Confidence boost if VWAP also near the FVG zone (confluence)',
    ],
    entry_indicators=[
        'FVG_BULL', 'FVG_BEAR', 'IFVG_BULL_ENTRY', 'IFVG_BEAR_ENTRY',
        'BPR_BULL', 'BPR_BEAR', 'DELTA', 'VWAP',
    ],

    exit_rules=[
        'Target: Opposite liquidity zone or 2.0x risk',
        'Stop: Beyond the FVG boundary (opposite side)',
        'Trail: Breakeven after 1:1 R',
        'Time exit: Close by 15:00 ET',
    ],
    stop_method='Beyond FVG boundary',
    target_method='Next liquidity pool or 2.0x risk',
    risk_reward_ratio=2.5,

    max_risk_pct=0.50,
    max_daily_loss_pct=1.0,
    max_trades_per_day=3,
    position_sizing='risk / (stop_distance * point_value)',

    expected_win_rate=0.50,
    expected_rr=2.5,
    expected_expectancy=0.75,
    pass_rate_estimate='medium',

    consistency_compliant=True,
    microscalp_compliant=True,
    eod_flat=True,
    backtestable=True,

    implementation_notes=(
        'FVG detection already exists in indicators/ict_models.py. The existing '
        'strategy uses FVG as confluence but not as primary entry. This strategy '
        'makes FVG fill the PRIMARY trigger with order block + delta as confirmations. '
        'Implementation requires adding order block detection (last opposing close '
        'before an impulsive move) to ict_models.py.'
    ),
)


# ============================================================================
#  CATEGORY B: FUNDED PAYOUT CONSISTENCY STRATEGIES
# ============================================================================

STRATEGY_07_MICRO_BUFFER = StrategyResearch(
    id='MICRO_BUFFER',
    name='Micro Contract Buffer Builder',
    category='payout',
    source='https://masterytraderacademy.com/prop-firm-stay-funded-withdrawal-strategy/',
    description=(
        'Start every funded account with micro contracts ONLY to build a profit '
        'buffer above the trailing drawdown. Do NOT scale to minis until buffer '
        'exceeds trailing drawdown threshold. Community consensus: this is THE '
        '#1 reason traders stay funded vs lose accounts. Build $1,500+ buffer '
        'with micros before increasing size. After payouts, drop back to micros '
        'for 3-5 days to rebuild buffer.'
    ),
    instruments=['MNQ', 'MES'],
    timeframe='1min',
    session_window='10:00-13:00 ET',

    entry_rules=[
        '1. Use SAME entry logic as primary strategy (ORB, momentum, etc.)',
        '2. Position size: micro contracts only until buffer > $1,500',
        '3. Buffer tiers: 0-500 = 1 micro, 500-1000 = 3 micros, 1000-1500 = 5 micros',
        '4. After buffer built: scale to standard size per prop firm scaling rules',
        '5. After payout: drop back to 1-3 micros for minimum 3 trading days',
    ],
    entry_indicators=['SAME_AS_PRIMARY_STRATEGY'],

    exit_rules=[
        'Same as primary strategy but with tighter stops during buffer phase',
        'Hard daily cap: Stop trading after +$300 profit (buffer phase)',
        'Stop trading after 2 consecutive losses (buffer phase)',
    ],
    stop_method='Tighter stops: 0.3x IB range during buffer phase',
    target_method='Same as primary strategy',
    risk_reward_ratio=2.0,

    max_risk_pct=0.25,         # Very conservative during buffer
    max_daily_loss_pct=0.50,
    max_trades_per_day=3,
    position_sizing='Micro-only tiered: 1/3/5 micros based on buffer size',

    expected_win_rate=0.50,
    expected_rr=2.0,
    expected_expectancy=0.50,
    pass_rate_estimate='high',

    consistency_compliant=True,
    microscalp_compliant=True,
    eod_flat=True,
    backtestable=True,

    implementation_notes=(
        'Implementation is a WRAPPER around any existing strategy that modifies '
        'position sizing based on current buffer above trailing drawdown. Requires '
        'tracking equity curve and trailing drawdown in engine/position.py. The '
        'strategy itself does not change — only the risk management layer changes.'
    ),
)

STRATEGY_08_PAYOUT_PROTECT = StrategyResearch(
    id='PAYOUT_PROTECT',
    name='Trailing Drawdown Protection Mode',
    category='payout',
    source='https://maventrading.com/blog/trailing-drawdown-prop-trading',
    description=(
        'Once funded, the trailing drawdown becomes the primary risk. This strategy '
        'enforces protective behaviors: never let trailing drawdown catch up to '
        'current equity. Key rule: always leave $500-$1,000 cushion above drawdown '
        'after any payout. Community data shows traders who maintain 2x buffer '
        'above drawdown stay funded 3x longer than those who trade at the edge.'
    ),
    instruments=['MNQ', 'MES', 'NQ', 'ES'],
    timeframe='1min',
    session_window='10:00-13:00 ET',

    entry_rules=[
        '1. SKIP trading if buffer above trailing DD < $500',
        '2. Reduce size by 50% if buffer < $1,000',
        '3. Normal size only when buffer > $1,500',
        '4. After ANY losing day, reduce next day size by 50%',
        '5. After 2 consecutive losers, pause trading for 1 day',
    ],
    entry_indicators=['ACCOUNT_EQUITY', 'TRAILING_DRAWDOWN', 'BUFFER_SIZE'],

    exit_rules=[
        'Tighten all stops by 25% when buffer < $1,000',
        'Move stops to breakeven faster (after 0.5R instead of 1.0R)',
        'Hard daily stop: quit after -$400 (funded) vs -$1,500 (eval)',
    ],
    stop_method='Tightened stops based on buffer size',
    target_method='Standard targets but take partials earlier',
    risk_reward_ratio=2.0,

    max_risk_pct=0.30,
    max_daily_loss_pct=0.50,
    max_trades_per_day=3,
    position_sizing='Buffer-dependent: skip/half/full based on cushion',

    expected_win_rate=0.52,
    expected_rr=2.0,
    expected_expectancy=0.56,
    pass_rate_estimate='high',

    consistency_compliant=True,
    microscalp_compliant=True,
    eod_flat=True,
    backtestable=True,

    implementation_notes=(
        'This is a RISK MANAGEMENT OVERLAY, not a new entry strategy. It wraps '
        'around the existing position manager to modify sizing and stops based on '
        'the equity buffer above trailing drawdown. Requires modifying '
        'engine/position.py to expose buffer_above_trailing_dd as a state variable '
        'that strategies can read.'
    ),
)

STRATEGY_09_CONSISTENCY_PACER = StrategyResearch(
    id='CONSISTENCY_PACER',
    name='Consistency Rule Compliance Pacer',
    category='payout',
    source='https://tradeify.co/post/handle-tradeify-consistency-rule',
    description=(
        'Enforces the 35-40% consistency rule by capping daily profits. If today\'s '
        'profit would exceed 35-40% of total accumulated profits, stop trading. '
        'This prevents "one big day" from blocking payouts. Community insight: '
        'the consistency rule is the #1 reason payouts get denied. Successful '
        'traders use daily profit caps of $300-$500 during evaluation to ensure '
        'profits are spread across 5+ winning days.'
    ),
    instruments=['MNQ', 'MES', 'NQ', 'ES'],
    timeframe='1min',
    session_window='10:00-13:00 ET',

    entry_rules=[
        '1. Before each trade, compute: would max profit cause consistency violation?',
        '2. If today_profit / total_profit would exceed 0.35, SKIP the trade',
        '3. Daily profit cap: min($500, 0.30 * total_accumulated_profit)',
        '4. After hitting daily cap, stop trading for the day',
        '5. If near profit target (within 20%), reduce to 1 micro contract only',
    ],
    entry_indicators=['DAILY_PNL', 'TOTAL_PNL', 'CONSISTENCY_RATIO'],

    exit_rules=[
        'Standard exits but with daily profit cap enforcement',
        'If closing a trade would breach consistency rule, scale down target',
    ],
    stop_method='Standard stops',
    target_method='Capped at daily profit limit',
    risk_reward_ratio=2.0,

    max_risk_pct=0.50,
    max_daily_loss_pct=1.0,
    max_trades_per_day=3,
    position_sizing='Adaptive: reduce near daily cap',

    expected_win_rate=0.50,
    expected_rr=1.5,
    expected_expectancy=0.25,
    pass_rate_estimate='high',

    consistency_compliant=True,  # BY DESIGN
    microscalp_compliant=True,
    eod_flat=True,
    backtestable=True,

    implementation_notes=(
        'This is a COMPLIANCE OVERLAY that wraps the sim_prop_firm.py Monte Carlo. '
        'Modifies the simulation to enforce daily caps and check consistency ratios '
        'before each simulated trade. Critical for accurate evaluation pass simulation.'
    ),
)

STRATEGY_10_SESSION_WINDOW = StrategyResearch(
    id='SESSION_WINDOW',
    name='Session-Restricted Peak Hours Trading',
    category='payout',
    source='https://damnpropfirms.com/prop-firms/core-strategy-execution-nq-es-specific/',
    description=(
        'Restricts all trading to peak liquidity windows only. Community data '
        'shows 9:30-12:00 ET captures 70%+ of daily range with cleanest moves. '
        'The overlap between European close and US open (9:30-12:00) delivers '
        'the cleanest directional movement. By restricting trading to these hours, '
        'traders reduce overtrading, improve execution quality, and stay sharp.'
    ),
    instruments=['NQ', 'MNQ', 'ES', 'MES'],
    timeframe='1min',
    session_window='09:30-12:00 ET',

    entry_rules=[
        '1. ONLY trade between 09:30 and 12:00 ET',
        '2. NO trades after 12:00 ET regardless of setup quality',
        '3. First 30 mins (09:30-10:00): ORB setups only, no counter-trend',
        '4. 10:00-12:00: Full strategy menu (momentum, pullback, mean reversion)',
        '5. Max 2 trades in first 30 mins, max 3 total for the session',
    ],
    entry_indicators=['TIME', 'SESSION_CONTEXT'],

    exit_rules=[
        'All positions must be closed by 12:30 ET (30 min buffer)',
        'Standard stops and targets but with 12:30 hard cutoff',
    ],
    stop_method='Standard',
    target_method='Standard but capped by time exit',
    risk_reward_ratio=2.0,

    max_risk_pct=0.75,
    max_daily_loss_pct=1.5,
    max_trades_per_day=3,
    position_sizing='Standard',

    expected_win_rate=0.55,
    expected_rr=2.0,
    expected_expectancy=0.65,
    pass_rate_estimate='high',

    consistency_compliant=True,
    microscalp_compliant=True,
    eod_flat=True,
    backtestable=True,

    implementation_notes=(
        'Simple time filter overlay. Enhances existing TimeFilter in '
        'filters/time_filter.py. Key change: tighten from current 10:30-15:30 '
        'window to 09:30-12:00 for evaluation speed. Wider window for funded.'
    ),
)


# ============================================================================
#  CATEGORY C: HYBRID EVALUATION + FUNDED STRATEGIES
# ============================================================================

STRATEGY_11_DUAL_REGIME = StrategyResearch(
    id='DUAL_REGIME',
    name='Dual-Regime: Aggressive Eval → Conservative Funded',
    category='hybrid',
    source='https://www.fxempire.com/news/article/how-to-pass-any-prop-firm-evaluation-in-2026',
    description=(
        'Automatically switches between aggressive (evaluation) and conservative '
        '(funded) parameter sets. During evaluation: maximize trades, wider stops, '
        'larger size to hit $9,000 fast. After funded: reduce size, tighten stops, '
        'enforce daily caps to protect trailing drawdown. This IS the existing '
        'dual_strategy.py approach but with the community-validated parameters. '
        'Key finding: 31 MNQ eval / 20 MNQ funded is optimal for $150K account.'
    ),
    instruments=['MNQ'],
    timeframe='1min',
    session_window='10:00-13:00 ET',

    entry_rules=[
        'EVALUATION MODE:',
        '  - 31 MNQ contracts',
        '  - Pure order flow: delta + CVD + imbalance (no HTF filters)',
        '  - Daily loss limit: $1,500',
        '  - Max 15 trades/day',
        '  - Max 5 consecutive losses before stopping',
        '',
        'FUNDED MODE:',
        '  - 20 MNQ contracts (or micro tiered by buffer)',
        '  - Add 5-min CVD filter + VWAP confirmation',
        '  - Daily loss limit: $800',
        '  - Max 10 trades/day',
        '  - Max 3 consecutive losses before stopping',
    ],
    entry_indicators=[
        'DELTA_PERCENTILE', 'VOLUME_SPIKE', 'CVD', 'IMBALANCE_RATIO',
        'VWAP', 'EMA20', 'EMA50',
    ],

    exit_rules=[
        'EVAL: Standard stops/targets, maximize trade count',
        'FUNDED: Tighter stops, breakeven faster, daily profit cap',
    ],
    stop_method='EVAL: 0.4x IB | FUNDED: 0.3x IB',
    target_method='EVAL: 2.0x IB | FUNDED: 1.5x IB',
    risk_reward_ratio=2.0,

    max_risk_pct=0.75,
    max_daily_loss_pct=1.0,
    max_trades_per_day=15,
    position_sizing='EVAL: 31 MNQ fixed | FUNDED: tiered by buffer',

    expected_win_rate=0.44,
    expected_rr=2.0,
    expected_expectancy=0.32,
    pass_rate_estimate='high',

    consistency_compliant=True,
    microscalp_compliant=True,
    eod_flat=True,
    backtestable=True,

    implementation_notes=(
        'This IS the existing dual_strategy.py. Documented here to capture the '
        'community-validated parameters and show that our existing approach aligns '
        'with what successful funded traders recommend. The 31/20 MNQ split and '
        '10:00-13:00 window match community best practices.'
    ),
)

STRATEGY_12_MULTI_ACCOUNT = StrategyResearch(
    id='MULTI_ACCOUNT',
    name='Sequential Multi-Account Factory',
    category='hybrid',
    source='https://tradeify.co/post/managing-multiple-prop-firm-accounts',
    description=(
        'Run 3-5 evaluations sequentially (not simultaneously) to eliminate '
        'correlation risk. Pass account 1, start funded trading. Use profits '
        'from account 1 to fund account 2 evaluation. Diversify across '
        'drawdown types (2x trailing + 3x EOD). Community insight: multiple '
        'accounts with different drawdown rules is TRUE diversification. '
        'Expected timeline: 5 funded accounts within 2-3 months.'
    ),
    instruments=['MNQ'],
    timeframe='1min',
    session_window='10:00-13:00 ET',

    entry_rules=[
        '1. Account 1: Pass evaluation with aggressive strategy',
        '2. Account 1 funded: Switch to conservative + micro buffer',
        '3. Account 2: Start evaluation (can overlap with funded trading on 1)',
        '4. Repeat until 5 funded accounts',
        '5. Diversify: 2 accounts trailing DD, 3 accounts EOD DD',
        '6. Never enter same trade on all accounts simultaneously',
    ],
    entry_indicators=['SAME_AS_PRIMARY_STRATEGY'],

    exit_rules=[
        'Per-account risk limits (not aggregate)',
        'Per-account daily loss caps',
        'Stagger payouts: don\'t withdraw from all accounts same week',
    ],
    stop_method='Per-account standard stops',
    target_method='Per-account standard targets',
    risk_reward_ratio=2.0,

    max_risk_pct=0.75,
    max_daily_loss_pct=1.0,
    max_trades_per_day=3,
    position_sizing='Per-account micro tiered',

    expected_win_rate=0.50,
    expected_rr=2.0,
    expected_expectancy=0.50,
    pass_rate_estimate='high',

    consistency_compliant=True,
    microscalp_compliant=True,
    eod_flat=True,
    backtestable=True,

    implementation_notes=(
        'This is the EVALUATION_FACTORY_SYSTEM.md approach already documented. '
        'The Monte Carlo sim in sim_prop_firm.py already supports this. Key '
        'addition: stagger payouts and diversify drawdown types across accounts.'
    ),
)


# ============================================================================
#  AGGREGATE COLLECTION
# ============================================================================

ALL_STRATEGIES = [
    STRATEGY_01_ORB_VWAP,
    STRATEGY_02_LIQUIDITY_SWEEP,
    STRATEGY_03_MOMENTUM_SCALP,
    STRATEGY_04_MEAN_REVERT_VWAP,
    STRATEGY_05_TREND_FOLLOW_EMA,
    STRATEGY_06_ICT_FVG_OB,
    STRATEGY_07_MICRO_BUFFER,
    STRATEGY_08_PAYOUT_PROTECT,
    STRATEGY_09_CONSISTENCY_PACER,
    STRATEGY_10_SESSION_WINDOW,
    STRATEGY_11_DUAL_REGIME,
    STRATEGY_12_MULTI_ACCOUNT,
]

EVALUATION_STRATEGIES = [s for s in ALL_STRATEGIES if s.category == 'evaluation']
PAYOUT_STRATEGIES = [s for s in ALL_STRATEGIES if s.category == 'payout']
HYBRID_STRATEGIES = [s for s in ALL_STRATEGIES if s.category == 'hybrid']
BACKTESTABLE_STRATEGIES = [s for s in ALL_STRATEGIES if s.backtestable]


def print_strategy_summary():
    """Print a summary table of all researched strategies."""
    print("=" * 120)
    print("  EVALUATION & PAYOUT STRATEGY RESEARCH — SUMMARY")
    print("=" * 120)

    for cat_name, cat_strats in [
        ('EVALUATION PASSING', EVALUATION_STRATEGIES),
        ('FUNDED PAYOUT', PAYOUT_STRATEGIES),
        ('HYBRID', HYBRID_STRATEGIES),
    ]:
        print(f"\n{'─' * 120}")
        print(f"  {cat_name} STRATEGIES")
        print(f"{'─' * 120}")
        print(f"{'ID':<22} {'Name':<45} {'WR%':>5} {'R:R':>5} {'E[R]':>6} {'Pass':>6} {'Backtest':>8}")
        print(f"{'─' * 22} {'─' * 45} {'─' * 5} {'─' * 5} {'─' * 6} {'─' * 6} {'─' * 8}")
        for s in cat_strats:
            print(
                f"{s.id:<22} {s.name[:45]:<45} "
                f"{s.expected_win_rate * 100:>4.0f}% {s.expected_rr:>5.1f} "
                f"{s.expected_expectancy:>+5.2f} {s.pass_rate_estimate:>6} "
                f"{'YES' if s.backtestable else 'NO':>8}"
            )

    print(f"\n{'=' * 120}")
    print(f"  Total strategies: {len(ALL_STRATEGIES)}")
    print(f"  Backtestable: {len(BACKTESTABLE_STRATEGIES)}")
    print(f"  Evaluation: {len(EVALUATION_STRATEGIES)} | Payout: {len(PAYOUT_STRATEGIES)} | Hybrid: {len(HYBRID_STRATEGIES)}")
    print(f"{'=' * 120}")


if __name__ == '__main__':
    print_strategy_summary()

    print("\n\nDETAILED STRATEGY DESCRIPTIONS:")
    print("=" * 120)
    for s in ALL_STRATEGIES:
        print(f"\n{'─' * 120}")
        print(f"  [{s.id}] {s.name}")
        print(f"  Category: {s.category} | Source: {s.source}")
        print(f"{'─' * 120}")
        print(f"\n  {s.description}\n")
        print(f"  Instruments: {', '.join(s.instruments)}")
        print(f"  Timeframe: {s.timeframe} | Session: {s.session_window}")
        print(f"\n  ENTRY RULES:")
        for rule in s.entry_rules:
            print(f"    {rule}")
        print(f"\n  EXIT RULES:")
        for rule in s.exit_rules:
            print(f"    {rule}")
        print(f"\n  RISK MANAGEMENT:")
        print(f"    Stop: {s.stop_method}")
        print(f"    Target: {s.target_method}")
        print(f"    R:R = {s.risk_reward_ratio}")
        print(f"    Max risk/trade: {s.max_risk_pct}% | Max daily loss: {s.max_daily_loss_pct}%")
        print(f"    Max trades/day: {s.max_trades_per_day}")
        print(f"    Position sizing: {s.position_sizing}")
        print(f"\n  EXPECTED PERFORMANCE:")
        print(f"    Win rate: {s.expected_win_rate * 100:.0f}% | R:R: {s.expected_rr:.1f}")
        print(f"    Expectancy: {s.expected_expectancy:+.2f}R per trade")
        print(f"    Pass rate estimate: {s.pass_rate_estimate}")
        print(f"\n  COMPLIANCE:")
        print(f"    Consistency: {'✓' if s.consistency_compliant else '✗'}")
        print(f"    Microscalp: {'✓' if s.microscalp_compliant else '✗'}")
        print(f"    EOD flat: {'✓' if s.eod_flat else '✗'}")
        print(f"\n  IMPLEMENTATION:")
        print(f"    Backtestable: {'YES' if s.backtestable else 'NO'}")
        print(f"    {s.implementation_notes}")
