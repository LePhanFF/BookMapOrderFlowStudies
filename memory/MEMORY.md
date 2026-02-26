# Project Memory - BookMapOrderFlowStudies-2

## Project Overview
NQ/MNQ futures trading system with Dalton Market Profile day-type strategies.
Backtest engine + diagnostic scripts for prop firm evaluation (Tradeify $150K Select Flex).

## Key Architecture
- `scripts/run_backtest.py` — main entry point (args: --instrument, --strategies, --filters)
- `engine/backtest.py` — unified bar-by-bar engine, dynamic day type classification
- `strategy/` — one file per strategy, all extend `StrategyBase`
- `config/constants.py` — all magic numbers
- `data/loader.py` → `data/session.py` (filter_rth) → `data/features.py` (compute_all_features)
- Python env: `E:\anaconda\python.exe`
- Branch: `dev-local-test`

## v14 Report Target (62 sessions, Nov 18 2025 - Feb 16 2026)
Report: 72 trades, ~82% WR, ~$17,513 net, PF ~12, MaxDD -$407

| Strategy | Trades | WR | Net |
|---|---|---|---|
| Trend Day Bull | 8 | 75% | $1,074 |
| P-Day | 8 | 75% | $1,075 |
| B-Day IBL Fade | 4 | 100% | $2,285 |
| Edge Fade OPTIMIZED | 17 | 94% | $7,696 |
| IBH Sweep+Fail | 4 | 100% | $582 |
| Bear Accept Short | 11 | 64% | $995 |
| Opening Range Rev | 20 | 80% | $3,807 |

## Current Engine Status (as of last run)
77 trades, 70.1% WR, $14,077 net — strategies 1-3 MATCH PERFECTLY.

### What Works (EXACT MATCH)
- **Trend Day Bull**: 8 trades, 75% WR, $1,074 ✓
- **P-Day**: 8 trades, 75% WR, $1,075 ✓
- **B-Day IBL Fade**: 4 trades, 100% WR, $2,285 ✓

### What Needs Work
- **Edge Fade**: 29 trades, 75.9% WR, $8,957 (report: 17, 94%, $7,696)
  - 3 optimization filters implemented (IB<200, before 13:30, NOT bearish ext_down>0.3)
  - Report used backward-looking session classification (look-ahead bias)
  - Engine uses real-time filter → more trades pass, lower WR but HIGHER net P&L
  - This is actually a BETTER result for live trading

- **IBH Sweep+Fail**: 2 trades, 0% WR, -$731 (report: 4, 100%, $582)
  - FUNDAMENTAL ISSUE: Report's 4 trades used HTF confluence (4H FVG, London levels)
  - Engine only has RTH data → can't check overnight/London/4H levels
  - Need ETH data in engine to match report's selectivity
  - Current filters: b_day + PDH proximity + wick rejection + delta < 0
  - See `strategy/ibh_sweep.py` and `diagnostics/diagnostic_ibh_fade_short.py`

- **Bear Accept Short**: 8 trades, 75% WR, $144 (report: 11, 64%, $995)
  - Fewer trades because engine's dynamic day_type classification is stricter
  - trend_down only fires when ext > 1.0x IB below — report included 0.5x too
  - Higher WR (75% vs 64%) suggests we're missing some valid losing trades
  - See `strategy/bear_accept.py`

- **Opening Range Rev**: 18 trades, 55.6% WR, $1,273 (report: 20, 80%, $3,807)
  - Uses prior_session_high/low as proxy for overnight levels (not exact)
  - Report used actual ETH overnight H/L from full data
  - Strategy scans IB bars in on_session_start(), emits cached signal
  - Need full ETH data access to improve overnight level detection
  - See `strategy/or_reversal.py`

## Critical Known Issues

### PositionManager max_drawdown blocks trades
`config/constants.py`: `DEFAULT_MAX_DRAWDOWN = 4000` (ABSOLUTE, not % based)
On 142-session full data, early losses exceed $4K → blocks ALL future trades.
Fix: pass `max_drawdown=999999` to PositionManager for testing.

### ETH Data Access
Engine receives RTH-filtered data only. Strategies needing overnight/London levels
(IBH Sweep, OR Reversal) can't access ETH bars.
Fix needed: pass full DataFrame or pre-compute overnight levels before RTH filter.

### Backward-Looking Bias in v14 Report
The diagnostic scripts classified sessions AFTER completion (full-session ext_down).
The engine classifies in real-time. This creates a gap for:
- Edge Fade: diagnostic excluded sessions that ENDED bearish; engine only knows current state
- Bear Accept: diagnostic used end-of-session day type; engine uses dynamic classification

### sys.path after file moves
All scripts in `scripts/` and `diagnostics/` use `Path(__file__).resolve().parent.parent`
for project root (were moved from root during repo cleanup).

## Strategy Implementation Details

### Edge Fade Optimization Filters (strategy/edge_fade.py)
```
EDGE_FADE_MAX_IB_RANGE = 200.0     # Was 350 (report: IB < 200)
EDGE_FADE_LAST_ENTRY_TIME = 13:30  # Was 15:20 (report: before 13:30)
EDGE_FADE_MAX_BEARISH_EXT = 0.3    # NEW: ext_down tracked from bar['low']
```
Bearish filter uses real-time max_ext_down (not session-end classification).

### SuperTrend Bull REMOVED from core
Report's 7 strategies don't include SuperTrend Bull. Removed from CORE_STRATEGIES.

### get_core_strategies() now returns 7 strategies
TrendDayBull, PDayStrategy, BDayStrategy, EdgeFadeStrategy,
IBHSweepFail, BearAcceptShort, OpeningRangeReversal

## File Locations
- Report: `docs/reports/2026.02.19-63days_final_report.md`
- Risk scaling: `docs/reports/eval_risk_scaling_study.md`
- Trade logs: `output/trade_log_v14.csv` (original), `output/trade_log_v14_62days_validation.csv`
- Diagnostics: `diagnostics/diagnostic_*.py` (24 files)
- NinjaTrader code: `ninjatrader/*.cs`

## Prop Firm Rules (Tradeify $150K Select Flex)
- Profit target: $9,000
- Trailing DD: $4,500 (EOD)
- No daily loss limit, no time limit
- Risk scaling study: 1.5x recommended (77 days median, 99.8% pass with 1 reset)
- See `scripts/sim_eval_risk_scaling.py`

## Next Steps
1. Add ETH data access to engine for overnight level computation
2. Fix IBH Sweep with proper overnight/London/4H confluence data
3. Improve OR Reversal with actual overnight H/L (not prior session proxy)
4. Run 200-session report once 7-strategy portfolio is validated
5. Re-run Monte Carlo risk scaling with updated trade distribution
