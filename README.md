# Dalton Market Profile Trading System

**Instrument**: MNQ (Micro E-mini Nasdaq-100 Futures)
**Platform**: NinjaTrader 8
**Target**: Tradeify $150K Select Flex Evaluation ($9K profit, $4.5K trailing DD)

---

## 259-Session Backtest Results (Feb 2025 - Feb 2026)

### Full Portfolio (7 Strategies)

| Metric | Value |
|--------|-------|
| Sessions | 259 |
| Trades | 283 |
| Win Rate | 55.5% |
| Net P&L | **$19,462** |
| Profit Factor | 1.58 |
| Max Drawdown | $3,334 |

### Solo Strategy Performance

| Strategy | Trades | WR | Net P&L | MaxDD | Prop Firm |
|----------|--------|----|---------|-------|-----------|
| **OR Reversal** | 65 | 61.5% | **$13,962** | $1,363 | **PASSES** |
| **Edge Fade** | 96 | 56.2% | **$8,434** | $2,374 | Too Slow |
| B-Day | 19 | 42.1% | $758 | $3,375 | Too Slow |
| P-Day | 23 | 34.8% | -$514 | $1,585 | Loses |
| Trend Bull | 21 | 38.1% | -$589 | $1,651 | Loses |
| Bear Accept | 58 | 67.2% | -$2,553 | $3,001 | Loses |
| IBH Sweep | 1 | 0.0% | -$37 | $37 | Loses |

**Best combo: OR Reversal + Edge Fade = $22,396 net, DD $2,262 (50% of prop firm limit)**

---

## Quick Start

```bash
# Run full portfolio backtest (259 sessions)
E:\anaconda\python.exe scripts/run_backtest.py --strategies core

# Run specific strategy
E:\anaconda\python.exe scripts/run_backtest.py --strategies or_reversal

# Solo strategy accounts (each strategy in own account)
E:\anaconda\python.exe scripts/diag_solo_strategy_accounts.py

# Monthly P&L grid
E:\anaconda\python.exe scripts/diag_monthly_report.py
```

### Data Setup

1. Export 1-min volumetric bars from NinjaTrader (delta, CVD, imbalance columns)
2. Save as CSV in `csv/` (gitignored, >100MB each)
3. Required columns: timestamp, session_date, open, high, low, close, volume, vol_ask, vol_bid, vol_delta, cumulative_delta, delta_percentile, imbalance_ratio, volume_spike, vwap, vwap_upper1-3, vwap_lower1-3, ema20, ema50, ema200, rsi14, atr14

---

## Project Structure

```
BookMapOrderFlowStudies-2/
├── config/                  # Configuration
│   ├── constants.py         # All strategy parameters (magic numbers)
│   └── instruments.py       # Instrument specs (tick size, multiplier)
│
├── data/                    # Data pipeline
│   ├── loader.py            # CSV data loader
│   ├── session.py           # RTH session boundaries
│   └── features.py          # Feature engineering
│
├── engine/                  # Backtest engine
│   ├── backtest.py          # Unified bar-by-bar engine, dynamic day-type classification
│   ├── execution.py         # Order execution
│   ├── equity.py            # Equity curve tracking
│   ├── trade.py             # Trade model
│   └── tpo_classifier.py    # TPO-based day-type classification
│
├── strategy/                # Strategy implementations (extend StrategyBase)
│   ├── base.py              # StrategyBase class
│   ├── day_type.py          # Day-type classification logic
│   ├── signal.py            # Signal generation
│   ├── neutral_day.py       # Neutral day (Edge Fade, OR Reversal)
│   ├── super_trend_bull.py  # B-Day IBL Fade
│   ├── super_trend_bear.py  # IBH Sweep + Fail
│   └── trend_bear.py        # Bear Acceptance Short
│
├── filters/                 # Trade filters
│   ├── order_flow_filter.py # Delta/CVD/imbalance quality gate
│   ├── time_filter.py       # Session time windows
│   ├── trend_filter.py      # HTF trend detection
│   └── volatility_filter.py # ATR-based filters
│
├── indicators/              # Technical indicators
│   ├── technical.py         # VWAP, ATR, etc.
│   └── ict_models.py        # ICT concepts (FVG, OB)
│
├── profile/                 # Market profile analysis
│   ├── tpo_profile.py       # TPO profiles
│   ├── volume_profile.py    # Volume profiles
│   ├── ib_analysis.py       # Initial balance analysis
│   ├── dpoc_migration.py    # DPOC tracking
│   └── wick_parade.py       # Wick analysis
│
├── prop/                    # Prop firm simulation
│   ├── pipeline.py          # Multi-account pipeline
│   ├── account.py           # Account tracking
│   ├── rules.py             # Prop firm rules (Tradeify, etc.)
│   └── sizer.py             # Position sizing
│
├── reporting/               # Output and metrics
│   ├── metrics.py           # Performance calculations
│   ├── comparison.py        # Strategy comparison
│   ├── day_analyzer.py      # Daily P&L analysis
│   └── trade_log.py         # Trade logging
│
├── export/                  # Export tools
│   └── ninjatrader.py       # NinjaTrader C# strategy export
│
├── scripts/                 # Runners and diagnostic scripts
│   ├── run_backtest.py      # Main backtest entry point
│   ├── diag_*.py            # Diagnostic/analysis scripts (15+)
│   ├── sim_prop_firm.py     # Prop firm simulation
│   └── strategy_report.py   # Report generation
│
├── diagnostics/             # Legacy diagnostic scripts
│   └── diagnostic_*.py      # Older analysis scripts (25+)
│
├── tests/                   # Test suite
│   ├── test_backtest_engine.py
│   ├── test_data_loader.py
│   └── test_playbook_strategies.py
│
├── docs/                    # Documentation
│   ├── design/              # Architecture and system design
│   ├── guides/              # Setup and usage guides
│   ├── reports/             # Performance reports and analysis
│   └── roadmap/             # Development plans and to-do
│
├── ninjatrader/             # NinjaTrader 8 C# strategies
├── research/                # Strategy research studies
├── csv/                     # Market data (gitignored, >100MB)
├── output/                  # Backtest output (trade logs)
├── prompts/                 # LLM prompts for analysis
├── rockit-framework/        # Rockit analysis framework (legacy)
└── archive/                 # Deprecated code
```

---

## Key Findings

1. **Current strategy-specific targets are optimal** -- beats all fixed R targets and trails
2. **Trail/BE regresses performance** ($13,986 vs $19,462 baseline)
3. **IB > 100pt filter adds $2,077** -- best avoidance filter
4. **Do NOT cap high IB** -- cutting IB < 350 loses $11,092
5. **OR Reversal SHORT side dominates** (72% WR, $11,395 vs LONG 48% WR, $2,567)
6. **Bear Accept has structural 0.30 R:R** -- not fixable with trade management
7. **Dec 2025 = 39% of annual P&L** ($7,582) -- goldilocks volatility regime
8. **Monte Carlo 95th percentile DD: $4,550** at 1 MNQ -- tight for prop firm

---

## Reports

| Report | Description |
|--------|-------------|
| [259-Day Deep Analysis](docs/reports/2026.02.21-259days_deep_analysis.md) | MFE, trail stops, zero-trade diagnosis |
| [259-Day Monthly Grid](docs/reports/2026.02.21-259days_monthly_report.md) | Monthly P&L breakdown |
| [259-Day Portfolio Deep Dive](docs/reports/2026.02.21-259days_portfolio_deep_dive.md) | Profit distribution, sizing, weakness |
| [259-Day Solo Breakdown](docs/reports/2026.02.21-259days-solobreakdown.md) | Each strategy in its own account |
| [Regime Gates Report](docs/reports/2026.02.20-142days_regime_gates_report.md) | Volatility regime filtering |
| [Design Document](docs/design/design-document.md) | Portfolio architecture |
| [NinjaTrader Setup](docs/guides/NINJATRADER_SETUP_GUIDE.md) | NT8 deployment guide |

---

*Last Updated: February 21, 2026*
