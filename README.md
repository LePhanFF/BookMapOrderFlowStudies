# Order Flow Strategy for Prop Firm Trading

**Status**: Production Ready (v14)
**Portfolio**: 7-Strategy Dalton Playbook + Order Flow Quality Gate
**Instrument**: MNQ (Micro E-mini Nasdaq-100)
**Platform**: NinjaTrader 8
**Target**: Tradeify $150K Select Flex Evaluation

---

## Latest Results (v14 - Feb 19, 2026)

| Portfolio | Trades | Win Rate | Expectancy | Net P&L | Max DD | Profit Factor |
|-----------|--------|----------|------------|---------|--------|---------------|
| **Intraday (6 strategies)** | 52 | 83% | $264/trade | $13,706 | -$351 | 18.35 |
| **Opening Range (1 strategy)** | 20 | 80% | $190/trade | $3,807 | -$407 | 6.30 |
| **COMBINED (7 strategies)** | **72** | **~82%** | **~$243/trade** | **~$17,513** | **-$407** | **~12** |

Annualized estimate: **~$71,200/year** on 1 MNQ contract.
Backtest: 62 RTH sessions, Nov 18 2025 - Feb 16 2026, MNQ at $2/pt.

**Data update**: CSV folder now contains data back to August 2025 (200+ sessions). Re-running backtests against this expanded dataset is the next priority.

---

## The 7 Strategies

| # | Strategy | Direction | Time | Trades | WR | Exp/Trade | Net P&L |
|---|----------|-----------|------|--------|-----|-----------|---------|
| 1 | Trend Day Bull (VWAP pullback) | LONG | 10:30-13:00 | 8 | 75% | $134 | $1,074 |
| 2 | P-Day VWAP Pullback | LONG | 10:30-13:00 | 8 | 75% | $134 | $1,075 |
| 3 | B-Day IBL Fade | LONG | 10:30-14:00 | 4 | 100% | $571 | $2,285 |
| 4 | Edge Fade (optimized) | LONG | 10:30-13:30 | 17 | 94% | $453 | $7,696 |
| 5 | IBH Sweep+Fail | SHORT | 10:30-14:00 | 4 | 100% | $146 | $582 |
| 6 | Bear Acceptance Short | SHORT | 10:30-14:00 | 11 | 64% | $90 | $995 |
| 7 | Opening Range Reversal | BOTH | 9:30-10:30 | 20 | 80% | $190 | $3,807 |

Strategies 1-6 run after the Initial Balance (10:30-16:00). Strategy 7 runs during the opening range (9:30-10:30). No time overlap = fully additive.

---

## Key Discovery: Playbook Beats Pure Order Flow

The original approach used pure 1-minute order flow signals (delta + CVD). The v14 playbook approach uses Dalton Market Profile day-type classification as the primary filter, with order flow as a quality gate.

| Approach | Win Rate | Expectancy | Sharpe |
|----------|----------|------------|--------|
| Pure Order Flow (standalone) | 44-56% | $37-84/trade | ~2 |
| **Playbook + OF Quality Gate** | **80-82%** | **$190-264/trade** | **~19** |

Full comparison: [Order Flow vs Playbook Analysis](docs/reports/order_flow_vs_playbook_comparison.md)

---

## 10 Rules You Must Never Break

1. **NEVER short on b_day or neutral** (0-14% WR on NQ)
2. **NEVER enter Edge Fade after 13:30** (0% WR, PM morph kills mean reversion)
3. **NEVER enter Edge Fade when IB > 200 pts** (0-36% WR on wide IB)
4. **NEVER enter Edge Fade on bearish days** (ext_down > 0.3x = 37% WR)
5. **NEVER use FVG as standalone entry on 1-min bars** (16% WR, counter-indicator)
6. **NEVER use trailing stops on VWAP pullback entries** (reduces P&L 50-80%)
7. **NEVER use CVDFilter on B-Day trades** (kills entries that naturally have CVD < MA)
8. **Day type is THE filter** - classify the day FIRST, then pick the strategy
9. **EOD exit captures the most P&L** for trend/p-day strategies
10. **Pre-entry delta < -500 on VWAP pullbacks = skip** (aggressive selling)

---

## Quick Start

1. Copy `ninjatrader/DualOrderFlow_Evaluation.cs` and `ninjatrader/DualOrderFlow_Funded.cs` to `Documents\NinjaTrader 8\bin\Custom\Strategies\`
2. Compile in NinjaTrader (`Tools` > `Edit NinjaScript` > `Strategy` > F5)
3. Create MNQ 1-minute chart and add strategy
4. Full setup: [NinjaTrader Setup Guide](docs/guides/NINJATRADER_SETUP_GUIDE.md)

---

## Project Structure

```
BookMapOrderFlowStudies-2/
|
|-- ninjatrader/                    # NinjaTrader 8 deployment scripts
|   |-- DualOrderFlow_Evaluation.cs # NT8 evaluation mode
|   +-- DualOrderFlow_Funded.cs     # NT8 funded mode
|
|-- strategy/                       # Python strategy implementations
|   |-- dual_strategy.py            # Dual mode backtesting
|   |-- strategy_5min_proper.py     # 5-min timeframe strategy
|   |-- base.py                     # Base strategy class
|   |-- b_day.py, p_day.py, ...     # Dalton day-type strategies
|   |-- edge_fade.py                # Edge fade strategy
|   |-- trend_bull.py, trend_bear.py
|   +-- signal.py                   # Signal generation
|
|-- engine/                         # Backtesting engine
|   |-- backtest.py                 # Core backtest runner
|   |-- execution.py                # Order execution
|   |-- position.py                 # Position management
|   |-- equity.py                   # Equity tracking
|   +-- trade.py                    # Trade model
|
|-- data/                           # Data loading and processing
|   |-- loader.py                   # CSV data loader
|   |-- session.py                  # Session boundaries
|   +-- features.py                 # Feature engineering
|
|-- config/                         # Configuration
|   |-- constants.py                # Strategy parameters
|   +-- instruments.py              # Instrument specs
|
|-- filters/                        # Trade filters
|   |-- order_flow_filter.py        # Delta/CVD/imbalance filters
|   |-- time_filter.py              # Session time windows
|   |-- trend_filter.py             # HTF trend detection
|   +-- volatility_filter.py        # ATR-based filters
|
|-- indicators/                     # Technical indicators
|   |-- technical.py                # VWAP, ATR, etc.
|   +-- ict_models.py              # ICT concepts
|
|-- profile/                        # Market profile analysis
|   |-- tpo_profile.py              # TPO profiles
|   |-- volume_profile.py           # Volume profiles
|   |-- ib_analysis.py              # Initial balance
|   +-- dpoc_migration.py           # DPOC tracking
|
|-- prop/                           # Prop firm simulation
|   |-- pipeline.py                 # Multi-account pipeline
|   |-- account.py                  # Account management
|   |-- rules.py                    # Prop firm rules
|   +-- sizer.py                    # Position sizing
|
|-- reporting/                      # Output and metrics
|   |-- metrics.py                  # Performance metrics
|   |-- comparison.py               # Strategy comparison
|   |-- day_analyzer.py             # Daily analysis
|   +-- trade_log.py                # Trade logging
|
|-- export/                         # Export tools
|   +-- ninjatrader.py              # NinjaTrader export
|
|-- scripts/                        # Runners and utilities
|   |-- run_backtest.py             # Single strategy backtest
|   |-- run_playbook_backtests.py   # Playbook suite runner
|   |-- run_all_playbook_strategies.py
|   |-- sim_prop_firm.py            # Prop firm simulation
|   |-- strategy_report.py          # Generate reports
|   |-- live_trading_wrapper.py     # Paper/live trading
|   +-- monitoring_dashboard.py     # Real-time tracking
|
|-- diagnostics/                    # Analysis and debugging scripts
|   |-- diagnostic_full_portfolio_report.py
|   |-- diagnostic_opening_range_*.py
|   |-- diagnostic_bearish_day*.py
|   |-- diagnostic_entry_models.py
|   +-- ... (25 diagnostic scripts)
|
|-- tests/                          # Test suite
|   |-- test_backtest_engine.py
|   |-- test_data_loader.py
|   +-- test_playbook_strategies.py
|
|-- docs/                           # Documentation
|   |-- design/                     # Architecture and system design
|   |-- guides/                     # Setup and usage guides
|   |-- reports/                    # Performance reports and analysis
|   +-- roadmap/                    # Development plans
|
|-- research/                       # Strategy research studies
|   +-- strategy-studies/           # Formal study write-ups
|
|-- csv/                            # Market data (Aug 2025 - Feb 2026, 200+ sessions)
|-- prompts/                        # LLM prompts for analysis
|-- rockit-framework/               # Rockit analysis framework
|-- archive/                        # Deprecated code and old analysis
+-- output/                         # Backtest output
```

---

## Reports and Analysis

### Current Reports
| Report | Description |
|--------|-------------|
| [v14 Final Report (Feb 19)](docs/reports/2026.02.19_final_report.md) | Complete 7-strategy playbook with entry/exit rules, NinjaTrader automation notes, and performance tables |
| [Order Flow vs Playbook Comparison](docs/reports/order_flow_vs_playbook_comparison.md) | Why playbook (80% WR) beats pure order flow (44-56% WR) |

### Research Studies
| Study | Description |
|-------|-------------|
| [Master Index](research/strategy-studies/MASTER_INDEX.md) | Comparison of 4 mechanical strategies (OF, ORB, Trend, Mean Reversion) |
| [Opening Range Breakout](research/strategy-studies/OPENING_RANGE_BREAKOUT_STUDY.md) | ORB strategy analysis for prop firms |
| [Trend Following Breakout](research/strategy-studies/TREND_FOLLOWING_BREAKOUT_STUDY.md) | Multi-timeframe momentum trading |
| [Mean Reversion](research/strategy-studies/MEAN_REVERSION_STUDY.md) | Bollinger Band and RSI counter-trend study |
| [Two Hour Trader](research/strategy-studies/TWO_HOUR_TRADER_STUDY.md) | Options-based mechanical trading (79% WR in live evals) |

### Design Documents
| Document | Description |
|----------|-------------|
| [Design Document (v14)](docs/design/design-document.md) | Portfolio architecture for 7-strategy system |
| [Dual Mode Strategy](docs/design/DUAL_MODE_STRATEGY.md) | Evaluation (Maniac) vs Funded (Sniper) framework |
| [Dual Strategy System](docs/design/DUAL_STRATEGY_SYSTEM.md) | Tiered Imbalance+Volume+CVD and Delta+CVD approach |
| [Evaluation Factory](docs/design/EVALUATION_FACTORY_SYSTEM.md) | Sequential evaluation passing and scaling plan |
| [Final Strategy](docs/design/FINAL_STRATEGY.md) | Complete order flow implementation guide |
| [Prop Firm Strategy](docs/design/prop_firm_strategy.md) | Tradeify $150K strategy with Dalton + Order Flow |

### Guides
| Guide | Description |
|-------|-------------|
| [NinjaTrader Setup](docs/guides/NINJATRADER_SETUP_GUIDE.md) | Complete NT8 setup for automated prop firm trading |
| [Quick Reference](docs/guides/NT_STRATEGY_QUICK_REFERENCE.md) | Dual order flow NinjaTrader script reference |
| [Automation Guide](docs/guides/AUTOMATION_GUIDE.md) | Automated order entry, NQ vs ES validation |
| [Enhanced Strategies](docs/guides/ENHANCED_STRATEGIES_GUIDE.md) | Bookmap-style features using OHLCV + delta |

### Roadmap
| Document | Description |
|----------|-------------|
| [Automation Architecture](docs/roadmap/AUTOMATION_ARCHITECTURE.md) | NT8 + Python Signal API with local and GCP deployment |
| [Development Roadmap](docs/roadmap/DEVELOPMENT_ROADMAP.md) | 3-strategy roadmap: trend, mean reversion, options |
| [Responsible Acceleration](docs/roadmap/RESPONSIBLE_ACCELERATION.md) | Risk management plan for evaluation passing |

### Archived Reports
Older reports superseded by the v14 playbook approach are in [`docs/reports/archive/`](docs/reports/archive/).

---

## How to Add New Data

1. Export volumetric bars from NinjaTrader (1-min bars with delta, CVD, imbalance)
2. Save as CSV in `csv/`: `NQ_Volumetric_1.csv`, `ES_Volumetric_1.csv`, `YM_Volumetric_1.csv`
3. Required columns: timestamp, session_date, open, high, low, close, volume, vol_ask, vol_bid, vol_delta, cumulative_delta, delta_percentile, imbalance_ratio, volume_spike, vwap, vwap_upper1-3, vwap_lower1-3, ema20, ema50, ema200, rsi14, atr14
4. Run intraday backtest: `python scripts/run_backtest.py --strategies core`
5. Run opening range analysis: `python diagnostics/diagnostic_opening_range_v2.py`

---

## Prop Firm Evaluation Plan

- **Profit target**: $9,000 (Tradeify $150K)
- **Trailing drawdown**: $4,500 (EOD)
- **Trades to pass**: ~37 trades at $243 avg expectancy
- **Timeline**: ~4-5 weeks at 1.2 trades/day
- **Max DD risk**: -$407 = 9% of drawdown allowance

---

*Repository Version: 3.0 (v14 playbook)*
*Last Updated: February 19, 2026*
