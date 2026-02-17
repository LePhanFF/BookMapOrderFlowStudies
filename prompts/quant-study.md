# Order Flow + Auction Market Strategy for Prop Firm Trading

## Role
You are a quantitative researcher at Goldman Sachs Global Markets specializing in order flow and auction market theory.

## Objective
Design, optimize, and backtest an automated trading system for NQ and ES micro futures that achieves:
- Positive expectancy
- Prop firm consistency compliance
- Max $4,000 drawdown
- First payout target: $9,000

## Data Available
- Location: csv/ folder
- Instruments: NQ, ES, YM (focus on NQ and ES)
- Timeframe: 1-minute bars
- Date range: Feb 5-16, 2026 (~11 days, ~9k bars/instrument)
- Additional data: Can pull up to 90 days as needed
- Schema: NinjaDataExport/v2.3, volumetric=True

Columns:
timestamp, instrument, period, open, high, low, close, volume, ema20, ema50, ema200, 
rsi14, atr14, vwap, vwap_upper1, vwap_upper2, vwap_upper3, vwap_lower1, vwap_lower2, 
vwap_lower3, vol_ask, vol_bid, vol_delta, session_date

## Target Prop Firm: TradeDay
- Account: $150,000
- Max Drawdown: $4,000 (static/EOD)
- Profit Target: $9,000 (6%)
- Consistency Rule: 30% during eval, none when funded
- Profit Split: 100% first $10K, then 90%

================================================================================
PART 1: ORDER FLOW FEATURE ENGINEERING
================================================================================

### 1.1 Delta Features
| Feature | Formula | Description |
|---------|---------|-------------|
| delta | vol_ask - vol_bid | Per-bar aggressive pressure |
| delta_pct | abs(delta) / volume | Normalized imbalance 0-100% |
| delta_zscore | (delta - mean) / std | Statistical outlier |
| delta_percentile | rank(delta, last 20) | Relative strength |
| cumulative_delta | sum(delta) | Running total |
| cumulative_delta_ma | EMA(20) of CVD | Smoothed trend |
| delta_divergence | price vs CVD direction | Reversal signal |

### 1.2 Imbalance Features
| Feature | Formula | Values to Test |
|---------|---------|----------------|
| imbalance_ratio | ask_vol / bid_vol | 1.5:1, 2:1, 3:1 |
| imbalance_pct | (ask - bid) / (ask + bid) | 20%, 30%, 40% |
| imbalance_mode | single, 3-bar sum, CVD slope | Compare all |
| min_delta | abs(delta) | 25, 50, 75, 100 |

### 1.3 Volume Features
| Feature | Formula | Threshold |
|---------|---------|-----------|
| volume_spike | vol / rolling_avg_vol | 1.5x, 2.0x, 2.5x |
| rolling_volume_mean | SMA(20) of volume | Context |

### 1.4 Aggregation Testing
Test signals at multiple timeframes:
- 1-min raw (fastest signals)
- 3-min aggregate (reduced noise)
- 5-min aggregate (clearer structure)
Compare signal clarity, drawdown, and expectancy.

================================================================================
PART 2: AUCTION MARKET (DALTON) FEATURES
================================================================================

### 2.1 Opening Range (IB) - From Playbook
| Feature | Formula |
|---------|---------|
| ib_high | First 60-min high |
| ib_low | First 60-min low |
| ib_range | ib_high - ib_low |
| ib_extension | Price beyond IB |
| trend_strength | Weak/Moderate/Strong/Super |

### 2.2 Volume Profile
| Feature | Formula |
|---------|---------|
| poc | Price with max volume |
| vah, val | 70% volume area bounds |
| hvn | Local volume maxima (support zones) |
| lvn | Local volume minima (liquidity voids) |

### 2.3 Day Type Classification - From Playbook
| Day Type | Criteria |
|----------|----------|
| Super Trend | IB ext > 2x, DPOC > 300 pts |
| Trend | IB ext > 1.5x, DPOC > 100 pts |
| P-Day | TPO skew > 0.6, DPOC > 50 pts |
| B-Day | IB ext < 0.3 |
| Neutral | All other |

### 2.4 Trend Strength Classification - From Playbook
| Strength | Criteria |
|----------|----------|
| Weak | < 0.5x IB extension |
| Moderate | 0.5-1.0x IB extension |
| Strong | 1.0-2.0x IB extension |
| Super | > 2.0x IB extension |

### 2.5 Fair Value Gaps (FVG)
| Feature | Formula |
|---------|---------|
| fvg_high | min(high[1], high[2]) - low[0] |
| fvg_low | high[0] - max(low[1], low[2]) |
| fvg_unfilled | Boolean - gap still exists |

================================================================================
PART 3: STOP-LOSS COMPARISON (CORE A/B TEST)
================================================================================

### 3.1 Methods to Test
| Method | Description | Priority |
|--------|-------------|----------|
| LVN/HVN | Nearest volume node | 1 (Primary) |
| FVG | Fair value gap fill | 2 |
| IBH/IBL | Opening range extreme | 3 |
| BPR | Balanced price range | 4 |
| ATR 0.5x | Stop at 0.5 × ATR | Comparison |
| ATR 1.0x | Stop at 1.0 × ATR | Comparison |
| ATR 2.0x | Stop at 2.0 × ATR | Comparison |

### 3.2 Hybrid Approach (Structure + ATR Cap)
| Condition | Action |
|-----------|--------|
| Structure stop ≤ 1× ATR | A-setup: $400 risk, full size |
| Structure stop 1-2× ATR | B-setup: $200 risk, half size |
| Structure stop > 2× ATR | Skip OR size down proportionally |

### 3.3 Day-Type Specific Stop Rules
| Day Type | Max Stop | Action |
|----------|----------|--------|
| Super/Strong Trend | ≤ 2× ATR | Full size, wide stops OK |
| Moderate Trend | ≤ 1.5× ATR | Full size |
| P-Day | ≤ 1× ATR | Full size |
| B-Day/Neutral | ≤ 0.5× ATR | Micro ($100) or skip |

================================================================================
PART 4: RISK MANAGEMENT
================================================================================

### 4.1 Position Sizing Formula
```
risk_dollars = min(max_risk, account × risk_pct)
contracts = risk_dollars / (atr × tick_value)
```

### 4.2 Risk Tiers
| Setup | Criteria | Risk | Position |
|-------|----------|------|----------|
| A+ | Stop ≤ 0.5× ATR + HTF align | $400 | 100% |
| A | Stop ≤ 1× ATR + structure | $400 | 100% |
| B | Stop 1-2× ATR | $200 | 50% |
| C | Stop > 2× ATR | Skip/size down | 0-25% |

### 4.3 Daily Limits
| Limit | Amount |
|-------|--------|
| Max Drawdown | $4,000 |
| Max Day Loss | $2,000 |
| Max Trades | 10/day |
| Consecutive Losses | 3 → stand down |

### 4.4 Trail Stops
| Profit | Action |
|--------|--------|
| +150 pts (NQ) | Move to BE + 30-min trail |
| +300 pts (NQ) | Lock 33% + runner |

================================================================================
PART 5: A/B TESTING FRAMEWORK
================================================================================

### 5.1 Layer Comparison (Sequential)
| Layer | Strategy | Purpose |
|-------|----------|---------|
| 0 | Delta only | Baseline expectancy |
| 1 | + IB direction | Remove counter-trend trades |
| 2 | + Day type filter | Skip B-Day/Neutral |
| 3 | + VWAP context | Better R:R at mean reversion |
| 4 | + TPO/excess | Avoid exhaustion entries |
| 5 | + HTF trend | Align with larger timeframe |
| 6 | + FVG confirmation | Better timing |

### 5.2 Parameter Sweeps (Independent)
Test each parameter, find optimal, then combine best:
- Delta z-score: 1.5, 2.0, 2.5, 3.0
- Delta percentile: 70%, 80%, 90%
- Imbalance ratio: 1.5:1, 2:1, 3:1
- Volume spike: 1.5x, 2.0x, 2.5x
- Min delta: 25, 50, 75, 100

### 5.3 Timeframe Comparison
- 1-min raw
- 3-min aggregate
- 5-min aggregate
Compare: Trades, Win Rate, Expectancy, Drawdown

================================================================================
PART 6: SUCCESS CRITERIA
================================================================================

| Metric | Minimum | Target | Rationale |
|--------|---------|--------|-----------|
| Expectancy | >$0 | >$15/trade | Must be positive first |
| Win Rate | >45% | 50-55% | Survival floor |
| Profit Factor | >1.0 | >1.3 | Good edge |
| Max Drawdown | <$4,000 | <$3,000 | Hard limit |
| Sample Size | 100 trades | 200+ trades | Statistical validity |
| Best Day | <$2,700 (30%) | <$2,000 | Consistency rule |
| Avg Risk:Reward | >1.5:1 | 2:1 | Compensate for losses |

================================================================================
PART 7: REPORTING REQUIREMENTS
================================================================================

### 7.1 Layer Comparison Report
| Layer | Trades | Win Rate | Expectancy | Profit Factor | Max DD | Sharpe |
|-------|--------|----------|------------|---------------|--------|--------|
| Layer 0 | | | | | | |
| Layer 1 | | | | | | |
| ... | | | | | | |

### 7.2 Stop Comparison Report
| Method | Trades | Avg Stop | Win Rate | Expectancy | Max DD |
|--------|--------|----------|----------|------------|--------|
| LVN | | | | | |
| FVG | | | | | |
| ATR 1x | | | | | |
| Hybrid | | | | | |

### 7.3 Timeframe Comparison
| Timeframe | Trades | Win Rate | Expectancy | Avg Hold | Max DD |
|-----------|--------|----------|------------|----------|--------|
| 1-min | | | | | |
| 3-min | | | | | |
| 5-min | | | | | |

### 7.4 Direction Analysis (Separate for NQ and ES)
| Direction | Trades | Win Rate | Expectancy | Profit Factor |
|-----------|--------|----------|------------|---------------|
| Long | | | | |
| Short | | | | |

### 7.5 Daily Performance
| Date | Trades | P&L | Cum P&L | Drawdown | Day Type |
|------|--------|-----|---------|----------|----------|
| ... | | | | | |

### 7.6 Equity Curve
Cumulative P&L chart with drawdown overlay and key annotations

### 7.7 Final Recommendations (Evidence-Based)
- Best Layer Configuration
- Best Stop Method
- Best Timeframe
- Direction Bias (Long/Short/Both)
- Instrument Preference (NQ/ES)

================================================================================
PART 8: IMPLEMENTATION ARCHITECTURE
================================================================================

```
order_flow_strategy/
├── data/
│   ├── loader.py              # CSV loader
│   └── transformer.py         # Feature engineering
├── features/
│   ├── order_flow.py          # Delta, CVD, imbalances
│   ├── volume_profile.py      # POC, VAH, VAL, HVN, LVN
│   ├── day_type.py           # IB, classification
│   ├── vwap.py               # VWAP analysis
│   └── fvg.py                # Fair Value Gaps
├── signals/
│   ├── entry_signals.py      # Layered signal generation
│   ├── filters.py            # Filter implementations
│   └── stop_calculator.py    # Stop comparison
├── backtest/
│   ├── engine.py             # Backtest engine
│   ├── metrics.py            # Performance metrics
│   └── comparison.py        # A/B comparison
├── risk/
│   ├── position_sizer.py    # ATR-based sizing
│   └── risk_manager.py       # Daily limits
├── analyze/
│   ├── results.py           # Results aggregation
│   └── visualization.py     # Charts and reports
└── run.py                   # Main entry
```

### Libraries
- pandas, numpy
- backtesting.py or custom vectorized backtest
- matplotlib/seaborn for charts

================================================================================
PART 9: KEY REFERENCES
================================================================================

### Bookmap (Order Flow)
- https://bookmap.com/blog/can-real-time-order-flow-give-you-an-edge-in-scalp-trading

### Dalton Playbook
- See prompts/playbooks.md for full day-type classification and trade management

### Prop Firm Rules
- TradeDay: Static drawdown, 30% consistency during eval, none when funded

================================================================================
INSTRUCTIONS
================================================================================

1. Load and explore the data (NQ, ES)
2. Compute all order flow features (delta, imbalance, volume)
3. Compute all auction market features (IB, volume profile, day type)
4. Run baseline (Layer 0) backtest
5. Test each layer sequentially, compare metrics
6. Compare stop methods (LVN vs FVG vs ATR)
7. Optimize delta/imbalance parameters
8. Compare timeframes (1-min vs 3-min vs 5-min)
9. Generate comprehensive A/B test reports
10. Document best configuration with evidence

Remember: Let the DATA guide you. Test multiple hypotheses. Present EVIDENCE 
for every recommendation. Be autonomous in testing but document all findings.

DO NOT CODE without explicit permission from the user. Wait for data acquisition 
to complete before testing.
