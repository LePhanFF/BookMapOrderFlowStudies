# Order Flow Strategy Design Document
## Version 1.3 - Prop Firm Optimized

**Date**: February 16, 2026
**Target**: TradeDay $150K Account
**Objective**: First payout of $9,000

---

## 1. Executive Summary

This document outlines a quantitative trading strategy combining:
1. **Order Flow Analysis** (Bookmap-style: delta, imbalances, CVD)
2. **Auction Market Theory** (Dalton: volume profile, day types, IB)
3. **Prop Firm Compliance** (TradeDay rules: $4K drawdown, consistency)

The strategy will be tested using A/B comparison to find optimal parameters.

---

## 2. Data Specification

| Attribute | Value |
|-----------|-------|
| Instruments | NQ (Nasdaq), ES (S&P 500) |
| Timeframe | 1-minute bars |
| Current Data | ~11 days (Feb 5-16, 2026) |
| Available Data | Up to 90 days (as needed) |
| Source | NinjaTrader Volumetric Export |

---

## 3. Feature Engineering

### 3.1 Order Flow Features

| Feature | Calculation | Purpose |
|---------|-------------|---------|
| `delta` | vol_ask - vol_bid | Aggressive pressure |
| `delta_pct` | abs(delta) / volume | Normalized 0-100% |
| `delta_zscore` | (delta - mean) / std | Outlier detection |
| `delta_percentile` | rank(last 20 bars) | Relative strength |
| `cumulative_delta` | running sum | Trend direction |
| `delta_divergence` | price vs CVD | Reversal signal |
| `imbalance_ratio` | ask / bid | Strength >2:1 |
| `volume_spike` | vol / SMA(20) | Unusual activity |

### 3.2 Imbalance Modes

| Mode | Definition |
|------|------------|
| Single Bar | Current bar delta only |
| 3-Bar Sum | Sum of last 3 bars |
| CVD Slope | Rate of cumulative delta change |

### 3.3 Parameters to Optimize

| Parameter | Test Values |
|-----------|-------------|
| Delta Z-Score | 1.5, 2.0, 2.5, 3.0 |
| Delta Percentile | 70%, 80%, 90% |
| Imbalance Ratio | 1.5:1, 2:1, 3:1 |
| Volume Spike | 1.5x, 2.0x, 2.5x |
| Min Delta | 25, 50, 75, 100 |

---

## 4. Auction Market Features

### 4.1 Opening Range (IB)

| Feature | Calculation |
|---------|-------------|
| ib_high | First 60-minute high |
| ib_low | First 60-minute low |
| ib_range | ib_high - ib_low |
| ib_extension | Price beyond IB |

### 4.2 Day Type Classification

| Day Type | Criteria |
|----------|----------|
| Super Trend | IB ext > 2×, DPOC > 300 pts |
| Trend | IB ext > 1.5×, DPOC > 100 pts |
| P-Day | TPO skew > 0.6, DPOC > 50 pts |
| B-Day | IB ext < 0.3 |
| Neutral | All other |

### 4.3 Volume Profile

| Feature | Purpose |
|---------|---------|
| POC | Point of control (fair value) |
| VAH/VAL | Value area high/low (70%) |
| HVN | High volume node (support/resistance) |
| LVN | Low volume node (liquidity void) |

### 4.4 Fair Value Gaps

| Feature | Calculation |
|---------|-------------|
| Bull FVG | min(high[1],high[2]) - low[0] |
| Bear FVG | high[0] - max(low[1],low[2]) |

---

## 5. A/B Testing Framework

### 5.1 Layer Comparison

| Layer | Strategy | Filter Added |
|-------|----------|--------------|
| 0 | Delta only | None |
| 1 | + IB Direction | Trade in IB break direction |
| 2 | + Day Type | Skip B-Day/Neutral |
| 3 | + VWAP | Price vs VWAP bands |
| 4 | + TPO Excess | Avoid exhaustion |
| 5 | + HTF Trend | Daily EMA alignment |
| 6 | + FVG | Confirm with gap fill |

### 5.2 Stop Comparison

| Method | Priority | Description |
|--------|----------|-------------|
| LVN | 1 | Nearest low volume node |
| FVG | 2 | Fair value gap |
| IBH/IBL | 3 | Opening range extreme |
| BPR | 4 | Prior balance range |
| ATR 1× | Baseline | 1× ATR stop |

### 5.3 Timeframe Comparison

| Timeframe | Aggregation | Use Case |
|-----------|-------------|----------|
| 1-min | Raw | Fastest signals |
| 3-min | 3-bar sum | Moderate filtering |
| 5-min | 5-bar sum | Clearest structure |

---

## 6. Risk Management

### 6.1 Position Sizing

```
risk_dollars = min($400, account × 0.25%)
contracts = risk_dollars / (atr × tick_value)
```

### 6.2 Risk Tiers

| Setup | Criteria | Risk | Position |
|-------|----------|------|----------|
| A+ | Stop ≤ 0.5× ATR + HTF align | $400 | 100% |
| A | Stop ≤ 1× ATR + structure | $400 | 100% |
| B | Stop 1-2× ATR | $200 | 50% |
| C | Stop > 2× ATR | Skip/size down | 0-25% |

### 6.3 Daily Limits

| Limit | Amount |
|-------|--------|
| Max Drawdown | $4,000 |
| Max Day Loss | $2,000 |
| Max Trades | 10 |
| Consecutive Losses | 3 → stand down |

### 6.4 Trail Rules

| Profit Level | Action |
|--------------|--------|
| +150 pts (NQ) | Move to BE + 30-min trail |
| +300 pts (NQ) | Lock 33% + runner |

---

## 7. Success Criteria

| Metric | Minimum | Target |
|--------|---------|--------|
| Expectancy | > $0 | > $15/trade |
| Win Rate | > 45% | 50-55% |
| Profit Factor | > 1.0 | > 1.3 |
| Max Drawdown | < $4,000 | < $3,000 |
| Sample Size | 100 trades | 200+ |
| Best Day | < $2,700 | < $2,000 |

---

## 8. Reporting

### 8.1 Required Reports

1. **Layer Comparison** - Trades, WR, Expectancy, PF, DD, Sharpe
2. **Stop Comparison** - Avg stop, WR, Expectancy, DD
3. **Timeframe Comparison** - 1/3/5-min performance
4. **Direction Analysis** - Long vs Short per instrument
5. **Daily Performance** - Date, trades, P&L, DD, day type
6. **Equity Curve** - Cumulative with drawdown overlay

### 8.2 Final Output

- Best layer configuration
- Best stop method
- Best timeframe
- Direction bias recommendation
- Instrument preference (NQ/ES)

---

## 9. Evidence-Based Decisions

### Principles

1. **Let data guide** - Test all hypotheses with evidence
2. **No curve fitting** - Use walk-forward validation
3. **Statistical significance** - Require 100+ trades
4. **Consistency** - Same performance across days

### Decision Priority

1. Stop method (biggest impact on DD)
2. Day type filter (biggest impact on WR)
3. Delta parameters (fine-tuning)
4. Timeframe (confirmation)

---

## 10. Implementation Status

| Phase | Status |
|-------|--------|
| Data Loading | Pending (waiting for 60-day data) |
| Feature Engineering | Pending |
| Layer 0 Baseline | Pending |
| Layer Comparisons | Pending |
| Stop Comparison | Pending |
| Parameter Optimization | Pending |
| Final Report | Pending |

---

## 11. Questions for User

1. **Data readiness**: When will 60-day data be available?
2. **Testing start**: Should I notify you before running backtests?
3. **Priority**: Any specific comparison to prioritize first?

---

*Document Version: 1.3*
*Last Updated: February 16, 2026*
*Status: Ready for data acquisition completion*
