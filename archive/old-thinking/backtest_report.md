# BACKTEST RESULTS REPORT
## Order Flow Strategy - NQ Futures

**Date**: 2026-02-16
**Data Range**: Nov 2025 - Feb 2026 (~62 trading days)
**Instrument**: NQ (Nasdaq), ES (S&P 500)

---

## EXECUTIVE SUMMARY

| Metric | Value |
|--------|-------|
| **Best Strategy** | Delta Percentile > 95% |
| **Win Rate** | 50.4% |
| **Expectancy** | $1.27/trade |
| **Profit Factor** | 1.14 |
| **Total Trades** | 838 |
| **Instrument** | NQ preferred over ES |

---

## LAYER COMPARISON (NQ)

| Layer | Strategy | Trades | WR% | Exp$ | PF |
|-------|----------|--------|-----|------|-----|
| 0 | Delta only (Pct>80) | 2145 | 48.4 | $0.20 | 1.03 |
| 1 | + IB Direction | 516 | 43.4 | -$1.48 | 0.81 |
| 2 | + Day Type | 385 | 45.2 | -$0.76 | 0.92 |
| 3 | + VWAP | 385 | 45.2 | -$0.76 | 0.92 |

**Finding**: IB filter makes strategy WORSE

---

## PARAMETER SWEEP RESULTS

### Delta Percentile Threshold

| Threshold | Trades | WR% | Exp$ | PF |
|-----------|--------|-----|------|-----|
| >60% | 2989 | 47.2 | -$0.31 | 0.98 |
| >70% | 2636 | 48.3 | $0.10 | 1.02 |
| >80% | 2145 | 48.4 | $0.20 | 1.03 |
| >85% | 1806 | 49.0 | $0.69 | 1.08 |
| >90% | 1392 | 49.2 | $0.99 | 1.11 |
| **>95%** | **838** | **50.4** | **$1.27** | **1.14** |

**Finding**: Higher delta threshold = better performance

---

### Volume Spike Filter

| Filter | Trades | WR% | Exp$ | PF |
|--------|--------|-----|------|-----|
| None (Pct>85) | 1806 | 49.0 | $0.69 | 1.08 |
| >1.2x | 950 | 50.1 | $0.97 | 1.11 |
| >1.5x | 544 | 52.4 | $0.85 | 1.10 |
| >2.0x | 217 | 47.9 | -$0.46 | 0.97 |

**Finding**: Volume spike 1.2x improves expectancy

---

### Time of Day

| Hour | Trades | WR% | Exp$ | PF |
|------|--------|-----|------|-----|
| 10:00-11:00 | 322 | 52.2 | $1.04 | 1.10 |
| 11:00-12:00 | 332 | 50.3 | $1.13 | 1.14 |
| 12:00-13:00 | 324 | 48.8 | $0.14 | 1.03 |
| 13:00-14:00 | 310 | 49.0 | -$0.21 | 1.00 |
| 14:00-15:00 | 316 | 47.5 | -$1.61 | 0.85 |

**Finding**: 10:00-12:00 is best; 14:00-15:00 is worst

---

## INSTRUMENT COMPARISON

| Instrument | Strategy | Trades | WR% | Exp$ | PF |
|------------|----------|--------|-----|------|-----|
| **NQ** | Pct>95 | 838 | 50.4 | $1.27 | 1.14 |
| ES | Pct>95 | 828 | 48.1 | -$0.08 | 1.01 |

**Finding**: NQ significantly outperforms ES for this strategy

---

## KEY INSIGHTS

1. **Delta Percentile > 95%** is the best single parameter
2. **Volume spike filter (1.2x)** improves performance
3. **10:00-12:00** is the best trading window
4. **IB filter hurts performance** - do not use
5. **NQ > ES** for this strategy
6. **Profit factor > 1.0** is achievable

---

## RECOMMENDED STRATEGY

```
Entry Conditions:
- Delta percentile > 95%
- Volume spike > 1.2x (optional)
- Time: 10:00 - 12:00 (optional)

Exit Rules:
- Stop: 1x ATR
- Target: 2x ATR (1:2 R:R)
- Max bars held: 5

Position Sizing:
- Risk: $400/trade max
- Adjust for ATR
```

---

## PROJECTED PERFORMANCE

| Metric | Value |
|--------|-------|
| Expected trades/day | ~14 |
| Expectancy | $1.27/trade |
| Expected daily P&L | ~$18/day |
| Days to $9K target | ~500 days |

**Note**: With 62 days of data and 838 trades, more data needed for statistical significance.

---

## NEXT STEPS

1. [ ] Test with more historical data
2. [ ] Implement proper risk management
3. [ ] Add position sizing
4. [ ] Test on live paper trading
5. [ ] Consider combining with day type filter for entry timing

---

*Report generated from backtest_engine.py*
