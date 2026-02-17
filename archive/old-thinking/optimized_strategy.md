# OPTIMIZED ORDER FLOW STRATEGY - FINAL RESULTS

## BEST PARAMETERS FOUND

| Parameter | Value |
|-----------|-------|
| **Entry Signal** | Delta percentile > 95% |
| **Stop** | 0.6 × ATR |
| **Max Bars Held** | 8 bars |
| **Time Window** | 13:00 - 15:00 ET |
| **Instrument** | NQ (Nasdaq) |

---

## PERFORMANCE METRICS

| Metric | Value |
|--------|-------|
| **Win Rate** | 41.8% |
| **Expectancy** | **$3.37/trade** |
| **Profit Factor** | **1.43** |
| **Risk:Reward** | **1.99:1** |
| **Total Trades** | 285 |
| **Data Days** | 62 |

---

## COMPARISON: BEFORE vs AFTER OPTIMIZATION

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Expectancy | $1.27 | $3.37 | **+165%** |
| Profit Factor | 1.14 | 1.43 | **+25%** |
| R:R | 1.10:1 | 1.99:1 | **+81%** |
| Time Window | All day | 13:00-15:00 | Better quality |

---

## KEY FINDINGS

### 1. Time of Day (CRITICAL)
| Window | Expectancy | PF |
|--------|------------|-----|
| 9:30-11:00 | -$1.50 | 0.91 |
| 10:00-12:00 | +$2.68 | 1.22 |
| 13:00-15:00 | **+$3.19** | **1.46** |

**Finding**: Afternoon session is significantly better

### 2. Stop Size
| Stop | Expectancy | PF |
|------|------------|-----|
| 0.5x ATR | +$3.22 | 1.46 |
| **0.6x ATR** | **+$3.37** | **1.43** |
| 0.75x ATR | +$1.14 | 1.13 |
| 1.0x ATR | +$1.41 | 1.15 |

**Finding**: 0.6x ATR is optimal

### 3. Max Bars Held
| Bars | Expectancy | PF |
|------|------------|-----|
| 5 | +$1.58 | 1.19 |
| **8** | **+$3.37** | **1.43** |
| 10 | +$3.19 | 1.46 |
| 15 | +$3.15 | 1.45 |

**Finding**: 8 bars is optimal

### 4. Delta Threshold
| Threshold | Expectancy | PF |
|-----------|------------|-----|
| >85% | +$0.28 | 1.04 |
| >90% | +$1.06 | 1.13 |
| **>95%** | **+$3.37** | **1.43** |

**Finding**: Higher threshold = better

---

## PROJECTED DAILY PERFORMANCE

Based on 62 days / 285 trades:

| Metric | Value |
|--------|-------|
| Avg trades/day | ~4.6 |
| Expected daily P&L | ~$15.50 |
| Days to $9K target | ~580 days |

**Note**: With better parameters and afternoon session, actual may differ.

---

## RISK MANAGEMENT

| Rule | Value |
|------|-------|
| Max risk/trade | $400 |
| Stop | 0.6 × ATR |
| Target | 1.2 × ATR (2× risk) |
| Max bars | 8 |
| Session | 13:00-15:00 only |

---

## ENTRY RULES

```
1. Time: 13:00 - 15:00 ET
2. Delta percentile > 95%
3. Delta > 0 (long) or Delta < 0 (short)
4. Entry: Next bar close after signal
5. Stop: 0.6 × ATR
6. Target: 2× ATR (1.2 × stop)
7. Exit: Stop, target, or 8 bars
```

---

## NEXT STEPS

1. [ ] Test on more data (90+ days)
2. [ ] Paper trade to validate
3. [ ] Add position sizing
4. [ ] Track daily drawdown
5. [ ] Consider combining with volume spike filter

---

*Generated: 2026-02-16*
*Data: NQ Futures, Nov 2025 - Feb 2026*
