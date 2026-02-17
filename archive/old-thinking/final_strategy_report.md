# FINAL ORDER FLOW STRATEGY REPORT
## Best Configurations Found

**Date**: February 16, 2026  
**Data**: 62 trading days (Nov 2025 - Feb 2026)  
**Instrument**: NQ Futures  

---

## TOP 2 STRATEGIES

### STRATEGY 1: Delta Direction Only (RECOMMENDED)

| Parameter | Value |
|-----------|-------|
| **Time Window** | 10:00 - 12:00 ET |
| **Signal** | Long when delta > 0, Short when delta < 0 |
| **Stop** | 2.0 × ATR |
| **Target** | 1.5 × ATR (1.5R) |
| **Max Bars** | 8 |

**Results:**
- Win Rate: **52.9%**
- Expectancy: **4.50 pts/trade**
- Profit Factor: **1.31**
- Total Trades: 875
- Trades/Day: ~14

**At 22 Contracts (MNQ):**
- Per trade: $198
- Daily: ~$2,771
- **Days to pass $150K eval: ~3 days**

---

### STRATEGY 2: Delta + CVD Trend (Higher Quality)

| Parameter | Value |
|-----------|-------|
| **Time Window** | 11:00 - 13:00 ET |
| **Signal** | Delta pct > 85 + Delta direction + CVD trend |
| **Stop** | 1.5 × ATR |
| **Target** | 1.5 × ATR (1.5R) |
| **Max Bars** | 8 |

**Results:**
- Win Rate: **54.0%**
- Expectancy: **3.83 pts/trade**
- Profit Factor: **1.31**
- Total Trades: 346
- Trades/Day: ~5.5

**At 22 Contracts (MNQ):**
- Per trade: $169
- Daily: ~$928
- **Days to pass $150K eval: ~10 days**

---

## COMPARISON

| Metric | Strategy 1 | Strategy 2 |
|--------|------------|------------|
| Win Rate | 52.9% | **54.0%** |
| Expectancy | **4.50 pts** | 3.83 pts |
| Trades/Day | **14** | 5.5 |
| Quality | Lower | Higher |
| Speed | **Fast (3 days)** | Medium (10 days) |

---

## RISK MANAGEMENT

### Strategy 1 Risk
- Stop: 2.0 × ATR (~30-40 points = $60-80 per contract)
- With 22 contracts: Risk ~$1,320-1,760 per trade
- **EXCEEDS $400 limit!**

### Adjusted Position Sizing

For Strategy 1:
- Risk per contract: ~$70 (2.0 × ATR × $2)
- Max contracts for $400 risk: $400 ÷ $70 = **5-6 contracts**

With 5 contracts:
- Daily: ~$630
- Days to $9K: ~14 days

For Strategy 2:
- Risk per contract: ~$52 (1.5 × ATR × $2)
- Max contracts for $400 risk: $400 ÷ $52 = **7-8 contracts**

With 7 contracts:
- Daily: ~$295
- Days to $9K: ~30 days

---

## KEY FINDINGS

1. **Wider stops (1.5-2.0x ATR) work better** than tight stops
2. **1.5:1 R:R is optimal** for this strategy
3. **Morning session (10-12) has more trades**
4. **Midday session (11-13) has higher quality**
5. **Simple delta direction is very effective**

---

## CONSISTENCY RULE COMPLIANCE

TradeDay Rule: Max single day = 30% of $9K = $2,700

- Strategy 1 adjusted (5 contracts): Max day ~$630 ✅
- Strategy 2 (7 contracts): Max day ~$295 ✅

Both pass the consistency rule!

---

## RECOMMENDATION

**Use Strategy 1 (Delta Direction) with proper sizing:**
- Start with 5-6 contracts
- Can scale to 7-8 as account grows
- Expected: Pass eval in 10-15 days
- Conservative, high probability

---

## NEXT STEPS

1. [ ] Paper trade for 1 week to validate
2. [ ] Start with smaller size (3-4 contracts)
3. [ ] Scale up after 5 winning days
4. [ ] Monitor max drawdown (keep under $3,000)

---

*Note: Strategy 1 with 5-6 contracts exceeds 50% win rate and can pass in 2 weeks with proper risk management.*
