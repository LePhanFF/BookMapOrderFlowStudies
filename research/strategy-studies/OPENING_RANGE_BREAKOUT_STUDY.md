# Opening Range Breakout (ORB) Strategy Study
## Mechanical Breakout Trading for Prop Firms

**Status**: 📋 RESEARCH PHASE  
**Instruments**: NQ, ES (MNQ, MES for sizing)  
**Target Win Rate**: 55%+  
**Target R:R**: 2:1+  
**Risk per Trade**: $400  
**Study Date**: February 2026  

---

## 1. Strategy Overview

### Core Concept
Trade the breakout of the opening range (first X minutes of trading session) when price breaks above/below the established range with momentum.

### Why It Works
1. **Opening range sets the tone** for the day
2. **Institutional participation** is highest at open
3. **Clear levels** to trade against (high/low of range)
4. **Mechanical** - no interpretation needed
5. **Proven** - up 400% in some variations

---

## 2. Entry Rules (Mechanical)

### Opening Range Definition
```
Time Window Options:
- 5-minute ORB (9:30-9:35)
- 15-minute ORB (9:30-9:45)
- 30-minute ORB (9:30-10:00)
- 60-minute ORB (9:30-10:30) - Traditional

Recommended: 15-minute ORB (best balance)
```

### Setup Conditions

**Long Entry:**
```
IF:
  1. Time is within 15 minutes of open (9:30-9:45)
  2. Price breaks above Opening Range High (ORH)
  3. Breakout bar closes above ORH
  4. Volume > average (confirming breakout)
  5. Optional: VIX < 25 (avoid chop)
  
THEN:
  Enter LONG at market or limit (breakout bar close + 1 tick)
  Stop: Below Opening Range Low (ORL)
  Target: 2:1 R:R minimum
```

**Short Entry:**
```
IF:
  1. Time is within 15 minutes of open
  2. Price breaks below Opening Range Low (ORL)
  3. Breakout bar closes below ORL
  4. Volume > average
  
THEN:
  Enter SHORT at market or limit
  Stop: Above Opening Range High (ORH)
  Target: 2:1 R:R minimum
```

### Trend Filter (Optional but Recommended)
```
Only take breakouts IN DIRECTION of:
  - Premarket trend (if clear)
  - Overnight high/low breaks
  - Previous day close direction
  
OR use HTF trend:
  - Price above 5-min EMA20 = bullish bias
  - Price below 5-min EMA20 = bearish bias
```

---

## 3. Exit Rules

### Stop Loss (Hard)
```
Stop Location: Opposite side of opening range
  - Longs: Stop below ORL
  - Shorts: Stop above ORH
  
Stop Distance: ORH - ORL (the range itself)
  
Example:
  - ORH: 4500.00
  - ORL: 4495.00
  - Range: 5.00 points
  - Long entry: 4500.25
  - Stop: 4494.75 (below ORL)
  - Risk: 5.50 points
```

### Profit Targets

**Conservative (2:1 R:R)**
```
Target = Entry + (2 × Risk)
Example:
  - Risk: 5.50 points
  - Target: 11.00 points
  - MNQ value: $22/point = $242 profit
```

**Aggressive (Extended targets)**
```
Target 1: 2:1 (exit 50%)
Target 2: 3:1 (exit 25%)
Target 3: 4:1 (runner, trail stop)
```

### Time Exit
```
If position open at 11:00 AM (90 min after open):
  - Exit at market
  - Take whatever P&L
  - Don't hold through lunch
```

### Trailing Stop (Optional)
```
Once in profit > 1R:
  - Move stop to breakeven
  - Trail with 20-period EMA
  - Exit on EMA cross
```

---

## 4. Position Sizing

### MNQ (Micro Nasdaq)
```
Range: 5-15 points typical
Risk per trade: $400

Sizing:
  - If range = 10 points
  - Risk = 10 pts × $2/pt = $20 per contract
  - Contracts = $400 ÷ $20 = 20 contracts
  
Example Trade:
  - Entry: 4500.00
  - Stop: 4490.00 (10 pts)
  - Target: 4520.00 (20 pts, 2:1)
  - Contracts: 20 MNQ
  - Risk: $400
  - Target: $800
```

### MES (Micro S&P)
```
Range: 3-8 points typical
Risk per trade: $400

Sizing:
  - If range = 5 points
  - Risk = 5 pts × $5/pt = $25 per contract
  - Contracts = $400 ÷ $25 = 16 contracts
```

### Dynamic Sizing
```
Range > 15 points (high volatility):
  - Reduce size by 50%
  - OR skip trade
  
Range < 5 points (low volatility):
  - Increase size by 25%
  - OR require 3:1 R:R
```

---

## 5. Instrument Comparison

| Instrument | Avg Range | Tick Value | Contracts for $400 | Best For |
|------------|-----------|------------|-------------------|----------|
| **MNQ** | 8-12 pts | $2 | 20-25 | High volatility |
| **MES** | 4-7 pts | $5 | 12-16 | Lower volatility |
| **NQ** | 8-12 pts | $20 | 2-3 | Large accounts |
| **ES** | 4-7 pts | $50 | 1-2 | Large accounts |

**Recommendation**: MNQ for prop firms (best volatility, manageable sizing)

---

## 6. Parameters to Test

### A. Opening Range Time
```
1. 5-minute ORB (aggressive, more signals)
2. 15-minute ORB (balanced)
3. 30-minute ORB (conservative, fewer signals)
4. 60-minute ORB (traditional, most conservative)
```

### B. Entry Confirmation
```
1. Close beyond range (wait for bar close)
2. Immediate breakout (enter on touch)
3. 2-bar confirmation (2 closes beyond)
4. Volume confirmation (break + volume spike)
```

### C. Trend Filter
```
1. No filter (take all breakouts)
2. Premarket trend only
3. 5-min EMA20 filter
4. Daily trend alignment
5. VIX filter (avoid VIX > 25)
```

### D. False Breakout Filter
```
1. None (take every breakout)
2. Reject if breakout < 2 ticks beyond range
3. Reject if immediate return to range
4. Require 5-min hold beyond range
```

---

## 7. Expected Performance (Hypothesis)

Based on reported results:

| Metric | Conservative | Target | Aggressive |
|--------|-------------|--------|-----------|
| **Win Rate** | 45% | 55% | 65% |
| **Avg Win** | 15 pts | 20 pts | 25 pts |
| **Avg Loss** | 10 pts | 10 pts | 10 pts |
| **R:R Ratio** | 1.5:1 | 2:1 | 2.5:1 |
| **Profit Factor** | 1.4 | 2.0 | 2.8 |
| **Trades/Day** | 1 | 1-2 | 2-3 |
| **Expectancy** | +$45 | +$110 | +$200 |

### Daily P&L Projection (MNQ)
```
Win Rate: 55%
Trades: 1.5/day
Avg Win: 20 pts × $2 × 20 ctr = $800
Avg Loss: 10 pts × $2 × 20 ctr = $400

Daily: (1.5 × 0.55 × $800) - (1.5 × 0.45 × $400) = $660 - $270 = $390/day

Monthly (20 days): $7,800
Pass $9K eval: 12 days
```

**Note**: Lower trade frequency than order flow, but higher win rate compensates.

---

## 8. ORB Variations to Test

### Variation A: Classic ORB
```
- 15-min range
- Enter on breakout
- Stop opposite side
- 2:1 target
```

### Variation B: ORB with Pullback
```
- Wait for breakout
- Wait for pullback to breakout level
- Enter on rejection
- Tighter stop
```

### Variation C: ORB + VWAP
```
- Only trade breakouts toward VWAP
  - Price below VWAP + breaks down = short
  - Price above VWAP + breaks up = long
- Aligns with intraday trend
```

### Variation D: ORB + Prior Day Levels
```
- Mark prior day high/low
- ORB breakout + break of prior day level = stronger signal
- Higher conviction, better R:R
```

### Variation E: Failed ORB (Fade)
```
- Breakout fails (returns to range within 5 min)
- Fade the breakout
- Stop beyond original breakout point
- Target: Other side of range
```

---

## 9. Advantages vs Order Flow Strategy

### Advantages
1. **Simpler**: No complex order flow calculations
2. **Higher win rate**: 55%+ vs 44%
3. **Clear levels**: Objective entry/stop
4. **Works in trending markets**: Captures momentum
5. **Less time intensive**: 1-2 trades vs 11 trades
6. **No data requirements**: Don't need Level 2

### Disadvantages
1. **Lower frequency**: 1-2 trades/day vs 11
2. **Misses chop**: Doesn't trade range-bound days
3. **Requires trend**: Fails in choppy/no-trend markets
4. **Gap risk**: Large gaps = larger stops
5. **Fixed time**: Must trade first 15-60 min

---

## 10. Implementation Plan

### Phase 1: Data Collection
```
- [ ] Get NQ/MNQ 1-min data (90 days)
- [ ] Calculate opening ranges for each day
- [ ] Mark breakout signals
- [ ] Track outcomes
```

### Phase 2: Backtest
```
- [ ] Test all 5 variations
- [ ] Optimize range time (5/15/30/60 min)
- [ ] Find best trend filters
- [ ] Compare to order flow results
```

### Phase 3: Paper Trade
```
- [ ] 2 weeks live market
- [ ] Verify opening range calculations
- [ ] Check breakout fills
- [ ] Measure slippage
```

### Phase 4: Live
```
- [ ] Trade first evaluation account
- [ ] Start with 15-minute ORB
- [ ] Scale up after validation
```

---

## 11. Risk Management

### ORB-Specific Risks

**False Breakout Risk**
```
- Price breaks range then immediately reverses
- Solution: Require close beyond range + volume
- Or: Wait for 5-min confirmation
```

**Gap Risk**
```
- Large overnight gap = huge opening range
- Solution: Skip if range > 2× ATR
- Or: Reduce size by 50%
```

**Chop Risk**
```
- No clear trend = whipsaws
- Solution: Add trend filter (EMA20)
- Skip if price oscillating around VWAP
```

### Daily Limits
```
Max Trades: 2/day (ORB only)
Max Loss: $800 (2 trades)
Time Limit: Done by 11:00 AM
No ORB: If VIX > 30 or premarket chop
```

---

## 12. Questions to Answer

1. **Best opening range time?** (5/15/30/60 min)
2. **Win rate with trend filter vs without?**
3. **Is MNQ or MES better for ORB?**
4. **How does ORB compare to order flow expectancy?**
5. **What is max drawdown with ORB?**
6. **Can we combine ORB + Order Flow?** (ORB breakout + Delta confirmation)
7. **Which variation is best?**

---

## 13. Comparison Summary

| Strategy | Win Rate | Trades/Day | Daily P&L | Complexity |
|----------|----------|-----------|-----------|------------|
| Order Flow | 44% | 11 | $1,027 | High |
| ORB | 55% | 1-2 | $390 | Low |
| Two Hour | 60% | 3-5 | $500-700 | Medium |

**Best for**: Traders who prefer simpler, higher-win-rate strategies with fewer trades.

---

## 14. Next Steps

**Ready to proceed?**

If you want to study ORB strategy:
1. I'll code the ORB backtest engine
2. Test all 5 variations
3. Compare across MNQ/MES
4. Find optimal parameters

**Vote**: Should we study ORB strategy next?

---

*Document Version: 1.0*  
*Status: Research Phase*  
*Priority: MEDIUM (proven strategy, mechanical)*
