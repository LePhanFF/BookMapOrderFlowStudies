# Trend-Following Breakout Strategy Study
## Multi-Timeframe Momentum Trading

**Status**: 📋 RESEARCH PHASE  
**Instruments**: NQ, ES (MNQ, MES for sizing)  
**Target Win Rate**: 55%+  
**Target R:R**: 2:1 to 4:1  
**Risk per Trade**: $400  
**Study Date**: February 2026  

---

## 1. Strategy Overview

### Core Concept
Trade breakouts of significant levels (20-period high/low, prior day high/low, VWAP bands) in the direction of the established trend. Higher timeframe trend filters reduce false breakouts.

### Why It Works
1. **Trend continuation** has statistical edge
2. **Clear mechanical rules** - no discretion
3. **Multiple timeframe alignment** increases win rate
4. **Breakouts capture momentum** phases
5. **Works across market conditions** (trending, not chop)

---

## 2. Entry Rules (Mechanical)

### Trend Definition (Higher Timeframe)
```
Timeframe: 5-minute or 15-minute
Bullish Trend:
  - Price > 20-period EMA
  - 20 EMA > 50 EMA
  - Price making higher highs/lows

Bearish Trend:
  - Price < 20-period EMA
  - 20 EMA < 50 EMA
  - Price making lower highs/lows
```

### Entry Conditions

**Version A: 20-Period High/Low Breakout**
```
Long Entry:
  IF:
    1. 5-min trend is bullish (above 20 EMA)
    2. Price breaks above 20-period high
    3. Volume > average (momentum confirmation)
    4. Time: 10:00-11:30 or 13:30-15:00 (avoid lunch)
    
  THEN:
    Enter LONG at breakout + 1 tick
    Stop: Below breakout bar low
    Target: 3:1 R:R (extended target for trends)

Short Entry:
  (Opposite logic)
```

**Version B: Prior Day High/Lod Breakout**
```
Long Entry:
  IF:
    1. Price breaks above prior day high
    2. 5-min trend bullish
    3. Opening range already established
    4. Volume confirming
    
  THEN:
    Enter LONG
    Stop: Below prior day high (now support)
    Target: 2:1 to 4:1 R:R
```

**Version C: VWAP Band Breakout**
```
Long Entry:
  IF:
    1. Price above VWAP (bullish bias)
    2. Price breaks above upper VWAP band (+2 std dev)
    3. 5-min trend bullish
    4. Not extended (>2 ATR from VWAP)
    
  THEN:
    Enter LONG
    Stop: Below VWAP
    Target: 2:1 R:R
```

**Version D: Opening Drive Continuation**
```
If ORB (Opening Range Breakout) succeeded:
  - Trend established in first 30 min
  - Wait for pullback to VWAP or 20 EMA
  - Enter in direction of opening trend
  - Ride the morning momentum
```

---

## 3. Exit Rules

### Stop Loss
```
Fixed Stop:
  - Below entry bar low (longs)
  - Above entry bar high (shorts)
  - Typically 8-15 points for MNQ

Trailing Stop (once in profit):
  - Trail with 20-period EMA
  - Exit on close below EMA (longs)
  - Or: Trail with 1.5 ATR
```

### Profit Targets

**Conservative**
```
Target 1: 2:1 R:R (exit 50%)
Target 2: 3:1 R:R (exit 25%)
Target 3: Trail stop (runner 25%)
```

**Aggressive (Trend Following)**
```
Target 1: 2:1 R:R (exit 33%)
Target 2: 4:1 R:R (exit 33%)
Target 3: 6:1 R:R or trail (runner 34%)
```

**Time-Based**
```
Exit by 11:30 AM if morning trade
Exit by 3:30 PM if afternoon trade
Avoid overnight/weekend
```

---

## 4. Multi-Timeframe Filter (Critical)

### The 3-Timeframe Rule
```
Trend Alignment Required:
  - 15-min: Trend direction (highest weight)
  - 5-min: Entry timeframe
  - 1-min: Precision entry

Only trade when:
  - 15-min trend = 5-min trend
  - Entry on 5-min aligns with 15-min
  - 1-min for optimal fill

Example:
  - 15-min bullish (price > 20 EMA)
  - 5-min pullback to 20 EMA
  - 1-min bullish engulfing = ENTER LONG
```

### Trend Strength Filter
```
ADX Indicator (optional but powerful):
  - ADX > 25: Strong trend, take all signals
  - ADX 20-25: Moderate trend, reduce size
  - ADX < 20: Weak trend/chop, skip signals
```

---

## 5. Position Sizing

### MNQ Sizing
```
Risk per trade: $400

Standard breakout (10-15 pt stop):
  - Contracts = $400 / (12 pts × $2) = ~16 contracts

Aggressive trend (20+ pt stop for runner):
  - Reduce to 10 contracts
  - Target 4:1+ R:R
```

### Dynamic Sizing by Trend Strength
```
Strong trend (ADX > 30):
  - Full size (16-20 contracts)
  - Extended targets (3:1+)

Moderate trend (ADX 20-30):
  - 75% size (12-15 contracts)
  - Standard targets (2:1)

Weak trend (ADX < 20):
  - 50% size or skip
  - Tight targets (1.5:1)
```

---

## 6. Parameters to Test

### A. Trend Definition
```
1. EMA 20/50 cross (standard)
2. Price above/below 20 EMA only
3. ADX + DI+/-DI
4. MACD alignment
5. Supertrend indicator
```

### B. Breakout Level
```
1. 20-period high/low
2. 50-period high/low
3. Prior day high/low
4. VWAP bands
5. Opening range extension
```

### C. Entry Timing
```
1. Immediate breakout
2. Close beyond level
3. Pullback to level
4. 2-bar confirmation
5. Volume spike confirmation
```

### D. Time Windows
```
1. All day (9:30-16:00)
2. Morning only (9:30-11:30)
3. Afternoon only (13:30-16:00)
4. Exclude lunch (11:30-13:30)
```

---

## 7. Expected Performance (Hypothesis)

| Metric | Conservative | Target | Aggressive |
|--------|-------------|--------|-----------|
| **Win Rate** | 50% | 58% | 65% |
| **Avg Win** | 25 pts | 35 pts | 50 pts |
| **Avg Loss** | 12 pts | 12 pts | 12 pts |
| **R:R Ratio** | 2:1 | 3:1 | 4:1 |
| **Profit Factor** | 2.0 | 2.8 | 3.5 |
| **Trades/Day** | 2-3 | 3-4 | 4-5 |
| **Expectancy** | +$130 | +$280 | +$450 |

### Daily P&L Projection (MNQ)
```
Win Rate: 58%
Trades: 3/day
Avg Win: 35 pts × $2 × 16 ctr = $1,120
Avg Loss: 12 pts × $2 × 16 ctr = $384

Daily: (3 × 0.58 × $1,120) - (3 × 0.42 × $384) = $1,949 - $484 = $1,465/day

Monthly (20 days): $29,300
Pass $9K eval: 6 days
```

**Note**: This is optimistic - requires strong trending markets.

---

## 8. Market Regimes

### Best Conditions
```
✅ Strong trend (ADX > 30)
✅ Opening drive momentum
✅ Post-news continuation
✅ Break of major level (prior day H/L)
✅ Volume > 150% average
```

### Worst Conditions
```
❌ Chop (ADX < 20)
❌ Inside day (range-bound)
❌ Late afternoon fade
❌ High VIX (>30)
❌ Pre-holiday session
```

### Daily Bias Check
```
Before trading, check:
1. Overnight range vs 20-day average
2. Premarket volume (high = good)
3. Economic calendar (avoid news)
4. VIX level (<25 preferred)
5. Gap size (large gaps = caution)
```

---

## 9. Advantages vs Other Strategies

### vs Order Flow
- **Higher win rate**: 58% vs 44%
- **Fewer trades**: 3-4 vs 11/day
- **Trend focus**: Better in strong trends
- **Clear filters**: HTF alignment reduces noise

### vs ORB
- **All day trading**: Not limited to open
- **Trend confirmation**: Higher quality setups
- **Better R:R**: 3:1+ vs 2:1
- **More selective**: Quality over quantity

### vs Mean Reversion
- **Pro-trend**: Not fighting momentum
- **Extended moves**: Capture full trend
- **Works in trends**: MR only works in range

---

## 10. Implementation Plan

### Phase 1: Multi-Timeframe Setup
```
- [ ] Configure 15-min, 5-min, 1-min charts
- [ ] Add EMAs (20, 50)
- [ ] Add ADX indicator
- [ ] Mark prior day high/low
- [ ] Calculate VWAP bands
```

### Phase 2: Signal Detection
```
- [ ] Code trend detection (15-min HTF)
- [ ] Code breakout detection (5-min)
- [ ] Code entry precision (1-min)
- [ ] Backtest 90 days
```

### Phase 3: Optimization
```
- [ ] Find best trend filter
- [ ] Optimize breakout level
- [ ] Test time windows
- [ ] Validate win rate
```

### Phase 4: Live Trading
```
- [ ] Paper trade 2 weeks
- [ ] Verify MTF alignment
- [ ] Check slippage on breakouts
- [ ] Go live with small size
```

---

## 11. Risk Management

### Trend-Following Risks

**Late Entry Risk**
```
- Entering trend too late
- Solution: Use pullback entries
- Wait for 5-min pullback to EMA
```

**Trend Reversal Risk**
```
- Trend ends suddenly
- Solution: ADX filter
- Exit if ADX drops below 20
```

**Chop Risk**
```
- Whipsaws in range
- Solution: Skip if ADX < 25
- Only trade strong trends
```

### Daily Limits
```
Max Trades: 5/day
Max Loss: $1,200 (3 trades)
Time: No entries after 3:00 PM
ADX Filter: Skip if ADX < 20 on 15-min
```

---

## 12. Questions to Answer

1. **Best HTF timeframe?** (15-min vs 30-min vs 1-hour)
2. **Is ADX filter worth it?**
3. **Best breakout level?** (20-period vs prior day vs VWAP)
4. **Trend following vs ORB - which is better?**
5. **How does performance vary by market regime?**
6. **Can we combine with order flow?** (Trend + Delta)

---

## 13. Hybrid Strategy Idea

**Trend Following + Order Flow**
```
1. 15-min trend bullish
2. Wait for pullback to 5-min EMA
3. Enter when:
   - Price touches EMA
   - Delta > 85% (aggressive buying)
   - CVD rising
   
This combines:
  - HTF trend (high win rate)
  - Pullback entry (good R:R)
  - Order flow confirmation (timing)
```

**Expected**: 60%+ win rate, 3:1 R:R

---

## 14. Comparison Summary

| Strategy | Win Rate | Trades/Day | R:R | Best Market |
|----------|----------|-----------|-----|-------------|
| Order Flow | 44% | 11 | 2:1 | All |
| ORB | 55% | 1-2 | 2:1 | Opening drive |
| Trend Following | 58% | 3-4 | 3:1 | Trending |
| Two Hour | 60% | 3-5 | 2:1 | Opening |

**Best for**: Trending markets, patient traders who want higher R:R.

---

## 15. Next Steps

**Ready to proceed?**

If you want to study Trend-Following:
1. Build multi-timeframe backtest engine
2. Test all 4 variations
3. Optimize trend filters
4. Compare to current results

**Vote**: Should we study Trend-Following strategy next?

---

*Document Version: 1.0*  
*Status: Research Phase*  
*Priority: HIGH (highest R:R potential)*
