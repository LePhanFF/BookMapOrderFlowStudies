# Mean Reversion Strategy Study
## Bollinger Band & RSI Counter-Trend Trading

**Status**: 📋 RESEARCH PHASE  
**Instruments**: NQ, ES (MNQ, MES for sizing)  
**Target Win Rate**: 60%+  
**Target R:R**: 1.5:1 to 2:1  
**Risk per Trade**: $400  
**Study Date**: February 2026  

---

## 1. Strategy Overview

### Core Concept
Trade counter-trend when price reaches extreme levels (Bollinger Band touches, RSI overbought/oversold) and shows signs of reversal in range-bound markets.

### Why It Works
1. **Markets are range-bound 70% of the time**
2. **Clear mechanical levels** (bands, RSI)
3. **High win rate** (65-70% in chop)
4. **Quick trades** - capture the snapback
5. **Works when trend strategies fail**

### When It FAILS
- ❌ Strong trending markets (ADX > 30)
- ❌ News-driven moves
- ❌ Opening drives
- ❌ Breakout days

---

## 2. Entry Rules (Mechanical)

### Market Regime Filter (CRITICAL)
```
Only trade if:
  - ADX < 25 (no strong trend)
  - Price within 2 ATR of VWAP (range-bound)
  - Not within 30 min of open (avoid volatility)
  - VIX < 25 (normal volatility)
```

### Setup A: Bollinger Band Touch
```
Long Entry (Counter-Trend Shorts):
  IF:
    1. Price touches or breaks below lower Bollinger Band
    2. RSI < 30 (oversold)
    3. ADX < 25 (range-bound)
    4. Time: 10:30-11:30 or 13:00-15:00
    5. Volume declining (exhaustion)
    
  THEN:
    Enter LONG at close of touch candle
    Stop: Below swing low (2-3 points)
    Target: VWAP or middle Bollinger Band (1.5:1 R:R)

Short Entry (Counter-Trend Longs):
  IF:
    1. Price touches or breaks above upper Bollinger Band
    2. RSI > 70 (overbought)
    3. ADX < 25
    4. Time window
    5. Volume declining
    
  THEN:
    Enter SHORT
    Stop: Above swing high
    Target: VWAP or middle band
```

### Setup B: RSI Extreme with Divergence
```
Long Entry:
  IF:
    1. RSI < 25 (extreme oversold)
    2. Price makes lower low, RSI makes higher low (divergence)
    3. ADX < 25
    4. Hammer/doji candle (reversal pattern)
    
  THEN:
    Enter LONG
    Stop: Below candle low
    Target: 20 EMA or prior resistance

Short Entry:
  (Opposite logic)
```

### Setup C: VWAP Rejection
```
Long Entry:
  IF:
    1. Price extends >1.5 ATR below VWAP
    2. Price rejects with wick
    3. RSI < 40 (oversold but not extreme)
    4. ADX < 25
    
  THEN:
    Enter LONG
    Stop: Below wick low
    Target: VWAP (1.5:1 R:R)
```

---

## 3. Exit Rules

### Stop Loss (Tight)
```
Stop: 2-5 points for MNQ
  - Must be tight because:
    - If reversion doesn't happen quickly, it won't
    - Mean reversion is fast
    - If price keeps going, trend is stronger than expected
```

### Profit Targets

**Conservative**
```
Target 1: Middle Bollinger Band (50% exit)
Target 2: Opposite Bollinger Band (50% exit)
```

**Aggressive**
```
Target 1: 1.5:1 R:R (exit 50%)
Target 2: 2:1 R:R (exit 50%)
```

### Time Exit
```
If position not profitable within 5 bars (5 minutes):
  - Exit at market
  - Reversion failed, move on
```

### Trailing Stop (Once in Profit)
```
Move stop to breakeven once +0.5R profit
Trail with 1.5 ATR
```

---

## 4. Bollinger Band Parameters

### Standard Settings
```
Length: 20 periods
Std Dev: 2.0

Alternative:
  Length: 20
  Std Dev: 2.5 (wider bands, fewer signals, higher quality)
```

### Band Width Filter
```
Band Width = (Upper - Lower) / Middle

If Band Width > 5% of price:
  - High volatility
  - Reduce size by 50%
  - Or skip

If Band Width < 2% of price:
  - Low volatility (squeeze)
  - Wait for breakout
  - Don't trade mean reversion
```

---

## 5. Position Sizing

### MNQ Sizing
```
Risk per trade: $400

Tight stop (3-5 points):
  - Risk per contract: 4 pts × $2 = $8
  - Contracts: $400 / $8 = 50 contracts (too many!)
  
Realistic:
  - Stop: 5 points = $10 risk
  - Contracts: $400 / $10 = 40 contracts
  - Adjust down to 20-25 for slippage
```

### Key Insight
**Mean reversion allows LARGE position sizes because:**
1. Tight stops (2-5 points)
2. Quick resolution (5 min max)
3. High win rate (65%+) compensates for smaller wins

---

## 6. RSI Parameters

### Settings
```
Length: 14 periods (standard)
Overbought: 70
Oversold: 30

Extreme Levels:
  Overbought: 80 (stronger signal)
  Oversold: 20 (stronger signal)
```

### RSI Divergence
```
Bullish Divergence (Long Signal):
  - Price makes lower low
  - RSI makes higher low
  - Indicates weakening selling pressure

Bearish Divergence (Short Signal):
  - Price makes higher high
  - RSI makes lower high
  - Indicates weakening buying pressure
```

---

## 7. Expected Performance (Hypothesis)

| Metric | Range-Bound | Trending | Overall |
|--------|-------------|----------|---------|
| **Win Rate** | 65-70% | 35-40% | 55-60% |
| **Avg Win** | 8 pts | 5 pts | 7 pts |
| **Avg Loss** | 5 pts | 6 pts | 5 pts |
| **R:R Ratio** | 1.6:1 | 0.8:1 | 1.4:1 |
| **Profit Factor** | 2.5 | 0.7 | 1.8 |
| **Trades/Day** | 4-6 | 2-3 (losses) | 3-4 |
| **Expectancy** | +$90 | -$40 | +$50 |

### Daily P&L Projection (MNQ)
```
Win Rate: 60% (accounting for trending days)
Trades: 4/day
Avg Win: 7 pts × $2 × 20 ctr = $280
Avg Loss: 5 pts × $2 × 20 ctr = $200

Daily: (4 × 0.60 × $280) - (4 × 0.40 × $200) = $672 - $320 = $352/day

Monthly (20 days): $7,040
Pass $9K eval: 26 days
```

**Note**: Lower daily P&L but higher win rate = psychological comfort.

---

## 8. Market Regime Detection

### Range-Bound (Trade Mean Reversion)
```
ADX < 25
Price oscillating around VWAP
Bollinger Bands flat or sloping slightly
RSI oscillating 30-70
Daily range < 1.5× average
```

### Trending (AVOID Mean Reversion)
```
ADX > 30
Price above/below VWAP consistently
Bollinger Bands expanding
RSI staying >60 or <40
Daily range > 2× average
```

### Transition (Caution)
```
ADX 25-30
Mixed signals
Reduce size by 50%
Or skip trading
```

---

## 9. Advantages vs Other Strategies

### Advantages
1. **Highest win rate**: 65%+ in chop
2. **Quick trades**: In and out fast
3. **Large position size**: Tight stops
4. **Works in sideways markets**: When trends fail
5. **Psychological comfort**: Winning more often

### Disadvantages
1. **Low R:R**: 1.5:1 typical
2. **Market dependent**: Fails in trends
3. **Requires regime detection**: Must know when NOT to trade
4. **Lower expectancy per trade**: $50 vs $94 (order flow)
5. **Counter-trend stress**: Hard to fight momentum

### When to Use
- Range-bound markets
- Afternoon chop (1:00-3:00 PM)
- Low volatility periods
- When trend strategies are losing

---

## 10. Implementation Plan

### Phase 1: Regime Detection
```
- [ ] Code ADX calculation
- [ ] Code Bollinger Band width
- [ ] Determine market regime
- [ ] Only trade mean reversion in range
```

### Phase 2: Signal Generation
```
- [ ] BB touch detection
- [ ] RSI extreme detection
- [ ] Divergence detection
- [ ] Volume confirmation
```

### Phase 3: Backtest
```
- [ ] Test on 90 days
- [ ] Separate range vs trend days
- [ ] Calculate regime-specific performance
- [ ] Find optimal filters
```

### Phase 4: Live
```
- [ ] Manual regime detection first
- [ ] Automate after validation
- [ ] Start small (10 contracts)
- [ ] Scale up after success
```

---

## 11. Risk Management

### Mean Reversion Specific Risks

**Trend Continuation Risk**
```
- Price keeps going (not reversing)
- Solution: Tight stops (2-5 points)
- Time exit if no reversal in 5 min
```

**False Reversal Risk**
```
- Temporary bounce then continuation
- Solution: Require RSI divergence
- Volume confirmation
```

**Gap Risk**
```
- Large gaps create extreme readings
- Solution: Skip first 30 min after large gap
- Wait for stabilization
```

### Daily Limits
```
Max Trades: 6/day
Max Loss: $800 (4 trades)
Time: Avoid 9:30-10:00 (opening chop)
Regime: Skip if ADX > 30
```

---

## 12. Hybrid Strategy Idea

**Mean Reversion + Order Flow**
```
1. Price at lower Bollinger Band
2. RSI < 30 (oversold)
3. ADX < 25 (range-bound)
4. Delta > 85% (aggressive buying at support)
5. CVD rising (institutional accumulation)

This combines:
  - Technical extreme (BB + RSI)
  - Market regime (ADX)
  - Order flow confirmation (Delta + CVD)
  
Expected: 70%+ win rate, 2:1 R:R
```

---

## 13. Comparison Summary

| Strategy | Win Rate | R:R | Best Market | Trades/Day |
|----------|----------|-----|-------------|------------|
| Order Flow | 44% | 2:1 | All | 11 |
| ORB | 55% | 2:1 | Opening | 1-2 |
| Trend Following | 58% | 3:1 | Trending | 3-4 |
| Mean Reversion | 65% | 1.5:1 | Range-bound | 4-6 |

**Best for**: Range-bound markets, traders who prefer high win rates.

---

## 14. Questions to Answer

1. **Can we detect range vs trend reliably?**
2. **What is optimal BB period/std dev?**
3. **Is RSI divergence worth the complexity?**
4. **How does MR compare to order flow overall?**
5. **Best time of day for MR?**
6. **Can we combine MR + Order Flow?**

---

## 15. Next Steps

**Ready to proceed?**

If you want to study Mean Reversion:
1. Build regime detection engine
2. Test BB + RSI combinations
3. Separate results by market type
4. Compare expectancy

**Vote**: Should we study Mean Reversion strategy next?

---

*Document Version: 1.0*  
*Status: Research Phase*  
*Priority: MEDIUM (complementary to trend strategies)*
