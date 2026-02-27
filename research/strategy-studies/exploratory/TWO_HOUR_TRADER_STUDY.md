# Two Hour Trader Strategy Study
## Options-Based Mechanical Trading for Prop Firms

**Status**: 📋 RESEARCH PHASE  
**Instrument**: SPX Options (or SPY for sizing)  
**Target Win Rate**: 55%+  
**Target R:R**: 2:1 to 3:1  
**Risk per Trade**: $400  
**Study Date**: February 2026  

---

## 1. Strategy Overview

### Source
Based on the "Two Hour Trader Framework" and "AutoPilot Trader" systems that have demonstrated:
- **79% win rate** in live prop firm evaluations
- **6.76 profit factor**
- Passed $50K evaluation in 18 days
- 955+ verified trades

### Core Concept
Trade the first 2 hours of market open (9:30-11:30 AM ET) when:
- Volatility is highest
- Institutional volume flows
- Momentum is strongest
- Mean reversion is most predictable

---

## 2. Entry Rules (Mechanical)

### Time Window
```
Primary: 9:30-11:30 AM ET (First 2 hours)
Optional: 3:00-4:00 PM ET (Power hour - not in original)
Avoid: 11:30 AM - 2:00 PM (Chop/lunch)
```

### Setup Conditions

**Version A: Momentum Breakout**
```
IF:
  1. Time is within first 2 hours (9:30-11:30)
  2. SPX breaks above/below 5-minute opening range high/low
  3. Volume > 20-period average (confirming breakout)
  4. VIX < 25 (avoid high volatility chop)
  
THEN:
  Enter call (breakout up) or put (breakout down)
  Strike: ATM or slight OTM (0.30-0.45 delta)
  Expiration: 0-2 DTE (same day or next)
```

**Version B: Mean Reversion (Counter-Trend)**
```
IF:
  1. Time is within first 2 hours
  2. Price extends >1.5 ATR from VWAP
  3. RSI > 70 (short) or RSI < 30 (long)
  4. Volume declining (exhaustion signal)
  
THEN:
  Enter counter-trend position
  Strike: OTM 0.30 delta
  Expiration: 0-1 DTE
```

**Version C: VWAP Bounce (Trend Continuation)**
```
IF:
  1. Price above VWAP (bullish bias) or below VWAP (bearish)
  2. Price pulls back to VWAP
  3. Candle rejects VWAP with wick
  4. Time window active
  
THEN:
  Enter in direction of VWAP trend
  Strike: ATM
  Expiration: 0-2 DTE
```

---

## 3. Exit Rules

### Stop Loss (Hard)
```
Options stop: 40-50% of premium paid
  - $400 risk / 0.50 delta = ~$800 option cost
  - Stop at $320-400 loss (40-50%)
  
OR

Underlying-based stop:
  - SPX moves X points against position
  - Typically 10-15 SPX points
```

### Profit Target
```
Target 1: 100% gain (2:1 R:R)
  - Exit 50% of position
  
Target 2: 200% gain (4:1 R:R)
  - Exit 25% of position
  
Target 3: Runner (trailing stop)
  - Exit final 25% on momentum shift
```

### Time Exit
```
Hard exit: 11:30 AM ET if position open
  - Don't hold through lunch chop
  - Take whatever P&L
```

---

## 4. Position Sizing

### SPX Options (Full Size)
```
Capital Required: $3,000-5,000 per trade
Risk per Trade: $400 (max)
Contract Cost: ~$800-1,200 (ATM 0.50 delta)
Contracts per Trade: 3-5

Example:
  - Buy 3 SPX calls @ $400 each = $1,200 cost
  - Stop: $160 loss per contract (40%)
  - Total risk: $480 (slightly over, adjust to 2-3 contracts)
```

### SPY Options (Reduced Size)
```
Capital Required: $800-1,500 per trade
Risk per Trade: $400
Contract Cost: ~$200-400 (ATM)
Contracts per Trade: 2-3

Example:
  - Buy 3 SPY calls @ $300 each = $900 cost
  - Stop: $135 loss per contract (45%)
  - Total risk: $405 ✅
```

### QQQ Options (Tech-focused)
```
Similar to SPY
May have higher volatility than SPX
Good for NQ correlation
```

---

## 5. Instrument Comparison

| Instrument | Cost | Risk/Contract | Contracts for $400 Risk | Liquidity | Best For |
|------------|------|---------------|------------------------|-----------|----------|
| **SPX** | $800-1,200 | ~$400 | 3-4 | ⭐⭐⭐⭐⭐ | Large accounts |
| **SPY** | $200-400 | ~$135 | 3 | ⭐⭐⭐⭐⭐ | Medium accounts |
| **QQQ** | $300-500 | ~$150 | 2-3 | ⭐⭐⭐⭐⭐ | Tech bias |
| **NQ Options** | $1,500+ | ~$600 | 1 | ⭐⭐⭐ | High volatility |
| **ES Options** | $800-1,000 | ~$400 | 1 | ⭐⭐⭐⭐ | Futures traders |

**Recommendation**: Start with SPY options for testing, scale to SPX for production.

---

## 6. Parameters to Test

### A. Time Window Variations
```
1. Strict 9:30-10:30 (first hour only)
2. 9:30-11:30 (original Two Hour)
3. 9:30-11:30 + 14:00-16:00 (add power hour)
4. Only 9:30-10:00 (power 30 min)
```

### B. Strike Selection
```
1. ATM (0.50 delta)
2. Slight OTM (0.40 delta)
3. OTM (0.30 delta) - higher R:R
4. ITM (0.60 delta) - higher win rate
```

### C. Expiration
```
1. 0 DTE (same day) - highest gamma
2. 1 DTE (next day) - slightly lower theta
3. 2 DTE - more time
```

### D. Technical Filters
```
1. VWAP slope (trend confirmation)
2. VIX level (volatility filter)
3. Premarket range (expectation)
4. Gap size (avoid large gaps)
```

---

## 7. Expected Performance (Hypothesis)

Based on reported results:

| Metric | Conservative | Target | Aggressive |
|--------|-------------|--------|------------|
| **Win Rate** | 55% | 65% | 79% (reported) |
| **Avg Win** | 80% gain | 100% gain | 150% gain |
| **Avg Loss** | 45% loss | 45% loss | 45% loss |
| **R:R Ratio** | 1.8:1 | 2.2:1 | 3.3:1 |
| **Profit Factor** | 2.0 | 3.0 | 6.76 (reported) |
| **Trades/Day** | 2-3 | 3-5 | 5-8 |
| **Expectancy** | +$120 | +$180 | +$280 |

### Daily P&L Projection (SPY Options)
```
Win Rate: 60%
Trades: 4/day
Avg Win: $200
Avg Loss: $135

Daily: (4 × 0.60 × $200) - (4 × 0.40 × $135) = $480 - $216 = $264/day

Monthly (20 days): $5,280
Pass $9K eval: 17 days
```

### Daily P&L Projection (SPX Options)
```
Win Rate: 60%
Trades: 3/day
Avg Win: $400
Avg Loss: $270

Daily: (3 × 0.60 × $400) - (3 × 0.40 × $270) = $720 - $324 = $396/day

Monthly (20 days): $7,920
Pass $9K eval: 23 days
```

**Note**: Lower trade frequency with SPX but higher per-trade profit.

---

## 8. Advantages vs Order Flow Strategy

### Advantages
1. **Higher win rate potential**: 60-79% vs 44%
2. **Defined risk**: Can't lose more than premium paid
3. **No overnight risk**: Day trade only
4. **Time decay works**: Theta decay in your favor intraday
5. **Less capital intensive**: SPY requires $1K vs MNQ $6K margin
6. **No margin calls**: Max loss is premium paid

### Disadvantages
1. **Time decay**: Must be right quickly
2. **Volatility crush**: IV can drop suddenly
3. **Spread costs**: Options have wider spreads
4. **Liquidity**: Less liquid than futures
5. **Complexity**: Greeks (delta, theta, gamma, vega)
6. **Commissions**: Higher per contract

---

## 9. Implementation Plan

### Phase 1: Data Collection
```
- [ ] Get SPX/SPY options data (1-min OHLC)
- [ ] Get SPX underlying data for signals
- [ ] 90 days minimum (same as order flow study)
```

### Phase 2: Backtest
```
- [ ] Test all three entry variations
- [ ] Optimize parameters
- [ ] Compare to order flow strategy
```

### Phase 3: Paper Trade
```
- [ ] 2 weeks on live market
- [ ] Verify fills
- [ ] Check slippage
```

### Phase 4: Live
```
- [ ] Start with SPY (smaller)
- [ ] Scale to SPX
- [ ] Trade evaluation
```

---

## 10. Risk Management

### Options-Specific Risks

**Delta Risk** (Directional)
```
- ATM options: ~0.50 delta
- For every $1 SPX moves, option moves $0.50
- Manage with stop on underlying
```

**Theta Risk** (Time Decay)
```
- 0 DTE: High theta decay (good for sellers, bad for buyers)
- Must capture move quickly
- Exit by 11:30 AM to avoid afternoon decay
```

**Vega Risk** (Volatility)
```
- VIX crush after open = option value drops
- Enter during elevated IV (better for mean reversion)
- Avoid when VIX > 30
```

**Gamma Risk** (Acceleration)
```
- 0 DTE = very high gamma
- Small moves = big P&L swings
- Manage position size carefully
```

### Daily Limits
```
Max Loss: $800 (2 trades worth)
Max Trades: 5/day
Time Stop: 11:30 AM hard stop
Consecutive Losses: 2 → reduce size
```

---

## 11. Questions to Answer

1. **Does 60%+ win rate hold up in backtest?**
2. **What is actual expectancy with $400 risk?**
3. **Which entry method works best?** (Momentum vs Mean Reversion vs VWAP)
4. **SPY vs SPX - which is better for prop firms?**
5. **Can we automate this in NinjaTrader or need Options API?**
6. **What is max drawdown with options?**
7. **How does this compare to order flow strategy?**

---

## 12. Next Steps

**Ready to proceed?**

If you want to study this strategy:
1. I'll research options data sources
2. Build options backtest engine
3. Test all three variations
4. Compare to current order flow results

**Vote**: Should we study Two Hour Trader strategy next?

---

*Document Version: 1.0*  
*Status: Research Phase*  
*Priority: HIGH (proven results in prop firms)*
