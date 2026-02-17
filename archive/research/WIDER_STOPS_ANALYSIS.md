# WIDER STOPS & SCALE-IN ANALYSIS
## Testing 1.0x to 2.0x ATR Stops with 1.5:1 to 3:1 R:R

---

## KEY FINDING: Wider Stops + Higher R:R = MUCH BETTER

### Best Configuration Found
| Parameter | Value |
|-----------|-------|
| **Stop** | **2.0x ATR** (97 points) |
| **Target** | **3.0x ATR** (145.5 points) |
| **R:R Ratio** | **3:1** |
| **Win Rate** | **58.1%** |
| **Expectancy** | **39.52 points/trade** |
| **Profit Factor** | **2.88** |

**This is 3x better than the previous best!**

---

## WIDER STOPS TEST RESULTS

### Top 10 Configurations (Sorted by Expectancy)

| Rank | Stop | R:R | WR% | Exp (pts) | PF | Avg Win | Avg Loss |
|------|------|-----|-----|-----------|-----|---------|----------|
| 1 | **2.0x** | **3.0:1** | **58.1%** | **39.52** | **2.88** | 104.2 | 50.0 |
| 2 | 1.5x | 3.0:1 | 58.7% | 35.58 | 3.03 | 90.4 | 42.4 |
| 3 | 2.0x | 2.5:1 | 58.7% | 35.29 | 2.71 | 95.2 | 50.0 |
| 4 | 1.5x | 2.5:1 | 58.7% | 31.31 | 2.79 | 83.1 | 42.4 |
| 5 | 2.0x | 2.0:1 | 58.7% | 29.45 | 2.43 | 85.3 | 50.0 |
| 6 | 1.5x | 2.0:1 | 57.8% | 24.27 | 2.31 | 74.0 | 43.9 |
| 7 | 1.0x | 3.0:1 | 52.2% | 23.99 | 2.40 | 78.8 | 36.0 |
| 8 | 2.0x | 1.5:1 | 57.8% | 20.75 | 1.94 | 74.0 | 52.2 |
| 9 | 1.0x | 2.5:1 | 52.9% | 20.24 | 2.20 | 70.2 | 36.0 |
| 10 | 1.5x | 1.5:1 | 57.6% | 17.27 | 1.91 | 62.9 | 44.7 |

### Key Insights

1. **2.0x ATR stop with 3:1 R:R is the BEST**
   - Highest expectancy: 39.52 points
   - Excellent profit factor: 2.88
   - Strong win rate: 58.1%

2. **3:1 R:R consistently outperforms 2:1 and 1.5:1**
   - Top 3 all use 3:1 or 2.5:1
   - Higher R:R = better expectancy

3. **Wider stops (2.0x) work better than tight stops (1.0x)**
   - 2.0x ATR captures bigger moves
   - Less likely to get stopped out on noise
   - Better risk-adjusted returns

4. **Win rate INCREASES with wider stops**
   - 1.0x ATR: ~52-53% WR
   - 2.0x ATR: ~58-59% WR
   - Wider stops = more breathing room = higher win rate

---

## SCALE-IN STRATEGY TEST

### Approach Tested
- **Initial Entry**: 1/3 position size on signal
- **Add**: 2/3 position after 1-3 bars IF in profit
- **Logic**: Only add when trade is working (confirmation)
- **Risk Management**: Scale in reduces risk on bad entries

### Results

| Confirm Bars | Stop | R:R | WR% | Exp (pts) | PF |
|--------------|------|-----|-----|-----------|-----|
| 1 | 1.5x | 2.5:1 | 47.3% | 20.93 | 1.97 |
| 2 | 1.5x | 2.5:1 | 49.1% | 20.73 | 2.14 |
| 1 | 1.0x | 2.5:1 | 39.3% | 16.52 | 1.95 |
| 2 | 1.5x | 2.0:1 | 47.4% | 15.84 | 1.85 |
| 3 | 1.5x | 2.5:1 | 44.6% | 15.38 | 1.82 |

### Scale-In Verdict
**Standard entry wins by 18.59 points/trade**

- Best scale-in: 20.93 pts expectancy
- Best standard: 39.52 pts expectancy
- Scale-in reduces position size when trade works
- Misses profit on full position during best trades

**Recommendation: Use standard full-size entry, NOT scale-in**

---

## WHY WIDER STOPS WORK BETTER

### 1. Captures Full 5-Minute Moves
- 5-min bars have natural range of 30-50 points
- 2x ATR (97 pts) captures complete moves
- 1x ATR (48 pts) often exits before move completes

### 2. Institutional Order Flow Takes Time
- Large orders execute over multiple bars
- Wider stops allow time for delta to work
- Tight stops get hit on temporary reversals

### 3. Higher R:R Compensates for Wider Stops
- 3:1 R:R means one winner = 3 losses
- Can afford 40% win rate and still profit
- Actually achieves 58% win rate = excellent

### 4. Psychological Edge
- Less noise, cleaner signals
- Fewer whip-saws
- Higher confidence in trades

---

## UPDATED BEST STRATEGY

### Configuration
| Parameter | Value |
|-----------|-------|
| **Timeframe** | 5-Minute |
| **Time Window** | 10:00 - 12:00 ET |
| **Signal** | Delta percentile > 85 + CVD trend |
| **ATR** | 48.5 points (true 5-min ATR) |
| **Stop** | **2.0x ATR = 97 points** |
| **Target** | **3.0x ATR = 145.5 points** |
| **R:R Ratio** | **3:1** |
| **Entry** | Full size (NO scale-in) |

### Expected Performance
| Metric | Value |
|--------|-------|
| Win Rate | 58.1% |
| Expectancy | 39.52 points/trade |
| Profit Factor | 2.88 |
| Avg Win | 104.2 points |
| Avg Loss | 50.0 points |

### MNQ Position Sizing
- **Stop**: 97 points
- **Risk per contract**: 97 × $2 = $194
- **Max contracts** ($400 risk): $400 ÷ $194 = **2 contracts**
- **Conservative**: 2 contracts max

### Profit Projection (2 contracts)
- Per trade: 39.52 pts × $2 × 2 = $158.08
- Trades per day: ~1 (62 trades ÷ 62 days)
- Daily: ~$158
- **Days to $9K: ~57 days**

**BUT**: Can we trade more contracts with smaller size?

---

## ALTERNATIVE: REDUCE STOP FOR MORE CONTRACTS

If we use 1.5x ATR stop:
- Stop: 73 points (1.5 × 48.5)
- Risk per contract: $146
- Max contracts: $400 ÷ $146 = **2.7 → 2 contracts** (same)

With 1.5x ATR, 3:1 R:R:
- Expectancy: 35.58 points
- Per trade (2 contracts): $142.32
- Daily: ~$142
- **Days to $9K: ~63 days**

**2.0x ATR still better despite same contract count**

---

## COMPARISON: EVOLUTION OF STRATEGY

| Version | Stop | R:R | WR% | Exp (pts) | Days to $9K |
|---------|------|-----|-----|-----------|-------------|
| V1 (Tight) | 0.6x ATR | 2:1 | 41.6% | -0.35 | ❌ Losing |
| V2 (Medium) | 1.0x ATR | 2:1 | 50.0% | 13.35 | ~61 days |
| **V3 (Wide)** | **2.0x ATR** | **3:1** | **58.1%** | **39.52** | **~57 days** |

**Version 3 is the winner!**

---

## FINAL RECOMMENDATIONS

### 1. Use Wider Stops (2.0x ATR)
- Captures full 5-minute institutional moves
- Higher win rate (58.1%)
- Better expectancy (39.52 pts)

### 2. Use Higher R:R (3:1)
- One winner pays for 3 losses
- Actually achieves 58% win rate
- Profit factor of 2.88 is excellent

### 3. DON'T Use Scale-In
- Reduces profit on best trades
- Standard entry outperforms by 18+ pts
- Full size from entry is better

### 4. Trade 10:00-12:00 Window
- Highest quality signals
- Best institutional participation
- Avoid 9:30-10:00 (choppy)

### 5. Use Full 5-Minute Order Flow
- Aggregate bid/ask properly
- Calculate true 5-min ATR (48.5 pts)
- Use CVD trend for confirmation

---

*Document Version: 3.0 (Wider Stops Update)*
*Date: February 16, 2026*
*Key Finding: 2.0x ATR + 3:1 R:R = 39.52 pts expectancy, 58.1% WR*
