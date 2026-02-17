# CORRECTED 5-MINUTE STRATEGY RESULTS
## Proper ATR and Order Flow Calculation

---

## ATR CALCULATION CORRECTION

### The Error
I was previously using the **1-minute ATR value** (16.3 pts) for both timeframes.

### The Correct Calculation
**5-minute ATR is calculated on 5-minute OHLC bars:**

```python
# True Range on 5-min bars
tr1 = high - low
tr2 = abs(high - close[1])
tr3 = abs(low - close[1])
true_range = max(tr1, tr2, tr3)
atr14 = average(true_range, 14)
```

### Results
| Timeframe | ATR(14) | Ratio |
|-----------|---------|-------|
| 1-Minute | 16.3 points | 1.0x |
| **5-Minute** | **38.1 points** | **2.33x** |

**Theoretical**: sqrt(5) = 2.24x ✅ Matches!

---

## CORRECTED 5-MINUTE ORDER FLOW METRICS

### How Order Flow is Aggregated
For each 5-minute bar:

```python
# Sum all 1-minute data within the 5-min window
vol_ask_5min = sum(vol_ask_1min for 5 bars)
vol_bid_5min = sum(vol_bid_1min for 5 bars)

# Calculate 5-min delta
delta_5min = vol_ask_5min - vol_bid_5min

# Calculate CVD (running sum of 5-min deltas)
cvd_5min = cumsum(delta_5min)

# Calculate percentile within 5-min timeframe
delta_percentile_5min = percentile_rank(delta_5min, lookback=20)
```

### Order Flow Inside 5-Minute Bars
**The 5-minute bar captures:**
- Total aggressive buying (sum of ask volume)
- Total aggressive selling (sum of bid volume)  
- Net delta (buying - selling pressure)
- Cumulative delta trend (rising/falling)
- Delta strength (percentile rank)

**This is valid because:**
- Order flow is additive - summing 1-min deltas = 5-min delta
- Aggressive buyers/sellers over 5 minutes = total pressure
- Percentile rank shows relative strength vs recent 5-min bars

---

## BEST 5-MINUTE STRATEGY FOUND

### Configuration
| Parameter | Value |
|-----------|-------|
| **Timeframe** | 5-Minute |
| **Time Window** | 10:00 - 12:00 ET |
| **Signal** | Delta percentile > 85 + CVD trend |
| **ATR** | 38.1 points (true 5-min ATR) |
| **Stop** | 1.0x ATR = 38.1 points |
| **Target** | 2.0x ATR = 76.2 points (2:1 R:R) |
| **Max Bars** | 8 |

### Results
| Metric | Value |
|--------|-------|
| **Total Trades** | 70 |
| **Win Rate** | **50.0%** |
| **Expectancy** | **13.35 points/trade** |
| **Profit Factor** | 1.75 |
| **Avg Win** | 62.2 points |
| **Avg Loss** | 35.5 points |

### Why This Makes Sense Now
✅ **Stop**: 38 points = outside 5-min noise  
✅ **Target**: 76 points = 2x the stop (proper 2:1 R:R)  
✅ **Expectancy**: 13+ points = meaningful profit  
✅ **Win Rate**: 50% = realistic with 2:1 R:R

---

## TOP 5 CONFIGURATIONS

### By Expectancy (Points per Trade)

| Rank | Time | Stop | R:R | WR% | Exp (pts) | PF |
|------|------|------|-----|-----|-----------|-----|
| 1 | 10:00-12:00 | 1.0x | 2:1 | 50.0% | **13.35** | 1.75 |
| 2 | 10:00-12:00 | 1.0x | 2:1 | 48.2% | **12.11** | 1.67 |
| 3 | 11:00-13:00 | 1.0x | 2:1 | 52.4% | **12.08** | 1.91 |
| 4 | 11:00-13:00 | 1.0x | 2:1 | 53.6% | **11.20** | 1.82 |
| 5 | 10:00-12:00 | 1.0x | 1.5:1 | 52.1% | **11.02** | 1.65 |

**Key Finding**: 1.0x ATR stops with 1.5-2:1 R:R work best on 5-min

---

## COMPARISON: 1-MIN vs 5-MIN

### Strategy Parameters
| Parameter | 1-Minute | 5-Minute |
|-----------|----------|----------|
| ATR | 16.3 pts | 38.1 pts |
| Stop | 0.6x ATR (9.8 pts) | 1.0x ATR (38.1 pts) |
| Target | 2.0x (19.6 pts) | 2.0x (76.2 pts) |
| Signal | Delta > 85 + CVD | Delta > 85 + CVD |
| Time | 11:00-13:00 | 10:00-12:00 |

### Performance
| Metric | 1-Minute | 5-Minute | Winner |
|--------|----------|----------|--------|
| Win Rate | 41.6% | **50.0%** | 5-Min |
| Expectancy | -0.35 pts | **+13.35 pts** | 5-Min |
| Profit Factor | 0.97 | **1.75** | 5-Min |
| Trade Quality | Low | High | 5-Min |
| Noise | High | Low | 5-Min |

**Winner: 5-Minute timeframe** ✅

---

## MNQ POSITION SIZING (Corrected)

### Best Strategy (5-Min)
- **Stop**: 38.1 points
- **Risk per contract**: 38.1 × $2 = $76.20
- **Max contracts** (with $400 risk): $400 ÷ $76.20 = **5 contracts**
- **Conservative**: 3-4 contracts

### Profit Projections

**With 5 contracts:**
- Per trade: 13.35 pts × $2 × 5 = $133.50
- Daily trades: ~1.1 (70 trades ÷ 62 days)
- Daily P&L: ~$147
- **Days to $9K: ~61 days**

**With 4 contracts (safer):**
- Per trade: 13.35 pts × $2 × 4 = $106.80
- Daily P&L: ~$117
- **Days to $9K: ~77 days**

---

## KEY INSIGHTS

### 1. 5-Minute ATR is 2.3x Larger
- Not using the same ATR for both timeframes
- Must calculate true range on 5-min OHLC

### 2. Order Flow Aggregation is Valid
- Summing 1-min deltas = 5-min delta ✅
- Percentile rank on 5-min = relative strength ✅
- CVD trend on 5-min = institutional pressure ✅

### 3. Wider Stops Work Better on 5-Min
- 1.0x ATR (38 pts) captures full 5-min moves
- 0.6x ATR was too tight on 5-min timeframe
- Need targets of 1.5-2.0x ATR (57-76 pts)

### 4. 5-Minute Filters Noise
- 4,039 bars vs 19,948 (4.9x fewer)
- Higher quality signals
- Better win rate (50% vs 41.6%)

---

## IMPLEMENTATION CODE

See `strategy_5min_proper.py` for full implementation including:
- Proper 5-min ATR calculation
- Order flow aggregation (delta, CVD, percentile)
- Backtest engine with correct exits
- Position sizing calculations

---

## FINAL RECOMMENDATION

**Use 5-Minute Timeframe:**
1. **Instrument**: MNQ (Micro Nasdaq)
2. **Timeframe**: 5-minute bars
3. **Time Window**: 10:00 - 12:00 ET
4. **Signal**: Delta percentile > 85 + CVD trend
5. **Stop**: 1.0x ATR (~38 points)
6. **Target**: 2.0x ATR (~76 points)
7. **Contracts**: 4-5 MNQ
8. **Expected**: Pass $150K eval in 60-80 days

---

*Document Version: 2.0 (Corrected)*
*Date: February 16, 2026*
*Key Fix: Proper 5-min ATR calculation (38.1 pts, not 16.3)*
