# ENHANCED ORDER FLOW STRATEGIES
## Bookmap-Style Features Implementation

**Date**: February 16, 2026  
**Based on**: Bookmap Article - Real-Time Order Flow Analysis  
**Data**: NQ Futures, 1-minute bars, 62 trading days  

---

## OVERVIEW

This document describes enhanced order flow strategies incorporating Bookmap-style features that can be calculated from standard OHLCV + delta data (no Level 2 required).

**Key Insight**: Adding volume spike and imbalance filters improves expectancy from 1.52 to 1.56 points/trade.

---

## STRATEGY 1: ORIGINAL BASELINE (Delta > 85 + CVD)

### Description
Basic momentum strategy using delta percentile and cumulative delta trend.

### Logic
```
LONG:
- Delta percentile > 85 (top 15% of recent delta readings)
- Delta > 0 (net buying)
- CVD (Cumulative Delta) is rising (institutional trend up)

SHORT:
- Delta percentile > 85
- Delta < 0 (net selling)
- CVD is falling
```

### Parameters
| Parameter | Value |
|-----------|-------|
| Delta Lookback | 20 bars |
| Delta Threshold | 85th percentile |
| CVD Period | 20-bar EMA |

### Performance
| Metric | Value |
|--------|-------|
| Total Trades | 677 |
| Win Rate | 44.2% |
| Expectancy | 1.52 pts/trade |
| Profit Factor | 1.19 |
| Daily Trades | 10.9 |
| Daily P&L (31 MNQ) | $993 |
| Days to $9K | 9 |

### Pros
✅ Simple, easy to understand  
✅ High trade frequency  
✅ Proven profitability  

### Cons
❌ No volume confirmation  
❌ No imbalance filtering  
❌ Can enter on weak momentum  

### Tweaks to Test
- Delta threshold: Try 80, 85, 90
- CVD period: Try 10, 20, 30 bars
- Add volume filter: volume > 1.2x average

---

## STRATEGY 2: ABSORPTION DETECTION

### Description
Identifies when large orders are being absorbed at a price level (high delta/volume but tight range). This often precedes breakouts or reversals as institutions accumulate/distribute.

### Logic
```
ABSORPTION LONG:
- Delta percentile > 80 (strong buying pressure)
- Price range < 70% of 20-bar average range (tight consolidation)
- Volume > 1.2x 20-bar average (high activity)

ABSORPTION SHORT:
- Delta percentile > 80
- Price range < 70% of average
- Volume > 1.2x average
```

### Formula
```python
price_range = high - low
price_range_pct = price_range / close * 100
avg_range_20 = rolling_mean(price_range_pct, 20)

absorption_long = (
    delta_percentile > 80 AND
    delta > 0 AND
    price_range_pct < avg_range_20 * 0.7 AND
    volume > volume_ma20 * 1.2
)
```

### Parameters
| Parameter | Value |
|-----------|-------|
| Delta Threshold | 80th percentile |
| Range Compression | < 70% of average |
| Volume Threshold | > 1.2x average |
| Lookback | 20 bars |

### Signals Found
- Absorption Long: 21 signals
- Absorption Short: 0 signals

### Performance (with CVD filter)
| Metric | Value |
|--------|-------|
| Total Trades | Very few (21 long only) |
| Win Rate | N/A (insufficient data) |
| Expectancy | N/A |

### Analysis
⚠️ **Too few signals** - only 21 absorption events in 62 days  
⚠️ No short absorption detected in this dataset  

### Recommendation
❌ **Not viable as standalone strategy**  
✅ **Use as confirmation filter only**

---

## STRATEGY 3: IMBALANCE + VOLUME + CVD ⭐ WINNER

### Description
Identifies significant bid/ask imbalances with volume confirmation and trend alignment. This is the strongest enhancement found.

### Logic
```
LONG ENTRY:
- Imbalance percentile > 85 (top 15% ask/bid ratio)
- Volume spike > 1.5x average (confirmation)
- CVD rising (trend aligned)
- Delta > 0

SHORT ENTRY:
- Imbalance percentile < 15 (bottom 15% ask/bid ratio)
- Volume spike > 1.5x average
- CVD falling (trend aligned)
- Delta < 0
```

### Formula
```python
# Calculate smoothed imbalance
imbalance_raw = vol_ask / vol_bid
imbalance_smooth = rolling_mean(imbalance_raw, 5)
imbalance_percentile = percentile_rank(imbalance_smooth, 20)

# Volume spike
volume_spike = volume > (volume_ma20 * 1.5)

# Entry signal
long_signal = (
    imbalance_percentile > 85 AND
    volume_spike AND
    cvd_rising AND
    delta > 0
)
```

### Parameters
| Parameter | Value |
|-----------|-------|
| Imbalance Period | 5-bar smoothing |
| Imbalance Threshold | >85th or <15th percentile |
| Volume Threshold | >1.5x 20-bar average |
| CVD Period | 20-bar EMA |

### Performance
| Metric | Value |
|--------|-------|
| Total Trades | 261 |
| Win Rate | 42.1% |
| Expectancy | **1.56 pts/trade** ⭐ |
| Profit Factor | **1.21** ⭐ |
| Daily Trades | 4.2 |
| Daily P&L (31 MNQ) | $408 |
| Days to $9K | 22 |

### vs Original Strategy
| Metric | Original | Imbalance+Vol | Improvement |
|--------|----------|---------------|-------------|
| Expectancy | 1.52 | 1.56 | +2.6% |
| Profit Factor | 1.19 | 1.21 | +1.7% |
| Win Rate | 44.2% | 42.1% | -2.1% |
| Trades/Day | 10.9 | 4.2 | -61% |

### Analysis
✅ **Higher quality trades** (better expectancy & PF)  
✅ **Volume confirmation** filters weak signals  
✅ **Imbalance detection** shows true aggression  
⚠️ **Fewer trades** (more selective)  
⚠️ **Lower win rate** but better R:R  

### Recommendation
🏆 **Use as primary strategy for quality over quantity**  
🏆 **Best for patient traders**  
⚠️ May want to trade multiple time windows to increase frequency

### Tweaks to Test
- Imbalance threshold: Try 80, 85, 90
- Volume multiplier: Try 1.3x, 1.5x, 2.0x
- Add VWAP filter: Only trade above/below VWAP
- Combine with original for more signals

---

## STRATEGY 4: VWAP BOUNCE/REJECTION

### Description
Uses VWAP (Volume Weighted Average Price) as dynamic support/resistance combined with delta confirmation.

### Logic
```
VWAP BOUNCE LONG:
- Price above VWAP (bullish context)
- Low touched VWAP or upper band (tested support)
- Delta > 0 (buying at support)
- Delta percentile > 75

VWAP REJECTION SHORT:
- Price below VWAP (bearish context)
- High touched VWAP or lower band (tested resistance)
- Delta < 0 (selling at resistance)
- Delta percentile > 75
```

### Formula
```python
# VWAP distance
distance_from_vwap = (close - vwap) / atr14

# Bounce detection
above_vwap = close > vwap
tested_vwap = low <= vwap_upper1  # Tested VWAP +1 std

vwap_bounce_long = (
    above_vwap AND
    tested_vwap AND
    delta > 0 AND
    delta_percentile > 75
)
```

### Parameters
| Parameter | Value |
|-----------|-------|
| VWAP Bands | ±1 standard deviation |
| Delta Threshold | 75th percentile |
| Touch Threshold | Low <= VWAP upper band |

### Performance
| Metric | Value |
|--------|-------|
| Total Trades | 451 |
| Win Rate | 42.8% |
| Expectancy | 0.54 pts/trade |
| Profit Factor | 1.07 |
| Daily Trades | 7.3 |

### Analysis
⚠️ **Lower expectancy** than original (0.54 vs 1.52)  
⚠️ VWAP bands may be too tight in this data  
⚠️ Too many false bounces  

### Recommendation
❌ **Not recommended as primary strategy**  
✅ **Use as secondary confirmation only**

---

## STRATEGY 5: COMBINATION (ALL FILTERS)

### Description
Combines delta, volume spike, imbalance, and CVD for highest quality signals.

### Logic
```
COMBO LONG:
- Delta percentile > 80
- Delta > 0
- CVD rising
- (Volume spike OR Absorption)
- Imbalance percentile > 60

COMBO SHORT:
- Delta percentile > 80
- Delta < 0
- CVD falling
- (Volume spike OR Absorption)
- Imbalance percentile < 40
```

### Performance
| Metric | Value |
|--------|-------|
| Total Trades | 186 |
| Win Rate | 44.6% |
| Expectancy | 0.32 pts/trade |
| Profit Factor | 1.05 |
| Daily Trades | 3.0 |

### Analysis
⚠️ **Too many filters** = too few signals  
⚠️ **Low expectancy** despite high win rate  
⚠️ Over-filtered  

### Recommendation
❌ **Too restrictive** - avoid  
✅ **Use fewer filters for better balance**

---

## SUMMARY COMPARISON

| Rank | Strategy | Trades | WR% | Exp (pts) | PF | Quality |
|------|----------|--------|-----|-----------|-----|---------|
| 1 | **Imbalance+Vol+CVD** ⭐ | 261 | 42.1% | **1.56** | **1.21** | Highest |
| 2 | Original (Delta+CVD) | 677 | 44.2% | 1.52 | 1.19 | High |
| 3 | VWAP Bounce | 451 | 42.8% | 0.54 | 1.07 | Medium |
| 4 | Combo (All Filters) | 186 | 44.6% | 0.32 | 1.05 | Low |
| - | Absorption Only | 21 | N/A | N/A | N/A | Too few |

---

## FINAL RECOMMENDATIONS

### Option A: Quality Focus (Recommended)
**Use Strategy 3: Imbalance + Volume + CVD**
- Best expectancy (1.56 pts)
- Best profit factor (1.21)
- Highest quality signals
- Trade 4-5 times/day
- Pass in ~22 days

### Option B: Quantity Focus
**Use Strategy 1: Original Delta > 85 + CVD**
- More trades (11/day)
- Still profitable (1.52 exp)
- Pass in ~9 days
- More commissions

### Option C: Hybrid Approach
**Use both strategies:**
1. Imbalance+Vol+CVD as primary (A+ setups)
2. Original Delta+CVD as secondary (A setups)
3. Different sizing for each

---

## IMPLEMENTATION CODE

See `strategy_5min_proper.py` for:
- Feature calculation functions
- Backtest engine
- Signal generation
- Performance metrics

---

## NEXT STEPS

1. [ ] Paper trade Strategy 3 for 1 week
2. [ ] Compare fills vs backtest
3. [ ] Optimize parameters (imbalance threshold, volume multiplier)
4. [ ] Test on ES data
5. [ ] Add VWAP context to Strategy 3
6. [ ] Consider trading 2 time windows (10-12 + 13-15)

---

**Document Version**: 1.0  
**Best Strategy**: Imbalance + Volume + CVD (1.56 expectancy)  
**Date**: February 16, 2026
