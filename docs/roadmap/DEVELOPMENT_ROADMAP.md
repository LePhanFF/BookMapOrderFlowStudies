# Development Roadmap: 3-Strategy Study
## Trend Following + Mean Reversion + Two Hour Options

**Branch**: dev-next-study  
**Status**: 🚀 IMPLEMENTATION PHASE  
**Last Updated**: February 16, 2026  

---

## 🎯 Selected Strategies

### 1. **Trend Following Breakout** ⭐ PRIMARY
- **Instruments**: MNQ, MES futures
- **Win Rate Target**: 58%
- **R:R Target**: 3:1
- **Why**: Best combination of win rate, R:R, and expectancy

### 2. **Mean Reversion** ⭐ COMPLEMENTARY
- **Instruments**: MNQ, MES futures
- **Win Rate Target**: 65% (in chop)
- **R:R Target**: 1.5:1
- **Why**: Counter-trend for range-bound days, smooths equity curve

### 3. **Two Hour Trader (Options)** ⭐ OPTIONS DIVERSIFICATION
- **Instruments**: QQQ, SPY, NQ options, ES options
- **Win Rate Target**: 60-79%
- **R:R Target**: 2:1
- **Why**: Different instrument class, highest reported win rate

---

## 📊 Strategy Comparison (Why These 3?)

| Strategy | Market Condition | Win Rate | R:R | Best For |
|----------|------------------|----------|-----|----------|
| **Trend Following** | Trending | 58% | 3:1 | Big moves |
| **Mean Reversion** | Range-bound | 65% | 1.5:1 | Chop days |
| **Two Hour** | Opening volatility | 60-79% | 2:1 | Options, high WR |

**Combined Portfolio**: Works in ALL market conditions

---

## 🗓️ Implementation Timeline

### **Week 1: Data & Infrastructure** (Feb 17-23)

#### Day 1-2: Data Collection
```
Priority 1: Futures Data (MNQ, MES)
- [ ] Download 90 days 1-min data
- [ ] Download 5-min data for HTF
- [ ] Calculate indicators (EMA20/50, ADX, Bollinger Bands)
- [ ] Store in processed format

Priority 2: Options Data Research
- [ ] Research data sources (CBOE, Polygon.io, Tradier)
- [ ] SPX/SPY options chain data
- [ ] QQQ options data
- [ ] NQ/ES futures options
- [ ] Cost analysis
```

#### Day 3-4: Build Core Engines
```
- [ ] Create trend_following_engine.py
- [ ] Create mean_reversion_engine.py
- [ ] Create options_backtest_framework.py
- [ ] Shared utilities (data loader, metrics calculator)
```

#### Day 5-7: Testing Infrastructure
```
- [ ] Unit tests for all engines
- [ ] Validation on sample data
- [ ] Performance benchmarks
```

### **Week 2: Backtesting & Optimization** (Feb 24-Mar 2)

#### Trend Following Backtest
```
- [ ] Test 20-period high/low breakout
- [ ] Test prior day high/low breakout
- [ ] Test VWAP band breakout
- [ ] Optimize HTF timeframe (15-min vs 30-min)
- [ ] Find best trend filter (ADX threshold)
- [ ] Run 90-day backtest
```

#### Mean Reversion Backtest
```
- [ ] Test Bollinger Band touch (2.0 vs 2.5 std)
- [ ] Test RSI extreme (30/70 vs 25/75)
- [ ] Test divergence detection
- [ ] Optimize regime filter (ADX < 25)
- [ ] Run 90-day backtest
```

#### Options Research
```
- [ ] Finalize data source
- [ ] Build options pricing model
- [ ] Test entry variations (momentum, mean reversion, VWAP)
- [ ] Paper trade analysis (if data available)
```

### **Week 3: Analysis & Comparison** (Mar 3-9)

#### Performance Analysis
```
- [ ] Calculate all metrics (WR, expectancy, drawdown, PF)
- [ ] Compare to Order Flow baseline
- [ ] Risk-adjusted returns (Sharpe, Sortino)
- [ ] Consecutive loss analysis
- [ ] Drawdown recovery analysis
```

#### Optimization
```
- [ ] Parameter sweeps for each strategy
- [ ] Walk-forward analysis
- [ ] Market regime testing (trend vs chop)
- [ ] Time-of-day analysis
```

### **Week 4: Implementation** (Mar 10-16)

#### NinjaTrader Scripts
```
- [ ] Code TrendFollowing_Breakout.cs
- [ ] Code MeanReversion_BB.cs
- [ ] Code TwoHour_Options.cs (if data supports)
- [ ] Test compilation
```

#### Documentation
```
- [ ] Final strategy guides
- [ ] NT8 setup instructions
- [ ] Risk management protocols
- [ ] Comparison report
```

### **Week 5+: Paper Trading** (Mar 17+)

#### Validation
```
- [ ] 2 weeks paper trading each strategy
- [ ] Verify fills match backtest
- [ ] Check slippage
- [ ] Refine parameters
```

---

## 🔬 Data Requirements

### Futures Data (MNQ, MES)
```
Source: Existing csv/ folder + additional download
Timeframes: 1-min, 5-min, 15-min
Period: 90 days (Nov 2025 - Feb 2026)
Indicators needed:
  - EMA 20, 50
  - ADX (14)
  - Bollinger Bands (20, 2.0)
  - RSI (14)
  - ATR (14)
  - VWAP
```

### Options Data (SPY, QQQ, SPX)
```
Source: To be determined
  - Polygon.io ($199/month)
  - CBOE Delayed (free)
  - Tradier ($10/month)
  - Or paper trade analysis

Data needed:
  - Option chain (strikes, expirations)
  - 1-min OHLC for underlying
  - Greeks (delta, theta, vega)
  - Volume and open interest
  - Bid-ask spreads

Time Period: 90 days
Strikes: ATM, +/- 5 strikes
Expirations: 0-7 DTE
```

---

## 🏗️ Architecture

### File Structure
```
research/strategy-studies/
├── MASTER_INDEX.md (✅ Done)
├── TREND_FOLLOWING_STUDY.md (✅ Done)
├── MEAN_REVERSION_STUDY.md (✅ Done)
├── TWO_HOUR_TRADER_STUDY.md (✅ Done)
└── OPENING_RANGE_BREAKOUT_STUDY.md (✅ Done - observe only)

src/
├── backtest_engines/
│   ├── __init__.py
│   ├── trend_following_engine.py
│   ├── mean_reversion_engine.py
│   └── options_backtest_engine.py
├── data/
│   ├── futures_loader.py
│   └── options_loader.py
├── indicators/
│   ├── technical.py (EMA, ADX, BB, RSI)
│   └── options_greeks.py
├── analysis/
│   ├── metrics.py
│   ├── comparison.py
│   └── visualization.py
└── tests/
    └── test_engines.py

results/
├── trend_following/
│   ├── backtest_results.csv
│   ├── optimization_report.md
│   └── equity_curve.png
├── mean_reversion/
│   └── ...
└── two_hour/
    └── ...

ninjatrader/
├── TrendFollowing_Breakout.cs
├── MeanReversion_BB.cs
└── TwoHour_Options.cs
```

---

## 📈 Success Metrics

### For Each Strategy
```
Must Achieve:
  - Win Rate > 50%
  - Profit Factor > 1.3
  - Expectancy > $50/trade
  - Max Drawdown < $3,000
  - Consecutive losses < 7

Nice to Have:
  - Win Rate > 55%
  - Profit Factor > 2.0
  - Expectancy > $100/trade
  - Smooth equity curve
```

### Comparison to Order Flow
```
Target: Beat or match Order Flow performance
Order Flow Baseline:
  - Win Rate: 44.2%
  - Daily P&L: $1,027 (31 contracts)
  - Expectancy: $94/trade
  - Drawdown: ~$2,500
```

---

## ⚠️ Risk Management

### Per Strategy
```
Trend Following:
  - Max daily loss: $1,200 (3 trades)
  - Stop: Technical level (breakout failure)
  - Time: No entries after 3:00 PM
  - ADX filter: Skip if ADX < 20

Mean Reversion:
  - Max daily loss: $800 (4 trades)
  - Stop: 2-5 points (tight)
  - Time: 5-bar max hold
  - Regime: Skip if ADX > 30

Two Hour Options:
  - Max daily loss: $800 (2 trades)
  - Stop: 40-50% of premium
  - Time: Hard exit 11:30 AM
  - VIX filter: Skip if VIX > 30
```

---

## 🎯 Immediate Next Steps

### Today (Feb 16)
```
1. ✅ Create this roadmap
2. [ ] Set up src/ directory structure
3. [ ] Build trend_following_engine.py (start with this - highest priority)
4. [ ] Test with existing MNQ data
```

### Tomorrow (Feb 17)
```
1. [ ] Complete trend following backtest
2. [ ] Generate initial results
3. [ ] Start mean reversion engine
4. [ ] Research options data sources
```

---

## 📝 Notes

### Opening Range Breakout
**Status**: Skip for now, observe only
**Reason**: Low frequency (1-2 trades/day) leads to lower daily P&L despite good win rate
**Action**: May revisit later as complementary strategy

### Options Data Challenge
**Issue**: Options data is expensive ($199/month for Polygon.io)
**Solutions**:
  1. Start with paper trading analysis (free)
  2. Use delayed data (CBOE)
  3. Focus on futures strategies first
  4. Add options later when profitable

### Priority Order
1. **Trend Following** (highest potential, futures-based)
2. **Mean Reversion** (complementary, futures-based)
3. **Two Hour Options** (requires data research)

---

## ✅ Checklist

### Week 1
- [ ] Data infrastructure ready
- [ ] Trend Following engine coded
- [ ] Mean Reversion engine coded
- [ ] Initial backtests run
- [ ] Options data source selected

### Week 2
- [ ] All backtests complete
- [ ] Parameters optimized
- [ ] Results documented
- [ ] Comparison to Order Flow done

### Week 3
- [ ] Winning strategy identified
- [ ] NinjaTrader script drafted
- [ ] Risk management tested
- [ ] Ready for paper trading

---

**Ready to start building? Let's code the Trend Following engine first!** 🚀

*Document Version: 1.0*  
*Status: Roadmap Complete - Ready for Implementation*
