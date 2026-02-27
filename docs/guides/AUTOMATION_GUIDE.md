# AUTOMATION GUIDE
## Automated Order Entry for Order Flow Strategy

---

## NQ vs ES VERDICT

**NQ (Nasdaq) is significantly better:**

| Metric | NQ | ES | Winner |
|--------|-----|-----|--------|
| Expectancy | **1.52 pts** | 0.20 pts | **NQ** |
| Daily P&L | **$993** | $685 | **NQ** |
| Days to Pass | **9 days** | 13 days | **NQ** |
| Win Rate | 44.2% | 44.8% | Similar |
| ATR | 16.3 pts | 3.1 pts | NQ more volatile |

**Trade NQ (MNQ - Micro Nasdaq)**, not ES.

---

## AUTOMATION OPTIONS

### Option 1: NinjaTrader Automated Strategy (Recommended)

**Pros:**
- Native support for order flow data
- Built-in ATM (Advanced Trade Management)
- Can read Bookmap/NinjaTrader volumetric data
- Supports limit orders, stop orders, OCO

**Cons:**
- Requires NinjaTrader license
- Learning curve for C# strategy development

**Implementation:**
```csharp
// Pseudo-code for NinjaTrader strategy
protected override void OnBarUpdate()
{
    // Calculate Delta Percentile
    double delta = Volume[0].AskVolume - Volume[0].BidVolume;
    double deltaPct = CalculatePercentile(delta, 20);
    
    // Check CVD trend
    bool cvdRising = CumulativeDelta[0] > SMA(CumulativeDelta, 20)[0];
    
    // Entry logic
    if (deltaPct > 85 && delta > 0 && cvdRising && ToTime(Time[0]) >= 100000 && ToTime(Time[0]) < 130000)
    {
        double stopDist = ATR(14)[0] * 0.4;
        double targetDist = stopDist * 2.0;
        
        EnterLong(31);  // 31 MNQ contracts
        SetStopLoss(CalculationMode.Price, Close[0] - stopDist);
        SetProfitTarget(CalculationMode.Price, Close[0] + targetDist);
    }
}
```

### Option 2: Bookmap + API Integration

**Pros:**
- Real-time order flow visualization
- Can export signals via API
- Best for manual confirmation + auto-execution

**Cons:**
- Not fully automated
- Requires API bridge to broker

**Implementation:**
```python
# Pseudo-code for Bookmap API integration
import bookmap

def on_order_flow_update(ask_volume, bid_volume, price):
    delta = ask_volume - bid_volume
    delta_percentile = calculate_percentile(delta, lookback=20)
    
    if delta_percentile > 85:
        if time_in_range(10, 0, 13, 0):
            send_order(
                symbol='MNQ',
                side='BUY',
                quantity=31,
                order_type='LIMIT',
                price=price,
                stop_loss=price - (atr * 0.4),
                take_profit=price + (atr * 0.8)
            )
```

### Option 3: Python + Interactive Brokers API

**Pros:**
- Full control
- Can integrate with existing backtest code
- Free

**Cons:**
- Need to handle order flow data feed separately
- More complex to maintain
- Latency considerations

**Implementation:**
```python
from ib_insync import *
import pandas as pd

class OrderFlowStrategy:
    def __init__(self):
        self.ib = IB()
        self.ib.connect('127.0.0.1', 7497, clientId=1)
        self.mnq = Future('MNQ', '202403', 'CME')
        
    def on_bar(self, bar):
        # Calculate signals
        delta = bar.askVolume - bar.bidVolume
        delta_pct = self.calculate_percentile(delta)
        
        if self.should_enter_long(delta_pct, bar):
            self.enter_long(bar.close)
    
    def enter_long(self, price):
        atr = self.calculate_atr()
        stop_price = price - (atr * 0.4)
        target_price = price + (atr * 0.8)
        
        # Bracket order: Entry + Stop + Target
        bracket = self.ib.bracketOrder(
            action='BUY',
            quantity=31,
            limitPrice=price,
            takeProfitPrice=target_price,
            stopLossPrice=stop_price
        )
        
        for order in bracket:
            self.ib.placeOrder(self.mnq, order)
```

---

## ORDER TYPES COMPARISON

### Limit Orders (Recommended)

**Pros:**
- ✅ Better fills (no slippage)
- ✅ Lower fees on some platforms
- ✅ Control entry price

**Cons:**
- ❌ May not fill in fast markets
- ❌ Missed entry if price gaps

**Best For:**
- Patient entries
- Liquid markets (NQ is very liquid)
- When signal is clear

### Market Orders

**Pros:**
- ✅ Guaranteed fill
- ✅ Fast execution

**Cons:**
- ❌ Slippage (1-2 ticks typical)
- ❌ Higher fees

**Best For:**
- Breaking out of range
- News events
- When you MUST get in

### Recommendation: **Use Limit Orders**

Place limit order at signal bar close or 1 tick better.

---

## AUTOMATED EXECUTION RULES

### Entry Rules
```
IF:
  - Time is 10:00-13:00 ET
  - Delta percentile > 85
  - Delta direction matches trade (long=delta>0, short=delta<0)
  - CVD is trending in trade direction
  - Not already in position

THEN:
  - Place limit order at bar close
  - Quantity: 31 MNQ contracts
  - Stop: 0.4x ATR (6.4 points)
  - Target: 2.0x ATR (12.8 points)
  - Max hold: 8 bars
```

### Exit Rules (OCO - One Cancels Other)
```
OCO Bracket:
  - Stop Loss: Entry - 6.4 points
  - Profit Target: Entry + 12.8 points
  
IF either hits:
  - Cancel the other order automatically
  - Close position
```

### Time-Based Exit
```
IF position held for 8 bars:
  - Close at market
  - Take whatever P&L
```

### Risk Management Overrides
```
Daily Loss Limit: $2,000
IF daily P&L <= -$2,000:
  - Stop trading for day
  - Close any open positions

Consecutive Losses: 5
IF 5 losses in a row:
  - Reduce size by 50%
  - Take 30 min break
  - Reassess strategy
```

---

## IMPLEMENTATION CHECKLIST

### Phase 1: Setup (Week 1)
- [ ] Choose platform (NinjaTrader recommended)
- [ ] Open futures account with margin
- [ ] Connect to data feed (Bookmap or NinjaTrader)
- [ ] Paper trade for 1 week
- [ ] Verify signals match backtest

### Phase 2: Basic Automation (Week 2)
- [ ] Program entry logic
- [ ] Test with 1 contract
- [ ] Verify OCO orders work
- [ ] Check fills vs backtest

### Phase 3: Full Automation (Week 3)
- [ ] Scale to 31 contracts
- [ ] Add risk management rules
- [ ] Monitor for 1 week
- [ ] Compare results to backtest

### Phase 4: Go Live (Week 4+)
- [ ] Start with 10 contracts
- [ ] Scale up after 5 winning days
- [ ] Daily review of trades
- [ ] Weekly strategy review

---

## TECHNICAL REQUIREMENTS

### Hardware
- Reliable internet (fiber preferred)
- Backup internet (mobile hotspot)
- Computer with 16GB+ RAM
- UPS (uninterruptible power supply)

### Software
- NinjaTrader 8+ (or equivalent)
- Real-time data feed (Bookmap, DTN IQFeed)
- VPS (Virtual Private Server) for 24/7 operation
- Backup system on secondary computer

### Data
- Level 2 order book (for order flow)
- 1-minute or 5-minute bars
- Historical data for backtesting
- Real-time delta calculations

---

## MONITORING & ALERTS

### Daily Monitoring
- Number of trades taken
- Win/loss count
- P&L vs expected
- Any technical issues

### Alerts
- Position entered
- Position exited (stop or target)
- Daily loss limit reached
- Connection issues
- Unusual market conditions

### Reporting
```
Daily Report:
- Date: 
- Trades: X
- Wins: X, Losses: X
- P&L: $X
- vs Expected: X%
- Issues: 
```

---

## RECOMMENDED SETUP

**For Fast Implementation:**
1. **Platform**: NinjaTrader 8
2. **Broker**: Interactive Brokers or AMP Futures
3. **Data**: Bookmap + NinjaTrader data
4. **Strategy**: Import backtest logic into NinjaScript
5. **Testing**: 1 week paper trade
6. **Go Live**: Start with 10 contracts, scale up

**Expected Timeline:**
- Setup: 1 week
- Paper trading: 1 week
- Live with small size: 2 weeks
- Full size (31 contracts): Week 4
- Pass eval: Week 5-6

---

## IMPORTANT WARNINGS

⚠️ **Automated trading is risky:**
- Always have manual override
- Monitor first week closely
- Have stop-loss in broker system (not just strategy)
- Test in simulation first
- Keep position size conservative initially

⚠️ **Technical failures happen:**
- Internet can go down
- Platform can crash
- API can disconnect
- Always have backup plan

⚠️ **Markets change:**
- Strategy may stop working
- Regular review required
- Be ready to turn off automation

---

*Document Version: 1.0*
*Recommended: Use NinjaTrader with limit orders*
*Start with paper trading for 1 week minimum*
