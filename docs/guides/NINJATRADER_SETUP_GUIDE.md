# NINJATRADER 8 AUTOMATED SETUP GUIDE
## Complete Implementation for Evaluation Factory

---

## OVERVIEW

This guide walks you through setting up **fully automated** order flow trading on NinjaTrader 8 for passing prop firm evaluations.

**What you'll build:**
- Automated strategy that enters/exits without manual intervention
- Risk management that stops trading when limits hit
- Real-time monitoring of P&L, trades, and status
- ATM (Advanced Trade Management) for bracket orders

**Time to setup:** 2-3 hours  
**Skill level:** Intermediate (basic NinjaTrader knowledge)

---

## PREREQUISITES

### Required Software
1. **NinjaTrader 8** (download from ninjatrader.com)
2. **Data Feed** (one of these):
   - NinjaTrader Lifetime License (includes data)
   - Kinetick Data Feed
   - Interactive Brokers connection
   - **Note:** You need volumetric/Level 2 data for order flow

3. **Prop Firm Account** (evaluation):
   - TradeDay, Apex, Topstep, etc.
   - Account connected to NinjaTrader

### Required Hardware
- Computer with 16GB+ RAM
- Reliable internet connection (fiber preferred)
- Backup internet (mobile hotspot)

---

## STEP 1: INSTALL NINJATRADER 8

### Download & Install
1. Go to https://ninjatrader.com/download-ninjatrader/
2. Download NinjaTrader 8
3. Run installer (Windows only)
4. Follow setup wizard
5. **Important:** Choose "Simulated Data" for now (we'll add real data later)

### Activate License
1. Open NinjaTrader 8
2. Go to `Help` → `License Key`
3. Enter your license key
4. Or use free version (limited features, but works for testing)

---

## STEP 2: CONNECT DATA FEED

### For TradeDay/Prop Firm:
1. Get your account credentials from prop firm
2. In NinjaTrader: `Connections` → `Configure`
3. Click `Add` → Select your prop firm (TradeDay, Rithmic, etc.)
4. Enter credentials
5. Click `Enable`
6. Wait for connection status to show "Connected"

### For Volumetric Data (Order Flow):
**Critical:** You need bid/ask volume data (Level 2)
1. Contact your data provider
2. Ensure "Volumetric Data" or "Level 2" is enabled
3. In NinjaTrader: `Tools` → `Options` → `Data`
4. Check "Enable volumetric data"

---

## STEP 3: COPY STRATEGY FILE

### Locate Strategy Directory
```
Navigate to:
Documents\NinjaTrader 8\bin\Custom\Strategies\
```

**Or in NinjaTrader:**
1. `Tools` → `Edit NinjaScript` → `Strategy`
2. This opens the strategies folder

### Copy Strategy File
1. Copy `DualOrderFlowEvaluation.cs` from this project
2. Paste into the `Strategies` folder above
3. **File path should be:**
   ```
   Documents\NinjaTrader 8\bin\Custom\Strategies\DualOrderFlowEvaluation.cs
   ```

---

## STEP 4: COMPILE STRATEGY

### Open Strategy Editor
1. In NinjaTrader: `Tools` → `Edit NinjaScript` → `Strategy`
2. Find `DualOrderFlowEvaluation` in the list
3. Click on it to open

### Compile the Code
1. Press **F5** (or click the green "Compile" button)
2. Check the "NinjaScript Output" window at bottom
3. You should see: **"Compilation successful"**

### If You Get Errors:
**Common issues:**

1. **"Series<double> not found"**
   - Make sure you're using NinjaTrader 8.0.25.0 or higher
   - Update NinjaTrader: `Help` → `Check for Updates`

2. **"Volume[0].AskVolume not found"**
   - You don't have volumetric data enabled
   - Contact data provider to add Level 2 data

3. **"ATR not found"**
   - Make sure `using NinjaTrader.NinjaScript;` is at top of file

4. **Other errors**
   - Copy the exact error message
   - Check NinjaTrader forums or support

---

## STEP 5: SETUP CHART

### Open MNQ Chart
1. `File` → `New` → `Chart`
2. In the Data Series window:
   - **Instrument:** MNQ (Micro E-mini Nasdaq-100)
   - **Type:** Minute
   - **Value:** 1
   - **Trading hours:** US Equities RTH (9:30-16:00)
3. Click `OK`

### Add Indicators (Optional but Helpful)
1. Right-click chart → `Indicators`
2. Add these indicators:
   - **VWAP** (Volume Weighted Average Price)
   - **EMA(20)** (Exponential Moving Average)
   - **ATR(14)** (Average True Range) - helps visualize stop distance
3. Click `Apply` then `OK`

### Set Chart Appearance
1. Right-click chart → `Properties`
2. Set colors you like (dark theme recommended)
3. **Important:** Check "Show quick buttons" (top right)
4. Click `OK`

---

## STEP 6: ADD STRATEGY TO CHART

### Enable Strategy
1. Right-click chart → `Strategies`
2. Click the `+` button (Add Strategy)
3. Select `DualOrderFlowEvaluation` from list
4. Click `OK`

### Configure Parameters
**You'll see these settings - fill them in:**

**1. Risk Parameters:**
```
Max Contracts: 31 (for Account 1-2)
Daily Loss Limit: -1500 (stop if hit)
Max Consecutive Losses: 5
Max Trades Per Day: 15
```

**2. Time Window:**
```
Session Start: 100000 (10:00 AM)
Session End: 130000 (1:00 PM)
```

**3. Signal Parameters:**
```
Delta Period: 20
Delta Threshold: 85
Imbalance Period: 20
Imbalance Threshold: 85
Volume Spike Multiplier: 1.5
```

**4. Exit Parameters:**
```
ATR Period: 14
Stop Multiplier: 0.4
Reward Multiplier: 2.0
Max Hold Bars: 8
```

### Set Order Type
1. In the strategy panel, scroll down
2. **ATM Strategy:** Select `DualStrategyATM` (or create one - see Step 7)
3. **Quantity:** Set to 31 (or whatever you're using)
4. **Time In Force:** GTC (Good Till Canceled)

### Enable Strategy
1. Check the box next to strategy name
2. Click `Apply`
3. Click `OK`

**You should see:**
- Strategy info text appears on chart (top left)
- "Dual Order Flow - EVALUATION MODE"
- Shows contracts and session time

---

## STEP 7: CREATE ATM STRATEGY (BRACKET ORDERS)

### What is ATM?
ATM = Advanced Trade Management. It automatically places:
- Entry order
- Stop loss order  
- Profit target order
- All linked as OCO (One Cancels Other)

### Create ATM Template
1. `Tools` → `Edit ATM Strategy`
2. Click `New`
3. Name: `DualStrategyATM`
4. Configure:

**Stop Strategy:**
```
Stop Loss: Enabled
Profit Target: Enabled
Tick Offset: 0
```

**Auto Breakeven:**
```
Enabled: False (we handle this in code)
```

**Auto Trail:**
```
Enabled: False (not using trailing stops)
```

5. Click `OK`

### Alternative: Use Strategy's Built-in OCO
The NinjaScript code already handles OCO, so ATM is optional but recommended for backup.

---

## STEP 8: TEST IN SIMULATION MODE

### Critical: Test Before Going Live!

### Enable Simulation
1. In NinjaTrader control panel (top left)
2. Make sure it says **"Simulated"** not "Live"
3. If it says "Live", click dropdown and select "Simulated"

### Run Backtest
1. On your chart with strategy enabled
2. Press the **Playback** button (looks like play button)
3. Or: Right-click chart → `Playback`
4. Select date range (last 30 days)
5. Click `Start`

### Watch Strategy Trade
You should see:
- Green arrows (LONG entries)
- Red arrows (SHORT entries)
- Diamonds at exits (green = win, red = loss)
- Daily P&L updating (top right)

### Verify Settings
1. Check that trades only happen 10:00-13:00
2. Verify position size is correct
3. Confirm stops are placed correctly
4. Check that max loss limits work

### Common Issues in Simulation
**No trades happening:**
- Check time zone (should be ET)
- Verify session hours (100000-130000)
- Make sure you have volumetric data

**Wrong position size:**
- Check `MaxContracts` parameter
- Verify ATM strategy quantity

**Stops not placed:**
- Check if ATM strategy is selected
- Verify you have ATR indicator on chart

---

## STEP 9: GO LIVE (EVALUATION ACCOUNT)

### Only After Successful Testing!

### Switch to Live
1. Close all charts
2. In Control Panel, click dropdown where it says "Simulated"
3. Select your prop firm account (e.g., "TradeDay")
4. Wait for "Connected" status

### Reopen Chart
1. `File` → `Recent` → Select your MNQ chart
2. Strategy should still be there (check parameters!)
3. Verify it says "LIVE" in the top right

### Enable Strategy
1. Right-click chart → `Strategies`
2. Check the box next to `DualOrderFlowEvaluation`
3. **CRITICAL:** Click `Apply` then `OK`
4. Strategy is now LIVE and will auto-trade

### Monitor First Hour
1. Watch the first 2-3 trades closely
2. Verify:
   - Entry prices match signal bar close
   - Stops at correct price
   - Targets at correct price
   - Quantity correct (31 contracts)

### Daily Routine
**9:30 AM:**
1. Open NinjaTrader
2. Connect to prop firm account
3. Open MNQ chart with strategy
4. Verify strategy is enabled
5. Check that "Daily P&L" shows $0

**10:00 AM:**
- Strategy starts automatically

**1:00 PM:**
- Strategy stops automatically
- Review trades in `Account` → `Orders` tab

**4:00 PM:**
- Check daily P&L
- Verify no rule violations
- Plan next day

---

## STEP 10: MANAGE MULTIPLE ACCOUNTS (FACTORY MODE)

### Setup for Sequential Passing

**Account 1 (Current):**
```
Chart 1: MNQ 1-min
Strategy: DualOrderFlowEvaluation
Parameters: 31 contracts, aggressive
Status: Trading now
```

**After Account 1 Passes (Day 9):**
1. Disable strategy on Chart 1
2. Switch to Account 2 in Connections
3. Open Chart 2 (MNQ 1-min)
4. Add strategy with same parameters
5. Enable and trade Account 2

### Tracking Spreadsheet
Create Excel/Google Sheets:

| Account | Start Date | Status | Days | P&L | Pass Date |
|---------|-----------|--------|------|-----|-----------|
| 1 | 2026-02-16 | Trading | 3 | $3,200 | - |
| 2 | - | Waiting | - | - | - |
| 3 | - | Waiting | - | - | - |

---

## RISK MANAGEMENT FEATURES

### Built-in Safeguards

**1. Daily Loss Limit:**
- Automatically stops trading if daily P&L hits limit
- Resets next day
- Displays "DAILY LOSS LIMIT REACHED" on chart

**2. Consecutive Losses:**
- Tracks losing streak
- Stops trading after 5 consecutive losses
- Prevents tilt trading

**3. Max Trades:**
- Limits to 15 trades/day
- Prevents over-trading
- Can adjust per account

**4. Time Window:**
- Only trades 10:00-13:00 ET
- Ignores other times
- Prevents overnight exposure

**5. Session Close:**
- Exits all positions at 15:55
- Avoids after-hours risk

---

## MONITORING & ALERTS

### On-Chart Display

**Top Left:**
```
Dual Order Flow - EVALUATION MODE
Contracts: 31
Session: 10:00-13:00 ET
```

**Top Right:**
```
Daily P&L: $1,240 | Trades: 5/15 | Cons. Losses: 0/5
```

**Status Messages:**
- White text: Normal operation
- Yellow text: Warning (down $500+)
- Orange text: Caution (down $1000+)
- Red text: Stopped (hit limit)

### Order Notifications
To get alerts:
1. `Tools` → `Alerts`
2. Click `New`
3. Set conditions:
   - Order filled
   - Daily P&L threshold
   - Connection lost
4. Configure email/SMS notifications

---

## TROUBLESHOOTING

### Strategy Not Trading

**Check these:**
1. Is market open? (9:30-16:00 ET)
2. Is it within trading hours? (10:00-13:00)
3. Is strategy enabled? (check box)
4. Is there enough data? (need 20 bars minimum)
5. Do you have volumetric data?

### Wrong Position Size

**Possible causes:**
1. ATM strategy overriding
2. MaxContracts parameter wrong
3. Account margin limit

**Fix:**
- Disable ATM strategy
- Set MaxContracts in strategy parameters
- Check account allows that size

### Orders Rejected

**Common reasons:**
1. Insufficient margin
2. Outside trading hours
3. Invalid stop price
4. Connection issues

**Fix:**
- Check account margin
- Verify session hours
- Check NinjaScript output for errors

### Connection Lost

**What to do:**
1. Don't panic
2. Check internet connection
3. NinjaTrader will auto-reconnect
4. If position open, manage manually
5. Consider manual stop in broker platform as backup

---

## BACKUP & SAFETY

### Always Have Manual Override

**Before going live:**
1. Test manual entry/exit buttons
2. Know how to disable strategy instantly
3. Have broker platform open (TradeDay web/app)
4. Set manual stops in broker as backup

### Emergency Stop
**If strategy goes crazy:**
1. Right-click chart → `Strategies`
2. Uncheck strategy box
3. Click `Apply`
4. Manually close any open positions

### Daily Backup
1. Export daily trade log: `Account` → `Orders` → `Export`
2. Save to cloud (Google Drive, Dropbox)
3. Track in spreadsheet

---

## NEXT STEPS

### Once Running
1. **Paper trade 1 week** (even with good backtest)
2. **Verify fills match backtest** expectations
3. **Check daily** that limits work
4. **Review weekly** and optimize if needed

### When Account 1 Passes
1. **Withdraw profit** (if allowed)
2. **Fund Account 2** evaluation
3. **Repeat process**
4. **Build spreadsheet** to track all accounts

### When All 5 Funded
1. **Switch to conservative mode** (2 contracts, wider stops)
2. **Trade different time windows** per account
3. **Scale gradually** as cushion builds
4. **Withdraw monthly** 50% of profits

---

## SUPPORT & RESOURCES

### NinjaTrader Resources
- **NinjaScript Help:** `Help` → `NinjaScript Editor Help`
- **Forum:** https://ninjatrader.com/support/forum/
- **Documentation:** https://ninjatrader.com/support/nt8-help/

### Prop Firm Support
- **TradeDay:** Support portal in dashboard
- **Technical:** Ask about NinjaTrader connection
- **Rules:** Clarify any restrictions

### Code Issues
- Check `Documents\NinjaTrader 8\trace\` for error logs
- Post on NinjaTrader forums with code snippet
- Or contact me with specific error message

---

## QUICK START CHECKLIST

- [ ] NinjaTrader 8 installed
- [ ] License activated
- [ ] Data feed connected (volumetric data!)
- [ ] Strategy file copied to Strategies folder
- [ ] Strategy compiled successfully (F5)
- [ ] MNQ 1-min chart created
- [ ] Strategy added to chart
- [ ] Parameters configured (31 contracts, 10-1pm)
- [ ] ATM strategy created (optional)
- [ ] Tested in simulation mode (1 week)
- [ ] Verified trades match expectations
- [ ] Switched to live account
- [ ] Enabled strategy
- [ ] Monitoring first trades
- [ ] Daily trade log exported

---

## SUMMARY

**You now have:**
1. ✅ Complete NinjaScript strategy
2. ✅ Step-by-step setup guide
3. ✅ Automated risk management
4. ✅ Real-time monitoring
5. ✅ Factory mode for multiple accounts

**Timeline:**
- Setup: 2-3 hours today
- Testing: 1 week
- Account 1: Pass in 9 days
- All 5 accounts: Funded in 65 days

**Ready to start?** Open NinjaTrader and begin Step 1!

---

*Document Version: 1.0*  
*Strategy: DualOrderFlowEvaluation.cs*  
*Platform: NinjaTrader 8*  
*Instrument: MNQ (Micro E-mini Nasdaq-100)*
