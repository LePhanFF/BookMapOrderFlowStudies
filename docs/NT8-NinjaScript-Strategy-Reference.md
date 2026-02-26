# NinjaTrader 8 NinjaScript Strategy Development Reference

Comprehensive reference for building automated NQ/MNQ strategies in NinjaTrader 8
without ATM strategies. Focused on reliability and manual order management.

---

## 1. Strategy Class Lifecycle (OnStateChange)

Every NinjaScript strategy inherits from `Strategy` and processes through a defined
sequence of `State` values. OnStateChange() fires each time the state transitions.

### State Progression (in order)

```
SetDefaults -> Configure -> Active -> DataLoaded -> Historical -> Transition -> Realtime -> Terminated
```

### Complete Lifecycle Template

```csharp
public class MyStrategy : Strategy
{
    // Class-level variables (survive across states)
    private Order entryOrder = null;
    private Order stopOrder  = null;
    private Order targetOrder = null;
    private bool breakEvenTriggered = false;

    protected override void OnStateChange()
    {
        if (State == State.SetDefaults)
        {
            // LEAN -- only UI-visible properties here
            Description = "OR Reversal Strategy";
            Name        = "ORReversal";
            Calculate   = Calculate.OnBarClose;  // or OnEachTick for trailing
            EntriesPerDirection     = 1;
            EntryHandling           = EntryHandling.UniqueEntries;
            IsExitOnSessionCloseStrategy = true;
            ExitOnSessionCloseSeconds    = 60;  // flatten 60s before session end
            IsFillLimitOnTouch      = false;
            TraceOrders             = true;  // essential for debugging
            BarsRequiredToTrade     = 5;
            IsInstantiatedOnEachOptimizationIteration = true;

            // Connection loss: Recalculate replays history on reconnect
            ConnectionLossHandling  = ConnectionLossHandling.Recalculate;
            DisconnectDelaySeconds  = 10;

            // Custom properties
            StopTicks    = 40;
            TargetTicks  = 80;
        }
        else if (State == State.Configure)
        {
            // Add secondary data series if needed
            // AddDataSeries(BarsPeriodType.Minute, 5);

            // Add custom time filter (optional)
            // AddDataSeries("NQ 03-26", BarsPeriodType.Minute, 1);
        }
        else if (State == State.DataLoaded)
        {
            // Instantiate indicators HERE (data is available)
            // myIndicator = MyCustomIndicator(param1, param2);
            // AddChartIndicator(myIndicator);

            // Initialize any tracking variables
            breakEvenTriggered = false;
        }
        else if (State == State.Historical)
        {
            // Strategy is processing historical bars
            // Good place for one-time historical setup
        }
        else if (State == State.Transition)
        {
            // Moving from historical to realtime
            // Historical bars are done, realtime about to begin
        }
        else if (State == State.Realtime)
        {
            // Now processing live data
            // Check for existing positions from prior session
            if (Position.MarketPosition == MarketPosition.Long)
                Print("Strategy enabled with existing LONG position");
        }
        else if (State == State.Terminated)
        {
            // Cleanup resources (file handles, timers, etc.)
        }
    }
}
```

### Key Gotchas

- **State.SetDefaults runs MANY times** -- NinjaTrader calls it whenever it scans
  strategies for the UI list. Keep it fast with no resource allocation.
- **State.DataLoaded** is where you instantiate indicators and series objects.
  `Bars`, `Close`, `High` etc. are NOT available before this state.
- **State.Transition** is brief. Do NOT submit orders here.
- OnBarUpdate() only fires starting from State.Historical onward.

---

## 2. Order Management (Managed Approach -- No ATM)

The managed approach uses `EnterLong()`, `EnterShort()`, `SetStopLoss()`,
`SetProfitTarget()`, `ExitLong()`, `ExitShort()`. NinjaTrader handles OCO
grouping and order lifecycle automatically.

### Entry Methods

```csharp
// Market entry
EnterLong(1, "ORRevLong");     // 1 contract, signal name "ORRevLong"
EnterShort(1, "ORRevShort");   // 1 contract, signal name "ORRevShort"

// Limit entry
EnterLongLimit(1, limitPrice, "ORRevLong");
EnterShortLimit(1, limitPrice, "ORRevShort");

// Stop entry (buy stop above market, sell stop below market)
EnterLongStopMarket(1, stopPrice, "ORRevLong");
EnterShortStopMarket(1, stopPrice, "ORRevShort");

// Stop-limit entry
EnterLongStopLimit(1, limitPrice, stopPrice, "ORRevLong");
```

### Protective Orders (Set Methods)

```csharp
// MUST be called BEFORE the entry method, or in OnStateChange Configure/DataLoaded
// These persist until changed or position is flat

// Fixed stop loss: 40 ticks from entry
SetStopLoss("ORRevLong", CalculationMode.Ticks, 40, false);

// Fixed profit target: 80 ticks from entry
SetProfitTarget("ORRevLong", CalculationMode.Ticks, 80);

// Stop at specific price
SetStopLoss("ORRevLong", CalculationMode.Price, 25300.0, false);

// Profit target at specific price
SetProfitTarget("ORRevLong", CalculationMode.Price, 25500.0);

// Percentage-based stop (rarely used for futures)
SetStopLoss("ORRevLong", CalculationMode.Percent, 1.0, false);
```

### Exit Methods

```csharp
// Exit specific named entry
ExitLong(1, "ExitORRev", "ORRevLong");  // qty, exit name, entry signal name
ExitShort(1, "ExitORRev", "ORRevShort");

// Exit at limit price
ExitLongLimit(1, limitPrice, "ExitORRev", "ORRevLong");
ExitShortLimit(1, limitPrice, "ExitORRev", "ORRevShort");

// Exit at stop price
ExitLongStopMarket(1, stopPrice, "ExitORRev", "ORRevLong");
```

### Complete Managed Entry Example

```csharp
protected override void OnBarUpdate()
{
    if (CurrentBar < BarsRequiredToTrade) return;

    // Only trade during RTH
    if (ToTime(Time[0]) < 93000 || ToTime(Time[0]) > 155900) return;

    // Flat -- look for entries
    if (Position.MarketPosition == MarketPosition.Flat)
    {
        breakEvenTriggered = false;  // reset for next trade

        if (longSignal)
        {
            SetStopLoss("GoLong", CalculationMode.Price, sweepLow - stopBuffer);
            SetProfitTarget("GoLong", CalculationMode.Price, targetPrice);
            EnterLong(1, "GoLong");
        }
        else if (shortSignal)
        {
            SetStopLoss("GoShort", CalculationMode.Price, sweepHigh + stopBuffer);
            SetProfitTarget("GoShort", CalculationMode.Price, targetPrice);
            EnterShort(1, "GoShort");
        }
    }
}
```

### Important Managed Approach Rules

1. **One active order per direction at a time.** Cannot submit a second long entry
   while first is pending.
2. **SetStopLoss/SetProfitTarget apply to the NEXT entry** if called while flat,
   or to the CURRENT position if called while in a trade.
3. **Calling EnterLong while short** auto-reverses (closes short + opens long).
   Guard against this with `Position.MarketPosition` checks.
4. **Signal names matter.** Use unique names per entry type. SetStopLoss/SetProfitTarget
   use the signal name to associate with the correct entry.
5. **SetStopLoss persists.** After a breakeven move, you MUST reset the stop before
   the next entry or it will use the last value.

---

## 3. Managed vs Unmanaged Approach

### Managed Approach

**Pros:**
- Simpler code, fewer edge cases
- NinjaTrader handles OCO grouping automatically
- SetStopLoss/SetProfitTarget auto-link to entries
- Works with Strategy Analyzer backtesting
- Position tracking is automatic
- Best for single-entry, fixed stop/target strategies

**Cons:**
- Cannot have two opposing orders simultaneously
- Limited to one pending entry per direction (per unique signal)
- Cannot scale in/out with independent stop/targets per scale
- SetStopLoss cannot be called from OnOrderUpdate/OnExecutionUpdate
- Less control over exact order timing

### Unmanaged Approach

**Pros:**
- Full control over every order
- Multiple simultaneous orders in any direction
- Independent stop/targets per partial position
- Can submit orders from OnExecutionUpdate (fastest protective order submission)
- OCO grouping is manual (more flexible)
- Can handle complex multi-leg strategies

**Cons:**
- Significantly more code
- YOU manage position tracking, OCO cancellation, fill confirmation
- More error-prone if not careful
- Must handle partial fills explicitly
- Does NOT work with Strategy Analyzer optimization (backtest only via Playback)

### Unmanaged Approach Setup

```csharp
protected override void OnStateChange()
{
    if (State == State.SetDefaults)
    {
        IsUnmanaged = true;  // MUST be set here
    }
}
```

### Unmanaged Order Submission Pattern

```csharp
private Order entryOrder   = null;
private Order stopOrder    = null;
private Order profitOrder  = null;

protected override void OnBarUpdate()
{
    if (Position.MarketPosition == MarketPosition.Flat && entryOrder == null)
    {
        if (longSignal)
        {
            // CRITICAL: Do NOT assign the return value directly.
            // Capture the order in OnOrderUpdate instead.
            SubmitOrderUnmanaged(0, OrderAction.Buy, OrderType.Market,
                1, 0, 0, "", "EntryLong");
        }
    }
}

protected override void OnOrderUpdate(Order order, double limitPrice,
    double stopPrice, int quantity, int filled, double averageFillPrice,
    OrderState orderState, DateTime time, ErrorCode error, string comment)
{
    // Capture order references safely (NOT from SubmitOrderUnmanaged return)
    if (order.Name == "EntryLong")
        entryOrder = order;
    if (order.Name == "StopLong")
        stopOrder = order;
    if (order.Name == "TargetLong")
        profitOrder = order;

    // Clean up on terminal states
    if (orderState == OrderState.Cancelled || orderState == OrderState.Rejected)
    {
        if (order.Name == "EntryLong")  entryOrder = null;
        if (order.Name == "StopLong")   stopOrder = null;
        if (order.Name == "TargetLong") profitOrder = null;
    }
}

protected override void OnExecutionUpdate(Execution execution, string executionId,
    double price, int quantity, MarketPosition marketPosition,
    string orderId, DateTime time)
{
    // Entry filled -> submit protective orders
    if (entryOrder != null && entryOrder == execution.Order
        && execution.Order.OrderState == OrderState.Filled)
    {
        string ocoId = Guid.NewGuid().ToString();

        // Stop loss
        SubmitOrderUnmanaged(0, OrderAction.Sell, OrderType.StopMarket,
            execution.Quantity, 0, stopPrice, ocoId, "StopLong");

        // Profit target
        SubmitOrderUnmanaged(0, OrderAction.Sell, OrderType.Limit,
            execution.Quantity, targetPrice, 0, ocoId, "TargetLong");
    }

    // Stop or target filled -> cleanup
    if (stopOrder != null && stopOrder == execution.Order
        && execution.Order.OrderState == OrderState.Filled)
    {
        stopOrder   = null;
        profitOrder = null;  // OCO should cancel the other
        entryOrder  = null;
    }
    if (profitOrder != null && profitOrder == execution.Order
        && execution.Order.OrderState == OrderState.Filled)
    {
        profitOrder = null;
        stopOrder   = null;
        entryOrder  = null;
    }
}
```

### Recommendation for OR Reversal + Edge Fade

**Use the MANAGED approach.** Here is why:

1. Our strategies take ONE entry per signal with a fixed stop and target. No scaling.
2. Managed approach works perfectly with Strategy Analyzer for validation.
3. SetStopLoss with CalculationMode.Price allows dynamic stop adjustment (trailing/BE).
4. The ATM "hanging" issue does not apply -- we are NOT using ATM at all.
5. The managed approach handles reconnection/recalculation automatically.
6. Significantly less code = fewer bugs in production.

The unmanaged approach is only needed if we later want multiple simultaneous
entries with independent exits, which our current strategy design does not require.

---

## 4. Trailing Stop Implementation

### Method 1: Built-in SetTrailStop (Simple)

```csharp
// In OnStateChange or before entry:
// Trail by 20 ticks from the highest favorable price
SetTrailStop("GoLong", CalculationMode.Ticks, 20, false);
```

This auto-trails: for longs, the stop follows price up by staying 20 ticks below
the highest price since entry. For shorts, it follows price down.

**Limitation:** Cannot coexist with SetStopLoss on the same signal name.
SetStopLoss always takes precedence.

### Method 2: Manual Trailing with SetStopLoss (Full Control)

This is the recommended approach for custom trail logic.

```csharp
private double trailStopPrice = 0;
private double highestSinceEntry = 0;
private double lowestSinceEntry = double.MaxValue;
private int trailTicks = 20;

protected override void OnStateChange()
{
    if (State == State.SetDefaults)
    {
        // Use OnEachTick for realtime trailing precision
        Calculate = Calculate.OnEachTick;
    }
}

protected override void OnBarUpdate()
{
    // --- LONG position trailing ---
    if (Position.MarketPosition == MarketPosition.Long)
    {
        // Track highest price since entry
        if (High[0] > highestSinceEntry)
            highestSinceEntry = High[0];

        double newStop = highestSinceEntry - trailTicks * TickSize;

        // Only move stop UP (never down for longs)
        if (newStop > trailStopPrice)
        {
            trailStopPrice = newStop;
            SetStopLoss(CalculationMode.Price, trailStopPrice);
        }
    }

    // --- SHORT position trailing ---
    else if (Position.MarketPosition == MarketPosition.Short)
    {
        if (Low[0] < lowestSinceEntry)
            lowestSinceEntry = Low[0];

        double newStop = lowestSinceEntry + trailTicks * TickSize;

        // Only move stop DOWN (never up for shorts)
        if (newStop < trailStopPrice || trailStopPrice == 0)
        {
            trailStopPrice = newStop;
            SetStopLoss(CalculationMode.Price, trailStopPrice);
        }
    }

    // --- Flat: reset tracking ---
    else
    {
        highestSinceEntry = 0;
        lowestSinceEntry  = double.MaxValue;
        trailStopPrice    = 0;
    }
}
```

### Method 3: Step Trailing (Move Stop in Defined Steps)

```csharp
// Trail in 20-tick steps: after each 20 ticks of profit, move stop up 20 ticks
private int stepSize = 20;
private int stepsCompleted = 0;

protected override void OnBarUpdate()
{
    if (Position.MarketPosition == MarketPosition.Long)
    {
        double entryPrice = Position.AveragePrice;
        double currentProfit = (Close[0] - entryPrice) / TickSize;
        int expectedSteps = (int)(currentProfit / stepSize);

        if (expectedSteps > stepsCompleted)
        {
            stepsCompleted = expectedSteps;
            double newStop = entryPrice + (stepsCompleted - 1) * stepSize * TickSize;
            SetStopLoss(CalculationMode.Price, Math.Max(newStop, entryPrice));
        }
    }
    else
    {
        stepsCompleted = 0;
    }
}
```

### Important Notes on Trailing

- **Calculate.OnEachTick** gives the most responsive trailing in realtime but
  is slower in backtesting. Consider `Calculate.OnBarClose` for backtesting
  and switching in `State.Realtime`.
- **Always move stops in ONE direction only.** Long stops go up, short stops go down.
- **Reset all tracking variables when flat** to prevent stale values on next entry.

---

## 5. Breakeven Stop Implementation

### Basic Breakeven After X Ticks Profit

```csharp
private bool breakEvenTriggered = false;
private int breakEvenTriggerTicks = 30;  // trigger after 30 ticks profit
private int breakEvenOffsetTicks  = 2;   // place stop 2 ticks beyond entry (small profit)

protected override void OnBarUpdate()
{
    // --- Entry logic ---
    if (Position.MarketPosition == MarketPosition.Flat)
    {
        breakEvenTriggered = false;  // CRITICAL: reset before each new trade

        if (longSignal)
        {
            SetStopLoss("GoLong", CalculationMode.Ticks, 40, false);
            SetProfitTarget("GoLong", CalculationMode.Ticks, 80);
            EnterLong(1, "GoLong");
        }
    }

    // --- Breakeven management ---
    if (Position.MarketPosition == MarketPosition.Long && !breakEvenTriggered)
    {
        double ticksInProfit = (Close[0] - Position.AveragePrice) / TickSize;

        if (ticksInProfit >= breakEvenTriggerTicks)
        {
            // Move stop to entry + small offset
            double bePrice = Position.AveragePrice + breakEvenOffsetTicks * TickSize;
            SetStopLoss(CalculationMode.Price, bePrice);
            breakEvenTriggered = true;
            Print(Time[0] + " BREAKEVEN triggered at " + bePrice);
        }
    }

    if (Position.MarketPosition == MarketPosition.Short && !breakEvenTriggered)
    {
        double ticksInProfit = (Position.AveragePrice - Close[0]) / TickSize;

        if (ticksInProfit >= breakEvenTriggerTicks)
        {
            double bePrice = Position.AveragePrice - breakEvenOffsetTicks * TickSize;
            SetStopLoss(CalculationMode.Price, bePrice);
            breakEvenTriggered = true;
            Print(Time[0] + " BREAKEVEN triggered at " + bePrice);
        }
    }
}
```

### Combined Breakeven + Trailing

```csharp
private bool breakEvenTriggered = false;
private bool trailingActive = false;
private double trailStopPrice = 0;
private double highestSinceEntry = 0;

private int beTriggerTicks   = 30;   // breakeven after 30 ticks
private int trailTriggerTicks = 50;  // start trailing after 50 ticks
private int trailDistTicks    = 25;  // trail distance

protected override void OnBarUpdate()
{
    if (Position.MarketPosition == MarketPosition.Long)
    {
        double entryPrice = Position.AveragePrice;
        double ticksProfit = (Close[0] - entryPrice) / TickSize;

        if (High[0] > highestSinceEntry)
            highestSinceEntry = High[0];

        // Phase 3: Trailing (overrides breakeven)
        if (ticksProfit >= trailTriggerTicks)
        {
            trailingActive = true;
            double newStop = highestSinceEntry - trailDistTicks * TickSize;
            if (newStop > trailStopPrice)
            {
                trailStopPrice = newStop;
                SetStopLoss(CalculationMode.Price, trailStopPrice);
            }
        }
        // Phase 2: Breakeven
        else if (ticksProfit >= beTriggerTicks && !breakEvenTriggered)
        {
            SetStopLoss(CalculationMode.Price, entryPrice + 2 * TickSize);
            breakEvenTriggered = true;
            trailStopPrice = entryPrice + 2 * TickSize;
        }
        // Phase 1: Initial stop (set at entry, no action needed here)
    }
    else
    {
        // Reset all flags when flat
        breakEvenTriggered = false;
        trailingActive = false;
        trailStopPrice = 0;
        highestSinceEntry = 0;
    }
}
```

### Critical Reset Rule

**You MUST reset `SetStopLoss` before the next entry.** If you moved the stop to
breakeven ($25,402) and then go flat, the next `EnterLong` will still use $25,402
as the stop unless you call `SetStopLoss` with new values before the next entry.
Always set stops BEFORE calling EnterLong/EnterShort.

---

## 6. Position Management

### Checking Position State

```csharp
// Current market position
MarketPosition pos = Position.MarketPosition;
// Values: MarketPosition.Flat, MarketPosition.Long, MarketPosition.Short

// Average entry price
double avgPrice = Position.AveragePrice;

// Number of contracts
int qty = Position.Quantity;

// Unrealized P&L
double unrealizedPnL = Position.GetUnrealizedProfitLoss(PerformanceUnit.Currency, Close[0]);
double unrealizedTicks = Position.GetUnrealizedProfitLoss(PerformanceUnit.Ticks, Close[0]);
double unrealizedPct = Position.GetUnrealizedProfitLoss(PerformanceUnit.Percent, Close[0]);
```

### Position-Aware Logic Pattern

```csharp
protected override void OnBarUpdate()
{
    switch (Position.MarketPosition)
    {
        case MarketPosition.Flat:
            HandleFlatState();
            break;
        case MarketPosition.Long:
            HandleLongPosition();
            break;
        case MarketPosition.Short:
            HandleShortPosition();
            break;
    }
}

private void HandleFlatState()
{
    // Reset all position tracking
    ResetTradeState();

    // Check for new entry signals
    if (longConditionMet)
    {
        SetStopLoss("GoLong", CalculationMode.Price, stopPrice);
        SetProfitTarget("GoLong", CalculationMode.Price, targetPrice);
        EnterLong(1, "GoLong");
    }
}

private void HandleLongPosition()
{
    // Monitor breakeven/trail
    ManageBreakeven();
    ManageTrail();

    // Time-based exit check
    if (ToTime(Time[0]) >= 155900)
    {
        ExitLong("TimeExit", "GoLong");
    }
}

private void ResetTradeState()
{
    breakEvenTriggered = false;
    trailStopPrice = 0;
    highestSinceEntry = 0;
    // ... reset all per-trade variables
}
```

### OnPositionUpdate Event

```csharp
protected override void OnPositionUpdate(Position position, double averagePrice,
    int quantity, MarketPosition marketPosition)
{
    if (marketPosition == MarketPosition.Flat)
    {
        Print(Time[0] + " Position closed. P&L: "
            + SystemPerformance.AllTrades[SystemPerformance.AllTrades.Count - 1].ProfitCurrency);
    }
    else if (marketPosition == MarketPosition.Long)
    {
        Print(Time[0] + " LONG entry at " + averagePrice + " x" + quantity);
        highestSinceEntry = averagePrice;  // initialize tracking
    }
    else if (marketPosition == MarketPosition.Short)
    {
        Print(Time[0] + " SHORT entry at " + averagePrice + " x" + quantity);
        lowestSinceEntry = averagePrice;
    }
}
```

---

## 7. Multiple Entries Per Session

### Setup for Multiple Unique Entries

```csharp
protected override void OnStateChange()
{
    if (State == State.SetDefaults)
    {
        // Allow 2 entries in the same direction (e.g., OR Rev + Edge Fade both long)
        EntriesPerDirection = 2;
        EntryHandling = EntryHandling.UniqueEntries;
    }
}

protected override void OnBarUpdate()
{
    // Each entry uses a UNIQUE signal name
    if (orReversalSignal && Position.MarketPosition == MarketPosition.Flat)
    {
        SetStopLoss("ORRev", CalculationMode.Price, orStopPrice);
        SetProfitTarget("ORRev", CalculationMode.Price, orTargetPrice);
        EnterLong(1, "ORRev");
    }

    if (edgeFadeSignal && Position.MarketPosition != MarketPosition.Short)
    {
        SetStopLoss("EdgeFade", CalculationMode.Price, efStopPrice);
        SetProfitTarget("EdgeFade", CalculationMode.Price, efTargetPrice);
        EnterLong(1, "EdgeFade");
    }
}
```

### Separate Strategies Approach (Recommended)

For OR Reversal and Edge Fade, **use two separate strategy instances** rather than
one combined strategy. Reasons:

1. Each has different time windows (9:30-10:00 vs 10:30-13:30)
2. Each has different stop/target logic
3. Independent enable/disable per strategy
4. Cleaner Position tracking (each strategy tracks its own position)
5. Replikanto copies each strategy independently to multiple accounts
6. NinjaTrader Position object is per-strategy-instance

### Per-Session Trade Limit

```csharp
private int todayTradeCount = 0;
private DateTime lastTradeDate = DateTime.MinValue;
private int maxTradesPerDay = 2;

protected override void OnBarUpdate()
{
    // Reset counter on new session
    if (Time[0].Date != lastTradeDate)
    {
        todayTradeCount = 0;
        lastTradeDate = Time[0].Date;
    }

    // Guard against over-trading
    if (todayTradeCount >= maxTradesPerDay)
        return;

    if (Position.MarketPosition == MarketPosition.Flat && entrySignal)
    {
        EnterLong(1, "MyEntry");
        todayTradeCount++;
    }
}
```

### Tracking Multiple Named Entries

```csharp
// When using EntryHandling.UniqueEntries, exits must reference the entry name
protected override void OnBarUpdate()
{
    // Exit specific entry by signal name
    if (Position.MarketPosition == MarketPosition.Long)
    {
        if (exitORRevCondition)
            ExitLong("ExitORRev", "ORRev");  // only exits the ORRev portion

        if (exitEdgeFadeCondition)
            ExitLong("ExitEdge", "EdgeFade");  // only exits the EdgeFade portion
    }
}
```

---

## 8. Time-Based Exits

### Method 1: Programmatic Time Check (Recommended)

```csharp
protected override void OnBarUpdate()
{
    int currentTime = ToTime(Time[0]);  // returns HHMMSS as integer

    // Prevent new entries after 13:30 (Edge Fade cutoff)
    if (currentTime > 133000)
        return;

    // Flatten at 15:59 regardless of P&L
    if (currentTime >= 155900 && Position.MarketPosition != MarketPosition.Flat)
    {
        if (Position.MarketPosition == MarketPosition.Long)
            ExitLong("EODFlatten", "GoLong");
        else if (Position.MarketPosition == MarketPosition.Short)
            ExitShort("EODFlatten", "GoShort");
    }
}
```

### Method 2: ExitOnSessionClose Property

```csharp
protected override void OnStateChange()
{
    if (State == State.SetDefaults)
    {
        IsExitOnSessionCloseStrategy = true;
        ExitOnSessionCloseSeconds    = 60;  // flatten 60 seconds before session end
    }
}
```

This is a built-in property. NinjaTrader will auto-submit market orders to flatten
the position X seconds before the session template's close time.

**Caveat:** ExitOnSessionClose uses the Trading Hours template close time.
For NQ RTH that is typically 16:00 ET. If you want to flatten at 15:59, either:
- Set `ExitOnSessionCloseSeconds = 60` (60 seconds before 16:00), or
- Use the programmatic approach for exact control.

### Method 3: Session-Aware with SessionIterator

```csharp
private SessionIterator sessionIterator;

protected override void OnStateChange()
{
    if (State == State.DataLoaded)
    {
        sessionIterator = new SessionIterator(Bars);
    }
}

protected override void OnBarUpdate()
{
    // Get exact session end time
    sessionIterator.GetNextSession(Time[0], false);
    DateTime sessionEnd = sessionIterator.ActualSessionEnd;

    // Flatten 1 minute before session end
    if (Time[0] >= sessionEnd.AddMinutes(-1)
        && Position.MarketPosition != MarketPosition.Flat)
    {
        if (Position.MarketPosition == MarketPosition.Long)
            ExitLong("SessionEnd", "GoLong");
        else
            ExitShort("SessionEnd", "GoShort");
    }
}
```

### Time Window Guards (Entry Filters)

```csharp
// OR Reversal: only scan during 9:30-10:00
private bool IsORWindow()
{
    int t = ToTime(Time[0]);
    return t >= 93000 && t <= 100000;
}

// Edge Fade: only enter during 10:30-13:30
private bool IsEdgeFadeWindow()
{
    int t = ToTime(Time[0]);
    return t >= 103000 && t <= 133000;
}
```

---

## 9. Connection Handling

### ConnectionLossHandling Options

```csharp
protected override void OnStateChange()
{
    if (State == State.SetDefaults)
    {
        // Option 1: Recalculate (DEFAULT, RECOMMENDED)
        // On reconnect, re-processes historical bars to rebuild state
        ConnectionLossHandling = ConnectionLossHandling.Recalculate;

        // Option 2: KeepRunning
        // Strategy continues from where it left off, no recalculation
        // WARNING: Strategy state may be inconsistent if bars were missed
        // ConnectionLossHandling = ConnectionLossHandling.KeepRunning;

        // Option 3: StopStrategy
        // Strategy is disabled on disconnect
        // ConnectionLossHandling = ConnectionLossHandling.StopStrategy;

        // How long to wait before stopping (seconds)
        DisconnectDelaySeconds = 10;
    }
}
```

### How Recalculate Works

1. Connection drops. Timer starts (DisconnectDelaySeconds).
2. If reconnected within DisconnectDelaySeconds -> strategy continues as if nothing happened.
3. If not reconnected in time -> strategy is stopped.
4. On reconnect -> strategy re-enables, replays historical bars, rebuilds position state.
5. If historically calculated orders match live working orders -> orders persist.
6. If mismatch -> live orders are cancelled and replaced.

### OnConnectionStatusUpdate Event

```csharp
protected override void OnConnectionStatusUpdate(ConnectionStatusEventArgs connectionStatusUpdate)
{
    if (connectionStatusUpdate.Status == ConnectionStatus.Connected)
    {
        Print(Time[0] + " CONNECTION RESTORED: " + connectionStatusUpdate.Connection.Options.Name);
    }
    else if (connectionStatusUpdate.Status == ConnectionStatus.ConnectionLost)
    {
        Print(Time[0] + " CONNECTION LOST: " + connectionStatusUpdate.Connection.Options.Name);
        // Do NOT submit orders here -- just log/alert
    }
}
```

### Best Practices for Reliability

1. **Use `ConnectionLossHandling.Recalculate`** for strategies with position state.
   It ensures the strategy rebuilds correctly after reconnect.
2. **Set `DisconnectDelaySeconds = 10-30`** -- long enough to survive brief blips,
   short enough to not miss critical price action.
3. **TraceOrders = true** for production debugging.
4. **Never submit orders in OnConnectionStatusUpdate** -- use it only for logging.
5. **Avoid `Calculate.OnEachTick` unless necessary** -- it creates more points of
   failure during connection instability.
6. **Monitor strategy tab** -- NinjaTrader shows strategy status
   (Enabled/Disabled/Error) in the Strategies tab.

### Reconnect Behavior with Tradovate

Tradovate (your data feed) has its own reconnect logic. NinjaTrader sees the
connection status through the Tradovate adapter. Key points:

- Tradovate reconnects automatically in most cases
- NinjaTrader will show "Connection Lost" then "Connected"
- With `Recalculate`, the strategy re-processes bars on reconnect
- Working orders on Tradovate's server persist through brief disconnects
- On extended outage, orders may expire depending on TIF (Time In Force)

---

## 10. Strategy to Indicator Communication

### Hosting an Indicator Inside a Strategy

```csharp
public class ORReversalStrategy : Strategy
{
    // Declare indicator at class level
    private ORReversalIndicator orIndicator;

    protected override void OnStateChange()
    {
        if (State == State.DataLoaded)
        {
            // Instantiate indicator with parameters
            orIndicator = ORReversalIndicator(15, 30, 100);  // OR bars, EOR bars, min IB

            // Optionally display on chart
            AddChartIndicator(orIndicator);
        }
    }

    protected override void OnBarUpdate()
    {
        // Access indicator's public properties and Series
        if (orIndicator.SignalDirection[0] == 1)  // Series<double> plot
        {
            double stopPrice = orIndicator.StopPrice;     // public property
            double targetPrice = orIndicator.TargetPrice;  // public property

            SetStopLoss("GoLong", CalculationMode.Price, stopPrice);
            SetProfitTarget("GoLong", CalculationMode.Price, targetPrice);
            EnterLong(1, "GoLong");
        }
    }
}
```

### Exposing Values from Indicator to Strategy

In the indicator code, expose values as public properties and/or Series:

```csharp
public class ORReversalIndicator : Indicator
{
    // --- Series (bar-indexed, accessible as myIndicator.SignalDirection[0]) ---
    [Browsable(false)]
    [XmlIgnore]
    public Series<double> SignalDirection { get; set; }

    [Browsable(false)]
    [XmlIgnore]
    public Series<double> IBRange { get; set; }

    // --- Simple properties (snapshot values, not bar-indexed) ---
    [Browsable(false)]
    [XmlIgnore]
    public double StopPrice { get; set; }

    [Browsable(false)]
    [XmlIgnore]
    public double TargetPrice { get; set; }

    [Browsable(false)]
    [XmlIgnore]
    public double ORHigh { get; set; }

    [Browsable(false)]
    [XmlIgnore]
    public double ORLow { get; set; }

    [Browsable(false)]
    [XmlIgnore]
    public string DayType { get; set; }

    protected override void OnStateChange()
    {
        if (State == State.SetDefaults)
        {
            Name = "ORReversalIndicator";
            IsOverlay = true;
            AddPlot(Brushes.Transparent, "SignalDirection");
            AddPlot(Brushes.Transparent, "IBRange");
        }
        else if (State == State.DataLoaded)
        {
            SignalDirection = new Series<double>(this);
            IBRange         = new Series<double>(this);
        }
    }

    protected override void OnBarUpdate()
    {
        // Compute signals, set public properties
        SignalDirection[0] = 0;  // 0=none, 1=long, -1=short

        if (sweepDetected && reversalConfirmed)
        {
            SignalDirection[0] = direction;
            StopPrice   = computedStop;
            TargetPrice = computedTarget;
        }

        ORHigh   = orHigh;
        ORLow    = orLow;
        DayType  = classifiedDayType;
    }
}
```

### Important: Update() Call for Non-Series Properties

When a strategy reads a non-Series public property (like `StopPrice`), the indicator
may not have processed the current bar yet. Force it:

```csharp
// In the indicator's property getter:
public double StopPrice
{
    get
    {
        Update();  // Forces indicator to process current bar
        return stopPrice;
    }
    set { stopPrice = value; }
}
```

Or call `Update()` explicitly from the strategy:

```csharp
protected override void OnBarUpdate()
{
    orIndicator.Update();  // Ensure indicator is current

    if (orIndicator.SignalDirection[0] == 1)
    {
        // Safe to read StopPrice now
    }
}
```

### Architecture for OR Reversal + Edge Fade

Recommended structure:

```
Indicators/
  ORReversalIndicator.cs     -- OR/EOR zones, sweep detection, reversal signals
  EdgeFadeIndicator.cs       -- IB range, edge detection, MR signals
  IBClassifier.cs            -- Day type classification (shared by both)

Strategies/
  ORReversalStrategy.cs      -- Hosts ORReversalIndicator, manages orders
  EdgeFadeStrategy.cs        -- Hosts EdgeFadeIndicator, manages orders
```

Each strategy is a thin "order manager" that:
1. Hosts its corresponding indicator
2. Reads signals from the indicator
3. Manages entry/exit orders with breakeven/trailing
4. Handles time windows and session management

The indicator does all the heavy computation (OR/EOR ranges, sweep detection,
delta analysis, day type classification, IB expansion checks).

### Volumetric Data Access in Indicators

From the existing `CsvMarketExporterTick.cs` in your NT8 install:

```csharp
using NinjaTrader.NinjaScript.BarsTypes;

// In OnBarUpdate:
NinjaTrader.NinjaScript.BarsTypes.VolumetricBarsType volumetricBar =
    Bars.BarsSeries.BarsType as NinjaTrader.NinjaScript.BarsTypes.VolumetricBarsType;

if (volumetricBar != null)
{
    long buyVol  = volumetricBar.Volumes[CurrentBar].TotalBuyingVolume;
    long sellVol = volumetricBar.Volumes[CurrentBar].TotalSellingVolume;
    long delta   = volumetricBar.Volumes[CurrentBar].BarDelta;
    long volume  = volumetricBar.Volumes[CurrentBar].TotalVolume;

    // Cumulative delta since session open
    // (you'd accumulate this yourself in a running sum)
    cumulativeDelta += delta;
}
```

**Requirement:** The chart must use "Volumetric" bar type for this cast to succeed.
Standard OHLC bars will return null for the volumetric cast.

---

## Quick Reference: Complete Minimal Strategy Template

```csharp
#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Windows.Media;
using System.Xml.Serialization;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Indicators;
using NinjaTrader.NinjaScript.DrawingTools;
using NinjaTrader.Core.FloatingPoint;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class ORReversalStrategy : Strategy
    {
        // --- Indicator reference ---
        // private ORReversalIndicator orIndicator;

        // --- Order tracking ---
        private bool breakEvenTriggered = false;
        private double highestSinceEntry = 0;
        private int todayTradeCount = 0;
        private DateTime lastTradeDate = DateTime.MinValue;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "OR Reversal - Judas Swing";
                Name        = "ORReversalStrategy";
                Calculate   = Calculate.OnBarClose;
                EntriesPerDirection          = 1;
                EntryHandling                = EntryHandling.UniqueEntries;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds    = 60;
                IsFillLimitOnTouch           = false;
                TraceOrders                  = true;
                BarsRequiredToTrade          = 30;
                IsInstantiatedOnEachOptimizationIteration = true;
                ConnectionLossHandling       = ConnectionLossHandling.Recalculate;
                DisconnectDelaySeconds       = 10;
                MaxTradesPerDay              = 1;
                StopTicks                    = 40;
                TargetMultiplier             = 2.0;
            }
            else if (State == State.DataLoaded)
            {
                // orIndicator = ORReversalIndicator(...);
                // AddChartIndicator(orIndicator);
            }
        }

        protected override void OnBarUpdate()
        {
            if (CurrentBar < BarsRequiredToTrade) return;
            if (State == State.Historical) return;  // optional: skip historical

            int t = ToTime(Time[0]);

            // Reset daily counter
            if (Time[0].Date != lastTradeDate)
            {
                todayTradeCount = 0;
                lastTradeDate = Time[0].Date;
            }

            // --- FLAT: Look for entries ---
            if (Position.MarketPosition == MarketPosition.Flat)
            {
                breakEvenTriggered = false;
                highestSinceEntry = 0;

                // Time window: 9:30 - 10:00 only
                if (t < 93000 || t > 100000) return;
                if (todayTradeCount >= MaxTradesPerDay) return;

                // TODO: Read signal from orIndicator
                bool longSignal = false;   // orIndicator.SignalDirection[0] == 1;
                bool shortSignal = false;  // orIndicator.SignalDirection[0] == -1;
                double stopPrice = 0;      // orIndicator.StopPrice;
                double targetPrice = 0;    // orIndicator.TargetPrice;

                if (longSignal)
                {
                    SetStopLoss("ORRevLong", CalculationMode.Price, stopPrice);
                    SetProfitTarget("ORRevLong", CalculationMode.Price, targetPrice);
                    EnterLong(1, "ORRevLong");
                    todayTradeCount++;
                }
                else if (shortSignal)
                {
                    SetStopLoss("ORRevShort", CalculationMode.Price, stopPrice);
                    SetProfitTarget("ORRevShort", CalculationMode.Price, targetPrice);
                    EnterShort(1, "ORRevShort");
                    todayTradeCount++;
                }
            }

            // --- LONG: Manage position ---
            else if (Position.MarketPosition == MarketPosition.Long)
            {
                // Breakeven
                if (!breakEvenTriggered)
                {
                    double ticksProfit = (Close[0] - Position.AveragePrice) / TickSize;
                    if (ticksProfit >= 30)
                    {
                        SetStopLoss(CalculationMode.Price,
                            Position.AveragePrice + 2 * TickSize);
                        breakEvenTriggered = true;
                    }
                }

                // EOD flatten
                if (t >= 155900)
                    ExitLong("EODExit", "ORRevLong");
            }

            // --- SHORT: Manage position ---
            else if (Position.MarketPosition == MarketPosition.Short)
            {
                if (!breakEvenTriggered)
                {
                    double ticksProfit = (Position.AveragePrice - Close[0]) / TickSize;
                    if (ticksProfit >= 30)
                    {
                        SetStopLoss(CalculationMode.Price,
                            Position.AveragePrice - 2 * TickSize);
                        breakEvenTriggered = true;
                    }
                }

                if (t >= 155900)
                    ExitShort("EODExit", "ORRevShort");
            }
        }

        protected override void OnExecutionUpdate(Execution execution,
            string executionId, double price, int quantity,
            MarketPosition marketPosition, string orderId, DateTime time)
        {
            // Log fills
            Print(time + " FILL: " + execution.Order.Name
                + " @ " + price + " x" + quantity
                + " pos=" + marketPosition);
        }

        protected override void OnPositionUpdate(Position position,
            double averagePrice, int quantity, MarketPosition marketPosition)
        {
            if (marketPosition == MarketPosition.Flat)
            {
                Print("Position FLAT. Resetting state.");
                breakEvenTriggered = false;
                highestSinceEntry = 0;
            }
        }

        protected override void OnConnectionStatusUpdate(
            ConnectionStatusEventArgs connectionStatusUpdate)
        {
            Print("Connection: " + connectionStatusUpdate.Status
                + " " + connectionStatusUpdate.Connection.Options.Name);
        }

        #region Properties
        [Range(1, 10), NinjaScriptProperty]
        [Display(Name = "Max Trades Per Day", GroupName = "Parameters", Order = 0)]
        public int MaxTradesPerDay { get; set; }

        [Range(1, 200), NinjaScriptProperty]
        [Display(Name = "Stop (Ticks)", GroupName = "Parameters", Order = 1)]
        public int StopTicks { get; set; }

        [Range(0.5, 5.0), NinjaScriptProperty]
        [Display(Name = "Target Multiplier (R)", GroupName = "Parameters", Order = 2)]
        public double TargetMultiplier { get; set; }
        #endregion
    }
}
```

---

## Key Takeaways for OR Reversal + Edge Fade Implementation

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Order approach | **Managed** | Single entry/exit, fixed targets, works with backtester |
| ATM strategy | **NO** | ATM hangs; managed approach is native and reliable |
| Stop method | **SetStopLoss(Price)** | Dynamic price control for BE moves |
| Target method | **SetProfitTarget(Price)** | Fixed 2R target (OR Rev) / IB midpoint (Edge Fade) |
| Trailing | **Not used** | Backtest showed trail regresses by 45% |
| Breakeven | **Optional** | Can implement but research shows fixed targets are optimal |
| Architecture | **2 separate strategies** | Independent time windows, stops, enables |
| Indicators | **Hosted inside strategy** | Indicator computes signals, strategy manages orders |
| Connection | **Recalculate** | Rebuilds state correctly on reconnect |
| Volumetric | **VolumetricBarsType cast** | Real delta from OrderFlow+ bars |
| Multi-account | **Replikanto** | Already installed, copies orders to all Apex accounts |

---

## Sources

- [NinjaScript Lifecycle](https://ninjatrader.com/support/helpguides/nt8/understanding_the_lifecycle_of.htm)
- [OnStateChange / State](https://ninjatrader.com/support/helpGuides/nt8/state.htm)
- [Managed Approach](https://ninjatrader.com/support/helpGuides/nt8/managed_approach.htm)
- [SubmitOrderUnmanaged](https://ninjatrader.com/support/helpguides/nt8/submitorderunmanaged.htm)
- [SetTrailStop](https://ninjatrader.com/support/helpguides/nt8/settrailstop.htm)
- [OnOrderUpdate](https://ninjatrader.com/support/helpGuides/nt8/onorderupdate.htm)
- [OnExecutionUpdate](https://ninjatrader.com/support/helpguides/nt8/onexecutionupdate.htm)
- [ConnectionLossHandling](https://ninjatrader.com/support/helpguides/nt8/connectionlosshandling.htm)
- [AddChartIndicator](https://ninjatrader.com/support/helpguides/nt8/addchartindicator.htm)
- [Adding Indicators to Strategies](https://ninjatrader.com/support/helpGuides/nt8/adding_indicators_to_strategie.htm)
- [Modifying Stop Loss Price (Reference Sample)](https://ninjatrader.com/support/helpGuides/nt8/modifying_the_price_of_stop_lo.htm)
- [SampleOnOrderUpdate (Reference Sample)](https://ninjatrader.com/support/helpguides/nt8/using_onorderupdate_and_onexec.htm)
- [Breakeven Stop Forum Discussion](https://forum.ninjatrader.com/forum/ninjatrader-8/strategy-development/1089279-how-to-move-a-stoploss-to-breakeven)
- [Trailing Stop Forum Discussion](https://forum.ninjatrader.com/forum/ninjatrader-8/strategy-development/1123406-how-to-program-stoploss-and-profit-trailing)
- [Managed vs Unmanaged Forum](https://forum.ninjatrader.com/forum/ninjatrader-8/strategy-development/96401-managed-vs-unmanaged-trades)
- [Connection Loss Best Practices](https://forum.ninjatrader.com/forum/ninjatrader-8/strategy-development/1313373-best-practices-to-mitigate-connection-lost)
- [Multiple Entries Forum](https://forum.ninjatrader.com/forum/ninjatrader-8/strategy-development/1213509-multiple-entries-multiple-targets)
- [EntriesPerDirection Forum](https://forum.ninjatrader.com/forum/ninjatrader-8/strategy-development/1294598-entries-per-direction)
- [Strategy Indicator Communication Forum](https://forum.ninjatrader.com/forum/ninjatrader-8/strategy-development/1183635-using-custom-indicator-in-strategy)
- [Flatten at Time Forum](https://forum.ninjatrader.com/forum/ninjatrader-8/strategy-development/1212792-close-all-open-orders-flatten-all-open-positions-based-on-a-time-filter)
