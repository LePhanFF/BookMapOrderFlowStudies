# NT8 Indicator Design: 80P + 20P + VA Edge Fade
**Date:** 2026-02-24 | **Version:** 1.0

## Build Order

1. **PriorSessionVA.cs** (shared indicator) -- dependency for 80P + VA Edge
2. **EightyPercentSignal.cs** (80P Rule indicator)
3. **VAEdgeFadeSignal.cs** (VA Edge Fade indicator)
4. **TwentyPercentSignal.cs** (20P Rule indicator)

---

## Shared Indicator: PriorSessionVA.cs

### Purpose
Computes and plots the **prior session ETH** Value Area (VAH, VAL, POC).
Must use ETH session (18:00 - 16:00), NOT RTH only.

### State
```csharp
private SessionIterator _sessionIterator;
private DateTime _currentSessionDate;

// Prior session VA levels (persist across bars within session)
private double _priorVAH, _priorVAL, _priorPOC;
private bool _hasVA;

// Volume profile computation
private SortedDictionary<double, long> _priorVolumeByPrice;
private double _tickSize;
```

### OnBarUpdate Logic
```
1. On new session start:
   - Finalize previous session's volume profile
   - Compute VA from accumulated volume (TPO 70% method):
     a. Find POC (price level with highest volume)
     b. Expand up/down from POC until 70% of total volume captured
     c. Top of expansion = VAH, bottom = VAL
   - Store as _priorVAH, _priorVAL, _priorPOC
   - _hasVA = true

2. On each bar:
   - Accumulate volume at each price level in current session
   - Draw prior VA as:
     * VAH line (magenta, DashStyle.Dash, width 2)
     * VAL line (magenta, DashStyle.Dash, width 2)
     * POC line (white, DashStyle.Solid, width 2)
     * Shaded rectangle between VAH and VAL (magenta, opacity 15%)
```

### Plot Outputs
| Plot | Color | Style |
|------|-------|-------|
| PriorVAH | Magenta | Dash, 2px |
| PriorVAL | Magenta | Dash, 2px |
| PriorPOC | White | Solid, 2px |
| VA Zone | Magenta | Rectangle, 15% opacity |

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| ValueAreaPercent | 70 | % of volume for VA (standard is 70%) |
| UseETH | true | Use ETH (18:00-16:00) vs RTH only |
| ShowZone | true | Show shaded VA rectangle |

### Public Properties (for hosted indicators)
```csharp
public double PriorVAH { get { return _priorVAH; } }
public double PriorVAL { get { return _priorVAL; } }
public double PriorPOC { get { return _priorPOC; } }
public bool HasVA { get { return _hasVA; } }
```

---

## Indicator 1: EightyPercentSignal.cs

### Purpose
Detects the 80% Rule: price opens outside prior VA, re-enters, expect full VA traverse.

### State
```csharp
// Hosted indicators
private PriorSessionVA _priorVA;
private ATR _atr;
private NinjaTrader.NinjaScript.Indicators.OrderFlowVWAP _vwap;

// Session state
private string _openLocation;     // "ABOVE_VA", "BELOW_VA", "INSIDE_VA"
private bool _openDetected;
private bool _reEntryDetected;    // Price has re-entered the VA
private bool _signalFired;
private int _rthBarCount;
private bool _rthStarted;

// Double top/bottom tracking (for Retest model)
private int _touchCount;          // How many times price has touched VA edge from outside
private double _lastTouchExtreme; // High/Low of last touch for double top/bottom
private bool _isDoubleTopBottom;

// Entry tracking
private double _entryPrice, _stopPrice, _targetPrice;
private string _entryModel;       // "ACCEPTANCE", "LIMIT_50PCT", "RETEST_DBL"
```

### OnBarUpdate Logic

```
PHASE 1: Detect open location (bar 0 of RTH)
  if Close[0] > PriorVAH:
    _openLocation = "ABOVE_VA"
  elif Close[0] < PriorVAL:
    _openLocation = "BELOW_VA"
  else:
    _openLocation = "INSIDE_VA"  // 80P does not apply

  if _openLocation == "INSIDE_VA": return  // No setup today

PHASE 2: Track re-entry (subsequent bars)
  if _openLocation == "ABOVE_VA":
    // SHORT setup: price opened above, we want it to drop into VA
    if Close[0] < PriorVAH and !_reEntryDetected:
      _reEntryDetected = true
      _touchCount += 1

    // Track exits (price goes back above VA)
    if Close[0] > PriorVAH and _reEntryDetected:
      _touchCount += 1  // Another touch = potential double top

    // RETEST model: signal on 2nd re-entry (double top at VAH)
    if _touchCount >= 2 and Close[0] < PriorVAH:
      _isDoubleTopBottom = true

  elif _openLocation == "BELOW_VA":
    // LONG setup: price opened below, we want it to rise into VA
    if Close[0] > PriorVAL and !_reEntryDetected:
      _reEntryDetected = true
      _touchCount += 1

    if Close[0] < PriorVAL and _reEntryDetected:
      _touchCount += 1

    if _touchCount >= 2 and Close[0] > PriorVAL:
      _isDoubleTopBottom = true

PHASE 3: Signal generation
  Time gate: only between 9:30 and 13:00 ET

  MODEL A: Acceptance Close
    if _reEntryDetected and first candle close inside VA:
      SIGNAL

  MODEL B: Limit 50% VA Depth
    if _reEntryDetected:
      limit_price = VA_edge + (PriorPOC - VA_edge) * 0.5
      if Low[0] <= limit_price (LONG) or High[0] >= limit_price (SHORT):
        SIGNAL at limit_price

  MODEL C: Retest Double Top/Bottom (RECOMMENDED)
    if _isDoubleTopBottom:
      SIGNAL at VA edge (limit)

PHASE 4: Stop & Target
  // CRITICAL: Stop = VA edge + 10pt buffer. NEVER widen.
  if _openLocation == "BELOW_VA":  // LONG
    stop = PriorVAL - 10.0
    risk = entry - stop
    target = entry + 4 * risk  // 4R target

  elif _openLocation == "ABOVE_VA":  // SHORT
    stop = PriorVAH + 10.0
    risk = stop - entry
    target = entry - 4 * risk

PHASE 5: Draw signal
  - Arrow (big, green LONG / red SHORT)
  - Labeled box: ">>> 80P LONG <<<" or ">>> 80P SHORT <<<"
  - Entry line (gold), Stop line (red), Target line (green)
  - VA zone already drawn by PriorSessionVA
```

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| EntryModel | Retest | Acceptance, Limit50, Retest |
| TargetMultiple | 4.0 | R-multiple for target |
| StopBuffer | 10.0 | Points beyond VA edge for stop |
| EntryCutoffHour | 13 | Last hour for entries (ET) |
| DebugMode | false | Show skip reasons |

### Visual Elements
| Element | Color | Description |
|---------|-------|-------------|
| Open location label | Cyan | "ABOVE VA" or "BELOW VA" at bar 0 |
| Re-entry marker | Yellow dot | When price first enters VA |
| Touch count | Small label | "T1", "T2" at each VA edge touch |
| Signal arrow | Green/Red | Big up/down arrow at signal bar |
| Signal box | Green/Red bg | ">>> 80P LONG <<<" with model name |
| Entry line | Gold, thick | Labeled "ENTRY" |
| Stop line | Red, thick | Labeled "STOP (VA+10)" |
| Target line | Green, thick | Labeled "TARGET (4R)" |

---

## Indicator 2: TwentyPercentSignal.cs

### Purpose
Detects the 20% Rule: IB extension breakout confirmed by 3 consecutive 5-min closes.

### State
```csharp
// IB tracking
private double _ibHigh, _ibLow, _ibRange, _ibMid;
private bool _ibComplete;
private int _rthBarCount;

// Breakout tracking
private string _breakoutSide;     // "BULL" (above IBH) or "BEAR" (below IBL)
private int _consecutiveCloses;   // Count of 5-min closes beyond IB boundary
private bool _signalFired;

// ATR for stop
private ATR _atr;

// 5-min secondary series
// Requires AddDataSeries(BarsPeriodType.Minute, 5)
private int _secondaryIdx;
```

### OnBarUpdate Logic

```
// Only process 5-min bars for close counting
if BarsInProgress == 1 (5-min series):

  PHASE 1: Track IB (first 60 min)
    if time < 10:30:
      Update _ibHigh, _ibLow from 5-min bars
      return

  PHASE 2: Detect breakout direction
    Time gate: 10:30 - 13:00 ET

    if Close > _ibHigh:
      if _breakoutSide != "BULL":
        _breakoutSide = "BULL"
        _consecutiveCloses = 0
      _consecutiveCloses += 1

    elif Close < _ibLow:
      if _breakoutSide != "BEAR":
        _breakoutSide = "BEAR"
        _consecutiveCloses = 0
      _consecutiveCloses += 1

    else:
      _consecutiveCloses = 0  // Reset if price is back inside IB

  PHASE 3: Signal on 3rd consecutive close
    if _consecutiveCloses >= 3 and !_signalFired:
      _signalFired = true

      atr = _atr[0]
      risk = 2 * atr  // 2 ATR stop

      if _breakoutSide == "BULL":
        entry = Close
        stop = entry - risk
        target = entry + 2 * risk  // 2R
        SIGNAL: LONG

      elif _breakoutSide == "BEAR":
        entry = Close
        stop = entry + risk
        target = entry - 2 * risk  // 2R
        SIGNAL: SHORT

// Primary series (1-min): draw levels and signals
if BarsInProgress == 0:
  Draw IB lines, breakout count labels, signal arrows
```

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| ConsecutiveCloses | 3 | Number of 5-min closes needed |
| ATRMultiplier | 2.0 | ATR multiple for stop |
| TargetMultiple | 2.0 | R-multiple for target |
| EntryCutoffHour | 13 | Last hour for entries (ET) |

### Visual Elements
| Element | Color | Description |
|---------|-------|-------------|
| IBH line | Cyan, solid | IB High level |
| IBL line | Cyan, solid | IB Low level |
| Close count | Small label | "1/3", "2/3", "3/3" near breakout bars |
| Signal arrow | Green/Red | Big arrow on 3rd confirming close |
| Signal box | ">>> 20P LONG <<<" with close count |
| Entry/Stop/Target | Gold/Red/Green thick lines |

### Required Data Series
```csharp
protected override void OnStateChange()
{
    if (State == State.Configure)
    {
        AddDataSeries(BarsPeriodType.Minute, 5);  // 5-min for close counting
    }
}
```

---

## Indicator 3: VAEdgeFadeSignal.cs

### Purpose
Detects VA Edge Fade: price pokes outside prior VA edge, fails, fades back in.

### State
```csharp
// Hosted indicators
private PriorSessionVA _priorVA;
private ATR _atr;
private NinjaTrader.NinjaScript.Indicators.OrderFlowVWAP _vwap;

// Session state
private bool _rthStarted;
private int _rthBarCount;

// Edge interaction tracking
private bool _touchedVAH;        // Price has been at or above VAH
private bool _touchedVAL;        // Price has been at or below VAL
private double _sweepExtreme;    // Furthest point beyond VA edge
private int _touchCount;         // For 2nd-test models

// Signal state
private bool _signalFired;
private string _entryModel;      // "INVERSION", "2x5MIN", "LIMIT_SWEEP", "LIMIT_EDGE"

// 5-min secondary series (for 2x5min model)
private int _consecutiveInsideCloses;  // 5-min closes back inside VA
```

### OnBarUpdate Logic

```
Time gate: 9:30 - 13:00 ET

PHASE 1: Track VA edge interactions
  if High[0] >= PriorVAH:
    _touchedVAH = true
    _sweepExtreme = Max(_sweepExtreme, High[0])

  if Low[0] <= PriorVAL:
    _touchedVAL = true
    _sweepExtreme = Min(_sweepExtreme, Low[0])

PHASE 2: Detect fade back into VA

  MODEL A: Inversion Candle (1st touch)
    if _touchedVAH and previous bar closed above VAH and current bar closes below VAH:
      // Bearish inversion (engulfing-like) at VAH
      SIGNAL: SHORT
      stop = entry + 2 * ATR
      target = entry - 4 * risk (swing) or entry - (entry - PriorPOC) for POC target

    if _touchedVAL and previous bar closed below VAL and current bar closes above VAL:
      // Bullish inversion at VAL
      SIGNAL: LONG

  MODEL B: 2x 5-min Close Inside VA
    // On 5-min bars:
    if _touchedVAH and Close < PriorVAH:
      _consecutiveInsideCloses += 1
    else:
      _consecutiveInsideCloses = 0

    if _consecutiveInsideCloses >= 2:
      SIGNAL: SHORT (fading from above VA)

  MODEL C: Limit at Sweep Extreme
    if _touchedVAH and price returns below VAH:
      // Place limit SHORT at _sweepExtreme (catch the retest)
      entry = _sweepExtreme
      stop = entry + 2 * ATR
      target = entry - risk (1R) or PriorPOC

  MODEL D: Limit at VA Edge (2nd test)
    if _touchCount >= 2:
      // On 2nd touch of VA edge
      entry = VA edge
      stop = entry + 2 * ATR (SHORT) or entry - 2 * ATR (LONG)
      target = PriorPOC

PHASE 3: Draw signal
  Signal arrow, entry/stop/target lines, model label
```

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| EntryModel | LimitEdge | Inversion, TwoFiveMin, LimitSweep, LimitEdge |
| ATRMultiplier | 2.0 | ATR multiple for stop |
| TargetMode | POC | POC, OneR, TwoR, FourR |
| RequireSecondTouch | false | Only signal on 2nd VA edge touch |
| EntryCutoffHour | 13 | Last hour for entries (ET) |

### Visual Elements
| Element | Color | Description |
|---------|-------|-------------|
| VA zone | From PriorSessionVA | VAH/VAL/POC (shared) |
| VA edge touch marker | Yellow dot | Where price first touches edge |
| Sweep extreme | Orange dash | Furthest point beyond VA |
| Inversion candle | Highlighted bar | Pattern recognition |
| Signal arrow | Green/Red | Big arrow at signal |
| Signal box | ">>> VA FADE SHORT <<<" with model |
| Entry/Stop/Target | Gold/Red/Green thick lines |

---

## Implementation Notes

### Critical Gotchas from Research

1. **Candle-based stops on retest entries = catastrophic (5-14% WR).** Always use fixed stops (VA+10 or 2 ATR).
2. **0.5 ATR trail universally fails.** No trailing stops. Fixed target exits only.
3. **Must use ETH VA, not RTH.** ETH captures overnight liquidity accumulation.
4. **Hard 13:00 ET entry cutoff.** Afternoon entries degrade significantly.
5. **80P stop widening destroys P&L.** VA edge + 10pt is optimal. Never widen.
6. **2 ATR is the universal stop** for 20P and VA Edge Fade.

### Data Requirements

All indicators need:
- 1-minute primary bars (for precise levels and signals)
- ATR(14) indicator
- OrderFlowVWAP (for VWAP alignment checks)
- PriorSessionVA (shared, for VA levels)

Additional:
- 20P needs `AddDataSeries(BarsPeriodType.Minute, 5)` for 5-min close counting
- VA Edge Fade needs 5-min secondary if using 2x5min model
- Volumetric bars optional (for delta, same candle fallback as OR Rev)

### Compilation Dependencies
```
1. PriorSessionVA.cs       (compile first, no dependencies)
2. EightyPercentSignal.cs  (depends on PriorSessionVA)
3. VAEdgeFadeSignal.cs     (depends on PriorSessionVA)
4. TwentyPercentSignal.cs  (no custom dependencies)
```
