# BookMapOrderFlowStudies-2 — Project Instructions

## Safety Rules
- **NEVER delete files in NinjaTrader 8 folder** — only CREATE or EDIT
- **NEVER delete files outside the GitHub repo** without explicit permission
- NT8 folder: `C:\Users\lehph\Documents\NinjaTrader 8\` — READ and WRITE only
- Always commit to `feature/next-improvements` branch, never force-push main

## Key Paths
- **NT8 Custom dir:** `C:\Users\lehph\Documents\NinjaTrader 8\bin\Custom\`
- **NT8 Error logs:** `C:\Users\lehph\Documents\NinjaTrader 8\log\log.YYYYMMDD.*.en.txt`
- **NT8 Trace logs:** `C:\Users\lehph\Documents\NinjaTrader 8\Rockit*Trace\*.log`
- **Python env:** `E:\anaconda\python.exe`
- **Debug checklist:** `docs/NT8-Indicator-Debug-Checklist.md`

---

## NT8 Indicator Visual Annotation Workflow

**IMPORTANT: Follow this workflow EVERY TIME you create or modify an NT8 indicator.**
The user has repeatedly encountered the same visual annotation bugs. This section prevents rework.

### Required Visual Elements (ALL Rockit Indicators)

Every indicator that fires a signal MUST have ALL of these annotations. Reference: `RockitORReversalSignal.cs`.

#### 1. Signal Arrow + Offset Callout (at signal fire)
```csharp
// Arrow at signal candle
Draw.ArrowDown(this, "Prefix_Arrow_" + d, true, 0, High[0] + 20 * TickSize, SignalColor);

// Dotted connector line to offset callout
Draw.Line(this, "Prefix_SigLine_" + d, false, 0, High[0] + 20 * TickSize, -barsToNoon, textY,
    SignalColor, DashStyleHelper.Dot, 1);

// Offset callout box (colored background: DarkRed=SHORT, DarkGreen=LONG)
Draw.Text(this, "Prefix_Box_" + d, true, ">>> SIGNAL <<<\n" + reason,
    -barsToNoon, textY, 0, Brushes.White, new SimpleFont("Arial", 11),
    TextAlignment.Left, Brushes.Transparent, bgBrush, 80);
```

#### 2. TradingView-Style R:R Zones (entry→target, entry→stop)
```csharp
// Green profit zone (entry → target, opacity 35)
Draw.Rectangle(this, "Prefix_TPZone_" + d, false, barsAgo, entry, 0, target,
    Brushes.Transparent, Brushes.Green, 35);

// Red risk zone (entry → stop, opacity 35)
Draw.Rectangle(this, "Prefix_SLZone_" + d, false, barsAgo, entry, 0, stop,
    Brushes.Transparent, Brushes.Red, 35);
```

#### 3. Entry/Stop/Target Lines + Labels
```csharp
// Entry line (solid), Stop line (dashed red), Target line (dashed green)
Draw.Line(this, "Prefix_Entry_" + d, false, barsAgo, entry, 0, entry,
    SignalColor, DashStyleHelper.Solid, 2);
Draw.Line(this, "Prefix_Stop_" + d, false, barsAgo, stop, 0, stop,
    Brushes.Red, DashStyleHelper.Dash, 2);
Draw.Line(this, "Prefix_Tgt_" + d, false, barsAgo, target, 0, target,
    Brushes.Lime, DashStyleHelper.Dash, 2);

// Right-aligned labels at bar 0, font 10
Draw.Text(this, "Prefix_EntryLbl_" + d, true, "ENTRY " + entry.ToString("F2"),
    0, entry, 0, SignalColor, new SimpleFont("Arial", 10),
    TextAlignment.Right, Brushes.Transparent, Brushes.Transparent, 0);
Draw.Text(this, "Prefix_StopLbl_" + d, true, "STOP " + stop.ToString("F2"),
    0, stop, 0, Brushes.Red, new SimpleFont("Arial", 10),
    TextAlignment.Right, Brushes.Transparent, Brushes.Transparent, 0);
Draw.Text(this, "Prefix_TgtLbl_" + d, true, "TARGET " + target.ToString("F2"),
    0, target, 0, Brushes.Lime, new SimpleFont("Arial", 10),
    TextAlignment.Right, Brushes.Transparent, Brushes.Transparent, 0);
```

#### 4. Key Session Levels (non-cluttered)
- IB High/Low: dashed lines, session-scoped only (not extending across sessions)
- VA High/Low/POC: if applicable, redraw every bar
- London H/L, Overnight H/L, PDH/PDL: labeled lines within session only

### Critical Rules (Prevent Recurring Bugs)

#### Right Edge = `0` (ALWAYS)
```
NEVER use negative values (future bars) — causes lines to extend beyond session
NEVER calculate rightEdge from time-to-close — pushes zones too far right
ALWAYS use 0 (current bar) as right edge for all Draw calls
Zones grow from signal bar to current bar, then freeze when ResetSession clears _signalFired
```

#### DrawSignalLevels MUST Be Reachable After Signal Fires
```csharp
// WRONG — zones never drawn after signal fires:
if (_signalFired) return;
// ... 100 lines later ...
DrawSignalLevels();

// CORRECT — draw first, then return:
if (_signalFired)
{
    if (isRTH && tod < new TimeSpan(16, 0, 0))
        DrawSignalLevels();
    return;
}
```

#### Session Scoping (Prevent Cross-Session Bleed)
- All draw tags use `_sessionDate.ToString("MMdd")` suffix
- `ResetSession()` MUST clear: `_signalFired = false`, `_signalBar = 0`, `_signalBarPrimary = 0`
- IsFirstBarOfSession or RTH start triggers ResetSession
- Old session drawings freeze at their last state (right edge was `0` at last bar)

#### barsAgo Clamping (Prevent Crash)
```csharp
int barsAgo = Math.Min(CurrentBar - _signalBar, Math.Max(CurrentBar - 1, 0));
// If _signalBar from different BarsInProgress, use BarsArray[0].Count - 1
```

#### NEVER Use Hosted Indicators (CacheIndicator is Broken)
```
NEVER call RockitPriorSessionVA(...) or similar hosted indicator patterns.
NT8's CacheIndicator creates temp instances that terminate without processing bars.
OnBarUpdate is never called on the cached instance → HasVA stays false forever.
ALWAYS compute shared data (VA, IB levels, etc.) INLINE inside the indicator that needs it.
```

#### Try-Catch All OnBarUpdate (Prevent Silent Death)
```csharp
protected override void OnBarUpdate()
{
    if (BarsInProgress != 0) return;
    if (CurrentBar < 10) return;
    try { OnBarUpdateInner(); }
    catch (Exception ex) { /* log but keep processing */ }
}
```
NT8 permanently disables an indicator after ONE unhandled exception. If a hosted indicator crashes, it kills the host too.

#### Draw Time Window = Full RTH (16:00)
```csharp
// WRONG — trade at 12:55 only shows zone for 5 minutes:
if (tod < new TimeSpan(13, 0, 0)) DrawSignalLevels();

// CORRECT — zones visible all session:
if (tod < new TimeSpan(16, 0, 0)) DrawSignalLevels();
```

### Debug Workflow (When Annotations Are Missing)

Follow `docs/NT8-Indicator-Debug-Checklist.md` in order:

1. **Check error log** for crashes — `log/log.YYYYMMDD.*.en.txt`, search indicator name
2. **Check trace log** (newest file in `Rockit*Trace/`) — verify state flags, signal fires
3. **Check draw reachability** — search for `return;` before `DrawSignalLevels`
4. **Check draw parameters** — barsAgo >= 0, right edge = 0, entry/stop/target non-zero
5. **Check session reset** — _signalFired clears, _signalBar resets
6. **Check hosted indicators** — PriorVA crash kills 80P; wrap in try-catch
7. **If NinjaScriptProperty changed** — user must remove + re-add indicator

### Trace Logging Standard

Every indicator MUST write trace logs to `Documents\NinjaTrader 8\Rockit{Name}Trace\`:
```csharp
private void Trace(string msg)
{
    string dir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
        "NinjaTrader 8", "Rockit" + Name + "Trace");
    Directory.CreateDirectory(dir);
    string file = Path.Combine(dir, Name + "_trace_" + DateTime.Now.ToString("yyyyMMdd_HHmmss") + ".log");
    // Append to file...
}
```

Log these events:
- Session start/reset with bar number and HasVA/state flags
- Signal fire with direction, entry, stop, target, reason
- DrawSignalLevels called with barsAgo, prices, BarsInProgress
- Any skip/filter reason (no VA, wrong regime, time gate, etc.)
- Errors caught by try-catch

### Indicator Annotation Checklist (Quick Verify)

Before declaring an indicator "done", verify on chart:
- [ ] Arrow points at signal candle
- [ ] Dotted line connects arrow to offset callout
- [ ] Callout box has colored background (red=SHORT, green=LONG), white text
- [ ] Green rectangle from entry to target (opacity 35)
- [ ] Red rectangle from entry to stop (opacity 35)
- [ ] Solid entry line, dashed red stop line, dashed green target line
- [ ] Right-aligned price labels for entry, stop, target
- [ ] Key levels (IB H/L, VA, London, etc.) visible but NOT extending beyond session
- [ ] No lines/zones bleeding into next session
- [ ] No zones pushed to end of session (should start at signal bar)
- [ ] Alert sound plays on signal fire
- [ ] Trace log written with correct state progression

---

## NT8 NinjaScript C# Coding Rules (Hard-Learned)

**IMPORTANT: Follow these rules EVERY TIME you write or modify NT8 C# code.**
Every rule here was learned from a real production bug that cost hours of debugging.

### Indicator Skeleton (Copy This for Every New Indicator)

```csharp
#region Using declarations
using System;
using System.IO;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Windows.Media;
using System.Xml.Serialization;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.Gui.Tools;
using NinjaTrader.NinjaScript.BarsTypes;       // for VolumetricBarsType
using NinjaTrader.NinjaScript.DrawingTools;
#endregion

namespace NinjaTrader.NinjaScript.Indicators
{
    public class RockitMySignal : Indicator
    {
        private SessionIterator _sessionIterator;
        private DateTime        _lastTradingDay = DateTime.MinValue;
        private DateTime        _sessionDate;
        private bool            _traceEnabled;
        private string          _traceFile;

        // --- State (reset each session) ---
        private bool   _signalFired;
        private int    _signalBar;
        private double _entryPrice, _stopPrice, _targetPrice;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Rockit My Signal";
                Name = "RockitMySignal";
                Calculate = Calculate.OnBarClose;
                IsOverlay = true;
                IsSuspendedWhileInactive = true;
                // Properties with defaults here
                EnableTraceLog = true;
            }
            else if (State == State.DataLoaded)
            {
                _sessionIterator = new SessionIterator(Bars);
                if (EnableTraceLog)
                {
                    _traceEnabled = true;
                    string dir = Path.Combine(NinjaTrader.Core.Globals.UserDataDir,
                        "Rockit" + Name + "Trace");
                    Directory.CreateDirectory(dir);
                    _traceFile = Path.Combine(dir, Name + "_trace_"
                        + DateTime.Now.ToString("yyyyMMdd_HHmmss") + ".log");
                    Trace("=== " + Name + " Trace Started ===");
                }
            }
        }

        protected override void OnBarUpdate()
        {
            if (BarsInProgress != 0) return;
            if (CurrentBar < 10) return;
            try { OnBarUpdateInner(); }
            catch (Exception ex) { Trace("ERROR: " + ex.Message); }
        }

        private void OnBarUpdateInner()
        {
            // Session boundary
            if (Bars.IsFirstBarOfSession)
            {
                _sessionIterator.GetNextSession(Time[0], true);
                DateTime td = _sessionIterator.GetTradingDay(Time[0]);
                if (td != _lastTradingDay)
                {
                    _lastTradingDay = td;
                    ResetSession();
                    _sessionDate = td;
                }
            }

            TimeSpan tod = Time[0].TimeOfDay;

            // Pre-market level tracking (bar-by-bar, not backward scan)
            if (tod >= new TimeSpan(3, 0, 0) && tod < new TimeSpan(8, 30, 0))
            {
                TrackPremarketLevels();
                return;
            }

            // RTH logic
            bool isRTH = tod >= new TimeSpan(9, 30, 0) && tod < new TimeSpan(16, 0, 0);
            if (!isRTH) return;

            // Draw zones BEFORE early-return on _signalFired
            if (_signalFired)
            {
                DrawSignalLevels();
                return;
            }

            // ... signal detection logic ...
        }

        private void ResetSession() { /* clear ALL state */ }
        private void TrackPremarketLevels() { /* London/Asia/ON bar-by-bar */ }
        private void DrawSignalLevels() { /* all Draw.* calls here */ }
        private void Trace(string msg) { /* append to trace file */ }
    }
}
```

### Session & Time Handling

#### NEVER use `Time[0].Date` for session identification
```csharp
// WRONG — Sunday pre-market bar returns Sunday date:
DateTime sessionDate = Time[0].Date;

// CORRECT — returns actual CME trading day:
_sessionIterator.GetNextSession(Time[0], true);
DateTime tradingDay = _sessionIterator.GetTradingDay(Time[0]);

// ALWAYS add _lastTradingDay guard (IsFirstBarOfSession fires multiple times):
if (tradingDay != _lastTradingDay)
{
    _lastTradingDay = tradingDay;
    ResetSession();
}
```

#### Pre-market levels: ALWAYS track bar-by-bar, NEVER backward scan
```csharp
// WRONG — backward scan fails on first session and holidays:
for (int i = CurrentBar - 1; i >= 0; i--)
    if (Time[i].TimeOfDay >= londonStart) { londonHigh = High[i]; break; }

// CORRECT — accumulate during the window:
if (tod >= new TimeSpan(3, 0, 0) && tod < new TimeSpan(8, 30, 0))
{
    if (High[0] > _londonHigh || _londonHigh == 0) _londonHigh = High[0];
    if (Low[0]  < _londonLow  || _londonLow  == 0) _londonLow  = Low[0];
}
// Asia: 20:00+, Overnight: 18:00-9:30, PDH/PDL: carried from prior IB range
```

#### Time windows — use TimeSpan, not DateTime comparison
```csharp
TimeSpan tod = Time[0].TimeOfDay;
if (tod >= new TimeSpan(9, 30, 0) && tod < new TimeSpan(16, 0, 0)) { /* RTH */ }
if (tod >= new TimeSpan(3, 0, 0) && tod < new TimeSpan(8, 30, 0)) { /* London */ }
```

### Volumetric Data Access

```csharp
// Cast to VolumetricBarsType — the ONLY way to get order flow:
private long GetBarDelta()
{
    var bType = Bars.BarsSeries.BarsType as VolumetricBarsType;
    if (bType != null)
    {
        try { return bType.Volumes[CurrentBar].BarDelta; }
        catch { }
    }
    // Fallback: estimate from candle body ratio
    double body = Close[0] - Open[0];
    double range = High[0] - Low[0];
    return range > 0 ? (long)((body / range) * Volume[0]) : 0;
}
```
**Always try-catch volumetric access** — fails if chart not using volumetric bars.

### Volume Profile / VA Inline Computation

```csharp
// NEVER use hosted indicators — compute VA inline:
private SortedDictionary<double, long> _volByPrice = new SortedDictionary<double, long>();
private double _tickSz = 0.25;  // NQ tick size

private void AccumulateVolume()
{
    long barVol = (long)Volume[0];
    if (barVol <= 0 || High[0] <= Low[0])
    {
        double key = RoundTick(Close[0]);
        _volByPrice[key] = _volByPrice.GetValueOrDefault(key) + Math.Max(barVol, 1);
        return;
    }
    int tickCount = Math.Max((int)Math.Round((High[0] - Low[0]) / _tickSz) + 1, 1);
    long volPerTick = barVol / tickCount;
    long remainder = barVol % tickCount;
    int slot = 0;
    for (double p = Low[0]; p <= High[0] + _tickSz * 0.5; p += _tickSz)
    {
        double key = RoundTick(p);
        _volByPrice[key] = _volByPrice.GetValueOrDefault(key)
                         + volPerTick + (slot++ == 0 ? remainder : 0);
    }
}

private double RoundTick(double price)
{
    return _tickSz > 0 ? Math.Round(price / _tickSz) * _tickSz : price;
}

private void FinalizeVA()  // Call at end of prior session
{
    if (_volByPrice.Count == 0) return;
    long totalVol = 0; double poc = 0; long pocVol = 0;
    foreach (var kv in _volByPrice)
    {
        totalVol += kv.Value;
        if (kv.Value > pocVol) { pocVol = kv.Value; poc = kv.Key; }
    }
    // Expand from POC until 70% volume captured
    long targetVol = (long)(totalVol * 0.70);
    var prices = _volByPrice.Keys.ToList();
    int pocIdx = prices.IndexOf(poc);
    long vaVol = pocVol; int lo = pocIdx, hi = pocIdx;
    while (vaVol < targetVol)
    {
        bool canUp = hi < prices.Count - 1, canDown = lo > 0;
        if (!canUp && !canDown) break;
        long upV = canUp ? _volByPrice[prices[hi + 1]] : 0;
        long dnV = canDown ? _volByPrice[prices[lo - 1]] : 0;
        if (canUp && (!canDown || upV >= dnV)) { hi++; vaVol += _volByPrice[prices[hi]]; }
        else { lo--; vaVol += _volByPrice[prices[lo]]; }
    }
    _priorPOC = poc; _priorVAH = prices[hi]; _priorVAL = prices[lo]; _hasVA = true;
}
```

### NinjaScriptProperty Patterns

```csharp
// Standard range property
[NinjaScriptProperty]
[Range(1.0, 5.0)]
[Display(Name = "Target Multiple (R)", Order = 1, GroupName = "Parameters")]
public double TargetMultiple { get; set; }

// Bool property
[NinjaScriptProperty]
[Display(Name = "Enable Trace Log", Order = 10, GroupName = "Debug")]
public bool EnableTraceLog { get; set; }

// Brush (color) property — REQUIRES serialization pair
[NinjaScriptProperty] [XmlIgnore]
[Display(Name = "Signal Color", Order = 5, GroupName = "Display")]
public Brush SignalColor { get; set; }

[Browsable(false)]
public string SignalColorSerializable
{
    get { return Serialize.BrushToString(SignalColor); }
    set { SignalColor = Serialize.StringToBrush(value); }
}

// Enum property
public enum TargetMode { POC, OneR, TwoR, FourR }

[NinjaScriptProperty]
[Display(Name = "Target Mode", Order = 2, GroupName = "Parameters")]
public TargetMode MyTargetMode { get; set; }
```
**CRITICAL:** If you add/remove/rename a `[NinjaScriptProperty]`, the user MUST remove and re-add the indicator from the chart. The cache signature changes and NT8 won't load old settings.

### ResetSession() — Complete Pattern

```csharp
private void ResetSession()
{
    _signalFired = false;
    _signalBar = 0;
    _entryPrice = 0; _stopPrice = 0; _targetPrice = 0;

    // Range levels
    _ibHigh = 0; _ibLow = 0;
    _eorHigh = 0; _eorLow = double.MaxValue;  // MaxValue for lows!
    _orHigh = 0; _orLow = 0;

    // Pre-market (DO NOT reset PDH/PDL — carry forward from prior session)
    _londonHigh = 0; _londonLow = 0;
    _overnightHigh = 0; _overnightLow = 0;
    // _pdh, _pdl intentionally NOT reset

    // State machine
    _smState = "WAITING";

    // Volume profile
    _volByPrice.Clear();
    _cvd = 0;

    // Acceptance counters
    _acceptCount = 0;
}
```
**Key rule:** `double.MaxValue` for low extremes (not 0, which would be "valid" price). Guard DrawSessionSummary with `if (_eorLow == double.MaxValue) return;`.

### 5-Minute Bar Zone Detection

```csharp
// WRONG — close-based misses wicks:
bool inZone = Close[0] >= zoneBottom && Close[0] <= zoneTop;

// CORRECT — wick-based catches touches:
bool inZone = High[0] >= zoneBottom && Low[0] <= zoneTop;
// For directional: e.g., SHORT entry zone
bool inFiftyPctZone = High[0] >= fiftyPct - buffer && Close[0] <= fiftyPct + buffer;
```
On 5-min bars, price wicks into zones but often closes back outside. Always use High[0]/Low[0] for zone detection.

### CVD (Cumulative Delta) Tracking

```csharp
private long _cvd;
private long _cvdAtSweep;      // snapshot at key event
private long _cvdAtReversal;   // snapshot at key event

// Every bar:
long delta = GetBarDelta();
_cvd += delta;

// Divergence check:
long cvdSinceEvent = _cvd - _cvdAtReversal;
bool bearishDivergence = cvdSinceEvent < 0;  // sellers in control despite price rise
```

### Closest-Level Detection (Not First Match)

```csharp
// WRONG — first match may not be closest:
if (_eorHigh >= _londonHigh - threshold) { level = "LDN H"; }
else if (_eorHigh >= _overnightHigh - threshold) { level = "ON H"; }

// CORRECT — check all, keep closest:
double bestDist = double.MaxValue;
string bestLevel = "";
foreach (var (name, price) in new[] { ("LDN H", _londonHigh), ("ON H", _overnightHigh),
                                       ("PDH", _pdh) })
{
    if (price > 0 && _eorHigh >= price - threshold)
    {
        double dist = Math.Abs(_eorHigh - price);
        if (dist < bestDist) { bestDist = dist; bestLevel = name; }
    }
}
// Also cap: bestDist must be <= 1x EOR range (no 400pt "sweeps")
```

### Dual-Sweep Preference

```csharp
if (sweptHigh && sweptLow)
{
    double highDepth = _eorHigh - GetRefPrice(highLevel);
    double lowDepth = GetRefPrice(lowLevel) - _eorLow;
    if (lowDepth > highDepth) sweptHigh = false;  // prefer deeper
    else sweptLow = false;
}
```

### OrderFlowVWAP Usage

```csharp
// In State.DataLoaded — always try-catch:
try
{
    _vwap = OrderFlowVWAP(VWAPResolution.Standard,
        Bars.TradingHours, VWAPStandardDeviations.Three, 1, 2, 3);
}
catch { _vwap = null; }

// In OnBarUpdate — null-check before use:
if (_vwap != null && _vwap[0] > 0) { double vwapPrice = _vwap[0]; }
```

### Trace Helper (Standard Implementation)

```csharp
private void Trace(string msg)
{
    if (!_traceEnabled || string.IsNullOrEmpty(_traceFile)) return;
    try
    {
        using (StreamWriter sw = new StreamWriter(_traceFile, true))  // append mode
            sw.WriteLine(DateTime.Now.ToString("HH:mm:ss.fff") + " | " + msg);
    }
    catch { }  // Silent fail — never crash on logging
}
```
Log on every state transition, signal fire, skip reason, and caught exception.

### Compilation & Reload Workflow

1. Save `.cs` to `bin\Custom\Indicators\`
2. In NT8: right-click → Compile (or F5 in NinjaScript editor)
3. Check Output window for compile errors
4. **If [NinjaScriptProperty] changed:** remove indicator from chart, re-add
5. Otherwise: right-click chart → Reload NinjaScript
6. Check error log: `log/log.YYYYMMDD.*.en.txt`
7. Check trace: newest file in `Rockit*Trace/`

### Common Compilation Errors & Fixes

| Error | Fix |
|-------|-----|
| `CS0246 'VolumetricBarsType' not found` | Add `using NinjaTrader.NinjaScript.BarsTypes;` |
| `CS0246 'SimpleFont' not found` | Use `NinjaTrader.Gui.Tools.SimpleFont` or `new Gui.Tools.SimpleFont(...)` |
| `CS0246 'TextAlignment' not found` | Use `System.Windows.TextAlignment` |
| `CS1061 'Brush' does not contain 'X'` | Brushes are `System.Windows.Media.Brushes` |
| `CS0019 operator '<' on TimeSpan` | Use `tod >= new TimeSpan(h, m, s)` not `tod >= time(h,m)` |
| `'Bars' does not exist in current context` | You're in State.SetDefaults — move to State.DataLoaded |
| Indicator disabled after first bar | Unhandled exception — wrap OnBarUpdate in try-catch |
| Draw objects invisible | Check: right edge = 0, barsAgo >= 0, price != 0, draw called every bar |

### Strategy Lifecycle (When Converting Indicator → Strategy)

```
State flow: SetDefaults → Configure → Active → DataLoaded → Historical → Transition → Realtime → Terminated
```

```csharp
// SetDefaults — LEAN, UI properties only:
Calculate = Calculate.OnBarClose;
EntriesPerDirection = 1;
EntryHandling = EntryHandling.UniqueEntries;
IsExitOnSessionCloseStrategy = true;
ExitOnSessionCloseSeconds = 60;
TraceOrders = true;  // ESSENTIAL for debugging

// Configure — add data series:
AddDataSeries(BarsPeriodType.Minute, 5);

// DataLoaded — instantiate indicators:
_atr = ATR(Close, 14);
_sessionIterator = new SessionIterator(Bars);
```

### Order Management (Managed Approach — ALWAYS Use This)

```csharp
// Entry
EnterLong(1, "MyLong");
SetStopLoss("MyLong", CalculationMode.Price, stopPrice, false);
SetProfitTarget("MyLong", CalculationMode.Price, targetPrice);

// NEVER use ATM strategies — hangs on partial fills, race conditions
// NEVER widen stops after entry — destroys P&L
// NEVER use trailing stops — backtested: 45% P&L regression
// Fixed target exits ONLY (proven by 259-session backtest)
```

---

## Python Backtest Workflow

- Entry point: `python scripts/run_backtest.py --strategies "Opening Range Rev" --instrument NQ`
- Full portfolio: `python scripts/run_backtest.py --instrument NQ`
- Python env: `E:\anaconda\python.exe`
- PositionManager: always use `max_drawdown=999999` (default 4000 blocks trades)
- CSV data in `csv/` (gitignored, local only)
