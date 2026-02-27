# NT8 Indicator Visual Annotation Debug Checklist

Use this checklist when an NT8 Rockit indicator's visual annotations (lines, zones, labels, arrows) are missing or broken.

---

## 1. Check for Crashes (kills the indicator silently)

- [ ] **Read error log**: `Documents/NinjaTrader 8/log/log.YYYYMMDD.*.en.txt`
  - Search for `Error` + indicator name
  - Common crash: `"accessing a series [barsAgo] with a value of X when there are only Y bars"`
  - **Fix**: Clamp all `barsAgo` to `Math.Min(value, CurrentBar - 1)`
  - **Fix**: Wrap entire OnBarUpdate in try-catch so one bad bar doesn't kill the indicator forever
  - If a hosted indicator crashes (e.g., PriorVA), it takes the host (80P) down with it

- [ ] **Check trace files**: `Documents/NinjaTrader 8/Rockit*Trace/*.log`
  - Read the NEWEST trace file (sorted by timestamp in filename)
  - Confirm `HasVA`, `_signalFired`, and other state flags are correct
  - If trace file doesn't exist → indicator never initialized (crash in State.DataLoaded?)

## 2. Check Draw Call Reachability

- [ ] **Early-return gates**: Search for `return;` statements BEFORE `DrawSignalLevels()`
  - BUG PATTERN: `if (_signalFired) return;` placed before `DrawSignalLevels()` → zones never drawn
  - **Fix**: Move DrawSignalLevels call ABOVE the early-return, or split into:
    ```csharp
    if (_signalFired) { DrawSignalLevels(); return; }
    ```

- [ ] **Time gates**: Check `tod < TimeSpan(H, 0, 0)` around DrawSignalLevels
  - If cutoff is 13:00 but trade fires at 12:55, zones only draw for 5 minutes
  - **Fix**: Use `tod < new TimeSpan(16, 0, 0)` (full RTH)

- [ ] **BarsInProgress gates**: On multi-series indicators (20P), confirm Draw runs on correct series
  - Signals fire on secondary (BarsInProgress==1)
  - Draw runs on primary (BarsInProgress==0) — verify it's not blocked

## 3. Check Draw Parameters

- [ ] **barsAgo (left edge)**: `CurrentBar - _signalBar`
  - Must be >= 0 and <= CurrentBar - 1
  - If `_signalBar` was set from a different BarsInProgress, use `BarsArray[0].Count - 1` not `CurrentBar`

- [ ] **Right edge**: Use `0` (current bar)
  - `0` = current bar at draw time → zones grow as bars process → freeze when _signalFired resets
  - DON'T use negative values (future bars) — causes lines to extend beyond session
  - DON'T calculate rightEdge from time-to-close — pushes zones too far right

- [ ] **Tags**: Must be unique per session
  - Pattern: `"Prefix_" + _sessionDate.ToString("MMdd")`
  - Verify no tag collisions across sessions (MMdd is unique within a year)

- [ ] **Price values**: Verify entry/stop/target are non-zero
  - `if (_entryPrice == 0) return;` guard at top of DrawSignalLevels

## 4. Check Session Reset

- [ ] **ResetSession()** must clear `_signalFired = false`
  - Without this, old session's drawings keep updating into new session
  - Verify `IsFirstBarOfSession` or RTH start calls ResetSession

- [ ] **_signalBar / _signalBarPrimary** must reset to 0
  - Stale values cause huge `barsAgo` on next session

## 5. Hosted Indicator Issues (PriorVA, etc.)

- [ ] **Hosted indicator crash = host dies**
  - If PriorVA crashes, 80P never gets HasVA=true
  - **Fix**: Wrap hosted indicator's entire OnBarUpdate in try-catch

- [ ] **showZone / showLabels / showIBLevels** params
  - When hosting: verify params match what you want visible
  - Example: `RockitPriorSessionVA(VA%, showZone=true, showLabels=true, showIBLevels=true, ...)`

- [ ] **NinjaScriptProperty changes require remove + re-add**
  - If you add/remove/rename a `[NinjaScriptProperty]`, the cache signature changes
  - User must remove the indicator from chart and re-add it

## 6. Compile & Reload Steps

1. Save .cs file
2. In NT8: right-click Indicators → "Compile" (or F5 in NinjaScript editor)
3. Check Output window for compile errors
4. If NinjaScriptProperty changed: remove indicator from chart, re-add
5. Otherwise: right-click chart → Reload NinjaScript
6. Check error log for runtime errors
7. Check trace file for correct state progression

## 7. Standard TV-Style Drawing Pattern (Reference)

```csharp
// Arrow at signal candle
Draw.ArrowDown(this, "Prefix_Arrow_" + d, true, 0, High[0] + 20 * TickSize, SignalColor);

// Dotted line to offset callout at noon
Draw.Line(this, "Prefix_SigLine_" + d, false, 0, High[0] + 20 * TickSize, -barsToNoon, textY,
    SignalColor, DashStyleHelper.Dot, 1);

// Offset callout with colored background
Draw.Text(this, "Prefix_Box_" + d, true, ">>> SIGNAL <<<\n" + reason,
    -barsToNoon, textY, 0, Brushes.White, new SimpleFont("Arial", 11),
    TextAlignment.Left, Brushes.Transparent, Brushes.DarkRed, 80);

// Green profit zone (entry → target, opacity 35)
Draw.Rectangle(this, "Prefix_TPZone_" + d, false, barsAgo, entry, 0, target,
    Brushes.Transparent, Brushes.Green, 35);

// Red risk zone (entry → stop, opacity 35)
Draw.Rectangle(this, "Prefix_SLZone_" + d, false, barsAgo, entry, 0, stop,
    Brushes.Transparent, Brushes.Red, 35);

// Entry line (solid), Stop line (dashed red), Target line (dashed green)
Draw.Line(this, "Prefix_Entry_" + d, false, barsAgo, entry, 0, entry,
    SignalColor, DashStyleHelper.Solid, 2);
Draw.Line(this, "Prefix_Stop_" + d, false, barsAgo, stop, 0, stop,
    Brushes.Red, DashStyleHelper.Dash, 2);
Draw.Line(this, "Prefix_Tgt_" + d, false, barsAgo, target, 0, target,
    Brushes.Lime, DashStyleHelper.Dash, 2);

// Labels at current bar, right-aligned, font 10
Draw.Text(this, "Prefix_EntryLbl_" + d, true, "ENTRY " + entry.ToString("F2"),
    0, entry, 0, SignalColor, new SimpleFont("Arial", 10),
    TextAlignment.Right, Brushes.Transparent, Brushes.Transparent, 0);
```

## Common Bugs Found (2026-02-25)

| Bug | Indicator | Root Cause | Fix |
|-----|-----------|------------|-----|
| HasVA always false | 80P / PriorVA | PriorVA crash at bar 276 kills indicator forever | try-catch around entire OnBarUpdate |
| TV zones never drawn | Acceptance, 80P | `if (_signalFired) return;` before DrawSignalLevels | Move draw call before early-return |
| Lines extend across sessions | 20P | Right edge used `-200` (future bars) | Use `0` (current bar) |
| Lines too far right | All | rightEdge calculated to 16:00 | Use `0` (current bar) |
| IB lines too cluttered | 20P | Fixed `-120` bar extension | Session-scoped `barsToClose` |
| No VA zone visible | PriorVA | DrawPriorVALevels only called once at session start | Redraw on every bar |
| "value of 5 when only 4 bars" | PriorVA | Draw.Line barsAgo > available bars | Clamp lookback, try-catch |
