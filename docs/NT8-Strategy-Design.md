# NinjaTrader 8 Strategy Design Document

## Current State (2026-02-25)

### Indicators Built (Rockit Suite)

| File | Strategy | Direction | Notes |
|------|----------|-----------|-------|
| `RockitORReversalSignal.cs` | OR Reversal (Judas Swing) | LONG + SHORT | State machine: SWEEP→REVERSAL→RETEST→FIRE |
| `RockitORAcceptanceSignal.cs` | OR Acceptance (London Break) | LONG + SHORT | Separate chart; bias-scored |
| `RockitVAEdgeFadeSignal.cs` | Edge Fade (IB MR) | LONG ONLY | Prior session VA levels |
| `RockitEightyPercentSignal.cs` | 80% Rule | LONG + SHORT | VA edge acceptance |
| `RockitPriorSessionVA.cs` | Prior Session VA | Visual only | VA High/Low/POC display |
| `ORReversalSignal.cs` | OR Reversal (original) | LONG + SHORT | Kept for reference |
| `EdgeFadeSignal.cs` | Edge Fade (original) | LONG ONLY | Kept for reference |

**Core Portfolio (Python backtest, 259 sessions):** 149 trades, 61.1% WR, $23,516 net, PF 2.42, MaxDD $2,262, **zero losing months**

---

## Why Edge Fade is LONG ONLY (No Short)

From 259-session backtest: **ALL NQ short-side mean reversion is negative expectancy.**
- NQ has structural long bias (tech growth, passive inflows)
- Short MR entries at IB high get steamrolled by continuation
- Only profitable NQ shorts are momentum/sweep-based (OR Reversal SHORT)
- Edge Fade SHORT was explicitly tested and disabled

---

## Architecture: Indicator → Strategy Framework

### Phase 1: Indicators (CURRENT — visual signals + alerts)
```
Indicators/
  ORReversalSignal.cs    -- draws OR/EOR zones, detects sweep, fires signal
  EdgeFadeSignal.cs      -- draws IB zone, detects edge entry, fires signal
```

### Phase 2: Strategies (NEXT — automated execution)
```
Strategies/
  ORReversalStrategy.cs  -- hosts ORReversalSignal, manages orders
  EdgeFadeStrategy.cs    -- hosts EdgeFadeSignal, manages orders
```

### Phase 3: Framework Strategies (FUTURE — adaptable to new setups)
```
Indicators/
  IBClassifier.cs        -- shared: computes IB H/L/mid/range, day type, regime
  SignalBase.cs           -- shared: common signal state, level tracking
Strategies/
  ManagedStrategyBase.cs -- shared: order management, trail/BE, EOD flatten
```

---

## Strategy Execution Design (No ATM)

### Why Not ATM
ATM strategies in NinjaTrader sometimes hang on:
- Partial fills
- Connection drops/reconnects
- Multiple order modifications in quick succession
- Race conditions between ATM and chart trader

### Managed Approach (Recommended)
NinjaScript's managed approach handles OCO grouping automatically and works with Strategy Analyzer. The strategy hosts the indicator, reads signals, and manages orders directly.

### Order Management Flow
```
OnBarUpdate():
  1. Check indicator for new signal
  2. If signal AND flat → EnterLong/EnterShort with signal name
  3. SetStopLoss(CalculationMode.Price, stopPrice)
  4. SetProfitTarget(CalculationMode.Price, targetPrice)

OnPositionUpdate():
  5. On entry fill → record entry price, start tracking
  6. On flat → reset state, log trade result

OnBarUpdate() while in position:
  7. Check trail/BE conditions
  8. Check time-based exit (EOD 15:59)
  9. Adjust stop via SetStopLoss(CalculationMode.Price, newStop)
```

### Stop Management Options
```csharp
// Fixed stop (default for both strategies)
SetStopLoss("entry1", CalculationMode.Price, stopPrice, false);

// Breakeven after X ticks profit
if (Position.MarketPosition == MarketPosition.Long
    && Close[0] >= entryPrice + breakEvenTriggerTicks * TickSize)
{
    SetStopLoss("entry1", CalculationMode.Price, entryPrice + 2 * TickSize, false);
    breakEvenTriggered = true;
}

// Trailing stop (manual control)
if (Position.MarketPosition == MarketPosition.Long)
{
    highSinceEntry = Math.Max(highSinceEntry, High[0]);
    double newStop = highSinceEntry - trailTicks * TickSize;
    if (newStop > currentStop)
        SetStopLoss("entry1", CalculationMode.Price, newStop, false);
}
```

### Time-Based Exit (EOD Flatten)
```csharp
// Safety: auto-flatten at 15:59
if (ToTime(Time[0]) >= 155900 && Position.MarketPosition != MarketPosition.Flat)
{
    if (Position.MarketPosition == MarketPosition.Long) ExitLong("EOD");
    else ExitShort("EOD");
}

// Also enable built-in safety net
IsExitOnSessionCloseStrategy = true;
ExitOnSessionCloseSeconds = 60;
```

### Connection Loss Handling
```csharp
// In SetDefaults:
ConnectionLossHandling = ConnectionLossHandling.Recalculate;
DisconnectDelaySeconds = 10;
```

---

## OR Reversal Signal — Logic Detail (v3, 2026-02-25)

### Levels Tracked
| Level | Color | Source |
|-------|-------|--------|
| OR High/Low/Mid | DodgerBlue (shaded rect) | First 15 RTH bars (9:30-9:44) |
| EOR High/Low | (internal) | First 30 RTH bars (9:30-9:59) |
| London High/Low | Yellow | 03:00-09:30 ET (pre-market scan) |
| Overnight High/Low | Orange | Prior close → RTH open |
| Prior Day High/Low | Magenta | Previous session RTH H/L |

### State Machine
```
WAITING → SWEEP detected (EOR extreme near London/ON/PDH/PDL)
       → REVERSAL (price closes through EOR mid — direction confirmed)
       → RETEST (30%+ bounce back toward OR mid)
       → FIRE (retest fails at 50% level with neg delta / CVD divergence)
```

### Signal Logic (v3 changes in **bold**)
1. **Build OR** (15 bars) → **Build EOR** (30 bars)
2. **Classify drive** (first 5 bars): DRIVE_UP / DRIVE_DOWN / ROTATION
3. **Detect sweep**: Closest level to EOR extreme within 17% EOR range
4. **Dual-sweep**: If both sides swept, keep deeper penetration
5. **NOT FADING filter REMOVED** — DRIVE_UP + HIGH sweep = classic Judas swing (allowed)
6. **Reversal**: Price closes through EOR mid (working mid = EOR mid on 5-min, OR mid on 1-min)
7. **Retest**: Price bounces 30%+ back toward OR mid
8. **Entry zone**: **50% retest level** = `reversal_extreme + (sweep_extreme - reversal_extreme) * 0.5`
9. **Delta/CVD**: Bar delta negative (SHORT) or CVD declining since reversal
10. **Stop: 2 × ATR(14) from entry** (was: swept level + 15% EOR range)
11. **Target**: 2R (fixed)

### Signal Rules
- **SHORT**: HIGH sweep → reversal below EOR mid → retest to 50% level → turn down + neg delta
- **LONG**: LOW sweep → reversal above EOR mid → retest to 50% level → turn up + pos delta
- **ALL drives allowed** — DRIVE_UP + HIGH sweep is the canonical Judas swing
- **One signal per session** (first valid signal only)
- **IB regime**: Skip if EOR range < 50 pts

### Why 50% Retest + 2 ATR Stop
- **Problem fixed**: Old logic entered on first bar below OR mid with neg delta → entries far from structure, wide stops (e.g. 2/06)
- **New logic**: Wait for price to bounce back to 50% of (sweep_high → reversal_low), enter on turn with neg delta
- **Result**: Entry is closer to the swept structure → tighter, more defined risk → 2 ATR stop is now reasonable

---

## OR Acceptance Signal — Logic Detail (NEW, 2026-02-25)

**Separate indicator** (`RockitORAcceptanceSignal.cs`) for a **separate chart**.
Pattern: price BREAKS a London level and HOLDS it — continuation, not reversal.

### Distinction from Judas Swing
| | Judas Swing (OR Reversal) | London Acceptance (OR Acceptance) |
|---|---|---|
| EOR vs London Low | EOR High **above** London Low (fake spike) | EOR High **below** London Low (true break) |
| Trade | Fade the spike → SHORT | Follow the break → SHORT continuation |
| Entry | On failed retest of OR mid | On 50% bounce back toward broken level |
| Stop | 2 ATR above entry | 2 ATR above entry (same) |

### Acceptance Detection
- **SHORT**: `eor_high < london_low` AND `or_low < london_low` (no spike above = true acceptance)
- **LONG**: `eor_low > london_high` AND `or_high > london_high`

### Directional Bias Score (resolves dual-level ambiguity)
When both London levels are near EOR, context determines direction:

| Signal | Bias |
|--------|------|
| Asia Low swept (`eor_low < asia_low`) | +1 bullish |
| London Low swept (`eor_low < london_low`) | +1 bullish |
| PDL held (`eor_low > pdl`) | +1 bullish |
| Asia High swept (`eor_high > asia_high`) | −1 bearish |
| London High swept (`eor_high > london_high`) | −1 bearish |
| PDH held (`eor_high < pdh`) | −1 bearish |

**Rule**: SHORT only if `bias <= 0`; LONG only if `bias >= 0`; skip if strongly opposite.

**Example**: Asia Low swept + London Low swept → bias = +2 → LONG, target London High ✅

### Entry Logic
1. EOR completes (10:00), acceptance condition confirmed, bias checked
2. `fifty_pct = ib_low + (london_low - ib_low) * 0.50` (for SHORT acceptance)
3. Scan IB bars for: price in 50% zone ± 0.5 ATR AND turning down AND neg delta
4. Invalidate if price reclaims the broken level
5. Stop: `acceptance_level + 0.5 ATR` (just above the broken level)
6. Target: 2R

### File Locations
- Python: `strategy/or_acceptance.py`
- NT8: `C:\Users\lehph\Documents\NinjaTrader 8\bin\Custom\Indicators\RockitORAcceptanceSignal.cs`
- Trace log: `Documents/NinjaTrader 8/RockitORAcceptanceTrace/`

---

## Edge Fade Signal — Logic Detail

### Levels Tracked
| Level | Color | Source |
|-------|-------|--------|
| IB High/Low | DodgerBlue (shaded rect) | First 60 RTH bars (9:30-10:30) |
| IB Mid (target) | DodgerBlue dashed | (IB High + IB Low) / 2 |
| Edge Ceiling | DarkGreen dotted | IB Low + 25% IB range |
| Entry/Stop/Target | Green/Red/Cyan | Dynamic per signal |

### Signal Logic (LONG ONLY)
1. **Build IB** (60 bars) → compute high/low/mid/range
2. **IB regime**: Skip if range < 150 pts (NORMAL+ only)
3. **IB expansion**: Skip if range < 1.2x rolling 5-day average
4. **Bearish extension**: Skip if price dropped > 30% below IBL
5. **Edge zone**: Price must be in lower 25% of IB (IBL to IBL + 25%)
6. **Delta positive**: Volumetric or candle body
7. **OF quality ≥ 2/3**: Delta + volume spike + buy imbalance
8. **R:R ≥ 1:1**: (IB mid - entry) / (entry - stop) ≥ 1.0
9. **Fire signal**: Entry @ close, Stop @ IBL - 15%, Target @ IB mid

### Signal Rules
- **LONG ONLY** — no shorts ever
- **Time window**: 10:30-13:30 only (PM morph kills MR)
- **Cooldown**: 20 bars between entries (multiple signals OK per session)
- **CVD divergence**: Price near lows + CVD rising → HIGH confidence

---

## Strategies Worth Porting (Priority Tiers)

### Tier 1 — MUST PORT (Production Ready)
| Strategy | Trades | WR | Net | Notes |
|----------|--------|-----|------|-------|
| OR Reversal | 56 | 66% | $14,030 | SHORT dominates (75% WR) |
| Edge Fade | 93 | 58% | $9,486 | LONG ONLY, highest frequency |

### Tier 2 — SHOULD PORT (Complementary)
| Strategy | Trades | WR | Net | Notes |
|----------|--------|-----|------|-------|
| Trend Bull | ~20 | ~60% | ~$3K | VWAP pullback on trend-up days |
| Trend Bear | ~15 | ~55% | ~$2K | VWAP rejection on trend-down days |
| B-Day | 7 | 71% | $1,806 | Balance day IBL fade, low frequency |

### Tier 2.5 — RESEARCH FIRST
| Strategy | Issue |
|----------|-------|
| IB Retest | BEAR-only, breakeven — needs more edge |
| Bear Accept | 11 trades, 64% WR — tiny sample |
| P-Day | Requires p-day classification, lower edge |

### Tier 3 — NICE TO HAVE
| Strategy | Issue |
|----------|-------|
| IBH Sweep+Fail | 4 trades, 100% WR — too rare |

---

## Adaptable Framework Design

All strategies share common patterns that should be abstracted:

### Shared Components
1. **IB Classifier** — Computes IB H/L/mid/range, regime (LOW/MED/NORMAL/HIGH)
2. **Session Manager** — Tracks RTH bar count, session boundaries, overnight levels
3. **Delta Reader** — Volumetric first, candle fallback
4. **Level Tracker** — London/Overnight/PDH/PDL with labeled drawing
5. **Signal State** — Entry/stop/target/direction with drawing management

### Strategy Template (for new strategies)
```csharp
public class NewStrategy : Strategy
{
    private IBClassifier _ib;
    private ORReversalSignal _orSignal;  // or any indicator

    // Managed order management
    // OnBarUpdate: read signal → enter → manage stop/target → EOD exit
    // All stop/trail/BE logic in shared base class
}
```

### Adding a New Strategy
1. Create indicator in `Indicators/` with signal logic
2. Create strategy in `Strategies/` that hosts the indicator
3. Strategy handles: entry, stop, target, trail, BE, EOD, connection loss
4. Indicator handles: level computation, zone drawing, signal detection

---

## Key Backtest Findings (DO NOT VIOLATE)

1. **Fixed targets ARE optimal** — trailing/BE regressed by 45%
2. **IB regime filter is critical** — skip LOW regime always
3. **NQ shorts lose on MR** — only sweep/momentum shorts work (OR Reversal, OR Acceptance)
4. **PM morph kills MR** — no entries after 13:30 for Edge Fade
5. **Don't cap high IB, don't skip after bad days**
6. **December = 27% of annual P&L** — don't over-optimize for it
7. **2 ATR stop + 50% retest entry** — tighter than swept-level stop, avoids deep entries (v3)
8. **NOT FADING filter removed** — DRIVE_UP + HIGH sweep IS the Judas swing, not a fade (v3)
9. **Acceptance ≠ Reversal** — separate indicator/strategy, use bias score to resolve dual-level setups

---

## File Locations

| File | Path |
|------|------|
| OR Reversal (Rockit) | `NinjaTrader 8\bin\Custom\Indicators\RockitORReversalSignal.cs` |
| OR Acceptance (Rockit) | `NinjaTrader 8\bin\Custom\Indicators\RockitORAcceptanceSignal.cs` |
| Edge Fade (Rockit) | `NinjaTrader 8\bin\Custom\Indicators\RockitVAEdgeFadeSignal.cs` |
| 80% Rule (Rockit) | `NinjaTrader 8\bin\Custom\Indicators\RockitEightyPercentSignal.cs` |
| Prior Session VA | `NinjaTrader 8\bin\Custom\Indicators\RockitPriorSessionVA.cs` |
| OR Reversal (original) | `NinjaTrader 8\bin\Custom\Indicators\ORReversalSignal.cs` |
| Edge Fade (original) | `NinjaTrader 8\bin\Custom\Indicators\EdgeFadeSignal.cs` |
| Strategy Reference | `docs\NT8-NinjaScript-Strategy-Reference.md` |
| OR Trade Design | `docs\OR-trade-design.md` |
| Acceptance Trade Design | `docs\or-acceptance-trade-design.md` (TODO) |
| Backtest Reports | `docs\reports\` |
| Python OR Reversal | `strategy\or_reversal.py` |
| Python OR Acceptance | `strategy\or_acceptance.py` |
| Python Edge Fade | `strategy\edge_fade.py` |

## Trace Log Locations

| Indicator | Log Directory |
|-----------|--------------|
| RockitORReversalSignal | `Documents\NinjaTrader 8\RockitORReversalTrace\` |
| RockitORAcceptanceSignal | `Documents\NinjaTrader 8\RockitORAcceptanceTrace\` |
