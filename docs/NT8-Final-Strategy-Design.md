# NinjaTrader 8 -- Final Strategy Design Document
**Date:** 2026-02-24 | **Version:** 3.0 | **Author:** Claude + Manual NT8 Testing

## Strategy Portfolio Overview

Five strategies organized in two tiers:

### Tier 1: Production (259-session backtest validated)

| Strategy | Trades/Mo | WR | Monthly P&L | PF | Max DD |
|----------|-----------|-----|------------|-----|--------|
| OR Reversal | 4.3 | 67.9% | $1,175 | 5.18 | $922 |
| Edge Fade | 7.2 | 58.1% | $730 | 1.75 | $1,854 |
| **Tier 1 Total** | **11.5** | **61.7%** | **$1,905** | -- | **$2,323** |

### Tier 2: New (Research-validated, needs Python backtest)

| Strategy | Trades/Mo | WR | Monthly P&L | PF | Max DD |
|----------|-----------|-----|------------|-----|--------|
| 80P Rule | 2.2 | 65.7% | $1,922* | 3.45 | ~$1,200 |
| 20P Rule | 3.5 | 57.0% | $496* | 1.78 | ~$800 |
| VA Edge Fade | 8.0 | 72.4% | $533-1,309* | 5.38 | ~$600 |

*\*Research estimates from separate backtest system; needs validation on unified engine*

---

## Strategy 1: Opening Range Reversal (OR Rev)

### Concept
ICT "Judas Swing" -- price sweeps pre-market liquidity in the first 30 minutes,
then reverses. We enter the reversal.

### Performance (NT8-synced, 2026-02-24)
- 56 trades, 67.9% WR, $15,275 net, PF 5.18, Sharpe 11.08
- SHORT: ~75% WR (dominant) | LONG: ~54% WR
- Expectancy: $273/trade

### Entry Rules

**Time Window:** 9:30 - 10:00 ET (scan first 30 1-min bars)

1. **Compute OR/EOR:**
   - OR = bars 0-14 (9:30-9:44): OR High, OR Low, OR Mid
   - EOR = bars 0-29 (9:30-9:59): EOR High, EOR Low, EOR Range

2. **Sweep Detection (proximity-based, pick CLOSEST):**
   - Candidates: Overnight H/L, PDH/PDL, Asia H/L, **London H/L**
   - Match: `abs(EOR extreme - candidate) < EOR_range * 0.17`
   - Max distance cap: `distance <= EOR_range` (reject 400pt "sweeps")
   - If BOTH high and low swept: keep the **deeper** penetration only

3. **Opening Drive Filter (NOT fading):**
   - Classify first 5 bars: DRIVE_UP / DRIVE_DOWN / ROTATION
   - DRIVE_UP + SHORT = FADE -> **SKIP** (40% WR)
   - DRIVE_DOWN + LONG = FADE -> **SKIP**
   - All other combos = allowed

4. **Reversal Confirmation:**
   - Scan bars after EOR extreme for close beyond **OR midpoint**
   - VWAP alignment: `abs(close - VWAP) < EOR_range * 0.17`
   - Delta OR CVD: bar delta < 0 (SHORT) **OR** CVD declining from extreme

5. **Entry:** Close of confirming bar

### Stop & Target

| Component | SHORT | LONG |
|-----------|-------|------|
| **Stop** | Swept level + EOR_range * 15% | Swept level - EOR_range * 15% |
| **Target** | Entry - 2R | Entry + 2R |
| **Risk** | stop - entry | entry - stop |

**Note on ATR stops:** 2.5 ATR stop tested at +35% P&L vs current stop, but higher DD.
Recommended for personal accounts only. Prop firms should use swept level stop (lowest DD).

### Regime Filter
- Skip LOW IB (< 100 pts)
- Allow: MED (100-150), NORMAL (150-250), HIGH (250+)

### NT8 Indicator: ORReversalSignal.cs
- Shaded OR zone rectangle, labeled OR High/Low/Mid levels
- London H/L (yellow), Overnight H/L (orange), PDH/PDL (magenta)
- Sweep detection arrows with swept level label
- Entry/Stop/Target thick labeled lines (gold/red/green)
- Debug mode shows skip reason when no signal fires

---

## Strategy 2: Edge Fade (IB Edge Mean Reversion)

### Concept
On balance days, price visits IB lower edge with buyer confirmation, fade back to IB mid.

### Performance (unchanged)
- 93 trades, 58.1% WR, $9,486 net, PF 1.75
- LONG ONLY (short MR on NQ = negative expectancy)

### Entry Rules

**Time Window:** 10:30 - 13:30 ET (no entries after 13:30 -- PM morph)

1. **IB Expansion:** IB range >= 1.2x 5-day rolling average
2. **Price Location:** Close in lower 25% of IB range
3. **Order Flow Quality (2 of 3):**
   - Bar delta > 0
   - Volume spike >= 1.0x average
   - Buy imbalance detected
4. **CVD Divergence:** Optional HIGH confidence upgrade
5. **Entry:** Close of qualifying bar

### Stop & Target

| Component | Value |
|-----------|-------|
| **Stop** | IBL - IB_range * 15% |
| **Target** | IB midpoint (~1.6R) |

### Regime Filter
- Skip LOW (< 100) and MED (100-150) IB
- Allow: NORMAL (150-250), HIGH (250+)

### NT8 Indicator: EdgeFadeSignal.cs
- Shaded IB zone, labeled IB Mid (TARGET), Edge Ceiling
- Signal arrows + entry/stop/target lines
- Works on volumetric OR regular 1-min charts

---

## Strategy 3: 80P Rule (Value Area Acceptance)

### Concept
80% Rule: when price opens outside prior session's Value Area, fails to break away,
and re-enters the VA, there's an 80% probability it traverses the full VA.

### Entry Rules

**Time Window:** 9:30 - 13:00 ET (hard cutoff)

**Prerequisite:** Price opens OUTSIDE prior ETH Value Area (above VAH or below VAL)

1. **Entry Model -- 100% Retest Double Top/Bottom (Recommended):**
   - Price pokes outside VA, re-enters, pokes out again (double top/bottom)
   - Enter on the second rejection (limit order at VA edge)
   - **WR: 65.7%, PF 3.45** (best risk-adjusted)

2. **Alternative -- Limit 50% VA Depth:**
   - Price re-enters VA, set limit at 50% depth into VA
   - **$1,922/mo** (best absolute P&L)

3. **Alternative -- Acceptance Close:**
   - Wait for candle to close inside VA
   - Baseline model, lower WR

### Stop & Target

| Component | Value |
|-----------|-------|
| **Stop** | VA edge + 10pt buffer (NEVER widen -- widens = destroys P&L) |
| **Target** | 4R (optimal based on extensive testing) |

### Critical Rules
- Use **ETH** Value Area (not RTH) -- ETH captures overnight liquidity
- **Candle-based stops on retest entries = catastrophic** (5-14% WR, avoid!)
- No 0.5 ATR trail (universally fails)
- Hard 13:00 entry cutoff

### NT8 Indicator: EightyPercentSignal.cs (TO BUILD)

**Required components:**
- Prior session ETH Value Area overlay (VAH / VAL / POC lines)
- Open location detection (above/below VA)
- Re-entry detection (first close inside VA from outside)
- Double top/bottom pattern recognition at VA edge
- Limit order placement zone (50% VA depth)
- Entry arrow + stop/target lines

**Shared indicators needed:**
- `PriorSessionVA.cs` -- computes prior session ETH Value Area
- `IBTracker.cs` -- tracks current IB High/Low/Range

---

## Strategy 4: 20P Rule (IB Extension Breakout)

### Concept
20% Rule: when IB sets up strongly, price extends beyond IB with conviction.
Confirmed by 3 consecutive 5-min closes beyond IB boundary.

### Entry Rules

**Time Window:** 10:30 - 13:00 ET

1. **IB Extension:** Price breaks above IBH (LONG) or below IBL (SHORT)
2. **Confirmation:** 3 consecutive 5-min closes beyond the IB boundary
3. **Entry:** Close of 3rd confirming 5-min bar

### Stop & Target

| Component | Value |
|-----------|-------|
| **Stop** | 2x ATR(14) from entry (median risk ~32 pts) |
| **Target** | 2R |

### Performance
- ~$496/mo, PF 1.78 (weakest of the new strategies but additive)

### NT8 Indicator: TwentyPercentSignal.cs (TO BUILD)

**Required components:**
- IB boundary lines (IBH / IBL)
- 5-min secondary bar series for close confirmation counting
- ATR(14) for stop calculation
- Breakout arrow after 3rd confirming close
- Entry/Stop/Target lines

---

## Strategy 5: VA Edge Fade (Value Area Edge Mean Reversion)

### Concept
Price pokes outside VA edge, fails, fades back in. Similar to Edge Fade but uses
prior session Value Area instead of current IB.

### Entry Rules

**Time Window:** 9:30 - 13:00 ET

**Models (best to worst):**

1. **Limit at sweep extreme (1st touch):**
   - Price sweeps beyond VA edge, place limit at the extreme
   - WR: 72.4%, PF 5.38, $533/mo

2. **Limit at VA edge + 2 ATR + POC target (2nd test):**
   - On second test of VA edge, limit at edge, target POC
   - $1,309/mo (highest absolute P&L)

3. **2x 5-min close inside VA:**
   - Wait for 2 consecutive 5-min closes inside VA from outside
   - $1,381/mo combined

4. **Inversion candle + Swing + 4R (1st touch):**
   - Bearish/bullish engulfing at VA edge
   - $1,453/mo

### Stop & Target

| Component | Value |
|-----------|-------|
| **Stop** | 2x ATR(14) from entry |
| **Target** | Varies by model: POC, 1R, 2R, or 4R |

### Critical Rules
- Same as 80P: ETH Value Area only, no candle stops on retests, no 0.5 ATR trail

### NT8 Indicator: VAEdgeFadeSignal.cs (TO BUILD)

**Required components:**
- Prior session ETH VA overlay (shared with 80P)
- VA edge touch/sweep detection
- Inversion candle pattern recognition
- 5-min secondary series for close counting
- Multiple entry model support (limit vs market)
- Entry/Stop/Target lines with model label

---

## Shared Indicators (TO BUILD)

### PriorSessionVA.cs
- Computes prior session **ETH** (18:00 - 16:00) Value Area
- Outputs: VAH, VAL, POC
- Plots as horizontal lines/shaded zone on current session
- Used by: 80P, VA Edge Fade

### IBTracker.cs
- Tracks IB High, IB Low, IB Range, IB Mid in real-time
- Detects IB extension (price beyond IBH/IBL)
- Computes 5-day rolling IB average for expansion check
- Used by: 20P, Edge Fade

### VAEdgeDetector.cs
- Detects price interaction with VA edges (touch, sweep, re-entry)
- Tracks open location relative to VA
- Used by: 80P, VA Edge Fade

---

## Portfolio Combinations

### Option 1: Maximum Monthly P&L
**OR Rev + Edge Fade + 80P + VA Edge + 20P**
- Est. 30+ trades/mo, ~$3,700-4,500/mo at 5 MNQ
- Highest complexity, most indicators to monitor

### Option 2: High Confidence (Conservative)
**OR Rev + Edge Fade + 80P Retest**
- ~16 trades/mo, ~$3,000/mo at 5 MNQ
- Lowest DD, highest combined PF

### Option 3: Current Production + Easy Add
**OR Rev + Edge Fade + 20P**
- ~15 trades/mo, ~$2,400/mo
- 20P shares IB infrastructure with Edge Fade, easiest to add

---

## Implementation Priority

1. **PriorSessionVA.cs** -- shared dependency for 80P + VA Edge
2. **EightyPercentSignal.cs** -- highest per-trade P&L
3. **VAEdgeFadeSignal.cs** -- fills trade frequency gap
4. **TwentyPercentSignal.cs** -- additive, lower priority
5. **IBTracker.cs** -- may already be partially covered by existing indicators

---

## Key Constants

```
# IB Regime Buckets (NQ/MNQ)
LOW:    < 100 pts   (skip for OR Rev, Edge Fade)
MED:    100-150 pts  (skip for Edge Fade)
NORMAL: 150-250 pts
HIGH:   >= 250 pts

# Sweep Detection
SWEEP_THRESHOLD = EOR_range * 0.17
MAX_SWEEP_DISTANCE = EOR_range

# Stops
OR_STOP_BUFFER = 15% of EOR range (above/below swept level)
EDGE_FADE_STOP = IBL - 15% IB range
80P_STOP = VA edge + 10pt (NEVER widen)
20P_STOP = 2x ATR(14)
VA_EDGE_STOP = 2x ATR(14)

# Targets
OR_TARGET = 2R (fixed, optimal)
EDGE_FADE_TARGET = IB midpoint (~1.6R)
80P_TARGET = 4R (optimal based on testing)
20P_TARGET = 2R
VA_EDGE_TARGET = varies (POC, 1R, 2R, or 4R)

# Time Cutoffs
OR_WINDOW = 9:30-10:00
EDGE_FADE_WINDOW = 10:30-13:30
80P_CUTOFF = 13:00
20P_WINDOW = 10:30-13:00
VA_EDGE_CUTOFF = 13:00
```
