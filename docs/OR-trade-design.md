# Opening Range Reversal Trade Design
**Updated:** 2026-02-24 (NT8 sync v2)

## Strategy Overview

**Name:** Opening Range Reversal (OR Rev)
**Time Window:** 9:30 - 10:00 ET (detection in first 30 bars of RTH)
**Direction:** LONG and SHORT
**Performance:** 56 trades, 67.9% WR, $15,275 net, PF 5.18, Sharpe 11.08, MaxDD $922 (260 sessions)
**SHORT dominates:** ~75% WR vs LONG ~54% WR
**Expectancy:** $273/trade

## Concept: ICT "Judas Swing"

In the first 30 minutes of RTH, price makes a false move to sweep pre-market liquidity
(overnight H/L, PDH/PDL, Asia H/L, **London H/L**), then reverses. We enter on the reversal
after the sweep extreme, when price crosses back through the OR midpoint with delta/CVD confirmation.

**Key insight:** Do NOT fade the opening drive. If the first 5 bars drive UP strongly,
do NOT short (that's a fade = 40% WR). Only take reversals that go WITH the drive or
occur in rotational opens.

## Required Data from Rockit Modules

From the JSON snapshot, the strategy needs:

```json
{
  "premarket": {
    "overnight_high": 25500.0,
    "overnight_low": 25300.0,
    "asia_high": 25480.0,
    "asia_low": 25320.0,
    "london_high": 25490.0,
    "london_low": 25340.0,
    "pdh": 25520.0,
    "pdl": 25280.0
  },
  "intraday": {
    "ib": {
      "ib_high": ...,
      "ib_low": ...,
      "current_close": ...,
      "vwap": ...
    }
  }
}
```

Plus 1-minute OHLCV+delta bars for the first 30 minutes of RTH.

## Detection Logic

### Phase 1: Compute OR and EOR

```python
OR_BARS = 15    # First 15 bars (9:30-9:44) = Opening Range
EOR_BARS = 30   # First 30 bars (9:30-9:59) = Extended Opening Range

or_bars = session_bars[:OR_BARS]
or_high = or_bars['high'].max()
or_low  = or_bars['low'].min()
or_mid  = (or_high + or_low) / 2

eor_bars = session_bars[:EOR_BARS]
eor_high  = eor_bars['high'].max()
eor_low   = eor_bars['low'].min()
eor_range = eor_high - eor_low
```

### Phase 2: Classify Opening Drive (first 5 bars)

```python
DRIVE_THRESHOLD = 0.4  # 40% of the first-5-bar range

first_5 = session_bars[:5]
open_price = first_5[0]['open']
close_5th  = first_5[4]['close']
drive_range = first_5['high'].max() - first_5['low'].min()
drive_pct = (close_5th - open_price) / drive_range if drive_range > 0 else 0

if drive_pct > 0.4:
    opening_drive = 'DRIVE_UP'
elif drive_pct < -0.4:
    opening_drive = 'DRIVE_DOWN'
else:
    opening_drive = 'ROTATION'
```

### Phase 3: Sweep Detection (Proximity-Based, Pick CLOSEST)

Check if EOR extreme is near a key level. **Pick the CLOSEST match** (not first match).
Cap max distance at EOR range to prevent false sweeps of distant levels.

```python
SWEEP_THRESHOLD = eor_range * 0.17  # ~17% of EOR range

# Key levels to check (NEW: includes London H/L)
high_candidates = [
    ('ON_HIGH', overnight_high), ('PDH', pdh),
    ('ASIA_HIGH', asia_high), ('LDN_HIGH', london_high)
]
low_candidates = [
    ('ON_LOW', overnight_low), ('PDL', pdl),
    ('ASIA_LOW', asia_low), ('LDN_LOW', london_low)
]

# Find closest swept level within threshold AND within EOR range
def find_closest(eor_extreme, candidates):
    best = None
    for name, lvl in candidates:
        if lvl is None: continue
        dist = abs(eor_extreme - lvl)
        if dist < SWEEP_THRESHOLD and dist <= eor_range:
            if best is None or dist < best[1]:
                best = (lvl, dist, name)
    return best

swept_high = find_closest(eor_high, high_candidates)
swept_low  = find_closest(eor_low, low_candidates)
```

### Phase 3b: Dual-Sweep Depth Comparison

If BOTH high and low are swept, keep only the **deeper** penetration:

```python
if swept_high and swept_low:
    high_depth = eor_high - swept_high.level
    low_depth  = swept_low.level - eor_low
    if high_depth >= low_depth:
        swept_low = None   # HIGH sweep wins
    else:
        swept_high = None  # LOW sweep wins
```

### Phase 4: Reversal Confirmation (Delta OR CVD)

After the sweep extreme bar, scan for reversal. **Accept bar delta OR CVD divergence**
(some valid entries have neutral bar delta but clear CVD trend).

```python
# Compute running CVD from bar 0 to current bar
cvd_series = ib_bars['delta'].fillna(0).cumsum()

# SHORT SETUP (Judas swing UP then reversal DOWN)
if swept_high and opening_drive != 'DRIVE_UP':  # NOT fading
    high_bar_idx = eor_bars['high'].argmax()
    cvd_at_extreme = cvd_series[high_bar_idx]

    for j, bar in enumerate(session_bars[high_bar_idx+1 : high_bar_idx+31]):
        if bar['close'] < or_mid:                                    # Reversed below OR mid
            if abs(bar['close'] - bar['vwap']) < eor_range * 0.17:   # VWAP aligned
                cvd_at_entry = cvd_series[high_bar_idx + j + 1]
                cvd_declining = cvd_at_entry < cvd_at_extreme
                if bar['delta'] < 0 or cvd_declining:                # Delta OR CVD
                    # SIGNAL: SHORT
                    stop   = swept_high.level + eor_range * 0.15  # Stop at SWEPT LEVEL
                    risk   = stop - bar['close']
                    target = bar['close'] - 2 * risk   # 2R target
                    break

# LONG SETUP (Judas swing DOWN then reversal UP)
if swept_low and opening_drive != 'DRIVE_DOWN':  # NOT fading
    low_bar_idx = eor_bars['low'].argmin()
    cvd_at_extreme = cvd_series[low_bar_idx]

    for j, bar in enumerate(session_bars[low_bar_idx+1 : low_bar_idx+31]):
        if bar['close'] > or_mid:                                    # Reversed above OR mid
            if abs(bar['close'] - bar['vwap']) < eor_range * 0.17:   # VWAP aligned
                cvd_at_entry = cvd_series[low_bar_idx + j + 1]
                cvd_rising = cvd_at_entry > cvd_at_extreme
                if bar['delta'] > 0 or cvd_rising:                   # Delta OR CVD
                    # SIGNAL: LONG
                    stop   = swept_low.level - eor_range * 0.15  # Stop at SWEPT LEVEL
                    risk   = bar['close'] - stop
                    target = bar['close'] + 2 * risk   # 2R target
                    break
```

### Phase 5: Risk Management

```python
MIN_RISK = eor_range * 0.03   # Minimum risk (too tight = noise stop)
MAX_RISK = eor_range * 1.3    # Maximum risk (too wide = bad R:R)

if risk < MIN_RISK or risk > MAX_RISK:
    skip  # Invalid risk parameters
```

### ATR Stop Alternative (for personal accounts)

2.5 ATR stop tested at +35% P&L vs swept level stop, but higher DD.
**Recommended for personal accounts with $1M+ balance only.**

```python
# Alternative: ATR-based stop
atr_stop = ATR(14) * 2.5
stop = entry + atr_stop  # SHORT
stop = entry - atr_stop  # LONG
# Still target 2R from this wider stop
```

## IB Regime Filter

Only trade when IB range is MED, NORMAL, or HIGH:
- **LOW (IB < 100 pts):** Skip -- no clear directional commitment
- **MED (100-150 pts):** Trade
- **NORMAL (150-250 pts):** Trade
- **HIGH (>= 250 pts):** Trade

## Exit Rules

| Exit | Condition | Priority |
|------|-----------|----------|
| **Target** | 2R from entry | Primary exit |
| **Stop** | Swept level + 15% EOR range | Hard stop |
| **VWAP Breach PM** | Price crosses VWAP after 13:00 | Afternoon reversal protection |
| **EOD** | 15:59 ET | Flat by close |

## Changes from v1 (2026-02-24 NT8 Sync)

| Change | Impact |
|--------|--------|
| Added London H/L to sweep candidates | +15 unfiltered trades |
| Proximity-based sweep (closest match) | Prevents false 400pt "sweeps" |
| Dual-sweep depth comparison | Picks correct side when both swept |
| Stop at swept level (not EOR extreme) | More logical stop placement |
| CVD divergence as delta alternative | +2 trades, +$1K net (best single change) |
| ~~EOR mid entry zone~~ | **REVERTED**: -$7K on 1-min bars |

## Source File Reference

Full implementation: `strategy/or_reversal.py` in BookMapOrderFlowStudies-2 repo.
NT8 indicator: `Indicators/ORReversalSignal.cs` in NinjaTrader 8 Custom folder.
