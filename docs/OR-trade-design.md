# Opening Range Reversal Trade Design

## Strategy Overview

**Name:** Opening Range Reversal (OR Rev)
**Time Window:** 9:30 - 10:00 ET (detection in first 30 bars of RTH)
**Direction:** LONG and SHORT
**Performance:** 56 trades, 66.1% WR, $14,030 net, PF 4.61, MaxDD $795 (259 sessions)
**SHORT dominates:** 75% WR, $10,751 (32 trades) vs LONG 54% WR, $3,279 (24 trades)

## Concept: ICT "Judas Swing"

In the first 30 minutes of RTH, price makes a false move to sweep pre-market liquidity
(overnight H/L, PDH/PDL, Asia H/L), then reverses. We enter on the reversal after the
sweep extreme, when price crosses back through the OR midpoint with delta confirmation.

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

### Phase 3: Sweep Detection

Check if EOR extreme is near a key level:

```python
SWEEP_THRESHOLD = eor_range * 0.17  # ~17% of EOR range

# Key levels to check
high_levels = [overnight_high, pdh, asia_high]  # filter None
low_levels  = [overnight_low, pdl, asia_low]     # filter None

swept_high = any(abs(eor_high - lvl) < SWEEP_THRESHOLD for lvl in high_levels)
swept_low  = any(abs(eor_low  - lvl) < SWEEP_THRESHOLD for lvl in low_levels)
```

### Phase 4: Reversal Confirmation

After the sweep extreme bar, scan for reversal:

```python
# SHORT SETUP (Judas swing UP then reversal DOWN)
if swept_high and opening_drive != 'DRIVE_UP':  # NOT fading
    # Find the bar that made the EOR high
    high_bar_idx = eor_bars['high'].argmax()
    # Scan bars after the high for reversal
    for bar in session_bars[high_bar_idx+1 : high_bar_idx+31]:
        if bar['close'] < or_mid:                                    # Reversed below OR mid
            if abs(bar['close'] - bar['vwap']) < eor_range * 0.17:   # VWAP aligned
                if bar['delta'] < 0:                                  # Seller confirmation
                    # SIGNAL: SHORT
                    stop   = eor_high + eor_range * 0.15
                    risk   = stop - bar['close']
                    target = bar['close'] - 2 * risk   # 2R target
                    break

# LONG SETUP (Judas swing DOWN then reversal UP)
if swept_low and opening_drive != 'DRIVE_DOWN':  # NOT fading
    low_bar_idx = eor_bars['low'].argmin()
    for bar in session_bars[low_bar_idx+1 : low_bar_idx+31]:
        if bar['close'] > or_mid:                                    # Reversed above OR mid
            if abs(bar['close'] - bar['vwap']) < eor_range * 0.17:   # VWAP aligned
                if bar['delta'] > 0:                                  # Buyer confirmation
                    # SIGNAL: LONG
                    stop   = eor_low - eor_range * 0.15
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

## IB Regime Filter

Only trade when IB range is MED, NORMAL, or HIGH:
- **LOW (IB < 100 pts):** Skip -- no clear directional commitment
- **MED (100-150 pts):** Trade
- **NORMAL (150-250 pts):** Trade
- **HIGH (>= 250 pts):** Trade

## Exit Rules

| Exit | Condition | Priority |
|------|-----------|----------|
| **Target** | 2R from entry | Primary exit (33.8% of trades) |
| **Stop** | Sweep extreme + 15% EOR range | Hard stop (25.8%) |
| **VWAP Breach PM** | Price crosses VWAP after 13:00 | Afternoon reversal protection (18.8%) |
| **EOD** | 15:59 ET | Flat by close (21.6%) |

## What the LLM Should Look For in the JSON Snapshot

When `current_et_time` is between 09:45 and 10:00, check:

1. **Premarket sweep setup:** Did the EOR high sweep `overnight_high`, `pdh`, or `asia_high`?
   Or did the EOR low sweep `overnight_low`, `pdl`, or `asia_low`?
2. **Opening drive classification:** Was the first 5 minutes a strong drive (>40% directional)
   or rotation? Do NOT short a strong drive-up. Do NOT long a strong drive-down.
3. **Reversal confirmation:** Is price now on the other side of OR midpoint?
4. **VWAP alignment:** Is price near VWAP (within ~17% of EOR range)?
5. **Delta/order flow:** Is delta confirming the reversal direction?

## Sample JSON Snapshot Fields Used

```json
{
  "premarket": {
    "overnight_high": 25500.0,
    "overnight_low": 25300.0,
    "pdh": 25520.0,
    "pdl": 25280.0,
    "asia_high": 25480.0,
    "asia_low": 25320.0
  },
  "intraday": {
    "ib": {
      "ib_high": 25450.0,
      "ib_low": 25350.0,
      "ib_range": 100.0,
      "vwap": 25400.0,
      "atr_regime": "normal"
    }
  }
}
```

## Source File Reference

Full implementation: `strategy/or_reversal.py` (284 lines) in BookMapOrderFlowStudies-2 repo.
