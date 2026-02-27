# Edge Fade Trade Design (Edge-to-Mid Mean Reversion)

## Strategy Overview

**Name:** Edge Fade
**Time Window:** 10:00 - 13:30 ET (post-OR, before PM morph)
**Direction:** LONG ONLY (all NQ shorts are negative expectancy in this setup)
**Performance:** 93 trades, 58.1% WR, $9,486 net, PF 1.75 (259 sessions)
**Day Types:** b_day, neutral (balance/range days)

## Concept: IB Edge Mean Reversion

On balance days, when price visits the lower 25% of the IB range with buyer confirmation,
fade back toward the IB midpoint. This captures the natural oscillation of range-bound days
where the IB acts as a "gravity well."

**Key insight:** This is LONG ONLY. All short-side mean reversion on NQ loses money.
Only trade when IB is wider than the recent 5-day average (expansion = overextension =
better mean reversion). Narrow IB = choppy/low vol = poor reversion.

## Required Data from Rockit Modules

From the JSON snapshot:

```json
{
  "intraday": {
    "ib": {
      "ib_high": 25450.0,
      "ib_low": 25350.0,
      "ib_range": 100.0,
      "ib_mid": 25400.0,
      "vwap": 25400.0,
      "price_location": "lower_third"
    },
    "volume_profile": {
      "poc": 25405.0,
      "vah": 25440.0,
      "val": 25360.0
    },
    "fvg_detection": {
      "fvg_bull_zones": [{"bottom": 25355.0, "top": 25370.0}]
    },
    "dpoc_migration": {
      "regime": "balancing_choppy"
    }
  },
  "core_confluences": {
    "day_type_classification": "b_day"
  }
}
```

## Detection Logic

### Phase 1: Session Setup (at IB end, ~10:30)

```python
COOLDOWN_BARS = 20          # Bars between entries
IB_EXPANSION_RATIO = 1.2   # IB must be >= 1.2x recent 5-day avg
LAST_ENTRY_TIME = "13:30"  # No entries after 13:30 (PM morph kills MR)
MAX_BEARISH_EXT = 0.30     # Max extension below IBL (bearish days = 37% WR)

# Check IB expansion (wider IB = overextended = good MR)
ib_history = last_5_sessions_ib_ranges  # [120, 150, 130, 140, 160]
avg_ib = mean(ib_history)
ib_ratio = current_ib_range / avg_ib

if ib_ratio < 1.2:
    skip_session()  # IB not expanded enough, choppy session
```

### Phase 2: Entry Zone Detection (per bar)

```python
EDGE_ZONE_PCT = 0.25   # Lower 25% of IB = entry zone
EDGE_STOP_BUFFER = 0.15  # Stop: IBL - 15% of IB range
MIN_RR = 1.0            # Minimum 1:1 reward/risk

edge_ceiling = ib_low + ib_range * 0.25

# Price must be in the lower 25% of IB (the "edge")
if ib_low < close < edge_ceiling:
    # Entry zone confirmed
    pass
```

### Phase 3: Order Flow Confirmation

```python
# Delta must be positive (buyers present at the edge)
if bar['delta'] <= 0:
    skip()

# OF quality: 2-of-3 bullish signals required
of_quality = sum([
    delta_percentile >= 60,    # Delta above 60th percentile
    imbalance_ratio > 1.0,     # Buy imbalance
    volume_spike >= 1.0,       # Volume at/above normal
])
if of_quality < 2:
    skip()

# Bearish extension filter: if price extended >30% below IBL, skip
# (these are bearish days, MR back to mid only works 37% of the time)
max_ext_below_ibl = (ib_low - session_low) / ib_range
if max_ext_below_ibl >= 0.30:
    skip()
```

### Phase 4: Entry Confirmation Models (boost confidence)

These are optional boosters. When present, confidence = 'high' (larger position).

```python
# 1. CVD Divergence: price near lows but CVD rising
#    57.1% WR (vs 33.3% baseline), +$14/trade edge
def check_cvd_divergence(price_history_10bars, cvd_history_10bars):
    price_position = (close - min_low) / (max_high - min_low)
    cvd_rising = cvd[-1] > cvd[-3]  # CVD slope positive
    return price_position < 0.40 and cvd_rising

# 2. FVG Confluence: entry inside a bullish Fair Value Gap
#    57.1% WR, +$13/trade edge
def check_fvg_confluence(bar):
    return bar['fvg_bull'] == True  # From fvg_detection module

# 3. 5-min Inversion: prior 5-min candle bearish, current bullish
#    45.5% WR, +$5.5/trade edge
def check_5m_inversion(candles_5m):
    prior = candles_5m[-2]  # bearish: close < open
    current = candles_5m[-1]  # bullish: close > open
    return prior['close'] < prior['open'] and current['close'] > current['open']

has_confirmation = any([cvd_divergence, fvg_confluence, inversion_5m])
confidence = 'high' if has_confirmation else 'medium'
```

### Phase 5: Signal Generation

```python
stop   = ib_low - ib_range * 0.15    # Below IBL with buffer
target = ib_mid                       # Target = IB midpoint

risk   = close - stop
reward = target - close

if reward / risk >= 1.0:  # Minimum 1:1 R:R
    signal = {
        'direction': 'LONG',
        'entry': close,
        'stop': stop,
        'target': target,
        'setup_type': 'EDGE_TO_MID',
        'confidence': confidence,
    }
```

## IB Regime Filter

Only trade when IB range is NORMAL or HIGH:
- **LOW (IB < 100 pts):** Skip -- insufficient range for mean reversion
- **MED (100-150 pts):** Skip -- loses money in this regime
- **NORMAL (150-250 pts):** Trade
- **HIGH (>= 250 pts):** Trade

## Exit Rules

| Exit | Condition | Priority |
|------|-----------|----------|
| **Target** | IB midpoint | Primary exit |
| **Stop** | IBL - 15% IB range | Hard stop |
| **VWAP Breach PM** | Price crosses VWAP after 13:00 | EXEMPT for Edge Fade (MR trades oscillate near VWAP) |
| **EOD** | 15:59 ET | Flat by close |

## What the LLM Should Look For in the JSON Snapshot

When `current_et_time` is between 10:30 and 13:30, and day type is `b_day` or `neutral`:

1. **Price location:** Is price in the lower third of IB? (`price_location: "lower_third"`)
2. **IB expansion:** Is today's IB range wider than recent average? (Check `ib_range` vs historical)
3. **Order flow:** Is delta positive? Are buyers stepping in at the edge?
4. **Bearish extension:** Has price NOT dropped more than 30% below IBL (that's a breakdown, not MR)
5. **Confluence boosters:** Is there a bullish FVG zone at this price? CVD divergence? 5-min inversion?
6. **DPOC regime:** `balancing_choppy` or `stabilizing_hold_forming_floor` = good for MR.
   `trending_on_the_move` = bad for MR.

## DISABLED: Failed Breakdown Entry

The "failed breakdown" variant (price breaks below IBL, then reclaims above) was tested
and shows 62% WR in simple sim but the engine's VWAP breach PM exit kills profitability
(-$987 net, 45% WR in real backtest). Disabled until VWAP breach PM handling is adjusted.

## Sample JSON Snapshot Fields Used

```json
{
  "intraday": {
    "ib": {
      "ib_high": 25450.0,
      "ib_low": 25350.0,
      "ib_range": 100.0,
      "ib_mid": 25400.0,
      "vwap": 25400.0,
      "price_location": "lower_third",
      "atr_regime": "normal"
    },
    "wick_parade": {
      "bullish_count": 3,
      "bearish_count": 1
    },
    "dpoc_migration": {
      "regime": "balancing_choppy",
      "dpoc_current": 25405.0
    },
    "volume_profile": {
      "poc": 25405.0,
      "vah": 25440.0,
      "val": 25360.0
    },
    "fvg_detection": {
      "fvg_bull_zones": [{"bottom": 25355.0, "top": 25370.0}]
    }
  },
  "core_confluences": {
    "day_type_classification": "b_day",
    "ib_acceptance": false,
    "price_location": "lower_third_hug"
  }
}
```

## Source File Reference

Full implementation: `strategy/edge_fade.py` (516 lines) in BookMapOrderFlowStudies-2 repo.
