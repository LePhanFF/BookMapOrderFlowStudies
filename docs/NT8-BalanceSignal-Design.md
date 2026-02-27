# RockitBalanceSignal — Design Document

## Overview

Quality-filtered mean-reversion strategy for balance/inside/neutral days. Uses a 6-factor scoring system to separate high-quality setups from mechanical noise.

**Two entry modes:**
1. **VA Edge Fade** — price touches prior VAH/VAL, fades back inside with Dalton acceptance
2. **Wide IB Reclaim** — IB range 350-500pt, price breaks IB edge then reclaims back inside

## Python Strategy: `strategy/balance_signal.py`

### Session Context Requirements

| Field | Source | Purpose |
|-------|--------|---------|
| `prior_va_high/low/poc` or `prior_vp_vah/val/poc` | engine/backtest.py | Prior session VA levels |
| `prior_tpo_shape` | TPOProfileAdapter | Profile shape (P/b/D) for F1 |
| `tpo_poc_location` | TPOProfileAdapter | POC position ratio (0-1) |
| `atr14` | Feature column | Volatility measure |
| `vwap` | Session context (updated per bar) | VWAP for F4 |
| `day_type`, `trend_strength` | Engine classifier | Signal metadata |
| `bar_time` | Engine | Time gate |
| `vp_data` | VolumeProfileAdapter | Volume bins for F2/F3 |
| `ib_high/low/range` | on_session_start params | IB levels |

### 6-Factor Scoring System (0-10 points)

#### F1: Profile Shape (0-2 pts)
- P-shape (POC upper third) → +2 LONG, 0 SHORT
- b-shape (POC lower third) → +2 SHORT, 0 LONG
- D-shape (centered) → +1 either
- Fallback: `tpo_poc_location` ratio

#### F2: HVN/LVN at VA Edge (0-2 pts)
- HVN at faded edge → +2 (strong support/resistance)
- Neutral volume → +1
- LVN at edge → +0 (weak level, likely to break)
- Uses prior session volume bins (80th/20th percentile thresholds)

#### F3: Volume Distribution Skew (0-1 pt)
- More volume below POC + fading VAL = sellers exhausted → +1 LONG
- More volume above POC + fading VAH = buyers exhausted → +1 SHORT
- Threshold: 55% of total volume on exhaustion side

#### F4: VWAP Confirmation (0-1 pt)
- LONG: price ≤ VWAP, VWAP slope flat/rising → +1
- SHORT: price ≥ VWAP, VWAP slope flat/falling → +1

#### F5: IB Containment (0-2 pts)
- IB fully inside prior VA → +1
- IB narrow (< 50% VA range) → +1

#### F6: Delta/CVD Divergence (0-2 pts)
- **Absorption**: delta opposes direction but price stable → +1
- **CVD divergence**: CVD trend disagrees with 5-bar price trend → +1

### Minimum Score: 5 (configurable via `BALANCE_MIN_QUALITY_SCORE`)

### Mode 1: VA Edge Fade

```
Detection:
  1. Price high >= prior_vah (touch detected)
  2. Price close < prior_vah (faded back inside)
  3. acceptance_below_vah >= 2 (Dalton 2x1min early acceptance)
  4. score >= MIN_QUALITY_SCORE
  → Emit SHORT signal

  Same logic inverted for VAL → LONG

Stop: VAH + 15% VA range (SHORT) / VAL - 15% VA range (LONG)
Target: Dynamic (see below)
```

### Mode 2: Wide IB Reclaim

```
Gate: IB range 350-500pt only

Detection:
  1. prev_close < ib_low AND current_close > ib_low (reclaim)
  2. acceptance_above_ibl >= 2
  3. score >= MIN_QUALITY_SCORE
  4. F2 (HVN/LVN) >= 1 (HVN at edge required)
  → Emit LONG signal

  Same logic inverted for IBH → SHORT

Stop: IBL - 10% IB range (LONG) / IBH + 10% IB range (SHORT)
Target: Dynamic (see below)
```

### Dynamic Target

| Mode | Default | P-shape Developing | b-shape Developing | BPR Active |
|------|---------|-------------------|-------------------|------------|
| VA LONG | Prior POC | (POC+DPOC)/2 (extended) | VWAP (tightened) | BPR midpoint |
| VA SHORT | Prior POC | VWAP (tightened) | (POC+DPOC)/2 (extended) | BPR midpoint |
| IB LONG | IB midpoint | (mid+DPOC)/2 | VWAP | BPR midpoint |
| IB SHORT | IB midpoint | VWAP | (mid+DPOC)/2 | BPR midpoint |

DPOC migration threshold: > 10% of VA range from prior POC.

### Dalton Acceptance Tracking

- **Early acceptance**: 2 consecutive closes on the right side of the level
- **Inside acceptance**: 6 consecutive closes (30-min period equivalent)
- Counters: `_accept_above_val`, `_accept_below_vah`, `_accept_above_ibl`, `_accept_below_ibh`
- Resets on any close on wrong side

### BPR Tracking

- Uses pre-computed `in_bpr`, `fvg_bull_bottom/top`, `fvg_bear_bottom/top` from `indicators/ict_models.py`
- BPR zone = overlap of bull FVG and bear FVG boundaries
- Deactivates when price moves > 1 BPR-range away

### Constants (`config/constants.py`)

```python
BALANCE_MIN_QUALITY_SCORE = 5
BALANCE_COOLDOWN_BARS = 20
BALANCE_LAST_ENTRY_TIME = time(13, 30)
BALANCE_VA_STOP_BUFFER_PCT = 0.15
BALANCE_VA_MIN_RR = 1.0
BALANCE_IB_RECLAIM_MIN = 350
BALANCE_IB_RECLAIM_MAX = 500
BALANCE_IB_STOP_BUFFER_PCT = 0.10
BALANCE_ACCEPT_EARLY = 2
BALANCE_ACCEPT_INSIDE = 6
BALANCE_HVN_PERCENTILE = 80
BALANCE_LVN_PERCENTILE = 20
BALANCE_EDGE_PROXIMITY_PCT = 0.10
```

### Key Python Source Files (READ BEFORE NT8 PORT)

| File | What to Study |
|------|---------------|
| `strategy/balance_signal.py` | **Primary reference** — all 6 scoring methods, both signal modes, acceptance tracking, BPR state, dynamic target, DPOC |
| `strategy/edge_fade.py` | Template for OF quality scoring, CVD divergence detection, 5-min candle aggregation, FVG/IFVG confluence checks |
| `strategy/or_reversal.py` | State machine pattern (WAITING→SWEEP→REVERSAL→RETEST→FIRE), IB bar scanning, EOR level tracking |
| `strategy/b_day.py` | B-day confidence gating, adaptive IB range cap, regime-aware volatility filter, multi-touch tracking |
| `strategy/base.py` | StrategyBase ABC — `on_session_start()`, `on_bar()`, `Signal` lifecycle |
| `strategy/signal.py` | Signal dataclass — `entry_price`, `stop_price`, `target_price`, `metadata`, `confidence` |
| `indicators/ict_models.py` | FVG/BPR detection logic — 3-bar gap pattern, zone lifecycle, overlap computation (must be ported inline for NT8) |
| `config/constants.py` | All `BALANCE_*` constants — thresholds, buffers, time gates |
| `tests/test_balance_signal.py` | Unit tests for every scoring factor — use as acceptance criteria for NT8 port |
| `scripts/analyze_balance_signal.py` | Post-backtest analysis — per-factor efficacy, score thresholds, monthly P&L |

### Backtest Results (2026-02-26, 260 sessions NQ)

| Threshold | Trades | WR | Net P&L | Max DD | PF |
|-----------|--------|-----|---------|--------|-----|
| Score >= 5 | 4 | 100% | $5,403 | $0 | inf |
| Score >= 4 | 14 | 35.7% | $1,173 | $2,174 | 1.25 |

- **p_day: 75% WR, +$1,937** — best day type
- **b_day: 12.5% WR, -$634** — should be excluded or require higher score
- All trades are VA_EDGE_FADE mode (Wide IB Reclaim never triggered)
- All 4 winners at score>=5 hit TARGET exit

---

## NT8 Indicator: `RockitBalanceSignal.cs` (Phase 2)

### Port Checklist

- [ ] Same 6-factor scoring (inline computation, no hosted indicators)
- [ ] VA Edge Fade + Wide IB Reclaim modes
- [ ] Same thresholds (tuned by Python backtest)
- [ ] TV-style annotations (per CLAUDE.md):
  - Arrow at signal bar
  - Dotted connector to offset callout
  - Callout box with score breakdown
  - Green/Red R:R zones
  - Entry/Stop/Target lines + labels
- [ ] BPR zones drawn as yellow rectangles
- [ ] Prior VA inline computation (volume profile bins from prior RTH bars)
- [ ] Trace logging to `Documents\NinjaTrader 8\RockitBalanceSignalTrace\`
- [ ] Try-catch OnBarUpdateInner wrapper
- [ ] Session scoping: `_sessionDate.ToString("MMdd")` suffix on all draw tags
- [ ] Right edge = 0 always
- [ ] Draw window = full RTH (16:00)

### NT8-Specific Considerations

1. **Volume Profile Inline**: Compute prior session VP bins using `Bars.GetVolume(i)` over prior RTH bars. No hosted `RockitPriorSessionVA`.
2. **5-min bars**: Use wick (High[0]/Low[0]) for zone detection, not close.
3. **Session date**: Use `_sessionIterator.GetTradingDay()`, never `Time[0].Date`.
4. **BPR**: Must compute FVGs inline (3-bar pattern detection) rather than relying on Python's pre-computed columns.
5. **Score callout example**:
   ```
   >>> BALANCE SHORT <<<
   Score: 7/10
   F1:2 F2:2 F3:1 F4:0 F5:1 F6:1
   Entry: 20085 | Stop: 20115 | Tgt: 20000
   ```

---

## Analysis Script: `scripts/analyze_balance_signal.py`

Measures:
1. Per-factor contribution (WR at max vs 0 for each factor)
2. Score threshold analysis (optimal cutoff: 4, 5, 6, 7, 8+)
3. Mode breakdown (VA Edge Fade vs Wide IB Reclaim, by direction)
4. BPR accuracy (with vs without BPR)
5. Acceptance filter value
6. Day type distribution
7. Monthly P&L grid
