# Order Flow vs Playbook: Comprehensive Comparison

**Instrument:** MNQ (Micro Nasdaq Futures)
**Period:** Nov 18, 2025 - Feb 16, 2026 (62 sessions)
**Branch:** v13-order-flow-optimization

---

## 1. The 4-Way Comparison

| Approach | Trades | Win Rate | Net P&L | Expectancy/Trade | Sharpe |
|:---------|-------:|---------:|--------:|-----------------:|-------:|
| **Playbook + OF Quality Gate** | **20** | **80.0%** | **$4,434** | **$222** | **18.81** |
| Playbook Alone (v12 baseline) | 15 | 86.7% | $3,221 | $215 | 12.19 |
| OF All Models (best per session) | 62 | 56.5% | $5,229 | $84 | - |
| OF + 30-min HTF Filter (best combo) | 34 | 52.9% | $1,743 | $51 | - |
| OF Standalone (best single model) | 22 | 50.0% | $814 | $37 | - |

> The Playbook + OF Quality Gate delivers **3x the expectancy** of standalone order flow
> with a win rate 25+ percentage points higher.

---

## 2. Standalone OF Model Results

Five standalone order flow entry models tested across all 62 sessions. All entries long-only,
stop at 0.4x IB range, EOD exit, max 1 entry per session per model.

| OF Model | Entry Criteria | Trades | Win Rate | Net P&L | $/Trade |
|:---------|:---------------|-------:|---------:|--------:|--------:|
| delta_surge | delta_zscore > 1.5, delta > 0, vol_spike > 1.3 | 61 | 45.9% | +$296 | +$5 |
| delta_momentum | 5-bar delta sum > 300, accel > 0, pctl >= 70 | 59 | 45.8% | +$749 | +$13 |
| absorption | imbalance > 1.2, vol_spike > 1.5, delta > 0, above VWAP | 46 | 41.3% | +$32 | +$1 |
| cvd_divergence | Price at low but CVD higher than prev low's CVD | 22 | 50.0% | +$814 | +$37 |
| high_conviction | 4+ of 5 signals (dz>1, imb>1.15, vol>1.3, pctl>=80, >VWAP) | 62 | 38.7% | -$188 | -$3 |

**Key takeaway:** OF models fire on nearly every session (61-62/62) with ~45% win rates.
They detect *intensity*, not *direction*. Even requiring 4+ simultaneous signals (high_conviction)
produces the worst results: 38.7% WR and negative P&L.

---

## 3. HTF Trend Filter Impact on OF Models

Adding a 30-minute EMA8 > EMA21 trend filter was the single best improvement for standalone OF.

| OF Model | No Filter | + 5-min EMA | + 15-min EMA | + 30-min EMA |
|:---------|----------:|------------:|-------------:|-------------:|
| delta_surge | +$296 (45.9%) | +$478 (53.3%) | +$524 (50.0%) | +$545 (53.1%) |
| delta_momentum | +$749 (45.8%) | +$239 (44.4%) | -$111 (44.8%) | **+$1,388 (55.2%)** |
| absorption | +$32 (41.3%) | -$326 (40.0%) | +$667 (46.9%) | +$171 (44.1%) |
| cvd_divergence | +$814 (50.0%) | -$334 (25.0%) | +$581 (50.0%) | +$162 (40.0%) |
| high_conviction | -$188 (38.7%) | -$506 (39.3%) | -$498 (39.4%) | **+$1,743 (52.9%)** |

> 30-min EMA is the only consistent HTF filter. It turned high_conviction from -$188 to +$1,743.
> 5-min EMA is too noisy. 15-min is inconsistent.

---

## 4. Which OF Model Works Best for Each Playbook Strategy?

Checked which of the 5 standalone OF models would have also fired at each Playbook entry bar.

### B-Day (IBL Fade) - 4 entries, 4 winners

| OF Model | Catches | Winners Caught | Losers Caught | P&L Caught |
|:---------|--------:|---------------:|--------------:|-----------:|
| **cvd_divergence** | **4/4 (100%)** | **4/4** | **0/0** | **+$2,285** |
| delta_surge | 1/4 (25%) | 1/4 | 0/0 | +$498 |
| high_conviction | 1/4 (25%) | 1/4 | 0/0 | +$498 |
| absorption | 0/4 (0%) | 0/4 | 0/0 | - |
| delta_momentum | 0/4 (0%) | 0/4 | 0/0 | - |

> **CVD divergence is a perfect match for B-Day.** The IBL fade is structurally a CVD divergence:
> price sweeps the IB low while buying pressure (CVD) holds up, signaling absorption by buyers.

### Trend Day Bull (VWAP Pullback) - 8 entries, 6 winners / 2 losses

| OF Model | Catches | Winners Caught | Losers Caught | P&L Caught |
|:---------|--------:|---------------:|--------------:|-----------:|
| **cvd_divergence** | **4/8 (50%)** | **4/6** | **0/2** | **+$671** |
| high_conviction | 3/8 (38%) | 2/6 | 1/2 | +$274 |
| delta_surge | 1/8 (13%) | 1/6 | 0/2 | +$37 |
| absorption | 1/8 (13%) | 1/6 | 0/2 | +$37 |
| delta_momentum | 1/8 (13%) | 1/6 | 0/2 | +$169 |

> CVD divergence catches 67% of Trend Bull winners and 0% of losers.
> high_conviction catches 1 of 2 losers (negative discrimination).

### P-Day (VWAP Long) - 8 entries, 6 winners / 2 losses

| OF Model | Catches | Winners Caught | Losers Caught | P&L Caught |
|:---------|--------:|---------------:|--------------:|-----------:|
| **cvd_divergence** | **4/8 (50%)** | **4/6** | **0/2** | **+$672** |
| high_conviction | 3/8 (38%) | 2/6 | 1/2 | +$275 |
| delta_momentum | 2/8 (25%) | 2/6 | 0/2 | +$207 |
| delta_surge | 0/8 (0%) | - | - | - |
| absorption | 0/8 (0%) | - | - | - |

> Same pattern as Trend Bull. CVD divergence is the best discriminator.

---

## 5. CVD Divergence: The Playbook's Natural OF Signal

Across all 13 unique Playbook entry bars:

| Metric | CVD Divergence | Next Best (high_conviction) |
|:-------|---------------:|----------------------------:|
| Catches Playbook winners | **9/11 (82%)** | 4/11 (36%) |
| Catches Playbook losers | **0/2 (0%)** | 1/2 (50%) |
| Discrimination (W% - L%) | **+82 pp** | -14 pp |
| Total P&L caught | +$2,994 | +$809 |

**Why CVD divergence aligns with the Playbook:**

The Playbook entries are all *pullback* entries (VWAP pullback for Trend/P-Day, IBL fade for B-Day).
A pullback entry is inherently a CVD divergence - price is pulling back (lower) while underlying
buying pressure (cumulative delta) holds steady or rises. The other OF models look for *momentum
spikes* (high z-scores, volume explosions), which is the opposite of what a pullback looks like.

---

## 6. OF Feature Profile by Strategy

Average order flow features at entry bars, broken out by strategy and outcome.

### Winners

| Strategy | Avg Delta | Delta Z | Percentile | Imbalance | Vol Spike | QG Score |
|:---------|----------:|--------:|-----------:|----------:|----------:|---------:|
| B-Day | +200 | 1.31 | 92nd | 1.238 | 1.42x | 2.8/3 |
| Trend Bull | +137 | 1.05 | 84th | 1.260 | 1.29x | 2.8/3 |
| P-Day | +114 | 0.80 | 83rd | 1.260 | 1.13x | 2.8/3 |

### Losers (both -$2.74 breakeven stops)

| Strategy | Avg Delta | Delta Z | Percentile | Imbalance | Vol Spike | QG Score |
|:---------|----------:|--------:|-----------:|----------:|----------:|---------:|
| Trend Bull | +115 | 0.94 | 85th | 1.231 | 1.08x | 2.5/3 |
| P-Day | +115 | 0.94 | 85th | 1.231 | 1.08x | 2.5/3 |

> B-Day entries show the strongest OF readings (delta +200, z-score 1.31, 92nd percentile).
> Winner vs loser differentiation is subtle on the current sample - both "losses" are
> breakeven stops at -$2.74, not real losses.

---

## 7. Model Count at Entry vs Outcome

How many standalone OF models fire simultaneously at each Playbook entry:

| Models Firing | Entries | Win/Loss | Net P&L | Interpretation |
|:--------------|--------:|---------:|--------:|:---------------|
| 0 models | 1 | 0W / 1L | -$3 | No OF confirmation = only loss |
| 1 model | 9 | 8W / 1L | +$2,827 | Quiet pullback, usually CVD div |
| 3 models | 2 | 2W / 0L | +$536 | Strong OF + pullback |
| 4 models | 1 | 1W / 0L | +$37 | Explosive OF (rare at entries) |

> The typical winning Playbook entry has just 1 OF model firing (CVD divergence).
> These are steady pullbacks with underlying buying pressure, not explosive momentum bars.

---

## 8. Recommendations

### Current Best Setup: Playbook + OF Quality Gate (2-of-3)
- 20 trades, 80.0% WR, $4,434 net, $222/trade, Sharpe 18.81
- Quality gate: delta_percentile >= 60, imbalance_ratio > 1.0, volume_spike >= 1.0 (need 2 of 3)

### Potential Enhancement: Add CVD Divergence as Confluence
- CVD divergence catches 82% of winners and 0% of losers within the Playbook
- Could be used as additional confidence scoring, not a hard filter (only 50% coverage on Trend/P-Day)
- Best application: **B-Day confirmation** (100% of B-Day winners have CVD divergence)

### What NOT to Do
- Do not use OF signals as standalone entries (45-50% WR, $5-37/trade)
- Do not require high_conviction (4+ signals) - it's the worst performer (-$188)
- Do not use momentum-based OF models (delta_surge, absorption) to filter pullback entries
- 5-min and 15-min HTF filters are too inconsistent to rely on

---

*Generated from diagnostic_of_standalone.py and diagnostic_playbook_of_breakdown.py*
*Data: 62 RTH sessions, MNQ 1-minute bars with order flow (delta, volume, imbalance)*
