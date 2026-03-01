# Strategy Research — Master Reference

**Last Updated**: 2026-03-01 (v3.3 — OR Acceptance entry model diagnosis)
**Instrument**: MNQ (Micro E-mini Nasdaq-100)
**Data**: 259 RTH sessions, 2,000+ trades analyzed
**Accounts**: 5 x TradeDay/Tradeify Lightning $150K

---

## TLDR — Recommended Portfolio

**Three complementary strategies, all LONG-biased, covering every market day.**

| Strategy | When | Trades/Mo | WR | PF | $/Mo (5 MNQ) | Study |
|----------|------|-----------|-----|-----|-------------|-------|
| **80P Rule** (Limit 50% VA + 4R) | Open outside VA, re-enters | 4.0 | 44.7% | 2.57 | **$1,922** | [80p-rule/](80p-rule/) |
| **Balance Day IBL Fade** (B-day + first touch) | Balance days, IBL touch | 3.4 | 82% | 9.35 | **$1,730** | [balance-day-edge-fade/](balance-day-edge-fade/) |
| **20P IB Extension** (2xATR + 2R) | IB breakout continuation | 3.7 | 45.5% | 1.78 | **$496** | [20p-rule/](20p-rule/) |
| **OR Reversal** (Judas Swing) | Sweep + reversal at open | 9.6 | 60.9% | 2.96 | **$2,720** | [exploratory/](exploratory/) |
| **OR Acceptance** (Break + Hold) | Level break + continuation | 5.9 | 43.7% | 1.32 | **$264** | [exploratory/](exploratory/) |
| **Combined** | — | **~26** | — | — | **~$7,132** | — |

**For evaluation passing** (fast, aggressive): Use [Dual Order Flow](dual-order-flow/) at 31 MNQ — $1,027/day, pass $9K in ~9 days.

---

## Phase Roadmap

| Phase | Mode | Strategy | Contracts | Target |
|-------|------|----------|-----------|--------|
| **1** (Days 1–9) | Evaluation | Dual Order Flow (Maniac) | 31 MNQ | Pass $9K eval |
| **2** (Days 10–20) | Funded cushion | Same, reduced risk | 20 MNQ | Build $5K buffer |
| **3** (Day 21+) | Funded production | 80P + B-Day Fade + 20P | 5 MNQ | $4,148/mo/acct |
| **4** (Month 3+) | Scale | Pass accounts 2–5 | 5 MNQ each | $20K–40K/mo total |

---

## Strategy Definitions (Entry / Exit / Stop)

### 1. 80% Rule — Limit 50% VA Entry + 4R Target

**Concept**: RTH opens outside prior day's Value Area, then re-enters. Price is expected to traverse to the opposite VA boundary.

| Parameter | Value |
|-----------|-------|
| **VA Source** | ETH 1-day (overnight + RTH) — captures institutional positioning |
| **Setup** | RTH open prints outside VA (above VAH or below VAL) |
| **Entry** | Limit order at 50% of VA depth (midpoint of VA = POC area) |
| **Confirmation** | 2x consecutive 5-min closes inside VA |
| **Direction** | LONG when open < VAL; SHORT when open > VAH |
| **Stop** | VA edge + 10 pts buffer |
| **Target** | 4R (4x the stop distance) |
| **R:R** | 4:1 |
| **Max hold** | EOD |
| **Frequency** | 4.0 trades/month |
| **Win Rate** | 44.7% |
| **Profit Factor** | 2.57 |
| **Monthly P&L** | $1,922 (5 MNQ) |

**Alternative configs** (see [80p-rule/2026.02.22-80p-rule-strategy-report-v3.md](80p-rule/2026.02.22-80p-rule-strategy-report-v3.md)):
- **Maximum Win Rate**: 100% retest + 2R target → 65.7% WR, PF 3.45, $915/mo
- **Maximum Frequency**: Acceptance model + 4R → 38.3% WR, PF 1.70, $955/mo
- **Highest WR/PF combo**: 2x 5-min acceptance + double-top retest + target POC → 66.7% WR, PF 6.83, $762/mo

**Critical rule**: Retest entries MUST use VA-edge stops (58–66% WR). Candle-extreme stops = 5–14% WR (catastrophic).

**Code**: [`strategy/eighty_percent_rule.py`](../../strategy/eighty_percent_rule.py)

---

### 2. Balance Day IBL Fade — First-Touch LONG

**Concept**: On balance/rotation days, price tests the IB low but fails to break through. Fade the touch back to IB midpoint.

| Parameter | Value |
|-----------|-------|
| **Day type** | B-day (or b-shape IB POC at 10:30 as real-time proxy) |
| **Setup** | Price touches IBL after 10:30 ET |
| **Entry** | 30-min close back above IBL (acceptance confirmation) |
| **Filter** | First touch only ("fade the first test, respect the second") |
| **Direction** | LONG only (NQ structural long bias: 71% WR long vs 57% short) |
| **Stop** | IBL − 10% of IB range |
| **Target** | IB midpoint |
| **Cutoff** | No entries after 14:00 ET |
| **Frequency** | 3.4 trades/month |
| **Win Rate** | 82% |
| **Profit Factor** | 9.35 |
| **Monthly P&L** | $1,730 (5 MNQ) |

**Day-type tiers** (see [balance-day-edge-fade/2026.02.27-balance-day-edge-fade-study.md](balance-day-edge-fade/2026.02.27-balance-day-edge-fade-study.md)):
- **Tier 1 (B-day IBL Long)**: 76% WR, PF 5.89, $1,352/mo
- **Tier 2 (First-Touch Long, any day)**: 82% WR, PF 9.35, $1,730/mo
- **Tier 3 (Neutral day both directions)**: 69% WR, PF 2.09, $1,610/mo

**DO NOT TRADE**: P-day SHORT (29% WR, PF 0.53 — fights bullish skew).

**DPOC migration filter**: Stabilizing DPOC confirms rotation. Low retention (<40%) = skip (22% WR). Neutral day + stabilizing DPOC + IBL LONG = 71% WR.

**Code**: [`strategy/b_day.py`](../../strategy/b_day.py), [`strategy/neutral_day.py`](../../strategy/neutral_day.py), [`strategy/p_day.py`](../../strategy/p_day.py), [`strategy/day_confidence.py`](../../strategy/day_confidence.py)

---

### 3. 20P Rule — IB Extension Continuation (2xATR Stop + 2R)

**Concept**: When IB extends >20% of its range, the breakout has momentum. Ride the continuation with structure-based stops.

| Parameter | Value |
|-----------|-------|
| **Setup** | IB extends >20% of range |
| **Entry** | 3x consecutive 5-min closes beyond IB boundary |
| **Direction** | Direction of the IB extension |
| **Stop** | 2x ATR below entry (LONG) / above entry (SHORT) |
| **Target** | 2R (2x the stop distance) |
| **Risk/trade** | ~32 pts median |
| **Frequency** | 3.7 trades/month |
| **Win Rate** | 45.5% |
| **Profit Factor** | 1.78 |
| **Monthly P&L** | $496 (5 MNQ) |

**Why v2 (structure stops) > v1 (IB boundary stops)**:
- Risk: 32 pts vs 219 pts (85% reduction)
- PF: 1.78 vs 1.25
- Clean exits: 24 stops / 20 targets / 0 EOD vs 6/11/27

**Code**: (NinjaTrader implementation — see [comparisons/2026.02.24-ninjatrader-implementation-design.md](comparisons/2026.02.24-ninjatrader-implementation-design.md))

---

### 4. OR Reversal (Judas Swing) — Sweep + Reversal at Open

**Concept**: Price sweeps a key overnight level (London/Asia/PDH) in the first 15 min, then reverses. Enter on the reversal with order flow confirmation.

| Parameter | Value |
|-----------|-------|
| **Setup** | EOR sweeps a key level (London, Asia, PDH/PDL) then reverses |
| **Entry** | Closest-level retest with CVD divergence + delta confirmation |
| **Direction** | Both (LONG 75% WR dominant) |
| **Stop** | Beyond the sweep extreme + 0.5 ATR buffer |
| **Target** | 2R |
| **Frequency** | 9.6 trades/month |
| **Win Rate** | 60.9% |
| **Profit Factor** | 2.96 |
| **Monthly P&L** | $2,720 (5 MNQ) |

**Code**: [`strategy/or_reversal.py`](../../strategy/or_reversal.py)

---

### 5. OR Acceptance — Level Break + Continuation

**Concept**: Price breaks a key level at the open and HOLDS it (acceptance, not fake sweep). Enter on the 50% retest of the continuation move.

| Parameter | Value |
|-----------|-------|
| **Acceptance check** | 70% of IB (60-bar) closes on correct side of level, <=5 wick violations |
| **Levels** | London H/L, Asia H/L, PDH/PDL |
| **Entry** | 50% retest of acceptance range + delta/CVD confirmation |
| **Direction** | Both (SHORT 44.8% WR, LONG 42.9% WR) |
| **Stop** | Acceptance level + 0.5 ATR buffer |
| **Target** | 2R |
| **Frequency** | 5.9 trades/month |
| **Win Rate** | 43.7% |
| **Profit Factor** | 1.32 |
| **Monthly P&L** | $264 (5 MNQ) |

**v2 optimization** (2026-02-27): Expanded from London-only (0 trades) to all reference levels, relaxed acceptance conditions. See [exploratory/2026.02.27-or-acceptance-optimization.md](exploratory/2026.02.27-or-acceptance-optimization.md).

**v3 proposed** (2026-03-01): Entry model fix — limit retest at acceptance level instead of 50% retrace. Entries close to level (≤20 pts risk) = 96% WR. Current 50% retrace entries (avg 70 pts risk) = 44% WR. See [exploratory/2026.03.01-premarket-acceptance-conditions.md](exploratory/2026.03.01-premarket-acceptance-conditions.md).

**Code**: [`strategy/or_acceptance.py`](../../strategy/or_acceptance.py)

---

### 6. Dual Order Flow — Evaluation Mode (Pass Fast)

**Concept**: Pure 1-min order flow signals (imbalance + delta + CVD) for maximum trade frequency to pass prop firm evaluations quickly.

| Parameter | Evaluation (Maniac) | Funded (Sniper) |
|-----------|---------------------|-----------------|
| **Contracts** | 31 MNQ | 20 MNQ |
| **Window** | 10:00–13:00 ET | 10:00–13:00 ET |
| **Stop** | 0.4x ATR (~6.5 pts) | 0.4x ATR (~6.5 pts) |
| **Target** | 2.0x stop (~13 pts) | 2.0x stop (~13 pts) |
| **Max hold** | 8 bars (8 min) | 8 bars (8 min) |
| **Win Rate** | 44.2% | 52.0% |
| **Trades/day** | 10.9 | 7.1 |
| **Daily P&L** | $1,027 | $794 |
| **Daily loss limit** | $1,500 | $1,500 |
| **Max trades/day** | 15 | 15 |

**Funded mode adds**: 5-min CVD alignment + 5-min VWAP context → +10–12% WR, −45% trades, +100% expectancy, PF > 2.0.

**Entry (Tier 1 — Full Size)**: Imbalance pctl > 85% + Volume > 1.5x avg + CVD confirms + Delta direction matches
**Entry (Tier 2 — Standard)**: Delta pctl > 85% + CVD confirms + No Tier 1 active

**Code**: [`strategy/signal.py`](../../strategy/signal.py), NinjaTrader: `DualOrderFlow_Evaluation.cs`, `DualOrderFlow_Funded.cs`
**Study**: [dual-order-flow/FINAL_EVALUATION_STRATEGIES_REPORT.md](dual-order-flow/FINAL_EVALUATION_STRATEGIES_REPORT.md)

---

## Key Research Findings

1. **NQ has structural long bias** — every short strategy fails except extreme liquidity sweeps (15+ pts on B-days). Default LONG.
2. **5-min timeframe >> 1-min** for quality — 50% WR / +13.35 pts (5-min) vs 41.6% / -0.35 pts (1-min).
3. **Acceptance is the #1 filter** — 30-min (or 2x 5-min) acceptance provides +28pp edge.
4. **Stop placement matters more than entry depth** — VA-edge stops (58–66% WR) vs candle-extreme (5–14%).
5. **Balance days are 43% of sessions** and detectable at 10:30 via IB POC shape.
6. **DPOC stabilizing regime** confirms balance rotation — use as secondary filter, not primary.
7. **Five strategies are complementary** — 80P (VA days), B-Day Fade (balance), 20P (breakout), OR Rev (sweep), OR Accept (break).
8. **OR Reversal is the top performer** — 60.9% WR, PF 2.96, $2,720/mo. LONG direction dominates (75% WR).
9. **OR Acceptance needed expanded levels** — London-only produced 0 trades; adding Asia H/L + PDH/PDL → 71 trades.
10. **IB window (60 bars) beats EOR (30 bars) for acceptance** — 43.7% vs 40.7% WR, more time for true acceptance to establish.
11. **Acceptance is the most common open type** (37% of sessions). 70% of the time, acceptance leads to continuation in the first 30 min (avg 101 pts).
12. **Choppy premarket → strong acceptance**: High premarket chop score (>0.4) = 46.7% WR, PF 1.42. Low chop = 27.3% WR. Chop that resolves into acceptance is stronger than a trending premarket continuation.
13. **LRLR trendline breaks are too rare (~10%) to use as an acceptance filter** and don't improve trade outcomes (33% WR with vs 45% without).
14. **Post-London directional strength (0.53 vs 0.39)** is the best feature distinguishing acceptance from Judas opens.
15. **OR Acceptance v2 entry model is the root cause of low WR** — 50% retrace puts entry 50-100 pts from the acceptance level. Entries ≤20 pts from level = 96% WR. Entries >60 pts = 15% WR. Fix: limit retest at the acceptance level after 2x 5-min acceptance.
16. **BOTH sessions (Judas + Acceptance on different levels) lose money** — 25% WR, PF 0.58. Skip them.

---

## What NOT to Trade

| Setup | WR | Why |
|-------|-----|-----|
| P-day SHORT | 29% | Fights bullish skew + NQ long bias |
| VWAP sweep-fail (any direction) | 45% | VWAP oscillation is noise |
| 2R target on IB fades | 47% | Overshoots mean-reversion range |
| OR Acceptance with wide stop (>60 pts) | 15% | Entry too far from acceptance level |
| BOTH sessions (Judas+Accept diff levels) | 25% | Conflicting signals = noise |
| Candle-extreme stops on retest entries | 5–14% | Stop is inside VA — always hits |
| Any SHORT without extreme sweep filter | <40% | NQ mean-reverts up, not down |

---

## Prop Firm Compliance

**TradeDay / Tradeify Lightning $150K**

| Rule | Requirement | Our Strategy |
|------|-------------|-------------|
| Profit target | $9,000 | $1,027/day eval → 9 days |
| EOD trailing drawdown | $4,500 | Max observed: $2,500 |
| Daily loss limit | $3,750 | Self-imposed: $1,500 |
| Min hold time | 10 seconds | 30 sec–8 min |
| Consistency (1st payout) | No day > 20% | Best day 14.5% |

---

## Folder Structure

```
research/strategy-studies/
├── README.md                        ← You are here (master reference)
├── 80p-rule/                        ← 80% Rule (Dalton framework)
│   ├── 2026.02.22-...-report.md         v1 (initial)
│   ├── 2026.02.22-...-report-v2.md      v2 (refined)
│   └── 2026.02.22-...-report-v3.md      v3 (LATEST — use this)
├── 20p-rule/                        ← 20% IB Extension
│   ├── 2026.02.23-20p-rule-study.md     v1 (IB boundary stops)
│   └── 2026.02.24-...-v2-structure-stops.md  v2 (LATEST — 2xATR stops)
├── balance-day-edge-fade/           ← Balance Day IBL/IBH Fade
│   ├── 2026.02.26-...-plan.md           Planning doc
│   └── 2026.02.27-...-study.md          Full study (LATEST)
├── va-edge-fade/                    ← VA Edge Fade
│   └── 2026.02.24-va-edge-fade-study.md
├── bracket-breakout/                ← Bracket / Breakout Study
│   └── 2026.02.25-bracket-breakout-study.md
├── va-breakout-continuation/        ← VA Breakout Continuation
│   └── 2026.02.26-va-breakout-continuation-study.md
├── dual-order-flow/                 ← Evaluation + Funded Order Flow
│   ├── FINAL_EVALUATION_STRATEGIES_REPORT.md
│   ├── evaluation_strategies.py
│   └── execution_playbook_70wr.py
├── comparisons/                     ← Cross-strategy analysis
│   ├── 2026.02.24-master-strategy-comparison.md
│   ├── 2026.02.24-option3-drawdown-analysis.md
│   ├── 2026.02.24-ninjatrader-implementation-design.md
│   └── backtest_results_research_strategies.py
├── exploratory/                     ← Early research + OR strategies
│   ├── 2026.02.27-or-acceptance-optimization.md  ← OR Acceptance optimization
│   ├── 2026.03.01-premarket-acceptance-conditions.md  ← Premarket study (LATEST)
│   ├── TWO_HOUR_TRADER_STUDY.md
│   ├── OPENING_RANGE_BREAKOUT_STUDY.md
│   ├── TREND_FOLLOWING_BREAKOUT_STUDY.md
│   └── MEAN_REVERSION_STUDY.md
└── legacy/                          ← Early session notes
    ├── 2026.02.18.md
    └── 2026.02.19.md
```

## Implementation Code

| Strategy | Python Code | NinjaTrader |
|----------|------------|-------------|
| 80P Rule | [`strategy/eighty_percent_rule.py`](../../strategy/eighty_percent_rule.py) | — |
| B-Day Fade | [`strategy/b_day.py`](../../strategy/b_day.py) | — |
| Neutral Day | [`strategy/neutral_day.py`](../../strategy/neutral_day.py) | — |
| P-Day (DO NOT SHORT) | [`strategy/p_day.py`](../../strategy/p_day.py) | — |
| Day Confidence | [`strategy/day_confidence.py`](../../strategy/day_confidence.py) | — |
| Day Type Classifier | [`strategy/day_type.py`](../../strategy/day_type.py) | — |
| DPOC Migration | [`rockit-framework/modules/dpoc_migration.py`](../../rockit-framework/modules/dpoc_migration.py) | — |
| Pipeline Orchestrator | [`rockit-framework/orchestrator.py`](../../rockit-framework/orchestrator.py) | — |
| OR Reversal (Judas Swing) | [`strategy/or_reversal.py`](../../strategy/or_reversal.py) | — |
| OR Acceptance (Break+Hold) | [`strategy/or_acceptance.py`](../../strategy/or_acceptance.py) | — |
| Premarket Study | [`scripts/study_premarket_acceptance.py`](../../scripts/study_premarket_acceptance.py) | — |
| Dual Order Flow (Eval) | [`strategy/signal.py`](../../strategy/signal.py) | `DualOrderFlow_Evaluation.cs` |
| Dual Order Flow (Funded) | [`strategy/signal.py`](../../strategy/signal.py) | `DualOrderFlow_Funded.cs` |

---

*Version: 3.3 — OR Acceptance entry model diagnosis + v3 proposed fix.*
*Previous: v3.1 (OR Rev + Accept), v3.0 (reorganized), MASTER_INDEX.md (deprecated)*
