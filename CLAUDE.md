# Project: BookMap Order Flow Studies

## Overview

MNQ (Micro E-mini Nasdaq-100) trading strategy research and implementation for prop firm funded accounts. All strategies target 5 MNQ contracts per trade on TradeDay/Tradeify Lightning $150K accounts.

## Strategy Studies Workflow

### After ANY research session that produces new findings:

1. **Save the study** in the appropriate subfolder under `research/strategy-studies/`:
   - `80p-rule/` — 80% Rule (Dalton VA framework)
   - `20p-rule/` — 20% IB Extension
   - `balance-day-edge-fade/` — Balance day IB edge fades
   - `va-edge-fade/` — Value Area edge fades
   - `bracket-breakout/` — Bracket/breakout studies
   - `va-breakout-continuation/` — VA breakout continuation
   - `dual-order-flow/` — Evaluation + funded order flow
   - `comparisons/` — Cross-strategy analysis
   - New topics get their own subfolder

2. **Update `research/strategy-studies/README.md`** — this is the **single source of truth**:
   - Update the **TLDR portfolio table** if the recommended portfolio changes
   - Update or add the **Strategy Definition** section with exact entry/exit/stop/target
   - Update **WR, PF, $/mo** numbers if new backtests improve on prior results
   - Add the new study to the **Folder Structure** tree
   - Link new code files in the **Implementation Code** table
   - Update the **"Last Updated"** date at the top
   - Add any new "What NOT to Trade" findings

3. **Naming convention** for study files: `YYYY.MM.DD-<topic>-<version>.md`
   - Example: `2026.02.27-balance-day-edge-fade-study.md`
   - Mark superseded versions clearly in the README (e.g., "v1 (initial)" vs "v3 (LATEST — use this)")

### What the README must always contain:

- **TLDR table**: The current recommended portfolio with trades/mo, WR, PF, $/mo for each strategy
- **Phase roadmap**: Evaluation → cushion → funded → scale
- **Strategy definitions**: For each active strategy — exact entry, exit, stop, target, confirmation, filters, frequency
- **Key findings**: Numbered list of the most important research conclusions
- **What NOT to trade**: Strategies/setups that are proven losers
- **Folder structure**: Updated tree showing all subfolders and which file is LATEST
- **Code links**: Table mapping each strategy to its Python/NinjaTrader implementation

### When adding new strategy code:

- Place in `strategy/` directory
- Inherit from `strategy/base.py` if applicable
- Update the Implementation Code table in the README

## Key Constraints (Never Violate)

- **NQ has structural long bias** — default LONG. Only short with extreme sweep filter (15+ pts, B-day only).
- **VA-edge stops required** for any retest entry. Never use candle-extreme stops (5–14% WR catastrophe).
- **5-min timeframe preferred** over 1-min for quality signals in funded mode.
- **P-day SHORT is forbidden** — 29% WR, PF 0.53.
- **Prop firm compliance**: daily loss limit $1,500 self-imposed, no day > 20% of profits, min 10s hold.

## Repository Structure

```
BookMapOrderFlowStudies/
├── CLAUDE.md                          ← This file (project conventions)
├── README.md                          ← Project overview
├── research/
│   └── strategy-studies/
│       ├── README.md                  ← MASTER REFERENCE (always read this first)
│       ├── 80p-rule/                  ← 80% Rule studies
│       ├── 20p-rule/                  ← 20% IB Extension studies
│       ├── balance-day-edge-fade/     ← Balance day fade studies
│       ├── va-edge-fade/              ← VA edge fade studies
│       ├── bracket-breakout/          ← Bracket breakout studies
│       ├── va-breakout-continuation/  ← VA breakout continuation
│       ├── dual-order-flow/           ← Evaluation/funded order flow
│       ├── comparisons/               ← Cross-strategy comparisons
│       ├── exploratory/               ← Early research (not backtested)
│       └── legacy/                    ← Old session notes
├── strategy/                          ← Python strategy implementations
├── rockit-framework/                  ← Analysis pipeline and modules
└── ninjatrader/                       ← NinjaTrader C# scripts
```

## Starting a New Session

1. Read `research/strategy-studies/README.md` first — it has the current state of all strategies
2. Check which strategies are recommended vs deprecated
3. Understand the exact entry/exit/stop rules before making changes
4. Any new research must update the README when complete
