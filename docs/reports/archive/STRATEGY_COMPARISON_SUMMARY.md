# STRATEGY COMPARISON SUMMARY
## Evaluation vs Live Funded

---

## SIDE-BY-SIDE COMPARISON

| Metric | **EVALUATION** | **LIVE FUNDED** |
|--------|---------------|-----------------|
| **Goal** | Pass ASAP (9-12 days) | Stay funded forever |
| **Timeframe** | 1-minute bars | 1-minute bars |
| **Session** | 10:00-13:00 ET | 10:00-13:00 ET |

---

## POSITION SIZING

| | Evaluation | Live Funded |
|---|-----------|-------------|
| **Phase 1** | 15 contracts (Days 1-3) | 15 contracts (Month 1) |
| **Phase 2** | 20 contracts (Days 4-7) | 20 contracts (Month 2-3) |
| **Phase 3** | 31 contracts (Days 8-12) | 25 contracts (Month 4+) |
| **Max Risk/Trade** | $196 → $261 → $404 | $196 → $261 → $326 |
| **Max Daily Risk** | $1,500 | $1,000 |

---

## STRATEGY PARAMETERS

| Parameter | Evaluation | Live Funded |
|-----------|-----------|-------------|
| **Stop Loss** | 0.4x ATR (~6.5 pts) | 0.4x ATR (~6.5 pts) |
| **Profit Target** | 2.0x stop (~13 pts) | 2.0x stop (~13 pts) |
| **R:R Ratio** | 2:1 | 2:1 |
| **Time Exit** | 8 bars | 8 bars |
| **Max Hold** | 8 minutes | 8 minutes |

---

## PERFORMANCE METRICS

### Evaluation (31 Contracts, Full Size)

| Metric | Value |
|--------|-------|
| **Win Rate** | 44.2% |
| **Expectancy** | 1.52 points/trade |
| **Dollar Expectancy** | $94.24/trade (31 contracts × $2 × 1.52) |
| **Trades/Day** | 10.9 |
| **Daily P&L** | $1,027 |
| **Profit Factor** | 1.19 |
| **Avg Win** | 12.8 points ($793) |
| **Avg Loss** | 6.4 points ($397) |
| **Max Win Streak** | 8 trades |
| **Max Loss Streak** | 7 trades |
| **Days to Pass** | 9-12 days |

### Live Funded (20 Contracts, Moderate)

| Metric | Value |
|--------|-------|
| **Win Rate** | 44.2% (same strategy) |
| **Expectancy** | 1.52 points/trade (same) |
| **Dollar Expectancy** | $60.80/trade (20 contracts × $2 × 1.52) |
| **Trades/Day** | 10.9 |
| **Daily P&L** | $663 |
| **Profit Factor** | 1.19 |
| **Avg Win** | 12.8 points ($512) |
| **Avg Loss** | 6.4 points ($256) |
| **Max Win Streak** | 8 trades |
| **Max Loss Streak** | 7 trades |
| **Monthly Target** | $8,000-12,000 |

---

## SIGNAL BREAKDOWN

### Strategy A (Tier 1) - Imbalance + Volume + CVD

| | Evaluation | Live Funded |
|---|-----------|-------------|
| **Trigger** | Imbalance > 85% + Volume > 1.5x + CVD trend |
| **Position Size** | 31 contracts | 15-25 contracts |
| **Trades/Day** | 4.2 |
| **Win Rate** | 42.1% |
| **Expectancy** | 1.56 points/trade |
| **Priority** | High quality, full size |

### Strategy B (Tier 2) - Delta + CVD

| | Evaluation | Live Funded |
|---|-----------|-------------|
| **Trigger** | Delta > 85% + CVD trend (Strategy A not present) |
| **Position Size** | 15 contracts | 10-15 contracts |
| **Trades/Day** | 6.7 |
| **Win Rate** | 44.2% |
| **Expectancy** | 1.52 points/trade |
| **Priority** | Fill gaps, half size |

---

## RISK MANAGEMENT

### Evaluation

| Risk Parameter | Limit |
|---------------|-------|
| **Daily Loss Limit** | $1,500 (75% of prop firm $2K limit) |
| **Max Consecutive Losses** | 5 trades |
| **Position Size Reduction** | After 3 consecutive losses |
| **Emergency Stop** | -$2,000 (prop firm limit) |
| **Reset Risk** | 5% probability |

### Live Funded

| Risk Parameter | Limit |
|---------------|-------|
| **Daily Loss Limit** | $1,000 (50% of $2K limit) |
| **Max Consecutive Losses** | 4 trades |
| **Position Size Reduction** | After 2 consecutive losses |
| **Emergency Stop** | -$1,500 |
| **Days/Week** | 3-4 days (not every day) |
| **Skip Days** | Choppy/volatile days |

---

## EXPECTED OUTCOMES

### Evaluation (12-Day Timeline)

| Phase | Days | Contracts | Daily P&L | Cumulative |
|-------|------|-----------|-----------|------------|
| Phase 1 (Build) | 1-3 | 15 | $497 | $1,491 |
| Phase 2 (Scale) | 4-7 | 20 | $663 | $4,143 |
| Phase 3 (Push) | 8-12 | 31 | $1,027 | $9,278 |

**Total:** $9,278 profit in 12 days with 5% reset risk

### Live Funded (Monthly Compounding)

| Month | Contracts | Daily P&L | Monthly P&L | Cushion | Withdrawal |
|-------|-----------|-----------|-------------|---------|------------|
| 1 | 15 | $497 | $8,000 | $8,000 | $0 |
| 2 | 20 | $663 | $12,000 | $20,000 | $0 |
| 3 | 20 | $663 | $12,000 | $32,000 | $0 |
| 4 | 25 | $829 | $15,000 | $39,500 | $7,500 |
| 5+ | 25-30 | $800-1,000 | $15,000+ | Growing | $7,500-10,000 |

**Year 1 Target:** $100,000+ profit, $50,000 cushion

---

## KEY DIFFERENCES

### Evaluation Focus
- **Speed**: Pass in 9-12 days
- **Aggressive sizing**: Up to 31 contracts
- **Higher daily targets**: $1,000+/day
- **Acceptable risk**: 5% reset rate
- **All-in approach**: Maximum trades/day

### Live Funded Focus
- **Longevity**: Trade for years, not days
- **Conservative sizing**: 15-25 contracts
- **Moderate targets**: $500-800/day
- **Capital preservation**: 50% daily loss buffer
- **Quality over quantity**: Skip bad days

---

## STRATEGY SIGNALS (Entry Rules)

### Long Entry
```
Time: 10:00-13:00 ET
Delta percentile: > 85
Delta direction: > 0 (positive)
CVD trend: Rising
Imbalance: > 85% (Strategy A only)
Volume: > 1.5x average (Strategy A only)
```

### Short Entry
```
Time: 10:00-13:00 ET
Delta percentile: > 85
Delta direction: < 0 (negative)
CVD trend: Falling
Imbalance: > 85% (Strategy A only)
Volume: > 1.5x average (Strategy A only)
```

### Exit Rules (Both)
```
Stop Loss: Entry - 0.4x ATR (6.4 points)
Profit Target: Entry + 2.0x stop (12.8 points)
Time Exit: 8 bars (8 minutes)
Order Type: OCO Bracket (One Cancels Other)
```

---

## SUMMARY TABLE

| | Evaluation | Live Funded |
|---|-----------|-------------|
| **Contracts** | 15→20→31 | 15→20→25 |
| **Win Rate** | 44.2% | 44.2% |
| **Expectancy** | 1.52 pts/trade | 1.52 pts/trade |
| **Daily P&L** | $497→$1,027 | $497→$829 |
| **Days Target** | 12 | 30/month |
| **Reset Risk** | 5% | N/A (live) |
| **Daily Loss** | $1,500 | $1,000 |
| **Philosophy** | Pass fast | Trade forever |

---

## BOTTOM LINE

**Evaluation**: Trade aggressively to pass in 12 days (44% WR, 31 contracts max)

**Live**: Trade conservatively to compound forever (44% WR, 20 contracts sustained)

**The strategy is the same - only the sizing and psychology change.**

---

*Document Version: 1.0*
*Strategy: Dual Tier Order Flow (Delta + CVD)*
*Instrument: MNQ (Micro Nasdaq)*
