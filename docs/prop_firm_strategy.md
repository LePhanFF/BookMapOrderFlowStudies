# Prop Firm Strategy: Tradeify Select Flex $150K

**System:** Dalton Playbook + Order Flow Quality Gate (v13)
**Instrument:** MNQ (Micro Nasdaq Futures)
**Simulation:** 10,000 Monte Carlo runs per scenario using actual backtest trade distribution

---

## 1. Tradeify Select Flex $150K Rules

### Evaluation Phase

| Rule | Value |
|:-----|------:|
| Profit Target | $9,000 |
| EOD Trailing Max Drawdown | $4,500 |
| Daily Loss Limit | None |
| Consistency Rule | No single day > 40% of total profit |
| Minimum Trading Days | 3 |
| Contract Limit | 12 minis / 120 micros |
| Monthly Cost | ~$359 |

### Funded Phase (Select Flex)

| Rule | Value |
|:-----|------:|
| EOD Trailing Drawdown | $4,500 (locks when profit exceeds $4,600) |
| Daily Loss Limit | None |
| Consistency Rule | None |
| Profit Split | 90% trader / 10% firm |
| Payout Frequency | Every 5 winning days |
| Minimum Winning Day | $250 |
| Max Payout | $5,000 per request (up to 50% of total profit) |

### Funded Scaling (MNQ Contracts by Equity Above Start)

| EOD Equity Above Start | Max Minis | Max Micros |
|:-----------------------|----------:|-----------:|
| $0 - $1,499 | 3 | 30 |
| $1,500 - $1,999 | 4 | 40 |
| $2,000 - $2,999 | 5 | 50 |
| $3,000 - $4,499 | 8 | 80 |
| $4,500+ | 12 | 120 |

> **Critical:** You start funded with only 30 MNQ (3 minis). You don't get full 120 MNQ until
> you've already earned $4,500+ above starting balance — which means you've nearly hit the
> drawdown lock threshold. Scaling is a major constraint in the early funded phase.

---

## 2. Our Trade Distribution (v13 Backtest)

From 62 sessions (Nov 2025 - Feb 2026), the Playbook + OF Quality Gate produced:

| Metric | Value |
|:-------|------:|
| Unique trade sessions | 12 out of 62 (19.4%) |
| Trades per month | ~4.3 |
| Win rate | 80% (10W / 2L, losses are -$2.74 breakeven stops) |
| Avg points per contract (winners) | +53.0 pts |
| Avg points per contract (losers) | -0.2 pts |
| Median points per contract | +43.7 pts |
| Max points per contract | +153.5 pts |

### Per-Session P&L at Different Sizing (1 contract = $2/point)

| Sizing | Avg P&L/Trade | Max Risk/Trade | Trades to $9K Target |
|:-------|-------------:|--------------:|--------------------:|
| 2 MNQ | +$188 | ~$160 | 48 trades (11 months) |
| 4 MNQ | +$377 | ~$320 | 24 trades (6 months) |
| 6 MNQ | +$565 | ~$480 | 16 trades (4 months) |
| 10 MNQ | +$941 | ~$800 | 10 trades (2.5 months) |

---

## 3. Monte Carlo Results: Three Scenarios

We model three performance levels to account for live trading degradation:

| Scenario | Win Rate | Winner Haircut | Loss Size | Rationale |
|:---------|:--------:|:--------------:|:---------:|:----------|
| **Backtest-Exact** | 80% | None | -$2.74 (BE stop) | Raw backtest output |
| **Realistic Live** | 70% | 20% off winners | -$80 to -$140 real stops | Slippage, hesitation, wider stops |
| **Stress Test** | 60% | 35% off winners | -$120 to -$220 real stops | Extended drawdown / regime change |

### 3a. Evaluation Phase: Pass Rate & Timeline

| Scenario | Sizing | Pass Rate | Median Days | Median Months | Blown |
|:---------|:-------|----------:|------------:|--------------:|------:|
| Backtest-Exact | 4 MNQ | 100% | 126 days | 5.7 mo | 0% |
| Backtest-Exact | 6 MNQ | 100% | 85 days | 3.9 mo | 0% |
| Backtest-Exact | 10 MNQ | 100% | 52 days | 2.4 mo | 0% |
| **Realistic Live** | **4 MNQ** | **77%** | **324 days** | **14.7 mo** | **23%** |
| **Realistic Live** | **6 MNQ** | **88%** | **225 days** | **10.2 mo** | **12%** |
| **Realistic Live** | **10 MNQ** | **75%** | **115 days** | **5.2 mo** | **25%** |
| Stress Test | 6 MNQ | 0.7% | - | - | 99% |
| Stress Test | 10 MNQ | 3.0% | - | - | 97% |

> **Key Insight:** The backtest-exact scenario is unrealistically optimistic (losses are only -$2.74).
> In realistic live conditions (70% WR with real stop-outs), the optimal sizing is **6 MNQ** —
> it has the highest pass rate (88%) while keeping timeline manageable (10 months median).
> Going to 10 MNQ is faster (5 months) but increases blow-up risk to 25%.

### 3b. Funded Phase: 1-Year Survival & Payouts

| Scenario | Sizing | 1-Year Survival | Avg Payouts | Median Withdrawn | Avg Withdrawn |
|:---------|:-------|----------------:|------------:|-----------------:|--------------:|
| Backtest-Exact | 4 MNQ | 100% | 5.3 | $12,333 | $12,497 |
| Backtest-Exact | 6 MNQ | 100% | 6.1 | $19,245 | $19,473 |
| **Realistic Live** | **4 MNQ** | **96.5%** | **4.0** | **$3,829** | **$4,002** |
| **Realistic Live** | **6 MNQ** | **77.4%** | **4.0** | **$5,455** | **$5,771** |
| Realistic Live | 10 MNQ | 41.4% | 4.0 | $7,607 | $8,576 |
| Stress Test | 4 MNQ | 24.3% | 0.0 | $0 | $276 |

> **Critical Finding:** At realistic live performance, the **4 MNQ size has 96.5% 1-year survival**
> but only withdraws ~$4,000/year. The 6 MNQ size withdraws more (~$5,771) but has 22.6% blow-up risk.
> This is the core tension: more size = more income but higher ruin probability.

### 3c. Net Annual Income (After Eval Costs)

| Scenario | Sizing | Eval Cost | 1yr Withdrawal | Net/Year | Net/Month |
|:---------|:-------|----------:|---------------:|---------:|----------:|
| Backtest-Exact | 4 MNQ | $2,086 | $12,497 | **$10,411** | **$868** |
| Backtest-Exact | 6 MNQ | $1,421 | $19,473 | **$18,052** | **$1,504** |
| Backtest-Exact | 10 MNQ | $883 | $32,585 | **$31,701** | **$2,642** |
| Realistic Live | 4 MNQ | $6,883 | $4,002 | **-$3,021** | **-$252** |
| Realistic Live | 6 MNQ | $4,391 | $5,771 | **$77** | **$6** |
| Realistic Live | 10 MNQ | $2,765 | $8,576 | **$789** | **$66** |

> **Reality check:** At realistic live performance with a single account, the economics are
> break-even at best. The eval subscription cost ($359/month for 10+ months) eats most
> of the funded-phase income. This is why **multi-account** and **execution quality** matter.

---

## 4. The Gap: Backtest vs Live

The simulation exposes the critical gap between backtest and live performance:

| Factor | Backtest | Realistic Live | Impact |
|:-------|:---------|:---------------|:-------|
| Win Rate | 80% | 70% | 10% more losses (real stop-outs, not breakeven) |
| Winner Size | Full | 80% of backtest | Slippage, partial fills, early exits |
| Loser Size | -$2.74 (BE stop) | -$80 to -$140 | Real stop-outs vs perfect breakeven |
| Expectancy/trade | +$188 (2 MNQ) | +$60 (2 MNQ) | 3x degradation |

**The single biggest lever is reducing the gap between backtest and live performance:**

1. **Execution discipline** — Enter at signal, not 5 bars late
2. **Stop management** — Use actual backtest stops (VWAP - 0.4x IB), don't widen in fear
3. **No discretionary overrides** — If the Playbook says trade, trade. If it says wait, wait.
4. **Patient entry** — The system only trades 4.3 times/month. Do not force trades.

If live performance is closer to 75% WR with 10% degradation (instead of 70% / 20%), the
economics improve dramatically because the evaluation timeline shortens and fewer accounts blow up.

---

## 5. Recommended Prop Firm Game Plan

### Phase 1: Evaluation (Target: 3-6 months)

| Parameter | Setting | Rationale |
|:----------|:--------|:----------|
| **Sizing** | 6 MNQ | Best pass rate (88%) at realistic live performance |
| **Max risk/trade** | ~$480 (6 MNQ x 80 pt stop x $2/pt) | Well within $4,500 DD budget |
| **Trades/month** | ~4.3 (Playbook signals only) | No forcing, no discretionary adds |
| **$/trade expected** | +$179 (realistic) to +$565 (backtest) | Depends on execution quality |
| **Target timeline** | 16-25 trades to pass | 4-6 months at 4.3 trades/month |

**Evaluation Risk Management:**
- Max single-trade risk: $480 (10.7% of $4,500 drawdown)
- After 2 consecutive losses: reduce to 4 MNQ for next trade
- If drawdown reaches $2,000: reduce to 2 MNQ only
- If drawdown reaches $3,500: **STOP trading, forfeit eval, restart fresh**
- Never risk more than the remaining drawdown can absorb 2 max losses

**Consistency Rule Compliance (40%):**
- At $9,000 target with 16 trades: max single day = $3,600
- Our largest backtest win at 6 MNQ would be ~$1,800 (153 pts x 6 x $2)
- We're naturally compliant — no single Playbook trade can hit 40% of $9,000

### Phase 2: Funded — Early (First $4,600)

| Parameter | Setting | Rationale |
|:----------|:--------|:----------|
| **Sizing** | 4 MNQ (capped by scaling anyway) | Only 30 MNQ allowed at $0-$1,499 equity |
| **Goal** | Reach $4,600 to lock drawdown | After lock, drawdown floor freezes at $100 above start |
| **Approach** | Conservative growth, no big swings | One blown account = restart eval from scratch |

**Scaling Awareness:**
- At $0-$1,499 profit: max 30 MNQ (3 minis). Use 4 MNQ.
- At $1,500-$1,999: max 40 MNQ. Still use 4 MNQ.
- At $2,000+: max 50 MNQ. Can move to 6 MNQ.
- At $4,500+: full 120 MNQ. Drawdown locks. Scale to 6-8 MNQ.

### Phase 3: Funded — Payouts

| Parameter | Setting |
|:----------|:--------|
| **Sizing** | 6-8 MNQ (drawdown locked, full scaling available) |
| **Payout trigger** | Every 5 winning days ($250+ each) |
| **Payout amount** | Up to $5,000 (50% of profit, 90% split) |
| **Expected payout cycle** | Every 5-8 weeks at 4.3 trades/month |

**Payout Example (Realistic Live, 4 MNQ):**

| Payout # | Month | Balance Before | You Receive (90%) | Balance After |
|:---------|------:|---------------:|-------------------:|--------------:|
| 1 | 3.0 | $765 | $344 | $383 |
| 2 | 5.1 | $1,608 | $723 | $804 |
| 3 | 7.4 | $2,251 | $1,013 | $1,126 |
| 4 | 8.4 | $3,260 | $1,467 | $1,630 |
| 5 | 11.3 | $3,166 | $1,425 | $1,583 |
| **Total Year 1** | | | **$4,973** | |

---

## 6. Multi-Account Strategy

Since the Playbook generates identical signals regardless of how many accounts you trade,
running multiple accounts multiplies income with zero additional analysis time.

### Expected Annual Income by # of Accounts

Using **Realistic Live (70% WR) + 6 MNQ** sizing:

| Accounts | Annual Eval Cost | Annual Withdrawals | Net Income/Year | Net Income/Month |
|:---------|:----------------|:-------------------|:---------------|:----------------|
| 1 | $4,391 | $4,468 | ~$77 | ~$6 |
| 2 | $8,782 | $8,936 | ~$153 | ~$13 |
| 3 | $13,174 | $13,404 | ~$230 | ~$19 |
| 5 | $21,956 | $22,340 | ~$383 | ~$32 |

Using **Backtest Performance (80% WR) + 6 MNQ** (if execution matches backtest):

| Accounts | Annual Eval Cost | Annual Withdrawals | Net Income/Year | Net Income/Month |
|:---------|:----------------|:-------------------|:---------------|:----------------|
| 1 | $1,421 | $19,473 | **$18,052** | **$1,504** |
| 2 | $2,843 | $38,946 | **$36,103** | **$3,009** |
| 3 | $4,264 | $58,418 | **$54,155** | **$4,513** |
| 5 | $7,107 | $97,364 | **$90,258** | **$7,522** |

> **The entire business case hinges on execution quality.** At backtest-level performance,
> 3 accounts yield $54K/year. At degraded live performance, 3 accounts barely break even.
> The difference is 10% win rate and 20% winner size — that's the gap discipline must close.

---

## 7. Critical Risk Factors

### What Can Go Wrong

| Risk | Probability | Impact | Mitigation |
|:-----|:-----------|:-------|:-----------|
| Live WR drops to 60% | Medium | Account blown in weeks | Stop trading at 3 consecutive losses, review system |
| Regime change (bear market) | Medium | All-long system stops working | Monitor: if 5+ sessions with no acceptance, pause |
| Execution slippage | High | Winners shrink 10-30% | Use limit orders at VWAP, not market orders |
| Overtrading (forcing entries) | High | WR drops below 70% | Playbook rules are mechanical — follow them exactly |
| Scaling limits kill sizing | Certain | Can't use 6 MNQ until $2K+ profit | Start conservative, don't fight the scaling |
| 40% consistency violation | Low | Can't pass eval despite hitting $9K | Our max single-day P&L at 6 MNQ is ~$1,800 < $3,600 |

### The #1 Rule

**Do not add discretionary trades.** The Playbook trades ~4 times per month. That feels slow.
You will be tempted to take "obvious" setups outside the system. The standalone OF study proved
that pure order flow entries have 45% WR and negative expectancy. Every discretionary trade you
add degrades your edge toward coin-flip territory.

---

## 8. Pass Timeline Estimates

### How Long to Pass the $9,000 Evaluation?

| If Your Live Performance Is... | With 6 MNQ | With 10 MNQ |
|:-------------------------------|:-----------|:------------|
| Backtest-exact (80% WR, tiny losses) | 4 months (88% pass rate: 100%) | 2.5 months (100%) |
| Realistic (70% WR, real stops) | 10 months (88% pass rate) | 5 months (75% pass rate) |
| Degraded (65% WR) | Not viable | Not viable |

### After Passing: How Long to First Payout?

| Performance | Sizing | Months to First Payout |
|:------------|:-------|:----------------------|
| Backtest-exact | 4 MNQ | 1.9 months |
| Backtest-exact | 6 MNQ | 1.7 months |
| Realistic Live | 4 MNQ | 2.3 months |
| Realistic Live | 6 MNQ | 2.3 months |

### Total Timeline: Start to First Dollar

| Scenario | Eval Time | Funded to 1st Payout | Total |
|:---------|:----------|:---------------------|:------|
| Best case (backtest perf, 10 MNQ) | 2.5 months | 1.3 months | **~4 months** |
| **Expected (realistic, 6 MNQ)** | **10 months** | **2.3 months** | **~12 months** |
| Conservative (realistic, 4 MNQ) | 15 months | 2.3 months | ~17 months |

---

## 9. Honest Assessment

### The Math Works If...

1. **Live execution matches 75%+ WR** (vs 80% backtest)
2. **Winner degradation stays under 15%** (vs 20% modeled in realistic scenario)
3. **You don't overtrade** — 4 trades/month, not 20
4. **You run 2-3 accounts** to amortize the fixed cost of signal generation

### The Math Breaks If...

1. Live WR drops below 65% (stress test: 0.7% eval pass rate)
2. Losses are full stop-outs (-80 to -110 pts) instead of breakeven
3. You add discretionary trades that dilute the Playbook's 80% WR
4. Market regime shifts to sustained bear (all-long system stops working)

### Bottom Line

The Playbook + OF Quality Gate has a genuine statistical edge. The challenge is preserving that
edge through live execution. With disciplined execution and 2-3 simultaneous accounts, the
system can generate $3,000-4,500/month at backtest-level performance, or break-even at degraded
live performance. **The gap between those two outcomes is entirely about execution discipline.**

---

*Generated by sim_prop_firm.py — 10,000 Monte Carlo simulations per scenario*
*Trade data: v13 Playbook + OF Quality Gate, 62 sessions, MNQ*
*Prop firm rules sourced from [Tradeify Help Center](https://help.tradeify.co) and [SaveOnPropFirms](https://saveonpropfirms.com/blog/tradeify-select-guide)*
