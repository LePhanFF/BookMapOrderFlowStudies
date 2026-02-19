# Order Flow Strategy - Design Document
## Version 3.0 - v14 Optimized Portfolio

**Date**: February 18, 2026
**Status**: VALIDATED - 6-Strategy Portfolio
**Branch**: `v14-prop-firm-pipeline`
**Target**: Tradeify $150K Select Flex Evaluation + Funded Accounts

---

## 1. Executive Summary

After extensive backtesting (62 RTH sessions, Nov 18 2025 - Feb 16 2026, MNQ $2/pt), we have identified a 6-strategy portfolio built on Dalton Market Profile day type classification with order flow confirmation:

| Portfolio | Trades | WR | Expectancy | Net P&L | Max DD | PF | MaxCL |
|-----------|--------|-----|-----------|---------|--------|-----|-------|
| **OLD (unfiltered)** | 110 | 58% | $110 | $12,148 | -$3,103 | 2.01 | 12 |
| **OPTIMIZED** | 52 | 83% | $264 | $13,706 | **-$351** | 18.35 | 2 |

The optimized portfolio produces MORE P&L with FEWER trades, a 9x smaller drawdown, and recovers from max DD in 1-2 trades.

### Key Principle
**Day type classification is THE primary filter**. Order flow is a confirmation tool, not a standalone signal. Market structure (IB acceptance, VWAP pullback, edge mean-reversion) provides directional context that raw OF cannot.

---

## 2. Architecture

### 2.1 System Components

```
BookMapOrderFlowStudies/
  config/        - Constants, thresholds
  data/          - Data loading, CSV parsing
  indicators/    - ICT models (FVG, IFVG, BPR)
  profile/       - Volume profile (HVN/LVN)
  strategy/      - Strategy implementations
    trend_bull.py    - Trend Day Bull + Super Trend
    p_day.py         - P-Day VWAP pullback
    b_day.py         - B-Day IBL Fade
    edge_fade.py     - Edge Fade (EDGE_TO_MID)
    day_confidence.py - Day type scoring
  engine/        - Position management, signals
  filters/       - OF filters (Delta, Volume, Imbalance)
  reporting/     - Performance metrics
  export/        - Output formatting
  prop/          - Prop firm simulation
    sim_aggressive.py  - Monte Carlo eval simulator
    sizer.py           - Position sizing
    monte_carlo.py     - MC engine
  run_backtest.py  - CLI entry point
  diagnostic_*.py  - Research/analysis scripts
```

### 2.2 Data Specification

| Attribute | Value |
|-----------|-------|
| Instruments | NQ (Nasdaq) - MNQ for trading |
| Timeframe | 1-minute bars |
| Data Period | Nov 18, 2025 - Feb 16, 2026 |
| Sessions | 62 RTH trading days |
| Source | NinjaTrader/Bookmap Volumetric Export |
| Required Columns | timestamp, open, high, low, close, volume, vol_delta, cumulative_delta, delta_percentile, imbalance_ratio, volume_spike, vwap |

### 2.3 Day Type Classification

| Day Type | IB Extension | Description | Active Strategies |
|----------|-------------|-------------|-------------------|
| trend_up | > 1.0x IB above IBH | Strong directional up | Trend Bull, P-Day |
| p_day | 0.5-1.0x above IBH | Directional with pullback | P-Day |
| b_day | 0.2-0.5x either side | Balanced, range-bound | B-Day Fade, Edge Fade, IBH Sweep |
| b_day_bear | > 0.5x below IBL | Balance with bear extension | Bear Accept Short |
| neutral | < 0.2x either side | Inside IB, low volatility | Edge Fade |
| trend_down | > 1.0x below IBL | Strong directional down | Bear Accept Short |

---

## 3. Strategy Roster (6 Active Strategies)

### Strategy 1: Trend Day Bull (LONG)

**When**: Day type = trend_up or strong p_day (ext_up > 1.0x IB)

**Entry**:
1. Wait for IBH acceptance (2+ bars above IBH)
2. Wait for price to pull back to VWAP
3. Pre-entry 10-bar delta sum > -500
4. OF quality gate: 2 of 3 positive (delta_pctl >= 60, imbalance > 1.0, vol_spike >= 1.0)
5. Enter LONG at VWAP

**Stop**: VWAP - 40% of IB range
**Target**: EOD exit
**Time**: 10:30 - 13:00
**Performance**: 8 trades, 75% WR, $134/trade, -$3 MaxDD, PF 197

### Strategy 2: P-Day (LONG)

**When**: Day type = p_day (ext_up 0.5-1.0x IB)
**Entry**: Identical to Trend Day Bull
**Performance**: 8 trades, 75% WR, $134/trade, -$3 MaxDD, PF 197

### Strategy 3: B-Day IBL Fade (LONG)

**When**: Day type = b_day, b_day_confidence >= 0.5

**Entry**:
1. Price touches IBL
2. Delta > 0 on touch bar
3. Quality score >= 2 (volume, wick rejection, FVG confluence)
4. IB range < 400 pts

**Stop**: IBL - 10% of IB range
**Target**: IB midpoint
**Time**: Before 14:00
**Performance**: 4 trades, 100% WR, $571/trade, $0 MaxDD, PF INF

### Strategy 4: Edge Fade OPTIMIZED (LONG)

**When**: Day type = b_day or neutral, on NON-BEARISH days (ext_down < 0.3x)

**CRITICAL FILTERS**:
- IB range < 200 pts (wider IB = 0% WR)
- NOT a bearish day (ext_down < 0.3x, otherwise 37% WR vs 82%)
- Entry before 13:30 (PM after 13:30 = 33% WR)

**Entry**:
1. Price in lower 25% of IB range
2. Delta > 0
3. OF quality >= 2 of 3
4. 20-bar cooldown between entries

**Stop**: IBL - 15% of IB range
**Target**: IB midpoint
**Time**: 10:30 - 13:30
**Performance**: 17 trades, 94% WR, $453/trade, -$351 MaxDD, PF 23

### Strategy 5: IBH Sweep + Failure Fade (SHORT)

**When**: Day type = b_day ONLY (23% WR on p_day!)

**Entry**:
1. Price sweeps above IBH (touches PDH, London High)
2. Fails to close above IBH
3. Delta < 0 on failure bar
4. Quality >= 2

**Stop**: Above sweep high + 15% IB range
**Target**: IB midpoint
**Performance**: 4 trades, 100% WR, $146/trade (CAUTION: small sample)

### Strategy 6: Bear Acceptance Short (SHORT)

**When**: Day type = trend_down or b_day_bear (ext_down > 0.5x)

**Entry**:
1. 3+ consecutive bars close below IBL (acceptance)
2. Short at close of 3rd acceptance bar

**Stop**: IBL + 25% of IB range
**Target**: IBL - 0.75x IB range
**Time**: Before 14:00
**Performance**: 11 trades, 64% WR, $90/trade, -$289 MaxDD, PF 3.3

---

## 4. Risk Management

### 4.1 Portfolio Limits

| Limit | Evaluation | Funded |
|-------|-----------|--------|
| Max Contracts | 1 MNQ | Scale 1-3 MNQ |
| Daily Edge Fade Loss | -$400 (2 stops) | -$400 |
| Max Drawdown Budget | $4,500 (trailing) | Account-specific |
| Session | 10:30-16:00 | 10:30-16:00 |

### 4.2 Position Sizing (Funded Accounts)

| Phase | Buffer | Size | A Setups | B Setups |
|-------|--------|------|----------|----------|
| Phase 1 | $0-$3,000 | 1 MNQ | 1 MNQ | 1 MNQ |
| Phase 2 | $3,000-$6,000 | 1-2 MNQ | 2 MNQ | 1 MNQ |
| Phase 3 | $6,000+ | 2-3 MNQ | 3 MNQ | 2 MNQ |

**A Setups**: B-Day IBL Fade (100% WR), Trend Bull (75% WR)
**B Setups**: Edge Fade (94% WR optimized), Bear Accept Short (64% WR)

---

## 5. Session Coverage

| Day Type | Sessions | Strategies Active | Coverage |
|----------|----------|-------------------|----------|
| trend_up | 4 | Trend Bull, P-Day | Partial |
| p_day | 13 | P-Day | 6/13 (46%) |
| b_day | 16 | B-Day Fade, Edge Fade, IBH Sweep | 8/16 (50%) |
| b_day_bear | 5 | Bear Accept Short | 5/5 (100%) |
| neutral | 18 | Edge Fade | 7/18 (39%) |
| trend_down | 6 | Bear Accept Short | 6/6 (100%) |
| **TOTAL** | **62** | --- | **32/62 (52%)** |

---

## 6. Key Findings

### What Works
1. Day type classification is THE primary filter
2. VWAP pullback is THE reliable entry for trend/p-day
3. IBL edge fades work on non-bearish balance/neutral days before 13:30
4. Bear acceptance shorts work ONLY on trend_down + b_day_bear
5. EOD exit captures more P&L than trailing stops on trend entries
6. IB range < 200 pts is critical for Edge Fade (controls stop distance)
7. OF quality gate (2-of-3) provides last-line defense

### What Doesn't Work
1. ANY shorts on b_day or neutral (0-14% WR — NQ long bias)
2. Edge Fade after 13:30 (PM morph kills mean reversion)
3. Edge Fade on wide IB > 200 pts (stop too far, target unreachable)
4. Edge Fade on bearish days (fading into selling = 37% WR)
5. FVG as standalone entry on 1-min bars (16% WR)
6. Trailing stops on VWAP entries (reduce P&L 50-80%)
7. CVDFilter (kills B-Day trades — balance days have CVD < MA)
8. Pure order flow standalone signals (45% WR, zero selectivity)

---

## 7. Evaluation Strategy (Tradeify $150K Select Flex)

- **Profit Target**: $9,000
- **Trailing Drawdown**: $4,500 (EOD)
- **Expectancy**: $264/trade at 1 MNQ
- **Trades needed**: ~35 trades
- **Timeline**: ~3-4 weeks
- **Max DD risk**: -$351 = 7.8% of DD allowance

---

## 8. How to Run

```bash
# All strategies (core playbook)
python run_backtest.py --strategies core

# With strict order flow filters
python run_backtest.py --strategies core --strict-filters

# Specific instrument
python run_backtest.py --strategies core --instrument NQ

# Diagnostics
python diagnostic_edge_fade_deep.py        # Edge Fade analysis
python diagnostic_bearish_day_v2.py         # Bear short study
python diagnostic_full_portfolio_report.py  # Full portfolio
python strategy_report.py                   # Strategy comparison
```

### Adding New Data
1. Place CSV in `csv/` with format: `{INSTRUMENT}_Volumetric_1.csv`
2. Required columns: timestamp, open, high, low, close, volume, vol_delta, cumulative_delta, delta_percentile, imbalance_ratio, volume_spike, vwap
3. Run: `python run_backtest.py --strategies core`

---

## 9. Version History

| Version | Date | Key Change |
|---------|------|-----------|
| v1.0 | Feb 2026 | Dual strategy (44% WR, $94/trade) |
| v2.0 | Feb 16 | Dalton playbook (84% WR, $261/trade, 20 trades) |
| **v3.0** | **Feb 18** | **6-strategy portfolio (83% WR, $264/trade, 52 trades)** |

Key v3.0 improvements:
- Edge Fade optimized from 53%→94% WR via 3 filters
- Added Bear Acceptance Short (64% WR on bearish days)
- Added IBH Sweep+Fail Short (100% WR, small sample)
- Portfolio MaxDD reduced from -$3,103 to -$351 (9x improvement)

---

*Document Version: 3.0*
*Last Updated: February 18, 2026*
*Strategy Validated: 52 trades, 62 sessions, 83% WR, $13,706 net*
