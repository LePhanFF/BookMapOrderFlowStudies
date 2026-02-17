# Order Flow Strategy - Project Roadmap

## Status: IN PROGRESS

Last Updated: 2026-02-16

---

## Completed Tasks

### Phase 1: Design & Planning ✅

| Task | Status | Notes |
|------|--------|-------|
| quant-study.md | ✅ | Comprehensive prompt created |
| design-document.md | ✅ | Strategy design v1.3 |
| ict.md | ✅ | ICT/SMC concepts parked |

### Phase 2: Data Pipeline ✅

| Task | Status | Notes |
|------|--------|-------|
| data_loader.py | ✅ | Loads NQ/ES data |
| RTH filter | ✅ | 9:30-15:00 session |
| Order flow features | ✅ | Delta, CVD, imbalances |
| Day type classification | ✅ | IB-based classification |
| IB features | ✅ | Opening range features |

### Phase 3: Testing 🔄

| Task | Status | Notes |
|------|--------|-------|
| test_data_loader.py | 🔄 | Unit tests created |
| Run tests | ⏳ | Need to execute |

---

## Current Work

### Running Unit Tests
- Testing data_loader.py functions
- Validating calculations

---

## Pending Tasks

### Phase 4: Backtest Engine
- [ ] Build backtest engine
- [ ] Implement signal generation
- [ ] Implement risk management

### Phase 5: A/B Testing
- [ ] Layer 0 baseline (delta only)
- [ ] Layer 1 + IB direction
- [ ] Layer 2 + Day type
- [ ] Layer 3 + VWAP
- [ ] Layer 4 + TPO
- [ ] Layer 5 + HTF
- [ ] Layer 6 + FVG

### Phase 6: Stop Comparison
- [ ] ATR stops
- [ ] LVN/HVN stops
- [ ] FVG stops
- [ ] Hybrid approach

### Phase 7: Optimization
- [ ] Parameter sweeps
- [ ] Timeframe comparison (1-min vs 5-min)
- [ ] Direction analysis

### Phase 8: Reports
- [ ] Layer comparison report
- [ ] Stop comparison report
- [ ] Equity curves
- [ ] Final recommendations

---

## Key Files

| File | Purpose |
|------|---------|
| data_loader.py | Data loading & features |
| test_data_loader.py | Unit tests |
| order_flow_strategy.py | Main strategy |
| backtest_engine.py | Backtest logic |
| reports.py | Generate reports |

---

## Data Summary

| Instrument | Rows | Date Range | Sessions |
|------------|------|------------|----------|
| NQ | 83,322 | Nov 2025 - Feb 2026 | 62 |
| ES | 82,543 | Nov 2025 - Feb 2026 | 62 |

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Account | $150,000 |
| Max Drawdown | $4,000 |
| Max Risk/Trade | $400 |
| Target Profit | $9,000 |
| Session | US RTH 9:30-15:00 |

---

## Next Steps

1. Run unit tests
2. Fix any failures
3. Build backtest engine
4. Run Layer 0 baseline
