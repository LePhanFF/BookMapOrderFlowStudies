# Constants & Multipliers Curve-Fitting Audit

Generated: 2026-02-20

## Summary

| Category | Count | % |
|---|---|---|
| STRUCTURAL (market/Dalton framework) | 19 | 14% |
| MEASURED (adaptive/rolling) | 2 | 2% |
| SEMI-ARBITRARY (logical but optimized) | 47 | 35% |
| ARBITRARY (magic numbers, high risk) | 66 | 49% |
| **TOTAL** | **134** | |

## Tier 1: Critical Risk (directly control entry/exit/P&L)

| Constant | File | Value | Problem | Fix |
|---|---|---|---|---|
| `SWEEP_THRESHOLD_PTS` | or_reversal.py | 20 pts | Absolute, not scaled | % of EOR range |
| `VWAP_ALIGNED_PTS` | or_reversal.py | 20 pts | Absolute, not scaled | % of EOR/IB range |
| `EDGE_FADE_IB_EXPANSION_RATIO` | edge_fade.py | 1.2 | 5-session lookback is fragile | 10-20 session lookback |
| `EDGE_FADE_MAX_BEARISH_EXT` | edge_fade.py | 0.3 | Backtest-derived cutoff | Regime-conditioned WR |
| `BEAR_ACCEPT_MIN_EXT` | bear_accept.py | 0.4 | Gates 100% of bear entries | Percentile of recent extensions |
| `150` / `250` regime thresholds | backtest.py | 150/250 pts | NQ-specific absolute | Percentile of rolling IB dist |
| `STOP_VWAP_BUFFER` | trend_bull/p_day | 0.40 | Sets risk on every trade | ATR-scaled or rolling bar range |
| `PRE_DELTA_RATIO` | trend_bull/p_day | 3.0 | Admitted backtest conversion | Rolling percentile of delta dist |

## Tier 2: Important (filters/confidence thresholds)

| Constant | File | Value | Problem | Fix |
|---|---|---|---|---|
| `0.375` confidence | trend_bull.py | 0.375 | 3 decimal places = over-optimization | Round to 0.4 |
| `0.5` b_day confidence | b_day.py | 0.5 | All B-Day trades gated | Regime-sliding threshold |
| `0.3` b_day confidence | ibh_sweep.py | 0.3 | Different from b_day's 0.5 | Unify |
| `EDGE_ZONE_PCT` | edge_fade.py | 0.25 | Arbitrary boundary | IB quartile |
| `DRIVE_THRESHOLD` | or_reversal.py | 0.4 | Opening drive classification | Rolling dist of opening moves |
| `EDGE_FADE_IB_LOOKBACK` | edge_fade.py | 5 | Too short, noisy | Increase to 15-20 |

## Tier 3: Lower Risk (cooldowns, lookbacks)

| Constant | File | Value | Fix |
|---|---|---|---|
| `PYRAMID_COOLDOWN_BARS` | constants.py | 12 | Express as % of IB period |
| `BDAY_COOLDOWN_BARS` | constants.py | 30 | Tie to session time remaining |
| `EDGE_FADE_COOLDOWN_BARS` | edge_fade.py | 20 | Same |
| `MORPH_TO_TREND_BREAKOUT_POINTS` | constants.py | 30 pts | IB-scaled (0.2x IB) |
| `PM_MORPH_BREAKOUT_POINTS` | constants.py | 15 pts | IB-scaled |
| `MORPH_TO_TREND_TARGET_POINTS` | constants.py | 150 pts | IB-scaled or ATR-scaled |

## Key Findings

1. **9 absolute-point constants** are the biggest risk -- they break on other instruments and shift with volatility regimes
2. **Edge Fade has 16 magic numbers** -- highest curve-fit risk of any strategy
3. **Confidence thresholds inconsistently calibrated** (0.3, 0.375, 0.5 across strategies for same scorer)
4. **Stop buffers are IB-scaled (good)** but magnitudes (0.10-0.50) are undocumented
5. **day_confidence.py has 20+ hand-tuned thresholds** that interact with each other
6. **Time gates (13:00-14:00) are low risk** but could be expressed as % of session

## Recommended Actions (by effort/impact)

1. Convert 9 absolute-point constants to IB/ATR-scaled (low effort, high impact)
2. Extend Edge Fade lookback from 5 to 15-20 sessions (one-line change)
3. Unify confidence thresholds (0.4 trend, 0.5 b_day)
4. Run sensitivity analysis: sweep Tier 1 constants +/-25%, measure P&L stability
5. Validate on 248-session combined dataset when re-exported data is ready
