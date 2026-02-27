"""
Named constants for the backtest system.
All magic numbers from the codebase consolidated here.
"""

from datetime import time

# ── Session Times (US Eastern) ──────────────────────────────────
RTH_START = time(9, 30)
RTH_END = time(16, 0)           # FIXED: was 15:00 in old data_loader.py
ETH_START = time(18, 0)
IB_END = time(10, 30)           # Initial Balance ends 60 min after open
IB_BARS_1MIN = 60               # Number of 1-min bars in IB period
LONDON_CLOSE = time(11, 30)     # No new pyramids after this
PM_SESSION_START = time(13, 0)  # PM management begins
EOD_CUTOFF = time(15, 30)       # Force close before this

# ── IB Extension Thresholds (Dalton) ────────────────────────────
IB_EXT_WEAK = 0.5               # < 0.5x = weak
IB_EXT_MODERATE = 1.0           # 0.5-1.0x = moderate
IB_EXT_STRONG = 2.0             # 1.0-2.0x = strong, > 2.0x = super

# ── Day Type Extension Thresholds ───────────────────────────────
# Extension from IB midpoint / IB range
DAY_TYPE_TREND_THRESHOLD = 1.0
DAY_TYPE_P_DAY_THRESHOLD = 0.5
DAY_TYPE_B_DAY_THRESHOLD = 0.2

# ── Risk Management Defaults ───────────────────────────────────
DEFAULT_ACCOUNT_SIZE = 150_000
DEFAULT_MAX_DRAWDOWN = 4_000
DEFAULT_MAX_RISK_PER_TRADE = 400
DEFAULT_MAX_DAY_LOSS = 2_000
DEFAULT_MAX_TRADES_PER_DAY = 10
DEFAULT_MAX_CONTRACTS = 30

# ── Execution Defaults ─────────────────────────────────────────
DEFAULT_SLIPPAGE_TICKS = 1      # 1 tick of slippage per side

# ── Feature Engineering ─────────────────────────────────────────
DELTA_ROLLING_WINDOW = 20
DELTA_PERCENTILE_THRESHOLD = 80
VOLUME_ROLLING_WINDOW = 20
CVD_EMA_SPAN = 20

# ── Strategy Parameters ────────────────────────────────────────
ACCEPTANCE_MIN_BARS = 2         # Minimum bars for acceptance confirmation (bullish)
ACCEPTANCE_MIN_BARS_BEAR = 3    # More conservative for bear (long bias in market)
ACCEPTANCE_IDEAL_BARS = 6       # Ideal bars for high-confidence acceptance
PYRAMID_COOLDOWN_BARS = 12      # Bars between pyramid entries
MAX_PYRAMIDS_STANDARD = 2
MAX_PYRAMIDS_AGGRESSIVE = 3
TREND_TARGET_IB_MULTIPLE = 2.5  # Target = entry + 2.5 * IB range
TREND_STOP_IB_BUFFER = 0.3     # Stop = IB edge + 30% of IB range (wider for trend days)
EMA_PULLBACK_THRESHOLD = 0.15  # Within 15% of IB range from EMA

# ── B-Day Strategy ──────────────────────────────────────────────
BDAY_COOLDOWN_BARS = 30
BDAY_STOP_IB_BUFFER = 0.1      # Stop 10% beyond IB edge
BDAY_POOR_STRUCTURE_LOOKBACK = 10

# ── Morph Strategy ──────────────────────────────────────────────
# Morph thresholds (IB-scaled ratios, not absolute points)
PM_MORPH_BREAKOUT_RATIO = 0.10        # Min breakout = 10% of AM range (was 15 pts, ~10% of median IB)
MORPH_TO_TREND_BREAKOUT_RATIO = 0.20  # Min breakout = 20% of IB range (was 30 pts, ~20% of median IB)
MORPH_TO_TREND_TARGET_RATIO = 1.0     # Target = 1.0x IB range from entry (was 150 pts, ~1x median IB)

# ── Value Area ──────────────────────────────────────────────────
VALUE_AREA_PCT = 0.70

# ── VWAP Breach Threshold ──────────────────────────────────────
VWAP_BREACH_RATIO = 0.07       # VWAP breach = 7% of IB range (was 10 pts absolute)
VWAP_BREACH_POINTS = 10        # Legacy fallback for warmup (used when IB range unavailable)

# ── Balance Signal Strategy ───────────────────────────────────
BALANCE_MIN_QUALITY_SCORE = 4       # Minimum total score (out of 10) to fire signal
BALANCE_COOLDOWN_BARS = 20          # Bars between entries per mode
BALANCE_LAST_ENTRY_TIME = time(13, 30)  # No entries after 13:30 (PM morph kills MR)

# VA Edge Fade mode
BALANCE_VA_STOP_BUFFER_PCT = 0.15   # Stop: VAH/VAL + 15% of VA range beyond edge
BALANCE_VA_MIN_RR = 1.0             # Minimum reward/risk ratio

# Wide IB Reclaim mode
BALANCE_IB_RECLAIM_MIN = 350        # Minimum IB range (pts) for wide IB reclaim
BALANCE_IB_RECLAIM_MAX = 500        # Maximum IB range (pts) for wide IB reclaim
BALANCE_IB_STOP_BUFFER_PCT = 0.10   # Stop: IB edge - 10% IB range

# Dalton acceptance
BALANCE_ACCEPT_EARLY = 2            # 2 consecutive closes = early acceptance
BALANCE_ACCEPT_INSIDE = 6           # 6 consecutive closes = inside (30-min) acceptance

# HVN/LVN detection
BALANCE_HVN_PERCENTILE = 80         # Volume bins above this = HVN
BALANCE_LVN_PERCENTILE = 20         # Volume bins below this = LVN
BALANCE_EDGE_PROXIMITY_PCT = 0.10   # HVN/LVN within 10% of VA range from edge
