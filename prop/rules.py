"""
Tradeify Select Flex $150K - Rule Constants

All parameters from Tradeify's Select Flex program for the $150K account tier.
Source: https://tradeify.com (verified Feb 2026)
"""

# --- Account Setup ---
STARTING_BALANCE = 150_000.0
EVAL_COST = 219.0  # Monthly subscription for $150K tier

# --- Evaluation Phase ---
EVAL_PROFIT_TARGET = 9_000.0  # Must reach $159,000 equity
EVAL_MAX_TRAILING_DD = 4_500.0  # EOD trailing drawdown limit
# No daily loss limit in eval
# No time limit in eval

# --- Funded Phase ---
FUNDED_MAX_TRAILING_DD = 4_500.0  # Same as eval
FUNDED_DD_LOCK_THRESHOLD = 100.0  # DD floor locks at start_balance + $100
CONSISTENCY_RULE_PCT = 0.40  # No single day > 40% of total profit
PROFIT_SPLIT = 0.90  # 90/10 split (trader keeps 90%)

# --- Payout Rules ---
PAYOUT_MIN_WINNING_DAYS = 5  # Need 5 winning days ($250+ each)
PAYOUT_WINNING_DAY_MIN = 250.0  # Minimum daily P&L to count as winning day
PAYOUT_MAX_AMOUNT = 5_000.0  # Max $5K per payout
PAYOUT_PROCESSING_DAYS = 7  # 7 calendar days to process
PAYOUT_CYCLE_TO_LIVE = 6  # Transition to live trading after 6th payout cycle

# --- Scaling Tiers (MNQ contracts) ---
# Tradeify lists limits in MINIS (NQ). We trade MICROS (MNQ) at 10:1 ratio.
# Equity above starting balance -> max MNQ contracts
SCALING_TIERS = [
    (0, 30),         # $0-$1,499: 3 minis = 30 MNQ
    (1_500, 40),     # $1,500-$1,999: 4 minis = 40 MNQ
    (2_000, 50),     # $2K-$2,999: 5 minis = 50 MNQ
    (3_000, 80),     # $3K-$4,499: 8 minis = 80 MNQ
    (4_500, 120),    # $4,500+: 12 minis = 120 MNQ
]

# --- Pipeline Budget ---
MAX_FUNDED_ACCOUNTS = 5  # Target 5 funded accounts
MAX_RESETS_PER_SLOT = 2  # Budget 2 resets per account slot
MAX_TOTAL_EVAL_ATTEMPTS = MAX_FUNDED_ACCOUNTS * (1 + MAX_RESETS_PER_SLOT)  # 15 total

# --- Risk Sizing (Eval Phase - Aggressive) ---
# Default eval sizing (can be overridden by sim scenarios)
EVAL_TYPE_A_RISK = 1_500.0  # $1,500 risk for Type A setups (B-Day, high-conf trend)
EVAL_TYPE_B_RISK = 750.0  # $750 risk for Type B setups (standard VWAP)
EVAL_DD_REDUCTION_THRESHOLD = 2_000.0  # Halve risk when DD > $2,000
EVAL_DD_STOP_THRESHOLD = 3_500.0  # Stop trading when DD > $3,500

# Eval sizing schemes for simulation comparison
EVAL_SIZING_SCHEMES = {
    'Conservative (A=$1K, B=$450)': {'type_a': 1_000, 'type_b': 450},
    'Balanced (A=$1.5K, B=$750)': {'type_a': 1_500, 'type_b': 750},
    'Aggressive (A=$2K, B=$1K)': {'type_a': 2_000, 'type_b': 1_000},
    'Full Send (A=$2.5K, B=$1.5K)': {'type_a': 2_500, 'type_b': 1_500},
}

# --- Risk Sizing (Funded Phase - Conservative) ---
FUNDED_PHASE1_RISK = 200.0  # Building buffer (equity < start + $1,000)
FUNDED_PHASE2_RISK = 300.0  # DD locked (equity >= start + $1,000)
FUNDED_PHASE3_TYPE_B_RISK = 450.0  # Buffer built (equity >= start + $2,000)
FUNDED_PHASE3_TYPE_A_RISK = 600.0  # Type A in funded phase 3
FUNDED_DD_STOP_THRESHOLD = 3_000.0  # Stop trading when DD > $3,000 in funded
FUNDED_BUFFER_PHASE2 = 1_000.0  # Equity above start to enter phase 2
FUNDED_BUFFER_PHASE3 = 2_000.0  # Equity above start to enter phase 3

# --- MNQ Instrument ---
MNQ_POINT_VALUE = 2.0  # $2 per full NQ point
MNQ_TICK_SIZE = 0.25
MNQ_COMMISSION_PER_SIDE = 0.62  # $0.62 per contract per side
MNQ_COMMISSION_RT = MNQ_COMMISSION_PER_SIDE * 2  # $1.24 round-trip

# --- Simulation Parameters ---
TRADE_FREQUENCY = 38 / 62  # 61.3% - from v14 backtest (38 sessions with trades / 62 total)
# v14 added Edge Fade (EDGE_TO_MID) which fires on b_day/neutral sessions
# Average ~1.8 trades/session (110 trades / 62 sessions raw, 42 deduped / 62)
# Multi-trade sessions: ~10% of active sessions have 2+ distinct entries
MULTI_TRADE_PROBABILITY = 0.10  # Probability of 2nd trade on active sessions
TRADING_DAYS_PER_MONTH = 21
MONTE_CARLO_RUNS = 10_000
SIM_DURATION_DAYS = 504  # 2 years of trading days


def get_max_contracts(equity_above_start: float) -> int:
    """Return max MNQ contracts allowed at current equity level."""
    max_contracts = SCALING_TIERS[0][1]  # Default: 3
    for threshold, contracts in SCALING_TIERS:
        if equity_above_start >= threshold:
            max_contracts = contracts
    return max_contracts
