"""
Order Flow + Auction Market Strategy
=====================================
A/B Testing Framework for Prop Firm Trading

Target: TradeDay $150K Account
Objective: First payout of $9,000
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, time
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = '/home/lphan/jupyterlab/BookMapOrderFlowStudies/csv/'
ACCOUNT_SIZE = 150000
MAX_DRAWDOWN = 4000
MAX_RISK_PER_TRADE = 400
MAX_DAY_LOSS = 2000
TARGET_PROFIT = 9000

# Trading hours (US Eastern)
SESSION_START = time(9, 30)
SESSION_END = time(15, 0)
IB_START = time(9, 30)
IB_END = time(10, 30)

print("=" * 60)
print("ORDER FLOW + AUCTION MARKET STRATEGY")
print("=" * 60)
print(f"Account: ${ACCOUNT_SIZE:,}")
print(f"Max Drawdown: ${MAX_DRAWDOWN:,}")
print(f"Target Profit: ${TARGET_PROFIT:,}")
print("=" * 60)
