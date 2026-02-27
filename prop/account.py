"""
PropAccount - Single prop firm account state machine.

Tracks a single Tradeify account through its lifecycle:
  EVAL -> FUNDED -> PAYOUT_READY -> BLOWN

Handles EOD trailing drawdown, DD floor locking, winning day counting,
scaling tier enforcement, and payout eligibility.
"""
from enum import Enum
from typing import Optional
from prop import rules


class Phase(Enum):
    EVAL = 'EVAL'
    FUNDED = 'FUNDED'
    PAYOUT_READY = 'PAYOUT_READY'
    BLOWN = 'BLOWN'


class PropAccount:
    """Single prop firm account with full lifecycle tracking."""

    def __init__(self, account_id: int, start_day: int = 0):
        self.account_id = account_id
        self.start_day = start_day
        self.phase = Phase.EVAL

        # Balance tracking
        self.starting_balance = rules.STARTING_BALANCE
        self.equity = self.starting_balance
        self.high_water_mark = self.starting_balance
        self.max_trailing_dd = rules.EVAL_MAX_TRAILING_DD

        # Funded phase DD floor
        self.dd_locked = False
        self.dd_floor = 0.0  # Set when DD locks in funded phase

        # Daily tracking
        self.daily_pnl = 0.0
        self.winning_days = 0
        self.total_winning_days = 0  # Lifetime (doesn't reset on payout)
        self.trading_days = 0

        # Trade tracking
        self.total_trades = 0
        self.total_pnl = 0.0
        self.best_day_pnl = 0.0  # For consistency rule

        # Payout tracking
        self.payouts_completed = 0
        self.total_payouts_amount = 0.0

        # Timeline
        self.days_elapsed = 0
        self.eval_days = 0  # Days spent in eval
        self.funded_days = 0  # Days spent in funded

        # Resets for this account slot
        self.resets = 0

    @property
    def is_active(self) -> bool:
        return self.phase not in (Phase.BLOWN,)

    @property
    def is_eval(self) -> bool:
        return self.phase == Phase.EVAL

    @property
    def is_funded(self) -> bool:
        return self.phase in (Phase.FUNDED, Phase.PAYOUT_READY)

    @property
    def equity_above_start(self) -> float:
        return self.equity - self.starting_balance

    @property
    def current_dd(self) -> float:
        """Current drawdown from high water mark."""
        return self.high_water_mark - self.equity

    @property
    def dd_limit(self) -> float:
        """The DD level at which account blows."""
        if self.dd_locked:
            return self.dd_floor
        return self.high_water_mark - self.max_trailing_dd

    @property
    def funded_risk_phase(self) -> int:
        """Which funded sizing phase (1/2/3) based on buffer."""
        if not self.is_funded:
            return 0
        buffer = self.equity_above_start
        if buffer >= rules.FUNDED_BUFFER_PHASE3:
            return 3
        elif buffer >= rules.FUNDED_BUFFER_PHASE2:
            return 2
        else:
            return 1

    def get_max_contracts(self) -> int:
        """Max MNQ contracts based on current scaling tier."""
        return rules.get_max_contracts(max(0, self.equity_above_start))

    def on_trade(self, net_pnl: float, setup_type: str = '') -> bool:
        """
        Process a completed trade. Returns True if account still active.

        Args:
            net_pnl: Net P&L in dollars (after commission/slippage)
            setup_type: Signal setup type for tracking
        """
        self.equity += net_pnl
        self.daily_pnl += net_pnl
        self.total_pnl += net_pnl
        self.total_trades += 1

        # Check if blown (intra-day check - some firms check real-time)
        if self._check_blown():
            return False

        return True

    def process_eod(self) -> dict:
        """
        End-of-day processing. Updates HWM, checks winning day, checks blown.

        Returns dict with EOD status info.
        """
        self.days_elapsed += 1
        if self.is_eval:
            self.eval_days += 1
        elif self.is_funded:
            self.funded_days += 1

        # Only count as trading day if we had trades
        had_trades = self.daily_pnl != 0.0
        if had_trades:
            self.trading_days += 1

        result = {
            'account_id': self.account_id,
            'phase': self.phase.value,
            'equity': self.equity,
            'daily_pnl': self.daily_pnl,
            'dd': self.current_dd,
            'had_trades': had_trades,
        }

        # Update HWM at EOD (Tradeify uses EOD trailing DD)
        if self.equity > self.high_water_mark:
            self.high_water_mark = self.equity

        # Check winning day ($250+)
        if self.daily_pnl >= rules.PAYOUT_WINNING_DAY_MIN:
            self.winning_days += 1
            self.total_winning_days += 1

        # Track best day for consistency rule
        if self.daily_pnl > self.best_day_pnl:
            self.best_day_pnl = self.daily_pnl

        # Check blown at EOD
        if self._check_blown():
            result['blown'] = True
        else:
            result['blown'] = False

        # Check eval passed
        if self.is_eval and self._check_eval_passed():
            result['eval_passed'] = True
        else:
            result['eval_passed'] = False

        # Check funded DD lock
        if self.is_funded and not self.dd_locked:
            if self.equity_above_start >= rules.FUNDED_DD_LOCK_THRESHOLD:
                self.dd_locked = True
                self.dd_floor = self.starting_balance + rules.FUNDED_DD_LOCK_THRESHOLD
                result['dd_locked'] = True

        # Check payout eligibility
        if self.is_funded and self.winning_days >= rules.PAYOUT_MIN_WINNING_DAYS:
            if self.phase != Phase.PAYOUT_READY:
                self.phase = Phase.PAYOUT_READY
            result['payout_ready'] = True
        else:
            result['payout_ready'] = False

        # Reset daily tracking
        self.daily_pnl = 0.0

        return result

    def transition_to_funded(self):
        """Transition from EVAL to FUNDED after passing."""
        if self.phase != Phase.EVAL:
            raise ValueError(f"Cannot transition to FUNDED from {self.phase}")
        self.phase = Phase.FUNDED
        # Reset HWM to current equity for funded phase
        self.high_water_mark = self.equity
        self.max_trailing_dd = rules.FUNDED_MAX_TRAILING_DD
        self.winning_days = 0
        self.best_day_pnl = 0.0

    def request_payout(self) -> float:
        """
        Process a payout. Returns the payout amount.
        Deducts from equity, resets winning days.
        """
        if self.phase != Phase.PAYOUT_READY:
            return 0.0

        # Calculate payout amount
        profit = self.equity - self.starting_balance
        if profit <= 0:
            return 0.0

        # Check consistency rule: no single day > 40% of total profit
        if self.best_day_pnl > profit * rules.CONSISTENCY_RULE_PCT:
            # Consistency rule violated - cap payout
            pass  # Still allow payout, just noting the risk

        # Payout is 90% of profit, capped at $5K
        payout_amount = min(profit * rules.PROFIT_SPLIT, rules.PAYOUT_MAX_AMOUNT)

        # Deduct from equity
        self.equity -= payout_amount
        self.total_payouts_amount += payout_amount
        self.payouts_completed += 1

        # Reset for next payout cycle
        self.winning_days = 0
        self.best_day_pnl = 0.0

        # Update HWM after payout withdrawal
        self.high_water_mark = self.equity

        # Return to FUNDED phase
        self.phase = Phase.FUNDED

        return payout_amount

    def _check_blown(self) -> bool:
        """Check if account has exceeded max drawdown."""
        if self.phase == Phase.BLOWN:
            return True

        blown = False
        if self.dd_locked:
            # Funded with locked DD floor
            blown = self.equity <= self.dd_floor
        else:
            # Trailing DD from HWM
            blown = self.equity <= self.high_water_mark - self.max_trailing_dd

        if blown:
            self.phase = Phase.BLOWN

        return blown

    def _check_eval_passed(self) -> bool:
        """Check if eval profit target has been reached."""
        return (self.equity - self.starting_balance) >= rules.EVAL_PROFIT_TARGET

    def _check_consistency(self) -> bool:
        """Check if consistency rule is satisfied for payout."""
        if self.total_pnl <= 0:
            return False
        return self.best_day_pnl <= self.total_pnl * rules.CONSISTENCY_RULE_PCT

    def __repr__(self) -> str:
        return (f"PropAccount(id={self.account_id}, phase={self.phase.value}, "
                f"equity=${self.equity:,.0f}, dd=${self.current_dd:,.0f}, "
                f"trades={self.total_trades}, win_days={self.winning_days})")
