"""
PropPipeline - Multi-account pipeline manager.

Sequential evaluation strategy:
  1. Start with 1 eval account
  2. Pass eval -> transitions to FUNDED, immediately start next eval
  3. Build up to 5 funded accounts
  4. Replace blown funded accounts via new evals
  5. Budget: 5 account slots x 2 resets each = 15 total eval attempts max

Two funded execution modes:
  - Rotation: Trade 1 funded account per day, rotate daily
  - Copy Trade: Same signal to ALL funded accounts simultaneously (smaller size each)
"""
from typing import List, Dict, Optional, Tuple
from prop import rules
from prop.account import PropAccount, Phase
from prop.sizer import PropSizer


class PropPipeline:
    """Manages sequential eval + multi-account funded pipeline."""

    def __init__(self, mode: str = 'copy_trade', max_funded: int = 5,
                 eval_type_a_risk: float = None, eval_type_b_risk: float = None):
        """
        Args:
            mode: 'rotation' or 'copy_trade'
            max_funded: Max number of funded accounts to maintain
            eval_type_a_risk: Override Type A risk for eval phase
            eval_type_b_risk: Override Type B risk for eval phase
        """
        assert mode in ('rotation', 'copy_trade'), f"Invalid mode: {mode}"
        self.mode = mode
        self.max_funded = max_funded

        # Account tracking
        self.funded_accounts: List[PropAccount] = []
        self.eval_account: Optional[PropAccount] = None
        self.retired_accounts: List[PropAccount] = []  # Blown or completed

        # Rotation state
        self.rotation_index = 0

        # Pipeline counters
        self.evals_started = 0
        self.evals_passed = 0
        self.evals_blown = 0
        self.funded_blown = 0
        self.next_account_id = 1

        # Financial tracking
        self.total_eval_cost = 0.0
        self.total_payouts = 0.0

        # Sizer with custom eval risk
        self.sizer = PropSizer(
            eval_type_a_risk=eval_type_a_risk,
            eval_type_b_risk=eval_type_b_risk,
        )

        # Milestone snapshots: day -> snapshot dict
        self.milestones: Dict[int, dict] = {}
        self.milestone_days = [63, 126, 189, 252, 378, 504]  # 3,6,9,12,18,24 months

        # Current simulation day
        self.current_day = 0

    @property
    def net_income(self) -> float:
        return self.total_payouts - self.total_eval_cost

    @property
    def reset_rate(self) -> float:
        total = self.evals_passed + self.evals_blown
        if total == 0:
            return 0.0
        return self.evals_blown / total

    @property
    def active_funded_count(self) -> int:
        return len(self.funded_accounts)

    @property
    def has_eval_budget(self) -> bool:
        """Check if we can still start new evals (within budget)."""
        return self.evals_started < rules.MAX_TOTAL_EVAL_ATTEMPTS

    def start_new_eval(self, day: int = 0):
        """Start a new evaluation account."""
        if not self.has_eval_budget:
            return  # Budget exhausted

        if self.eval_account is not None:
            return  # Already have an active eval

        account = PropAccount(account_id=self.next_account_id, start_day=day)
        self.next_account_id += 1
        self.eval_account = account
        self.evals_started += 1
        self.total_eval_cost += rules.EVAL_COST

    def on_signal(self, strategy_name: str, setup_type: str, confidence: str,
                  stop_distance_pts: float, trade_points: float) -> List[Tuple[PropAccount, int, float]]:
        """
        Route a trade signal to all active accounts.

        Args:
            strategy_name: e.g., 'B-Day', 'Trend Day Bull'
            setup_type: e.g., 'B_DAY_IBL_FADE', 'VWAP_PULLBACK'
            confidence: 'low', 'medium', 'high'
            stop_distance_pts: Stop distance in NQ points
            trade_points: Actual trade result in NQ points (from backtest)

        Returns:
            List of (account, contracts, net_pnl) tuples for accounts that took the trade
        """
        results = []
        setup_grade = self.sizer.classify_setup(strategy_name, setup_type, confidence)

        # Always route to eval account
        if self.eval_account and self.eval_account.is_active:
            contracts = self.sizer.calculate_contracts(
                self.eval_account, setup_grade, stop_distance_pts
            )
            if contracts > 0:
                net_pnl = self.sizer.calculate_net_pnl(trade_points, contracts)
                self.eval_account.on_trade(net_pnl, setup_type)
                results.append((self.eval_account, contracts, net_pnl))

        # Route to funded accounts based on mode
        if self.funded_accounts:
            if self.mode == 'rotation':
                # Only trade the current rotation account
                if self.rotation_index < len(self.funded_accounts):
                    acct = self.funded_accounts[self.rotation_index]
                    if acct.is_active:
                        contracts = self.sizer.calculate_contracts(
                            acct, setup_grade, stop_distance_pts
                        )
                        if contracts > 0:
                            net_pnl = self.sizer.calculate_net_pnl(trade_points, contracts)
                            acct.on_trade(net_pnl, setup_type)
                            results.append((acct, contracts, net_pnl))

            elif self.mode == 'copy_trade':
                # Trade ALL funded accounts simultaneously
                for acct in self.funded_accounts:
                    if acct.is_active:
                        contracts = self.sizer.calculate_contracts(
                            acct, setup_grade, stop_distance_pts
                        )
                        if contracts > 0:
                            net_pnl = self.sizer.calculate_net_pnl(trade_points, contracts)
                            acct.on_trade(net_pnl, setup_type)
                            results.append((acct, contracts, net_pnl))

        return results

    def process_eod(self, day: int) -> dict:
        """
        End-of-day processing for all accounts.

        Returns summary dict with events that occurred.
        """
        self.current_day = day
        events = {
            'day': day,
            'eval_passed': False,
            'eval_blown': False,
            'funded_blown': [],
            'payouts': [],
        }

        # Process eval account EOD
        if self.eval_account:
            result = self.eval_account.process_eod()

            if result.get('eval_passed'):
                # Eval passed! Transition to funded
                self.eval_account.transition_to_funded()
                self.funded_accounts.append(self.eval_account)
                self.evals_passed += 1
                events['eval_passed'] = True
                self.eval_account = None

                # Start next eval if we need more funded accounts
                if self.active_funded_count < self.max_funded:
                    self.start_new_eval(day)

            elif result.get('blown'):
                # Eval blown - retire and start new one
                self.retired_accounts.append(self.eval_account)
                self.evals_blown += 1
                events['eval_blown'] = True
                self.eval_account = None

                # Start new eval if budget allows
                self.start_new_eval(day)

        # Process funded accounts EOD
        blown_indices = []
        for i, acct in enumerate(self.funded_accounts):
            result = acct.process_eod()

            if result.get('blown'):
                blown_indices.append(i)
                self.funded_blown += 1
                events['funded_blown'].append(acct.account_id)

            elif result.get('payout_ready'):
                # Process payout
                payout = acct.request_payout()
                if payout > 0:
                    self.total_payouts += payout
                    events['payouts'].append((acct.account_id, payout))

        # Remove blown funded accounts (reverse order to preserve indices)
        for i in sorted(blown_indices, reverse=True):
            retired = self.funded_accounts.pop(i)
            self.retired_accounts.append(retired)

        # Start new eval if a funded account blew up and we have no active eval
        if blown_indices and self.eval_account is None:
            self.start_new_eval(day)

        # Advance rotation index
        if self.mode == 'rotation' and self.funded_accounts:
            self.rotation_index = (self.rotation_index + 1) % len(self.funded_accounts)

        # Snapshot milestones
        if day in self.milestone_days:
            self._snapshot_milestone(day)

        return events

    def _snapshot_milestone(self, day: int):
        """Record pipeline state at a milestone day."""
        months = day / 21  # ~21 trading days per month

        self.milestones[day] = {
            'month': round(months),
            'funded_active': self.active_funded_count,
            'evals_started': self.evals_started,
            'evals_passed': self.evals_passed,
            'evals_blown': self.evals_blown,
            'funded_blown': self.funded_blown,
            'total_eval_cost': self.total_eval_cost,
            'total_payouts': self.total_payouts,
            'net_income': self.net_income,
            'reset_rate': self.reset_rate,
            'funded_equity': [a.equity for a in self.funded_accounts],
            'avg_funded_equity': (
                sum(a.equity for a in self.funded_accounts) / len(self.funded_accounts)
                if self.funded_accounts else 0
            ),
            'funded_payouts': [a.payouts_completed for a in self.funded_accounts],
        }

    def get_final_summary(self) -> dict:
        """Get summary of entire pipeline run."""
        return {
            'mode': self.mode,
            'days_simulated': self.current_day,
            'funded_active': self.active_funded_count,
            'evals_started': self.evals_started,
            'evals_passed': self.evals_passed,
            'evals_blown': self.evals_blown,
            'funded_blown': self.funded_blown,
            'reset_rate': self.reset_rate,
            'total_eval_cost': self.total_eval_cost,
            'total_payouts': self.total_payouts,
            'net_income': self.net_income,
            'milestones': self.milestones,
        }

    def __repr__(self) -> str:
        eval_status = f"eval={self.eval_account.account_id}" if self.eval_account else "no eval"
        return (f"PropPipeline(mode={self.mode}, funded={self.active_funded_count}, "
                f"{eval_status}, net=${self.net_income:,.0f})")
