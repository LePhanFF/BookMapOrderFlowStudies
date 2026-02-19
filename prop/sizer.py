"""
PropSizer - Phase-aware position sizing with Type A/B setup classification.

Type A (high conviction): B-Day IBL Fade, high-confidence Trend Bull -> $1,000 risk in eval
Type B (standard): VWAP pullbacks, P-Day, pyramids -> $450 risk in eval
Funded phase uses conservative sizing with buffer-based scaling.
"""
import math
from prop import rules
from prop.account import PropAccount, Phase


class PropSizer:
    """Position sizer that adjusts risk based on account phase and setup quality."""

    # Type A setups: highest conviction entries from v13 backtest
    TYPE_A_SETUPS = {
        ('B-Day', 'B_DAY_IBL_FADE'),           # 100% WR (4/4), avg +$571
        ('Trend Day Bull', 'VWAP_PULLBACK'),    # Only with confidence='high'
    }

    def __init__(self, eval_type_a_risk=None, eval_type_b_risk=None):
        """Allow custom eval risk amounts for simulation comparison."""
        self.eval_type_a_risk = eval_type_a_risk or rules.EVAL_TYPE_A_RISK
        self.eval_type_b_risk = eval_type_b_risk or rules.EVAL_TYPE_B_RISK

    def classify_setup(self, strategy_name: str, setup_type: str,
                       confidence: str = 'medium') -> str:
        """
        Classify a signal as Type A or Type B.

        Type A: B-Day IBL Fade (always) or high-confidence Trend Bull VWAP
        Type B: Everything else
        """
        key = (strategy_name, setup_type)

        if key == ('B-Day', 'B_DAY_IBL_FADE'):
            return 'A'
        elif key == ('Trend Day Bull', 'VWAP_PULLBACK') and confidence == 'high':
            return 'A'
        else:
            return 'B'

    def calculate_risk(self, account: PropAccount, setup_grade: str) -> float:
        """
        Calculate dollar risk for a trade based on account phase and setup grade.

        Args:
            account: The PropAccount to size for
            setup_grade: 'A' or 'B' from classify_setup()

        Returns:
            Dollar risk amount (0.0 if should skip trade)
        """
        if account.phase == Phase.BLOWN:
            return 0.0

        if account.is_eval:
            return self._eval_risk(account, setup_grade)
        elif account.is_funded:
            return self._funded_risk(account, setup_grade)
        else:
            return 0.0

    def _eval_risk(self, account: PropAccount, setup_grade: str) -> float:
        """Aggressive sizing during evaluation."""
        dd = account.current_dd

        # Stop trading if DD too deep
        if dd > rules.EVAL_DD_STOP_THRESHOLD:
            return 0.0

        # Base risk by setup type (use instance overrides if set)
        if setup_grade == 'A':
            base_risk = self.eval_type_a_risk
        else:
            base_risk = self.eval_type_b_risk

        # Halve risk if in drawdown
        if dd > rules.EVAL_DD_REDUCTION_THRESHOLD:
            base_risk *= 0.5

        return base_risk

    def _funded_risk(self, account: PropAccount, setup_grade: str) -> float:
        """Conservative sizing during funded phase."""
        dd = account.current_dd

        # Stop trading if DD too deep in funded
        if dd > rules.FUNDED_DD_STOP_THRESHOLD:
            return 0.0

        phase = account.funded_risk_phase

        if phase == 1:
            # Building buffer - minimal risk
            return rules.FUNDED_PHASE1_RISK
        elif phase == 2:
            # DD locked, moderate risk
            return rules.FUNDED_PHASE2_RISK
        else:
            # Buffer built, can size up
            if setup_grade == 'A':
                return rules.FUNDED_PHASE3_TYPE_A_RISK
            else:
                return rules.FUNDED_PHASE3_TYPE_B_RISK

    def calculate_contracts(self, account: PropAccount, setup_grade: str,
                            stop_distance_pts: float) -> int:
        """
        Calculate number of MNQ contracts for a trade.

        Args:
            account: The PropAccount
            setup_grade: 'A' or 'B'
            stop_distance_pts: Distance from entry to stop in NQ points

        Returns:
            Number of MNQ contracts (0 if should skip)
        """
        risk_dollars = self.calculate_risk(account, setup_grade)
        if risk_dollars <= 0:
            return 0

        if stop_distance_pts <= 0:
            return 0

        # contracts = risk_dollars / (stop_pts * point_value)
        raw_contracts = risk_dollars / (stop_distance_pts * rules.MNQ_POINT_VALUE)

        # Floor to integer, minimum 1 if we're trading at all
        contracts = max(1, math.floor(raw_contracts))

        # Cap at scaling tier
        max_contracts = account.get_max_contracts()
        contracts = min(contracts, max_contracts)

        return contracts

    def calculate_net_pnl(self, points: float, contracts: int) -> float:
        """
        Calculate net P&L for a completed trade.

        Args:
            points: P&L in NQ points (positive = profit)
            contracts: Number of MNQ contracts

        Returns:
            Net P&L in dollars after commissions
        """
        gross = points * rules.MNQ_POINT_VALUE * contracts
        commission = rules.MNQ_COMMISSION_RT * contracts
        return gross - commission
