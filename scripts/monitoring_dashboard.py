"""
Monitoring Dashboard
====================
Real-time trade monitoring and performance tracking

Features:
- Real-time P&L display
- Trade statistics
- Position tracking
- Alert generation
- Performance metrics

Usage:
    dashboard = MonitoringDashboard()
    dashboard.update_trade(trade_data)
    dashboard.display_status()
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import json
from pathlib import Path
import logging

from dual_strategy import TradeSignal, SignalType
from live_trading_wrapper import Position, TradingState


logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Trading alert"""
    timestamp: datetime
    level: str  # 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    message: str
    acknowledged: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level,
            'message': self.message,
            'acknowledged': self.acknowledged
        }


@dataclass
class SessionStats:
    """Statistics for current trading session"""
    session_start: datetime
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade_duration: float = 0.0
    
    # Strategy breakdown
    tier_1_trades: int = 0
    tier_1_pnl: float = 0.0
    tier_2_trades: int = 0
    tier_2_pnl: float = 0.0
    
    def update(self, position: Position):
        """Update stats with new position"""
        self.total_trades += 1
        self.total_pnl += position.pnl
        
        if position.pnl > 0:
            self.winning_trades += 1
            self.largest_win = max(self.largest_win, position.pnl)
        else:
            self.losing_trades += 1
            self.largest_loss = min(self.largest_loss, position.pnl)
        
        # Strategy breakdown
        if position.signal.signal_type == SignalType.TIER_1:
            self.tier_1_trades += 1
            self.tier_1_pnl += position.pnl
        else:
            self.tier_2_trades += 1
            self.tier_2_pnl += position.pnl
    
    @property
    def win_rate(self) -> float:
        return self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0
    
    @property
    def profit_factor(self) -> float:
        total_wins = sum(p.pnl for p in [] if p.pnl > 0)  # Placeholder
        total_losses = abs(sum(p.pnl for p in [] if p.pnl < 0))  # Placeholder
        return total_wins / total_losses if total_losses > 0 else float('inf')


class MonitoringDashboard:
    """
    Real-time Monitoring Dashboard
    
    Tracks:
    - Current P&L
    - Open position
    - Daily statistics
    - Trading performance
    - Alerts and warnings
    """
    
    def __init__(self, max_alerts: int = 100, alert_log_file: Optional[str] = None):
        """
        Initialize dashboard
        
        Args:
            max_alerts: Maximum number of alerts to keep in memory
            alert_log_file: File to log alerts to
        """
        self.session_stats = SessionStats(session_start=datetime.now())
        self.alerts: deque = deque(maxlen=max_alerts)
        self.current_state: Optional[TradingState] = None
        self.trade_history: List[Position] = []
        
        self.alert_log_file = alert_log_file
        
        # Alert thresholds
        self.daily_loss_threshold = -1500.0
        self.consecutive_loss_threshold = 3
        self.large_loss_threshold = -500.0
        
        logger.info("MonitoringDashboard initialized")
    
    def update_state(self, state: TradingState):
        """Update with current trading state"""
        self.current_state = state
        
        # Check for alerts
        self._check_alerts()
    
    def update_trade(self, position: Position):
        """Update with completed trade"""
        self.trade_history.append(position)
        self.session_stats.update(position)
        
        # Check for trade-specific alerts
        self._check_trade_alerts(position)
    
    def _check_alerts(self):
        """Check for system alerts"""
        if not self.current_state:
            return
        
        # Daily loss alert
        if self.current_state.daily_pnl <= self.daily_loss_threshold:
            if self.current_state.daily_pnl <= -2000:
                self.add_alert(
                    'CRITICAL',
                    f"DAILY LOSS LIMIT REACHED: ${self.current_state.daily_pnl:.2f}. STOP TRADING!"
                )
            else:
                self.add_alert(
                    'WARNING',
                    f"Daily loss approaching limit: ${self.current_state.daily_pnl:.2f}"
                )
        
        # Consecutive losses alert
        if self.current_state.consecutive_losses >= self.consecutive_loss_threshold:
            self.add_alert(
                'WARNING',
                f"Consecutive losses: {self.current_state.consecutive_losses}. Consider reducing size."
            )
        
        # Check if trading disabled
        if not self.current_state.trading_enabled:
            self.add_alert(
                'ERROR',
                "Trading has been disabled. Manual intervention required."
            )
    
    def _check_trade_alerts(self, position: Position):
        """Check alerts for specific trade"""
        # Large loss alert
        if position.pnl <= self.large_loss_threshold:
            self.add_alert(
                'WARNING',
                f"Large loss: ${position.pnl:.2f} on {position.signal.direction} trade"
            )
        
        # Quick loss alert (stopped out quickly)
        if position.pnl < 0 and position.bars_held <= 2:
            self.add_alert(
                'INFO',
                f"Quick stop: {position.bars_held} bars, ${position.pnl:.2f}"
            )
    
    def add_alert(self, level: str, message: str):
        """Add a new alert"""
        alert = Alert(
            timestamp=datetime.now(),
            level=level,
            message=message
        )
        self.alerts.append(alert)
        
        # Log alert
        logger.log(
            getattr(logging, level, logging.INFO),
            f"ALERT [{level}]: {message}"
        )
        
        # Write to file if configured
        if self.alert_log_file:
            with open(self.alert_log_file, 'a') as f:
                f.write(f"{datetime.now().isoformat()} [{level}] {message}\n")
    
    def get_unacknowledged_alerts(self) -> List[Alert]:
        """Get all unacknowledged alerts"""
        return [a for a in self.alerts if not a.acknowledged]
    
    def acknowledge_alert(self, alert_idx: int):
        """Acknowledge an alert"""
        if 0 <= alert_idx < len(self.alerts):
            self.alerts[alert_idx].acknowledged = True
    
    def display_status(self) -> str:
        """
        Generate status display string
        
        Returns formatted string for display
        """
        lines = []
        lines.append("=" * 60)
        lines.append("TRADING DASHBOARD - " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        lines.append("=" * 60)
        
        # Session stats
        lines.append("\n📊 SESSION STATISTICS")
        lines.append("-" * 60)
        lines.append(f"Session Duration: {datetime.now() - self.session_stats.session_start}")
        lines.append(f"Total Trades: {self.session_stats.total_trades}")
        lines.append(f"Win Rate: {self.session_stats.win_rate*100:.1f}%")
        lines.append(f"Total P&L: ${self.session_stats.total_pnl:,.2f}")
        lines.append(f"Largest Win: ${self.session_stats.largest_win:,.2f}")
        lines.append(f"Largest Loss: ${self.session_stats.largest_loss:,.2f}")
        
        # Strategy breakdown
        lines.append("\n📈 STRATEGY BREAKDOWN")
        lines.append("-" * 60)
        lines.append(f"Tier 1 (Full Size): {self.session_stats.tier_1_trades} trades, ${self.session_stats.tier_1_pnl:,.2f}")
        lines.append(f"Tier 2 (Half Size): {self.session_stats.tier_2_trades} trades, ${self.session_stats.tier_2_pnl:,.2f}")
        
        # Current state
        if self.current_state:
            lines.append("\n💼 CURRENT STATE")
            lines.append("-" * 60)
            lines.append(f"Daily P&L: ${self.current_state.daily_pnl:,.2f}")
            lines.append(f"Daily Trades: {self.current_state.daily_trades}")
            lines.append(f"Consecutive Losses: {self.current_state.consecutive_losses}")
            lines.append(f"Trading Enabled: {'✅' if self.current_state.trading_enabled else '❌'}")
            
            if self.current_state.current_position:
                pos = self.current_state.current_position
                lines.append(f"\nOpen Position: {pos.signal.direction} {pos.signal.position_size} MNQ")
                lines.append(f"Entry: {pos.signal.entry_price:.2f} | Stop: {pos.signal.stop_price:.2f} | Target: {pos.signal.target_price:.2f}")
                lines.append(f"Bars Held: {pos.bars_held}")
        
        # Recent alerts
        unack = self.get_unacknowledged_alerts()
        if unack:
            lines.append("\n🚨 UNACKNOWLEDGED ALERTS")
            lines.append("-" * 60)
            for i, alert in enumerate(unack[-5:], 1):  # Show last 5
                lines.append(f"{i}. [{alert.level}] {alert.message}")
        
        # Performance projection
        if self.session_stats.total_trades > 5:
            lines.append("\n🎯 PERFORMANCE PROJECTION")
            lines.append("-" * 60)
            avg_daily_pnl = self.session_stats.total_pnl / max(1, len(self.trade_history) / 11)  # Assuming ~11 trades/day
            days_to_pass = max(0, (9000 - self.session_stats.total_pnl) / avg_daily_pnl) if avg_daily_pnl > 0 else float('inf')
            lines.append(f"Avg Daily P&L: ${avg_daily_pnl:,.2f}")
            lines.append(f"Est. Days to Pass: {days_to_pass:.1f}")
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)
    
    def print_status(self):
        """Print status to console"""
        print(self.display_status())
    
    def get_recent_trades(self, n: int = 10) -> List[Dict]:
        """Get recent trades as list of dictionaries"""
        recent = self.trade_history[-n:]
        return [self._position_to_dict(p) for p in recent]
    
    def _position_to_dict(self, position: Position) -> Dict:
        """Convert position to dictionary"""
        return {
            'entry_time': position.entry_time.isoformat(),
            'exit_time': position.exit_time.isoformat() if position.exit_time else None,
            'direction': position.signal.direction,
            'signal_type': position.signal.signal_type.name,
            'position_size': position.signal.position_size,
            'entry_price': position.signal.entry_price,
            'exit_price': position.exit_price,
            'exit_reason': position.exit_reason.value if position.exit_reason else None,
            'pnl': position.pnl,
            'bars_held': position.bars_held
        }
    
    def export_report(self, filename: Optional[str] = None) -> str:
        """
        Export full report to file
        
        Returns: Path to saved file
        """
        if filename is None:
            filename = f"trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'session_stats': {
                'session_start': self.session_stats.session_start.isoformat(),
                'total_trades': self.session_stats.total_trades,
                'winning_trades': self.session_stats.winning_trades,
                'losing_trades': self.session_stats.losing_trades,
                'win_rate': self.session_stats.win_rate,
                'total_pnl': self.session_stats.total_pnl,
                'largest_win': self.session_stats.largest_win,
                'largest_loss': self.session_stats.largest_loss,
                'tier_1_trades': self.session_stats.tier_1_trades,
                'tier_1_pnl': self.session_stats.tier_1_pnl,
                'tier_2_trades': self.session_stats.tier_2_trades,
                'tier_2_pnl': self.session_stats.tier_2_pnl
            },
            'alerts': [a.to_dict() for a in self.alerts],
            'trades': [self._position_to_dict(p) for p in self.trade_history]
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report exported to {filename}")
        return filename
    
    def reset(self):
        """Reset dashboard for new session"""
        self.session_stats = SessionStats(session_start=datetime.now())
        self.alerts.clear()
        self.current_state = None
        self.trade_history.clear()
        logger.info("Dashboard reset for new session")


class SimpleConsoleDashboard:
    """
    Simple console-based dashboard for basic monitoring
    
    Usage:
        dashboard = SimpleConsoleDashboard()
        dashboard.update(pnl=100, position=position)
        dashboard.draw()
    """
    
    def __init__(self):
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.win_count = 0
        self.loss_count = 0
        self.current_position = None
        self.last_update = datetime.now()
    
    def update(self, daily_pnl: float = None, total_trades: int = None,
               win_count: int = None, loss_count: int = None,
               position: Position = None):
        """Update dashboard values"""
        if daily_pnl is not None:
            self.daily_pnl = daily_pnl
        if total_trades is not None:
            self.total_trades = total_trades
        if win_count is not None:
            self.win_count = win_count
        if loss_count is not None:
            self.loss_count = loss_count
        if position is not None:
            self.current_position = position
        
        self.last_update = datetime.now()
    
    def draw(self):
        """Draw simple console output"""
        win_rate = self.win_count / max(1, self.total_trades) * 100
        
        print(f"\n{'='*50}")
        print(f"TRADING DASHBOARD - {self.last_update.strftime('%H:%M:%S')}")
        print(f"{'='*50}")
        print(f"Daily P&L:    ${self.daily_pnl:>10,.2f}")
        print(f"Total Trades: {self.total_trades:>10}")
        print(f"Win Rate:     {win_rate:>10.1f}%")
        print(f"Wins/Losses:  {self.win_count}/{self.loss_count}")
        
        if self.current_position:
            print(f"\nOpen: {self.current_position.signal.direction} "
                  f"{self.current_position.signal.position_size} MNQ")
        else:
            print(f"\nNo open position")
        
        print(f"{'='*50}\n")


# Convenience functions
def create_dashboard(max_alerts: int = 100, 
                    alert_log_file: Optional[str] = None) -> MonitoringDashboard:
    """Factory function to create dashboard"""
    return MonitoringDashboard(max_alerts, alert_log_file)


if __name__ == '__main__':
    print("Monitoring Dashboard")
    print("=" * 50)
    print("\nFeatures:")
    print("  - Real-time P&L tracking")
    print("  - Trade statistics")
    print("  - Alert generation")
    print("  - Performance projection")
    print("\nUsage:")
    print("  from monitoring_dashboard import MonitoringDashboard")
    print("  dashboard = MonitoringDashboard()")
    print("  dashboard.update_state(trading_state)")
    print("  dashboard.print_status()")
    print("")
    print("  # Or use simple version:")
    print("  from monitoring_dashboard import SimpleConsoleDashboard")
    print("  dash = SimpleConsoleDashboard()")
    print("  dash.update(daily_pnl=500, total_trades=5)")
    print("  dash.draw()")
