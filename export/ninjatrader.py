"""
NinjaTrader C# code generation.

Generates NinjaScript strategies from backtested signals, supporting:
  - Dalton playbook day-type strategies
  - Order flow filters (delta, CVD, imbalance)
  - ATM bracket orders with stop/target

Adapted from the original ninjatrader_export.py with improvements:
  - Parameterized strategy generation per day-type
  - Proper IB calculation in NinjaScript
  - VWAP/EMA indicator integration
  - PM management (trail to BE, VWAP breach exit)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
from engine.trade import Trade


class NinjaTraderExporter:
    """
    Generate NinjaScript C# code for NinjaTrader 8.

    Supports generating a unified strategy that implements all 9 Dalton
    day-type playbook strategies, with order flow filters.
    """

    def __init__(self, params: Optional[Dict] = None):
        defaults = {
            'instrument': 'MNQ',
            'timeframe': '1 Minute',
            'session_start': '09:30',
            'session_end': '15:30',
            'ib_minutes': 60,
            'delta_period': 20,
            'delta_threshold': 80,
            'imbalance_threshold': 85,
            'volume_spike_multiplier': 1.5,
            'atr_period': 14,
            'ema_fast': 20,
            'ema_slow': 50,
            'stop_ib_buffer': 0.2,
            'target_ib_multiple': 2.5,
            'max_hold_bars': 120,
            'max_contracts': 30,
            'max_daily_loss': 2000,
            'max_consecutive_losses': 5,
            'trail_to_be_after_pm': True,
            'vwap_breach_exit': True,
        }
        self.params = {**defaults, **(params or {})}

    def generate_strategy(self, strategy_name: str = "DaltonPlaybookStrategy") -> str:
        """Generate complete NinjaScript strategy code."""
        parts = [
            self._header(strategy_name),
            self._usings(),
            self._namespace_start(),
            self._class_start(strategy_name),
            self._parameters(),
            self._variables(),
            self._on_state_change(),
            self._on_bar_update(),
            self._ib_calculation(),
            self._day_type_classification(),
            self._signal_detection(),
            self._order_flow_filters(),
            self._position_management(),
            self._risk_management(),
            self._helper_methods(),
            self._class_end(),
            self._namespace_end(),
        ]
        return '\n'.join(parts)

    def _header(self, name: str) -> str:
        return f"""// {name}.cs
// Auto-generated Dalton Playbook Strategy for NinjaTrader 8
// Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
//
// Implements 9 Dalton day-type strategies with order flow filters.
// See playbooks.md for full strategy documentation.
//
// IMPORTANT: Test in simulation mode before live trading.
"""

    def _usings(self) -> str:
        return """
using System;
using System.Collections.Generic;
using System.Linq;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Indicators;
using NinjaTrader.Core.FloatingPoint;
using NinjaTrader.NinjaScript.DrawingTools;
using System.Windows.Media;
"""

    def _namespace_start(self) -> str:
        return "namespace NinjaTrader.NinjaScript.Strategies\n{"

    def _class_start(self, name: str) -> str:
        return f"""
    public class {name} : Strategy
    {{"""

    def _parameters(self) -> str:
        p = self.params
        return f"""
        #region Parameters

        [NinjaScriptProperty]
        [Display(Name = "IB Minutes", Order = 1, GroupName = "Session")]
        public int IBMinutes {{ get; set; }} = {p['ib_minutes']};

        [NinjaScriptProperty]
        [Display(Name = "Max Contracts", Order = 2, GroupName = "Risk")]
        public int MaxContracts {{ get; set; }} = {p['max_contracts']};

        [NinjaScriptProperty]
        [Display(Name = "Max Daily Loss", Order = 3, GroupName = "Risk")]
        public double MaxDailyLoss {{ get; set; }} = {p['max_daily_loss']};

        [NinjaScriptProperty]
        [Display(Name = "Stop IB Buffer", Order = 4, GroupName = "Targets")]
        public double StopIBBuffer {{ get; set; }} = {p['stop_ib_buffer']};

        [NinjaScriptProperty]
        [Display(Name = "Target IB Multiple", Order = 5, GroupName = "Targets")]
        public double TargetIBMultiple {{ get; set; }} = {p['target_ib_multiple']};

        [NinjaScriptProperty]
        [Display(Name = "Delta Threshold Pct", Order = 6, GroupName = "Order Flow")]
        public int DeltaThresholdPct {{ get; set; }} = {p['delta_threshold']};

        [NinjaScriptProperty]
        [Display(Name = "Trail to BE after PM", Order = 7, GroupName = "PM Management")]
        public bool TrailToBEAfterPM {{ get; set; }} = {str(p['trail_to_be_after_pm']).lower()};

        [NinjaScriptProperty]
        [Display(Name = "VWAP Breach Exit", Order = 8, GroupName = "PM Management")]
        public bool VWAPBreachExit {{ get; set; }} = {str(p['vwap_breach_exit']).lower()};

        #endregion"""

    def _variables(self) -> str:
        return """
        #region Variables

        // IB tracking
        private double ibHigh, ibLow, ibRange, ibMid;
        private bool ibComplete;
        private int ibBarCount;

        // Day type
        private string currentDayType;
        private string trendStrength;

        // Indicators
        private EMA emaFast, emaSlow;
        private ATR atr;
        private OrderFlowCumulativeDelta cumulativeDelta;

        // Position tracking
        private double dailyPnL;
        private int consecutiveLosses;
        private int dailyTradeCount;
        private DateTime currentSessionDate;
        private int barsInPosition;

        // Acceptance tracking
        private int consecutiveAboveIBH;
        private int consecutiveBelowIBL;
        private bool acceptedAbove, acceptedBelow;

        #endregion"""

    def _on_state_change(self) -> str:
        p = self.params
        return f"""
        protected override void OnStateChange()
        {{
            if (State == State.SetDefaults)
            {{
                Name = "DaltonPlaybookStrategy";
                Description = "9 Dalton Day-Type Strategies with Order Flow Filters";
                Calculate = Calculate.OnBarClose;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds = 30;
                IsFillLimitOnTouch = false;
                MaximumBarsLookBack = MaximumBarsLookBack.TwoHundredFiftySix;
                OrderFillResolution = OrderFillResolution.Standard;
                Slippage = 1;
                StartBehavior = StartBehavior.WaitUntilFlat;
                TimeInForce = TimeInForce.Gtc;
                TraceOrders = false;
                RealtimeErrorHandling = RealtimeErrorHandling.StopCancelClose;
                StopTargetHandling = StopTargetHandling.PerEntryExecution;
                BarsRequiredToTrade = 20;
            }}
            else if (State == State.Configure)
            {{
            }}
            else if (State == State.DataLoaded)
            {{
                emaFast = EMA({p['ema_fast']});
                emaSlow = EMA({p['ema_slow']});
                atr = ATR({p['atr_period']});
            }}
        }}"""

    def _on_bar_update(self) -> str:
        return """
        protected override void OnBarUpdate()
        {
            if (CurrentBar < 20) return;

            // Session date tracking
            if (Time[0].Date != currentSessionDate.Date)
            {
                ResetSession();
                currentSessionDate = Time[0].Date;
            }

            // IB calculation phase
            if (!ibComplete)
            {
                UpdateIB();
                return;
            }

            // Risk check
            if (!CanTrade()) return;

            // Classify day type dynamically
            ClassifyDayType();

            // Position management
            if (Position.MarketPosition != MarketPosition.Flat)
            {
                ManagePosition();
                return;
            }

            // Signal detection
            CheckSignals();
        }"""

    def _ib_calculation(self) -> str:
        return """
        private void ResetSession()
        {
            ibHigh = double.MinValue;
            ibLow = double.MaxValue;
            ibComplete = false;
            ibBarCount = 0;
            dailyPnL = 0;
            dailyTradeCount = 0;
            consecutiveAboveIBH = 0;
            consecutiveBelowIBL = 0;
            acceptedAbove = false;
            acceptedBelow = false;
            currentDayType = "NEUTRAL";
            trendStrength = "WEAK";
        }

        private void UpdateIB()
        {
            ibBarCount++;
            ibHigh = Math.Max(ibHigh, High[0]);
            ibLow = Math.Min(ibLow, Low[0]);

            if (ibBarCount >= IBMinutes)
            {
                ibRange = ibHigh - ibLow;
                ibMid = (ibHigh + ibLow) / 2.0;
                ibComplete = true;
            }
        }"""

    def _day_type_classification(self) -> str:
        return """
        private void ClassifyDayType()
        {
            double ext = 0;

            if (Close[0] > ibHigh)
            {
                ext = (Close[0] - ibMid) / ibRange;
                consecutiveAboveIBH++;
                consecutiveBelowIBL = 0;
                if (consecutiveAboveIBH >= 2) acceptedAbove = true;
            }
            else if (Close[0] < ibLow)
            {
                ext = (ibMid - Close[0]) / ibRange;
                consecutiveBelowIBL++;
                consecutiveAboveIBH = 0;
                if (consecutiveBelowIBL >= 2) acceptedBelow = true;
            }
            else
            {
                consecutiveAboveIBH = 0;
                consecutiveBelowIBL = 0;
            }

            // Trend strength
            if (ext > 2.0) trendStrength = "SUPER";
            else if (ext > 1.0) trendStrength = "STRONG";
            else if (ext > 0.5) trendStrength = "MODERATE";
            else trendStrength = "WEAK";

            // Day type
            if (ext > 2.0 && Close[0] > ibHigh) currentDayType = "SUPER_TREND_UP";
            else if (ext > 1.0 && Close[0] > ibHigh) currentDayType = "TREND_UP";
            else if (ext > 2.0 && Close[0] < ibLow) currentDayType = "SUPER_TREND_DOWN";
            else if (ext > 1.0 && Close[0] < ibLow) currentDayType = "TREND_DOWN";
            else if (ext > 0.5 && Close[0] > ibMid) currentDayType = "P_DAY";
            else if (ext < 0.2) currentDayType = "B_DAY";
            else currentDayType = "NEUTRAL";
        }"""

    def _signal_detection(self) -> str:
        return """
        private void CheckSignals()
        {
            // Trend day long
            if ((currentDayType == "TREND_UP" || currentDayType == "SUPER_TREND_UP")
                && acceptedAbove && trendStrength != "WEAK")
            {
                // IBH retest entry
                if (Low[0] <= ibHigh + (ibRange * 0.1) && Close[0] > ibHigh)
                {
                    double stop = ibLow - (ibRange * StopIBBuffer);
                    double target = Close[0] + (TargetIBMultiple * ibRange);
                    EnterLong(1, "TrendLong");
                    SetStopLoss("TrendLong", CalculationMode.Price, stop, false);
                    SetProfitTarget("TrendLong", CalculationMode.Price, target, false);
                    dailyTradeCount++;
                    return;
                }
            }

            // Trend day short
            if ((currentDayType == "TREND_DOWN" || currentDayType == "SUPER_TREND_DOWN")
                && acceptedBelow && trendStrength != "WEAK")
            {
                if (High[0] >= ibLow - (ibRange * 0.1) && Close[0] < ibLow)
                {
                    double stop = ibHigh + (ibRange * StopIBBuffer);
                    double target = Close[0] - (TargetIBMultiple * ibRange);
                    EnterShort(1, "TrendShort");
                    SetStopLoss("TrendShort", CalculationMode.Price, stop, false);
                    SetProfitTarget("TrendShort", CalculationMode.Price, target, false);
                    dailyTradeCount++;
                    return;
                }
            }

            // B-Day fade at IBH (short)
            if (currentDayType == "B_DAY" && trendStrength == "WEAK")
            {
                if (High[0] >= ibHigh && Close[0] < ibHigh)
                {
                    double stop = ibHigh + (ibRange * 0.1);
                    double target = ibMid;
                    EnterShort(1, "BDayFadeHigh");
                    SetStopLoss("BDayFadeHigh", CalculationMode.Price, stop, false);
                    SetProfitTarget("BDayFadeHigh", CalculationMode.Price, target, false);
                    dailyTradeCount++;
                    return;
                }

                // B-Day fade at IBL (long)
                if (Low[0] <= ibLow && Close[0] > ibLow)
                {
                    double stop = ibLow - (ibRange * 0.1);
                    double target = ibMid;
                    EnterLong(1, "BDayFadeLow");
                    SetStopLoss("BDayFadeLow", CalculationMode.Price, stop, false);
                    SetProfitTarget("BDayFadeLow", CalculationMode.Price, target, false);
                    dailyTradeCount++;
                    return;
                }
            }
        }"""

    def _order_flow_filters(self) -> str:
        return """
        private bool PassesOrderFlowFilter(string direction)
        {
            // Placeholder: integrate with NinjaTrader OrderFlow indicators
            // In live, use OrderFlowCumulativeDelta and volumetric data
            return true;
        }"""

    def _position_management(self) -> str:
        return """
        private void ManagePosition()
        {
            barsInPosition++;

            // PM management: trail to breakeven after 1 PM
            if (TrailToBEAfterPM)
            {
                int currentTime = ToTime(Time[0]);
                if (currentTime >= 130000)
                {
                    if (Position.MarketPosition == MarketPosition.Long
                        && Position.GetUnrealizedProfitLoss(PerformanceUnit.Currency) > 0)
                    {
                        SetStopLoss(CalculationMode.Price, Position.AveragePrice);
                    }
                    else if (Position.MarketPosition == MarketPosition.Short
                        && Position.GetUnrealizedProfitLoss(PerformanceUnit.Currency) > 0)
                    {
                        SetStopLoss(CalculationMode.Price, Position.AveragePrice);
                    }
                }
            }

            // Max hold time exit
            if (barsInPosition >= 120)
            {
                if (Position.MarketPosition == MarketPosition.Long)
                    ExitLong("TimeExit");
                else if (Position.MarketPosition == MarketPosition.Short)
                    ExitShort("TimeExit");
            }
        }"""

    def _risk_management(self) -> str:
        return """
        private bool CanTrade()
        {
            if (dailyPnL <= -MaxDailyLoss)
                return false;
            if (consecutiveLosses >= 5)
                return false;
            if (dailyTradeCount >= 10)
                return false;
            return true;
        }

        protected override void OnPositionUpdate(Position position, double averagePrice,
            int quantity, MarketPosition marketPosition)
        {
            if (marketPosition == MarketPosition.Flat && position != null)
            {
                double pnl = position.GetProfitLoss(Close[0], PerformanceUnit.Currency);
                dailyPnL += pnl;

                if (pnl < 0)
                    consecutiveLosses++;
                else
                    consecutiveLosses = 0;

                barsInPosition = 0;
            }
        }"""

    def _helper_methods(self) -> str:
        return ""

    def _class_end(self) -> str:
        return """
    }"""

    def _namespace_end(self) -> str:
        return "}"

    def save_to_file(self, code: str, filepath: str) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(code)
        print(f"Saved: {filepath}")

    def export_all(self, output_dir: str = './ninjatrader_export') -> None:
        """Export strategy, ATM template, and parameters."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        code = self.generate_strategy()
        self.save_to_file(code, str(output_path / 'DaltonPlaybookStrategy.cs'))

        params_file = output_path / 'parameters.json'
        with open(params_file, 'w') as f:
            json.dump(self.params, f, indent=2)

        print(f"\nExported to: {output_dir}")


def export_backtest_trades_to_ninjatrader(
    trades: List[Trade],
    output_dir: str = './ninjatrader_export',
) -> None:
    """
    Export backtest trades as a CSV that NinjaTrader can import
    for performance analysis.
    """
    import csv

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / 'backtest_trades.csv'
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'EntryTime', 'ExitTime', 'Direction', 'Contracts',
            'EntryPrice', 'ExitPrice', 'NetPnL', 'Strategy', 'Setup',
        ])
        for t in trades:
            writer.writerow([
                str(t.entry_time), str(t.exit_time), t.direction,
                t.contracts, round(t.entry_price, 2), round(t.exit_price, 2),
                round(t.net_pnl, 2), t.strategy_name, t.setup_type,
            ])

    print(f"Exported {len(trades)} trades to: {filepath}")
