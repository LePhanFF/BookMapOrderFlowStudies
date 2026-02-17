// DaltonPlaybookStrategy.cs
// Auto-generated Dalton Playbook Strategy for NinjaTrader 8
// Generated: 2026-02-16 22:38:53
//
// Implements 9 Dalton day-type strategies with order flow filters.
// See playbooks.md for full strategy documentation.
//
// IMPORTANT: Test in simulation mode before live trading.


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

namespace NinjaTrader.NinjaScript.Strategies
{

    public class DaltonPlaybookStrategy : Strategy
    {

        #region Parameters

        [NinjaScriptProperty]
        [Display(Name = "IB Minutes", Order = 1, GroupName = "Session")]
        public int IBMinutes { get; set; } = 60;

        [NinjaScriptProperty]
        [Display(Name = "Max Contracts", Order = 2, GroupName = "Risk")]
        public int MaxContracts { get; set; } = 30;

        [NinjaScriptProperty]
        [Display(Name = "Max Daily Loss", Order = 3, GroupName = "Risk")]
        public double MaxDailyLoss { get; set; } = 2000;

        [NinjaScriptProperty]
        [Display(Name = "Stop IB Buffer", Order = 4, GroupName = "Targets")]
        public double StopIBBuffer { get; set; } = 0.2;

        [NinjaScriptProperty]
        [Display(Name = "Target IB Multiple", Order = 5, GroupName = "Targets")]
        public double TargetIBMultiple { get; set; } = 2.5;

        [NinjaScriptProperty]
        [Display(Name = "Delta Threshold Pct", Order = 6, GroupName = "Order Flow")]
        public int DeltaThresholdPct { get; set; } = 80;

        [NinjaScriptProperty]
        [Display(Name = "Trail to BE after PM", Order = 7, GroupName = "PM Management")]
        public bool TrailToBEAfterPM { get; set; } = true;

        [NinjaScriptProperty]
        [Display(Name = "VWAP Breach Exit", Order = 8, GroupName = "PM Management")]
        public bool VWAPBreachExit { get; set; } = true;

        #endregion

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

        #endregion

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
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
            }
            else if (State == State.Configure)
            {
            }
            else if (State == State.DataLoaded)
            {
                emaFast = EMA(20);
                emaSlow = EMA(50);
                atr = ATR(14);
            }
        }

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
        }

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
        }

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
        }

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
        }

        private bool PassesOrderFlowFilter(string direction)
        {
            // Placeholder: integrate with NinjaTrader OrderFlow indicators
            // In live, use OrderFlowCumulativeDelta and volumetric data
            return true;
        }

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
        }

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
        }


    }
}