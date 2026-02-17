# NinjaTrader 8 Automated Strategy
# Dual Order Flow Strategy for Evaluation Factory
# Version 1.0

#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Gui.SuperDom;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.Core.FloatingPoint;
using NinjaTrader.NinjaScript.DrawingTools;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class DualOrderFlowEvaluation : Strategy
    {
        #region Variables
        
        // Order Flow Data Series
        private Series<double> deltaSeries;
        private Series<double> cvdSeries;
        private Series<double> imbalanceSeries;
        private Series<double> volumeMASeries;
        
        // Tracking variables
        private int consecutiveLosses;
        private double dailyPnL;
        private DateTime lastTradeDate;
        private int barsInTrade;
        private bool inPosition;
        private string currentStrategy; // "A" or "B"
        
        // Configuration
        private bool tradingEnabled;
        private int maxTradesPerDay;
        private int tradesToday;
        
        // Order tracking
        private Order entryOrder;
        private Order stopOrder;
        private Order targetOrder;
        
        #endregion

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Dual Order Flow Strategy for Prop Firm Evaluation";
                Name = "DualOrderFlowEvaluation";
                
                // Risk Parameters (Set these in the strategy panel)
                MaxContracts = 31;
                DailyLossLimit = -1500;
                MaxConsecutiveLosses = 5;
                MaxTradesPerDay = 15;
                
                // Time Window
                SessionStartTime = 100000; // 10:00 AM
                SessionEndTime = 130000;   // 1:00 PM
                
                // Signal Parameters
                DeltaPeriod = 20;
                DeltaThreshold = 85;
                ImbalancePeriod = 20;
                ImbalanceThreshold = 85;
                VolumeSpikeMultiplier = 1.5;
                
                // Exit Parameters
                ATRPeriod = 14;
                StopMultiplier = 0.4;
                RewardMultiplier = 2.0;
                MaxHoldBars = 8;
                
                // Initialize
                Calculate = Calculate.OnBarClose;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds = 30;
                IsFillLimitOnTouch = false;
                MaximumBarsLookBack = MaximumBarsLookBack.TwoHundredFiftySix;
                OrderFillResolution = OrderFillResolution.Standard;
                Slippage = 0;
                StartBehavior = StartBehavior.WaitUntilFlat;
                TimeInForce = TimeInForce.Gtc;
                TraceOrders = false;
                RealtimeErrorHandling = RealtimeErrorHandling.StopCancelClose;
                StopTargetHandling = StopTargetHandling.PerEntryExecution;
                BarsRequiredToTrade = 20;
                
                // Initialize variables
                consecutiveLosses = 0;
                dailyPnL = 0;
                tradesToday = 0;
                tradingEnabled = true;
                inPosition = false;
            }
            else if (State == State.Configure)
            {
                // Add data series if needed
                AddDataSeries(Data.BarsPeriodType.Minute, 1);
            }
            else if (State == State.DataLoaded)
            {
                // Initialize series
                deltaSeries = new Series<double>(this);
                cvdSeries = new Series<double>(this);
                imbalanceSeries = new Series<double>(this);
                volumeMASeries = new Series<double>(this);
                
                // Initialize tracking
                lastTradeDate = DateTime.MinValue;
                
                // Draw info on chart
                Draw.TextFixed(this, "StrategyInfo", 
                    "Dual Order Flow - EVALUATION MODE\n" +
                    "Contracts: " + MaxContracts + "\n" +
                    "Session: 10:00-13:00 ET", 
                    TextPosition.TopLeft, Brushes.White, new SimpleFont("Arial", 10), Brushes.Transparent, Brushes.Transparent, 0);
            }
        }

        #region Properties
        
        [NinjaScriptProperty]
        [Display(Name = "Max Contracts", Description = "Maximum position size", Order = 1, GroupName = "1. Risk Parameters")]
        public int MaxContracts { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Daily Loss Limit", Description = "Stop trading if daily P&L reaches this", Order = 2, GroupName = "1. Risk Parameters")]
        public double DailyLossLimit { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Max Consecutive Losses", Description = "Stop after this many consecutive losses", Order = 3, GroupName = "1. Risk Parameters")]
        public int MaxConsecutiveLosses { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Max Trades Per Day", Description = "Maximum trades allowed per day", Order = 4, GroupName = "1. Risk Parameters")]
        public int MaxTradesPerDay { get; set; }
        
        [NinjaScriptProperty]
        [Range(0, 235959)]
        [Display(Name = "Session Start", Description = "Start time in HHMMSS format", Order = 1, GroupName = "2. Time Window")]
        public int SessionStartTime { get; set; }
        
        [NinjaScriptProperty]
        [Range(0, 235959)]
        [Display(Name = "Session End", Description = "End time in HHMMSS format", Order = 2, GroupName = "2. Time Window")]
        public int SessionEndTime { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Delta Period", Description = "Lookback period for delta percentile", Order = 1, GroupName = "3. Signal Parameters")]
        public int DeltaPeriod { get; set; }
        
        [NinjaScriptProperty]
        [Range(0, 100)]
        [Display(Name = "Delta Threshold", Description = "Delta percentile threshold (0-100)", Order = 2, GroupName = "3. Signal Parameters")]
        public double DeltaThreshold { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Imbalance Period", Description = "Lookback period for imbalance percentile", Order = 3, GroupName = "3. Signal Parameters")]
        public int ImbalancePeriod { get; set; }
        
        [NinjaScriptProperty]
        [Range(0, 100)]
        [Display(Name = "Imbalance Threshold", Description = "Imbalance percentile threshold (0-100)", Order = 4, GroupName = "3. Signal Parameters")]
        public double ImbalanceThreshold { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Volume Spike Multiplier", Description = "Volume > MA × this for spike", Order = 5, GroupName = "3. Signal Parameters")]
        public double VolumeSpikeMultiplier { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "ATR Period", Description = "Period for ATR calculation", Order = 1, GroupName = "4. Exit Parameters")]
        public int ATRPeriod { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Stop Multiplier", Description = "Stop = ATR × this multiplier", Order = 2, GroupName = "4. Exit Parameters")]
        public double StopMultiplier { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Reward Multiplier", Description = "Target = Stop × this multiplier", Order = 3, GroupName = "4. Exit Parameters")]
        public double RewardMultiplier { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Max Hold Bars", Description = "Exit after this many bars", Order = 4, GroupName = "4. Exit Parameters")]
        public int MaxHoldBars { get; set; }
        
        #endregion

        protected override void OnBarUpdate()
        {
            // Wait for enough bars
            if (CurrentBar < BarsRequiredToTrade)
                return;
            
            // Check if new day
            if (Time[0].Date != lastTradeDate.Date)
            {
                dailyPnL = 0;
                consecutiveLosses = 0;
                tradesToday = 0;
                tradingEnabled = true;
                lastTradeDate = Time[0].Date;
                
                // Clear daily status text
                Draw.TextFixed(this, "DailyStatus", "", TextPosition.TopRight);
            }
            
            // Calculate order flow features
            CalculateOrderFlowFeatures();
            
            // Update status display
            UpdateStatusDisplay();
            
            // Check if we should be trading
            if (!ShouldTrade())
                return;
            
            // Manage existing position
            if (Position.MarketPosition != MarketPosition.Flat)
            {
                ManagePosition();
                return;
            }
            
            // Look for entry signals
            CheckEntrySignals();
        }

        private void CalculateOrderFlowFeatures()
        {
            // Get volume data (requires volumetric data feed)
            double askVol = Volume[0].AskVolume;
            double bidVol = Volume[0].BidVolume;
            double totalVol = Volume[0].TotalVolume;
            
            // Calculate delta
            double delta = askVol - bidVol;
            deltaSeries[0] = delta;
            
            // Calculate CVD (Cumulative Delta)
            if (CurrentBar == 0)
                cvdSeries[0] = delta;
            else
                cvdSeries[0] = cvdSeries[1] + delta;
            
            // Calculate imbalance
            double imbalance = askVol / Math.Max(bidVol, 1);
            imbalanceSeries[0] = EMA(imbalanceSeries, 5)[0];
            
            // Volume MA
            volumeMASeries[0] = SMA(Volume, 20)[0];
        }

        private bool ShouldTrade()
        {
            // Check time window
            int currentTime = ToTime(Time[0]);
            if (currentTime < SessionStartTime || currentTime > SessionEndTime)
                return false;
            
            // Check if trading disabled
            if (!tradingEnabled)
            {
                Draw.TextFixed(this, "TradeStatus", "TRADING DISABLED", TextPosition.TopRight, Brushes.Red);
                return false;
            }
            
            // Check daily loss limit
            if (dailyPnL <= DailyLossLimit)
            {
                tradingEnabled = false;
                Draw.TextFixed(this, "TradeStatus", "DAILY LOSS LIMIT REACHED", TextPosition.TopRight, Brushes.Red);
                return false;
            }
            
            // Check consecutive losses
            if (consecutiveLosses >= MaxConsecutiveLosses)
            {
                tradingEnabled = false;
                Draw.TextFixed(this, "TradeStatus", "MAX CONSECUTIVE LOSSES", TextPosition.TopRight, Brushes.Red);
                return false;
            }
            
            // Check max trades per day
            if (tradesToday >= MaxTradesPerDay)
            {
                Draw.TextFixed(this, "TradeStatus", "MAX TRADES REACHED", TextPosition.TopRight, Brushes.Yellow);
                return false;
            }
            
            return true;
        }

        private void CheckEntrySignals()
        {
            // Calculate percentiles
            double deltaPercentile = CalculatePercentile(deltaSeries, DeltaPeriod);
            double imbalancePercentile = CalculatePercentile(imbalanceSeries, ImbalancePeriod);
            
            // CVD trend
            double cvdMA = SMA(cvdSeries, 20)[0];
            bool cvdRising = cvdSeries[0] > cvdMA;
            bool cvdFalling = cvdSeries[0] < cvdMA;
            
            // Volume spike
            bool volumeSpike = Volume[0].TotalVolume > (volumeMASeries[0] * VolumeSpikeMultiplier);
            
            // Delta direction
            bool deltaPositive = deltaSeries[0] > 0;
            bool deltaNegative = deltaSeries[0] < 0;
            
            // ATR for stops
            double atr = ATR(ATRPeriod)[0];
            
            // Strategy A: Imbalance + Volume + CVD (Tier 1)
            bool signalALong = (imbalancePercentile > ImbalanceThreshold) && 
                               volumeSpike && 
                               cvdRising && 
                               deltaPositive;
            
            bool signalAShort = (imbalancePercentile > ImbalanceThreshold) && 
                                volumeSpike && 
                                cvdFalling && 
                                deltaNegative;
            
            // Strategy B: Delta + CVD (Tier 2)
            bool signalBLong = !signalALong && 
                               (deltaPercentile > DeltaThreshold) && 
                               cvdRising && 
                               deltaPositive;
            
            bool signalBShort = !signalAShort && 
                                (deltaPercentile > DeltaThreshold) && 
                                cvdFalling && 
                                deltaNegative;
            
            // Execute Strategy A (Full size)
            if (signalALong)
            {
                EnterTrade("LONG", "A", atr);
            }
            else if (signalAShort)
            {
                EnterTrade("SHORT", "A", atr);
            }
            // Execute Strategy B (Half size if using tiered sizing)
            else if (signalBLong)
            {
                EnterTrade("LONG", "B", atr);
            }
            else if (signalBShort)
            {
                EnterTrade("SHORT", "B", atr);
            }
        }

        private void EnterTrade(string direction, string strategy, double atr)
        {
            // Calculate position size
            int quantity = MaxContracts;
            
            // For Strategy B, could use half size (optional)
            // if (strategy == "B") quantity = quantity / 2;
            
            // Calculate stops
            double stopDistance = atr * StopMultiplier;
            double targetDistance = stopDistance * RewardMultiplier;
            
            double entryPrice = Close[0];
            double stopPrice;
            double targetPrice;
            string label = "Strategy" + strategy + direction;
            
            if (direction == "LONG")
            {
                stopPrice = entryPrice - stopDistance;
                targetPrice = entryPrice + targetDistance;
                
                // Enter long
                EnterLong(quantity, label);
                SetStopLoss(label, CalculationMode.Price, stopPrice, false);
                SetProfitTarget(label, CalculationMode.Price, targetPrice, false);
                
                // Draw entry
                Draw.ArrowUp(this, "Entry" + CurrentBar, false, label, 0, Brushes.Lime);
            }
            else
            {
                stopPrice = entryPrice + stopDistance;
                targetPrice = entryPrice - targetDistance;
                
                // Enter short
                EnterShort(quantity, label);
                SetStopLoss(label, CalculationMode.Price, stopPrice, false);
                SetProfitTarget(label, CalculationMode.Price, targetPrice, false);
                
                // Draw entry
                Draw.ArrowDown(this, "Entry" + CurrentBar, false, label, 0, Brushes.Red);
            }
            
            // Track position
            inPosition = true;
            currentStrategy = strategy;
            barsInTrade = 0;
            tradesToday++;
            
            // Log entry
            Print(string.Format("{0}: ENTER {1} {2} contracts @ {3:F2} - Strategy {4}", 
                Time[0], direction, quantity, entryPrice, strategy));
        }

        private void ManagePosition()
        {
            barsInTrade++;
            
            // Time exit
            if (barsInTrade >= MaxHoldBars)
            {
                string label = currentStrategy + (Position.MarketPosition == MarketPosition.Long ? "LONG" : "SHORT");
                
                if (Position.MarketPosition == MarketPosition.Long)
                    ExitLong("Time Exit");
                else
                    ExitShort("Time Exit");
                
                Print(string.Format("{0}: TIME EXIT after {1} bars", Time[0], barsInTrade));
            }
        }

        private double CalculatePercentile(Series<double> series, int period)
        {
            if (CurrentBar < period)
                return 50;
            
            double currentValue = series[0];
            int count = 0;
            
            for (int i = 0; i < period && i <= CurrentBar; i++)
            {
                if (series[i] <= currentValue)
                    count++;
            }
            
            return (double)count / Math.Min(period, CurrentBar + 1) * 100;
        }

        private void UpdateStatusDisplay()
        {
            string status = string.Format(
                "Daily P&L: ${0:F0} | Trades: {1}/{2} | Cons. Losses: {3}/{4}",
                dailyPnL, tradesToday, MaxTradesPerDay, consecutiveLosses, MaxConsecutiveLosses);
            
            Brush statusColor = Brushes.White;
            if (dailyPnL < -500) statusColor = Brushes.Yellow;
            if (dailyPnL < -1000) statusColor = Brushes.Orange;
            if (dailyPnL <= DailyLossLimit) statusColor = Brushes.Red;
            
            Draw.TextFixed(this, "DailyStatus", status, TextPosition.TopRight, statusColor);
        }

        protected override void OnPositionUpdate(Position position, double averagePrice,
            int quantity, MarketPosition marketPosition)
        {
            if (position.MarketPosition == MarketPosition.Flat && inPosition)
            {
                // Position closed
                inPosition = false;
                
                // Calculate P&L
                double tradePnL = position.GetProfitLoss(Close[0], PerformanceUnit.Currency);
                dailyPnL += tradePnL;
                
                // Track consecutive losses
                if (tradePnL < 0)
                    consecutiveLosses++;
                else
                    consecutiveLosses = 0;
                
                // Draw exit marker
                string exitType = tradePnL > 0 ? "WIN" : "LOSS";
                Brush exitColor = tradePnL > 0 ? Brushes.Green : Brushes.Red;
                Draw.Diamond(this, "Exit" + CurrentBar, false, exitType, 0, exitColor);
                
                // Log exit
                Print(string.Format("{0}: EXIT {1} P&L: ${2:F2} | Daily: ${3:F2}", 
                    Time[0], exitType, tradePnL, dailyPnL));
            }
        }

        protected override void OnOrderUpdate(Order order, double limitPrice, double stopPrice,
            int quantity, int filled, double averageFillPrice, OrderState orderState,
            DateTime time, ErrorCode error, string nativeError)
        {
            // Handle order errors
            if (error != ErrorCode.NoError)
            {
                Print(string.Format("ORDER ERROR: {0} - {1}", error, nativeError));
                
                // Disable trading on critical errors
                if (error == ErrorCode.OrderRejected || error == ErrorCode.InsufficientFunds)
                {
                    tradingEnabled = false;
                    Draw.TextFixed(this, "TradeStatus", "ORDER ERROR - TRADING DISABLED", 
                        TextPosition.TopRight, Brushes.Red);
                }
            }
        }

        public override string DisplayName
        {
            get { return "Dual Order Flow Evaluation"; }
        }
    }
}
