// NinjaTrader 8 Strategy
// Dual Order Flow - EVALUATION MODE (Pure Order Flow)
// Version 1.0
// Purpose: Pass prop firm evaluation FAST - no HTF filters, maximum aggression

#region Using declarations
using System;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Windows.Media;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.Core.FloatingPoint;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class DualOrderFlow_Evaluation : Strategy
    {
        #region Variables
        private Series<double> deltaSeries;
        private Series<double> cvdSeries;
        private Series<double> imbalanceSeries;
        private Series<double> volumeMASeries;
        
        private int consecutiveLosses;
        private double dailyPnL;
        private DateTime lastTradeDate;
        private int barsInTrade;
        private bool inPosition;
        private string currentStrategy;
        private int tradesToday;
        private bool tradingEnabled;
        #endregion

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = @"EVALUATION MODE: Pure 1-min order flow for fast passing. 
                Aggressive sizing, maximum trades, no HTF filters. 
                Use this to pass evaluation accounts quickly.";
                Name = "DualOrderFlow_Evaluation";
                
                // Risk - EVALUATION AGGRESSIVE
                MaxContracts = 31;
                DailyLossLimit = -1500;
                MaxConsecutiveLosses = 5;
                MaxTradesPerDay = 15;
                
                // Time - Core session only
                SessionStartTime = 100000;
                SessionEndTime = 130000;
                
                // Signals
                DeltaPeriod = 20;
                DeltaThreshold = 85;
                ImbalancePeriod = 20;
                ImbalanceThreshold = 85;
                VolumeSpikeMultiplier = 1.5;
                
                // Exits
                ATRPeriod = 14;
                StopMultiplier = 0.4;
                RewardMultiplier = 2.0;
                MaxHoldBars = 8;
                
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
                
                consecutiveLosses = 0;
                dailyPnL = 0;
                tradesToday = 0;
                tradingEnabled = true;
                inPosition = false;
            }
            else if (State == State.DataLoaded)
            {
                deltaSeries = new Series<double>(this);
                cvdSeries = new Series<double>(this);
                imbalanceSeries = new Series<double>(this);
                volumeMASeries = new Series<double>(this);
                lastTradeDate = DateTime.MinValue;
                
                Draw.TextFixed(this, "StrategyInfo", 
                    "⚡ EVALUATION MODE - MANIAC ⚡\n" +
                    "Strategy: Pure 1-Min Order Flow\n" +
                    "Contracts: " + MaxContracts + " | WR: 44% | Trades: 11/day\n" +
                    "Session: 10:00-13:00 ET | Pass: 9 days",
                    TextPosition.TopLeft, Brushes.Yellow, new SimpleFont("Arial", 11), Brushes.DarkBlue, Brushes.Transparent, 0);
            }
        }

        #region Properties
        [NinjaScriptProperty]
        [Display(Name = "Max Contracts", Description = "MAXIMUM size for evaluation speed", Order = 1, GroupName = "1. Risk Parameters")]
        public int MaxContracts { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Daily Loss Limit", Description = "STOP trading if hit (evaluation aggressive)", Order = 2, GroupName = "1. Risk Parameters")]
        public double DailyLossLimit { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Max Consecutive Losses", Description = "Stop after this many losses", Order = 3, GroupName = "1. Risk Parameters")]
        public int MaxConsecutiveLosses { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Max Trades Per Day", Description = "Maximum trades (15 = aggressive)", Order = 4, GroupName = "1. Risk Parameters")]
        public int MaxTradesPerDay { get; set; }
        
        [NinjaScriptProperty]
        [Range(0, 235959)]
        [Display(Name = "Session Start", Description = "HHMMSS format", Order = 1, GroupName = "2. Time")]
        public int SessionStartTime { get; set; }
        
        [NinjaScriptProperty]
        [Range(0, 235959)]
        [Display(Name = "Session End", Description = "HHMMSS format", Order = 2, GroupName = "2. Time")]
        public int SessionEndTime { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Delta Period", Order = 1, GroupName = "3. Signal Parameters")]
        public int DeltaPeriod { get; set; }
        
        [NinjaScriptProperty]
        [Range(0, 100)]
        [Display(Name = "Delta Threshold", Description = "Delta percentile (0-100)", Order = 2, GroupName = "3. Signal Parameters")]
        public double DeltaThreshold { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Imbalance Period", Order = 3, GroupName = "3. Signal Parameters")]
        public int ImbalancePeriod { get; set; }
        
        [NinjaScriptProperty]
        [Range(0, 100)]
        [Display(Name = "Imbalance Threshold", Description = "Imbalance percentile (0-100)", Order = 4, GroupName = "3. Signal Parameters")]
        public double ImbalanceThreshold { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Volume Spike Multiplier", Order = 5, GroupName = "3. Signal Parameters")]
        public double VolumeSpikeMultiplier { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "ATR Period", Order = 1, GroupName = "4. Exit Parameters")]
        public int ATRPeriod { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Stop Multiplier", Description = "0.4 = tight stops", Order = 2, GroupName = "4. Exit Parameters")]
        public double StopMultiplier { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Reward Multiplier", Description = "2.0 = 2:1 R:R", Order = 3, GroupName = "4. Exit Parameters")]
        public double RewardMultiplier { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Max Hold Bars", Description = "8 bars = 8 minutes", Order = 4, GroupName = "4. Exit Parameters")]
        public int MaxHoldBars { get; set; }
        #endregion

        protected override void OnBarUpdate()
        {
            if (CurrentBar < BarsRequiredToTrade)
                return;
            
            // Reset daily counters
            if (Time[0].Date != lastTradeDate.Date)
            {
                dailyPnL = 0;
                consecutiveLosses = 0;
                tradesToday = 0;
                tradingEnabled = true;
                lastTradeDate = Time[0].Date;
                Draw.TextFixed(this, "DailyStatus", "", TextPosition.TopRight);
            }
            
            CalculateOrderFlowFeatures();
            UpdateStatusDisplay();
            
            if (!ShouldTrade())
                return;
            
            if (Position.MarketPosition != MarketPosition.Flat)
            {
                ManagePosition();
                return;
            }
            
            CheckEntrySignals();
        }

        private void CalculateOrderFlowFeatures()
        {
            double askVol = Volume[0].AskVolume;
            double bidVol = Volume[0].BidVolume;
            double delta = askVol - bidVol;
            deltaSeries[0] = delta;
            
            if (CurrentBar == 0)
                cvdSeries[0] = delta;
            else
                cvdSeries[0] = cvdSeries[1] + delta;
            
            double imbalance = askVol / Math.Max(bidVol, 1);
            imbalanceSeries[0] = EMA(imbalanceSeries, 5)[0];
            volumeMASeries[0] = SMA(Volume, 20)[0];
        }

        private bool ShouldTrade()
        {
            int currentTime = ToTime(Time[0]);
            if (currentTime < SessionStartTime || currentTime > SessionEndTime)
                return false;
            
            if (!tradingEnabled)
                return false;
            
            if (dailyPnL <= DailyLossLimit)
            {
                tradingEnabled = false;
                Draw.TextFixed(this, "TradeStatus", "🛑 DAILY LOSS LIMIT", TextPosition.TopRight, Brushes.Red);
                return false;
            }
            
            if (consecutiveLosses >= MaxConsecutiveLosses)
            {
                tradingEnabled = false;
                Draw.TextFixed(this, "TradeStatus", "🛑 MAX LOSSES", TextPosition.TopRight, Brushes.Red);
                return false;
            }
            
            if (tradesToday >= MaxTradesPerDay)
                return false;
            
            return true;
        }

        private void CheckEntrySignals()
        {
            double deltaPercentile = CalculatePercentile(deltaSeries, DeltaPeriod);
            double imbalancePercentile = CalculatePercentile(imbalanceSeries, ImbalancePeriod);
            
            double cvdMA = SMA(cvdSeries, 20)[0];
            bool cvdRising = cvdSeries[0] > cvdMA;
            bool cvdFalling = cvdSeries[0] < cvdMA;
            
            bool volumeSpike = Volume[0].TotalVolume > (volumeMASeries[0] * VolumeSpikeMultiplier);
            
            bool deltaPositive = deltaSeries[0] > 0;
            bool deltaNegative = deltaSeries[0] < 0;
            
            double atr = ATR(ATRPeriod)[0];
            
            // Strategy A: Imbalance + Volume + CVD
            bool signalALong = (imbalancePercentile > ImbalanceThreshold) && volumeSpike && cvdRising && deltaPositive;
            bool signalAShort = (imbalancePercentile > ImbalanceThreshold) && volumeSpike && cvdFalling && deltaNegative;
            
            // Strategy B: Delta + CVD only
            bool signalBLong = !signalALong && (deltaPercentile > DeltaThreshold) && cvdRising && deltaPositive;
            bool signalBShort = !signalAShort && (deltaPercentile > DeltaThreshold) && cvdFalling && deltaNegative;
            
            if (signalALong)
                EnterTrade("LONG", "A", atr);
            else if (signalAShort)
                EnterTrade("SHORT", "A", atr);
            else if (signalBLong)
                EnterTrade("LONG", "B", atr);
            else if (signalBShort)
                EnterTrade("SHORT", "B", atr);
        }

        private void EnterTrade(string direction, string strategy, double atr)
        {
            int quantity = MaxContracts;
            double stopDistance = atr * StopMultiplier;
            double targetDistance = stopDistance * RewardMultiplier;
            
            double entryPrice = Close[0];
            double stopPrice;
            double targetPrice;
            string label = "Eval_" + strategy + "_" + direction;
            
            if (direction == "LONG")
            {
                stopPrice = entryPrice - stopDistance;
                targetPrice = entryPrice + targetDistance;
                EnterLong(quantity, label);
                SetStopLoss(label, CalculationMode.Price, stopPrice, false);
                SetProfitTarget(label, CalculationMode.Price, targetPrice, false);
                Draw.ArrowUp(this, "Entry" + CurrentBar, false, label, 0, Brushes.Lime);
            }
            else
            {
                stopPrice = entryPrice + stopDistance;
                targetPrice = entryPrice - targetDistance;
                EnterShort(quantity, label);
                SetStopLoss(label, CalculationMode.Price, stopPrice, false);
                SetProfitTarget(label, CalculationMode.Price, targetPrice, false);
                Draw.ArrowDown(this, "Entry" + CurrentBar, false, label, 0, Brushes.Red);
            }
            
            inPosition = true;
            currentStrategy = strategy;
            barsInTrade = 0;
            tradesToday++;
            
            Print(string.Format("{0}: ENTER {1} {2} contracts @ {3:F2} [Eval Mode]", 
                Time[0], direction, quantity, entryPrice));
        }

        private void ManagePosition()
        {
            barsInTrade++;
            if (barsInTrade >= MaxHoldBars)
            {
                if (Position.MarketPosition == MarketPosition.Long)
                    ExitLong("Time Exit");
                else
                    ExitShort("Time Exit");
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
                "💰 P&L: ${0:F0} | 📊 Trades: {1}/{2} | 🔥 Streak: {3}/{4}",
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
                inPosition = false;
                double tradePnL = position.GetProfitLoss(Close[0], PerformanceUnit.Currency);
                dailyPnL += tradePnL;
                
                if (tradePnL < 0)
                    consecutiveLosses++;
                else
                    consecutiveLosses = 0;
                
                string exitType = tradePnL > 0 ? "✅ WIN" : "❌ LOSS";
                Brush exitColor = tradePnL > 0 ? Brushes.Green : Brushes.Red;
                Draw.Diamond(this, "Exit" + CurrentBar, false, exitType, 0, exitColor);
                
                Print(string.Format("{0}: {1} ${2:F2} | Daily: ${3:F2}", 
                    Time[0], exitType, tradePnL, dailyPnL));
            }
        }

        public override string DisplayName
        {
            get { return "⚡ Dual Order Flow - EVALUATION"; }
        }
    }
}
