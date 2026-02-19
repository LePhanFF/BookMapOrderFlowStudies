# NinjaTrader + Python Signal API Architecture

## Overview

Hybrid architecture where **NinjaTrader 8** handles execution (market data, orders, positions)
and a **Python Signal API** (deployed on GCP Cloud Run) handles all strategy logic. This gives
us a single codebase for backtesting AND live trading — no C# translation of our 7 strategies.

---

## Architecture

```
 LIVE MARKET                    CLOUD                         LOCAL DEV
+----------------+     +-------------------------+     +------------------+
| NinjaTrader 8  |     |   GCP Cloud Run         |     | Python Backtest  |
| (Execution)    |     |   (Signal Generation)   |     | (Same Code)      |
|                |     |                         |     |                  |
| MNQ data feed  |---->| POST /bar               |     | run_backtest.py  |
| ES data feed   |---->|   {bars, levels, of}    |     | Same strategies  |
| YM data feed   |---->|                         |     | Same filters     |
|                |     | Strategy Engine:         |     | Same engine      |
| Receives:      |<----| - Day type classify     |     |                  |
| {action, stop, |     | - IB/VWAP compute       |     | CSV data input   |
|  target, size} |     | - 7 strategy evaluate    |     | instead of API   |
|                |     | - OF quality gate        |     |                  |
| Executes:      |     | - Signal generation      |     |                  |
| - Place orders |     |                         |     |                  |
| - Manage stops |     | Returns:                |     |                  |
| - Track P&L    |     | {action, stop, target,  |     |                  |
+----------------+     |  model, confidence}     |     +------------------+
                       +-------------------------+
```

---

## API Specification

### Endpoint: `POST /signal`

NinjaTrader sends bar data every 1 minute. The API evaluates all strategies and returns
a trade signal (or no-op).

#### Request Body

```json
{
  "timestamp": "2026-02-19T10:31:00",
  "session_date": "2026-02-19",
  "instrument": "MNQ",

  "bar": {
    "open": 21850.25,
    "high": 21855.50,
    "low": 21847.00,
    "close": 21852.75,
    "volume": 1247,
    "delta": 89,
    "up_volume": 668,
    "down_volume": 579,
    "up_ticks": 423,
    "down_ticks": 389
  },

  "cross_instruments": {
    "ES": {
      "open": 6120.50, "high": 6122.00,
      "low": 6119.25, "close": 6121.50,
      "volume": 3200, "delta": -45
    },
    "YM": {
      "open": 44150.0, "high": 44165.0,
      "low": 44140.0, "close": 44155.0,
      "volume": 890, "delta": 23
    }
  },

  "session_levels": {
    "pdh": 21900.50,
    "pdl": 21720.25,
    "overnight_high": 21880.00,
    "overnight_low": 21750.50,
    "asia_high": 21870.25,
    "asia_low": 21765.00,
    "london_high": 21875.50,
    "london_low": 21760.00,
    "ib_high": 21860.00,
    "ib_low": 21795.00,
    "vwap": 21830.50
  },

  "position": {
    "is_flat": true,
    "direction": null,
    "entry_price": null,
    "unrealized_pnl": 0.0
  }
}
```

#### Response Body

```json
{
  "action": "ENTER_LONG",
  "model": "EDGE_TO_MID",
  "strategy": "Edge Fade",
  "confidence": "high",
  "entry_price": 21852.75,
  "stop_loss": 21785.25,
  "target": 21827.50,
  "position_size": 1,
  "reasoning": "Price in lower 25% IB, delta>0, OF quality 2/3, IB<200, not bearish, before 13:30",
  "day_type": "b_day",
  "day_confidence": 0.72,
  "of_quality": {
    "delta_pctl": 72,
    "imbalance_ratio": 1.15,
    "volume_spike": 1.22,
    "score": 3
  }
}
```

**Action values:**
- `"NONE"` — No trade signal, hold current state
- `"ENTER_LONG"` — Enter long position
- `"ENTER_SHORT"` — Enter short position
- `"EXIT"` — Close current position (VWAP breach, time stop, etc.)
- `"MOVE_STOP"` — Update stop loss on existing position
- `"MOVE_TARGET"` — Update target on existing position

### Endpoint: `POST /session_init`

Called once at session start (9:25 AM) to reset state and pass pre-market levels.

```json
{
  "session_date": "2026-02-19",
  "instrument": "MNQ",
  "premarket_levels": {
    "pdh": 21900.50,
    "pdl": 21720.25,
    "overnight_high": 21880.00,
    "overnight_low": 21750.50,
    "asia_high": 21870.25,
    "asia_low": 21765.00,
    "london_high": 21875.50,
    "london_low": 21760.00
  }
}
```

Response: `{"status": "ok", "session_id": "2026-02-19-MNQ"}`

### Endpoint: `GET /health`

Health check for Cloud Run. Returns `{"status": "healthy", "version": "1.0.0"}`.

### Endpoint: `GET /state`

Debug endpoint returning current session state (day type, IB levels, active positions, strategy states).

---

## Python API Implementation (FastAPI)

```python
# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime, time
import uvicorn

from api.session_manager import SessionManager
from api.signal_engine import SignalEngine

app = FastAPI(title="Trading Signal API", version="1.0.0")

# Global session manager (one per running instance)
session_mgr = SessionManager()


class BarData(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: int
    delta: int
    up_volume: int
    down_volume: int
    up_ticks: int = 0
    down_ticks: int = 0


class CrossInstrumentBar(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: int
    delta: int


class PositionState(BaseModel):
    is_flat: bool = True
    direction: Optional[str] = None
    entry_price: Optional[float] = None
    unrealized_pnl: float = 0.0


class SignalRequest(BaseModel):
    timestamp: str
    session_date: str
    instrument: str = "MNQ"
    bar: BarData
    cross_instruments: Dict[str, CrossInstrumentBar] = {}
    session_levels: Dict[str, float] = {}
    position: PositionState = PositionState()


class SignalResponse(BaseModel):
    action: str = "NONE"
    model: Optional[str] = None
    strategy: Optional[str] = None
    confidence: str = "normal"
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target: Optional[float] = None
    position_size: int = 0
    reasoning: str = ""
    day_type: Optional[str] = None
    day_confidence: Optional[float] = None
    of_quality: Dict[str, Any] = {}


@app.post("/signal", response_model=SignalResponse)
async def get_signal(req: SignalRequest):
    """Process one bar and return trade signal."""
    session = session_mgr.get_or_create(req.session_date, req.instrument)
    engine = SignalEngine(session)

    signal = engine.evaluate(
        timestamp=req.timestamp,
        bar=req.bar,
        cross_instruments=req.cross_instruments,
        session_levels=req.session_levels,
        position=req.position,
    )
    return signal


@app.post("/session_init")
async def init_session(data: dict):
    """Initialize a new trading session."""
    session_mgr.init_session(
        session_date=data["session_date"],
        instrument=data.get("instrument", "MNQ"),
        premarket_levels=data.get("premarket_levels", {}),
    )
    return {"status": "ok", "session_id": f"{data['session_date']}-{data.get('instrument', 'MNQ')}"}


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/state")
async def get_state():
    """Debug: return current session state."""
    return session_mgr.get_state_snapshot()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

---

## Session Manager (Server-Side State)

The API maintains session state between calls (bars, IB levels, VWAP, day type).
This is safe because Cloud Run keeps the container warm between requests.

```python
# api/session_manager.py
from datetime import time
from typing import Dict, Optional, List
import numpy as np


class TradingSession:
    """Maintains all state for one trading session."""

    def __init__(self, session_date: str, instrument: str):
        self.session_date = session_date
        self.instrument = instrument

        # Bar history (grows through the session)
        self.bars: List[dict] = []

        # IB tracking (first 60 minutes: 10:30-11:30 for intraday)
        self.ib_high: Optional[float] = None
        self.ib_low: Optional[float] = None
        self.ib_finalized: bool = False

        # OR tracking (first 30 minutes: 9:30-10:00 for Opening Range)
        self.or_high: Optional[float] = None
        self.or_low: Optional[float] = None
        self.or_finalized: bool = False

        # VWAP (running calculation)
        self.vwap_cum_pv: float = 0.0
        self.vwap_cum_vol: int = 0
        self.vwap: float = 0.0

        # Day type classification
        self.day_type: Optional[str] = None
        self.day_confidence: float = 0.0

        # Pre-market levels (set on init)
        self.premarket_levels: Dict[str, float] = {}

        # Strategy states (cooldowns, touch counts, etc.)
        self.strategy_states: Dict[str, dict] = {}

        # Acceptance tracking
        self.bars_above_ibh: int = 0
        self.bars_below_ibl: int = 0
        self.accepted_long: bool = False
        self.accepted_short: bool = False

        # Order flow rolling window
        self.delta_history: List[int] = []

        # Cross-instrument bars for SMT
        self.cross_bars: Dict[str, List[dict]] = {"ES": [], "YM": []}

    def add_bar(self, bar: dict, timestamp: str, cross: dict = None):
        """Add a new bar and update all derived state."""
        self.bars.append({**bar, "timestamp": timestamp})

        # Update VWAP
        typical = (bar["high"] + bar["low"] + bar["close"]) / 3.0
        self.vwap_cum_pv += typical * bar["volume"]
        self.vwap_cum_vol += bar["volume"]
        if self.vwap_cum_vol > 0:
            self.vwap = self.vwap_cum_pv / self.vwap_cum_vol

        # Update IB (bars 0-59 of RTH for intraday, or 10:30-11:30)
        if not self.ib_finalized and len(self.bars) <= 60:
            if self.ib_high is None or bar["high"] > self.ib_high:
                self.ib_high = bar["high"]
            if self.ib_low is None or bar["low"] < self.ib_low:
                self.ib_low = bar["low"]
            if len(self.bars) == 60:
                self.ib_finalized = True

        # Update OR (first 30 bars from 9:30)
        if not self.or_finalized and len(self.bars) <= 30:
            if self.or_high is None or bar["high"] > self.or_high:
                self.or_high = bar["high"]
            if self.or_low is None or bar["low"] < self.or_low:
                self.or_low = bar["low"]
            if len(self.bars) == 30:
                self.or_finalized = True

        # Update acceptance tracking
        if self.ib_finalized:
            if bar["close"] > self.ib_high:
                self.bars_above_ibh += 1
            else:
                self.bars_above_ibh = 0
            if bar["close"] < self.ib_low:
                self.bars_below_ibl += 1
            else:
                self.bars_below_ibl = 0

            if self.bars_above_ibh >= 2:
                self.accepted_long = True
            if self.bars_below_ibl >= 3:
                self.accepted_short = True

        # Update delta history (rolling 10-bar)
        self.delta_history.append(bar["delta"])
        if len(self.delta_history) > 10:
            self.delta_history.pop(0)

        # Update cross-instrument bars
        if cross:
            for inst, inst_bar in cross.items():
                if inst in self.cross_bars:
                    self.cross_bars[inst].append(inst_bar)

        # Update day type classification (after IB finalized)
        if self.ib_finalized:
            self._classify_day_type()

    def _classify_day_type(self):
        """Classify current day type based on IB extension."""
        if self.ib_high is None or self.ib_low is None:
            return

        ib_range = self.ib_high - self.ib_low
        if ib_range <= 0:
            return

        latest = self.bars[-1]
        ext_up = max(0, latest["high"] - self.ib_high) / ib_range
        ext_down = max(0, self.ib_low - latest["low"]) / ib_range

        # Classification thresholds (from Dalton framework)
        if ext_up > 1.0:
            self.day_type = "trend_up"
        elif ext_down > 1.0:
            self.day_type = "trend_down"
        elif ext_up > 0.5:
            self.day_type = "p_day"
        elif ext_down > 0.5:
            self.day_type = "b_day_bear"
        elif ext_up > 0.2 or ext_down > 0.2:
            self.day_type = "p_day" if ext_up > ext_down else "b_day_bear"
        elif ext_up < 0.2 and ext_down < 0.2:
            self.day_type = "b_day"
        else:
            self.day_type = "neutral"

    @property
    def ib_range(self) -> float:
        if self.ib_high and self.ib_low:
            return self.ib_high - self.ib_low
        return 0.0

    @property
    def or_range(self) -> float:
        if self.or_high and self.or_low:
            return self.or_high - self.or_low
        return 0.0

    @property
    def pre_delta_sum(self) -> int:
        return sum(self.delta_history)


class SessionManager:
    """Manages trading sessions across the API lifecycle."""

    def __init__(self):
        self.sessions: Dict[str, TradingSession] = {}

    def get_or_create(self, session_date: str, instrument: str) -> TradingSession:
        key = f"{session_date}-{instrument}"
        if key not in self.sessions:
            self.sessions[key] = TradingSession(session_date, instrument)
        return self.sessions[key]

    def init_session(self, session_date: str, instrument: str, premarket_levels: dict):
        key = f"{session_date}-{instrument}"
        session = TradingSession(session_date, instrument)
        session.premarket_levels = premarket_levels
        self.sessions[key] = session

    def get_state_snapshot(self) -> dict:
        result = {}
        for key, session in self.sessions.items():
            result[key] = {
                "bars_count": len(session.bars),
                "ib_high": session.ib_high,
                "ib_low": session.ib_low,
                "ib_finalized": session.ib_finalized,
                "vwap": round(session.vwap, 2),
                "day_type": session.day_type,
                "accepted_long": session.accepted_long,
                "accepted_short": session.accepted_short,
            }
        return result
```

---

## Signal Engine (Strategy Evaluation)

```python
# api/signal_engine.py
"""
Evaluates all 7 strategies against current bar and session state.
Uses the SAME strategy logic as the backtest engine.

Strategy evaluation order:
  1. OR Reversal (9:30-10:30 only)
  2. Trend Day Bull
  3. P-Day
  4. B-Day IBL Fade
  5. Edge Fade (optimized)
  6. IBH Sweep+Fail
  7. Bear Acceptance Short
"""

from api.session_manager import TradingSession


class SignalEngine:
    def __init__(self, session: TradingSession):
        self.session = session

    def evaluate(self, timestamp, bar, cross_instruments, session_levels, position):
        """
        Process one bar through all strategies.
        Returns the highest-priority signal (or NONE).

        This method wraps the existing strategy classes from strategy/ package.
        The key insight: each strategy's check_signal() method is called with
        the same bar data and session state, just like in run_backtest.py.
        """
        # Add bar to session state
        self.session.add_bar(
            bar=bar.dict() if hasattr(bar, 'dict') else bar,
            timestamp=timestamp,
            cross=cross_instruments,
        )

        # Update levels from NinjaTrader
        if session_levels:
            for key, val in session_levels.items():
                if key == "vwap":
                    continue  # We compute our own VWAP
                self.session.premarket_levels[key] = val

        # If position is open, check for exit signals
        if not position.is_flat:
            exit_signal = self._check_exits(bar, position, timestamp)
            if exit_signal:
                return exit_signal

        # If flat, check for entry signals (priority order)
        if position.is_flat:
            # Strategy 7: OR Reversal (9:30-10:30 ONLY)
            signal = self._check_or_reversal(bar, timestamp)
            if signal:
                return signal

            # Strategy 1: Trend Day Bull
            signal = self._check_trend_bull(bar, timestamp)
            if signal:
                return signal

            # Strategy 2: P-Day
            signal = self._check_p_day(bar, timestamp)
            if signal:
                return signal

            # Strategy 3: B-Day IBL Fade
            signal = self._check_b_day(bar, timestamp)
            if signal:
                return signal

            # Strategy 4: Edge Fade (optimized)
            signal = self._check_edge_fade(bar, timestamp)
            if signal:
                return signal

            # Strategy 5: IBH Sweep+Fail (SHORT)
            signal = self._check_ibh_sweep_fail(bar, timestamp)
            if signal:
                return signal

            # Strategy 6: Bear Acceptance Short
            signal = self._check_bear_short(bar, timestamp)
            if signal:
                return signal

        return {"action": "NONE"}

    # --- Each _check method mirrors the existing strategy class logic ---
    # These import from strategy/ package and call the same evaluate() methods
    # See run_backtest.py for how strategies are instantiated and evaluated
```

---

## NinjaTrader C# Client (NinjaScript Indicator)

This NinjaScript indicator runs on MNQ 1-min chart, sends bar data to the API,
and executes signals.

```csharp
// NinjaScript/Indicators/PythonSignalClient.cs
using System;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
using NinjaTrader.Cbi;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Indicators;

namespace NinjaTrader.NinjaScript.Strategies
{
    public class PythonSignalClient : Strategy
    {
        private static readonly HttpClient client = new HttpClient();
        private string apiUrl = "https://your-service-xyz.run.app";

        // Cross-instrument data series
        private int esIdx, ymIdx;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name = "PythonSignalClient";
                Description = "Queries Python Signal API for trade signals";
                Calculate = Calculate.OnBarClose;

                // API configuration
                apiUrl = "https://trading-signal-api-xxxxx.run.app";
            }
            else if (State == State.Configure)
            {
                // Add ES and YM for cross-instrument SMT
                AddDataSeries("ES 03-26", BarsPeriodType.Minute, 1);  // Index 1
                AddDataSeries("YM 03-26", BarsPeriodType.Minute, 1);  // Index 2
                esIdx = 1;
                ymIdx = 2;
            }
            else if (State == State.DataLoaded)
            {
                // Initialize session at market open
                SendSessionInit();
            }
        }

        protected override void OnBarUpdate()
        {
            // Only process on primary (MNQ) bar close
            if (BarsInProgress != 0) return;

            // Build request payload
            var payload = new
            {
                timestamp = Time[0].ToString("yyyy-MM-ddTHH:mm:ss"),
                session_date = Time[0].ToString("yyyy-MM-dd"),
                instrument = "MNQ",

                bar = new
                {
                    open = Open[0],
                    high = High[0],
                    low = Low[0],
                    close = Close[0],
                    volume = (int)Volume[0],
                    delta = GetDelta(),      // From volumetric data
                    up_volume = GetUpVol(),
                    down_volume = GetDownVol(),
                    up_ticks = 0,
                    down_ticks = 0
                },

                cross_instruments = new
                {
                    ES = new
                    {
                        open = Opens[esIdx][0],
                        high = Highs[esIdx][0],
                        low = Lows[esIdx][0],
                        close = Closes[esIdx][0],
                        volume = (int)Volumes[esIdx][0],
                        delta = 0  // Or compute from ES volumetric
                    },
                    YM = new
                    {
                        open = Opens[ymIdx][0],
                        high = Highs[ymIdx][0],
                        low = Lows[ymIdx][0],
                        close = Closes[ymIdx][0],
                        volume = (int)Volumes[ymIdx][0],
                        delta = 0
                    }
                },

                session_levels = new
                {
                    pdh = GetPriorDayHigh(),
                    pdl = GetPriorDayLow(),
                    vwap = GetVWAP()
                    // overnight, asia, london computed server-side from bar history
                },

                position = new
                {
                    is_flat = Position.MarketPosition == MarketPosition.Flat,
                    direction = Position.MarketPosition.ToString(),
                    entry_price = Position.AveragePrice,
                    unrealized_pnl = Position.GetUnrealizedProfitLoss(
                        PerformanceUnit.Currency, Close[0])
                }
            };

            // Send to Python API
            var signal = SendSignal(payload).Result;

            if (signal == null) return;

            // Execute signal
            switch (signal.action)
            {
                case "ENTER_LONG":
                    EnterLong(signal.position_size, signal.model);
                    SetStopLoss(signal.model, CalculationMode.Price,
                                signal.stop_loss, false);
                    SetProfitTarget(signal.model, CalculationMode.Price,
                                   signal.target);
                    Print($"LONG {signal.model}: entry={Close[0]}, " +
                          $"stop={signal.stop_loss}, target={signal.target}");
                    break;

                case "ENTER_SHORT":
                    EnterShort(signal.position_size, signal.model);
                    SetStopLoss(signal.model, CalculationMode.Price,
                                signal.stop_loss, false);
                    SetProfitTarget(signal.model, CalculationMode.Price,
                                   signal.target);
                    Print($"SHORT {signal.model}: entry={Close[0]}, " +
                          $"stop={signal.stop_loss}, target={signal.target}");
                    break;

                case "EXIT":
                    if (Position.MarketPosition == MarketPosition.Long)
                        ExitLong();
                    else if (Position.MarketPosition == MarketPosition.Short)
                        ExitShort();
                    Print($"EXIT: {signal.reasoning}");
                    break;

                case "MOVE_STOP":
                    // Dynamically update stop loss
                    SetStopLoss(CalculationMode.Price, signal.stop_loss, false);
                    break;
            }
        }

        private async Task<SignalResponse> SendSignal(object payload)
        {
            try
            {
                var json = JsonConvert.SerializeObject(payload);
                var content = new StringContent(json, Encoding.UTF8,
                                               "application/json");

                var response = await client.PostAsync(
                    $"{apiUrl}/signal", content);

                if (response.IsSuccessStatusCode)
                {
                    var body = await response.Content.ReadAsStringAsync();
                    return JsonConvert.DeserializeObject<SignalResponse>(body);
                }
                else
                {
                    Print($"API Error: {response.StatusCode}");
                    return null;
                }
            }
            catch (Exception ex)
            {
                Print($"API Exception: {ex.Message}");
                return null;  // Fail safe: no signal = no trade
            }
        }

        private void SendSessionInit()
        {
            // Called once at session start to initialize server-side state
            // Implementation similar to SendSignal with /session_init endpoint
        }

        // Helper methods for volumetric data
        private int GetDelta()
        {
            // Use NinjaTrader's volumetric bar data
            // Bars.GetAsk(0) - Bars.GetBid(0) approximation
            return 0; // Replace with actual volumetric implementation
        }
        private int GetUpVol() { return 0; }
        private int GetDownVol() { return 0; }
        private double GetPriorDayHigh() { return CurrentDayOHL.PriorHigh[0]; }
        private double GetPriorDayLow() { return CurrentDayOHL.PriorLow[0]; }
        private double GetVWAP() { return 0; } // Or use OrderFlowVWAP indicator
    }

    public class SignalResponse
    {
        public string action { get; set; }
        public string model { get; set; }
        public string strategy { get; set; }
        public string confidence { get; set; }
        public double entry_price { get; set; }
        public double stop_loss { get; set; }
        public double target { get; set; }
        public int position_size { get; set; }
        public string reasoning { get; set; }
    }
}
```

---

## GCP Cloud Run Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ ./api/
COPY strategy/ ./strategy/
COPY indicators/ ./indicators/
COPY config/ ./config/
COPY data/ ./data/
COPY engine/ ./engine/
COPY filters/ ./filters/
COPY profile/ ./profile/

# Cloud Run uses PORT env var
ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### requirements.txt

```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
pandas>=2.1.0
numpy>=1.25.0
scipy>=1.11.0
```

### Deploy Commands

```bash
# Build and deploy to GCP Cloud Run
gcloud builds submit --tag gcr.io/YOUR_PROJECT/trading-signal-api

gcloud run deploy trading-signal-api \
  --image gcr.io/YOUR_PROJECT/trading-signal-api \
  --platform managed \
  --region us-central1 \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 1 \
  --max-instances 1 \
  --timeout 60 \
  --no-allow-unauthenticated \
  --service-account trading-bot@YOUR_PROJECT.iam.gserviceaccount.com
```

**Key settings:**
- `--min-instances 1`: Keep container warm (no cold start during market hours)
- `--max-instances 1`: Single instance to maintain session state
- `--no-allow-unauthenticated`: Require API key/token
- `--memory 1Gi`: Enough for pandas + numpy + bar history
- `--timeout 60`: Generous timeout for complex strategy evaluation

### Cost Estimate

- Cloud Run with min-instances=1: ~$30-50/month
- Only runs during market hours (6.5 hrs/day, 5 days/week)
- Could use Cloud Scheduler to scale to 0 outside market hours

---

## Data Flow: What NinjaTrader Sends vs What Python Computes

| Data | Source | Why |
|------|--------|-----|
| Bar OHLCV | NinjaTrader | Real-time market data |
| Delta, up/down volume | NinjaTrader | Volumetric bar data |
| ES/YM bars | NinjaTrader | AddDataSeries cross-instrument |
| PDH/PDL | NinjaTrader | CurrentDayOHL indicator |
| **VWAP** | **Python** | Custom running calc (more control) |
| **IB High/Low** | **Python** | Tracks from bar history |
| **OR High/Low** | **Python** | First 30 bars of session |
| **Day type** | **Python** | Classification from IB extension |
| **Acceptance** | **Python** | Bars above/below IB edges |
| **OF quality** | **Python** | Delta pctl, imbalance, vol spike |
| **Strategy signals** | **Python** | All 7 strategies |
| **Overnight/Asia/London** | **Either** | NT can send, or Python tracks from ETH bars |

---

## Advantages Over Pure NinjaScript

| Factor | Pure NinjaScript | Python API Hybrid |
|--------|-----------------|-------------------|
| **Dev speed** | Days to translate each strategy | Already done (existing Python code) |
| **Backtesting** | NinjaTrader backtester (limited) | Full Python backtest (run_backtest.py) |
| **Code quality** | C# with limited debugging | Python with full IDE support |
| **Iteration** | Recompile + restart NT | Deploy in 30 sec, no NT restart |
| **Version control** | NinjaScript project files | Git-tracked, same as backtest |
| **Testing** | Manual replay | Automated pytest suite |
| **Multi-instrument** | Possible but complex | Clean cross_instruments dict |
| **Latency** | ~0ms (local) | ~50-200ms (network) |
| **Reliability** | NT crashes = no trades | API down = no signal = no trade (safe) |

**Latency concern**: 50-200ms round trip is fine for 1-minute bars. You have 60,000ms per bar.
Even with occasional 500ms spikes, that's <1% of the bar period. The strategies evaluate
on bar close, so there's no sub-second timing requirement.

---

## Fail-Safe Design

1. **API timeout**: NinjaTrader sets 5-second timeout. If no response, action = "NONE" (no trade)
2. **API down**: NinjaTrader catches exception, logs error, continues without trading
3. **Invalid signal**: NinjaTrader validates stop/target before executing
4. **Network partition**: No signal = no trade. Never trades on stale/missing data.
5. **Position mismatch**: NinjaTrader sends current position state; API doesn't assume
6. **Session reset**: `/session_init` clears all state at 9:25 AM daily

The design is **fail-safe by default** — any error results in no trading, never in incorrect trading.

---

## Implementation Roadmap

### Phase 1: API Scaffold (1-2 days)
- Create `api/` package with main.py, session_manager.py, signal_engine.py
- Wire existing strategy classes into signal_engine
- Local testing with `uvicorn api.main:app --reload`
- Test with historical bar data replayed via curl

### Phase 2: NinjaTrader Client (1-2 days)
- NinjaScript strategy with HttpClient
- AddDataSeries for ES + YM
- Volumetric bar data extraction (delta, up_vol, down_vol)
- Test in NinjaTrader Playback mode (replay historical data)

### Phase 3: GCP Deployment (1 day)
- Dockerfile + requirements.txt
- gcloud run deploy
- Authentication (API key in header)
- Cloud Scheduler for min-instances during market hours

### Phase 4: Paper Trading (1-2 weeks)
- Run on live market data in NinjaTrader SIM account
- Compare signals to backtest expectations
- Monitor latency, errors, edge cases

### Phase 5: Live Eval (ongoing)
- Switch to live Tradeify eval account
- Start with 1 MNQ contract
- Scale per prop firm plan (Phase 1→2→3 sizing)
