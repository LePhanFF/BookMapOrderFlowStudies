You are a quantitative researcher at Goldman Sach Global markets.  I need a complete time series forecasting model for NQ, ES, YM.

I have provided data in csv format in csv folder

#schema: NinjaDataExport/v2.3, volumetric=True
timestamp,instrument,period,open,high,low,close,volume,ema20,ema50,ema200,rsi14,atr14,vwap,vwap_upper1,vwap_upper2,vwap_upper3,vwap_lower1,vwap_lower2,vwap_lower3,vol_ask,vol_bid,vol_delta,session_date

The data is generated from ninja trader platform for every 1 minute.


I want you to design a system similar to bookmap

https://bookmap.com/blog/can-real-time-order-flow-give-you-an-edge-in-scalp-trading


Optimize the system for expectancy , win rate, manage average win size and everage losing size, make sure no outlier that can wipe out an portfolio.  Assume you are using a 150k prop firm challenge using $4500 in EOD drawdown.

Your goal is to design a system optimize  it, manage its risks, then design a real time python that we can execute every 1 to 5 minutes to find good setups.

Please feel free to do the design, and test.  I use python installed under e:\anaconda 

