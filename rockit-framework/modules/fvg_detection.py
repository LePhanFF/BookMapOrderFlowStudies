# modules/fvg_detection.py
import pandas as pd

def get_fvg_detection(df_extended, df_current, current_time_str="11:45"):
    """
    Detect Fair Value Gaps (FVG) on daily, 4H, 1H, 90min, 15min, and 5min timeframes from last 3-5 days.
    - Standard ICT 3-candle rule
    - Unfilled FVGs only (not mitigated up to current_time)
    - Bullish (discount) / Bearish (premium)
    - Added BPR detection: Checks if FVGs are inverted by opposing engulfing candles.
    - Iteration check: If current/last FVG engulfs past FVGs on 5min/15min.
    """
    current_time = pd.to_datetime(current_time_str).time()
    available_df = df_current[df_current.index.time <= current_time].copy()

    def detect_fvg(df, timeframe):
        fvgs = []
        for i in range(2, len(df)):
            prev = df.iloc[i-2]
            curr = df.iloc[i-1]
            next_bar = df.iloc[i]
            
            # Bullish FVG: prev high < next low
            if prev['high'] < next_bar['low']:
                fvgs.append({
                    'type': 'bullish',
                    'top': next_bar['low'],
                    'bottom': prev['high'],
                    'time': df.index[i].strftime('%Y-%m-%d %H:%M')
                })
            
            # Bearish FVG: prev low > next high
            if prev['low'] > next_bar['high']:
                fvgs.append({
                    'type': 'bearish',
                    'top': prev['low'],
                    'bottom': next_bar['high'],
                    'time': df.index[i].strftime('%Y-%m-%d %H:%M')
                })
        
        # Filter unfilled (not mitigated by price up to current_time)
        unfilled = []
        for fvg in fvgs:
            mitigated = False
            fvg_idx = df.index.get_loc(pd.to_datetime(fvg['time']))
            for j in range(fvg_idx + 1, len(df)):
                bar = df.iloc[j]
                if fvg['type'] == 'bullish' and bar['low'] <= fvg['bottom']:
                    mitigated = True
                    break
                if fvg['type'] == 'bearish' and bar['high'] >= fvg['top']:
                    mitigated = True
                    break
            if not mitigated:
                unfilled.append(fvg)
        
        return unfilled

    def detect_bpr(df, fvgs, timeframe):
        bpr_events = []
        for fvg in fvgs:
            fvg_idx = df.index.get_loc(pd.to_datetime(fvg['time']))
            for i in range(fvg_idx + 1, len(df)):
                bar = df.iloc[i]
                if fvg['type'] == 'bearish' and bar['low'] < fvg['bottom'] and bar['high'] > fvg['top'] and bar['close'] > bar['open']:
                    bpr_events.append({
                        'type': 'bullish_bpr',
                        'fvg_inverted': fvg,
                        'engulfing_time': df.index[i].strftime('%Y-%m-%d %H:%M')
                    })
                elif fvg['type'] == 'bullish' and bar['high'] > fvg['top'] and bar['low'] < fvg['bottom'] and bar['close'] < bar['open']:
                    bpr_events.append({
                        'type': 'bearish_bpr',
                        'fvg_inverted': fvg,
                        'engulfing_time': df.index[i].strftime('%Y-%m-%d %H:%M')
                    })
        return bpr_events

    def check_engulfed_past_fvgs(fvgs, timeframe):
        engulfed = []
        if len(fvgs) < 2:
            return engulfed
        
        current_fvg = fvgs[-1]
        for past_fvg in fvgs[:-1]:
            if current_fvg['type'] != past_fvg['type']:
                if (current_fvg['bottom'] <= past_fvg['bottom'] and current_fvg['top'] >= past_fvg['top']):
                    engulfed.append({
                        'engulfing_fvg': current_fvg,
                        'engulfed_fvg': past_fvg,
                        'timeframe': timeframe
                    })
        return engulfed

    # Resample to timeframes
    daily = available_df.resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    fourh = available_df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    oneh = available_df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    ninetymin = available_df.resample('90min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    fifteenmin = available_df.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    fivemin = available_df.resample('5min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    # Detect FVGs
    daily_fvg = detect_fvg(daily, "daily")
    fourh_fvg = detect_fvg(fourh, "4h")
    oneh_fvg = detect_fvg(oneh, "1h")
    ninetymin_fvg = detect_fvg(ninetymin, "90min")
    fifteenmin_fvg = detect_fvg(fifteenmin, "15min")
    fivemin_fvg = detect_fvg(fivemin, "5min")

    # Detect BPR
    daily_bpr = detect_bpr(daily, daily_fvg, "daily")
    fourh_bpr = detect_bpr(fourh, fourh_fvg, "4h")
    oneh_bpr = detect_bpr(oneh, oneh_fvg, "1h")
    ninetymin_bpr = detect_bpr(ninetymin, ninetymin_fvg, "90min")
    fifteenmin_bpr = detect_bpr(fifteenmin, fifteenmin_fvg, "15min")
    fivemin_bpr = detect_bpr(fivemin, fivemin_fvg, "5min")

    # Check engulfed past FVGs on 5min and 15min
    fivemin_engulfed = check_engulfed_past_fvgs(fivemin_fvg, "5min")
    fifteenmin_engulfed = check_engulfed_past_fvgs(fifteenmin_fvg, "15min")

    return {
        "daily_fvg": daily_fvg,
        "4h_fvg": fourh_fvg,
        "1h_fvg": oneh_fvg,
        "90min_fvg": ninetymin_fvg,
        "15min_fvg": fifteenmin_fvg,
        "5min_fvg": fivemin_fvg,
        "daily_bpr": daily_bpr,
        "4h_bpr": fourh_bpr,
        "1h_bpr": oneh_bpr,
        "90min_bpr": ninetymin_bpr,
        "15min_bpr": fifteenmin_bpr,
        "5min_bpr": fivemin_bpr,
        "5min_engulfed": fivemin_engulfed,
        "15min_engulfed": fifteenmin_engulfed,
        "note": "Unfilled FVGs last 3-5 periods – bullish = discount zone, bearish = premium zone. BPR = opposing engulfing inversion. Engulfed = current FVG fully covers past opposite-type FVG."
    }