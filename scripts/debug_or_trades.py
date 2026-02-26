"""Debug OR Reversal detection for Feb 4 and Feb 6 2026."""
import pandas as pd
import numpy as np
from datetime import timedelta

# Load data
df = pd.read_csv('csv/NQ_Volumetric_1.csv', low_memory=False, comment='#')
for col in ['open','high','low','close','volume','vwap','vol_ask','vol_bid','vol_delta']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
df['session_date'] = pd.to_datetime(df['session_date'], errors='coerce')

def find_closest(extreme, candidates, thresh, eor_r):
    best_level, best_name, best_dist = None, None, float('inf')
    for name, lvl in candidates:
        if lvl is None: continue
        dist = abs(extreme - lvl)
        if dist < thresh and dist <= eor_r and dist < best_dist:
            best_dist = dist
            best_level = lvl
            best_name = name
    return best_level, best_name

for target_date in ['2026-02-04', '2026-02-06']:
    print(f'\n{"="*80}')
    print(f'ANALYSIS: {target_date}')
    print(f'{"="*80}')

    sd = pd.Timestamp(target_date)
    sess = df[df['session_date'] == sd].copy()

    # Get RTH bars (9:30-16:00)
    rth = sess[(sess['timestamp'].dt.hour >= 9) & (sess['timestamp'].dt.hour < 16)].copy()
    rth = rth[(rth['timestamp'].dt.hour > 9) | (rth['timestamp'].dt.minute >= 30)].copy()
    rth = rth.reset_index(drop=True)

    print(f'RTH bars: {len(rth)}')

    if len(rth) < 60:
        print('Not enough bars')
        continue

    # IB = first 60 bars
    ib_bars = rth.iloc[:60]
    ib_high = ib_bars['high'].max()
    ib_low = ib_bars['low'].min()
    ib_range = ib_high - ib_low

    # OR = first 15 bars (9:30-9:44)
    or_bars = rth.iloc[:15]
    or_high = or_bars['high'].max()
    or_low = or_bars['low'].min()
    or_mid = (or_high + or_low) / 2

    # EOR = first 30 bars (9:30-9:59)
    eor_bars = rth.iloc[:30]
    eor_high = eor_bars['high'].max()
    eor_low = eor_bars['low'].min()
    eor_range = eor_high - eor_low
    eor_mid = (eor_high + eor_low) / 2

    print(f'\nOR: high={or_high}, low={or_low}, mid={or_mid:.2f}')
    print(f'EOR: high={eor_high}, low={eor_low}, range={eor_range:.2f}, mid={eor_mid:.2f}')
    print(f'IB: high={ib_high}, low={ib_low}, range={ib_range:.2f}')

    # Opening drive from first 5 bars
    first_5 = rth.iloc[:5]
    open_price = first_5.iloc[0]['open']
    close_5th = first_5.iloc[4]['close']
    drive_range = first_5['high'].max() - first_5['low'].min()
    if drive_range > 0:
        drive_pct = (close_5th - open_price) / drive_range
    else:
        drive_pct = 0

    if drive_pct > 0.4:
        opening_drive = 'DRIVE_UP'
    elif drive_pct < -0.4:
        opening_drive = 'DRIVE_DOWN'
    else:
        opening_drive = 'ROTATION'

    print(f'\nOpening Drive: open={open_price}, close_5th={close_5th}, drive_range={drive_range:.2f}')
    print(f'  drive_pct={drive_pct:.4f} -> {opening_drive}')

    # Overnight levels
    prev_day = sd - timedelta(days=1)
    if prev_day.weekday() == 5: prev_day -= timedelta(days=1)
    elif prev_day.weekday() == 6: prev_day -= timedelta(days=2)

    on_mask = (df['timestamp'] >= prev_day + timedelta(hours=18)) & (df['timestamp'] < sd + timedelta(hours=9, minutes=30))
    on_bars_df = df[on_mask]
    on_high = on_bars_df['high'].max() if len(on_bars_df) > 0 else None
    on_low = on_bars_df['low'].min() if len(on_bars_df) > 0 else None

    london_mask = (df['timestamp'] >= sd + timedelta(hours=2)) & (df['timestamp'] < sd + timedelta(hours=5))
    london = df[london_mask]
    ldn_high = london['high'].max() if len(london) > 0 else None
    ldn_low = london['low'].min() if len(london) > 0 else None

    prior_rth_mask = (df['timestamp'] >= prev_day + timedelta(hours=9, minutes=30)) & (df['timestamp'] <= prev_day + timedelta(hours=16))
    prior_rth = df[prior_rth_mask]
    pdh = prior_rth['high'].max() if len(prior_rth) > 0 else None
    pdl = prior_rth['low'].min() if len(prior_rth) > 0 else None

    print(f'\nKey Levels:')
    print(f'  ON High={on_high}, ON Low={on_low}')
    print(f'  London High={ldn_high}, London Low={ldn_low}')
    print(f'  PDH={pdh}, PDL={pdl}')

    # Sweep detection
    sweep_threshold = eor_range * 0.17
    print(f'\nSweep threshold: {sweep_threshold:.2f} (17% of EOR range {eor_range:.2f})')

    high_candidates = [('ON_HIGH', on_high), ('PDH', pdh), ('LDN_HIGH', ldn_high)]
    low_candidates = [('ON_LOW', on_low), ('PDL', pdl), ('LDN_LOW', ldn_low)]

    print(f'\nHigh sweep analysis (EOR high = {eor_high}):')
    for name, lvl in high_candidates:
        if lvl is not None:
            dist = abs(eor_high - lvl)
            within = 'YES' if dist < sweep_threshold and dist <= eor_range else 'NO'
            print(f'  {name}={lvl}, dist={dist:.2f}, within threshold={within}')

    print(f'\nLow sweep analysis (EOR low = {eor_low}):')
    for name, lvl in low_candidates:
        if lvl is not None:
            dist = abs(eor_low - lvl)
            within = 'YES' if dist < sweep_threshold and dist <= eor_range else 'NO'
            print(f'  {name}={lvl}, dist={dist:.2f}, within threshold={within}')

    swept_high_level, swept_high_name = find_closest(eor_high, high_candidates, sweep_threshold, eor_range)
    swept_low_level, swept_low_name = find_closest(eor_low, low_candidates, sweep_threshold, eor_range)

    print(f'\nSweep results:')
    print(f'  High swept: {swept_high_name}={swept_high_level}')
    print(f'  Low swept: {swept_low_name}={swept_low_level}')

    # Dual sweep
    if swept_high_level is not None and swept_low_level is not None:
        high_depth = eor_high - swept_high_level
        low_depth = swept_low_level - eor_low
        print(f'  DUAL SWEEP - high_depth={high_depth:.2f}, low_depth={low_depth:.2f}')
        if high_depth >= low_depth:
            swept_low_level = None; swept_low_name = None
            print(f'  -> Keeping HIGH sweep')
        else:
            swept_high_level = None; swept_high_name = None
            print(f'  -> Keeping LOW sweep')

    if swept_high_level is None and swept_low_level is None:
        print('\n*** NO SWEEP DETECTED -- strategy would NOT fire ***')
        # Still show delta/CVD and price action

    # CVD and delta analysis
    deltas = rth['vol_delta'].fillna(0)
    cvd = deltas.cumsum()

    print(f'\n--- Delta/CVD for first 30 bars (EOR) ---')
    for i in range(30):
        bar = rth.iloc[i]
        ts = bar['timestamp'].strftime('%H:%M')
        d = bar['vol_delta'] if pd.notna(bar['vol_delta']) else 0
        print(f'  Bar {i:2d} ({ts}): O={bar["open"]:>9.2f} H={bar["high"]:>9.2f} L={bar["low"]:>9.2f} C={bar["close"]:>9.2f} delta={d:>6.0f} CVD={cvd.iloc[i]:>8.0f}')

    if swept_high_level is None and swept_low_level is None:
        # Show post-EOR price action anyway
        print(f'\n--- Price action 10:00-10:30 (post-EOR) ---')
        for i in range(30, 61):
            if i >= len(rth): break
            bar = rth.iloc[i]
            ts = bar['timestamp'].strftime('%H:%M')
            d = bar['vol_delta'] if pd.notna(bar['vol_delta']) else 0
            print(f'  Bar {i:2d} ({ts}): O={bar["open"]:>9.2f} H={bar["high"]:>9.2f} L={bar["low"]:>9.2f} C={bar["close"]:>9.2f} delta={d:>6.0f} CVD={cvd.iloc[i]:>8.0f}')
        print(f'\n--- Session outcome ---')
        print(f'  Session high: {rth["high"].max()}, low: {rth["low"].min()}')
        print(f'  Close: {rth.iloc[-1]["close"]}')
        continue

    # Check for reversal signal
    print(f'\n--- Signal Check ---')
    vwap_threshold = eor_range * 0.17
    max_risk = eor_range * 1.3

    if swept_high_level is not None and opening_drive != 'DRIVE_UP':
        print(f'Checking SHORT setup (sweep of {swept_high_name}={swept_high_level})')
        high_bar_idx = eor_bars['high'].idxmax()
        cvd_at_extreme = cvd.iloc[high_bar_idx]
        print(f'  High extreme at bar idx {high_bar_idx}, CVD at extreme={cvd_at_extreme:.0f}')

        post_high = rth.loc[high_bar_idx:]
        found = False
        for j in range(1, min(30, len(post_high))):
            bar = post_high.iloc[j]
            price = bar['close']
            vwap = bar.get('vwap', np.nan)
            delta = bar.get('vol_delta', 0) or 0
            cvd_at_entry = cvd.loc[post_high.index[j]]
            cvd_declining = cvd_at_entry < cvd_at_extreme

            ts = bar['timestamp'].strftime('%H:%M')
            below_or_mid = price < or_mid
            vwap_ok = not pd.isna(vwap) and abs(price - vwap) <= vwap_threshold
            delta_ok = delta < 0 or cvd_declining

            stop = swept_high_level + eor_range * 0.15
            risk = stop - price
            risk_ok = risk >= eor_range * 0.03 and risk <= max_risk
            target = price - 2 * risk if risk > 0 else 0

            if below_or_mid:
                print(f'  Post-extreme bar {j} ({ts}): close={price:.2f}, or_mid={or_mid:.2f}, below_mid={below_or_mid}, vwap={vwap:.2f}, vwap_ok={vwap_ok}, delta={delta:.0f}, cvd={cvd_at_entry:.0f}, cvd_declining={cvd_declining}, risk={risk:.2f}, risk_ok={risk_ok}')
                if vwap_ok and delta_ok and risk_ok:
                    print(f'  *** SHORT SIGNAL: entry={price:.2f}, stop={stop:.2f}, target={target:.2f} ***')
                    found = True
                    break
        if not found:
            print(f'  No valid SHORT reversal found in post-extreme bars')

    if swept_low_level is not None and opening_drive != 'DRIVE_DOWN':
        print(f'Checking LONG setup (sweep of {swept_low_name}={swept_low_level})')
        low_bar_idx = eor_bars['low'].idxmin()
        cvd_at_extreme = cvd.iloc[low_bar_idx]
        print(f'  Low extreme at bar idx {low_bar_idx}, CVD at extreme={cvd_at_extreme:.0f}')

        post_low = rth.loc[low_bar_idx:]
        found = False
        for j in range(1, min(30, len(post_low))):
            bar = post_low.iloc[j]
            price = bar['close']
            vwap = bar.get('vwap', np.nan)
            delta = bar.get('vol_delta', 0) or 0
            cvd_at_entry = cvd.loc[post_low.index[j]]
            cvd_rising = cvd_at_entry > cvd_at_extreme

            ts = bar['timestamp'].strftime('%H:%M')
            above_or_mid = price > or_mid
            vwap_ok = not pd.isna(vwap) and abs(price - vwap) <= vwap_threshold
            delta_ok = delta > 0 or cvd_rising

            stop = swept_low_level - eor_range * 0.15
            risk = price - stop
            risk_ok = risk >= eor_range * 0.03 and risk <= max_risk
            target = price + 2 * risk if risk > 0 else 0

            if above_or_mid:
                print(f'  Post-extreme bar {j} ({ts}): close={price:.2f}, or_mid={or_mid:.2f}, above_mid={above_or_mid}, vwap={vwap:.2f}, vwap_ok={vwap_ok}, delta={delta:.0f}, cvd={cvd_at_entry:.0f}, cvd_rising={cvd_rising}, risk={risk:.2f}, risk_ok={risk_ok}')
                if vwap_ok and delta_ok and risk_ok:
                    print(f'  *** LONG SIGNAL: entry={price:.2f}, stop={stop:.2f}, target={target:.2f} ***')
                    found = True
                    break
        if not found:
            print(f'  No valid LONG reversal found in post-extreme bars')

    # Show post-EOR price action
    print(f'\n--- Price action 10:00-10:30 (post-EOR) ---')
    for i in range(30, 61):
        if i >= len(rth): break
        bar = rth.iloc[i]
        ts = bar['timestamp'].strftime('%H:%M')
        d = bar['vol_delta'] if pd.notna(bar['vol_delta']) else 0
        print(f'  Bar {i:2d} ({ts}): O={bar["open"]:>9.2f} H={bar["high"]:>9.2f} L={bar["low"]:>9.2f} C={bar["close"]:>9.2f} delta={d:>6.0f} CVD={cvd.iloc[i]:>8.0f}')

    print(f'\n--- Session outcome ---')
    print(f'  Session high: {rth["high"].max()}, low: {rth["low"].min()}')
    print(f'  Close: {rth.iloc[-1]["close"]}')
