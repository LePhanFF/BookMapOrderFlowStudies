"""One-shot script to apply two targeted line-level fixes to RockitORReversalSignal.cs"""
import sys

path = r'C:\Users\lehph\Documents\NinjaTrader 8\bin\Custom\Indicators\RockitORReversalSignal.cs'

with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# ---------- Fix 1: SHORT arm (HIGH sweep) ----------
old1 = (
    '\t\t\t\t// 50% retest entry zone: centered on midpoint between _reversalExtreme and _eorHigh\n'
    '\t\t\t\tdouble fiftyPct = _reversalExtreme + (_eorHigh - _reversalExtreme) * 0.5;\n'
    '\t\t\t\tbool inFiftyPctZone = close >= fiftyPct - _eorRange * 0.15 && close <= fiftyPct + _eorRange * 0.15;'
)
new1 = (
    '\t\t\t\t// 50% retest entry zone: centered on midpoint between _reversalExtreme and _eorHigh\n'
    '\t\t\t\tdouble fiftyPct = _reversalExtreme + (_eorHigh - _reversalExtreme) * 0.5;\n'
    '\t\t\t\t// Use High[0] for zone entry: on 5-min bars the wick touches the zone but close drops back\n'
    '\t\t\t\tbool inFiftyPctZone = High[0] >= fiftyPct - _eorRange * 0.15 && close <= fiftyPct + _eorRange * 0.15;'
)

if old1 in content:
    content = content.replace(old1, new1, 1)
    print('Fix 1 (SHORT/HIGH): APPLIED')
else:
    print('Fix 1 (SHORT/HIGH): NOT FOUND -- checking raw bytes around line 680...')
    lines = content.splitlines()
    for i in range(678, 685):
        print(f'  line {i+1}: {repr(lines[i])}')

# ---------- Fix 2: LONG arm (LOW sweep) ----------
old2 = (
    '\t\t\t\t// 50% retest entry zone: centered on midpoint between _eorLow and _reversalExtreme\n'
    '\t\t\t\tdouble fiftyPct = _reversalExtreme - (_reversalExtreme - _eorLow) * 0.5;\n'
    '\t\t\t\tbool inFiftyPctZone = close >= fiftyPct - _eorRange * 0.15 && close <= fiftyPct + _eorRange * 0.15;'
)
new2 = (
    '\t\t\t\t// 50% retest entry zone: centered on midpoint between _eorLow and _reversalExtreme\n'
    '\t\t\t\tdouble fiftyPct = _reversalExtreme - (_reversalExtreme - _eorLow) * 0.5;\n'
    '\t\t\t\t// Use Low[0] for zone entry: on 5-min bars the wick touches the zone but close bounces back\n'
    '\t\t\t\tbool inFiftyPctZone = Low[0] <= fiftyPct + _eorRange * 0.15 && close >= fiftyPct - _eorRange * 0.15;'
)

if old2 in content:
    content = content.replace(old2, new2, 1)
    print('Fix 2 (LONG/LOW):  APPLIED')
else:
    print('Fix 2 (LONG/LOW):  NOT FOUND -- checking raw bytes around line 745...')
    lines = content.splitlines()
    for i in range(743, 750):
        print(f'  line {i+1}: {repr(lines[i])}')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print('Done.')
