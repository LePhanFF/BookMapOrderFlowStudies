import sys

path = r'C:\Users\lehph\Documents\NinjaTrader 8\bin\Custom\Indicators\RockitORReversalSignal.cs'

with open(path, 'r', encoding='utf-8-sig') as f:
    content = f.read()

orig_len = len(content)
print(f"File read OK. Length={orig_len} chars")

# ── Change 1 ──────────────────────────────────────────────────────────────────
# After `_sessionIterator = new SessionIterator(Bars);` insert `_atr = ATR(14);`
old1 = '_sessionIterator = new SessionIterator(Bars);'
new1 = '_sessionIterator = new SessionIterator(Bars);\n\t\t\t\t_atr = ATR(14);'
c1 = content.count(old1)
print(f"Change 1 occurrences: {c1}")
assert c1 == 1, f"Expected 1, got {c1}"
content = content.replace(old1, new1)
print("Change 1 applied.")

# ── Change 2 (SHORT arm) ──────────────────────────────────────────────────────
old2 = (
    "if (retestReachedMid && inEntryZone && turningDown && ofConfirm)\n"
    "\t\t\t\t{\n"
    "\t\t\t\t\t// FIRE SHORT \u2014 stop at swept reference level + small buffer\n"
    "\t\t\t\t\tdouble entry = close;\n"
    "\t\t\t\t\tdouble stop = _sweptRefPrice + _eorRange * StopBufferPct;\n"
    "\t\t\t\t\tdouble risk = stop - entry;\n"
    "\t\t\t\t\tdouble target = entry - TargetMultiple * risk;\n"
    "\n"
    "\t\t\t\t\tif (risk > 20 && risk < 300)\n"
    "\t\t\t\t\t{\n"
    "\t\t\t\t\t\t_sweepLevel = _sweepLevelName;\n"
    "\t\t\t\t\t\tTrace(\"[SM] >>>> FIRING SHORT via RETEST FAILURE at bar \" + rthBar + \" <<<<\");\n"
    "\t\t\t\t\t\tTrace(\"[SM] Entry near mid=\" + mid.ToString(\"F2\") + \" | retestPeak=\" + _retestPeak.ToString(\"F2\")\n"
    "\t\t\t\t\t\t\t+ \" \u2192 turned down at \" + close.ToString(\"F2\")\n"
    "\t\t\t\t\t\t\t+ \" | OF: barDelta=\" + delta + \" cvdSinceRev=\" + cvdSinceReversal);\n"
    "\t\t\t\t\t\tFireSignal(\"SHORT\", entry, stop, target);\n"
    "\t\t\t\t\t}\n"
    "\t\t\t\t\telse\n"
    "\t\t\t\t\t{\n"
    "\t\t\t\t\t\t_lastSkipReason = \"SHORT: risk=\" + risk.ToString(\"F0\") + \" out of bounds [20-300]\";\n"
    "\t\t\t\t\t\tTrace(\"[SM] \" + _lastSkipReason);\n"
    "\t\t\t\t\t}\n"
    "\t\t\t\t}"
)

new2 = (
    "// 50% retest entry zone: centered on midpoint between _reversalExtreme and _eorHigh\n"
    "\t\t\t\tdouble fiftyPct = _reversalExtreme + (_eorHigh - _reversalExtreme) * 0.5;\n"
    "\t\t\t\tbool inFiftyPctZone = close >= fiftyPct - _eorRange * 0.15 && close <= fiftyPct + _eorRange * 0.15;\n"
    "\n"
    "\t\t\t\tif (retestReachedMid && inFiftyPctZone && turningDown && ofConfirm)\n"
    "\t\t\t\t{\n"
    "\t\t\t\t\t// FIRE SHORT \u2014 stop 2 ATR above entry (tighter, entry-relative)\n"
    "\t\t\t\t\tdouble entry = close;\n"
    "\t\t\t\t\tdouble atrVal = _atr != null && CurrentBar >= 14 ? _atr[0] : _eorRange * 0.15;\n"
    "\t\t\t\t\tdouble stop = entry + 2.0 * atrVal;\n"
    "\t\t\t\t\tdouble risk = stop - entry;\n"
    "\t\t\t\t\tdouble target = entry - TargetMultiple * risk;\n"
    "\n"
    "\t\t\t\t\tif (risk > 10 && risk < 300)\n"
    "\t\t\t\t\t{\n"
    "\t\t\t\t\t\t_sweepLevel = _sweepLevelName;\n"
    "\t\t\t\t\t\tTrace(\"[SM] >>>> FIRING SHORT via RETEST FAILURE at bar \" + rthBar + \" <<<<\");\n"
    "\t\t\t\t\t\tTrace(\"[SM] Entry=\" + entry.ToString(\"F2\") + \" 50%Lvl=\" + fiftyPct.ToString(\"F2\")\n"
    "\t\t\t\t\t\t\t+ \" ATR=\" + atrVal.ToString(\"F1\") + \" stop=\" + stop.ToString(\"F2\")\n"
    "\t\t\t\t\t\t\t+ \" risk=\" + risk.ToString(\"F0\") + \" | OF: barDelta=\" + delta + \" cvdSinceRev=\" + cvdSinceReversal);\n"
    "\t\t\t\t\t\tFireSignal(\"SHORT\", entry, stop, target);\n"
    "\t\t\t\t\t}\n"
    "\t\t\t\t\telse\n"
    "\t\t\t\t\t{\n"
    "\t\t\t\t\t\t_lastSkipReason = \"SHORT: risk=\" + risk.ToString(\"F0\") + \" out of bounds [10-300]\";\n"
    "\t\t\t\t\t\tTrace(\"[SM] \" + _lastSkipReason);\n"
    "\t\t\t\t\t}\n"
    "\t\t\t\t}"
)

c2 = content.count(old2)
print(f"Change 2 occurrences: {c2}")
assert c2 == 1, f"Expected 1, got {c2}"
content = content.replace(old2, new2)
print("Change 2 applied.")

# ── Change 3 (LONG arm) ───────────────────────────────────────────────────────
old3 = (
    "if (retestReachedMid && inEntryZone && turningUp && ofConfirm)\n"
    "\t\t\t\t{\n"
    "\t\t\t\t\tdouble entry = close;\n"
    "\t\t\t\t\tdouble stop = _sweptRefPrice - _eorRange * StopBufferPct;\n"
    "\t\t\t\t\tdouble risk = entry - stop;\n"
    "\t\t\t\t\tdouble target = entry + TargetMultiple * risk;\n"
    "\n"
    "\t\t\t\t\tif (risk > 20 && risk < 300)\n"
    "\t\t\t\t\t{\n"
    "\t\t\t\t\t\t_sweepLevel = _sweepLevelName;\n"
    "\t\t\t\t\t\tTrace(\"[SM] >>>> FIRING LONG via RETEST FAILURE at bar \" + rthBar + \" <<<<\");\n"
    "\t\t\t\t\t\tTrace(\"[SM] Entry near mid=\" + mid.ToString(\"F2\") + \" | retestPeak=\" + _retestPeak.ToString(\"F2\")\n"
    "\t\t\t\t\t\t\t+ \" \u2192 turned up at \" + close.ToString(\"F2\")\n"
    "\t\t\t\t\t\t\t+ \" | OF: barDelta=\" + delta + \" cvdSinceRev=\" + cvdSinceReversal);\n"
    "\t\t\t\t\t\tFireSignal(\"LONG\", entry, stop, target);\n"
    "\t\t\t\t\t}\n"
    "\t\t\t\t\telse\n"
    "\t\t\t\t\t{\n"
    "\t\t\t\t\t\t_lastSkipReason = \"LONG: risk=\" + risk.ToString(\"F0\") + \" out of bounds [20-300]\";\n"
    "\t\t\t\t\t\tTrace(\"[SM] \" + _lastSkipReason);\n"
    "\t\t\t\t\t}\n"
    "\t\t\t\t}"
)

new3 = (
    "// 50% retest entry zone: centered on midpoint between _eorLow and _reversalExtreme\n"
    "\t\t\t\tdouble fiftyPct = _reversalExtreme - (_reversalExtreme - _eorLow) * 0.5;\n"
    "\t\t\t\tbool inFiftyPctZone = close >= fiftyPct - _eorRange * 0.15 && close <= fiftyPct + _eorRange * 0.15;\n"
    "\n"
    "\t\t\t\tif (retestReachedMid && inFiftyPctZone && turningUp && ofConfirm)\n"
    "\t\t\t\t{\n"
    "\t\t\t\t\t// FIRE LONG \u2014 stop 2 ATR below entry (tighter, entry-relative)\n"
    "\t\t\t\t\tdouble entry = close;\n"
    "\t\t\t\t\tdouble atrVal = _atr != null && CurrentBar >= 14 ? _atr[0] : _eorRange * 0.15;\n"
    "\t\t\t\t\tdouble stop = entry - 2.0 * atrVal;\n"
    "\t\t\t\t\tdouble risk = entry - stop;\n"
    "\t\t\t\t\tdouble target = entry + TargetMultiple * risk;\n"
    "\n"
    "\t\t\t\t\tif (risk > 10 && risk < 300)\n"
    "\t\t\t\t\t{\n"
    "\t\t\t\t\t\t_sweepLevel = _sweepLevelName;\n"
    "\t\t\t\t\t\tTrace(\"[SM] >>>> FIRING LONG via RETEST FAILURE at bar \" + rthBar + \" <<<<\");\n"
    "\t\t\t\t\t\tTrace(\"[SM] Entry=\" + entry.ToString(\"F2\") + \" 50%Lvl=\" + fiftyPct.ToString(\"F2\")\n"
    "\t\t\t\t\t\t\t+ \" ATR=\" + atrVal.ToString(\"F1\") + \" stop=\" + stop.ToString(\"F2\")\n"
    "\t\t\t\t\t\t\t+ \" risk=\" + risk.ToString(\"F0\") + \" | OF: barDelta=\" + delta + \" cvdSinceRev=\" + cvdSinceReversal);\n"
    "\t\t\t\t\t\tFireSignal(\"LONG\", entry, stop, target);\n"
    "\t\t\t\t\t}\n"
    "\t\t\t\t\telse\n"
    "\t\t\t\t\t{\n"
    "\t\t\t\t\t\t_lastSkipReason = \"LONG: risk=\" + risk.ToString(\"F0\") + \" out of bounds [10-300]\";\n"
    "\t\t\t\t\t\tTrace(\"[SM] \" + _lastSkipReason);\n"
    "\t\t\t\t\t}\n"
    "\t\t\t\t}"
)

c3 = content.count(old3)
print(f"Change 3 occurrences: {c3}")
assert c3 == 1, f"Expected 1, got {c3}"
content = content.replace(old3, new3)
print("Change 3 applied.")

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"File written. New length={len(content)} chars (delta={len(content)-orig_len})")
