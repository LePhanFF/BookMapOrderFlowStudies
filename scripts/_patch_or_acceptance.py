#!/usr/bin/env python
"""
One-shot patch for RockitORAcceptanceSignal.cs
Changes:
  1. ResetSession() - add resets for _asiaHigh/_asiaLow/_pdh/_pdl/_biasScore
  2. LoadLondonLevels() - add Asia session and prior-day scans
  3. EOR completion block - add bias score computation + updated acceptance conditions
"""
import sys

path = r'C:\Users\lehph\Documents\NinjaTrader 8\bin\Custom\Indicators\RockitORAcceptanceSignal.cs'

with open(path, encoding='utf-8-sig') as f:
    content = f.read()

original_len = len(content)

# ─── Change 1: ResetSession() ─────────────────────────────────────────────────
old1 = (
    "\t\t_londonHigh      = 0; _londonLow = 0;\n"
    "\t\t_hasLondonLevels = false;"
)
new1 = (
    "\t\t_londonHigh = 0; _londonLow = 0;\n"
    "\t\t_asiaHigh   = 0; _asiaLow   = 0;\n"
    "\t\t_pdh        = 0; _pdl       = 0;\n"
    "\t\t_hasLondonLevels = false;\n"
    "\t\t_biasScore = 0;"
)
if old1 not in content:
    print("ERROR: Change 1 search string not found"); sys.exit(1)
content = content.replace(old1, new1, 1)
print("Change 1 applied: ResetSession() new field resets (lines ~463-464)")

# ─── Change 2: LoadLondonLevels() - append Asia + PDH/PDL after London block ──
old2 = (
    "\t\t\telse\n"
    "\t\t\t{\n"
    "\t\t\t\tTrace(\"London levels: not found (\" + scanned + \" bars scanned)\");\n"
    "\t\t\t}\n"
    "\t\t}"
)
new2 = (
    "\t\t\telse\n"
    "\t\t\t{\n"
    "\t\t\t\tTrace(\"London levels: not found (\" + scanned + \" bars scanned)\");\n"
    "\t\t\t}\n"
    "\n"
    "\t\t\t// Asia session: 20:00-00:00 prior evening ET\n"
    "\t\t\tDateTime asiaStart = today.AddDays(-1).AddHours(20);\n"
    "\t\t\tDateTime asiaEnd   = today.AddHours(0);\n"
    "\t\t\tdouble ah = 0, al = double.MaxValue;\n"
    "\t\t\tint ascanned = 0;\n"
    "\t\t\tfor (int i = 1; i <= Math.Min(CurrentBar, 700); i++)\n"
    "\t\t\t{\n"
    "\t\t\t\tDateTime t = Time[i];\n"
    "\t\t\t\tif (t < asiaStart) break;\n"
    "\t\t\t\tif (t >= asiaStart && t <= asiaEnd)\n"
    "\t\t\t\t{\n"
    "\t\t\t\t\tif (High[i] > ah) ah = High[i];\n"
    "\t\t\t\t\tif (Low[i]  < al) al = Low[i];\n"
    "\t\t\t\t\tascanned++;\n"
    "\t\t\t\t}\n"
    "\t\t\t}\n"
    "\t\t\tif (ascanned > 0 && ah > 0 && al < double.MaxValue)\n"
    "\t\t\t{\n"
    "\t\t\t\t_asiaHigh = ah; _asiaLow = al;\n"
    "\t\t\t\tTrace(\"Asia levels: H=\" + _asiaHigh.ToString(\"F2\") + \" L=\" + _asiaLow.ToString(\"F2\") + \" (\" + ascanned + \" bars)\");\n"
    "\t\t\t}\n"
    "\n"
    "\t\t\t// Prior RTH day: find PDH/PDL from yesterday 9:30-16:00\n"
    "\t\t\tDateTime pdStart = today.AddDays(-1).AddHours(9).AddMinutes(30);\n"
    "\t\t\tDateTime pdEnd   = today.AddDays(-1).AddHours(16);\n"
    "\t\t\tdouble ph = 0, pl = double.MaxValue;\n"
    "\t\t\tint pscanned = 0;\n"
    "\t\t\tfor (int i = 1; i <= Math.Min(CurrentBar, 800); i++)\n"
    "\t\t\t{\n"
    "\t\t\t\tDateTime t = Time[i];\n"
    "\t\t\t\tif (t < pdStart) break;\n"
    "\t\t\t\tif (t >= pdStart && t <= pdEnd)\n"
    "\t\t\t\t{\n"
    "\t\t\t\t\tif (High[i] > ph) ph = High[i];\n"
    "\t\t\t\t\tif (Low[i]  < pl) pl = Low[i];\n"
    "\t\t\t\t\tpscanned++;\n"
    "\t\t\t\t}\n"
    "\t\t\t}\n"
    "\t\t\tif (pscanned > 0 && ph > 0 && pl < double.MaxValue)\n"
    "\t\t\t{\n"
    "\t\t\t\t_pdh = ph; _pdl = pl;\n"
    "\t\t\t\tTrace(\"Prior day: H=\" + _pdh.ToString(\"F2\") + \" L=\" + _pdl.ToString(\"F2\") + \" (\" + pscanned + \" bars)\");\n"
    "\t\t\t}\n"
    "\t\t}"
)
if old2 not in content:
    print("ERROR: Change 2 search string not found"); sys.exit(1)
content = content.replace(old2, new2, 1)
print("Change 2 applied: LoadLondonLevels() Asia + PDH/PDL scan added (lines ~440-444)")

# ─── Change 3: EOR completion - bias score + updated acceptance conditions ────
old3 = (
    "\t\t\t\t\t// ── Check acceptance conditions ────────────────────────────────\n"
    "\t\t\t\t\t// SHORT: entire EOR traded below London Low\n"
    "\t\t\t\t\tif (_londonLow > 0 && _eorHigh < _londonLow)\n"
    "\t\t\t\t\t{\n"
    "\t\t\t\t\t\t_acceptanceShort  = true;\n"
    "\t\t\t\t\t\t_acceptanceLevel  = _londonLow;\n"
    "\t\t\t\t\t\t_fiftyPct         = _eorLow + (_londonLow - _eorLow) * 0.5;\n"
    "\t\t\t\t\t\tTrace(\"ACCEPTANCE SHORT: EOR_H=\" + _eorHigh.ToString(\"F2\")\n"
    "\t\t\t\t\t\t\t+ \" < LDN_L=\" + _londonLow.ToString(\"F2\")\n"
    "\t\t\t\t\t\t\t+ \" | 50%=\" + _fiftyPct.ToString(\"F2\"));\n"
    "\t\t\t\t\t}\n"
    "\t\t\t\t\t// LONG: entire EOR traded above London High\n"
    "\t\t\t\t\telse if (_londonHigh > 0 && _eorLow > _londonHigh)\n"
    "\t\t\t\t\t{\n"
    "\t\t\t\t\t\t_acceptanceLong   = true;\n"
    "\t\t\t\t\t\t_acceptanceLevel  = _londonHigh;\n"
    "\t\t\t\t\t\t_fiftyPct         = _eorHigh - (_eorHigh - _londonHigh) * 0.5;\n"
    "\t\t\t\t\t\tTrace(\"ACCEPTANCE LONG: EOR_L=\" + _eorLow.ToString(\"F2\")\n"
    "\t\t\t\t\t\t\t+ \" > LDN_H=\" + _londonHigh.ToString(\"F2\")\n"
    "\t\t\t\t\t\t\t+ \" | 50%=\" + _fiftyPct.ToString(\"F2\"));\n"
    "\t\t\t\t\t}\n"
    "\t\t\t\t\telse\n"
    "\t\t\t\t\t{"
)
new3 = (
    "\t\t\t\t\t// ── Directional bias score ─────────────────────────────────────────\n"
    "\t\t\t\t\t_biasScore = 0;\n"
    "\t\t\t\t\tif (_asiaLow    > 0 && _eorLow  < _asiaLow)    _biasScore += 1; // Asia Low swept\n"
    "\t\t\t\t\tif (_londonLow  > 0 && _eorLow  < _londonLow)  _biasScore += 1; // LDN Low swept\n"
    "\t\t\t\t\tif (_pdl        > 0 && _eorLow  > _pdl)         _biasScore += 1; // PDL held\n"
    "\t\t\t\t\tif (_asiaHigh   > 0 && _eorHigh > _asiaHigh)   _biasScore -= 1; // Asia High swept\n"
    "\t\t\t\t\tif (_londonHigh > 0 && _eorHigh > _londonHigh)  _biasScore -= 1; // LDN High swept\n"
    "\t\t\t\t\tif (_pdh        > 0 && _eorHigh < _pdh)         _biasScore -= 1; // PDH held\n"
    "\t\t\t\t\tTrace(\"BiasScore=\" + _biasScore\n"
    "\t\t\t\t\t\t+ \" (AsiaL=\" + _asiaLow.ToString(\"F0\") + \" AsiaH=\" + _asiaHigh.ToString(\"F0\")\n"
    "\t\t\t\t\t\t+ \" PDL=\" + _pdl.ToString(\"F0\") + \" PDH=\" + _pdh.ToString(\"F0\") + \")\");\n"
    "\n"
    "\t\t\t\t\t// ── Check acceptance conditions ─────────────────────────────────────\n"
    "\t\t\t\t\t// SHORT: entire EOR below London Low, bias not bullish\n"
    "\t\t\t\t\tif (_londonLow > 0 && _eorHigh < _londonLow && _biasScore <= 0)\n"
    "\t\t\t\t\t{\n"
    "\t\t\t\t\t\t_acceptanceShort  = true;\n"
    "\t\t\t\t\t\t_acceptanceLevel  = _londonLow;\n"
    "\t\t\t\t\t\t_fiftyPct         = _eorLow + (_londonLow - _eorLow) * 0.5;\n"
    "\t\t\t\t\t\tTrace(\"ACCEPTANCE SHORT: EOR_H=\" + _eorHigh.ToString(\"F2\")\n"
    "\t\t\t\t\t\t\t+ \" < LDN_L=\" + _londonLow.ToString(\"F2\")\n"
    "\t\t\t\t\t\t\t+ \" bias=\" + _biasScore + \" | 50%=\" + _fiftyPct.ToString(\"F2\"));\n"
    "\t\t\t\t\t}\n"
    "\t\t\t\t\t// LONG: entire EOR above London High, bias not bearish\n"
    "\t\t\t\t\telse if (_londonHigh > 0 && _eorLow > _londonHigh && _biasScore >= 0)\n"
    "\t\t\t\t\t{\n"
    "\t\t\t\t\t\t_acceptanceLong   = true;\n"
    "\t\t\t\t\t\t_acceptanceLevel  = _londonHigh;\n"
    "\t\t\t\t\t\t_fiftyPct         = _eorHigh - (_eorHigh - _londonHigh) * 0.5;\n"
    "\t\t\t\t\t\tTrace(\"ACCEPTANCE LONG: EOR_L=\" + _eorLow.ToString(\"F2\")\n"
    "\t\t\t\t\t\t\t+ \" > LDN_H=\" + _londonHigh.ToString(\"F2\")\n"
    "\t\t\t\t\t\t\t+ \" bias=\" + _biasScore + \" | 50%=\" + _fiftyPct.ToString(\"F2\"));\n"
    "\t\t\t\t\t}\n"
    "\t\t\t\t\telse\n"
    "\t\t\t\t\t{"
)
if old3 not in content:
    print("ERROR: Change 3 search string not found"); sys.exit(1)
content = content.replace(old3, new3, 1)
print("Change 3 applied: EOR bias score + updated acceptance conditions (lines ~204-225)")

# Write output
with open(path, encoding='utf-8', mode='w') as f:
    f.write(content)

print()
print("File written successfully.")
print("Chars before: %d  |  Chars after: %d  |  Delta: +%d" % (
    original_len, len(content), len(content) - original_len))
