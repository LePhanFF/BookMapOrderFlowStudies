#!/usr/bin/env python3
"""Fix two bugs in RockitORAcceptanceSignal.cs"""

path = 'C:/Users/lehph/Documents/NinjaTrader 8/bin/Custom/Indicators/RockitORAcceptanceSignal.cs'

with open(path, 'r', encoding='utf-8-sig') as f:
    content = f.read()

# ── Fix 1: Replace session detection + IB tracking + CVD + London loading block ──
# We'll search for the distinct non-unicode strings to find the block boundaries.
# The block starts at: if (_sessionIterator.IsNewSession(
# and ends after: LoadLondonLevels();  (the first occurrence)

import re

old_block = None
new_block = None

# Locate the block by searching for the key lines we know exist
marker_start = '\t\t\t// ─── Session boundary ─────────────────────────────────────────────────'
# Since unicode dashes may cause issues, let's just find by ASCII parts
# Find "IsNewSession" line
idx_new_session = content.find('_sessionIterator.IsNewSession(Time[0], false)')
if idx_new_session == -1:
    print("ERROR: Could not find _sessionIterator.IsNewSession")
else:
    print(f"Found IsNewSession at index {idx_new_session}")

# Find the line start of the comment before it
# Go backward to find the line with "Session boundary"
line_before = content.rfind('\n', 0, idx_new_session)
line_before2 = content.rfind('\n', 0, line_before)
comment_start = line_before2 + 1
print(f"Block starts at index {comment_start}")
print(f"Preview: {repr(content[comment_start:comment_start+80])}")

# Find the end: the first LoadLondonLevels(); line after the block start
idx_load_london = content.find('LoadLondonLevels();', idx_new_session)
if idx_load_london == -1:
    print("ERROR: Could not find LoadLondonLevels()")
else:
    print(f"Found LoadLondonLevels() at index {idx_load_london}")

# Find the end of that line (the newline after LoadLondonLevels();)
idx_end_of_line = content.find('\n', idx_load_london)
block_end = idx_end_of_line + 1
print(f"Block ends at index {block_end}")
print(f"Old block (repr):\n{repr(content[comment_start:block_end])}")
