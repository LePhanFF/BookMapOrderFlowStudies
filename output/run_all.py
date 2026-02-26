"""
Run both backtest commands and save results to files.
Execute this with: E:\anaconda\python.exe output\run_all.py
"""
import subprocess
import os
import sys

BASE = r"C:\Users\lehph\Documents\GitHub\BookMapOrderFlowStudies-2"
PYTHON = r"E:\anaconda\python.exe"
SCRIPT = os.path.join(BASE, "scripts", "run_backtest.py")
OUT = os.path.join(BASE, "output")

def run_cmd(strategies, instrument, outfile):
    cmd = [PYTHON, SCRIPT, "--strategies", strategies, "--instrument", instrument]
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE)
    combined = result.stdout + ("\n--- STDERR ---\n" + result.stderr if result.stderr.strip() else "")
    with open(os.path.join(OUT, outfile), "w", encoding="utf-8") as f:
        f.write(combined)
    print(f"Saved to {outfile}")
    print(combined)
    return result.returncode

# Command 1
rc1 = run_cmd("Opening Range Rev", "NQ", "cmd1_result.txt")
print(f"\nCommand 1 exit code: {rc1}")

# Command 2
rc2 = run_cmd("OR Acceptance", "NQ", "cmd2_result.txt")
print(f"\nCommand 2 exit code: {rc2}")

if rc2 != 0:
    print("\nCommand 2 failed -- trying fallback 'Acceptance'")
    rc2b = run_cmd("Acceptance", "NQ", "cmd2_fallback_result.txt")
    print(f"Fallback exit code: {rc2b}")

print("\n=== ALL DONE ===")
