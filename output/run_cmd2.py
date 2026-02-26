import subprocess, sys
result = subprocess.run(
    [r"E:\anaconda\python.exe",
     r"C:\Users\lehph\Documents\GitHub\BookMapOrderFlowStudies-2\scripts\run_backtest.py",
     "--strategies", "OR Acceptance",
     "--instrument", "NQ"],
    capture_output=True, text=True,
    cwd=r"C:\Users\lehph\Documents\GitHub\BookMapOrderFlowStudies-2"
)
out = result.stdout + result.stderr
with open(r"C:\Users\lehph\Documents\GitHub\BookMapOrderFlowStudies-2\output\cmd2_result.txt", "w") as f:
    f.write(out)
print("=== STDOUT ===")
print(result.stdout)
print("=== STDERR ===")
print(result.stderr)
