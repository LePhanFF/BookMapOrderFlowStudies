"""Run unit tests and write results to file for review."""
import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_balance_signal.py", "-v", "--tb=short"],
    capture_output=True, text=True, cwd=r"C:\Users\lehph\Documents\GitHub\BookMapOrderFlowStudies-2"
)

output = result.stdout + "\n" + result.stderr
with open(r"C:\Users\lehph\Documents\GitHub\BookMapOrderFlowStudies-2\test_results.txt", "w") as f:
    f.write(output)

print(output)
sys.exit(result.returncode)
