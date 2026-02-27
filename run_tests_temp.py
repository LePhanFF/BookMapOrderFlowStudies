"""Temporary test runner - writes output to a file."""
import subprocess
import sys

result = subprocess.run(
    [r"E:\anaconda\python.exe", "-m", "pytest",
     r"tests/test_balance_signal.py", "-v"],
    capture_output=True, text=True,
    cwd=r"C:\Users\lehph\Documents\GitHub\BookMapOrderFlowStudies-2"
)

output_path = r"C:\Users\lehph\Documents\GitHub\BookMapOrderFlowStudies-2\test_output.txt"
with open(output_path, "w") as f:
    f.write("=== STDOUT ===\n")
    f.write(result.stdout)
    f.write("\n=== STDERR ===\n")
    f.write(result.stderr)
    f.write(f"\n=== RETURN CODE: {result.returncode} ===\n")

print(f"Output written to {output_path}")
