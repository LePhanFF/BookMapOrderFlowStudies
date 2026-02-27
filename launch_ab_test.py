"""
Launcher: runs test_or_rev_ab.py inline (not subprocess) and saves output.
"""
import sys
import io
import contextlib
from pathlib import Path

project_root = Path("C:/Users/lehph/Documents/GitHub/BookMapOrderFlowStudies-2")
sys.path.insert(0, str(project_root))
out_file = project_root / "output" / "ab_run_output.txt"

# Import and run the A/B test main()
import importlib.util
spec = importlib.util.spec_from_file_location(
    "test_or_rev_ab",
    str(project_root / "scripts" / "test_or_rev_ab.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    with contextlib.redirect_stderr(buf):
        try:
            mod.main()
        except Exception as e:
            import traceback
            print(f"\nERROR: {e}")
            traceback.print_exc()

result = buf.getvalue()
print(result)
out_file.parent.mkdir(parents=True, exist_ok=True)
out_file.write_text(result, encoding="utf-8")
print(f"\n[Saved to {out_file}]", file=sys.__stdout__)
