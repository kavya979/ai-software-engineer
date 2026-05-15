"""
Test Runner — executes every generated Python file in a sandbox subprocess.
Captures stdout/stderr and flags any runtime errors.
"""

import os
import subprocess
import tempfile
from state import AgentState


def test_runner_node(state: AgentState) -> AgentState:
    print("\n🧪 [Test Runner] Running generated files...")

    results: dict[str, str] = {}
    all_passed = True

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Write all files so inter-file imports work
        for filename, code in state["code_files"].items():
            fpath = os.path.join(tmp_dir, filename)
            with open(fpath, "w") as f:
                f.write(code)

        # Execute each file individually
        for filename in state["code_files"]:
            fpath = os.path.join(tmp_dir, filename)
            print(f"   Running {filename}...")

            proc = subprocess.run(
                ["python", fpath],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=tmp_dir,
            )

            output = proc.stdout + proc.stderr
            results[filename] = output.strip() or "(no output)"

            if proc.returncode != 0 or "Error" in proc.stderr or "Traceback" in proc.stderr:
                all_passed = False
                print(f"   ❌ {filename} failed:")
                for line in proc.stderr.splitlines()[:5]:
                    print(f"      {line}")
            else:
                print(f"   ✅ {filename} OK")

    return {
        **state,
        "test_results": results,
        "tests_passed": all_passed,
    }
