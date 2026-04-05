# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # insurance-telematics: New Integration and Edge-Case Tests (April 2026)
# MAGIC
# MAGIC Runs the two new test files added in the April 2026 coverage improvement:
# MAGIC
# MAGIC - `test_integration.py` — end-to-end pipeline tests covering all stage seams
# MAGIC - `test_edge_cases.py` — edge cases: empty data, single-trip drivers, NaN sensors, etc.

# COMMAND ----------
# MAGIC %pip install polars>=1.0 statsmodels>=0.14.5 scipy>=1.10 numpy>=2.0 hmmlearn>=0.3.3 scikit-learn>=1.3 pyarrow>=14.0 pandas>=2.0 pytest

# COMMAND ----------
import subprocess, sys, os

result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", "/Workspace/insurance-telematics", "--quiet"],
    capture_output=True, text=True
)
print(result.stdout[-2000:] if result.stdout else "")
if result.stderr:
    print("STDERR:", result.stderr[-1000:])

# COMMAND ----------
# Run integration tests
result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        "/Workspace/insurance-telematics/tests/test_integration.py",
        "-v", "--tb=short", "--no-header",
    ],
    capture_output=True, text=True,
    cwd="/Workspace/insurance-telematics",
)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-1000:])
assert result.returncode == 0, f"Integration tests failed (exit code {result.returncode})"

# COMMAND ----------
# Run edge-case tests
result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        "/Workspace/insurance-telematics/tests/test_edge_cases.py",
        "-v", "--tb=short", "--no-header",
    ],
    capture_output=True, text=True,
    cwd="/Workspace/insurance-telematics",
)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-1000:])
assert result.returncode == 0, f"Edge-case tests failed (exit code {result.returncode})"

# COMMAND ----------
# Run the full test suite to check for regressions
result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        "/Workspace/insurance-telematics/tests/",
        "-v", "--tb=short", "--no-header",
        "-q",
    ],
    capture_output=True, text=True,
    cwd="/Workspace/insurance-telematics",
)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-1000:])
print(f"\nFull suite exit code: {result.returncode}")
assert result.returncode == 0, f"Full suite failed (exit code {result.returncode})"
