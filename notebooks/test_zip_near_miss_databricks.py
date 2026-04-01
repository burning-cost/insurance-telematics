# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # ZIPNearMissModel — Databricks Test Runner
# MAGIC
# MAGIC Runs the full test suite for `zip_near_miss.py` on Databricks serverless compute.
# MAGIC Upload this notebook and the project source, then run all cells.

# COMMAND ----------
# MAGIC %pip install polars>=1.0 statsmodels>=0.14.5 scipy>=1.10 numpy>=2.0 hmmlearn>=0.3.3 scikit-learn>=1.3 pyarrow>=14.0 pandas>=2.0 pytest

# COMMAND ----------
import subprocess, sys, os

# Install the library from source (editable)
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", "/Workspace/insurance-telematics", "--quiet"],
    capture_output=True, text=True
)
print(result.stdout[-2000:] if result.stdout else "")
print(result.stderr[-2000:] if result.stderr else "")

# COMMAND ----------
# Run the ZIP near-miss tests
result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        "/Workspace/insurance-telematics/tests/test_zip_near_miss.py",
        "-v", "--tb=short", "--no-header",
    ],
    capture_output=True, text=True,
    cwd="/Workspace/insurance-telematics",
)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-1000:])
assert result.returncode == 0, f"Tests failed (exit code {result.returncode})"

# COMMAND ----------
# Run the full test suite
result_full = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        "/Workspace/insurance-telematics/tests/",
        "-v", "--tb=short", "--no-header",
    ],
    capture_output=True, text=True,
    cwd="/Workspace/insurance-telematics",
)
print(result_full.stdout)
if result_full.stderr:
    print("STDERR:", result_full.stderr[-1000:])
assert result_full.returncode == 0, f"Full test suite failed (exit code {result_full.returncode})"
print("\nAll tests passed.")
