"""
Submit the new integration and edge-case tests to Databricks serverless compute.
Uses jobs/create + run_now (runs/submit is currently disabled in this workspace).
"""

import base64
import json
import os
import pathlib
import sys
import time
import uuid

# Load creds
env_path = pathlib.Path.home() / ".config" / "burning-cost" / "databricks.env"
for line in env_path.read_text().splitlines():
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()

RUN_ID = uuid.uuid4().hex[:8]
WORKSPACE_DIR = "/Workspace/Shared/insurance-telematics-new-tests"
NOTEBOOK_PATH = f"{WORKSPACE_DIR}/run_tests_{RUN_ID}"
BASE = pathlib.Path("/home/ralph/repos/insurance-telematics")


def upload_file(local_path: pathlib.Path, workspace_path: str) -> None:
    content = local_path.read_bytes()
    encoded = base64.b64encode(content).decode()
    parent = "/".join(workspace_path.split("/")[:-1])
    try:
        w.workspace.mkdirs(path=parent)
    except Exception:
        pass
    w.workspace.import_(
        path=workspace_path,
        content=encoded,
        overwrite=True,
        format=ImportFormat.AUTO,
    )


def upload_tree(local_dir: pathlib.Path, workspace_base: str, extensions: set) -> int:
    count = 0
    for f in sorted(local_dir.rglob("*")):
        if f.is_file() and f.suffix in extensions and "__pycache__" not in str(f):
            rel = f.relative_to(local_dir.parent)
            ws_path = f"{workspace_base}/{rel}"
            upload_file(f, ws_path)
            count += 1
    return count


print("Uploading source and tests...")
n = upload_tree(BASE / "src", WORKSPACE_DIR, {".py"})
n += upload_tree(BASE / "tests", WORKSPACE_DIR, {".py"})
upload_file(BASE / "pyproject.toml", f"{WORKSPACE_DIR}/pyproject.toml")
print(f"  Uploaded {n} files")

# Build notebook that installs deps, installs package, runs tests
NOTEBOOK_CONTENT = f"""\
# Databricks notebook source
# MAGIC %pip install polars>=1.0 statsmodels>=0.14.5 scipy>=1.10 numpy>=2.0 hmmlearn>=0.3.3 scikit-learn>=1.3 pyarrow>=14.0 pandas>=2.0 pytest --quiet

# COMMAND ----------

import subprocess, sys, os, shutil, tempfile

WORKSPACE_SRC = "{WORKSPACE_DIR}"

LOCAL_DIR = tempfile.mkdtemp(prefix="ins_tel_")
print(f"Working directory: {{LOCAL_DIR}}")


def copy_dir(src, dst):
    os.makedirs(dst, exist_ok=True)
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        rel = os.path.relpath(root, src)
        dest_dir = os.path.join(dst, rel)
        os.makedirs(dest_dir, exist_ok=True)
        for fname in files:
            if not fname.endswith('.pyc'):
                shutil.copy2(os.path.join(root, fname), os.path.join(dest_dir, fname))


copy_dir(WORKSPACE_SRC + "/src", LOCAL_DIR + "/src")
copy_dir(WORKSPACE_SRC + "/tests", LOCAL_DIR + "/tests")
shutil.copy2(WORKSPACE_SRC + "/pyproject.toml", LOCAL_DIR + "/pyproject.toml")
print(f"Copied project to {{LOCAL_DIR}}")

r_install = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", LOCAL_DIR, "--quiet"],
    capture_output=True, text=True
)
if r_install.returncode != 0:
    print("Install error:", r_install.stderr[:2000])
else:
    print("insurance-telematics installed successfully")

# COMMAND ----------

print("=== Running integration tests ===")
r_int = subprocess.run(
    [sys.executable, "-m", "pytest",
     LOCAL_DIR + "/tests/test_integration.py",
     "-v", "--tb=short", "--no-header", "-p", "no:cacheprovider"],
    capture_output=True, text=True, cwd=LOCAL_DIR
)
print(r_int.stdout[-15000:])
if r_int.stderr:
    print("STDERR:", r_int.stderr[-500:])

# COMMAND ----------

print("=== Running edge-case tests ===")
r_edge = subprocess.run(
    [sys.executable, "-m", "pytest",
     LOCAL_DIR + "/tests/test_edge_cases.py",
     "-v", "--tb=short", "--no-header", "-p", "no:cacheprovider"],
    capture_output=True, text=True, cwd=LOCAL_DIR
)
print(r_edge.stdout[-15000:])
if r_edge.stderr:
    print("STDERR:", r_edge.stderr[-500:])

# COMMAND ----------

print("=== Running full test suite ===")
r_all = subprocess.run(
    [sys.executable, "-m", "pytest",
     LOCAL_DIR + "/tests",
     "-v", "--tb=short", "--no-header", "-p", "no:cacheprovider", "-q"],
    capture_output=True, text=True, cwd=LOCAL_DIR
)
print(r_all.stdout[-15000:])
if r_all.stderr:
    print("STDERR:", r_all.stderr[-500:])

# COMMAND ----------

total_pass = r_int.returncode == 0 and r_edge.returncode == 0 and r_all.returncode == 0
if total_pass:
    msg = "ALL TESTS PASSED"
    print(f"\\n=== {{msg}} ===")
    try:
        dbutils.notebook.exit(msg)
    except NameError:
        pass
else:
    codes = f"integration={{r_int.returncode}} edge={{r_edge.returncode}} full={{r_all.returncode}}"
    msg = f"TESTS FAILED ({{codes}})"
    print(f"\\n=== {{msg}} ===")
    try:
        dbutils.notebook.exit(msg)
    except NameError:
        pass
"""

# Upload test notebook
print(f"Uploading notebook to {NOTEBOOK_PATH} ...")
nb_bytes = NOTEBOOK_CONTENT.encode("utf-8")
nb_b64 = base64.b64encode(nb_bytes).decode()
w.workspace.import_(
    path=NOTEBOOK_PATH,
    content=nb_b64,
    overwrite=True,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
)
print(f"Notebook uploaded to {NOTEBOOK_PATH}")

# Create job (jobs/create works when runs/submit is disabled)
print("Creating Databricks job...")
created_job = w.jobs.create(
    name=f"telematics-new-tests-{RUN_ID}",
    tasks=[
        jobs.Task(
            task_key="pytest",
            notebook_task=jobs.NotebookTask(
                notebook_path=NOTEBOOK_PATH,
            ),
            environment_key="default",
        )
    ],
    environments=[
        jobs.JobEnvironment(
            environment_key="default",
            spec=jobs.compute.Environment(
                client="2",
                dependencies=[
                    "polars>=1.0",
                    "statsmodels>=0.14.5",
                    "scipy>=1.10",
                    "numpy>=2.0",
                    "hmmlearn>=0.3.3",
                    "scikit-learn>=1.3",
                    "pyarrow>=14.0",
                    "pandas>=2.0",
                    "pytest",
                ],
            ),
        )
    ],
)
job_id = created_job.job_id
print(f"Created job ID: {job_id}")

# Run now
run_response = w.jobs.run_now(job_id=job_id)
run_id = run_response.run_id
print(f"Started run ID: {run_id}")

# Poll
while True:
    state = w.jobs.get_run(run_id=run_id)
    life = str(state.state.life_cycle_state)
    result_state = str(state.state.result_state)
    print(f"  [{time.strftime('%H:%M:%S')}] {life} / {result_state}      ", end="\r")
    if any(x in life for x in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR")):
        print()
        break
    time.sleep(20)

# Fetch output
for task in (state.tasks or []):
    try:
        output = w.jobs.get_run_output(run_id=task.run_id)
        print("\n" + "=" * 60)
        print("TEST OUTPUT:")
        print("=" * 60)
        if output.notebook_output and output.notebook_output.result:
            print(output.notebook_output.result)
        elif output.error:
            print("ERROR:", output.error)
            if output.error_trace:
                print("TRACE:", output.error_trace[:5000])
        else:
            print("(no output captured in notebook_output)")
    except Exception as e:
        print(f"Could not fetch output for task {task.task_key}: {e}")

# Clean up job
try:
    w.jobs.delete(job_id=job_id)
    print(f"Deleted job {job_id}")
except Exception:
    pass

final = str(state.state.result_state)
print(f"\nFinal result: {final}")
sys.exit(0 if "SUCCESS" in final else 1)
