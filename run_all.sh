#!/usr/bin/env bash
# Runs every (framework, task_category) combination. Resumable: skip if result exists.
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"

for fw in crewai langgraph metagpt autogen; do
  for task in gsm8k tool_use collab; do
    echo "=== $fw / $task ==="
    python -m benchmark.harness --framework "$fw" --task "$task" || echo "[warn] $fw/$task partial fail"
  done
done

python benchmark/analyze.py
