#!/usr/bin/env bash
# DEF-159: wait for run1 (PID $1) to exit, then run2 and run3 sequentially.
# Sequential is mandatory: reproduce_baseline.sh re-points the ./logs symlink,
# so two concurrent runs would cross-contaminate AgentDojo's per-task cache.
# Usage: ./run_queue.sh <run1-pid>      (progress: tail run_queue.log, run2.log, run3.log)
set -u
cd "$(dirname "$0")"
while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "== run1 exited at $(date); starting run2" >> run_queue.log
MODEL=openai:gpt-4o-2024-08-06 ./reproduce_baseline.sh run2 > run2.log 2>&1
echo "== run2 exited rc=$? at $(date); starting run3" >> run_queue.log
MODEL=openai:gpt-4o-2024-08-06 ./reproduce_baseline.sh run3 > run3.log 2>&1
echo "== run3 exited rc=$? at $(date)" >> run_queue.log
