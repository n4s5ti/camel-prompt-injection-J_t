#!/usr/bin/env bash
# DEF-159: run the three baseline runs sequentially (the ./logs symlink forbids concurrency).
#
#   ./run_queue.sh <pid>          wait for PID to exit, then run2, run3            (original mode)
#   ./run_queue.sh --wait-credits probe OpenAI every 10 min until the account has credits, then
#                                 run1 (resumes from AgentDojo's per-task cache), run2, run3.
#
# A run that dies on exhausted credits is resumed automatically once credits return.
# Progress: run_queue.log, run1.log / run2.log / run3.log. Kill with: pkill -f run_queue.sh
set -u
cd "$(dirname "$0")"
MODEL="${MODEL:-openai:gpt-4o-2024-08-06}"

log() { echo "== $* at $(date)" >> run_queue.log; }

have_credits() {
  uv run --env-file .env python - <<'EOF' >/dev/null 2>&1
import os, sys, openai
c = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
try:
    c.chat.completions.create(model="gpt-4o-2024-08-06", max_tokens=1, messages=[{"role": "user", "content": "hi"}])
except openai.RateLimitError as e:
    sys.exit(2 if "insufficient_quota" in str(e) or "credit" in str(e) else 0)
sys.exit(0)
EOF
}

run_one() {  # run_one TAG  — resumes itself after a credit outage
  MODEL="$MODEL" ./reproduce_baseline.sh "$1" > "$1.log" 2>&1
  local rc=$?
  log "$1 exited rc=$rc"
  if [ $rc -ne 0 ] && grep -q "insufficient_quota\|credit_balance_exhausted" "$1.log"; then
    log "$1 stopped on exhausted credits; waiting for credits before continuing"
    until have_credits; do sleep 600; done
    log "credits available; resuming $1"
    run_one "$1"
  fi
}

if [ "${1:-}" = "--wait-credits" ]; then
  log "waiting for OpenAI credits (probe every 10 min)"
  until have_credits; do sleep 600; done
  log "credits available; starting run1 (resume)"
  run_one run1
else
  while kill -0 "$1" 2>/dev/null; do sleep 60; done
  log "PID $1 exited; starting run2"
fi
run_one run2
run_one run3
log "queue finished"
