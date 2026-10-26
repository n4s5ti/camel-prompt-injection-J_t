#!/usr/bin/env bash
# DEF-167: arm B ("camel-live") — policies enforced at generation time, denials fed back to the P-LLM.
# Same log-dir convention as reproduce_baseline.sh so arm B lands next to arms A/undefended of the same run:
#   logs_runN/<model>+camel+live/<suite>/...  and  logs_runN/rejection_log/<model>+camel+live.jsonl
# Must NOT run concurrently with reproduce_baseline.sh (shared ./logs symlink).
#
# Usage:  ./run_arm_b.sh [RUN_TAG]      e.g. ./run_arm_b.sh run1

set -euo pipefail
MODEL="${MODEL:-openai:gpt-4o-2024-08-06}"
RUN_TAG="${1:-run1}"
SUITES=(workspace slack travel banking)
SUITE_ARGS=(); for s in "${SUITES[@]}"; do SUITE_ARGS+=(--suites "$s"); done

export LOGDIR="logs_${RUN_TAG}"
mkdir -p "$LOGDIR"
rm -f logs && ln -s "$LOGDIR" logs

echo "== [$RUN_TAG] arm B camel-live  model=$MODEL  logs -> $LOGDIR"
echo "== 1/2 camel-live, no attack (utility)"
uv run --env-file .env main.py "$MODEL" --live-policies "${SUITE_ARGS[@]}"
echo "== 2/2 camel-live, under attack (utility + security)"
uv run --env-file .env main.py "$MODEL" --live-policies --run-attack "${SUITE_ARGS[@]}"
echo "== done [$RUN_TAG] arm B. Record per-suite utility/security into DEF-167 next to arm A's."
echo "   Verify the log: uv run python -c \"from camel.rejection_log import verify_chain as v; print(v('$LOGDIR/rejection_log/${MODEL#*:}+camel+live.jsonl'))\""
