#!/usr/bin/env bash
# DEF-159: establish the AgentDojo baseline for P-DEF-12 (paper reference: 84% undefended / 77% camel+secpol,
# on models we are NOT using -- see MVP-NOTES.md §5/§9). Model fixed to gpt-oss-120b served by Cerebras:
# OpenAI-compatible, returns logprobs + accepts seed. Do not change the model after this runs.
# .env needs:  OPENAI_API_KEY=<cerebras key>   OPENAI_BASE_URL=https://api.cerebras.ai/v1
# First run `uv run --env-file .env check_model_name.py openai:gpt-oss-120b` once.
#
# Usage:  ./reproduce_baseline.sh [RUN_TAG]
# Each invocation is one "run" for variance; do three. Logs land in ./logs/<pipeline>/<suite>/...
# AgentDojo caches results per task in logs/ (force_rerun=False), so a repeat run with the same
# model reuses cached tasks. To get an independent run, use a separate logs dir per RUN_TAG.

set -euo pipefail
MODEL="${MODEL:-openai:gpt-4o-2024-08-06}"
RUN_TAG="${1:-run1}"
SUITES=(workspace slack travel banking)
# cyclopts wants the flag repeated per value, not "--suites a b c"
SUITE_ARGS=(); for s in "${SUITES[@]}"; do SUITE_ARGS+=(--suites "$s"); done

export LOGDIR="logs_${RUN_TAG}"
mkdir -p "$LOGDIR"
# main.py hardcodes Path("./logs"); point it at this run's dir.
rm -f logs && ln -s "$LOGDIR" logs

echo "== [$RUN_TAG] model=$MODEL  logs -> $LOGDIR"
echo

echo "== 1/4 undefended, no attack (utility)"
uv run --env-file .env main.py "$MODEL" --use-original "${SUITE_ARGS[@]}"

echo "== 2/4 undefended, under attack (utility + security)"
uv run --env-file .env main.py "$MODEL" --use-original --run-attack "${SUITE_ARGS[@]}"

echo "== 3/4 camel (no policies), no attack + under attack  [required before replay]"
uv run --env-file .env main.py "$MODEL" "${SUITE_ARGS[@]}"
uv run --env-file .env main.py "$MODEL" --run-attack "${SUITE_ARGS[@]}"

echo "== 4/4 camel + security policies (post-hoc replay of step 3)"
uv run --env-file .env main.py "$MODEL" --replay-with-policies "${SUITE_ARGS[@]}"
uv run --env-file .env main.py "$MODEL" --replay-with-policies --run-attack "${SUITE_ARGS[@]}"

echo
echo "== done [$RUN_TAG]. Record per-suite utility/security from the output above into DEF-159."
echo "   Paper reference (different models): undefended ~0.84, camel+secpol ~0.77, camel+secpol ASR ~0.0."
echo "   Our pre-registered checks: undefended utility > camel+secpol utility; camel+secpol ASR ~ 0."
