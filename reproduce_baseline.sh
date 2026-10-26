#!/usr/bin/env bash
# DEF-159: reproduce the CaMeL AgentDojo baseline (paper: 84% undefended / 77% camel+secpol).
# Requires OPENAI_API_KEY in .env (or exported). Model fixed to gpt-4o-2024-08-06 because
# it exposes logprobs (see MVP-NOTES.md §5). Do not change the model after this runs.
#
# Usage:  ./reproduce_baseline.sh [RUN_TAG]
# Each invocation is one "run" for variance; do three. Logs land in ./logs/<pipeline>/<suite>/...
# AgentDojo caches results per task in logs/ (force_rerun=False), so a repeat run with the same
# model reuses cached tasks. To get an independent run, use a separate logs dir per RUN_TAG.

set -euo pipefail
MODEL="${MODEL:-openai:gpt-4o-2024-08-06}"
RUN_TAG="${1:-run1}"
SUITES=(workspace slack travel banking)

export LOGDIR="logs_${RUN_TAG}"
mkdir -p "$LOGDIR"
# main.py hardcodes Path("./logs"); point it at this run's dir.
rm -f logs && ln -s "$LOGDIR" logs

echo "== [$RUN_TAG] model=$MODEL  logs -> $LOGDIR"
echo

echo "== 1/4 undefended, no attack (utility)"
uv run --env-file .env main.py "$MODEL" --use-original --suites "${SUITES[@]}"

echo "== 2/4 undefended, under attack (utility + security)"
uv run --env-file .env main.py "$MODEL" --use-original --run-attack --suites "${SUITES[@]}"

echo "== 3/4 camel (no policies), no attack + under attack  [required before replay]"
uv run --env-file .env main.py "$MODEL" --suites "${SUITES[@]}"
uv run --env-file .env main.py "$MODEL" --run-attack --suites "${SUITES[@]}"

echo "== 4/4 camel + security policies (post-hoc replay of step 3)"
uv run --env-file .env main.py "$MODEL" --replay-with-policies --suites "${SUITES[@]}"
uv run --env-file .env main.py "$MODEL" --replay-with-policies --run-attack --suites "${SUITES[@]}"

echo
echo "== done [$RUN_TAG]. Record per-suite utility/security from the output above into DEF-159."
echo "   Paper targets (overall): undefended utility ~0.84, camel+secpol utility ~0.77, camel+secpol security ~0.0 ASR."
