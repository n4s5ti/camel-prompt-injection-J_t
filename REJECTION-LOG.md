# Rejection log — schema and rules (P-DEF-12 / DEF-160)

One JSONL record per **tool-call decision** at CaMeL's policy boundary. This is the dataset every later signal
(J_t, delta, λ, W_c) is computed from. Nothing in a record comes from the model's narration.

- Writer: `src/camel/rejection_log.py::RejectionLogger.record` — the only write path (test-enforced).
- Hook: `src/camel/interpreter/interpreter.py::_eval_call`, immediately after `check_policy`, *before* the
  denial is raised. A rejected call is recorded whether or not anything recovers from it.
- File: `<logdir>/rejection_log/<pipeline_name>.jsonl`, e.g. `logs_run1/rejection_log/gpt-4o-2024-08-06+camel+secpol.jsonl`.
  One file per (run, arm). `run_id` defaults to the resolved logdir name; override with `CAMEL_RUN_ID`.
- Backfill: after each suite, `main.py` stamps `task_outcome` onto that suite's records from AgentDojo's
  `utility_results`.
- Verify: `python -c "from camel.rejection_log import verify_chain; print(verify_chain('<file>'))"`.

## What is and is not logged

| Logged | Not logged |
| --- | --- |
| AgentDojo tool calls (`send_email`, `list_files`, …) | Builtins (`print`, `len`, `str.upper`, …) |
| `query_ai_assistant` (always accepted; it is a no-side-effect tool) | Interpreter-defined functions and classes |
| Both arms: live `PrivilegedLLM` and the post-hoc `PrivilegedLLMReplayer`; arm B `+camel+live` (DEF-167) | Tasks AgentDojo serves from its result cache (`force_rerun=False` → pipeline never runs) |

## Arm B: what the planner sees on a denial (DEF-167)

With `--live-policies`, `EvalArgs.denial_mode = ERROR`: the interpreter returns a *trusted* `CaMeLException`
wrapping `SecurityPolicyDeniedError(sanitized_denial_message(...))` instead of raising. The text is built from
`{tool, policy, argument, code}` only — e.g. `Tool 'send_money' was denied by security policy 'send_money' on
argument 'recipient': untrusted_source.` — never from `Denied.reason`, which interpolates raw values. The
exception carries no dependencies, so `format_camel_exception` shows it rather than redacting it, and the
existing retry loop (`PrivilegedLLM.query`, `max_attempts=10`) handles it like any interpretation error:
namespace preserved up to the denied statement, denied call never executed. `tests/test_camel_live.py` holds
the canary tests for both the dependency-level and the policy-level denial paths.

Policy subjects are exactly the callables that `_eval_call` can deny; the same predicate decides logging.
The undefended (`--use-original`) pipeline has no interpreter and produces no log.

## Record

Keys appear in this order (`rejection_log.required_fields()`). "Null" marks keys that may legitimately be `null`;
everything else is non-null in every record (test-enforced, smoke-verified on both arms).

| Key | Type | Null | Meaning |
| --- | --- | --- | --- |
| `schema` | int | | `1` |
| `run_id` | str | | `CAMEL_RUN_ID` or logdir name (`logs_run1`) |
| `seed` | int | yes | `CAMEL_SEED`; OpenAI accepts no seed → null. "3 runs", not "3 seeds". |
| `arm` | str | | pipeline name: `…+camel` (live, no policies), `…+camel+secpol` (replay), `…+camel+live` (DEF-167) |
| `suite` | str | | `workspace` / `slack` / `travel` / `banking` |
| `task_id` | str | | user task id |
| `injection_task_id` | str | yes | null when not under attack |
| `epoch` | int | | `CAMEL_EPOCH`, default 0 (DEF-165) |
| `task_run_n` | int | | 1-based pipeline invocation for this task. AgentDojo re-invokes the whole pipeline up to 3× when the final model output is `None` (a task that ended in a traceback does this); `attempt_n` restarts each time. Order attempts by (`task_run_n`, `attempt_n`, `step_n`). |
| `attempt_n` | int | | 1-based index of the generated program within the invocation. Live: each P-LLM retry. Replay: each saved turn. |
| `step_n` | int | | 1-based index of the decision within the task (continues across attempts) |
| `tool` | str | | tool name |
| `args_canonical` | object | | canonicalized arguments (below) |
| `args_hash` | str | | sha256 of `{"tool", "args": args_canonical}` canonical JSON |
| `args_hash_seen_before` | bool | | same `args_hash` already recorded earlier in this task (any attempt). Feeds λ "recursive". |
| `decision` | str | | `accepted` / `rejected` / `halted_for_confirmation` (reserved; CaMeL never halts today) |
| `policy_failed` | str | yes | policy pattern that denied (`send_email`, `engine.dependencies`, `engine.default`, `base`); null when accepted |
| `field_failed` | str | yes | argument that failed; null when accepted or engine-level |
| `reason` | str | yes | `DenialCode` value (`untrusted_source`, `readers_cannot_read`, `not_public`, `file_not_shared`, `depends_on_private`, `no_policy_matched`); null when accepted. **Never the free-text reason.** |
| `provenance_summary` | object | | per argument: `{"sources": [...], "readers": "public" \| [...]}` |
| `n_dependencies` | int | | control-flow dependencies at the call (`dependencies` in `_eval_call`) |
| `dependencies_public` | bool | | all dependencies have public readers |
| `conf_t` | float | yes | `exp(mean token logprob)` ∈ (0, 1] over the attempt's code block; null for replay and for providers without logprobs |
| `conf_t_scope` | str | yes | `code_block` or `content` (no fence found) |
| `conf_t_n_tokens` | int | yes | tokens scored |
| `conf_t_n_excluded` | int | yes | tokens dropped from the mean because OpenAI returned the `-9999.0` "no logprob" sentinel. Seen on long retry contexts; one such token would otherwise collapse `conf_t` to ~1e-30. |
| `code_hash` | str | yes | sha256 of the attempt's raw P-LLM output |
| `delta_to_goal`, `lambda`, `looked_good`, `failed_t`, `J_t`, `verifier_miss` | | yes | null until DEF-161/163/164 |
| `task_outcome` | str | yes | `pass` / `fail`, backfilled after the suite; null if the process died before backfill |
| `ts` | float | | unix time at write |
| `prev_hash` | str | | hash of the previous record in the file; `"0"*64` for the first |
| `hash` | str | | sha256(`prev_hash` + "\n" + canonical JSON of the record minus `hash`, `prev_hash`, `task_outcome`) |

### Provenance labels

`user`, `camel` (interpreter-derived), `assistant`, `trusted_tool`, `tool:<name>`, `tool:<name>[inner,…]` for
tools whose output carries inner sources (e.g. email senders). Readers are `"public"` or the sorted reader set.

## Canonicalization

`canonicalize_args(tool, kwargs)`:

1. Drop fields listed in `VOLATILE_FIELDS[tool]`.
2. Unwrap CaMeL values to raw Python, then normalize: strings are stripped of surrounding whitespace;
   enums → value; datetimes → ISO-8601; timedeltas → seconds; pydantic models → `model_dump()`;
   sets → sorted lists; classes → `<class Name>`; anything else → `repr`.
3. Serialize with sorted keys, compact separators, `ensure_ascii=False`.

Two calls hash equal iff tool name and canonical args are identical. Argument *values* are stored (this is the
broker-side dataset, never shown to the model). The free-text denial reason is not stored.

### Volatile fields per AgentDojo v1.2 tool

**None identified.** Every tool argument in the four suites is task-determined (ids, addresses, amounts,
dates and text from the task or from tool results). There is no timestamp, nonce, or session field that the
model must mint. `VOLATILE_FIELDS` is therefore empty; the table exists so the rule has one home. If a
tool is added whose argument legitimately varies between semantically identical calls, list it there with a
rationale here.

Reviewed tools: workspace (email, calendar, drive), slack, travel (hotel/restaurant/car/flight/calendar/email),
banking (send/schedule/update transaction, password, user info, file).

## Hash chain

Append-only within a file. Each record commits to its predecessor. `task_outcome` is excluded from the hash
because it is unknown at write time; the backfill rewrites the file atomically and re-verifies the chain
before replacing it. A new `RejectionLogger` on an existing file resumes from the last hash, so a crashed
run restarted with the same `runN` continues the chain (AgentDojo re-runs only uncached tasks). If a task
was mid-flight at the crash, its partial records remain and the rerun appends a second sequence for the same
task; analysis should take the sequence with the highest `ts` per task.

## Who can write

- The P-LLM output is executed by the CaMeL interpreter, which has no import statement, no file IO, and a
  namespace containing only builtins and AgentDojo tools. `tests/test_rejection_log.py` asserts the
  namespace cannot reach `open`, `__import__`, the module, or the logger, and that `import os` is an
  interpreter error.
- `RejectionLogger.record` is the only `.record(` call site under `src/camel` (test-enforced).
- The Q-LLM (`query_ai_assistant`) returns data, never code, and has no handle on the logger.

## conf_t

Requested via `logprobs=True` on chat completions (`LogprobOpenAILLM`, a drop-in for agentdojo's
`OpenAILLM`). The scored span is the **first fenced code block** in the assistant content — what the
interpreter runs — located by character offsets over the returned token strings. `conf_t = exp(mean logprob)`.
Content tokens only; reasoning tokens are never scored. o-series models return no logprobs → null, and
`LogprobOpenAILLM` does not ask for them there.
