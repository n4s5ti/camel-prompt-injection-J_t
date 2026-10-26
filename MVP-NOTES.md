# Enforced Learning MVP — CaMeL codebase findings (2026-10-26)

Linear: P-DEF-12. These are facts from reading the code, not the paper. They change the experiment design.

## 1. The published "+camel+secpol" number is a post-hoc replay, not live enforcement

Pipeline has three modes (`src/camel/models.py::make_tools_pipeline`):

| Mode | Flag | Policy engine | What happens on `Denied` |
|---|---|---|---|
| undefended | `--use-original` | none | n/a |
| camel | (default) | `ADNoSecurityPolicyEngine` (always `Allowed`) | never happens |
| camel+secpol | `--replay-with-policies` | real per-suite engine | **task terminates** (`replay_privileged_llm.py:189`) |

`--replay-with-policies` reads the saved trace from `logs/<model>+camel/...`, re-executes the saved code with the real policy engine, and on `SecurityPolicyDeniedError` returns immediately. The P-LLM is never invoked during replay. **The model never sees a denial in the paper's numbers.** The utility gap is "tasks whose no-policy code would have tripped a policy," with zero repair attempted.

Consequence: the MVP's "learning under enforcement" hypothesis needs a **live-policy baseline** that doesn't exist yet. Without it, "memory helps" is confounded with "retrying at all helps."

## 2. Three-arm design (replaces two-arm)

| Arm | Policy at generation time | Within-task retry on denial | Cross-task memory |
|---|---|---|---|
| A. camel-replay (paper) | no (post-hoc) | no | no |
| B. **camel-live** (new baseline) | yes | yes, up to `max_attempts=10` | no |
| C. camel-live + J_t memory (MVP) | yes | yes | yes |

B − A = value of within-task feedback. C − B = value of learning. Only C − B tests the hypothesis.

## 3. Where the hooks go

- **Policy choke point:** `src/camel/interpreter/interpreter.py:2050` — one `check_policy(tool_name, kwargs, dependencies)` call per tool invocation, returns `Allowed | Denied(reason)`. Raises `SecurityPolicyDeniedError` at :2063. **DEF-160's log record is written here.**
- **Retry loop:** `src/camel/pipeline_elements/privileged_llm.py::PrivilegedLLM.query` — `for _ in range(self.max_attempts)`, calls `_generate_and_interpret_code`, breaks on no `interpretation_error`. `make_error_messages` (:120) formats the error back to the P-LLM. **This is the within-task repair loop. It already exists; it just never receives policy denials** because (a) the default engine never denies and (b) `run_code` has no `except SecurityPolicyDeniedError`, so a denial would propagate as an uncaught exception and crash the task.
- **camel-live = two changes:** pass the real engine to `PrivilegedLLM` in `make_tools_pipeline`, and catch `SecurityPolicyDeniedError` in `run_code` / `_generate_and_interpret_code`, converting it to an error message so the retry loop handles it.
- **Memory retrieval point (DEF-162):** the system prompt or the first user message in `_generate_and_interpret_code` (`privileged_llm_messages` construction, ~:300).

## 4. Security requirement for camel-live: sanitize the denial reason

`security_policy.py::check_policy` builds `Denied(f"{tool_name} is state-changing and depends on private values {non_public_variables}.")` — **the reason string contains raw untrusted values.** In replay mode that's harmless (model never sees it). In live mode it would feed untrusted data straight into the P-LLM, which is exactly the channel CaMeL exists to close.

`format_camel_exception` redacts untrusted `CaMeLException`s, but `SecurityPolicyDeniedError` is a plain exception and bypasses that path.

Rule: what the P-LLM sees on denial is `{policy_name, tool_name, arg_name}` only. Never values. Never the untrusted-source content. This is the same rule DEF-160 already has for the log (`reason` = structured enum), now with a security reason behind it, not just a data-quality one.

## 5. Logprobs: constraint on model choice

- agentdojo 0.1.34's LLM layer requests no logprobs (`grep -r logprob agent_pipeline/llms/` → nothing).
- The P-LLM output is a **whole Python program**, not a discrete tool call. `conf_t` for the MVP = mean token logprob over the generated code block for the attempt that produced the denied call. (Per-line attribution is v2.)
- Providers: OpenAI chat completions supports `logprobs=True` on non-reasoning models (`gpt-4o-2024-08-06`, `gpt-4.1-2025-04-14`). **o3/o4-mini do not return logprobs.** Anthropic does not expose logprobs. Gemini: limited/unreliable. Local via vLLM (OpenAI-compatible) does.
- Decision for DEF-159: **`openai:gpt-4o-2024-08-06`** unless a local vLLM model is preferred. Subclass `agent_pipeline.OpenAILLM` to pass `logprobs=True` and stash the mean on the message / extra_args.
- No `seed` is passed to OpenAI; temperature is 0.0. "3 seeds" becomes "3 runs"; temp-0 is not deterministic at OpenAI. Report range across runs.

## 6. Task counts (AgentDojo v1.2) — statistical power

| suite | user tasks | injection tasks | attack pairs |
|---|---|---|---|
| workspace | 40 | 14 | 560 |
| slack | 21 | 5 | 105 |
| travel | 20 | 7 | 140 |
| **banking (held-out)** | **16** | **9** | **144** |
| total | 97 | 35 | 949 |

Held-out utility is measured on **16 tasks**. One task = 6.25 points. The pre-registered "beyond run-to-run variance" bar on banking will be coarse. Report counts, not just percentages. Consider also holding out slack (21) if train-set size allows; decide before DEF-165, not after.

## 9. Model decision (2026-10-26, supersedes §5's gpt-4o choice)

**`gpt-oss-120b` served by Cerebras** via `OPENAI_BASE_URL=https://api.cerebras.ai/v1`. Verified against Cerebras's API reference: chat completions accepts `logprobs`, `top_logprobs`, `seed`; returns `choices[].logprobs` for content tokens and `reasoning_logprobs` separately; tool calling supported. No client code changes — the `openai` SDK and pydantic_ai both honor `OPENAI_BASE_URL`.

Consequences, pre-registered:
- The paper never ran gpt-oss-120b, so **DEF-159 is "establish our baseline," not "reproduce 84/77."** The paper's numbers are a reference. The checks that carry over: undefended utility > camel+secpol utility; camel+secpol ASR ≈ 0. A later reader seeing e.g. 70/63 must not call it a failed reproduction.
- gpt-oss-120b is a reasoning model (MoE, ~5B active, `reasoning_effort` default medium). Not 4o-class in behavior; closer to o4-mini. `conf_t` is computed over **content** tokens only (`choices[].logprobs`), never `reasoning_logprobs`.
- `seed` is accepted (best-effort). Pass it; still report ranges across runs.
- AgentDojo's `important_instructions` attack addresses the model by its registered self-name (`MODEL_NAMES`). Added `gpt-oss-120b → "ChatGPT"` and `qwen-3.8-27b → "Qwen"` in `models.py`; **verify with `check_model_name.py` before the first run.** A wrong name weakens the attack and inflates security numbers.
- Cerebras `gpt-oss-120b` rejects `response_format` + `tools` in one request. The quarantined LLM uses pydantic_ai structured output; if it trips this, switch Q-LLM to `qwen-3.8-27b` via `--q-llm openai:qwen-3.8-27b` and record it.
- Rate limits: a full run is ~10–20M tokens. Needs the paid tier; free tier's daily cap will kill the run mid-suite.

## 7. Reproduce commands

See `reproduce_baseline.sh`. Order matters: arm A requires the `+camel` run to exist first (replay reads its logs).

## 8. Not changed

Paper claims, AgentDojo task definitions, attack (`important_instructions`), policy engines. Everything in this doc is observation of the artifact as released.
