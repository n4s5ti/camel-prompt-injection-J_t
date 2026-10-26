# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DEF-160: structured rejection log — schema, hash chain, backfill, interpreter hook, planner isolation."""

import ast
import json
import math
import pathlib

import pytest
from agentdojo import functions_runtime, task_suite

from camel import rejection_log, security_policy
from camel.capabilities import Capabilities, readers, sources
from camel.interpreter import interpreter, value
from camel.interpreter import namespace as ns
from camel.pipeline_elements import agentdojo_function
from camel.pipeline_elements.logprob_openai_llm import score_tokens
from camel.pipeline_elements.security_policies import ADNoSecurityPolicyEngine, WorkspaceSecurityPolicyEngine
from camel.security_policy import DenialCode, Denied

REPO = pathlib.Path(__file__).resolve().parents[1]
POLICIES_DIR = REPO / "src" / "camel" / "pipeline_elements" / "security_policies"

# Nullable by schema (see REJECTION-LOG.md). Everything else must be non-null in every record.
NULLABLE = {
    "seed",
    "injection_task_id",
    "policy_failed",
    "field_failed",
    "reason",
    "conf_t",
    "conf_t_scope",
    "conf_t_n_tokens",
    "conf_t_n_excluded",
    "code_hash",
    "delta_to_goal",
    "lambda",
    "looked_good",
    "failed_t",
    "J_t",
    "verifier_miss",
    "task_outcome",
}


def _user_str(s: str) -> value.CaMeLStr:
    return value.CaMeLStr.from_raw(s, Capabilities.default(), ())


def _tool_str(s: str, tool: str = "list_files", public: bool = True) -> value.CaMeLStr:
    rdrs = readers.Public() if public else frozenset({"alice@example.com"})
    return value.CaMeLStr.from_raw(s, Capabilities(frozenset({sources.Tool(tool)}), rdrs), ())


def _workspace():
    suite = task_suite.get_suite("v1.2", "workspace")
    env = suite.load_and_inject_default_environment({})
    runtime = functions_runtime.FunctionsRuntime(suite.tools)
    builtins_namespace = ns.Namespace.with_builtins()
    namespace = builtins_namespace.add_variables(
        agentdojo_function.make_agentdojo_namespace(builtins_namespace, runtime, env)
    )
    return env, namespace


# ---- schema / canonicalization / chain -------------------------------------------------------------------


def test_canonicalization_is_order_independent_and_strips_volatile(monkeypatch):
    monkeypatch.setitem(rejection_log.VOLATILE_FIELDS, "t", frozenset({"nonce"}))
    a = rejection_log.canonicalize_args("t", {"b": _user_str(" x "), "a": _user_str("y"), "nonce": _user_str("1")})
    b = rejection_log.canonicalize_args("t", {"a": _user_str("y"), "b": _user_str("x"), "nonce": _user_str("2")})
    assert a == {"a": "y", "b": "x"}
    assert rejection_log.args_hash("t", a) == rejection_log.args_hash("t", b)
    assert rejection_log.args_hash("t", a) != rejection_log.args_hash("u", a)


def test_chain_backfill_and_required_fields(tmp_path):
    path = tmp_path / "log.jsonl"
    log = rejection_log.RejectionLogger(path, run_id="r", arm="x+camel")
    log.begin_task("workspace", "user_task_1", None)
    log.begin_attempt(conf_t=0.5, conf_t_scope="code_block", conf_t_n_tokens=3, code="send_email()")
    r1 = log.record("send_email", {"recipients": _user_str("a@b")}, [], security_policy.Allowed())
    r2 = log.record("send_email", {"recipients": _user_str("a@b")}, [], security_policy.Allowed())
    log.begin_attempt(conf_t=0.4)
    r3 = log.record(
        "delete_file",
        {"file_id": _tool_str("7")},
        [_tool_str("p", public=False)],
        Denied("x", code=DenialCode.UNTRUSTED_SOURCE, field="file_id", policy="delete_file"),
    )
    log.end_task()
    log.begin_task("workspace", "user_task_2", "injection_task_0")
    log.begin_attempt()
    r4 = log.record("send_email", {}, [], security_policy.Allowed())
    log.end_task()

    assert (r1["attempt_n"], r1["step_n"], r1["args_hash_seen_before"]) == (1, 1, False)
    assert (r2["attempt_n"], r2["step_n"], r2["args_hash_seen_before"]) == (1, 2, True)
    assert (r3["attempt_n"], r3["step_n"], r3["decision"]) == (2, 3, "rejected")
    assert (r3["policy_failed"], r3["field_failed"], r3["reason"]) == ("delete_file", "file_id", "untrusted_source")
    assert r3["provenance_summary"] == {"file_id": {"sources": ["tool:list_files"], "readers": "public"}}
    assert r3["dependencies_public"] is False and r3["n_dependencies"] == 1
    assert r1["conf_t"] == 0.5 and r3["conf_t"] == 0.4 and r4["conf_t"] is None
    assert r4["task_id"] == "user_task_2" and r4["injection_task_id"] == "injection_task_0"
    assert r1["prev_hash"] == rejection_log.GENESIS_HASH and r2["prev_hash"] == r1["hash"]
    # The free-text Denied.reason ("x") is never stored; only the structured code is.
    assert all(rec["reason"] in (None, "untrusted_source") for rec in (r1, r2, r3, r4))

    for rec in (r1, r2, r3, r4):
        assert tuple(rec.keys()) == rejection_log.required_fields()
        for k, v in rec.items():
            if k not in NULLABLE:
                assert v is not None, k

    assert rejection_log.verify_chain(path) == 4

    # Backfill: only our run/arm, only matching tasks; chain still verifies; idempotent.
    n = log.backfill_task_outcomes({("user_task_1", ""): True, ("user_task_2", "injection_task_0"): False})
    assert n == 4
    recs = list(rejection_log.iter_records(path))
    assert [r["task_outcome"] for r in recs] == ["pass", "pass", "pass", "fail"]
    assert rejection_log.verify_chain(path) == 4
    assert log.backfill_task_outcomes({("user_task_1", ""): True}) == 0

    # Resume: a new logger on the same file continues the chain.
    log2 = rejection_log.RejectionLogger(path, run_id="r", arm="x+camel")
    log2.begin_task("workspace", "user_task_3", None)
    log2.begin_attempt()
    r5 = log2.record("send_email", {}, [], security_policy.Allowed())
    assert r5["prev_hash"] == recs[-1]["hash"]
    assert rejection_log.verify_chain(path) == 5


def test_tampering_breaks_chain(tmp_path):
    path = tmp_path / "log.jsonl"
    log = rejection_log.RejectionLogger(path, run_id="r", arm="a")
    log.begin_task("s", "t", None)
    log.begin_attempt()
    log.record("send_email", {"body": _user_str("hi")}, [], security_policy.Allowed())
    log.record("send_email", {"body": _user_str("hi")}, [], security_policy.Allowed())
    lines = path.read_text().splitlines()
    rec = json.loads(lines[0])
    rec["decision"] = "rejected"
    lines[0] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        rejection_log.verify_chain(path)


# ---- the engine produces structured denials ---------------------------------------------------------------


def test_every_denied_call_site_carries_a_code():
    sites = 0
    for src in [*POLICIES_DIR.glob("*.py"), REPO / "src" / "camel" / "security_policy.py"]:
        for node in ast.walk(ast.parse(src.read_text())):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Denied":
                sites += 1
                assert any(k.arg == "code" for k in node.keywords), f"{src.name}:{node.lineno} Denied() without code="
    assert sites >= 33


def test_engine_fills_policy_and_codes():
    env, _ = _workspace()
    engine = WorkspaceSecurityPolicyEngine(env)
    d = engine.check_policy("delete_file", {"file_id": _tool_str("7")}, [])
    assert isinstance(d, Denied)
    assert (d.code, d.field, d.policy) == (DenialCode.UNTRUSTED_SOURCE, "file_id", "delete_file")

    d = engine.check_policy("delete_file", {"file_id": _user_str("7")}, [_tool_str("p", public=False)])
    assert isinstance(d, Denied)
    assert (d.code, d.field, d.policy) == (DenialCode.DEPENDS_ON_PRIVATE, None, "engine.dependencies")

    d = engine.check_policy("some_new_tool", {}, [])
    assert isinstance(d, Denied)
    assert (d.code, d.policy) == (DenialCode.NO_POLICY_MATCHED, "engine.default")

    assert isinstance(engine.check_policy("delete_file", {"file_id": _user_str("7")}, []), security_policy.Allowed)


# ---- interpreter hook ----------------------------------------------------------------------------------------


def _run(code: str, engine, logger):
    env, namespace = _workspace()
    eval_args = interpreter.EvalArgs(engine(env), interpreter.MetadataEvalMode.NORMAL, logger)
    # parse_and_interpret_code takes the P-LLM's markdown output, i.e. a fenced code block.
    return interpreter.parse_and_interpret_code(f"```python\n{code}\n```", namespace, [], [], eval_args)


def test_interpreter_records_accepted_and_rejected_calls(tmp_path):
    path = tmp_path / "log.jsonl"
    log = rejection_log.RejectionLogger(path, run_id="r", arm="x+camel+secpol")
    log.begin_task("workspace", "user_task_0", None)
    log.begin_attempt(conf_t=0.9, code="...")
    code = "files = list_files()\nn = len(files)\nprint(n)\ndelete_file(file_id=files[0].id_)\n"
    with pytest.raises(security_policy.SecurityPolicyDeniedError):
        _run(code, WorkspaceSecurityPolicyEngine, log)
    log.end_task()

    recs = list(rejection_log.iter_records(path))
    # list_files (accepted) and delete_file (rejected). len/print are builtins → not recorded.
    assert [(r["tool"], r["decision"]) for r in recs] == [("list_files", "accepted"), ("delete_file", "rejected")]
    rej = recs[1]
    assert (rej["policy_failed"], rej["field_failed"], rej["reason"]) == ("delete_file", "file_id", "untrusted_source")
    srcs = rej["provenance_summary"]["file_id"]["sources"]
    assert "tool:list_files" in srcs and "camel" in srcs  # came out of a tool, through interpreter indexing
    assert rej["args_canonical"] == {"file_id": recs[1]["args_canonical"]["file_id"]}  # raw id string, canonical
    assert rej["conf_t"] == 0.9 and rej["attempt_n"] == 1 and rej["step_n"] == 2
    assert rejection_log.verify_chain(path) == 2


def test_no_policy_engine_still_logs_accepts(tmp_path):
    path = tmp_path / "log.jsonl"
    log = rejection_log.RejectionLogger(path, run_id="r", arm="x+camel")
    log.begin_task("workspace", "user_task_0", None)
    log.begin_attempt()
    _run("files = list_files()\ndelete_file(file_id=files[0].id_)\n", ADNoSecurityPolicyEngine, log)
    log.end_task()
    assert [(r["tool"], r["decision"]) for r in rejection_log.iter_records(path)] == [
        ("list_files", "accepted"),
        ("delete_file", "accepted"),
    ]


def test_no_logger_means_no_side_effects(tmp_path):
    _run("files = list_files()\n", ADNoSecurityPolicyEngine, None)
    assert list(tmp_path.iterdir()) == []


def test_logger_outside_task_is_inert(tmp_path):
    path = tmp_path / "log.jsonl"
    log = rejection_log.RejectionLogger(path, run_id="r", arm="x")
    _run("files = list_files()\n", ADNoSecurityPolicyEngine, log)  # no begin_task → nothing written
    assert not path.exists()


# ---- the planner cannot write the log ----------------------------------------------------------------------


def test_planner_namespace_cannot_reach_the_log():
    _, namespace = _workspace()
    reachable = {k: v for k, v in namespace.variables.items()}
    forbidden = {"open", "__import__", "exec", "eval", "compile", "RejectionLogger", "rejection_log", "os", "sys"}
    assert forbidden.isdisjoint(reachable), forbidden & set(reachable)
    for v in reachable.values():
        raw = getattr(v, "raw", None)
        assert raw is not rejection_log and not isinstance(raw, rejection_log.RejectionLogger)
    # and the language has no imports: `import os` is an interpreter error, not an import.
    res, *_ = _run("import os\n", ADNoSecurityPolicyEngine, None)
    assert isinstance(res, interpreter.result.Error)


def test_record_is_the_only_writer():
    """Exactly one call site writes to the log file, and it is inside RejectionLogger."""
    src = (REPO / "src" / "camel").rglob("*.py")
    writers = []
    for p in src:
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "record":
                writers.append(f"{p.relative_to(REPO)}:{node.lineno}")
    assert writers == ["src/camel/interpreter/interpreter.py:" + writers[0].split(":")[1]], writers


# ---- conf_t ---------------------------------------------------------------------------------------------------


def test_score_tokens_prefers_code_block():
    tokens = ["Here", " is", " code", ":\n", "```", "python", "\n", "x", " =", " 1", "\n", "```", "\n", "done"]
    lps = [-5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -0.1, -0.2, -0.3, -0.4, -5.0, -5.0, -5.0]
    conf, scope, n, excluded = score_tokens(tokens, lps)
    assert scope == "code_block" and n == 4 and excluded == 0
    assert conf == pytest.approx(math.exp(-0.25))
    conf, scope, n, excluded = score_tokens(["plain", " text"], [-1.0, -3.0])
    assert scope == "content" and n == 2 and conf == pytest.approx(math.exp(-2.0))
    assert score_tokens([], []) == (None, "content", 0, 0)


def test_score_tokens_drops_openai_sentinel():
    """OpenAI returns -9999.0 for tokens it cannot score; one of them must not collapse conf_t to ~0."""
    tokens = ["```", "\n", "a", " b", " c", "\n", "```"]
    lps = [-1.0, -1.0, -0.1, -9999.0, -0.3, -1.0, -1.0]
    conf, scope, n, excluded = score_tokens(tokens, lps)
    assert scope == "code_block" and n == 3 and excluded == 1  # 'a', ' c' and the newline inside the fence
    assert conf == pytest.approx(math.exp((-0.1 - 0.3 - 1.0) / 3))
    assert score_tokens(["x"], [-9999.0]) == (None, "content", 0, 1)


def test_task_run_n_counts_agentdojo_reinvocations(tmp_path):
    log = rejection_log.RejectionLogger(tmp_path / "log.jsonl", run_id="r", arm="a")
    runs = []
    for key in [("s", "t1", None), ("s", "t1", None), ("s", "t1", None), ("s", "t2", None), ("s", "t1", None)]:
        log.begin_task(*key)
        log.begin_attempt()
        runs.append(log.record("send_email", {}, [], security_policy.Allowed())["task_run_n"])
        log.end_task()
    assert runs == [1, 2, 3, 1, 1]
