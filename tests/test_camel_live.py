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

"""DEF-167: camel-live (arm B) — denials become sanitized, retryable errors; the canary never reaches the P-LLM."""

from collections.abc import Sequence

import pytest
from agentdojo import agent_pipeline, functions_runtime, task_suite
from agentdojo import types as ad_types

from camel import rejection_log, security_policy
from camel.capabilities import Capabilities, sources
from camel.interpreter import interpreter, result, value
from camel.interpreter import namespace as ns
from camel.pipeline_elements import agentdojo_function
from camel.pipeline_elements.privileged_llm import PrivilegedLLM, make_error_messages
from camel.pipeline_elements.security_policies import WorkspaceSecurityPolicyEngine
from camel.security_policy import DenialCode, Denied, sanitized_denial_message

CANARY = "canary7f3a9c@secret-leak.example"  # a valid email so it survives AgentDojo's arg validation


def _workspace():
    suite = task_suite.get_suite("v1.2", "workspace")
    env = suite.load_and_inject_default_environment({})
    runtime = functions_runtime.FunctionsRuntime(suite.tools)
    builtins_namespace = ns.Namespace.with_builtins()
    namespace = builtins_namespace.add_variables(
        agentdojo_function.make_agentdojo_namespace(builtins_namespace, runtime, env)
    )
    return env, runtime, namespace


def _private(s: str) -> value.CaMeLStr:
    """A value that came out of a tool and is readable only by someone else — i.e. private and untrusted."""
    return value.CaMeLStr.from_raw(
        s, Capabilities(frozenset({sources.Tool("get_received_emails")}), frozenset({"someone@else"})), ()
    )


def _all_text(messages: Sequence[ad_types.ChatMessage]) -> str:
    out = []
    for m in messages:
        content = m.get("content")
        if content:
            out.append(ad_types.get_text_content_as_str(content))
    return "\n".join(out)


# ---- the message itself -------------------------------------------------------------------------------


def test_sanitized_message_has_structure_and_no_values():
    d = Denied(
        f"depends on private values ['{CANARY}']", code=DenialCode.DEPENDS_ON_PRIVATE, policy="engine.dependencies"
    )
    msg = sanitized_denial_message("send_email", d)
    assert CANARY not in msg
    assert "send_email" in msg and "engine.dependencies" in msg and "depends_on_private" in msg
    d = Denied(CANARY, code=DenialCode.UNTRUSTED_SOURCE, field="file_id", policy="delete_file")
    msg = sanitized_denial_message("delete_file", d)
    assert CANARY not in msg and "argument 'file_id'" in msg


# ---- interpreter: ERROR mode vs RAISE mode ---------------------------------------------------------------


def _interpret(code: str, namespace, env, mode, logger=None):
    eval_args = interpreter.EvalArgs(
        WorkspaceSecurityPolicyEngine(env), interpreter.MetadataEvalMode.NORMAL, logger, mode
    )
    return interpreter.parse_and_interpret_code(f"```python\n{code}\n```", namespace, [], [], eval_args)


def test_raise_mode_is_unchanged():
    env, _, namespace = _workspace()
    with pytest.raises(security_policy.SecurityPolicyDeniedError):
        _interpret("delete_file(file_id=list_files()[0].id_)", namespace, env, interpreter.DenialMode.RAISE)


def test_error_mode_returns_trusted_error_and_keeps_namespace():
    env, _, namespace = _workspace()
    code = "files = list_files()\nn = len(files)\ndelete_file(file_id=files[0].id_)\nnever = 1"
    res, new_ns, tool_calls, _ = _interpret(code, namespace, env, interpreter.DenialMode.ERROR)
    assert isinstance(res, result.Error)
    err = res.error
    assert isinstance(err.exception, security_policy.SecurityPolicyDeniedError)
    assert err.nodes[-1].lineno == 3
    assert "files" in new_ns.variables and "n" in new_ns.variables and "never" not in new_ns.variables
    assert [tc.function for tc in tool_calls if not tc.is_builtin] == ["list_files"]  # delete_file never ran


def test_canary_never_reaches_the_planner_via_dependency_denial():
    """The engine's own reason string interpolates the private values. In live mode that text must not leak."""
    env, _, namespace = _workspace()
    namespace = namespace.add_variables({"secret": _private(CANARY)})
    # Control flow depends on the private value itself -> engine-level DEPENDS_ON_PRIVATE whose `reason`
    # interpolates `non_public_variables`, i.e. contains CANARY verbatim.
    code = "if secret:\n    send_email(recipients=['a@b.com'], subject='s', body='b')"
    engine_reason = WorkspaceSecurityPolicyEngine(env).check_policy("send_email", {}, [namespace.variables["secret"]])
    assert isinstance(engine_reason, Denied) and CANARY in engine_reason.reason  # the hole is real upstream
    res, *_ = _interpret(code, namespace, env, interpreter.DenialMode.ERROR)
    assert isinstance(res, result.Error)
    messages = make_error_messages(code, res.error)
    text = _all_text(messages)
    assert CANARY not in text
    assert "redacted" not in text  # the sanitized message is trusted and therefore shown, not blanked
    assert "depends_on_private" in text and "send_email" in text


def test_canary_never_reaches_the_planner_via_policy_denial():
    """Policy-level reasons embed recipient sets and reader sets. Use real tool data so the taint chain is the
    one production sees: body from one received email, recipient from another → READERS_CANNOT_READ on body,
    and the engine's reason contains that recipient address verbatim."""
    env, _, namespace = _workspace()
    captured: list[Denied] = []

    class Spy(WorkspaceSecurityPolicyEngine):
        def check_policy(self, tool_name, kwargs, dependencies):
            res = super().check_policy(tool_name, kwargs, dependencies)
            if isinstance(res, Denied):
                captured.append(res)
            return res

    code = "ems = get_received_emails()\nsend_email(recipients=[ems[1].sender], subject='s', body=ems[0].body)"
    eval_args = interpreter.EvalArgs(Spy(env), interpreter.MetadataEvalMode.NORMAL, None, interpreter.DenialMode.ERROR)
    res, new_ns, *_ = interpreter.parse_and_interpret_code(f"```python\n{code}\n```", namespace, [], [], eval_args)
    assert isinstance(res, result.Error)
    assert len(captured) == 1 and captured[0].code == DenialCode.READERS_CANNOT_READ and captured[0].field == "body"
    leaked_recipient = new_ns.variables["ems"].raw[1].sender
    assert leaked_recipient in captured[0].reason  # upstream's free text really does carry the value
    text = _all_text(make_error_messages(code, res.error))
    assert leaked_recipient not in text
    assert "send_email" in text and "argument 'body'" in text and "readers_cannot_read" in text


# ---- the retry loop actually retries -------------------------------------------------------------------


class _ScriptedLLM(agent_pipeline.BasePipelineElement):
    """Returns pre-written programs in order; records every message it was shown."""

    name = "scripted"

    def __init__(self, programs: list[str]) -> None:
        self.programs = list(programs)
        self.seen: list[Sequence[ad_types.ChatMessage]] = []

    def query(self, query, runtime, env=functions_runtime.EmptyEnv(), messages=[], extra_args={}):
        self.seen.append(list(messages))
        code = self.programs.pop(0)
        out = ad_types.ChatAssistantMessage(
            role="assistant",
            content=[ad_types.text_content_block_from_string(f"```python\n{code}\n```")],
            tool_calls=None,
        )
        extra_args["conf_t"] = 0.5
        return query, runtime, env, [*messages, out], extra_args


def test_live_pipeline_retries_after_denial_and_logs_both_attempts(tmp_path):
    env, runtime, _ = _workspace()
    llm = _ScriptedLLM(
        [
            "files = list_files()\ndelete_file(file_id=files[0].id_)",  # denied: file_id from a tool
            "print(len(files))",  # repair: no side effect, reuses the preserved namespace
        ]
    )
    log = rejection_log.RejectionLogger(tmp_path / "log.jsonl", run_id="r", arm="x+camel+live")
    p_llm = PrivilegedLLM(
        llm,
        WorkspaceSecurityPolicyEngine,
        "openai:gpt-4o",
        rejection_logger=log,
        denial_mode=interpreter.DenialMode.ERROR,
    )
    extra_args = {"task_suite_name": "workspace", "user_task_id": "user_task_0", "injection_task_id": None}
    _, _, _, messages, _ = p_llm.query("delete the first file", runtime, env, [], extra_args)

    assert llm.programs == []  # both programs consumed → the loop retried exactly once
    # The second call saw the sanitized denial, and only that.
    retry_prompt = _all_text(llm.seen[1])
    assert "delete_file" in retry_prompt and "untrusted_source" in retry_prompt and "argument 'file_id'" in retry_prompt
    assert "redacted" not in retry_prompt
    recs = list(rejection_log.iter_records(tmp_path / "log.jsonl"))
    assert [(r["attempt_n"], r["tool"], r["decision"]) for r in recs] == [
        (1, "list_files", "accepted"),
        (1, "delete_file", "rejected"),
    ]
    assert recs[1]["reason"] == "untrusted_source" and recs[1]["conf_t"] == 0.5
    assert messages[-1]["role"] == "assistant"  # task ended cleanly, not in a traceback


def test_live_pipeline_gives_up_after_max_attempts(tmp_path):
    env, runtime, _ = _workspace()
    bad = "delete_file(file_id=list_files()[0].id_)"
    llm = _ScriptedLLM([bad, bad, bad])
    p_llm = PrivilegedLLM(
        llm, WorkspaceSecurityPolicyEngine, "openai:gpt-4o", max_attempts=3, denial_mode=interpreter.DenialMode.ERROR
    )
    _, _, _, messages, _ = p_llm.query("q", runtime, env, [], {"task_suite_name": "workspace", "user_task_id": "u"})
    assert llm.programs == [] and len(llm.seen) == 3
    assert CANARY not in _all_text(messages)
    assert messages[-1]["role"] == "assistant" and messages[-1]["content"] is None  # upstream's traceback terminator


def test_make_tools_pipeline_live_wiring(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from camel import models

    p = models.make_tools_pipeline(
        "openai:gpt-4o-2024-08-06",
        False,
        False,
        "important_instructions",
        "medium",
        None,
        "workspace",
        None,
        interpreter.MetadataEvalMode.NORMAL,
        None,
        tmp_path,
        live_policies=True,
    )
    assert p.name == "gpt-4o-2024-08-06+camel+live"
    assert "gpt-4o-2024-08-06+camel+live" in models.MODEL_NAMES
    element = list(p.elements)[-1]
    assert isinstance(element, PrivilegedLLM)
    assert element.security_policy_engine is WorkspaceSecurityPolicyEngine
    assert element.denial_mode == interpreter.DenialMode.ERROR
    assert element.rejection_logger is not None and element.rejection_logger.arm == "gpt-4o-2024-08-06+camel+live"
    with pytest.raises(ValueError):
        models.make_tools_pipeline(
            "openai:gpt-4o-2024-08-06",
            False,
            True,
            "important_instructions",
            "medium",
            None,
            "workspace",
            None,
            interpreter.MetadataEvalMode.NORMAL,
            None,
            tmp_path,
            live_policies=True,
        )
