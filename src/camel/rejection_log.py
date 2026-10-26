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

"""Structured rejection log (P-DEF-12 / DEF-160).

One JSONL record per tool-call *decision* taken by the CaMeL interpreter at the policy boundary
(`interpreter._eval_call`, right after `SecurityPolicyEngine.check_policy`). Builtins (`print`, `len`, ...)
and interpreter-defined functions/classes are not policy subjects and are not logged.

Every field is written broker-side from interpreter state. Nothing in a record comes from model narration.
The planner cannot reach this module: the CaMeL interpreter exposes no file IO and no imports, and the
namespace it evaluates in never contains the logger (see `tests/test_rejection_log.py`).

Schema, canonicalization rules and the hash-chain rule are documented in `REJECTION-LOG.md`.
"""

from __future__ import annotations

import dataclasses
import datetime
import enum
import hashlib
import json
import os
import threading
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pydantic

from camel.capabilities import get_all_readers, get_all_sources
from camel.capabilities import readers as readers_mod
from camel.capabilities import sources as sources_mod
from camel.interpreter.value import CaMeLValue
from camel.security_policy import Denied, SecurityPolicyResult

SCHEMA_VERSION = 1
GENESIS_HASH = "0" * 64

# Per-tool argument fields that are volatile (would differ between semantically identical calls) and are
# therefore stripped from `args_canonical` before hashing. AgentDojo v1.2 tools take only task-determined
# arguments (ids, addresses, dates given in the task text, model-authored text); no field has been identified
# as volatile. The table exists so the rule has one home. Rationale per entry belongs in REJECTION-LOG.md.
VOLATILE_FIELDS: dict[str, frozenset[str]] = {}

# Fields excluded from the hash. `task_outcome` is backfilled after the task ends, so it cannot be covered by
# a hash computed at write time; `hash`/`prev_hash` are the chain itself.
_UNHASHED_FIELDS = frozenset({"hash", "prev_hash", "task_outcome"})

# Signals that later issues compute from the log. Present in every record, null until those issues land.
_DOWNSTREAM_NULL_FIELDS = ("delta_to_goal", "lambda", "looked_good", "failed_t", "J_t", "verifier_miss")

DECISION_ACCEPTED = "accepted"
DECISION_REJECTED = "rejected"
DECISION_HALTED = "halted_for_confirmation"  # reserved; CaMeL never asks for confirmation today


class Decision(str, enum.Enum):
    ACCEPTED = DECISION_ACCEPTED
    REJECTED = DECISION_REJECTED
    HALTED_FOR_CONFIRMATION = DECISION_HALTED


def _jsonable(x: Any) -> Any:
    """Converts a (possibly CaMeL-wrapped) value into a deterministic JSON-serializable structure."""
    if isinstance(x, CaMeLValue):
        x = x.raw
    if x is None or isinstance(x, bool | int | str):
        return x.strip() if isinstance(x, str) else x
    if isinstance(x, float):
        return x if x == x and x not in (float("inf"), float("-inf")) else repr(x)
    if isinstance(x, enum.Enum):
        return _jsonable(x.value)
    if isinstance(x, datetime.datetime | datetime.date | datetime.time):
        return x.isoformat()
    if isinstance(x, datetime.timedelta):
        return x.total_seconds()
    if isinstance(x, pydantic.BaseModel):
        return _jsonable(x.model_dump())
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, set | frozenset):
        return sorted((_jsonable(v) for v in x), key=lambda v: json.dumps(v, sort_keys=True, default=str))
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, type):
        return f"<class {x.__name__}>"
    if isinstance(x, bytes):
        return x.hex()
    return repr(x)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def canonicalize_args(tool: str, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical argument map: volatile fields stripped, values JSON-normalized, keys sorted at dump time."""
    volatile = VOLATILE_FIELDS.get(tool, frozenset())
    return {name: _jsonable(v) for name, v in kwargs.items() if name not in volatile}


def args_hash(tool: str, args_canonical: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json({"tool": tool, "args": args_canonical}).encode()).hexdigest()


def _source_label(s: sources_mod.Source) -> str:
    if isinstance(s, sources_mod.Tool):
        if s.inner_sources:
            inner = ",".join(sorted(_source_label(i) if not isinstance(i, str) else i for i in s.inner_sources))
            return f"tool:{s.tool_name}[{inner}]"
        return f"tool:{s.tool_name}"
    return {
        sources_mod.SourceEnum.User: "user",
        sources_mod.SourceEnum.CaMeL: "camel",
        sources_mod.SourceEnum.Assistant: "assistant",
        sources_mod.SourceEnum.TrustedToolSource: "trusted_tool",
    }.get(s, str(s))


def provenance_of(v: CaMeLValue) -> dict[str, Any]:
    srcs, _ = get_all_sources(v)
    rdrs, _ = get_all_readers(v)
    return {
        "sources": sorted(_source_label(s) for s in srcs),
        "readers": "public" if isinstance(rdrs, readers_mod.Public) else sorted(str(r) for r in rdrs),
    }


def record_hash(record: Mapping[str, Any], prev_hash: str) -> str:
    body = {k: v for k, v in record.items() if k not in _UNHASHED_FIELDS}
    return hashlib.sha256((prev_hash + "\n" + canonical_json(body)).encode()).hexdigest()


@dataclasses.dataclass
class TaskContext:
    suite: str
    task_id: str
    injection_task_id: str | None
    attempt_n: int = 0
    step_n: int = 0
    seen_hashes: set[str] = dataclasses.field(default_factory=set)
    conf_t: float | None = None
    conf_t_scope: str | None = None
    conf_t_n_tokens: int | None = None
    code_hash: str | None = None


class RejectionLogger:
    """Append-only, hash-chained JSONL writer. The only thing in the codebase that writes rejection records.

    Lifecycle per task: `begin_task` → (`begin_attempt` → `record`*)* → `end_task`.
    After a suite finishes, `backfill_task_outcomes` stamps `task_outcome` onto every record of each task.
    """

    def __init__(self, path: Path | str, *, run_id: str, arm: str, seed: int | None = None, epoch: int = 0) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self.arm = arm
        self.seed = seed
        self.epoch = epoch
        self._ctx: TaskContext | None = None
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._prev_hash = self._read_last_hash()

    # ---- lifecycle -------------------------------------------------------------------------------------

    def begin_task(self, suite: str, task_id: str, injection_task_id: str | None) -> None:
        self._ctx = TaskContext(suite=suite, task_id=task_id, injection_task_id=injection_task_id)

    def begin_attempt(
        self,
        *,
        conf_t: float | None = None,
        conf_t_scope: str | None = None,
        conf_t_n_tokens: int | None = None,
        code: str | None = None,
    ) -> None:
        ctx = self._require_ctx()
        ctx.attempt_n += 1
        ctx.conf_t = conf_t
        ctx.conf_t_scope = conf_t_scope
        ctx.conf_t_n_tokens = conf_t_n_tokens
        ctx.code_hash = hashlib.sha256(code.encode()).hexdigest() if code is not None else None

    def end_task(self) -> None:
        self._ctx = None

    @property
    def in_task(self) -> bool:
        return self._ctx is not None

    # ---- the one write path ----------------------------------------------------------------------------

    def record(
        self,
        tool: str,
        kwargs: Mapping[str, CaMeLValue],
        dependencies: Iterable[CaMeLValue],
        result: SecurityPolicyResult,
    ) -> dict[str, Any]:
        ctx = self._require_ctx()
        ctx.step_n += 1
        args_canonical = canonicalize_args(tool, kwargs)
        h = args_hash(tool, args_canonical)
        seen_before = h in ctx.seen_hashes
        ctx.seen_hashes.add(h)

        if isinstance(result, Denied):
            decision = DECISION_REJECTED
            policy_failed = result.policy
            field_failed = result.field
            reason = str(result.code) if result.code is not None else None
        else:
            decision, policy_failed, field_failed, reason = DECISION_ACCEPTED, None, None, None

        deps = list(dependencies)
        rec: dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "run_id": self.run_id,
            "seed": self.seed,
            "arm": self.arm,
            "suite": ctx.suite,
            "task_id": ctx.task_id,
            "injection_task_id": ctx.injection_task_id,
            "epoch": self.epoch,
            "attempt_n": ctx.attempt_n,
            "step_n": ctx.step_n,
            "tool": tool,
            "args_canonical": args_canonical,
            "args_hash": h,
            "args_hash_seen_before": seen_before,
            "decision": decision,
            "policy_failed": policy_failed,
            "field_failed": field_failed,
            "reason": reason,
            "provenance_summary": {name: provenance_of(v) for name, v in kwargs.items()},
            "n_dependencies": len(deps),
            "dependencies_public": all(_is_public_readers(d) for d in deps),
            "conf_t": ctx.conf_t,
            "conf_t_scope": ctx.conf_t_scope,
            "conf_t_n_tokens": ctx.conf_t_n_tokens,
            "code_hash": ctx.code_hash,
            **{k: None for k in _DOWNSTREAM_NULL_FIELDS},
            "task_outcome": None,
            "ts": time.time(),
        }
        with self._lock:
            rec["prev_hash"] = self._prev_hash
            rec["hash"] = record_hash(rec, self._prev_hash)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(canonical_json(rec) + "\n")
                f.flush()
            self._prev_hash = rec["hash"]
        return rec

    # ---- post-hoc ----------------------------------------------------------------------------------------

    def backfill_task_outcomes(self, outcomes: Mapping[tuple[str, str], bool]) -> int:
        """Stamps `task_outcome` ("pass"/"fail") onto every record whose task is in `outcomes`.

        `outcomes` is AgentDojo's `utility_results`: `{(user_task_id, injection_task_id or ""): bool}`.
        Only records of this logger's `run_id`/`arm` with `task_outcome is None` are touched. The hash chain
        excludes `task_outcome`, so it stays valid; it is re-verified before the file is replaced.
        """
        with self._lock:
            if not self.path.exists():
                return 0
            lines = self.path.read_text(encoding="utf-8").splitlines()
            updated = 0
            out: list[str] = []
            for line in lines:
                rec = json.loads(line)
                key = (rec["task_id"], rec["injection_task_id"] or "")
                if (
                    rec["task_outcome"] is None
                    and rec["run_id"] == self.run_id
                    and rec["arm"] == self.arm
                    and key in outcomes
                ):
                    rec["task_outcome"] = "pass" if outcomes[key] else "fail"
                    updated += 1
                out.append(canonical_json(rec))
            if updated:
                tmp = self.path.with_suffix(self.path.suffix + ".tmp")
                tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
                verify_chain(tmp)
                os.replace(tmp, self.path)
            return updated

    # ---- internals ---------------------------------------------------------------------------------------

    def _require_ctx(self) -> TaskContext:
        if self._ctx is None:
            raise RuntimeError("RejectionLogger used outside of a task; call begin_task() first")
        return self._ctx

    def _read_last_hash(self) -> str:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return GENESIS_HASH
        last = None
        with self.path.open("rb") as f:
            for raw in f:
                if raw.strip():
                    last = raw
        if last is None:
            return GENESIS_HASH
        return json.loads(last)["hash"]


def _is_public_readers(v: CaMeLValue) -> bool:
    return isinstance(get_all_readers(v)[0], readers_mod.Public)


def iter_records(path: Path | str) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def verify_chain(path: Path | str) -> int:
    """Walks the file and re-derives every hash. Returns the record count; raises ValueError on the first break."""
    prev = GENESIS_HASH
    n = 0
    for rec in iter_records(path):
        if rec["prev_hash"] != prev:
            raise ValueError(f"chain break at record {n}: prev_hash {rec['prev_hash']} != {prev}")
        expected = record_hash(rec, prev)
        if rec["hash"] != expected:
            raise ValueError(f"hash mismatch at record {n}: {rec['hash']} != {expected}")
        prev = rec["hash"]
        n += 1
    return n


def required_fields() -> tuple[str, ...]:
    """Every key a record must carry. Nullable keys are documented in REJECTION-LOG.md."""
    return (
        "schema",
        "run_id",
        "seed",
        "arm",
        "suite",
        "task_id",
        "injection_task_id",
        "epoch",
        "attempt_n",
        "step_n",
        "tool",
        "args_canonical",
        "args_hash",
        "args_hash_seen_before",
        "decision",
        "policy_failed",
        "field_failed",
        "reason",
        "provenance_summary",
        "n_dependencies",
        "dependencies_public",
        "conf_t",
        "conf_t_scope",
        "conf_t_n_tokens",
        "code_hash",
        *_DOWNSTREAM_NULL_FIELDS,
        "task_outcome",
        "ts",
        "prev_hash",
        "hash",
    )
