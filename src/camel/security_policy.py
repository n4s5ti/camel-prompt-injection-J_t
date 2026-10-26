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

"""Security policies for tools."""

import dataclasses
import enum
import fnmatch
from collections.abc import Iterable, Mapping
from typing import Protocol

from camel.capabilities import get_all_readers, is_public, readers
from camel.interpreter.value import CaMeLValue


@dataclasses.dataclass(frozen=True)
class Allowed:
    """Allowed access to the tool."""

    ...


class DenialCode(str, enum.Enum):
    """Machine-readable predicate that failed. The rejection log (DEF-160) stores this, never the free-text reason.

    Engine-level codes (no `field`, raised before any per-tool policy runs):
      DEPENDS_ON_PRIVATE  — control flow reaching this call depends on a non-public value.
      NO_POLICY_MATCHED   — tool has side effects and no policy pattern matched; default deny.
    Policy-level codes (always carry the `field` that failed):
      UNTRUSTED_SOURCE    — argument must come from the user/trusted sources (`is_trusted` is False).
      READERS_CANNOT_READ — argument is not readable by the recipients/participants/file readers.
      NOT_PUBLIC          — argument must be public (`is_public` is False).
      FILE_NOT_SHARED     — an attached drive file is not shared with every recipient.
    """

    DEPENDS_ON_PRIVATE = "depends_on_private"
    NO_POLICY_MATCHED = "no_policy_matched"
    UNTRUSTED_SOURCE = "untrusted_source"
    READERS_CANNOT_READ = "readers_cannot_read"
    NOT_PUBLIC = "not_public"
    FILE_NOT_SHARED = "file_not_shared"

    def __str__(self) -> str:
        return self.value


@dataclasses.dataclass(frozen=True)
class Denied:
    """Denied access to the tool."""

    reason: str
    """Reason for denial. Free text; may contain raw values. Never shown to the P-LLM in live mode (DEF-167)
    and never written to the rejection log (DEF-160) — the structured fields below are what gets recorded."""
    code: DenialCode | None = None
    """Which predicate failed."""
    field: str | None = None
    """Which argument failed the predicate. `None` for engine-level denials."""
    policy: str | None = None
    """Name of the policy (tool-name pattern) that denied. Filled in by the engine if the policy left it unset."""


SecurityPolicyResult = Allowed | Denied


class SecurityPolicy(Protocol):
    def __call__(
        self,
        tool_name: str,
        kwargs: Mapping[str, CaMeLValue],
    ) -> SecurityPolicyResult: ...


NO_SIDE_EFFECT_TOOLS = {
    # Query AI assistant function
    "query_ai_assistant",
}


def base_security_policy(
    tool_name: str,
    kwargs: Mapping[str, CaMeLValue],
    no_side_effect_tools: set[str],
) -> SecurityPolicyResult:
    """Base security policy. If *anything* in the arguments is *not* public, then access is denied."""
    r = [get_all_readers(data) for data in kwargs.values()]
    if any(reader != readers.Public() for reader in r) and tool_name not in no_side_effect_tools:
        failed = next((name for name, data in kwargs.items() if get_all_readers(data)[0] != readers.Public()), None)
        return Denied("Data is not public.", code=DenialCode.NOT_PUBLIC, field=failed, policy="base")
    return Allowed()


class SecurityPolicyDeniedError(Exception): ...


class SecurityPolicyEngine(Protocol):
    policies: list[tuple[str, SecurityPolicy]]
    no_side_effect_tools: set[str]

    def check_policy(
        self,
        tool_name: str,
        kwargs: Mapping[str, CaMeLValue],
        dependencies: Iterable[CaMeLValue],
    ) -> SecurityPolicyResult:
        """Checks if the tool is allowed to be executed with the given data.

        Policies in `POLICIES` are evaluated in order. If any evaluates to Allowed(),
        then the tool is executed.
        """
        if tool_name in self.no_side_effect_tools:
            return Allowed()
        non_public_variables = [d.raw for d in dependencies if not is_public(d)]
        if len(non_public_variables) > 0:
            return Denied(
                f"{tool_name} is state-changing and depends on private values {non_public_variables}.",
                code=DenialCode.DEPENDS_ON_PRIVATE,
                policy="engine.dependencies",
            )
        for policy_name, policy in self.policies:
            if fnmatch.fnmatch(tool_name, policy_name):
                res = policy(tool_name, kwargs)
                if isinstance(res, Denied) and res.policy is None:
                    res = dataclasses.replace(res, policy=policy_name)
                return res
        return Denied(
            "No security policy matched for tool. Defaulting to denial.",
            code=DenialCode.NO_POLICY_MATCHED,
            policy="engine.default",
        )


class NoSecurityPolicyEngine(SecurityPolicyEngine):
    def __init__(self) -> None:
        self.policies = []
        self.no_side_effect_tools = set()

    def check_policy(
        self,
        tool_name: str,
        kwargs: Mapping[str, CaMeLValue],
        dependencies: Iterable[CaMeLValue],
    ) -> SecurityPolicyResult:
        return Allowed()
