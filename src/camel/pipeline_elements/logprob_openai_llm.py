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

"""OpenAI LLM element that also returns token log-probabilities (P-DEF-12 / DEF-160, `conf_t`).

Behaves exactly like `agentdojo.agent_pipeline.OpenAILLM` — same message conversion, same retry policy —
plus `logprobs=True` on the request. After each call it writes into `extra_args`:

  conf_t          exp(mean token logprob) over the scored span, in (0, 1]; None if the provider returned none
  conf_t_scope    "code_block" if a ```fenced``` block was found and scored, else "content"
  conf_t_n_tokens number of tokens scored
  conf_t_n_excluded number of sentinel (-9999.0) tokens dropped from the mean

The span is the *first* fenced code block in the assistant content (what the interpreter will run). Reasoning
models (o-series) do not return logprobs; for them `conf_t` is None and the request does not ask for them.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

import openai
from agentdojo.agent_pipeline.llms import openai_llm as _oai
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage
from openai import NOT_GIVEN
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionReasoningEffort,
    ChatCompletionToolParam,
)
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_random_exponential

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*[ \t]*\r?\n(.*?)```", re.S)
# OpenAI reports tokens it cannot score as logprob == -9999.0. One such token in a 150-token block drives
# exp(mean) to ~1e-30, so they are excluded from the mean and counted separately.
SENTINEL_LOGPROB = -9990.0


@retry(
    wait=wait_random_exponential(multiplier=1, max=40),
    stop=stop_after_attempt(3),
    reraise=True,
    retry=retry_if_not_exception_type((openai.BadRequestError, openai.UnprocessableEntityError)),
)
def _chat_completion_request_with_logprobs(
    client: openai.OpenAI,
    model: str,
    messages: Sequence[ChatCompletionMessageParam],
    tools: Sequence[ChatCompletionToolParam],
    reasoning_effort: ChatCompletionReasoningEffort | None,
    temperature: float | None,
    logprobs: bool,
):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools or NOT_GIVEN,
        tool_choice="auto" if tools else NOT_GIVEN,
        temperature=temperature or NOT_GIVEN,
        reasoning_effort=reasoning_effort or NOT_GIVEN,
        logprobs=True if logprobs else NOT_GIVEN,
    )


def score_tokens(tokens: Sequence[str], logprobs: Sequence[float]) -> tuple[float | None, str, int, int]:
    """Returns (conf_t, scope, n_scored, n_excluded). conf_t = exp(mean logprob) over the first fenced code
    block if one exists in the concatenated tokens, else over all content tokens. Sentinel logprobs
    (<= SENTINEL_LOGPROB) are excluded from the mean and reported in n_excluded. None when nothing to score."""
    if not tokens:
        return None, "content", 0, 0
    text = "".join(tokens)
    m = _FENCE_RE.search(text)
    if m is None:
        chosen = list(logprobs)
        scope = "content"
    else:
        start, end = m.span(1)
        chosen = []
        pos = 0
        for tok, lp in zip(tokens, logprobs, strict=True):
            tok_start, tok_end = pos, pos + len(tok)
            pos = tok_end
            if tok_end <= start or tok_start >= end:
                continue
            chosen.append(lp)
        scope = "code_block"
        if not chosen:  # degenerate fence (empty body)
            chosen = list(logprobs)
            scope = "content"
    scored = [lp for lp in chosen if lp > SENTINEL_LOGPROB]
    n_excluded = len(chosen) - len(scored)
    if not scored:
        return None, scope, 0, n_excluded
    mean_lp = sum(scored) / len(scored)
    return math.exp(mean_lp), scope, len(scored), n_excluded


class LogprobOpenAILLM(_oai.OpenAILLM):
    """`OpenAILLM` that requests logprobs and reports `conf_t` through `extra_args`."""

    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        reasoning_effort: ChatCompletionReasoningEffort | None = None,
        temperature: float | None = 0.0,
        request_logprobs: bool | None = None,
    ) -> None:
        super().__init__(client, model, reasoning_effort, temperature)
        # o-series models reject `logprobs`; default to asking only for non-reasoning models.
        self.request_logprobs = (not _oai._is_reasoning_model(model)) if request_logprobs is None else request_logprobs

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        openai_messages = [_oai._message_to_openai(message, self.model) for message in messages]
        openai_tools = [_oai._function_to_openai(tool) for tool in runtime.functions.values()]
        completion = _chat_completion_request_with_logprobs(
            self.client,
            self.model,
            openai_messages,
            openai_tools,
            self.reasoning_effort,
            self.temperature,
            self.request_logprobs,
        )
        choice = completion.choices[0]
        output = _oai._openai_to_assistant_message(choice.message)
        conf_t, scope, n, n_excluded = None, "content", 0, 0
        if choice.logprobs is not None and choice.logprobs.content:
            conf_t, scope, n, n_excluded = score_tokens(
                [t.token for t in choice.logprobs.content], [t.logprob for t in choice.logprobs.content]
            )
        extra_args["conf_t"] = conf_t
        extra_args["conf_t_scope"] = scope
        extra_args["conf_t_n_tokens"] = n
        extra_args["conf_t_n_excluded"] = n_excluded
        messages = [*messages, output]
        return query, runtime, env, messages, extra_args
