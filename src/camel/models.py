import os
import time
from pathlib import Path

import anthropic
import openai
from agentdojo import agent_pipeline, functions_runtime
from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.models import MODEL_NAMES
from google import genai
from openai.types.chat import ChatCompletionReasoningEffort
from pydantic_ai.models import KnownModelName

from camel.interpreter.interpreter import MetadataEvalMode
from camel.pipeline_elements.anthropic_tool_filter import AnthropicLLMToolFilter
from camel.pipeline_elements.logprob_openai_llm import LogprobOpenAILLM
from camel.pipeline_elements.privileged_llm import PrivilegedLLM
from camel.pipeline_elements.replay_privileged_llm import PrivilegedLLMReplayer, UserInjectionTasksGetter
from camel.pipeline_elements.security_policies import (
    ADNoSecurityPolicyEngine,
    AgentDojoSecurityPolicyEngine,
    BankingSecurityPolicyEngine,
    SlackSecurityPolicyEngine,
    TravelSecurityPolicyEngine,
    WorkspaceSecurityPolicyEngine,
)
from camel.rejection_log import RejectionLogger

_thinking_efforts = ["low", "medium", "high"]
_oai_thinking_models = {
    "o4-mini-2025-04-16": "ChatGPT",
    "o3-2025-04-16": "ChatGPT",
    "o3-mini-2025-01-31": "ChatGPT",
    "o1-2024-12-17": "ChatGPT",
    "codex-mini-latest": "ChatGPT",
}
_oai_thinking_models_with_effort = {
    f"{model}-{effort}": name for model, name in _oai_thinking_models.items() for effort in _thinking_efforts
}
_supported_model_names = {
    "gemini-2.5-flash-preview-05-20": "AI model developed by Google",
    "gemini-2.5-pro-preview-06-05": "AI model developed by Google",
    "gemini-2.0-flash-lite-001": "AI model developed by Google",
    "claude-3-5-haiku-20241022": "Claude",
    "claude-3-5-sonnet-20241022": "Claude",
    "claude-3-7-sonnet-20250219": "Claude",
    "claude-sonnet-4-20250514": "Claude",
    "claude-opus-4-20250514": "Claude",
    "gpt-4o-2024-08-06": "GPT-4",
    "gpt-4o-mini-2024-07-18": "GPT-4",
    "gpt-4.1-2025-04-14": "ChatGPT",
    "gpt-4.1-nano-2025-04-14": "ChatGPT",
    # --- P-DEF-12 additions: served via an OpenAI-compatible endpoint (Cerebras), set OPENAI_BASE_URL.
    # The value is the model's self-identification, used verbatim by AgentDojo's important_instructions
    # attack ("...to you, {name}..."). Verify with `uv run check_model_name.py` before the first run.
    "gpt-oss-120b": "ChatGPT",
    "qwen-3.8-27b": "Qwen",
} | _oai_thinking_models_with_effort
suffixes = ["", "+camel", "+camel+secpol", "+camel+secpol+strict"]

CAMEL_MODEL_NAMES = {f"{model}{suffix}": name for model, name in _supported_model_names.items() for suffix in suffixes}

_SECURITY_POLICY_ENGINES: dict[str, type[AgentDojoSecurityPolicyEngine]] = {
    "workspace": WorkspaceSecurityPolicyEngine,
    "travel": TravelSecurityPolicyEngine,
    "banking": BankingSecurityPolicyEngine,
    "slack": SlackSecurityPolicyEngine,
}


def _is_oai_reasoning_model(model: str) -> bool:
    return "o4" in model or "o3" in model or "o1" in model or "codex" in model


def make_rejection_logger(logdir: Path | None, pipeline_name: str) -> RejectionLogger | None:
    """DEF-160: one JSONL file per (run, pipeline). `None` when no logdir is given (tests, ad-hoc use).

    run_id defaults to the resolved log directory name (`logs_run1`, ...) so the symlink dance in
    `reproduce_baseline.sh` still yields distinct ids. Override with CAMEL_RUN_ID; CAMEL_EPOCH / CAMEL_SEED
    are read for the record fields (OpenAI accepts no seed, so CAMEL_SEED is normally unset → null).
    """
    if logdir is None:
        return None
    run_id = os.getenv("CAMEL_RUN_ID") or Path(logdir).resolve().name
    seed_env = os.getenv("CAMEL_SEED")
    return RejectionLogger(
        Path(logdir) / "rejection_log" / f"{pipeline_name}.jsonl",
        run_id=run_id,
        arm=pipeline_name,
        seed=int(seed_env) if seed_env else None,
        epoch=int(os.getenv("CAMEL_EPOCH", "0")),
    )


MODEL_NAMES.update(CAMEL_MODEL_NAMES)


class Sleep(agent_pipeline.BasePipelineElement):
    def __init__(self, amount: int) -> None:
        super().__init__()
        self._amount = amount

    def query(
        self,
        query: str,
        runtime,
        env=functions_runtime.EmptyEnv(),
        messages=[],
        extra_args={},
    ) -> tuple:
        if self._amount > 0:
            time.sleep(self._amount)
        return query, runtime, env, messages, extra_args


def make_tools_pipeline(
    model: KnownModelName,
    use_original: bool,
    replay_with_policies: bool,
    attack_name: str,
    reasoning_effort: ChatCompletionReasoningEffort,
    thinking_budget_tokens: int | None,
    suite: str,
    ad_defense: str | None,
    eval_mode: MetadataEvalMode,
    q_llm: KnownModelName | None,
    logdir: Path | None = None,
) -> agent_pipeline.AgentPipeline:
    if "google" in model:
        # vertexai.init(project=os.getenv("GCP_PROJECT"), location=os.getenv("GCP_LOCATION"))
        # llm = GoogleLLM(model.split(":")[1])
        # client = genai.Client(vertexai=True, project=os.getenv("GCP_PROJECT"), location=os.getenv("GCP_LOCATION"))
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        if model == "google:gemini-2.0-flash-lite-001":
            max_tokens = 8192
        else:
            max_tokens = 65535
        llm = agent_pipeline.GoogleLLM(model.split(":")[1], client, max_tokens=max_tokens)
    elif "openai" in model:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # reasoning models do not support temperature and their "system" message is called "developer" message.
        # LogprobOpenAILLM == agentdojo's OpenAILLM + logprobs (conf_t, DEF-160); it skips logprobs for o-series.
        if _is_oai_reasoning_model(model):
            llm = LogprobOpenAILLM(client, model.split(":")[1], reasoning_effort, None)
        else:
            llm = LogprobOpenAILLM(client, model.split(":")[1], None)
    elif "anthropic" in model:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        if thinking_budget_tokens:
            max_tokens = 8192 + thinking_budget_tokens
        else:
            max_tokens = 8192
        llm = agent_pipeline.AnthropicLLM(
            client, model.split(":")[1], thinking_budget_tokens=thinking_budget_tokens, max_tokens=max_tokens
        )
    else:
        raise ValueError("Invalid model")

    llm.name = model.split(":")[1]

    engine = _SECURITY_POLICY_ENGINES[suite]

    if use_original:
        if "exp" in model:
            print("Adding 'Sleep' pipeline element.")
            tools_loop = agent_pipeline.ToolsExecutionLoop([agent_pipeline.ToolsExecutor(), Sleep(6), llm])
        else:
            tools_loop = agent_pipeline.ToolsExecutionLoop([agent_pipeline.ToolsExecutor(), llm])

        tools_pipeline = agent_pipeline.AgentPipeline(
            [agent_pipeline.SystemMessage(load_system_message(None)), agent_pipeline.InitQuery(), llm, tools_loop]
        )

        if ad_defense == "tool_filter" and isinstance(llm, agent_pipeline.AnthropicLLM):
            tools_pipeline = agent_pipeline.AgentPipeline(
                [
                    agent_pipeline.SystemMessage(load_system_message(None)),
                    agent_pipeline.InitQuery(),
                    AnthropicLLMToolFilter(llm.client, llm.name),
                    llm,
                    tools_loop,
                ]
            )
        elif ad_defense is not None:
            tools_pipeline = agent_pipeline.AgentPipeline.from_config(
                agent_pipeline.PipelineConfig(
                    llm=llm, defense=ad_defense, system_message_name=None, system_message=None
                )
            )

        if _is_oai_reasoning_model(model):
            tools_pipeline.name = f"{model.split(':')[1]}-{reasoning_effort}"
        elif "anthropic" in model and (("3-7-sonnet" in model or "sonnet-4" in model) and thinking_budget_tokens):
            tools_pipeline.name = f"{model.split(':')[1]}-{thinking_budget_tokens}"
        else:
            tools_pipeline.name = model.split(":")[1]
        if ad_defense is not None:
            tools_pipeline.name += f"+{ad_defense}"

    elif replay_with_policies:
        if _is_oai_reasoning_model(model):
            pipeline_name = f"{model.split(':')[1]}-{reasoning_effort}+camel"
        else:
            pipeline_name = f"{model.split(':')[1]}+camel"
        # Used for logging
        if "openai" in model and _is_oai_reasoning_model(model):
            secpol_name = f"{model.split(':')[1]}-{reasoning_effort}+camel+secpol"
        elif "anthropic" in model and (("3-7-sonnet" in model or "sonnet-4" in model) and thinking_budget_tokens):
            secpol_name = f"{model.split(':')[1]}-{thinking_budget_tokens}+camel+secpol"
        else:
            secpol_name = f"{model.split(':')[1]}+camel+secpol"
        if eval_mode == MetadataEvalMode.STRICT:
            secpol_name += "+strict"
        rejection_logger = make_rejection_logger(logdir, secpol_name)
        tools_pipeline = agent_pipeline.AgentPipeline(
            [
                agent_pipeline.InitQuery(),
                UserInjectionTasksGetter(),
                PrivilegedLLMReplayer(pipeline_name, attack_name, engine, eval_mode, rejection_logger),
            ]
        )
        tools_pipeline.name = secpol_name
        tools_pipeline.rejection_logger = rejection_logger  # type: ignore[attr-defined]
    else:
        # Used for logging
        if "openai" in model and _is_oai_reasoning_model(model):
            camel_name = f"{model.split(':')[1]}-{reasoning_effort}+camel"
        elif "anthropic" in model and (("3-7-sonnet" in model or "sonnet-4" in model) and thinking_budget_tokens):
            camel_name = f"{model.split(':')[1]}-{thinking_budget_tokens}+camel"
        else:
            camel_name = f"{model.split(':')[1]}+camel"

        if q_llm:
            camel_name += f"-q:{q_llm.split(':')[1]}"
        rejection_logger = make_rejection_logger(logdir, camel_name)
        tools_pipeline = agent_pipeline.AgentPipeline(
            [
                # Adds the user query to the history
                agent_pipeline.InitQuery(),
                # DEF-160: puts suite / user task / injection task ids into extra_args for the logger
                UserInjectionTasksGetter(),
                # Generates the code and writes it to `extra_args`
                PrivilegedLLM(
                    llm,
                    ADNoSecurityPolicyEngine,
                    q_llm or model,
                    rejection_logger=rejection_logger,
                ),
            ]
        )
        tools_pipeline.name = camel_name
        tools_pipeline.rejection_logger = rejection_logger  # type: ignore[attr-defined]

    return tools_pipeline
