"""Ask the configured model what it is, so the MODEL_NAMES entry in src/camel/models.py matches.

AgentDojo's important_instructions attack addresses the model by this name. A mismatch weakens the
attack and inflates security numbers, so check it once per model before running anything.

Usage:  uv run --env-file .env check_model_name.py openai:gpt-oss-120b
"""

import os
import sys

import openai

model = sys.argv[1].split(":", 1)[1] if len(sys.argv) > 1 else "gpt-oss-120b"
client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])  # honors OPENAI_BASE_URL
print(f"base_url = {client.base_url}")

r = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "What model are you? Answer in at most ten words."}],
    temperature=0.0,
    logprobs=True,
    top_logprobs=1,
)
choice = r.choices[0]
print(f"model    = {r.model}")
print(f"answer   = {choice.message.content!r}")
lp = choice.logprobs
n = len(lp.content) if lp and lp.content else 0
mean = sum(t.logprob for t in lp.content) / n if n else float("nan")
print(f"logprobs = {'YES' if n else 'NO'} ({n} content tokens, mean logprob {mean:.3f})")
print()
print("Set src/camel/models.py::_supported_model_names[<model>] to the name the model uses for itself above.")
