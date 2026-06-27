"""MiniMax planner via Anthropic SDK with validated JSON output and heuristic fallback."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from evolora.models.core import AgentPlan, EvalResult, LoraHyperparams, TrainingDataSpec

_SYSTEM_PROMPT = """You are a LoRA fine-tuning strategist.
Given evaluation failures, propose targeted training data and hyperparameters.
Respond ONLY with valid JSON matching this schema (no markdown, no explanation):
{
  "hyperparams": {
    "r": <int 1-64, power of two>,
    "lora_alpha": <int 1-256>,
    "lora_dropout": <float 0-0.5>,
    "learning_rate": <float 0-0.1>,
    "num_epochs": <int 1-5>,
    "batch_size": <int 1-32>,
    "warmup_steps": <int 0-500>,
    "weight_decay": <float 0-0.5>
  },
  "data_spec": {
    "examples": [{"prompt": "...", "completion": "..."}],
    "rationale": "...",
    "max_examples": <int 1-200>
  },
  "rationale": "...",
  "focus_areas": ["..."]
}"""


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks that some models emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _parse_plan(raw: str) -> AgentPlan:
    text = _strip_think(raw)
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    data: dict[str, Any] = json.loads(text)
    return AgentPlan(**data)


class MiniMaxPlanner:
    """Calls MiniMax via the Anthropic SDK. Falls back to HeuristicPlanner on failure."""

    def __init__(self, api_key: str, model: str, base_url: str) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url

    def _make_client(self):
        import anthropic

        return anthropic.AsyncAnthropic(
            api_key=self._api_key,
            base_url=self._base_url,
        )

    def _build_user_prompt(
        self,
        iteration: int,
        baseline_score: float,
        current_score: float,
        failures: list[EvalResult],
    ) -> str:
        failure_summary = [
            {"sample_id": f.sample_id, "score": f.score, "details": f.details}
            for f in failures[:10]  # cap context — never include expected answers
        ]
        return json.dumps({
            "iteration": iteration,
            "baseline_score": baseline_score,
            "current_score": current_score,
            "failure_count": len(failures),
            "sample_failures": failure_summary,
            "task": "structured customer spending summary (JSON output)",
            "instruction": (
                "Propose training data and hyperparameters to improve the model. "
                "Do NOT include expected answers in your plan."
            ),
        })

    async def plan(
        self,
        iteration: int,
        baseline_score: float,
        current_score: float,
        failures: list[EvalResult],
    ) -> tuple[AgentPlan, bool]:
        """Return (plan, fallback_used). Falls back to HeuristicPlanner on failure."""
        client = self._make_client()
        user_prompt = self._build_user_prompt(iteration, baseline_score, current_score, failures)

        for attempt in range(3):
            try:
                resp = await client.messages.create(
                    model=self._model,
                    system=_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=0.3,
                    max_tokens=1500,
                )
                raw = resp.content[0].text if resp.content else ""
                plan = _parse_plan(raw)
                return plan, False
            except (json.JSONDecodeError, ValidationError, KeyError, IndexError):
                if attempt == 2:
                    break
            except Exception:
                break

        fallback = HeuristicPlanner().plan(iteration, baseline_score, current_score, failures)
        return fallback, True


class HeuristicPlanner:
    """Rule-based fallback — no API calls required."""

    def plan(
        self,
        iteration: int,
        baseline_score: float,
        current_score: float,
        failures: list[EvalResult],
    ) -> AgentPlan:
        r = min(64, 8 * (2 ** min(iteration - 1, 2)))
        lr = max(5e-5, 2e-4 / (iteration + 1))

        examples = [
            {
                "prompt": (
                    'Customers: [{"name":"Alice","purchases":[100,200]}, '
                    '{"name":"Bob","purchases":[50]}]. Summarize.'
                ),
                "completion": (
                    '{"top_customer":"Alice","top_customer_total":300,'
                    '"customer_count":2,"total_revenue":350,'
                    '"summary":"Alice leads with $300 in purchases."}'
                ),
            }
            for _ in range(min(5 + iteration * 2, 20))
        ]

        return AgentPlan(
            hyperparams=LoraHyperparams(r=r, lora_alpha=r * 2, learning_rate=lr),
            data_spec=TrainingDataSpec(examples=examples, rationale="Heuristic fallback plan"),
            rationale=f"Heuristic plan for iteration {iteration} (MiniMax unavailable)",
            focus_areas=["json_format", "field_accuracy"],
        )


def get_planner(
    use_minimax: bool,
    api_key: str = "",
    model: str = "MiniMax-M2.7-highspeed",
    base_url: str = "https://api.minimax.io/anthropic",
):
    if use_minimax and api_key:
        return MiniMaxPlanner(api_key=api_key, model=model, base_url=base_url)
    return HeuristicPlanner()
