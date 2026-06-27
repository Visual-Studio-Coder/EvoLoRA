"""MiniMax planner via the OpenAI SDK using bounded tool-calling, with a heuristic fallback.

MiniMax drives each iteration through three tools (see ``agent/tools.py``): ``create_evals``
to declare the evaluation focus, ``add_training_examples`` to synthesize targeted training
data, and ``start_training_model`` to choose LoRA hyperparameters from fixed safe choice sets
and launch. If the model returns a legacy single-shot JSON plan instead, ``_parse_plan``
handles it; if anything fails, ``HeuristicPlanner`` produces a valid plan with no API calls.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from evolora.agent.tools import TOOLS, coerce_hyperparams
from evolora.models.core import AgentPlan, EvalResult, LoraHyperparams, TrainingDataSpec

_TOOL_SYSTEM_PROMPT = """You are a LoRA fine-tuning strategist driving a bounded, auditable
self-improvement loop for a small model on a structured-JSON task. Improve the model by
calling these tools, in order:
  1. create_evals — state the criteria a correct answer must satisfy (call once, first).
  2. add_training_examples — synthesize targeted prompt/completion pairs for the observed
     failures (call one or more times). Never copy the evaluation ground-truth answers.
  3. start_training_model — pick LoRA hyperparameters from the allowed values and launch
     (call exactly once, last).
Keep training data focused and de-duplicated. After start_training_model is called, stop."""


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


def _assemble_plan(
    iteration: int,
    hyperparams: dict,
    examples: list[dict[str, str]],
    criteria: list[str],
    rationale_bits: list[str],
    training_sample_count: int | None,
) -> AgentPlan:
    """Build a validated AgentPlan from the accumulated tool calls."""
    max_examples = training_sample_count or max(1, min(len(examples), 500))
    detail = " | ".join(b for b in rationale_bits if b.strip())
    trace = (
        f"create_evals({len(criteria)}) -> add_training_examples({len(examples)}) -> "
        f"start_training_model(r={hyperparams['r']}, alpha={hyperparams['lora_alpha']}, "
        f"lr={hyperparams['learning_rate']:g}, epochs={hyperparams['num_epochs']}, "
        f"batch={hyperparams['batch_size']})"
    )
    rationale = f"{trace} :: {detail}" if detail else trace
    return AgentPlan(
        hyperparams=LoraHyperparams(**hyperparams),
        data_spec=TrainingDataSpec(
            examples=examples,
            rationale=detail or "MiniMax tool-driven training data",
            max_examples=max_examples,
        ),
        rationale=rationale,
        focus_areas=criteria[:5] or ["json_format", "field_accuracy"],
    )


class MiniMaxPlanner:
    """Drives MiniMax through the three bounded tools. Falls back to HeuristicPlanner on failure."""

    def __init__(self, api_key: str, model: str, base_url: str) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url

    def _make_client(self):
        from openai import AsyncOpenAI

        return AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
        )

    def _build_user_prompt(
        self,
        iteration: int,
        baseline_score: float,
        current_score: float,
        failures: list[EvalResult],
        training_sample_count: int | None = None,
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
            "requested_training_sample_count": training_sample_count,
            "instruction": (
                "Use create_evals, then add_training_examples, then start_training_model. "
                "If requested_training_sample_count is not null, add exactly that many training "
                "examples. If it is null, choose a sensible number yourself. "
                "Do NOT include expected answers in your training data."
            ),
        })

    async def plan(
        self,
        iteration: int,
        baseline_score: float,
        current_score: float,
        failures: list[EvalResult],
        training_sample_count: int | None = None,
    ) -> tuple[AgentPlan, bool]:
        """Return (plan, fallback_used). Drives the three tools; falls back on any failure."""
        client = self._make_client()
        user_prompt = self._build_user_prompt(
            iteration, baseline_score, current_score, failures, training_sample_count
        )
        messages: list[dict] = [
            {"role": "system", "content": _TOOL_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        criteria: list[str] = []
        examples: list[dict[str, str]] = []
        rationale_bits: list[str] = []
        hyperparams: dict | None = None

        try:
            for _round in range(6):  # bounded tool-calling turns
                resp = await client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0.3,
                    max_tokens=2000,
                )
                msg = resp.choices[0].message
                tool_calls = msg.tool_calls or []

                if not tool_calls:
                    # Model answered without tools — accept a legacy single-shot JSON plan.
                    content = (msg.content or "").strip()
                    if content:
                        return _parse_plan(content), False
                    break

                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                })

                done = False
                for tc in tool_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}

                    if name == "create_evals":
                        criteria = [str(c) for c in args.get("criteria", [])][:10]
                        result = f"Recorded {len(criteria)} eval criteria."
                    elif name == "add_training_examples":
                        accepted = [
                            {"prompt": str(e["prompt"]), "completion": str(e["completion"])}
                            for e in args.get("examples", [])
                            if isinstance(e, dict) and e.get("prompt") and e.get("completion")
                        ]
                        examples.extend(accepted)
                        if args.get("rationale"):
                            rationale_bits.append(str(args["rationale"]))
                        result = f"Accepted {len(accepted)} examples ({len(examples)} total)."
                    elif name == "start_training_model":
                        hyperparams = coerce_hyperparams(args)
                        result = f"Training launched with {hyperparams}."
                        done = True
                    else:
                        result = f"Unknown tool {name!r} ignored."

                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

                if done:
                    break

            if hyperparams is None or not examples:
                raise ValueError("agent did not produce a complete tool-driven plan")

            plan = _assemble_plan(
                iteration, hyperparams, examples, criteria, rationale_bits, training_sample_count
            )
            return plan, False
        except (json.JSONDecodeError, ValidationError, ValueError, KeyError, IndexError, AttributeError):
            pass
        except Exception:
            pass

        fallback = HeuristicPlanner().plan(
            iteration, baseline_score, current_score, failures, training_sample_count
        )
        return fallback, True


class HeuristicPlanner:
    """Rule-based fallback — no API calls required."""

    def plan(
        self,
        iteration: int,
        baseline_score: float,
        current_score: float,
        failures: list[EvalResult],
        training_sample_count: int | None = None,
    ) -> AgentPlan:
        r = min(64, 8 * (2 ** min(iteration - 1, 2)))
        lr = max(5e-5, 2e-4 / (iteration + 1))
        example_count = training_sample_count or min(5 + iteration * 2, 20)

        examples = [
            {
                "prompt": (
                    f'Customers: [{{"name":"Alice","purchases":[{100 + i},{200 + i}]}}, '
                    f'{{"name":"Bob","purchases":[{50 + i}]}}]. Summarize.'
                ),
                "completion": (
                    f'{{"top_customer":"Alice","top_customer_total":{300 + (2 * i)},'
                    f'"customer_count":2,"total_revenue":{350 + (3 * i)},'
                    f'"summary":"Alice leads with ${300 + (2 * i)} in purchases."}}'
                ),
            }
            for i in range(example_count)
        ]

        return AgentPlan(
            hyperparams=LoraHyperparams(r=r, lora_alpha=r * 2, learning_rate=lr),
            data_spec=TrainingDataSpec(
                examples=examples,
                rationale="Heuristic fallback plan",
                max_examples=example_count,
            ),
            rationale=f"Heuristic plan for iteration {iteration} (MiniMax unavailable)",
            focus_areas=["json_format", "field_accuracy"],
        )


def get_planner(
    use_minimax: bool,
    api_key: str = "",
    model: str = "MiniMax-M2.7-highspeed",
    base_url: str = "https://api.minimax.io/v1",
):
    if use_minimax and api_key:
        return MiniMaxPlanner(api_key=api_key, model=model, base_url=base_url)
    return HeuristicPlanner()
