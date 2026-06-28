"""MiniMax planner via the OpenAI SDK using bounded tool-calling, with a heuristic fallback.

MiniMax drives each iteration through three tools (see ``agent/tools.py``): ``create_evals``
to declare the evaluation focus, ``add_training_examples`` to synthesize targeted training
data, and ``start_training_model`` to choose LoRA hyperparameters from fixed safe choice sets
and launch. If the model returns a legacy single-shot JSON plan instead, ``_parse_plan``
handles it; if anything fails, ``HeuristicPlanner`` produces a valid plan with no API calls.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

from pydantic import ValidationError

from evolora.agent.tools import TOOLS, coerce_hyperparams, extract_training_payload
from evolora.models.core import AgentPlan, EvalResult, LoraHyperparams, TrainingDataSpec

_TOOL_SYSTEM_PROMPT = """You are a LoRA fine-tuning strategist driving a bounded, auditable
self-improvement loop for a small model on a structured-JSON task. Improve the model by
calling these tools, in order:
  1. create_evals — state the criteria a correct answer must satisfy (call once, first).
  2. add_training_examples — synthesize targeted prompt/completion pairs for the observed
     failures as a single training_json object (call one or more times). Never copy the
     evaluation ground-truth answers.
  3. start_training_model — pick LoRA hyperparameters from the allowed values and launch
     (call exactly once, last).
Every training prompt's input must be SELF-CONTAINED: the model sees only the prompt. For SQL
goals, EVERY prompt MUST begin with a schema block of `CREATE TABLE` statements (columns + types)
for every table the completion references, then a blank line, then the request — do this for all
examples. For extraction/parsing goals, include the source record in the prompt.
Keep training data focused and de-duplicated. After start_training_model is called, stop."""

_EVAL_GEN_SYSTEM = """You create the evaluation set for fine-tuning a model on the user's task.
Call the create_evals tool with `criteria` (what a correct answer must satisfy) and
`eval_examples` — concrete {prompt, expected_output} pairs where expected_output is the exact
correct JSON object the model should produce. Make examples varied, realistic, and objectively
checkable.

CRITICAL: every prompt's input must be SELF-CONTAINED — the model sees only the prompt. For SQL
goals, EVERY prompt MUST begin with a schema block of one or more `CREATE TABLE` statements
(column names + types) for every table the query references, then a blank line, then the request.
Example prompt:
  CREATE TABLE customers (customer_id INT, name TEXT, age INT, city TEXT);
  CREATE TABLE orders (order_id INT, customer_id INT, amount DECIMAL);

  Write a query returning each customer's name and total order amount.
Do this for ALL examples, no exceptions. For extraction/parsing goals, include the source text or
record in the prompt. Never rely on context the prompt doesn't state."""

MINIMAX_TOOL_MAX_TOKENS = 8000
MINIMAX_EVAL_MAX_TOKENS = 8000
MINIMAX_MAX_TOOL_ROUNDS = 8
MINIMAX_EXAMPLE_BATCH_SIZE = 10


class MiniMaxPlanError(RuntimeError):
    """Raised when MiniMax cannot complete the bounded tool workflow."""


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks that some models emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json(raw: str):
    """Strip think-blocks and markdown fences, then parse JSON (list or object)."""
    text = _strip_think(raw)
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)


def _parse_plan(raw: str) -> AgentPlan:
    data: dict[str, Any] = _extract_json(raw)
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
        self.last_error = ""

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
        goal: str = "",
        example_batch_size: int = MINIMAX_EXAMPLE_BATCH_SIZE,
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
            "user_goal": goal or None,
            "requested_training_sample_count": training_sample_count,
            "instruction": (
                "Use create_evals, then add_training_examples with a training_json object, "
                "then start_training_model. "
                "If user_goal is provided, tailor the eval criteria and training examples toward "
                "it (while keeping outputs as strict JSON). "
                "If requested_training_sample_count is not null, add exactly that many training "
                "examples. Use multiple add_training_examples calls with at most "
                f"{example_batch_size} examples per call. If it is null, choose a sensible "
                "number yourself. "
                "Every training completion must be a valid JSON object string. For SQL tasks, "
                'wrap the answer as {"sql": "<query>"}. '
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
        goal: str = "",
    ) -> tuple[AgentPlan, bool]:
        """Return (plan, fallback_used). Drives the three tools; falls back on any failure."""
        errors: list[str] = []
        for batch_size in (MINIMAX_EXAMPLE_BATCH_SIZE, 5):
            try:
                plan = await self._plan_with_tools(
                    iteration,
                    baseline_score,
                    current_score,
                    failures,
                    training_sample_count,
                    goal,
                    batch_size,
                )
                self.last_error = ""
                return plan
            except (
                MiniMaxPlanError,
                json.JSONDecodeError,
                ValidationError,
                ValueError,
                KeyError,
                IndexError,
                AttributeError,
            ) as exc:
                errors.append(f"{exc.__class__.__name__}: {exc}")
                continue
            except Exception as exc:
                errors.append(f"{exc.__class__.__name__}: {exc}")
                continue

        self.last_error = " | ".join(errors[-2:]) if errors else "unknown MiniMax planner failure"
        fallback = HeuristicPlanner().plan(
            iteration, baseline_score, current_score, failures, training_sample_count, goal
        )
        return fallback, True

    async def _plan_with_tools(
        self,
        iteration: int,
        baseline_score: float,
        current_score: float,
        failures: list[EvalResult],
        training_sample_count: int | None,
        goal: str,
        example_batch_size: int,
    ) -> tuple[AgentPlan, bool]:
        client = self._make_client()
        user_prompt = self._build_user_prompt(
            iteration,
            baseline_score,
            current_score,
            failures,
            training_sample_count,
            goal,
            example_batch_size,
        )
        messages: list[dict] = [
            {"role": "system", "content": _TOOL_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        criteria: list[str] = []
        examples: list[dict[str, str]] = []
        rationale_bits: list[str] = []
        hyperparams: dict | None = None

        for _round in range(MINIMAX_MAX_TOOL_ROUNDS):  # bounded tool-calling turns
            resp = await client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.3,
                max_tokens=MINIMAX_TOOL_MAX_TOKENS,
            )
            choice = resp.choices[0]
            msg = choice.message
            tool_calls = msg.tool_calls or []

            if not tool_calls:
                # Model answered without tools — accept a legacy single-shot JSON plan.
                content = (msg.content or "").strip()
                if content:
                    return _parse_plan(content), False
                break

            assistant_tool_calls = []
            parsed_calls = []
            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    finish = getattr(choice, "finish_reason", "")
                    raise MiniMaxPlanError(
                        f"MiniMax returned invalid JSON for {tc.function.name} "
                        f"(finish_reason={finish})"
                    ) from exc
                assistant_tool_calls.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )
                parsed_calls.append((tc, args))

            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": assistant_tool_calls,
            })

            done = False
            for tc, args in parsed_calls:
                name = tc.function.name

                if name == "create_evals":
                    criteria = [str(c) for c in args.get("criteria", [])][:10]
                    result = f"Recorded {len(criteria)} eval criteria."
                elif name == "add_training_examples":
                    accepted, rationale = extract_training_payload(args)
                    examples.extend(accepted)
                    if rationale:
                        rationale_bits.append(rationale)
                    remaining = (
                        max(training_sample_count - len(examples), 0)
                        if training_sample_count is not None
                        else "agent choice"
                    )
                    result = (
                        f"Accepted {len(accepted)} examples ({len(examples)} total). "
                        f"Remaining requested examples: {remaining}. "
                        f"Continue with at most {example_batch_size} examples per call."
                    )
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
            raise MiniMaxPlanError("agent did not produce a complete tool-driven plan")

        plan = _assemble_plan(
            iteration, hyperparams, examples, criteria, rationale_bits, training_sample_count
        )
        return plan, False

    async def generate_evals(self, goal: str, count: int = 5) -> list[dict]:
        """Have MiniMax CALL the create_evals tool to produce an objective eval set.

        Returns a list of {"prompt": str, "expected": dict}. Returns [] on any
        failure so the orchestrator can fall back to its default eval set.
        """
        client = self._make_client()
        create_evals_tools = [t for t in TOOLS if t["function"]["name"] == "create_evals"]
        # A fresh seed per call nudges MiniMax to produce a NEW, diverse eval set each run
        # instead of converging on the same canonical examples (low temp made it look static).
        variation_seed = random.randint(1, 9_999_999)
        user_prompt = json.dumps({
            "goal": goal,
            "count": count,
            "variation_seed": variation_seed,
            "instruction": (
                f"Call create_evals with criteria and exactly {count} eval_examples for this "
                "goal. Generate a FRESH, DIVERSE set each time: vary the scenarios, difficulty, "
                "and edge cases, and do NOT reuse a fixed canonical set — use variation_seed "
                f"{variation_seed} to diversify. Each expected_output must be the single correct "
                "JSON output for its prompt."
            ),
        })
        try:
            resp = await client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _EVAL_GEN_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                tools=create_evals_tools,
                tool_choice={"type": "function", "function": {"name": "create_evals"}},
                temperature=0.9,
                max_tokens=MINIMAX_EVAL_MAX_TOKENS,
            )
            tool_calls = resp.choices[0].message.tool_calls or []
            args = json.loads(tool_calls[0].function.arguments or "{}") if tool_calls else {}
        except Exception:
            return []

        evals: list[dict] = []
        for item in args.get("eval_examples", []):
            if not isinstance(item, dict):
                continue
            prompt = item.get("prompt")
            expected = item.get("expected_output")
            if prompt and isinstance(expected, dict):
                evals.append({"prompt": str(prompt), "expected": expected})
            if len(evals) >= count:
                break
        return evals


class HeuristicPlanner:
    """Rule-based fallback — no API calls required."""

    def plan(
        self,
        iteration: int,
        baseline_score: float,
        current_score: float,
        failures: list[EvalResult],
        training_sample_count: int | None = None,
        goal: str = "",
    ) -> AgentPlan:
        r = min(64, 8 * (2 ** min(iteration - 1, 2)))
        lr = max(5e-5, 2e-4 / (iteration + 1))
        example_count = training_sample_count or min(5 + iteration * 2, 20)
        goal_note = f" for goal: {goal}" if goal else ""

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
            rationale=f"Heuristic plan for iteration {iteration} (MiniMax unavailable){goal_note}",
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
