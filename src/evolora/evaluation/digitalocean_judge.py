"""DigitalOcean LLM-as-judge integration with a mock-safe fallback."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from evolora.models.core import AgentPlan, EvalResult, JudgeReport


class CandidateJudge(Protocol):
    @property
    def is_mock(self) -> bool: ...

    async def judge(
        self,
        *,
        goal: str,
        task_name: str,
        base_model_id: str,
        iteration: int,
        score: float,
        adaptive_score: float | None,
        plan: AgentPlan,
        eval_results: list[EvalResult],
        responses: dict[str, str],
    ) -> JudgeReport: ...


def _strip_json(raw: str) -> Any:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)


def _clip(value: Any, limit: int = 800) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _heuristic_report(
    *,
    score: float,
    adaptive_score: float | None,
    eval_results: list[EvalResult],
    source: str = "heuristic",
    fallback_note: str = "",
) -> JudgeReport:
    failed = [r for r in eval_results if not r.passed]
    rating = score if adaptive_score is None else (score * 0.75) + (adaptive_score * 0.25)
    weaknesses = [
        f"{r.sample_id}: {_clip(r.details, 180)}"
        for r in failed[:3]
    ] or ["No failed locked-eval samples detected."]
    summary = (
        f"{source} judge rates the candidate {rating:.2f}. "
        f"Locked score is {score:.2f}"
        + (f"; adaptive diagnostic is {adaptive_score:.2f}." if adaptive_score is not None else ".")
    )
    if fallback_note:
        summary += f" {fallback_note}"
    return JudgeReport(
        rating=max(0.0, min(1.0, rating)),
        summary=summary,
        strengths=["Objective eval is complete and auditable."],
        weaknesses=weaknesses,
        recommended_focus=[r.sample_id for r in failed[:5]] or ["preserve current behavior"],
        source=source,
        is_mock=True,
    )


class HeuristicJudge:
    """No-network judge used when DigitalOcean is not configured."""

    @property
    def is_mock(self) -> bool:
        return True

    async def judge(
        self,
        *,
        goal: str,
        task_name: str,
        base_model_id: str,
        iteration: int,
        score: float,
        adaptive_score: float | None,
        plan: AgentPlan,
        eval_results: list[EvalResult],
        responses: dict[str, str],
    ) -> JudgeReport:
        return _heuristic_report(
            score=score,
            adaptive_score=adaptive_score,
            eval_results=eval_results,
            source="heuristic_do_judge_fallback",
            fallback_note="DigitalOcean model access key is not configured.",
        )


class DigitalOceanJudge:
    """Calls DigitalOcean Inference with an OpenAI-compatible chat completion."""

    def __init__(self, *, api_key: str, base_url: str, model: str) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/") + "/"
        self._model = model

    @property
    def is_mock(self) -> bool:
        return False

    def _make_client(self):
        from openai import AsyncOpenAI

        return AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)

    async def judge(
        self,
        *,
        goal: str,
        task_name: str,
        base_model_id: str,
        iteration: int,
        score: float,
        adaptive_score: float | None,
        plan: AgentPlan,
        eval_results: list[EvalResult],
        responses: dict[str, str],
    ) -> JudgeReport:
        payload = {
            "specialization_goal": goal or task_name,
            "base_model_id": base_model_id,
            "iteration": iteration,
            "locked_score": score,
            "adaptive_score": adaptive_score,
            "planner_focus": plan.focus_areas,
            "planner_rationale": _clip(plan.rationale, 600),
            "eval_observations": [
                {
                    "sample_id": result.sample_id,
                    "objective_score": result.score,
                    "passed": result.passed,
                    "details": result.details,
                    "model_response": _clip(responses.get(result.sample_id, ""), 1000),
                }
                for result in eval_results[:8]
            ],
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are DigitalOcean Inference acting as an LLM-as-a-judge for an "
                    "auditable LoRA specialization loop. Judge whether the trained adapter "
                    "matches the specialization goal. Return ONLY JSON with keys: rating "
                    "(0..1), summary, strengths (array), weaknesses (array), recommended_focus "
                    "(array). Do not include markdown."
                ),
            },
            {"role": "user", "content": json.dumps(payload, sort_keys=True)},
        ]
        try:
            resp = await self._make_client().chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.1,
                max_completion_tokens=800,
            )
            parsed = _strip_json(resp.choices[0].message.content or "{}")
            if not isinstance(parsed, dict):
                raise ValueError("judge response was not a JSON object")
            return JudgeReport(
                rating=float(parsed.get("rating", score)),
                summary=str(parsed.get("summary", "")).strip() or "DigitalOcean judge completed.",
                strengths=[str(x) for x in parsed.get("strengths", [])][:5],
                weaknesses=[str(x) for x in parsed.get("weaknesses", [])][:5],
                recommended_focus=[str(x) for x in parsed.get("recommended_focus", [])][:5],
                source=f"digitalocean:{self._model}",
                is_mock=False,
            )
        except Exception as exc:
            return _heuristic_report(
                score=score,
                adaptive_score=adaptive_score,
                eval_results=eval_results,
                source="digitalocean_fallback",
                fallback_note=f"DigitalOcean judge failed safely: {exc.__class__.__name__}.",
            )


def get_judge(
    *,
    api_key: str = "",
    base_url: str = "https://inference.do-ai.run/v1/",
    model: str = "llama3.3-70b-instruct",
) -> CandidateJudge:
    if api_key:
        return DigitalOceanJudge(api_key=api_key, base_url=base_url, model=model)
    return HeuristicJudge()
