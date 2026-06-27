"""Gemini 2.5 Flash LLM-as-judge — drop-in alongside the DigitalOcean judge.

Add to .env:
    GOOGLE_API_KEY=your_key_here

The rest of the code stays untouched. Just swap get_judge() in config.py
to use get_gemini_judge() when GOOGLE_API_KEY is set.
"""

from __future__ import annotations

import json
import re
from typing import Any

from evolora.models.core import AgentPlan, EvalResult, JudgeReport


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
    fallback_note: str = "",
) -> JudgeReport:
    failed = [r for r in eval_results if not r.passed]
    rating = score if adaptive_score is None else (score * 0.75) + (adaptive_score * 0.25)
    weaknesses = [
        f"{r.sample_id}: {_clip(r.details, 180)}" for r in failed[:3]
    ] or ["No failed locked-eval samples detected."]
    summary = (
        f"gemini_fallback judge rates the candidate {rating:.2f}. "
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
        source="gemini_fallback",
        is_mock=True,
    )


class GeminiJudge:
    """Uses Gemini 2.5 Flash as the LLM judge via the new google-genai SDK."""

    def __init__(self, *, api_key: str, model: str = "gemini-2.5-flash") -> None:
        self._api_key = api_key
        self._model = model

    @property
    def is_mock(self) -> bool:
        return False

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
        import asyncio
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self._api_key)

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

        system_prompt = (
            "You are an expert LLM judge evaluating a LoRA fine-tuning specialization loop. "
            "Judge whether the trained adapter matches the specialization goal based on the "
            "eval observations provided. Return ONLY a valid JSON object with these keys: "
            "rating (float 0.0 to 1.0), summary (string), strengths (array of strings), "
            "weaknesses (array of strings), recommended_focus (array of strings). "
            "No markdown, no explanation outside the JSON."
        )

        user_message = (
            f"{system_prompt}\n\nEvaluation data:\n{json.dumps(payload, sort_keys=True)}"
        )

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=self._model,
                    contents=user_message,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=800,
                    ),
                ),
            )

            raw = response.text or "{}"
            parsed = _strip_json(raw)

            if not isinstance(parsed, dict):
                raise ValueError("Gemini judge response was not a JSON object")

            return JudgeReport(
                rating=float(parsed.get("rating", score)),
                summary=str(parsed.get("summary", "")).strip() or "Gemini judge completed.",
                strengths=[str(x) for x in parsed.get("strengths", [])][:5],
                weaknesses=[str(x) for x in parsed.get("weaknesses", [])][:5],
                recommended_focus=[str(x) for x in parsed.get("recommended_focus", [])][:5],
                source=f"gemini:{self._model}",
                is_mock=False,
            )

        except Exception as exc:
            return _heuristic_report(
                score=score,
                adaptive_score=adaptive_score,
                eval_results=eval_results,
                fallback_note=f"Gemini judge failed safely: {exc.__class__.__name__}: {exc}",
            )


def get_gemini_judge(
    *,
    api_key: str = "",
    model: str = "gemini-2.5-flash",
) -> "GeminiJudge | None":
    """Returns a GeminiJudge if api_key is set, else None (caller should fall back)."""
    if api_key:
        return GeminiJudge(api_key=api_key, model=model)
    return None
