"""MiniMax retrain decision layer fed by the DigitalOcean judge report."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from evolora.models.core import JudgeReport, RetrainDecision


class RetrainAdvisor(Protocol):
    @property
    def is_mock(self) -> bool: ...

    async def decide(
        self,
        *,
        goal: str,
        rating: float,
        target_score: float,
        iteration: int,
        max_iterations: int,
        judge_report: JudgeReport,
    ) -> RetrainDecision: ...


def _extract_json(raw: str) -> Any:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)


def _heuristic_decision(
    *,
    rating: float,
    target_score: float,
    iteration: int,
    max_iterations: int,
    judge_report: JudgeReport,
    source: str = "heuristic",
    note: str = "",
) -> RetrainDecision:
    has_room = iteration < max_iterations
    retrain = has_room and rating < target_score
    reason = (
        f"{source} recommends {'another iteration' if retrain else 'stopping'}: "
        f"judge rating {rating:.2f}, target {target_score:.2f}, iteration "
        f"{iteration}/{max_iterations}."
    )
    if note:
        reason += f" {note}"
    return RetrainDecision(
        retrain_recommended=retrain,
        confidence=0.65 if retrain else 0.55,
        reason=reason,
        suggested_focus=judge_report.recommended_focus[:5],
        source=source,
        is_mock=True,
    )


class HeuristicRetrainAdvisor:
    @property
    def is_mock(self) -> bool:
        return True

    async def decide(
        self,
        *,
        goal: str,
        rating: float,
        target_score: float,
        iteration: int,
        max_iterations: int,
        judge_report: JudgeReport,
    ) -> RetrainDecision:
        return _heuristic_decision(
            rating=rating,
            target_score=target_score,
            iteration=iteration,
            max_iterations=max_iterations,
            judge_report=judge_report,
            source="heuristic_minimax_decision_fallback",
            note="MiniMax API key is not configured.",
        )


class MiniMaxRetrainAdvisor:
    """Asks MiniMax whether the DigitalOcean judge result justifies retraining."""

    def __init__(self, *, api_key: str, base_url: str, model: str) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._model = model

    @property
    def is_mock(self) -> bool:
        return False

    def _make_client(self):
        from openai import AsyncOpenAI

        return AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)

    async def decide(
        self,
        *,
        goal: str,
        rating: float,
        target_score: float,
        iteration: int,
        max_iterations: int,
        judge_report: JudgeReport,
    ) -> RetrainDecision:
        payload = {
            "specialization_goal": goal,
            "digitalocean_judge_rating": rating,
            "target_score": target_score,
            "iteration": iteration,
            "max_iterations": max_iterations,
            "judge_summary": judge_report.summary,
            "judge_strengths": judge_report.strengths,
            "judge_weaknesses": judge_report.weaknesses,
            "judge_recommended_focus": judge_report.recommended_focus,
            "instruction": (
                "Decide whether another bounded LoRA training iteration is necessary. "
                "Recommend retraining only if the rating is below target, there is room "
                "for another iteration, and the focus areas are actionable."
            ),
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are MiniMax supervising a bounded self-improvement loop. "
                    "Return ONLY JSON with keys: retrain_recommended (boolean), confidence "
                    "(0..1), reason, suggested_focus (array)."
                ),
            },
            {"role": "user", "content": json.dumps(payload, sort_keys=True)},
        ]
        try:
            resp = await self._make_client().chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.1,
                # M2.7 is a reasoning model: it spends tokens on a <think> block before the
                # JSON answer. 500 truncated it (finish_reason=length) -> parse failure ->
                # silent heuristic fallback. 4000 lets it finish thinking AND emit the JSON.
                max_tokens=4000,
            )
            parsed = _extract_json(resp.choices[0].message.content or "{}")
            if not isinstance(parsed, dict):
                raise ValueError("decision response was not a JSON object")
            return RetrainDecision(
                retrain_recommended=bool(parsed.get("retrain_recommended", False)),
                confidence=float(parsed.get("confidence", 0.5)),
                reason=str(parsed.get("reason", "")).strip() or "MiniMax decision completed.",
                suggested_focus=[str(x) for x in parsed.get("suggested_focus", [])][:5],
                source=f"minimax:{self._model}",
                is_mock=False,
            )
        except Exception as exc:
            return _heuristic_decision(
                rating=rating,
                target_score=target_score,
                iteration=iteration,
                max_iterations=max_iterations,
                judge_report=judge_report,
                source="minimax_decision_fallback",
                note=f"MiniMax decision failed safely: {exc.__class__.__name__}.",
            )


def get_retrain_advisor(
    *,
    api_key: str = "",
    base_url: str = "https://api.minimax.io/v1",
    model: str = "MiniMax-M2.7-highspeed",
) -> RetrainAdvisor:
    if api_key:
        return MiniMaxRetrainAdvisor(api_key=api_key, base_url=base_url, model=model)
    return HeuristicRetrainAdvisor()
