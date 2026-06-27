"""LLM-as-a-judge evaluation over the canonical evals.json record format.

A canonical eval record (the shape pushed to / returned from the VM):

    {
      "input": str,            # the prompt/question given to the model
      "expected_output": str,  # the ground-truth reference answer
      "actual_output": str,    # starts empty, filled by the finetuned model during eval
      "score": int | None,     # starts null, filled by the LLM-as-a-judge (0-10)
      "reason": str            # starts empty, filled by the judge to explain the score
    }

The agent generates ``input`` + ``expected_output``; the VM runs the adapter and fills
``actual_output``; ``LLMJudgeEvaluator`` fills ``score`` (0-10) and ``reason`` using a
DigitalOcean inference model. The filled records can then be persisted (e.g. MongoDB).
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

EVAL_FIELDS = ("input", "expected_output", "actual_output", "score", "reason")


def make_eval_records(items: list[dict]) -> list[dict]:
    """Build canonical eval records from generated evals.

    Accepts the agent's ``{input|prompt, expected_output|expected}`` shape (expected may be a
    dict/list — it is JSON-stringified) and returns records with empty
    ``actual_output``/``score``/``reason`` ready to write to ``evals.json``.
    """
    records: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        inp = item.get("input", item.get("prompt", ""))
        exp = item.get("expected_output", item.get("expected", ""))
        if isinstance(exp, (dict, list)):
            exp = json.dumps(exp, sort_keys=True)
        records.append({
            "input": str(inp),
            "expected_output": str(exp),
            "actual_output": str(item.get("actual_output", "") or ""),
            "score": item.get("score"),
            "reason": str(item.get("reason", "") or ""),
        })
    return records


_JUDGE_SYSTEM = """You are a strict evaluation judge for a fine-tuned model. Given a question
(input), the expected correct output, and the model's actual output, score how well the actual
output matches the expected output on a 0-10 scale (10 = fully correct and equivalent in
meaning; 0 = wrong or empty). Respond with ONLY a JSON object:
{"score": <integer 0-10>, "reason": "<one short sentence>"}."""


def _parse_json(raw: str) -> dict:
    """Strip think-blocks / fences and parse the first JSON object found."""
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:])
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}


def _coerce_score(value: Any) -> int | None:
    try:
        return max(0, min(10, int(round(float(value)))))
    except (TypeError, ValueError):
        return None


class LLMJudgeEvaluator:
    """Fill ``score`` (0-10) + ``reason`` for eval records using an LLM judge."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://inference.do-ai.run/v1/",
        model: str = "llama3.3-70b-instruct",
        concurrency: int = 5,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._concurrency = max(1, concurrency)

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    def _make_client(self):
        from openai import AsyncOpenAI

        return AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)

    async def judge(self, records: list[dict]) -> tuple[float, list[dict]]:
        """Score each record. Returns (aggregate_score_0to1, filled_records).

        Records missing ``actual_output`` score 0. If the judge is unconfigured the records
        are returned unchanged with a 0.0 aggregate (caller can fall back).
        """
        if not records:
            return 0.0, []
        if not self.configured:
            return 0.0, [dict(r) for r in records]

        client = self._make_client()
        semaphore = asyncio.Semaphore(self._concurrency)

        async def score_one(record: dict) -> dict:
            filled = dict(record)
            if not str(filled.get("actual_output", "")).strip():
                filled["score"] = 0
                filled["reason"] = "no model output produced"
                return filled
            async with semaphore:
                try:
                    score, reason = await self._score(client, filled)
                    filled["score"] = score
                    filled["reason"] = reason
                except Exception as exc:  # pragma: no cover - network/judge failure
                    filled["reason"] = f"judge error: {exc}"
            return filled

        scored = await asyncio.gather(*(score_one(r) for r in records))
        graded = [r["score"] for r in scored if isinstance(r.get("score"), (int, float))]
        aggregate = (sum(graded) / len(graded) / 10.0) if graded else 0.0
        return aggregate, list(scored)

    async def _score(self, client, record: dict) -> tuple[int | None, str]:
        user = json.dumps({
            "input": record.get("input", ""),
            "expected_output": record.get("expected_output", ""),
            "actual_output": record.get("actual_output", ""),
        })
        resp = await client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content or ""
        data = _parse_json(raw)
        return _coerce_score(data.get("score")), str(data.get("reason", "")).strip()


def get_llm_judge_evaluator(
    api_key: str = "",
    base_url: str = "https://inference.do-ai.run/v1/",
    model: str = "llama3.3-70b-instruct",
) -> LLMJudgeEvaluator:
    return LLMJudgeEvaluator(api_key=api_key, base_url=base_url, model=model)
