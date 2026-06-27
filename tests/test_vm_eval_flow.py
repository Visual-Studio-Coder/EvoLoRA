"""VM eval handoff: backend yields eval_records -> orchestrator scores via the LLM-judge."""

import pytest

from evolora.demo.task import LOCKED_EVAL_SET
from evolora.evaluation.llm_judge import LLMJudgeEvaluator
from evolora.models.core import ArtifactMeta, RunConfig
from evolora.orchestration.orchestrator import Orchestrator


class _RemoteBackendWithEvals:
    """Fake remote backend: skips the GPU, yields VM-style eval_records (actual filled)."""

    is_mock = False
    name = "remote"

    async def train(self, run_id, iteration, plan, base_model_id, remote_payload=None):
        async def stream():
            yield {"phase": "train", "message": "training", "done": False}
            yield {
                "done": True,
                "artifact": ArtifactMeta(
                    run_id=run_id,
                    iteration=iteration,
                    adapter_path="/workspace/lora_model",
                    score=0.0,
                    checksum="x",
                    is_mock=False,
                ),
                "eval_records": [
                    {"input": "q1", "expected": "e1", "actual": "good", "score": None},
                    {"input": "q2", "expected": "e2", "actual": "bad", "score": None},
                ],
                "cost_usd": 0.0,
                "duration_s": 0.0,
            }

        return stream()

    async def health_check(self) -> bool:
        return True


class _StubJudge(LLMJudgeEvaluator):
    def __init__(self):
        super().__init__(api_key="x")

    async def _score(self, client, record):
        return (10, "match") if record["actual"] == "good" else (0, "wrong")


@pytest.mark.asyncio
async def test_orchestrator_scores_vm_eval_records_with_llm_judge():
    cfg = RunConfig(max_iterations=1, training_backend="remote", target_score=1.0)
    orch = Orchestrator(
        config=cfg,
        eval_set=LOCKED_EVAL_SET,
        training_backend=_RemoteBackendWithEvals(),
        llm_judge=_StubJudge(),
    )

    async for _ in await orch.run():
        pass

    rec = orch._record
    assert rec.iterations
    it = rec.iterations[0]
    # judge scored the VM records: mean(10, 0)/10 = 0.5
    assert abs(it.score - 0.5) < 1e-6
    assert len(it.eval_results) == 2
    assert it.eval_results[0].details.get("reason") == "match"
    assert it.eval_results[1].details.get("reason") == "wrong"
