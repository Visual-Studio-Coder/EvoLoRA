"""Integration tests for the orchestrator state machine."""

import pytest

from evolora.agent.planner import HeuristicPlanner, MiniMaxPlanner
from evolora.demo.task import ADAPTIVE_EVAL_SET, LOCKED_EVAL_SET
from evolora.models.core import (
    AgentPlan,
    ArtifactMeta,
    JudgeReport,
    RetrainDecision,
    RunConfig,
    RunStatus,
    StopReason,
)
from evolora.models.events import EventKind
from evolora.orchestration.orchestrator import Orchestrator
from evolora.persistence.store import InMemoryRunStore


async def _collect(orch: Orchestrator) -> tuple[list, object]:
    events = []
    async for ev in await orch.run():
        events.append(ev)
    return events, orch._record


def _make_orch(**kwargs) -> Orchestrator:
    defaults = dict(max_iterations=2, target_score=0.99, patience=5)
    defaults.update(kwargs)
    cfg = RunConfig(**defaults)
    return Orchestrator(
        config=cfg,
        eval_set=LOCKED_EVAL_SET,
        adaptive_eval_set=ADAPTIVE_EVAL_SET,
        run_store=InMemoryRunStore(),
    )


class StaticJudge:
    is_mock = True

    def __init__(self, rating: float = 0.4) -> None:
        self.rating = rating

    async def judge(self, **kwargs):
        return JudgeReport(
            rating=self.rating,
            summary="Needs tighter JSON specialization.",
            weaknesses=["field accuracy"],
            recommended_focus=["field accuracy"],
            source="test_judge",
            is_mock=True,
        )


class StaticRetrainAdvisor:
    is_mock = True

    def __init__(self, retrain: bool) -> None:
        self.retrain = retrain

    async def decide(self, **kwargs):
        return RetrainDecision(
            retrain_recommended=self.retrain,
            confidence=0.8,
            reason="test decision",
            suggested_focus=["field accuracy"],
            source="test_advisor",
            is_mock=True,
        )


class EvalGatePlanner(MiniMaxPlanner):
    def __init__(self) -> None:
        pass

    async def generate_evals(self, goal: str, count: int = 5) -> list[dict]:
        return [{"prompt": f"{goal}: produce JSON", "expected": {"ok": True}}]

    async def plan(self, *args, **kwargs):
        return HeuristicPlanner().plan(*args, **kwargs), False


class FallbackReasonPlanner(MiniMaxPlanner):
    def __init__(self) -> None:
        self.last_error = "BadRequestError: invalid function arguments json string"

    async def plan(self, *args, **kwargs):
        return HeuristicPlanner().plan(*args, **kwargs), True


class StaticLLMJudge:
    async def judge(self, records: list[dict]) -> tuple[float, list[dict]]:
        scored = []
        scores = []
        for record in records:
            score = 2 if record.get("actual") == "base" else 8
            scored.append({**record, "score": score, "reason": "test judge"})
            scores.append(score)
        return sum(scores) / len(scores) / 10.0, scored


class RemoteBackendWithBaseline:
    is_mock = False
    name = "remote"

    def __init__(self) -> None:
        self.baseline_payload: dict | None = None
        self.train_payload: dict | None = None

    async def evaluate_base(self, remote_payload=None):
        self.baseline_payload = remote_payload

        async def stream():
            yield {"phase": "baseline", "message": "baseline live log", "done": False}
            yield {
                "done": True,
                "eval_records": [{"input": "q", "expected": "a", "actual": "base"}],
            }

        return stream()

    async def train(
        self,
        run_id: str,
        iteration: int,
        plan: AgentPlan,
        base_model_id: str,
        remote_payload=None,
    ):
        self.train_payload = remote_payload

        async def stream():
            yield {"phase": "train", "message": "unsloth step 1", "done": False}
            yield {
                "done": True,
                "artifact": ArtifactMeta(
                    run_id=run_id,
                    iteration=iteration,
                    adapter_path="/workspace/lora_model",
                    score=0.0,
                    checksum="remote-test",
                    is_mock=False,
                ),
                "eval_records": [{"input": "q", "expected": "a", "actual": "trained"}],
                "cost_usd": 0.0,
                "duration_s": 0.0,
            }

        return stream()

    async def health_check(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_happy_path_completes():
    orch = _make_orch()
    events, rec = await _collect(orch)
    assert rec.status in {RunStatus.COMPLETE, RunStatus.FAILED}
    kinds = [e.kind for e in events]
    assert EventKind.RUN_STARTED in kinds
    assert EventKind.EVAL_SET_LOCKED in kinds
    assert EventKind.BASELINE_COMPLETE in kinds


@pytest.mark.asyncio
async def test_max_iterations_stop():
    orch = _make_orch(max_iterations=1, target_score=1.0, patience=99)
    events, rec = await _collect(orch)
    assert rec.status == RunStatus.COMPLETE
    assert rec.stop_reason == StopReason.MAX_ITERATIONS


@pytest.mark.asyncio
async def test_cancellation():
    orch = _make_orch(max_iterations=5)
    orch.cancel()
    events, rec = await _collect(orch)
    assert rec.status == RunStatus.CANCELLED
    assert rec.stop_reason == StopReason.CANCELLED


@pytest.mark.asyncio
async def test_event_ordering():
    orch = _make_orch(max_iterations=1)
    events, _ = await _collect(orch)
    kinds = [e.kind for e in events]
    assert kinds[0] == EventKind.RUN_STARTED
    assert kinds[1] == EventKind.EVAL_SET_LOCKED
    assert kinds[2] == EventKind.STATUS_CHANGED  # baseline status
    assert kinds[3] == EventKind.BASELINE_COMPLETE


@pytest.mark.asyncio
async def test_generated_evals_require_user_approval_before_locking():
    cfg = RunConfig(
        max_iterations=1, target_score=1.0, goal="custom receipt parser", require_retrain_approval=True
    )
    orch = Orchestrator(
        config=cfg,
        eval_set=LOCKED_EVAL_SET,
        planner=EvalGatePlanner(),
        run_store=InMemoryRunStore(),
    )

    events = []
    async for event in await orch.run():
        events.append(event)
        if event.kind == EventKind.EVAL_APPROVAL_REQUIRED:
            assert event.data["evals"] == [
                {
                    "input": "custom receipt parser: produce JSON",
                    "expected": '{"ok": true}',
                }
            ]
            orch.submit_retrain_approval(False)

    kinds = [event.kind for event in events]
    assert EventKind.EVAL_APPROVAL_REQUIRED in kinds
    assert EventKind.EVAL_SET_LOCKED not in kinds
    assert orch._record.status == RunStatus.CANCELLED
    assert orch._record.stop_reason == StopReason.CANCELLED


@pytest.mark.asyncio
async def test_remote_baseline_uses_vm_records_and_llm_judge():
    backend = RemoteBackendWithBaseline()
    cfg = RunConfig(max_iterations=1, target_score=1.0, training_backend="remote")
    orch = Orchestrator(
        config=cfg,
        eval_set=LOCKED_EVAL_SET,
        training_backend=backend,
        llm_judge=StaticLLMJudge(),
        run_store=InMemoryRunStore(),
    )

    events, rec = await _collect(orch)
    log_messages = [event.message for event in events if event.kind == EventKind.LOG]

    assert backend.baseline_payload is not None
    assert backend.train_payload is not None
    assert rec.baseline_score == 0.2
    assert rec.iterations[0].score == 0.8
    assert "[remote:baseline] baseline live log" in log_messages
    assert "[remote:train] unsloth step 1" in log_messages


@pytest.mark.asyncio
async def test_fallback_event_includes_minimax_failure_reason():
    cfg = RunConfig(max_iterations=1, target_score=1.0)
    orch = Orchestrator(
        config=cfg,
        eval_set=LOCKED_EVAL_SET,
        planner=FallbackReasonPlanner(),
        run_store=InMemoryRunStore(),
    )

    events, _ = await _collect(orch)
    fallback_event = next(event for event in events if event.kind == EventKind.AGENT_FALLBACK_USED)

    assert "invalid function arguments" in fallback_event.message
    assert fallback_event.data["reason"] == "BadRequestError: invalid function arguments json string"


@pytest.mark.asyncio
async def test_best_iteration_preserved():
    orch = _make_orch(max_iterations=2, target_score=1.0)
    _, rec = await _collect(orch)
    # best_score should be >= baseline
    assert rec.best_score >= rec.baseline_score


@pytest.mark.asyncio
async def test_exact_training_sample_count_is_enforced():
    orch = _make_orch(max_iterations=1, target_score=1.0, training_sample_count=12)
    _, rec = await _collect(orch)
    assert rec.iterations
    assert len(rec.iterations[0].plan.data_spec.examples) == 12
    assert rec.iterations[0].plan.data_spec.max_examples == 12


@pytest.mark.asyncio
async def test_training_examples_stack_across_iterations():
    cfg = RunConfig(max_iterations=2, target_score=1.0, training_sample_count=10)
    orch = Orchestrator(
        config=cfg,
        eval_set=LOCKED_EVAL_SET,
        adaptive_eval_set=ADAPTIVE_EVAL_SET,
        judge=StaticJudge(rating=0.3),
        retrain_advisor=StaticRetrainAdvisor(retrain=True),
        run_store=InMemoryRunStore(),
    )
    _, rec = await _collect(orch)

    assert len(rec.iterations) == 2
    # Each iteration adds training_sample_count more examples (stacking): 10 -> 20.
    assert len(rec.iterations[0].plan.data_spec.examples) == 10
    assert len(rec.iterations[1].plan.data_spec.examples) == 20
    assert rec.iterations[1].plan.data_spec.max_examples == 20


@pytest.mark.asyncio
async def test_no_secrets_in_prompts():
    """Eval prompts passed to runner must not contain expected answers."""
    prompts = LOCKED_EVAL_SET.prompts_only()
    for p in prompts:
        assert "expected" not in p
        assert "top_customer_total" not in str(p)


@pytest.mark.asyncio
async def test_in_memory_store_persists():
    orch = _make_orch(max_iterations=1)
    _, rec = await _collect(orch)
    stored = await orch._store.get(rec.run_id)
    assert stored is not None
    assert stored.run_id == rec.run_id


@pytest.mark.asyncio
async def test_mongo_serialization_shape():
    orch = _make_orch(max_iterations=1)
    _, rec = await _collect(orch)
    from evolora.persistence.store import InMemoryRunStore
    store = orch._store
    assert isinstance(store, InMemoryRunStore)
    doc = store.to_mongo_doc(rec)
    assert "_id" in doc
    assert doc["_id"] == rec.run_id


@pytest.mark.asyncio
async def test_judge_and_retrain_decision_are_recorded():
    cfg = RunConfig(max_iterations=1, target_score=1.0)
    orch = Orchestrator(
        config=cfg,
        eval_set=LOCKED_EVAL_SET,
        adaptive_eval_set=ADAPTIVE_EVAL_SET,
        judge=StaticJudge(rating=0.42),
        retrain_advisor=StaticRetrainAdvisor(retrain=True),
    )
    events, rec = await _collect(orch)
    kinds = [e.kind for e in events]

    assert EventKind.JUDGE_STARTED in kinds
    assert EventKind.JUDGE_COMPLETE in kinds
    assert EventKind.RETRAIN_DECISION_RECEIVED in kinds
    assert rec.iterations[0].judge_report is not None
    assert rec.iterations[0].judge_report.rating == 0.42
    assert rec.iterations[0].retrain_decision is not None
    assert rec.iterations[0].retrain_decision.retrain_recommended is True


@pytest.mark.asyncio
async def test_below_target_auto_iterates_without_asking():
    # While the judge still wants more training (below target), the loop auto-iterates
    # without prompting — no USER_APPROVAL_REQUIRED until the model is 'good enough'.
    cfg = RunConfig(max_iterations=2, target_score=1.0, require_retrain_approval=True)
    orch = Orchestrator(
        config=cfg,
        eval_set=LOCKED_EVAL_SET,
        adaptive_eval_set=ADAPTIVE_EVAL_SET,
        judge=StaticJudge(rating=0.25),
        retrain_advisor=StaticRetrainAdvisor(retrain=True),  # judge wants more training
        run_store=InMemoryRunStore(),
    )
    events, rec = await _collect(orch)

    # Key behavior: it never paused to ask, and ran more than one iteration on its own.
    assert EventKind.USER_APPROVAL_REQUIRED not in [e.kind for e in events]
    assert len(rec.iterations) == 2


@pytest.mark.asyncio
async def test_judge_acceptance_stops_without_more_training():
    cfg = RunConfig(max_iterations=3, target_score=1.0)
    orch = Orchestrator(
        config=cfg,
        eval_set=LOCKED_EVAL_SET,
        adaptive_eval_set=ADAPTIVE_EVAL_SET,
        judge=StaticJudge(rating=0.91),
        retrain_advisor=StaticRetrainAdvisor(retrain=False),
    )
    _, rec = await _collect(orch)

    assert len(rec.iterations) == 1
    assert rec.status == RunStatus.COMPLETE
    assert rec.stop_reason == StopReason.JUDGE_ACCEPTED


@pytest.mark.asyncio
async def test_keep_training_gate_lets_user_stop_when_judge_accepts():
    cfg = RunConfig(max_iterations=2, target_score=1.0, require_retrain_approval=True)
    orch = Orchestrator(
        config=cfg,
        eval_set=LOCKED_EVAL_SET,
        adaptive_eval_set=ADAPTIVE_EVAL_SET,
        judge=StaticJudge(rating=0.95),
        retrain_advisor=StaticRetrainAdvisor(retrain=False),  # judge says good enough
        run_store=InMemoryRunStore(),
    )
    events = []
    async for event in await orch.run():
        events.append(event)
        if event.kind == EventKind.USER_APPROVAL_REQUIRED:
            assert event.data.get("approval_type") == "keep_training"
            orch.submit_retrain_approval(False)  # user accepts the model -> stop

    assert EventKind.USER_APPROVAL_REQUIRED in [e.kind for e in events]
    assert orch._record.stop_reason == StopReason.JUDGE_ACCEPTED
    assert len(orch._record.iterations) == 1


@pytest.mark.asyncio
async def test_adaptive_hardens_evals_when_base_model_aces_them():
    class HardeningPlanner(MiniMaxPlanner):
        def __init__(self):
            self.difficulties: list[str] = []

        async def generate_evals(self, goal, count=5, difficulty="standard"):
            self.difficulties.append(difficulty)
            tag = "hard" if difficulty == "hard" else "easy"
            return [{"prompt": f"{tag}-q{i}", "expected": {"ok": True}} for i in range(count)]

        async def plan(self, *args, **kwargs):
            return HeuristicPlanner().plan(*args, **kwargs), False

    planner = HardeningPlanner()
    cfg = RunConfig(max_iterations=1, target_score=1.0, goal="sql queries")
    orch = Orchestrator(
        config=cfg, eval_set=LOCKED_EVAL_SET, planner=planner, run_store=InMemoryRunStore()
    )

    # Force the base model to ace the first (easy) eval set so hardening triggers.
    scores = iter([0.9, 0.3])

    async def fake_eval(adapter_path=None, eval_set=None):
        return next(scores, 0.3), [], {}

    orch._eval = fake_eval

    events = []
    async for event in await orch.run():
        events.append(event)
        if event.kind == EventKind.EVAL_APPROVAL_REQUIRED:
            orch.submit_retrain_approval(True)

    assert "hard" in planner.difficulties  # regenerated a harder eval set
    baselines = [e for e in events if e.kind == EventKind.BASELINE_COMPLETE]
    assert len(baselines) == 2  # initial baseline + re-baseline on harder evals


@pytest.mark.asyncio
async def test_keep_training_gate_continues_when_user_wants_more():
    cfg = RunConfig(max_iterations=2, target_score=1.0, require_retrain_approval=True)
    orch = Orchestrator(
        config=cfg,
        eval_set=LOCKED_EVAL_SET,
        adaptive_eval_set=ADAPTIVE_EVAL_SET,
        judge=StaticJudge(rating=0.95),
        retrain_advisor=StaticRetrainAdvisor(retrain=False),
        run_store=InMemoryRunStore(),
    )
    async for event in await orch.run():
        if event.kind == EventKind.USER_APPROVAL_REQUIRED:
            orch.submit_retrain_approval(True)  # keep training to make it smarter

    # judge accepted at iter 1 but the user pushed on; ran the 2nd (max) iteration too
    assert len(orch._record.iterations) == 2
