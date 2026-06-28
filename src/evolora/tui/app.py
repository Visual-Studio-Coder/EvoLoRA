"""Textual TUI for EvoLoRA."""

from __future__ import annotations

import traceback
from datetime import datetime

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Input, ProgressBar, RichLog, Static

from evolora.agent.planner import get_planner
from evolora.config import get_config
from evolora.demo.task import ADAPTIVE_EVAL_SET, LOCKED_EVAL_SET
from evolora.evaluation.digitalocean_judge import get_judge
from evolora.models.core import RunConfig
from evolora.models.events import Event, EventKind
from evolora.observability.run_logger import default_log_dir
from evolora.orchestration.orchestrator import Orchestrator
from evolora.orchestration.retrain_advisor import get_retrain_advisor
from evolora.training.backends import get_backend
from evolora.training.runner import get_runner


class SectionTitle(Static):
    """Small green terminal-style section label."""


# Maps the run-state label shown in the status bar to a mascot mood.
_STATE_MOOD = {
    "READY": "idle", "INVALID": "sad",
    "STARTING": "run", "RUNNING": "run",
    "LOCKED": "think", "BASELINE": "look",
    "PLANNING": "think", "VALIDATE": "run",
    "TRAINING": "run", "TRAINED": "happy",
    "EVALUATE": "look", "EVAL": "look",
    "JUDGE": "look", "JUDGED": "think",
    "DECIDE": "think", "APPROVE": "think",
    "APPROVED": "run", "DECLINED": "idle",
    "BEST": "happy", "ITERATION": "run",
    "STOP": "idle", "DONE": "happy",
    "CANCEL": "sad", "CANCELLED": "sad", "FAILED": "sad",
}


class Mascot(Static):
    """A little terminal cat that idles/blinks and reacts to the run state.

    Always animated (alive): each mood is a list of frames cycled on a timer; active
    moods also scamper across the bar. Mirrors the agent via set_mood().
    """

    # mood -> (animation frames, caption, scampers across the bar)
    MOODS = {
        "idle": (["(=^.^=)", "(=^.^=)", "(=^.^=)", "(=^.^=)", "(=^.^=)", "(=^-^=)"], "purring", False),
        "think": (["(=o.o=)?", "(=o.o=) ", "(=O.o=)?", "(=o.O=)?"], "thinking", True),
        "run": (["(=^.^=)", "(=^o^=)"], "chasing", True),
        "look": (["(=O.O=)", "(=o.o=)", "(=O.O=)", "(=-.-=)"], "watching", True),
        "happy": (["(=^w^=)", "(=^o^=)"], "yay!", False),
        "sad": (["(=;_;=)", "(=T_T=)"], "mrrp", False),
    }

    def __init__(self, **kwargs) -> None:
        self._mood = "idle"
        self._pos = 0
        self._dir = 1
        self._anim = 0
        self._track = 16
        super().__init__(self._frame(), **kwargs)

    def on_mount(self) -> None:
        self.set_interval(0.18, self._tick)

    def set_mood(self, mood: str) -> None:
        self._mood = mood if mood in self.MOODS else "idle"
        self.update(self._frame())

    def _tick(self) -> None:
        self._anim += 1
        if self.MOODS[self._mood][2]:  # scampers
            self._pos += self._dir
            if self._pos >= self._track:
                self._pos, self._dir = self._track, -1
            elif self._pos <= 0:
                self._pos, self._dir = 0, 1
        self.update(self._frame())

    def _frame(self) -> str:
        frames, caption, _ = self.MOODS[self._mood]
        face = frames[self._anim % len(frames)]
        cell = ((" " * self._pos) + face).ljust(self._track + 10)
        return f"[#39ff14]{cell}[/][#005a1c]{caption}[/]"


class EvoLoRAApp(App[None]):
    """Native Textual interface for the EvoLoRA loop."""

    TITLE = "EvoLoRA"
    SUB_TITLE = "Auditable self-improvement loop"

    BINDINGS = [
        Binding("ctrl+r", "start_run", "Start run"),
        Binding("ctrl+x", "cancel_run", "Cancel run"),
        Binding("q", "quit", "Quit"),
    ]

    CSS = """
    Screen {
        background: #070c07;
        color: #007a1e;
    }

    #frame {
        height: 100%;
        width: 100%;
        layout: vertical;
        background: #070c07;
    }

    #topbar {
        height: 3;
        padding: 0 2;
        background: #040904;
        border-bottom: solid #0a4018;
        layout: horizontal;
        align: center middle;
    }

    #brand {
        width: auto;
        content-align: left middle;
        text-style: bold;
        color: #39ff14;
    }

    #mascot {
        width: 1fr;
        height: 1;
        margin: 0 2;
        content-align: left middle;
    }

    #clock {
        width: 22;
        content-align: right middle;
        color: #003010;
    }

    #main {
        height: 1fr;
        layout: horizontal;
        background: #001007;
    }

    #left-column {
        width: 1fr;
        min-width: 64;
        layout: vertical;
    }

    #right-column {
        width: 38;
        min-width: 34;
        layout: vertical;
    }

    .panel {
        background: #080d08;
        border: solid #0a5a24;
        padding: 1 2;
    }

    #agent-panel {
        height: 1fr;
    }

    #examples-panel {
        height: 1fr;
    }

    #config-panel {
        height: 14;
    }

    #hyperparam-panel {
        height: 9;
    }

    #hyperparam-panel.flash {
        border: solid #39ff14;
        background: #0d1f0d;
    }

    #metrics-panel {
        height: 1fr;
    }

    SectionTitle {
        height: 1;
        color: #007a1e;
        text-style: bold;
        background: #080d08;
    }

    RichLog {
        height: 1fr;
        background: #080d08;
        color: #006622;
        scrollbar-background: #040904;
        scrollbar-color: #004d18;
    }

    #config-values, #metrics-values {
        height: auto;
        color: #004018;
    }

    #loss-strip {
        height: 4;
        color: #00cc33;
    }

    ProgressBar {
        height: 1;
        margin: 1 0 0 0;
    }

    #statusbar {
        height: 3;
        padding: 0 2;
        background: #040904;
        border-top: solid #0a4018;
        color: #007a1e;
        layout: horizontal;
        align: center middle;
    }

    #run-state {
        width: 16;
        content-align: left middle;
        color: #39ff14;
        text-style: bold;
    }

    #status-text {
        width: 1fr;
        content-align: left middle;
        color: #004d18;
    }

    #score-text {
        width: 34;
        content-align: right middle;
        color: #39ff14;
    }

    #inputbar {
        height: 4;
        padding: 0 2;
        background: #040904;
        border-top: solid #0a4018;
        layout: horizontal;
        align: center middle;
    }

    #goal-input {
        width: 1fr;
        background: #060c06;
        color: #39ff14;
        border: solid #002a0c;
    }

    #sample-label {
        width: 10;
        margin: 0 0 0 1;
        content-align: right middle;
        color: #004018;
    }

    #sample-count-input {
        width: 10;
        background: #060c06;
        color: #39ff14;
        border: solid #002a0c;
    }

    #start-button {
        min-width: 14;
        margin: 0 0 0 1;
        background: #00280d;
        color: #39ff14;
        border: solid #006622;
    }

    #cancel-button {
        min-width: 12;
        margin: 0 0 0 1;
        background: #120606;
        color: #aa5544;
        border: solid #3d1612;
    }

    #approve-retrain-button {
        min-width: 7;
        margin: 0 0 0 1;
        background: #00280d;
        color: #39ff14;
        border: solid #006622;
    }

    #decline-retrain-button {
        min-width: 6;
        margin: 0 0 0 1;
        background: #120606;
        color: #aa5544;
        border: solid #3d1612;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._orchestrator: Orchestrator | None = None
        self._run_active = False
        self._loss_values: list[float] = []
        self._baseline = 0.0
        self._best = 0.0
        self._current = 0.0
        self._judge_rating: float | None = None
        self._requested_sample_count: int | None = 30
        self._goal = ""
        self._approval_context: str | None = None
        self._hyperparams: dict = {}

    def compose(self) -> ComposeResult:
        with Container(id="frame"):
            with Horizontal(id="topbar"):
                yield Static("> EvoLoRA", id="brand")
                yield Mascot(id="mascot")
                yield Static("", id="clock")

            with Horizontal(id="main"):
                with Vertical(id="left-column"):
                    with Vertical(id="agent-panel", classes="panel"):
                        yield SectionTitle("-- AGENT REASONING -----------------------------")
                        yield RichLog(
                            id="agent-log",
                            markup=True,
                            highlight=False,
                            wrap=True,
                            auto_scroll=True,
                        )
                    with Vertical(id="examples-panel", classes="panel"):
                        yield SectionTitle("-- TRAINING EXAMPLES ----------------------------")
                        yield RichLog(
                            id="examples-log",
                            markup=True,
                            highlight=False,
                            wrap=True,
                            auto_scroll=True,
                        )

                with Vertical(id="right-column"):
                    with Vertical(id="config-panel", classes="panel"):
                        yield SectionTitle("-- LORA CONFIG")
                        yield Static("", id="config-values")
                    with Vertical(id="hyperparam-panel", classes="panel"):
                        yield SectionTitle("-- HYPERPARAMETERS")
                        yield Static("", id="hyperparam-values")
                    with Vertical(id="metrics-panel", classes="panel"):
                        yield SectionTitle("-- TRAINING METRICS")
                        yield Static("", id="metrics-values")
                        yield ProgressBar(total=100, show_eta=False, id="training-progress")
                        yield Static("", id="loss-strip")

            with Horizontal(id="statusbar"):
                yield Static("READY", id="run-state")
                yield Static("mock backend idle | locked eval pending | agent: heuristic/MiniMax", id="status-text")
                yield Static("baseline -- | best --", id="score-text")

            with Horizontal(id="inputbar"):
                yield Input(
                    placeholder="What kind of specialized model would you like to build today?",
                    id="goal-input",
                )
                yield Static("# samples", id="sample-label")
                yield Input(
                    value="30",
                    placeholder="auto",
                    restrict=r"[0-9]*",
                    max_length=6,
                    id="sample-count-input",
                )
                yield Button("START", id="start-button")
                yield Button("CANCEL", id="cancel-button", disabled=True)
                yield Button("YES", id="approve-retrain-button", disabled=True)
                yield Button("NO", id="decline-retrain-button", disabled=True)

    def on_mount(self) -> None:
        self.set_interval(1, self._update_clock)
        self._update_clock()
        self._update_config_panel()
        self._update_hyperparam_panel()
        self._update_metrics_panel()
        self.query_one("#goal-input", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start-button":
            self.action_start_run()
        elif event.button.id == "cancel-button":
            self.action_cancel_run()
        elif event.button.id == "approve-retrain-button":
            self.action_answer_retrain(True)
        elif event.button.id == "decline-retrain-button":
            self.action_answer_retrain(False)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id in {"goal-input", "sample-count-input"}:
            self.action_start_run()

    def action_start_run(self) -> None:
        if self._run_active:
            self._agent_log().write("[yellow][!] Run already in progress[/]")
            return

        sample_count = self._parse_sample_count()
        if sample_count == 0:
            return

        self._requested_sample_count = sample_count
        self._goal = self.query_one("#goal-input", Input).value.strip()
        self._update_config_panel()
        self._run_active = True
        self._loss_values.clear()
        self._baseline = 0.0
        self._best = 0.0
        self._current = 0.0
        self._judge_rating = None
        self._approval_context = None
        self.query_one("#start-button", Button).disabled = True
        self.query_one("#cancel-button", Button).disabled = False
        # Lock the goal + sample inputs for the duration of the run.
        self.query_one("#goal-input", Input).disabled = True
        self.query_one("#sample-count-input", Input).disabled = True
        self._set_retrain_buttons(False)
        self._agent_log().clear()
        self._examples_log().clear()
        if self._goal:
            self._agent_log().write(f"[bright_green][>][/] Use case sent to agent: [bold]{self._goal}[/]")
        sample_label = (
            f"exactly {sample_count} training samples"
            if sample_count is not None
            else "agent-selected training sample count"
        )
        self._set_state("STARTING", f"building mock-first EvoLoRA run | {sample_label}")
        self.query_one("#training-progress", ProgressBar).update(total=100, progress=0)
        self.run_worker(self._run_evolora(), group="evolora-run", exclusive=True)

    def action_cancel_run(self) -> None:
        if self._orchestrator is not None:
            self._orchestrator.cancel()
        self._set_state("CANCEL", "cancellation requested")
        self._agent_log().write("[red][!] Cancellation requested[/]")

    def action_answer_retrain(self, approved: bool) -> None:
        if self._orchestrator is None:
            return
        self._orchestrator.submit_retrain_approval(approved)
        self._set_retrain_buttons(False)
        context = self._approval_context or "retrain"
        self._approval_context = None
        answer = "approved" if approved else "declined"
        label = "Generated eval set" if context == "evals" else "Retrain"
        self._agent_log().write(f"[yellow][user][/] {label} {answer}")

    async def _run_evolora(self) -> None:
        cfg = get_config()
        run_config = RunConfig(
            max_iterations=cfg.max_iterations,
            target_score=cfg.target_score,
            improvement_threshold=cfg.improvement_threshold,
            patience=cfg.patience,
            training_backend=cfg.training_backend,
            model_runner=cfg.model_runner,
            base_model_id=cfg.base_model_id,
            training_sample_count=self._requested_sample_count,
            goal=self._goal,
            require_retrain_approval=True,
        )

        backend = get_backend(cfg.training_backend)
        runner = get_runner(cfg.model_runner)
        # Use the real MiniMax tool-calling planner whenever a key is configured — the agent
        # plans for real even when training stays on the mock backend (real reasoning, mock GPU).
        planner = get_planner(
            use_minimax=cfg.minimax_available,
            api_key=cfg.minimax_api_key,
            model=cfg.minimax_model,
            base_url=cfg.minimax_base_url,
        )
        judge = get_judge(
            api_key=cfg.digital_ocean_model_access_key,
            base_url=cfg.digitalocean_inference_base_url,
            model=cfg.digital_ocean_judge_model,
        )
        retrain_advisor = get_retrain_advisor(
            api_key=cfg.minimax_api_key,
            model=cfg.minimax_model,
            base_url=cfg.minimax_base_url,
        )

        self._orchestrator = Orchestrator(
            config=run_config,
            eval_set=LOCKED_EVAL_SET,
            planner=planner,
            training_backend=backend,
            model_runner=runner,
            adaptive_eval_set=ADAPTIVE_EVAL_SET,
            judge=judge,
            retrain_advisor=retrain_advisor,
        )

        try:
            async for event in await self._orchestrator.run():
                self._apply_event(event)
        except Exception as exc:  # pragma: no cover - defensive UI boundary
            error_path = self._write_tui_exception(exc)
            self._set_state("FAILED", str(exc))
            self._agent_log().write(f"[red][x] TUI run failed:[/] {exc}")
            if error_path:
                self._agent_log().write(f"[red][x] Full traceback saved:[/] {error_path}")
        finally:
            self._run_active = False
            self.query_one("#start-button", Button).disabled = False
            self.query_one("#cancel-button", Button).disabled = True
            self.query_one("#goal-input", Input).disabled = False
            self.query_one("#sample-count-input", Input).disabled = False
            self._set_retrain_buttons(False)
            self._update_examples_from_record()

    def _apply_event(self, event: Event) -> None:
        kind = event.kind
        data = event.data

        if kind == EventKind.RUN_STARTED:
            mode = "MOCK" if data.get("mock") else "REAL"
            self._set_state("RUNNING", f"{mode} run {event.run_id[:8]} started")
            self._agent_log().write(f"[green][OK][/] EvoLoRA run started: [bold]{event.run_id[:8]}[/]")
            readable_log = self._current_readable_log_path()
            if readable_log:
                self._agent_log().write(f"[cyan][log][/] Full run log: {readable_log}")
            return

        if kind == EventKind.EVAL_SET_LOCKED:
            self._set_state("LOCKED", event.message)
            self._agent_log().write(f"[cyan][OK][/] Locked evaluation hash: {data.get('hash', '')[:16]}...")
            self._update_config_panel(eval_hash=str(data.get("hash", ""))[:12])
            return

        if kind == EventKind.STATUS_CHANGED:
            self._set_state("BASELINE", event.message)
            self._agent_log().write(f"[blue][>][/] {event.message}")
            return

        if kind == EventKind.BASELINE_COMPLETE:
            self._baseline = float(data.get("score", 0.0))
            self._best = self._baseline
            self._set_state("BASELINE", event.message)
            self._agent_log().write(f"[blue][OK][/] Baseline score: [bold]{self._baseline:.3f}[/]")
            self._update_score_text()
            return

        if kind == EventKind.PLANNING_STARTED:
            self._set_state("PLANNING", event.message)
            self._agent_log().write(f"[yellow][>][/] {event.message}")
            return

        if kind == EventKind.AGENT_FALLBACK_USED:
            reason = str(data.get("reason", "")).strip()
            detail = f": {reason}" if reason else ""
            self._agent_log().write(f"[red][!][/] MiniMax unavailable; using heuristic fallback{detail}")
            return

        if kind == EventKind.EVAL_APPROVAL_REQUIRED:
            evals = data.get("evals", []) or []
            self._approval_context = "evals"
            self._set_state("APPROVE", f"Review {len(evals)} generated evals | YES or NO")
            self._agent_log().write(
                f"[yellow][?][/] MiniMax generated [bold]{len(evals)}[/] eval examples. Approve them before locking the benchmark."
            )
            self._render_eval_approval(evals)
            self._set_retrain_buttons(True)
            return

        if kind == EventKind.PLAN_RECEIVED:
            focus = ", ".join(data.get("focus_areas", []) or ["json_format", "field_accuracy"])
            self._agent_log().write(f"[green][OK][/] Plan received. Focus: [bold]{focus}[/]")
            rationale = str(data.get("rationale", "")).strip()
            if rationale:
                self._examples_log().write(f"[green]planner rationale[/]: {rationale}")
            self._update_config_panel(focus=focus)
            return

        if kind == EventKind.VALIDATION_COMPLETE:
            self._set_state("VALIDATE", event.message)
            self._agent_log().write(f"[green][OK][/] {event.message}")
            hyperparams = data.get("hyperparams")
            if hyperparams:
                changed = hyperparams != self._hyperparams
                self._hyperparams = dict(hyperparams)
                self._update_hyperparam_panel(hyperparams)
                if changed:
                    self._flash_hyperparams()
            return

        if kind == EventKind.TRAINING_STARTED:
            self._set_state("TRAINING", f"{event.message} | backend={data.get('backend', 'mock')}")
            self._agent_log().write(f"[green][>][/] Training started on [bold]{data.get('backend', 'mock')}[/] backend")
            return

        if kind == EventKind.TRAINING_PROGRESS:
            step = int(data.get("step", 0))
            total = int(data.get("total_steps", 1))
            loss = float(data.get("loss", 0.0))
            self._loss_values.append(loss)
            self._set_state("TRAINING", f"step {step} / {total} | loss {loss:.4f}")
            self.query_one("#training-progress", ProgressBar).update(total=total, progress=step)
            self._update_metrics_panel(step=step, total_steps=total, loss=loss)
            if step == 1 or step == total or step % 5 == 0:
                self._agent_log().write(f"[green][train][/] step {step}/{total} loss={loss:.4f}")
            return

        if kind == EventKind.TRAINING_COMPLETE:
            self._set_state("TRAINED", f"training complete | mock={data.get('is_mock', True)}")
            self._agent_log().write("[green][OK][/] Training complete; mock adapter artifact created")
            return

        if kind == EventKind.EVAL_STARTED:
            self._set_state("EVALUATE", event.message)
            self._agent_log().write(f"[blue][>][/] {event.message}")
            return

        if kind == EventKind.EVAL_COMPLETE:
            self._current = float(data.get("score", 0.0))
            self._set_state("EVAL", event.message)
            self._agent_log().write(f"[blue][OK][/] Locked evaluation score: [bold]{self._current:.3f}[/]")
            self._update_score_text()
            return

        if kind == EventKind.ADAPTIVE_COMPLETE:
            self._agent_log().write(f"[green][diag][/] Adaptive challenge score: {float(data.get('score', 0.0)):.3f}")
            return

        if kind == EventKind.JUDGE_STARTED:
            self._set_state("JUDGE", event.message)
            self._agent_log().write(f"[cyan][>][/] {event.message}")
            return

        if kind == EventKind.JUDGE_COMPLETE:
            self._judge_rating = float(data.get("rating", 0.0))
            source = str(data.get("source", "judge"))
            mode = "mock/fallback" if data.get("is_mock") else "real"
            summary = str(data.get("summary", "")).strip()
            self._set_state("JUDGED", f"{mode} judge rating {self._judge_rating:.2f}")
            self._agent_log().write(
                f"[cyan][judge][/] {mode} {source} rating: [bold]{self._judge_rating:.2f}[/]"
            )
            if summary:
                self._examples_log().write(f"[cyan]judge summary[/]: {summary}")
            for weakness in data.get("weaknesses", [])[:3]:
                self._examples_log().write(f"[yellow]judge focus[/]: {weakness}")
            self._update_metrics_panel()
            return

        if kind == EventKind.RETRAIN_DECISION_RECEIVED:
            recommendation = "retrain" if data.get("retrain_recommended") else "stop"
            actor = "heuristic advisor" if data.get("is_mock") else "MiniMax"
            reason = str(data.get("reason", "")).strip()
            self._set_state("DECIDE", f"{actor} recommends {recommendation}")
            self._agent_log().write(
                f"[yellow][decision][/] {actor} recommends [bold]{recommendation}[/]"
            )
            if reason:
                self._examples_log().write(f"[yellow]decision reason[/]: {reason}")
            return

        if kind == EventKind.USER_APPROVAL_REQUIRED:
            rating = float(data.get("rating", 0.0))
            self._approval_context = "retrain"
            self._set_state("APPROVE", f"Retrain? judge rating {rating:.2f} | YES or NO")
            self._agent_log().write(
                f"[yellow][?][/] Retraining is recommended. Judge rating: [bold]{rating:.2f}[/]. Choose YES or NO."
            )
            self._set_retrain_buttons(True)
            return

        if kind == EventKind.USER_APPROVAL_RECEIVED:
            approved = bool(data.get("approved", False))
            self._set_state("APPROVED" if approved else "DECLINED", event.message)
            self._approval_context = None
            self._set_retrain_buttons(False)
            return

        if kind == EventKind.BEST_UPDATED:
            self._best = float(data.get("score", self._best))
            self._set_state("BEST", event.message)
            self._agent_log().write(f"[green][OK][/] {event.message}")
            self._update_score_text()
            return

        if kind == EventKind.ITERATION_COMPLETE:
            self._current = float(data.get("score", self._current))
            self._best = float(data.get("best", self._best))
            self._set_state("ITERATION", event.message)
            self._agent_log().write(f"[green][OK][/] {event.message}")
            self._update_score_text()
            self._update_examples_from_record()
            return

        if kind == EventKind.STOP_CONDITION_MET:
            self._set_state("STOP", event.message)
            self._agent_log().write(f"[yellow][stop][/] {event.message}")
            return

        if kind in {EventKind.RUN_COMPLETE, EventKind.RUN_CANCELLED, EventKind.RUN_FAILED}:
            state = {
                EventKind.RUN_COMPLETE: "DONE",
                EventKind.RUN_CANCELLED: "CANCELLED",
                EventKind.RUN_FAILED: "FAILED",
            }[kind]
            self._set_state(state, event.message)
            self._best = float(data.get("best_score", self._best))
            self._agent_log().write(f"[bold green][{state}][/] {event.message}")
            self._update_score_text()
            return

        if kind == EventKind.LOG:
            self._agent_log().write(event.message)

    def _update_examples_from_record(self) -> None:
        if self._orchestrator is None:
            return
        record = self._orchestrator._record
        if not record.iterations:
            return
        latest = record.iterations[-1]
        examples = latest.plan.data_spec.examples[:3]
        log = self._examples_log()
        log.write(f"[green]iteration {latest.iteration} training data[/]: {len(latest.plan.data_spec.examples)} examples")
        for index, example in enumerate(examples, start=1):
            prompt = str(example.get("prompt", "")).replace("\n", " ")[:100]
            completion = str(example.get("completion", "")).replace("\n", " ")[:100]
            log.write(f"[#003a10]#{index:03d}[/] [#00aa44]{prompt}[/]")
            log.write(f"      [#007a1e]{completion}[/]")

    def _render_eval_approval(self, evals: list[dict]) -> None:
        log = self._examples_log()
        log.write("[yellow]generated evals awaiting approval[/]")
        for index, item in enumerate(evals[:10], start=1):
            prompt = str(item.get("input", "")).replace("\n", " ")[:120]
            expected = str(item.get("expected", "")).replace("\n", " ")[:160]
            log.write(f"[#ffb000]eval #{index:02d} input[/]: {prompt}")
            log.write(f"[#d97b00]expected[/]: {expected}")
        if len(evals) > 10:
            log.write(f"[#805000]... {len(evals) - 10} more eval examples hidden[/]")

    def _update_clock(self) -> None:
        self.query_one("#clock", Static).update(datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))

    def _update_config_panel(
        self,
        *,
        eval_hash: str | None = None,
        focus: str | None = None,
    ) -> None:
        cfg = get_config()
        lines = [
            ("base_model", cfg.base_model_id.split("/")[-1]),
            ("backend", cfg.training_backend),
            # In remote mode the mock runner is bypassed — baseline + eval run the real model
            # on the VM and are LLM-judged. Show that instead of the unused "mock" label.
            ("runner", "vm (remote)" if cfg.training_backend == "remote" else cfg.model_runner),
            ("agent", "MiniMax" if cfg.minimax_available else "heuristic"),
            (
                "samples",
                str(self._requested_sample_count)
                if self._requested_sample_count is not None
                else "auto",
            ),
            ("eval_hash", eval_hash or "pending"),
            ("target", f"{cfg.target_score:.2f}"),
            ("max_iters", str(cfg.max_iterations)),
        ]
        if focus:
            lines.append(("focus", focus[:20]))

        content = "\n".join(f"[#004018]{name:<11}[/] [#39ff14]{value}[/]" for name, value in lines)
        self.query_one("#config-values", Static).update(content)

    def _update_hyperparam_panel(self, hyperparams: dict | None = None) -> None:
        hp = hyperparams if hyperparams is not None else self._hyperparams
        if not hp:
            self.query_one("#hyperparam-values", Static).update(
                "[#003010]awaiting first plan…[/]"
            )
            return
        lr = hp.get("learning_rate")
        lr_text = f"{lr:.1e}" if isinstance(lr, (int, float)) else str(lr)
        rows = [
            ("rank (r)", hp.get("r", "--")),
            ("alpha", hp.get("lora_alpha", "--")),
            ("learning_rate", lr_text),
            ("epochs", hp.get("num_epochs", "--")),
            ("batch", hp.get("batch_size", "--")),
        ]
        content = "\n".join(f"[#004018]{name:<13}[/] [#39ff14]{value}[/]" for name, value in rows)
        self.query_one("#hyperparam-values", Static).update(content)

    def _flash_hyperparams(self) -> None:
        """Pulse the hyperparameter panel border to highlight a hyperparameter change."""
        try:
            panel = self.query_one("#hyperparam-panel")
        except Exception:
            return

        def pulse(remaining: int) -> None:
            if remaining <= 0:
                panel.remove_class("flash")
                return
            panel.toggle_class("flash")
            self.set_timer(0.16, lambda: pulse(remaining - 1))

        pulse(6)

    def _update_metrics_panel(
        self,
        *,
        step: int = 0,
        total_steps: int = 0,
        loss: float | None = None,
    ) -> None:
        loss_text = "--" if loss is None else f"{loss:.4f}"
        content = "\n".join(
            [
                f"[#004018]step[/]        [#39ff14]{step}[/][#002a0c] / {total_steps or '--'}[/]",
                f"[#004018]train_loss[/]  [#39ff14]{loss_text}[/]",
                f"[#004018]baseline[/]    [#00cc33]{self._baseline:.3f}[/]",
                f"[#004018]current[/]     [#00cc33]{self._current:.3f}[/]",
                f"[#004018]best[/]        [#39ff14]{self._best:.3f}[/]",
                f"[#004018]judge[/]       [#39ff14]{self._judge_rating:.3f}[/]" if self._judge_rating is not None else "[#004018]judge[/]       [#003010]--[/]",
            ]
        )
        self.query_one("#metrics-values", Static).update(content)
        self.query_one("#loss-strip", Static).update(self._loss_sparkline())

    def _loss_sparkline(self) -> str:
        if not self._loss_values:
            return "[#003010]loss curve pending[/]"
        bars = " ▁▂▃▄▅▆▇█"
        values = self._loss_values[-24:]
        low = min(values)
        high = max(values)
        spread = max(high - low, 0.0001)
        points = []
        for value in values:
            idx = int((1.0 - ((value - low) / spread)) * (len(bars) - 1))
            points.append(bars[max(0, min(idx, len(bars) - 1))])
        return "[#00cc33]" + "".join(points) + "[/]"

    def _set_state(self, state: str, detail: str) -> None:
        self.query_one("#run-state", Static).update(state)
        self.query_one("#status-text", Static).update(detail)
        self.query_one("#mascot", Mascot).set_mood(_STATE_MOOD.get(state, "idle"))

    def _set_retrain_buttons(self, enabled: bool) -> None:
        self.query_one("#approve-retrain-button", Button).disabled = not enabled
        self.query_one("#decline-retrain-button", Button).disabled = not enabled

    def _parse_sample_count(self) -> int | None:
        raw = self.query_one("#sample-count-input", Input).value.strip()
        if not raw:
            return None
        count = int(raw)
        if not 1 <= count <= 100000:
            self._set_state("INVALID", "training sample count must be blank or between 1 and 100000")
            self._agent_log().write("[red][x] Training sample count must be blank or between 1 and 100000[/]")
            self.query_one("#sample-count-input", Input).focus()
            return 0
        return count

    def _update_score_text(self) -> None:
        self.query_one("#score-text", Static).update(
            f"baseline {self._baseline:.3f} | current {self._current:.3f} | best {self._best:.3f}"
        )
        self._update_metrics_panel()

    def _agent_log(self) -> RichLog:
        return self.query_one("#agent-log", RichLog)

    def _examples_log(self) -> RichLog:
        return self.query_one("#examples-log", RichLog)

    def _current_readable_log_path(self) -> str:
        if self._orchestrator is None:
            return ""
        logger = getattr(self._orchestrator, "_run_logger", None)
        path = getattr(logger, "readable_path", None)
        return str(path) if path else ""

    def _write_tui_exception(self, exc: Exception) -> str:
        try:
            log_dir = default_log_dir()
            log_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            path = log_dir / f"tui-error-{stamp}.log"
            path.write_text(
                "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                encoding="utf-8",
            )
            return str(path)
        except Exception:
            return ""
