"""Textual TUI for EvoLoRA."""

from __future__ import annotations

from datetime import datetime

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Input, ProgressBar, RichLog, Static

from evolora.agent.planner import get_planner
from evolora.config import get_config
from evolora.demo.task import ADAPTIVE_EVAL_SET, LOCKED_EVAL_SET
from evolora.models.core import RunConfig
from evolora.models.events import Event, EventKind
from evolora.orchestration.orchestrator import Orchestrator
from evolora.training.backends import get_backend
from evolora.training.runner import get_runner


class SectionTitle(Static):
    """Small green terminal-style section label."""


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
        border-bottom: solid #001007;
        layout: horizontal;
        align: center middle;
    }

    #brand {
        width: 1fr;
        content-align: left middle;
        text-style: bold;
        color: #39ff14;
    }

    #clock {
        width: 22;
        content-align: right middle;
        color: #003010;
    }

    .tab {
        width: auto;
        min-width: 11;
        height: 1;
        margin: 0 1 0 0;
        padding: 0 1;
        content-align: center middle;
        color: #003d10;
        border: solid #001a07;
    }

    .tab.active {
        color: #39ff14;
        background: #00280d;
        border: solid #00641e;
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
        border: solid #001a07;
        padding: 1 2;
    }

    #agent-panel {
        height: 1fr;
    }

    #examples-panel {
        height: 1fr;
    }

    #config-panel {
        height: 18;
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
        border-top: solid #001007;
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
        border-top: solid #001507;
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
    """

    def __init__(self) -> None:
        super().__init__()
        self._orchestrator: Orchestrator | None = None
        self._run_active = False
        self._loss_values: list[float] = []
        self._baseline = 0.0
        self._best = 0.0
        self._current = 0.0
        self._requested_sample_count: int | None = 30

    def compose(self) -> ComposeResult:
        with Container(id="frame"):
            with Horizontal(id="topbar"):
                yield Static("> EvoLoRA", id="brand")
                yield Static("TRAIN", classes="tab active")
                yield Static("EVALUATE", classes="tab")
                yield Static("EXPORT", classes="tab")
                yield Static("SETTINGS", classes="tab")
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
                    max_length=3,
                    id="sample-count-input",
                )
                yield Button("START", id="start-button")
                yield Button("CANCEL", id="cancel-button", disabled=True)

    def on_mount(self) -> None:
        self.set_interval(1, self._update_clock)
        self._update_clock()
        self._seed_design_content()
        self._update_config_panel()
        self._update_metrics_panel()
        self.query_one("#goal-input", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start-button":
            self.action_start_run()
        elif event.button.id == "cancel-button":
            self.action_cancel_run()

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
        self._update_config_panel()
        self._run_active = True
        self._loss_values.clear()
        self._baseline = 0.0
        self._best = 0.0
        self._current = 0.0
        self.query_one("#start-button", Button).disabled = True
        self.query_one("#cancel-button", Button).disabled = False
        self._agent_log().clear()
        self._examples_log().clear()
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
        )

        backend = get_backend(cfg.training_backend)
        runner = get_runner(cfg.model_runner)
        planner = get_planner(
            use_minimax=cfg.minimax_available and cfg.training_backend != "mock",
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
        )

        try:
            async for event in await self._orchestrator.run():
                self._apply_event(event)
        except Exception as exc:  # pragma: no cover - defensive UI boundary
            self._set_state("FAILED", str(exc))
            self._agent_log().write(f"[red][x] TUI run failed:[/] {exc}")
        finally:
            self._run_active = False
            self.query_one("#start-button", Button).disabled = False
            self.query_one("#cancel-button", Button).disabled = True
            self._update_examples_from_record()

    def _apply_event(self, event: Event) -> None:
        kind = event.kind
        data = event.data

        if kind == EventKind.RUN_STARTED:
            mode = "MOCK" if data.get("mock") else "REAL"
            self._set_state("RUNNING", f"{mode} run {event.run_id[:8]} started")
            self._agent_log().write(f"[green][OK][/] EvoLoRA run started: [bold]{event.run_id[:8]}[/]")
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
            self._agent_log().write("[red][!][/] MiniMax unavailable; using heuristic fallback")
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

    def _seed_design_content(self) -> None:
        agent = self._agent_log()
        examples = self._examples_log()
        agent.write("[green][OK][/] Parsed use case description")
        agent.write("[green][OK][/] Domain identified: structured customer spending summary")
        agent.write("[green][OK][/] Locked benchmark and adaptive diagnostics are separate")
        agent.write("[bright_green][>][/] Ready to synthesize targeted training examples")
        agent.write("[#003a10]    JSON output | numeric totals | no markdown fences[/]")

        examples.write("[#003a10]#001[/] [#005a1e]user >[/] Summarize customers and purchases as strict JSON")
        examples.write("[#003a10]#002[/] [#005a1e]asst >[/] {top_customer, top_customer_total, customer_count, total_revenue, summary}")
        examples.write("[#003a10]#003[/] [#005a1e]rule >[/] adaptive challenge scores stay diagnostic, not official")

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
            ("runner", cfg.model_runner),
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

    def _parse_sample_count(self) -> int | None:
        raw = self.query_one("#sample-count-input", Input).value.strip()
        if not raw:
            return None
        count = int(raw)
        if not 1 <= count <= 500:
            self._set_state("INVALID", "training sample count must be blank or between 1 and 500")
            self._agent_log().write("[red][x] Training sample count must be blank or between 1 and 500[/]")
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
