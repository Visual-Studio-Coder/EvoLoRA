"""Textual TUI for EvoLoRA."""

from __future__ import annotations

import asyncio
import os
import traceback
from datetime import datetime

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Input, ProgressBar, RichLog, Select, Static

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
from evolora.voice import VoiceService


class SectionTitle(Static):
    """Small green terminal-style section label."""


class MicButton(Button):
    """Push-to-talk button: hold (mouse down) to dictate, release to stop.

    Textual doesn't deliver key-up events, but it does deliver mouse up/down, so a
    held button gives true hold-to-talk. Mouse capture keeps the release reliable even
    if the pointer drifts off the button while held.
    """

    class HoldStart(Message):
        pass

    class HoldEnd(Message):
        pass

    def on_mouse_down(self, event: events.MouseDown) -> None:
        self.capture_mouse()
        self.post_message(self.HoldStart())

    def on_mouse_up(self, event: events.MouseUp) -> None:
        self.release_mouse()
        self.post_message(self.HoldEnd())


# Maps the run-state label shown in the status bar to a mascot mood.
_STATE_MOOD = {
    "READY": "idle",
    "INVALID": "sad",
    "STARTING": "run",
    "RUNNING": "run",
    "LOCKED": "think",
    "BASELINE": "look",
    "PLANNING": "think",
    "VALIDATE": "run",
    "TRAINING": "run",
    "TRAINED": "happy",
    "EVALUATE": "look",
    "EVAL": "look",
    "JUDGE": "look",
    "JUDGED": "think",
    "DECIDE": "think",
    "APPROVE": "think",
    "APPROVED": "run",
    "DECLINED": "idle",
    "BEST": "happy",
    "ITERATION": "run",
    "STOP": "idle",
    "DONE": "happy",
    "CANCEL": "sad",
    "CANCELLED": "sad",
    "FAILED": "sad",
}


class Mascot(Static):
    """A little terminal cat that idles/blinks and reacts to the run state.

    Always animated (alive): each mood is a list of frames cycled on a timer; active
    moods also scamper across the bar. Mirrors the agent via set_mood().
    """

    # mood -> (animation frames, caption, scampers across the bar)
    MOODS = {
        "idle": (
            ["(=^.^=)", "(=^.^=)", "(=^.^=)", "(=^.^=)", "(=^.^=)", "(=^-^=)"],
            "purring",
            False,
        ),
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
        Binding("ctrl+y", "copy_log", "Copy agent log"),
        Binding("ctrl+m", "toggle_mute", "Mute voice"),
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

    #mute-button {
        min-width: 10;
        height: 1;
        margin: 0 0 0 2;
        background: #00280d;
        color: #39ff14;
        border: none;
    }

    #mute-button.muted {
        background: #120606;
        color: #aa5544;
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
        height: 18;
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

    #mic-button {
        min-width: 11;
        margin: 0 0 0 1;
        background: #00280d;
        color: #39ff14;
        border: solid #006622;
    }

    #mic-button.live {
        background: #063b30;
        color: #5ff5e0;
        border: solid #2ad6c0;
        text-style: bold;
    }

    #chat-toggle {
        min-width: 12;
        margin: 0 0 0 1;
        background: #001a14;
        color: #2ad6c0;
        border: solid #0a5a4a;
    }

    #chat-toggle.active {
        background: #063b30;
        color: #5ff5e0;
        border: solid #2ad6c0;
        text-style: bold;
    }

    #model-select {
        width: 30;
        margin: 0 0 0 1;
        display: none;
    }

    #base-model-select {
        width: 1fr;
        margin: 0 0 1 0;
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
        self._chat_mode = False
        self._chat_busy = False
        self._loss_values: list[float] = []
        self._baseline = 0.0
        self._best = 0.0
        self._current = 0.0
        self._judge_rating: float | None = None
        self._requested_sample_count: int | None = 30
        self._goal = ""
        self._approval_context: str | None = None
        self._hyperparams: dict = {}
        self._voice: VoiceService | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def compose(self) -> ComposeResult:
        with Container(id="frame"):
            with Horizontal(id="topbar"):
                yield Static("> EvoLoRA", id="brand")
                yield Mascot(id="mascot")
                yield Static("", id="clock")
                yield Button("\U0001f50a ON", id="mute-button")

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
                        yield Select(
                            [
                                ("Phi-3-mini (fast)", "unsloth/Phi-3-mini-4k-instruct"),
                                ("Llama 3.1 8B", "unsloth/Meta-Llama-3.1-8B-Instruct"),
                            ],
                            id="base-model-select",
                            allow_blank=False,
                            value=self._default_base_model(),
                        )
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
                yield Static(
                    "mock backend idle | locked eval pending | agent: heuristic/MiniMax",
                    id="status-text",
                )
                yield Static("baseline -- | best --", id="score-text")

            with Horizontal(id="inputbar"):
                yield Input(
                    placeholder="What kind of specialized model would you like to build today?",
                    id="goal-input",
                )
                yield MicButton("\U0001f3a4 HOLD", id="mic-button")
                yield Static("# samples", id="sample-label")
                yield Input(
                    value="30",
                    placeholder="auto",
                    restrict=r"[0-9]*",
                    max_length=6,
                    id="sample-count-input",
                )
                yield Select(
                    [("latest (lora_model)", "lora_model")],
                    id="model-select",
                    allow_blank=False,
                    value="lora_model",
                )
                yield Button("START", id="start-button")
                yield Button("CANCEL", id="cancel-button", disabled=True)
                yield Button("CHAT", id="chat-toggle")
                yield Button("YES", id="approve-retrain-button", disabled=True)
                yield Button("NO", id="decline-retrain-button", disabled=True)

    def on_mount(self) -> None:
        self.set_interval(1, self._update_clock)
        self._update_clock()
        self._update_config_panel()
        self._update_hyperparam_panel()
        self._update_metrics_panel()
        self.query_one("#goal-input", Input).focus()
        self._init_voice()
        self._maybe_autostart()

    def _maybe_autostart(self) -> None:
        """For unattended/recorded runs: if EVOLORA_AUTOSTART is set, pre-fill the goal +
        sample count from env and kick off the run automatically (no keyboard needed)."""
        if os.getenv("EVOLORA_AUTOSTART", "").strip().lower() not in {"1", "true", "yes", "on"}:
            return
        goal = os.getenv("EVOLORA_GOAL", "").strip()
        samples = os.getenv("EVOLORA_SAMPLES", "").strip()
        if goal:
            self.query_one("#goal-input", Input).value = goal
        if samples:
            self.query_one("#sample-count-input", Input).value = samples
        self.set_timer(2.5, self.action_start_run)

    # ------------------------------------------------------------------ voice

    def _init_voice(self) -> None:
        """Bring up voice (dictation + narrator) — best-effort, never blocks the TUI."""
        # Never start audio/network/global-key listeners inside the test suite.
        if "PYTEST_CURRENT_TEST" in os.environ:
            return
        self._loop = asyncio.get_running_loop()
        self._voice = VoiceService.create(get_config(), self._loop)
        self._voice.register_dictation_handlers(self._voice_interim, self._voice_final)
        self.run_worker(self._voice_start(), group="voice-init")

    async def _voice_start(self) -> None:
        assert self._voice is not None
        status = await self._voice.start()
        self._voice.start_ptt_key()
        self._refresh_mute_button()
        self._agent_log().write(f"[#2ad6c0][voice][/] {status}")
        if self._voice.enabled:
            ptt = get_config().ptt_key.upper()
            self._agent_log().write(
                f"[#2ad6c0][voice][/] Hold the [bold]\U0001f3a4 MIC[/] button (or {ptt}) to dictate. "
                "Narrator is live; [bold]Ctrl+M[/] or the corner button mutes all sound."
            )

    def _voice_interim(self, text: str) -> None:
        """Live transcript → goal/chat input box (called on the TUI loop)."""
        try:
            inp = self.query_one("#goal-input", Input)
            inp.value = text
            inp.cursor_position = len(text)
        except Exception:
            pass

    def _voice_final(self, text: str) -> None:
        self._voice_interim(text)
        try:
            self.query_one("#goal-input", Input).focus()
        except Exception:
            pass

    def on_mic_button_hold_start(self, event: MicButton.HoldStart) -> None:
        if not (self._voice and self._voice.enabled):
            self._agent_log().write("[#805000][mic] voice is off (no LiveKit creds / device)[/]")
            return
        self.query_one("#mic-button", MicButton).add_class("live")
        self.run_worker(self._voice.begin_dictation(), group="voice-dictation-start")

    def on_mic_button_hold_end(self, event: MicButton.HoldEnd) -> None:
        if not (self._voice and self._voice.enabled):
            return
        self.query_one("#mic-button", MicButton).remove_class("live")
        self.run_worker(self._voice.end_dictation(), group="voice-dictation-stop")

    def action_toggle_mute(self) -> None:
        if self._voice is None:
            return
        self._voice.toggle_mute()
        self._refresh_mute_button()

    def _refresh_mute_button(self) -> None:
        try:
            btn = self.query_one("#mute-button", Button)
        except Exception:
            return
        if self._voice is None or not self._voice.enabled:
            btn.label = "\U0001f507 OFF"
            btn.add_class("muted")
        elif self._voice.muted:
            btn.label = "\U0001f507 MUTED"
            btn.add_class("muted")
        else:
            btn.label = "\U0001f50a ON"
            btn.remove_class("muted")

    async def on_unmount(self) -> None:
        if self._voice is not None:
            await self._voice.aclose()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start-button":
            self.action_start_run()
        elif event.button.id == "cancel-button":
            self.action_cancel_run()
        elif event.button.id == "chat-toggle":
            self.action_toggle_chat()
        elif event.button.id == "mute-button":
            self.action_toggle_mute()
        elif event.button.id == "approve-retrain-button":
            self.action_answer_retrain(True)
        elif event.button.id == "decline-retrain-button":
            self.action_answer_retrain(False)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if self._chat_mode and event.input.id == "goal-input":
            self._send_chat(event.value.strip())
            return
        if event.input.id in {"goal-input", "sample-count-input"}:
            self.action_start_run()

    def action_toggle_chat(self) -> None:
        """Switch between training mode and chatting with the trained model.

        Disabled while a run is in progress — you can't chat mid-training.
        """
        if self._run_active:
            self._agent_log().write("[yellow][!] Can't chat while a training run is active[/]")
            return
        self._chat_mode = not self._chat_mode
        goal_input = self.query_one("#goal-input", Input)
        chat_button = self.query_one("#chat-toggle", Button)
        if self._chat_mode:
            chat_button.label = "EXIT CHAT"
            chat_button.add_class("active")
            self.query_one("#start-button", Button).disabled = True
            self.query_one("#model-select").display = True
            self.query_one("#sample-label").display = False
            self.query_one("#sample-count-input").display = False
            goal_input.placeholder = "Ask the selected model… (Enter to send)"
            goal_input.value = ""
            self._agent_log().write(
                "[bright_green][chat][/] Chat mode ON — pick a model from the dropdown and type a "
                "message. First reply loads the model (~30s)."
            )
            self._set_state("CHAT", "chatting with a trained model")
            self.run_worker(self._populate_models(), group="evolora-models", exclusive=True)
        else:
            chat_button.label = "CHAT"
            chat_button.remove_class("active")
            self.query_one("#start-button", Button).disabled = self._run_active
            self.query_one("#model-select").display = False
            self.query_one("#sample-label").display = True
            self.query_one("#sample-count-input").display = True
            goal_input.placeholder = "What kind of specialized model would you like to build today?"
            self._agent_log().write("[bright_green][chat][/] Chat mode OFF")
            self._set_state("READY", "training mode")
        goal_input.focus()

    async def _populate_models(self) -> None:
        """Fill the model dropdown with the base model + previously trained adapters."""
        cfg = get_config()
        options: list[tuple[str, str]] = [
            (f"base: {cfg.base_model_id.split('/')[-1]}", cfg.base_model_id)
        ]
        if cfg.training_backend == "remote":
            try:
                backend = get_backend("remote")
                for path in await backend.list_adapters():
                    label = (
                        "latest (lora_model)"
                        if path == "lora_model"
                        else path.replace("adapters/", "")
                    )
                    options.append((label, path))
            except Exception as exc:
                self._agent_log().write(f"[#805000]Could not list trained models: {exc}[/]")
        select = self.query_one("#model-select", Select)
        select.set_options(options)
        values = [value for _, value in options]
        select.value = "lora_model" if "lora_model" in values else values[0]

    def _selected_model(self) -> str:
        try:
            value = self.query_one("#model-select", Select).value
        except Exception:
            return "lora_model"
        if value is None or value is Select.BLANK:
            return "lora_model"
        return str(value)

    _BASE_MODELS = (
        "unsloth/Phi-3-mini-4k-instruct",
        "unsloth/Meta-Llama-3.1-8B-Instruct",
    )

    def _default_base_model(self) -> str:
        configured = get_config().base_model_id
        return configured if configured in self._BASE_MODELS else self._BASE_MODELS[0]

    def _selected_base_model(self) -> str:
        try:
            value = self.query_one("#base-model-select", Select).value
        except Exception:
            return self._default_base_model()
        if value is None or value is Select.BLANK:
            return self._default_base_model()
        return str(value)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "base-model-select":
            self._update_config_panel()

    def _send_chat(self, prompt: str) -> None:
        if not prompt or self._chat_busy:
            return
        self.query_one("#goal-input", Input).value = ""
        self._agent_log().write(f"[bold bright_green]you ›[/] {prompt}")
        self._agent_log().write("[#2a7a2a]model is thinking…[/]")
        self.run_worker(self._chat_worker(prompt), group="evolora-chat", exclusive=True)

    async def _chat_worker(self, prompt: str) -> None:
        self._chat_busy = True
        self.query_one("#goal-input", Input).disabled = True
        try:
            cfg = get_config()
            if cfg.training_backend != "remote":
                self._agent_log().write(
                    "[red][x][/] Chat needs TRAINING_BACKEND=remote (a VM with a trained adapter)."
                )
                return
            backend = get_backend("remote")
            model_dir = self._selected_model()
            reply = await backend.chat(prompt, model_dir)
            short = model_dir.replace("adapters/", "").split("/")[-1]
            self._agent_log().write(
                f"[bold cyan]model ({short}) ›[/] {reply or '(empty response)'}"
            )
        except Exception as exc:
            self._agent_log().write(f"[red][x] chat failed:[/] {exc}")
            self._agent_log().write(
                "[#805000]Tip: chat works after a real remote run produces lora_model on the VM.[/]"
            )
        finally:
            self._chat_busy = False
            if self._chat_mode:
                self.query_one("#goal-input", Input).disabled = False
                self.query_one("#goal-input", Input).focus()

    def action_start_run(self) -> None:
        if self._run_active:
            self._agent_log().write("[yellow][!] Run already in progress[/]")
            return
        if self._chat_mode:
            self._agent_log().write("[yellow][!] Exit chat mode before starting a run[/]")
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
        self.query_one("#chat-toggle", Button).disabled = True  # no chatting mid-training
        # Lock the goal + sample inputs + base-model picker for the duration of the run.
        self.query_one("#goal-input", Input).disabled = True
        self.query_one("#sample-count-input", Input).disabled = True
        self.query_one("#base-model-select", Select).disabled = True
        self._set_retrain_buttons(False)
        self._agent_log().clear()
        self._examples_log().clear()
        if self._goal:
            self._agent_log().write(
                f"[bright_green][>][/] Use case sent to agent: [bold]{self._goal}[/]"
            )
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
        if context == "keep_training":
            self._agent_log().write(
                f"[yellow][user][/] {'Keep training' if approved else 'Stop — model accepted'}"
            )
            return
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
            base_model_id=self._selected_base_model(),
            training_sample_count=self._requested_sample_count,
            goal=self._goal,
            # AUTO_APPROVE=true -> no approval gates (eval set + keep-training); fully autonomous.
            require_retrain_approval=not cfg.auto_approve,
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
            self.query_one("#chat-toggle", Button).disabled = False
            self.query_one("#goal-input", Input).disabled = False
            self.query_one("#sample-count-input", Input).disabled = False
            self.query_one("#base-model-select", Select).disabled = False
            self._set_retrain_buttons(False)
            self._update_examples_from_record()

    def _apply_event(self, event: Event) -> None:
        # Feed the live event stream to the narrator first (non-blocking, best-effort).
        # This is the ONLY coupling to voice — it can never affect the run.
        if self._voice is not None:
            self._voice.observe(event)

        kind = event.kind
        data = event.data

        if kind == EventKind.RUN_STARTED:
            mode = "MOCK" if data.get("mock") else "REAL"
            self._set_state("RUNNING", f"{mode} run {event.run_id[:8]} started")
            self._agent_log().write(
                f"[green][OK][/] EvoLoRA run started: [bold]{event.run_id[:8]}[/]"
            )
            readable_log = self._current_readable_log_path()
            if readable_log:
                self._agent_log().write(f"[cyan][log][/] Full run log: {readable_log}")
            return

        if kind == EventKind.EVAL_SET_LOCKED:
            self._set_state("LOCKED", event.message)
            self._agent_log().write(
                f"[cyan][OK][/] Locked evaluation hash: {data.get('hash', '')[:16]}..."
            )
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
            self._agent_log().write(
                f"[red][!][/] MiniMax unavailable; using heuristic fallback{detail}"
            )
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
            self._agent_log().write(
                f"[green][>][/] Training started on [bold]{data.get('backend', 'mock')}[/] backend"
            )
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
            self._agent_log().write(
                "[green][OK][/] Training complete; mock adapter artifact created"
            )
            return

        if kind == EventKind.EVAL_STARTED:
            self._set_state("EVALUATE", event.message)
            self._agent_log().write(f"[blue][>][/] {event.message}")
            return

        if kind == EventKind.EVAL_COMPLETE:
            self._current = float(data.get("score", 0.0))
            self._set_state("EVAL", event.message)
            self._agent_log().write(
                f"[blue][OK][/] Locked evaluation score: [bold]{self._current:.3f}[/]"
            )
            self._update_score_text()
            return

        if kind == EventKind.ADAPTIVE_COMPLETE:
            self._agent_log().write(
                f"[green][diag][/] Adaptive challenge score: {float(data.get('score', 0.0)):.3f}"
            )
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
            if str(data.get("approval_type", "retrain")) == "keep_training":
                self._approval_context = "keep_training"
                self._set_state("APPROVE", f"Keep training? rating {rating:.2f} | YES=more NO=stop")
                self._agent_log().write(
                    f"[yellow][?][/] Good enough (rating [bold]{rating:.2f}[/]). Keep training to make it "
                    "smarter? [bold]YES[/] = another round, [bold]NO[/] = stop here."
                )
            else:
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
        log.write(
            f"[green]iteration {latest.iteration} training data[/]: {len(latest.plan.data_spec.examples)} examples"
        )
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
            # base_model shown by the dropdown above
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
            self.query_one("#hyperparam-values", Static).update("[#003010]awaiting first plan…[/]")
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
                f"[#004018]judge[/]       [#39ff14]{self._judge_rating:.3f}[/]"
                if self._judge_rating is not None
                else "[#004018]judge[/]       [#003010]--[/]",
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
            self._set_state(
                "INVALID", "training sample count must be blank or between 1 and 100000"
            )
            self._agent_log().write(
                "[red][x] Training sample count must be blank or between 1 and 100000[/]"
            )
            self.query_one("#sample-count-input", Input).focus()
            return 0
        return count

    def _update_score_text(self) -> None:
        self.query_one("#score-text", Static).update(
            f"baseline {self._baseline:.3f} | current {self._current:.3f} | best {self._best:.3f}"
        )
        self._update_metrics_panel()

    def action_copy_log(self) -> None:
        """Copy the full agent reasoning log to the system clipboard (Ctrl+Y).

        Lets you grab the exact text (errors, decisions) to paste elsewhere, since
        the TUI's mouse capture can make manual selection awkward in some terminals.
        """
        log = self._agent_log()
        text = "\n".join(strip.text for strip in log.lines)
        self.copy_to_clipboard(text)
        self.query_one("#status-text", Static).update(
            f"[#39ff14]copied agent log to clipboard ({len(log.lines)} lines)[/]"
        )

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
