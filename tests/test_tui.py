"""Regression coverage for the Textual app input wiring."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Input

import evolora.tui.app as tui_app
from evolora.config import Config
from evolora.models.core import RunConfig
from evolora.models.events import Event, EventKind
from evolora.tui.app import EvoLoRAApp


async def _empty_events():
    if False:
        yield None


@pytest.fixture
def captured_tui_configs(monkeypatch: pytest.MonkeyPatch) -> list[RunConfig]:
    captured: list[RunConfig] = []

    class SpyOrchestrator:
        def __init__(self, *, config: RunConfig, **_: object) -> None:
            captured.append(config)
            self._record = SimpleNamespace(config=config, iterations=[])

        async def run(self):
            return _empty_events()

        def cancel(self) -> None:
            pass

    monkeypatch.setattr(tui_app, "Orchestrator", SpyOrchestrator)
    monkeypatch.setattr(
        tui_app,
        "get_config",
        lambda: Config(max_iterations=1, training_backend="mock", model_runner="mock"),
    )
    monkeypatch.setattr(tui_app, "get_backend", lambda _name: object())
    monkeypatch.setattr(tui_app, "get_runner", lambda _name: object())
    monkeypatch.setattr(tui_app, "get_planner", lambda **_: object())
    return captured


async def _start_tui_run(
    captured: list[RunConfig],
    *,
    sample_count_value: str | None = None,
    goal: str = "",
) -> RunConfig:
    app = EvoLoRAApp()

    async with app.run_test(size=(120, 40)) as pilot:
        sample_input = app.query_one("#sample-count-input", Input)
        assert sample_input.value == "30"

        if sample_count_value is not None:
            sample_input.value = sample_count_value
        app.query_one("#goal-input", Input).value = goal

        app.action_start_run()
        for _ in range(20):
            await pilot.pause(0.05)
            if captured:
                break

    assert captured
    return captured[-1]


@pytest.mark.asyncio
async def test_tui_default_sample_count_and_goal_feed_run_config(
    captured_tui_configs: list[RunConfig],
) -> None:
    config = await _start_tui_run(
        captured_tui_configs,
        goal="summarize customer spend as strict JSON",
    )

    assert config.training_sample_count == 30
    assert config.goal == "summarize customer spend as strict JSON"
    assert config.require_retrain_approval is True


@pytest.mark.asyncio
async def test_tui_blank_sample_count_leaves_choice_to_agent(
    captured_tui_configs: list[RunConfig],
) -> None:
    config = await _start_tui_run(
        captured_tui_configs,
        sample_count_value="",
    )

    assert config.training_sample_count is None


@pytest.mark.asyncio
async def test_tui_eval_approval_event_enables_yes_no_buttons() -> None:
    app = EvoLoRAApp()

    async with app.run_test(size=(120, 40)):
        app._apply_event(
            Event(
                kind=EventKind.EVAL_APPROVAL_REQUIRED,
                run_id="run-1",
                message="Approve generated evals?",
                data={"evals": [{"input": "q", "expected": '{"ok": true}'}]},
            )
        )

        assert app._approval_context == "evals"
        assert app.query_one("#approve-retrain-button", Button).disabled is False
        assert app.query_one("#decline-retrain-button", Button).disabled is False


@pytest.mark.asyncio
async def test_tui_validation_event_updates_hyperparameter_pane() -> None:
    from textual.widgets import Static

    app = EvoLoRAApp()
    hp = {"r": 16, "lora_alpha": 32, "learning_rate": 1e-4, "num_epochs": 3, "batch_size": 1}

    async with app.run_test(size=(120, 40)):
        app._apply_event(
            Event(
                kind=EventKind.VALIDATION_COMPLETE,
                run_id="run-1",
                message="Plan validated: 30 examples",
                data={"hyperparams": hp},
            )
        )

        assert app._hyperparams == hp
        rendered = str(app.query_one("#hyperparam-values", Static).render())
        assert "16" in rendered  # rank
        assert "1.0e-04" in rendered  # learning_rate formatted


@pytest.mark.asyncio
async def test_tui_copy_log_action_copies_agent_log(monkeypatch) -> None:
    app = EvoLoRAApp()
    copied: list[str] = []

    async with app.run_test(size=(120, 40)):
        monkeypatch.setattr(app, "copy_to_clipboard", lambda text: copied.append(text))
        app._agent_log().write("distinctive reasoning line 42")
        app.action_copy_log()

    assert copied
    assert "distinctive reasoning line 42" in copied[0]


@pytest.mark.asyncio
async def test_tui_chat_toggle_flips_mode_and_is_blocked_during_run() -> None:
    app = EvoLoRAApp()

    async with app.run_test(size=(120, 40)):
        chat_button = app.query_one("#chat-toggle", Button)

        app.action_toggle_chat()
        assert app._chat_mode is True
        assert str(chat_button.label) == "EXIT CHAT"
        assert app.query_one("#start-button", Button).disabled is True

        app.action_toggle_chat()
        assert app._chat_mode is False
        assert str(chat_button.label) == "CHAT"

        # While a run is active, toggling chat is refused (no chatting mid-training).
        app._run_active = True
        app.action_toggle_chat()
        assert app._chat_mode is False


@pytest.mark.asyncio
async def test_tui_model_dropdown_lists_trained_models(monkeypatch) -> None:
    from textual.widgets import Select

    monkeypatch.setattr(
        tui_app,
        "get_config",
        lambda: Config(training_backend="remote", base_model_id="org/Phi-3-mini"),
    )

    class FakeBackend:
        async def list_adapters(self):
            return ["lora_model", "adapters/sql-abc"]

    monkeypatch.setattr(tui_app, "get_backend", lambda _name: FakeBackend())

    app = EvoLoRAApp()
    async with app.run_test(size=(120, 40)):
        await app._populate_models()
        select = app.query_one("#model-select", Select)
        assert select.value == "lora_model"  # defaults to the latest trained model
        select.value = "adapters/sql-abc"  # a populated adapter option is selectable
        assert app._selected_model() == "adapters/sql-abc"


def test_tui_exception_writer_saves_traceback(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tui_app, "default_log_dir", lambda: tmp_path)
    app = EvoLoRAApp()

    path = app._write_tui_exception(RuntimeError("giant tui error"))

    assert path
    assert Path(path).parent == tmp_path
    text = Path(path).read_text(encoding="utf-8")
    assert "RuntimeError: giant tui error" in text
