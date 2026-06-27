"""EvoLoRA CLI — entry point for all commands."""

from __future__ import annotations

import asyncio
import sys

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="evolora", help="EvoLoRA: auditable bounded self-improvement loop.")
console = Console()


@app.command()
def doctor() -> None:
    """Check environment and service connectivity."""
    from evolora.config import get_config

    cfg = get_config()
    table = Table(title="EvoLoRA Doctor", show_header=True)
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")

    def row(name, ok, detail=""):
        status = "[green]OK[/green]" if ok else "[red]MISSING[/red]"
        table.add_row(name, status, detail)

    row("Python >= 3.11", sys.version_info >= (3, 11), sys.version.split()[0])
    row("MINIMAX_API_KEY", cfg.minimax_available, "set" if cfg.minimax_available else "not set — mock mode")
    row(
        "DIGITAL_OCEAN_MODEL_ACCESS_KEY",
        cfg.digital_ocean_judge_available,
        "set" if cfg.digital_ocean_judge_available else "not set — heuristic judge",
    )
    row("MONGODB_URI", cfg.mongo_available, "set" if cfg.mongo_available else "not set — in-memory")
    row("TRAINING_BACKEND", True, cfg.training_backend)
    row("MODEL_RUNNER", True, cfg.model_runner)
    row("ARTIFACT_DIR", True, cfg.artifact_dir)

    console.print(table)

    if not cfg.minimax_available:
        console.print("[yellow]MiniMax not configured — heuristic planner will be used.[/yellow]")
    if not cfg.digital_ocean_judge_available:
        console.print("[yellow]DigitalOcean judge not configured — heuristic judge will be used.[/yellow]")
    if not cfg.mongo_available:
        console.print("[yellow]MongoDB not configured — using in-memory store.[/yellow]")


@app.command()
def demo(
    mock: bool = typer.Option(True, "--mock/--no-mock", help="Force mock backend"),
    iterations: int = typer.Option(3, help="Max iterations"),
) -> None:
    """Run a demo EvoLoRA loop (mock by default)."""
    asyncio.run(_demo(mock=mock, iterations=iterations))


async def _demo(mock: bool, iterations: int) -> None:
    import os

    from evolora.agent.planner import get_planner
    from evolora.config import get_config
    from evolora.demo.task import ADAPTIVE_EVAL_SET, LOCKED_EVAL_SET
    from evolora.evaluation.digitalocean_judge import get_judge
    from evolora.models.core import RunConfig
    from evolora.orchestration.orchestrator import Orchestrator
    from evolora.orchestration.retrain_advisor import get_retrain_advisor
    from evolora.training.backends import get_backend
    from evolora.training.runner import get_runner

    cfg = get_config()
    if mock:
        os.environ["TRAINING_BACKEND"] = "mock"

    run_config = RunConfig(
        max_iterations=iterations,
        target_score=cfg.target_score,
        training_backend="mock" if mock else cfg.training_backend,
        model_runner="mock" if mock else cfg.model_runner,
        base_model_id=cfg.base_model_id,
    )

    backend = get_backend("mock" if mock else cfg.training_backend)
    runner = get_runner("mock" if mock else cfg.model_runner)
    planner = get_planner(
        use_minimax=cfg.minimax_available and not mock,
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
        api_key=cfg.minimax_api_key if not mock else "",
        model=cfg.minimax_model,
        base_url=cfg.minimax_base_url,
    )

    orch = Orchestrator(
        config=run_config,
        eval_set=LOCKED_EVAL_SET,
        planner=planner,
        training_backend=backend,
        model_runner=runner,
        adaptive_eval_set=ADAPTIVE_EVAL_SET,
        judge=judge,
        retrain_advisor=retrain_advisor,
    )

    mode = "[yellow][MOCK][/yellow]" if mock else "[cyan][REAL][/cyan]"
    console.rule(f"EvoLoRA Demo {mode}")

    async for event in await orch.run():
        _print_event(event)

    console.rule("Done")


def _print_event(event) -> None:
    from evolora.models.events import EventKind

    labels = {
        EventKind.RUN_STARTED: "[bold green]START   [/]",
        EventKind.EVAL_SET_LOCKED: "[bold cyan]LOCK    [/]",
        EventKind.BASELINE_COMPLETE: "[bold blue]BASELINE[/]",
        EventKind.PLANNING_STARTED: "[bold yellow]PLAN    [/]",
        EventKind.PLAN_RECEIVED: "[bold yellow]PLAN OK [/]",
        EventKind.AGENT_FALLBACK_USED: "[bold red]FALLBACK[/]",
        EventKind.TRAINING_STARTED: "[bold]TRAIN   [/]",
        EventKind.TRAINING_PROGRESS: None,  # skip progress spam in CLI
        EventKind.TRAINING_COMPLETE: "[bold green]TRAINED [/]",
        EventKind.EVAL_COMPLETE: "[bold blue]EVAL    [/]",
        EventKind.ADAPTIVE_COMPLETE: "[bold]ADAPTIVE[/]",
        EventKind.JUDGE_STARTED: "[bold cyan]JUDGE   [/]",
        EventKind.JUDGE_COMPLETE: "[bold cyan]JUDGED  [/]",
        EventKind.RETRAIN_DECISION_RECEIVED: "[bold yellow]DECIDE  [/]",
        EventKind.USER_APPROVAL_REQUIRED: "[bold yellow]APPROVE [/]",
        EventKind.USER_APPROVAL_RECEIVED: "[bold yellow]ANSWER  [/]",
        EventKind.ITERATION_COMPLETE: "[bold]ITER    [/]",
        EventKind.BEST_UPDATED: "[bold green]BEST    [/]",
        EventKind.STOP_CONDITION_MET: "[bold red]STOP    [/]",
        EventKind.RUN_COMPLETE: "[bold green]DONE    [/]",
        EventKind.RUN_FAILED: "[bold red]FAILED  [/]",
        EventKind.RUN_CANCELLED: "[bold red]CANCEL  [/]",
        EventKind.LOG: "        ",
    }
    label = labels.get(event.kind)
    if label is None:
        return
    console.print(f"{label} {event.message}")


@app.command("smoke-minimax")
def smoke_minimax() -> None:
    """Smoke-test MiniMax connectivity (requires MINIMAX_API_KEY)."""
    asyncio.run(_smoke_minimax())


async def _smoke_minimax() -> None:
    from evolora.config import get_config

    cfg = get_config()
    if not cfg.minimax_available:
        console.print("[red]MINIMAX_API_KEY not set.[/red]")
        raise typer.Exit(1)
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=cfg.minimax_api_key,
            base_url=cfg.minimax_base_url,
        )
        resp = await client.chat.completions.create(
            model=cfg.minimax_model,
            messages=[
                {"role": "user", "content": 'Reply with {"ok": true}'},
            ],
            max_tokens=20,
        )
        content = resp.choices[0].message.content or ""
        console.print(f"[green]MiniMax OK ({cfg.minimax_model}):[/green] {content}")
    except Exception as exc:
        console.print(f"[red]MiniMax FAILED:[/red] {exc}")
        raise typer.Exit(1)


@app.command("history")
def history(limit: int = typer.Option(10, help="Number of runs to show")) -> None:
    """Show run history (in-memory only until MongoDB is wired)."""
    console.print("[yellow]History requires a persistent RunStore. Wire MongoDB to enable.[/yellow]")


@app.command("tui")
def tui() -> None:
    """Launch the Textual TUI (built by Codex)."""
    try:
        from evolora.tui.app import EvoLoRAApp  # type: ignore

        EvoLoRAApp().run()
    except ImportError:
        console.print("[yellow]TUI not yet built — Codex is on it.[/yellow]")


if __name__ == "__main__":
    app()
