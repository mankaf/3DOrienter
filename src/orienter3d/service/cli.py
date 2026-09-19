from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from .backends import MockReconstructionBackend, ReconstructionBackend, TripoSRBackend
from .contracts import OrientationOptions, PipelineOptions, ServiceError
from .pipeline import run_pipeline

app = typer.Typer(help="Run the offline image-to-oriented-STL pipeline.")
console = Console()


def _backend(
    name: str,
    triposr_repo: Path | None,
    triposr_python: Path | None,
    triposr_model: str,
    triposr_min_free_vram_mb: int,
) -> ReconstructionBackend:
    normalized = name.lower()
    if normalized == "mock":
        return MockReconstructionBackend()
    if normalized == "triposr":
        if triposr_repo is None or triposr_python is None:
            raise typer.BadParameter("TripoSR requires --triposr-repo and --triposr-python")
        if triposr_min_free_vram_mb < 0:
            raise typer.BadParameter("TripoSR free-VRAM threshold cannot be negative")
        return TripoSRBackend(
            repository=triposr_repo,
            python_executable=triposr_python,
            pretrained_model=triposr_model,
            min_free_vram_mb=triposr_min_free_vram_mb,
        )
    raise typer.BadParameter("Backend must be 'mock' or 'triposr'")


@app.command()
def run(
    image: Annotated[Path, typer.Argument(help="PNG, JPEG, or WebP input image")],
    jobs_dir: Annotated[Path, typer.Option(help="Private root for controlled job directories")] = Path(
        "data/jobs"
    ),
    backend: Annotated[str, typer.Option(help="Reconstruction backend: mock or triposr")] = "mock",
    triposr_repo: Annotated[Path | None, typer.Option(help="Pinned TripoSR checkout")] = None,
    triposr_python: Annotated[
        Path | None, typer.Option(help="Python executable in the isolated TripoSR environment")
    ] = None,
    triposr_model: Annotated[
        str, typer.Option(help="Hugging Face model id or private local model directory")
    ] = "stabilityai/TripoSR",
    triposr_min_free_vram_mb: Annotated[
        int, typer.Option(help="Reject GPU work below this free-VRAM threshold")
    ] = 7_000,
    job_id: Annotated[str | None, typer.Option(help="Optional safe idempotency/job identifier")] = None,
    target_max_mm: Annotated[float, typer.Option(help="Scale longest mesh dimension to millimeters")] = 100.0,
    samples: Annotated[int, typer.Option(help="Random orientation candidates")] = 400,
    seed: Annotated[int, typer.Option(help="Orientation search seed")] = 42,
) -> None:
    """Generate, validate, scale, orient, and report one mesh."""
    selected_backend = _backend(
        backend,
        triposr_repo,
        triposr_python,
        triposr_model,
        triposr_min_free_vram_mb,
    )
    options = PipelineOptions(
        target_max_dimension_mm=target_max_mm,
        orientation=OrientationOptions(samples=samples, seed=seed),
    )
    try:
        execution = run_pipeline(image, jobs_dir, selected_backend, options, job_id)
    except ServiceError as exc:
        console.print(f"[red]{exc.code.value}:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    console.print(f"[green]Job succeeded:[/green] {execution.report.job_id}")
    console.print(f"Workspace: {execution.workspace_path}")
    console.print(f"Report: {execution.report_path}")


if __name__ == "__main__":
    app()
