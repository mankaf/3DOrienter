from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console
from rich.table import Table
from scipy.spatial.transform import Rotation

from .ai_optimizer import (
    build_finalists,
    build_prompt,
    query_lmstudio,
    resolve_lmstudio_settings,
    utility,
)
from .mesh_io import MeshLoadError, apply_orientation, export_mesh, load_mesh
from .scoring import OrientationScorer, ScoreWeights
from .search import candidate_rotations, optimize_orientation

app = typer.Typer(
    name="3dorienter",
    no_args_is_help=True,
    help="Optimize STL orientation for FDM support removal and surface quality.",
)
console = Console()


def _build_scorer(
    mesh,
    max_overhang: float,
    layer_height: float,
    support_weight: float,
    quality_weight: float,
    height_weight: float,
    stability_weight: float,
) -> OrientationScorer:
    return OrientationScorer(
        mesh,
        max_overhang_deg=max_overhang,
        layer_height_mm=layer_height,
        weights=ScoreWeights(
            support=support_weight,
            quality=quality_weight,
            height=height_weight,
            stability=stability_weight,
        ),
    )


def _read_mesh_or_exit(path: Path):
    if not path.exists():
        console.print(f"[red]Input file not found:[/red] {path}")
        raise typer.Exit(code=2)
    try:
        return load_mesh(path)
    except MeshLoadError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc


def _print_metrics(title: str, metrics) -> None:
    table = Table(title=title)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    values = {
        "total score (lower is better)": f"{metrics.score:.6f}",
        "support cost": f"{metrics.support_cost:.6f}",
        "weighted overhang area": f"{metrics.overhang_area_mm2:.2f} mm²",
        "support volume proxy": f"{metrics.support_volume_proxy_mm3:.2f} mm³",
        "surface quality cost": f"{metrics.quality_cost:.6f}",
        "height": f"{metrics.height_mm:.2f} mm",
        "bed contact ratio": f"{metrics.bed_contact_ratio:.4f}",
    }
    for key, value in values.items():
        table.add_row(key, value)
    console.print(table)


@app.command()
def analyze(
    input_file: Annotated[Path, typer.Argument(help="Input STL/OBJ/PLY/3MF file")],
    rotation_xyz: Annotated[
        tuple[float, float, float],
        typer.Option("--rotation", help="XYZ Euler rotation in degrees"),
    ] = (0.0, 0.0, 0.0),
    max_overhang: Annotated[float, typer.Option(help="Allowed overhang angle in degrees")] = 45.0,
    layer_height: Annotated[float, typer.Option(help="Layer height in millimeters")] = 0.2,
) -> None:
    """Score one specified orientation without changing the file."""
    mesh = _read_mesh_or_exit(input_file)
    scorer = _build_scorer(mesh, max_overhang, layer_height, 0.65, 0.25, 0.05, 0.05)
    matrix = Rotation.from_euler("xyz", rotation_xyz, degrees=True).as_matrix()
    _print_metrics("Orientation analysis", scorer.evaluate(matrix))


@app.command()
def optimize(
    input_file: Annotated[Path, typer.Argument(help="Input STL/OBJ/PLY/3MF file")],
    output_file: Annotated[Path, typer.Option("--output", "-o", help="Oriented output file")],
    report_file: Annotated[
        Path | None, typer.Option("--report", help="Optional JSON report path")
    ] = None,
    samples: Annotated[int, typer.Option(help="Uniform random SO(3) candidates")] = 400,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    face_candidates: Annotated[
        int, typer.Option(help="Largest faces to try against the build plate")
    ] = 24,
    yaw_steps: Annotated[int, typer.Option(help="Yaw variants per face candidate")] = 4,
    max_overhang: Annotated[float, typer.Option(help="Allowed overhang angle in degrees")] = 45.0,
    layer_height: Annotated[float, typer.Option(help="Layer height in millimeters")] = 0.2,
    support_weight: Annotated[float, typer.Option(help="Weight for support burden")] = 0.65,
    quality_weight: Annotated[float, typer.Option(help="Weight for surface quality")] = 0.25,
    height_weight: Annotated[float, typer.Option(help="Weight for print height")] = 0.05,
    stability_weight: Annotated[float, typer.Option(help="Weight for bed contact")] = 0.05,
) -> None:
    """Search candidate rotations and export the best reoriented mesh."""
    if samples < 0 or face_candidates < 0 or yaw_steps < 1:
        console.print("[red]samples and face-candidates must be non-negative; yaw-steps >= 1[/red]")
        raise typer.Exit(code=2)

    mesh = _read_mesh_or_exit(input_file)
    try:
        scorer = _build_scorer(
            mesh,
            max_overhang,
            layer_height,
            support_weight,
            quality_weight,
            height_weight,
            stability_weight,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    results = optimize_orientation(
        mesh,
        scorer,
        random_samples=samples,
        seed=seed,
        face_candidates=face_candidates,
        yaw_steps=yaw_steps,
        keep_top=10,
    )
    best = results[0]
    oriented = apply_orientation(mesh, best.rotation)
    export_mesh(oriented, output_file)

    console.print(f"[green]Saved oriented mesh:[/green] {output_file}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        euler = Rotation.from_matrix(best.rotation).as_euler("xyz", degrees=True)
    console.print("Best rotation XYZ: " + ", ".join(f"{value:.3f}°" for value in euler.tolist()))
    console.print(f"Candidate source: {best.source}")
    _print_metrics("Best orientation", best.metrics)

    if report_file is not None:
        payload = {
            "input_file": str(input_file),
            "output_file": str(output_file),
            "settings": {
                "samples": samples,
                "seed": seed,
                "face_candidates": face_candidates,
                "yaw_steps": yaw_steps,
                "max_overhang_deg": max_overhang,
                "layer_height_mm": layer_height,
                "weights": {
                    "support": support_weight,
                    "quality": quality_weight,
                    "height": height_weight,
                    "stability": stability_weight,
                },
            },
            "best": best.to_dict(),
            "top_candidates": [result.to_dict() for result in results],
        }
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]Saved report:[/green] {report_file}")


_AI_OBJECTIVES = {
    "balanced": {
        "quality": 0.30,
        "support_removal": 0.25,
        "strength": 0.25,
        "flexibility": 0.10,
        "print_time": 0.10,
    },
    "quality": {
        "quality": 0.55,
        "support_removal": 0.20,
        "strength": 0.15,
        "flexibility": 0.05,
        "print_time": 0.05,
    },
    "easy_supports": {
        "quality": 0.25,
        "support_removal": 0.50,
        "strength": 0.15,
        "flexibility": 0.05,
        "print_time": 0.05,
    },
    "strength": {
        "quality": 0.15,
        "support_removal": 0.15,
        "strength": 0.55,
        "flexibility": 0.05,
        "print_time": 0.10,
    },
    "flexibility": {
        "quality": 0.15,
        "support_removal": 0.15,
        "strength": 0.15,
        "flexibility": 0.45,
        "print_time": 0.10,
    },
}


@app.command("ai-optimize")
def ai_optimize(
    input_file: Annotated[Path, typer.Argument(help="Input STL/OBJ/PLY/3MF file")],
    output_file: Annotated[Path, typer.Option("--output", "-o", help="Oriented output file")],
    report_file: Annotated[
        Path | None, typer.Option("--report", help="Optional JSON report path")
    ] = None,
    priority: Annotated[
        str,
        typer.Option(help="balanced, quality, easy-supports, strength, or flexibility"),
    ] = "balanced",
    support_types: Annotated[
        str, typer.Option(help="Comma-separated support choices: normal,tree")
    ] = "normal,tree",
    load_direction: Annotated[
        tuple[float, float, float],
        typer.Option("--load-direction", help="Load direction XYZ in model coordinates"),
    ] = (0.0, 0.0, 1.0),
    force_n: Annotated[
        float, typer.Option("--force", help="Approximate applied force in newtons")
    ] = 250.0,
    walls: Annotated[int, typer.Option(help="Wall/perimeter count")] = 4,
    infill_percent: Annotated[float, typer.Option("--infill", help="Infill percentage")] = 25.0,
    tpu_fraction: Annotated[
        float, typer.Option(help="TPU fraction from 0.0 to 1.0 for PLA+TPU printing")
    ] = 0.0,
    samples: Annotated[int, typer.Option(help="Uniform random SO(3) candidates")] = 400,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    face_candidates: Annotated[
        int, typer.Option(help="Largest faces to try against the build plate")
    ] = 24,
    yaw_steps: Annotated[int, typer.Option(help="Yaw variants per face candidate")] = 4,
    max_overhang: Annotated[float, typer.Option(help="Allowed overhang angle in degrees")] = 45.0,
    layer_height: Annotated[float, typer.Option(help="Layer height in millimeters")] = 0.2,
    lmstudio_url: Annotated[
        str | None, typer.Option("--lmstudio-url", help="LM Studio server base URL")
    ] = None,
    lmstudio_model: Annotated[
        str | None, typer.Option("--lmstudio-model", help="Loaded LM Studio model ID")
    ] = None,
    max_regret: Annotated[
        float,
        typer.Option(help="Maximum simulator utility loss allowed before overriding the AI"),
    ] = 0.01,
) -> None:
    """Use LM Studio to rank simulated FDM orientation and support candidates."""
    priority_key = priority.strip().lower().replace("-", "_")
    if priority_key not in _AI_OBJECTIVES:
        console.print(
            "[red]priority must be balanced, quality, easy-supports, strength, or flexibility[/red]"
        )
        raise typer.Exit(code=2)

    parsed_supports = tuple(
        item.strip().lower() for item in support_types.split(",") if item.strip()
    )
    if not parsed_supports or any(item not in {"normal", "tree"} for item in parsed_supports):
        console.print("[red]support-types must contain only normal and/or tree[/red]")
        raise typer.Exit(code=2)
    parsed_supports = tuple(dict.fromkeys(parsed_supports))

    if (
        samples < 0
        or face_candidates < 0
        or yaw_steps < 1
        or force_n < 0
        or walls < 1
        or not 0.0 <= infill_percent <= 100.0
        or not 0.0 <= tpu_fraction <= 1.0
        or max_regret < 0
        or float(np.linalg.norm(load_direction)) <= 0
    ):
        console.print("[red]Invalid AI optimization settings[/red]")
        raise typer.Exit(code=2)

    mesh = _read_mesh_or_exit(input_file)
    objective = dict(_AI_OBJECTIVES[priority_key])
    try:
        scorer = _build_scorer(mesh, max_overhang, layer_height, 0.65, 0.25, 0.05, 0.05)
        finalists = build_finalists(
            mesh,
            scorer,
            samples=samples,
            seed=seed,
            face_candidates=face_candidates,
            yaw_steps=yaw_steps,
            support_types=parsed_supports,
            layer_height_mm=layer_height,
            load_direction_model=load_direction,
            force_n=force_n,
            walls=walls,
            infill_percent=infill_percent,
            tpu_fraction=tpu_fraction,
            objective=objective,
        )
        prompt = build_prompt(
            scenario=priority_key,
            load_direction_model=load_direction,
            force_n=force_n,
            walls=walls,
            infill_percent=infill_percent,
            tpu_fraction=tpu_fraction,
            objective=objective,
            candidates=finalists,
        )
        url, model_name = resolve_lmstudio_settings(lmstudio_url, lmstudio_model)
        ai_candidate_id, raw_output = query_lmstudio(
            url=url,
            model=model_name,
            prompt=prompt,
        )
    except (ValueError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    if not 0 <= ai_candidate_id < len(finalists):
        console.print(f"[red]LM Studio selected invalid candidate {ai_candidate_id}[/red]")
        raise typer.Exit(code=2)

    ai_choice = finalists[ai_candidate_id]
    simulator_best = max(finalists, key=lambda candidate: utility(candidate, objective))
    ai_utility = utility(ai_choice, objective)
    best_utility = utility(simulator_best, objective)
    regret = best_utility - ai_utility
    overridden = regret > max_regret
    chosen = simulator_best if overridden else ai_choice

    rotation = np.asarray(chosen["rotation_matrix"], dtype=float)
    oriented = apply_orientation(mesh, rotation)
    export_mesh(oriented, output_file)

    console.print(f"[green]Saved oriented mesh:[/green] {output_file}")
    console.print(f"LM Studio model: {model_name}")
    console.print(f"AI candidate: {ai_candidate_id}")
    console.print(f"Selected candidate: {chosen['candidate_id']}")
    console.print(
        "Verification: "
        + ("[yellow]simulator override[/yellow]" if overridden else "[green]AI accepted[/green]")
    )
    console.print(f"AI utility regret: {regret:.6f}")

    table_title = "Simulator-verified final print plan" if overridden else "AI-selected print plan"
    table = Table(title=table_title)
    table.add_column("Setting")
    table.add_column("Value", justify="right")
    table.add_row(
        "rotation XYZ",
        ", ".join(f"{float(value):.3f}°" for value in chosen["rotation_xyz_deg"]),
    )
    table.add_row("support type", str(chosen["support_type"]))
    table.add_row("support density", f"{float(chosen['support_density_percent']):.1f}%")
    table.add_row("interface layers", str(chosen["interface_layers"]))
    table.add_row("Z gap", f"{float(chosen['z_gap_mm']):.2f} mm")
    table.add_row("quality score", f"{float(chosen['quality_score']):.6f}")
    table.add_row("strength score", f"{float(chosen['strength_score']):.6f}")
    table.add_row("flexibility score", f"{float(chosen['flexibility_score']):.6f}")
    table.add_row("support removal cost", f"{float(chosen['support_removal_cost']):.6f}")
    table.add_row(
        "support material proxy",
        f"{float(chosen['support_material_proxy_mm3']):.2f} mm³",
    )
    table.add_row("print time proxy", f"{float(chosen['print_time_proxy']):.2f}")
    console.print(table)

    if report_file is not None:
        payload = {
            "input_file": str(input_file),
            "output_file": str(output_file),
            "lmstudio": {
                "url": url,
                "model": model_name,
                "raw_output": raw_output,
            },
            "settings": {
                "priority": priority_key,
                "objective": objective,
                "support_types": list(parsed_supports),
                "load_direction_model": list(load_direction),
                "force_n": force_n,
                "walls": walls,
                "infill_percent": infill_percent,
                "tpu_fraction": tpu_fraction,
                "samples": samples,
                "seed": seed,
                "face_candidates": face_candidates,
                "yaw_steps": yaw_steps,
                "max_overhang_deg": max_overhang,
                "layer_height_mm": layer_height,
                "max_regret": max_regret,
            },
            "ai_candidate_id": ai_candidate_id,
            "simulator_best_candidate_id": simulator_best["candidate_id"],
            "ai_utility": ai_utility,
            "simulator_best_utility": best_utility,
            "utility_regret": regret,
            "simulator_override": overridden,
            "decision_source": "simulator_override" if overridden else "ai",
            "ai_recommendation": ai_choice,
            "simulator_best": simulator_best,
            "final_plan": chosen,
            "chosen": chosen,
            "finalists": finalists,
        }
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]Saved report:[/green] {report_file}")


@app.command("make-dataset")
def make_dataset(
    input_dir: Annotated[Path, typer.Argument(help="Directory containing mesh files")],
    output_jsonl: Annotated[Path, typer.Option("--output", "-o", help="JSONL dataset path")],
    samples_per_model: Annotated[int, typer.Option(help="Random candidates per model")] = 300,
    seed: Annotated[int, typer.Option(help="Base random seed")] = 42,
    face_candidates: Annotated[int, typer.Option(help="Largest face-aligned candidates")] = 24,
    yaw_steps: Annotated[int, typer.Option(help="Yaw variants per face candidate")] = 4,
    max_overhang: Annotated[float, typer.Option(help="Allowed overhang angle in degrees")] = 45.0,
    layer_height: Annotated[float, typer.Option(help="Layer height in millimeters")] = 0.2,
) -> None:
    """Generate orientation/score labels for later machine-learning experiments."""
    suffixes = {".stl", ".obj", ".ply", ".3mf", ".off", ".glb", ".gltf"}
    files = sorted(path for path in input_dir.rglob("*") if path.suffix.lower() in suffixes)
    if not files:
        console.print(f"[red]No supported mesh files found in:[/red] {input_dir}")
        raise typer.Exit(code=2)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for model_index, path in enumerate(files):
            try:
                mesh = load_mesh(path)
                scorer = _build_scorer(mesh, max_overhang, layer_height, 0.65, 0.25, 0.05, 0.05)
                candidates = candidate_rotations(
                    mesh,
                    random_samples=samples_per_model,
                    seed=seed + model_index,
                    face_candidates=face_candidates,
                    yaw_steps=yaw_steps,
                )
            # A malformed file must not abort a long multi-file batch. The exception is
            # reported with its source path, and processing continues with the next mesh.
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                console.print(f"[yellow]Skipping {path}: {exc}[/yellow]")
                continue

            for candidate_index, (matrix, source) in enumerate(candidates):
                metrics = scorer.evaluate(matrix)
                record = {
                    "model": str(path.relative_to(input_dir)),
                    "candidate_index": candidate_index,
                    "source": source,
                    "rotation_matrix": np.asarray(matrix).tolist(),
                    "quaternion_xyzw": Rotation.from_matrix(matrix).as_quat().tolist(),
                    "metrics": metrics.to_dict(),
                }
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                written += 1

    console.print(f"[green]Wrote {written} labeled orientations:[/green] {output_jsonl}")
    if skipped:
        console.print(f"[yellow]Skipped {skipped} unreadable model(s).[/yellow]")


if __name__ == "__main__":
    app()
