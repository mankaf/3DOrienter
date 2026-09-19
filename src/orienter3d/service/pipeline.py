from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..mesh_io import apply_orientation, export_mesh
from ..scoring import OrientationScorer
from ..search import optimize_orientation
from .backends import ReconstructionBackend
from .contracts import (
    ArtifactManifest,
    ErrorCode,
    OrientationReport,
    PipelineEvent,
    PipelineOptions,
    PipelineReport,
    PipelineState,
    ServiceError,
)
from .image_preflight import prepare_image
from .mesh_validation import validate_and_normalize_mesh

SAFE_JOB_ID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class JobWorkspace:
    root: Path
    input_dir: Path
    generated_dir: Path
    output_dir: Path
    reports_dir: Path

    @classmethod
    def create(cls, jobs_root: Path, job_id: str) -> JobWorkspace:
        if not SAFE_JOB_ID.fullmatch(job_id):
            raise ServiceError(ErrorCode.INVALID_JOB_ID, "Job ID contains unsafe characters")
        root = jobs_root.resolve() / job_id
        if root.exists():
            raise ServiceError(ErrorCode.JOB_EXISTS, "Job workspace already exists")
        input_dir = root / "input"
        generated_dir = root / "generated"
        output_dir = root / "output"
        reports_dir = root / "reports"
        for directory in (input_dir, generated_dir, output_dir, reports_dir):
            directory.mkdir(parents=True, exist_ok=False)
        return cls(root, input_dir, generated_dir, output_dir, reports_dir)


@dataclass(frozen=True)
class PipelineExecution:
    report: PipelineReport
    workspace_path: Path
    report_path: Path


def run_pipeline(
    image_path: Path,
    jobs_root: Path,
    backend: ReconstructionBackend,
    options: PipelineOptions | None = None,
    job_id: str | None = None,
) -> PipelineExecution:
    """Run image preflight, reconstruction, mesh validation, and orientation."""
    options = options or PipelineOptions()
    job_id = job_id or uuid.uuid4().hex
    workspace = JobWorkspace.create(jobs_root, job_id)
    created_at = _now()
    events: list[PipelineEvent] = []

    def mark(state: PipelineState) -> None:
        events.append(PipelineEvent(state=state, timestamp_utc=_now()))

    mark(PipelineState.ACCEPTED)
    mark(PipelineState.PREFLIGHT)
    normalized_image = workspace.input_dir / "input.png"
    image_report = prepare_image(image_path, normalized_image, options.image_limits)

    mark(PipelineState.RECONSTRUCTING)
    raw_mesh = workspace.generated_dir / "raw.obj"
    reconstruction = backend.reconstruct(normalized_image, raw_mesh)

    mark(PipelineState.VALIDATING)
    normalized_mesh = workspace.generated_dir / "normalized.stl"
    mesh, mesh_report = validate_and_normalize_mesh(
        reconstruction.mesh_path,
        normalized_mesh,
        options.mesh_limits,
        options.target_max_dimension_mm,
    )

    mark(PipelineState.ORIENTING)
    orientation = options.orientation
    scorer = OrientationScorer(
        mesh,
        max_overhang_deg=orientation.max_overhang,
        layer_height_mm=orientation.layer_height,
    )
    results = optimize_orientation(
        mesh,
        scorer,
        random_samples=orientation.samples,
        seed=orientation.seed,
        face_candidates=orientation.face_candidates,
        yaw_steps=orientation.yaw_steps,
        keep_top=orientation.keep_top,
    )
    best = results[0]
    oriented_path = workspace.output_dir / "oriented.stl"
    export_mesh(apply_orientation(mesh, best.rotation), oriented_path)
    best_payload = best.to_dict()

    mark(PipelineState.SUCCEEDED)
    report_path = workspace.reports_dir / "pipeline-report.json"
    report = PipelineReport(
        job_id=job_id,
        state=PipelineState.SUCCEEDED,
        created_at_utc=created_at,
        completed_at_utc=_now(),
        image=image_report,
        reconstruction=reconstruction.report,
        mesh=mesh_report,
        orientation=OrientationReport(
            source=str(best_payload["source"]),
            rotation_matrix=best_payload["rotation_matrix"],
            euler_xyz_deg=best_payload["euler_xyz_deg"],
            metrics=best_payload["metrics"],
            candidates_retained=len(results),
        ),
        artifacts=ArtifactManifest(
            normalized_image="input/input.png",
            normalized_mesh="generated/normalized.stl",
            oriented_stl="output/oriented.stl",
            report="reports/pipeline-report.json",
        ),
        events=tuple(events),
    )
    temporary_report = report_path.with_suffix(".tmp")
    temporary_report.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    temporary_report.replace(report_path)
    return PipelineExecution(report, workspace.root, report_path)
