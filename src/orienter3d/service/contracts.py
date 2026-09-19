from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PipelineState(str, Enum):
    ACCEPTED = "accepted"
    PREFLIGHT = "preflight"
    RECONSTRUCTING = "reconstructing"
    VALIDATING = "validating"
    ORIENTING = "orienting"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ErrorCode(str, Enum):
    INVALID_JOB_ID = "invalid_job_id"
    JOB_EXISTS = "job_exists"
    IMAGE_NOT_FOUND = "image_not_found"
    IMAGE_TOO_LARGE = "image_too_large"
    IMAGE_DIMENSIONS_EXCEEDED = "image_dimensions_exceeded"
    INVALID_IMAGE = "invalid_image"
    BACKEND_NOT_CONFIGURED = "backend_not_configured"
    BACKEND_BUSY = "backend_busy"
    BACKEND_FAILED = "backend_failed"
    BACKEND_TIMEOUT = "backend_timeout"
    MESH_INVALID = "mesh_invalid"
    MESH_TOO_COMPLEX = "mesh_too_complex"
    PIPELINE_FAILED = "pipeline_failed"


class ServiceError(RuntimeError):
    """Expected pipeline failure with a stable machine-readable code."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class ImageLimits(FrozenModel):
    max_file_bytes: int = Field(default=20 * 1024 * 1024, ge=1)
    max_width: int = Field(default=4096, ge=1)
    max_height: int = Field(default=4096, ge=1)
    allowed_formats: tuple[str, ...] = ("JPEG", "PNG", "WEBP")


class MeshLimits(FrozenModel):
    max_file_bytes: int = Field(default=100 * 1024 * 1024, ge=1)
    max_vertices: int = Field(default=2_000_000, ge=3)
    max_faces: int = Field(default=4_000_000, ge=1)
    max_components: int = Field(default=64, ge=1)
    max_degenerate_face_ratio: float = Field(default=0.10, ge=0.0, le=1.0)


class OrientationOptions(FrozenModel):
    samples: int = Field(default=400, ge=0, le=20_000)
    seed: int = 42
    face_candidates: int = Field(default=24, ge=0, le=1_000)
    yaw_steps: int = Field(default=4, ge=1, le=360)
    max_overhang: float = Field(default=45.0, gt=0.0, lt=90.0)
    layer_height: float = Field(default=0.2, gt=0.0, le=2.0)
    keep_top: int = Field(default=10, ge=1, le=100)


class PipelineOptions(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    target_max_dimension_mm: float = Field(default=100.0, gt=0.0, le=1_000.0)
    image_limits: ImageLimits = Field(default_factory=ImageLimits)
    mesh_limits: MeshLimits = Field(default_factory=MeshLimits)
    orientation: OrientationOptions = Field(default_factory=OrientationOptions)


class ImageReport(FrozenModel):
    sha256: str
    source_format: str
    width: int
    height: int
    normalized_mode: Literal["RGB"] = "RGB"
    normalized_filename: str


class ReconstructionReport(FrozenModel):
    backend: str
    model: str
    revision: str
    duration_seconds: float = Field(ge=0.0)
    deterministic: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


class MeshReport(FrozenModel):
    vertices: int
    faces: int
    components: int
    watertight: bool
    winding_consistent: bool
    degenerate_face_ratio: float = Field(ge=0.0, le=1.0)
    original_extents: tuple[float, float, float]
    normalized_extents_mm: tuple[float, float, float]
    scale_factor: float = Field(gt=0.0)
    warnings: tuple[str, ...] = ()
    passed: bool = True


class OrientationReport(FrozenModel):
    source: str
    rotation_matrix: list[list[float]]
    euler_xyz_deg: list[float]
    metrics: dict[str, float]
    candidates_retained: int


class PipelineEvent(FrozenModel):
    state: PipelineState
    timestamp_utc: str


class ArtifactManifest(FrozenModel):
    normalized_image: str
    normalized_mesh: str
    oriented_stl: str
    report: str


class PipelineReport(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    job_id: str
    state: PipelineState
    created_at_utc: str
    completed_at_utc: str
    image: ImageReport
    reconstruction: ReconstructionReport
    mesh: MeshReport
    orientation: OrientationReport
    artifacts: ArtifactManifest
    events: tuple[PipelineEvent, ...]
