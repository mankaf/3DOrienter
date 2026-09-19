from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from ..mesh_io import MeshLoadError, export_mesh, load_mesh
from .contracts import ErrorCode, MeshLimits, MeshReport, ServiceError


def validate_and_normalize_mesh(
    source: Path,
    destination: Path,
    limits: MeshLimits,
    target_max_dimension_mm: float,
) -> tuple[trimesh.Trimesh, MeshReport]:
    """Apply the service quality gate, scale to millimeters, and export STL."""
    if not source.is_file() or source.stat().st_size == 0:
        raise ServiceError(ErrorCode.MESH_INVALID, "Reconstruction produced no mesh data")
    if source.stat().st_size > limits.max_file_bytes:
        raise ServiceError(ErrorCode.MESH_TOO_COMPLEX, "Generated mesh file exceeds size limit")

    try:
        mesh = load_mesh(source)
    except MeshLoadError as exc:
        raise ServiceError(ErrorCode.MESH_INVALID, "Generated mesh could not be loaded") from exc

    vertices = len(mesh.vertices)
    faces = len(mesh.faces)
    if vertices > limits.max_vertices or faces > limits.max_faces:
        raise ServiceError(
            ErrorCode.MESH_TOO_COMPLEX,
            f"Generated mesh has {vertices} vertices and {faces} faces",
        )

    try:
        components = int(mesh.body_count)
    except (ValueError, TypeError, IndexError) as exc:
        raise ServiceError(ErrorCode.MESH_INVALID, "Mesh connectivity analysis failed") from exc
    if components > limits.max_components:
        raise ServiceError(
            ErrorCode.MESH_INVALID,
            f"Generated mesh has {components} disconnected components",
        )

    areas = np.asarray(mesh.area_faces, dtype=float)
    degenerate_ratio = float(np.mean(areas <= 1e-12)) if len(areas) else 1.0
    if degenerate_ratio > limits.max_degenerate_face_ratio:
        raise ServiceError(
            ErrorCode.MESH_INVALID,
            f"Degenerate face ratio {degenerate_ratio:.3f} exceeds quality limit",
        )

    extents = np.asarray(mesh.extents, dtype=float)
    if extents.shape != (3,) or not np.isfinite(extents).all() or float(extents.max()) <= 1e-9:
        raise ServiceError(ErrorCode.MESH_INVALID, "Generated mesh has invalid dimensions")

    scale_factor = target_max_dimension_mm / float(extents.max())
    mesh.apply_scale(scale_factor)
    normalized_extents = tuple(float(value) for value in mesh.extents)
    warnings: list[str] = []
    if not mesh.is_watertight:
        warnings.append("mesh_not_watertight")
    if not mesh.is_winding_consistent:
        warnings.append("winding_inconsistent")
    if components > 1:
        warnings.append("multiple_components")

    try:
        export_mesh(mesh, destination)
    except ValueError as exc:
        raise ServiceError(ErrorCode.MESH_INVALID, "Normalized mesh export failed") from exc

    return mesh, MeshReport(
        vertices=vertices,
        faces=faces,
        components=components,
        watertight=bool(mesh.is_watertight),
        winding_consistent=bool(mesh.is_winding_consistent),
        degenerate_face_ratio=degenerate_ratio,
        original_extents=tuple(float(value) for value in extents),
        normalized_extents_mm=normalized_extents,
        scale_factor=scale_factor,
        warnings=tuple(warnings),
    )
