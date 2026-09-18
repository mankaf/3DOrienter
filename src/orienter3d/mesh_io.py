from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


class MeshLoadError(ValueError):
    """Raised when an input file cannot be converted to one triangle mesh."""


def load_mesh(path: Path, *, repair: bool = True) -> trimesh.Trimesh:
    """Load an STL or another trimesh-supported format as a single mesh."""
    try:
        loaded = trimesh.load(path, force="scene", process=False)
    except Exception as exc:  # pragma: no cover - depends on parser backends
        raise MeshLoadError(f"Could not load '{path}': {exc}") from exc

    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise MeshLoadError(f"'{path}' contains no mesh geometry")
        mesh = loaded.to_geometry()
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise MeshLoadError(f"'{path}' did not resolve to a triangle mesh")

    mesh = mesh.copy()
    if mesh.faces.shape[0] == 0 or mesh.vertices.shape[0] == 0:
        raise MeshLoadError(f"'{path}' contains an empty mesh")

    if repair:
        mesh.remove_infinite_values()
        mesh.merge_vertices()
        mesh.remove_unreferenced_vertices()
        # These operations are conservative: they repair winding/normals without
        # attempting risky hole filling or topology reconstruction.
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_normals(mesh, multibody=True)

    if not np.isfinite(mesh.vertices).all():
        raise MeshLoadError(f"'{path}' contains non-finite vertex coordinates")
    return mesh


def apply_orientation(mesh: trimesh.Trimesh, rotation: np.ndarray) -> trimesh.Trimesh:
    """Rotate a mesh, place its lowest point on Z=0, and center it in XY."""
    if rotation.shape != (3, 3):
        raise ValueError("rotation must be a 3x3 matrix")

    result = mesh.copy()
    result.vertices = np.asarray(result.vertices) @ rotation.T

    mins = result.vertices.min(axis=0)
    maxs = result.vertices.max(axis=0)
    center_xy = 0.5 * (mins[:2] + maxs[:2])
    result.apply_translation([-center_xy[0], -center_xy[1], -mins[2]])
    return result


def export_mesh(mesh: trimesh.Trimesh, path: Path) -> None:
    """Export a mesh, inferring the format from the file suffix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mesh.export(path)
    except Exception as exc:  # pragma: no cover - format/backend dependent
        raise ValueError(f"Could not export '{path}': {exc}") from exc
