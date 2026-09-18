from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import permutations, product

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from .scoring import OrientationMetrics, OrientationScorer


@dataclass(frozen=True)
class OrientationResult:
    rotation: np.ndarray
    metrics: OrientationMetrics
    source: str

    def to_dict(self) -> dict[str, object]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            euler = Rotation.from_matrix(self.rotation).as_euler("xyz", degrees=True)
        return {
            "source": self.source,
            "rotation_matrix": self.rotation.tolist(),
            "euler_xyz_deg": euler.tolist(),
            "metrics": self.metrics.to_dict(),
        }


def _rotation_align_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))

    if dot > 1.0 - 1e-10:
        return np.eye(3)
    if dot < -1.0 + 1e-10:
        # Pick a stable axis orthogonal to source for a 180 degree rotation.
        basis = np.array([1.0, 0.0, 0.0])
        if abs(source[0]) > 0.8:
            basis = np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, basis)
        axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(np.pi * axis).as_matrix()

    skew = np.array(
        [[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]]
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - dot) / (np.linalg.norm(cross) ** 2))


def _cube_rotations() -> list[np.ndarray]:
    rotations: list[np.ndarray] = []
    eye = np.eye(3)
    for perm in permutations(range(3)):
        permuted = eye[list(perm)]
        for signs in product((-1.0, 1.0), repeat=3):
            matrix = np.diag(signs) @ permuted
            if np.linalg.det(matrix) > 0.5:
                rotations.append(matrix)
    return rotations


def candidate_rotations(
    mesh: trimesh.Trimesh,
    *,
    random_samples: int,
    seed: int,
    face_candidates: int = 24,
    yaw_steps: int = 4,
) -> list[tuple[np.ndarray, str]]:
    """Create deterministic, face-aligned, axis-aligned, and random candidates."""
    candidates: list[tuple[np.ndarray, str]] = [(np.eye(3), "identity")]

    for matrix in _cube_rotations():
        candidates.append((matrix, "axis"))

    areas = np.asarray(mesh.area_faces)
    normals = np.asarray(mesh.face_normals)
    if len(areas):
        largest = np.argsort(areas)[::-1][: max(face_candidates, 0)]
        for face_index in largest:
            align = _rotation_align_vectors(normals[face_index], np.array([0.0, 0.0, -1.0]))
            for yaw_index in range(max(yaw_steps, 1)):
                yaw = Rotation.from_euler("z", 360.0 * yaw_index / max(yaw_steps, 1), degrees=True)
                candidates.append((yaw.as_matrix() @ align, f"face:{int(face_index)}"))

    if random_samples > 0:
        rng = np.random.default_rng(seed)
        random_matrices = Rotation.random(random_samples, rng=rng).as_matrix()
        candidates.extend((matrix, "random") for matrix in random_matrices)

    # Deduplicate matrices after rounding, preserving the first source label.
    unique: dict[tuple[float, ...], tuple[np.ndarray, str]] = {}
    for matrix, source in candidates:
        key = tuple(np.round(matrix, decimals=8).ravel())
        unique.setdefault(key, (matrix, source))
    return list(unique.values())


def optimize_orientation(
    mesh: trimesh.Trimesh,
    scorer: OrientationScorer,
    *,
    random_samples: int = 400,
    seed: int = 42,
    face_candidates: int = 24,
    yaw_steps: int = 4,
    keep_top: int = 10,
) -> list[OrientationResult]:
    candidates = candidate_rotations(
        mesh,
        random_samples=random_samples,
        seed=seed,
        face_candidates=face_candidates,
        yaw_steps=yaw_steps,
    )
    results = [
        OrientationResult(rotation=matrix, metrics=scorer.evaluate(matrix), source=source)
        for matrix, source in candidates
    ]
    results.sort(key=lambda item: item.metrics.score)
    return results[: max(1, keep_top)]
