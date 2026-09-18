from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import trimesh

_EPS = 1e-12


@dataclass(frozen=True)
class ScoreWeights:
    support: float = 0.65
    quality: float = 0.25
    height: float = 0.05
    stability: float = 0.05

    def normalized(self) -> ScoreWeights:
        values = np.array([self.support, self.quality, self.height, self.stability], dtype=float)
        if np.any(values < 0):
            raise ValueError("score weights must be non-negative")
        total = float(values.sum())
        if total <= 0:
            raise ValueError("at least one score weight must be positive")
        values /= total
        return ScoreWeights(*values.tolist())


@dataclass(frozen=True)
class OrientationMetrics:
    score: float
    support_cost: float
    support_area_ratio: float
    support_volume_ratio: float
    quality_cost: float
    height_ratio: float
    stability_cost: float
    bed_contact_ratio: float
    height_mm: float
    overhang_area_mm2: float
    support_volume_proxy_mm3: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class OrientationScorer:
    """Fast orientation scorer using geometric FDM proxies.

    The scorer intentionally avoids a full slicer. It estimates:
    - support burden from downward overhang area and height-to-bed,
    - surface quality from the area of sloped faces prone to stair stepping,
    - print height,
    - first-layer stability from nearly coplanar bed-contact triangles.
    """

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        *,
        max_overhang_deg: float = 45.0,
        layer_height_mm: float = 0.2,
        bed_tolerance_mm: float | None = None,
        weights: ScoreWeights | None = None,
    ) -> None:
        if not 0.0 < max_overhang_deg < 90.0:
            raise ValueError("max_overhang_deg must be between 0 and 90")
        if layer_height_mm <= 0:
            raise ValueError("layer_height_mm must be positive")

        self.mesh = mesh
        self.vertices = np.asarray(mesh.vertices, dtype=float)
        self.faces = np.asarray(mesh.faces, dtype=np.int64)
        self.normals = np.asarray(mesh.face_normals, dtype=float)
        self.areas = np.asarray(mesh.area_faces, dtype=float)
        self.centroids = np.asarray(mesh.triangles_center, dtype=float)

        self.total_area = max(float(self.areas.sum()), _EPS)
        self.contact_reference_area = max(float(self.areas.max(initial=0.0)), _EPS)
        center = self.vertices.mean(axis=0)
        radius = max(float(np.linalg.norm(self.vertices - center, axis=1).max()), _EPS)
        self.diagonal = max(2.0 * radius, _EPS)
        self.reference_volume = max(self.total_area * self.diagonal, _EPS)

        self.max_overhang_deg = max_overhang_deg
        self.overhang_normal_threshold = float(np.cos(np.deg2rad(max_overhang_deg)))
        self.layer_height_mm = layer_height_mm
        self.bed_tolerance_mm = (
            max(layer_height_mm * 0.55, 1e-4)
            if bed_tolerance_mm is None
            else max(float(bed_tolerance_mm), 1e-6)
        )
        self.weights = (weights or ScoreWeights()).normalized()

    def evaluate(self, rotation: np.ndarray) -> OrientationMetrics:
        if rotation.shape != (3, 3):
            raise ValueError("rotation must be a 3x3 matrix")

        rotated_normals = self.normals @ rotation.T
        vertex_z = self.vertices @ rotation[2, :]
        min_z = float(vertex_z.min())
        max_z = float(vertex_z.max())
        height = max(max_z - min_z, 0.0)

        centroid_z = self.centroids @ rotation[2, :] - min_z
        nz = np.clip(rotated_normals[:, 2], -1.0, 1.0)
        transformed_vertex_z = vertex_z - min_z
        face_vertex_z = transformed_vertex_z[self.faces]
        touching = np.max(face_vertex_z, axis=1) <= self.bed_tolerance_mm

        # A downward face needs support when its plane is shallower than the
        # allowed overhang angle. Severity ramps continuously at the threshold.
        downward = -nz
        severity = np.clip(
            (downward - self.overhang_normal_threshold)
            / max(1.0 - self.overhang_normal_threshold, _EPS),
            0.0,
            1.0,
        )
        # Bed-contact faces are supported directly by the build plate.
        severity = severity * (~touching)
        weighted_overhang_area = self.areas * severity
        overhang_area = float(weighted_overhang_area.sum())
        support_area_ratio = overhang_area / self.total_area

        projected_area = self.areas * np.clip(downward, 0.0, 1.0)
        support_volume_proxy = float((projected_area * centroid_z * severity).sum())
        raw_volume_ratio = support_volume_proxy / self.reference_volume
        # Saturation keeps a large proxy from dominating every other metric.
        support_volume_ratio = float(1.0 - np.exp(-max(raw_volume_ratio, 0.0)))
        support_cost = 0.45 * support_area_ratio + 0.55 * support_volume_ratio

        # Stair-step proxy: zero for vertical and exactly horizontal planes,
        # highest on intermediate slopes. Downward quality loss is already
        # additionally represented by support cost.
        abs_nz = np.abs(nz)
        slope_penalty = 2.0 * abs_nz * np.sqrt(np.maximum(1.0 - abs_nz**2, 0.0))
        quality_cost = float(np.dot(self.areas, slope_penalty) / self.total_area)

        height_ratio = min(height / self.diagonal, 1.0)

        downward_facing = nz < -0.5
        contact_area = float(self.areas[touching & downward_facing].sum())

        # The largest input triangle is a stable, rotation-invariant reference.
        # Multiple coplanar triangles may sum past it, so cap the ratio at one.
        bed_contact_ratio = min(contact_area / self.contact_reference_area, 1.0)
        stability_cost = 1.0 - bed_contact_ratio

        w = self.weights
        score = (
            w.support * support_cost
            + w.quality * quality_cost
            + w.height * height_ratio
            + w.stability * stability_cost
        )

        return OrientationMetrics(
            score=float(score),
            support_cost=float(support_cost),
            support_area_ratio=float(support_area_ratio),
            support_volume_ratio=float(support_volume_ratio),
            quality_cost=float(quality_cost),
            height_ratio=float(height_ratio),
            stability_cost=float(stability_cost),
            bed_contact_ratio=float(bed_contact_ratio),
            height_mm=float(height),
            overhang_area_mm2=float(overhang_area),
            support_volume_proxy_mm3=float(support_volume_proxy),
        )
