import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from orienter3d.scoring import OrientationScorer


def test_box_prefers_large_face_down_and_has_contact():
    mesh = trimesh.creation.box(extents=(40.0, 20.0, 10.0))
    scorer = OrientationScorer(mesh)

    flat = scorer.evaluate(np.eye(3))
    tall = scorer.evaluate(Rotation.from_euler("y", 90, degrees=True).as_matrix())

    assert flat.height_mm < tall.height_mm
    assert flat.bed_contact_ratio > tall.bed_contact_ratio
    assert flat.score < tall.score


def test_rotation_score_is_finite():
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=10.0)
    scorer = OrientationScorer(mesh)
    matrix = Rotation.from_euler("xyz", [23.0, -51.0, 9.0], degrees=True).as_matrix()
    metrics = scorer.evaluate(matrix)

    assert np.isfinite(metrics.score)
    assert 0.0 <= metrics.quality_cost <= 1.0 + 1e-9
    assert 0.0 <= metrics.height_ratio <= 1.0 + 1e-9
