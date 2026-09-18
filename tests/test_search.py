import trimesh

from orienter3d.scoring import OrientationScorer
from orienter3d.search import optimize_orientation


def test_optimizer_returns_sorted_results():
    mesh = trimesh.creation.box(extents=(10.0, 20.0, 30.0))
    scorer = OrientationScorer(mesh)
    results = optimize_orientation(mesh, scorer, random_samples=12, seed=7, keep_top=5)

    assert len(results) == 5
    assert [item.metrics.score for item in results] == sorted(
        item.metrics.score for item in results
    )
