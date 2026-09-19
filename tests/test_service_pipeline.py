import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import trimesh
from PIL import Image

from orienter3d.service import (
    MockReconstructionBackend,
    PipelineOptions,
    ServiceError,
    run_pipeline,
)
from orienter3d.service.contracts import ErrorCode, ImageLimits, OrientationOptions, SlicingReport
from orienter3d.service.image_preflight import prepare_image


def test_mock_pipeline_writes_safe_bundle(tmp_path):
    image_path = tmp_path / "source.png"
    Image.new("RGBA", (64, 48), (10, 120, 240, 128)).save(image_path)
    options = PipelineOptions(
        target_max_dimension_mm=80.0,
        orientation=OrientationOptions(samples=8, seed=7, keep_top=4),
    )

    execution = run_pipeline(
        image_path,
        tmp_path / "jobs",
        MockReconstructionBackend(),
        options,
        job_id="safe-job-1",
    )

    payload = json.loads(execution.report_path.read_text(encoding="utf-8"))
    serialized = json.dumps(payload)
    assert payload["state"] == "succeeded"
    assert payload["reconstruction"]["backend"] == "mock"
    assert str(tmp_path) not in serialized
    assert payload["artifacts"]["oriented_stl"] == "output/oriented.stl"

    normalized = trimesh.load(execution.workspace_path / "generated" / "normalized.stl")
    oriented = trimesh.load(execution.workspace_path / "output" / "oriented.stl")
    assert oriented.bounds[0][2] == pytest.approx(0.0, abs=1e-6)
    assert max(normalized.extents) == pytest.approx(80.0, rel=1e-4)


@dataclass(frozen=True)
class _StubSlicer:
    def slice(self, stl_path: Path, output_path: Path) -> SlicingReport:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("stub gcode.3mf\n", encoding="utf-8")
        return SlicingReport(
            engine="stub",
            engine_version="0",
            print_time_seconds=120.0,
            filament_used_mm=1000.0,
            filament_used_g=3.0,
            support_used=False,
        )


def test_pipeline_attaches_slicing_report_when_slicer_provided(tmp_path):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (64, 48), "white").save(image_path)
    options = PipelineOptions(orientation=OrientationOptions(samples=4, seed=1, keep_top=2))

    execution = run_pipeline(
        image_path,
        tmp_path / "jobs",
        MockReconstructionBackend(),
        options,
        job_id="sliced-job-1",
        slicer=_StubSlicer(),
    )

    payload = json.loads(execution.report_path.read_text(encoding="utf-8"))
    assert payload["slicing"]["print_time_seconds"] == pytest.approx(120.0)
    assert payload["artifacts"]["sliced_project"] == "output/oriented.gcode.3mf"
    assert (execution.workspace_path / "output" / "oriented.gcode.3mf").is_file()


def test_preflight_rejects_oversized_dimensions(tmp_path):
    image_path = tmp_path / "wide.png"
    Image.new("RGB", (33, 10), "white").save(image_path)

    with pytest.raises(ServiceError) as raised:
        prepare_image(
            image_path,
            tmp_path / "normalized.png",
            ImageLimits(max_width=32, max_height=32),
        )

    assert raised.value.code is ErrorCode.IMAGE_DIMENSIONS_EXCEEDED


def test_pipeline_rejects_unsafe_job_id(tmp_path):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (16, 16), "white").save(image_path)

    with pytest.raises(ServiceError) as raised:
        run_pipeline(
            image_path,
            tmp_path / "jobs",
            MockReconstructionBackend(),
            job_id="../escape",
        )

    assert raised.value.code is ErrorCode.INVALID_JOB_ID
