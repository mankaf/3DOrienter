import json
import subprocess

import pytest

from orienter3d.service.backends import TripoSRBackend
from orienter3d.service.contracts import ErrorCode, ServiceError


def _configured_backend(tmp_path, **overrides):
    repository = tmp_path / "triposr"
    repository.mkdir()
    (repository / "run.py").write_text("# test stub\n", encoding="utf-8")
    python = tmp_path / "python"
    python.write_text("test stub\n", encoding="utf-8")
    return TripoSRBackend(repository=repository, python_executable=python, **overrides)


def test_triposr_backend_rejects_busy_gpu(tmp_path, monkeypatch):
    backend = _configured_backend(tmp_path, min_free_vram_mb=7_000)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "1024\n", ""),
    )

    with pytest.raises(ServiceError) as raised:
        backend.reconstruct(tmp_path / "input.png", tmp_path / "raw.obj")

    assert raised.value.code is ErrorCode.BACKEND_BUSY


def test_triposr_report_does_not_expose_local_model_path(tmp_path, monkeypatch):
    backend = _configured_backend(
        tmp_path,
        pretrained_model="/private/models/triposr",
        model_id="stabilityai/TripoSR",
        min_free_vram_mb=7_000,
    )
    output = tmp_path / "generated" / "raw.obj"

    def fake_run(command, **kwargs):
        if command[0] == "nvidia-smi":
            return subprocess.CompletedProcess(command, 0, "8000\n", "")
        generated = output.parent / "triposr-output" / "0" / "mesh.obj"
        generated.parent.mkdir(parents=True)
        generated.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = backend.reconstruct(tmp_path / "input.png", output)
    serialized = json.dumps(result.report.model_dump(mode="json"))

    assert result.report.model == "stabilityai/TripoSR"
    assert "/private/models/triposr" not in serialized
