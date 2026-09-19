from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import trimesh

from .contracts import ErrorCode, ReconstructionReport, ServiceError


@dataclass(frozen=True)
class ReconstructionOutput:
    mesh_path: Path
    report: ReconstructionReport


class ReconstructionBackend(Protocol):
    def reconstruct(self, image_path: Path, output_path: Path) -> ReconstructionOutput:
        """Generate one geometry-only mesh from a normalized image."""


@dataclass(frozen=True)
class MockReconstructionBackend:
    """Deterministic local backend for pipeline and infrastructure tests only."""

    name: str = "mock"
    model: str = "deterministic-icosphere"
    revision: str = "1"

    def reconstruct(self, image_path: Path, output_path: Path) -> ReconstructionOutput:
        started = time.perf_counter()
        digest = hashlib.sha256(image_path.read_bytes()).digest()
        scales = np.array([0.75 + digest[index] / 510.0 for index in range(3)])
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        mesh.apply_scale(scales)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(output_path)
        return ReconstructionOutput(
            mesh_path=output_path,
            report=ReconstructionReport(
                backend=self.name,
                model=self.model,
                revision=self.revision,
                duration_seconds=time.perf_counter() - started,
                deterministic=True,
                metadata={"test_backend": True},
            ),
        )


@dataclass(frozen=True)
class TripoSRBackend:
    """Isolated subprocess adapter for the official TripoSR run.py CLI."""

    repository: Path
    python_executable: Path
    revision: str = "107cefdc244c39106fa830359024f6a2f1c78871"
    pretrained_model: str = "stabilityai/TripoSR"
    model_id: str = "stabilityai/TripoSR"
    device: str = "cuda:0"
    chunk_size: int = 8192
    mc_resolution: int = 256
    foreground_ratio: float = 0.85
    timeout_seconds: int = 900
    min_free_vram_mb: int = 7_000
    extra_environment: dict[str, str] = field(default_factory=dict)

    def reconstruct(self, image_path: Path, output_path: Path) -> ReconstructionOutput:
        run_script = self.repository.resolve() / "run.py"
        python = self.python_executable.resolve()
        if not run_script.is_file() or not python.is_file():
            raise ServiceError(
                ErrorCode.BACKEND_NOT_CONFIGURED,
                "TripoSR repository or isolated Python executable is missing",
            )

        try:
            gpu_status = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                    "--id=0",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
                shell=False,
            )
        except OSError as exc:
            raise ServiceError(
                ErrorCode.BACKEND_NOT_CONFIGURED,
                "nvidia-smi is required for GPU admission control",
            ) from exc
        try:
            free_vram_mb = int(gpu_status.stdout.strip().splitlines()[0])
        except (IndexError, ValueError) as exc:
            raise ServiceError(
                ErrorCode.BACKEND_NOT_CONFIGURED,
                "Could not read free GPU memory",
            ) from exc
        if gpu_status.returncode != 0 or free_vram_mb < self.min_free_vram_mb:
            raise ServiceError(
                ErrorCode.BACKEND_BUSY,
                f"GPU has {free_vram_mb} MiB free; {self.min_free_vram_mb} MiB required",
            )

        backend_output = output_path.parent / "triposr-output"
        if backend_output.exists():
            raise ServiceError(ErrorCode.BACKEND_FAILED, "TripoSR output directory already exists")

        command = [
            str(python),
            str(run_script),
            str(image_path.resolve()),
            "--device",
            self.device,
            "--pretrained-model-name-or-path",
            self.pretrained_model,
            "--chunk-size",
            str(self.chunk_size),
            "--mc-resolution",
            str(self.mc_resolution),
            "--foreground-ratio",
            str(self.foreground_ratio),
            "--output-dir",
            str(backend_output),
            "--model-save-format",
            "obj",
        ]
        started = time.perf_counter()
        environment = os.environ.copy()
        environment.update(self.extra_environment)
        try:
            completed = subprocess.run(
                command,
                cwd=self.repository,
                env=environment,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ServiceError(ErrorCode.BACKEND_TIMEOUT, "TripoSR reconstruction timed out") from exc
        except OSError as exc:
            raise ServiceError(ErrorCode.BACKEND_FAILED, "Could not start TripoSR") from exc

        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "unknown error")[-2_000:]
            raise ServiceError(ErrorCode.BACKEND_FAILED, f"TripoSR failed: {detail}")

        generated = backend_output / "0" / "mesh.obj"
        if not generated.is_file() or generated.stat().st_size == 0:
            raise ServiceError(ErrorCode.BACKEND_FAILED, "TripoSR produced no mesh")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generated, output_path)
        metadata: dict[str, Any] = {
            "device": self.device,
            "chunk_size": self.chunk_size,
            "mc_resolution": self.mc_resolution,
            "foreground_ratio": self.foreground_ratio,
        }
        return ReconstructionOutput(
            mesh_path=output_path,
            report=ReconstructionReport(
                backend="triposr",
                model=self.model_id,
                revision=self.revision,
                duration_seconds=time.perf_counter() - started,
                deterministic=False,
                metadata=metadata,
            ),
        )
