from __future__ import annotations

import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from xml.etree import ElementTree

from .contracts import ErrorCode, ServiceError, SlicingReport

_SLICE_INFO_PATH = "Metadata/slice_info.config"


class SlicerBackend(Protocol):
    def slice(self, stl_path: Path, output_path: Path) -> SlicingReport:
        """Slice one mesh and return parsed time/filament/support metrics."""


def _parse_slice_info(archive_path: Path) -> SlicingReport:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            raw = archive.read(_SLICE_INFO_PATH)
    except (KeyError, zipfile.BadZipFile) as exc:
        raise ServiceError(
            ErrorCode.SLICER_OUTPUT_INVALID,
            "Slicer output did not contain the expected slice metadata",
        ) from exc

    try:
        root = ElementTree.fromstring(raw)
        header = {
            item.get("key"): item.get("value") for item in root.findall("./header/header_item")
        }
        plate = root.find("./plate")
        if plate is None:
            raise ValueError("missing <plate> element")
        metadata = {item.get("key"): item.get("value") for item in plate.findall("./metadata")}
        prediction_seconds = float(metadata["prediction"])
        support_used = metadata.get("support_used") == "true"
        filament_used_mm = 0.0
        filament_used_g = 0.0
        for filament in plate.findall("./filament"):
            filament_used_mm += float(filament.get("used_m", "0")) * 1000.0
            filament_used_g += float(filament.get("used_g", "0"))
    except (ElementTree.ParseError, KeyError, ValueError, TypeError) as exc:
        raise ServiceError(
            ErrorCode.SLICER_OUTPUT_INVALID,
            "Slicer output metadata could not be parsed",
        ) from exc

    return SlicingReport(
        engine=header.get("X-BBL-Client-Name", "unknown"),
        engine_version=header.get("X-BBL-Client-Version", "unknown"),
        print_time_seconds=prediction_seconds,
        filament_used_mm=filament_used_mm,
        filament_used_g=filament_used_g,
        support_used=support_used,
    )


@dataclass(frozen=True)
class OrcaFamilySlicerBackend:
    """Headless slicer adapter for CLIs sharing OrcaSlicer's Slic3r::CLI surface.

    Works with ElegooSlicer, OrcaSlicer, and BambuStudio. Requires ``--slice`` mode
    (not a bare ``--help``, which boots the full GUI and its network/printer stack
    instead of the headless pipeline).
    """

    executable: Path
    process_profile: Path
    machine_profile: Path
    filament_profile: Path
    timeout_seconds: int = 300

    def slice(self, stl_path: Path, output_path: Path) -> SlicingReport:
        for required, code in (
            (self.executable, ErrorCode.SLICER_NOT_CONFIGURED),
            (self.process_profile, ErrorCode.SLICER_NOT_CONFIGURED),
            (self.machine_profile, ErrorCode.SLICER_NOT_CONFIGURED),
            (self.filament_profile, ErrorCode.SLICER_NOT_CONFIGURED),
        ):
            if not required.is_file():
                raise ServiceError(code, f"Slicer dependency is missing: {required}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        settings = f"{self.process_profile.resolve()};{self.machine_profile.resolve()}"
        command = [
            str(self.executable.resolve()),
            str(stl_path.resolve()),
            "--load-settings",
            settings,
            "--load-filaments",
            str(self.filament_profile.resolve()),
            "--slice",
            "0",
            "--export-3mf",
            str(output_path.resolve()),
        ]
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ServiceError(ErrorCode.SLICER_TIMEOUT, "Slicing timed out") from exc
        except OSError as exc:
            raise ServiceError(ErrorCode.SLICER_FAILED, "Could not start the slicer") from exc
        duration = time.perf_counter() - started

        if completed.returncode != 0 or not output_path.is_file():
            detail = (completed.stderr or completed.stdout or "unknown error")[-2_000:]
            raise ServiceError(
                ErrorCode.SLICER_FAILED,
                f"Slicing failed after {duration:.1f}s: {detail}",
            )

        return _parse_slice_info(output_path)
