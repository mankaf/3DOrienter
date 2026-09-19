import subprocess
import zipfile

import pytest

from orienter3d.service.contracts import ErrorCode, ServiceError
from orienter3d.service.slicing import OrcaFamilySlicerBackend

# Captured verbatim from a real `elegoo-slicer.exe --slice 0 --export-3mf` run
# against a flat-bottomed 10mm test cube, so the parser is checked against the
# actual on-disk schema rather than a guessed one.
_REAL_SLICE_INFO_CONFIG = b"""<?xml version="1.0" encoding="UTF-8"?>
<config>
  <header>
    <header_item key="X-BBL-Client-Type" value="slicer"/>
    <header_item key="X-BBL-Client-Version" value="01.05.03.05"/>
    <header_item key="X-BBL-Client-Name" value="ElegooSlicer"/>
    <header_item key="OrcaSlicer-Version" value="2.4.2"/>
  </header>
  <plate>
    <metadata key="index" value="1"/>
    <metadata key="prediction" value="365"/>
    <metadata key="weight" value=""/>
    <metadata key="support_used" value="false"/>
    <object identify_id="14" name="probe_box.stl_id_0_copy_0" skipped="false" />
    <filament id="1" tray_info_idx="" type="PLA" color="#F2754E" used_m="0.48" used_g="12.34" group_id="0" nozzle_diameter="0.40" volume_type="Standard" used_for_object="true" used_for_support="false"/>
  </plate>
</config>
"""


def _configured_backend(tmp_path, **overrides):
    executable = tmp_path / "elegoo-slicer.exe"
    process_profile = tmp_path / "process.json"
    machine_profile = tmp_path / "machine.json"
    filament_profile = tmp_path / "filament.json"
    for path in (executable, process_profile, machine_profile, filament_profile):
        path.write_text("test stub\n", encoding="utf-8")
    return OrcaFamilySlicerBackend(
        executable=executable,
        process_profile=process_profile,
        machine_profile=machine_profile,
        filament_profile=filament_profile,
        **overrides,
    )


def test_slicer_parses_real_slice_info_schema(tmp_path, monkeypatch):
    backend = _configured_backend(tmp_path)
    output = tmp_path / "oriented.gcode.3mf"

    def fake_run(command, **kwargs):
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("Metadata/slice_info.config", _REAL_SLICE_INFO_CONFIG)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    report = backend.slice(tmp_path / "oriented.stl", output)

    assert report.engine == "ElegooSlicer"
    assert report.print_time_seconds == pytest.approx(365.0)
    assert report.filament_used_mm == pytest.approx(480.0)
    assert report.filament_used_g == pytest.approx(12.34)
    assert report.support_used is False


def test_slicer_reports_failure_detail(tmp_path, monkeypatch):
    backend = _configured_backend(tmp_path)

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 4_294_967_196, "", "Errors")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(ServiceError) as raised:
        backend.slice(tmp_path / "oriented.stl", tmp_path / "out.3mf")

    assert raised.value.code is ErrorCode.SLICER_FAILED
    assert "Errors" in str(raised.value)


def test_slicer_rejects_missing_dependency(tmp_path):
    backend = _configured_backend(tmp_path)
    missing = OrcaFamilySlicerBackend(
        executable=tmp_path / "does-not-exist.exe",
        process_profile=backend.process_profile,
        machine_profile=backend.machine_profile,
        filament_profile=backend.filament_profile,
    )

    with pytest.raises(ServiceError) as raised:
        missing.slice(tmp_path / "oriented.stl", tmp_path / "out.3mf")

    assert raised.value.code is ErrorCode.SLICER_NOT_CONFIGURED


def test_slicer_times_out(tmp_path, monkeypatch):
    backend = _configured_backend(tmp_path, timeout_seconds=1)

    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(cmd=command, timeout=1)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(ServiceError) as raised:
        backend.slice(tmp_path / "oriented.stl", tmp_path / "out.3mf")

    assert raised.value.code is ErrorCode.SLICER_TIMEOUT
