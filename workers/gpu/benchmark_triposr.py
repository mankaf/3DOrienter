from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from orienter3d import service


class BenchmarkResult:
    """Single benchmark result record."""

    __slots__ = (
        "error_code",
        "error_message",
        "image",
        "job_id",
        "peak_vram_mb",
        "repeat_index",
        "stage_durations",
        "success",
        "timestamp_utc",
        "total_wall_seconds",
        "warm",
    )

    def __init__(
        self,
        image: str,
        repeat_index: int,
        warm: bool,
        success: bool,
        total_wall_seconds: float | None = None,
        stage_durations: dict[str, float] | None = None,
        peak_vram_mb: float | None = None,
        job_id: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        self.image = image
        self.repeat_index = repeat_index
        self.warm = warm
        self.success = success
        self.total_wall_seconds = total_wall_seconds
        self.stage_durations = stage_durations or {}
        self.peak_vram_mb = peak_vram_mb
        self.job_id = job_id
        self.error_code = error_code
        self.error_message = error_message
        self.timestamp_utc = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "repeat_index": self.repeat_index,
            "warm": self.warm,
            "success": self.success,
            "total_wall_seconds": self.total_wall_seconds,
            "stage_durations": self.stage_durations,
            "peak_vram_mb": self.peak_vram_mb,
            "job_id": self.job_id,
            "error_code": self.error_code,
            "error_message": self.error_message,
        }


def _parse_timestamp(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


_SUPPORTED_EXTENSIONS = frozenset({"png", "jpg", "jpeg", "webp"})


def _discover_images(images_dir: Path) -> list[Path]:
    """Find all supported image files in directory, sorted by name."""
    if not images_dir.is_dir():
        raise ValueError(f"Images directory does not exist: {images_dir}")

    found = []
    for entry in os.listdir(images_dir):
        ext_lower = Path(entry).suffix[1:].lower()
        if ext_lower in _SUPPORTED_EXTENSIONS:
            full_path = images_dir / entry
            if full_path.is_file():
                found.append(full_path)

    return sorted(found, key=lambda p: p.name.lower())


def _vram_poller(
    event: threading.Event, interval_seconds: float, samples: list[int]
) -> None:
    """Poll VRAM usage until signaled to stop.

    Runs in a background thread so we can measure peak VRAM during the job without blocking.
    """
    while not event.is_set():
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    "--id=0",
                ],
                capture_output=True,
                text=True,
                check=False,
                shell=False,
            )

            if result.returncode == 0 and result.stdout.strip():
                used_mb = int(result.stdout.strip())
                samples.append(used_mb)
        except (subprocess.SubprocessError, OSError, ValueError):
            pass

        event.wait(interval_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark TripoSR reconstruction backend")
    parser.add_argument("--images-dir", type=Path, required=True, help="Directory of input images")
    parser.add_argument(
        "--triposr-repo", type=Path, required=True, help="TripoSR repository path"
    )
    parser.add_argument(
        "--triposr-python", type=Path, required=True, help="Isolated Python executable path"
    )
    parser.add_argument("--jobs-dir", type=Path, required=True, help="Output job workspace root")
    parser.add_argument(
        "--results",
        type=Path,
        default="benchmark_results.jsonl",
        help="Output JSONL results file (default: benchmark_results.jsonl)",
    )
    parser.add_argument(
        "--triposr-model",
        type=str,
        default="stabilityai/TripoSR",
        help="TripoSR pretrained model name or path",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Number of times to run each image")
    parser.add_argument(
        "--sample-interval-seconds",
        type=float,
        default=0.5,
        help="VRAM polling interval in seconds (default: 0.5)",
    )

    args = parser.parse_args()

    images_dir = args.images_dir.resolve()
    jobs_dir = args.jobs_dir.resolve()
    results_path = Path(args.results).resolve()

    if not images_dir.is_dir():
        print(f"Error: Images directory does not exist: {images_dir}", file=sys.stderr)
        sys.exit(1)

    image_files = _discover_images(images_dir)
    if not image_files:
        print("No supported image files found in the specified directory", file=sys.stderr)
        sys.exit(1)

    results_path.parent.mkdir(parents=True, exist_ok=True)

    total_jobs = 0
    success_count = 0
    failure_counts: dict[str, int] = {}
    successful_wall_times: list[float] = []

    for image_file in image_files:
        filename = image_file.name
        stem = Path(filename).stem

        for repeat_index in range(args.repeat):
            total_jobs += 1
            warm = repeat_index > 0
            job_id = f"{stem}-{'cold' if not warm else 'warm'}-{repeat_index}"

            backend = service.TripoSRBackend(
                repository=args.triposr_repo,
                python_executable=args.triposr_python,
                pretrained_model=args.triposr_model,
            )

            event = threading.Event()
            samples: list[int] = []
            poller_thread = threading.Thread(target=_vram_poller, args=(event, args.sample_interval_seconds, samples))
            poller_thread.start()

            try:
                execution = service.run_pipeline(
                    image_file,
                    jobs_dir,
                    backend,
                    job_id=job_id,
                )

                event.set()
                poller_thread.join()
                peak_vram_mb = max(samples) if samples else None

                events = execution.report.events
                timestamps: list[datetime] = [_parse_timestamp(e.timestamp_utc) for e in events]

                stage_durations: dict[str, float] = {}
                total_wall_seconds = 0.0

                for i in range(1, len(events)):
                    prev_ts = timestamps[i - 1]
                    curr_ts = timestamps[i]
                    duration = (curr_ts - prev_ts).total_seconds()
                    state_name = events[i].state.value
                    stage_durations[state_name] = round(duration, 3)

                total_wall_seconds = (timestamps[-1] - timestamps[0]).total_seconds()

                result = BenchmarkResult(
                    image=filename,
                    repeat_index=repeat_index,
                    warm=warm,
                    success=True,
                    total_wall_seconds=round(total_wall_seconds, 3),
                    stage_durations=stage_durations,
                    peak_vram_mb=peak_vram_mb,
                    job_id=job_id,
                )

            except service.ServiceError as exc:
                event.set()
                poller_thread.join()
                peak_vram_mb = max(samples) if samples else None

                result = BenchmarkResult(
                    image=filename,
                    repeat_index=repeat_index,
                    warm=warm,
                    success=False,
                    error_code=exc.code.value,
                    error_message=str(exc),
                    peak_vram_mb=peak_vram_mb,
                )

            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result.to_dict()) + "\n")
                f.flush()

            if result.success:
                success_count += 1
                successful_wall_times.append(result.total_wall_seconds)
            else:
                failure_counts[result.error_code] = failure_counts.get(result.error_code, 0) + 1

    print(f"Total jobs: {total_jobs}")
    print(f"Successful: {success_count}")
    print("Failed by error code:")
    for code, count in sorted(failure_counts.items()):
        print(f"  {code}: {count}")

    if successful_wall_times:
        p50 = statistics.median(successful_wall_times)
        n = len(successful_wall_times)
        sorted_times = sorted(successful_wall_times)
        p95_index = int(0.95 * n + 0.5 - 1)
        p95_index = max(0, min(p95_index, n - 1))
        p95 = sorted_times[p95_index]

        print("Wall time (successful jobs):")
        print(f"  P50: {p50:.3f}s")
        print(f"  P95: {p95:.3f}s")


if __name__ == "__main__":
    main()
