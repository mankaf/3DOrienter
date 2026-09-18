from __future__ import annotations

import json
import os
import random
import subprocess
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from .scoring import OrientationScorer
from .search import candidate_rotations
from .training_data import simulate_print_configuration

OBJECTIVES: dict[str, dict[str, float]] = {
    "balanced": {
        "quality": 0.30,
        "support_removal": 0.25,
        "strength": 0.25,
        "flexibility": 0.10,
        "print_time": 0.10,
    },
    "quality": {
        "quality": 0.55,
        "support_removal": 0.20,
        "strength": 0.15,
        "flexibility": 0.05,
        "print_time": 0.05,
    },
    "easy_supports": {
        "quality": 0.25,
        "support_removal": 0.50,
        "strength": 0.15,
        "flexibility": 0.05,
        "print_time": 0.05,
    },
    "strength": {
        "quality": 0.15,
        "support_removal": 0.15,
        "strength": 0.55,
        "flexibility": 0.05,
        "print_time": 0.10,
    },
    "flexibility": {
        "quality": 0.15,
        "support_removal": 0.15,
        "strength": 0.15,
        "flexibility": 0.45,
        "print_time": 0.10,
    },
}

_CONFIG_PRESETS = ((10.0, 0, 0.24), (15.0, 2, 0.20), (20.0, 3, 0.16))


def read_dotenv(path: Path = Path(".env.3dorienter")) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _windows_gateway_url() -> str | None:
    try:
        output = subprocess.check_output(
            ["ip", "route", "show", "default"], text=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.SubprocessError):
        return None
    fields = output.split()
    if "via" not in fields:
        return None
    index = fields.index("via") + 1
    if index >= len(fields):
        return None
    return f"http://{fields[index]}:1234"


def resolve_lmstudio_settings(url: str | None, model: str | None) -> tuple[str, str]:
    dotenv = read_dotenv()
    resolved_url = (
        url
        or os.environ.get("LM_STUDIO_URL")
        or dotenv.get("LM_STUDIO_URL")
        or _windows_gateway_url()
        or "http://localhost:1234"
    ).rstrip("/")
    resolved_model = (
        model
        or os.environ.get("LM_STUDIO_MODEL")
        or dotenv.get("LM_STUDIO_MODEL")
        or "3dorienter-v3"
    )
    return resolved_url, resolved_model


def utility(candidate: dict[str, Any], weights: dict[str, float]) -> float:
    time_score = 1.0 / (1.0 + float(candidate["print_time_proxy"]) / 500.0)
    return float(
        weights["quality"] * float(candidate["quality_score"])
        + weights["support_removal"] * (1.0 - float(candidate["support_removal_cost"]))
        + weights["strength"] * float(candidate["strength_score"])
        + weights["flexibility"] * float(candidate["flexibility_score"])
        + weights["print_time"] * time_score
    )


def _candidate_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        tuple(round(float(value), 4) for value in candidate["rotation_xyz_deg"]),
        candidate["support_type"],
        float(candidate["support_density_percent"]),
        int(candidate["interface_layers"]),
        float(candidate["z_gap_mm"]),
    )


def _pick_diverse_finalists(
    pool: list[dict[str, Any]], weights: dict[str, float], seed: int
) -> list[dict[str, Any]]:
    selectors = [
        lambda c: utility(c, weights),
        lambda c: float(c["quality_score"]),
        lambda c: 1.0 - float(c["support_removal_cost"]),
        lambda c: float(c["strength_score"]),
        lambda c: float(c["flexibility_score"]),
        lambda c: 1.0 / (1.0 + float(c["print_time_proxy"]) / 500.0),
    ]
    selected: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    def add(candidate: dict[str, Any]) -> None:
        key = _candidate_key(candidate)
        if key not in seen:
            seen.add(key)
            selected.append(candidate)

    for support_type in sorted({str(c["support_type"]) for c in pool}):
        matching = [c for c in pool if c["support_type"] == support_type]
        if matching:
            add(max(matching, key=lambda c: utility(c, weights)))
    for selector in selectors:
        add(max(pool, key=selector))
    for candidate in sorted(pool, key=lambda c: utility(c, weights), reverse=True):
        add(candidate)
        if len(selected) >= 8:
            break
    selected = selected[:8]
    if len(selected) < 2:
        raise ValueError("not enough distinct candidates for AI ranking")
    random.Random(seed).shuffle(selected)
    for candidate_id, candidate in enumerate(selected):
        candidate["candidate_id"] = candidate_id
    return selected


def build_finalists(
    mesh: trimesh.Trimesh,
    scorer: OrientationScorer,
    *,
    samples: int,
    seed: int,
    face_candidates: int,
    yaw_steps: int,
    support_types: tuple[str, ...],
    layer_height_mm: float,
    load_direction_model: tuple[float, float, float],
    force_n: float,
    walls: int,
    infill_percent: float,
    tpu_fraction: float,
    objective: dict[str, float],
) -> list[dict[str, Any]]:
    rotations = candidate_rotations(
        mesh,
        random_samples=samples,
        seed=seed,
        face_candidates=face_candidates,
        yaw_steps=yaw_steps,
    )
    orientation_rows = []
    for matrix, source in rotations:
        metrics = scorer.evaluate(matrix)
        orientation_rows.append((metrics.score, matrix, source, metrics))
    orientation_rows.sort(key=lambda row: row[0])
    orientation_rows = orientation_rows[: min(48, len(orientation_rows))]

    pool: list[dict[str, Any]] = []
    for _, matrix, source, metrics in orientation_rows:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            rotation_xyz = (
                Rotation.from_matrix(matrix).as_euler("xyz", degrees=True).round(3).tolist()
            )
        for support_type in support_types:
            for density, interface_layers, z_gap_mm in _CONFIG_PRESETS:
                simulation = simulate_print_configuration(
                    metrics,
                    matrix,
                    support_type=support_type,
                    support_density_percent=density,
                    interface_layers=interface_layers,
                    z_gap_mm=z_gap_mm,
                    layer_height_mm=layer_height_mm,
                    load_direction_model=load_direction_model,
                    force_n=force_n,
                    walls=walls,
                    infill_percent=infill_percent,
                    tpu_fraction=tpu_fraction,
                ).to_dict()
                pool.append(
                    {
                        "rotation_matrix": np.asarray(matrix).tolist(),
                        "rotation_xyz_deg": rotation_xyz,
                        "source": source,
                        "support_type": support_type,
                        "support_density_percent": density,
                        "interface_layers": interface_layers,
                        "z_gap_mm": z_gap_mm,
                        "quality_score": round(float(simulation["quality_score"]), 6),
                        "strength_score": round(float(simulation["strength_score"]), 6),
                        "flexibility_score": round(float(simulation["flexibility_score"]), 6),
                        "support_removal_cost": round(float(simulation["support_removal_cost"]), 6),
                        "support_material_proxy_mm3": round(
                            float(simulation["support_material_proxy_mm3"]), 2
                        ),
                        "print_time_proxy": round(float(simulation["print_time_proxy"]), 2),
                        "orientation_metrics": metrics.to_dict(),
                    }
                )
    return _pick_diverse_finalists(pool, objective, seed)


def build_prompt(
    *,
    scenario: str,
    load_direction_model: tuple[float, float, float],
    force_n: float,
    walls: int,
    infill_percent: float,
    tpu_fraction: float,
    objective: dict[str, float],
    candidates: list[dict[str, Any]],
) -> str:
    public_candidates = [
        {
            key: value
            for key, value in candidate.items()
            if key not in {"rotation_matrix", "orientation_metrics", "source"}
        }
        for candidate in candidates
    ]
    request = {
        "scenario": scenario,
        "load_direction_model": list(load_direction_model),
        "force_n": force_n,
        "walls": walls,
        "infill_percent": infill_percent,
        "tpu_fraction": tpu_fraction,
        "objective": objective,
        "candidates": public_candidates,
    }
    return (
        "3DORIENTER_RANKING_V1\n"
        'Select exactly one candidate from the supplied JSON. Return only {"chosen_candidate":N}, '
        "where N is an integer candidate_id.\n"
        "INPUT_JSON:\n" + json.dumps(request, separators=(",", ":")) + "\nOUTPUT_JSON:\n"
    )


def query_lmstudio(
    *, url: str, model: str, prompt: str, timeout_seconds: float = 120.0
) -> tuple[int, str]:
    payload = json.dumps(
        {"model": model, "prompt": prompt, "temperature": 0, "max_tokens": 24}
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{url}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LM Studio request failed: {exc}") from exc
    try:
        text = str(body["choices"][0]["text"]).strip()
        result = json.loads(text)
        candidate_id = int(result["chosen_candidate"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"LM Studio returned invalid output: {body!r}") from exc
    return candidate_id, text
