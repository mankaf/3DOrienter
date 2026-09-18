from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .scoring import OrientationMetrics

_EPS = 1e-12

_SUPPORT_PRESETS = {
    "normal": {"volume_factor": 1.0, "contact_factor": 1.0, "removal_factor": 1.0},
    "tree": {"volume_factor": 0.62, "contact_factor": 0.58, "removal_factor": 0.72},
}


@dataclass(frozen=True)
class PrintSimulation:
    support_type: str
    support_density_percent: float
    interface_layers: int
    z_gap_mm: float
    walls: int
    infill_percent: float
    tpu_fraction: float
    support_material_proxy_mm3: float
    support_contact_proxy_mm2: float
    support_removal_cost: float
    quality_score: float
    strength_score: float
    flexibility_score: float
    print_time_proxy: float
    load_z_alignment: float

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def _unit_vector(values: tuple[float, float, float]) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    length = float(np.linalg.norm(vector))
    if length <= _EPS:
        raise ValueError("load direction must not be the zero vector")
    return vector / length


def simulate_print_configuration(
    metrics: OrientationMetrics,
    rotation: np.ndarray,
    *,
    support_type: str,
    support_density_percent: float,
    interface_layers: int,
    z_gap_mm: float,
    layer_height_mm: float,
    load_direction_model: tuple[float, float, float] = (0.0, 0.0, 1.0),
    force_n: float = 100.0,
    walls: int = 3,
    infill_percent: float = 20.0,
    tpu_fraction: float = 0.0,
) -> PrintSimulation:
    """Return fast physics-inspired labels for AI ranking.

    These are deterministic proxies, not finite-element or slicer results.
    They let us build the data pipeline before adding voxel FEA and slicer
    integration.
    """
    if support_type not in _SUPPORT_PRESETS:
        raise ValueError(f"unsupported support type: {support_type}")
    if layer_height_mm <= 0 or z_gap_mm <= 0 or force_n < 0:
        raise ValueError("layer height and z gap must be positive; force cannot be negative")

    preset = _SUPPORT_PRESETS[support_type]
    density = float(np.clip(support_density_percent, 1.0, 100.0))
    interfaces = max(int(interface_layers), 0)
    walls = max(int(walls), 1)
    infill = float(np.clip(infill_percent, 0.0, 100.0))
    tpu = float(np.clip(tpu_fraction, 0.0, 1.0))

    density_factor = max(density / 15.0, 0.15)
    support_material = metrics.support_volume_proxy_mm3 * preset["volume_factor"] * density_factor
    support_contact = (
        metrics.overhang_area_mm2 * preset["contact_factor"] * (1.0 + 0.06 * interfaces)
    )

    gap_removal_factor = float(np.clip(0.2 / z_gap_mm, 0.45, 2.0))
    removal_raw = (
        preset["removal_factor"]
        * (0.55 * metrics.support_area_ratio + 0.45 * metrics.support_volume_ratio)
        * gap_removal_factor
        * (1.0 + 0.05 * interfaces)
    )
    support_removal_cost = float(np.clip(removal_raw, 0.0, 1.0))

    interface_factor = 1.10 if interfaces == 0 else max(0.62, 0.86 - 0.04 * interfaces)
    underside_loss = (
        metrics.support_area_ratio
        * preset["contact_factor"]
        * interface_factor
        * float(np.clip(z_gap_mm / 0.2, 0.5, 2.0))
    )
    quality_loss = float(np.clip(0.72 * metrics.quality_cost + 0.28 * underside_loss, 0.0, 1.0))
    quality_score = 1.0 - quality_loss

    load_model = _unit_vector(load_direction_model)
    load_world = load_model @ np.asarray(rotation, dtype=float).T
    load_z_alignment = float(abs(load_world[2]))

    process_factor = float(np.clip(0.50 + 0.08 * walls + 0.004 * infill, 0.45, 1.20))
    material_strength_factor = 1.0 - 0.45 * tpu
    layer_anisotropy_factor = 1.0 - 0.45 * load_z_alignment
    force_factor = 1.0 / (1.0 + force_n / 2500.0)
    strength_score = float(
        np.clip(
            process_factor * material_strength_factor * layer_anisotropy_factor * force_factor,
            0.0,
            1.0,
        )
    )

    flexibility_score = float(
        np.clip(
            0.04 + 0.84 * tpu + 0.08 * load_z_alignment + 0.0015 * max(0.0, 30.0 - infill),
            0.0,
            1.0,
        )
    )

    layer_count_proxy = metrics.height_mm / max(layer_height_mm, _EPS)
    print_time_proxy = float(layer_count_proxy + support_material / 25.0)

    return PrintSimulation(
        support_type=support_type,
        support_density_percent=density,
        interface_layers=interfaces,
        z_gap_mm=float(z_gap_mm),
        walls=walls,
        infill_percent=infill,
        tpu_fraction=tpu,
        support_material_proxy_mm3=float(support_material),
        support_contact_proxy_mm2=float(support_contact),
        support_removal_cost=support_removal_cost,
        quality_score=float(quality_score),
        strength_score=strength_score,
        flexibility_score=flexibility_score,
        print_time_proxy=print_time_proxy,
        load_z_alignment=load_z_alignment,
    )
