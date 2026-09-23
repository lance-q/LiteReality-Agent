"""Independent, missing-data-tolerant score terms for scene-constraint candidates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import CandidateContext, CandidatePose, CandidateScoringHook


@dataclass(frozen=True)
class ScoreWeights:
    center: float = 4.0
    dimensions: float = 0.5
    multiview_center: float = 0.5
    support: float = 5.0
    upright: float = 0.25
    wall: float = 2.0
    collision: float = 5.0
    mask: float = 1.0
    depth: float = 1.0


def _robust_center(vertices: np.ndarray) -> np.ndarray:
    return (np.quantile(vertices, 0.02, axis=0) + np.quantile(vertices, 0.98, axis=0)) / 2.0


def _major_horizontal_axis(vertices: np.ndarray) -> np.ndarray | None:
    points = np.asarray(vertices, dtype=float)[:, (0, 2)]
    centered = points - np.median(points, axis=0)
    covariance = centered.T @ centered
    values, vectors = np.linalg.eigh(covariance)
    if values[-1] <= 1e-12:
        return None
    axis_xz = vectors[:, -1]
    return np.array([axis_xz[0], 0.0, axis_xz[1]])


def _wall_crossing(vertices: np.ndarray, context: CandidateContext) -> float:
    wall = context.nearest_wall
    if wall is None:
        return 0.0
    relative = vertices - wall.center_world_m
    along = relative @ wall.direction_world
    relevant = np.abs(along) <= wall.width_m / 2 + 0.05
    if not np.any(relevant):
        return 0.0
    signed = relative[relevant] @ wall.normal_world
    negative = max(0.0, -float(np.min(signed)) - 0.03)
    positive = max(0.0, float(np.max(signed)) - 0.03)
    return min(negative, positive) / 0.10 if negative > 0 and positive > 0 else 0.0


def score_candidate(
    candidate: CandidatePose,
    context: CandidateContext,
    *,
    weights: ScoreWeights,
    require_support: bool,
    require_upright: bool,
    wall_alignment: bool,
    minimum_support_coverage: float,
    hooks: tuple[CandidateScoringHook, ...] = (),
) -> tuple[float, dict[str, float]]:
    vertices = candidate.transformed_vertices_world_m
    center = _robust_center(vertices)
    center_delta = center - context.target_center_world_m
    center_error = float(np.linalg.norm(center_delta[(0, 2)]) / 0.10 + abs(center_delta[1]) / 0.25)

    dimensions = np.ptp(vertices, axis=0)
    target_dimensions = np.asarray(context.rgbd_dimensions_m, dtype=float)
    observable = target_dimensions > 1e-5
    if np.any(observable):
        dimension_error = float(
            np.mean(
                np.abs(
                    np.log(
                        np.maximum(dimensions[observable], 1e-6)
                        / target_dimensions[observable]
                    )
                )
            )
        )
    else:
        dimension_error = 0.0

    observations = context.observation_centers_world_m
    multiview_error = (
        float(np.median(np.linalg.norm(observations - center, axis=1)) / 0.15)
        if len(observations)
        else 0.0
    )

    if candidate.support is None:
        support_error = 1.0 if require_support else 0.0
    else:
        coverage = candidate.support_coverage if candidate.support_coverage is not None else 0.0
        support_error = max(0.0, minimum_support_coverage - coverage) / max(
            minimum_support_coverage, 1e-6
        )
        support_error += abs(candidate.support_snap_m) / 0.50

    upright_error = (
        0.0 if candidate.upright_axis != "current" else (0.25 if require_upright else 0.0)
    )
    wall_error = 0.0
    if wall_alignment and context.nearest_wall is not None:
        major_axis = _major_horizontal_axis(vertices)
        if major_axis is not None:
            wall_error = 1.0 - abs(float(major_axis @ context.nearest_wall.direction_world))

    robust_bottom = float(np.quantile(vertices[:, 1], 0.02))
    floor_penetration = 0.0
    if context.floor is not None:
        floor_penetration = max(0.0, context.floor.height_world_y_m - robust_bottom - 0.03) / 0.10
    collision_error = floor_penetration + _wall_crossing(vertices, context)

    terms = {
        "center": center_error,
        "dimensions": dimension_error,
        "multiview_center": multiview_error,
        "support": support_error,
        "upright": upright_error,
        "wall": wall_error,
        "collision": collision_error,
    }
    for hook in hooks:
        for name, value in hook(candidate, context).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"scoring hook returned invalid {name}={value}")
            terms[name] = float(value)

    total = sum(getattr(weights, name, 1.0) * value for name, value in terms.items())
    return float(total), terms
