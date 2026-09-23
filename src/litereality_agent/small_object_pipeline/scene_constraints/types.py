"""Shared types for the per-object scene-constraint solver.

All coordinates are LiteReality/ARKit world coordinates in metres.  World +Y is up and
homogeneous matrices multiply column vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

SupportMode = Literal["none", "floor", "horizontal", "wall"]


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray, tolerance_m: float = 0.0) -> bool:
    """Ray-cast containment with an inexpensive tolerance based on edge distance."""
    x, y = np.asarray(point, dtype=float)
    vertices = np.asarray(polygon, dtype=float)
    if len(vertices) < 3:
        return False
    inside = False
    previous = vertices[-1]
    for current in vertices:
        x1, y1 = previous
        x2, y2 = current
        edge = current - previous
        edge_length_sq = float(edge @ edge)
        if edge_length_sq > 1e-12 and tolerance_m > 0:
            parameter = float(np.clip((point - previous) @ edge / edge_length_sq, 0.0, 1.0))
            if np.linalg.norm(point - (previous + parameter * edge)) <= tolerance_m:
                return True
        if (y1 > y) != (y2 > y):
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


@dataclass(frozen=True)
class SupportSurface:
    name: str
    category: str
    height_world_y_m: float
    footprint_world_xz_m: np.ndarray
    source: str

    def contains_xz(self, point_xz_m: np.ndarray, *, tolerance_m: float = 0.0) -> bool:
        return _point_in_polygon(
            np.asarray(point_xz_m, dtype=float),
            self.footprint_world_xz_m,
            tolerance_m,
        )

    def coverage(self, points_xz_m: np.ndarray, *, tolerance_m: float = 0.0) -> float:
        points = np.asarray(points_xz_m, dtype=float)
        if len(points) == 0:
            return 0.0
        supported = sum(self.contains_xz(point, tolerance_m=tolerance_m) for point in points)
        return float(supported / len(points))


@dataclass(frozen=True)
class WallSurface:
    name: str
    center_world_m: np.ndarray
    direction_world: np.ndarray
    normal_world: np.ndarray
    width_m: float
    height_m: float


@dataclass(frozen=True)
class CandidatePose:
    candidate_id: str
    T_world_from_object: np.ndarray
    rotation_world_from_object: np.ndarray
    uniform_scale: float
    upright_axis: str
    yaw_deg: float
    support: SupportSurface | None
    support_snap_m: float
    support_coverage: float | None
    transformed_vertices_world_m: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class CandidateContext:
    label: str
    profile_name: str
    target_center_world_m: np.ndarray
    rgbd_dimensions_m: np.ndarray
    observation_centers_world_m: np.ndarray
    floor: SupportSurface | None
    nearest_wall: WallSurface | None


class CandidateScoringHook(Protocol):
    """Extension point for later mask reprojection or depth rendering terms."""

    def __call__(self, candidate: CandidatePose, context: CandidateContext) -> dict[str, float]: ...

