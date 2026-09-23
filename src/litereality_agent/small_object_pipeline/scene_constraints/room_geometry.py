"""Read conservative support and wall primitives directly from a RoomPlan USDZ.

RoomPlan USDA matrices in these captures use row-vector convention (translation in the final row).
The resulting surfaces are converted immediately into the authoritative ARKit Y-up world frame.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np

from litereality_agent.agent.tools.shared.stitch_wall_image.stitch_wall import load_wall_planes

from .types import SupportSurface, WallSurface

_NUMBER = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
_CUBE_LOCAL = np.array(
    [[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)],
    dtype=float,
)


def _numbers(value: str) -> list[float]:
    return [float(item) for item in re.findall(_NUMBER, value)]


def _matrix(usda: str) -> np.ndarray:
    match = re.search(r"matrix4d\s+xformOp:transform\s*=\s*\(\s*\((.*?)\)\s*\)", usda, re.S)
    if not match:
        return np.eye(4)
    values = _numbers(match.group(1))
    if len(values) != 16:
        raise ValueError(f"RoomPlan transform contains {len(values)} values, expected 16")
    return np.asarray(values, dtype=float).reshape(4, 4)


def _transform_points_row(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.c_[points, np.ones(len(points))]
    return (homogeneous @ transform)[:, :3]


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Dependency-free monotonic-chain hull for small RoomPlan footprints."""
    unique = sorted({(float(point[0]), float(point[1])) for point in np.asarray(points)})
    if len(unique) <= 2:
        return np.asarray(unique, dtype=float)

    def cross(origin, a, b) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _floor_surface(name: str, usda: str) -> SupportSurface | None:
    points_match = re.search(r"point3f\[\]\s+points\s*=\s*\[(.*?)\]", usda, re.S)
    if not points_match:
        return None
    tuples = re.findall(r"\(([^()]*)\)", points_match.group(1))
    points = np.asarray([_numbers(value)[:3] for value in tuples], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        return None
    world = _transform_points_row(points, _matrix(usda))
    height = float(np.max(world[:, 1]))
    top = world[world[:, 1] >= height - 0.01]
    footprint = _convex_hull(top[:, (0, 2)])
    if len(footprint) < 3:
        return None
    return SupportSurface(name, "floor", height, footprint, "RoomPlan floor mesh")


def _box_surface(name: str, category: str, usda: str) -> SupportSurface | None:
    scale_match = re.search(r"double3\s+xformOp:scale\s*=\s*\((.*?)\)", usda, re.S)
    if not scale_match:
        return None
    scale = np.asarray(_numbers(scale_match.group(1)), dtype=float)
    if scale.shape != (3,) or np.any(scale <= 0):
        return None
    transform = _matrix(usda)
    axes = transform[:3, :3]
    vertical_axis = int(np.argmax(np.abs(axes @ np.array([0.0, 1.0, 0.0]))))
    vertical_alignment = abs(float(axes[vertical_axis] @ np.array([0.0, 1.0, 0.0])))
    if vertical_alignment < 0.90:
        return None
    corners = _transform_points_row(_CUBE_LOCAL * scale, transform)
    height = float(np.max(corners[:, 1]))
    top = corners[corners[:, 1] >= height - 0.01]
    footprint = _convex_hull(top[:, (0, 2)])
    if len(footprint) < 3:
        footprint = _convex_hull(corners[:, (0, 2)])
    if len(footprint) < 3:
        return None
    return SupportSurface(name, category.lower(), height, footprint, "RoomPlan parametric box")


def load_room_constraints(usdz_path: Path | str) -> tuple[list[SupportSurface], list[WallSurface]]:
    path = Path(usdz_path)
    supports: list[SupportSurface] = []
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            floor_match = re.search(r"assets/Parametric/Floors/(Floor\d+)\.usda$", member)
            box_match = re.search(r"assets/Parametric/([^/]+)/([^/]+)\.usda$", member)
            if not floor_match and not box_match:
                continue
            usda = archive.read(member).decode("utf-8")
            if floor_match:
                surface = _floor_surface(floor_match.group(1), usda)
            else:
                category, name = box_match.groups()
                if category in {"Walls", "Floors"}:
                    continue
                surface = _box_surface(name, category, usda)
            if surface is not None:
                supports.append(surface)

    walls = [
        WallSurface(
            plane.name,
            np.asarray(plane.center, dtype=float),
            np.asarray(plane.u_axis, dtype=float),
            np.asarray(plane.normal, dtype=float),
            plane.width_m,
            plane.height_m,
        )
        for plane in load_wall_planes(path)
    ]
    return supports, walls


def nearest_wall(
    point_world_m: np.ndarray, walls: list[WallSurface], *, maximum_distance_m: float
) -> tuple[WallSurface | None, float | None]:
    point = np.asarray(point_world_m, dtype=float)
    best: tuple[WallSurface, float] | None = None
    for wall in walls:
        relative = point - wall.center_world_m
        along = float(relative @ wall.direction_world)
        if abs(along) > wall.width_m / 2 + 0.25:
            continue
        distance = abs(float(relative @ wall.normal_world))
        if distance <= maximum_distance_m and (best is None or distance < best[1]):
            best = (wall, distance)
    return best if best is not None else (None, None)
