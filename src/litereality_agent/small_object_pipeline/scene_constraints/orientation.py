"""Deterministic upright and yaw candidate generation without geometry deformation."""

from __future__ import annotations

import math

import numpy as np

from .types import WallSurface

_WORLD_UP = np.array([0.0, 1.0, 0.0])
_SIGNED_LOCAL_AXES = (
    ("+X", np.array([1.0, 0.0, 0.0])),
    ("-X", np.array([-1.0, 0.0, 0.0])),
    ("+Y", np.array([0.0, 1.0, 0.0])),
    ("-Y", np.array([0.0, -1.0, 0.0])),
    ("+Z", np.array([0.0, 0.0, 1.0])),
    ("-Z", np.array([0.0, 0.0, -1.0])),
)


def rotation_y(yaw_rad: float) -> np.ndarray:
    cosine, sine = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=float) / np.linalg.norm(source)
    target = np.asarray(target, dtype=float) / np.linalg.norm(target)
    cross = np.cross(source, target)
    dot = float(np.clip(source @ target, -1.0, 1.0))
    if np.linalg.norm(cross) <= 1e-10:
        if dot > 0:
            return np.eye(3)
        helper = np.array([1.0, 0.0, 0.0])
        if abs(float(source @ helper)) > 0.9:
            helper = np.array([0.0, 0.0, 1.0])
        axis = np.cross(source, helper)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    skew = np.array(
        [[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]]
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - dot) / float(cross @ cross))


def _horizontal_reference(local_up: np.ndarray) -> np.ndarray:
    for axis in np.eye(3):
        if abs(float(axis @ local_up)) < 0.5:
            return axis
    raise AssertionError("a signed coordinate axis always has a perpendicular coordinate axis")


def _yaw_to_direction(
    base_rotation: np.ndarray, local_reference: np.ndarray, direction: np.ndarray
) -> float:
    current = base_rotation @ local_reference
    current_angle = math.atan2(float(current[2]), float(current[0]))
    target_angle = math.atan2(float(direction[2]), float(direction[0]))
    # rotation_y(theta) changes horizontal atan2(z, x) by -theta.
    return current_angle - target_angle


def _deduplicate(
    candidates: list[tuple[str, str, float, np.ndarray]],
) -> list[tuple[str, str, float, np.ndarray]]:
    kept: list[tuple[str, str, float, np.ndarray]] = []
    for candidate in candidates:
        if not any(np.allclose(candidate[3], accepted[3], atol=1e-7) for accepted in kept):
            kept.append(candidate)
    return kept


def generate_orientation_candidates(
    initial_rotation: np.ndarray,
    *,
    enforce_upright: bool,
    nearest_wall: WallSurface | None,
    include_cardinal_yaws: bool = True,
) -> list[tuple[str, str, float, np.ndarray]]:
    """Return ``(id, upright_axis, yaw_degrees, rotation)`` candidates."""
    initial = np.asarray(initial_rotation, dtype=float)
    candidates = [("current", "current", 0.0, initial)]
    if not enforce_upright:
        return candidates

    for axis_name, local_up in _SIGNED_LOCAL_AXES:
        base = _rotation_between(local_up, _WORLD_UP)
        local_reference = _horizontal_reference(local_up)
        yaw_values = [0.0]
        if include_cardinal_yaws:
            yaw_values.extend(math.radians(value) for value in (90.0, 180.0, 270.0))
        if nearest_wall is not None:
            wall_direction = nearest_wall.direction_world.copy()
            wall_direction[1] = 0.0
            wall_direction /= np.linalg.norm(wall_direction)
            wall_yaw = _yaw_to_direction(base, local_reference, wall_direction)
            yaw_values.extend(wall_yaw + math.radians(value) for value in (0.0, 90.0, 180.0, 270.0))
        for yaw in yaw_values:
            rotation = rotation_y(yaw) @ base
            yaw_deg = float(math.degrees(yaw) % 360.0)
            candidates.append((f"up_{axis_name}_yaw_{yaw_deg:.3f}", axis_name, yaw_deg, rotation))
    return _deduplicate(candidates)


def uniform_scale_and_rotation(T_world_from_object: np.ndarray) -> tuple[float, np.ndarray]:
    linear = np.asarray(T_world_from_object, dtype=float)[:3, :3]
    singular_values = np.linalg.svd(linear, compute_uv=False)
    if not np.allclose(singular_values, singular_values[0], rtol=1e-6, atol=1e-6):
        raise ValueError(
            f"scene constraints require uniform scale; singular values={singular_values}"
        )
    scale = float(np.mean(singular_values))
    approximate = linear / scale
    left, _, right = np.linalg.svd(approximate)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("initial object rotation is not orthonormal")
    return scale, rotation
