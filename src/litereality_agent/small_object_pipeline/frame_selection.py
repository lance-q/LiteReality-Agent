"""Pose-motion keyframe selection without removing useful multi-view overlap."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Keyframe:
    frame_id: int
    translation_from_previous_m: float
    rotation_from_previous_deg: float


def _rotation_distance_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = a.T @ b
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def discover_complete_frame_ids(scan_dir: Path | str) -> list[int]:
    scan_dir = Path(scan_dir)
    ids = []
    for metadata_path in sorted(scan_dir.glob("frame_*.json")):
        frame_id = int(metadata_path.stem.rsplit("_", 1)[1])
        if (scan_dir / f"frame_{frame_id:05d}.jpg").is_file() and (
            scan_dir / f"depth_{frame_id:05d}.png"
        ).is_file():
            ids.append(frame_id)
    return ids


def select_keyframes(
    scan_dir: Path | str,
    *,
    translation_threshold_m: float,
    rotation_threshold_deg: float,
    frame_ids: list[int] | None = None,
) -> list[Keyframe]:
    if translation_threshold_m < 0 or rotation_threshold_deg < 0:
        raise ValueError("keyframe thresholds cannot be negative")
    scan_dir = Path(scan_dir)
    candidates = frame_ids or discover_complete_frame_ids(scan_dir)
    selected: list[Keyframe] = []
    last_T = None
    for frame_id in candidates:
        data = json.loads((scan_dir / f"frame_{frame_id:05d}.json").read_text())
        T_world_from_camera_arkit = np.asarray(data["cameraPoseARFrame"], dtype=float).reshape(4, 4)
        if last_T is None:
            selected.append(Keyframe(frame_id, 0.0, 0.0))
            last_T = T_world_from_camera_arkit
            continue
        translation_m = float(
            np.linalg.norm(T_world_from_camera_arkit[:3, 3] - last_T[:3, 3])
        )
        rotation_deg = _rotation_distance_deg(
            last_T[:3, :3], T_world_from_camera_arkit[:3, :3]
        )
        if translation_m >= translation_threshold_m or rotation_deg >= rotation_threshold_deg:
            selected.append(Keyframe(frame_id, translation_m, rotation_deg))
            last_T = T_world_from_camera_arkit
    return selected
