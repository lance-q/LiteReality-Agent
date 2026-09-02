"""Greedy multi-view association in authoritative LiteReality world coordinates."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ObjectTrack:
    object_id: str
    label: str
    observations: list[dict] = field(default_factory=list)

    @property
    def center_world_m(self) -> np.ndarray:
        return np.median(
            np.asarray([item["world_center_m"] for item in self.observations], dtype=float), axis=0
        )

    @property
    def dimensions_m(self) -> np.ndarray:
        return np.median(
            np.asarray([item["dimensions_m"] for item in self.observations], dtype=float), axis=0
        )


def _sizes_compatible(a: np.ndarray, b: np.ndarray, max_ratio: float) -> bool:
    a = np.maximum(np.asarray(a, dtype=float), 1e-4)
    b = np.maximum(np.asarray(b, dtype=float), 1e-4)
    return bool(np.max(np.maximum(a / b, b / a)) <= max_ratio)


def associate_observations(
    observations: list[dict], *, centroid_distance_m: float, max_size_ratio: float
) -> list[ObjectTrack]:
    tracks: list[ObjectTrack] = []
    for observation in sorted(observations, key=lambda item: (item["frame_id"], item["detection_id"])):
        center = np.asarray(observation["world_center_m"], dtype=float)
        candidates = [
            track
            for track in tracks
            if track.label == observation["label"]
            and np.linalg.norm(track.center_world_m - center) <= centroid_distance_m
            and _sizes_compatible(track.dimensions_m, observation["dimensions_m"], max_size_ratio)
        ]
        if candidates:
            track = min(candidates, key=lambda item: np.linalg.norm(item.center_world_m - center))
        else:
            track = ObjectTrack(f"object_{len(tracks):03d}", observation["label"])
            tracks.append(track)
        track.observations.append(observation)
    return tracks
