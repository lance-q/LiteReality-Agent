"""Greedy multi-view association in authoritative LiteReality world coordinates."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def robust_observation_center(
    observations: list[dict],
    *,
    minimum_relative_reliability: float = 0.1,
    consensus_radius_multiplier: float = 2.5,
) -> tuple[np.ndarray, dict]:
    """Reliability-weighted center after rejecting weak and spatially inconsistent views.

    Reliability is the valid depth point count multiplied by valid-depth fraction. For three or
    more surviving views, a medoid consensus rejects centers far from the closest view cluster.
    This keeps a sparse/poor-depth observation from pulling placement halfway toward an outlier.
    """
    if not observations:
        raise ValueError("at least one observation is required")
    centers = np.asarray([item["world_center_m"] for item in observations], dtype=float)
    reliability = np.asarray(
        [
            max(1.0, float(item.get("valid_depth_pixel_count", 1)))
            * max(0.0, float(item.get("valid_depth_fraction", 1.0)))
            for item in observations
        ],
        dtype=float,
    )
    quality_mask = reliability >= minimum_relative_reliability * float(reliability.max())
    candidate_indices = np.flatnonzero(quality_mask)
    if len(candidate_indices) >= 3:
        candidate_centers = centers[candidate_indices]
        distances = np.linalg.norm(
            candidate_centers[:, None, :] - candidate_centers[None, :, :], axis=-1
        )
        medoid_local_index = int(np.argmin(np.median(distances, axis=1)))
        medoid_distances = distances[medoid_local_index]
        nonzero = medoid_distances[medoid_distances > 1e-9]
        # The closest neighbor defines the local agreement scale; using all medoid distances
        # would let one remote view inflate its own acceptance radius.
        typical_distance_m = float(np.min(nonzero)) if len(nonzero) else 0.0
        consensus_radius_m = max(0.03, consensus_radius_multiplier * typical_distance_m)
        candidate_indices = candidate_indices[medoid_distances <= consensus_radius_m]
    else:
        consensus_radius_m = None
    selected_weights = reliability[candidate_indices]
    center = np.average(centers[candidate_indices], axis=0, weights=selected_weights)
    diagnostics = {
        "method": "valid-depth-reliability weighted center with medoid consensus",
        "observation_frame_ids": [int(item["frame_id"]) for item in observations],
        "reliability_weights": reliability.tolist(),
        "inlier_frame_ids": [int(observations[index]["frame_id"]) for index in candidate_indices],
        "consensus_radius_m": consensus_radius_m,
    }
    return center, diagnostics


@dataclass
class ObjectTrack:
    object_id: str
    label: str
    observations: list[dict] = field(default_factory=list)

    @property
    def center_world_m(self) -> np.ndarray:
        center, _ = robust_observation_center(self.observations)
        return center

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
