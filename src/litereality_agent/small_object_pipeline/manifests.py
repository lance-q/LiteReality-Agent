"""Small, lossless JSON manifests used by later pipeline phases."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ObservationManifest:
    frame_id: int
    detection_id: int
    label: str
    mask_path: str
    bbox_full_image: tuple[float, float, float, float]
    world_center_m: np.ndarray
    T_world_from_camera_cv: np.ndarray

    def save(self, path: Path | str) -> None:
        data = {
            "frame_id": self.frame_id,
            "detection_id": self.detection_id,
            "label": self.label,
            "mask_path": self.mask_path,
            "bbox_full_image": list(self.bbox_full_image),
            "world_center_m": np.asarray(self.world_center_m).tolist(),
            "T_world_from_camera_cv": np.asarray(self.T_world_from_camera_cv).tolist(),
        }
        Path(path).write_text(json.dumps(data, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path | str) -> "ObservationManifest":
        data = json.loads(Path(path).read_text())
        return cls(
            frame_id=int(data["frame_id"]),
            detection_id=int(data["detection_id"]),
            label=str(data["label"]),
            mask_path=str(data["mask_path"]),
            bbox_full_image=tuple(float(value) for value in data["bbox_full_image"]),
            world_center_m=np.asarray(data["world_center_m"], dtype=np.float64),
            T_world_from_camera_cv=np.asarray(data["T_world_from_camera_cv"], dtype=np.float64),
        )
