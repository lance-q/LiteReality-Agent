"""Strict loading of one LiteReality Scanner RGB-D frame."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .coordinate_frames import CameraIntrinsics, opencv_camera_pose_from_arkit


@dataclass(frozen=True)
class ScannerFrame:
    frame_id: int
    rgb_path: Path
    depth_path: Path
    confidence_path: Path | None
    rgb_width_px: int
    rgb_height_px: int
    depth_mm: np.ndarray
    confidence: np.ndarray | None
    intrinsics_rgb: CameraIntrinsics
    intrinsics_depth: CameraIntrinsics
    T_world_from_camera_arkit: np.ndarray
    T_world_from_camera_cv: np.ndarray


def _required(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"required non-empty capture file is missing: {path}")
    return path


def load_scanner_frame(scan_dir: Path | str, frame_id: int) -> ScannerFrame:
    scan_dir = Path(scan_dir).resolve()
    token = f"{frame_id:05d}"
    rgb_path = _required(scan_dir / f"frame_{token}.jpg")
    depth_path = _required(scan_dir / f"depth_{token}.png")
    json_path = _required(scan_dir / f"frame_{token}.json")
    confidence_path = scan_dir / f"conf_{token}.png"
    if not confidence_path.is_file() or confidence_path.stat().st_size == 0:
        confidence_path = None

    with Image.open(rgb_path) as rgb:
        rgb_width_px, rgb_height_px = rgb.size
        orientation = rgb.getexif().get(274, 1)
        if orientation != 1:
            raise ValueError(
                f"{rgb_path.name} has EXIF orientation {orientation}; pixel coordinates would be ambiguous"
            )

    depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_mm is None or depth_mm.ndim != 2 or depth_mm.dtype != np.uint16:
        raise ValueError(f"{depth_path} must be a readable single-channel uint16 PNG")
    depth_height_px, depth_width_px = depth_mm.shape

    confidence = None
    if confidence_path is not None:
        confidence = cv2.imread(str(confidence_path), cv2.IMREAD_UNCHANGED)
        if confidence is None or confidence.shape != depth_mm.shape:
            raise ValueError("confidence image must have exactly the depth image dimensions")

    metadata = json.loads(json_path.read_text())
    if metadata.get("frame_index") != frame_id:
        raise ValueError(
            f"{json_path.name} frame_index={metadata.get('frame_index')} does not match {frame_id}"
        )
    K = np.asarray(metadata["intrinsics"], dtype=np.float64).reshape(3, 3)
    if not np.allclose(K[2], [0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"unexpected intrinsic bottom row in {json_path}")
    intrinsics_rgb = CameraIntrinsics(
        rgb_width_px, rgb_height_px, K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    )
    intrinsics_depth = intrinsics_rgb.scaled_to(depth_width_px, depth_height_px)
    T_world_from_camera_arkit = np.asarray(
        metadata["cameraPoseARFrame"], dtype=np.float64
    ).reshape(4, 4)
    T_world_from_camera_cv = opencv_camera_pose_from_arkit(T_world_from_camera_arkit)

    return ScannerFrame(
        frame_id=frame_id,
        rgb_path=rgb_path,
        depth_path=depth_path,
        confidence_path=confidence_path,
        rgb_width_px=rgb_width_px,
        rgb_height_px=rgb_height_px,
        depth_mm=depth_mm,
        confidence=confidence,
        intrinsics_rgb=intrinsics_rgb,
        intrinsics_depth=intrinsics_depth,
        T_world_from_camera_arkit=T_world_from_camera_arkit,
        T_world_from_camera_cv=T_world_from_camera_cv,
    )
