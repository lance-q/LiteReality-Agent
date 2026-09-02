"""Mask-aware metric point-cloud construction from one scanner frame."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .coordinate_frames import apply_transform, backproject_pixels
from .scanner_frame import ScannerFrame


@dataclass(frozen=True)
class WorldPointCloud:
    points_world_m: np.ndarray
    colors_rgb: np.ndarray
    robust_center_world_m: np.ndarray
    robust_dimensions_world_m: np.ndarray
    median_depth_m: float
    valid_depth_pixel_count: int


def load_full_image_binary_mask(mask_path: Path | str, frame: ScannerFrame) -> np.ndarray:
    """Load one independent object mask; non-zero is object and dimensions must equal RGB."""
    with Image.open(mask_path) as image:
        if image.size != (frame.rgb_width_px, frame.rgb_height_px):
            raise ValueError(
                f"mask size {image.size} must exactly match RGB "
                f"{(frame.rgb_width_px, frame.rgb_height_px)}"
            )
        return np.asarray(image.convert("L")) > 0


def _sample_rgb_at_depth_pixels(frame: ScannerFrame, pixels_uv: np.ndarray) -> np.ndarray:
    bgr = cv2.imread(str(frame.rgb_path), cv2.IMREAD_COLOR)
    if bgr is None or bgr.shape[:2] != (frame.rgb_height_px, frame.rgb_width_px):
        raise ValueError("RGB decode dimensions differ from scanner frame metadata")
    u_depth, v_depth = pixels_uv[:, 0], pixels_uv[:, 1]
    u_rgb = np.clip(
        np.floor((u_depth + 0.5) * frame.rgb_width_px / frame.depth_mm.shape[1]).astype(int),
        0,
        frame.rgb_width_px - 1,
    )
    v_rgb = np.clip(
        np.floor((v_depth + 0.5) * frame.rgb_height_px / frame.depth_mm.shape[0]).astype(int),
        0,
        frame.rgb_height_px - 1,
    )
    return bgr[v_rgb, u_rgb, ::-1]


def world_point_cloud_from_frame(
    frame: ScannerFrame,
    full_image_mask: np.ndarray | None = None,
    *,
    min_confidence: int | None = 1,
    min_depth_m: float = 0.05,
    max_depth_m: float = 8.0,
    reject_depth_outliers: bool = False,
    quantile_low: float = 0.02,
    quantile_high: float = 0.98,
) -> WorldPointCloud:
    if not 0 <= quantile_low < quantile_high <= 1:
        raise ValueError("robust quantiles must satisfy 0 <= low < high <= 1")
    valid = (frame.depth_mm > min_depth_m * 1000.0) & (frame.depth_mm < max_depth_m * 1000.0)
    if frame.confidence is not None and min_confidence is not None:
        valid &= frame.confidence >= min_confidence
    if full_image_mask is not None:
        mask = np.asarray(full_image_mask, dtype=np.uint8)
        if mask.shape != (frame.rgb_height_px, frame.rgb_width_px):
            raise ValueError("full_image_mask must exactly match the RGB image dimensions")
        mask_depth = cv2.resize(
            mask,
            (frame.depth_mm.shape[1], frame.depth_mm.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        valid &= mask_depth
    rows_v, columns_u = np.nonzero(valid)
    if len(columns_u) == 0:
        raise ValueError(f"frame {frame.frame_id} mask has no valid depth pixels")
    depth_m = frame.depth_mm[rows_v, columns_u].astype(np.float64) / 1000.0

    if reject_depth_outliers and len(depth_m) >= 8:
        median = np.median(depth_m)
        mad = np.median(np.abs(depth_m - median))
        if mad > 0:
            keep = np.abs(depth_m - median) <= 3.5 * 1.4826 * mad
        else:
            low, high = np.quantile(depth_m, [quantile_low, quantile_high])
            keep = (depth_m >= low) & (depth_m <= high)
        rows_v, columns_u, depth_m = rows_v[keep], columns_u[keep], depth_m[keep]
    if len(depth_m) == 0:
        raise ValueError("all depth pixels were rejected as outliers")

    pixels_uv = np.column_stack((columns_u, rows_v))
    points_camera_cv_m = backproject_pixels(pixels_uv, depth_m, frame.intrinsics_depth)
    points_world_m = apply_transform(frame.T_world_from_camera_cv, points_camera_cv_m)
    colors_rgb = _sample_rgb_at_depth_pixels(frame, pixels_uv)
    lower, upper = np.quantile(points_world_m, [quantile_low, quantile_high], axis=0)
    return WorldPointCloud(
        points_world_m=points_world_m,
        colors_rgb=colors_rgb,
        robust_center_world_m=(lower + upper) / 2.0,
        robust_dimensions_world_m=upper - lower,
        median_depth_m=float(np.median(depth_m)),
        valid_depth_pixel_count=len(depth_m),
    )


def write_ascii_ply(path: Path | str, cloud: WorldPointCloud) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(cloud.points_world_m)
    colors = np.asarray(cloud.colors_rgb)
    if points.shape != (len(colors), 3) or colors.shape[1:] != (3,):
        raise ValueError("PLY points and colors must both be Nx3")
    with path.open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(points)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point, color in zip(points, colors):
            stream.write(
                f"{point[0]:.7g} {point[1]:.7g} {point[2]:.7g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
    return path
