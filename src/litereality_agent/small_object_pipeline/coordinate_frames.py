"""Explicit coordinate-frame operations for scanner RGB-D data.

Conventions
-----------
All 3D points are column vectors.  Homogeneous transforms are named
``T_<destination>_from_<source>`` and are applied as ``p_destination = T @ p_source``.

ARKit/RoomPlan world is right-handed, metres, +Y up.  ARKit camera coordinates are +X right,
+Y up, -Z forward.  Depth intrinsics use the OpenCV/Open3D camera basis (+X right, +Y down,
+Z forward).  Therefore::

    T_world_from_camera_cv = T_world_from_camera_arkit @ diag(1, -1, -1, 1)

This is the same two-column flip performed by LiteReality's scanner loader before it writes an
``extrinsic_N.npy`` file.  No world-axis conversion occurs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

T_ARKIT_CAMERA_FROM_OPENCV_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass(frozen=True)
class CameraIntrinsics:
    width_px: int
    height_px: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float

    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx_px, 0.0, self.cx_px], [0.0, self.fy_px, self.cy_px], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def scaled_to(self, width_px: int, height_px: int) -> "CameraIntrinsics":
        if self.width_px <= 0 or self.height_px <= 0 or width_px <= 0 or height_px <= 0:
            raise ValueError("intrinsic image dimensions must be positive")
        sx = width_px / self.width_px
        sy = height_px / self.height_px
        return CameraIntrinsics(
            width_px=width_px,
            height_px=height_px,
            fx_px=self.fx_px * sx,
            fy_px=self.fy_px * sy,
            cx_px=self.cx_px * sx,
            cy_px=self.cy_px * sy,
        )


def validate_homogeneous_transform(matrix: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
        raise ValueError(f"{name} has invalid homogeneous bottom row: {matrix[3].tolist()}")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-3):
        raise ValueError(f"{name} rotation is not right-handed")
    return matrix


def opencv_camera_pose_from_arkit(T_world_from_camera_arkit: np.ndarray) -> np.ndarray:
    """Convert only the camera basis; output remains in authoritative ARKit world coordinates."""
    T_world_from_camera_arkit = validate_homogeneous_transform(
        T_world_from_camera_arkit, "T_world_from_camera_arkit"
    )
    return T_world_from_camera_arkit @ T_ARKIT_CAMERA_FROM_OPENCV_CAMERA


def backproject_pixels(
    pixels_uv: np.ndarray, depth_m: np.ndarray, intrinsics: CameraIntrinsics
) -> np.ndarray:
    """Backproject full depth-image pixels to OpenCV camera coordinates.

    ``pixels_uv`` is Nx2 in (u, v) order and ``depth_m`` is N metric Z-depth values, not ray
    distance.  Returned ``points_camera_cv`` is Nx3 with +X right, +Y down, +Z forward.
    """
    pixels_uv = np.asarray(pixels_uv, dtype=np.float64)
    depth_m = np.asarray(depth_m, dtype=np.float64)
    if pixels_uv.ndim != 2 or pixels_uv.shape[1] != 2:
        raise ValueError(f"pixels_uv must have shape (N, 2), got {pixels_uv.shape}")
    if depth_m.shape != (len(pixels_uv),):
        raise ValueError("depth_m must contain exactly one value per pixel")
    if not np.isfinite(depth_m).all() or np.any(depth_m <= 0):
        raise ValueError("depth_m must be finite and strictly positive")
    u, v = pixels_uv[:, 0], pixels_uv[:, 1]
    z = depth_m
    x = (u - intrinsics.cx_px) * z / intrinsics.fx_px
    y = (v - intrinsics.cy_px) * z / intrinsics.fy_px
    return np.column_stack((x, y, z))


def apply_transform(T_destination_from_source: np.ndarray, points_source: np.ndarray) -> np.ndarray:
    """Apply a 4x4 column-vector transform to an Nx3 point array."""
    transform = np.asarray(T_destination_from_source, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4)")
    points_source = np.asarray(points_source, dtype=np.float64)
    if points_source.ndim != 2 or points_source.shape[1] != 3:
        raise ValueError("points_source must have shape (N, 3)")
    homogeneous = np.column_stack((points_source, np.ones(len(points_source))))
    return (transform @ homogeneous.T).T[:, :3]


@dataclass(frozen=True)
class CropTransform:
    """Invert a resized detection tile back to original full-image pixel coordinates."""

    crop_x_px: float
    crop_y_px: float
    crop_width_px: float
    crop_height_px: float
    network_width_px: int
    network_height_px: int

    def network_to_full_pixels(self, pixels_uv_network: np.ndarray) -> np.ndarray:
        pixels = np.asarray(pixels_uv_network, dtype=np.float64)
        if pixels.ndim != 2 or pixels.shape[1] != 2:
            raise ValueError("pixels_uv_network must have shape (N, 2)")
        if min(
            self.crop_width_px,
            self.crop_height_px,
            self.network_width_px,
            self.network_height_px,
        ) <= 0:
            raise ValueError("crop and network dimensions must be positive")
        out = pixels.copy()
        out[:, 0] = self.crop_x_px + pixels[:, 0] * self.crop_width_px / self.network_width_px
        out[:, 1] = self.crop_y_px + pixels[:, 1] * self.crop_height_px / self.network_height_px
        return out
