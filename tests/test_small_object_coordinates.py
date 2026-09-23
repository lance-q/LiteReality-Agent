from __future__ import annotations

import numpy as np
import pytest

from litereality_agent.small_object_pipeline.coordinate_frames import (
    CameraIntrinsics,
    CropTransform,
    apply_transform,
    backproject_pixels,
    opencv_camera_pose_from_arkit,
)
from litereality_agent.small_object_pipeline.manifests import ObservationManifest


def test_crop_coordinate_inversion_restores_full_image_pixels():
    crop = CropTransform(
        crop_x_px=500,
        crop_y_px=300,
        crop_width_px=600,
        crop_height_px=600,
        network_width_px=1200,
        network_height_px=1200,
    )
    result = crop.network_to_full_pixels(np.array([[0, 0], [600, 300], [1200, 1200]]))
    np.testing.assert_allclose(result, [[500, 300], [800, 450], [1100, 900]])


def test_synthetic_backprojection_uses_opencv_camera_basis():
    intrinsics = CameraIntrinsics(640, 480, fx_px=100, fy_px=200, cx_px=320, cy_px=240)
    points = backproject_pixels(
        np.array([[320, 240], [420, 440]], dtype=float),
        np.array([2.0, 4.0]),
        intrinsics,
    )
    np.testing.assert_allclose(points, [[0, 0, 2], [4, 4, 4]])


def test_arkit_to_depth_camera_basis_and_world_transform():
    T_world_from_camera_arkit = np.eye(4)
    T_world_from_camera_arkit[:3, 3] = [1, 2, 3]
    T_world_from_camera_cv = opencv_camera_pose_from_arkit(T_world_from_camera_arkit)
    points_world = apply_transform(T_world_from_camera_cv, np.array([[1, 2, 3]]))
    # OpenCV +Y down / +Z forward map to ARKit -Y / -Z; X is unchanged.
    np.testing.assert_allclose(points_world, [[2, 0, 0]])


def test_transform_persistence_is_numerically_lossless(tmp_path):
    transform = np.array(
        [[0, 0, -1, 1.25], [0, 1, 0, 0.75], [1, 0, 0, -2.5], [0, 0, 0, 1]],
        dtype=float,
    )
    manifest = ObservationManifest(
        frame_id=123,
        detection_id=2,
        label="cup",
        mask_path="masks/detection_002.png",
        bbox_full_image=(10.5, 20.25, 80.5, 100.75),
        world_center_m=np.array([1.3, 0.81, -2.1]),
        T_world_from_camera_cv=transform,
    )
    path = tmp_path / "observation.json"
    manifest.save(path)
    loaded = ObservationManifest.load(path)
    np.testing.assert_array_equal(loaded.T_world_from_camera_cv, transform)
    np.testing.assert_array_equal(loaded.world_center_m, manifest.world_center_m)
    assert loaded.bbox_full_image == manifest.bbox_full_image


def test_backprojection_rejects_nonpositive_depth():
    intrinsics = CameraIntrinsics(1, 1, 1, 1, 0, 0)
    with pytest.raises(ValueError, match="strictly positive"):
        backproject_pixels(np.array([[0, 0]]), np.array([0.0]), intrinsics)
