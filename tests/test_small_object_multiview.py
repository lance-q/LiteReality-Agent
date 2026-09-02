from __future__ import annotations

import json

import numpy as np

from litereality_agent.small_object_pipeline.detection import (
    InstanceDetection,
    deduplicate_same_frame,
    generate_tiles,
)
from litereality_agent.small_object_pipeline.frame_selection import select_keyframes
from litereality_agent.small_object_pipeline.object_association import associate_observations


def _write_frame(scan, frame_id, x_m, yaw_deg):
    angle = np.radians(yaw_deg)
    rotation = np.array(
        [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[0, 3] = x_m
    (scan / f"frame_{frame_id:05d}.json").write_text(
        json.dumps({"cameraPoseARFrame": transform.reshape(-1).tolist()})
    )
    (scan / f"frame_{frame_id:05d}.jpg").write_bytes(b"x")
    (scan / f"depth_{frame_id:05d}.png").write_bytes(b"x")


def test_keyframes_use_configurable_translation_or_rotation(tmp_path):
    for values in [(0, 0.0, 0.0), (1, 0.05, 2.0), (2, 0.21, 3.0), (3, 0.22, 20.0)]:
        _write_frame(tmp_path, *values)
    selected = select_keyframes(
        tmp_path, translation_threshold_m=0.18, rotation_threshold_deg=12.0
    )
    assert [item.frame_id for item in selected] == [0, 2, 3]


def test_overlapping_tiles_cover_trailing_image_edges():
    tiles = generate_tiles(1000, 800, 400, 400, 0.25)
    assert any(tile.crop_x_px + tile.crop_width_px == 1000 for tile in tiles)
    assert any(tile.crop_y_px + tile.crop_height_px == 800 for tile in tiles)


def test_same_frame_duplicates_merge_but_separate_masks_remain():
    shape = (100, 100)
    mask_a = np.zeros(shape, bool)
    mask_a[10:30, 10:30] = True
    mask_b = np.zeros(shape, bool)
    mask_b[11:31, 11:31] = True
    mask_c = np.zeros(shape, bool)
    mask_c[60:80, 60:80] = True
    detections = [
        InstanceDetection(1, 0, "cup", 0.9, [10, 10, 30, 30], mask_a),
        InstanceDetection(1, 1, "cup", 0.8, [11, 11, 31, 31], mask_b),
        InstanceDetection(1, 2, "cup", 0.7, [60, 60, 80, 80], mask_c),
    ]
    kept = deduplicate_same_frame(detections, bbox_iou_threshold=0.5, mask_iou_threshold=0.5)
    assert len(kept) == 2
    assert kept[0].mask_full_image is not kept[1].mask_full_image


def test_near_multiview_observations_associate_but_distant_same_label_does_not():
    def observation(frame, center):
        return {
            "frame_id": frame,
            "detection_id": 0,
            "label": "cup",
            "world_center_m": center,
            "dimensions_m": [0.1, 0.12, 0.1],
        }

    tracks = associate_observations(
        [observation(1, [1.3, 0.81, -2.1]), observation(2, [1.32, 0.8, -2.09]), observation(3, [3, 1, 0])],
        centroid_distance_m=0.25,
        max_size_ratio=2.0,
    )
    assert [len(track.observations) for track in tracks] == [2, 1]
