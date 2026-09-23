"""Tiling, coordinate restoration, and same-frame duplicate suppression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coordinate_frames import CropTransform


@dataclass(frozen=True)
class ImageTile:
    crop_x_px: int
    crop_y_px: int
    crop_width_px: int
    crop_height_px: int

    def transform(self, network_width_px: int, network_height_px: int) -> CropTransform:
        return CropTransform(
            self.crop_x_px,
            self.crop_y_px,
            self.crop_width_px,
            self.crop_height_px,
            network_width_px,
            network_height_px,
        )


@dataclass
class InstanceDetection:
    frame_id: int
    detection_id: int
    label: str
    score: float
    bbox_full_image: list[float]
    mask_full_image: np.ndarray
    source_tile: ImageTile | None = None


def generate_tiles(
    image_width_px: int,
    image_height_px: int,
    tile_width_px: int,
    tile_height_px: int,
    overlap_fraction: float,
) -> list[ImageTile]:
    if not 0 <= overlap_fraction < 1:
        raise ValueError("tile overlap must be in [0, 1)")
    if min(image_width_px, image_height_px, tile_width_px, tile_height_px) <= 0:
        raise ValueError("image and tile dimensions must be positive")
    step_x = max(1, round(tile_width_px * (1 - overlap_fraction)))
    step_y = max(1, round(tile_height_px * (1 - overlap_fraction)))
    xs = list(range(0, max(1, image_width_px - tile_width_px + 1), step_x))
    ys = list(range(0, max(1, image_height_px - tile_height_px + 1), step_y))
    xs.append(max(0, image_width_px - tile_width_px))
    ys.append(max(0, image_height_px - tile_height_px))
    return [
        ImageTile(x, y, min(tile_width_px, image_width_px - x), min(tile_height_px, image_height_px - y))
        for y in sorted(set(ys))
        for x in sorted(set(xs))
    ]


def bbox_iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("mask IoU requires equal full-image dimensions")
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection / union) if union else 0.0


def deduplicate_same_frame(
    detections: list[InstanceDetection],
    *,
    bbox_iou_threshold: float,
    mask_iou_threshold: float,
) -> list[InstanceDetection]:
    kept: list[InstanceDetection] = []
    for candidate in sorted(detections, key=lambda item: item.score, reverse=True):
        duplicate = any(
            candidate.label == accepted.label
            and (
                bbox_iou(candidate.bbox_full_image, accepted.bbox_full_image) >= bbox_iou_threshold
                or mask_iou(candidate.mask_full_image, accepted.mask_full_image) >= mask_iou_threshold
            )
            for accepted in kept
        )
        if not duplicate:
            candidate.detection_id = len(kept)
            kept.append(candidate)
    return kept
