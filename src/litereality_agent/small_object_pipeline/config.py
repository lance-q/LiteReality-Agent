"""Configuration for the small multi-view MVP; all thresholds are explicit and serializable."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SmallObjectConfig:
    # Scanner captures are already spaced by roughly 0.18 m. These defaults reduce the
    # 96-frame example scan to a practical subset while retaining multi-view overlap.
    keyframe_translation_threshold_m: float = 0.50
    keyframe_rotation_threshold_deg: float = 25.0
    detection_box_threshold: float = 0.25
    detection_text_threshold: float = 0.25
    same_frame_bbox_iou_threshold: float = 0.55
    same_frame_mask_iou_threshold: float = 0.60
    association_centroid_distance_m: float = 0.35
    association_max_size_ratio: float = 2.5
    robust_quantile_low: float = 0.02
    robust_quantile_high: float = 0.98
    min_valid_depth_pixels: int = 20
    tile_width_px: int | None = None
    tile_height_px: int | None = None
    tile_overlap_fraction: float = 0.25

    def to_dict(self) -> dict:
        return asdict(self)
