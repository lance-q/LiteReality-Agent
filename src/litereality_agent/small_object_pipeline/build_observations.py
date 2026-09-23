"""Detection masks -> metric observations -> associated physical object manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .depth_backprojection import (
    load_full_image_binary_mask,
    world_point_cloud_from_frame,
    write_ascii_ply,
)
from .object_association import associate_observations, robust_observation_center
from .scanner_frame import load_scanner_frame
from .view_selection import choose_best_observation


def _observation(
    scan_dir: Path, output_dir: Path, detection: dict, *, min_confidence: int | None = 1
) -> dict:
    frame_id = int(detection["frame_id"])
    detection_id = int(detection["detection_id"])
    frame = load_scanner_frame(scan_dir, frame_id)
    mask = load_full_image_binary_mask(detection["mask_path"], frame)
    cloud = world_point_cloud_from_frame(
        frame,
        mask,
        min_confidence=min_confidence,
        reject_depth_outliers=True,
    )
    ply_path = output_dir / "observations" / f"frame_{frame_id:05d}_detection_{detection_id:03d}.ply"
    write_ascii_ply(ply_path, cloud)
    depth_mask = cv2.resize(
        mask.astype(np.uint8),
        (frame.depth_mm.shape[1], frame.depth_mm.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    bbox = detection["bbox_full_image"]
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    grayscale = cv2.imread(str(frame.rgb_path), cv2.IMREAD_GRAYSCALE)
    crop = grayscale[max(0, y1) : min(frame.rgb_height_px, y2), max(0, x1) : min(frame.rgb_width_px, x2)]
    sharpness = float(cv2.Laplacian(crop, cv2.CV_64F).var()) if crop.size else 0.0
    boundary_margin = min(x1, y1, frame.rgb_width_px - x2, frame.rgb_height_px - y2)
    return {
        **detection,
        "world_pointcloud_path": str(ply_path.resolve()),
        "world_center_m": cloud.robust_center_world_m.tolist(),
        "dimensions_m": cloud.robust_dimensions_world_m.tolist(),
        "median_depth_m": cloud.median_depth_m,
        "valid_depth_pixel_count": cloud.valid_depth_pixel_count,
        "minimum_depth_confidence": min_confidence,
        "valid_depth_fraction": cloud.valid_depth_pixel_count / max(1, int(depth_mask.sum())),
        "mask_area_fraction": int(mask.sum()) / mask.size,
        "boundary_margin_fraction": max(0.0, boundary_margin) / min(frame.rgb_width_px, frame.rgb_height_px),
        "sharpness": sharpness,
        "T_world_from_camera_cv": frame.T_world_from_camera_cv.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scan_dir", type=Path)
    parser.add_argument("detections_manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--centroid-distance-m", type=float, default=0.35)
    parser.add_argument("--max-size-ratio", type=float, default=2.5)
    parser.add_argument("--min-confidence", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--min-detection-score", type=float, default=0.0)
    parser.add_argument("--allowed-labels", nargs="*")
    parser.add_argument("--max-object-dimension-m", type=float)
    parser.add_argument("--skip-invalid-observations", action="store_true")
    args = parser.parse_args()
    raw = json.loads(args.detections_manifest.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    observations = []
    rejected = []
    allowed_labels = set(args.allowed_labels) if args.allowed_labels else None
    for frame_manifest in raw["frames"]:
        for detection in frame_manifest["detections"]:
            if float(detection["score"]) < args.min_detection_score:
                continue
            if allowed_labels is not None and detection["label"] not in allowed_labels:
                continue
            try:
                observation = _observation(
                    args.scan_dir,
                    args.output_dir,
                    detection,
                    min_confidence=args.min_confidence,
                )
            except (OSError, ValueError) as error:
                if not args.skip_invalid_observations:
                    raise
                rejected.append(
                    {
                        "frame_id": detection["frame_id"],
                        "detection_id": detection["detection_id"],
                        "reason": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            if args.max_object_dimension_m is not None and max(
                observation["dimensions_m"]
            ) > args.max_object_dimension_m:
                rejected.append(
                    {
                        "frame_id": detection["frame_id"],
                        "detection_id": detection["detection_id"],
                        "reason": "robust RGB-D dimensions exceed small-object limit",
                    }
                )
                continue
            observations.append(observation)
    tracks = associate_observations(
        observations,
        centroid_distance_m=args.centroid_distance_m,
        max_size_ratio=args.max_size_ratio,
    )
    objects = []
    for track in tracks:
        best, selection = choose_best_observation(track.observations)
        robust_center, center_diagnostics = robust_observation_center(track.observations)
        objects.append(
            {
                "object_id": track.object_id,
                "label": track.label,
                "observations": track.observations,
                "best_observation": {
                    "frame_id": best["frame_id"],
                    "detection_id": best["detection_id"],
                    "rgb_path": str((args.scan_dir / f"frame_{best['frame_id']:05d}.jpg").resolve()),
                    "mask_path": best["mask_path"],
                    "selection": selection,
                },
                "rgbd_world_center_m": robust_center.tolist(),
                "world_center_estimation": center_diagnostics,
                "rgbd_dimensions_m": track.dimensions_m.tolist(),
                "sam3d_asset": None,
                "T_world_from_object": None,
            }
        )
    manifest = {
        "scan_dir": str(args.scan_dir.resolve()),
        "minimum_depth_confidence": args.min_confidence,
        "minimum_detection_score": args.min_detection_score,
        "allowed_labels": sorted(allowed_labels) if allowed_labels else None,
        "maximum_object_dimension_m": args.max_object_dimension_m,
        "rejected_observations": rejected,
        "objects": objects,
    }
    output_path = args.output_dir / "objects_manifest.json"
    output_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {output_path}: {len(observations)} observations -> {len(objects)} objects")
    for item in objects:
        print(
            f"{item['object_id']} {item['label']}: {len(item['observations'])} views; "
            f"best frame {item['best_observation']['frame_id']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
