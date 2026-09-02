"""Phase 1/2 command line: inspect scanner coordinates and export world-space depth PLYs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import SmallObjectConfig
from .depth_backprojection import (
    load_full_image_binary_mask,
    world_point_cloud_from_frame,
    write_ascii_ply,
)
from .frame_selection import discover_complete_frame_ids, select_keyframes
from .object_alignment import align_sam3d_glb, alignment_diagnostics_dict
from .scanner_frame import load_scanner_frame


def _print_inspection(scan_dir: Path, frame_id: int) -> None:
    frame = load_scanner_frame(scan_dir, frame_id)
    print(f"scan: {scan_dir}")
    print(f"RGB: {frame.rgb_path} ({frame.rgb_width_px}x{frame.rgb_height_px})")
    print(
        f"depth: {frame.depth_path} ({frame.depth_mm.shape[1]}x{frame.depth_mm.shape[0]}, "
        "uint16 millimetres)"
    )
    print(f"intrinsics source: {scan_dir / f'frame_{frame_id:05d}.json'}::intrinsics")
    print("camera pose source: same JSON::cameraPoseARFrame")
    print("raw pose: T_world_from_camera_arkit; column vectors; ARKit camera -Z forward")
    print("depth camera: +X right, +Y down, +Z forward")
    print("world: authoritative ARKit/RoomPlan, right-handed, +Y up, metres")
    print("T_world_from_camera_cv = T_world_from_camera_arkit @ diag(1,-1,-1,1)")
    print("K_depth = scale(K_rgb, depth_width/rgb_width, depth_height/rgb_height)")
    print("T_world_from_camera_cv:")
    for row in frame.T_world_from_camera_cv:
        print("  " + " ".join(f"{value: .8f}" for value in row))


def _validate(args: argparse.Namespace) -> None:
    frame = load_scanner_frame(args.scan_dir, args.frame)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    full_cloud = world_point_cloud_from_frame(
        frame, min_confidence=args.min_confidence, max_depth_m=args.max_depth_m
    )
    full_path = write_ascii_ply(output_dir / f"frame_{args.frame:05d}_depth_world.ply", full_cloud)
    print(f"wrote {full_path} ({full_cloud.valid_depth_pixel_count} points)")

    report = {
        "frame_id": args.frame,
        "units": "metres",
        "world_coordinate_system": "ARKit/RoomPlan right-handed, +Y up",
        "T_world_from_camera_cv": frame.T_world_from_camera_cv.tolist(),
        "full_depth_world_ply": str(full_path),
        "full_depth_world_bounds_m": {
            "center": full_cloud.robust_center_world_m.tolist(),
            "dimensions": full_cloud.robust_dimensions_world_m.tolist(),
        },
    }
    if args.mask is not None:
        mask = load_full_image_binary_mask(args.mask, frame)
        object_cloud = world_point_cloud_from_frame(
            frame,
            mask,
            min_confidence=args.min_confidence,
            max_depth_m=args.max_depth_m,
            reject_depth_outliers=True,
        )
        object_path = write_ascii_ply(
            output_dir / f"frame_{args.frame:05d}_object_world.ply", object_cloud
        )
        report["object"] = {
            "mask_path": str(args.mask),
            "world_ply": str(object_path),
            "valid_depth_pixel_count": object_cloud.valid_depth_pixel_count,
            "median_depth_m": object_cloud.median_depth_m,
            "robust_center_world_m": object_cloud.robust_center_world_m.tolist(),
            "robust_dimensions_world_m": object_cloud.robust_dimensions_world_m.tolist(),
        }
        print(f"wrote {object_path} ({object_cloud.valid_depth_pixel_count} points)")
    report_path = output_dir / f"frame_{args.frame:05d}_validation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {report_path}")


def _align_sam3d(args: argparse.Namespace) -> None:
    validation = json.loads(args.validation_json.read_text())
    if "object" not in validation:
        raise ValueError("validation JSON contains no masked object measurement")
    measurement = validation["object"]
    alignment = align_sam3d_glb(
        args.sam3d_glb,
        args.output_glb,
        np.asarray(measurement["robust_center_world_m"], dtype=np.float64),
        scale_mode=args.scale_mode,
        uniform_scale=args.uniform_scale,
    )
    report = alignment_diagnostics_dict(
        alignment,
        rgbd_dimensions_m=np.asarray(measurement["robust_dimensions_world_m"], dtype=np.float64),
    )
    report.update({
        "sam3d_glb": str(args.sam3d_glb),
        "aligned_world_glb": str(args.output_glb),
    })
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output_glb}")
    print(f"wrote {args.output_json}")


def _select_keyframes(args: argparse.Namespace) -> None:
    selected = select_keyframes(
        args.scan_dir,
        translation_threshold_m=args.translation_threshold_m,
        rotation_threshold_deg=args.rotation_threshold_deg,
    )
    complete = discover_complete_frame_ids(args.scan_dir)
    report = {
        "scan_dir": str(args.scan_dir.resolve()),
        "selection_rule": "first frame, then translation OR rotation from last selected frame",
        "translation_threshold_m": args.translation_threshold_m,
        "rotation_threshold_deg": args.rotation_threshold_deg,
        "complete_frame_count": len(complete),
        "selected_frame_count": len(selected),
        "selected_frames": [
            {
                "frame_id": item.frame_id,
                "translation_from_previous_selected_m": item.translation_from_previous_m,
                "rotation_from_previous_selected_deg": item.rotation_from_previous_deg,
            }
            for item in selected
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}: selected {len(selected)}/{len(complete)} complete frames")


def main() -> int:
    defaults = SmallObjectConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("scan_dir", type=Path)
    inspect_parser.add_argument("--frame", type=int, default=0)
    keyframe_parser = subparsers.add_parser("select-keyframes")
    keyframe_parser.add_argument("scan_dir", type=Path)
    keyframe_parser.add_argument("--output", type=Path, required=True)
    keyframe_parser.add_argument(
        "--translation-threshold-m",
        type=float,
        default=defaults.keyframe_translation_threshold_m,
    )
    keyframe_parser.add_argument(
        "--rotation-threshold-deg",
        type=float,
        default=defaults.keyframe_rotation_threshold_deg,
    )
    validate_parser = subparsers.add_parser("validate-depth")
    validate_parser.add_argument("scan_dir", type=Path)
    validate_parser.add_argument("--frame", type=int, required=True)
    validate_parser.add_argument("--mask", type=Path)
    validate_parser.add_argument("--output-dir", type=Path, required=True)
    validate_parser.add_argument("--min-confidence", type=int, choices=(0, 1, 2), default=1)
    validate_parser.add_argument("--max-depth-m", type=float, default=8.0)
    align_parser = subparsers.add_parser("align-sam3d")
    align_parser.add_argument("--sam3d-glb", type=Path, required=True)
    align_parser.add_argument("--validation-json", type=Path, required=True)
    align_parser.add_argument("--output-glb", type=Path, required=True)
    align_parser.add_argument("--output-json", type=Path, required=True)
    align_parser.add_argument("--scale-mode", choices=("none", "uniform"), default="none")
    align_parser.add_argument("--uniform-scale", type=float, default=1.0)
    args = parser.parse_args()
    if args.command == "inspect":
        _print_inspection(args.scan_dir, args.frame)
    elif args.command == "select-keyframes":
        _select_keyframes(args)
    elif args.command == "validate-depth":
        _validate(args)
    else:
        _align_sam3d(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
