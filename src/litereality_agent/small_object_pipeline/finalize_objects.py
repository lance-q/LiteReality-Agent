"""Metric-align one cached SAM3D asset per associated physical object."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from .object_alignment import (
    _as_mesh,
    align_sam3d_glb,
    alignment_diagnostics_dict,
    estimate_uniform_scale_from_rgbd_dimensions,
)
from .object_association import robust_observation_center


def _limit_center_correction(
    original_center_m: np.ndarray, refined_center_m: np.ndarray, maximum_correction_m: float
) -> tuple[np.ndarray, float, bool]:
    delta = refined_center_m - original_center_m
    distance = float(np.linalg.norm(delta))
    if distance <= maximum_correction_m or distance <= 1e-12:
        return refined_center_m, distance, False
    return original_center_m + delta * (maximum_correction_m / distance), distance, True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("objects_manifest", type=Path)
    parser.add_argument("--sam3d-output-dir", type=Path, required=True)
    parser.add_argument("--aligned-output-dir", type=Path, required=True)
    parser.add_argument("--scale-mode", choices=("none", "uniform"), default="none")
    parser.add_argument(
        "--uniform-scale",
        type=float,
        help="explicit scalar; omitted in uniform mode means estimate from RGB-D",
    )
    parser.add_argument(
        "--refine-world-center",
        action="store_true",
        help="re-estimate placement from reliable, spatially consistent observations",
    )
    parser.add_argument("--max-center-correction-m", type=float, default=0.05)
    args = parser.parse_args()
    manifest = json.loads(args.objects_manifest.read_text())
    args.aligned_output_dir.mkdir(parents=True, exist_ok=True)
    for item in manifest["objects"]:
        object_id = item["object_id"]
        sam3d_glb = args.sam3d_output_dir / object_id / "object.glb"
        success = args.sam3d_output_dir / object_id / "success.json"
        if not sam3d_glb.is_file() or not success.is_file():
            raise FileNotFoundError(f"no successful cached SAM3D output for {object_id}")
        rgbd_dimensions_m = np.asarray(item["rgbd_dimensions_m"], dtype=float)
        uniform_scale = 1.0
        scale_estimation_method = "disabled"
        axis_scale_ratios = None
        if args.scale_mode == "uniform":
            if args.uniform_scale is None:
                source_mesh = _as_mesh(trimesh.load(sam3d_glb, force="scene", process=False))
                source_dimensions = np.ptp(np.asarray(source_mesh.vertices, dtype=float), axis=0)
                try:
                    uniform_scale, axis_scale_ratios = estimate_uniform_scale_from_rgbd_dimensions(
                        source_dimensions, rgbd_dimensions_m
                    )
                    scale_estimation_method = "median of corresponding RGB-D/SAM3D extent ratios"
                except ValueError as error:
                    if "no observable positive extent" not in str(error):
                        raise
                    peer_scales = []
                    for peer in manifest["objects"]:
                        if peer is item or peer["label"] != item["label"]:
                            continue
                        peer_dimensions = np.asarray(peer["rgbd_dimensions_m"], dtype=float)
                        if not np.any(peer_dimensions > 1e-8):
                            continue
                        peer_asset = args.sam3d_output_dir / peer["object_id"] / "object.glb"
                        peer_mesh = _as_mesh(trimesh.load(peer_asset, force="scene", process=False))
                        peer_source_dimensions = np.ptp(
                            np.asarray(peer_mesh.vertices, dtype=float), axis=0
                        )
                        peer_scale, _ = estimate_uniform_scale_from_rgbd_dimensions(
                            peer_source_dimensions, peer_dimensions
                        )
                        peer_scales.append(peer_scale)
                    if not peer_scales:
                        raise ValueError(
                            f"{object_id} has no RGB-D extent and no same-label scale prior"
                        ) from error
                    uniform_scale = float(np.median(peer_scales))
                    axis_scale_ratios = np.full(3, np.nan)
                    scale_estimation_method = "median same-label scale prior; RGB-D extent unobservable"
            else:
                uniform_scale = args.uniform_scale
                scale_estimation_method = "explicit command-line scalar"
        target_center_world_m = np.asarray(item["rgbd_world_center_m"], dtype=float)
        if args.refine_world_center:
            proposed_center, refinement = robust_observation_center(item["observations"])
            target_center_world_m, proposed_distance_m, was_limited = _limit_center_correction(
                target_center_world_m, proposed_center, args.max_center_correction_m
            )
            refinement.update(
                {
                    "original_center_world_m": np.asarray(
                        item["rgbd_world_center_m"], dtype=float
                    ).tolist(),
                    "proposed_center_world_m": proposed_center.tolist(),
                    "applied_center_world_m": target_center_world_m.tolist(),
                    "proposed_correction_m": proposed_distance_m,
                    "maximum_correction_m": args.max_center_correction_m,
                    "correction_was_limited": was_limited,
                }
            )
            item["rgbd_world_center_m"] = target_center_world_m.tolist()
            item["world_center_refinement"] = refinement
        aligned_glb = args.aligned_output_dir / f"{object_id}_world.glb"
        alignment = align_sam3d_glb(
            sam3d_glb,
            aligned_glb,
            target_center_world_m,
            scale_mode=args.scale_mode,
            uniform_scale=uniform_scale,
        )
        alignment_report = alignment_diagnostics_dict(
            alignment,
            rgbd_dimensions_m=rgbd_dimensions_m,
        )
        alignment_report.update(
            {
                "sam3d_glb": str(sam3d_glb.resolve()),
                "aligned_world_glb": str(aligned_glb.resolve()),
                "sam3d_cache_reused": True,
                "uniform_scale_estimation_method": scale_estimation_method,
                "axis_scale_ratios_diagnostic": (
                    axis_scale_ratios.tolist() if axis_scale_ratios is not None else None
                ),
            }
        )
        alignment_json = args.aligned_output_dir / f"{object_id}_alignment.json"
        alignment_json.write_text(json.dumps(alignment_report, indent=2) + "\n")
        item["sam3d_asset"] = str(sam3d_glb.resolve())
        item["aligned_asset"] = str(aligned_glb.resolve())
        item["alignment_manifest"] = str(alignment_json.resolve())
        item["T_world_from_object"] = alignment.T_world_from_object.tolist()
        item["scale_mode"] = alignment.scale_mode
        item["uniform_scale"] = alignment.uniform_scale
    args.objects_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"updated {args.objects_manifest} with {len(manifest['objects'])} aligned assets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
