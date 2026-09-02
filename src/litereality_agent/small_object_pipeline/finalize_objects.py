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
                uniform_scale, axis_scale_ratios = estimate_uniform_scale_from_rgbd_dimensions(
                    source_dimensions, rgbd_dimensions_m
                )
                scale_estimation_method = "median of corresponding RGB-D/SAM3D extent ratios"
            else:
                uniform_scale = args.uniform_scale
                scale_estimation_method = "explicit command-line scalar"
        aligned_glb = args.aligned_output_dir / f"{object_id}_world.glb"
        alignment = align_sam3d_glb(
            sam3d_glb,
            aligned_glb,
            np.asarray(item["rgbd_world_center_m"], dtype=float),
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
