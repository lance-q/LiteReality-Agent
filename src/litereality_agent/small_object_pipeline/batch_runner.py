"""One-command reconstruction, alignment, and optimized room export after manifest review."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    workspace = Path(config["workspace"])
    manifest = Path(config["objects_manifest"])
    sam3d_output_dir = Path(config.get("sam3d_output_dir", workspace / "sam3d_outputs"))
    aligned_output_dir = Path(config.get("aligned_output_dir", workspace / "aligned_objects"))
    package = "litereality_agent.small_object_pipeline"
    source_env = {**os.environ, "PYTHONPATH": config.get("pythonpath", "src")}

    _run(
        [
            sys.executable,
            "-m",
            f"{package}.sam3d_batch",
            str(manifest),
            "--python",
            config["sam3d_python"],
            "--sam3d-repo",
            config["sam3d_repo"],
            "--output-dir",
            str(sam3d_output_dir),
            "--scanner-upright",
        ],
        env=source_env,
    )
    _run(
        [
            sys.executable,
            "-m",
            f"{package}.finalize_objects",
            str(manifest),
            "--sam3d-output-dir",
            str(sam3d_output_dir),
            "--aligned-output-dir",
            str(aligned_output_dir),
            "--scale-mode",
            "uniform",
        ],
        env=source_env,
    )

    for object_id, validated_asset in config.get("validated_alignment_overrides", {}).items():
        destination = aligned_output_dir / f"{object_id}_world.glb"
        shutil.copy2(validated_asset, destination)
        print(f"restored validated alignment: {object_id} <- {validated_asset}")

    if config.get("scene_constraints_enabled", True):
        constraint_command = [
            sys.executable,
            "-m",
            f"{package}.scene_constraints.cli",
            str(manifest),
            "--room-usdz",
            config["room_usdz"],
            "--sam3d-output-dir",
            str(sam3d_output_dir),
            "--aligned-output-dir",
            str(aligned_output_dir),
        ]
        for object_id in config.get("validated_alignment_overrides", {}):
            constraint_command.extend(["--validated-object-id", object_id])
        if config.get("scene_constraint_profile_overrides"):
            constraint_command.extend(
                ["--profile-overrides", config["scene_constraint_profile_overrides"]]
            )
        if config.get("scene_constraint_solver_config"):
            constraint_command.extend(
                ["--solver-config", config["scene_constraint_solver_config"]]
            )
        if config.get("scene_constraint_diagnostics_only", False):
            constraint_command.append("--diagnostics-only")
        _run(constraint_command, env=source_env)

    manifest_data = json.loads(manifest.read_text())
    labels: dict[str, int] = {}
    asset_arguments = []
    for item in manifest_data["objects"]:
        label = item["label"]
        labels[label] = labels.get(label, 0) + 1
        scene_label = f"{label}_{labels[label]:02d}"
        asset = aligned_output_dir / f"{item['object_id']}_world.glb"
        asset_arguments.append(f"{scene_label}={asset}")

    merge_worker = Path(__file__).with_name("batch_room_merge.py")
    output_dir = workspace / "merged"
    common = [
        config["blender"],
        "--background",
        "--python",
        str(merge_worker),
        "--",
        config["room_usdz"],
        str(workspace / "aligned_objects_optimized.glb"),
    ]
    optimization = [
        "--decimate-ratio",
        str(config.get("delivery_decimate_ratio", 0.35)),
        "--draco",
    ]
    _run(
        common
        + [
            str(output_dir / "room_plus_small_objects_optimized.blend"),
            str(output_dir / "room_plus_small_objects_optimized.glb"),
        ]
        + asset_arguments
        + optimization
    )
    exclusions = config.get("visible_debug_exclusions", [])
    if exclusions:
        visible_worker = Path(__file__).with_name("visible_scene_export.py")
        _run(
            [
                config["blender"],
                "--background",
                "--python",
                str(visible_worker),
                "--",
                str(output_dir / "room_plus_small_objects_optimized.blend"),
                str(output_dir / "room_plus_small_objects_optimized_visible.blend"),
                str(output_dir / "room_plus_small_objects_optimized_visible.glb"),
            ]
            + exclusions
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
