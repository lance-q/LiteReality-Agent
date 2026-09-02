"""Run SAM3D once per associated object, using successful outputs as cache entries."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("objects_manifest", type=Path)
    parser.add_argument("--python", type=Path, required=True, help="SAM3D environment Python")
    parser.add_argument("--sam3d-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.objects_manifest.read_text())
    worker = Path(__file__).with_name("sam3d_worker.py")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for item in manifest["objects"]:
        best = item["best_observation"]
        command = [
            str(args.python),
            str(worker),
            "--sam3d-repo",
            str(args.sam3d_repo),
            "--image",
            best["rgb_path"],
            "--mask",
            best["mask_path"],
            "--output-dir",
            str(args.output_dir / item["object_id"]),
        ]
        if args.force:
            command.append("--force")
        print(
            f"{item['object_id']}: frame {best['frame_id']} detection "
            f"{best['detection_id']}"
        )
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
