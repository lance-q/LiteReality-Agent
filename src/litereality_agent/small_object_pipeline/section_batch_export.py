"""Export one optimized objects GLB and room-context GLB per spatial section."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sections_index", type=Path)
    parser.add_argument("--aligned-dir", type=Path, required=True)
    parser.add_argument("--room-usdz", type=Path, required=True)
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decimate-ratio", type=float, default=0.2)
    args = parser.parse_args()
    index = json.loads(args.sections_index.read_text())
    worker = Path(__file__).with_name("batch_room_merge.py")
    for section, metadata in index["sections"].items():
        manifest = json.loads(Path(metadata["manifest"]).read_text())
        section_dir = args.output_dir / section
        assets = [
            f"{item['object_id']}_{item['label']}="
            f"{args.aligned_dir / (item['object_id'] + '_world.glb')}"
            for item in manifest["objects"]
        ]
        command = [
            str(args.blender), "--background", "--python", str(worker), "--",
            str(args.room_usdz), str(section_dir / f"{section}_objects.glb"),
            str(section_dir / f"room_plus_{section}.blend"),
            str(section_dir / f"room_plus_{section}.glb"), *assets,
            "--decimate-ratio", str(args.decimate_ratio), "--draco",
        ]
        print(f"exporting {section}: {len(assets)} objects", flush=True)
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
