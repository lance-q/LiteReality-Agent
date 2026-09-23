"""Split an object manifest into deterministic LiteReality-world spatial sections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def section_for(center: list[float]) -> str:
    x_m, _, z_m = center
    if x_m >= 2.8:
        return "living_sofa"
    if z_m <= -1.0:
        return "kitchen"
    if z_m >= 1.0:
        return "dining_entry"
    return "central_storage"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.manifest.read_text())
    grouped: dict[str, list[dict]] = {}
    for item in source["objects"]:
        grouped.setdefault(section_for(item["rgbd_world_center_m"]), []).append(item)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = {"coordinate_system": "LiteReality/ARKit world", "sections": {}}
    for name, objects in sorted(grouped.items()):
        output = args.output_dir / f"{name}_objects_manifest.json"
        output.write_text(json.dumps({**source, "objects": objects}, indent=2) + "\n")
        index["sections"][name] = {
            "object_count": len(objects),
            "manifest": str(output.resolve()),
        }
    (args.output_dir / "sections_index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps(index, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
