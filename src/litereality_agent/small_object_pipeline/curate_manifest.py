"""Create a reconstruction manifest from reviewed physical-object IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument("--object-ids", nargs="+", required=True)
    args = parser.parse_args()
    source = json.loads(args.source_manifest.read_text())
    requested = set(args.object_ids)
    selected = [item for item in source["objects"] if item["object_id"] in requested]
    found = {item["object_id"] for item in selected}
    if found != requested:
        raise ValueError(f"unknown object IDs: {sorted(requested - found)}")
    curated = {
        **{key: value for key, value in source.items() if key != "objects"},
        "curation": {
            "method": "visual review of full-image masks before expensive SAM3D inference",
            "source_manifest": str(args.source_manifest.resolve()),
            "selected_object_ids": args.object_ids,
        },
        "objects": selected,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(curated, indent=2) + "\n")
    print(f"wrote {args.output_manifest}: {len(selected)} selected physical objects")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
