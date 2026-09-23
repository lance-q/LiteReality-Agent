"""Merge a secondary manifest into a primary one using world-space duplicate rejection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("primary", type=Path)
    parser.add_argument("secondary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--secondary-prefix", default="broad_")
    parser.add_argument("--minimum-secondary-views", type=int, default=2)
    parser.add_argument("--duplicate-distance-m", type=float, default=0.15)
    args = parser.parse_args()
    primary = json.loads(args.primary.read_text())
    secondary = json.loads(args.secondary.read_text())
    merged = list(primary["objects"])
    centers = [np.asarray(item["rgbd_world_center_m"], dtype=float) for item in merged]
    rejected = []
    for item in secondary["objects"]:
        if len(item["observations"]) < args.minimum_secondary_views:
            continue
        center = np.asarray(item["rgbd_world_center_m"], dtype=float)
        if centers and min(float(np.linalg.norm(center - other)) for other in centers) < args.duplicate_distance_m:
            rejected.append(item["object_id"])
            continue
        item = dict(item)
        item["object_id"] = args.secondary_prefix + item["object_id"]
        merged.append(item)
        centers.append(center)
    output = {
        **primary,
        "objects": merged,
        "merge_diagnostics": {
            "primary_count": len(primary["objects"]),
            "secondary_multiview_candidates": sum(
                len(item["observations"]) >= args.minimum_secondary_views
                for item in secondary["objects"]
            ),
            "secondary_duplicates_rejected": rejected,
            "final_count": len(merged),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["merge_diagnostics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
