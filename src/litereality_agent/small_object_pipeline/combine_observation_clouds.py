"""Combine two world-space observation PLYs with fixed comparison colors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def _vertices(path: Path) -> np.ndarray:
    loaded = trimesh.load(path, process=False)
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError(f"invalid point cloud vertices in {path}")
    return vertices


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("green_cloud", type=Path)
    parser.add_argument("red_cloud", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    green_vertices = _vertices(args.green_cloud)
    red_vertices = _vertices(args.red_cloud)
    vertices = np.vstack([green_vertices, red_vertices])
    colors = np.vstack(
        [
            np.tile([0, 255, 0, 255], (len(green_vertices), 1)),
            np.tile([255, 0, 0, 255], (len(red_vertices), 1)),
        ]
    ).astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trimesh.points.PointCloud(vertices, colors=colors).export(args.output)
    manifest = {
        "coordinate_system": "LiteReality/ARKit world, metres, +Y up",
        "green": {"path": str(args.green_cloud.resolve()), "points": len(green_vertices)},
        "red": {"path": str(args.red_cloud.resolve()), "points": len(red_vertices)},
        "combined": {"path": str(args.output.resolve()), "points": len(vertices)},
    }
    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.output}: {len(green_vertices)} green + {len(red_vertices)} red points")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
