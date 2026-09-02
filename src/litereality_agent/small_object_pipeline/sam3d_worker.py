"""Isolated SAM3D image+mask worker, executed with the existing SAM3D environment.

This file imports no LiteReality dependencies so it can run inside the untouched upstream
environment.  A non-empty ``object.glb`` plus ``success.json`` is the cache key.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _tensor_json(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().tolist()
    return np.asarray(value).tolist()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam3d-repo", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output_glb = args.output_dir / "object.glb"
    success_path = args.output_dir / "success.json"
    if not args.force and output_glb.is_file() and output_glb.stat().st_size and success_path.is_file():
        print(f"cache hit: {output_glb}")
        return 0
    with Image.open(args.image) as image_file, Image.open(args.mask) as mask_file:
        image = np.asarray(image_file.convert("RGB"), dtype=np.uint8)
        mask = np.asarray(mask_file.convert("L")) > 0
    if image.shape[:2] != mask.shape:
        raise ValueError(f"SAM3D mask {mask.shape} must exactly match image {image.shape[:2]}")
    if not mask.any():
        raise ValueError("SAM3D mask is empty")
    sys.path.insert(0, str(args.sam3d_repo / "notebook"))
    sys.path.insert(0, str(args.sam3d_repo))
    from inference import Inference  # noqa: PLC0415

    inference = Inference(str(args.sam3d_repo / "checkpoints/hf/pipeline.yaml"), compile=False)
    output = inference(image, mask, seed=args.seed)
    glb = output.get("glb")
    if glb is None:
        raise RuntimeError("SAM3D returned no GLB")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    glb.export(output_glb)
    metadata = {
        "image": str(args.image.resolve()),
        "mask": str(args.mask.resolve()),
        "seed": args.seed,
        "sam3d_coordinate_system": "canonical object-local GLB, Y-up, arbitrary scale",
        "predicted_local_to_camera": {
            "quaternion_wxyz": _tensor_json(output["rotation"]),
            "translation": _tensor_json(output["translation"]),
            "isotropic_scale": _tensor_json(output["scale"]),
            "authoritative_for_metric_world": False,
        },
        "glb": str(output_glb.resolve()),
    }
    success_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {output_glb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
