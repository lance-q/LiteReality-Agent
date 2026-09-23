"""Render best observations from an object manifest for pre-SAM3D review."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("objects_manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--min-views", type=int, default=1)
    parser.add_argument("--min-selection-score", type=float, default=0.0)
    parser.add_argument("--tile-width", type=int, default=360)
    args = parser.parse_args()
    manifest = json.loads(args.objects_manifest.read_text())
    objects = [
        item
        for item in manifest["objects"]
        if len(item["observations"]) >= args.min_views
        and float(item["best_observation"]["selection"]["score"]) >= args.min_selection_score
    ]
    columns = 4
    tile_height = int(args.tile_width * 0.75) + 42
    sheet = Image.new(
        "RGB",
        (columns * args.tile_width, max(1, math.ceil(len(objects) / columns)) * tile_height),
        "#202020",
    )
    font = ImageFont.load_default()
    for index, item in enumerate(objects):
        best = item["best_observation"]
        image = Image.open(best["rgb_path"]).convert("RGB")
        mask = np.asarray(Image.open(best["mask_path"]).convert("L")) > 0
        pixels = np.asarray(image).copy()
        pixels[mask] = (0.55 * pixels[mask] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
        ys, xs = np.nonzero(mask)
        if len(xs):
            pad = max(40, int(0.35 * max(np.ptp(xs), np.ptp(ys))))
            x1, x2 = max(0, int(xs.min()) - pad), min(image.width, int(xs.max()) + pad + 1)
            y1, y2 = max(0, int(ys.min()) - pad), min(image.height, int(ys.max()) + pad + 1)
            pixels = pixels[y1:y2, x1:x2]
        annotated = Image.fromarray(pixels).transpose(Image.Transpose.ROTATE_270)
        annotated.thumbnail((args.tile_width, tile_height - 42))
        x = (index % columns) * args.tile_width
        y = (index // columns) * tile_height
        sheet.paste(annotated, (x, y))
        caption = (
            f"{item['object_id']} {item['label']} | frame {best['frame_id']} | "
            f"views {len(item['observations'])} | score {best['selection']['score']:.2f}"
        )
        ImageDraw.Draw(sheet).text((x + 4, y + tile_height - 36), caption, fill="white", font=font)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=92)
    print(f"wrote {args.output}: {len(objects)} candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
