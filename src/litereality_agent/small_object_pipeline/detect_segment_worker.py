"""GPU worker for GroundingDINO detection followed by one SAM mask per instance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from litereality_agent.models.grounding_dino.inference import detect

from .detection import (
    ImageTile,
    InstanceDetection,
    deduplicate_same_frame,
    generate_tiles,
)
from .segmentation import segment_boxes


def _detect_frame(args, frame_id: int) -> dict:
    image_path = args.scan_dir / f"frame_{frame_id:05d}.jpg"
    image = Image.open(image_path).convert("RGB")
    candidates: list[tuple[list[float], float, ImageTile | None]] = []
    sources: list[tuple[Image.Image, ImageTile | None]] = [(image, None)]
    if args.tile_width and args.tile_height:
        for tile in generate_tiles(
            image.width, image.height, args.tile_width, args.tile_height, args.tile_overlap
        ):
            sources.append(
                (
                    image.crop(
                        (
                            tile.crop_x_px,
                            tile.crop_y_px,
                            tile.crop_x_px + tile.crop_width_px,
                            tile.crop_y_px + tile.crop_height_px,
                        )
                    ),
                    tile,
                )
            )
    for source_image, tile in sources:
        for result in detect(
            source_image,
            args.prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            upright=True,
        ):
            box = list(result.box)
            if tile is not None:
                box = [
                    box[0] + tile.crop_x_px,
                    box[1] + tile.crop_y_px,
                    box[2] + tile.crop_x_px,
                    box[3] + tile.crop_y_px,
                ]
            box[0], box[2] = max(0.0, box[0]), min(float(image.width), box[2])
            box[1], box[3] = max(0.0, box[1]), min(float(image.height), box[3])
            if box[2] > box[0] and box[3] > box[1]:
                candidates.append((box, float(result.score), tile))

    # Cheap pre-NMS avoids sending near-identical boxes through SAM. Final dedup also uses mask IoU.
    pre_nms: list[tuple[list[float], float, ImageTile | None]] = []
    from .detection import bbox_iou

    for candidate in sorted(candidates, key=lambda item: item[1], reverse=True):
        if not any(bbox_iou(candidate[0], accepted[0]) >= args.bbox_iou for accepted in pre_nms):
            pre_nms.append(candidate)
    masks = segment_boxes(image, [item[0] for item in pre_nms], args.segment_model)
    detections = [
        InstanceDetection(
            frame_id=frame_id,
            detection_id=index,
            label=args.label,
            score=candidate[1],
            bbox_full_image=candidate[0],
            mask_full_image=mask,
            source_tile=candidate[2],
        )
        for index, (candidate, mask) in enumerate(zip(pre_nms, masks))
    ]
    detections = deduplicate_same_frame(
        detections,
        bbox_iou_threshold=args.bbox_iou,
        mask_iou_threshold=args.mask_iou,
    )
    frame_dir = args.output_dir / f"frame_{frame_id:05d}"
    mask_dir = frame_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    manifest_detections = []
    for detection in detections:
        mask_path = mask_dir / f"detection_{detection.detection_id:03d}.png"
        Image.fromarray(detection.mask_full_image.astype(np.uint8) * 255).save(mask_path)
        tile = detection.source_tile
        manifest_detections.append(
            {
                "frame_id": frame_id,
                "detection_id": detection.detection_id,
                "label": detection.label,
                "score": detection.score,
                "bbox_full_image": detection.bbox_full_image,
                "mask_path": str(mask_path.resolve()),
                "mask_pixel_area": int(detection.mask_full_image.sum()),
                "source_crop": (
                    {
                        "crop_x": tile.crop_x_px,
                        "crop_y": tile.crop_y_px,
                        "crop_width": tile.crop_width_px,
                        "crop_height": tile.crop_height_px,
                        "network_width": tile.crop_width_px,
                        "network_height": tile.crop_height_px,
                    }
                    if tile
                    else None
                ),
            }
        )
    manifest = {
        "frame_id": frame_id,
        "rgb_path": str(image_path.resolve()),
        "rgb_size": [image.width, image.height],
        "prompt": args.prompt,
        "raw_detection_count": len(candidates),
        "detections": manifest_detections,
    }
    (frame_dir / "detections.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"frame {frame_id}: {len(candidates)} raw -> {len(detections)} instances")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scan_dir", type=Path)
    parser.add_argument("--frames", nargs="+", type=int, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--box-threshold", type=float, default=0.15)
    parser.add_argument("--text-threshold", type=float, default=0.15)
    parser.add_argument("--bbox-iou", type=float, default=0.55)
    parser.add_argument("--mask-iou", type=float, default=0.60)
    parser.add_argument("--tile-width", type=int)
    parser.add_argument("--tile-height", type=int)
    parser.add_argument("--tile-overlap", type=float, default=0.25)
    parser.add_argument("--segment-model", default="facebook/sam-vit-base")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests = [_detect_frame(args, frame_id) for frame_id in args.frames]
    (args.output_dir / "detections_manifest.json").write_text(
        json.dumps({"frames": manifests}, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
