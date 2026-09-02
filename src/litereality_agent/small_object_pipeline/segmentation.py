"""Box-prompted SAM instance segmentation with full-image binary mask output."""

from __future__ import annotations

import os

import numpy as np

_MODEL = None


def default_model_id() -> str:
    return os.environ.get("LR_SEGMENT_MODEL", "facebook/sam-vit-base")


def _load(model_id: str | None = None):
    global _MODEL
    model_id = model_id or default_model_id()
    if _MODEL is not None and _MODEL[3] == model_id:
        return _MODEL
    import torch
    from transformers import SamModel, SamProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = SamProcessor.from_pretrained(model_id)
    model = SamModel.from_pretrained(model_id).to(device).eval()
    _MODEL = processor, model, device, model_id
    return _MODEL


def segment_boxes(image, boxes_xyxy: list[list[float]], model_id: str | None = None) -> list[np.ndarray]:
    """Return one independent boolean mask per box, each exactly matching the input image."""
    if not boxes_xyxy:
        return []
    import torch

    processor, model, device, _ = _load(model_id)
    inputs = processor(images=image, input_boxes=[boxes_xyxy], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=True)
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )[0]
    scores = outputs.iou_scores[0].cpu()
    results = []
    for index in range(len(boxes_xyxy)):
        best = int(torch.argmax(scores[index]).item())
        mask = masks[index, best].numpy().astype(bool)
        if mask.shape != image.size[::-1]:
            raise RuntimeError(f"SAM returned {mask.shape}, expected {image.size[::-1]}")
        results.append(mask)
    return results
