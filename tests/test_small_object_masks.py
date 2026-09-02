from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from litereality_agent.small_object_pipeline.depth_backprojection import (
    load_full_image_binary_mask,
)


def test_sam3d_mask_must_exactly_match_input_image(tmp_path):
    path = tmp_path / "mask.png"
    Image.fromarray(np.zeros((8, 10), dtype=np.uint8)).save(path)
    frame = SimpleNamespace(rgb_width_px=10, rgb_height_px=8)
    assert load_full_image_binary_mask(path, frame).shape == (8, 10)


def test_sam3d_mask_dimension_mismatch_fails_loudly(tmp_path):
    path = tmp_path / "mask.png"
    Image.fromarray(np.zeros((7, 10), dtype=np.uint8)).save(path)
    frame = SimpleNamespace(rgb_width_px=10, rgb_height_px=8)
    with pytest.raises(ValueError, match="exactly match RGB"):
        load_full_image_binary_mask(path, frame)
