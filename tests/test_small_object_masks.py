from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from litereality_agent.small_object_pipeline.depth_backprojection import (
    load_full_image_binary_mask,
)
from litereality_agent.small_object_pipeline.sam3d_worker import prepare_sam3d_input


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


def test_scanner_upright_rotates_image_and_mask_together():
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    mask = np.zeros((2, 3), dtype=bool)
    image[0, 0] = [10, 20, 30]
    mask[0, 0] = True
    rotated_image, rotated_mask = prepare_sam3d_input(image, mask, scanner_upright=True)
    assert rotated_image.shape == (3, 2, 3)
    assert rotated_mask.shape == (3, 2)
    assert rotated_mask[0, 1]
    np.testing.assert_array_equal(rotated_image[0, 1], [10, 20, 30])
