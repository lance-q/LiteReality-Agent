from __future__ import annotations

import json
import sys

import numpy as np
import pytest
import trimesh

from litereality_agent.small_object_pipeline.finalize_objects import _limit_center_correction
from litereality_agent.small_object_pipeline.object_alignment import (
    align_mesh_to_world_anchor,
    alignment_diagnostics_dict,
    estimate_uniform_scale_from_rgbd_dimensions,
    validate_shape_preserving_linear_transform,
)
from litereality_agent.small_object_pipeline.sam3d_worker import main as sam3d_worker_main


def test_center_refinement_correction_is_conservatively_limited() -> None:
    applied, proposed_distance, limited = _limit_center_correction(
        np.zeros(3), np.array([0.2, 0.0, 0.0]), 0.05
    )
    np.testing.assert_allclose(applied, [0.05, 0.0, 0.0])
    assert proposed_distance == 0.2
    assert limited


def _pairwise_distances(vertices: np.ndarray) -> np.ndarray:
    offsets = vertices[:, None, :] - vertices[None, :, :]
    return np.linalg.norm(offsets, axis=-1)


def test_rigid_alignment_preserves_pairwise_geometry_and_aspect_ratios():
    mesh = trimesh.creation.box(extents=[1, 2, 4])
    source_vertices = np.asarray(mesh.vertices).copy()
    target_center = np.array([1.3, 0.81, -2.1])
    aligned, result = align_mesh_to_world_anchor(mesh, target_center)

    np.testing.assert_allclose(
        _pairwise_distances(np.asarray(aligned.vertices)),
        _pairwise_distances(source_vertices),
        atol=1e-12,
    )
    np.testing.assert_allclose(aligned.bounds.mean(axis=0), target_center)
    np.testing.assert_allclose(result.source_aspect_ratios, result.aligned_aspect_ratios)
    np.testing.assert_allclose(result.transform_singular_values, [1, 1, 1])


def test_uniform_scale_changes_every_distance_by_one_scalar_only():
    mesh = trimesh.creation.icosphere(subdivisions=1)
    source_distances = _pairwise_distances(np.asarray(mesh.vertices))
    aligned, result = align_mesh_to_world_anchor(
        mesh, np.array([2.0, 3.0, 4.0]), scale_mode="uniform", uniform_scale=0.25
    )
    np.testing.assert_allclose(
        _pairwise_distances(np.asarray(aligned.vertices)), source_distances * 0.25, atol=1e-12
    )
    np.testing.assert_allclose(result.transform_singular_values, [0.25, 0.25, 0.25])
    np.testing.assert_allclose(result.source_aspect_ratios, result.aligned_aspect_ratios)


def test_rgbd_uniform_scale_is_median_of_axis_ratios():
    scale, ratios = estimate_uniform_scale_from_rgbd_dimensions(
        np.array([1.0, 2.0, 4.0]), np.array([0.1, 0.4, 1.2])
    )
    np.testing.assert_allclose(ratios, [0.1, 0.2, 0.3])
    assert scale == pytest.approx(0.2)


def test_non_uniform_scale_is_rejected():
    with pytest.raises(ValueError, match="non-uniform object deformation"):
        validate_shape_preserving_linear_transform(
            np.diag([0.5773, 0.9229, 0.1902]), scale_mode="uniform"
        )


def test_rgbd_dimensions_are_diagnostic_only_and_do_not_modify_vertices():
    mesh = trimesh.creation.box(extents=[1, 2, 4])
    aligned, result = align_mesh_to_world_anchor(mesh, np.array([1.0, 2.0, 3.0]))
    vertices_before_report = np.asarray(aligned.vertices).copy()
    report = alignment_diagnostics_dict(result, rgbd_dimensions_m=np.array([9.0, 0.1, 7.0]))
    np.testing.assert_array_equal(aligned.vertices, vertices_before_report)
    assert report["rgbd_dimensions_m_diagnostic"] == [9.0, 0.1, 7.0]
    assert report["geometry_source"] == "sam3d"


def test_sam3d_worker_reuses_successful_cached_asset(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "object_000"
    output_dir.mkdir()
    (output_dir / "object.glb").write_bytes(b"cached")
    (output_dir / "success.json").write_text(json.dumps({"ok": True}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sam3d_worker.py",
            "--sam3d-repo",
            str(tmp_path / "unused-repo"),
            "--image",
            str(tmp_path / "unused.jpg"),
            "--mask",
            str(tmp_path / "unused.png"),
            "--output-dir",
            str(output_dir),
        ],
    )
    assert sam3d_worker_main() == 0
    assert "cache hit" in capsys.readouterr().out
