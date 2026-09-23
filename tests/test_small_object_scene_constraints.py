"""Pure CPU tests for scene-constraint arithmetic (not executed during implementation)."""

from __future__ import annotations

import numpy as np
import trimesh

from litereality_agent.small_object_pipeline.scene_constraints.orientation import (
    generate_orientation_candidates,
    uniform_scale_and_rotation,
)
from litereality_agent.small_object_pipeline.scene_constraints.profiles import profile_for_label
from litereality_agent.small_object_pipeline.scene_constraints.solver import (
    SceneConstraintSolver,
    SolverConfig,
)
from litereality_agent.small_object_pipeline.scene_constraints.types import SupportSurface


def test_signed_axes_can_all_map_to_world_up():
    candidates = generate_orientation_candidates(
        np.eye(3), enforce_upright=True, nearest_wall=None, include_cardinal_yaws=False
    )
    by_axis = {axis: rotation for _, axis, _, rotation in candidates if axis != "current"}
    source = {
        "+X": [1, 0, 0],
        "-X": [-1, 0, 0],
        "+Y": [0, 1, 0],
        "-Y": [0, -1, 0],
        "+Z": [0, 0, 1],
        "-Z": [0, 0, -1],
    }
    assert set(by_axis) == set(source)
    for name, vector in source.items():
        np.testing.assert_allclose(by_axis[name] @ vector, [0, 1, 0], atol=1e-7)


def test_floor_snap_preserves_uniform_scale_and_geometry():
    solver = object.__new__(SceneConstraintSolver)
    solver.config = SolverConfig()
    mesh = trimesh.creation.box(extents=[0.4, 0.8, 0.3])
    vertices = np.asarray(mesh.vertices)
    anchor = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    floor = SupportSurface(
        "Floor0",
        "floor",
        0.0,
        np.array([[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]]),
        "test",
    )
    candidate = solver._make_candidate(
        candidate_id="floor",
        source_vertices=vertices,
        source_anchor=anchor,
        target_center_world_m=np.array([1.0, 0.7, 2.0]),
        uniform_scale=0.5,
        rotation=np.eye(3),
        upright_axis="+Y",
        yaw_deg=0.0,
        support=floor,
        maximum_support_snap_m=1.0,
    )
    assert candidate is not None
    singular_values = np.linalg.svd(candidate.T_world_from_object[:3, :3], compute_uv=False)
    np.testing.assert_allclose(singular_values, [0.5, 0.5, 0.5])
    assert abs(np.quantile(candidate.transformed_vertices_world_m[:, 1], 0.02)) < 1e-8


def test_profile_defaults_are_conservative():
    assert profile_for_label("trash can").name == "floor_standing"
    assert profile_for_label("microwave").name == "wall_relative_supported"
    assert profile_for_label("pillow").name == "free_form"
    assert profile_for_label("unrecognized artifact").name == "free_form"


def test_uniform_scale_extraction_rejects_nonuniform_transform():
    transform = np.eye(4)
    transform[:3, :3] = np.diag([1.0, 2.0, 1.0])
    try:
        uniform_scale_and_rotation(transform)
    except ValueError as error:
        assert "uniform scale" in str(error)
    else:
        raise AssertionError("non-uniform transform was accepted")
