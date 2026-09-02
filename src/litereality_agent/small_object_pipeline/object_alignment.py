"""Shape-preserving placement of canonical SAM3D geometry in LiteReality world coordinates.

SAM3D is authoritative for object-local geometry. RGB-D supplies only the target world anchor.
All transforms use column vectors. The default is rigid::

    p_world = R_world_from_object @ p_object + translation_world_m

The source anchor is the center of the complete local-space mesh bounding box. It is translated
onto the robust median RGB-D world center. The canonical SAM3D Y-up orientation is preserved
(``R = identity``) until its optional predicted camera rotation is verified end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import trimesh

ScaleMode = Literal["none", "uniform"]


@dataclass(frozen=True)
class MetricAlignment:
    T_world_from_object: np.ndarray
    scale_mode: ScaleMode
    uniform_scale: float
    rotation_matrix: np.ndarray
    translation_world_m: np.ndarray
    source_anchor_object: np.ndarray
    target_center_world_m: np.ndarray
    source_dimensions: np.ndarray
    aligned_dimensions: np.ndarray
    source_aspect_ratios: np.ndarray
    aligned_aspect_ratios: np.ndarray
    transform_singular_values: np.ndarray
    export_max_centered_vertex_delta: float | None = None
    export_max_sampled_pairwise_distance_delta: float | None = None


def alignment_diagnostics_dict(
    alignment: MetricAlignment, *, rgbd_dimensions_m: np.ndarray | None = None
) -> dict:
    """Return the persistent, explicit geometry-preservation alignment schema."""
    report = {
        "alignment_method": "rigid SAM3D geometry preservation with RGB-D world placement",
        "geometry_source": "sam3d",
        "scale_mode": alignment.scale_mode,
        "uniform_scale": alignment.uniform_scale,
        "source_anchor_policy": "center of complete SAM3D local-space mesh bounding box",
        "source_anchor_object": alignment.source_anchor_object.tolist(),
        "target_center_world_m": alignment.target_center_world_m.tolist(),
        "rotation_matrix": alignment.rotation_matrix.tolist(),
        "translation_world_m": alignment.translation_world_m.tolist(),
        "T_world_from_object": alignment.T_world_from_object.tolist(),
        "sam3d_dimensions_diagnostic": alignment.source_dimensions.tolist(),
        "aligned_dimensions_diagnostic": alignment.aligned_dimensions.tolist(),
        "source_aspect_ratios": alignment.source_aspect_ratios.tolist(),
        "aligned_aspect_ratios": alignment.aligned_aspect_ratios.tolist(),
        "transform_singular_values": alignment.transform_singular_values.tolist(),
        "export_max_centered_vertex_delta": alignment.export_max_centered_vertex_delta,
        "export_max_sampled_pairwise_distance_delta": (
            alignment.export_max_sampled_pairwise_distance_delta
        ),
    }
    if rgbd_dimensions_m is not None:
        report["rgbd_dimensions_m_diagnostic"] = np.asarray(
            rgbd_dimensions_m, dtype=float
        ).tolist()
    return report


def _as_mesh(asset: trimesh.Trimesh | trimesh.Scene) -> trimesh.Trimesh:
    if isinstance(asset, trimesh.Trimesh):
        return asset.copy()
    meshes = [geometry for geometry in asset.dump() if isinstance(geometry, trimesh.Trimesh)]
    if not meshes:
        raise ValueError("SAM3D GLB contains no triangle mesh")
    return trimesh.util.concatenate(meshes)


def _dimensions(vertices: np.ndarray) -> np.ndarray:
    return np.ptp(vertices, axis=0)


def _aspect_ratios(dimensions: np.ndarray) -> np.ndarray:
    largest = float(np.max(dimensions))
    if largest <= 1e-12:
        raise ValueError("SAM3D mesh has degenerate dimensions")
    return np.asarray(dimensions, dtype=np.float64) / largest


def estimate_uniform_scale_from_rgbd_dimensions(
    source_dimensions: np.ndarray, rgbd_dimensions_m: np.ndarray
) -> tuple[float, np.ndarray]:
    """Estimate one robust scalar; never return independent per-axis mesh scales.

    SAM3D GLB and LiteReality are both Y-up in this stage. The median of the three corresponding
    extent ratios prevents one partial or occluded RGB-D extent from dominating the result.
    """
    source = np.asarray(source_dimensions, dtype=np.float64)
    target = np.asarray(rgbd_dimensions_m, dtype=np.float64)
    if source.shape != (3,) or target.shape != (3,):
        raise ValueError("source and RGB-D dimensions must each have shape (3,)")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("source and RGB-D dimensions must be finite")
    if np.any(source <= 1e-8) or np.any(target <= 0):
        raise ValueError("source and RGB-D dimensions must be strictly positive")
    axis_ratios = target / source
    return float(np.median(axis_ratios)), axis_ratios


def validate_shape_preserving_linear_transform(
    linear: np.ndarray, *, scale_mode: ScaleMode, tolerance: float = 1e-6
) -> np.ndarray:
    """Return singular values, rejecting shear or non-uniform deformation."""
    linear = np.asarray(linear, dtype=np.float64)
    if linear.shape != (3, 3) or not np.isfinite(linear).all():
        raise ValueError("object-to-world linear transform must be a finite 3x3 matrix")
    singular_values = np.linalg.svd(linear, compute_uv=False)
    if not np.allclose(singular_values, singular_values[0], rtol=tolerance, atol=tolerance):
        raise ValueError(
            f"non-uniform object deformation is forbidden; singular values={singular_values}"
        )
    if scale_mode == "none" and not np.allclose(
        singular_values, 1.0, rtol=tolerance, atol=tolerance
    ):
        raise ValueError(
            f"scale_mode='none' requires a rigid transform; singular values={singular_values}"
        )
    if np.linalg.det(linear) <= 0:
        raise ValueError("object-to-world transform must not mirror geometry")
    return singular_values


def align_mesh_to_world_anchor(
    mesh: trimesh.Trimesh,
    target_center_world_m: np.ndarray,
    *,
    scale_mode: ScaleMode = "none",
    uniform_scale: float = 1.0,
    rotation_matrix: np.ndarray | None = None,
) -> tuple[trimesh.Trimesh, MetricAlignment]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) < 4 or not np.isfinite(vertices).all():
        raise ValueError("SAM3D mesh must contain at least four finite vertices")
    target_center = np.asarray(target_center_world_m, dtype=np.float64)
    if target_center.shape != (3,) or not np.isfinite(target_center).all():
        raise ValueError("RGB-D target center must be a finite shape-(3,) vector")
    if scale_mode not in ("none", "uniform"):
        raise ValueError(f"unsupported scale mode: {scale_mode!r}")
    if scale_mode == "none" and not np.isclose(uniform_scale, 1.0):
        raise ValueError("scale_mode='none' requires uniform_scale=1.0")
    if not np.isfinite(uniform_scale) or uniform_scale <= 0:
        raise ValueError("uniform scale must be finite and positive")
    rotation = np.eye(3) if rotation_matrix is None else np.asarray(rotation_matrix, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("rotation_matrix must have shape (3,3)")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-6
    ):
        raise ValueError("rotation_matrix must be a proper orthonormal rotation")

    applied_scale = 1.0 if scale_mode == "none" else float(uniform_scale)
    linear = applied_scale * rotation
    singular_values = validate_shape_preserving_linear_transform(linear, scale_mode=scale_mode)
    source_low = vertices.min(axis=0)
    source_high = vertices.max(axis=0)
    source_anchor = (source_low + source_high) / 2.0
    translation = target_center - linear @ source_anchor
    T_world_from_object = np.eye(4)
    T_world_from_object[:3, :3] = linear
    T_world_from_object[:3, 3] = translation
    aligned = mesh.copy()
    aligned.apply_transform(T_world_from_object)
    source_dimensions = _dimensions(vertices)
    aligned_dimensions = _dimensions(np.asarray(aligned.vertices, dtype=np.float64))
    result = MetricAlignment(
        T_world_from_object=T_world_from_object,
        scale_mode=scale_mode,
        uniform_scale=applied_scale,
        rotation_matrix=rotation,
        translation_world_m=translation,
        source_anchor_object=source_anchor,
        target_center_world_m=target_center,
        source_dimensions=source_dimensions,
        aligned_dimensions=aligned_dimensions,
        source_aspect_ratios=_aspect_ratios(source_dimensions),
        aligned_aspect_ratios=_aspect_ratios(aligned_dimensions),
        transform_singular_values=singular_values,
    )
    return aligned, result


def align_sam3d_glb(
    sam3d_glb_path: Path | str,
    output_glb_path: Path | str,
    target_center_world_m: np.ndarray,
    *,
    scale_mode: ScaleMode = "none",
    uniform_scale: float = 1.0,
    rotation_matrix: np.ndarray | None = None,
) -> MetricAlignment:
    mesh = _as_mesh(trimesh.load(sam3d_glb_path, force="scene", process=False))
    aligned, result = align_mesh_to_world_anchor(
        mesh,
        target_center_world_m,
        scale_mode=scale_mode,
        uniform_scale=uniform_scale,
        rotation_matrix=rotation_matrix,
    )
    output = Path(output_glb_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    aligned.export(output)
    # Reload the actual artifact and verify that GLB serialization did not deform it.
    exported = _as_mesh(trimesh.load(output, force="scene", process=False))
    source_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    exported_vertices = np.asarray(exported.vertices, dtype=np.float64)
    if source_vertices.shape != exported_vertices.shape:
        raise ValueError("aligned GLB export changed the SAM3D vertex count")
    restored_vertices = (
        exported_vertices - result.translation_world_m
    ) @ np.linalg.inv(result.T_world_from_object[:3, :3]).T
    centered_delta = float(np.max(np.abs(restored_vertices - source_vertices)))
    sample_indices = np.linspace(
        0, len(source_vertices) - 1, min(256, len(source_vertices)), dtype=int
    )
    source_sample = source_vertices[sample_indices]
    exported_sample = exported_vertices[sample_indices]
    source_distances = np.linalg.norm(
        source_sample[:, None, :] - source_sample[None, :, :], axis=-1
    )
    exported_distances = np.linalg.norm(
        exported_sample[:, None, :] - exported_sample[None, :, :], axis=-1
    )
    expected_distances = source_distances * result.uniform_scale
    pairwise_delta = float(np.max(np.abs(exported_distances - expected_distances)))
    if centered_delta > 1e-5 or pairwise_delta > 1e-5:
        raise ValueError(
            "aligned GLB export failed geometry-preservation validation: "
            f"centered_delta={centered_delta}, pairwise_delta={pairwise_delta}"
        )
    exported_dimensions = _dimensions(exported_vertices)
    return replace(
        result,
        aligned_dimensions=exported_dimensions,
        aligned_aspect_ratios=_aspect_ratios(exported_dimensions),
        export_max_centered_vertex_delta=centered_delta,
        export_max_sampled_pairwise_distance_delta=pairwise_delta,
    )
