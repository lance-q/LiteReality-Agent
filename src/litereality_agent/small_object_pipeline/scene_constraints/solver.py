"""Independent discrete scene-constraint solving for each uniformly scaled object."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import trimesh

from ..object_alignment import (
    _as_mesh,
    _aspect_ratios,
    validate_shape_preserving_linear_transform,
)
from .orientation import generate_orientation_candidates, uniform_scale_and_rotation
from .profiles import ConstraintProfile, load_profile_overrides, profile_for_label
from .room_geometry import load_room_constraints, nearest_wall
from .scoring import ScoreWeights, score_candidate
from .types import CandidateContext, CandidatePose, CandidateScoringHook, SupportSurface


@dataclass(frozen=True)
class SolverConfig:
    robust_bottom_quantile: float = 0.02
    footprint_band_quantile: float = 0.10
    footprint_tolerance_m: float = 0.03
    support_lookup_tolerance_m: float = 0.12
    minimum_support_coverage: float = 0.70
    minimum_score_improvement: float = 0.05
    minimum_distinct_score_margin: float = 0.02
    weights: ScoreWeights = field(default_factory=ScoreWeights)

    def __post_init__(self) -> None:
        quantiles = (self.robust_bottom_quantile, self.footprint_band_quantile)
        if not all(0.0 <= value < 0.5 for value in quantiles):
            raise ValueError("bottom/footprint quantiles must be in [0, 0.5)")
        if self.footprint_band_quantile < self.robust_bottom_quantile:
            raise ValueError("footprint band quantile must not be below bottom quantile")
        if not 0.0 <= self.minimum_support_coverage <= 1.0:
            raise ValueError("minimum support coverage must be in [0, 1]")
        nonnegative = (
            self.footprint_tolerance_m,
            self.support_lookup_tolerance_m,
            self.minimum_score_improvement,
            self.minimum_distinct_score_margin,
            *asdict(self.weights).values(),
        )
        if not all(np.isfinite(value) and value >= 0 for value in nonnegative):
            raise ValueError(
                "scene-constraint thresholds and weights must be finite and non-negative"
            )


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _candidate_record(candidate: CandidatePose, total: float, terms: dict[str, float]) -> dict:
    return {
        "candidate_id": candidate.candidate_id,
        "upright_axis": candidate.upright_axis,
        "yaw_deg": candidate.yaw_deg,
        "support_surface": candidate.support.name if candidate.support else None,
        "support_height_world_y_m": (
            candidate.support.height_world_y_m if candidate.support else None
        ),
        "support_snap_m": candidate.support_snap_m,
        "support_coverage": candidate.support_coverage,
        "score_terms": terms,
        "total_score": total,
        "T_world_from_object": candidate.T_world_from_object,
    }


class SceneConstraintSolver:
    def __init__(
        self,
        room_usdz: Path | str,
        *,
        config: SolverConfig | None = None,
        profile_overrides: dict[str, str] | None = None,
        scoring_hooks: tuple[CandidateScoringHook, ...] = (),
    ) -> None:
        self.room_usdz = Path(room_usdz)
        self.config = config or SolverConfig()
        self.profile_overrides = profile_overrides or {}
        self.scoring_hooks = scoring_hooks
        self.supports, self.walls = load_room_constraints(self.room_usdz)
        self.floor = next(
            (surface for surface in self.supports if surface.category == "floor"), None
        )

    def _support_candidates(
        self,
        profile: ConstraintProfile,
        target_center_world_m: np.ndarray,
    ) -> list[SupportSurface]:
        if profile.support_mode == "floor":
            return [self.floor] if self.floor is not None else []
        if profile.support_mode != "horizontal":
            return []
        target_xz = target_center_world_m[(0, 2)]
        supports = [
            surface
            for surface in self.supports
            if surface.category in profile.support_categories
            and surface.contains_xz(
                target_xz, tolerance_m=self.config.support_lookup_tolerance_m
            )
        ]
        # A low object may genuinely rest on the floor even if its semantic profile usually means
        # table-supported.  Keep this as a candidate, never a forced fallback.
        if (
            self.floor is not None
            and target_center_world_m[1] <= self.floor.height_world_y_m + 0.60
        ):
            supports.append(self.floor)
        return supports

    def _make_candidate(
        self,
        *,
        candidate_id: str,
        source_vertices: np.ndarray,
        source_anchor: np.ndarray,
        target_center_world_m: np.ndarray,
        uniform_scale: float,
        rotation: np.ndarray,
        upright_axis: str,
        yaw_deg: float,
        support: SupportSurface | None,
        maximum_support_snap_m: float,
    ) -> CandidatePose | None:
        linear = uniform_scale * rotation
        validate_shape_preserving_linear_transform(linear, scale_mode="uniform")
        translation = target_center_world_m - linear @ source_anchor
        vertices = source_vertices @ linear.T + translation
        support_snap_m = 0.0
        coverage = None
        if support is not None:
            bottom = float(np.quantile(vertices[:, 1], self.config.robust_bottom_quantile))
            support_snap_m = support.height_world_y_m - bottom
            if abs(support_snap_m) > maximum_support_snap_m:
                return None
            translation = translation.copy()
            translation[1] += support_snap_m
            vertices = vertices.copy()
            vertices[:, 1] += support_snap_m
            band_height = float(np.quantile(vertices[:, 1], self.config.footprint_band_quantile))
            footprint_points = vertices[vertices[:, 1] <= band_height + 1e-8][:, (0, 2)]
            coverage = support.coverage(
                footprint_points, tolerance_m=self.config.footprint_tolerance_m
            )
        transform = np.eye(4)
        transform[:3, :3] = linear
        transform[:3, 3] = translation
        return CandidatePose(
            candidate_id,
            transform,
            rotation,
            uniform_scale,
            upright_axis,
            yaw_deg,
            support,
            support_snap_m,
            coverage,
            vertices,
        )

    def _initial_candidate(
        self, source_vertices: np.ndarray, transform: np.ndarray, scale: float, rotation: np.ndarray
    ) -> CandidatePose:
        return CandidatePose(
            "initial",
            transform.copy(),
            rotation,
            scale,
            "current",
            0.0,
            None,
            0.0,
            None,
            source_vertices @ transform[:3, :3].T + transform[:3, 3],
        )

    def solve_object(
        self,
        item: dict,
        source_mesh: trimesh.Trimesh | None,
        *,
        validated_override: bool = False,
    ) -> tuple[CandidatePose | None, dict]:
        object_id = str(item["object_id"])
        label = str(item["label"])
        profile = profile_for_label(label, self.profile_overrides)
        initial_transform = np.asarray(item["T_world_from_object"], dtype=float)
        scale, initial_rotation = uniform_scale_and_rotation(initial_transform)
        target_center = np.asarray(item["rgbd_world_center_m"], dtype=float)
        rgbd_dimensions = np.asarray(item["rgbd_dimensions_m"], dtype=float)
        observation_centers = np.asarray(
            [observation["world_center_m"] for observation in item.get("observations", [])],
            dtype=float,
        ).reshape((-1, 3))
        wall, wall_distance = nearest_wall(
            target_center, self.walls, maximum_distance_m=profile.maximum_wall_distance_m
        )
        diagnostics = {
            "solver": "discrete per-object LiteReality scene constraints v1",
            "object_id": object_id,
            "label": label,
            "constraint_profile": profile.name,
            "initial_T_world_from_object": initial_transform,
            "uniform_scale_preserved": scale,
            "selected_support_surface": None,
            "support_height_world_y_m": None,
            "nearest_wall": wall.name if wall else None,
            "nearest_wall_distance_m": wall_distance,
            "tested_candidates": [],
            "selected_candidate": None,
            "final_T_world_from_object": initial_transform,
            "validated_override_preserved": validated_override,
            "applied": False,
            "warnings": [],
        }
        if validated_override:
            diagnostics["warnings"].append(
                "validated alignment override: automatic pose modification skipped"
            )
            return None, _jsonable(diagnostics)
        if profile.name == "free_form":
            diagnostics["warnings"].append("weak/free-form profile: retained existing alignment")
            return None, _jsonable(diagnostics)
        if profile.support_mode == "wall":
            diagnostics["warnings"].append(
                "wall-mounted translation is reserved for a future solver; "
                "retained existing alignment"
            )
            return None, _jsonable(diagnostics)
        if source_mesh is None:
            raise ValueError(f"source SAM3D mesh is required for unconstrained object {object_id}")
        source_vertices = np.asarray(source_mesh.vertices, dtype=float)
        source_anchor = (source_vertices.min(axis=0) + source_vertices.max(axis=0)) / 2.0

        context = CandidateContext(
            label,
            profile.name,
            target_center,
            rgbd_dimensions,
            observation_centers,
            self.floor,
            wall,
        )
        initial = self._initial_candidate(
            source_vertices, initial_transform, scale, initial_rotation
        )
        candidates = [initial]
        supports = self._support_candidates(profile, target_center)
        orientation_candidates = generate_orientation_candidates(
            initial_rotation,
            enforce_upright=(
                profile.enforce_upright and not (profile.wall_alignment and wall is None)
            ),
            nearest_wall=wall,
        )
        for orientation_id, upright_axis, yaw_deg, rotation in orientation_candidates:
            for support in [None, *supports]:
                candidate = self._make_candidate(
                    candidate_id=f"{orientation_id}|support={support.name if support else 'none'}",
                    source_vertices=source_vertices,
                    source_anchor=source_anchor,
                    target_center_world_m=target_center,
                    uniform_scale=scale,
                    rotation=rotation,
                    upright_axis=upright_axis,
                    yaw_deg=yaw_deg,
                    support=support,
                    maximum_support_snap_m=profile.maximum_support_snap_m,
                )
                if candidate is not None and not any(
                    np.allclose(
                        candidate.T_world_from_object,
                        previous.T_world_from_object,
                        atol=1e-7,
                    )
                    for previous in candidates
                ):
                    candidates.append(candidate)

        scored: list[tuple[float, CandidatePose, dict[str, float]]] = []
        for candidate in candidates:
            total, terms = score_candidate(
                candidate,
                context,
                weights=self.config.weights,
                require_support=profile.support_mode in ("floor", "horizontal"),
                require_upright=profile.enforce_upright,
                wall_alignment=profile.wall_alignment,
                minimum_support_coverage=self.config.minimum_support_coverage,
                hooks=self.scoring_hooks,
            )
            scored.append((total, candidate, terms))
        scored.sort(key=lambda value: (value[0], value[1].candidate_id))
        diagnostics["tested_candidates"] = [
            _candidate_record(candidate, total, terms) for total, candidate, terms in scored
        ]
        initial_score = next(
            total for total, candidate, _ in scored if candidate.candidate_id == "initial"
        )
        best_score, best, best_terms = scored[0]
        distinct_scores = sorted({round(total, 10) for total, _, _ in scored})
        margin = distinct_scores[1] - distinct_scores[0] if len(distinct_scores) > 1 else None
        improvement = initial_score - best_score
        diagnostics["initial_total_score"] = initial_score
        diagnostics["best_total_score"] = best_score
        diagnostics["score_improvement"] = improvement
        diagnostics["distinct_score_margin"] = margin

        if not supports and profile.support_mode in ("floor", "horizontal"):
            diagnostics["warnings"].append("no plausible support surface found")
        if profile.wall_alignment and wall is None:
            diagnostics["warnings"].append(
                "no sufficiently near wall; wall-relative rotation disabled"
            )
        if best.support is None and profile.support_mode in ("floor", "horizontal"):
            diagnostics["warnings"].append(
                "best candidate has no support; retained existing alignment"
            )
            return None, _jsonable(diagnostics)
        if improvement < self.config.minimum_score_improvement:
            diagnostics["warnings"].append(
                "constraint candidate did not improve the initial score enough"
            )
            return None, _jsonable(diagnostics)
        if margin is not None and margin < self.config.minimum_distinct_score_margin:
            diagnostics["warnings"].append(
                "candidate scores are ambiguous; retained existing alignment"
            )
            return None, _jsonable(diagnostics)
        if (
            best.support_coverage is not None
            and best.support_coverage < self.config.minimum_support_coverage
        ):
            diagnostics["warnings"].append("selected candidate has insufficient support footprint")
            return None, _jsonable(diagnostics)

        diagnostics.update(
            {
                "selected_candidate": _candidate_record(best, best_score, best_terms),
                "selected_support_surface": best.support.name if best.support else None,
                "support_height_world_y_m": (
                    best.support.height_world_y_m if best.support else None
                ),
                "selected_upright_axis": best.upright_axis,
                "selected_yaw_deg": best.yaw_deg,
                "final_T_world_from_object": best.T_world_from_object,
                "applied": True,
            }
        )
        if best_terms["collision"] > 0:
            diagnostics["warnings"].append(
                "selected candidate retains possible floor/wall penetration "
                "within conservative tolerances"
            )
        return best, _jsonable(diagnostics)

    @staticmethod
    def export_candidate(
        source_mesh: trimesh.Trimesh, candidate: CandidatePose, destination: Path | str
    ) -> None:
        linear = candidate.T_world_from_object[:3, :3]
        singular_values = validate_shape_preserving_linear_transform(linear, scale_mode="uniform")
        if not np.allclose(singular_values, candidate.uniform_scale, rtol=1e-6, atol=1e-6):
            raise ValueError("scene constraint candidate changed the persisted uniform scale")
        constrained = source_mesh.copy()
        constrained.apply_transform(candidate.T_world_from_object)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.scene-constraint.tmp.glb")
        constrained.export(temporary)
        exported = _as_mesh(trimesh.load(temporary, force="scene", process=False))
        source_vertices = np.asarray(source_mesh.vertices, dtype=float)
        exported_vertices = np.asarray(exported.vertices, dtype=float)
        if source_vertices.shape != exported_vertices.shape:
            raise ValueError("scene-constrained GLB export changed the SAM3D vertex count")
        restored = (
            exported_vertices - candidate.T_world_from_object[:3, 3]
        ) @ np.linalg.inv(linear).T
        if float(np.max(np.abs(restored - source_vertices))) > 1e-5:
            raise ValueError("scene-constrained GLB export deformed or reordered SAM3D geometry")
        temporary.replace(destination)


def constrain_manifest(
    objects_manifest: Path | str,
    *,
    room_usdz: Path | str,
    sam3d_output_dir: Path | str,
    aligned_output_dir: Path | str,
    validated_object_ids: set[str] | None = None,
    profile_overrides_path: Path | str | None = None,
    diagnostics_only: bool = False,
    config: SolverConfig | None = None,
) -> dict:
    manifest_path = Path(objects_manifest)
    manifest = json.loads(manifest_path.read_text())
    output_dir = Path(aligned_output_dir)
    protected = validated_object_ids or set()
    solver = SceneConstraintSolver(
        room_usdz,
        config=config,
        profile_overrides=load_profile_overrides(profile_overrides_path),
    )
    applied_count = 0
    for item in manifest["objects"]:
        object_id = str(item["object_id"])
        source_mesh = None
        if object_id not in protected:
            source_path = Path(
                item.get("sam3d_asset")
                or Path(sam3d_output_dir) / object_id / "object.glb"
            )
            source_mesh = _as_mesh(trimesh.load(source_path, force="scene", process=False))
        candidate, diagnostics = solver.solve_object(
            item, source_mesh, validated_override=object_id in protected
        )
        if diagnostics_only and candidate is not None:
            diagnostics["applied"] = False
            diagnostics["warnings"].append(
                "diagnostics-only mode: candidate was not exported"
            )
        elif candidate is not None:
            if source_mesh is None:
                raise AssertionError("unprotected candidate must retain its source mesh")
            destination = output_dir / f"{object_id}_world.glb"
            solver.export_candidate(source_mesh, candidate, destination)
            item["T_world_from_object"] = candidate.T_world_from_object.tolist()
            item["aligned_asset"] = str(destination.resolve())
            item["constraint_profile"] = diagnostics["constraint_profile"]
            applied_count += 1

        item["scene_constraints"] = diagnostics
        alignment_path = Path(
            item.get("alignment_manifest") or output_dir / f"{object_id}_alignment.json"
        )
        alignment = json.loads(alignment_path.read_text()) if alignment_path.is_file() else {}
        alignment["scene_constraints"] = diagnostics
        if candidate is not None and not diagnostics_only:
            vertices = candidate.transformed_vertices_world_m
            alignment.update(
                {
                    "alignment_method": "uniform SAM3D scale + discrete scene constraints",
                    "rotation_matrix": candidate.rotation_world_from_object.tolist(),
                    "translation_world_m": candidate.T_world_from_object[:3, 3].tolist(),
                    "T_world_from_object": candidate.T_world_from_object.tolist(),
                    "aligned_dimensions_diagnostic": np.ptp(vertices, axis=0).tolist(),
                    "aligned_aspect_ratios": _aspect_ratios(np.ptp(vertices, axis=0)).tolist(),
                    "transform_singular_values": [candidate.uniform_scale] * 3,
                }
            )
        alignment_path.parent.mkdir(parents=True, exist_ok=True)
        alignment_path.write_text(json.dumps(alignment, indent=2) + "\n")
        item["alignment_manifest"] = str(alignment_path.resolve())

    manifest["scene_constraint_solver"] = {
        "enabled": True,
        "room_usdz": str(Path(room_usdz).resolve()),
        "diagnostics_only": diagnostics_only,
        "validated_object_ids": sorted(protected),
        "objects_modified": applied_count,
        "config": _jsonable(asdict(solver.config)),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest["scene_constraint_solver"]
