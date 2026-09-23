"""Lightweight semantic profiles; unknown labels deliberately receive weak constraints."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from .types import SupportMode


@dataclass(frozen=True)
class ConstraintProfile:
    name: str
    support_mode: SupportMode
    enforce_upright: bool
    wall_alignment: bool = False
    support_categories: tuple[str, ...] = ()
    maximum_support_snap_m: float = 0.35
    maximum_wall_distance_m: float = 1.0


PROFILES = {
    "floor_standing": ConstraintProfile(
        "floor_standing", "floor", True, maximum_support_snap_m=0.60
    ),
    "supported": ConstraintProfile(
        "supported",
        "horizontal",
        True,
        support_categories=("table", "storage", "stove", "oven", "dishwasher", "sink"),
    ),
    "wall_relative_supported": ConstraintProfile(
        "wall_relative_supported",
        "horizontal",
        True,
        wall_alignment=True,
        support_categories=("table", "storage", "stove", "oven", "dishwasher", "sink"),
        maximum_wall_distance_m=1.25,
    ),
    "free_form": ConstraintProfile("free_form", "none", False),
    "wall_mounted": ConstraintProfile(
        "wall_mounted", "wall", False, wall_alignment=True, maximum_wall_distance_m=0.75
    ),
}

_FLOOR_STANDING = (
    "trash_can",
    "trash_bin",
    "garbage_bin",
    "waste_bin",
    "bin_trash",
    "floor_basket",
    "standing_plant",
    "vacuum",
)
_WALL_RELATIVE = (
    "microwave",
    "rectangular_appliance",
    "television",
    "tv",
    "cabinet",
    "shelf",
)
_SUPPORTED = (
    "kettle",
    "pumpkin",
    "toaster",
    "coffee_maker",
    "blender",
    "rice_cooker",
    "pressure_cooker",
    "bowl",
    "plate",
    "dish",
    "pot",
    "pan",
    "cup",
    "mug",
    "bottle",
    "book",
    "speaker",
    "router",
    "printer",
    "laptop",
    "monitor",
    "toy",
    "figurine",
    "candle",
    "container",
    "jar",
    "can",
    "food_package",
)
_FREE_FORM = ("pillow", "cushion", "blanket", "clothing", "towel", "cable", "headphones")


def _normalized(label: str) -> str:
    return "_".join(label.strip().lower().replace("-", "_").split())


def _contains_term(label: str, term: str) -> bool:
    return (
        label == term
        or label.startswith(f"{term}_")
        or label.endswith(f"_{term}")
        or f"_{term}_" in label
    )


def default_profile_name(label: str) -> str:
    normalized = _normalized(label)
    if any(_contains_term(normalized, token) for token in _FREE_FORM):
        return "free_form"
    if any(_contains_term(normalized, token) for token in _FLOOR_STANDING):
        return "floor_standing"
    if any(_contains_term(normalized, token) for token in _WALL_RELATIVE):
        return "wall_relative_supported"
    if any(_contains_term(normalized, token) for token in _SUPPORTED):
        return "supported"
    return "free_form"


def load_profile_overrides(path: Path | str | None) -> dict[str, str]:
    if path is None:
        return {}
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in data.items()
    ):
        raise ValueError("profile override JSON must map labels to profile names")
    unknown = sorted(set(data.values()) - set(PROFILES))
    if unknown:
        raise ValueError(f"unknown constraint profiles: {unknown}")
    return {_normalized(label): profile for label, profile in data.items()}


def profile_for_label(
    label: str,
    overrides: dict[str, str] | None = None,
    *,
    maximum_support_snap_m: float | None = None,
) -> ConstraintProfile:
    normalized = _normalized(label)
    name = (overrides or {}).get(normalized, default_profile_name(normalized))
    profile = PROFILES[name]
    if maximum_support_snap_m is not None and profile.support_mode in ("floor", "horizontal"):
        profile = replace(profile, maximum_support_snap_m=maximum_support_snap_m)
    return profile
