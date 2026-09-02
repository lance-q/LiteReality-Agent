"""Simple, inspectable best-observation scoring."""

from __future__ import annotations


def score_observation(observation: dict) -> tuple[float, dict]:
    factors = {
        "mask_area_fraction": float(observation["mask_area_fraction"]),
        "valid_depth_fraction": float(observation["valid_depth_fraction"]),
        "boundary_margin_fraction": float(observation["boundary_margin_fraction"]),
        "sharpness_normalized": min(float(observation["sharpness"]) / 500.0, 1.0),
    }
    score = (
        0.35 * factors["mask_area_fraction"]
        + 0.30 * factors["valid_depth_fraction"]
        + 0.15 * factors["boundary_margin_fraction"]
        + 0.20 * factors["sharpness_normalized"]
    )
    return score, factors


def choose_best_observation(observations: list[dict]) -> tuple[dict, dict]:
    if not observations:
        raise ValueError("cannot choose a view without observations")
    scored = []
    for observation in observations:
        score, factors = score_observation(observation)
        scored.append((score, observation, factors))
    score, observation, factors = max(scored, key=lambda item: item[0])
    return observation, {"score": score, "factors": factors}
