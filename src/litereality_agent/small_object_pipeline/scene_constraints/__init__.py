"""Conservative scene constraints for uniformly aligned SAM3D objects."""

from .profiles import ConstraintProfile, profile_for_label
from .solver import SceneConstraintSolver, SolverConfig

__all__ = ["ConstraintProfile", "SceneConstraintSolver", "SolverConfig", "profile_for_label"]
