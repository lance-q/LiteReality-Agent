"""Apply conservative scene constraints after uniform SAM3D metric alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .scoring import ScoreWeights
from .solver import SolverConfig, constrain_manifest


def _load_config(path: Path | None) -> SolverConfig:
    if path is None:
        return SolverConfig()
    data = json.loads(path.read_text())
    weights = ScoreWeights(**data.pop("weights", {}))
    return SolverConfig(weights=weights, **data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("objects_manifest", type=Path)
    parser.add_argument("--room-usdz", type=Path, required=True)
    parser.add_argument("--sam3d-output-dir", type=Path, required=True)
    parser.add_argument("--aligned-output-dir", type=Path, required=True)
    parser.add_argument("--validated-object-id", action="append", default=[])
    parser.add_argument("--profile-overrides", type=Path)
    parser.add_argument("--solver-config", type=Path)
    parser.add_argument("--diagnostics-only", action="store_true")
    args = parser.parse_args()
    result = constrain_manifest(
        args.objects_manifest,
        room_usdz=args.room_usdz,
        sam3d_output_dir=args.sam3d_output_dir,
        aligned_output_dir=args.aligned_output_dir,
        validated_object_ids=set(args.validated_object_id),
        profile_overrides_path=args.profile_overrides,
        diagnostics_only=args.diagnostics_only,
        config=_load_config(args.solver_config),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
