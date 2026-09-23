"""Install this repository's pipeline into a separate LiteReality-Agent checkout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _destination(checkout: Path) -> Path:
    package = checkout.resolve() / "src" / "litereality_agent"
    if not (package / "__init__.py").is_file():
        raise SystemExit(f"not a LiteReality-Agent checkout: {checkout}")
    return package / "small_object_pipeline"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--litereality-agent", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--link", action="store_true", help="create a development symlink")
    mode.add_argument("--copy", action="store_true", help="copy a standalone source snapshot")
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[1]
    source = repository / "src" / "litereality_agent" / "small_object_pipeline"
    destination = _destination(args.litereality_agent)

    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() and destination.resolve() == source.resolve() and args.link:
            print(f"overlay already linked: {destination}")
            return 0
        raise SystemExit(
            f"refusing to replace existing path: {destination}\n"
            "Move it aside explicitly, then run this installer again."
        )

    if args.link:
        destination.symlink_to(source, target_is_directory=True)
        print(f"linked {destination} -> {source}")
    else:
        shutil.copytree(source, destination)
        print(f"copied {source} -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
