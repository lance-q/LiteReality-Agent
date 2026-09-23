# LiteReality small-object reconstruction

This repository extends
[LiteReality-Agent](https://github.com/ZheningHuang/LiteReality-Agent) with an
RGB-D small-object pipeline backed by
[SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects). LiteReality Scanner and
LiteReality-Agent remain authoritative for the metric room and ARKit world frame; SAM3D supplies
the geometry and appearance of individual objects.

```text
LiteReality Scanner capture
-> LiteReality global room reconstruction
-> RGB-D small-object detection and masks
-> multi-view association/deduplication
-> one SAM3D reconstruction per physical object
-> uniform metric alignment
-> scene-constraint solver
-> section export
```

LiteReality-Agent and SAM3D are external dependencies. Their source, model weights, scans, and
generated results are intentionally not vendored in this repository.

## Expected workspace

Use three sibling checkouts. The names can differ; command-line options provide their paths.

```text
<workspace>/
|-- litereality-small-objects/   # this repository
|-- LiteReality-Agent/           # upstream LiteReality-Agent clone
|-- sam-3d-objects/              # upstream SAM3D clone
|-- scans/                        # local Scanner captures (untracked)
`-- results/                      # local reconstruction workspaces (untracked)
```

The current development checkout may contain the full LiteReality-Agent tree around this overlay.
Those upstream files are deliberately ignored and remain available locally.

## Installation

LiteReality-Agent currently supports Python 3.10-3.12 and uses `uv`. Install it first using its
upstream instructions. The commands below create a development environment and expose this
overlay in that checkout without copying the two upstream repositories into Git:

```bash
mkdir <workspace>
cd <workspace>

# External dependencies: keep these as separate repositories.
git clone https://github.com/ZheningHuang/LiteReality-Agent.git LiteReality-Agent
git clone https://github.com/facebookresearch/sam-3d-objects.git sam-3d-objects

# This integration repository contains only the overlay code.
git clone https://github.com/lance-q/LiteReality-Agent.git litereality-small-objects

cd LiteReality-Agent
uv sync --group dev

cd ../litereality-small-objects
uv sync --group dev
uv run python scripts/install_overlay.py \
  --litereality-agent ../LiteReality-Agent \
  --link
```

`--link` creates a development symlink at
`LiteReality-Agent/src/litereality_agent/small_object_pipeline`. Use `--copy` for a standalone
copy instead. The installer refuses to replace an unrelated existing directory.

For normal development, use `--link`: edits in this integration checkout become immediately
visible to LiteReality-Agent. For a fixed deployment snapshot, use `--copy`; rerunning `--copy`
does not overwrite an existing pipeline, so remove or archive that destination deliberately before
installing a newer snapshot.

Set up SAM3D in its own environment by following its upstream README. The pipeline expects the
SAM3D checkout to contain `notebook/inference.py` and `checkpoints/hf/pipeline.yaml`. Blender is
needed only for room/section export. The known working LiteReality-Agent setup uses Blender 5.x.

The lightweight overlay dependencies are declared in `pyproject.toml`: NumPy, Pillow, OpenCV, and
trimesh. Detection and SAM3D retain their separate upstream environments.

## Model setup

Do not place model weights in this repository.

- GroundingDINO is supplied by the LiteReality-Agent detection setup.
- SAM instance segmentation is loaded through the existing LiteReality-Agent environment.
- SAM3D checkpoints belong under the separately cloned SAM3D repository, as documented upstream.

Pass the SAM3D repository and interpreter explicitly in the batch configuration or CLI. Do not
commit API keys, checkpoint paths, or machine-specific absolute paths.

## Scanner input

A LiteReality Scanner capture is expected to provide matching RGB frames, depth, confidence,
camera intrinsics, per-frame ARKit camera-to-world transforms, and `room.usdz`. World units are
metres and world +Y is vertical. Store captures outside this repository, such as
`<workspace>/scans/<scan-name>/`.

## Running the pipeline

Run pipeline modules from the LiteReality-Agent checkout after installing the overlay. The staged
commands and metadata formats are documented in [SMALL_OBJECT_PIPELINE.md](SMALL_OBJECT_PIPELINE.md).
A typical end-to-end batch uses:

```bash
cd <workspace>/LiteReality-Agent
uv run python -m litereality_agent.small_object_pipeline.batch_runner \
  <workspace>/results/<scan-name>/batch_config.json
```

Start from `configs/batch.example.json` and replace its relative example paths:

```bash
mkdir -p <workspace>/results/<scan-name>/small_objects
cp <workspace>/litereality-small-objects/configs/batch.example.json \
  <workspace>/results/<scan-name>/small_objects/batch_config.json
```

The configuration must point `sam3d_repo` at the separate SAM3D checkout and `sam3d_python` at
that project's Python environment. It must point `room_usdz` at the Scanner capture and `blender`
at the Blender executable. Paths may be absolute or relative to the directory from which the
batch runner is launched.

Important scene constraint controls are:

```json
{
  "scene_constraints_enabled": true,
  "scene_constraint_diagnostics_only": true
}
```

Diagnostics-only mode records proposed support/orientation corrections without replacing aligned
GLBs. Once reviewed, set it to `false`. Manually validated transforms listed under
`validated_alignment_overrides` are never overwritten by the solver.

The pipeline also supports individual stages for keyframe selection, detection/segmentation,
association, cached SAM3D reconstruction, uniform alignment, scene constraints, and section
export. See the companion document for commands that exist today.

## Outputs

The configured workspace contains generated data such as:

- `sam3d_outputs/`: cached SAM3D `object.glb` assets;
- `aligned_objects/`: uniformly scaled, rigidly placed world-space objects;
- `observations/`, masks, point clouds, and manifests;
- section object-only and room-plus-object GLBs;
- Blender working files and constraint diagnostics.

These outputs are development data and normally remain untracked. Keep useful validated caches
locally; copy or archive them separately when transferring a run.

## Development notes

SAM3D `object.glb` is immutable source geometry. Alignment is

```text
p_world = s * R * p_object + t
```

with one uniform scalar `s` and a rigid pose. Independent axis scales, shear, and vertex warping
are prohibited. RGB-D dimensions are diagnostics and scale evidence, not permission to deform a
mesh. The scene-constraint solver may select upright/yaw candidates and adjust placement against
floors, supports, and walls while retaining the same uniform scale.

Safe CPU-only checks, after the overlay is installed, are:

```bash
cd <workspace>/litereality-small-objects
uv run ruff check src tests scripts
PYTHONPATH=<workspace>/LiteReality-Agent/src:src uv run pytest -q tests
```

Do not run the SAM3D batch command without its configured GPU environment.

## Licensing and attribution

The retained Apache-2.0 `LICENSE` comes from LiteReality-Agent and is kept because this overlay is
derived from and integrates with that project. SAM 3D Objects remains external and is governed by
the license in its own repository.
