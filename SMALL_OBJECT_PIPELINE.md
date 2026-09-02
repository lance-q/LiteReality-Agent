# Small-object reconstruction

This pipeline detects small objects in a LiteReality Scanner capture, reconstructs them with
[SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects), and places the resulting
meshes in the LiteReality/RoomPlan world.

## Install

Install both projects by following their original documentation:

- [LiteReality Agent](https://github.com/lance-q/LiteReality-Agent)
- [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects)

SAM 3D Objects must have its own Python environment and downloaded checkpoints. Its repository
must contain `notebook/inference.py` and `checkpoints/hf/pipeline.yaml`.

## Run

Run these commands from the LiteReality Agent repository root. Replace the example paths, frame
IDs, prompt, and label with your own:

```bash
export SCAN_DIR="scans/Yanran_LivingRoom"
export OUTPUT_DIR="run/Yanran_LivingRoom/small_objects"
export SAM3D_REPO="/path/to/sam-3d-objects"
export SAM3D_PYTHON="/path/to/sam3d/environment/bin/python"
mkdir -p "$OUTPUT_DIR"
```

### 1. Select keyframes

```bash
uv run python -m litereality_agent.small_object_pipeline.run_pipeline \
  select-keyframes "$SCAN_DIR" \
  --output "$OUTPUT_DIR/keyframes.json"
```

Choose useful frame IDs from `keyframes.json`.

### 2. Detect and segment the object

```bash
uv run python -m litereality_agent.small_object_pipeline.detect_segment_worker \
  "$SCAN_DIR" \
  --frames 45 75 \
  --prompt "small orange pumpkin toy." \
  --label pumpkin_toy \
  --output-dir "$OUTPUT_DIR/detections"
```

### 3. Associate observations

```bash
uv run python -m litereality_agent.small_object_pipeline.build_observations \
  "$SCAN_DIR" \
  "$OUTPUT_DIR/detections/detections_manifest.json" \
  --output-dir "$OUTPUT_DIR"
```

### 4. Reconstruct with SAM 3D Objects

```bash
uv run python -m litereality_agent.small_object_pipeline.sam3d_batch \
  "$OUTPUT_DIR/objects_manifest.json" \
  --python "$SAM3D_PYTHON" \
  --sam3d-repo "$SAM3D_REPO" \
  --output-dir "$OUTPUT_DIR/sam3d_outputs"
```

Successful reconstructions are cached. Add `--force` to rerun them.

### 5. Align the meshes

```bash
uv run python -m litereality_agent.small_object_pipeline.finalize_objects \
  "$OUTPUT_DIR/objects_manifest.json" \
  --sam3d-output-dir "$OUTPUT_DIR/sam3d_outputs" \
  --aligned-output-dir "$OUTPUT_DIR/aligned_objects" \
  --scale-mode uniform
```

The aligned GLB files are written to `$OUTPUT_DIR/aligned_objects`. Use `--scale-mode none` to
preserve the original SAM 3D scale.

## Notes

- The scanner capture must contain matching RGB, depth, confidence, and camera JSON files.
- Detection requires a text prompt and explicit frame IDs.
- RGB-D provides metric position and scale; SAM 3D Objects provides object geometry and texture.
- The output GLB uses the LiteReality/RoomPlan world frame in metres with +Y up.
