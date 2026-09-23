"""Merge multiple ARKit-world aligned small-object GLBs into one RoomPlan scene.

Run with Blender, not CPython::

    blender --background --python batch_room_merge.py -- \
        ROOM.usdz OBJECTS.glb OUT.blend OUT.glb LABEL=OBJECT.glb [...]

All inputs are already expressed in the LiteReality/ARKit world frame. Blender's
importers perform the standard Y-up to Z-up conversion consistently, so this
worker deliberately applies no additional transform.
"""

from __future__ import annotations

import sys
from pathlib import Path

import bpy


def _parse_asset(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise ValueError(f"expected LABEL=PATH, got {value!r}")
    return label, Path(path)


def _import_objects(assets: list[tuple[str, Path]], *, decimate_ratio: float = 1.0) -> None:
    for label, asset_path in assets:
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=str(asset_path.resolve()))
        imported = set(bpy.data.objects) - before
        if not imported:
            raise RuntimeError(f"no objects imported from {asset_path}")
        for obj in imported:
            obj["litereality_small_object"] = True
            obj["small_object_label"] = label
            obj["aligned_asset_path"] = str(asset_path.resolve())
            obj["coordinate_system"] = "ARKit/RoomPlan world; imported from Y-up GLB"
            if obj.type == "MESH" and decimate_ratio < 1.0 and len(obj.data.polygons) >= 1000:
                bpy.context.view_layer.objects.active = obj
                obj.select_set(True)
                modifier = obj.modifiers.new(name="LiteReality delivery decimation", type="DECIMATE")
                modifier.ratio = decimate_ratio
                bpy.ops.object.modifier_apply(modifier=modifier.name)
                obj.select_set(False)


def _export_objects_only(
    output_glb: Path,
    assets: list[tuple[str, Path]],
    *,
    decimate_ratio: float,
    draco: bool,
) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    _import_objects(assets, decimate_ratio=decimate_ratio)
    output_glb.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(output_glb.resolve()),
        export_format="GLB",
        export_draco_mesh_compression_enable=draco,
    )


def main() -> int:
    if "--" not in sys.argv:
        raise SystemExit("expected arguments after --")
    args = sys.argv[sys.argv.index("--") + 1 :]
    exclude_room_objects: list[str] = []
    decimate_ratio = 1.0
    draco = False
    while "--exclude-room-object" in args:
        option_index = args.index("--exclude-room-object")
        if option_index + 1 >= len(args):
            raise SystemExit("--exclude-room-object requires an object name")
        exclude_room_objects.append(args[option_index + 1])
        del args[option_index : option_index + 2]
    if "--decimate-ratio" in args:
        option_index = args.index("--decimate-ratio")
        decimate_ratio = float(args[option_index + 1])
        del args[option_index : option_index + 2]
    if "--draco" in args:
        args.remove("--draco")
        draco = True
    if not 0.0 < decimate_ratio <= 1.0:
        raise ValueError("--decimate-ratio must be in (0, 1]")
    if len(args) < 5:
        raise SystemExit(
            "usage: batch_room_merge.py -- ROOM.usdz OBJECTS.glb OUT.blend OUT.glb "
            "LABEL=OBJECT.glb [LABEL=OBJECT.glb ...] [--exclude-room-object NAME]"
        )
    room_usdz, objects_glb, output_blend, output_glb = map(Path, args[:4])
    assets = [_parse_asset(value) for value in args[4:]]
    for source in [room_usdz, *(path for _, path in assets)]:
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(source)

    _export_objects_only(
        objects_glb, assets, decimate_ratio=decimate_ratio, draco=draco
    )

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.usd_import(filepath=str(room_usdz.resolve()))
    for name in exclude_room_objects:
        matches = [obj for obj in bpy.data.objects if obj.name == name or obj.name.startswith(name)]
        if not matches:
            raise ValueError(f"RoomPlan object {name!r} was not found for debug exclusion")
        for obj in matches:
            bpy.data.objects.remove(obj, do_unlink=True)
        print(f"excluded RoomPlan debug proxy {name}: {len(matches)} Blender object(s)")
    _import_objects(assets, decimate_ratio=decimate_ratio)
    output_blend.parent.mkdir(parents=True, exist_ok=True)
    output_glb.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_blend.resolve()))
    bpy.ops.export_scene.gltf(
        filepath=str(output_glb.resolve()),
        export_format="GLB",
        export_draco_mesh_compression_enable=draco,
    )
    print(f"wrote {objects_glb}")
    print(f"wrote {output_blend}")
    print(f"wrote {output_glb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
