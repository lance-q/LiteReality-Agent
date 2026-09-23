"""Blender worker: merge an ARKit-world aligned GLB with a RoomPlan USDZ debug room.

Both source formats declare Y-up. Blender's importers convert both into Blender Z-up, so no
manual axis flip is applied. Exporting GLB converts the merged scene back to standard Y-up.

Run with Blender, not CPython::

    blender --background --python room_merge.py -- ROOM.usdz OBJECT.glb OUT.blend OUT.glb
"""

from __future__ import annotations

import sys
from pathlib import Path

import bpy


def main() -> int:
    if "--" not in sys.argv:
        raise SystemExit("expected arguments after --")
    args = sys.argv[sys.argv.index("--") + 1 :]
    exclude_room_objects: list[str] = []
    if "--exclude-room-object" in args:
        option_index = args.index("--exclude-room-object")
        if option_index + 1 >= len(args):
            raise SystemExit("--exclude-room-object requires an object name")
        exclude_room_objects.append(args[option_index + 1])
        del args[option_index : option_index + 2]
    if len(args) != 4:
        raise SystemExit(
            "usage: room_merge.py -- ROOM.usdz OBJECT.glb OUT.blend OUT.glb "
            "[--exclude-room-object NAME]"
        )
    room_usdz, object_glb, output_blend, output_glb = map(Path, args)
    for source in (room_usdz, object_glb):
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(source)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.usd_import(filepath=str(room_usdz.resolve()))
    for name in exclude_room_objects:
        matches = [obj for obj in bpy.data.objects if obj.name == name or obj.name.startswith(name)]
        if not matches:
            raise ValueError(f"RoomPlan object {name!r} was not found for debug exclusion")
        for obj in matches:
            bpy.data.objects.remove(obj, do_unlink=True)
        print(f"excluded RoomPlan debug proxy {name}: {len(matches)} Blender object(s)")
    room_objects = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(object_glb.resolve()))
    for obj in set(bpy.data.objects) - room_objects:
        obj["litereality_small_object"] = True
        obj["coordinate_system"] = "ARKit/RoomPlan world; imported from Y-up GLB"
    output_blend.parent.mkdir(parents=True, exist_ok=True)
    output_glb.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_blend.resolve()))
    bpy.ops.export_scene.gltf(filepath=str(output_glb.resolve()), export_format="GLB")
    print(f"wrote {output_blend}")
    print(f"wrote {output_glb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
