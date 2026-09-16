"""Blender worker that removes debug proxies from an existing optimized room scene."""

from __future__ import annotations

import sys
from pathlib import Path

import bpy


def main() -> int:
    if "--" not in sys.argv:
        raise SystemExit("expected arguments after --")
    args = sys.argv[sys.argv.index("--") + 1 :]
    if len(args) < 3:
        raise SystemExit("usage: visible_scene_export.py -- IN.blend OUT.blend OUT.glb [NAME ...]")
    input_blend, output_blend, output_glb = map(Path, args[:3])
    exclusions = args[3:]
    bpy.ops.wm.open_mainfile(filepath=str(input_blend.resolve()))
    for name in exclusions:
        matches = [obj for obj in bpy.data.objects if obj.name == name or obj.name.startswith(name)]
        if not matches:
            raise ValueError(f"RoomPlan object {name!r} was not found for debug exclusion")
        for obj in matches:
            bpy.data.objects.remove(obj, do_unlink=True)
        print(f"excluded RoomPlan debug proxy {name}: {len(matches)} Blender object(s)")
    output_blend.parent.mkdir(parents=True, exist_ok=True)
    output_glb.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_blend.resolve()))
    bpy.ops.export_scene.gltf(
        filepath=str(output_glb.resolve()),
        export_format="GLB",
        export_draco_mesh_compression_enable=True,
    )
    print(f"wrote {output_blend}")
    print(f"wrote {output_glb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
