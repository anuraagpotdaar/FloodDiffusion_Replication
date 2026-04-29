"""
One-shot converter: SMPL Blender add-on `.blend` → `.glb` for the front-end.

The SMPL Blender add-on (static/models/smpl_blender_addon) ships a `.blend`
file containing two rigged meshes: SMPL-mesh-male (+ armature SMPL-male) and
SMPL-mesh-female (+ armature SMPL-female). This script extracts the male mesh
in its rest (T-)pose with shape keys baked, and exports it as a glTF binary
the JS front-end can rig from joint streams.

Usage (two ways):

  # 1) bpy as a pip package (no Blender app install needed)
  python -m pip install bpy
  python FloodDiffusion/tools/export_smpl_glb.py

  # 2) headless Blender from a system install (e.g. brew install --cask blender)
  /Applications/Blender.app/Contents/MacOS/Blender \
      --background --python FloodDiffusion/tools/export_smpl_glb.py

Output: FloodDiffusion/static/models/smpl_neutral.glb
"""

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(THIS_DIR)
BLEND_PATH = os.path.join(
    PROJECT_DIR, "static", "models", "smpl_blender_addon",
    "data", "smpl-model-20200803.blend",
)
OUT_PATH = os.path.join(PROJECT_DIR, "static", "models", "smpl_neutral.glb")
GENDER = "male"  # Add-on ships only male/female; "male" matches the reference figure.

try:
    import bpy
except ImportError:
    sys.stderr.write(
        "ERROR: this script must run inside Blender's Python.\n"
        "  Easiest fix:    python -m pip install bpy && python " + __file__ + "\n"
        "  Alternative:    brew install --cask blender, then run\n"
        "    /Applications/Blender.app/Contents/MacOS/Blender --background \\\n"
        "      --python " + __file__ + "\n"
    )
    sys.exit(1)


def main():
    if not os.path.exists(BLEND_PATH):
        sys.stderr.write(f"ERROR: source .blend not found at {BLEND_PATH}\n")
        sys.exit(2)

    print(f"Opening {BLEND_PATH}")
    bpy.ops.wm.open_mainfile(filepath=BLEND_PATH)

    mesh_name = f"SMPL-mesh-{GENDER}"
    arm_name = f"SMPL-{GENDER}"
    if mesh_name not in bpy.data.objects or arm_name not in bpy.data.objects:
        available = [o.name for o in bpy.data.objects]
        sys.stderr.write(
            f"ERROR: expected objects '{mesh_name}' and '{arm_name}' not found.\n"
            f"Objects in scene: {available}\n"
        )
        sys.exit(3)

    mesh = bpy.data.objects[mesh_name]
    arm = bpy.data.objects[arm_name]

    # Drop the other gender so it's not bundled into the export
    other = "female" if GENDER == "male" else "male"
    for n in (f"SMPL-mesh-{other}", f"SMPL-{other}"):
        if n in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[n], do_unlink=True)

    # Select mesh + armature for export
    bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm

    print(f"Exporting to {OUT_PATH}")
    bpy.ops.export_scene.gltf(
        filepath=OUT_PATH,
        export_format="GLB",
        use_selection=True,
        export_apply=True,           # apply modifiers
        export_skins=True,
        export_morph=False,          # skip blend shapes for runtime simplicity
        export_animations=False,
        export_yup=True,
    )

    if os.path.exists(OUT_PATH):
        size = os.path.getsize(OUT_PATH) / (1024 * 1024)
        print(f"OK: wrote {OUT_PATH} ({size:.2f} MB)")
    else:
        sys.stderr.write("ERROR: exporter returned but file is missing\n")
        sys.exit(4)


main()
