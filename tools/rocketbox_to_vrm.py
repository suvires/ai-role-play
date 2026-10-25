"""Converts a Microsoft Rocketbox *_facial.fbx avatar to VRM 1.0.

Get the avatar's Export/<name>_facial.fbx and its Textures/ folder (next to each other) from
https://github.com/microsoft/Microsoft-Rocketbox/tree/master/Assets/Avatars, then:

    # once: install the VRM add-on (here into a separate Blender profile)
    export BLENDER_USER_RESOURCES=$PWD/.blender-vrm
    blender --online-mode --command extension sync
    blender --online-mode --command extension install vrm --enable

    blender --background --python tools/rocketbox_to_vrm.py -- <in.fbx> <out.vrm> "<avatar name>"

It maps the 3ds Max Biped bones to the VRM humanoid, the Oculus visemes (AA_VI_*) to the
aa/ee/ih/oh/ou expressions and the ARKit blinks to blink, and keeps the model's A-pose
(the web client lowers the arms from any rest pose).
"""
import math
import sys

import bpy

src, out, name = sys.argv[sys.argv.index("--") + 1 :][:3]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.preferences.addon_enable(module="bl_ext.blender_org.vrm")
bpy.ops.import_scene.fbx(filepath=src)

arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
mesh = next(o for o in bpy.data.objects if o.type == "MESH")
# Keep the FBX object transforms (cm scale, Z rotation) as imported: the exporter handles
# them, while applying them with transform_apply collapses the skinned mesh to ~2 cm.

bones = arm.data.bones
head = lambda n: arm.matrix_world @ bones[n].head_local
for side, sign in (("L", 1), ("R", -1)):
    d = (head(f"Bip01 {side} Forearm") - head(f"Bip01 {side} UpperArm")).normalized()
    print(f"[convert] {side} arm direction {tuple(round(v, 2) for v in d)}, "
          f"{math.degrees(math.asin(-d.z)):.1f} deg below horizontal (x sign ok: {d.x * sign > 0})")

# 3ds Max Biped parents the thighs to the spine and the clavicles to the neck;
# VRM needs the legs under the hips and the shoulders under the (upper) chest
print("[convert] parents: thigh ->", bones["Bip01 L Thigh"].parent.name, "| clavicle ->", bones["Bip01 L Clavicle"].parent.name)
bpy.ops.object.select_all(action="DESELECT")
arm.select_set(True)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")
for side in "LR":
    arm.data.edit_bones[f"Bip01 {side} Thigh"].parent = arm.data.edit_bones["Bip01 Pelvis"]
    arm.data.edit_bones[f"Bip01 {side} Clavicle"].parent = arm.data.edit_bones["Bip01 Spine2"]
bpy.ops.object.mode_set(mode="OBJECT")
bones = arm.data.bones

# Drop the specular maps: they point to a missing Windows path and VRM doesn't use them
for mat in mesh.data.materials:
    for node in list(mat.node_tree.nodes):
        if node.type == "TEX_IMAGE" and node.image and "specular" in node.image.filepath:
            mat.node_tree.nodes.remove(node)

# Smaller file: keep only the visemes and the ARKit shapes used by the expressions below,
# and halve the body/opacity textures (the head keeps 2048 px so the face stays sharp)
USED_ARKIT = ("EyeBlink", "MouthSmile", "CheekSquint", "EyeSquint", "MouthFrown", "BrowInnerUp",
              "BrowDown", "NoseSneer", "BrowOuterUp", "EyeWide")
for key in list(mesh.data.shape_keys.key_blocks):
    if key.name != "Basis" and not key.name.startswith("AA_VI_") and not any(s in key.name for s in USED_ARKIT):
        mesh.shape_key_remove(key)
for image in bpy.data.images:
    if image.size[0] > 1024 and "head" not in image.filepath:
        image.scale(1024, 1024)

# The FBX carries vertex colours (3ds Max masks/baked shading, black under the nose and jaw).
# glTF viewers multiply the texture by them, so those areas render pitch black: drop them.
for attribute in list(mesh.data.color_attributes):
    mesh.data.color_attributes.remove(attribute)

# Rocketbox normal maps are DirectX-style (green = down); glTF expects OpenGL (green = up).
# Left as is, the lighting is inverted on the bumps: black patches under the nose and jaw.
for image in {i for i in bpy.data.images if "normal" in i.filepath.lower()}:
    pixels = list(image.pixels)
    pixels[1::4] = [1.0 - g for g in pixels[1::4]]
    image.pixels = pixels
    image.update()
    image.pack()  # export the edited pixels, not the original file on disk
    print(f"[convert] flipped green channel of {image.filepath.split('/')[-1]}")

ext = arm.data.vrm_addon_extension
ext.spec_version = "1.0"
vrm1 = ext.vrm1

# Humanoid: 3ds Max Biped names -> VRM 1.0 bones
human = vrm1.humanoid.human_bones
mapping = {
    "hips": "Pelvis", "spine": "Spine", "chest": "Spine1", "upper_chest": "Spine2",
    "neck": "Neck", "head": "Head", "left_eye": "LEye", "right_eye": "REye",
}
for side, s in (("left", "L"), ("right", "R")):
    mapping |= {
        f"{side}_shoulder": f"{s} Clavicle", f"{side}_upper_arm": f"{s} UpperArm",
        f"{side}_lower_arm": f"{s} Forearm", f"{side}_hand": f"{s} Hand",
        f"{side}_upper_leg": f"{s} Thigh", f"{side}_lower_leg": f"{s} Calf",
        f"{side}_foot": f"{s} Foot", f"{side}_toes": f"{s} Toe0",
        f"{side}_thumb_metacarpal": f"{s} Finger0", f"{side}_thumb_proximal": f"{s} Finger01",
        f"{side}_thumb_distal": f"{s} Finger02",
    }
    for finger, i in (("index", 1), ("middle", 2), ("ring", 3), ("little", 4)):
        for part, j in (("proximal", ""), ("intermediate", "1"), ("distal", "2")):
            mapping[f"{side}_{finger}_{part}"] = f"{s} Finger{i}{j}"
for vrm_bone, biped in mapping.items():
    getattr(human, vrm_bone).node.bone_name = f"Bip01 {biped}"

# Expressions: Oculus visemes and ARKit shapes already in the model
preset = vrm1.expressions.preset
def bind(expression, *shape_keys, weight=1.0):
    for key in shape_keys:
        b = getattr(preset, expression).morph_target_binds.add()
        b.node.mesh_object_name = mesh.name
        b.index = key
        b.weight = weight
bind("aa", "AA_VI_10_aa")
bind("ee", "AA_VI_11_E")
bind("ih", "AA_VI_12_I")
bind("oh", "AA_VI_13_O")
bind("ou", "AA_VI_14_U")
bind("blink_left", "AK_09_EyeBlinkLeft")
bind("blink_right", "AK_10_EyeBlinkRight")
bind("blink", "AK_09_EyeBlinkLeft", "AK_10_EyeBlinkRight")
# Emotions built from ARKit shapes (the client blends them with the visemes)
bind("happy", "AK_44_MouthSmileLeft", "AK_45_MouthSmileRight")
bind("happy", "AK_07_CheekSquintLeft", "AK_08_CheekSquintRight", weight=0.6)
bind("happy", "AK_19_EyeSquintLeft", "AK_20_EyeSquintRight", weight=0.3)
bind("relaxed", "AK_44_MouthSmileLeft", "AK_45_MouthSmileRight", weight=0.4)
bind("relaxed", "AK_19_EyeSquintLeft", "AK_20_EyeSquintRight", weight=0.2)
bind("sad", "AK_30_MouthFrownLeft", "AK_31_MouthFrownRight", weight=0.8)
bind("sad", "AK_03_BrowInnerUp", weight=0.9)
bind("angry", "AK_01_BrowDownLeft", "AK_02_BrowDownRight")
bind("angry", "AK_19_EyeSquintLeft", "AK_20_EyeSquintRight", weight=0.4)
bind("angry", "AK_50_NoseSneerLeft", "AK_51_NoseSneerRight", weight=0.4)
bind("surprised", "AK_03_BrowInnerUp", weight=0.8)
bind("surprised", "AK_04_BrowOuterUpLeft", "AK_05_BrowOuterUpRight")
bind("surprised", "AK_21_EyeWideLeft", "AK_22_EyeWideRight")
vrm1.look_at.type = "bone"
# Export the A-pose rest as is: the add-on's automatic T-pose conversion collapses this mesh
# (it has shape keys). The web client lowers the arms from whatever rest pose the model has.
vrm1.humanoid.pose = vrm1.humanoid.POSE_REST_POSITION_POSE.identifier

meta = vrm1.meta
meta.vrm_name = name
meta.authors.add().value = "Microsoft Rocketbox"
meta.copyright_information = "Copyright (c) 2020 Microsoft. MIT License."
meta.license_url = "https://vrm.dev/licenses/1.0/"
meta.other_license_url = "https://github.com/microsoft/Microsoft-Rocketbox/blob/master/LICENSE.md"
meta.avatar_permission = "everyone"
meta.commercial_usage = "corporation"
meta.credit_notation = "required"  # MIT: keep the copyright notice
meta.allow_redistribution = True
meta.modification = "allowModificationRedistribution"

bpy.ops.object.select_all(action="SELECT")
bpy.context.view_layer.objects.active = arm
from bl_ext.blender_org.vrm.editor.validation import WM_OT_vrm_validator

class _Errors(list):
    def add(self):
        e = type("E", (), {})()
        self.append(e)
        return e

errors = _Errors()
WM_OT_vrm_validator.detect_errors(bpy.context, errors, arm.name)
for e in errors:
    print(f"[convert] validation {'ERROR' if e.severity == 0 else 'warning'}: {e.message}")

result = bpy.ops.export_scene.vrm(filepath=out, ignore_warning=True)
print("[convert] export:", result)
