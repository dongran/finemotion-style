#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MotionBuilder script (run inside Autodesk MotionBuilder) for retargeting
Bandai-Namco-Research-Motiondataset BVH clips to the provided SMPL skeleton.

Key difference vs LaFAN1:
- Bandai BVH uses joint names like UpperLeg_L / Shoulder_L etc.
- We characterize via an explicit HIK slot mapping loaded from hik_mapping_bandai.json.

Workflow (per BVH):
1) Open SMPL T-pose FBX (with a Character already set up; named 'Character')
2) Import BVH clip (optionally with a prepended T-pose first frame)
3) Create & characterize a new Character for the BVH skeleton using mapping json
4) Retarget + bake animation onto SMPL skeleton
5) Export SMPL BVH to: <bvhForC>/output/<clip>Re.bvh
"""

import json
import os

import pyfbsdk
from pyfbsdk import (
    FBApplication,
    FBBodyNodeId,
    FBCharacter,
    FBCharacterInputType,
    FBCharacterPlotWhere,
    FBModelSkeleton,
    FBModelList,
    FBGetSelectedModels,
    FBPlotOptions,
    FBSystem,
)


SCRIPT_DIR = os.path.dirname(__file__)
with open(os.path.join(SCRIPT_DIR, "bandai_config.json"), encoding="utf-8") as _f:
    CONFIG = json.load(_f)


def _config_path(value):
    return os.path.abspath(os.path.join(SCRIPT_DIR, os.path.expanduser(value)))


def message_box(title, message, button="OK"):
    if CONFIG.get("interactive", True):
        return pyfbsdk.FBMessageBox(title, message, button)
    print("%s: %s" % (title, message))


FOLDER_PATH = _config_path(CONFIG["input_directory"])
OUTPUT_DIR_NAME = "output"
MAPPING_JSON = os.path.join(SCRIPT_DIR, "hik_mapping_bandai.json")

# Bandai BVH often has an extra top node: ROOT joint_Root { JOINT Hips { ... } }
# In sampled files this node has all-zero motion, so we can safely remove it.
REMOVE_EXTRA_ROOT = True
EXTRA_ROOT_NAME = "joint_Root"

# 批量处理：留空则处理 bvhForC 下全部 BVH；填入则只处理一个文件名（用于单测）
ONLY_BVH_NAME = CONFIG.get("only_bvh_name", "")

# 如果你想每次跑批量前清空 output 目录，设为 True
CLEAR_OUTPUT_DIR = False

if not CONFIG.get("smpl_fbx"):
    raise RuntimeError("Set smpl_fbx in bandai_config.json to your own characterized target FBX.")
FBX_T_POSE_PATH = _config_path(CONFIG["smpl_fbx"])


def _load_mapping(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Mapping json not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


print(f"DEBUG: FOLDER_PATH = {FOLDER_PATH}")
print(f"DEBUG: FBX_T_POSE_PATH = {FBX_T_POSE_PATH}")
print(f"DEBUG: MAPPING_JSON = {MAPPING_JSON}")

if not os.path.exists(FOLDER_PATH):
    msg = f"[Error] Input folder not found:\n{FOLDER_PATH}\n\nPlease run Step1 to generate bvhForC first."
    print(msg)
    message_box("Script Error", msg, "OK")
    FOLDER_PATH = None

if FBX_T_POSE_PATH is not None and not os.path.exists(FBX_T_POSE_PATH):
    msg = f"[Error] SMPL FBX not found:\n{FBX_T_POSE_PATH}"
    print(msg)
    message_box("Script Error", msg, "OK")
    FBX_T_POSE_PATH = None


def _find_joint_by_name(joint_name, num, file_base=None):
    """
    BVH import often namespaces joints like: 'BVH 1:Hips'
    This helper tries a few common patterns, then falls back to suffix match.
    """
    try:
        if hasattr(pyfbsdk, "FBFindModelByName"):
            obj = pyfbsdk.FBFindModelByName(joint_name)
            if obj:
                return obj
    except Exception:
        pass

    candidates = [
        f"BVH {num}:{joint_name}",
        f"BVH {num}::{joint_name}",
        f"BVH:{num}:{joint_name}",
        f"BVH:{joint_name}",
        joint_name,
    ]
    if file_base:
        candidates.extend(
            [
                f"{file_base}:{joint_name}",
                f"{file_base}::{joint_name}",
                f"BVH {num}:{file_base}:{joint_name}",
                f"BVH {num}:{file_base}::{joint_name}",
            ]
        )

    for label in candidates:
        try:
            obj = pyfbsdk.FBFindModelByLabelName(label)
            if obj:
                return obj
        except Exception:
            pass

    for skel in FBSystem().Scene.ModelSkeletons:
        try:
            name = getattr(skel, "Name", "")
            label = getattr(skel, "LabelName", "")
            long_name = getattr(skel, "LongName", "")

            if name == joint_name or label == joint_name or long_name == joint_name:
                return skel

            for s in (name, label, long_name):
                if not s:
                    continue
                if s.endswith(":" + joint_name) or s.endswith("::" + joint_name):
                    return skel
                if ":" in s and s.split(":")[-1] == joint_name:
                    return skel
        except Exception:
            continue
    return None


def _debug_print_some_skeleton_names(limit=60):
    try:
        skels = list(FBSystem().Scene.ModelSkeletons)
    except Exception:
        return
    print(f"DEBUG: ModelSkeletons count = {len(skels)}")
    for i, skel in enumerate(skels[:limit]):
        try:
            print(
                "DEBUG: skel[%d] Name='%s' Label='%s' LongName='%s'"
                % (
                    i,
                    getattr(skel, "Name", ""),
                    getattr(skel, "LabelName", ""),
                    getattr(skel, "LongName", ""),
                )
            )
        except Exception:
            continue


def _safe_delete_model(model):
    """Best-effort delete for various MotionBuilder versions."""
    if model is None:
        return False
    # Most common
    try:
        if hasattr(model, "FBDelete"):
            model.FBDelete()
            return True
    except Exception:
        pass
    # Some versions expose a module-level function
    try:
        if hasattr(pyfbsdk, "FBDelete"):
            pyfbsdk.FBDelete(model)
            return True
    except Exception:
        pass
    # Fallback: try Delete()
    try:
        if hasattr(model, "Delete"):
            model.Delete()
            return True
    except Exception:
        pass
    return False


def _strip_extra_joint_root(num, file_base=None):
    """
    If the imported BVH has an extra 'joint_Root' above Hips, detach Hips and delete it.

    Note: For Bandai dataset we verified (spot check) that joint_Root's motion channels
    are all zeros, so reparenting does not change the animation.
    """
    root = _find_joint_by_name(EXTRA_ROOT_NAME, num=num, file_base=file_base)
    if root is None:
        return

    hips = _find_joint_by_name("Hips", num=num, file_base=file_base)
    if hips is None:
        print(f"[Warning] Found {EXTRA_ROOT_NAME} but could not find Hips; skip removal.")
        return

    # Detach hips from joint_Root (if needed) before deleting the root node.
    try:
        if getattr(hips, "Parent", None) == root:
            try:
                hips.Parent = None
            except Exception:
                # Some MB versions prefer explicit RootModel
                hips.Parent = FBSystem().Scene.RootModel
            print(f"DEBUG: Detached Hips from {EXTRA_ROOT_NAME}")
    except Exception:
        pass

    deleted = _safe_delete_model(root)
    if deleted:
        print(f"DEBUG: Deleted extra BVH root node: {EXTRA_ROOT_NAME}")
    else:
        print(f"[Warning] Could not delete extra BVH root node: {EXTRA_ROOT_NAME}")


def characterize_bvh_character(character_name, num, file_base=None):
    """Create and characterize a Character for the imported BVH skeleton."""
    mapping = _load_mapping(MAPPING_JSON)
    new_character = FBCharacter(f"{character_name}{num}")

    missing = []
    for slot_name, joint_name in mapping.items():
        # ignore comments in json
        if slot_name.startswith("//"):
            continue
        mapping_slot = new_character.PropertyList.Find(slot_name + "Link")
        if mapping_slot is None:
            missing.append(f"HIK property {slot_name}Link is unavailable")
            continue
        joint_obj = _find_joint_by_name(joint_name, num=num, file_base=file_base)
        if joint_obj:
            mapping_slot.append(joint_obj)
            try:
                print(
                    "DEBUG: mapped slot '%s' -> joint '%s' (Name='%s' Label='%s' LongName='%s')"
                    % (
                        slot_name,
                        joint_name,
                        getattr(joint_obj, "Name", ""),
                        getattr(joint_obj, "LabelName", ""),
                        getattr(joint_obj, "LongName", ""),
                    )
                )
            except Exception:
                pass
        else:
            missing.append(f"{slot_name} -> {joint_name}")

    if missing:
        print(f"[Error] Missing BVH joints for mapping entries:\n- " + "\n- ".join(missing))
        _debug_print_some_skeleton_names(limit=80)
        message_box(
            "Characterize Error",
            "Could not find BVH joints for mapping entries. See console for details.\n\nMissing:\n- "
            + "\n- ".join(missing),
            "OK",
        )
        return new_character, False

    characterized = new_character.SetCharacterizeOn(True)
    if not characterized:
        print(new_character.GetCharacterizeError())
    else:
        FBApplication().CurrentCharacter = new_character
    return new_character, bool(characterized)


def plot_character_to_skeleton(character):
    """Bake the validated target; PlotAnimation returns success in the SDK."""
    if FBApplication().CurrentCharacter != character:
        raise RuntimeError("The current Character is not the validated SMPL target.")

    plot_options = FBPlotOptions()
    plot_options.ConstantKeyReducerKeepOneKey = False
    plot_options.PlotAllTakes = False
    plot_options.PlotOnFrame = True
    plot_options.PlotPeriod = pyfbsdk.FBTime(0, 0, 0, 1)
    plot_options.PlotTranslationOnRootOnly = False
    plot_options.PreciseTimeDiscontinuities = False
    plot_options.RotationFilterToApply = pyfbsdk.FBRotationFilter.kFBRotationFilterUnroll
    plot_options.UseConstantKeyReducer = False

    if not character.PlotAnimation(
        FBCharacterPlotWhere.kFBCharacterPlotOnSkeleton,
        plot_options,
    ):
        raise RuntimeError("SMPL target PlotAnimation failed; no BVH will be exported.")


def find_smpl_target(character_name="Character"):
    """Resolve the target before importing the source, avoiding name ambiguity."""
    matches = [character for character in FBSystem().Scene.Characters
               if character.Name == character_name]
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one target Character named %r; found %d."
                           % (character_name, len(matches)))
    target = matches[0]
    if not target.GetCharacterize():
        raise RuntimeError("Target Character must already be characterized in the SMPL FBX.")
    root = target.GetModel(FBBodyNodeId.kFBHipsNodeId)
    roots = [model for model in FBSystem().Scene.ModelSkeletons
             if model.Name == "Pelvis"]
    if (root is None or not isinstance(root, FBModelSkeleton)
            or root.Name != "Pelvis" or len(roots) != 1 or roots[0] != root):
        raise RuntimeError("Target Hips mapping must identify the unique Pelvis skeleton.")
    return target, root


def activate_smpl_character(target, source):
    """Connect the explicitly resolved SMPL target to the characterized BVH source."""
    if target == source:
        raise RuntimeError("Target Character and source Character must be different.")
    if not target.GetCharacterize() or not source.GetCharacterize():
        raise RuntimeError("Both target and source Characters must be characterized.")
    target.InputCharacter = source
    target.InputType = FBCharacterInputType.kFBCharacterInputCharacter
    target.ActiveInput = True
    FBApplication().CurrentCharacter = target
    FBSystem().Scene.Evaluate()
    print("DEBUG: target='%s', input='%s', type=%s, active=%s, ready=%s, source_matches=%s"
          % (target.LongName,
             target.InputCharacter.LongName if target.InputCharacter else None,
             target.InputType, target.ActiveInput, target.ReadyForRetarget(),
             target.InputCharacter == source))
    # ReadyForRetarget describes the separate Retarget operation. In MotionBuilder
    # 2026 it can remain False for a valid live Character input; this workflow
    # bakes that live input with PlotAnimation and checks its documented result.
    if (target.InputCharacter != source or not target.ActiveInput
            or target.InputType != FBCharacterInputType.kFBCharacterInputCharacter):
        raise RuntimeError("Could not activate the source input on the SMPL target Character.")


def select_descendants(skeleton):
    skeleton.Selected = True
    selected = [skeleton]
    for child in skeleton.Children:
        if isinstance(child, FBModelSkeleton):
            selected.extend(select_descendants(child))
    return selected


def select_target_only(root):
    """Remove imported selections and verify that only target bones are selected."""
    old_selection = FBModelList()
    FBGetSelectedModels(old_selection)
    for model in old_selection:
        model.Selected = False
    expected = select_descendants(root)
    actual = FBModelList()
    FBGetSelectedModels(actual)
    # The query can return generic FBModel wrappers; compare their unique scene names.
    if ({model.LongName for model in actual}
            != {model.LongName for model in expected}):
        raise RuntimeError("Export selection includes objects outside the target skeleton.")
    print("DEBUG: target skeleton models: " + ", ".join(model.Name for model in expected))
    # MotionBuilder imports BVH End Sites as terminal skeleton models. The
    # characterized template has 24 animated joints plus those terminal models;
    # the subsequent canonical BVH finalizer retains the SMPL24 channels.
    animated = [model for model in expected
                if not (model.Name.endswith("_End") and len(model.Children) == 0)]
    canonical_names = [
        "Pelvis", "Left_hip", "Left_knee", "Left_ankle", "Left_foot",
        "Right_hip", "Right_knee", "Right_ankle", "Right_foot",
        "Spine1", "Spine2", "Spine3", "Neck", "Head",
        "Left_collar", "Left_shoulder", "Left_elbow", "Left_wrist", "Left_palm",
        "Right_collar", "Right_shoulder", "Right_elbow", "Right_wrist", "Right_palm",
    ]
    if [model.Name for model in animated] != canonical_names:
        raise RuntimeError("Target skeleton must match the 24-joint canonical SMPL BVH order; "
                           "only leaf _End models may be additional.")


def export_bvh(file_path):
    return FBApplication().FileExport(file_path)


def retarget_and_export(bvh_file_path, out_bvh_path, file_base):
    app = FBApplication()
    app.FileNew()

    if not app.FileOpen(FBX_T_POSE_PATH):
        raise RuntimeError(f"Could not open FBX: {FBX_T_POSE_PATH}")

    target, target_root = find_smpl_target("Character")

    if not app.FileImport(bvh_file_path):
        raise RuntimeError(f"Could not import BVH: {bvh_file_path}")

    if CONFIG.get("bake_fps", 120) != 120:
        raise RuntimeError("This release supports the recorded 120 fps bake; use bake_fps=120.")
    pyfbsdk.FBPlayerControl().SetTransportFps(pyfbsdk.FBTimeMode.kFBTimeMode120Frames)

    if REMOVE_EXTRA_ROOT:
        _strip_extra_joint_root(num=1, file_base=file_base)

    source, characterized = characterize_bvh_character("BVHCharacter", num=1, file_base=file_base)
    if not characterized:
        raise RuntimeError("BVH Characterization failed (mapping incomplete).")

    activate_smpl_character(target, source)
    plot_character_to_skeleton(target)

    # Select SMPL skeleton (Pelvis root) for clean export
    select_target_only(target_root)

    if not export_bvh(out_bvh_path):
        raise RuntimeError(f"Could not export BVH: {out_bvh_path}")
    if not os.path.isfile(out_bvh_path) or os.path.getsize(out_bvh_path) == 0:
        raise RuntimeError("FileExport reported success without a nonempty BVH file.")


def main():
    if FOLDER_PATH is None or FBX_T_POSE_PATH is None:
        print("Script execution aborted due to path errors.")
        return

    output_dir = os.path.join(FOLDER_PATH, OUTPUT_DIR_NAME)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif CLEAR_OUTPUT_DIR:
        try:
            for f in os.listdir(output_dir):
                if f.lower().endswith(".bvh"):
                    os.remove(os.path.join(output_dir, f))
            print(f"DEBUG: Cleared existing BVH files in: {output_dir}")
        except Exception as e:
            print(f"[Warning] Failed to clear output dir: {e}")

    bvh_files = [f for f in os.listdir(FOLDER_PATH) if f.lower().endswith(".bvh")]
    if not bvh_files:
        msg = f"No .bvh files found in:\n{FOLDER_PATH}"
        print(msg)
        message_box("Warning", msg, "OK")
        return

    if ONLY_BVH_NAME:
        bvh_files = [f for f in bvh_files if f == ONLY_BVH_NAME]
        if not bvh_files:
            msg = f"[Error] ONLY_BVH_NAME not found in bvhForC:\n{ONLY_BVH_NAME}"
            print(msg)
            message_box("Warning", msg, "OK")
            return

    success_count = 0
    for fname in bvh_files:
        file_base = os.path.splitext(fname)[0]
        bvh_path = os.path.join(FOLDER_PATH, fname)
        out_bvh_path = os.path.join(output_dir, file_base + "Re.bvh")
        if os.path.exists(out_bvh_path):
            raise RuntimeError("Refusing to overwrite existing output: " + out_bvh_path)
        try:
            print(f"Processing: {fname}")
            retarget_and_export(bvh_path, out_bvh_path, file_base=file_base)
            print(f"Exported: {out_bvh_path}")
            success_count += 1
        except Exception as e:
            print(f"[Error] {fname}: {e}")

    if success_count != len(bvh_files):
        raise RuntimeError("Retargeting failed for one or more clips; see the preceding error.")
    message_box(
        "Process Completed",
        f"Successfully processed {success_count}/{len(bvh_files)} files.\nCheck output folder:\n{output_dir}",
        "OK",
    )


main()
