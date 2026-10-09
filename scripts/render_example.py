#!/usr/bin/env python3
"""Render genuine source BVH and converted SMPL24 motion with a shared camera.

Requires numpy, torch, smplx, trimesh, pyrender, Pillow, imageio, imageio-ffmpeg.
SMPL model files must be obtained separately under their own licence.
No hand-pose synthesis, interpolation of axis angles, or pose correction is used.
The small BVH parser/FK below is original release code, not a vendored loader.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
from importlib.metadata import version
import io
import json
import math
import os
from pathlib import Path
import re

import numpy as np


def load_bvh(path):
    """Parse BVH hierarchy/channels, including End Sites, then apply ordered FK."""
    text = Path(path).read_text(encoding="utf-8-sig")
    hierarchy, motion = text.split("MOTION", 1)
    tokens = iter(re.findall(r"\{|\}|[^\s{}]+", hierarchy))
    names, parents, offsets, channels = [], [], [], []
    channel_count = 0

    def expect(value):
        found = next(tokens)
        if found != value:
            raise ValueError(f"BVH: expected {value}, got {found}")

    def joint(kind, parent):
        nonlocal channel_count
        name = next(tokens) if kind != "End" else f"{names[parent]}_End"
        if kind == "End":
            expect("Site")
        index = len(names)
        names.append(name)
        parents.append(parent)
        offsets.append([0., 0., 0.])
        channels.append([])
        expect("{")
        while True:
            token = next(tokens)
            if token == "}":
                break
            if token == "OFFSET":
                offsets[index] = [float(next(tokens)) for _ in range(3)]
            elif token == "CHANNELS":
                n = int(next(tokens))
                channels[index] = [(next(tokens), channel_count + c) for c in range(n)]
                channel_count += n
            elif token in ("JOINT", "End"):
                joint(token, index)
            else:
                raise ValueError(f"Unexpected BVH hierarchy token: {token}")

    expect("HIERARCHY")
    expect("ROOT")
    joint("ROOT", -1)
    lines = motion.strip().splitlines()
    frame_count = int(lines[0].split(":")[1])
    frame_time = float(lines[1].split(":")[1])
    values = np.loadtxt(io.StringIO("\n".join(lines[2:])), ndmin=2)
    if values.shape != (frame_count, channel_count):
        raise ValueError(f"BVH channel/frame mismatch: {values.shape}")
    return values, 1 / frame_time, names, np.asarray(parents), np.asarray(offsets), channels


def bvh_positions(values, parents, offsets, channels, position_channels="replace-offset"):
    frames, joints = len(values), len(parents)
    matrices = np.repeat(np.eye(4)[None, None], frames * joints, axis=0).reshape(frames, joints, 4, 4)
    for j in range(joints):
        local = np.repeat(np.eye(4)[None], frames, axis=0)
        local[:, :3, 3] = offsets[j]
        for name, index in channels[j]:
            op = np.repeat(np.eye(4)[None], frames, axis=0)
            axis = "XYZ".index(name[0].upper())
            if name.lower().endswith("position"):
                # Bandai's 6-channel joints store full local translations,
                # including the rest offset, matching the historical loader.
                if position_channels == "replace-offset":
                    local[:, axis, 3] = values[:, index]
                    continue
                op[:, axis, 3] = values[:, index]
            elif name.lower().endswith("rotation"):
                angle = np.deg2rad(values[:, index])
                a, b = (axis + 1) % 3, (axis + 2) % 3
                op[:, a, a] = op[:, b, b] = np.cos(angle)
                op[:, a, b], op[:, b, a] = -np.sin(angle), np.sin(angle)
            else:
                raise ValueError(f"Unsupported BVH channel: {name}")
            local = local @ op
        matrices[:, j] = local if parents[j] < 0 else matrices[:, parents[j]] @ local
    return matrices[:, :, :3, 3]


def look_at(eye, target):
    z = np.asarray(eye) - target
    z /= np.linalg.norm(z)
    x = np.cross([0., 1., 0.], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([x, y, z], axis=1)
    pose[:3, 3] = eye
    return pose


def skeleton_mesh(points, parents, names, trimesh):
    meshes = []
    for j, parent in enumerate(parents):
        if parent < 0 or names[parent].lower() == "joint_root" or any(s in names[j].lower() for s in ("thumb", "index", "middle", "ring", "pinky")):
            continue
        a, b = points[parent], points[j]
        if np.linalg.norm(b - a) < 1e-6:
            continue
        meshes.append(trimesh.creation.cylinder(radius=.012, sections=8, segment=np.stack([a, b])))
    for j, point in enumerate(points):
        if any(s in names[j].lower() for s in ("thumb", "index", "middle", "ring", "pinky", "_end", "joint_root")):
            continue
        sphere = trimesh.creation.icosphere(subdivisions=1, radius=.022)
        sphere.apply_translation(point)
        meshes.append(sphere)
    return trimesh.util.concatenate(meshes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bvh", type=Path, required=True)
    p.add_argument("--npz", type=Path, required=True)
    p.add_argument("--smpl-model-path", type=Path, required=True, help="Local folder with smpl/SMPL_NEUTRAL.pkl; not published")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--smpl-fps", type=float, default=30.)
    p.add_argument("--source-offset-seconds", type=float, default=0., help="BVH time corresponding to SMPL frame 0; recorded explicitly")
    p.add_argument("--bvh-scale", type=float, default=.01, help="Source units to metres")
    p.add_argument("--position-channels", choices=["replace-offset", "add-offset"], default="replace-offset",
        help="Bandai 6-channel translations already include local rest offsets; use add-offset only when appropriate for a different BVH")
    p.add_argument("--fps", type=float, default=15.)
    p.add_argument("--width", type=int, default=480, help="Per-panel width")
    p.add_argument("--height", type=int, default=480, help="Per-panel height")
    p.add_argument("--azimuth", type=float, default=150.)
    p.add_argument("--elevation", type=float, default=9.)
    p.add_argument("--max-seconds", type=float, default=0., help="0 renders the common duration")
    p.add_argument("--platform", choices=["egl", "osmesa"], default="egl")
    p.add_argument("--egl-device", type=int, default=0)
    p.add_argument("--software", action="store_true", help="Force Mesa software EGL vendor instead of a GPU")
    p.add_argument("--ground-align", action="store_true", help="Fixed display-only Y shifts put first-frame feet on the common ground; never alters the NPZ")
    args = p.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError("Use a new empty output directory to preserve prior renders")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PYOPENGL_PLATFORM"] = args.platform
    os.environ["EGL_DEVICE_ID"] = str(args.egl_device)
    if args.software:
        os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/usr/share/glvnd/egl_vendor.d/50_mesa.json"
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
        os.environ["EGL_PLATFORM"] = "surfaceless"
    # Compatibility for legacy Chumpy inside user-provided SMPL pickles.
    for name, value in {"bool": bool, "int": int, "float": float, "complex": complex, "object": object, "str": str, "unicode": str}.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    import torch
    import smplx
    import trimesh
    import pyrender
    if args.software:
        # pyrender 0.1.45 selects an EGL hardware device even under Mesa.
        # Select the surfaceless default display for llvmpipe instead.
        from pyrender.platforms import egl
        egl.get_device_by_index = lambda _: egl.EGLDevice(None)
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    torch.set_num_threads(4)
    values, source_fps, names, parents, offsets, channels = load_bvh(args.bvh)
    source = bvh_positions(values, parents, offsets, channels, args.position_channels) * args.bvh_scale
    data = np.load(args.npz, allow_pickle=False)
    poses, trans = data["poses"], data["trans"]
    if poses.shape != (len(trans), 24, 3) or trans.shape[1:] != (3,):
        raise ValueError("Expected SMPL24 poses (T,24,3) and trans (T,3)")
    if not np.isfinite(poses).all() or not np.isfinite(trans).all():
        raise ValueError("Nonfinite motion")
    duration = min((len(source) - 1) / source_fps - args.source_offset_seconds, (len(poses) - 1) / args.smpl_fps)
    if args.max_seconds > 0:
        duration = min(duration, args.max_seconds)
    if duration <= 0:
        raise ValueError("No common time range")
    times = np.arange(0, duration + 1e-7, 1 / args.fps)
    source_ids = np.clip(np.rint((times + args.source_offset_seconds) * source_fps).astype(int), 0, len(source) - 1)
    smpl_ids = np.clip(np.rint(times * args.smpl_fps).astype(int), 0, len(poses) - 1)
    source = source[source_ids]
    # Root translation only: independent fixed X/Z origins; no heading/scale/pose fitting.
    hip_index = names.index("Hips") if "Hips" in names else 0
    source_origin = source[0, hip_index].copy()
    source_origin[1] = 0
    source -= source_origin
    target_origin = trans[smpl_ids[0]].copy()
    target_origin[1] = 0
    translation = trans[smpl_ids].astype(np.float32) - target_origin
    body = smplx.create(str(args.smpl_model_path), model_type="smpl", gender="neutral", use_pca=False).to("cpu")
    vertices, joints = [], []
    betas = np.asarray(data["betas"], dtype=np.float32).reshape(-1)[:10] if "betas" in data else np.zeros(10, dtype=np.float32)
    if len(betas) != 10:
        raise ValueError("Expected 10 shape coefficients")
    with torch.inference_mode():
        for start in range(0, len(smpl_ids), 32):
            ids = smpl_ids[start:start + 32]
            pose = torch.as_tensor(poses[ids], dtype=torch.float32)
            out = body(global_orient=pose[:, 0], body_pose=pose[:, 1:].reshape(len(ids), 69),
                transl=torch.as_tensor(translation[start:start + len(ids)]), betas=torch.as_tensor(betas).repeat(len(ids), 1))
            vertices.append(out.vertices.numpy())
            joints.append(out.joints[:, :24].numpy())
    vertices, joints = np.concatenate(vertices), np.concatenate(joints)
    source_body_indices = [j for j, name in enumerate(names) if name.lower() != "joint_root"]
    source_y_before = [float(source[:, source_body_indices, 1].min()), float(source[:, source_body_indices, 1].max())]
    mesh_y_before = [float(vertices[:, :, 1].min()), float(vertices[:, :, 1].max())]
    ankle_y_before = [[float(joints[:, j, 1].min()), float(joints[:, j, 1].max())] for j in (7, 8)]
    source_hips_y_before = [float(source[:, hip_index, 1].min()), float(source[:, hip_index, 1].max())]
    source_ground_shift = target_ground_shift = 0.
    if args.ground_align:
        source_ground_shift = -float(source[0, source_body_indices, 1].min())
        target_ground_shift = -float(vertices[0, :, 1].min())
        source[:, :, 1] += source_ground_shift
        vertices[:, :, 1] += target_ground_shift
        joints[:, :, 1] += target_ground_shift
    bounds = np.stack([np.minimum(source[:, source_body_indices].min((0, 1)), vertices.min((0, 1))), np.maximum(source[:, source_body_indices].max((0, 1)), vertices.max((0, 1)))])
    floor_y = float(bounds[0, 1]) - .012
    target = (bounds[0] + bounds[1]) / 2
    radius = np.linalg.norm(bounds[1] - bounds[0]) / 2
    camera_distance = max(3., radius / math.sin(math.radians(22.5)) * 1.15)
    azimuth, elevation = np.deg2rad([args.azimuth, args.elevation])
    eye = target + camera_distance * np.array([np.sin(azimuth) * np.cos(elevation), np.sin(elevation), np.cos(azimuth) * np.cos(elevation)])
    pose = look_at(eye, target)
    scene = pyrender.Scene(bg_color=np.array([248, 249, 251, 255]), ambient_light=[.48, .48, .48])
    scene.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(45)), pose=pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.8), pose=pose)
    floor = trimesh.creation.box(extents=[6, .006, 6])
    floor.apply_translation([0, floor_y, 0])
    floor.visual.face_colors = [233, 235, 239, 255]
    scene.add(pyrender.Mesh.from_trimesh(floor, smooth=False))
    renderer = pyrender.OffscreenRenderer(args.width, args.height)
    renderer._platform.make_current()
    from OpenGL.GL import glGetString, GL_RENDERER
    gl_renderer = glGetString(GL_RENDERER).decode("utf-8")
    material_source = pyrender.MetallicRoughnessMaterial(baseColorFactor=[.14, .31, .52, 1], roughnessFactor=.8)
    material_target = pyrender.MetallicRoughnessMaterial(baseColorFactor=[.82, .40, .15, 1], roughnessFactor=.85)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 17)
        small = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = small = ImageFont.load_default()
    frames = []
    writer = imageio.get_writer(args.out_dir / "bandai_bvh_smpl.mp4", fps=args.fps, codec="libx264", quality=8, macro_block_size=1)
    try:
        for i, time in enumerate(times):
            panels = []
            for mesh, material in [(skeleton_mesh(source[i], parents, names, trimesh), material_source), (trimesh.Trimesh(vertices=vertices[i], faces=body.faces, process=False), material_target)]:
                node = scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True))
                image, _ = renderer.render(scene)
                panels.append(Image.fromarray(image))
                scene.remove_node(node)
            canvas = Image.new("RGB", (args.width * 2, args.height + 64), "#f8f9fb")
            canvas.paste(panels[0], (0, 48))
            canvas.paste(panels[1], (args.width, 48))
            draw = ImageDraw.Draw(canvas)
            draw.text((18, 12), "Source BVH · Bandai Namco", fill="#24364c", font=font)
            draw.text((args.width + 18, 12), "Converted SMPL · MotionBuilder", fill="#5e3824", font=font)
            draw.line((args.width, 0, args.width, args.height + 64), fill="#d4d8df")
            display_note = "same scale; fixed camera; display ground aligned" if args.ground_align else "same scale and fixed camera; original vertical origins"
            draw.text((18, args.height + 48), f"t = {time:0.2f} s   |   {display_note}", fill="#5e6671", font=small)
            writer.append_data(np.asarray(canvas))
            frames.append(canvas)
            if i % 20 == 0:
                print(f"Rendered {i + 1}/{len(times)}", flush=True)
    finally:
        writer.close()
        renderer.delete()
    frames[0].save(args.out_dir / "first_frame.png")
    # Cover at maximum wrist reach from hips (real frame, no posed illustration).
    hand_indices = [j for j, name in enumerate(names) if name.lower() in ("lefthand", "righthand", "hand_l", "hand_r")]
    cover = int(np.linalg.norm(source[:, hand_indices] - source[:, hip_index:hip_index + 1], axis=-1).max(1).argmax()) if hand_indices else len(frames) // 2
    frames[cover].save(args.out_dir / "bandai_bvh_smpl.png")
    selected = np.linspace(0, len(frames) - 1, 6).round().astype(int)
    sheet = Image.new("RGB", (args.width * 4, (args.height + 64) * 3), "white")
    for i, index in enumerate(selected):
        sheet.paste(frames[index], ((i % 2) * args.width * 2, (i // 2) * (args.height + 64)))
    sheet.save(args.out_dir / "bandai_contact_sheet.png")
    # Shared GIF palette avoids per-frame colour shimmer, timing preserves motion speed.
    gif_stride = max(1, int(round(args.fps / 10)))
    gif_frames = frames[::gif_stride]
    palette = Image.new("RGB", (args.width * 2, (args.height + 64) * len(selected)))
    for i, index in enumerate(selected):
        palette.paste(frames[index], (0, i * (args.height + 64)))
    palette = palette.quantize(colors=128)
    quantized = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in gif_frames]
    quantized[0].save(args.out_dir / "bandai_bvh_smpl.gif", save_all=True, append_images=quantized[1:],
        duration=int(round(1000 * gif_stride / args.fps)), loop=0, optimize=False)
    record = {"render_script_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "source_bvh": args.bvh.name, "source_sha256": hashlib.sha256(args.bvh.read_bytes()).hexdigest(),
        "smpl_npz": args.npz.name, "smpl_sha256": hashlib.sha256(args.npz.read_bytes()).hexdigest(),
        "source_fps": source_fps, "smpl_fps": args.smpl_fps, "source_offset_seconds": args.source_offset_seconds,
        "bvh_scale_to_metres": args.bvh_scale, "bvh_position_channels": args.position_channels, "render_fps": args.fps, "frames": len(times), "common_duration_seconds": duration,
        "viewport_per_panel_pixels": [args.width, args.height], "combined_frame_pixels": [args.width * 2, args.height + 64],
        "source_frame_indices": source_ids.tolist(), "smpl_frame_indices": smpl_ids.tolist(),
        "gif_frame_stride": gif_stride, "gif_frame_duration_ms": int(round(1000 * gif_stride / args.fps)),
        "camera": {"azimuth_degrees": args.azimuth, "elevation_degrees": args.elevation, "distance_metres": camera_distance,
            "target": target.tolist(), "shared_matrix": pose.tolist(), "fixed_for_all_frames": True},
        "display_translation_only": {"source_xz_origin": source_origin.tolist(), "smpl_xz_origin": target_origin.tolist(),
            "initial_ground_alignment_requested": args.ground_align, "source_y_shift_metres": source_ground_shift, "smpl_y_shift_metres": target_ground_shift},
        "ground_y_metres": floor_y, "cover_time_seconds": float(times[cover]), "contact_times_seconds": times[selected].tolist(),
        "smpl_model": {"type": "SMPL", "gender": "neutral", "betas": betas.tolist(), "model_assets_included": False,
            "credit": "Loper et al., SMPL: A Skinned Multi-Person Linear Model, ACM TOG 2015", "website": "https://smpl.is.tue.mpg.de/"},
        "notes": ["CPU SMPL evaluation; offscreen OpenGL renderer", "No synthetic fist/hand posing, motion repair, heading alignment, or dynamic camera", "Source and target skeleton proportions differ; no size fitting is applied", "Ground alignment, when requested, is a fixed first-frame display translation, not a motion-data correction", "BVH finger channels remain in input; small fingers and dummy joint_Root omitted only from skeleton drawing", "Nearest stored frames used; no motion interpolation", "This visualization is not an accuracy certificate"],
        "render_backend": {"platform": args.platform, "egl_device": args.egl_device, "software_requested": args.software, "gl_renderer": gl_renderer},
        "geometry_diagnostics_before_display_ground_alignment": {"source_y_range_metres": source_y_before, "mesh_y_range_metres": mesh_y_before,
            "left_ankle_y_range_metres": ankle_y_before[0], "right_ankle_y_range_metres": ankle_y_before[1],
            "left_foot_horizontal_span_metres": np.ptp(joints[:, 10][:, [0, 2]], axis=0).tolist(),
            "right_foot_horizontal_span_metres": np.ptp(joints[:, 11][:, [0, 2]], axis=0).tolist(),
            "source_hips_y_range_metres": source_hips_y_before},
        "versions": {"numpy": np.__version__, "torch": torch.__version__, "smplx": version("smplx"), "trimesh": trimesh.__version__, "pyrender": pyrender.__version__,
            "Pillow": version("Pillow"), "imageio": version("imageio"), "imageio-ffmpeg": version("imageio-ffmpeg")}}
    (args.out_dir / "render_manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_dir}: {len(frames)} frames, cover t={times[cover]:.2f}s")


if __name__ == "__main__":
    main()
