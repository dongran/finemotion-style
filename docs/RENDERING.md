# Render the paired Bandai example

The release preview renders the original Bandai BVH hierarchy on the left and
the converted SMPL24 motion as a neutral SMPL mesh on the right. Both panels use
the same fixed camera, timestamps and metre scale. The animation was generated
from the paired files, not an image-generation model or a manually posed body.

![Original BVH and converted SMPL](../assets/bandai_wave/bandai_bvh_smpl.gif)

The selected example is `dataset-2_wave-right-hand_normal_001`: standing,
raising the right arm, waving, then lowering the arm. The source has 187 frames
at approximately 30 Hz; the supplied SMPL sequence has 186 frames at 30 Hz.
The web preview uses the common duration, 62 frames at 10 Hz. Source and target
frame indices are saved in `assets/bandai_wave/render_manifest.json`.

## Dependencies

Install PyTorch for your platform, then the packages listed in
`scripts/requirements-render.txt`. The tested environment is Python 3.10,
PyTorch 2.10.0 CPU, NumPy 1.23.5, SMPL-X Python package 0.1.28 and pyrender
0.1.45. Linux headless rendering also needs a working EGL or OSMesa installation.

Obtain the SMPL model separately from [the SMPL project](https://smpl.is.tue.mpg.de/)
under its applicable terms. The model files, template FBX and body meshes are
not included in this repository. Point `--smpl-model-path` to a local folder
containing `smpl/SMPL_NEUTRAL.pkl`. The renderer uses the actual SMPL body model
with ten zero shape coefficients; it does not replace the motion with a generated
illustration. SMPL credit: Loper et al., *SMPL: A Skinned Multi-Person Linear Model*,
ACM Transactions on Graphics, 2015.

## Example command

Run from the release root, using your separately obtained SMPL model folder:

```bash
python scripts/render_example.py \
  --bvh examples/bandai_wave/input/dataset-2_wave-right-hand_normal_001.bvh \
  --npz examples/bandai_wave/smpl/dataset-2_wave-right-hand_normal_001Re_30hz.npz \
  --smpl-model-path /path/to/body_models \
  --out-dir generated/bandai_wave_preview \
  --smpl-fps 30 --fps 10 --ground-align --software
```

`--software` selects Linux Mesa surfaceless EGL, evaluated here with
`llvmpipe (LLVM 20.1.2, 256 bits)`. SMPL evaluation also runs on the CPU.
This preview did not use the server GPUs. On another machine with a working
offscreen GPU renderer, omit `--software` and select `--egl-device` if needed.
Use a new output directory for every run: existing renders are not overwritten.

The command writes an MP4, GIF, cover PNG, first-frame PNG, contact sheet and JSON
render manifest. The JSON records input SHA-256 hashes, camera matrix, actual
frame indices, display transforms and the original vertical geometry ranges.

## Reading the comparison correctly

- Bandai six-channel joints contain full local translation values. The default
  `--position-channels replace-offset` matches the historical pipeline; adding
  the hierarchy offset again would distort the skeleton.
- The source and target bodies have different skeleton proportions. Their sizes
  are shown at the same metre scale, without height fitting or heading fitting.
- Each sequence's initial root X/Z position is subtracted once to center its
  panel. This is a fixed display translation, not per-frame root locking;
  later root displacement is preserved. The origins are recorded in the manifest.
- `--ground-align` shifts each displayed sequence vertically by one fixed amount
  determined from its first frame. It makes the ground reference comparable
  between the BVH character and the neutral SMPL model. It does not rewrite the
  NPZ, correct foot sliding, or move the floor on later frames.
- Rendering samples the nearest stored source and target frames. It does not
  interpolate rotations or repair the motion. The common duration omits the
  unmatched source tail.
- Small fingers and the dummy BVH world root are omitted from the skeleton
  drawing. Their channels remain in the distributed original BVH. Standard
  SMPL24 does not reproduce articulated finger motion.

The contact sheet was visually inspected at six times over the clip. It shows
corresponding right-arm movement with no obvious body inversion or gross
self-intersection in those frames. This is a visual sanity check, not a claim
that the retargeting has zero error.

## Data and code attribution

Source motion: [Bandai Namco Research Motion Dataset](https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset),
dataset 2, `dataset-2_wave-right-hand_normal_001.bvh`, CC BY-NC 4.0. The adapted
motion and preview must retain the source credit, noncommercial terms and
modification notice described in `NOTICE.md`.

The MotionBuilder conversion follows our
[JoruriPuppet processing repository](https://github.com/dongran/tempo-changing-music2motion).
The release renderer is newly written glue code: a small BVH parser/FK plus calls
to the [SMPL-X Python implementation](https://github.com/vchoutas/smplx),
[trimesh](https://github.com/mikedh/trimesh),
[pyrender](https://github.com/mmatl/pyrender),
[Pillow](https://python-pillow.org/) and [ImageIO](https://imageio.readthedocs.io/).
The source conversion dependency and its retained notice are documented separately.
