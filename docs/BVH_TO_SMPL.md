# Reproduce the Bandai BVH to SMPL example

This walkthrough uses one clip, `dataset-2_wave-right-hand_normal_001.bvh`, from
[Bandai Namco Research Motion Dataset 2](https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset).
It extends our [JoruriPuppet / SIGGRAPH Asia 2025 conversion workflow](https://github.com/dongran/tempo-changing-music2motion)
with a Bandai-specific calibration pose and HIK mapping. The original source is
187 frames at approximately 30Hz. The distributed SMPL parameters are archived
results; the current scripts were also run through a fresh conversion in
**MotionBuilder 2026**.

## Pipeline at a glance

1. **Source preparation** — original Bandai BVH + source T-pose → calibrated BVH (**187 → 188 frames, 30Hz**).
2. **MotionBuilder retargeting** — source Character → SMPL target Character → baked target BVH (**749 frames, 120Hz**).
3. **SMPL24 conversion** — canonical target BVH → axis-angle rotations + translations in meters (**748 frames, 120Hz**).
4. **Release sampling** — every fourth frame, then remove the first sampled frame → SMPL NPZ (**186 frames, 30Hz**).

The downstream files pair a **20Hz HumanML263 motion feature** with a text file
of the same motion ID. Text preparation follows video/source-label description
→ visual review → spaCy lemma/POS → `caption#tokens#start#end`.

## Dependencies and target assets

Use Python with NumPy for preparation and finalization, and a licensed Autodesk
MotionBuilder installation for retargeting. The fresh run used MotionBuilder
2026 on Windows, its bundled Python 3.11.9, and Python 3.10/NumPy 1.23.5 for
finalization. spaCy and SMPL rendering dependencies are separate; see
[text preparation](TEXT_FORMAT.md) and [rendering](RENDERING.md).

Clone the 2025 processing code at the recorded revision:

```bash
git clone https://github.com/dongran/tempo-changing-music2motion.git
git -C tempo-changing-music2motion checkout 500615821e04b248b0e0f531ec1b8096bb948ef2
python -m pip install numpy==1.23.5
```

The entry points check the preparation/finalizer source hashes. This revision
contains the source-template calibration logic used here; a later upstream
revision reverted that logic. Keep the accompanying MIT notice.

Obtain or create a compatible **characterized SMPL target FBX** and a matching
**SMPL T-pose reference BVH**, respecting their model-asset terms. They are not
bundled here. The tested 2025 workflow uses `smpl-male-T-pose.fbx` and
`smpl-T.bvh`. The FBX must contain one characterized Character named `Character`,
with its Hips link pointing to `Pelvis` and the canonical 24-joint target
hierarchy. Arbitrary FBX rigs are not interchangeable with this target.

## 1. Prepare the source calibration frame

Run from this repository's root. Use a fresh working directory:

```bash
python scripts/bandai/prepare_bandai.py --bvh examples/bandai_wave/input/dataset-2_wave-right-hand_normal_001.bvh --tpose examples/bandai_wave/input/T-pose-bandai.bvh --tempo-repo tempo-changing-music2motion --work-dir work/bandai
```

This produces `work/bandai/bvhForC/dataset-2_wave-right-hand_normal_001.bvh`.
The included T-pose has the **Bandai source skeleton**, not the target skeleton.
The historical helper preserves source position channels, initializes rotations
from that pose, and zeros the upper-arm calibration rotations. One frame is
prepended: 187 → 188. Its historical numeric text formatting rounds some source
channels; it is not a byte-preserving rewrite. The bundled `_for_mb.bvh` is the
archived prepared input for comparison.

## 2. Characterize, retarget and bake in MotionBuilder

Edit `scripts/bandai/bandai_config.json`. Set `smpl_fbx` to your compatible target
FBX. `input_directory` defaults to `../../work/bandai/bvhForC`; relative paths
are resolved against the script directory. Absolute paths are also accepted.
Keep `bake_fps: 120` and the single selected source filename for this example.

In a fresh MotionBuilder session, execute
`scripts/bandai/PuppetToSmpl_Bandai.py` in the Python Editor. The script opens a
new scene for each clip, so save any open scene before running it. It performs
the following operations:

- Open the target FBX and resolve its Character and `Pelvis` before source import.
- Import the calibrated BVH, remove its inert `joint_Root`, and characterize
  the source using `hik_mapping_bandai.json` (for example, HIK `RightArm` links
  to source `UpperArm_R`). The selected clip's dummy-root channels are zero;
  recheck this assumption before adapting the script to other clips.
- Set the target's Character input to the source, enable live input, and bake
  the target skeleton with `PlotAnimation` at an explicit 120Hz transport rate.
- Verify source/target characterization, active input and the canonical target
  joint order; select only the target skeleton and export a nonempty BVH.

The result is
`work/bandai/bvhForC/output/dataset-2_wave-right-hand_normal_001Re.bvh`.
Existing outputs are not overwritten. In the verified 2026 run this export has
**749 frames** and a frame time of approximately `0.00833333` seconds.

For a batch run, set `interactive: false`, then use MotionBuilder's
[documented command-line script entry](https://help.autodesk.com/cloudhelp/2023/ENU/MotionBuilder-SDK/Getting-Started/MotionBuilder-Command-Line.html):

```text
motionbuilder.exe -batch C:/finemotion/scripts/bandai/PuppetToSmpl_Bandai.py
```

Use simple ASCII paths for the batch entry script and its working assets, as in
the tested run. Check its log and exported file before continuing; a successful
application launch alone does not establish successful retargeting.

## 3. Convert the target BVH to SMPL24 parameters

Use the same target reference hierarchy as the FBX:

```bash
python scripts/bandai/finalize_bandai.py --tempo-repo tempo-changing-music2motion --work-dir work/bandai --reference-bvh /path/to/smpl-T.bvh
```

The original finalizer standardizes the target hierarchy, removes the first
120Hz export frame, and writes `bvhSMPL/` and `npz/` beneath the work directory.
749 → **748 frames**. It uses the recorded **zxy** rotation conversion, then
reorders joints into SMPL24 order. Translations are multiplied by `0.01` to
convert centimeters to meters. This is the historical converter; it is not a
general-purpose automatic BVH Euler-order detector.

The NPZ contains `poses(T,24,3)` in axis-angle radians and `trans(T,3)` in meters.
The archived 120Hz NPZ has no FPS field; its rate is recorded in the manifest.
Shapes and units should always be checked before sending these arrays to a
body-model or training pipeline.

## 4. Apply the recorded 30Hz sampling rule

```bash
python scripts/bandai/to_30hz.py --input work/bandai/npz/dataset-2_wave-right-hand_normal_001Re.npz --output work/bandai/npz30/dataset-2_wave-right-hand_normal_001Re_30hz.npz
```

For both arrays the rule is `raw[::4][1:].astype(float32)`: sample every fourth
120Hz frame, then discard one sampled frame. 748 → 187 → **186 frames**. The
result also contains `mocap_framerate=30`. These frame-removal rules preserve
the historical example; do not apply them indiscriminately to other datasets
or repeatedly strip calibration frames.

## Validation and downstream training files

```bash
python scripts/verify_bandai_example.py
```

This checks the bundled file hashes, finite motion/features, shapes, text
pairing, and the exact archived 120→30Hz array relationship. The
[example manifest](../examples/bandai_wave/manifest.json) records the source,
frame history, annotation provenance and fresh-run validation.

The fresh MotionBuilder 2026 run recovered the same frame counts and finite
arrays. At 30Hz, the maximum absolute parameter differences from the archived
arrays were `0.00159168` radians for axis-angle components and `0.000140011`
meters for translation components. These are component differences, not a
body-surface accuracy metric. A second run of the final script produced the
same fresh BVH hash. Archived files remain the distributed reference; a new
MotionBuilder/model-template setup need not be byte-identical.

The bundled downstream feature is **123×263 at 20Hz**, with its matching text
under `training/`. It is an archived HumanML feature, not newly regenerated by
the four steps above. Feature construction additionally requires the method
project's joint/body-model processing, normalizations and dependencies. The
training loader uses feature-frame indices, not the SMPL NPZ's 30Hz indices.
This single example demonstrates the motion/text layout; it is not a complete
training dataset with splits, Mean/Std and model/vocabulary assets.

For captions, the process is motion video/source label → visually reviewed
sentence → spaCy lemma/POS → four-field text. The [text guide](TEXT_FORMAT.md)
provides the actual formatting rules and commands. VLM-generated sentences
still need visual review; POS tagging does not check what happened in the video.

Source motion and its adaptations are CC BY-NC 4.0. Conversion glue code is MIT;
target/model assets retain their independent terms. See [NOTICE.md](../NOTICE.md)
and [CITATIONS.bib](../CITATIONS.bib) for attribution to Bandai, the 2025
conversion work, HumanML3D and SMPL.
