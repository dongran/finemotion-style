# Attribution and license scope

## Bandai source and adapted motion

The bundled example is `dataset-2_wave-right-hand_normal_001.bvh` from **Bandai-Namco-Research-Motiondataset-2**, provided by Bandai Namco Research Inc. Original notice: © [2022] Bandai Namco Research Inc. All Rights Reserved.

Source: https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset

The original source, Bandai-skeleton calibration pose, prepared BVH, converted motion parameters, motion features, and motion previews are provided under **CC BY-NC 4.0**, with the original dataset-2 license retained at `licenses/BANDAI_DATASET2_LICENSE.txt`. License: https://creativecommons.org/licenses/by-nc/4.0/

Modifications: a calibration pose was inserted; the motion was retargeted onto a SMPL skeleton using MotionBuilder; rotations were converted to SMPL24 axis-angle; root translations were scaled from centimeters to meters; recorded frame-removal/sampling rules were applied; HumanML263 features and visual previews were produced. Display-only floor offsets in the previews are recorded separately and do not change distributed arrays. The short example caption is a visually reviewed annotation distributed under CC BY-NC 4.0.

Retain attribution, the license, and these modification notices. Noncommercial use only. The source creators do not endorse the conversion, this benchmark, or the method. Original license disclaimers continue to apply.

## Code and documentation

Project scripts and documentation are MIT licensed; see `LICENSE`. The foundational conversion implementation is [tempo-changing-music2motion / JoruriPuppet](https://github.com/dongran/tempo-changing-music2motion), Copyright (c) 2025 Ran Dong, MIT. Its original notice is retained at `licenses/TEMPO_CODE_LICENSE.txt`. The recorded helper revision is `500615821e04b248b0e0f531ec1b8096bb948ef2`.

The HumanML tokenizer reference rule follows [HumanML3D/text_process.py](https://github.com/EricGuo5513/HumanML3D/blob/main/text_process.py), Copyright (c) 2022 Chuan Guo, MIT. Its original notice is retained at `licenses/HUMANML3D_REPOSITORY_LICENSE.txt`.

MotionBuilder/pyfbsdk is Autodesk software and must be obtained separately. Historical video descriptions used [VideoLLaMA3](https://github.com/DAMO-NLP-SG/VideoLLaMA3); language annotations use [spaCy](https://spacy.io/). These dependencies retain their own licenses. They are not bundled or relicensed by this repository.

## SMPL and model assets

Mesh previews use a separately obtained SMPL neutral model with zero shape coefficients. Please credit [SMPL](https://smpl.is.tue.mpg.de/) and its authors; the citation is included in `CITATIONS.bib`. SMPL model weights, shape blend shapes, the characterized target FBX, and target skeleton templates are not distributed here. Obtain or generate suitable target assets under their applicable terms.

Motion parameters, model assets, and software have distinct provenance and license scope. The MIT software license does not override motion-data or model-asset terms.

## Future sources

Other source motions are acquired by users from their original providers. Future text/manifest downloads will retain source-specific notices; this example release does not claim redistribution rights over the full integrated corpus.
