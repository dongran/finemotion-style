#!/usr/bin/env python3
"""Validate the bundled motion/text example against its recorded manifest."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", type=Path,
        default=Path(__file__).resolve().parents[1] / "examples/bandai_wave")
    args = parser.parse_args()
    root = args.example.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = (root / item["path"]).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError("Missing or out-of-example path: " + item["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("SHA256 mismatch: " + item["path"])
    expected_raw, expected_final = manifest["raw_frames"], manifest["final_frames"]
    with np.load(root / manifest["smpl_120hz"], allow_pickle=False) as raw:
        with np.load(root / manifest["smpl_30hz"], allow_pickle=False) as final:
            if float(final["mocap_framerate"]) != 30:
                raise ValueError("Expected mocap_framerate=30")
            for name, trailing in (("poses", (24, 3)), ("trans", (3,))):
                if raw[name].shape != (expected_raw,) + trailing or final[name].shape != (expected_final,) + trailing:
                    raise ValueError("Unexpected motion shape: " + name)
                if not np.isfinite(raw[name]).all() or not np.isfinite(final[name]).all():
                    raise ValueError("Non-finite motion")
                np.testing.assert_array_equal(raw[name][::4][1:].astype(np.float32), final[name])
    features = np.load(root / manifest["training_features"], allow_pickle=False)
    if list(features.shape) != manifest["feature_shape"] or not np.isfinite(features).all():
        raise ValueError("Invalid HumanML263 features")
    lines = (root / manifest["training_text"]).read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        raise ValueError("Expected one reviewed example caption")
    caption, tokens, start, end = lines[0].split("#")
    if caption != manifest["caption"] or not tokens.strip():
        raise ValueError("Unexpected caption or empty POS tokens")
    for token in tokens.split():
        word, pos = token.rsplit("/", 1)
        if not word or not pos:
            raise ValueError("Invalid lemma/POS token")
    if not all(math.isfinite(float(x)) and float(x) == 0 for x in (start, end)):
        raise ValueError("Example text must address the whole clip, 0#0")
    print(json.dumps({"verified_files": len(manifest["files"]), "feature_shape": list(features.shape),
        "caption": "reviewed", "archived_120_to_30_relation": "exact_float32_match"}))


if __name__ == "__main__":
    main()
