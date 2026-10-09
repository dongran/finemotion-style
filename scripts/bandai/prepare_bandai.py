#!/usr/bin/env python3
"""Prepare Bandai input using the recovered 2025 source-specific helper.

Dependency: tempo-changing-music2motion at
500615821e04b248b0e0f531ec1b8096bb948ef2 (not the later revert).
"""
import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bvh", type=Path, required=True)
    p.add_argument("--tpose", type=Path, required=True)
    p.add_argument("--tempo-repo", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    args = p.parse_args()
    src, template, repo, work = [x.resolve() for x in
        (args.bvh, args.tpose, args.tempo_repo, args.work_dir)]
    helper = repo / "dataset/python/add_tpose_and_rename_clips.py"
    for path in (src, template, helper):
        if not path.is_file():
            raise FileNotFoundError(path)
    if work.exists():
        raise FileExistsError("Choose a fresh work directory: " + str(work))
    expected = "5d17d767af5a5fee65c598aa4746ce91dbedddcf6d819e28ccd02022a96b3570"
    if hashlib.sha256(helper.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != expected:
        raise ValueError("Unexpected helper version; use the pinned 5006158 revision.")
    sys.path.insert(0, str(helper.parent))
    spec = importlib.util.spec_from_file_location("bandai_prepare_helper", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The helper selects a reference file from the work directory by name.
    (work / "bvh").mkdir(parents=True)
    shutil.copyfile(template, work / "T-pose-bandai.bvh")
    shutil.copyfile(src, work / "bvh" / src.name)
    module.prepend_tpose_frame([str(work / "bvh" / src.name)], str(work / "bvhForC"))
    print(json.dumps({"prepared": str(work / "bvhForC" / src.name),
                      "helper_sha256": expected}))


if __name__ == "__main__":
    main()
