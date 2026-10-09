#!/usr/bin/env python3
"""Run the historical SMPL BVH/NPZ finalizer, with explicit paths.

This deliberately uses the original library's zxy rotation conversion.
It is separate from the later header-auto converter.
"""
import argparse
import hashlib
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tempo-repo", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--reference-bvh", type=Path, required=True)
    args = p.parse_args()
    repo, work, ref = [x.resolve() for x in (args.tempo_repo, args.work_dir, args.reference_bvh)]
    library = repo / "dataset/python/smpl_bvh_to_smpl_npz.py"
    expected = "c030469c76fcb3090f39b7c2e48109a4257c8bbe8e469aabf0558d41ef44e007"
    if hashlib.sha256(library.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != expected:
        raise ValueError("Unexpected finalizer version; use the pinned 5006158 revision.")
    if not ref.is_file():
        raise FileNotFoundError(ref)
    if not list((work / "bvhForC/output").glob("*.bvh")):
        raise FileNotFoundError("No MotionBuilder outputs in bvhForC/output")
    if any((work / name).exists() for name in ("bvhSMPL", "npz")):
        raise FileExistsError("bvhSMPL/npz already exists; use a fresh work directory")
    sys.path.insert(0, str(repo))
    from dataset.python import smpl_bvh_to_smpl_npz as conversion
    conversion.build_smpl_bvh(root=str(work), smpl_t_bvh_path=str(ref),
        input_bvh_dir="bvhForC/output", out_bvh_smpl_dir="bvhSMPL")
    conversion.build_smpl_npz(root=str(work), in_bvh_smpl_dir="bvhSMPL",
        out_npz_dir="npz", trans_scale=0.01)


if __name__ == "__main__":
    main()
