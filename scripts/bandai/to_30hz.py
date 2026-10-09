#!/usr/bin/env python3
"""Reproduce the verified Bandai example's 120Hz -> 30Hz arrays.

Historical conversion first removes one 120Hz frame in the BVH finalizer;
this command takes every fourth frame, then removes one more 30Hz frame.
Use only for an input known to have that history, not all arbitrary NPZ files.
"""
import argparse
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.input, allow_pickle=False) as data:
        poses, trans = data["poses"], data["trans"]
    if poses.ndim != 3 or poses.shape[1:] != (24, 3) or trans.shape != (len(poses), 3):
        raise ValueError("Expected poses(T,24,3), trans(T,3)")
    if len(poses) < 8 or not np.isfinite(poses).all() or not np.isfinite(trans).all():
        raise ValueError("Invalid or too short motion")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, poses=poses[::4][1:].astype(np.float32),
        trans=trans[::4][1:].astype(np.float32), mocap_framerate=np.float32(30.0))
    print("frames:", len(poses), "->", len(poses[::4][1:]))


if __name__ == "__main__":
    main()
