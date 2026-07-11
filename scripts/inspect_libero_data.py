#!/usr/bin/env python
"""Week 1 data-inspection pass over a LIBERO demo hdf5 file.

This is deliberately standalone (no project imports) so it can run the
moment a dataset file lands, before any modeling code exists. It reports the
things you'd want to know before trusting a dataset for imitation learning:
episode-length distribution, per-dimension action statistics (needed for
normalization later), the language instruction, and a handful of sample
frames -- plus a first-pass outlier flag on episode length, since an
anomalously short/long demo is often a recording glitch or a failed episode
that slipped into the "successful demos" file.

Usage:
    uv run python scripts/inspect_libero_data.py --file data/libero_datasets/libero_object/<task>_demo.hdf5
"""
import argparse
import json
import os

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ACTION_DIM_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to a LIBERO *_demo.hdf5 file")
    parser.add_argument(
        "--outdir",
        default=None,
        help="Where to write plots/sample frames/stats.json. "
        "Defaults to outputs/data_inspection/<task_name>/",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    task_name = os.path.splitext(os.path.basename(args.file))[0]
    outdir = args.outdir or os.path.join("outputs", "data_inspection", task_name)
    os.makedirs(outdir, exist_ok=True)

    f = h5py.File(args.file, "r")
    data = f["data"]
    demo_keys = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))

    problem_info = json.loads(data.attrs["problem_info"])
    language_instruction = problem_info["language_instruction"]

    print(f"Task: {task_name}")
    print(f"Language instruction: \"{language_instruction}\"")
    print(f"Num demos: {len(demo_keys)}")

    # --- episode length distribution ---
    lengths = np.array([data[k]["actions"].shape[0] for k in demo_keys])
    print(
        f"Episode length: min={lengths.min()} mean={lengths.mean():.1f} "
        f"max={lengths.max()} std={lengths.std():.1f}"
    )

    median_len = np.median(lengths)
    mad = np.median(np.abs(lengths - median_len)) + 1e-6  # robust spread
    outlier_mask = np.abs(lengths - median_len) / mad > 5
    outlier_demos = [demo_keys[i] for i in np.where(outlier_mask)[0]]
    if outlier_demos:
        print(
            f"[flagged] {len(outlier_demos)} length outliers (>5 MAD from median): "
            f"{outlier_demos} -- inspect these before trusting them as good demos"
        )
    else:
        print("[ok] no length outliers by robust z-score > 5")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(lengths, bins=20)
    ax.set_xlabel("episode length (steps)")
    ax.set_ylabel("count")
    ax.set_title(f"{task_name}: episode length distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "episode_length_hist.png"), dpi=120)
    plt.close(fig)

    # --- action statistics (per dimension, across all demos) ---
    all_actions = np.concatenate([data[k]["actions"][:] for k in demo_keys], axis=0)
    action_stats = {
        "dim_names": ACTION_DIM_NAMES,
        "min": all_actions.min(axis=0).tolist(),
        "max": all_actions.max(axis=0).tolist(),
        "mean": all_actions.mean(axis=0).tolist(),
        "std": all_actions.std(axis=0).tolist(),
    }
    print("Action stats (per dim):")
    for i, name in enumerate(ACTION_DIM_NAMES):
        print(
            f"  {name:8s} min={action_stats['min'][i]:+.3f} max={action_stats['max'][i]:+.3f} "
            f"mean={action_stats['mean'][i]:+.3f} std={action_stats['std'][i]:.3f}"
        )
    with open(os.path.join(outdir, "action_stats.json"), "w") as fp:
        json.dump(action_stats, fp, indent=2)

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for i, name in enumerate(ACTION_DIM_NAMES):
        ax = axes.flat[i]
        ax.hist(all_actions[:, i], bins=50)
        ax.set_title(name)
    axes.flat[-1].axis("off")
    fig.suptitle(f"{task_name}: per-dimension action distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "action_distributions.png"), dpi=120)
    plt.close(fig)

    # --- sample frames from demo_0 (start / mid / end) for a visual sanity check ---
    demo0 = data[demo_keys[0]]
    n = demo0["actions"].shape[0]
    for t, tag in [(0, "start"), (n // 2, "mid"), (n - 1, "end")]:
        for cam in ["agentview_rgb", "eye_in_hand_rgb"]:
            img = Image.fromarray(demo0["obs"][cam][t])
            img.save(os.path.join(outdir, f"{demo_keys[0]}_{cam}_{tag}.png"))

    print(f"\nWrote plots, sample frames, and action_stats.json to {outdir}/")


if __name__ == "__main__":
    main()
