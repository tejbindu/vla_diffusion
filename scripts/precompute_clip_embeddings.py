#!/usr/bin/env python
"""Precompute and cache frozen-CLIP vision/text embeddings for one or more
LIBERO task hdf5 files, so training never needs to run CLIP's forward pass.

    uv run python scripts/precompute_clip_embeddings.py \
        --data data/libero_datasets/libero_object/*.hdf5 \
        --cache-dir data/clip_cache
"""
import argparse
import json
import os

import h5py
import torch
from tqdm import tqdm

from vla_diffusion.data.clip_cache import cache_path_for
from vla_diffusion.models.clip_encoders import FrozenClipEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True, help="hdf5 file(s)")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--image-key", default="agentview_rgb")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    encoder = FrozenClipEncoder(device=args.device)

    for path in args.data:
        out_path = cache_path_for(path, args.cache_dir)
        if os.path.exists(out_path):
            print(f"Cache already exists, skipping: {out_path}")
            continue

        with h5py.File(path, "r") as f:
            data = f["data"]
            problem_info = json.loads(data.attrs["problem_info"])
            language = problem_info["language_instruction"]
            demo_keys = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))

            vision_cache = {}
            for demo_key in tqdm(demo_keys, desc=os.path.basename(path)):
                images = data[demo_key]["obs"][args.image_key][:]  # (T, H, W, 3) uint8
                embeds = []
                for start in range(0, len(images), args.batch_size):
                    batch = images[start : start + args.batch_size]
                    batch = torch.from_numpy(batch.copy()).float().permute(0, 3, 1, 2) / 255.0
                    embeds.append(encoder.encode_image(batch).cpu())
                vision_cache[demo_key] = torch.cat(embeds, dim=0)

            text_embed = encoder.encode_text([language])[0].cpu()

        torch.save({"vision": vision_cache, "text": text_embed, "language": language}, out_path)
        print(f"Wrote {out_path} ({len(demo_keys)} demos, language: \"{language}\")")


if __name__ == "__main__":
    main()
