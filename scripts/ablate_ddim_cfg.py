#!/usr/bin/env python
"""Week 5 ablation: DDIM inference-step count and CFG guidance scale, measured
open-loop (no sim rollout needed) against the same held-out validation split
train_vla.py used -- reconstructed with the same random seed from the paths
and val_fraction recorded in the checkpoint's config.json.

Reports, for each (num_inference_steps, guidance_scale) combination:
  - sample MSE: how close a DDIM-sampled action chunk gets to the real
    ground-truth chunk for held-out observations (the same metric
    train_vla.py's language_sensitivity_check uses, just without the
    wrong-language comparison)
  - mean wall-clock time per sampled chunk (the real latency tradeoff a
    diffusion policy pays vs. a single-forward-pass VLA)

    uv run python scripts/ablate_ddim_cfg.py --checkpoint outputs/vla_full/ema.pt
"""
import argparse
import json
import os
import time

import torch
from diffusers import DDIMScheduler
from torch.utils.data import DataLoader, random_split

from vla_diffusion.data.clip_cache import LiberoClipDataset
from vla_diffusion.data.libero_dataset import ACTION_DIM, PROPRIO_DIM
from vla_diffusion.models.vla_diffusion_policy import VLADiffusionPolicy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--clip-cache-dir", default="data/clip_cache")
    parser.add_argument("--num-inference-steps", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    parser.add_argument("--guidance-scales", type=float, nargs="+", default=[1.0, 2.5, 5.0])
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, val_loader, scheduler, chunk_size, num_inference_steps, guidance_scale, device, max_batches):
    total_se, total_n, total_time, n_chunks = 0.0, 0, 0.0, 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        vision_embed = batch["vision_embed"].to(device)
        text_embed = batch["text_embed"].to(device)
        proprio = batch["proprio"].to(device)
        gt_actions = batch["action_chunk"].to(device)
        mask = batch["action_mask"].to(device)

        start = time.perf_counter()
        sampled = model.sample(
            vision_embed, text_embed, proprio, chunk_size, scheduler,
            num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
        )
        elapsed = time.perf_counter() - start

        se = (((sampled - gt_actions) ** 2) * mask.unsqueeze(-1)).sum().item()
        total_se += se
        total_n += mask.sum().item() * gt_actions.shape[-1]
        total_time += elapsed
        n_chunks += vision_embed.shape[0]

    return {
        "sample_mse": total_se / max(total_n, 1),
        "mean_latency_ms_per_chunk": (total_time / n_chunks) * 1000,
    }


def main():
    args = parse_args()
    ckpt_dir = os.path.dirname(args.checkpoint)
    with open(os.path.join(ckpt_dir, "config.json")) as fp:
        train_config = json.load(fp)
    chunk_size = train_config["chunk_size"]

    dataset = LiberoClipDataset(
        train_config["data_paths"], clip_cache_dir=args.clip_cache_dir, chunk_size=chunk_size
    )
    n_val = max(1, int(len(dataset) * train_config["val_fraction"]))
    n_train = len(dataset) - n_val
    _, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    print(f"Held-out validation set: {len(val_set)} samples (same split train_vla.py used)")

    model = VLADiffusionPolicy(proprio_dim=PROPRIO_DIM, action_dim=ACTION_DIM)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.to(args.device).eval()

    scheduler = DDIMScheduler(
        num_train_timesteps=train_config["num_train_timesteps"],
        beta_schedule=train_config["beta_schedule"],
        prediction_type="epsilon",
    )

    results = []
    print(f"\n{'steps':>6} {'guidance':>9} {'sample_mse':>11} {'latency_ms':>11}")
    for steps in args.num_inference_steps:
        for scale in args.guidance_scales:
            metrics = evaluate(model, val_loader, scheduler, chunk_size, steps, scale, args.device, args.max_batches)
            print(f"{steps:>6} {scale:>9.1f} {metrics['sample_mse']:>11.5f} {metrics['mean_latency_ms_per_chunk']:>11.2f}")
            results.append({"num_inference_steps": steps, "guidance_scale": scale, **metrics})

    out_path = os.path.join(ckpt_dir, "ablation_ddim_cfg.json")
    with open(out_path, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
