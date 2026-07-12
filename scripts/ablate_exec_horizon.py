#!/usr/bin/env python
"""Week 5 ablation: receding-horizon replanning interval (exec_horizon).

exec_horizon=1 replans every step (most reactive, most sampling calls);
exec_horizon=chunk_size executes a full sampled chunk open-loop before
replanning (cheapest, least reactive to drift). This is the practical,
no-retraining version of the "chunk horizon" smoothness-vs-reactivity
tradeoff: chunk_size itself is fixed by the trained checkpoint, but how
often we replan within a chunk is a free eval-time choice.

    MUJOCO_GL=egl LIBERO_CONFIG_PATH=$(pwd)/.libero_config \
        uv run python scripts/ablate_exec_horizon.py \
        --checkpoint outputs/vla_full/ema.pt --num-episodes 5
"""
import argparse
import json
import os

import torch
from diffusers import DDIMScheduler

from vla_diffusion.data.libero_dataset import ACTION_DIM, PROPRIO_DIM
from vla_diffusion.eval.common import build_env
from vla_diffusion.eval.rollout_vla import run_episodes
from vla_diffusion.models.clip_encoders import FrozenClipEncoder
from vla_diffusion.models.vla_diffusion_policy import VLADiffusionPolicy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task-suite", default="libero_object")
    parser.add_argument(
        "--task-names", nargs="+",
        default=["salad_dressing", "cream_cheese", "orange_juice"],
        help="substring match against task names; one rollout set per task",
    )
    parser.add_argument("--exec-horizons", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--camera-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_dir = os.path.dirname(args.checkpoint)
    with open(os.path.join(ckpt_dir, "action_stats.json")) as fp:
        action_stats = json.load(fp)
    with open(os.path.join(ckpt_dir, "config.json")) as fp:
        train_config = json.load(fp)
    chunk_size = train_config["chunk_size"]

    model = VLADiffusionPolicy(proprio_dim=PROPRIO_DIM, action_dim=ACTION_DIM)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.to(args.device).eval()

    clip_encoder = FrozenClipEncoder(device=args.device)
    scheduler = DDIMScheduler(
        num_train_timesteps=train_config["num_train_timesteps"],
        beta_schedule=train_config["beta_schedule"],
        prediction_type="epsilon",
    )

    all_results = []
    print(f"{'task':>16} {'exec_horizon':>13} {'success_rate':>13} {'mean_latency_ms':>17} {'mean_steps':>11}")
    for task_name in args.task_names:
        env, task = build_env(args.task_suite, task_name, args.camera_size)
        for exec_horizon in args.exec_horizons:
            results = run_episodes(
                model, clip_encoder, scheduler, env, task, action_stats, chunk_size,
                num_episodes=args.num_episodes, max_steps=args.max_steps, exec_horizon=exec_horizon,
                num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
                device=args.device, verbose=False,
            )
            print(
                f"{task_name:>16} {exec_horizon:>13} {results['success_rate']:>13.1%} "
                f"{results['mean_sample_latency_s'] * 1000:>17.2f} {results['mean_episode_steps']:>11.1f}"
            )
            all_results.append({"task": task_name, "exec_horizon": exec_horizon, **results})
        env.close()

    out_path = os.path.join(ckpt_dir, "ablation_exec_horizon.json")
    with open(out_path, "w") as fp:
        json.dump(all_results, fp, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
