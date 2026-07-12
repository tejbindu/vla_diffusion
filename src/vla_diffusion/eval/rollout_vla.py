"""Closed-loop evaluation of a trained VLADiffusionPolicy inside the real
LIBERO sim, with CFG-guided DDIM sampling and receding-horizon control.

    MUJOCO_GL=egl LIBERO_CONFIG_PATH=$(pwd)/.libero_config \
        uv run python -m vla_diffusion.eval.rollout_vla \
        --checkpoint outputs/vla_run1/ema.pt \
        --task-name pick_up_the_salad_dressing_and_place_it_in_the_basket \
        --num-episodes 10 --num-inference-steps 10 --guidance-scale 1.0 --exec-horizon 4

Unlike training (which uses cached CLIP embeddings, since the demo frames
never change), eval must encode the live frame with CLIP on every replan --
there's nothing to cache when the observation is coming from a running sim.
The language embedding is the one exception: it's the same for the whole
episode, so it's computed once via task.language, not per step.
"""
import argparse
import json
import os
import time

import torch
from diffusers import DDIMScheduler

from vla_diffusion.data.libero_dataset import ACTION_DIM, PROPRIO_DIM
from vla_diffusion.eval.common import build_env, denormalize, get_image, get_proprio
from vla_diffusion.models.clip_encoders import FrozenClipEncoder
from vla_diffusion.models.vla_diffusion_policy import VLADiffusionPolicy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task-suite", default="libero_object")
    parser.add_argument("--task-name", required=True, help="substring match against task names in the suite")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--exec-horizon", type=int, default=4, help="replan every N steps (receding horizon)")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="1.0 = no CFG (skips the extra uncond pass)")
    parser.add_argument("--camera-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def run_episodes(
    model, clip_encoder, scheduler, env, task, action_stats, chunk_size,
    num_episodes, max_steps, exec_horizon, num_inference_steps, guidance_scale, device,
    verbose=True,
):
    text_embed = clip_encoder.encode_text([task.language]).to(device)

    successes = 0
    sample_latencies = []
    episode_steps = []
    for ep in range(num_episodes):
        obs = env.reset()
        action_plan = None
        plan_step = 0
        success = False
        t = 0
        for t in range(max_steps):
            if action_plan is None or plan_step >= exec_horizon:
                image = get_image(obs).unsqueeze(0).to(device)
                proprio = torch.from_numpy(get_proprio(obs)).unsqueeze(0).to(device)

                start = time.perf_counter()
                vision_embed = clip_encoder.encode_image(image)
                sampled = model.sample(
                    vision_embed, text_embed, proprio, chunk_size, scheduler,
                    num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                )[0].cpu().numpy()
                sample_latencies.append(time.perf_counter() - start)

                action_plan = denormalize(sampled, action_stats)
                plan_step = 0

            action = action_plan[plan_step]
            plan_step += 1
            obs, reward, done, info = env.step(action)

            if env.check_success():
                success = True
                break
            if done:
                break

        successes += int(success)
        episode_steps.append(t + 1)
        if verbose:
            print(f"episode {ep:3d}  success={success}  steps={t + 1}")

    return {
        "success_rate": successes / num_episodes,
        "successes": successes,
        "num_episodes": num_episodes,
        "mean_sample_latency_s": sum(sample_latencies) / len(sample_latencies),
        "mean_episode_steps": sum(episode_steps) / len(episode_steps),
    }


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

    env, task = build_env(args.task_suite, args.task_name, args.camera_size)
    print(f"Task: {task.name}  (\"{task.language}\")")

    results = run_episodes(
        model, clip_encoder, scheduler, env, task, action_stats, chunk_size,
        num_episodes=args.num_episodes, max_steps=args.max_steps, exec_horizon=args.exec_horizon,
        num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale, device=args.device,
    )
    env.close()

    print(
        f"\nSuccess rate: {results['successes']}/{results['num_episodes']} = {results['success_rate']:.1%}"
        f"  |  mean sample latency: {results['mean_sample_latency_s'] * 1000:.1f} ms"
        f"  |  mean episode length: {results['mean_episode_steps']:.1f} steps"
    )


if __name__ == "__main__":
    main()
