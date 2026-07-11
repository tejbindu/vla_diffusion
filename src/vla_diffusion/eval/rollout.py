"""Closed-loop evaluation of a trained BC-MLP policy inside the real LIBERO
sim (robosuite/MuJoCo), not just offline hdf5 replay.

    MUJOCO_GL=egl LIBERO_CONFIG_PATH=$(pwd)/.libero_config \
        uv run python -m vla_diffusion.eval.rollout \
        --checkpoint outputs/bc_mlp_run1/best.pt \
        --task-name pick_up_the_salad_dressing_and_place_it_in_the_basket \
        --num-episodes 10

Proprioception is assembled to exactly match training
(vla_diffusion.data.libero_dataset.PROPRIO_KEYS): joint_states,
gripper_states, then ee_states = concat(eef_pos, quat2axisangle(eef_quat)),
per LIBERO's own scripts/create_dataset.py.
"""
import argparse
import json
import os

import numpy as np
import torch
from robosuite.utils.transform_utils import quat2axisangle

from vla_diffusion.data.libero_dataset import ACTION_DIM, PROPRIO_DIM
from vla_diffusion.models.bc_mlp import BCMLPPolicy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task-suite", default="libero_object")
    parser.add_argument("--task-name", required=True, help="substring match against task names in the suite")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--exec-horizon", type=int, default=1, help="replan every N steps (receding horizon)")
    parser.add_argument("--camera-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def get_proprio(obs):
    joint_states = obs["robot0_joint_pos"]
    gripper_states = obs["robot0_gripper_qpos"]
    ee_states = np.hstack([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"])])
    return np.concatenate([joint_states, gripper_states, ee_states]).astype(np.float32)


def get_image(obs, camera="agentview_image"):
    img = torch.from_numpy(obs[camera].copy()).float().permute(2, 0, 1) / 255.0
    return img


def denormalize(action_norm, stats):
    action_min = np.array(stats["min"])
    action_max = np.array(stats["max"])
    return (action_norm + 1) / 2 * (action_max - action_min) + action_min


def build_env(task_suite, task_name_substr, camera_size):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bm = benchmark.get_benchmark_dict()[task_suite]()
    matches = [i for i, name in enumerate(bm.get_task_names()) if task_name_substr in name]
    if not matches:
        raise SystemExit(f"No task in {task_suite} matches '{task_name_substr}'. Options: {bm.get_task_names()}")
    task = bm.get_task(matches[0])
    bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    print(f"Task: {task.name}")
    env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=camera_size, camera_widths=camera_size)
    return env


def main():
    args = parse_args()
    ckpt_dir = os.path.dirname(args.checkpoint)
    with open(os.path.join(ckpt_dir, "action_stats.json")) as fp:
        action_stats = json.load(fp)
    with open(os.path.join(ckpt_dir, "config.json")) as fp:
        train_config = json.load(fp)
    chunk_size = train_config["chunk_size"]

    model = BCMLPPolicy(proprio_dim=PROPRIO_DIM, action_dim=ACTION_DIM, chunk_size=chunk_size)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.to(args.device).eval()

    env = build_env(args.task_suite, args.task_name, args.camera_size)

    successes = 0
    for ep in range(args.num_episodes):
        obs = env.reset()
        action_plan = None
        plan_step = 0
        success = False
        for t in range(args.max_steps):
            if action_plan is None or plan_step >= args.exec_horizon:
                image = get_image(obs).unsqueeze(0).to(args.device)
                proprio = torch.from_numpy(get_proprio(obs)).unsqueeze(0).to(args.device)
                with torch.no_grad():
                    pred = model(image, proprio)[0].cpu().numpy()
                action_plan = denormalize(pred, action_stats)
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
        print(f"episode {ep:3d}  success={success}  steps={t + 1}")

    env.close()
    print(f"\nSuccess rate: {successes}/{args.num_episodes} = {successes / args.num_episodes:.1%}")


if __name__ == "__main__":
    main()
