"""Shared closed-loop-eval utilities used by both rollout.py (BC-MLP) and
rollout_vla.py (the full VLA). Proprioception is assembled to exactly match
training (vla_diffusion.data.libero_dataset.PROPRIO_KEYS): joint_states,
gripper_states, then ee_states = concat(eef_pos, quat2axisangle(eef_quat)),
per LIBERO's own scripts/create_dataset.py.
"""
import os

import numpy as np
import torch
from robosuite.utils.transform_utils import quat2axisangle


def get_proprio(obs):
    joint_states = obs["robot0_joint_pos"]
    gripper_states = obs["robot0_gripper_qpos"]
    ee_states = np.hstack([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"])])
    return np.concatenate([joint_states, gripper_states, ee_states]).astype(np.float32)


def get_image(obs, camera="agentview_image"):
    return torch.from_numpy(obs[camera].copy()).float().permute(2, 0, 1) / 255.0


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
    env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=camera_size, camera_widths=camera_size)
    return env, task
