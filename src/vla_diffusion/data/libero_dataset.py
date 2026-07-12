"""PyTorch Dataset over LIBERO demo hdf5 files: action-chunked, normalized.

Proprioception is built to exactly match how LIBERO's own
`scripts/create_dataset.py` derives it from live robosuite obs, so a policy
trained here can be dropped into a live env at eval time without a feature
mismatch:
    joint_states = obs["robot0_joint_pos"]                      (7,)
    gripper_states = obs["robot0_gripper_qpos"]                 (2,)
    ee_states = concat(obs["robot0_eef_pos"],
                        quat2axisangle(obs["robot0_eef_quat"]))  (6,)
"""
import json

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

PROPRIO_KEYS = ["joint_states", "gripper_states", "ee_states"]  # 7 + 2 + 6 = 15
PROPRIO_DIM = 15
ACTION_DIM = 7


class LiberoChunkDataset(Dataset):
    def __init__(self, hdf5_paths, chunk_size=8, image_key="agentview_rgb", action_stats=None):
        self.chunk_size = chunk_size
        self.image_key = image_key
        self.hdf5_paths = list(hdf5_paths)
        self._files = {}  # opened lazily per worker process, see _get_file

        self.index = []  # (path, demo_key, start_t, episode_len)
        self.task_language = {}

        all_actions = []
        for path in self.hdf5_paths:
            with h5py.File(path, "r") as f:
                data = f["data"]
                problem_info = json.loads(data.attrs["problem_info"])
                self.task_language[path] = problem_info["language_instruction"]
                for demo_key in data.keys():
                    ep_len = data[demo_key]["actions"].shape[0]
                    all_actions.append(data[demo_key]["actions"][:])
                    for t in range(ep_len):
                        self.index.append((path, demo_key, t, ep_len))

        if action_stats is None:
            all_actions = np.concatenate(all_actions, axis=0)
            self.action_min = all_actions.min(axis=0).astype(np.float32)
            self.action_max = all_actions.max(axis=0).astype(np.float32)
        else:
            self.action_min = np.array(action_stats["min"], dtype=np.float32)
            self.action_max = np.array(action_stats["max"], dtype=np.float32)

    def _get_file(self, path):
        # Lazily opened so each DataLoader worker process owns its own h5py
        # handle instead of inheriting one across a fork.
        if path not in self._files:
            self._files[path] = h5py.File(path, "r")
        return self._files[path]

    def __len__(self):
        return len(self.index)

    def normalize_action(self, a):
        rng = self.action_max - self.action_min
        rng = np.where(rng < 1e-6, 1.0, rng)
        return (2 * (a - self.action_min) / rng - 1).astype(np.float32)

    def denormalize_action(self, a):
        rng = self.action_max - self.action_min
        return ((a + 1) / 2 * rng + self.action_min).astype(np.float32)

    def action_stats_dict(self):
        return {"min": self.action_min.tolist(), "max": self.action_max.tolist()}

    def _get_proprio_action_language(self, path, demo_key, t, ep_len):
        """The part of __getitem__ that doesn't touch images -- factored out
        so LiberoClipDataset (which uses cached CLIP embeddings instead of
        raw pixels) can reuse it without paying for an unused image decode.
        """
        demo = self._get_file(path)["data"][demo_key]

        proprio = np.concatenate(
            [demo["obs"][k][t] for k in PROPRIO_KEYS], axis=0
        ).astype(np.float32)

        end = min(t + self.chunk_size, ep_len)
        actions = self.normalize_action(demo["actions"][t:end].astype(np.float32))
        pad_len = self.chunk_size - actions.shape[0]
        mask = np.ones(self.chunk_size, dtype=np.float32)
        if pad_len > 0:
            pad = np.repeat(actions[-1:], pad_len, axis=0)
            actions = np.concatenate([actions, pad], axis=0)
            mask[self.chunk_size - pad_len :] = 0.0

        return {
            "proprio": torch.from_numpy(proprio),
            "action_chunk": torch.from_numpy(actions),
            "action_mask": torch.from_numpy(mask),
            "language": self.task_language[path],
        }

    def __getitem__(self, idx):
        path, demo_key, t, ep_len = self.index[idx]
        demo = self._get_file(path)["data"][demo_key]

        image = demo["obs"][self.image_key][t]
        image = torch.from_numpy(image.copy()).float().permute(2, 0, 1) / 255.0

        item = self._get_proprio_action_language(path, demo_key, t, ep_len)
        item["image"] = image
        return item
