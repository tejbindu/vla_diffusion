"""Precomputed-CLIP-embedding variant of LiberoChunkDataset.

Per the project's compute-minimizing plan: vision/text backbones are frozen,
so their embeddings are computed once (scripts/precompute_clip_embeddings.py)
and cached to disk, rather than re-running CLIP's forward pass on every
training step. Training then only updates the fusion transformer + diffusion
head, which is the whole point -- it's what keeps this trainable in a couple
of GPU-hours instead of a day.
"""
import os

import torch

from vla_diffusion.data.libero_dataset import LiberoChunkDataset


def cache_path_for(hdf5_path, cache_dir):
    base = os.path.splitext(os.path.basename(hdf5_path))[0]
    return os.path.join(cache_dir, f"{base}.clip_cache.pt")


class LiberoClipDataset(LiberoChunkDataset):
    def __init__(self, hdf5_paths, clip_cache_dir, chunk_size=8, action_stats=None):
        super().__init__(hdf5_paths, chunk_size=chunk_size, action_stats=action_stats)
        self.vision_cache = {}  # path -> {demo_key: (T, 512) tensor}
        self.text_cache = {}  # path -> (512,) tensor
        for path in self.hdf5_paths:
            cache_file = cache_path_for(path, clip_cache_dir)
            if not os.path.exists(cache_file):
                raise FileNotFoundError(
                    f"No CLIP cache for {path} at {cache_file}. "
                    "Run scripts/precompute_clip_embeddings.py first."
                )
            cached = torch.load(cache_file, weights_only=True)
            self.vision_cache[path] = cached["vision"]
            self.text_cache[path] = cached["text"]

    def __getitem__(self, idx):
        path, demo_key, t, ep_len = self.index[idx]
        item = self._get_proprio_action_language(path, demo_key, t, ep_len)
        item["vision_embed"] = self.vision_cache[path][demo_key][t]
        item["text_embed"] = self.text_cache[path]
        return item
