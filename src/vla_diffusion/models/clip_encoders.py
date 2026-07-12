"""Frozen CLIP vision + text encoders (ViT-B/32, original OpenAI weights via
open_clip). Used both offline (scripts/precompute_clip_embeddings.py caches
vision/text embeddings for training) and online (closed-loop eval, where
each live frame must be encoded on the fly since there's nothing to cache).

Preprocessing is reimplemented as plain tensor ops instead of open_clip's
PIL-based `preprocess` transform: our images already arrive as float CHW
tensors in [0, 1] (from LiberoChunkDataset / the live rollout obs), and
since LIBERO's cameras are always square, CLIP's Resize+CenterCrop
simplifies to a single resize -- no PIL round-trip needed, and the same
code path works for both a single live frame and a batched offline pass.
"""
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAME = "ViT-B-32-quickgelu"
PRETRAINED = "openai"
CLIP_INPUT_SIZE = 224
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
EMBED_DIM = 512


class FrozenClipEncoder(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
        self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        self.model = model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B, 3, H, W) float in [0, 1], H == W (true for all LIBERO cameras)
        images = F.interpolate(
            images, size=(CLIP_INPUT_SIZE, CLIP_INPUT_SIZE), mode="bicubic", align_corners=False, antialias=True
        )
        return (images - self.mean) / self.std

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        images = self._preprocess(images.to(self.mean.device))
        return self.model.encode_image(images).float()

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts).to(self.mean.device)
        return self.model.encode_text(tokens).float()
