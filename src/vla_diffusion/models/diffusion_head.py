"""FiLM-conditioned 1D denoising network for action-chunk diffusion, in the
style of Chi et al.'s Diffusion Policy.

This is a simplified, single-resolution version of the paper's temporal
U-Net: a stack of FiLM-conditioned residual 1D-conv blocks operating at the
full chunk length throughout, with no down/upsampling. The paper's
multi-resolution U-Net earns its complexity at longer action horizons
(16-64 steps); at our chunk_size of ~8 there isn't much signal to gain from
multiple temporal resolutions, so a flat residual stack is the simpler,
equally expressive choice for this scale.

Conditioning (observation embedding + diffusion timestep embedding) is
injected at every block via FiLM (feature-wise linear modulation) -- the
same mechanism the paper uses, and the reason a single global_cond vector
can steer denoising at every layer rather than only at the input.
"""
import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """Standard transformer/diffusion sinusoidal embedding of a scalar timestep."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class FiLMResidualBlock1D(nn.Module):
    """Conv1d -> GroupNorm -> FiLM(cond) -> Mish, twice, plus a residual path."""

    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=5, n_groups=8):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(min(n_groups, out_channels), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(min(n_groups, out_channels), out_channels)
        self.film = nn.Linear(cond_dim, out_channels * 2)
        self.act = nn.Mish()
        self.residual_proj = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T), cond: (B, cond_dim)
        h = self.act(self.norm1(self.conv1(x)))
        scale, shift = self.film(cond).chunk(2, dim=-1)  # each (B, C_out)
        h = h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.residual_proj(x)


class DiffusionActionHead(nn.Module):
    def __init__(
        self,
        action_dim,
        cond_dim,
        hidden_dims=(128, 256, 256),
        diffusion_step_embed_dim=128,
        kernel_size=5,
    ):
        super().__init__()
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )
        global_cond_dim = cond_dim + diffusion_step_embed_dim

        self.in_proj = nn.Conv1d(action_dim, hidden_dims[0], 1)
        self.blocks = nn.ModuleList(
            [
                FiLMResidualBlock1D(hidden_dims[i], hidden_dims[i + 1], global_cond_dim, kernel_size)
                for i in range(len(hidden_dims) - 1)
            ]
        )
        self.out_proj = nn.Conv1d(hidden_dims[-1], action_dim, 1)

    def forward(self, noisy_actions: torch.Tensor, timestep: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        noisy_actions: (B, T, action_dim)
        timestep: (B,) integer diffusion timesteps
        cond: (B, cond_dim) observation conditioning (image + proprio embedding)
        returns: predicted noise, (B, T, action_dim)
        """
        x = noisy_actions.transpose(1, 2)  # (B, action_dim, T)
        t_emb = self.diffusion_step_encoder(timestep)  # (B, dsed)
        global_cond = torch.cat([cond, t_emb], dim=-1)

        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x, global_cond)
        x = self.out_proj(x)
        return x.transpose(1, 2)  # (B, T, action_dim)
