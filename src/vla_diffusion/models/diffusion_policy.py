"""Diffusion policy: SimpleCNNEncoder + proprio -> obs_cond, feeding the
FiLM-conditioned DiffusionActionHead. Trained with a DDPM epsilon-prediction
loss; sampled at inference with DDIM for a small number of steps.
"""
import torch
import torch.nn as nn
from diffusers import DDIMScheduler

from vla_diffusion.models.diffusion_head import DiffusionActionHead
from vla_diffusion.models.encoders import SimpleCNNEncoder


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        proprio_dim=15,
        action_dim=7,
        img_embed_dim=128,
        hidden_dims=(128, 256, 256),
        diffusion_step_embed_dim=128,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.image_encoder = SimpleCNNEncoder(out_dim=img_embed_dim)
        cond_dim = img_embed_dim + proprio_dim
        self.action_head = DiffusionActionHead(
            action_dim=action_dim,
            cond_dim=cond_dim,
            hidden_dims=hidden_dims,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
        )

    def obs_cond(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        img_feat = self.image_encoder(image)
        return torch.cat([img_feat, proprio], dim=-1)

    def forward(self, noisy_actions: torch.Tensor, timestep: torch.Tensor, image: torch.Tensor, proprio: torch.Tensor):
        cond = self.obs_cond(image, proprio)
        return self.action_head(noisy_actions, timestep, cond)

    @torch.no_grad()
    def sample(
        self,
        image: torch.Tensor,
        proprio: torch.Tensor,
        chunk_size: int,
        scheduler: DDIMScheduler,
        num_inference_steps: int = 10,
    ) -> torch.Tensor:
        """DDIM sampling: iteratively denoise from Gaussian noise, conditioned
        on the observation. Returns an action chunk in normalized [-1, 1] space.
        """
        device = image.device
        batch_size = image.shape[0]
        cond = self.obs_cond(image, proprio)

        scheduler.set_timesteps(num_inference_steps, device=device)
        actions = torch.randn(batch_size, chunk_size, self.action_dim, device=device)

        for t in scheduler.timesteps:
            t_batch = t.expand(batch_size).to(device)
            pred_noise = self.action_head(actions, t_batch, cond)
            actions = scheduler.step(pred_noise, t, actions).prev_sample

        return actions
