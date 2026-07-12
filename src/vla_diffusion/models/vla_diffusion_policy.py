"""The flagship model: frozen CLIP vision+text embeddings -> learned fusion
transformer -> FiLM-conditioned diffusion action head, with classifier-free
guidance on the language conditioning.

Takes precomputed/live vision_embed and text_embed tensors rather than raw
images/strings -- encoding with the frozen CLIP backbone is a separate step
(cached offline during training via scripts/precompute_clip_embeddings.py,
or run live per-frame during closed-loop eval via FrozenClipEncoder).
"""
import torch
import torch.nn as nn
from diffusers import DDIMScheduler

from vla_diffusion.models.diffusion_head import DiffusionActionHead
from vla_diffusion.models.fusion_transformer import FusionTransformer


class VLADiffusionPolicy(nn.Module):
    def __init__(
        self,
        vision_dim=512,
        text_dim=512,
        proprio_dim=15,
        action_dim=7,
        fusion_d_model=256,
        fusion_heads=4,
        fusion_layers=3,
        diffusion_hidden_dims=(256, 256, 256),
        diffusion_step_embed_dim=128,
    ):
        super().__init__()
        self.action_dim = action_dim
        # Learned "no language given" marker for classifier-free guidance,
        # rather than literal zeros -- gives the model a fittable target for
        # what an unconditional prediction should look like.
        self.null_text_embed = nn.Parameter(torch.randn(text_dim) * 0.02)

        self.fusion = FusionTransformer(
            vision_dim=vision_dim,
            text_dim=text_dim,
            proprio_dim=proprio_dim,
            d_model=fusion_d_model,
            n_heads=fusion_heads,
            n_layers=fusion_layers,
            out_dim=fusion_d_model,
        )
        self.action_head = DiffusionActionHead(
            action_dim=action_dim,
            cond_dim=fusion_d_model,
            hidden_dims=diffusion_hidden_dims,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
        )

    def _drop_text_cond(self, text_embed: torch.Tensor, cond_dropout_prob: float) -> torch.Tensor:
        if cond_dropout_prob <= 0:
            return text_embed
        drop_mask = torch.rand(text_embed.shape[0], device=text_embed.device) < cond_dropout_prob
        text_embed = text_embed.clone()
        text_embed[drop_mask] = self.null_text_embed
        return text_embed

    def obs_cond(self, vision_embed, text_embed, proprio, cond_dropout_prob=0.0):
        text_embed = self._drop_text_cond(text_embed, cond_dropout_prob)
        return self.fusion(vision_embed, text_embed, proprio)

    def forward(self, noisy_actions, timestep, vision_embed, text_embed, proprio, cond_dropout_prob=0.0):
        cond = self.obs_cond(vision_embed, text_embed, proprio, cond_dropout_prob)
        return self.action_head(noisy_actions, timestep, cond)

    @torch.no_grad()
    def sample(
        self,
        vision_embed: torch.Tensor,
        text_embed: torch.Tensor,
        proprio: torch.Tensor,
        chunk_size: int,
        scheduler: DDIMScheduler,
        num_inference_steps: int = 10,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """DDIM sampling with classifier-free guidance:
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        guidance_scale == 1.0 skips the extra unconditional forward pass
        entirely (pure conditional sampling, no guidance).
        """
        device = vision_embed.device
        B = vision_embed.shape[0]
        cond = self.fusion(vision_embed, text_embed, proprio)
        if guidance_scale != 1.0:
            null_text = self.null_text_embed.unsqueeze(0).expand(B, -1)
            uncond = self.fusion(vision_embed, null_text, proprio)

        scheduler.set_timesteps(num_inference_steps, device=device)
        actions = torch.randn(B, chunk_size, self.action_dim, device=device)

        for t in scheduler.timesteps:
            t_batch = t.expand(B).to(device)
            if guidance_scale != 1.0:
                eps_cond = self.action_head(actions, t_batch, cond)
                eps_uncond = self.action_head(actions, t_batch, uncond)
                pred_noise = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                pred_noise = self.action_head(actions, t_batch, cond)
            actions = scheduler.step(pred_noise, t, actions).prev_sample

        return actions
