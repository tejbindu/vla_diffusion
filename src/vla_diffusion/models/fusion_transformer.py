"""Small learned transformer that fuses frozen CLIP vision + text embeddings
with proprioception into a single conditioning vector for the diffusion
action head.

Each modality contributes one pooled token (not patch-level vision tokens --
a deliberate simplification: CLIP's pooled embedding already summarizes the
frame, and a 3-4 token self-attention fusion is enough signal at this scale
without the extra compute of patch-level cross-attention). A learned
modality-type embedding tells the transformer which token is which, and a
CLS token's output after self-attention is the conditioning vector z.
"""
import torch
import torch.nn as nn

VISION, TEXT, PROPRIO = 0, 1, 2


class FusionTransformer(nn.Module):
    def __init__(
        self,
        vision_dim,
        text_dim,
        proprio_dim,
        d_model=256,
        n_heads=4,
        n_layers=3,
        out_dim=256,
    ):
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
        self.proprio_proj = nn.Linear(proprio_dim, d_model)
        self.modality_embed = nn.Embedding(3, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, out_dim)

    def forward(self, vision_embed: torch.Tensor, text_embed: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        B = vision_embed.shape[0]
        v = self.vision_proj(vision_embed) + self.modality_embed.weight[VISION]
        t = self.text_proj(text_embed) + self.modality_embed.weight[TEXT]
        p = self.proprio_proj(proprio) + self.modality_embed.weight[PROPRIO]
        cls = self.cls_token.expand(B, -1, -1).squeeze(1)

        tokens = torch.stack([cls, v, t, p], dim=1)  # (B, 4, d_model)
        out = self.transformer(tokens)
        return self.out_proj(out[:, 0])  # CLS token's output
