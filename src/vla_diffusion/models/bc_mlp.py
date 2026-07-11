"""BC-MLP baseline: a small CNN image encoder + MLP head predicting an
action chunk. Deliberately simple -- it exists to validate the data ->
train -> closed-loop-eval pipeline before the diffusion head (Week 3) and
the frozen vision-language fusion trunk (Week 4) are added on top of it.
"""
import torch
import torch.nn as nn


class SimpleCNNEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x):
        return self.proj(self.net(x))


class BCMLPPolicy(nn.Module):
    def __init__(self, proprio_dim=15, action_dim=7, chunk_size=8, img_embed_dim=128, hidden_dim=256):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.image_encoder = SimpleCNNEncoder(out_dim=img_embed_dim)
        self.head = nn.Sequential(
            nn.Linear(img_embed_dim + proprio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, chunk_size * action_dim),
        )

    def forward(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        img_feat = self.image_encoder(image)
        x = torch.cat([img_feat, proprio], dim=-1)
        return self.head(x).view(-1, self.chunk_size, self.action_dim)
