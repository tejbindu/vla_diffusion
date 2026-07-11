import torch


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over (B, T, D) tensors, ignoring padded timesteps.

    mask: (B, T), 1 for real timesteps and 0 for end-of-episode padding.
    """
    err = (pred - target) ** 2 * mask.unsqueeze(-1)
    return err.sum() / mask.sum().clamp(min=1) / pred.shape[-1]
