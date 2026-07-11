"""Week 3: train the diffusion policy on one or more LIBERO task hdf5 files.

    uv run python -m vla_diffusion.training.train_diffusion \
        --data data/libero_datasets/libero_object/pick_up_the_salad_dressing_and_place_it_in_the_basket_demo.hdf5 \
        --outdir outputs/diffusion_run1 --epochs 100

Sanity check for this week: the masked epsilon-prediction training loss
should trend down, and DDIM-sampled action chunks on held-out validation
observations should get closer to the ground-truth action chunk over
training (`ddim_sample_mse`) -- that's the "did it actually learn the task"
signal, checked without needing a full closed-loop sim rollout (Week 5).
"""
import argparse
import copy
import json
import os

import torch
from diffusers import DDIMScheduler, DDPMScheduler
from torch.utils.data import DataLoader, random_split

from vla_diffusion.data.libero_dataset import ACTION_DIM, PROPRIO_DIM, LiberoChunkDataset
from vla_diffusion.models.diffusion_policy import DiffusionPolicy
from vla_diffusion.training.data_utils import resolve_paths
from vla_diffusion.training.losses import masked_mse

NUM_TRAIN_TIMESTEPS = 100
BETA_SCHEDULE = "squaredcos_cap_v2"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True, help="hdf5 file(s) or glob pattern(s)")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--sample-every", type=int, default=10, help="epochs between DDIM sample-quality checks")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class EMA:
    """Exponential moving average of model weights. Standard for diffusion
    models: the raw, just-stepped weights tend to sample noisily, while the
    EMA shadow weights sample much more reliably -- this is what we actually
    checkpoint and evaluate with.
    """

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.mul_(self.decay).add_(p, alpha=1 - self.decay)


@torch.no_grad()
def sample_quality_check(model, val_loader, scheduler, chunk_size, num_inference_steps, device, max_batches=4):
    """DDIM-sample action chunks for held-out obs and compare to ground truth."""
    model.eval()
    total_se, total_n = 0.0, 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        image = batch["image"].to(device)
        proprio = batch["proprio"].to(device)
        gt_actions = batch["action_chunk"].to(device)
        mask = batch["action_mask"].to(device)

        sampled = model.sample(image, proprio, chunk_size, scheduler, num_inference_steps)
        se = ((sampled - gt_actions) ** 2 * mask.unsqueeze(-1)).sum()
        total_se += se.item()
        total_n += mask.sum().item() * gt_actions.shape[-1]
    return total_se / max(total_n, 1)


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    paths = resolve_paths(args.data)
    print(f"Training on {len(paths)} file(s): {[os.path.basename(p) for p in paths]}")

    dataset = LiberoChunkDataset(paths, chunk_size=args.chunk_size)
    n_val = max(1, int(len(dataset) * args.val_fraction))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )
    print(f"Dataset: {len(dataset)} samples ({n_train} train / {n_val} val)")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )

    model = DiffusionPolicy(proprio_dim=PROPRIO_DIM, action_dim=ACTION_DIM).to(args.device)
    ema = EMA(model, args.ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_scheduler = DDPMScheduler(
        num_train_timesteps=NUM_TRAIN_TIMESTEPS, beta_schedule=BETA_SCHEDULE, prediction_type="epsilon"
    )
    infer_scheduler = DDIMScheduler(
        num_train_timesteps=NUM_TRAIN_TIMESTEPS, beta_schedule=BETA_SCHEDULE, prediction_type="epsilon"
    )

    with open(os.path.join(args.outdir, "action_stats.json"), "w") as fp:
        json.dump(dataset.action_stats_dict(), fp, indent=2)
    with open(os.path.join(args.outdir, "config.json"), "w") as fp:
        json.dump(
            vars(args)
            | {
                "data_paths": paths,
                "num_train_timesteps": NUM_TRAIN_TIMESTEPS,
                "beta_schedule": BETA_SCHEDULE,
            },
            fp,
            indent=2,
        )

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            image = batch["image"].to(args.device)
            proprio = batch["proprio"].to(args.device)
            actions = batch["action_chunk"].to(args.device)
            mask = batch["action_mask"].to(args.device)

            noise = torch.randn_like(actions)
            timesteps = torch.randint(
                0, NUM_TRAIN_TIMESTEPS, (actions.shape[0],), device=args.device
            ).long()
            noisy_actions = train_scheduler.add_noise(actions, noise, timesteps)

            pred_noise = model(noisy_actions, timesteps, image, proprio)
            loss = masked_mse(pred_noise, noise, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(model)
            train_loss += loss.item() * image.shape[0]
        train_loss /= n_train

        log = f"epoch {epoch:3d}  train_loss={train_loss:.5f}"
        if (epoch + 1) % args.sample_every == 0 or epoch == args.epochs - 1:
            sample_mse = sample_quality_check(
                ema.shadow, val_loader, infer_scheduler, args.chunk_size, args.num_inference_steps, args.device
            )
            log += f"  ddim_sample_mse={sample_mse:.5f}"
        print(log)

    torch.save(model.state_dict(), os.path.join(args.outdir, "last.pt"))
    torch.save(ema.shadow.state_dict(), os.path.join(args.outdir, "ema.pt"))
    print(f"Done. Checkpoints + action_stats.json written to {args.outdir}")


if __name__ == "__main__":
    main()
