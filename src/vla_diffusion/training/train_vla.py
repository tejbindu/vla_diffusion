"""Week 4: train the full VLA (frozen CLIP + fusion transformer + diffusion
head) on multiple LIBERO-Object tasks at once, with classifier-free guidance
dropout on the language conditioning.

    uv run python -m vla_diffusion.training.train_vla \
        --data data/libero_datasets/libero_object/*.hdf5 \
        --clip-cache-dir data/clip_cache \
        --outdir outputs/vla_run1 --epochs 30

Requires scripts/precompute_clip_embeddings.py to have already been run for
the same --data files (into --clip-cache-dir).

Sanity check for this week isn't just "does loss go down" -- LIBERO-Object
tasks all share the *same* shelf scene (every grocery item is present in
every task's initial state); the only thing that tells them apart is the
language instruction. So a model that ignores language and only looks at
vision+proprio literally cannot solve more than one of these tasks. We use
that: language_sensitivity_check compares DDIM-sample MSE against ground
truth when conditioned on the correct instruction vs. a wrong (shuffled)
one from a different task in the batch. If the model actually uses
language, correct-language MSE should be clearly lower.
"""
import argparse
import copy
import json
import os

import torch
from diffusers import DDIMScheduler, DDPMScheduler
from torch.utils.data import DataLoader, random_split

from vla_diffusion.data.clip_cache import LiberoClipDataset
from vla_diffusion.data.libero_dataset import ACTION_DIM, PROPRIO_DIM
from vla_diffusion.models.vla_diffusion_policy import VLADiffusionPolicy
from vla_diffusion.training.data_utils import resolve_paths
from vla_diffusion.training.losses import masked_mse

NUM_TRAIN_TIMESTEPS = 100
BETA_SCHEDULE = "squaredcos_cap_v2"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--clip-cache-dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--cond-dropout-prob", type=float, default=0.1, help="CFG training dropout on language")
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class EMA:
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
def language_sensitivity_check(model, val_loader, scheduler, chunk_size, num_inference_steps, device, max_batches=4):
    """Correct-language vs. wrong-(shuffled)-language DDIM sample MSE. If the
    model actually conditions on language, correct should clearly beat wrong
    -- LIBERO-Object's shared shelf scene means vision+proprio alone can't
    disambiguate the task.
    """
    model.eval()
    correct_se, wrong_se, n = 0.0, 0.0, 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        vision_embed = batch["vision_embed"].to(device)
        text_embed = batch["text_embed"].to(device)
        proprio = batch["proprio"].to(device)
        gt_actions = batch["action_chunk"].to(device)
        mask = batch["action_mask"].to(device)

        wrong_text_embed = torch.roll(text_embed, shifts=1, dims=0)

        correct_sampled = model.sample(vision_embed, text_embed, proprio, chunk_size, scheduler, num_inference_steps)
        wrong_sampled = model.sample(vision_embed, wrong_text_embed, proprio, chunk_size, scheduler, num_inference_steps)

        correct_se += (((correct_sampled - gt_actions) ** 2) * mask.unsqueeze(-1)).sum().item()
        wrong_se += (((wrong_sampled - gt_actions) ** 2) * mask.unsqueeze(-1)).sum().item()
        n += mask.sum().item() * gt_actions.shape[-1]

    return correct_se / max(n, 1), wrong_se / max(n, 1)


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    paths = resolve_paths(args.data)
    print(f"Training on {len(paths)} file(s): {[os.path.basename(p) for p in paths]}")

    dataset = LiberoClipDataset(paths, clip_cache_dir=args.clip_cache_dir, chunk_size=args.chunk_size)
    n_val = max(1, int(len(dataset) * args.val_fraction))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )
    print(f"Dataset: {len(dataset)} samples across {len(paths)} tasks ({n_train} train / {n_val} val)")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )

    model = VLADiffusionPolicy(proprio_dim=PROPRIO_DIM, action_dim=ACTION_DIM).to(args.device)
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
            vision_embed = batch["vision_embed"].to(args.device)
            text_embed = batch["text_embed"].to(args.device)
            proprio = batch["proprio"].to(args.device)
            actions = batch["action_chunk"].to(args.device)
            mask = batch["action_mask"].to(args.device)

            noise = torch.randn_like(actions)
            timesteps = torch.randint(
                0, NUM_TRAIN_TIMESTEPS, (actions.shape[0],), device=args.device
            ).long()
            noisy_actions = train_scheduler.add_noise(actions, noise, timesteps)

            pred_noise = model(
                noisy_actions, timesteps, vision_embed, text_embed, proprio,
                cond_dropout_prob=args.cond_dropout_prob,
            )
            loss = masked_mse(pred_noise, noise, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(model)
            train_loss += loss.item() * vision_embed.shape[0]
        train_loss /= n_train

        log = f"epoch {epoch:3d}  train_loss={train_loss:.5f}"
        if (epoch + 1) % args.sample_every == 0 or epoch == args.epochs - 1:
            correct_mse, wrong_mse = language_sensitivity_check(
                ema.shadow, val_loader, infer_scheduler, args.chunk_size, args.num_inference_steps, args.device
            )
            log += f"  correct_lang_mse={correct_mse:.5f}  wrong_lang_mse={wrong_mse:.5f}"
        print(log)

    torch.save(model.state_dict(), os.path.join(args.outdir, "last.pt"))
    torch.save(ema.shadow.state_dict(), os.path.join(args.outdir, "ema.pt"))
    print(f"Done. Checkpoints + action_stats.json written to {args.outdir}")


if __name__ == "__main__":
    main()
