"""Week 2: train the BC-MLP baseline on one or more LIBERO task hdf5 files.

    uv run python -m vla_diffusion.training.train_bc \
        --data data/libero_datasets/libero_object/*.hdf5 \
        --outdir outputs/bc_mlp_run1 --epochs 50

This exists to validate the full data -> train -> checkpoint pipeline before
the diffusion action head (Week 3) replaces the MLP head.
"""
import argparse
import glob
import json
import os

import torch
from torch.utils.data import DataLoader, random_split

from vla_diffusion.data.libero_dataset import ACTION_DIM, PROPRIO_DIM, LiberoChunkDataset
from vla_diffusion.models.bc_mlp import BCMLPPolicy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True, help="hdf5 file(s) or glob pattern(s)")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def masked_mse(pred, target, mask):
    # mask: (B, T) -> broadcast over the action-dim axis
    err = (pred - target) ** 2 * mask.unsqueeze(-1)
    return err.sum() / mask.sum().clamp(min=1) / pred.shape[-1]


def resolve_paths(patterns):
    paths = []
    for p in patterns:
        matches = sorted(glob.glob(p))
        paths.extend(matches if matches else [p])
    if not paths:
        raise SystemExit(f"No files matched: {patterns}")
    return paths


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

    model = BCMLPPolicy(
        proprio_dim=PROPRIO_DIM, action_dim=ACTION_DIM, chunk_size=args.chunk_size
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    with open(os.path.join(args.outdir, "action_stats.json"), "w") as fp:
        json.dump(dataset.action_stats_dict(), fp, indent=2)
    with open(os.path.join(args.outdir, "config.json"), "w") as fp:
        json.dump(vars(args) | {"data_paths": paths}, fp, indent=2)

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            image = batch["image"].to(args.device)
            proprio = batch["proprio"].to(args.device)
            action_chunk = batch["action_chunk"].to(args.device)
            mask = batch["action_mask"].to(args.device)

            pred = model(image, proprio)
            loss = masked_mse(pred, action_chunk, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * image.shape[0]
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                image = batch["image"].to(args.device)
                proprio = batch["proprio"].to(args.device)
                action_chunk = batch["action_chunk"].to(args.device)
                mask = batch["action_mask"].to(args.device)
                pred = model(image, proprio)
                val_loss += masked_mse(pred, action_chunk, mask).item() * image.shape[0]
        val_loss /= n_val

        print(f"epoch {epoch:3d}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(args.outdir, "best.pt"))

    torch.save(model.state_dict(), os.path.join(args.outdir, "last.pt"))
    print(f"Done. Checkpoints + action_stats.json written to {args.outdir}")


if __name__ == "__main__":
    main()
