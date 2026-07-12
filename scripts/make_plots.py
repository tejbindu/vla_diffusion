#!/usr/bin/env python
"""Generate the README's result plots from the ablation JSON files
(outputs/vla_sanity/ablation_*.json) plus the Week 3/4 training histories.

The Week 3/4 numbers are hardcoded below because those particular runs
predate the history.json logging this project now writes automatically
(added in Week 6) -- they're the exact per-epoch values printed during
those training runs, transcribed once here rather than re-run. Any future
run (including the GPU one) will have a real outputs/<run>/history.json to
plot directly instead.

    uv run python scripts/make_plots.py
"""
import json
import os

import matplotlib.pyplot as plt

OUTDIR = "assets/plots"
ABLATION_DIR = "outputs/vla_sanity"

# -- palette (see dataviz skill references/palette.md) --
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
AQUA = "#1baf7a"
ORANGE = "#eb6834"

plt.rcParams.update({
    "font.family": "sans-serif",
    "text.color": INK,
    "axes.edgecolor": GRIDLINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

# Week 3: diffusion-only policy, single task (salad_dressing), 30 CPU epochs
WEEK3_LOSS = [
    0.24154, 0.13635, 0.11910, 0.10931, 0.10183, 0.09882, 0.09517, 0.09084, 0.09008, 0.08569,
    0.08563, 0.08395, 0.08427, 0.08357, 0.07870, 0.08000, 0.07724, 0.07795, 0.07735, 0.07315,
    0.07600, 0.07191, 0.07236, 0.07257, 0.07150, 0.06968, 0.06879, 0.06925, 0.06752, 0.06701,
]

# Week 4: full VLA, 3 tasks, 30 CPU epochs
WEEK4_LOSS = [
    0.15764, 0.09225, 0.08274, 0.07285, 0.07066, 0.06616, 0.06407, 0.06229, 0.06273, 0.06213,
    0.05900, 0.05859, 0.05610, 0.05657, 0.05670, 0.05498, 0.05520, 0.05437, 0.05478, 0.05499,
    0.05310, 0.05345, 0.05149, 0.05156, 0.05094, 0.05057, 0.05157, 0.04977, 0.04968, 0.04851,
]

# Week 4 language-sensitivity check, at the 3 epochs it was measured
LANG_SENS_EPOCHS = [9, 19, 29]
LANG_SENS_CORRECT = [0.17224, 0.10029, 0.07620]
LANG_SENS_WRONG = [0.15003, 0.11496, 0.08831]


def style_axes(ax, show_x_grid=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(GRIDLINE)
    ax.tick_params(length=0)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)
    if not show_x_grid:
        ax.xaxis.grid(False)


def plot_training_loss():
    fig, ax = plt.subplots(figsize=(7, 4.2))
    epochs = list(range(30))
    ax.plot(epochs, WEEK3_LOSS, color=BLUE, linewidth=2, solid_capstyle="round",
            marker="o", markersize=4, markevery=[len(epochs) - 1])
    ax.plot(epochs, WEEK4_LOSS, color=AQUA, linewidth=2, solid_capstyle="round",
            marker="o", markersize=4, markevery=[len(epochs) - 1])

    ax.annotate("Week 3: diffusion policy\n(1 task)", xy=(29, WEEK3_LOSS[-1]),
                xytext=(18, 0.145), color=BLUE, fontsize=10, fontweight="medium")
    ax.annotate("Week 4: full VLA\n(3 tasks + language)", xy=(29, WEEK4_LOSS[-1]),
                xytext=(18, 0.205), color=AQUA, fontsize=10, fontweight="medium")

    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss (masked ε-prediction MSE)")
    ax.set_title("Training loss, CPU sanity runs", loc="left", color=INK, fontsize=13, fontweight="semibold")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "training_loss.png"), dpi=160)
    plt.close(fig)


def plot_language_sensitivity():
    fig, ax = plt.subplots(figsize=(6, 4.2))
    x = range(len(LANG_SENS_EPOCHS))
    width = 0.32

    bars_correct = ax.bar([i - width / 2 for i in x], LANG_SENS_CORRECT, width=width,
                           color=BLUE, label="correct language")
    bars_wrong = ax.bar([i + width / 2 for i in x], LANG_SENS_WRONG, width=width,
                         color=ORANGE, label="wrong (shuffled) language")

    for bars in (bars_correct, bars_wrong):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003, f"{b.get_height():.3f}",
                    ha="center", va="bottom", fontsize=9, color=INK_SECONDARY)

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"epoch {e}" for e in LANG_SENS_EPOCHS])
    ax.set_ylabel("DDIM-sample MSE vs. ground truth")
    ax.set_title("Does the model actually use language?", loc="left", color=INK, fontsize=13, fontweight="semibold")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "language_sensitivity.png"), dpi=160)
    plt.close(fig)


def plot_ddim_ablation():
    with open(os.path.join(ABLATION_DIR, "ablation_ddim_cfg.json")) as fp:
        rows = json.load(fp)
    rows = [r for r in rows if r["guidance_scale"] == 1.0]
    rows.sort(key=lambda r: r["num_inference_steps"])
    steps = [r["num_inference_steps"] for r in rows]
    mse = [r["sample_mse"] for r in rows]
    latency = [r["mean_latency_ms_per_chunk"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.0))

    ax = axes[0]
    ax.plot(steps, mse, color=BLUE, linewidth=2, marker="o", markersize=7, solid_capstyle="round")
    ax.text(steps[-1], mse[-1] + 0.05, f"{mse[-1]:.3f}", color=INK_SECONDARY, fontsize=9, ha="center")
    ax.set_xscale("log")
    ax.set_xticks(steps)
    ax.set_xticklabels(steps)
    ax.set_xlabel("DDIM inference steps")
    ax.set_ylabel("sample MSE vs. ground truth")
    ax.set_title("Quality vs. steps", loc="left", color=INK, fontsize=12, fontweight="semibold")
    style_axes(ax)

    ax = axes[1]
    ax.plot(steps, latency, color=AQUA, linewidth=2, marker="o", markersize=7, solid_capstyle="round")
    ax.text(steps[-1], latency[-1] + 0.4, f"{latency[-1]:.1f} ms", color=INK_SECONDARY, fontsize=9, ha="right")
    ax.set_xscale("log")
    ax.set_xticks(steps)
    ax.set_xticklabels(steps)
    ax.set_xlabel("DDIM inference steps")
    ax.set_ylabel("latency per chunk (ms, CPU)")
    ax.set_title("Cost vs. steps", loc="left", color=INK, fontsize=12, fontweight="semibold")
    style_axes(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.suptitle("DDIM step count: quality plateaus by ~3-5 steps, cost keeps climbing",
                 x=0.02, ha="left", color=INK, fontsize=13, fontweight="semibold", y=0.98)
    fig.savefig(os.path.join(OUTDIR, "ddim_ablation.png"), dpi=160)
    plt.close(fig)


def plot_exec_horizon_ablation():
    with open(os.path.join(ABLATION_DIR, "ablation_exec_horizon.json")) as fp:
        rows = json.load(fp)

    horizons = sorted({r["exec_horizon"] for r in rows})
    max_steps = 100  # from the ablation run's --max-steps
    total_compute_s = []
    for h in horizons:
        matching = [r["mean_sample_latency_s"] for r in rows if r["exec_horizon"] == h]
        mean_latency = sum(matching) / len(matching)
        num_replans = max_steps / h
        total_compute_s.append(mean_latency * num_replans)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    bars = ax.bar([str(h) for h in horizons], total_compute_s, width=0.5, color=BLUE)
    for b, v in zip(bars, total_compute_s):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}s", ha="center",
                va="bottom", fontsize=10, color=INK_SECONDARY)

    ax.set_xlabel("exec_horizon (steps executed per replan)")
    ax.set_ylabel("total diffusion-sampling compute\nper 100-step episode (s, CPU)")
    ax.set_title("Replanning less often is ~linearly cheaper", loc="left",
                 color=INK, fontsize=13, fontweight="semibold")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "exec_horizon_ablation.png"), dpi=160)
    plt.close(fig)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    plot_training_loss()
    plot_language_sensitivity()
    plot_ddim_ablation()
    plot_exec_horizon_ablation()
    print(f"Wrote plots to {OUTDIR}/")


if __name__ == "__main__":
    main()
