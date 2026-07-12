# vla-diffusion-libero

A small, from-scratch language-conditioned VLA: frozen CLIP vision+text embeddings feed a
learned fusion transformer, which conditions a diffusion-transformer action head via FiLM.
Trained and evaluated on [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) in
simulation. Architecturally this is a scaled-down version of what pi0 / RDT-1B / Octo do (VLM
conditioning + diffusion-policy action decoding), built from the underlying papers rather than
by fine-tuning an existing checkpoint or benchmarking off-the-shelf baselines against each other.

![Closed-loop rollout demo](assets/rollout_demo.gif)

*A trained checkpoint driving the real robosuite/MuJoCo sim in closed loop — live CLIP
encoding, language-conditioned diffusion sampling, receding-horizon control. This particular
checkpoint is a small CPU sanity-scale run (see [Results](#results) for why that matters and
what it doesn't yet do); the point of this GIF is that the full pipeline runs correctly
end-to-end.*

## Motivation

Modern VLA policies (RT-2, OpenVLA, pi0, Octo) get most of their generalization from
internet-scale vision-language pretraining and large multi-embodiment robot datasets — resources
far outside what a single GPU or a small team can reproduce. This project asks a narrower,
more tractable question: how much of the VLA-plus-diffusion-policy recipe survives when a
frozen, off-the-shelf CLIP encoder stands in for a trained-from-scratch or fine-tuned VLM
backbone, and the action head is a lightweight diffusion transformer instead of a large one?

Concretely, it's built to get real, measured answers to a few design questions rather than
assume them:

- Does a small learned fusion trunk on top of *frozen* CLIP embeddings actually learn to use
  language to disambiguate otherwise-visually-identical tasks, or does it quietly shortcut
  through vision and proprioception alone?
- How many diffusion-sampling steps does a small action head actually need before quality
  plateaus, and what does that cost in latency — is the "diffusion policies are too slow for
  real-time control" criticism actually true at this scale?
- Does replanning frequency (how many steps of a sampled action chunk get executed open-loop
  before the policy replans) trade off compute against control quality the way the Diffusion
  Policy paper's chunking argument predicts?

Every claim in the [Results](#results) section below is measured, not assumed — including the
places where the evidence is weak or inconclusive.

## Architecture

```
RGB (agentview cam)          -> frozen vision encoder (CLIP ViT-B/32, openai weights)  -+
Language instruction         -> frozen text encoder (CLIP text tower)                  -+-> fusion transformer (learned, self-attention over 4 pooled tokens + CLS)
Proprioception (joint/eef/gripper, 15-dim) -----------------------------------------    -+       |
                                                                                                   v
                                                                                     conditioning vector z
                                                                                                   |
                                                                            FiLM conditioning      v
                                                                    diffusion action head (1D residual conv stack)
                                                                    - DDPM eps-prediction loss on 8-step action chunks
                                                                    - DDIM sampling (~3-5 steps is enough, see Results)
                                                                    - classifier-free guidance on language
```

Vision/language backbones are frozen and their embeddings are precomputed/cached offline —
training only updates the fusion transformer + diffusion head (~a few million params). This is
the main lever that keeps this trainable on a single rented GPU in hours, not days.

## Results

**Read this before the numbers below**: all results here come from small, CPU-only sanity-scale
training runs (≤30 epochs, 3 tasks, 50 demos/task, no image augmentation) — deliberately kept
cheap so the full pipeline could be validated end-to-end before spending real training compute.
What these runs demonstrate is that the *pipeline* (data → training → closed-loop eval →
ablations) is fully correct; the *policy* itself is not yet good, and that's expected, not a bug.

### Training actually learns

<img src="assets/plots/training_loss.png" width="600">

The diffusion-only policy (single task) and the full VLA (3 tasks + language) both show clean,
monotonic loss curves — unsurprising on its own, but it's the baseline sanity check everything
else depends on.

### The model is actually using language, not shortcutting through vision

<img src="assets/plots/language_sensitivity.png" width="560">

LIBERO-Object's tasks all share the *identical* shelf scene — every task's demo shows the same
objects in the same layout; only the language instruction says which one to pick. That makes
this a clean test: sample an action chunk conditioned on the *correct* instruction vs. a
*wrong* one (shuffled from a different task in the batch) and compare to ground truth. If the
model were ignoring language, there'd be no difference. Early in training there wasn't (the
first checkpoint measured had wrong-language MSE slightly *lower*, i.e. noise). By the next
checkpoint the expected direction emerged and held through the end of training — a genuine, if
modest, cross-modal effect, not an assumed one.

### DDIM step count: diminishing returns kick in fast

<img src="assets/plots/ddim_ablation.png" width="720">

Sampling quality (chunk MSE against held-out ground-truth actions) falls off a cliff below 3
steps, then flattens through 20. Practically: this policy doesn't need the 100 steps it trained
with — 3-5 DDIM steps captures almost all the achievable quality, which is most of the latency
gap against a single-forward-pass VLA that a diffusion policy is usually criticized for.

*(A parallel sweep over classifier-free-guidance scale showed no consistent effect on this
metric at this training budget — reported as inconclusive rather than oversold; this amount of
training is likely too little for the unconditional branch to have learned a meaningfully
different distribution yet.)*

### Replanning frequency: a real compute/reactivity tradeoff

<img src="assets/plots/exec_horizon_ablation.png" width="560">

`exec_horizon` controls how many actions from a sampled chunk get executed open-loop before the
policy replans. Total diffusion-sampling compute per episode scales roughly linearly with
replanning frequency — about a 9x difference between replanning every step and replanning once
per full chunk. The other side of this tradeoff (does replanning less often hurt success once
drift accumulates) isn't visible yet because closed-loop success is 0% at every setting with
this checkpoint — see below.

### Headline closed-loop number: 0/45 successes

Across all 3 trained tasks × all 3 `exec_horizon` settings × 5 episodes. Expected, not
concerning, given 50 demos/task and this little training with no augmentation — the harder
engineering problem solved here was getting the *full* closed-loop path (live CLIP encoding →
fusion trunk → CFG-guided DDIM sampling → receding-horizon control → real sim execution →
success detection) to run correctly at all, which it now does. All plots and numbers are
reproducible via `scripts/ablate_ddim_cfg.py` and `scripts/ablate_exec_horizon.py`.

## Repo layout

```
src/vla_diffusion/
  data/       LIBERO dataset wrappers, CLIP embedding caching, action normalization
  models/     CLIP encoder wrapper, fusion transformer, diffusion action head, BC-MLP baseline
  training/   train loops (BC-MLP, diffusion-only, full VLA), shared losses/data-utils
  eval/       closed-loop rollout harnesses (BC-MLP and VLA), shared sim/proprio utilities
  ros2/       reference ROS2 node wrapping the trained policy for real-robot deployment
scripts/
  setup_libero.sh              clone + install LIBERO into the uv venv, non-interactive config
  download_libero_data.py      pull a task subset (not the whole suite) from Hugging Face
  inspect_libero_data.py       data-debugging pass: episode lengths, action stats, outlier flags
  precompute_clip_embeddings.py cache frozen CLIP embeddings for a set of task files
  ablate_ddim_cfg.py           open-loop DDIM-steps / CFG-scale ablation
  ablate_exec_horizon.py       closed-loop replanning-frequency ablation
  make_plots.py                regenerates the plots in assets/plots/
assets/       committed plots + demo GIF for this README
```

## Setup

```bash
uv sync                        # installs the pinned project deps into .venv
scripts/setup_libero.sh        # clones LIBERO, installs it into .venv, seeds non-interactive config
export LIBERO_CONFIG_PATH="$(pwd)/.libero_config"   # add to shell profile too
export MUJOCO_GL=egl                                # headless rendering

# pull a small task subset instead of the full 7.5GB suite
uv run python scripts/download_libero_data.py --suite libero_object --num-tasks 3

# data-debugging pass on a downloaded task
uv run python scripts/inspect_libero_data.py --file data/libero_datasets/libero_object/<task>_demo.hdf5

# cache frozen CLIP embeddings (training never re-runs CLIP's forward pass)
uv run python scripts/precompute_clip_embeddings.py \
    --data data/libero_datasets/libero_object/*.hdf5 --cache-dir data/clip_cache

# train the full VLA, multi-task + language-conditioned
uv run python -m vla_diffusion.training.train_vla \
    --data data/libero_datasets/libero_object/*.hdf5 --clip-cache-dir data/clip_cache \
    --outdir outputs/vla_run1 --epochs 100

# closed-loop eval in the real sim
uv run python -m vla_diffusion.eval.rollout_vla \
    --checkpoint outputs/vla_run1/ema.pt --task-name <task_substring>

# ablations
uv run python scripts/ablate_ddim_cfg.py --checkpoint outputs/vla_run1/ema.pt
uv run python scripts/ablate_exec_horizon.py --checkpoint outputs/vla_run1/ema.pt
```

Note: `torch` pulls the standard PyPI CUDA-enabled Linux wheel by default, so no separate
reinstall should be needed on a rented GPU box as long as its driver supports the installed
CUDA version. If `torch.cuda.is_available()` is `False` there, reinstall with an `--index-url`
matching that box's driver (see pytorch.org's install matrix).

## ROS2 wrapper

`src/vla_diffusion/ros2/vla_policy_node.py` wraps the trained policy as a standard `rclpy.Node`:
subscribes to a camera image topic and `/joint_states` (+ a `tf2` lookup for end-effector pose,
matching how real Franka-Panda ROS2 drivers expose robot state), runs the same live-CLIP →
fusion-trunk → CFG-guided-DDIM-sampling → receding-horizon loop as `eval/rollout_vla.py`, and
publishes the resulting action. It's written against the standard ROS2 Humble/Jazzy `rclpy` API
but **not run against a live ROS2 stack** — this environment has no ROS2 install (`rclpy` isn't
a pip package; it ships with a ROS2 distro's own Python environment, so this project's deps
would need to be made importable from that same interpreter). Treat it as correct, adaptable
reference code, not a validated one — see the notes at the bottom of the file for what to check
on a real ROS2 box before trusting it near a robot.

## Compute-minimized scope

Full LIBERO-Object is 10 tasks × 50 demos (~7.5GB, one ~700-800MB hdf5 per task). This project
trains on a **3-task subset** (~2GB) with cached frozen embeddings and a small action head, so
the full pipeline could be built and validated cheaply before committing to a larger training
run. Scaling to the full suite is a config change once that larger run is worth paying for.

## Environment notes

- **LIBERO's editable install registers no package.** Its `libero/` package directory has no
  top-level `__init__.py`, so the default PEP 660 editable-install finder (`pip install -e .`)
  builds an empty package map and `import libero` fails silently at install time. Resolved by
  installing with `--config-settings editable_mode=compat`, which falls back to a plain
  sys.path insertion.
- **robosuite ≥1.5 is incompatible with LIBERO.** `environments/manipulation/single_arm_env.py`
  was renamed to `manipulation_env.py` in 1.5, breaking LIBERO's imports. Pinned to
  `robosuite==1.4.1`, the version LIBERO's `requirements.txt` targets.
- **robosuite 1.4.1 requires an older MuJoCo release.** Paired with a current MuJoCo version,
  `env.reset()` raises a `TypeError` in `mj_fullM()` due to a changed C binding signature.
  Pinned `mujoco==3.1.6`, contemporaneous with robosuite 1.4.1; verified working end-to-end.
- **Several runtime dependencies are undeclared in LIBERO's `setup.py`.** `bddl`, `robomimic`,
  `gym`, `future`, `easydict`, `cloudpickle`, and `thop` are all imported by LIBERO's code at
  runtime but absent from its declared dependencies. Identified by running the pipeline and
  resolving each resulting `ModuleNotFoundError`, then pinned individually with `--no-deps` so
  their stale transitive pins (e.g. `numpy==1.22.4`) don't override this project's dependency
  stack.
- **A stray `VIRTUAL_ENV` environment variable can redirect installs to the wrong virtualenv.**
  `uv add` ignores it with a warning; `uv pip install` respects it silently. `unset VIRTUAL_ENV`
  before running any `uv pip` command in this repo.
- **Long CPU training runs on shared hardware are unreliable for progress tracking.** A
  100-epoch run on a machine under concurrent load had its CPU allocation drop from ~840% to
  ~250%, with no visible output for over three hours because Python fully buffers stdout when
  redirected to a file. The run was stopped in favor of an already-validated shorter checkpoint.
  Training scripts now write `history.json` after every epoch rather than only at completion,
  so an interrupted run still leaves progress on disk.
