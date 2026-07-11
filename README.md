# vla-diffusion-libero

A small, from-scratch language-conditioned VLA: frozen vision-language encoders feeding a
diffusion-transformer action head, trained on [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
in simulation. Architecturally this is a scaled-down version of what pi0 / RDT-1B / Octo do
(VLM conditioning + diffusion policy action decoding) — the goal is to demonstrate the ability
to actually build this combination, not just fine-tune an existing checkpoint or benchmark two
off-the-shelf baselines against each other.

## Architecture

```
RGB (agentview + wrist cam) -> frozen vision encoder (CLIP ViT-B/32)   -+
Language instruction        -> frozen text encoder (CLIP text tower)   -+-> fusion transformer (learned)
Proprioception (joint/eef/gripper) -> small MLP                        -+       |
                                                                                 v
                                                              conditioning vector z
                                                                                 |
                                                            FiLM conditioning    v
                                                    diffusion action head (1D temporal denoiser)
                                                    - DDPM eps-prediction loss on action chunks
                                                    - DDIM sampling (~10 steps) at inference
                                                    - classifier-free guidance on language
```

Vision/language backbones are frozen and their embeddings are precomputed/cached offline —
training only updates the fusion transformer + diffusion head (~10-30M params). This is the
main lever that keeps this trainable on a single rented GPU in hours, not days.

## Compute-minimized MVP scope

Full LIBERO-Object is 10 tasks x 50 demos (~7.5GB, one ~700-800MB hdf5 per task). The MVP
trains on a **3-4 task subset** (~2-3GB) with cached frozen embeddings, bf16, and a small
action head. Scale up to the full suite only after the pipeline is validated end-to-end.

## Repo layout

```
src/vla_diffusion/
  data/       LIBERO dataset wrappers, embedding caching, action normalization
  models/     vision/text encoder wrappers, fusion trunk, diffusion action head
  training/   train loop, configs
  eval/       closed-loop rollout harness in LIBERO/robosuite
scripts/
  setup_libero.sh          clone + install LIBERO into the uv venv, non-interactive config
  download_libero_data.py  pull a task subset (not the whole suite) from Hugging Face
  inspect_libero_data.py   Week 1 data-debugging pass: episode lengths, action stats,
                            length-outlier flagging, sample frames
configs/      Hydra configs (ablations become config swaps)
notebooks/    exploratory work
```

## Setup

```bash
uv sync                        # installs the pinned project deps into .venv
scripts/setup_libero.sh        # clones LIBERO, installs it into .venv, seeds non-interactive config
export LIBERO_CONFIG_PATH="$(pwd)/.libero_config"   # add to shell profile too
export MUJOCO_GL=egl                                # headless rendering

# pull a small task subset instead of the full 7.5GB suite
uv run python scripts/download_libero_data.py --suite libero_object --num-tasks 3

# Week 1 data-debugging pass on a downloaded task
uv run python scripts/inspect_libero_data.py --file data/libero_datasets/libero_object/<task>_demo.hdf5

# Week 2: train + closed-loop-eval the BC-MLP baseline
uv run python -m vla_diffusion.training.train_bc \
    --data data/libero_datasets/libero_object/*.hdf5 --outdir outputs/bc_mlp_run1
uv run python -m vla_diffusion.eval.rollout \
    --checkpoint outputs/bc_mlp_run1/best.pt --task-name <task_substring>
```

Note: `torch` pulls the standard PyPI CUDA-enabled Linux wheel by default, so no separate
reinstall should be needed on the vast.ai box as long as its driver supports the installed
CUDA version. If `torch.cuda.is_available()` is `False` there, reinstall with an
`--index-url` matching that box's driver (see pytorch.org's install matrix).

## Gotchas hit while building this (kept here since they cost real debugging time)

- **LIBERO's editable install silently installs nothing.** Its `libero/` package dir has no
  top-level `__init__.py`, so `pip install -e .`'s default PEP 660 finder registers an empty
  package map (`import libero` fails with no error at install time). Fixed by installing with
  `--config-settings editable_mode=compat`, which falls back to a plain sys.path insertion.
- **robosuite>=1.5 breaks LIBERO's imports outright** — `single_arm_env.py` was renamed to
  `manipulation_env.py`. Pinned to `robosuite==1.4.1`, the version LIBERO's own (very stale)
  `requirements.txt` targets.
- **robosuite 1.4.1 + a modern mujoco crashes at `env.reset()`** with a `TypeError` in
  `mj_fullM()` — the C binding signature changed. Pinned `mujoco==3.1.6`, contemporaneous with
  robosuite 1.4.1's release; verified working end-to-end (reset/step/closed-loop rollout).
- **`bddl`, `robomimic`, `gym`, `future`, `easydict`, `cloudpickle`, `thop` are all real runtime
  imports** in LIBERO's code that its `setup.py` declares zero dependencies for. Found by
  actually running the pipeline and fixing each `ModuleNotFoundError` as it surfaced, then
  pinned individually with `--no-deps` so their own stale transitive pins (e.g. `numpy==1.22.4`)
  don't clobber the modern stack the rest of this project depends on.
- **A stray `VIRTUAL_ENV` env var from an unrelated project silently redirects `uv pip
  install`** to the wrong virtualenv (`uv add` ignores it and warns; `uv pip install` obeys it
  with no warning). `unset VIRTUAL_ENV` before any `uv pip` command in this repo.

## Status

- [x] LIBERO installed, benchmark suites enumerable, non-interactive config seeded
- [x] Per-task download helper (avoids pulling the full suite)
- [x] Data-inspection script validated on a real task file (`libero_object` /
      "pick up the salad dressing and place it in the basket", 50 demos) — already flagged
      5/50 episodes as length outliers worth a closer look before treating them as clean demos
- [x] Dataset + dataloader with action chunking and normalization (Week 2)
- [x] BC-MLP baseline validated end-to-end: trains (loss decreases), checkpoints, and its
      checkpoint runs a real closed-loop rollout in the robosuite/MuJoCo sim with correct
      proprio reconstruction from live obs. (3-epoch CPU smoke test only, 0% success as
      expected -- real training happens on the GPU box in Week 5's full run.)
- [ ] Diffusion action head, single-task overfit sanity check (Week 3)
- [ ] Vision-language fusion trunk + classifier-free guidance, multi-task training (Week 4)
- [ ] Full training run + closed-loop eval + ablations (DDIM steps, chunk horizon, CFG on/off)
      (Week 5)
- [ ] README writeup with plots/rollout GIFs, optional ROS2 wrapper, optional second suite
      (Week 6)
