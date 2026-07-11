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

# pull a small task subset instead of the full 7.5GB suite
uv run python scripts/download_libero_data.py --suite libero_object --num-tasks 3

# Week 1 data-debugging pass on a downloaded task
uv run python scripts/inspect_libero_data.py --file data/libero_datasets/libero_object/<task>_demo.hdf5
```

Note: `torch` pulls the standard PyPI CUDA-enabled Linux wheel by default, so no separate
reinstall should be needed on the vast.ai box as long as its driver supports the installed
CUDA version. If `torch.cuda.is_available()` is `False` there, reinstall with an
`--index-url` matching that box's driver (see pytorch.org's install matrix).

## Status

- [x] LIBERO installed, benchmark suites enumerable, non-interactive config seeded
- [x] Per-task download helper (avoids pulling the full suite)
- [x] Data-inspection script validated on a real task file (`libero_object` /
      "pick up the salad dressing and place it in the basket", 50 demos) — already flagged
      5/50 episodes as length outliers worth a closer look before treating them as clean demos
- [ ] Dataset + dataloader with action chunking and normalization (Week 2)
- [ ] BC-MLP baseline to validate the pipeline end-to-end (Week 2)
- [ ] Diffusion action head, single-task overfit sanity check (Week 3)
- [ ] Vision-language fusion trunk + classifier-free guidance, multi-task training (Week 4)
- [ ] Full training run + closed-loop eval + ablations (DDIM steps, chunk horizon, CFG on/off)
      (Week 5)
- [ ] README writeup with plots/rollout GIFs, optional ROS2 wrapper, optional second suite
      (Week 6)
