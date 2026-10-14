<p align="center">
  <img src="assets/qantara-logo.svg" alt="Qantara" width="200">
</p>

# Qantara
### Bridge-Flow Training for Multi-Paradigm JEPA Inference

[Ruslan Rakhimov](https://github.com/rakhimovv), [George Bredis](https://github.com/Natyren), [Yuriy Maksyuta](https://github.com/ex7remum), [Daniil Gavrilov](https://github.com/kefirski)

*ICML 2026 Workshop on Decision-Making from Offline Datasets to Online Adaptation (DEMO)*

**Abstract:** Joint-Embedding Predictive Architectures (JEPAs) underpin a growing family of
latent world models for control from raw pixels. Published JEPA world models for pixel-based
control (PLDM, DINO-WM, LeWM, V-JEPA-2-AC) each commit at training time to a single inference
paradigm, trajectory optimisation in a learned dynamics model, so a checkpoint trained to plan
cannot be read off as a policy without retraining. Deferring that choice to inference would let
the constraints that bind at deployment, per-step latency and whether a goal observation can be
supplied, pick the path. We present **Qantara**, an end-to-end JEPA of about 21M parameters
whose joint training objective pairs a Brownian-bridge interpolant between consecutive clean
latents on the state axis with noise-to-data flow matching on the action axis. One checkpoint
serves three inference paradigms without retraining: latent planning, behaviour-cloning action
sampling, and a video-inverse composition that predicts the next latent without action
conditioning, then recovers the action bridging the two. Training samples the two noise levels
along the edges of the (action-noise, state-noise) square, one edge per inference dispatch, so
each path is trained where it is queried. Replacing that sampler with uniform sampling over the
square at matched compute drops Push-T planning from 90.1 to 53.3 success rate (SR). On the LeWM
control suite at a matched 10-epoch budget, Qantara averages 91.2 SR over the four environments
and reaches 93.7 on OGBench-Cube, +7.7 over DINO-WM and +17.0 over LeWM, each measured against
the stronger of that baseline's published score and our retrain of it. From the same weights the
two goal-blind paths, which receive no goal image, reach 82.1-83.2 SR on Push-T and 70.8-72.8 on
Cube at roughly 15-65x lower inference cost, and fall to near-noise on Reacher, whose released
training corpus comes from a random policy.

<p align="center">
   <b>[ 📄 <a href="https://arxiv.org/abs/2607.04978">Paper</a> | 🤗 <a href="https://huggingface.co/papers/2607.04978">HF Paper</a> | 🌐 <a href="https://corl-team.github.io/qantara/">Website</a> | 🤗 <a href="https://huggingface.co/t-tech/qantara-checkpoints">Checkpoints</a> | 🧵 <a href="https://x.com/rusrakhimov/status/2074847486288806306">Thread</a> ]</b>
</p>

<br>

This repository contains the model implementation, training and evaluation code, the
Hydra configs, and the figure/table scripts needed to reproduce the paper's numbers. It
is derived from the open-source [Le-WM](https://github.com/lucas-maes/le-wm) JEPA world
model (forked at `ca231f9`, March 2026); the predictor, training objective, and inference
paths were substantially modified for this work.

If you use this code or the released checkpoints, please cite:

```bibtex
@misc{qantara2026,
  title     = {Qantara: Bridge-Flow Training for Multi-Paradigm JEPA Inference},
  author    = {Rakhimov, Ruslan and Bredis, George and Maksyuta, Yuriy and Gavrilov, Daniil},
  year      = {2026},
  note      = {ICML 2026 Workshop on Decision-Making from Offline Datasets to Online Adaptation (DEMO); non-archival},
  eprint    = {2607.04978},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url       = {https://arxiv.org/abs/2607.04978},
}
```

## ⚙️ Installation

```bash
uv sync
```

Dependencies and the Python constraint (`>=3.10,<3.14`) are declared in `pyproject.toml`.

Set MuJoCo's rendering backend before running:

```bash
export MUJOCO_GL=egl       # default; requires graphics drivers
# export MUJOCO_GL=osmesa  # software-rendering fallback if EGL is unavailable
```

Two environment variables are read throughout:

```bash
export STABLEWM_HOME=/path/to/datasets_and_checkpoints   # data + checkpoint root (required)
export CKPT_REPO=t-tech/qantara-checkpoints         # HF repo holding the released checkpoints
```

## 🗃️ Data

The four evaluation environments use HDF5 datasets in the format consumed by the
`stable_worldmodel` library:

| Task     | Source                          |
|----------|---------------------------------|
| PushT    | LeRobot / D4RL (PushT expert)   |
| TwoRoom  | OGBench (point-maze two-room)   |
| Cube     | OGBench (cube-single expert)    |
| Reacher  | DMControl (Reacher hard)        |

Place the `.h5` files under `$STABLEWM_HOME`. Dataset names are referenced without the
`.h5` extension in `config/train/data/*.yaml`.

## 🏋️ Training

Hydra configs:

- `config/train/lewm.yaml`: baseline JEPA world model (the predecessor we compare against).
- `config/train/qantara.yaml`: our model; inherits `lewm`.

```bash
uv run python train.py --config-name=qantara data=<DATA> subdir=<run-name>
```

`<DATA>` selects the dataset config from `config/train/data/`:

| Task    | `data=` value | Config file    |
|---------|---------------|----------------|
| PushT   | `pusht`       | `pusht.yaml`   |
| TwoRoom | `tworoom`     | `tworoom.yaml` |
| Cube    | `ogb`         | `ogb.yaml`     |
| Reacher | `dmc`         | `dmc.yaml`     |

`subdir` controls the output directory under `$STABLEWM_HOME/`. Two checkpoints are
written at the end of training: `{subdir}/qantara_object.ckpt` (pickled model, used by
eval) and `{subdir}/qantara_weights.pt` (state-dict).

The headline results use three training seeds (`seed=11,22,33`) per environment.

## 🎯 Evaluation

Per-environment eval configs live in `config/eval/`. The same checkpoint is queried in
three inference paradigms:

```bash
# 1. Latent planning (CEM), the headline planner
uv run python eval.py --config-name=pusht policy=<run-name>/qantara

# 2. Behaviour cloning (goal-blind)
uv run python eval.py --config-name=pusht policy=<run-name>/qantara +policy_kind=bc +bc_kind=bc

# 3. Video–inverse dynamics (goal-blind)
uv run python eval.py --config-name=pusht policy=<run-name>/qantara +policy_kind=bc +bc_kind=video_idm
```

`policy` is the checkpoint path relative to `$STABLEWM_HOME`, without the
`_object.ckpt` suffix.

## 🤗 Checkpoints

The headline checkpoints (4 envs × 3 seeds, for Qantara and the LeWM baseline) are
released on the Hugging Face Hub under flat naming:

```
qantara-<env>-s<seed>.ckpt      lewm-<env>-s<seed>.ckpt
# env  ∈ {pusht, tworoom, cube, reacher}      seed ∈ {11, 22, 33}
```

Download one directly:

```bash
huggingface-cli download "$CKPT_REPO" qantara-pusht-s11.ckpt --local-dir /tmp/ckpts
```

The figure/table scripts below auto-download what they need from `$CKPT_REPO`.

## 📈 Reproducing the paper figures and tables

| Paper artifact                        | Command                                                                                  |
|---------------------------------------|------------------------------------------------------------------------------------------|
| Decoded rollout figures               | `uv run python scripts/make_rollout_figure.py …` → `uv run python scripts/plot_rollout_figure.py --npz figs/rollout/<env>.npz --out <env>.pdf` |
| Inference cost (15–65× speedup)       | `bash scripts/run_timing.sh` → `uv run python scripts/agg_timing.py <roots…>` |

The success-rate and ablation tables are produced directly by the Training + Evaluation
commands above (multiple seeds × envs × the three dispatches).

## 🗂️ Repository layout

```
train.py            training entrypoint (Hydra)
eval.py             goal-conditioned planning eval (CEM / BC / video-inverse)
jepa.py             JEPA wrapper + rollout
module.py           model components (predictor, decoder, time embedding)
utils.py            data + checkpoint utilities
config/train/       training configs (qantara, lewm, data/*)
config/eval/        evaluation configs (per env, solvers, launcher)
scripts/            figure/table reproduction scripts
```

## 🙏 Acknowledgments

Built on top of [Le-WM](https://github.com/lucas-maes/le-wm) and the
`stable_worldmodel` / `stable_pretraining` libraries.

## 📜 License

MIT. See `LICENSE`.
