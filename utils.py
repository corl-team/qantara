import logging
import os
import sys
import warnings

import numpy as np
import torch
from pathlib import Path
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback


def resolve_num_gpus(trainer_cfg):
    """Resolve number of GPUs from Hydra trainer config (before Trainer exists)."""
    devices = trainer_cfg.get("devices", "auto")
    if devices == "auto" or devices is None:
        return torch.cuda.device_count() or 1
    try:
        return len(devices)  # list/tuple/ListConfig
    except TypeError:
        return int(devices)


def is_rank_zero():
    """True on rank 0 (or single-GPU). Works before and after dist init."""
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


def adjust_batch_size_for_gpus(cfg):
    """Divide loader.batch_size by num_gpus so scripts spec global batch.

    Lightning wraps each DataLoader with a DistributedSampler, so per-rank scaling
    is the only DDP adjustment we owe.
    """
    from omegaconf import open_dict
    num_gpus = resolve_num_gpus(cfg.trainer)
    if num_gpus <= 1:
        return num_gpus

    with open_dict(cfg):
        assert cfg.loader.batch_size % num_gpus == 0, \
            f"loader.batch_size ({cfg.loader.batch_size}) must be divisible by num_gpus ({num_gpus})"
        cfg.loader.batch_size //= num_gpus

    if is_rank_zero():
        print(f"Multi-GPU: {num_gpus} GPUs, per-rank loader.batch_size={cfg.loader.batch_size}")
    return num_gpus


def enable_cem_progress(every: int = 5):
    """Print a heartbeat every `every` CEM iterations so long solves don't look hung.
    Wraps `model.get_cost` (called once per inner CEM step) — works for both
    EpochPlanEvalCallback and eval.py since they share the same solver class.
    """
    import time as _time
    from stable_worldmodel.solver import cem as _cem
    if getattr(_cem.CEMSolver, "_progress_patched", False):
        return
    orig_solve = _cem.CEMSolver.solve

    def solve(self, *args, **kwargs):
        orig_get_cost = self.model.get_cost
        state = {"i": 0, "t0": _time.time()}
        # get_cost is called once per inner CEM iter, and the solver loops over
        # env batches of size `batch_size` — total calls = n_batches * n_steps.
        n_batches = (self.n_envs + self.batch_size - 1) // self.batch_size
        total = n_batches * self.n_steps

        def get_cost(*a, **kw):
            state["i"] += 1
            if state["i"] % every == 0:
                el = _time.time() - state["t0"]
                print(f"  CEM step {state['i']}/{total} ({el:.1f}s)", flush=True)
            return orig_get_cost(*a, **kw)

        self.model.get_cost = get_cost
        try:
            return orig_solve(self, *args, **kwargs)
        finally:
            self.model.get_cost = orig_get_cost

    _cem.CEMSolver.solve = solve
    _cem.CEMSolver._progress_patched = True


def silence_warnings():
    """Suppress verbose startup logs from Lightning, stable_pretraining, and loguru."""
    for _mod in ("lightning", "lightning.pytorch", "lightning_utilities"):
        logging.getLogger(_mod).setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", message=r".*Tensor Cores.*")
    warnings.filterwarnings("ignore", message=r".*precision=bf16.*")
    warnings.filterwarnings("ignore", message=r".*batch_size.*ambiguous.*")
    warnings.filterwarnings("ignore", message=r".*smaller than the logging interval.*")
    warnings.filterwarnings("ignore", message=r".*exists and is not empty.*")
    warnings.filterwarnings("ignore", message=r".*Length of split.*is 0.*")
    warnings.filterwarnings("ignore", message=r".*Total length of.*DataLoader.*is zero.*")

    # stable_pretraining uses loguru — reconfigure to ERROR to suppress its module
    # summary, manager emoji spam, env dump, CPU offload banner, and related chatter.
    from loguru import logger as _loguru
    _loguru.remove()
    _loguru.add(sys.stderr, level="ERROR")


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    """uint8 [0,255] -> ImageNet-normalized float32."""
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    std = torch.where(std < 1e-8, torch.ones_like(std), std)

    def norm_fn(x):
        return ((x - mean) / std).float()

    normalizer = dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)
    return normalizer


class TrainMonitor(Callback):
    """Print metrics to stdout at regular intervals."""

    def __init__(self, log_every):
        self.log_every = log_every

    def _print_metrics(self, trainer, prefix):
        step = trainer.global_step
        if step % self.log_every != 0 or not trainer.is_global_zero:
            return
        parts = [f"step={step}"]
        for k, v in sorted(trainer.callback_metrics.items()):
            if not k.startswith(prefix) or k.endswith("_step"):
                continue  # drop Lightning's on_step duplicate; bare key carries same value mid-epoch
            fmt = ".3e" if k.endswith("/lr") else ".4f"
            parts.append(f"{k}={v:{fmt}}")
        if len(parts) > 1:
            print(" | ".join(parts), flush=True)

    def on_fit_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            print("fit_start")

    def on_sanity_check_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            print("sanity_check_start")

    def on_sanity_check_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            print("sanity_check_end")

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            print(f"train_epoch_start epoch={trainer.current_epoch}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs):
        self._print_metrics(trainer, "fit/")

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            print(f"train_epoch_end epoch={trainer.current_epoch}")

    def on_validation_batch_end(self, trainer, pl_module, *args, **kwargs):
        self._print_metrics(trainer, "validate/")

    def on_fit_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            print("fit_end")

class ModelObjectCallBack(Callback):
    """Callback to pickle model object after each epoch."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        output_path = (
            self.dirpath
            / f"{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt"
        )

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._dump_model(pl_module.model, output_path)

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._dump_model(pl_module.model, output_path)

    def _dump_model(self, model, path):
        try:
            torch.save(model, path)
        except Exception as e:
            print(f"Error saving model object: {e}")


class WeightsCheckpointCallback(Callback):
    """Save model state_dict every epoch as a versioned file (`<stem>_ep{N:03d}.pt`).
    Complements the end-of-training object-pickle save — mid-training kills leave a
    resumable state_dict, and every past epoch is recoverable for post-hoc eval.
    """

    def __init__(self, path):
        super().__init__()
        self.path = Path(path)

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ep = trainer.current_epoch
        out = self.path.parent / f"{self.path.stem}_ep{ep:03d}{self.path.suffix}"
        torch.save(pl_module.model.state_dict(), out)


class EpochPlanEvalCallback(Callback):
    """Per-epoch in-process CEM planning eval on the training env.

    Wraps the live `pl_module.model` (which has `get_cost`) into a WorldModelPolicy each
    epoch — bypasses swm.policy.AutoCostModel (which loads from disk). Runs rank-zero only
    and logs `eval/success_rate`.

    Expensive dataset + env setup is deferred to the first eval and cached on the
    callback. Eval episode sampling is deterministic on `eval_cfg.seed` so the same
    start/goal pairs are reused across epochs.
    """

    def __init__(self, eval_cfg, every_n_epochs: int = 1):
        super().__init__()
        self.eval_cfg = eval_cfg
        self.every_n_epochs = every_n_epochs
        self._world = None
        self._dataset = None
        self._process = None
        self._transform = None
        self._start_steps = None
        self._episodes_idx = None

    def _setup(self):
        # Local imports keep train.py startup light for runs that don't use eval.
        import stable_pretraining as spt
        import stable_worldmodel as swm
        from sklearn import preprocessing
        from torchvision.transforms import v2 as tvt

        cfg = self.eval_cfg
        cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
        self._world = swm.World(**cfg.world, image_shape=(224, 224))

        def _img():
            return tvt.Compose([
                tvt.ToImage(),
                tvt.ToDtype(torch.float32, scale=True),
                tvt.Normalize(**spt.data.dataset_stats.ImageNet),
                tvt.Resize(size=cfg.eval.img_size),
            ])
        self._transform = {"pixels": _img(), "goal": _img()}

        dataset = swm.data.HDF5Dataset(
            cfg.eval.dataset_name,
            keys_to_cache=cfg.dataset.keys_to_cache,
            cache_dir=Path(swm.data.utils.get_cache_dir()),
        )
        col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
        ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

        process = {}
        for col in cfg.dataset.keys_to_cache:
            if col == "pixels":
                continue
            proc = preprocessing.StandardScaler()
            col_data = dataset.get_col_data(col)
            col_data = col_data[~np.isnan(col_data).any(axis=1)]
            proc.fit(col_data)
            process[col] = proc
            if col != "action":
                process[f"goal_{col}"] = process[col]
        self._process = process

        # Determine per-episode max_start_idx, then filter rows to valid starts, then
        # deterministically sample `num_eval` starts seeded by cfg.seed (mirrors eval.py).
        step_idx = dataset.get_col_data("step_idx")
        ep_col = dataset.get_col_data(col_name)
        lengths = np.array([np.max(step_idx[ep_col == ep]) + 1 for ep in ep_indices])
        max_start_idx = lengths - cfg.eval.goal_offset_steps - 1
        max_start_map = {ep: max_start_idx[i] for i, ep in enumerate(ep_indices)}
        max_start_per_row = np.array([max_start_map[ep] for ep in ep_col])
        valid = np.nonzero(step_idx <= max_start_per_row)[0]

        g = np.random.default_rng(cfg.seed)
        picked = g.choice(len(valid), size=cfg.eval.num_eval, replace=False)
        picked = np.sort(valid[picked])

        rows = dataset.get_row_data(picked)
        self._episodes_idx = rows[col_name].tolist()
        self._start_steps = rows["step_idx"].tolist()
        self._dataset = dataset

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        if self._world is None:
            self._setup()

        import hydra
        import stable_worldmodel as swm
        from omegaconf import OmegaConf

        cfg = self.eval_cfg
        jepa = pl_module.model
        was_training = jepa.training
        jepa.eval()
        requires_grad = [p.requires_grad for p in jepa.parameters()]
        jepa.requires_grad_(False)

        try:
            plan_config = swm.PlanConfig(**cfg.plan_config)
            solver = hydra.utils.instantiate(cfg.solver, model=jepa)
            policy = swm.policy.WorldModelPolicy(
                solver=solver,
                config=plan_config,
                process=self._process,
                transform=self._transform,
            )
            self._world.set_policy(policy)
            callables = OmegaConf.to_container(cfg.eval.callables, resolve=True) \
                if cfg.eval.get("callables") else None
            metrics = self._world.evaluate_from_dataset(
                self._dataset,
                start_steps=self._start_steps,
                goal_offset_steps=cfg.eval.goal_offset_steps,
                eval_budget=cfg.eval.eval_budget,
                episodes_idx=self._episodes_idx,
                callables=callables,
                save_video=False,
                video_path="/tmp",
            )
        finally:
            for p, rg in zip(jepa.parameters(), requires_grad):
                p.requires_grad_(rg)
            if was_training:
                jepa.train()

        sr = float(metrics["success_rate"])
        pl_module.log("eval/success_rate", sr, on_epoch=True,
                      rank_zero_only=True, logger=True, sync_dist=False)
        print(f"[eval] epoch={trainer.current_epoch} success_rate={sr:.4f}", flush=True)
