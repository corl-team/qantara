import os
from functools import partial
from pathlib import Path

from utils import silence_warnings
silence_warnings()

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf, open_dict
from stable_pretraining.callbacks.env_info import EnvironmentDumpCallback

from jepa import JEPA
from module import ARPredictor, Qantara, Embedder, ImageDecoder, MLP, SIGReg
from utils import (
    EpochPlanEvalCallback,
    TrainMonitor,
    WeightsCheckpointCallback,
    adjust_batch_size_for_gpus,
    enable_cem_progress,
    get_column_normalizer,
    get_img_preprocessor,
    is_rank_zero,
)

# ImageNet normalization constants (for decoder PSNR in [0,1] pixel space).
from stable_pretraining.data import dataset_stats as _ds_stats
_IMAGENET_MEAN = torch.tensor(_ds_stats.ImageNet["mean"]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor(_ds_stats.ImageNet["std"]).view(1, 3, 1, 1)


from module import _DINOv2Encoder, _ResNet18IN1kEncoder  # noqa: E402 — defined in module.py so eval.py can unpickle


def init_clearml(project, task_name, hparams):
    """Init ClearML task. Must be called before TensorBoardLogger is created.
    Returns (task, task.id); caller persists task.id alongside the ckpt and is
    responsible for `task.close()` at end of training so the atexit retry loop
    doesn't hang under server-side rate limits.
    """
    import warnings
    # Cap api retries (default 240×120s ≈ 8h) so a rate-limited write doesn't
    # stall task.close(). pyhocon last-wins merge — appending is fine.
    conf = Path.home() / "clearml.conf"
    text = conf.read_text() if conf.exists() else ""
    if "api.http.retries" not in text:
        conf.write_text(text + "\napi.http.retries { total: 5, connect: 5, read: 5, status: 5 }\n")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module=r"clearml\.utilities\.pyhocon")
        warnings.filterwarnings("ignore", module=r"pyparsing")
        from clearml import Task
        # Report inline rather than via a forked subprocess. The default subprocess
        # reporter is what `task.close()` blocks on (BackgroundMonitor.wait_for_sub_process
        # — no timeout), and the subprocess can wedge on rate-limited writes.
        Task._report_subprocess_enabled = False
        Task.set_random_seed(None)
        os.environ.setdefault("CLEARML_VCS_DIFF", "")
        task = Task.init(
            project_name=project, task_name=task_name,
            output_uri=False, reuse_last_task_id=False,
            # detect_repository=False: ClearML's repo scanner runs in the background and
            # task.close() blocks on it via _wait_for_repo_detection(timeout=300s) — a
            # 5-min hang on every job exit. We don't use the diff/commit metadata.
            auto_connect_frameworks={"pytorch": False, "detect_repository": False},
            auto_connect_streams=False,
            auto_resource_monitoring=False,
        )
    warnings.filterwarnings("ignore", module=r"clearml\.utilities\.pyhocon")
    warnings.filterwarnings("ignore", module=r"pyparsing")
    warnings.filterwarnings("ignore", message=r".*torch\.jit\.script_method")
    task.connect(hparams)
    return task, task.id


def load_eval_cfg(cfg):
    """Load config/eval/{config_name}.yaml for in-training per-epoch eval. The only
    dynamic `defaults:` entry that matters at runtime is `solver` (launcher is unused
    here); we resolve it by hand rather than spinning up a second Hydra context.

    `config_name` and `task` fall back to `cfg.data.eval_config` / `cfg.data.task` so
    callers don't need to repeat them — set them once in config/train/data/<name>.yaml.
    """
    edt = cfg.get("eval_during_train")
    if not edt or not edt.get("enabled"):
        return None
    config_name = edt.get("config_name") or cfg.data.get("eval_config")
    assert config_name, "eval_during_train.config_name unset and data.eval_config not defined"
    base = Path(__file__).parent / "config" / "eval"
    eval_cfg = OmegaConf.load(base / f"{config_name}.yaml")
    for d in eval_cfg.pop("defaults", []):
        if isinstance(d, (dict, DictConfig)) and "solver" in d:
            eval_cfg.solver = OmegaConf.load(base / "solver" / f"{d.solver}.yaml")
    eval_cfg.seed = int(cfg.seed)
    if edt.get("num_eval") is not None:
        eval_cfg.eval.num_eval = int(edt.num_eval)
    task = edt.get("task") or cfg.data.get("task")
    if task is not None and "task" in eval_cfg:
        eval_cfg.task = task
    for k, v in (edt.get("solver") or {}).items():
        if v is not None:
            eval_cfg.solver[k] = v
    return eval_cfg


def _decoder_step(module, output, dec_input, pixel_gt, detach, log_images_every=5000, mask=None):
    """Run pixel decoder on dec_input, write decoder_loss + decoder_psnr to output.

    dec_input / pixel_gt share (B, T', ...) layout; caller picks the time slice.
    Optional (B, T') `mask` restricts loss to a subset of frames (PSNR stays unmasked).
    Every `log_images_every` steps, log a GT-vs-pred strip to loggers exposing `add_image`.
    """
    if detach:
        dec_input = dec_input.detach()
    pixel_pred = module.model.decoder(dec_input.flatten(0, 1))
    pixel_gt = pixel_gt.flatten(0, 1)
    if mask is None:
        output["decoder_loss"] = F.mse_loss(pixel_pred, pixel_gt)
    else:
        per_frame_mse = (pixel_pred - pixel_gt).pow(2).mean(dim=(1, 2, 3))
        m = mask.flatten().to(per_frame_mse.dtype)
        output["decoder_loss"] = (per_frame_mse * m).sum() / m.sum().clamp(min=1.0)
    with torch.no_grad():
        mean = _IMAGENET_MEAN.to(pixel_gt.device)
        std  = _IMAGENET_STD.to(pixel_gt.device)
        gt_01   = (pixel_gt   * std + mean).clamp(0, 1)
        pred_01 = (pixel_pred * std + mean).clamp(0, 1)
        output["decoder_psnr"] = -10 * torch.log10(
            (pred_01 - gt_01).pow(2).mean().clamp(min=1e-10)
        )
        if log_images_every and module.global_step % log_images_every == 0:
            n_vis = min(4, pixel_gt.size(0))
            gt_row = torch.cat(list(gt_01[:n_vis]), dim=2)     # (3, H, nW)
            pred_row = torch.cat(list(pred_01[:n_vis]), dim=2)
            grid = torch.cat([gt_row, pred_row], dim=1)        # top=GT, bottom=pred
            for lg in module.loggers:
                if hasattr(lg, "experiment") and hasattr(lg.experiment, "add_image"):
                    lg.experiment.add_image("decoder/gt_vs_pred", grid, module.global_step)


def _probe_step(model, output, state_gt):
    """Detached state-probe loss on post-projector embedding. Returns loss or None."""
    if not hasattr(model, "post_proj_probe"):
        return None
    emb_flat = output["emb"].detach().flatten(0, 1).float()
    gt = state_gt.flatten(0, 1).float()
    output["probe_post_loss"] = F.mse_loss(model.post_proj_probe(emb_flat), gt)
    return output["probe_post_loss"]


def lejepa_forward(self, batch, stage, cfg):
    """Encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]          # (B, T, D)
    act_emb = output["act_emb"]  # (B, T, A_emb)

    # Collapse detector — per-dim std of post-projector emb (loss-kind-agnostic).
    with torch.no_grad():
        _per_dim_std = emb.flatten(0, 1).float().std(dim=0, unbiased=False)
        output["emb_std_mean"] = _per_dim_std.mean()
        output["emb_std_min"]  = _per_dim_std.min()

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:]                    # shift-n_preds target
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = (
        output["pred_loss"]
        + cfg.loss.sigreg.weight * output["sigreg_loss"]
    )

    # Decoder reconstructs pixels from the predicted next-state emb. Predictor output
    # covers frames [n_preds:], so pixel GT is sliced to match.
    if self.model.decoder is not None:
        _decoder_step(self, output, pred_emb, batch["pixels"][:, n_preds:],
                      detach=cfg.decoder.get("detach", True),
                      log_images_every=cfg.decoder.get("log_images_every", 5000))
        output["loss"] = output["loss"] + cfg.loss.decoder.weight * output["decoder_loss"]

    if "state" in batch:
        probe_total = _probe_step(self.model, output, batch["state"])
        if probe_total is not None:
            output["loss"] = output["loss"] + probe_total

    log_keys = {k for k in output if "loss" in k or "psnr" in k or k.startswith("emb_")}
    metrics_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if k in log_keys}
    if stage == "fit":
        metrics_dict["fit/lr"] = self.optimizers().param_groups[0]["lr"]
    self.log_dict(metrics_dict, on_step=True, on_epoch=True, sync_dist=True, logger=True)
    return output


def qantara_forward(self, batch, stage, cfg):
    """Qantara forward: encode → predictor → masked bridge-z + FM-a + SIGReg losses.
    Assumes every action is labeled; NaN sequence-boundary rows are replaced with 0
    and contribute to the FM action loss (negligible for small NaN fractions).
    """

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)
    emb = output["emb"]            # (B, T+1, D_emb) — post-projector z_0..z_T
    action = batch["action"]        # (B, T+1, D_act) — raw action at each frame
    B, Tp1, _ = emb.shape
    T = Tp1 - 1

    # Collapse detector — per-dim std of post-projector emb. Full collapse → 0 for all dims;
    # partial collapse → min→0 while mean>0. Cheap (single reduction) and loss-kind-agnostic.
    with torch.no_grad():
        _per_dim_std = emb.flatten(0, 1).float().std(dim=0, unbiased=False)
        output["emb_std_mean"] = _per_dim_std.mean()
        output["emb_std_min"]  = _per_dim_std.min()

    # Qantara target blocks (a_t, z_{t+1}) use a_0..a_{T-1}; a_T would pair with z_{T+1} which
    # isn't in the clip. Drop the last action row.
    z_clean = emb
    a_clean = action[:, :T]

    out = self.model.predictor(z_clean, a_clean)
    # FM losses masked to target blocks (see Qantara.forward for K / is_target semantics).
    # τ_a=1 positions are excluded from the action loss when mask_tau1=True (input is
    # clean a_t but target = a_t − ε_a is irreducibly stochastic in ε_a, leaving a
    # per-sample gradient noise floor that would propagate through the shared trunk).
    # The state axis under z_bridge always-masks τ_z=1.
    is_tgt = out["is_target"]
    mask_tau1_a = bool(cfg.loss.fm_a.get("mask_tau1", True))
    mask_z_b = is_tgt & (out["tau_z"] < 1.0)
    mask_a_b = is_tgt & (out["tau_a"] < 1.0) if mask_tau1_a else is_tgt
    mask_z = mask_z_b.float()
    mask_a = mask_a_b.float()
    z_err = (out["x_z"] - out["tgt_z"]).pow(2).mean(dim=-1)
    a_err = (out["v_a"] - out["tgt_a"]).pow(2).mean(dim=-1)

    output["fm_z_loss"] = (z_err * mask_z).sum() / mask_z.sum().clamp(min=1.0)
    output["fm_a_loss"] = (a_err * mask_a).sum() / mask_a.sum().clamp(min=1.0)
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))

    output["loss"] = (
        cfg.loss.fm_z.weight * output["fm_z_loss"]
        + cfg.loss.fm_a.weight * output["fm_a_loss"]
        + cfg.loss.sigreg.weight * output["sigreg_loss"]
    )

    # Decoder on x̂^z, masked to target blocks. Read the cem replica (best-conditioned
    # x_z: clean action in); fall back to modes[0] if cem is disabled.
    if self.model.decoder is not None:
        M = out["M"]
        modes = self.model.predictor.modes
        off = modes.index("cem") if "cem" in modes else 0
        _decoder_step(self, output, out["x_z"][off::M], batch["pixels"][:, 1:],
                      detach=cfg.decoder.get("detach", True),
                      log_images_every=cfg.decoder.get("log_images_every", 5000),
                      mask=out["is_target"][off::M])
        output["loss"] = output["loss"] + cfg.loss.decoder.weight * output["decoder_loss"]

    if "state" in batch:
        probe_total = _probe_step(self.model, output, batch["state"])
        if probe_total is not None:
            output["loss"] = output["loss"] + probe_total

    log_keys = {k for k in output if "loss" in k or "psnr" in k or k.startswith("emb_")}
    metrics_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if k in log_keys}
    if stage == "fit":
        metrics_dict["fit/lr"] = self.optimizers().param_groups[0]["lr"]
    # (τ^a, τ^z) mean over target slots — audits the mode-mixture composition (each
    # mode in mode_spec pins or samples τ per slot). Train stage only to keep eval logs clean.
    if stage == "fit":
        with torch.no_grad():
            tau_z = out["tau_z"][is_tgt].float()
            tau_a = out["tau_a"][is_tgt].float()
            if tau_z.numel() > 0:
                metrics_dict["fit/tau_z_mean"] = tau_z.mean()
            if tau_a.numel() > 0:
                metrics_dict["fit/tau_a_mean"] = tau_a.mean()
    self.log_dict(metrics_dict, on_step=True, on_epoch=True, sync_dist=True, logger=True)
    return output


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    wm_type = cfg.wm.get("type", "lewm")
    assert wm_type in ("lewm", "qantara"), f"unknown wm.type={wm_type}"

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    pl.seed_everything(cfg.seed, workers=True)
    enable_cem_progress(every=5)  # heartbeat during long CEM solves (in-train + eval.py)

    #########################
    ##       dataset       ##
    #########################

    print("building dataset")
    adjust_batch_size_for_gpus(cfg)

    ds_cfg = {k: v for k, v in cfg.data.dataset.items() if v is not None}
    dataset = swm.data.HDF5Dataset(**ds_cfg, transform=None)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)
            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    ##############################
    ##       model / optim      ##
    ##############################

    print("building model")

    encoder_kind = cfg.get("encoder_kind", "vit")
    if encoder_kind == "vit":
        encoder = spt.backbone.utils.vit_hf(
            cfg.encoder_scale, patch_size=cfg.patch_size, image_size=cfg.img_size,
            pretrained=False, use_mask_token=False,
        )
        hidden_dim = encoder.config.hidden_size
    elif encoder_kind in ("dinov2_s", "rn18_in1k"):
        # Frozen pretrained backbone ablations. Inputs are already ImageNet-normalized
        # by get_img_preprocessor (DINOv2 needs mult-of-14 img_size; ResNet is size-flex).
        # Capacity-matched: predictor trunk stays at headline width (cfg.wm.embed_dim,
        # default 192); the only learnable surface is the Linear shape-adapter below.
        # Pair with loss.sigreg.weight=0.0 (pretrained features are non-collapsed).
        encoder = _DINOv2Encoder(cfg.img_size) if encoder_kind == "dinov2_s" \
            else _ResNet18IN1kEncoder()
        encoder.requires_grad_(False)
        hidden_dim = cfg.wm.get("embed_dim", 192)
    else:
        raise ValueError(f"unknown encoder_kind={encoder_kind!r} (allowed: vit, dinov2_s, rn18_in1k)")

    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    if wm_type == "qantara":
        # Default num_frames falls back to the inherited LeWM split (history_size + num_preds);
        # override via `wm.num_frames=N` to set directly. Only the sum matters for Qantara.
        num_frames = cfg.wm.get("num_frames", cfg.wm.history_size + cfg.wm.num_preds)
        predictor = Qantara(
            num_frames=num_frames,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            action_dim=effective_act_dim,
            **cfg.predictor,
        )
    else:
        predictor = ARPredictor(
            num_frames=cfg.wm.history_size,
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            **cfg.predictor,
        )

    # Qantara noises in raw action space and projects internally — no external action_encoder
    # or pred_proj (heads are internal). Le-WM keeps both.
    if wm_type == "qantara":
        action_encoder = None
        predictor_proj = None
    else:
        action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
        predictor_proj = MLP(
            input_dim=hidden_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        )

    # Encoder post-CLS projector. BN-MLP mirrors the Le-WM paper recipe (BN1d hidden,
    # zero-init last Linear); the JEPA latent target distribution is BN-shaped, so the
    # projector's BN aligns the encoder's HF ViT output with it. Frozen-backbone path
    # uses a single Linear shape-adapter (no BN/MLP — DINOv2 features are non-collapsed
    # by pretraining, and the only job is encoder_out_dim → predictor's hidden_dim).
    if encoder_kind == "vit":
        projector = MLP(input_dim=hidden_dim, output_dim=embed_dim,
                        hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    else:
        projector = torch.nn.Linear(encoder.embed_dim, embed_dim)

    # Pixel decoder (optional).
    dec_cfg = cfg.get("decoder", {})
    dec_weight = cfg.loss.get("decoder", {}).get("weight", 0)
    decoder = None
    if dec_weight > 0:
        decoder = ImageDecoder(
            embed_dim=embed_dim,
            img_size=cfg.img_size,
            base_channels=dec_cfg.get("base_channels", 128),
        )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
        decoder=decoder,
    )
    if wm_type == "qantara":
        # Knobs read at inference by JEPA's Qantara rollout paths.
        # rollout_k: number of state-axis x̂-recursion steps per env step in CEM-style
        # latent planning. K=1 = single x-prediction call (exact along the chord at γ=0).
        world_model.rollout_k = cfg.wm.get("rollout_k", 1)
        # rollout_guidance_w: CFG scale w with the learned null-action proxy; 1.0 = off.
        world_model.rollout_guidance_w = cfg.wm.get("rollout_guidance_w", 1.0)
        # rollout_a_k: number of action-axis Euler v-integration steps for BC sampling.
        world_model.rollout_a_k = cfg.wm.get("rollout_a_k", 1)
        # bc_past_frames: frame-spaced past context for BC inference (default = history_size - 1
        # so it matches the training-time frame-spaced context). Read by JEPA.get_action.
        world_model.bc_past_frames = cfg.wm.get("bc_past_frames", max(0, cfg.wm.history_size - 1))
        # Raw action dim (env's action_space.shape[-1]). The predictor sees the
        # frameskip-expanded effective_act_dim; the BC dispatch in JEPA.get_action splits
        # the predicted concat into per-frame actions for the env-step buffer.
        world_model.action_dim_raw = int(cfg.wm.action_dim)

    # Online state probe (trained via main loss on detached post-projector embedding).
    if hasattr(cfg.wm, "state_dim"):
        world_model.post_proj_probe = torch.nn.Linear(embed_dim, cfg.wm.state_dim)

    # Scheduler config. Defaults to LinearWarmupCosineAnnealingLR with
    # smart-default (warmup_steps = 1% total). Override via `optimizer.scheduler.<key>=...`.
    # null sub-keys are stripped. If any sub-key is set (so spt's all-or-nothing smart-default
    # path is skipped), we backfill missing required kwargs (max_steps, warmup_steps,
    # warmup_start_lr, eta_min) ourselves using the same formula spt would, computed from
    # trainer.max_epochs × len(train_loader). Without the backfill, partial overrides crash
    # with `LinearWarmupCosineAnnealingLR.__init__() missing 1 required positional argument`.
    # `*_frac` knobs (warmup_frac, stable_frac, decay_frac) resolve to absolute step counts as
    # frac × est_total_steps — convenient for env-agnostic 10/70/20-style WSD configs.
    opt_dict = {k: v for k, v in dict(cfg.optimizer).items() if k != "scheduler"}
    sched_cfg = cfg.optimizer.get("scheduler", None)
    if sched_cfg is None:
        sched_dict = {"type": "LinearWarmupCosineAnnealingLR"}
    else:
        sched_dict = {k: v for k, v in dict(sched_cfg).items() if v is not None}
        sched_dict.setdefault("type", "LinearWarmupCosineAnnealingLR")
        est_total_steps = int(cfg.trainer.max_epochs) * max(1, len(train))
        for key in ("warmup", "stable", "decay"):
            frac_key = f"{key}_frac"
            if frac_key in sched_dict:
                sched_dict[f"{key}_steps"] = max(1, int(float(sched_dict.pop(frac_key)) * est_total_steps))
        if sched_dict.get("type") == "LinearWarmupCosineAnnealingLR" and len(sched_dict) > 1:
            # User override path — fill any missing required kwargs.
            sched_dict.setdefault("warmup_steps", max(1, int(0.01 * est_total_steps)))
            sched_dict.setdefault("max_steps", est_total_steps)
            sched_dict.setdefault("warmup_start_lr", 0.0)
            sched_dict.setdefault("eta_min", 0.0)
        elif sched_dict.get("type") == "WarmupStableDecay":
            # Local class — pass as partial to bypass spt's name-based factory (which only
            # looks up torch.optim.lr_scheduler + spt.optim.lr_scheduler globals).
            from module import WarmupStableDecay
            kwargs = {k: v for k, v in sched_dict.items() if k != "type"}
            sched_dict = partial(WarmupStableDecay, **kwargs)
    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": opt_dict,
            "scheduler": sched_dict,
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    forward_fn = qantara_forward if wm_type == "qantara" else lejepa_forward
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(forward_fn, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    # In hosted job runners (PLATFORM_JOB_NAME set), outputs/ is the synced dir.
    if os.environ.get("PLATFORM_JOB_NAME"):
        run_dir = Path("outputs") / run_id
    else:
        run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    loggers = [CSVLogger(save_dir=run_dir, name="", version="")]

    hparams = OmegaConf.to_container(cfg, resolve=True)
    clearml_task = None
    if cfg.clearml.enabled:
        task_name = os.environ.get("PLATFORM_JOB_NAME", cfg.clearml.task)
        clearml_task, task_id = init_clearml(cfg.clearml.project, task_name, hparams)
        # Persist task id so post-train scripts can attach same-task scalars.
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "clearml_task_id.txt").write_text(task_id)
    # TBLogger is always present — ClearML auto-intercepts TB writes when enabled.
    loggers.append(TensorBoardLogger(save_dir=run_dir, name="", version=""))
    if cfg.wandb.enabled:
        wb = WandbLogger(**cfg.wandb.config)
        wb.log_hyperparams(hparams)
        loggers.append(wb)

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    # Per-epoch versioned weights ckpts (`<stem>_ep{N:03d}.pt`) so every epoch is
    # recoverable and mid-training kills leave a resumable state_dict. End-of-train
    # object pickle is written below.
    weights_path = run_dir / f"{cfg.output_model_name}_weights.pt"

    callbacks = [
        TrainMonitor(log_every=cfg.trainer.log_every_n_steps),
        WeightsCheckpointCallback(weights_path),
    ]
    eval_cfg = load_eval_cfg(cfg)
    if eval_cfg is not None:
        callbacks.append(EpochPlanEvalCallback(
            eval_cfg,
            every_n_epochs=cfg.eval_during_train.get("every_n_epochs", 1),
        ))

    print("constructing Trainer")
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        logger=loggers,
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    # spt.Manager auto-injects EnvironmentDumpCallback; remove it (we don't use spt env logging)
    trainer.callbacks = [cb for cb in trainer.callbacks if not isinstance(cb, EnvironmentDumpCallback)]

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        seed=cfg.seed,
        ckpt_path=None,
    )

    manager()

    # Final weights are written per-epoch by WeightsCheckpointCallback; last epoch's
    # save is the authoritative state_dict. Object pickle is rank-zero only so N workers
    # don't race on the same large file; swm.policy.AutoCostModel loads this for eval.
    if is_rank_zero():
        obj_path = run_dir / f"{cfg.output_model_name}_object.ckpt"
        torch.save(world_model.model, obj_path)
        print(f"Saved object to {obj_path}")

    # ClearML's task.close() / atexit __shutdown() call self.flush(wait_for_uploads=True)
    # which calls reporter.wait_for_events(timeout=None) — an unbounded wait on the
    # daemon-thread event queue. Even with retry caps + subprocess disabled, lingering
    # threads (DataLoader workers, ClearML pollers, MUJOCO/CUDA handlers — 130+ here)
    # block process exit. Flush async to push pending events, then hard-exit.
    if clearml_task is not None:
        import threading as _thr
        import time as _t

        # Mark completed FIRST (single fast API call). The flush() that follows can
        # block on the daemon reporter queue under server load — if our watchdog
        # then kills the process, the task is at least flagged completed.
        try:
            clearml_task.mark_completed(force=True, ignore_errors=True)
        except Exception as e:
            print(f"ClearML mark_completed failed (continuing): {e}", flush=True)

        # Poll-drain: kick the daemon thread, then wait up to `budget` seconds for
        # the in-flight queue to empty. Exits as soon as the queue is empty (typical
        # case) instead of always waiting the full budget. Hard-exits if drain stalls.
        budget = 180
        try:
            clearml_task.flush(wait_for_uploads=False)
            reporter = getattr(clearml_task, "_Task__reporter", None)
            print(f"[clearml] draining reporter queue (up to {budget}s)…", flush=True)
            tic = _t.time()
            last_msg = tic
            while _t.time() - tic < budget:
                if reporter is None or not reporter.events_waiting():
                    break
                if _t.time() - last_msg > 30:
                    print(f"[clearml] still draining ({int(_t.time() - tic)}s elapsed)", flush=True)
                    last_msg = _t.time()
                _t.sleep(1.0)
            elapsed = _t.time() - tic
            if reporter and reporter.events_waiting():
                print(f"[clearml] drain budget {budget}s exhausted (queue still non-empty)", flush=True)
            else:
                print(f"[clearml] drained in {elapsed:.1f}s", flush=True)
        except Exception as e:
            print(f"ClearML drain failed (continuing): {e}", flush=True)
        os._exit(0)


if __name__ == "__main__":
    run()
