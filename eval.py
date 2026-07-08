import os

os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")

import json
import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm
from utils import enable_cem_progress


def attach_get_action_timer(policy):
    """Wrap policy.get_action to record per-call wall-clock (cuda-synced).
    Returns a list that the wrapped method appends to in seconds.
    """
    orig = policy.get_action
    times: list[float] = []
    cuda = torch.cuda.is_available()

    def timed(*a, **kw):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = orig(*a, **kw)
        if cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        return out

    policy.get_action = timed
    return times

def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset

@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run goal-conditioned planning evaluation."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # Mirror train.py's global torch state (TF32 + cuDNN algo choices) so in-train
    # callback and post-train eval share numerics.
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    # Qantara FM rollout samples ε via torch.randn() from global RNG (jepa.py:rollout,
    # module.py:rollout_{z,a}_step). Seed both torch and numpy so CEM rankings are
    # reproducible across processes at fixed seed.
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    enable_cem_progress(every=5)

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    policy = cfg.get("policy", "random")
    # Policy kind. "cem" = WorldModelPolicy wrapping the trained Qantara's get_cost
    # (goal-conditioned MPC). "bc" = FeedForwardPolicy → JEPA.get_action →
    # predictor.rollout_a_step (open-loop BC inference, no goal conditioning).
    policy_kind = cfg.get("policy_kind", "cem")
    assert policy_kind in ("cem", "bc"), f"policy_kind must be cem|bc, got {policy_kind!r}"
    # BC inference variant — see Qantara.rollout_a_step / rollout_joint_step.
    bc_kind = cfg.get("bc_kind", "bc")
    assert bc_kind in ("bc", "joint", "video_idm"), \
        f"bc_kind must be bc|joint|video_idm, got {bc_kind!r}"
    # Frame-spaced past-context for BC: 0 = single-frame; K>0 maintains a deque of
    # past (z, action_chunk) pairs (one entry per frameskip-block) and feeds rollout_a_step
    # with z_hist of length K+1 + a_hist of length K. Capped at num_frames-1 inside the model.
    # Default unset → use ckpt-baked attr (train.py sets it to wm.history_size-1).
    bc_past_frames = cfg.get("bc_past_frames", None)
    if bc_past_frames is not None:
        bc_past_frames = int(bc_past_frames)
        assert bc_past_frames >= 0, f"bc_past_frames must be ≥ 0, got {bc_past_frames}"

    if policy != "random":
        model = swm.policy.AutoCostModel(cfg.policy)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        # Inference-time overrides: flip rollout knobs on the JEPA post-load. train.py
        # bakes defaults (see config/train/qantara.yaml); hydra `+wm.rollout_k=2` at eval
        # time wins last. Keyspace matches train so there's one source of truth.
        # Bridge-marginal noise injection at intermediate τ in rollout_z_step is gated
        # by z_bridge_noise>0; there is no separate SDE flag.
        for attr in ("rollout_k", "rollout_guidance_w", "rollout_a_k"):
            v = cfg.get("wm", {}).get(attr) if "wm" in cfg else None
            if v is not None:
                setattr(model, attr, v)
        if policy_kind == "bc":
            # BC dispatch — open-loop next-action via predictor.rollout_a_step. No goal
            # conditioning at the model level; Qantara's BC mode is unconditional.
            assert hasattr(model, "get_action"), \
                "policy_kind=bc requires a model with get_action (Qantara-trained JEPA)."
            # If the pickled model lacks action_dim_raw, recover it from the action
            # StandardScaler's fitted dim (= env's raw action_space last dim) so
            # get_action's frameskip slicing returns (E, action_dim_raw) per call.
            if not hasattr(model, "action_dim_raw") and "action" in process:
                model.action_dim_raw = int(process["action"].scale_.shape[0])
            model.bc_inference_kind = bc_kind
            if bc_past_frames is not None:
                model.bc_past_frames = bc_past_frames  # explicit eval-time override
            elif not hasattr(model, "bc_past_frames"):
                model.bc_past_frames = 0  # default: single-frame BC
            policy = swm.policy.FeedForwardPolicy(
                model=model, process=process, transform=transform,
            )
        else:
            config = swm.PlanConfig(**cfg.plan_config)
            solver = hydra.utils.instantiate(cfg.solver, model=model)
            policy = swm.policy.WorldModelPolicy(
                solver=solver, config=config, process=process, transform=transform
            )

    else:
        policy = swm.policy.RandomPolicy()

    # Per-get_action wall-clock timer. Wraps the policy entrypoint (one env-step's
    # worth of policy work, batched over envs); cuda-synced so GPU work is included.
    # Skipped for the random baseline.
    timer_calls = attach_get_action_timer(policy) if cfg.policy != "random" else None

    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    # Map each dataset row’s episode_idx to its max_start_idx
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices), size=cfg.eval.num_eval, replace=False
    )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    start_time = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.callables, resolve=True) if cfg.eval.get("callables") else None,
        save_video=bool(cfg.eval.get("save_video", True)),
        video_path=results_path,
    )
    end_time = time.time()

    print(metrics)

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    timing = None
    if timer_calls is not None and len(timer_calls) > 0:
        ts = np.asarray(timer_calls, dtype=np.float64)
        timing = dict(
            n_calls=int(ts.size),
            mean_s=float(ts.mean()),
            std_s=float(ts.std()),
            median_s=float(np.median(ts)),
            p10_s=float(np.percentile(ts, 10)),
            p90_s=float(np.percentile(ts, 90)),
            total_s=float(ts.sum()),
            policy_kind=str(policy_kind),
            bc_kind=str(bc_kind) if policy_kind == "bc" else None,
            env=str(cfg.eval.dataset_name),
        )
        # Seed in filename — repeated runs append to results.txt; per-seed timing
        # JSONs must not clobber.
        timing_path = results_path.parent / f"{cfg.output.filename}.seed{cfg.seed}.timing.json"
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        with timing_path.open("w") as f:
            json.dump(timing, f, indent=2)

    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")
        if timing is not None:
            f.write(f"timing: {json.dumps(timing)}\n")


if __name__ == "__main__":
    run()
