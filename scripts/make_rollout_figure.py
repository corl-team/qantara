"""Multi-paradigm decoded rollout figure.

Mirrors LeWM Fig 7 (predictor rollouts) but extends across the three Qantara
inference paradigms. Per env, renders a 4-row strip:

  GT pixels     : real env trajectory from a held-out val sample
  demo          : decoder(predictor rollout under demonstration actions from
                  the dataset trajectory; the forward-dynamics primitive CEM
                  scores during candidate search)
  BC            : decoder(predictor rollout under BC-sampled actions)
                  goal-blind action source for the BC dispatch
  video / inv   : decoder(predictor rollout under video / inverse extracted actions)
                  goal-blind action source for the third dispatch

All four rows share the same starting z_0 (encoded from the trajectory's first
context frame); rows differ only in how the action sequence is generated.

Usage:
  uv run python scripts/make_rollout_figure.py --env pusht \\
      --ckpt /path/to/qantara_object.ckpt --out figs/rollout/pusht.pdf \\
      --horizon 6 --K-fwd 4 --K-act 2
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from module import Qantara                                     # noqa: E402  (unpickle)
from jepa import JEPA                                       # noqa: E402
from scripts.figure_utils import (  # noqa: E402
    denorm_to_01, decode_z, load_ckpt, encode_batch, build_horizon_loader,
)


@torch.no_grad()
def rollout_one(model, z_init: torch.Tensor, action_source: str,
                 demo_actions: torch.Tensor, horizon: int,
                 K_fwd: int, K_act: int) -> torch.Tensor:
    """One open-loop rollout starting from z_init (B=1, D).

    action_source ∈ {"demo", "bc", "video_idm"}. demo_actions: (1, horizon, A_act),
    used only when action_source == "demo".

    Returns predicted latents (1, horizon, D); does NOT include z_init.
    """
    pred = model.predictor
    HS = pred.num_frames - 1     # max context length the rollout primitives accept
    z_acc = z_init.clone()       # (1, num_hist=1, D)
    a_acc = torch.zeros(1, 0, pred.action_dim, device=z_init.device, dtype=z_init.dtype)
    out = []

    for k in range(horizon):
        # Slide window so num_hist ≤ HS (rollout_*_step assertion).
        cur = z_acc.size(1)
        hs = min(cur, HS)
        z_win = z_acc[:, cur - hs:cur]                     # (1, hs, D)
        a_win = a_acc[:, max(0, a_acc.size(1) - (hs - 1)):]  # (1, hs-1, A)

        if action_source == "demo":
            a_k = demo_actions[:, k]                       # (1, A)
        elif action_source == "bc":
            a_k = pred.rollout_a_step(z_win, a_win, K=K_act)
        elif action_source == "video_idm":
            a_k = pred.rollout_video_idm_step(z_win, a_win, K=K_act)
        else:
            raise ValueError(f"unknown action_source: {action_source}")

        z_next = pred.rollout_z_step(z_win, a_win, a_k, K=K_fwd, guidance_w=1.0)  # (1, D)
        out.append(z_next.unsqueeze(1))
        z_acc = torch.cat([z_acc, z_next.unsqueeze(1)], dim=1)
        a_acc = torch.cat([a_acc, a_k.unsqueeze(1)], dim=1)

    return torch.cat(out, dim=1)                            # (1, horizon, D)


@torch.no_grad()
def collect_rollouts(model, loader, device, horizon: int, K_fwd: int, K_act: int,
                      sample_idx: int = 0):
    """Pick one trajectory from the val loader; run all three rollouts.

    Returns a dict of (horizon+1, 3, H, W) RGB tensors keyed by row name. The +1
    is the context frame at t=0; rollout rows show t=1..horizon decoded.
    """
    pred = model.predictor
    assert isinstance(pred, Qantara), "Qantara-only — uses Qantara rollout primitives"

    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        # z_full: (B, T, D), a_full: (B, T, A). T should be ≥ horizon + 1 for a clean strip.
        z_full, a_full = encode_batch(model, batch, is_qantara=True)
        if z_full.size(0) <= sample_idx or z_full.size(1) < horizon + 1:
            continue
        # Pick one trajectory (B=1 slice).
        z_full = z_full[sample_idx:sample_idx + 1]                   # (1, T, D)
        a_full = a_full[sample_idx:sample_idx + 1]                   # (1, T, A)
        pixels = batch["pixels"][sample_idx:sample_idx + 1]          # (1, T, C, H, W)
        break
    else:
        raise RuntimeError("no eligible trajectory in val loader")

    z_init = z_full[:, :1]                                           # (1, 1, D), keep time dim
    demo_a = a_full[:, :horizon]                                     # (1, horizon, A)

    z_demo = rollout_one(model, z_init, "demo",      demo_a, horizon, K_fwd, K_act)
    z_bc   = rollout_one(model, z_init, "bc",        demo_a, horizon, K_fwd, K_act)
    z_vid  = rollout_one(model, z_init, "video_idm", demo_a, horizon, K_fwd, K_act)

    # Decode every column. Context frame at t=0 is shared across all rollout rows.
    gt_pixels = denorm_to_01(pixels[0, :horizon + 1])                # (horizon+1, 3, H, W)
    decoded_init = decode_z(model, z_init[:, 0])[0]                  # (3, H, W)

    def stitch(predicted_z):
        """(1, horizon, D) -> (horizon+1, 3, H, W) by prepending decoded z_0."""
        decoded = decode_z(model, predicted_z[0])                    # (horizon, 3, H, W)
        return torch.cat([decoded_init.unsqueeze(0), decoded], dim=0)

    return dict(
        gt=gt_pixels,
        demo=stitch(z_demo),
        bc=stitch(z_bc),
        video_idm=stitch(z_vid),
    )


def save_npz(samples: dict, out_path: Path) -> None:
    """Dump decoded RGB tensors as .npz so plotting can iterate without rerunning the model.

    All values are (T, 3, H, W) float32 in [0,1]. Keys: gt, demo, bc, video_idm.
    """
    np.savez(
        out_path,
        gt=samples["gt"].cpu().numpy().astype(np.float32),
        demo=samples["demo"].cpu().numpy().astype(np.float32),
        bc=samples["bc"].cpu().numpy().astype(np.float32),
        video_idm=samples["video_idm"].cpu().numpy().astype(np.float32),
    )
    print(f"[wrote] {out_path}")


def render_strip(env: str, samples: dict, out_path: Path,
                  drop_video_idm: bool = False) -> None:
    """4-row (or 3-row) strip per env. Cols = timesteps."""
    import matplotlib.pyplot as plt

    rows = [
        ("GT",         samples["gt"]),
        ("demo",     samples["demo"]),
        ("BC",         samples["bc"]),
    ]
    if not drop_video_idm:
        rows.append(("video / inv", samples["video_idm"]))

    n_cols = samples["gt"].size(0)
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.3 * n_cols, 1.4 * n_rows),
                              squeeze=False)
    for r, (label, imgs) in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r, c]
            img = imgs[c].cpu().permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"$t={c}$", fontsize=8)
            if c == 0:
                ax.set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")
    fig.suptitle(f"Qantara multi-paradigm decoded rollout — {env}", fontsize=10)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.02, wspace=0.02, hspace=0.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out_path} (and .png)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", required=True, help="pusht|tworoom|cube|reacher")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, type=Path,
                    help="Output PDF path; PNG saved alongside")
    ap.add_argument("--horizon", type=int, default=6,
                    help="Number of rollout steps to visualise (in addition to t=0 context)")
    ap.add_argument("--K-fwd", type=int, default=4,
                    help="K for rollout_z_step (CEM forward edge); headline = 4")
    ap.add_argument("--K-act", type=int, default=2,
                    help="K for rollout_a_step / rollout_video_idm_step; headline = 2")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--cache-dir", default=os.environ.get("STABLEWM_HOME"))
    ap.add_argument("--sample-idx", type=int, default=0,
                    help="Which sample within the first val batch to use as the trajectory")
    ap.add_argument("--drop-video-idm", action="store_true",
                    help="Render 3 rows (drop video / inv) if BC and video / inv look identical")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model = load_ckpt(args.ckpt, device)
    if model.decoder is None:
        raise SystemExit(f"ckpt {args.ckpt} has no trained decoder.")

    num_steps = args.horizon + 1
    loader = build_horizon_loader(args.env, args.batch_size, args.num_workers,
                                   args.cache_dir, num_steps=num_steps)

    samples = collect_rollouts(model, loader, device,
                                horizon=args.horizon,
                                K_fwd=args.K_fwd, K_act=args.K_act,
                                sample_idx=args.sample_idx)
    save_npz(samples, args.out.with_suffix(".npz"))
    render_strip(args.env, samples, args.out, drop_video_idm=args.drop_video_idm)


if __name__ == "__main__":
    main()
