"""Render the multi-paradigm decoded rollout figure from cached .npz tensors.

Pairs with scripts/make_rollout_figure.py — that script runs the model and dumps
decoded RGB strips to <env>.npz (keys: gt, demo, bc, video_idm; each (T, 3, H, W)
float32 in [0,1]). This script consumes those tensors so layout/styling can be
iterated without re-running the predictor + decoder.

Design changes vs. the inline render in make_rollout_figure.py:
  - tight tile packing (no inter-tile whitespace)
  - serif typography to match the LaTeX caption
  - explicit role tag on each row ("CEM forward primitive" / "policy-head sampling"
    / "inverse-dynamics actions") so the role of `demo` does not have to be read
    out of the caption
  - thin vertical separator between t=0 (encoded context) and t=1.. (open-loop
    rollout) so the reader sees where prediction begins
  - no figure-level suptitle (the LaTeX caption is the title)

Usage:
  uv run python scripts/plot_rollout_figure.py \\
      --npz figs/rollout/pusht.npz --out figs/rollout/pusht_v2.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROW_SPEC: list[tuple[str, str, str]] = [
    # (npz_key, primary label, sub-label)
    ("gt",        "ground truth",  "held-out trajectory pixels"),
    ("demo",      "demo",          "CEM forward primitive on recorded actions"),
    ("bc",        "BC",            "policy-head action sampling"),
    ("video_idm", "video-inv",     "inverse-dynamics action extraction"),
]


def load_npz(path: Path, drop_video_idm: bool) -> list[tuple[str, str, np.ndarray]]:
    data = np.load(path)
    rows = []
    for key, label, sub in ROW_SPEC:
        if key == "video_idm" and drop_video_idm:
            continue
        rows.append((label, sub, data[key]))
    return rows


def render(rows: list[tuple[str, str, np.ndarray]], out_path: Path,
           tile_size: float = 1.05, label_width: float = 1.7,
           title: str | None = None) -> None:
    n_rows = len(rows)
    n_cols = rows[0][2].shape[0]

    fig_w = label_width + tile_size * n_cols
    fig_h = tile_size * n_rows + (0.35 if title else 0.0)
    fig = plt.figure(figsize=(fig_w, fig_h))

    label_frac = label_width / fig_w
    top_pad = 0.30 / fig_h if title else 0.005
    gs = fig.add_gridspec(
        n_rows, n_cols,
        left=label_frac, right=1.0 - 0.005,
        bottom=0.005, top=1.0 - top_pad,
        wspace=0.0, hspace=0.0,
    )

    plt.rcParams.update({"font.family": "serif", "font.size": 9})

    for r, (label, sub, imgs) in enumerate(rows):
        for c in range(n_cols):
            ax = fig.add_subplot(gs[r, c])
            img = np.transpose(imgs[c], (1, 2, 0))
            ax.imshow(np.clip(img, 0.0, 1.0), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_color("#cccccc")
            if r == 0:
                if c == 0:
                    marker = "$t=0$  (context)"
                else:
                    marker = f"$t={c}$"
                ax.set_title(marker, fontsize=9, pad=3)

        # Row label block: bold primary on top, italic sub-label beneath.
        first_ax = fig.axes[r * n_cols]
        bbox = first_ax.get_position()
        y_mid = (bbox.y0 + bbox.y1) / 2
        # Primary label
        fig.text(label_frac - 0.012, y_mid + 0.022, label,
                 ha="right", va="center", fontsize=10, weight="bold")
        # Sub-label (smaller, lighter)
        fig.text(label_frac - 0.012, y_mid - 0.022, sub,
                 ha="right", va="center", fontsize=7.5, style="italic",
                 color="#555555")

    # Vertical separator between t=0 (context) and t=1 (first rollout step).
    # The line spans the figure between column 0 and column 1.
    if n_cols >= 2:
        col0 = fig.axes[0].get_position()
        col1 = fig.axes[1].get_position()
        x_sep = (col0.x1 + col1.x0) / 2
        # span the full vertical extent of the tile grid
        top = fig.axes[0].get_position().y1
        bot = fig.axes[(n_rows - 1) * n_cols].get_position().y0
        sep = Line2D([x_sep, x_sep], [bot, top],
                      transform=fig.transFigure,
                      color="#888888", linewidth=0.8, linestyle=(0, (3, 2)))
        fig.add_artist(sep)

    if title:
        fig.suptitle(title, fontsize=11, y=1.0 - 0.005)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_path.with_suffix(".png"), dpi=200,
                 bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[wrote] {out_path} (and .png)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True, type=Path,
                    help="Cached rollout tensors (from make_rollout_figure.py)")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output PDF path; PNG saved alongside")
    ap.add_argument("--drop-video-idm", action="store_true",
                    help="Render only 3 rows (drop video-inverse)")
    ap.add_argument("--title", default=None,
                    help="Optional figure title (omit when used inside LaTeX caption)")
    args = ap.parse_args()

    rows = load_npz(args.npz, drop_video_idm=args.drop_video_idm)
    render(rows, args.out, title=args.title)


if __name__ == "__main__":
    main()
