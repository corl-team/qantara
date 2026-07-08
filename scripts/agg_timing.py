"""Aggregate per-seed get_action wall-clock JSONs (eval.py timing sidecars) into a
(env × dispatch) table.

Reads results.txt.timing.json files emitted by eval.py and groups by (env,
policy_kind, bc_kind). Reports per-cell mean ± std of `mean_s` (across train
seeds), since each train seed produces one mean over many env-steps. Also
reports a CEM-relative speedup column for quick reading.

Usage:
    uv run python scripts/agg_timing.py <root1> [<root2> ...] [--out tab.tex]

Each <root> is searched recursively for *.timing.json. Cell key is
(env, dispatch) where dispatch ∈ {cem, bc, video_idm}. Multiple roots are
merged.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


DISPATCH_LABEL = {
    "cem": r"CEM latent planning",
    "bc": r"BC $\tau^z\!=\!0$ Euler",
    "video_idm": r"video--inverse",
}

ENV_ORDER = ["pusht", "tworoom", "cube", "reacher"]
ENV_LABEL = {
    "pusht": "Push-T",
    "tworoom": "Two-Room",
    "cube": "Cube",
    "reacher": "Reacher",
}


def load_one(path: Path) -> dict | None:
    try:
        with path.open() as f:
            d = json.load(f)
    except Exception as e:
        print(f"[skip] {path}: {e}")
        return None
    return d


def dispatch_key(d: dict) -> str:
    pk = d.get("policy_kind")
    if pk == "cem":
        return "cem"
    if pk == "bc":
        bk = d.get("bc_kind") or "bc"
        return bk
    return pk or "unknown"


def env_key(d: dict) -> str:
    e = d.get("env") or ""
    e = str(e).lower()
    for canonical in ENV_ORDER:
        if canonical in e:
            return canonical
    return e or "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", type=Path,
                    help="Directories searched recursively for *.timing.json")
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional path to write a LaTeX table fragment")
    ap.add_argument("--metric", default="mean_s",
                    choices=["mean_s", "median_s"])
    args = ap.parse_args()

    # cells[(env, dispatch)] -> list of per-seed metric values (one per file)
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    n_calls: dict[tuple[str, str], int] = defaultdict(int)

    for root in args.roots:
        for p in sorted(root.rglob("*.timing.json")):
            d = load_one(p)
            if not d:
                continue
            k = (env_key(d), dispatch_key(d))
            cells[k].append(float(d[args.metric]))
            counts[k] += 1
            n_calls[k] += int(d.get("n_calls", 0))

    if not cells:
        print("No timing JSONs found.")
        return

    dispatches_present: list[str] = []
    for d in ("cem", "bc", "video_idm"):
        if any(k[1] == d for k in cells):
            dispatches_present.append(d)

    envs_present = [e for e in ENV_ORDER if any(k[0] == e for k in cells)]

    # Console table
    head = ["env"] + [f"{DISPATCH_LABEL[d]} (s)" for d in dispatches_present]
    if "cem" in dispatches_present:
        head += ["CEM/BC", "CEM/vid--inv"]
    print("\t".join(head))
    for env in envs_present:
        row: list[str] = [ENV_LABEL[env]]
        cem_mean = None
        for disp in dispatches_present:
            vals = cells.get((env, disp), [])
            if not vals:
                row.append("-"); continue
            m = statistics.mean(vals)
            s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            row.append(f"{m:.4f}±{s:.4f} (n={len(vals)},N={n_calls[(env,disp)]})")
            if disp == "cem":
                cem_mean = m
        if "cem" in dispatches_present:
            for disp in ("bc", "video_idm"):
                if disp in dispatches_present:
                    vals = cells.get((env, disp), [])
                    if vals and cem_mean is not None:
                        m = statistics.mean(vals)
                        row.append(f"{cem_mean / m:.1f}x" if m > 0 else "-")
                    else:
                        row.append("-")
        print("\t".join(row))

    if args.out is not None:
        cols = "l" + "c" * len(dispatches_present) + ("cc" if "cem" in dispatches_present else "")
        lines = [
            r"\begin{tabular}{" + cols + "}",
            r"\toprule",
            "env & " + " & ".join(DISPATCH_LABEL[d] for d in dispatches_present)
            + ((" & CEM/BC & CEM/vid--inv" if "cem" in dispatches_present else "")) + r" \\",
            r"\midrule",
        ]
        for env in envs_present:
            cells_str: list[str] = [ENV_LABEL[env]]
            cem_mean = None
            for disp in dispatches_present:
                vals = cells.get((env, disp), [])
                if not vals:
                    cells_str.append("-"); continue
                m = statistics.mean(vals)
                s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                cells_str.append(f"{m:.3f}\\,$\\pm$\\,{s:.3f}")
                if disp == "cem":
                    cem_mean = m
            if "cem" in dispatches_present:
                for disp in ("bc", "video_idm"):
                    if disp in dispatches_present:
                        vals = cells.get((env, disp), [])
                        if vals and cem_mean is not None:
                            m = statistics.mean(vals)
                            cells_str.append(f"{cem_mean / m:.1f}\\,$\\times$" if m > 0 else "-")
                        else:
                            cells_str.append("-")
            lines.append(" & ".join(cells_str) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + "\n")
        print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
