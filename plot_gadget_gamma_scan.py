"""Figures for the gamma scan saved by gamma_gadget_scan.py.

One figure per metric (1 - F, 1 - F in code space, leakage, energy error).
Each figure has one panel per encoding:
    (a) first quantization      (b) particle conservation
and shows the oxygen (sd shell) and calcium (pf shell) isotopes together:
oxygen = blue, circles, solid; calcium = orange, squares, dashed.
Darker shade = more neutron pairs.

Usage
-----
  python plot_gadget_gamma_scan.py
  python plot_gadget_gamma_scan.py --indir data/gamma_scan --outdir figures/gamma_scan
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ENCODINGS = [("fq", "(a)"), ("pc", "(b)")]  # first quantization, particle conservation
METRICS = {
    "infidelity": (r"$1 - F$", "infidelity_vs_gamma.pdf"),
    "infidelity_code_space": (
        r"$1 - F$ (code space)",
        "infidelity_code_space_vs_gamma.pdf",
    ),
    "leakage": ("leakage", "leakage_vs_gamma.pdf"),
    "rel_energy_error": (
        r"$|E - E_{\mathrm{exact}}| / |E_{\mathrm{exact}}|$",
        "rel_energy_error_vs_gamma.pdf",
    ),
}
# element: (colormap, marker, linestyle)
STYLE = {
    "sd": ("Blues", "o", "-"),
    "pf": ("Oranges", "s", "--"),
}


def load(indir: str) -> list[dict]:
    runs = []
    for path in sorted(glob.glob(os.path.join(indir, "*.npz"))):
        z = np.load(path)
        runs.append({k: z[k] for k in z.files})
    for r in runs:
        for k in ("label", "encoding", "shell"):
            r[k] = str(r[k])
        r["n_pairs"] = int(r["n_pairs"])
    # oxygen first, then calcium; inside each by number of pairs
    order = {s: i for i, s in enumerate(STYLE)}
    runs.sort(key=lambda r: (order.get(r["shell"], 99), r["n_pairs"]))
    return runs


def color(run: dict, runs: list[dict]):
    """Same nucleus -> same colour in every panel."""
    cmap = plt.get_cmap(STYLE[run["shell"]][0])
    n_max = max(r["n_pairs"] for r in runs if r["shell"] == run["shell"])
    t = 0.45 + 0.5 * (run["n_pairs"] - 1) / max(n_max - 1, 1)
    return cmap(t)


def plot_metric(runs, metric, ylabel, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (enc, letter) in zip(axes, ENCODINGS):
        for r in (r for r in runs if r["encoding"] == enc):
            if r["shell"] not in STYLE:
                continue
            _, marker, ls = STYLE[r["shell"]]
            y = np.asarray(r[metric], dtype=float)
            ok = np.isfinite(y) & (y > 0)
            ax.plot(
                r["gammas"][ok],
                y[ok],
                marker=marker,
                ls=ls,
                ms=3.5,
                lw=1.3,
                color=color(r, runs),
                label=r["label"],
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\gamma$")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(letter, loc="left", fontweight="bold")
        ax.legend(fontsize=7, ncol=2, framealpha=0.85)
    axes[0].set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"-> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--indir", default="data/gamma_scan")
    ap.add_argument("--outdir", default="figures/gamma_scan")
    args = ap.parse_args()

    runs = load(args.indir)
    if not runs:
        raise SystemExit(f"no .npz files in {args.indir}")
    os.makedirs(args.outdir, exist_ok=True)
    for metric, (ylabel, fname) in METRICS.items():
        plot_metric(runs, metric, ylabel, os.path.join(args.outdir, fname))


if __name__ == "__main__":
    main()
