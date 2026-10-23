"""Figures for the gamma scan saved by gamma_gadget_scan.py.

One figure per metric (1 - F, 1 - F in code space, leakage, energy error),
each a 2 x 2 grid:

                    first quantization    particle conservation
    oxygen  (sd)           (a)                    (b)
    calcium (pf)           (c)                    (d)

Colour and marker = isotope (same nucleus looks the same in both encodings).

Plus the scatter infidelity_vs_dim_gamma<G>.pdf: 1 - F at the largest gamma
of the scan (G, or --scatter-gamma) for every nucleus and encoding, (a) vs logical Hilbert-space dimension C(D, N),
(b) vs number of qubits.  Colour = encoding, marker = shell.

Usage
-----
  python plot_gadget_gamma_scan.py
  python plot_gadget_gamma_scan.py --indir data/gamma_scan_cmax4 --outdir figures/gamma_scan_cmax4
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ENCODINGS = ["fq", "pc"]  # columns: first quantization, particle conservation
SHELLS = ["sd", "pf"]  # rows: oxygen, calcium
MARKERS = "osD^v<>ph*"
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


def load(indir: str) -> list[dict]:
    runs = []
    for path in sorted(glob.glob(os.path.join(indir, "*.npz"))):
        z = np.load(path)
        r = {k: z[k] for k in z.files}
        for k in ("label", "encoding", "shell"):
            r[k] = str(r[k])
        r["n_pairs"] = int(r["n_pairs"])
        runs.append(r)
    runs.sort(key=lambda r: r["n_pairs"])
    return runs


def style(run: dict, runs: list[dict]) -> dict:
    """Same nucleus -> same colour and marker in every panel."""
    n_max = max(r["n_pairs"] for r in runs if r["shell"] == run["shell"])
    t = (run["n_pairs"] - 1) / max(n_max - 1, 1)
    return dict(
        color=plt.get_cmap("viridis")(0.9 * t),
        marker=MARKERS[(run["n_pairs"] - 1) % len(MARKERS)],
    )


def plot_metric(runs, metric, ylabel, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    letters = iter("abcd")
    for i, shell in enumerate(SHELLS):
        for j, enc in enumerate(ENCODINGS):
            ax = axes[i, j]
            for r in runs:
                if r["shell"] != shell or r["encoding"] != enc:
                    continue
                y = np.asarray(r[metric], dtype=float)
                ok = np.isfinite(y) & (y > 0)
                ax.plot(
                    r["gammas"][ok],
                    y[ok],
                    ms=3.5,
                    lw=1.3,
                    label=r["label"],
                    **style(r, runs),
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)
            ax.set_title(f"({next(letters)})", loc="left", fontweight="bold")
            if ax.lines:
                ax.legend(
                    fontsize=7, ncol=2 if len(ax.lines) > 5 else 1, framealpha=0.85
                )
    for ax in axes[1]:
        ax.set_xlabel(r"$\gamma$")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"-> {path}")


def plot_scatter(runs, path, gamma0):
    enc_style = {"fq": ("C3", "first quantization"), "pc": ("C0", "particle conservation")}
    shell_marker = {"sd": ("o", "sd shell"), "pf": ("s", "pf shell")}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (xkey, xlabel) in zip(
        axes,
        [("dim", r"logical Hilbert-space dim. $C(D, N)$"), ("n_qubits", "number of qubits")],
    ):
        pts = []
        for r in runs:
            k = int(np.argmin(np.abs(r["gammas"] - gamma0)))
            y = float(r["infidelity"][k])
            if not np.isfinite(y) or y <= 0:
                continue
            x = int(r[xkey])
            ax.scatter(
                x,
                y,
                color=enc_style[r["encoding"]][0],
                marker=shell_marker[r["shell"]][0],
                s=36,
                zorder=3,
            )
            pts.append((x, y, r["label"], r["encoding"] == "fq"))
        if xkey == "dim":
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.autoscale_view()
        # labels: push apart the ones that would overlap (same dimension /
        # same qubit count, e.g. Ca48 and Ca52 both at C(10, 4) = C(10, 6))
        offsets = [(4, 3), (4, -9), (-4, 3), (-4, -9), (4, 11), (-4, 11)]
        placed = []  # (x_disp, y_disp) of used label anchors
        for x, y, label, bold in sorted(pts, key=lambda t: (t[0], t[1])):
            xd, yd = ax.transData.transform((x, y))
            for dx, dy in offsets:
                anchor = (xd + dx * 1.4, yd + dy * 1.4)
                if all(abs(anchor[0] - a) > 30 or abs(anchor[1] - b) > 9 for a, b in placed):
                    break
            placed.append(anchor)
            ax.annotate(
                label,
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7,
                ha="left" if dx > 0 else "right",
                fontweight="bold" if bold else "normal",
            )
        ax.set_xlabel(xlabel)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel(rf"$1 - F$  ($\gamma = {gamma0:g}$)")
    axes[0].set_title("(a)", loc="left", fontweight="bold")
    axes[1].set_title("(b)", loc="left", fontweight="bold")
    handles = [
        plt.Line2D([], [], color=c, marker="o", ls="", label=lab)
        for c, lab in enc_style.values()
    ] + [
        plt.Line2D([], [], color="k", marker=m, ls="", label=lab)
        for m, lab in shell_marker.values()
    ]
    axes[1].legend(handles=handles, fontsize=8, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"-> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--indir", default="data/gamma_scan_cmax4")
    ap.add_argument("--outdir", default="figures/gamma_scan_cmax4")
    ap.add_argument(
        "--scatter-gamma",
        type=float,
        default=None,
        help="gamma of the scatter plot (default: largest gamma of the scan)",
    )
    args = ap.parse_args()

    runs = load(args.indir)
    if not runs:
        raise SystemExit(f"no .npz files in {args.indir}")
    os.makedirs(args.outdir, exist_ok=True)
    for metric, (ylabel, fname) in METRICS.items():
        plot_metric(runs, metric, ylabel, os.path.join(args.outdir, fname))
    g0 = args.scatter_gamma or float(min(r["gammas"].max() for r in runs))
    plot_scatter(runs, os.path.join(args.outdir, f"infidelity_vs_dim_gamma{g0:g}.pdf"), g0)


if __name__ == "__main__":
    main()
