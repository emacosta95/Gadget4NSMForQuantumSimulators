"""Figures from gadget_gamma_scan.py output (data/gamma_scan/*.npz).

fig 1  1-F vs gamma, one panel per shell (O | Ca), colour = logical dim,
       solid = particle conservation (pc), dashed = first quantization (fq)
fig 2  1-F at --gamma-ref vs Hilbert-space dimension (logical dim, qubits)
fig 3  per nucleus/encoding: bar plot psi(a,b,c) exact vs gadget at the
       stored gammas (top --top configurations by exact weight)

python plot_gadget_gamma_scan.py --indir data/gamma_scan --outdir figures/gamma_scan
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")  # file output only: no display / Qt needed (WSL, ssh, cluster)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

STYLE = {"pc": dict(ls="-", marker="o"), "fq": dict(ls="--", marker="s")}
ENC_NAME = {"pc": "particle conservation", "fq": "first quantization"}


def load(indir):
    runs = []
    for f in sorted(glob.glob(os.path.join(indir, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        runs.append({k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files})
    return runs


def fig_infidelity_vs_gamma(runs, outdir, key):
    shells = sorted({r["shell"] for r in runs}, key=["sd", "pf"].index)
    dims = [r["dim"] for r in runs]
    # norm = LogNorm(vmin=min(dims), vmax=max(max(dims), min(dims) + 1))
    # cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(
        1, len(shells), figsize=(6 * len(shells), 4.5), squeeze=False
    )
    for ax, shell in zip(axes[0], shells):
        for r in sorted(
            (r for r in runs if r["shell"] == shell),
            key=lambda r: (r["encoding"], r["n_pairs"]),
        ):
            ax.loglog(
                r["gammas"],
                r[key],
                # color=cmap(norm(r["dim"])),
                ms=3,
                label=f"{r['label']} ({ENC_NAME[r['encoding']]})",
                **STYLE[r["encoding"]],
            )
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(
            {
                "infidelity": "1 - F",
                "infidelity_code_space": "1 - F (code space)",
                "leakage": "leakage",
                "rel_energy_error": r"$|\Delta E|/|E|$",
            }[key]
        )
        ax.set_title(f"{shell} shell")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=6, ncol=2)
    # sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    # fig.colorbar(sm, ax=axes[0].tolist(), label="logical Hilbert-space dim")
    path = os.path.join(outdir, f"{key}_vs_gamma.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_infidelity_vs_dim(runs, outdir, gamma_ref):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    markers = {"sd": "o", "pf": "s"}
    colors = {"pc": "C0", "fq": "C3"}
    for r in runs:
        i = int(np.argmin(np.abs(np.log(r["gammas"] / gamma_ref))))
        y = r["infidelity"][i]
        for ax, x in zip(axes, (r["dim"], r["n_qubits"])):
            ax.loglog(x, y, markers[r["shell"]], color=colors[r["encoding"]], ms=7)
            ax.annotate(
                r["label"],
                (x, y),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )
    axes[0].set_xlabel("logical Hilbert-space dim  C(D, N)")
    axes[1].set_xlabel("number of qubits")
    for ax in axes:
        ax.set_ylabel(f"1 - F   ($\\gamma \\approx {gamma_ref:g}$)")
        ax.grid(True, which="both", alpha=0.3)
    handles = [
        Line2D([], [], color=colors[e], marker="o", ls="", label=ENC_NAME[e])
        for e in colors
    ] + [
        Line2D([], [], color="k", marker=markers[s], ls="", label=f"{s} shell")
        for s in markers
    ]
    axes[1].legend(handles=handles, fontsize=8)
    path = os.path.join(outdir, f"infidelity_vs_dim_gamma{gamma_ref:g}.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_wavefunctions(r, outdir, top):
    wf_g = np.atleast_1d(r["wf_gammas"])
    psi_ex = r["psi_exact"]
    order = np.argsort(-np.abs(psi_ex))[:top]
    labels = ["(" + ",".join(map(str, c)) + ")" for c in r["configs"][order]]
    n_bars = 1 + len(wf_g)
    width = 0.8 / n_bars
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(order) * n_bars**0.5), 4))
    ax.bar(x - 0.4 + width / 2, psi_ex[order], width, color="k", label="exact")
    for k, g in enumerate(wf_g):
        psi_g = r["psi_gadget"][k]
        if np.all(np.isnan(psi_g)):
            continue
        j = int(np.argmin(np.abs(r["gammas"] - g)))
        ax.bar(
            x - 0.4 + width * (k + 1.5),
            psi_g[order],
            width,
            label=f"gadget $\\gamma$={g:g}  (1-F={r['infidelity'][j]:.1e})",
        )
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, fontsize=7)
    ax.set_xlabel("quasiparticle orbitals", fontsize=10)
    ax.set_ylabel(r"$\psi(\text{orbitals})$", fontsize=10)
    shown = "" if len(order) == r["dim"] else f", top {len(order)}/{r['dim']}"
    ax.set_title(
        f"{r['label']} - {ENC_NAME[r['encoding']]} ({r['n_qubits']} qubits{shown})"
    )
    ax.legend(fontsize=10)
    path = os.path.join(outdir, f"wf_{r['label']}_{r['isospin']}_{r['encoding']}.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    root = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--indir", default=os.path.join(root, "data", "gamma_scan"))
    ap.add_argument("--outdir", default=os.path.join(root, "figures", "gamma_scan"))
    ap.add_argument("--gamma-ref", type=float, default=300.0)
    ap.add_argument("--top", type=int, default=20)
    args, _ = ap.parse_known_args()
    os.makedirs(args.outdir, exist_ok=True)
    runs = load(args.indir)
    if not runs:
        raise SystemExit(f"no .npz in {args.indir}")
    for key in ("infidelity", "infidelity_code_space", "leakage", "rel_energy_error"):
        print(fig_infidelity_vs_gamma(runs, args.outdir, key))
    print(fig_infidelity_vs_dim(runs, args.outdir, args.gamma_ref))
    for r in runs:
        print(fig_wavefunctions(r, args.outdir, args.top))


if __name__ == "__main__":
    main()
