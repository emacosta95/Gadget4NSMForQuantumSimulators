"""Combined figure of the annealing from H_Z (AnnealingFromHZ.ipynb):

    2 x 3 panels, rows = encoding (PCE on top, FQE below),
    columns = nuclei (O18, O20, Ca42),

built from the curves saved in data/annealing_hz/<nucleus>_<enc>_gamma<G>_tau<T>.npz.
Each level e_i(t) is coloured by its population; the dashed line is
e(t) = <H(t)>; in first quantization the fermionic (antisymmetric under
register exchange) levels are dotted -- they exist only for N >= 2 registers,
i.e. for O20 here.

Usage
-----
  python plot_annealing_hz.py                      # gamma = 20, tau = 1000
  python plot_annealing_hz.py --gamma 20 --tau 1000 --nuclei O18 O20 Ca42
"""

from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

DATADIR, FIGDIR = "data/annealing_hz", "figures/annealing_hz"
ENCODINGS = ["pc", "fq"]
ENC_LABEL = {"pc": "PCE", "fq": "FQE"}
PRETTY = {"O18": r"$^{18}$O", "O20": r"$^{20}$O", "O22": r"$^{22}$O", "Ca42": r"$^{42}$Ca", "Ca44": r"$^{44}$Ca"}
POP_CMAP = LinearSegmentedColormap.from_list("pop", ["#d0d0d0", "#f4a259", "#c1121f", "#5a0a10"])
POP_NORM = Normalize(0.0, 1.0)


def load_result(path):
    z = np.load(path)
    r = SimpleNamespace(**{k: z[k] for k in z.files})
    for k in ("label", "encoding"):
        setattr(r, k, str(getattr(r, k)))
    for k in ("gamma", "tau", "fidelity_gadget", "fidelity_exact", "leakage"):
        setattr(r, k, float(getattr(r, k)))
    r.sector = np.asarray(r.sector).astype(str)
    return r


def draw_levels(ax, r, n_show=5, lw=2.2):
    """Levels of one run on `ax`, coloured by population."""
    t, g = r.times, r.gamma
    keep = [i for i, sec in enumerate(r.sector) if sec != "A"][:n_show]
    for i in keep:
        y = r.levels[:, i] * g
        pts = np.column_stack([t, y]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=POP_CMAP, norm=POP_NORM, linewidths=lw, zorder=2)
        lc.set_array(0.5 * (r.populations[:-1, i] + r.populations[1:, i]))
        ax.add_collection(lc)
    for i in [i for i, sec in enumerate(r.sector) if sec == "A"]:
        ax.plot(t, r.levels[:, i] * g, ls=":", lw=1.3, color="#3a6ea5", zorder=1)
    ax.plot(t, r.energy * g, "k--", lw=1.3, zorder=3)
    ax.set_xlim(t[0], t[-1])
    ys = np.concatenate([r.levels[:, keep].ravel() * g, r.energy * g])
    pad = 0.06 * (ys.max() - ys.min())
    ax.set_ylim(ys.min() - pad, ys.max() + pad)


def plot_grid(results, path, n_show=5, legend_title=None):
    """results: {(enc, nucleus): result}; rows = ENCODINGS, columns = nuclei."""
    nuclei = []
    for (_, nuc) in results:
        if nuc not in nuclei:
            nuclei.append(nuc)
    fig, axes = plt.subplots(2, len(nuclei), figsize=(16, 8.6), sharex="col")
    letters = iter("abcdefghi")
    has_fermionic = False
    for i, enc in enumerate(ENCODINGS):
        for j, nuc in enumerate(nuclei):
            ax = axes[i, j]
            r = results.get((enc, nuc))
            if r is None:
                ax.set_visible(False)
                continue
            draw_levels(ax, r, n_show=n_show)
            has_fermionic |= bool((r.sector == "A").any())
            ax.set_title(f"({next(letters)})", loc="left", fontweight="bold", fontsize=13)
            ax.text(0.03, 0.04, rf"{PRETTY.get(nuc, nuc)}, {ENC_LABEL[enc]}, $F={r.fidelity_gadget:.3f}$",
                    transform=ax.transAxes, fontsize=12, va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=11)
            if i == 1:
                ax.set_xlabel(r"$t\,\omega$", fontsize=13)
            if j == 0:
                ax.set_ylabel(r"$e_i\,\gamma$ [MeV]", fontsize=13)
    handles = [Line2D([], [], color="#c1121f", lw=2.2, label="levels $e_i(t)$, colour = population"),
               Line2D([], [], color="k", ls="--", lw=1.3, label=r"$e(t)=\langle H(t)\rangle$")]
    if has_fermionic:
        handles.insert(1, Line2D([], [], color="#3a6ea5", ls=":", lw=1.3,
                                 label="fermionic (antisymmetric) levels, FQE with $N\\geq 2$"))
        handles[0].set_label("bosonic (symmetric) levels, colour = population")
    axes[0, 0].legend(handles=handles, fontsize=9, loc="upper right", framealpha=0.9,
                      title=legend_title, title_fontsize=10)
    fig.subplots_adjust(left=0.05, right=0.93, bottom=0.08, top=0.95, wspace=0.18, hspace=0.18)
    cax = fig.add_axes([0.945, 0.08, 0.012, 0.87])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=POP_NORM, cmap=POP_CMAP), cax=cax)
    cb.set_label(r"population $|\langle\phi_i(t)|\psi(t)\rangle|^2$", fontsize=11)
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png", dpi=200)
    plt.close(fig)
    print("->", path + ".pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gamma", type=float, default=20.0)
    ap.add_argument("--tau", type=float, default=1000.0)
    ap.add_argument("--nuclei", nargs="+", default=["O18", "O20", "Ca42"])
    ap.add_argument("--n-show", type=int, default=5, help="bosonic levels drawn per panel")
    args = ap.parse_args()
    results = {}
    for enc in ENCODINGS:
        for nuc in args.nuclei:
            path = f"{DATADIR}/{nuc}_{enc}_gamma{args.gamma:g}_tau{args.tau:g}.npz"
            if os.path.exists(path):
                results[(enc, nuc)] = load_result(path)
            else:
                print("missing", path)
    tau_star = args.tau / args.gamma
    os.makedirs(FIGDIR, exist_ok=True)
    plot_grid(results, f"{FIGDIR}/annealing_hz_grid_" + "_".join(args.nuclei), n_show=args.n_show,
              legend_title=rf"$\gamma = {args.gamma:g}\,\omega,\ \tau = {tau_star:g}\,\gamma$")


if __name__ == "__main__":
    main()
