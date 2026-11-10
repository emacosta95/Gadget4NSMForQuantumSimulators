r"""
Fig. (LiH, no flip): hopping g1 of H_DOCI, exact vs the effective one of the
stoquastic-truncation ("no flip") route of lih_gadget_routes.py.

  (a) exact      g~1_AB = s_A s_B g1_AB   in the max-cut gauge s (best_sign_gauge)
  (b) no flip    -d_A d_B, the hopping the gadget realises (PCE, c_AB = 0): d is
                 the optimal rank-1 fit of the truncated target min(g~1_AB, 0)
                 (frustrated, still positive edges dropped; hatched), with the
                 same fit as the route (fit_drive, |d_A| <= D_MAX, seed 42)

The gauge is a change of orbital phases (same spectrum), so (a) is the exact
hopping in the basis the gadget sees. Same convention as plot_lih_couplings.py:
g1 = K/2 = 1/2 (ab|ab) in mHa [src/molecular_doci.py stores K, and the fit there
gives -d d^T ~ K; here d is rescaled by 1/sqrt 2 so that -d_A d_B ~ K/2].
Canonical RHF orbitals 0, 1, ... by increasing energy; tick labels carry the
gauge sign s_A. One colour scale for both panels.

Usage:
    python plot_lih_noflip_couplings.py [--basis sto-6g] [--R 1.5949] [--width 7.0]
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

from lih_gadget_routes import D_MAX  # noqa: E402
from src.gadget import fit_drive  # noqa: E402
from src.molecular_doci import best_sign_gauge, doci_couplings, lih  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--basis", default="sto-6g")
p.add_argument("--R", type=float, default=1.5949, help="Li-H distance [Angstrom]")
p.add_argument("--width", type=float, default=7.0,
               help="figure width in inches (3.4 = one PRR column, 7.0 = two)")
p.add_argument("--out", default="fig_couplings_LiH_noflip")
args = p.parse_args()

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "hatch.linewidth": 0.6,
    }
)

# ------------------------------------------------------- couplings + gauge
c = doci_couplings(lih(args.R), basis=args.basis, max_pairs=10)  # no frozen core
n = c["n_orb"]
s, gg, frustration = best_sign_gauge(c["g1"])
g_trunc = np.minimum(gg, 0.0)  # stoquastic truncation (project convention, = K)
drive = fit_drive(g_trunc, n_restarts=50, use_c=False, seed=42, d_max=D_MAX)
d = drive.d / np.sqrt(2)  # paper convention: -d_A d_B ~ K/2
g_exact = 0.5e3 * gg  # paper convention g1 = K/2, in mHa
g_gadget = -1e3 * np.outer(d, d)  # c_AB = 0: alpha = 1
dropped = g_exact > 0
iu = np.triu_indices(n, 1)
for M in (g_exact, g_gadget):
    np.fill_diagonal(M, np.nan)

n_occ = int(np.sum(c["mf"].mo_occ > 0)) - c["n_core"]
labels = [rf"${a}^{{{'+' if s[a] > 0 else '-'}}}$" for a in range(n)]
print(f"LiH {args.basis}, R={args.R} A: {n} orbitals, gauge s = {s}")
print(f"dropped {int(dropped[iu].sum())} of {len(iu[0])} edges, "
      f"frustration (dropped |g| weight) = {frustration:.3f}")
res = g_gadget[iu] - np.minimum(g_exact, 0)[iu]
print("d_A (paper convention) =", np.round(d, 4))
print(f"rank-1 fit vs truncated target: max |diff| = {np.abs(res).max():.2f} mHa, "
      f"rel. Frobenius = {np.linalg.norm(res) / np.linalg.norm(np.minimum(g_exact, 0)[iu]):.3f}")

# ------------------------------------------------------------------ plot
vmax = np.nanmax(np.abs(g_exact))
norm = Normalize(-vmax, vmax)
cmap = plt.get_cmap("RdBu_r")
fig, axes = plt.subplots(1, 2, figsize=(args.width, 0.4 * args.width),
                         constrained_layout=True)
panels = (
    (g_exact, r"(a) LiH, exact: $s_As_B\,g^{(1)}_{AB}$"),
    (g_gadget, r"(b) no flip, optimal $-d_A d_B$"),
)
for k, (M, title) in enumerate(panels):
    ax = axes[k]
    im = ax.imshow(M, cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r, g, b, _ = cmap(norm(M[i, j]))
            col = "k" if 0.299 * r + 0.587 * g + 0.114 * b > 0.5 else "w"
            if k == 1 and dropped[i, j]:  # dropped edge (fit target 0): hatched
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           hatch="////", ec="0.6", lw=0))
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                    fontsize=7, color=col)
    ax.set_xticks(range(n), labels)
    ax.set_yticks(range(n), labels if k == 0 else [])
    ax.tick_params(length=0)
    for a in (ax.axhline, ax.axvline):  # occupied | virtual RHF orbitals
        a(n_occ - 0.5, color="k", lw=0.6)
    ax.set_title(title, pad=3)
cb = fig.colorbar(im, ax=axes, shrink=0.9, aspect=18, pad=0.02)
cb.set_label("mHa", labelpad=1)
cb.ax.tick_params(labelsize=6)

fig.savefig(f"{args.out}.pdf", bbox_inches="tight")
fig.savefig(f"{args.out}.png", dpi=300, bbox_inches="tight")
print(f"saved {args.out}.pdf / .png")
