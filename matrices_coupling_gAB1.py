r"""
Fig. (DOCI couplings, LiH): g1 and g2 of H_DOCI in the same convention as the
NSM figure (plot_couplings_sd_pf.py), i.e. Eq. (H_NSMQ) form:

    h_A    = 2 h1~_{aa} + h2~_{aaaa}
    g1_AB  = 1/2 h2~_{aabb}                     (coefficient of X_A X_B + Y_A Y_B, A<B)
    g2_AB  = 2 (2 h2~_{abba} - h2~_{abab})      (coefficient of N_A N_B, A<B)

with h2~_{abcd} = (ad|bc) in chemists' notation (physicists' <ab|dc>), i.e.
h2~_{aabb} = (ab|ab) = K_ab,  h2~_{abba} = (aa|bb) = J_ab,  h2~_{abab} = K_ab.

NOTE: src/molecular_doci.py stores g1 = K without the 1/2 (the factor sits in
the operator a^dag a + h.c. <-> (XX+YY)/2 there). Here we use the paper's
convention, so g1 = K/2.

Orbitals: canonical RHF, labelled 0,1,2,... by increasing orbital energy.
Default: LiH, STO-6G, R = 1.5957 Angstrom (exp. r_e, Huber & Herzberg 1979), no frozen core
(6 spatial orbitals, 2 electron pairs).

Usage:
    python plot_couplings_LiH.py [--basis sto-6g] [--R 1.5949] [--ncore 0]
                                 [--width 3.4]
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize  # noqa
from pyscf import ao2mo, gto, mcscf, scf

p = argparse.ArgumentParser()
p.add_argument("--basis", default="sto-6g")
p.add_argument("--R", type=float, default=1.5957, help="Li-H distance [Angstrom]")
p.add_argument("--ncore", type=int, default=0, help="frozen doubly occupied orbitals")
p.add_argument(
    "--width",
    type=float,
    default=7.0,
    help="figure width in inches (3.4 = one PRR column, 7.0 = two)",
)
p.add_argument("--out", default="fig_couplings_LiH")
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
    }
)

# ------------------------------------------------------------------ RHF
mol = gto.M(
    atom=f"Li 0 0 0; H 0 0 {args.R}", basis=args.basis, spin=0, charge=0, verbose=0
)
mf = scf.RHF(mol)
mf.conv_tol = 1e-11
mf.kernel()
assert mf.converged

n_mo = mol.nao_nr()
n_core = args.ncore
n_elec = mol.nelectron - 2 * n_core
n_act = n_mo - n_core

# -------------------------------------------- active-space integrals
mc = mcscf.CASCI(mf, n_act, n_elec)
mc.ncore = n_core
h1e, E_core = mc.get_h1eff(mf.mo_coeff)  # h1~ (core-dressed), C
mo_act = mf.mo_coeff[:, n_core : n_core + n_act]
eri = ao2mo.restore(1, ao2mo.kernel(mol, mo_act), n_act)  # (pq|rs)

J = np.einsum("ppqq->pq", eri)  # (pp|qq) = h2~_{abba}
K = np.einsum("pqpq->pq", eri)  # (pq|pq) = h2~_{abab} = h2~_{aabb}

h = 2 * np.diag(h1e) + np.diag(J)
g1 = 0.5 * K
g2 = 2 * (2 * J - K)
for M in (g1, g2):
    np.fill_diagonal(M, np.nan)

# ------------------------------------------------- orbital labels
# canonical RHF orbitals, indexed by increasing orbital energy
labels = [str(i) for i in range(n_core, n_core + n_act)]

occ = mf.mo_occ[n_core : n_core + n_act]
print(
    f"LiH {args.basis}, R={args.R} A: {n_act} active orbitals, "
    f"{n_elec // 2} pairs, frozen core={n_core}, E_RHF={mf.e_tot:.6f} Ha"
)
print("orbitals :", labels)
print("occ      :", occ)
print("h_A [Ha] :", np.round(h, 4))
iu = np.triu_indices(n_act, 1)
print(
    f"g1 range [Ha]: [{np.nanmin(g1[iu]):.4f}, {np.nanmax(g1[iu]):.4f}]  (all >= 0: {np.all(g1[iu] >= -1e-12)})"
)
print(f"g2 range [Ha]: [{np.nanmin(g2[iu]):.4f}, {np.nanmax(g2[iu]):.4f}]")

# ------------------------------------------------------------- plot
# g1 (~1e-2 Ha) and g2 (~1 Ha) differ by two orders of magnitude, so each
# panel gets its own colour scale (both are non-negative for LiH).
fig = plt.figure(figsize=(args.width, 0.32 * args.width))
gs = fig.add_gridspec(1, 5, width_ratios=[1, 0.03, 0.18, 1, 0.03], wspace=0.08)
n_occ = int(np.sum(occ > 0))

# both in Hartree; g1 ~1e-2 Ha needs three decimals
panels = (
    (g1, r"$g^{(1)}_{AB}$", "Ha", "{:.3f}"),
    (g2, r"$g^{(2)}_{AB}$", "Ha", "{:.2f}"),
)
for k, (M, title, unit, fmt) in enumerate(panels):
    ax, cax = fig.add_subplot(gs[0, 3 * k]), fig.add_subplot(gs[0, 3 * k + 1])
    vals = M[iu]  # upper triangle, A<B
    lo = min(0.0, np.nanmin(vals))
    norm = Normalize(lo, np.nanmax(np.abs(vals)))
    im = ax.imshow(
        M, norm=norm, interpolation="nearest", aspect="auto"
    )  # default colormap
    cmap = im.get_cmap()
    for i in range(n_act):
        for j in range(n_act):
            if i != j:
                r, g, b, _ = cmap(norm(M[i, j]))
                c = "k" if 0.299 * r + 0.587 * g + 0.114 * b > 0.5 else "w"
                ax.text(
                    j,
                    i,
                    fmt.format(M[i, j]),
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=c,
                )
    ax.set_xticks(range(n_act), labels)
    ax.set_yticks(range(n_act), labels if k == 0 else [])
    ax.tick_params(length=0)
    for a in (ax.axhline, ax.axvline):  # occupied | virtual RHF orbitals
        a(n_occ - 0.5, color="k", lw=0.6)
    ax.set_title(f"({'ab'[k]}) LiH: {title}", pad=3)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(unit, labelpad=1)
    cb.ax.tick_params(labelsize=6)

fig.savefig(f"{args.out}.pdf", bbox_inches="tight")
fig.savefig(f"{args.out}.png", dpi=300, bbox_inches="tight")
print(f"saved {args.out}.pdf / .png")
