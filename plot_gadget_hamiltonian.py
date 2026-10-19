r"""
Fig. (couplings): g1 and g2 of H_NSM-Q for neutron pairs, in one row:
(a,b) sd-shell (USDB) g1, g2; (c,d) pf-shell (GXPF1A) g1, g2.

Sized for a full-width figure* in a two-column REVTeX (PRR) layout:
\textwidth = 7.0 in (178 mm). Include with
    \includegraphics[width=\textwidth]{fig_couplings_sd_pf.pdf}

Usage:
    python plot_couplings_sd_pf.py [path/to/matrix_elements_h_eff_2body]
"""

import sys
from collections import deque
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

DATA = sys.argv[1] if len(sys.argv) > 1 else "data/matrix_elements_h_eff_2body"
OUT = "fig_couplings_sd_pf"

# ---- figure geometry (inches): full two-column width, rectangular ----
FIG_W, FIG_H = 7.0, 2.25
ANNOTATE = {"sd": True, "pf": False}  # numbers in cells (pf 10x10 is too dense)

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
    }
)

# repo index -> orbital (inferred from h_A degeneracies; check vs NSMFermions)
SHELLS = {
    "sd": dict(
        onebody="one_body_nn_sd.npz",
        twobody="twobody_nn_sd.npz",
        name=r"$sd$-shell",
        groups=[(r"$0d_{5/2}$", 3), (r"$1s_{1/2}$", 1), (r"$0d_{3/2}$", 2)],
    ),
    "pf": dict(
        onebody="one_body_nn_pf.npz",
        twobody="twobody_nn_pf.npz",
        name=r"$pf$-shell",
        groups=[
            (r"$0f_{7/2}$", 4),
            (r"$1p_{3/2}$", 2),
            (r"$0f_{5/2}$", 3),
            (r"$1p_{1/2}$", 1),
        ],
    ),
}


def load_onebody(path):
    d = np.load(path)
    keys, vals = d["keys"], d["values"]
    D = int(keys.max()) + 1
    h, g1 = np.zeros(D), np.zeros((D, D))
    for (i, j), v in zip(keys, vals):
        if i == j:
            h[i] = v
        else:
            g1[i, j] = v
    if np.allclose(np.tril(g1, -1), 0):  # only upper triangle stored
        g1 = g1 + g1.T
    return h, g1


def load_twobody(path, D):
    d = np.load(path)
    g2 = np.zeros((D, D))
    for k, v in zip(d["keys"], d["values"]):
        i, j = int(k[0]), int(k[1])
        g2[i, j] = g2[j, i] = v
    return g2


def gauge(g1, tol=1e-12):
    """s_A = +-1 such that s_A s_B g1_AB <= 0 (BFS); returns s, frustrated edges."""
    D = len(g1)
    s = np.zeros(D, int)
    s[0] = 1
    q = deque([0])
    while q:
        A = q.popleft()
        for B in range(D):
            if B != A and abs(g1[A, B]) > tol and s[B] == 0:
                s[B] = -s[A] * int(np.sign(g1[A, B]))
                q.append(B)
    frustrated = [
        (A, B)
        for A in range(D)
        for B in range(A + 1, D)
        if s[A] * s[B] * g1[A, B] > tol
    ]
    return s, frustrated


def draw(ax, M, groups, vmax, annotate):
    D = len(M)
    M = M.astype(float).copy()
    np.fill_diagonal(M, np.nan)
    norm = Normalize(-vmax, vmax)  # symmetric range, default colormap (viridis)
    im = ax.imshow(M, norm=norm, interpolation="nearest")
    cmap = im.get_cmap()
    # orbital blocks: separator lines + one label per orbital
    edges = np.cumsum([0] + [n for _, n in groups])
    for e in edges[1:-1]:
        ax.axhline(e - 0.5, color="k", lw=0.6)
        ax.axvline(e - 0.5, color="k", lw=0.6)
    centres = [(edges[k] + edges[k + 1] - 1) / 2 for k in range(len(groups))]
    labels = [lab for lab, _ in groups]
    ax.set_xticks(centres, labels)
    ax.set_yticks(centres, labels)
    ax.tick_params(length=0)
    if annotate:
        for i in range(D):
            for j in range(D):
                if i != j:
                    r_, g_, b_, _ = cmap(norm(M[i, j]))
                    c = "k" if 0.299 * r_ + 0.587 * g_ + 0.114 * b_ > 0.5 else "w"
                    ax.text(
                        j,
                        i,
                        f"{M[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=4.2,
                        color=c,
                    )
    return im


fig = plt.figure(figsize=(FIG_W, FIG_H))
gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 0.16, 1, 1, 0.05], wspace=0.08)
axes = [fig.add_subplot(gs[0, k]) for k in (0, 1, 3, 4)]
cax = fig.add_subplot(gs[0, 5])
titles = [r"$g^{(1)}_{AB}$", r"$g^{(2)}_{AB}$"]
panel = iter("abcd")

mats = {}
for key in ["sd", "pf"]:
    cfg = SHELLS[key]
    h, g1 = load_onebody(f"{DATA}/{cfg['onebody']}")
    g2 = load_twobody(f"{DATA}/{cfg['twobody']}", len(h))
    s, fr = gauge(g1)  # gauge only reported in the text, not plotted
    print(f"{key}: gauge s = {s}, frustrated edges = {fr}")
    mats[key] = (g1, g2)
vmax = max(np.abs(M).max() for pair in mats.values() for M in pair)  # one shared scale

for r, key in enumerate(["sd", "pf"]):
    cfg = SHELLS[key]
    for c, M in enumerate(mats[key]):
        ax = axes[2 * r + c]
        im = draw(ax, M, cfg["groups"], vmax, ANNOTATE[key])
        ax.set_title(f"({next(panel)}) {cfg['name']}: {titles[c]}", pad=3)
        if c > 0:
            ax.set_yticklabels([])

cb = fig.colorbar(im, cax=cax)
cb.set_label("MeV", labelpad=2)
cb.ax.tick_params(labelsize=6.5)

fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}.png", dpi=300, bbox_inches="tight")
print(f"saved {OUT}.pdf / .png  ({FIG_W} x {FIG_H} in)")
