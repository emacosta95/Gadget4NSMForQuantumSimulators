"""
Fidelity map of the minor-embedded O18 gadget over a (gamma, J_F) grid.

Reproduces the construction of DWAVEembeddingO18.ipynb with a *fixed* embedding
seed and sweeps both parameters over several decades, so that the working region
and the failure regions are visible on the same plot.

The `--c-mode` switch controls how the c_AB corrections are regularised:

  reachable  (default)  clip alpha_AB to >= --alpha-min.  The ansatz can only
                        realise alpha_AB > 1/2, because the two-particle penalty
                        is gamma*(1+c_AB) and alpha_AB = (1/2)(1 + 1/(1+c_AB)),
                        so c_AB < -1 turns the penalty NEGATIVE and the particle
                        number constraint is destroyed.  This mode is the
                        smallest change that keeps the constraint intact.
  notebook              the original `c[|c|>2] = |c|; clip(c,-2,2)`.
  raw                   no regularisation at all.  For O18 pair (0,2) has
                        c = -16.08, so the doubly occupied state (0,2) drops
                        ~15*gamma BELOW the one-hot manifold and the ground
                        state leaves the N=1 sector everywhere.

Examples
--------
    PYTHONPATH=. python fidelity_map_O18.py
    PYTHONPATH=. python fidelity_map_O18.py --c-mode notebook --out map_notebook.pdf
    PYTHONPATH=. python fidelity_map_O18.py --c-mode raw      --out map_raw.pdf
    PYTHONPATH=. python fidelity_map_O18.py --n-gamma 40 --n-jf 40
"""

import argparse

import numpy as np
import networkx as nx
import dwave_networkx as dnx
import minorminer
import qutip as qt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from ManyBodyQutip.qutip_class import SpinOperator
from src.interaction_utils import EffectiveInteractionOptimizer2ndVersion

LOGICAL_QUBITS = 6
N_TOT = 1


# ────────────────────────────── target problem ──────────────────────────────


def load_target():
    data = np.load("data/matrix_elements_h_eff_2body/one_body_nn_sd.npz")
    keys, values = data["keys"], data["values"]
    diagonal_elements = np.zeros(LOGICAL_QUBITS)
    g_matrix = np.zeros((LOGICAL_QUBITS, LOGICAL_QUBITS))
    for a, (i, j) in enumerate(keys):
        if i != j:
            g_matrix[i, j] = values[a]
        else:
            diagonal_elements[i] = values[a]
    return g_matrix, diagonal_elements


def nsm_ground_state(g_matrix, diagonal_elements):
    """Exact N=1 ground state of the quasiparticle Hamiltonian."""
    n = LOGICAL_QUBITS
    H = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            H += SpinOperator(
                [("x", i, "x", j)], coupling=[0.5 * g_matrix[i, j]], size=n
            ).qutip_op
            H += SpinOperator(
                [("y", i, "y", j)], coupling=[0.5 * g_matrix[i, j]], size=n
            ).qutip_op
    for i in range(n):
        H += SpinOperator([("qz", i)], coupling=[diagonal_elements[i]], size=n).qutip_op
    N_op = sum(
        SpinOperator([("qz", i)], coupling=[1], size=n).qutip_op for i in range(n)
    )
    evals, evecs = H.eigenstates()
    sector = [
        i for i, psi in enumerate(evecs) if abs(qt.expect(N_op, psi) - N_TOT) < 1e-6
    ]
    k = sector[int(np.argmin(evals[sector]))]
    return evals[k], evecs[k]


def qutip_idx(occupied, n_qubits):
    """QuTiP MSB-first index of the basis state with `occupied` qubits set."""
    return sum(1 << (n_qubits - 1 - q) for q in occupied)


# ─────────────────────────── c_AB regularisation ────────────────────────────


def regularise_c(alpha_raw, c_raw, mode, alpha_min):
    """Return (c_used, notes). See module docstring for the modes."""
    c = c_raw.copy()
    notes = []
    if mode == "raw":
        bad = [
            (i, j)
            for i in range(LOGICAL_QUBITS)
            for j in range(i + 1, LOGICAL_QUBITS)
            if c[i, j] <= -1
        ]
        for i, j in bad:
            notes.append(
                f"pair ({i},{j}): c={c[i,j]:.3f} <= -1 -> two-particle penalty "
                f"gamma*(1+c) = {1+c[i,j]:.3f}*gamma is NEGATIVE, constraint destroyed"
            )
    elif mode == "notebook":
        c[np.abs(c) > 4] = np.abs(c[np.abs(c) > 4])
        for i in range(LOGICAL_QUBITS):
            for j in range(i + 1, LOGICAL_QUBITS):
                if not np.isclose(c[i, j], c_raw[i, j]):
                    notes.append(
                        f"pair ({i},{j}): c {c_raw[i,j]:.3f} -> {c[i,j]:.3f} "
                        f"(alpha {alpha_raw[i,j]:.3f} -> {0.5*(1+1/(1+c[i,j])):.3f})"
                    )
    elif mode == "reachable":
        alpha = np.where(np.isnan(alpha_raw), np.nan, np.maximum(alpha_raw, alpha_min))
        c = 1.0 / (2.0 * alpha - 1.0) - 1.0
        for i in range(LOGICAL_QUBITS):
            for j in range(i + 1, LOGICAL_QUBITS):
                if not np.isclose(alpha[i, j], alpha_raw[i, j]):
                    notes.append(
                        f"pair ({i},{j}): alpha {alpha_raw[i,j]:.3f} < 1/2 is "
                        f"UNREACHABLE, raised to {alpha[i,j]:.3f} (c={c[i,j]:.3f})"
                    )
    else:
        raise ValueError(mode)
    return c, notes


# ─────────────────────────────── the embedding ───────────────────────────────


def make_embedding(seed=0, pegasus_size=8):
    K = nx.complete_graph(LOGICAL_QUBITS)
    hardware = dnx.pegasus_graph(pegasus_size)
    embedding = minorminer.find_embedding(K, hardware, random_seed=seed)

    used_nodes = [n for chain in embedding.values() for n in chain]
    used_edges = [
        (a, b)
        for u, v in K.edges()
        for a in embedding[u]
        for b in embedding[v]
        if hardware.has_edge(a, b)
    ]
    p2i = {p: i for i, p in enumerate(used_nodes)}
    local_edges = [(p2i[u], p2i[v]) for u, v in used_edges]

    chains = {l: [p2i[p] for p in c] for l, c in embedding.items()}
    intra_edges = []
    for chain in chains.values():
        intra_edges += [(chain[k], chain[k + 1]) for k in range(len(chain) - 1)]
    intra_set = set(map(frozenset, intra_edges))
    inter_edges = [(u, v) for u, v in local_edges if frozenset((u, v)) not in intra_set]
    p2l = {p: l for l, c in chains.items() for p in c}

    # one coupler per logical pair, matching the notebook's `added_pairs` logic
    couplers, seen = [], set()
    for u, v in inter_edges:
        lu, lv = p2l[u], p2l[v]
        if lu == lv:
            continue
        pair = frozenset((lu, lv))
        if pair in seen:
            continue
        seen.add(pair)
        couplers.append((u, v, lu, lv))
    assert len(couplers) == 15, f"only {len(couplers)}/15 logical pairs coupled"

    path_ordered = all(
        hardware.has_edge(used_nodes[a], used_nodes[b]) for a, b in intra_edges
    )
    return dict(
        chains=chains,
        intra_edges=intra_edges,
        couplers=couplers,
        n_physical=len(used_nodes),
        path_ordered=path_ordered,
    )


# ──────────────────────── fast Hamiltonian assembler ────────────────────────


class Assembler:
    """Precomputes the operator structure once; each (gamma, J_F) is then a
    few vector operations plus one 2^n x 2^n dense diagonalisation."""

    def __init__(self, emb, c_matrix, diagonal_elements, d_opt):
        self.emb = emb
        self.c = c_matrix
        self.diag = diagonal_elements
        self.d_opt = d_opt
        n = emb["n_physical"]
        self.n = n
        self.dim = 1 << n
        i = np.arange(self.dim)
        # bits[p] = occupation of physical qubit p (QuTiP MSB-first convention)
        self.bits = np.array([(i >> (n - 1 - p)) & 1 for p in range(n)])
        self.flip = np.array([i ^ (1 << (n - 1 - p)) for p in range(n)])
        self.rows = i

    def transverse_field(self, J_F):
        d_phys = np.zeros(self.n)
        for logical, chain in self.emb["chains"].items():
            k = len(chain)
            d_logical = self.d_opt[logical] / np.sqrt(2)
            magnitude = (np.abs(d_logical) ** (1 / k)) * (J_F ** ((k - 1) / k))
            for idx, phys in enumerate(chain):
                if k == 1:
                    d_phys[phys] = d_logical
                elif idx < k - 1:
                    d_phys[phys] = +magnitude
                else:
                    d_phys[phys] = -np.sign(d_logical) * magnitude
        return d_phys

    def __call__(self, gamma, J_F):
        b = self.bits
        diag = np.zeros(self.dim)
        for u, v, lu, lv in self.emb["couplers"]:
            diag += (2 + self.c[lu, lv]) * gamma * b[u] * b[v]
        h_constraint = gamma * (1 - 2 * N_TOT)
        for logical, chain in self.emb["chains"].items():
            eps = h_constraint / len(chain) + self.diag[logical] / (gamma * len(chain))
            for phys in chain:
                diag += eps * b[phys]
        for a, c_ in self.emb["intra_edges"]:
            diag += -J_F * (1 - 2 * b[a]) * (1 - 2 * b[c_])
        diag += gamma * (N_TOT**2) + J_F * len(self.emb["intra_edges"])

        H = np.diag(diag)
        d_phys = self.transverse_field(J_F)
        for p in range(self.n):
            if d_phys[p] != 0.0:
                H[self.rows, self.flip[p]] += d_phys[p]
        return H


def analyse(H, emb, exact_amps, gamma, target_diag, g_matrix):
    n = emb["n_physical"]
    idxs = [qutip_idx(emb["chains"][l], n) for l in range(LOGICAL_QUBITS)]
    w, V = np.linalg.eigh(H)
    gs = V[:, 0]

    amps = gs[idxs]
    imax = int(np.argmax(np.abs(exact_amps)))
    if abs(amps[imax]) > 1e-14:
        amps = amps * np.sign(amps[imax])
    F = float(np.abs(np.dot(exact_amps, amps)) ** 2)
    norm = float(np.sum(amps**2))

    A = V[np.ix_(idxs, range(LOGICAL_QUBITS))]
    U, s, Vt = np.linalg.svd(A)
    W = U @ Vt
    H_eff = (W @ np.diag(w[:LOGICAL_QUBITS]) @ W.T) * gamma
    d_err = np.diag(H_eff) - target_diag
    d_err = np.abs(d_err - d_err.mean()).max()
    off = H_eff - np.diag(np.diag(H_eff))
    return F, norm, float(d_err), float(np.abs(off - g_matrix).max())


# ────────────────────────────────── main ────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gamma-min", type=float, default=1.0)
    ap.add_argument("--gamma-max", type=float, default=2000.0)
    ap.add_argument("--jf-min", type=float, default=10.0)
    ap.add_argument("--jf-max", type=float, default=1e5)
    ap.add_argument("--n-gamma", type=int, default=32)
    ap.add_argument("--n-jf", type=int, default=32)
    ap.add_argument(
        "--c-mode", choices=["reachable", "notebook", "raw"], default="notebook"
    )
    ap.add_argument(
        "--alpha-min",
        type=float,
        default=0.55,
        help="floor on alpha_AB for --c-mode reachable (must exceed 0.5)",
    )
    ap.add_argument("--seed", type=int, default=0, help="minorminer embedding seed")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    g_matrix, diagonal_elements = load_target()
    _, psi0 = nsm_ground_state(g_matrix, diagonal_elements)
    a = psi0.full().flatten()
    exact_amps = np.array(
        [a[qutip_idx([l], LOGICAL_QUBITS)] for l in range(LOGICAL_QUBITS)]
    )
    exact_amps = np.real(
        exact_amps * np.exp(-1j * np.angle(exact_amps[np.argmax(np.abs(exact_amps))]))
    )

    opt = EffectiveInteractionOptimizer2ndVersion(LOGICAL_QUBITS, n_restarts=100)
    d_opt, _ = opt.optimize_rank1(g_matrix)
    alpha_raw, c_raw, _, _ = opt.get_alpha_and_c(g_matrix, d_opt)
    c_matrix, notes = regularise_c(alpha_raw, c_raw, args.c_mode, args.alpha_min)

    print(f"c-mode = {args.c_mode}")
    for note in notes:
        print("  " + note)
    if not notes:
        print("  (no pair required regularisation)")

    emb = make_embedding(seed=args.seed)
    print(
        f"chains: {emb['chains']}  lengths {[len(c) for c in emb['chains'].values()]}  "
        f"path-ordered: {emb['path_ordered']}  n_physical={emb['n_physical']}"
    )

    H_of = Assembler(emb, c_matrix, diagonal_elements, d_opt)

    # cross-check the fast assembler against a QuTiP build at one point
    ref = 0.0
    n_ph = emb["n_physical"]
    for u, v, lu, lv in emb["couplers"]:
        ref += SpinOperator(
            [("qz", u, "qz", v)], coupling=[(2 + c_matrix[lu, lv]) * 37.0], size=n_ph
        ).qutip_op
    for logical, chain in emb["chains"].items():
        eps = 37.0 * (1 - 2 * N_TOT) / len(chain) + diagonal_elements[logical] / (
            37.0 * len(chain)
        )
        for phys in chain:
            ref += SpinOperator([("qz", phys)], coupling=[eps], size=n_ph).qutip_op
    for x, y in emb["intra_edges"]:
        ref += SpinOperator([("z", x, "z", y)], coupling=[-410.0], size=n_ph).qutip_op
    ref += (37.0 * N_TOT**2 + 410.0 * len(emb["intra_edges"])) * qt.tensor(
        [qt.qeye(2)] * n_ph
    )
    d_phys = H_of.transverse_field(410.0)
    for p in range(n_ph):
        ref += SpinOperator([("x", p)], coupling=[d_phys[p]], size=n_ph).qutip_op
    err = np.abs(ref.full() - H_of(37.0, 410.0)).max()
    assert err < 1e-9, f"fast assembler disagrees with QuTiP by {err}"
    print(f"fast assembler verified against QuTiP (max abs diff {err:.2e})")

    gammas = np.logspace(
        np.log10(args.gamma_min), np.log10(args.gamma_max), args.n_gamma
    )
    JFs = np.logspace(np.log10(args.jf_min), np.log10(args.jf_max), args.n_jf)

    F = np.zeros((args.n_gamma, args.n_jf))
    NRM = np.zeros_like(F)
    DIAG = np.zeros_like(F)
    OFF = np.zeros_like(F)
    for i, gamma in enumerate(gammas):
        for j, J_F in enumerate(JFs):
            F[i, j], NRM[i, j], DIAG[i, j], OFF[i, j] = analyse(
                H_of(gamma, J_F), emb, exact_amps, gamma, diagonal_elements, g_matrix
            )
        print(f"gamma={gamma:9.3f}  F: {F[i].min():.3f} .. {F[i].max():.3f}")

    best = np.unravel_index(np.argmax(F), F.shape)
    print(
        f"\nbest F = {F[best]:.5f} at gamma={gammas[best[0]]:.2f}, J_F={JFs[best[1]]:.1f}"
    )

    tag = args.c_mode
    np.savez(
        f"data/fidelity_map_O18_{tag}.npz",
        F=F,
        norm=NRM,
        diag_err=DIAG,
        off_err=OFF,
        gammas=gammas,
        JFs=JFs,
        c_mode=tag,
    )

    # ───────────────────────────── figures ─────────────────────────────
    INFID_FLOOR = 1e-6
    INF = np.clip(1.0 - F, INFID_FLOOR, 1.0)
    G, J = np.meshgrid(gammas, JFs, indexing="ij")
    gg = np.logspace(np.log10(gammas[0]), np.log10(gammas[-1]), 200)
    vmin = max(INFID_FLOOR, INF.min())

    fig = plt.figure(figsize=(12.5, 5.2))
    gs_ = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.28)

    # (a) infidelity map, log colour scale
    ax = fig.add_subplot(gs_[0])
    pc = ax.pcolormesh(
        G, J, INF, shading="nearest", cmap="magma_r", norm=LogNorm(vmin=vmin, vmax=1.0)
    )
    cs = ax.contour(
        G, J, INF, levels=[1e-2, 1e-1, 0.5], colors="k", linewidths=[2, 1.5, 1]
    )
    ax.clabel(
        cs,
        fmt=lambda v: f"$10^{{{np.log10(v):.0f}}}$" if v < 0.2 else "0.5",
        fontsize=10,
    )
    ax.plot(gg, gg**2, "c--", lw=2.2, label=r"$J_F=\gamma^2$")
    ax.plot(
        gammas[best[0]],
        JFs[best[1]],
        "c*",
        ms=18,
        mec="k",
        label=r"best $1-F$=" + f"{1-F[best]:.1e}",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(gammas[0], gammas[-1])
    ax.set_ylim(JFs[0], JFs[-1])
    ax.set_xlabel(r"$\gamma\,\omega^{-1}$", fontsize=17)
    ax.set_ylabel(r"$J_F\,\omega^{-1}$", fontsize=17)
    ax.set_title(r"$1-F$", fontsize=14)
    ax.legend(fontsize=11, loc="lower right", framealpha=0.9)
    ax.tick_params(labelsize=12)
    plt.colorbar(pc, ax=ax, label=r"$1-F$")

    # (b) cuts at fixed J_F
    ax = fig.add_subplot(gs_[1])
    sel = np.linspace(0, len(JFs) - 1, 5).astype(int)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(sel)))
    for col, j in zip(colors, sel):
        ax.plot(
            gammas,
            INF[:, j],
            "-o",
            ms=3.5,
            lw=2,
            color=col,
            label=rf"$J_F\omega^{{-1}}=${JFs[j]:.3g}",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(vmin * 0.7, 2.0)
    ax.set_xlabel(r"$\gamma\,\omega^{-1}$", fontsize=17)
    ax.set_ylabel(r"$1-F$", fontsize=17)
    ax.set_title(r"cuts at fixed $J_F$", fontsize=13)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=12)

    out = args.out or f"fidelity_map_O18_{tag}.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=140)
    plt.close(fig)

    # ── second figure: what drives the two failure regions ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    pc = ax.pcolormesh(G, J, NRM, shading="nearest", cmap="cividis", vmin=0, vmax=1)
    ax.plot(gg, gg**2, "r--", lw=2, label=r"$J_F=\gamma^2$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(gammas[0], gammas[-1])
    ax.set_ylim(JFs[0], JFs[-1])
    ax.set_xlabel(r"$\gamma\,\omega^{-1}$", fontsize=16)
    ax.set_ylabel(r"$J_F\,\omega^{-1}$", fontsize=16)
    ax.set_title(
        "Weight left in the one-hot sector\n(low = leakage / broken chains)",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=12)
    plt.colorbar(pc, ax=ax)

    ax = axes[1]
    span = diagonal_elements.max() - diagonal_elements.min()
    pc = ax.pcolormesh(
        G,
        J,
        np.clip(DIAG, 1e-3, None),
        shading="nearest",
        cmap="magma",
        norm=LogNorm(vmin=1e-2, vmax=max(1e2, DIAG.max())),
    )
    ax.contour(G, J, DIAG, levels=[span], colors="c", linewidths=2)
    ax.plot(gg, gg**2, "r--", lw=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(gammas[0], gammas[-1])
    ax.set_ylim(JFs[0], JFs[-1])
    ax.set_xlabel(r"$\gamma\,\omega^{-1}$", fontsize=16)
    ax.set_ylabel(r"$J_F\,\omega^{-1}$", fontsize=16)
    ax.set_title(
        "Spread of the effective logical\n"
        f"diagonal error (cyan = physical spread {span:.1f})",
        fontsize=12,
    )
    ax.tick_params(labelsize=12)
    plt.colorbar(pc, ax=ax)
    plt.tight_layout()
    diag_out = out.replace(".pdf", "_diagnostics.pdf")
    plt.savefig(diag_out, bbox_inches="tight", dpi=150)
    plt.savefig(diag_out.replace(".pdf", ".png"), bbox_inches="tight", dpi=140)
    print(f"wrote {out}, {diag_out} (+ .png) and data/fidelity_map_O18_{tag}.npz")


if __name__ == "__main__":
    main()
