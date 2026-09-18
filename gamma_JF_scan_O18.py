"""
Systematic study of the minor-embedded O18 gadget: fidelity vs (gamma, J_F).

Companion to DWAVEembeddingO18.ipynb. Reproduces the notebook's construction
exactly, with a *fixed* embedding seed, and adds three diagnostics:

  1. F(gamma, J_F)          -- the 2D fidelity map
  2. norm(gamma, J_F)       -- weight remaining in the one-hot logical sector
                               (separates "leakage" from "wrong logical state")
  3. gamma * H_eff          -- the effective 6x6 logical Hamiltonian actually
                               realised, compared against diag(eps) + g_matrix

The point of (2)+(3): at large gamma the embedded ground state stays perfectly
inside the one-hot sector (norm -> 1) while the fidelity collapses. The hopping
part of H_eff is essentially exact; it is the *diagonal* that is destroyed, by a
site-dependent chain-breaking self-energy that scales like gamma^2 / J_F.

Run:  PYTHONPATH=. python gamma_JF_scan_O18.py
"""

import numpy as np
import networkx as nx
import dwave_networkx as dnx
import minorminer
import qutip as qt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
            H += SpinOperator([("x", i, "x", j)], coupling=[0.5 * g_matrix[i, j]], size=n).qutip_op
            H += SpinOperator([("y", i, "y", j)], coupling=[0.5 * g_matrix[i, j]], size=n).qutip_op
    for i in range(n):
        H += SpinOperator([("qz", i)], coupling=[diagonal_elements[i]], size=n).qutip_op
    N_op = sum(SpinOperator([("qz", i)], coupling=[1], size=n).qutip_op for i in range(n))
    evals, evecs = H.eigenstates()
    sector = [i for i, psi in enumerate(evecs) if abs(qt.expect(N_op, psi) - N_TOT) < 1e-6]
    k = sector[int(np.argmin(evals[sector]))]
    return evals[k], evecs[k]


# ─────────────────────────────── the embedding ───────────────────────────────

def make_embedding(seed=0, pegasus_size=8):
    """Deterministic minor embedding of K6 into Pegasus, in local qubit indices."""
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
    phys_to_idx = {p: i for i, p in enumerate(used_nodes)}
    local_edges = [(phys_to_idx[u], phys_to_idx[v]) for u, v in used_edges]

    chains = {l: [phys_to_idx[p] for p in c] for l, c in embedding.items()}
    intra_edges = []
    for chain in chains.values():
        intra_edges += [(chain[k], chain[k + 1]) for k in range(len(chain) - 1)]

    intra_set = set(map(frozenset, intra_edges))
    inter_edges = [(u, v) for u, v in local_edges if frozenset((u, v)) not in intra_set]
    phys_to_logical = {p: l for l, c in chains.items() for p in c}

    # the notebook assumes chain lists are path-ordered; verify it
    path_ordered = all(
        hardware.has_edge(used_nodes[a], used_nodes[b]) for a, b in intra_edges
    )
    return dict(
        chains=chains,
        intra_edges=intra_edges,
        inter_edges=inter_edges,
        phys_to_logical=phys_to_logical,
        n_physical=len(used_nodes),
        path_ordered=path_ordered,
    )


def qutip_idx(occupied, n_qubits):
    """QuTiP MSB-first index of the basis state with `occupied` qubits set."""
    return sum(1 << (n_qubits - 1 - q) for q in occupied)


# ────────────────────────────── embedded gadget ──────────────────────────────

def build_embedded(gamma, J_F, emb, c_matrix, diagonal_elements, d_opt):
    chains = emb["chains"]
    n_ph = emb["n_physical"]
    p2l = emb["phys_to_logical"]
    identity = qt.tensor([qt.qeye(2)] * n_ph)

    hamiltonian_zz = 0.0
    added = set()
    for u, v in emb["inter_edges"]:
        lu, lv = p2l[u], p2l[v]
        if lu == lv:
            continue
        pair = frozenset((lu, lv))
        if pair in added:
            continue
        added.add(pair)
        hamiltonian_zz += SpinOperator(
            [("qz", u, "qz", v)],
            coupling=[(2 + c_matrix[lu, lv]) * gamma],
            size=n_ph,
        ).qutip_op

    h_constraint = gamma * (1 - 2 * N_TOT)
    H_diagonal = 0.0
    for logical, chain in chains.items():
        eps = h_constraint / len(chain) + diagonal_elements[logical] / (gamma * len(chain))
        for phys in chain:
            H_diagonal += SpinOperator([("qz", phys)], coupling=[eps], size=n_ph).qutip_op

    H_ferro = 0.0
    for a, b in emb["intra_edges"]:
        H_ferro += SpinOperator([("z", a, "z", b)], coupling=[-J_F], size=n_ph).qutip_op

    longitudinal = (
        hamiltonian_zz
        + gamma * (N_TOT**2) * identity
        + H_ferro
        + H_diagonal
        + J_F * len(emb["intra_edges"]) * identity
    )

    # transverse field: chain-flip is generated at k-th order, hence J_F**((k-1)/k)
    d_phys = np.zeros(n_ph)
    for logical, chain in chains.items():
        k = len(chain)
        d_logical = d_opt[logical] / np.sqrt(2)
        magnitude = (np.abs(d_logical) ** (1 / k)) * (J_F ** ((k - 1) / k))
        for i, phys in enumerate(chain):
            if k == 1:
                d_phys[phys] = d_logical
            elif i < k - 1:
                d_phys[phys] = +magnitude
            else:
                d_phys[phys] = -np.sign(d_logical) * magnitude

    transverse = sum(
        SpinOperator([("x", p)], coupling=[d_phys[p]], size=n_ph).qutip_op
        for p in range(n_ph)
    )
    return longitudinal + transverse, d_phys


# ─────────────────────────────── observables ────────────────────────────────

def analyse(H, emb, exact_amps, gamma):
    """Fidelity, one-hot norm, and the realised effective logical Hamiltonian."""
    chains, n_ph = emb["chains"], emb["n_physical"]
    idxs = [qutip_idx(chains[l], n_ph) for l in range(LOGICAL_QUBITS)]

    w, V = np.linalg.eigh(H.full())
    gs = V[:, 0]

    emb_amps = np.array([gs[i] for i in idxs])
    imax = int(np.argmax(np.abs(exact_amps)))
    if abs(emb_amps[imax]) > 1e-14:
        emb_amps = emb_amps * np.exp(-1j * np.angle(emb_amps[imax]))
    F = np.abs(np.vdot(exact_amps, emb_amps)) ** 2
    norm = float(np.sum(np.abs(emb_amps) ** 2))

    # Lowdin-orthogonalised projection of the 6 lowest levels onto the one-hot
    # subspace -> the effective logical Hamiltonian, rescaled by gamma
    A = V[np.ix_(idxs, range(LOGICAL_QUBITS))]
    U, s, Vt = np.linalg.svd(A)
    W = U @ Vt
    H_eff = np.real(W @ np.diag(w[:LOGICAL_QUBITS]) @ W.conj().T) * gamma
    return F, norm, H_eff, s.min()


# ────────────────────────────────── main ────────────────────────────────────

def main():
    g_matrix, diagonal_elements = load_target()
    E0, psi0 = nsm_ground_state(g_matrix, diagonal_elements)
    amps = psi0.full().flatten()
    exact_amps = np.array([amps[qutip_idx([l], LOGICAL_QUBITS)] for l in range(LOGICAL_QUBITS)])
    exact_amps = exact_amps * np.exp(-1j * np.angle(exact_amps[np.argmax(np.abs(exact_amps))]))
    print(f"Exact N=1 ground state energy: {E0:.6f}")

    opt = EffectiveInteractionOptimizer2ndVersion(LOGICAL_QUBITS, n_restarts=100)
    d_opt, _ = opt.optimize_rank1(g_matrix)
    alpha_matrix, c_matrix, report, _ = opt.get_alpha_and_c(g_matrix, d_opt)

    # the notebook's clipping of c -- record which pairs it distorts
    c_raw = c_matrix.copy()
    c_matrix[np.abs(c_matrix) > 2] = np.abs(c_matrix[np.abs(c_matrix) > 2])
    c_matrix = np.clip(c_matrix, -2, 2)
    clipped = [
        (i, j)
        for i in range(LOGICAL_QUBITS)
        for j in range(i + 1, LOGICAL_QUBITS)
        if not np.isclose(c_raw[i, j], c_matrix[i, j])
    ]
    if clipped:
        print(f"c_AB clipped on pairs {clipped} -> these couplings are systematically wrong")
        alpha_clipped = 0.5 * (1 + 1 / (1 + c_matrix))
        g_achievable = -alpha_clipped * np.outer(d_opt, d_opt)
        np.fill_diagonal(g_achievable, 0.0)
        _, psi_ach = nsm_ground_state(g_achievable, diagonal_elements)
        a = psi_ach.full().flatten()
        a = np.array([a[qutip_idx([l], LOGICAL_QUBITS)] for l in range(LOGICAL_QUBITS)])
        a = a * np.exp(-1j * np.angle(a[np.argmax(np.abs(a))]))
        print(f"  -> fidelity ceiling set by the clipping: {np.abs(np.vdot(exact_amps, a))**2:.6f}")

    emb = make_embedding(seed=0)
    print(f"chains: { {l: c for l, c in emb['chains'].items()} }")
    print(f"chain lengths: {[len(c) for c in emb['chains'].values()]}, "
          f"path-ordered chains: {emb['path_ordered']}")

    gammas = np.array([5, 10, 20, 30, 50, 100, 200, 400, 800])
    JFs = np.array([10, 30, 100, 300, 1000, 3000, 10000])

    F = np.zeros((len(gammas), len(JFs)))
    NRM = np.zeros_like(F)
    DIAGERR = np.zeros_like(F)

    target_diag = diagonal_elements
    for i, gamma in enumerate(gammas):
        for j, J_F in enumerate(JFs):
            H, _ = build_embedded(gamma, J_F, emb, c_matrix, diagonal_elements, d_opt)
            f, nrm, H_eff, _ = analyse(H, emb, exact_amps, gamma)
            F[i, j], NRM[i, j] = f, nrm
            err = np.diag(H_eff) - target_diag
            DIAGERR[i, j] = np.abs(err - err.mean()).max()
        print(f"gamma={gamma:6g} | F: " + " ".join(f"{x:.4f}" for x in F[i]))

    print("\ncolumns are J_F =", JFs)
    print("\nnorm remaining in the one-hot logical sector:")
    for i, gamma in enumerate(gammas):
        print(f"gamma={gamma:6g} | " + " ".join(f"{x:.4f}" for x in NRM[i]))
    print("\nspread of the effective logical diagonal error "
          f"(target spread is {target_diag.max()-target_diag.min():.2f}):")
    for i, gamma in enumerate(gammas):
        print(f"gamma={gamma:6g} | " + " ".join(f"{x:8.3f}" for x in DIAGERR[i]))

    np.savez("data/gamma_JF_scan_O18.npz", F=F, norm=NRM, diag_err=DIAGERR,
             gammas=gammas, JFs=JFs)

    # ── figure: fidelity map, with the J_F ~ gamma^2 crossover overlaid ──
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    G, J = np.meshgrid(gammas, JFs, indexing="ij")

    for ax, Z, title, kw in [
        (axes[0], F, "Fidelity", dict(vmin=0, vmax=1, cmap="viridis")),
        (axes[1], NRM, "Norm in one-hot sector", dict(vmin=0, vmax=1, cmap="viridis")),
        (axes[2], np.log10(DIAGERR + 1e-6), r"$\log_{10}$ diagonal error", dict(cmap="magma")),
    ]:
        pc = ax.pcolormesh(G, J, Z, shading="nearest", **kw)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$\gamma$", fontsize=15)
        ax.set_ylabel(r"$J_F$", fontsize=15)
        ax.set_title(title, fontsize=14)
        gg = np.logspace(np.log10(gammas[0]), np.log10(gammas[-1]), 50)
        ax.plot(gg, gg**2, "w--", lw=2, label=r"$J_F=\gamma^2$")
        ax.set_ylim(JFs[0], JFs[-1])
        ax.legend(fontsize=11)
        plt.colorbar(pc, ax=ax)

    plt.tight_layout()
    plt.savefig("gamma_JF_scan_O18.pdf", bbox_inches="tight")
    print("\nwrote gamma_JF_scan_O18.pdf and data/gamma_JF_scan_O18.npz")


if __name__ == "__main__":
    main()
