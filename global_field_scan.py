"""O18 gadget with a GLOBAL transverse field and tuned chain couplings, on the
same (gamma, chi = J_F / gamma) grid and in the same PCE convention as the
local-field minor-embedding maps (embedding_scan.py).

Scheme (GadgetO18_GlobalFieldChainTuning.ipynb):
  * every logical qubit a is a chain of k = 2 physical qubits (minorminer
    chains of length 1 padded with a free Pegasus neighbour): 12 qubits;
  * ONE global field  Omega = sqrt(J_F)  on every qubit;
  * chain bond  J_F,a = J_F r_a,  r_a = (1 - t_a^2 / J_F) / t_a  (exact k = 2
    inversion), so the chain doublet is split by exactly 2 t_a with
    t_a = |d_a| / sqrt 2: the chain plays the role of the local drive d_a;
  * signs of d_a fixed by the gauge sigma_a = sign(d_a) (-1)^(k_a - 1);
  * constraint couplers and longitudinal fields from the unembedded PCE
    gadget with c_AB = 0 (rank-1 drive) and the exact self-energy.

The local-field comparison uses the K6 embedding of the local-field maps
(chains of length 1 and 2, local drives d_a / k_a on the chain qubits).

Usage
-----
  python global_field_scan.py            # scan + figures
  python global_field_scan.py --plot-only

Outputs
-------
  data/systematic_analysis/global_field_O18.npz
  data/systematic_analysis/scan_embedding_O18_chi1e5.npz   (local field, same grid)
  figures/systematic_analysis/global_field_O18_embedding.pdf  (logical K6 + physical graph)
  figures/systematic_analysis/global_vs_local_O18.pdf         (1 - F(chi, gamma) maps)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import dwave_networkx as dnx  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from src.gadget import (  # noqa: E402
    GadgetHamiltonian,
    GadgetParameters,
    MinorEmbeddedEncoding,
    ParticleConservationEncoding,
    QuasiparticleProblem,
    find_minor_embedding,
    fit_drive,
)
from src.gadget.embedding import from_minorminer  # noqa: E402

ONEBODY = "data/matrix_elements_h_eff_2body/one_body_nn_sd.npz"
OUT = "data/systematic_analysis/global_field_O18.npz"
LOCAL = "data/systematic_analysis/scan_embedding_O18_chi1e5.npz"
FIGDIR = "figures/systematic_analysis"


# ----------------------------------------------------------------- set-up
def setup():
    problem = QuasiparticleProblem.from_npz(ONEBODY, n_levels=6, n_particles=1, label="O18")
    drive = fit_drive(problem.g1, n_restarts=30, use_c=False, seed=42)  # PCE: c_AB = 0
    D = problem.n_levels
    hardware = dnx.pegasus_graph(8)
    emb_raw = find_minor_embedding(D, hardware=hardware, tries=10, seed=1)
    chains_hw = {a: [emb_raw.hardware_of[q] for q in ch] for a, ch in emb_raw.chains.items()}
    used = {q for ch in chains_hw.values() for q in ch}
    for a, ch in chains_hw.items():  # pad length-1 chains -> every chain tunable
        if len(ch) == 1:
            free = next(v for v in hardware.neighbors(ch[0]) if v not in used)
            ch.append(free)
            used.add(free)
    emb = from_minorminer(chains_hw, hardware, nx.complete_graph(D))
    assert emb.check(verbose=False)
    k_len = np.array([len(emb.chains[a]) for a in range(D)])
    assert set(k_len.tolist()) == {2}
    return problem, drive, emb, k_len


def chain_doublet(k, J, Omega):
    """(half-splitting t, centre) of the two lowest levels of an isolated chain."""
    dim = 1 << k
    idx = np.arange(dim)
    spins = [1 - 2 * ((idx >> (k - 1 - q)) & 1) for q in range(k)]
    diag = np.zeros(dim)
    for q in range(k - 1):
        diag += -J * spins[q] * spins[q + 1] + J
    H = np.diag(diag)
    for q in range(k):
        H[idx, idx ^ (1 << (k - 1 - q))] += Omega
    w = np.linalg.eigvalsh(H)[:2]
    return 0.5 * (w[1] - w[0]), 0.5 * (w[0] + w[1])


class GlobalFieldO18:
    def __init__(self):
        self.problem, self.drive, self.emb, self.k_len = setup()
        self.D = self.problem.n_levels
        self.E_exact, self.psi_exact = self.problem.exact_groundstate()
        self.d = self.drive.d
        self.t_target = np.abs(self.d) / np.sqrt(2)
        self.sigma = np.sign(self.d) * (-1.0) ** (self.k_len - 1)
        self.p2l = self.emb.phys_to_logical()
        self.encoding = MinorEmbeddedEncoding(ParticleConservationEncoding(self.problem), self.emb)

    def ratios(self, J_F):
        return (1.0 - self.t_target**2 / J_F) / self.t_target  # exact k = 2 inversion

    def base_terms(self, gamma):
        H0 = GadgetHamiltonian(self.problem, ParticleConservationEncoding(self.problem), self.drive,
                               GadgetParameters(gamma=gamma, self_energy="exact", use_c=False))
        return H0._base_terms(H0.encoding)

    def hamiltonian(self, gamma, Omega, J_chain):
        zz, z, const = self.base_terms(gamma)
        emb, n = self.emb, self.emb.n_physical
        idx = np.arange(1 << n)
        N = [((idx >> (n - 1 - q)) & 1).astype(float) for q in range(n)]
        diag = np.full(1 << n, float(const))
        couplers = emb.couplers()
        for (a, b), w in zz.items():
            u, v = couplers[(a, b)]
            diag += w * N[u] * N[v]
        for a, h in z.items():
            for q in emb.chains[a]:
                diag += h / len(emb.chains[a]) * N[q]
        for u, v in emb.intra_edges:
            J = J_chain[self.p2l[u]]
            diag += -J * (1 - 2 * N[u]) * (1 - 2 * N[v]) + J
        H = np.diag(diag)
        for q in range(n):  # GLOBAL transverse field
            H[idx, idx ^ (1 << (n - 1 - q))] += Omega
        return H

    def point(self, gamma, J_F):
        Omega = np.sqrt(J_F)
        r = self.ratios(J_F)
        J_chain = {a: J_F * r[a] for a in range(self.D)}
        offset = sum(chain_doublet(int(self.k_len[a]), J_chain[a], Omega)[1] for a in range(self.D))
        w, v = sla.eigh(self.hamiltonian(gamma, Omega, J_chain), subset_by_index=[0, 1])
        amps = self.sigma * self.encoding.project(v[:, 0]).real
        norm2 = float(amps @ amps)
        F = float((amps @ self.psi_exact) ** 2)
        energy = (w[0] - offset) * gamma
        return dict(infidelity=1 - F, infidelity_code=1 - F / norm2, leakage=1 - norm2,
                    rel_energy_error=abs(energy - self.E_exact) / abs(self.E_exact),
                    gap=(w[1] - w[0]) * gamma, r=r)


# ------------------------------------------------------------------ scan
def scan(n_gamma, n_chi, chi_max):
    m = GlobalFieldO18()
    gammas = np.logspace(0, np.log10(400), n_gamma)
    chis = np.logspace(0, np.log10(chi_max), n_chi)
    keys = ("infidelity", "infidelity_code", "leakage", "rel_energy_error", "gap")
    grid = {k: np.full((n_gamma, n_chi), np.nan) for k in keys}
    if os.path.exists(OUT):  # resume
        old = np.load(OUT, allow_pickle=True)
        if np.allclose(old["gammas"], gammas) and np.allclose(old["ratios"], chis):
            grid = {k: old[k].copy() for k in keys}
    print(f"O18 global field: {m.emb.n_physical} physical qubits, chains {m.emb.chain_lengths()}, "
          f"r_a(J_F->inf) = {np.round(1 / m.t_target, 3)}", flush=True)
    for i, g in enumerate(gammas):
        if not np.isnan(grid["infidelity"][i]).any():
            continue
        t0 = time.time()
        for j, chi in enumerate(chis):
            r = m.point(g, chi * g)
            for k in keys:
                grid[k][i, j] = r[k]
        np.savez(OUT, gammas=gammas, ratios=chis, d=m.d, chains=m.k_len,
                 r_inf=1 / m.t_target, **grid)
        print(f"gamma={g:7.2f}  best 1-F={np.nanmin(grid['infidelity'][i]):.2e} "
              f"at chi={chis[np.nanargmin(grid['infidelity'][i])]:.3g}  [{time.time()-t0:.0f}s]", flush=True)
    print("->", OUT)


# --------------------------------------------------------------- figures
def log_edges(x):
    lx = np.log10(x)
    mid = 0.5 * (lx[1:] + lx[:-1])
    return 10 ** np.concatenate(([2 * lx[0] - mid[0]], mid, [2 * lx[-1] - mid[-1]]))


def plot_graph(path):
    m = GlobalFieldO18()
    emb, D, d = m.emb, m.D, m.d
    loc = find_minor_embedding(D, pegasus_size=8, tries=6, seed=1)  # local-field embedding
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    # (a) logical K6
    G = nx.complete_graph(D)
    pos = nx.circular_layout(G)
    w = np.array([abs(m.problem.g1[u, v]) for u, v in G.edges()])
    nx.draw_networkx_edges(G, pos, ax=axes[0], width=4 * w / w.max(), alpha=0.6)
    nodes = nx.draw_networkx_nodes(G, pos, ax=axes[0], node_color=d, cmap="coolwarm",
                                   vmin=-1.2, vmax=1.2, node_size=800)
    nx.draw_networkx_labels(G, pos, ax=axes[0], labels={a: str(a) for a in range(D)}, font_size=13)
    cb = fig.colorbar(nodes, ax=axes[0], shrink=0.8)
    cb.set_label("$d_a$", fontsize=14)
    axes[0].set_title("(a)", loc="left", fontweight="bold", fontsize=15)
    axes[0].axis("off")

    def draw_physical(ax, e, label, chain_w=None):
        Gp = nx.Graph()
        Gp.add_nodes_from(range(e.n_physical))
        p2l = e.phys_to_logical()
        Gp.add_edges_from(e.intra_edges, kind="chain")
        Gp.add_edges_from(e.couplers().values(), kind="coupler")
        # chain bonds short (heavy), couplers long: chains stay compact
        Hw = nx.Graph()
        Hw.add_nodes_from(Gp)
        Hw.add_edges_from(((u, v, {"w": 0.35 if Gp.edges[u, v]["kind"] == "chain" else 1.0}) for u, v in Gp.edges()))
        pp = nx.kamada_kawai_layout(Hw, weight="w")
        ce = [x for x in Gp.edges() if Gp.edges[x]["kind"] == "chain"]
        ke = [x for x in Gp.edges() if Gp.edges[x]["kind"] == "coupler"]
        nx.draw_networkx_edges(Gp, pp, edgelist=ke, ax=ax, style="--", alpha=0.4)
        widths = [5 if chain_w is None else 2 + 3 * chain_w[p2l[u]] for u, _ in ce]
        nx.draw_networkx_edges(Gp, pp, edgelist=ce, ax=ax, width=widths, edge_color="k")
        nx.draw_networkx_nodes(Gp, pp, ax=ax, node_color=[p2l[q] for q in Gp], cmap="tab10",
                               vmin=0, vmax=9, node_size=420)
        nx.draw_networkx_labels(Gp, pp, ax=ax, labels={q: p2l[q] for q in Gp}, font_size=10)
        ax.set_title(label, fontsize=12)
        ax.margins(0.12)
        ax.axis("off")

    # (b) local field: minorminer K6 embedding, local drives
    draw_physical(axes[1], loc, f"local field: {sum(loc.chain_lengths().values())} qubits, "
                                r"drive $d_a/k_a$ per chain qubit")
    axes[1].text(0.0, 1.02, "(b)", transform=axes[1].transAxes, fontweight="bold", fontsize=15)
    # (c) global field: padded chains, bond width ~ r_a
    r = 1 / m.t_target
    draw_physical(axes[2], emb, f"global field: {emb.n_physical} qubits, " r"$\Omega=\sqrt{J_F}$, "
                                r"$J_{F,a}=J_F\,r_a$", chain_w=r / r.max())
    axes[2].text(0.0, 1.02, "(c)", transform=axes[2].transAxes, fontweight="bold", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("->", path)


def plot_maps(path):
    loc = np.load(LOCAL, allow_pickle=True)
    glo = np.load(OUT, allow_pickle=True)
    dmax = float(np.abs(glo["d"]).max())
    panels = [("local field", loc["gammas"], loc["ratios"], loc["inf_emb"]),
              ("global field", glo["gammas"], glo["ratios"], glo["infidelity"])]
    vmin = min(np.nanmin(p[3]) for p in panels)
    norm = LogNorm(vmin=vmin, vmax=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, constrained_layout=True)
    for k, (ax, (title, g, chi, inf)) in enumerate(zip(axes, panels)):
        ge, ce = log_edges(g / dmax), log_edges(chi)
        mesh = ax.pcolormesh(ge, ce, np.clip(inf, vmin, 1).T, norm=norm, cmap="viridis", shading="flat")
        for jf in [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]:  # J_F / |d|_max = const
            ax.plot(ge, jf / ge, ls="--", lw=0.8, color="w", alpha=0.45)
        ax.set(xscale="log", yscale="log", xlim=(ge[0], ge[-1]), ylim=(ce[0], ce[-1]))
        ax.set_title(rf"$^{{18}}$O, {title}", fontsize=18)
        ax.set_title(f"({'ab'[k]})", loc="left", fontweight="bold", fontsize=15)
        ax.set_xlabel(r"$\gamma / |d|_{\max}$", fontsize=18)
        ax.tick_params(labelsize=13, which="both", direction="in", top=True, right=True, color="w")
    axes[0].set_ylabel(r"$\chi = J_F/\gamma$", fontsize=18)
    cb = fig.colorbar(mesh, ax=axes, pad=0.02, aspect=20)
    cb.set_label(r"$1-F(\chi,\gamma)$", fontsize=18)
    cb.ax.tick_params(labelsize=13)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("->", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-gamma", type=int, default=20)
    ap.add_argument("--n-chi", type=int, default=20)
    ap.add_argument("--chi-max", type=float, default=1e5)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    os.makedirs(FIGDIR, exist_ok=True)
    if not args.plot_only:
        scan(args.n_gamma, args.n_chi, args.chi_max)
        if not os.path.exists(LOCAL):  # local field on the same grid
            os.makedirs("data/systematic_analysis/_tmp_local", exist_ok=True)
            subprocess.run([sys.executable, "-W", "ignore", "-c",
                            "import numpy as np, embedding_scan as e;"
                            f"g=np.logspace(0,np.log10(400),{args.n_gamma}); r=np.logspace(0,np.log10({args.chi_max}),{args.n_chi});"
                            "e.scan('O18', g, r, 3, 'data/systematic_analysis/_tmp_local')"], check=True)
            os.replace("data/systematic_analysis/_tmp_local/scan_embedding_O18.npz", LOCAL)
            os.rmdir("data/systematic_analysis/_tmp_local")
    plot_graph(f"{FIGDIR}/global_field_O18_embedding.pdf")
    plot_maps(f"{FIGDIR}/global_vs_local_O18.pdf")


if __name__ == "__main__":
    main()
