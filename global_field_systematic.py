"""Global transverse field: 1 - F on the (gamma, chi = J_F / gamma) grid for the
oxygens and calciums, i.e. the global-field version of the local-field maps of
embedding_scan.py (figure scan_embedding_O_Ca.pdf in SystematicResults.ipynb).

Scheme (as in global_field_scan.py, here for any nucleus):
  * PCE with c_AB = 0, exact self-energy, minorminer embedding of K_D on
    Pegasus(8) (seed 1, the one of the local-field scans) with every length-1
    chain padded by a free neighbour: all chains have k = 2
    (sd: 12 qubits, pf: 20 qubits);
  * ONE global field  Omega = sqrt(J_F)  on every qubit;
  * chain bond  J_F,a = J_F r_a,  r_a = (1 - t_a^2 / J_F) / t_a,  t_a = |d_a|/sqrt 2,
    so the chain doublet is split by exactly 2 t_a (exact k = 2 inversion);
  * drive signs through the gauge sigma_a = sign(d_a) (-1)^(k_a - 1): a code
    state |n> picks up prod_a sigma_a^{n_a};
  * ground state within --max-flips (default 5) bit flips of the code space (library
    truncated solver, same as the local-field scans).

Usage
-----
  python global_field_systematic.py                      # O18 O20 O22, 20 x 20 grid

Output: data/systematic_analysis/global_field_scan_<label>.npz
        (gammas, ratios, inf_emb, dE_emb, chains, r_inf, d, max_flips)
"""

from __future__ import annotations

import argparse
import os
import time
import warnings

import dwave_networkx as dnx
import networkx as nx
import numpy as np

from embedding_scan import NUCLEI, problem
from src.gadget import (
    GadgetHamiltonian,
    GadgetParameters,
    MinorEmbeddedEncoding,
    ParticleConservationEncoding,
    find_minor_embedding,
    fit_drive,
)
from src.gadget.embedding import from_minorminer


def padded_embedding(D, seed=1, tries=6):
    """Local-field embedding of K_D with every length-1 chain padded to k = 2."""
    hardware = dnx.pegasus_graph(8)
    raw = find_minor_embedding(D, hardware=hardware, tries=tries, seed=seed)
    chains = {a: [raw.hardware_of[q] for q in ch] for a, ch in raw.chains.items()}
    used = {q for ch in chains.values() for q in ch}
    for a, ch in chains.items():
        if len(ch) == 1:
            free = next(v for v in hardware.neighbors(ch[0]) if v not in used)
            ch.append(free)
            used.add(free)
    emb = from_minorminer(chains, hardware, nx.complete_graph(D))
    assert emb.check(verbose=False)
    assert set(emb.chain_lengths().values()) == {2}
    return emb


def chain_doublet_centre(J, Omega):
    """Centre of the two lowest levels of an isolated k = 2 chain,
    H = -J Z Z + J + Omega (X1 + X2) (subtracted from the energy)."""
    H = np.array([[0, Omega, Omega, 0], [Omega, 2 * J, 0, Omega],
                  [Omega, 0, 2 * J, Omega], [0, Omega, Omega, 0]], float)
    w = np.linalg.eigvalsh(H)[:2]
    return 0.5 * (w[0] + w[1])


class GlobalFieldHamiltonian(GadgetHamiltonian):
    """Embedded PCE gadget with per-chain bonds J_chain[a] and one field Omega."""

    def __init__(self, prob, enc, drive, params, J_chain, Omega):
        self.J_chain, self.Omega = J_chain, Omega
        super().__init__(prob, enc, drive, params)

    def _embedded_terms(self, enc):
        zz, z, _, const = super()._embedded_terms(enc)
        emb = enc.embedding
        const -= self.params.J_F * len(emb.intra_edges)  # drop the uniform-J_F chains
        p2l = emb.phys_to_logical()
        zz_pauli = {}
        for u, v in emb.intra_edges:
            J = self.J_chain[p2l[u]]
            zz_pauli[(min(u, v), max(u, v))] = -J
            const += J
        return zz, z, zz_pauli, const

    def transverse_amplitudes(self):
        return np.full(self.encoding.n_qubits, self.Omega)


class GlobalFieldModel:
    def __init__(self, label, max_flips=5):
        self.label = label
        self.p = problem(label)
        self.E_exact, self.psi_exact = self.p.exact_groundstate()
        self.drive = fit_drive(self.p.g1, n_restarts=30, use_c=False, seed=42)
        self.d = self.drive.d
        self.emb = padded_embedding(self.p.n_levels)
        self.k = np.array([len(self.emb.chains[a]) for a in range(self.p.n_levels)])
        self.t = np.abs(self.d) / np.sqrt(2)
        sigma = np.sign(self.d) * (-1.0) ** (self.k - 1)
        # gauge sign of every logical configuration: prod over occupied levels
        self.enc = MinorEmbeddedEncoding(ParticleConservationEncoding(self.p), self.emb)
        self.gauge = np.array([np.prod(sigma[list(cfg)]) for cfg in self.enc.logical_basis])
        self.max_flips = max_flips

    def point(self, gamma, J_F):
        Omega = np.sqrt(J_F)
        r = (1.0 - self.t**2 / J_F) / self.t
        J_chain = J_F * r
        H = GlobalFieldHamiltonian(
            self.p, self.enc, self.drive,
            GadgetParameters(gamma=gamma, J_F=J_F, self_energy="exact", use_c=False),
            J_chain, Omega,
        )
        E0, psi = H._groundstate_truncated(self.max_flips)
        amps = self.gauge * self.enc.project(psi).real
        offset = sum(chain_doublet_centre(J_chain[a], Omega) for a in range(self.p.n_levels))
        energy = (E0 - offset) * gamma
        return dict(infidelity=1 - float(amps @ self.psi_exact) ** 2,
                    rel_energy_error=abs(energy - self.E_exact) / abs(self.E_exact),
                    dim=H.truncated_dim)


def scan(label, gammas, ratios, max_flips, outdir):
    m = GlobalFieldModel(label, max_flips)
    print(f"[{label}] global field: {m.emb.n_physical} physical qubits, "
          f"r_a(J_F->inf) = {np.round(1 / m.t, 3)}", flush=True)
    inf = np.full((len(gammas), len(ratios)), np.nan)
    dE = np.full_like(inf, np.nan)
    path = os.path.join(outdir, f"global_field_scan_{label}.npz")
    if os.path.exists(path):  # resume on the same grid
        old = np.load(path, allow_pickle=True)
        if np.allclose(old["gammas"], gammas) and np.allclose(old["ratios"], ratios):
            inf, dE = old["inf_emb"].copy(), old["dE_emb"].copy()
    for i, g in enumerate(gammas):
        if not np.isnan(inf[i]).any():
            continue
        t0 = time.time()
        for j, chi in enumerate(ratios):
            r = m.point(g, chi * g)
            inf[i, j], dE[i, j] = r["infidelity"], r["rel_energy_error"]
        np.savez(path, gammas=gammas, ratios=ratios, inf_emb=inf, dE_emb=dE,
                 chains=m.emb.chain_lengths(), r_inf=1 / m.t, d=m.d, max_flips=max_flips)
        print(f"[{label}] gamma={g:7.2f}  best 1-F={np.nanmin(inf[i]):.2e}  "
              f"[dim {r['dim']}, {time.time()-t0:.0f}s]", flush=True)
    print("->", path, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nuclei", nargs="+", default=["O18", "O20", "O22"], choices=list(NUCLEI))
    ap.add_argument("--n-gamma", type=int, default=None, help="default: 20 (sd), 12 (pf)")
    ap.add_argument("--n-ratio", type=int, default=None, help="default: 20 (sd), 12 (pf)")
    ap.add_argument("--gamma-max", type=float, default=400.0)
    ap.add_argument("--ratio-max", type=float, default=1000.0)
    ap.add_argument("--max-flips", type=int, default=5)
    ap.add_argument("--outdir", default="data/systematic_analysis")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    os.makedirs(args.outdir, exist_ok=True)
    for label in args.nuclei:
        n_def = 20 if NUCLEI[label][0] == "sd" else 12  # grids of the local-field maps
        gammas = np.logspace(0, np.log10(args.gamma_max), args.n_gamma or n_def)
        ratios = np.logspace(0, np.log10(args.ratio_max), args.n_ratio or n_def)
        scan(label, gammas, ratios, args.max_flips, args.outdir)


if __name__ == "__main__":
    main()
