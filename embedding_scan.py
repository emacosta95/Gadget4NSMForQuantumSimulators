"""Minor-embedded PCE gadget: 1 - F on the (gamma, chi = J_F / gamma) grid.

Same study as oxigen_analysis.py (data/systematic_analysis/scan_embedding_O*.npz,
figure in SystematicResults.ipynb), for any nucleus of the sd or pf shell.

  * PCE with c_AB = 0 (rank-1 drive), exact self-energy compensation
  * embedding of K_D on a Pegasus(8) graph (minorminer, seed 1): for the pf
    shell 10 logical qubits -> 16 physical qubits (six chains of length 2)
  * ground state in the space within --max-flips bit flips of the code space
    (3 is converged for chains of length <= 2: a code-to-code hop never goes
    more than 2 flips away from the code space; checked against 4 and 5 flips)

Usage
-----
  python embedding_scan.py --nuclei Ca42 Ca44 Ca46 --n-gamma 12 --n-ratio 12
  python embedding_scan.py --nuclei O18 O20 O22             # oxygen, same grid as before

Output: data/systematic_analysis/scan_embedding_<label>.npz with the keys of
oxigen_analysis.py (gammas, ratios, inf_gadget, inf_emb, dE_emb, chains).
"""

from __future__ import annotations

import argparse
import os
import time
import warnings

import numpy as np

from src.gadget import (
    GadgetHamiltonian,
    GadgetParameters,
    MinorEmbeddedEncoding,
    ParticleConservationEncoding,
    QuasiparticleProblem,
    find_minor_embedding,
    fit_drive,
)

DATA = "data/matrix_elements_h_eff_2body"
NUCLEI = {  # label: (shell, n_levels, n_pairs)
    "O18": ("sd", 6, 1), "O20": ("sd", 6, 2), "O22": ("sd", 6, 3),
    "Ca42": ("pf", 10, 1), "Ca44": ("pf", 10, 2), "Ca46": ("pf", 10, 3),
}


def problem(label):
    shell, D, N = NUCLEI[label]
    return QuasiparticleProblem.from_npz(
        f"{DATA}/one_body_nn_{shell}.npz",
        f"{DATA}/twobody_nn_{shell}.npz" if N > 1 else None,
        n_levels=D, n_particles=N, label=label,
    )


def scan(label, gammas, ratios, max_flips, outdir, use_c=False):
    p = problem(label)
    E0, _ = p.exact_groundstate()
    drive = fit_drive(p.g1, n_restarts=30, use_c=use_c, seed=42)
    emb = find_minor_embedding(p.n_levels, pegasus_size=8, tries=6, seed=1)
    assert emb.check(verbose=False)
    chains = emb.chain_lengths()
    print(f"[{label}] E0={E0:.4f}  chains={chains}  physical qubits={sum(chains.values())}", flush=True)

    enc = ParticleConservationEncoding(p)
    inf_gadget = np.empty(len(gammas))
    for i, g in enumerate(gammas):
        H = GadgetHamiltonian(p, enc, drive, GadgetParameters(gamma=g, self_energy="exact", use_c=use_c))
        inf_gadget[i] = 1 - H.check(verbose=False)["fidelity"]

    enc_emb = MinorEmbeddedEncoding(ParticleConservationEncoding(p), emb)
    inf_emb = np.full((len(gammas), len(ratios)), np.nan)
    dE_emb = np.full_like(inf_emb, np.nan)
    path = os.path.join(outdir, f"scan_embedding_{label}.npz")
    if os.path.exists(path):  # resume: keep the rows already computed on the same grid
        old = np.load(path, allow_pickle=True)
        if np.allclose(old["gammas"], gammas) and np.allclose(old["ratios"], ratios):
            inf_emb, dE_emb = old["inf_emb"].copy(), old["dE_emb"].copy()
            print(f"[{label}] resuming: {int((~np.isnan(inf_emb).any(1)).sum())} rows already done", flush=True)
    for i, g in enumerate(gammas):
        if not np.isnan(inf_emb[i]).any():
            continue
        t0 = time.time()
        for j, ratio in enumerate(ratios):
            H = GadgetHamiltonian(
                p, enc_emb, drive,
                GadgetParameters(gamma=g, J_F=ratio * g, self_energy="exact", use_c=use_c),
            )
            H.max_flips = max_flips
            r = H.check(verbose=False)
            inf_emb[i, j] = 1 - r["fidelity"]
            dE_emb[i, j] = r["relative_energy_error_corrected"]
        np.savez(path, gammas=gammas, ratios=ratios, inf_gadget=inf_gadget, inf_emb=inf_emb,
                 dE_emb=dE_emb, chains=chains, max_flips=max_flips, use_c=use_c)
        print(f"[{label}] gamma={g:7.2f}  best 1-F={np.nanmin(inf_emb[i]):.2e}  "
              f"(unembedded {inf_gadget[i]:.2e})  [{time.time()-t0:.0f}s]", flush=True)
    print("->", path, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nuclei", nargs="+", default=["Ca42", "Ca44", "Ca46"], choices=list(NUCLEI))
    ap.add_argument("--n-gamma", type=int, default=12)
    ap.add_argument("--n-ratio", type=int, default=12)
    ap.add_argument("--gamma-max", type=float, default=400.0)
    ap.add_argument("--ratio-max", type=float, default=1000.0)
    ap.add_argument("--max-flips", type=int, default=3)
    ap.add_argument("--outdir", default="data/systematic_analysis")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    os.makedirs(args.outdir, exist_ok=True)
    gammas = np.logspace(0, np.log10(args.gamma_max), args.n_gamma)
    ratios = np.logspace(0, np.log10(args.ratio_max), args.n_ratio)
    for label in args.nuclei:
        scan(label, gammas, ratios, args.max_flips, args.outdir)


if __name__ == "__main__":
    main()
