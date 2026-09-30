import numpy as np

from src.gadget import *

ONEBODY = "data/matrix_elements_h_eff_2body/one_body_nn_pf.npz"
TWOBODY = "data/matrix_elements_h_eff_2body/twobody_nn_pf.npz"


n_particles = [1, 2, 3, 4, 5]

gammas = np.logspace(0, np.log10(400), 10)  # 1 … 400
ratios = np.logspace(1, 4, 10)  # J_F/γ = 1 … 1000

results = {}  # results[n] = dict of arrays

for n_particle in n_particles:
    label = f"Ca{38 + 2*n_particle}"
    if n_particle == 1:
        p = QuasiparticleProblem.from_npz(
            ONEBODY, n_levels=10, n_particles=n_particle, label=label
        )
    else:
        p = QuasiparticleProblem.from_npz(
            ONEBODY, TWOBODY, n_levels=10, n_particles=n_particle, label=label
        )
    E0, psi_log = p.exact_groundstate()
    print(f"[1] {label}: E0 = {E0:.6f}")

    drive = fit_drive(p.g1, n_restarts=30, cap=4.0, seed=42)  # once
    emb = find_minor_embedding(p.n_levels, pegasus_size=8, tries=6, seed=1)  # once
    assert emb.check(verbose=False)
    print(f"    chains = {emb.chain_lengths()}")

    # --- (a) unembedded gadget: 1-F vs gamma
    enc = ParticleConservationEncoding(p)
    inf_gadget = np.empty(len(gammas))
    for i, gamma in enumerate(gammas):
        H = GadgetHamiltonian(
            p, enc, drive, GadgetParameters(gamma=gamma, self_energy="exact")
        )
        inf_gadget[i] = 1 - H.check(verbose=False)["fidelity"]

    # --- (b) embedded: 1-F on the (gamma, J_F/gamma) grid
    enc_emb = MinorEmbeddedEncoding(ParticleConservationEncoding(p), emb)
    inf_emb = np.full((len(gammas), len(ratios)), np.nan)
    dE_emb = np.full_like(inf_emb, np.nan)
    for i, gamma in enumerate(gammas):
        for j, ratio in enumerate(ratios):
            Jf = ratio * gamma
            H = GadgetHamiltonian(
                p,
                enc_emb,
                drive,
                GadgetParameters(gamma=gamma, J_F=Jf, self_energy="exact"),
            )
            r = H.check(verbose=False)
            inf_emb[i, j] = 1 - r["fidelity"]
            dE_emb[i, j] = r["relative_energy_error_corrected"]
        print(f"[6] {label} gamma={gamma:6.1f}  best 1-F={np.nanmin(inf_emb[i]):.2e}")

    results[n_particle] = dict(
        gammas=gammas,
        ratios=ratios,
        inf_gadget=inf_gadget,
        inf_emb=inf_emb,
        dE_emb=dE_emb,
        chains=emb.chain_lengths(),
    )
    np.savez(f"scan_embedding_{label}.npz", **results[n_particle])
