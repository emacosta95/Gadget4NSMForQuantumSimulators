import numpy as np

from src.gadget import *

ONEBODY = "data/matrix_elements_h_eff_2body/one_body_nn_sd.npz"
TWOBODY = "data/matrix_elements_h_eff_2body/twobody_nn_sd.npz"


n_particles = [1, 2, 3]

gammas = np.linspace(1, 400, 20)
Jfs = np.linspace(10, 400**2 / 0.01, 20)
infidelities_per_particle = {n: [] for n in n_particles}

for n_particle in n_particles:
    if n_particle == 1:
        p = QuasiparticleProblem.from_npz(
            ONEBODY, n_levels=6, n_particles=n_particle, label=f"O{16+n_particle*2}"
        )
    else:
        p = QuasiparticleProblem.from_npz(
            ONEBODY,
            TWOBODY,
            n_levels=6,
            n_particles=n_particle,
            label=f"O{16+n_particle*2}",
        )
    E0, psi = p.exact_groundstate()
    print(f"[1] exact spectrum ok, E0(O{16+n_particle*2}) = {E0:.6f}")

    for enc in (ParticleConservationEncoding(p), FirstQuantizationEncoding(p)):
        _, psi_log = p.exact_groundstate()
        psi_phys = enc.embed(psi_log)
        back = enc.project(psi_phys)
        assert np.allclose(back, psi_log, atol=1e-12), type(enc).__name__
        assert abs(enc.leakage(psi_phys)) < 1e-12
        probs = enc.probabilities(psi_phys)
        assert np.isclose(sum(probs.values()), 1.0)
        print(f"[2] {type(enc).__name__}: n_qubits={enc.n_qubits} round trip ok")

        drive = fit_drive(p.g1, n_restarts=30, cap=4.0, seed=42)
    enc = ParticleConservationEncoding(p)
    previous = 1.0
    for gamma in gammas:
        H = GadgetHamiltonian(
            p, enc, drive, GadgetParameters(gamma=gamma, self_energy="exact")
        )
        r = H.check(verbose=False)
        infidelity = 1 - r["fidelity"]
        print(f"[3] O18 gamma={gamma:7.0f}  1-F={infidelity:.3e}")
        assert infidelity < previous
        previous = infidelity
        infidelities_per_particle[n_particle].append(infidelity)

    for Jf in Jfs:
        for gamma in gammas:
            emb = find_minor_embedding(p.n_levels, pegasus_size=8, tries=6, seed=1)
            assert emb.check(verbose=False)
            enc = MinorEmbeddedEncoding(ParticleConservationEncoding(p), emb)
            gamma = 200.0
            H = GadgetHamiltonian(
                p,
                enc,
                drive,
                GadgetParameters(
                    gamma=gamma, J_F=suggested_J_F(gamma), self_energy="exact"
                ),
            )
            r = H.check(verbose=False)
            print(
                f"[6] embedded n_phys={enc.n_qubits} chains={emb.chain_lengths()} "
                f"1-F={1-r['fidelity']:.3e} dE_corrected={r['relative_energy_error_corrected']:.3e}"
            )
            assert 1 - r["fidelity"] < 1e-2
