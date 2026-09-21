#!/usr/bin/env python3
"""Smoke tests for src.gadget. Run once in your environment:

    python test_gadget.py

Checks, in order:
  1. exact logical Hamiltonian reproduces the NSM number-sector spectrum
  2. every encoding projects/embeds consistently (round trip, multiplicities)
  3. O18 one-hot gadget converges with gamma
  4. O20 first quantization: self_energy="exact" removes the gamma2 dependence
  5. driver is diagonal and its ground state is a valid logical configuration
  6. minor embedding: chains, coupler dedup, fidelity
"""

import numpy as np

from src.gadget import *

ONEBODY = "data/matrix_elements_h_eff_2body/one_body_nn_sd.npz"
TWOBODY = "data/matrix_elements_h_eff_2body/twobody_nn_sd.npz"


def problem_O18():
    return QuasiparticleProblem.from_npz(
        ONEBODY, n_levels=6, n_particles=1, label="O18"
    )


def problem_O20():
    return QuasiparticleProblem.from_npz(
        ONEBODY, TWOBODY, n_levels=6, n_particles=2, label="O20"
    )


def test_exact_spectrum():
    p = problem_O18()
    E0, psi = p.exact_groundstate()
    # one particle: the logical Hamiltonian is just eps + g1
    H = np.diag(p.eps) + p.g1
    assert np.allclose(p.logical_hamiltonian(), H)
    assert np.isclose(E0, np.linalg.eigvalsh(H)[0])
    print(f"[1] exact spectrum ok, E0(O18) = {E0:.6f}")


def test_two_body_scale():
    """g2 must not be halved: twobody_nn_sd.npz only lists i<j."""
    p = problem_O20()
    import numpy as np

    raw = np.load(TWOBODY)
    for key, value in zip(raw["keys"], raw["values"]):
        i, j = int(key[0]), int(key[1])
        assert np.isclose(p.g2[i, j], value), (i, j, p.g2[i, j], value)
    print("[1b] g2 matches the raw two-body matrix elements")


def test_encodings():
    p = problem_O20()
    for enc in (ParticleConservationEncoding(p), FirstQuantizationEncoding(p)):
        _, psi_log = p.exact_groundstate()
        psi_phys = enc.embed(psi_log)
        back = enc.project(psi_phys)
        assert np.allclose(back, psi_log, atol=1e-12), type(enc).__name__
        assert abs(enc.leakage(psi_phys)) < 1e-12
        probs = enc.probabilities(psi_phys)
        assert np.isclose(sum(probs.values()), 1.0)
        print(f"[2] {type(enc).__name__}: n_qubits={enc.n_qubits} round trip ok")


def test_O18_convergence():
    p = problem_O18()
    drive = fit_drive(p.g1, n_restarts=30, cap=4.0, seed=42)
    enc = ParticleConservationEncoding(p)
    previous = 1.0
    for gamma in (100.0, 300.0, 1000.0):
        H = GadgetHamiltonian(
            p, enc, drive, GadgetParameters(gamma=gamma, self_energy="exact")
        )
        r = H.check(verbose=False)
        infidelity = 1 - r["fidelity"]
        print(f"[3] O18 gamma={gamma:7.0f}  1-F={infidelity:.3e}")
        assert infidelity < previous
        previous = infidelity


def test_O20_selfenergy():
    p = problem_O20()
    drive = fit_drive(p.g1, n_restarts=30, cap=4.0, seed=42)
    enc = FirstQuantizationEncoding(p)
    out = {}
    for mode in ("uniform", "exact"):
        for ratio in (0.1, 1.0):
            H = GadgetHamiltonian(
                p,
                enc,
                drive,
                GadgetParameters(gamma=300.0, gamma2=300.0 * ratio, self_energy=mode),
            )
            out[(mode, ratio)] = 1 - H.check(verbose=False)["fidelity"]
            print(f"[4] O20 {mode:8s} gamma2={ratio}gamma  1-F={out[(mode,ratio)]:.3e}")
    # the whole point: with the full compensation gamma2 = gamma is as good
    assert out[("exact", 1.0)] < out[("uniform", 1.0)] / 5
    assert abs(out[("exact", 1.0)] - out[("exact", 0.1)]) < 0.3 * out[("exact", 0.1)]


def test_driver():
    p = problem_O20()
    drive = fit_drive(p.g1, n_restarts=10, cap=4.0, seed=42)
    enc = FirstQuantizationEncoding(p)
    H = GadgetHamiltonian(p, enc, drive, GadgetParameters(gamma=50.0))
    drv = H.driver()
    M = drv.total().full()
    assert np.allclose(M, np.diag(np.diag(M))), "driver must be diagonal"
    ada = QuantumAdiabatic4Gadget(H, drv)
    idx = ada.initial_index()
    assert (
        enc.leakage(ada.initial_state()) < 1e-12
    ), "initial state leaves the code space"
    print(f"[5] driver diagonal, initial physical index {idx}")


def test_embedding():
    p = problem_O18()
    drive = fit_drive(p.g1, n_restarts=30, cap=4.0, seed=42)
    emb = find_minor_embedding(p.n_levels, pegasus_size=8, tries=6, seed=1)
    assert emb.check(verbose=False)
    enc = MinorEmbeddedEncoding(ParticleConservationEncoding(p), emb)
    gamma = 200.0
    H = GadgetHamiltonian(
        p,
        enc,
        drive,
        GadgetParameters(gamma=gamma, J_F=suggested_J_F(gamma), self_energy="exact"),
    )
    r = H.check(verbose=False)
    print(
        f"[6] embedded n_phys={enc.n_qubits} chains={emb.chain_lengths()} "
        f"1-F={1-r['fidelity']:.3e} dE_corrected={r['relative_energy_error_corrected']:.3e}"
    )
    assert 1 - r["fidelity"] < 1e-2


if __name__ == "__main__":
    test_exact_spectrum()
    test_two_body_scale()
    test_encodings()
    test_O18_convergence()
    test_O20_selfenergy()
    test_driver()
    test_embedding()
    print("\nall checks passed")
