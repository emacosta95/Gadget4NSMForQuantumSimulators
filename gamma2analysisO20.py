#!/usr/bin/env python3
"""Systematic analysis of the Gadget fidelity vs gamma at different gamma2 for 20-O.

We work in the first quantization encoding: the two quasiparticles of the 20-O are
encoded in two registers of ``n_orbitals`` qubits each, so that a logical pair
(A, B) of the NSM Hamiltonian is represented by the two configurations (A_a, B_b)
and (B_a, A_b).

The Gadget longitudinal Hamiltonian carries two penalties:

    gamma  -> particle number constrain of each register (one particle per register)
    gamma2 -> hardcore boson constrain between the registers, which forbids the two
              particles to occupy the same single particle level

This script scans gamma while keeping gamma2 = r * gamma for several ratios r, and
for every (gamma, gamma2) it diagonalizes the total Gadget Hamiltonian, maps its
ground state back onto the two-body logical bitstrings and overlaps it with the
exact NSM ground state.

The result is written as a pickle holding, for every ratio, the gamma grid and the
corresponding fidelities.

Usage
-----
    python gamma2analysisO20.py
    python gamma2analysisO20.py --gamma-min 5 --gamma-max 300 --n-gammas 15 \
        --ratios 0.1 0.5 1.0 2.0 --output data/results/gamma2analysisO20.pkl
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import qutip as qt
import scipy.linalg

from ManyBodyQutip.qutip_class import SpinOperator
from NSMFermions.utils_quasiparticle_approximation import HardcoreBosonsBasis
from src.interaction_utils import EffectiveInteractionOptimizer2ndVersion
from src.utils import computational_basis, generate_particleconservation_basis

ONEBODY_PATH = "data/matrix_elements_h_eff_2body/one_body_nn_sd.npz"
TWOBODY_PATH = "data/matrix_elements_h_eff_2body/twobody_nn_sd.npz"


def load_couplings(n_orbitals, onebody_path=ONEBODY_PATH, twobody_path=TWOBODY_PATH):
    """Load the quasiparticle couplings g^(1)_AB and g^(2)_AB."""
    data_onebody = np.load(onebody_path)
    g_onebody = {}
    diagonal_elements = np.zeros(n_orbitals)
    g_matrix = np.zeros((n_orbitals, n_orbitals))
    for a, key in enumerate(data_onebody["keys"]):
        i, j = key
        value = data_onebody["values"][a]
        g_onebody[(i, j)] = value
        if i != j:
            g_matrix[i, j] = value
        else:
            diagonal_elements[i] = value

    data_twobody = np.load(twobody_path)
    g_twobody = {}
    for a, key in enumerate(data_twobody["keys"]):
        i, j, _, _ = key
        g_twobody[(i, j)] = data_twobody["values"][a]

    return g_onebody, g_twobody, diagonal_elements, g_matrix


def exact_nsm_groundstate(g_onebody, g_twobody, n_orbitals, nparticles=2):
    """Exact NSM ground state in the hardcore boson number sector.

    Returns the spectrum in the number sector and the ground state decomposed on
    the two-body logical bitstrings, ``{(A, B): amplitude}``.
    """
    particle_conserved_basis = generate_particleconservation_basis(
        size_a=n_orbitals, size_b=0, nparticles_a=nparticles, nparticles_b=0
    )
    hardcore_basis = HardcoreBosonsBasis(basis=particle_conserved_basis)

    hamiltonian = 0.0
    for (idx_a, idx_b), value in g_onebody.items():
        hamiltonian += value * hardcore_basis.adag_a_matrix(idx_a, idx_b)
    for (idx_a, idx_b), value in g_twobody.items():
        hamiltonian += value * hardcore_basis.adag_adag_a_a_matrix(
            idx_a, idx_b, idx_a, idx_b
        )

    energies, eigenstates = np.linalg.eigh(hamiltonian.todense())
    groundstate = np.asarray(eigenstates)[:, 0]

    psi_twobody_exact = {}
    for r, amplitude in enumerate(groundstate):
        if np.abs(amplitude) ** 2 <= 10**-10:
            continue
        idxs = np.nonzero(hardcore_basis.basis[r])[0]
        if len(idxs) != 2:
            continue
        key = tuple(int(x) for x in sorted(idxs))
        psi_twobody_exact[key] = psi_twobody_exact.get(key, 0.0) + float(amplitude)

    return energies, psi_twobody_exact


def fit_effective_interaction(g_matrix, n_orbitals, n_restarts=100, seed=None):
    """Fit the rank-1 drive d_A and the parametrized self energy c_AB."""
    if seed is not None:
        np.random.seed(seed)

    optimizer = EffectiveInteractionOptimizer2ndVersion(n_orbitals)
    d_opt, _ = optimizer.optimize_rank1(g_matrix)
    _, c_matrix, _, _ = optimizer.get_alpha_and_c(g_matrix, d_opt)

    # regularization of the self energy, as done in the notebooks
    c_matrix[np.abs(c_matrix) > 4] = np.abs(c_matrix[np.abs(c_matrix) > 4])

    return d_opt, c_matrix


def build_transverse_hamiltonian(d_opt, n_qubits):
    """Transverse field, identical on the two registers. It does not depend on gamma."""
    transverse_hamiltonian = 0.0
    for i in range(n_qubits):
        transverse_hamiltonian += SpinOperator(
            [("x", i)],
            coupling=[d_opt[i % (n_qubits // 2)] / np.sqrt(2)],
            size=n_qubits,
            verbose=1,
        ).qutip_op
    return transverse_hamiltonian


def build_longitudinal_hamiltonian(
    gamma,
    gamma2,
    n_qubits,
    diagonal_elements,
    c_matrix,
    g_twobody,
    identity_qubit_space,
    ntot=1,
    field=None,
):
    """Longitudinal (Ising) part of the Gadget in the first quantization encoding."""
    if field is None:
        field = diagonal_elements
    effective_longitudinal_field = field / gamma
    half = n_qubits // 2

    # particle constrain of each register
    hamiltonian_zz = 0.0
    for i in range(half):
        for j in range(i + 1, half):
            hamiltonian_zz += SpinOperator(
                [("qz", i, "qz", j)],
                coupling=[(2 + c_matrix[i, j]) * gamma],
                size=n_qubits,
                verbose=1,
            ).qutip_op
    for i in range(half, n_qubits):
        for j in range(i + 1, n_qubits):
            hamiltonian_zz += SpinOperator(
                [("qz", i, "qz", j)],
                coupling=[(2 + c_matrix[i - half, j - half]) * gamma],
                size=n_qubits,
                verbose=1,
            ).qutip_op

    # linear part of the particle number constrain + effective longitudinal field
    hamiltonian_z = 0.0
    for i in range(n_qubits):
        hamiltonian_z += SpinOperator(
            [("qz", i)],
            coupling=[
                effective_longitudinal_field[i % half] + gamma * (1 - 2 * ntot)
            ],
            size=n_qubits,
            verbose=1,
        ).qutip_op

    # hardcore boson constrain -> avoid the two particles on the same site
    for i in range(half):
        hamiltonian_zz += SpinOperator(
            [("qz", i, "qz", i + half)],
            coupling=[gamma2],
            size=n_qubits,
            verbose=1,
        ).qutip_op

    # two-body interaction, with the self energy contribution removed
    for i in range(half):
        for j in range(i + 1, half):
            hamiltonian_zz += SpinOperator(
                [("qz", i, "qz", j + half)],
                coupling=[g_twobody[(i, j)] / gamma],
                size=n_qubits,
                verbose=1,
            ).qutip_op
            hamiltonian_zz += SpinOperator(
                [("qz", i + half, "qz", j)],
                coupling=[g_twobody[(i, j)] / gamma],
                size=n_qubits,
                verbose=1,
            ).qutip_op

    return hamiltonian_zz + hamiltonian_z + 2 * gamma * (ntot**2) * identity_qubit_space


def groundstate(total_hamiltonian, solver="scipy"):
    """Lowest eigenpair of the total Gadget Hamiltonian.

    ``scipy`` uses a direct LAPACK call restricted to the lowest eigenpair, which
    is several times faster than the qutip/ARPACK path and, being direct, has a
    runtime independent of how spread the spectrum is. ``qutip`` keeps the
    original ``eigenstates(eigvals=1)`` route and is kept for cross checks.
    """
    if solver == "qutip":
        eigenvalues, eigenstates = total_hamiltonian.eigenstates(eigvals=1)
        return eigenvalues[0], eigenstates[0]

    matrix = total_hamiltonian.full()
    # the Gadget Hamiltonian is real symmetric (Z couplings plus a transverse X field)
    max_imaginary = np.max(np.abs(matrix.imag))
    if max_imaginary > 10**-12:
        raise ValueError(f"unexpected imaginary part in the Hamiltonian: {max_imaginary}")

    eigenvalues, eigenvectors = scipy.linalg.eigh(
        matrix.real, subset_by_index=[0, 0]
    )
    return eigenvalues[0], eigenvectors[:, 0]


def gadget_fidelity(state, basis, n_qubits, psi_twobody_exact, threshold=10**-2):
    """Fidelity between a Gadget state and the exact NSM ground state.

    The Gadget state is mapped back onto the two-body logical bitstrings: a logical
    pair (A, B) is encoded twice in this first quantization representation, so we
    sum the two amplitudes and normalize with the number of degenerate
    configurations. Only the configurations with one particle per register are
    retained, the others violate the constrain and do not represent a logical state.
    """
    amplitudes = (
        state.full().flatten() if hasattr(state, "full") else np.asarray(state).flatten()
    )
    half = n_qubits // 2

    psi_twobody_gadget = {}
    counts = {}
    for r, amplitude in enumerate(amplitudes):
        if np.abs(amplitude) <= threshold:
            continue
        idxs = np.nonzero(basis[r])[0]
        if len(idxs) != 2:
            continue
        a, b = idxs
        if not (a < half and b >= half):
            continue
        key = tuple(sorted((int(a) % half, int(b) % half)))
        psi_twobody_gadget[key] = psi_twobody_gadget.get(key, 0.0) + amplitude
        counts[key] = counts.get(key, 0) + 1
    psi_twobody_gadget = {
        key: value / np.sqrt(counts[key]) for key, value in psi_twobody_gadget.items()
    }

    overlap = 0.0
    for key, amplitude in psi_twobody_exact.items():
        overlap += np.conj(amplitude) * psi_twobody_gadget.get(key, 0.0)
    return float(np.abs(overlap) ** 2)


def run_sweep(gammas, ratios, n_qubits, basis, psi_twobody_exact, hamiltonian_kwargs,
              transverse_hamiltonian, solver="scipy", verbose=True):
    """Scan gamma for every gamma2 = ratio * gamma."""
    fidelities = {}
    energies = {}

    for ratio in ratios:
        fidelities_ratio = np.zeros(len(gammas))
        energies_ratio = np.zeros(len(gammas))
        for k, gamma in enumerate(gammas):
            start = time.time()
            total_hamiltonian = (
                build_longitudinal_hamiltonian(
                    gamma, ratio * gamma, n_qubits, **hamiltonian_kwargs
                )
                + transverse_hamiltonian
            )
            energy, state = groundstate(total_hamiltonian, solver=solver)
            fidelities_ratio[k] = gadget_fidelity(
                state, basis, n_qubits, psi_twobody_exact
            )
            energies_ratio[k] = energy
            if verbose:
                print(
                    f"  gamma2={ratio}*gamma  gamma={gamma:8.3f}  "
                    f"F={fidelities_ratio[k]:.6f}  ({time.time() - start:.1f}s)",
                    flush=True,
                )
        fidelities[ratio] = fidelities_ratio
        energies[ratio] = energies_ratio
        if verbose:
            print(f"ratio {ratio} done\n", flush=True)

    return fidelities, energies


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fidelity of the 20-O Gadget vs gamma, for several gamma2 = r*gamma."
    )
    parser.add_argument("--gamma-min", type=float, default=5.0)
    parser.add_argument("--gamma-max", type=float, default=300.0)
    parser.add_argument("--n-gammas", type=int, default=15)
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 1.0, 2.0],
        help="values of r in gamma2 = r * gamma",
    )
    parser.add_argument("--n-orbitals", type=int, default=6)
    parser.add_argument("--nparticles", type=int, default=2)
    parser.add_argument("--n-restarts", type=int, default=100)
    parser.add_argument(
        "--seed", type=int, default=42, help="seed of the interaction fit restarts"
    )
    parser.add_argument(
        "--solver",
        choices=["scipy", "qutip"],
        default="scipy",
        help="ground state solver: 'scipy' is a direct LAPACK call on the lowest "
        "eigenpair (fast), 'qutip' is the original eigenstates(eigvals=1) route",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/results/gamma2analysisO20.pkl")
    )
    return parser.parse_args()


def main():
    args = parse_args()

    n_orbitals = args.n_orbitals
    n_qubits = 2 * n_orbitals

    print("loading the quasiparticle couplings ...", flush=True)
    g_onebody, g_twobody, diagonal_elements, g_matrix = load_couplings(n_orbitals)

    print("diagonalizing the exact NSM Hamiltonian ...", flush=True)
    nsm_energies, psi_twobody_exact = exact_nsm_groundstate(
        g_onebody, g_twobody, n_orbitals, nparticles=args.nparticles
    )
    print(f"  exact NSM ground state energy: {nsm_energies[0]:.6f}", flush=True)

    print("fitting the effective interaction ...", flush=True)
    d_opt, c_matrix = fit_effective_interaction(
        g_matrix, n_orbitals, n_restarts=args.n_restarts, seed=args.seed
    )
    print(f"  d_opt: {d_opt}", flush=True)

    basis = computational_basis(n_qubits)
    identity_qubit_space = qt.tensor([qt.qeye(2)] * n_qubits)
    transverse_hamiltonian = build_transverse_hamiltonian(d_opt, n_qubits)

    hamiltonian_kwargs = dict(
        diagonal_elements=diagonal_elements,
        c_matrix=c_matrix,
        g_twobody=g_twobody,
        identity_qubit_space=identity_qubit_space,
    )

    gammas = np.linspace(args.gamma_min, args.gamma_max, args.n_gammas)
    print(
        f"scanning {len(gammas)} gammas x {len(args.ratios)} ratios "
        f"({len(gammas) * len(args.ratios)} diagonalizations) ...",
        flush=True,
    )
    fidelities, energies = run_sweep(
        gammas,
        args.ratios,
        n_qubits,
        basis,
        psi_twobody_exact,
        hamiltonian_kwargs,
        transverse_hamiltonian,
        solver=args.solver,
    )

    results = {
        "gammas": gammas,
        "gamma2_ratios": list(args.ratios),
        # gamma2 = ratio * gamma, fidelity of the Gadget gs vs the exact NSM gs
        "fidelities": fidelities,
        "gamma2_values": {ratio: ratio * gammas for ratio in args.ratios},
        "gadget_energies": energies,
        "nsm_groundstate_energy": float(nsm_energies[0]),
        "nsm_spectrum": nsm_energies,
        "d_opt": d_opt,
        "c_matrix": c_matrix,
        "psi_twobody_exact": psi_twobody_exact,
        "metadata": {
            "n_orbitals": n_orbitals,
            "n_qubits": n_qubits,
            "nparticles": args.nparticles,
            "ntot": 1,
            "n_restarts": args.n_restarts,
            "seed": args.seed,
            "solver": args.solver,
            "encoding": "first quantization, two registers",
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as file:
        pickle.dump(results, file)
    print(f"saved -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
