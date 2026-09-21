"""Logical (target) quasiparticle problem.

A `QuasiparticleProblem` is the *physics input* of the whole pipeline, and is
completely independent of how it will be encoded on hardware:

    H_log = sum_A  eps_A N_A
          + sum_{A<B} g1_AB (a^dag_A a_B + h.c.)          <-> (X_A X_B + Y_A Y_B)/2
          + sum_{A<B} g2_AB N_A N_B

with hardcore bosons (quasiparticle pairs) in the sector of `n_particles`.

This is the NSM quasiparticle Hamiltonian (O18: n_particles=1, O20/Ca42:
n_particles=2) and, with the same conventions, the molecular seniority-zero
(DOCI) Hamiltonian:  eps_a = 2h_aa + (aa|aa), g1_ab = (ab|ab),
g2_ab = 2[2(aa|bb) - (ab|ab)].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

Config = Tuple[int, ...]  # sorted tuple of occupied single-particle levels


def _symmetrize(matrix) -> np.ndarray:
    """Symmetric, zero-diagonal coupling matrix.

    The data files are not uniform: `one_body_nn_sd.npz` lists BOTH (i,j) and
    (j,i), while `twobody_nn_sd.npz` lists only i<j.  Averaging a
    triangle-only matrix would halve it, so a matrix with one empty triangle is
    mirrored and one with both is averaged.
    """
    M = np.asarray(matrix, dtype=float).copy()
    np.fill_diagonal(M, 0.0)
    upper, lower = np.triu(M, 1), np.tril(M, -1)
    if np.allclose(lower, 0.0):
        return upper + upper.T
    if np.allclose(upper, 0.0):
        return lower + lower.T
    return 0.5 * (M + M.T)


@dataclass
class QuasiparticleProblem:
    """Target Hamiltonian in the hardcore-boson number sector."""

    eps: np.ndarray  # (D,)   diagonal one-body, g^(1)_AA
    g1: np.ndarray  # (D,D)  off-diagonal hopping, zero diagonal
    g2: Optional[np.ndarray] = None  # (D,D)  density-density, zero diagonal
    n_particles: int = 1
    label: str = ""

    # ---------------------------------------------------------------- init
    def __post_init__(self):
        self.eps = np.asarray(self.eps, dtype=float)
        self.g1 = _symmetrize(self.g1)
        self.g2 = np.zeros_like(self.g1) if self.g2 is None else _symmetrize(self.g2)
        if self.eps.shape[0] != self.g1.shape[0]:
            raise ValueError("eps and g1 have inconsistent sizes")

    @property
    def n_levels(self) -> int:
        """D: number of single-particle levels (= logical qubits of the 2nd-quantized encoding)."""
        return self.eps.shape[0]

    # ------------------------------------------------------------ loaders
    @classmethod
    def from_npz(
        cls,
        onebody_path: str,
        twobody_path: Optional[str] = None,
        n_levels: Optional[int] = None,
        n_particles: int = 1,
        label: str = "",
    ) -> "QuasiparticleProblem":
        """Load the NSM files `one_body_nn_sd.npz` / `twobody_nn_sd.npz`."""
        data1 = np.load(onebody_path)
        keys, values = data1["keys"], data1["values"]
        D = n_levels or int(np.max(keys) + 1)
        eps = np.zeros(D)
        g1 = np.zeros((D, D))
        for a, (i, j) in enumerate(keys):
            if i == j:
                eps[i] = values[a]
            else:
                g1[i, j] = values[a]

        g2 = None
        if twobody_path is not None:
            data2 = np.load(twobody_path)
            g2 = np.zeros((D, D))
            for a, key in enumerate(data2["keys"]):
                i, j = int(key[0]), int(key[1])
                g2[i, j] = data2["values"][a]
        return cls(eps=eps, g1=g1, g2=g2, n_particles=n_particles, label=label)

    @classmethod
    def from_dicts(
        cls,
        g_onebody: Dict[Tuple[int, int], float],
        g_twobody: Optional[Dict[Tuple[int, int], float]] = None,
        n_levels: Optional[int] = None,
        n_particles: int = 1,
        label: str = "",
    ) -> "QuasiparticleProblem":
        """Build from the `{(i,j): value}` dictionaries used in the notebooks."""
        D = n_levels or (max(max(k) for k in g_onebody) + 1)
        eps = np.zeros(D)
        g1 = np.zeros((D, D))
        for (i, j), v in g_onebody.items():
            if i == j:
                eps[i] = v
            else:
                g1[i, j] = v
        g2 = None
        if g_twobody:
            g2 = np.zeros((D, D))
            for (i, j), v in g_twobody.items():
                g2[i, j] = v
        return cls(eps=eps, g1=g1, g2=g2, n_particles=n_particles, label=label)

    # ----------------------------------------------------- logical basis
    def logical_basis(self) -> list[Config]:
        """The COMMON basis: sorted tuples of occupied levels, |A>, |A,B>, ...

        Every encoding projects back onto this basis, which is what makes
        fidelities and probabilities comparable across encodings.
        """
        return [tuple(c) for c in combinations(range(self.n_levels), self.n_particles)]

    def logical_basis_array(self) -> np.ndarray:
        """Same basis as an occupation-number array of shape (dim, D)."""
        basis = np.zeros((self.dim, self.n_levels), dtype=int)
        for r, cfg in enumerate(self.logical_basis()):
            basis[r, list(cfg)] = 1
        return basis

    @property
    def dim(self) -> int:
        from math import comb

        return comb(self.n_levels, self.n_particles)

    # ------------------------------------------------------ exact solution
    def hardcore_basis(self):
        """`NSMFermions.HardcoreBosonsBasis` on the number sector.

        Built from `src.utils.generate_particleconservation_basis`, so its
        ordering is `itertools.combinations(range(D), N)` -- the same as
        `logical_basis()`.  Every encoding projects onto that ordering.
        """
        from NSMFermions.utils_quasiparticle_approximation import HardcoreBosonsBasis

        from src.utils import generate_particleconservation_basis

        basis = generate_particleconservation_basis(
            size_a=self.n_levels,
            size_b=0,
            nparticles_a=self.n_particles,
            nparticles_b=0,
        )
        return HardcoreBosonsBasis(basis=basis)

    def logical_hamiltonian(self) -> np.ndarray:
        """Dense H_log in `logical_basis()`, built with NSMFermions.

        Same construction as the notebooks: `adag_a_matrix` for eps and g1,
        `adag_adag_a_a_matrix(A, B, A, B)` for g2.
        """
        HBB = self.hardcore_basis()
        H = 0.0
        for A in range(self.n_levels):
            H = H + self.eps[A] * HBB.adag_a_matrix(A, A)
            for B in range(self.n_levels):
                if A == B or self.g1[A, B] == 0.0:
                    continue
                H = H + self.g1[A, B] * HBB.adag_a_matrix(A, B)
        for A in range(self.n_levels):
            for B in range(A + 1, self.n_levels):
                if self.g2[A, B] != 0.0:
                    H = H + self.g2[A, B] * HBB.adag_adag_a_a_matrix(A, B, A, B)
        return np.asarray(H.todense()) if hasattr(H, "todense") else np.asarray(H)

    def exact_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """(energies, eigenvectors) of H_log in the number sector."""
        return np.linalg.eigh(self.logical_hamiltonian())

    def exact_groundstate(self) -> Tuple[float, np.ndarray]:
        """(E0, psi0) with psi0 a real vector over `logical_basis()`."""
        w, v = self.exact_spectrum()
        return float(w[0]), np.asarray(v)[:, 0]

    # ------------------------------------------- reference spin Hamiltonian
    def qutip_hamiltonian(self):
        """Full 2^D spin form (XY + ZZ + Z), as built in the notebooks.

        Only for cross-checks: `logical_hamiltonian()` is the number-sector
        object every encoding is compared against.
        """
        from ManyBodyQutip.qutip_class import SpinOperator

        D = self.n_levels
        H = 0.0
        for i in range(D):
            for j in range(i + 1, D):
                for pauli in ("x", "y"):
                    H += SpinOperator(
                        [(pauli, i, pauli, j)],
                        coupling=[0.5 * self.g1[i, j]],
                        size=D,
                        verbose=0,
                    ).qutip_op
                if self.g2[i, j] != 0.0:
                    H += SpinOperator(
                        [("qz", i, "qz", j)],
                        coupling=[self.g2[i, j]],
                        size=D,
                        verbose=0,
                    ).qutip_op
        for i in range(D):
            H += SpinOperator(
                [("qz", i)], coupling=[self.eps[i]], size=D, verbose=0
            ).qutip_op
        return H

    # ---------------------------------------------------------- utilities
    def with_external_field(self, external_field: np.ndarray) -> "QuasiparticleProblem":
        """Same geometry, only a longitudinal field: the DRIVER problem.

        `g1 = 0`, `g2` kept (it is part of the classical constraint landscape),
        `eps -> external_field`.
        """
        return QuasiparticleProblem(
            eps=np.asarray(external_field, dtype=float),
            g1=np.zeros_like(self.g1),
            g2=self.g2.copy(),
            n_particles=self.n_particles,
            label=(self.label + " [driver]").strip(),
        )

    def default_driver_field(self) -> np.ndarray:
        """External field selecting the lowest `n_particles` levels as the
        initial product state (the choice made in the O20 notebook)."""
        field = np.zeros(self.n_levels)
        order = np.argsort(self.eps)[: self.n_particles]
        field[order] = self.eps[order]
        return field
