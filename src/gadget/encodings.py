"""Encodings = maps between the logical number sector and a qubit register.

Every encoding answers the same three questions, which is what makes results
from different notebooks comparable:

    physical_indices(cfg)  which computational-basis states represent |cfg>
    project(psi)           psi (2^n_qubits) -> amplitudes over problem.logical_basis()
    embed(psi_logical)     the inverse map, used to prepare initial states

`project` divides each logical amplitude by sqrt(multiplicity), so that a
perfectly encoded state projects back with unit norm whatever the redundancy of
the encoding (N! for first quantization, 1 otherwise).  The missing norm is the
leakage out of the constrained subspace.

Implemented encodings
---------------------
ParticleConservationEncoding  second quantization / one-hot: D qubits,
                              constraint gamma (sum_A N_A - N)^2
FirstQuantizationEncoding     N registers of D qubits, one particle each,
                              plus the hardcore penalty gamma2 between registers
MinorEmbeddedEncoding         any of the above, with each logical qubit replaced
                              by a ferromagnetic chain of physical qubits
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import permutations
from math import factorial
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .problem import Config, QuasiparticleProblem


def state_index(occupied: Iterable[int], n_qubits: int) -> int:
    """Index of the computational-basis state with `occupied` qubits set to 1.

    QuTiP/MSB convention: qubit 0 is the most significant bit, i.e. the same
    ordering as `src.utils.computational_basis`.
    """
    idx = 0
    for q in occupied:
        idx += 1 << (n_qubits - 1 - q)
    return int(idx)


def as_vector(psi) -> np.ndarray:
    """Accept a qutip Qobj, a scipy vector or a numpy array."""
    if hasattr(psi, "full"):
        return np.asarray(psi.full()).flatten()
    return np.asarray(psi).flatten()


class Encoding(ABC):
    """Base class. Subclasses only define `n_qubits` and `physical_indices`."""

    def __init__(self, problem: QuasiparticleProblem):
        self.problem = problem
        self.logical_basis: List[Config] = problem.logical_basis()
        self._index_of = {cfg: r for r, cfg in enumerate(self.logical_basis)}

    # ------------------------------------------------------------ abstract
    @property
    @abstractmethod
    def n_qubits(self) -> int: ...

    @abstractmethod
    def physical_indices(self, cfg: Config) -> List[int]:
        """Computational-basis indices encoding the logical configuration."""

    # ---------------------------------------------------------- machinery
    @property
    def dim(self) -> int:
        return 1 << self.n_qubits

    def computational_basis(self) -> np.ndarray:
        """(2^n, n) occupation array, row i = binary_repr(i)."""
        n = self.n_qubits
        return np.array(
            [list(np.binary_repr(i, width=n)) for i in range(1 << n)], dtype=int
        )

    def multiplicity(self, cfg: Config) -> int:
        return len(self.physical_indices(cfg))

    def logical_index_map(self) -> Dict[Config, List[int]]:
        return {cfg: self.physical_indices(cfg) for cfg in self.logical_basis}

    # ------------------------------------------------------------ project
    def project(self, psi, normalize: bool = False) -> np.ndarray:
        """psi (physical) -> amplitudes over `problem.logical_basis()`.

        Amplitudes of the degenerate representations of the same logical
        configuration are summed and divided by sqrt(multiplicity), which is the
        overlap with the correctly symmetrized encoded state.
        """
        vec = as_vector(psi)
        out = np.zeros(len(self.logical_basis), dtype=complex)
        for r, cfg in enumerate(self.logical_basis):
            idxs = self.physical_indices(cfg)
            out[r] = np.sum(vec[idxs]) / np.sqrt(len(idxs))
        if normalize:
            norm = np.linalg.norm(out)
            if norm > 0:
                out = out / norm
        return out

    def leakage(self, psi) -> float:
        """1 - (norm of the projection)^2: weight lost outside the code space."""
        return float(1.0 - np.sum(np.abs(self.project(psi)) ** 2))

    def probabilities(self, psi, normalize: bool = True) -> Dict[Config, float]:
        amps = self.project(psi, normalize=normalize)
        return {cfg: float(np.abs(a) ** 2) for cfg, a in zip(self.logical_basis, amps)}

    def fidelity(
        self, psi, psi_logical_ref: np.ndarray, normalize: bool = True
    ) -> float:
        """|<ref | P psi>|^2 with `psi_logical_ref` in the logical basis.

        `normalize=False` keeps the leakage inside the number, which is the
        honest figure of merit for the gadget; `True` reproduces the
        'fidelity within the code space' used in some notebooks.
        """
        amps = self.project(psi, normalize=normalize)
        ref = np.asarray(psi_logical_ref).flatten()
        return float(np.abs(np.vdot(ref, amps)) ** 2)

    def embed(self, psi_logical: np.ndarray) -> np.ndarray:
        """Logical amplitudes -> the physical state vector (uniform over copies)."""
        vec = np.zeros(self.dim, dtype=complex)
        for r, cfg in enumerate(self.logical_basis):
            idxs = self.physical_indices(cfg)
            vec[idxs] = psi_logical[r] / np.sqrt(len(idxs))
        return vec

    def basis_state(self, cfg: Config) -> np.ndarray:
        psi = np.zeros(len(self.logical_basis))
        psi[self._index_of[cfg]] = 1.0
        return self.embed(psi)

    def to_qutip(self, vec: np.ndarray):
        import qutip as qt

        n = self.n_qubits
        return qt.Qobj(np.asarray(vec).reshape(-1, 1), dims=[[2] * n, [1] * n])


# --------------------------------------------------------------------------
class ParticleConservationEncoding(Encoding):
    """Second quantization: D qubits, constraint gamma (sum_A N_A - N_tot)^2.

    One qubit per single-particle level, `n_particles` excitations.  This is the
    O18 (N=1) and O20 / Ca42 (N=2) 'particle conservation' encoding.
    """

    @property
    def n_qubits(self) -> int:
        return self.problem.n_levels

    @property
    def n_tot(self) -> int:
        return self.problem.n_particles

    def physical_indices(self, cfg: Config) -> List[int]:
        return [state_index(cfg, self.n_qubits)]


# --------------------------------------------------------------------------
class FirstQuantizationEncoding(Encoding):
    """N registers of D qubits, one particle per register.

    Qubit (r, A) has index r*D + A.  Constraints:
      * gamma (sum_A N_{r,A} - 1)^2 on each register  (one-hot per register)
      * gamma2 N_{r,A} N_{s,A} between registers      (hardcore boson)
    A logical configuration is represented N! times (register permutations);
    `project` handles the sqrt(N!).
    """

    @property
    def n_registers(self) -> int:
        return self.problem.n_particles

    @property
    def n_sites(self) -> int:
        return self.problem.n_levels

    @property
    def n_qubits(self) -> int:
        return self.n_registers * self.n_sites

    def qubit(self, register: int, level: int) -> int:
        return register * self.n_sites + level

    def register_of(self, qubit: int) -> int:
        return qubit // self.n_sites

    def level_of(self, qubit: int) -> int:
        return qubit % self.n_sites

    def physical_indices(self, cfg: Config) -> List[int]:
        return [
            state_index([self.qubit(r, A) for r, A in enumerate(perm)], self.n_qubits)
            for perm in permutations(cfg)
        ]

    def expected_multiplicity(self) -> int:
        return factorial(self.n_registers)


# --------------------------------------------------------------------------
class MinorEmbeddedEncoding(Encoding):
    """A base encoding with every logical qubit replaced by a chain.

    `chains` maps a logical qubit of the BASE encoding to the list of physical
    qubit indices of its chain.  A base configuration is represented by the
    physical state where every qubit of an occupied chain is 1.
    """

    def __init__(self, base: Encoding, embedding):
        """`embedding` is a `MinorEmbedding` (or a plain {logical: chain} dict)."""
        super().__init__(base.problem)
        self.base = base
        if hasattr(embedding, "chains"):
            self.embedding = embedding
            chains = embedding.chains
        else:
            self.embedding = None
            chains = embedding
        self.chains = {int(k): list(v) for k, v in chains.items()}
        missing = set(range(base.n_qubits)) - set(self.chains)
        if missing:
            raise ValueError(f"no chain given for base qubits {sorted(missing)}")
        self._n_physical = 1 + max(max(c) for c in self.chains.values())

    @property
    def base_n_tot(self) -> int:
        return getattr(self.base, "n_tot", 1)

    @property
    def n_qubits(self) -> int:
        return self._n_physical

    def physical_indices(self, cfg: Config) -> List[int]:
        out = []
        for base_idx in self.base.physical_indices(cfg):
            bits = np.binary_repr(base_idx, width=self.base.n_qubits)
            occupied = []
            for q, b in enumerate(bits):
                if b == "1":
                    occupied.extend(self.chains[q])
            out.append(state_index(occupied, self.n_qubits))
        return out

    def chain_lengths(self) -> Dict[int, int]:
        return {k: len(v) for k, v in self.chains.items()}
