"""The single Hamiltonian class every notebook should now call.

    H = H_long(gamma, gamma2, c_AB, g2, eps, N_tot)  +  H_trans(d)

built with `ManyBodyQutip.SpinOperator` exactly as in the notebooks, but with
the encoding-specific bookkeeping hidden in `encodings.py`.

Units: the whole Hamiltonian is written so that the *logical* energies come out
divided by gamma (i.e. compare `eigenvalue * gamma` with the exact NSM energy),
which is the convention of `GadgetO18/O20`.

The DRIVER Hamiltonian is the same object with `g1 = 0`, an external field in
place of eps, and no transverse field: `GadgetHamiltonian.driver()`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

from . import selfenergy as se
from .embedding import MinorEmbedding, physical_drive
from .encodings import (
    Encoding,
    FirstQuantizationEncoding,
    MinorEmbeddedEncoding,
    ParticleConservationEncoding,
)
from .fit import DriveFit
from .problem import QuasiparticleProblem


@dataclass
class GadgetParameters:
    """All hyperparameters in one place."""

    gamma: float = 100.0
    gamma2: Optional[float] = None  # default: gamma2 = gamma (now allowed)
    J_F: Optional[float] = None  # ferromagnetic chain strength (embedding only)
    self_energy: str = "exact"  # "none" | "uniform" | "exact"
    use_c: bool = True  # include the (2 + c_AB) gamma correction
    chain_counterterm: bool = False  # k = 2 amplitude counterterm

    @property
    def gamma2_value(self) -> float:
        return self.gamma if self.gamma2 is None else self.gamma2

    @property
    def gamma2_ratio(self) -> float:
        return self.gamma2_value / self.gamma


class GadgetHamiltonian:
    """Total gadget Hamiltonian for a (problem, encoding, drive) triple."""

    def __init__(
        self,
        problem: QuasiparticleProblem,
        encoding: Encoding,
        drive: DriveFit,
        params: GadgetParameters,
        transverse: bool = True,
    ):
        self.problem = problem
        self.encoding = encoding
        self.drive = drive
        self.params = params
        self.transverse_on = transverse
        self._long = None
        self._trans = None

    # ------------------------------------------------------------ helpers
    @property
    def n_qubits(self) -> int:
        return self.encoding.n_qubits

    @property
    def c_matrix(self) -> np.ndarray:
        if not self.params.use_c:
            return np.zeros_like(self.drive.c_matrix)
        return self.drive.c_matrix

    def _identity(self):
        import qutip as qt

        return qt.tensor([qt.qeye(2)] * self.n_qubits)

    def _theta(self, n_tot: int) -> np.ndarray:
        """Longitudinal compensation of the self-energy, per level, units 1/gamma."""
        mode = self.params.self_energy
        D = self.problem.n_levels
        if mode == "none":
            return np.zeros(D)
        if mode == "uniform":
            return np.full(D, se.theta_uniform(self.drive.d, n_tot))
        if mode in ("exact", "full"):
            if n_tot > 1 and isinstance(self.encoding, ParticleConservationEncoding):
                warnings.warn(
                    "exact theta_1 is derived for one particle per constrained "
                    "register; falling back to the uniform shift for the "
                    "second-quantized encoding with N_tot > 1.",
                    RuntimeWarning,
                )
                return np.full(D, se.theta_uniform(self.drive.d, n_tot))
            return se.theta_one_body(self.drive.d, self.c_matrix)
        raise ValueError(f"unknown self_energy mode {mode!r}")

    # ------------------------------------------------------------- builder
    def longitudinal(self):
        if self._long is None:
            enc = self.encoding
            if isinstance(enc, ParticleConservationEncoding):
                self._long = self._build_particle_conservation(enc)
            elif isinstance(enc, FirstQuantizationEncoding):
                self._long = self._build_first_quantization(enc)
            elif isinstance(enc, MinorEmbeddedEncoding):
                self._long = self._build_minor_embedded(enc)
            else:
                raise TypeError(f"no builder for encoding {type(enc).__name__}")
        return self._long

    def transverse(self):
        if self._trans is None:
            self._trans = self._build_transverse()
        return self._trans

    def total(self):
        H = self.longitudinal()
        if self.transverse_on:
            H = H + self.transverse()
        return H

    # ------------------------------------------- second quantization / one-hot
    def _build_particle_conservation(self, enc: ParticleConservationEncoding):
        from ManyBodyQutip.qutip_class import SpinOperator

        p, prm = self.problem, self.params
        gamma, D, n_tot = prm.gamma, p.n_levels, enc.n_tot
        c = self.c_matrix
        if n_tot > 1 and np.any(c != 0.0):
            # With N_tot > 1 a term c_AB gamma N_A N_B is NOT a correction to a
            # virtual state: it sits on the diagonal of the logical sector and
            # splits it at order gamma. c_AB is only meaningful when each
            # constrained register holds a single particle (N_tot = 1, or the
            # first-quantization encoding).
            warnings.warn(
                "c_AB is dropped for the second-quantized encoding with "
                "N_tot > 1: it would split the logical sector at order gamma. "
                "Use FirstQuantizationEncoding to keep the c_AB correction.",
                RuntimeWarning,
            )
            c = np.zeros_like(c)
        theta = self._theta(n_tot)

        H_zz = 0.0
        for i in range(D):
            for j in range(i + 1, D):
                coupling = (2.0 + c[i, j]) * gamma + p.g2[i, j] / gamma
                H_zz += SpinOperator(
                    [("qz", i, "qz", j)], coupling=[coupling], size=D, verbose=0
                ).qutip_op

        H_z = 0.0
        for i in range(D):
            h = (p.eps[i] + theta[i]) / gamma + gamma * (1 - 2 * n_tot)
            H_z += SpinOperator([("qz", i)], coupling=[h], size=D, verbose=0).qutip_op

        return H_zz + H_z + gamma * (n_tot**2) * self._identity()

    # ------------------------------------------------- first quantization / NxD
    def _build_first_quantization(self, enc: FirstQuantizationEncoding):
        from ManyBodyQutip.qutip_class import SpinOperator

        p, prm = self.problem, self.params
        gamma, gamma2 = prm.gamma, prm.gamma2_value
        D, N, n = enc.n_sites, enc.n_registers, enc.n_qubits
        c = self.c_matrix
        theta = self._theta(1)
        n_tot = 1  # one particle per register

        phi = (
            se.phi_symmetrized(self.drive.d, c, prm.gamma2_ratio)
            if prm.self_energy in ("exact", "full")
            else np.zeros((D, D))
        )

        H_zz = 0.0
        # one-hot constraint inside every register
        for r in range(N):
            for A in range(D):
                for B in range(A + 1, D):
                    H_zz += SpinOperator(
                        [("qz", enc.qubit(r, A), "qz", enc.qubit(r, B))],
                        coupling=[(2.0 + c[A, B]) * gamma],
                        size=n,
                        verbose=0,
                    ).qutip_op

        # hardcore boson + two-body + self-energy compensation between registers
        for r in range(N):
            for s in range(r + 1, N):
                for A in range(D):
                    H_zz += SpinOperator(
                        [("qz", enc.qubit(r, A), "qz", enc.qubit(s, A))],
                        coupling=[gamma2],
                        size=n,
                        verbose=0,
                    ).qutip_op
                for A in range(D):
                    for B in range(D):
                        if A == B:
                            continue
                        # both ordered couplers (A,B) and (B,A) are built, but only
                        # one of them is active on a given logical configuration,
                        # so each carries the FULL weight (no 1/2).
                        w = (p.g2[A, B] + phi[A, B]) / gamma
                        if w == 0.0:
                            continue
                        H_zz += SpinOperator(
                            [("qz", enc.qubit(r, A), "qz", enc.qubit(s, B))],
                            coupling=[w],
                            size=n,
                            verbose=0,
                        ).qutip_op

        H_z = 0.0
        for r in range(N):
            for A in range(D):
                h = (p.eps[A] + theta[A]) / gamma + gamma * (1 - 2 * n_tot)
                H_z += SpinOperator(
                    [("qz", enc.qubit(r, A))], coupling=[h], size=n, verbose=0
                ).qutip_op

        return H_zz + H_z + N * gamma * (n_tot**2) * self._identity()

    # ------------------------------------------------------- minor embedding
    def _build_minor_embedded(self, enc: MinorEmbeddedEncoding):
        """Map the base longitudinal Hamiltonian onto chains + couplers.

        Base ZZ couplings go on ONE representative hardware coupler per logical
        pair (dedup), base fields are spread uniformly over the chain, and every
        chain bond gets -J_F Z Z plus the constant that re-centres the spectrum.
        """
        from ManyBodyQutip.qutip_class import SpinOperator

        emb = enc.embedding
        if emb is None:
            raise ValueError("MinorEmbeddedEncoding needs a MinorEmbedding to build H")
        prm = self.params
        J_F = prm.J_F
        if J_F is None:
            raise ValueError("GadgetParameters.J_F is required for a minor embedding")

        n = enc.n_qubits
        zz, z = self._base_terms(enc.base)
        couplers = emb.couplers()

        H = 0.0
        for (i, j), w in zz.items():
            key = (min(i, j), max(i, j))
            if key not in couplers:
                raise ValueError(f"no hardware coupler for logical pair {key}")
            u, v = couplers[key]
            H += SpinOperator(
                [("qz", u, "qz", v)], coupling=[w], size=n, verbose=0
            ).qutip_op

        for i, h in z.items():
            chain = emb.chains[i]
            for q in chain:
                H += SpinOperator(
                    [("qz", q)], coupling=[h / len(chain)], size=n, verbose=0
                ).qutip_op

        for a, b in emb.intra_edges:
            H += SpinOperator(
                [("z", a, "z", b)], coupling=[-J_F], size=n, verbose=0
            ).qutip_op

        identity = self._identity()
        constant = self.params.gamma * (enc.base_n_tot**2)
        return H + constant * identity + J_F * len(emb.intra_edges) * identity

    def _base_terms(self, base: Encoding):
        """{(i,j): coupling} and {i: field} of the base (unembedded) Hamiltonian."""
        p, prm = self.problem, self.params
        gamma = prm.gamma
        c = self.c_matrix

        if isinstance(base, ParticleConservationEncoding):
            D, n_tot = p.n_levels, base.n_tot
            theta = self._theta(n_tot)
            zz = {
                (i, j): (2.0 + c[i, j]) * gamma + p.g2[i, j] / gamma
                for i in range(D)
                for j in range(i + 1, D)
            }
            z = {
                i: (p.eps[i] + theta[i]) / gamma + gamma * (1 - 2 * n_tot)
                for i in range(D)
            }
            return zz, z

        raise NotImplementedError(
            "minor embedding of the first-quantization encoding is not wired yet; "
            "embed each register separately, or open an issue in the notes."
        )

    # ------------------------------------------------------------ transverse
    def _build_transverse(self):
        from ManyBodyQutip.qutip_class import SpinOperator

        enc = self.encoding
        n = enc.n_qubits
        d = self.drive.d
        H = 0.0

        if isinstance(enc, ParticleConservationEncoding):
            for i in range(n):
                H += SpinOperator(
                    [("x", i)], coupling=[d[i] / np.sqrt(2)], size=n, verbose=0
                ).qutip_op
            return H

        if isinstance(enc, FirstQuantizationEncoding):
            for r in range(enc.n_registers):
                for A in range(enc.n_sites):
                    H += SpinOperator(
                        [("x", enc.qubit(r, A))],
                        coupling=[d[A] / np.sqrt(2)],
                        size=n,
                        verbose=0,
                    ).qutip_op
            return H

        if isinstance(enc, MinorEmbeddedEncoding):
            d_phys = physical_drive(
                d,
                enc.embedding,
                self.params.J_F,
                counterterm=self.params.chain_counterterm,
            )
            for q in range(n):
                if d_phys[q] == 0.0:
                    continue
                H += SpinOperator(
                    [("x", q)], coupling=[d_phys[q]], size=n, verbose=0
                ).qutip_op
            return H

        raise TypeError(f"no transverse builder for {type(enc).__name__}")

    # ---------------------------------------------------------------- driver
    def driver(
        self, external_field: Optional[np.ndarray] = None
    ) -> "GadgetHamiltonian":
        """The driver: same constraints, only an external longitudinal field.

        `g1 = 0` (so no drive is needed), `eps -> external_field`, transverse
        field off.  Its ground state is a classical product state, the starting
        point of the adiabatic protocol.
        """
        field = (
            self.problem.default_driver_field()
            if external_field is None
            else np.asarray(external_field, float)
        )
        driver_problem = self.problem.with_external_field(field)
        driver_drive = DriveFit(
            d=np.zeros_like(self.drive.d),
            c_matrix=self.c_matrix.copy(),
            cap=self.drive.cap,
        )
        driver_params = GadgetParameters(
            gamma=self.params.gamma,
            gamma2=self.params.gamma2,
            J_F=self.params.J_F,
            self_energy="none",  # no drive -> no self-energy to compensate
            use_c=self.params.use_c,
            chain_counterterm=self.params.chain_counterterm,
        )
        return GadgetHamiltonian(
            driver_problem, self.encoding, driver_drive, driver_params, transverse=False
        )

    # ----------------------------------------------------------- diagnostics
    def groundstate(self, solver: str = "scipy"):
        """(E0, psi0) of the total Hamiltonian; psi0 as a numpy vector."""
        import scipy.linalg

        H = self.total()
        if solver == "qutip":
            w, v = H.eigenstates(eigvals=1)
            return float(w[0]), np.asarray(v[0].full()).flatten()
        M = H.full()
        if np.max(np.abs(M.imag)) > 1e-12:
            raise ValueError("unexpected imaginary part in the Hamiltonian")
        w, v = scipy.linalg.eigh(M.real, subset_by_index=[0, 0])
        return float(w[0]), v[:, 0]

    def unembedded(self) -> "GadgetHamiltonian":
        """Same parameters on the base (unembedded) encoding."""
        if not isinstance(self.encoding, MinorEmbeddedEncoding):
            return self
        return GadgetHamiltonian(
            self.problem,
            self.encoding.base,
            self.drive,
            self.params,
            transverse=self.transverse_on,
        )

    def chain_energy_offset(self) -> float:
        """E0(embedded) - E0(base), in gadget units.

        OPEN ITEM: the physical drive on a chain of length k >= 2 has magnitude
        |d|^{1/k} J_F^{(k-1)/k}, which is huge, and its virtual domain-wall
        excitations produce an extra *diagonal* shift that is NOT compensated
        anywhere yet.  It is a constant on the logical sector (so the fidelity
        is unaffected) but it spoils the absolute energy.  Use this number to
        subtract it, or derive the counterterm.
        """
        return self.groundstate()[0] - self.unembedded().groundstate()[0]

    def check(self, verbose: bool = True) -> Dict[str, float]:
        """Standard consistency checks: energy, fidelity, leakage, degeneracy."""
        E0_exact, psi_exact = self.problem.exact_groundstate()
        E0, psi = self.groundstate()
        energy = E0 * self.params.gamma
        out = {
            "gadget_energy": energy,
            "exact_energy": E0_exact,
            "relative_energy_error": abs(energy - E0_exact) / abs(E0_exact),
            "fidelity": self.encoding.fidelity(psi, psi_exact, normalize=False),
            "leakage": self.encoding.leakage(psi),
        }
        if isinstance(self.encoding, MinorEmbeddedEncoding):
            offset = self.chain_energy_offset()
            out["chain_energy_offset"] = offset
            corrected = (E0 - offset) * self.params.gamma
            out["relative_energy_error_corrected"] = abs(corrected - E0_exact) / abs(
                E0_exact
            )
        if verbose:
            for k, v in out.items():
                print(f"{k:>24}: {v: .6e}")
        return out
