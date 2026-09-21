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



def _batched_pcg(B, R, X0=None, tol=1e-12, maxiter=3000):
    """Solve B X = R (B sparse SPD, R dense with several columns), Jacobi PCG."""
    dinv = 1.0 / B.diagonal()
    X = np.zeros_like(R) if X0 is None else X0.copy()
    res = R - B @ X
    Z = dinv[:, None] * res
    Pm = Z.copy()
    rz = np.sum(res * Z, axis=0)
    norm_R = np.linalg.norm(R, axis=0)
    norm_R[norm_R == 0] = 1.0
    for it in range(maxiter):
        if np.all(np.linalg.norm(res, axis=0) <= tol * norm_R):
            return X, it
        BP = B @ Pm
        pBp = np.sum(Pm * BP, axis=0)
        pp = np.sum(Pm * Pm, axis=0)
        if np.any(pBp < -1e-12 * np.maximum(pp, 1e-300)):
            raise np.linalg.LinAlgError(
                "H_QQ - E is not positive definite: some non-code state lies below "
                "the logical ground state (the embedding/gadget is not in its "
                "valid regime at these parameters)"
            )
        alpha = np.where(pBp > 0, rz / np.where(pBp > 0, pBp, 1.0), 0.0)
        X += alpha * Pm
        res -= alpha * BP
        Z = dinv[:, None] * res
        rz_new = np.sum(res * Z, axis=0)
        beta = rz_new / np.where(rz == 0, 1.0, rz)
        rz = rz_new
        Pm = Z + beta * Pm
    warnings.warn("batched PCG did not converge", RuntimeWarning)
    return X, maxiter


def _lowdin_groundstate(H, p_idx, tol_E=1e-12, max_outer=200):
    """Exact ground state of sparse H by Loewdin/Brillouin-Wigner partitioning.

    P = the code-space states (few), Q = the rest.  With
        X(E) = (H_QQ - E)^{-1} H_QP ,   H_eff(E) = H_PP - H_QP^T X(E),
    the ground energy is the root of g(E) = lambda_min(H_eff(E)) - E, with
    g'(E) = -(1 + |X c|^2).  H_QQ - E is SPD for every E below lambda_min(H_QQ)
    >= E0 (Cauchy interlacing), so each step is a batched Jacobi-PCG solve in Q
    -- no factorisation, no fill-in.  Safeguarded Newton on a bracket; a failed
    (indefinite) solve means E > lambda_min(H_QQ) and shrinks the bracket.

    Note: the chain self-energy of order h^2/J_F can be much larger than gamma,
    so lambda_min(H_PP) is NOT a safe starting point -- hence the bracket.
    """
    import scipy.sparse as sp

    m = H.shape[0]
    mask = np.zeros(m, dtype=bool)
    mask[p_idx] = True
    q_idx = np.nonzero(~mask)[0]
    H = H.tocsr()
    H_PP = H[p_idx][:, p_idx].toarray()
    H_QP = H[q_idx][:, p_idx].toarray()
    H_QQ = H[q_idx][:, q_idx].tocsr()
    I_Q = sp.identity(len(q_idx), format="csr")

    # Gershgorin lower bound (safe: g(lo) > 0) and variational upper bound
    diag = H.diagonal()
    offsum = np.asarray(abs(H).sum(axis=1)).ravel() - np.abs(diag)
    lo = float(np.min(diag - offsum))
    hi = float(np.linalg.eigvalsh(H_PP)[0])

    def evaluate(E, X0):
        try:
            X, _ = _batched_pcg(H_QQ - E * I_Q, H_QP, X0=X0)
        except np.linalg.LinAlgError:
            return None
        H_eff = H_PP - H_QP.T @ X
        w, v = np.linalg.eigh(0.5 * (H_eff + H_eff.T))
        c = v[:, 0]
        return X, float(w[0]) - E, 1.0 + float(np.sum((X @ c) ** 2)), c

    E, X = hi, None
    for _ in range(max_outer):
        r = evaluate(E, X)
        if r is None:  # indefinite: E above lambda_min(H_QQ)
            hi = E
            E = 0.5 * (lo + hi)
            X = None
            continue
        X, g, slope, c = r
        if g > 0:
            lo = E
        else:
            hi = E
        E_new = E + g / slope  # Newton step (g' = -slope)
        if not (lo <= E_new <= hi):
            E_new = 0.5 * (lo + hi)
        if abs(E_new - E) < tol_E * max(1.0, abs(E)) or hi - lo < tol_E:
            E = E_new
            break
        E = E_new
    r = evaluate(E, X)
    if r is None:
        raise RuntimeError("Loewdin solver: could not bracket the ground energy")
    X, g, _, c = r
    E = E + g
    x = np.zeros(m)
    x[p_idx] = c
    x[q_idx] = -X @ c
    x /= np.linalg.norm(x)
    return E, x


@dataclass
class GadgetParameters:
    """All hyperparameters in one place."""

    gamma: float = 100.0
    gamma2: Optional[float] = None  # default: gamma2 = gamma (now allowed)
    J_F: Optional[float] = None  # ferromagnetic chain strength (embedding only)
    self_energy: str = "exact"  # "none" | "uniform" | "exact"
    use_c: bool = True  # include the (2 + c_AB) gamma correction
    chain_counterterm: bool = False  # k = 2 amplitude counterterm
    coupler_mode: str = "single"  # "single" | "spread": inter-chain ZZ placement

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
    def base_encoding(self) -> Encoding:
        enc = self.encoding
        return enc.base if isinstance(enc, MinorEmbeddedEncoding) else enc

    @property
    def uses_parametrized_self_energy(self) -> bool:
        """Whether the c_AB (parametrised virtual self-energy) terms apply.

        c_AB gamma N_A N_B only reshapes VIRTUAL states when every constrained
        register holds one particle: the first-quantization encoding, or the
        one-hot (N_tot = 1) particle-conservation case (notes, sec. 4.1).  In
        the particle-conservation encoding with N_tot > 1 the same term sits on
        the diagonal of the logical sector (it would need (N_p+1)-body
        penalties), so it is switched off there -- in every code path,
        embedded or not.
        """
        if not self.params.use_c:
            return False
        base = self.base_encoding
        if isinstance(base, ParticleConservationEncoding):
            return base.n_tot == 1
        return True

    @property
    def c_matrix(self) -> np.ndarray:
        if not self.uses_parametrized_self_energy:
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
            if n_tot > 1 and isinstance(self.base_encoding, ParticleConservationEncoding):
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

    # ------------------------------------------------ longitudinal terms
    # Every longitudinal Hamiltonian is described by the same three objects,
    # built once in `_base_terms` and reused by the unembedded builders, the
    # minor-embedded builder and the matrix-free sparse solver:
    #     zz[(i, j)] : coefficient of N_i N_j  (i < j)
    #     z[i]       : coefficient of N_i
    #     const      : multiple of the identity
    # with N = (1 - Z)/2 the occupation ("qz" in ManyBodyQutip).

    def _base_terms(self, base: Encoding):
        """(zz, z, const) of the UNEMBEDDED longitudinal Hamiltonian of `base`."""
        p, prm = self.problem, self.params
        gamma = prm.gamma

        if isinstance(base, ParticleConservationEncoding):
            D, n_tot = p.n_levels, base.n_tot
            c = self.c_matrix  # zero unless n_tot == 1 (uses_parametrized_self_energy)
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
            return zz, z, gamma * n_tot**2

        if isinstance(base, FirstQuantizationEncoding):
            gamma2 = prm.gamma2_value
            D, N = base.n_sites, base.n_registers
            c = self.c_matrix
            theta = self._theta(1)
            n_tot = 1  # one particle per register
            phi = (
                se.phi_symmetrized(self.drive.d, c, prm.gamma2_ratio)
                if prm.self_energy in ("exact", "full")
                else np.zeros((D, D))
            )
            zz: Dict = {}
            # one-hot constraint (+ parametrised self-energy) inside every register
            for r in range(N):
                for A in range(D):
                    for B in range(A + 1, D):
                        zz[(base.qubit(r, A), base.qubit(r, B))] = (
                            2.0 + c[A, B]
                        ) * gamma
            # hard-core boson + two-body + self-energy compensation between registers
            for r in range(N):
                for s in range(r + 1, N):
                    for A in range(D):
                        zz[(base.qubit(r, A), base.qubit(s, A))] = gamma2
                    for A in range(D):
                        for B in range(D):
                            if A == B:
                                continue
                            # both ordered couplers (A,B) and (B,A) are built, but
                            # only one is active on a given logical configuration,
                            # so each carries the FULL weight (no 1/2).
                            w = (p.g2[A, B] + phi[A, B]) / gamma
                            if w != 0.0:
                                zz[(base.qubit(r, A), base.qubit(s, B))] = w
            z = {
                base.qubit(r, A): (p.eps[A] + theta[A]) / gamma
                + gamma * (1 - 2 * n_tot)
                for r in range(N)
                for A in range(D)
            }
            return zz, z, N * gamma * n_tot**2

        raise NotImplementedError(f"no longitudinal terms for {type(base).__name__}")

    def _embedded_terms(self, enc: MinorEmbeddedEncoding):
        """(zz, z, zz_pauli, const) of the minor-embedded longitudinal Hamiltonian.

        * base ZZ couplings go on the hardware couplers joining the two chains:
          ONE representative coupler (`coupler_mode="single"`, default) or split
          evenly over all of them (`"spread"`, which on the logical sector gives
          the same energy but a smaller spurious neighbour-conditioned field);
        * base fields are spread uniformly over the chain;
        * every chain bond gets -J_F Z Z, plus the constant that re-centres the
          spectrum (a fully aligned chain then costs 0).
        """
        emb = enc.embedding
        if emb is None:
            raise ValueError("MinorEmbeddedEncoding needs a MinorEmbedding to build H")
        J_F = self.params.J_F
        if J_F is None:
            raise ValueError("GadgetParameters.J_F is required for a minor embedding")
        mode = self.params.coupler_mode
        if mode not in ("single", "spread"):
            raise ValueError(f"unknown coupler_mode {mode!r}")

        base_zz, base_z, const = self._base_terms(enc.base)
        all_couplers = emb.all_couplers()

        zz: Dict = {}
        for (i, j), w in base_zz.items():
            key = (min(i, j), max(i, j))
            phys = all_couplers.get(key)
            if not phys:
                raise ValueError(
                    f"no hardware coupler for logical pair {key}: embed the "
                    "encoding's coupling graph (GadgetHamiltonian.logical_graph())"
                )
            if mode == "single":
                phys = phys[:1]
            for u, v in phys:
                k = (min(u, v), max(u, v))
                zz[k] = zz.get(k, 0.0) + w / len(phys)

        z: Dict = {}
        for i, h in base_z.items():
            chain = emb.chains[i]
            for q in chain:
                z[q] = z.get(q, 0.0) + h / len(chain)

        zz_pauli = {(min(a, b), max(a, b)): -J_F for a, b in emb.intra_edges}
        return zz, z, zz_pauli, const + J_F * len(emb.intra_edges)

    def _terms(self):
        """(zz, z, zz_pauli, const) of the longitudinal part, physical indices."""
        enc = self.encoding
        if isinstance(enc, MinorEmbeddedEncoding):
            return self._embedded_terms(enc)
        if isinstance(enc, (ParticleConservationEncoding, FirstQuantizationEncoding)):
            zz, z, const = self._base_terms(enc)
            return zz, z, {}, const
        raise TypeError(f"no builder for encoding {type(enc).__name__}")

    def _qutip_from_terms(self, zz, z, zz_pauli, const):
        from ManyBodyQutip.qutip_class import SpinOperator

        n = self.n_qubits
        H = const * self._identity()
        for (i, j), w in zz.items():
            if w != 0.0:
                H += SpinOperator(
                    [("qz", i, "qz", j)], coupling=[w], size=n, verbose=0
                ).qutip_op
        for i, h in z.items():
            if h != 0.0:
                H += SpinOperator([("qz", i)], coupling=[h], size=n, verbose=0).qutip_op
        for (a, b), w in zz_pauli.items():
            H += SpinOperator(
                [("z", a, "z", b)], coupling=[w], size=n, verbose=0
            ).qutip_op
        return H

    # kept for backward compatibility with notebooks that call them directly
    def _build_particle_conservation(self, enc: ParticleConservationEncoding):
        zz, z, const = self._base_terms(enc)
        return self._qutip_from_terms(zz, z, {}, const)

    def _build_first_quantization(self, enc: FirstQuantizationEncoding):
        zz, z, const = self._base_terms(enc)
        return self._qutip_from_terms(zz, z, {}, const)

    def _build_minor_embedded(self, enc: MinorEmbeddedEncoding):
        return self._qutip_from_terms(*self._embedded_terms(enc))

    def logical_graph(self, tol: float = 0.0):
        """networkx graph of the base encoding's ZZ couplings (|w| > tol).

        This is the graph to hand to `find_minor_embedding`: K_n for the
        particle-conservation encoding, (almost) K_{N n} for first quantization.
        """
        import networkx as nx

        enc = self.encoding
        base = enc.base if isinstance(enc, MinorEmbeddedEncoding) else enc
        zz, _, _ = self._base_terms(base)
        G = nx.Graph()
        G.add_nodes_from(range(base.n_qubits))
        G.add_edges_from(k for k, w in zz.items() if abs(w) > tol)
        return G

    # ------------------------------------------------ matrix-free operator
    def transverse_amplitudes(self) -> np.ndarray:
        """Coefficient of X_q on every physical qubit."""
        enc = self.encoding
        n = enc.n_qubits
        d = self.drive.d
        if not self.transverse_on:
            return np.zeros(n)
        if isinstance(enc, MinorEmbeddedEncoding):
            # logical amplitude of every BASE qubit, then chain-rescaled
            d_base = self._base_amplitudes(enc.base)
            return physical_drive(
                d_base, enc.embedding, self.params.J_F, scale=1.0,
                counterterm=self.params.chain_counterterm,
            )
        return self._base_amplitudes(enc)

    def _base_amplitudes(self, base: Encoding) -> np.ndarray:
        d = self.drive.d
        if isinstance(base, ParticleConservationEncoding):
            return d / np.sqrt(2)
        if isinstance(base, FirstQuantizationEncoding):
            return np.concatenate([d / np.sqrt(2)] * base.n_registers)
        raise TypeError(f"no transverse builder for {type(base).__name__}")

    def diagonal(self) -> np.ndarray:
        """Diagonal of H in the computational basis, without building H.

        Qubit q is bit (n-1-q) of the basis index (qutip tensor order),
        N_q = bit, Z_q = 1 - 2 bit.
        """
        zz, z, zz_pauli, const = self._terms()
        n = self.n_qubits
        idx = np.arange(1 << n, dtype=np.int64)
        bits = [((idx >> (n - 1 - q)) & 1).astype(np.uint8) for q in range(n)]
        del idx
        diag = np.full(1 << n, float(const))
        for (i, j), w in zz.items():
            if w != 0.0:
                diag += w * (bits[i] & bits[j])
        for i, h in z.items():
            if h != 0.0:
                diag += h * bits[i]
        for (a, b), w in zz_pauli.items():
            # Z_a Z_b = +1 if the bits agree, -1 otherwise
            diag += w * (1.0 - 2.0 * (bits[a] ^ bits[b]))
        return diag

    def linear_operator(self):
        """scipy LinearOperator for H (diagonal + single-qubit X), O(n 2^n) memory-light."""
        from scipy.sparse.linalg import LinearOperator

        n = self.n_qubits
        diag = self.diagonal()
        amps = self.transverse_amplitudes()
        active = [(q, a) for q, a in enumerate(amps) if a != 0.0]

        def matvec(x):
            x = np.asarray(x).reshape(-1)
            y = diag * x
            for q, a in active:
                xv = x.reshape(1 << q, 2, 1 << (n - 1 - q))
                y += a * xv[:, ::-1, :].reshape(-1)
            return y

        def matmat(X):
            return np.column_stack([matvec(X[:, k]) for k in range(X.shape[1])])

        dim = 1 << n
        return (
            LinearOperator((dim, dim), matvec=matvec, matmat=matmat, dtype=float),
            diag,
        )

    # ------------------------------------------------------------ transverse
    def _build_transverse(self):
        from ManyBodyQutip.qutip_class import SpinOperator

        n = self.n_qubits
        H = 0.0
        for q, a in enumerate(self.transverse_amplitudes()):
            if a != 0.0:
                H += SpinOperator([("x", q)], coupling=[a], size=n, verbose=0).qutip_op
        return H

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
            coupler_mode=self.params.coupler_mode,
        )
        return GadgetHamiltonian(
            driver_problem, self.encoding, driver_drive, driver_params, transverse=False
        )

    # ----------------------------------------------------------- diagnostics
    DENSE_MAX_QUBITS = 13
    max_flips = 5

    def groundstate(self, solver: str = "auto", tol: float = 1e-10, maxiter: int = 5000):
        """(E0, psi0) of the total Hamiltonian; psi0 as a numpy vector.

        solver:
          "auto"   dense for n <= DENSE_MAX_QUBITS, otherwise "sparse"
          "scipy"  dense scipy.linalg.eigh (the previous default)
          "qutip"  qutip eigenstates
          "sparse" matrix-free LOBPCG with a Jacobi preconditioner, started from
                   the encoded exact logical ground state. Exact, but slow when
                   J_F >> logical scale (bad conditioning): fine for the
                   unembedded encodings, slow for embedded ones.
          "truncated"  exact ground state in the subspace of basis states
                   within `self.max_flips` bit flips of the code space
                   (default 5), solved by self-consistent Loewdin partitioning
                   (code space vs rest) with batched preconditioned CG.  This is the natural perturbative truncation:
                   a chain of length k tunnels through states <= k flips away.
                   Check convergence with `truncation_scan`.  Default for
                   embedded Hamiltonians above DENSE_MAX_QUBITS.
        The result is cached (the Hamiltonian is immutable once built).
        """
        cache = getattr(self, "_gs_cache", None)
        if cache is not None and cache[0] == solver:
            return cache[1]
        if solver == "auto":
            if self.n_qubits <= self.DENSE_MAX_QUBITS:
                solver = "scipy"
            elif isinstance(self.encoding, MinorEmbeddedEncoding):
                solver = "truncated"
            else:
                solver = "sparse"
        if solver == "sparse":
            out = self._groundstate_sparse(tol=tol, maxiter=maxiter)
        elif solver == "truncated":
            out = self._groundstate_truncated(self.max_flips)
        elif solver == "qutip":
            w, v = self.total().eigenstates(eigvals=1)
            out = float(w[0]), np.asarray(v[0].full()).flatten()
        elif solver == "scipy":
            import scipy.linalg

            M = self.total().full()
            if np.max(np.abs(M.imag)) > 1e-12:
                raise ValueError("unexpected imaginary part in the Hamiltonian")
            w, v = scipy.linalg.eigh(M.real, subset_by_index=[0, 0])
            out = float(w[0]), v[:, 0]
        else:
            raise ValueError(f"unknown solver {solver!r}")
        self._gs_cache = (solver, out)
        return out

    def _groundstate_sparse(self, tol: float, maxiter: int):
        from scipy.sparse.linalg import LinearOperator, lobpcg

        A, diag = self.linear_operator()
        dim = diag.shape[0]

        # starting vector: the exact logical ground state, encoded
        _, psi_log = self.problem.exact_groundstate()
        x0 = np.real(self.encoding.embed(psi_log))
        rng = np.random.default_rng(0)
        x0 = x0 + 1e-6 * rng.standard_normal(dim)
        x0 /= np.linalg.norm(x0)
        e0 = float(x0 @ A.matvec(x0))

        # Jacobi preconditioner (diag - e0)^-1, regularised on the logical scale
        amps = self.transverse_amplitudes()
        reg = max(1e-3, float(np.max(np.abs(amps))) if amps.size else 1.0)
        inv = 1.0 / (np.abs(diag - e0) + reg)
        M = LinearOperator((dim, dim), matvec=lambda x: inv * np.asarray(x).reshape(-1),
                           matmat=lambda X: inv[:, None] * X, dtype=float)

        w, v = lobpcg(A, x0.reshape(-1, 1), M=M, largest=False, tol=tol,
                      maxiter=maxiter, verbosityLevel=0)
        psi = v[:, 0]
        res = np.linalg.norm(A.matvec(psi) - w[0] * psi)
        self.sparse_residual = float(res)
        if res > 1e3 * tol * max(1.0, abs(w[0])):
            warnings.warn(f"LOBPCG residual {res:.2e} above tolerance", RuntimeWarning)
        return float(w[0]), psi

    # --------------------------------------------------- truncated solver
    def truncated_space(self, max_flips: int) -> np.ndarray:
        """Sorted basis indices within `max_flips` bit flips of the code space."""
        n = self.n_qubits
        S = np.unique(np.concatenate(
            [np.asarray(v, dtype=np.int64) for v in self.encoding.logical_index_map().values()]
        ))
        masks = np.array([1 << (n - 1 - q) for q in range(n)], dtype=np.int64)
        frontier = S
        for _ in range(max_flips):
            new = np.unique((frontier[:, None] ^ masks[None, :]).ravel())
            new = np.setdiff1d(new, S, assume_unique=True)
            if new.size == 0:
                break
            S = np.union1d(S, new)
            frontier = new
        return S

    def _groundstate_truncated(self, max_flips: int):
        import scipy.sparse as sp
        from scipy.sparse.linalg import eigsh

        n = self.n_qubits
        S = self.truncated_space(max_flips)
        m = S.size
        zz, z, zz_pauli, const = self._terms()
        bits = [((S >> (n - 1 - q)) & 1).astype(np.uint8) for q in range(n)]
        diag = np.full(m, float(const))
        for (i, j), w in zz.items():
            if w != 0.0:
                diag += w * (bits[i] & bits[j])
        for i, h in z.items():
            if h != 0.0:
                diag += h * bits[i]
        for (a, b), w in zz_pauli.items():
            diag += w * (1.0 - 2.0 * (bits[a] ^ bits[b]))
        del bits
        rows, cols, vals = [np.arange(m)], [np.arange(m)], [diag]
        for q, amp in enumerate(self.transverse_amplitudes()):
            if amp == 0.0:
                continue
            T = S ^ (1 << (n - 1 - q))
            pos = np.searchsorted(S, T)
            pos[pos == m] = 0
            ok = S[pos] == T
            rows.append(np.nonzero(ok)[0]); cols.append(pos[ok])
            vals.append(np.full(int(ok.sum()), amp))
        Hs = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(m, m),
        )
        if m <= 4000:
            w, v = np.linalg.eigh(Hs.toarray())
            E0, x = float(w[0]), v[:, 0]
        else:
            code = np.unique(np.concatenate([
                np.asarray(v_, dtype=np.int64)
                for v_ in self.encoding.logical_index_map().values()
            ]))
            E0, x = _lowdin_groundstate(Hs, np.searchsorted(S, code))
        self.truncated_dim = int(m)
        if n > 26:
            raise NotImplementedError(
                "full-vector output is limited to 26 qubits; use truncated_space() "
                "and the subspace vector directly"
            )
        psi = np.zeros(1 << n)
        psi[S] = x
        self.truncation_weight_boundary = float(
            np.sum(x[np.isin(S, self._boundary(S, max_flips))] ** 2)
        ) if max_flips > 0 else 0.0
        return E0, psi

    def _boundary(self, S, max_flips):
        """States exactly `max_flips` flips from the code space (outer shell)."""
        inner = self.truncated_space(max_flips - 1)
        return np.setdiff1d(S, inner, assume_unique=True)

    def truncation_scan(self, flips=(3, 4, 5, 6), verbose: bool = True):
        """1-F and energy vs the truncation radius: use it to pick `max_flips`."""
        E_ex, psi_ex = self.problem.exact_groundstate()
        out = []
        for m in flips:
            self._gs_cache = None
            E0, psi = self._groundstate_truncated(m)
            F = self.encoding.fidelity(psi, psi_ex, normalize=False)
            out.append(dict(max_flips=m, dim=self.truncated_dim, E0=E0,
                            infidelity=1 - F,
                            boundary_weight=self.truncation_weight_boundary))
            if verbose:
                print(f"  flips={m}: dim={self.truncated_dim:>8d}  E0={E0:.10f}  "
                      f"1-F={1-F:.4e}  outer-shell weight={self.truncation_weight_boundary:.1e}")
        self._gs_cache = None
        return out

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
            # same, renormalised inside the code space (leakage removed)
            "fidelity_in_code_space": self.encoding.fidelity(
                psi, psi_exact, normalize=True
            ),
            "leakage": self.encoding.leakage(psi),
        }
        if hasattr(self, "sparse_residual"):
            out["sparse_residual"] = self.sparse_residual
        if hasattr(self, "truncated_dim"):
            out["truncated_dim"] = self.truncated_dim
            out["truncation_boundary_weight"] = self.truncation_weight_boundary
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
