"""Quantum adiabatic state preparation on top of the gadget.

    H(s) = (1 - s) H_driver + s H_gadget ,    s = t / tau

The driver is a purely classical (diagonal) Hamiltonian with the same
constraints, so the initial state is the product state minimizing its diagonal.
The transverse field is carried entirely by H_gadget, exactly as in the O20
notebook.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .encodings import Encoding
from .hamiltonian import GadgetHamiltonian


@dataclass
class AdiabaticResult:
    times: np.ndarray
    psi: np.ndarray  # final state, physical basis
    energy: np.ndarray  # <psi(t)|H(t)|psi(t)>
    spectrum: Optional[np.ndarray] = None  # (n_steps, nlevels), if tracked
    fidelity_exact: Optional[float] = None
    fidelity_gadget: Optional[float] = None
    leakage: Optional[float] = None

    def plot(self, gamma: float = 1.0, ax=None):
        import matplotlib.pyplot as plt

        ax = ax or plt.subplots(figsize=(8, 6))[1]
        if self.spectrum is not None:
            ax.plot(self.times, self.spectrum * gamma, linewidth=3)
        ax.plot(self.times, self.energy * gamma, "k--", label=r"$\langle H(t)\rangle$")
        ax.set_xlabel(r"$t\,\omega$", fontsize=18)
        ax.set_ylabel(r"$e_i\cdot\gamma$", fontsize=18)
        ax.legend(fontsize=12)
        return ax


class QuantumAdiabatic4Gadget:
    """Adiabatic sweep between a driver and a gadget Hamiltonian."""

    def __init__(
        self,
        gadget: GadgetHamiltonian,
        driver: Optional[GadgetHamiltonian] = None,
        schedule: Optional[Callable[[float], float]] = None,
    ):
        self.gadget = gadget
        self.driver = driver if driver is not None else gadget.driver()
        if self.driver.encoding is not gadget.encoding:
            raise ValueError("driver and gadget must share the same encoding")
        # s(t/tau); default linear ramp
        self.schedule = schedule or (lambda x: x)

    # ------------------------------------------------------------ helpers
    @property
    def encoding(self) -> Encoding:
        return self.gadget.encoding

    def initial_indices(self, tol: float = 1e-8) -> np.ndarray:
        """All degenerate classical minima of the driver (it is diagonal)."""
        diag = np.real(self.driver.total().full().diagonal())
        return np.where(diag <= diag.min() + tol * max(1.0, abs(diag.min())))[0]

    def initial_index(self) -> int:
        return int(self.initial_indices()[0])

    def initial_state(self, symmetrize: bool = True, tol: float = 1e-8) -> np.ndarray:
        """Ground state of the driver.

        In the first-quantization encoding the driver minimum is N!-fold
        degenerate (register permutations). H(s) commutes with the register
        swap, so starting from a single bitstring locks weight into the
        antisymmetric sector forever and caps the fidelity at 1/N!.
        `symmetrize=True` (default) starts from the symmetric combination; set
        it to False to reproduce the single-bitstring start of the notebooks.
        """
        psi = np.zeros(self.encoding.dim, dtype=complex)
        idxs = self.initial_indices(tol) if symmetrize else [self.initial_index()]
        psi[idxs] = 1.0 / np.sqrt(len(idxs))
        return psi

    # ----------------------------------------------------------------- run
    def run(
        self,
        tau: float = 50.0,
        steps_per_unit: int = 10,
        track_spectrum: bool = False,
        nlevels: int = 3,
        psi0: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> AdiabaticResult:
        from scipy.sparse.linalg import eigsh, expm_multiply

        H_drv = self.driver.total().data.as_scipy()
        H_gad = self.gadget.total().data.as_scipy()

        n_steps = max(int(steps_per_unit * tau), 2)
        times = np.linspace(0.0, tau, n_steps)
        dt = times[1] - times[0]

        psi = self.initial_state() if psi0 is None else np.asarray(psi0, complex)
        energy = np.zeros(n_steps)
        spectrum = np.zeros((n_steps, nlevels)) if track_spectrum else None

        for k, t in enumerate(times):
            s = self.schedule(t / tau)
            H_t = (1.0 - s) * H_drv + s * H_gad
            psi = expm_multiply(-1j * dt * H_t, psi)
            energy[k] = np.real(psi.conj().dot(H_t.dot(psi)))
            if track_spectrum:
                w = np.sort(
                    eigsh(H_t, k=nlevels, which="SA", return_eigenvectors=False)
                )
                spectrum[k] = w
            if verbose and k % max(n_steps // 10, 1) == 0:
                print(f"  s={s:.2f}  E={energy[k]:.6f}", flush=True)

        result = AdiabaticResult(times=times, psi=psi, energy=energy, spectrum=spectrum)

        _, psi_exact = self.gadget.problem.exact_groundstate()
        result.fidelity_exact = self.encoding.fidelity(psi, psi_exact, normalize=False)
        result.leakage = self.encoding.leakage(psi)
        _, psi_gadget = self.gadget.groundstate()
        result.fidelity_gadget = float(abs(np.vdot(psi_gadget, psi)) ** 2)
        return result

    # ------------------------------------------------------------- scanning
    def scan_tau(self, taus, **kwargs) -> dict:
        """Fidelity vs annealing time; the adiabatic-performance figure."""
        out = {
            "tau": np.asarray(taus, float),
            "fidelity_exact": [],
            "fidelity_gadget": [],
        }
        for tau in out["tau"]:
            r = self.run(tau=tau, **kwargs)
            out["fidelity_exact"].append(r.fidelity_exact)
            out["fidelity_gadget"].append(r.fidelity_gadget)
            print(f"tau={tau:8.2f}  F_exact={r.fidelity_exact:.6f}", flush=True)
        out["fidelity_exact"] = np.asarray(out["fidelity_exact"])
        out["fidelity_gadget"] = np.asarray(out["fidelity_gadget"])
        return out

    def minimum_gap(self, n_points: int = 51) -> tuple:
        """(s*, gap) of the instantaneous spectrum along the sweep."""
        from scipy.sparse.linalg import eigsh

        H_drv = self.driver.total().data.as_scipy()
        H_gad = self.gadget.total().data.as_scipy()
        ss = np.linspace(0.0, 1.0, n_points)
        gaps = np.zeros(n_points)
        for k, s in enumerate(ss):
            w = np.sort(
                eigsh(
                    (1 - s) * H_drv + s * H_gad,
                    k=2,
                    which="SA",
                    return_eigenvectors=False,
                )
            )
            gaps[k] = w[1] - w[0]
        i = int(np.argmin(gaps))
        return float(ss[i]), float(gaps[i])
