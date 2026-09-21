"""Drive fit: rank-1 d_A plus the per-pair self-energy parameter c_AB.

Second-order Brillouin-Wigner on the one-hot constraint gives

    g_AB^eff = - alpha_AB d_A d_B / gamma ,
    alpha_AB = (1/2) [ 1 + 1/(1 + c_AB) ],

so c_AB = 0 is the pure rank-1 gadget and c_AB is realized in hardware as a
correction (2 + c_AB) gamma to the constraint coupler of the pair (A,B).

This module is a thin, *stateful* wrapper around
`src.interaction_utils.EffectiveInteractionOptimizer2ndVersion`, so the
optimizer code stays where it already is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.interaction_utils import EffectiveInteractionOptimizer2ndVersion


@dataclass
class DriveFit:
    """Result of the two-stage fit. This is what the Hamiltonian classes take."""

    d: np.ndarray  # (D,)  transverse-field amplitudes
    c_matrix: np.ndarray  # (D,D) self-energy parameters, 0 on the diagonal
    alpha_matrix: Optional[np.ndarray] = None
    report: Optional[list] = None
    cap: Optional[float] = 4.0

    @property
    def n_levels(self) -> int:
        return self.d.shape[0]

    def reconstructed(
        self,
    ) -> np.ndarray:
        """-alpha_AB d_A d_B, i.e. the coupling actually realized."""
        alpha = (
            np.where(np.isnan(self.alpha_matrix), 1.0, self.alpha_matrix)
            if self.alpha_matrix is not None
            else alpha_from_c(self.c_matrix)
        )
        out = -alpha * np.outer(self.d, self.d)
        np.fill_diagonal(out, 0.0)
        return out

    def residual(self, g1: np.ndarray) -> np.ndarray:
        return self.reconstructed() - g1


def alpha_from_c(c: np.ndarray) -> np.ndarray:
    """alpha = (1/2)(1 + 1/(1+c)); alpha = 1 on the (unused) diagonal."""
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = 0.5 * (1.0 + 1.0 / (1.0 + c))
    alpha = np.where(np.isfinite(alpha), alpha, 1.0)
    np.fill_diagonal(alpha, 1.0)
    return alpha


def fit_drive(
    g1: np.ndarray,
    n_restarts: int = 100,
    cap: Optional[float] = 4.0,
    use_c: bool = True,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> DriveFit:
    """Stage 1: rank-1 d_A.  Stage 2: closed-form c_AB on the fixed d.

    Parameters
    ----------
    cap :
        Regularization used in the notebooks: `c[|c| > cap] = |c|`, which keeps
        the pair away from the c = -1 pole at the price of breaking the exact
        match of that pair. `None` disables it (exact c, but check stability:
        c_AB = -2 is the independent-set degeneracy).
    use_c :
        `False` returns the pure rank-1 gadget (c = 0), i.e. the original
        `EffectiveInteractionOptimizer` behaviour.
    """
    if seed is not None:
        np.random.seed(seed)

    n = g1.shape[0]
    opt = EffectiveInteractionOptimizer2ndVersion(n, n_restarts=n_restarts)
    d_opt, _ = opt.optimize_rank1(g1)

    if not use_c:
        return DriveFit(d=d_opt, c_matrix=np.zeros((n, n)), cap=None)

    alpha_matrix, c_matrix, report, _ = opt.get_alpha_and_c(g1, d_opt)
    if verbose:
        opt.print_report(g1, d_opt, alpha_matrix, c_matrix, report)

    c_matrix = np.nan_to_num(c_matrix, nan=0.0)
    if cap is not None:
        mask = np.abs(c_matrix) > cap
        c_matrix[mask] = np.abs(c_matrix[mask])
    np.fill_diagonal(c_matrix, 0.0)

    return DriveFit(
        d=d_opt, c_matrix=c_matrix, alpha_matrix=alpha_matrix, report=report, cap=cap
    )
