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
        Upper bound c_max on c_AB.  Any c_AB <= -1 (doubly occupied register
        state at (1 + c_AB) gamma, i.e. at or below the code space) and any
        c_AB > cap is set to `cap`.  This keeps the code space at the bottom of
        the spectrum and bounds the coupler (2 + c_AB) gamma, at the price of
        breaking the exact match of those pairs.  `None` disables it (exact c,
        no bounds).  For an exact fit inside ]-1, c_max] use
        `fit_drive_bounded`.
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
        c_matrix[(c_matrix <= -1.0) | (c_matrix > cap)] = cap
    np.fill_diagonal(c_matrix, 0.0)

    return DriveFit(
        d=d_opt, c_matrix=c_matrix, alpha_matrix=alpha_matrix, report=report, cap=cap
    )


# --------------------------------------------------------------------------
# Bounded fit: c_AB restricted to ]c_floor, c_max]
#
# A register with two particles (A, B) sits (1 + c_AB) gamma above the code
# space, so c_AB <= -1 puts that state below the logical subspace; a large
# c_AB stretches the energy scale (coupler (2 + c_AB) gamma).  `fit_drive`'s
# `cap` keeps c in ]-1, cap] by resetting the offending pairs to `cap`, which
# breaks their match and leaves a floor in 1 - F at large gamma.  Here d and c
# are fitted together inside the allowed window, so the match stays exact.
# --------------------------------------------------------------------------
def _c_from_alpha(alpha: np.ndarray) -> np.ndarray:
    return 1.0 / (2.0 * alpha - 1.0) - 1.0


def _bounded_fit_d(g1, c_min, c_max, d_starts, rng, n_restarts):
    """Best d for c in [c_min, c_max]; alpha is profiled out in closed form."""
    from scipy.optimize import least_squares

    n = g1.shape[0]
    iu = np.triu_indices(n, 1)
    gt = g1[iu]
    a_lo = 0.5 * (1.0 + 1.0 / (1.0 + c_max))
    a_hi = 0.5 * (1.0 + 1.0 / (1.0 + c_min))

    def alpha_of(d):
        dd = d[iu[0]] * d[iu[1]]
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(dd != 0.0, -gt / dd, a_lo)
        return np.clip(np.nan_to_num(a, nan=a_lo), a_lo, a_hi), dd

    def residual(d):
        a, dd = alpha_of(d)
        return gt + a * dd

    scale = np.sqrt(np.abs(gt).mean())
    starts = list(d_starts) + [
        rng.normal(0.0, 1.0, n) * scale * rng.uniform(0.5, 2.0)
        for _ in range(n_restarts)
    ]
    best = None
    for x0 in starts:
        r = least_squares(residual, x0, method="trf")
        err = float(np.abs(r.fun).max())
        if best is None or err < best[0]:
            best = (err, r.x)
    err, d = best
    a, _ = alpha_of(d)
    alpha = np.ones((n, n))
    alpha[iu] = a
    alpha[(iu[1], iu[0])] = a
    c = _c_from_alpha(alpha)
    np.fill_diagonal(c, 0.0)
    return err, d, c, alpha


def fit_drive_bounded(
    g1: np.ndarray,
    c_max: float = 8.0,
    c_min: Optional[float] = None,
    c_floor: float = -1.0,
    floor_margin: float = 1e-3,
    n_restarts: int = 150,
    tol: float = 1e-10,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> DriveFit:
    """Fit d and c together with every c_AB in ]c_floor, c_max].

    c_min = None (default) picks the largest c_min that still gives an exact
    fit (max |g_eff - g1| < tol), i.e. the exact fit with the widest gap
    (1 + c) gamma to the doubly occupied states.  If no exact fit exists in
    ]c_floor, c_max], the best fit at c_min = c_floor + floor_margin is
    returned.  Pass a number to fix c_min instead.
    """
    rng = np.random.default_rng(seed)
    n = g1.shape[0]
    np.random.seed(seed)
    d_rank1, _ = EffectiveInteractionOptimizer2ndVersion(
        n, n_restarts=30
    ).optimize_rank1(g1)

    def run(cm):
        return _bounded_fit_d(g1, cm, c_max, [d_rank1], rng, n_restarts)

    lowest = c_floor + floor_margin
    if c_min is None:
        best = run(lowest)
        c_used = lowest
        if best[0] < tol:  # exact somewhere: push c_min up (bisection)
            lo, hi = lowest, c_max
            for _ in range(12):
                mid = 0.5 * (lo + hi)
                trial = run(mid)
                if trial[0] < tol:
                    lo, best = mid, trial
                else:
                    hi = mid
            c_used = lo
    else:
        c_used = max(c_min, lowest)
        best = run(c_used)

    err, d, c, alpha = best
    if verbose:
        off = c[~np.eye(n, dtype=bool)]
        print(
            f"bounded fit: c in [{c_used:.3f}, {c_max}]  "
            f"max|g_eff - g1| = {err:.2e}  c range [{off.min():.3f}, {off.max():.3f}]"
        )
    return DriveFit(
        d=d,
        c_matrix=c,
        alpha_matrix=alpha,
        report=[dict(max_residual=err, c_min=c_used, c_max=c_max)],
        cap=None,
    )
