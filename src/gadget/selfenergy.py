"""Second-order self-energy of the virtual states, and its compensation.

Setup: a register carrying one particle at level A, constraint
`gamma (sum_A N_A - 1)^2` with the per-pair correction `(2 + c_AB) gamma`, and
the transverse field `V = sum_A (d_A/sqrt2) X_A`.

Single-qubit flips out of |A> and their virtual cost (in units of gamma):

  * flip A down (empty register)                        ->  1
  * flip B != A up (two particles in the register)      ->  1 + c_AB
  * flip B != A up ONTO the level occupied by another
    register (first-quantization encoding only)         ->  1 + c_AB + gamma2/gamma

Summing |<virt|V|A>|^2 / dE gives the diagonal shift, so the COMPENSATION to
add to the longitudinal field is, in units of 1/gamma,

    theta_1(A) = (1/2) [ d_A^2 + sum_{B != A} d_B^2 / (1 + c_AB) ]                (1)

plus, in the first-quantization encoding, the inter-register two-body term

    phi(A,B) = (1/2) d_B^2 [ 1/(1 + c_AB + gamma2/gamma) - 1/(1 + c_AB) ]         (2)

for the *ordered* pair (register at A, other register at B).  Equation (2) is
the bug fix: with it the compensation is exact at second order for any N, and
gamma2 = gamma becomes usable (best hardcore protection), instead of having to
push gamma2 small.  Note phi is NOT symmetric (d_B^2 and c_AB); the physical ZZ
coupler between the two registers gets the symmetrized phi(A,B) + phi(B,A).

Identity worth remembering (checked on O18):

    theta_1(A) - theta_uniform = -(1/d_A) sum_C d_C (g_AC + d_A d_C)

i.e. it is proportional to the gradient of the rank-1 loss, so *at the rank-1
optimum theta_1 is already uniform* even for c != 0.  The residual
non-uniformity in practice comes only from the |c| > cap regularization.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

SelfEnergyMode = Literal["none", "uniform", "exact", "full"]


def theta_uniform(d: np.ndarray, n_particles: int = 1) -> float:
    """The notebooks' uniform shift: 0.5 * sum_B d_B^2 / N_tot (units 1/gamma)."""
    return 0.5 * float(np.sum(d**2)) / n_particles


def theta_one_body(d: np.ndarray, c_matrix: np.ndarray) -> np.ndarray:
    """Eq. (1): exact per-level one-body compensation, shape (D,), units 1/gamma."""
    d2 = d**2
    denom = 1.0 + c_matrix
    D = d.shape[0]
    theta = np.zeros(D)
    for A in range(D):
        acc = d2[A]
        for B in range(D):
            if B == A:
                continue
            acc += d2[B] / denom[A, B]
        theta[A] = 0.5 * acc
    return theta


def phi_two_body(
    d: np.ndarray, c_matrix: np.ndarray, gamma2_ratio: float
) -> np.ndarray:
    """Eq. (2): phi[A, B], units 1/gamma, zero diagonal. NOT symmetric.

    `gamma2_ratio = gamma2 / gamma`.  With c = 0 all rows are equal and the
    correction collapses to a one-body field (times N-1); with c != 0 genuine
    inter-register ZZ terms are needed.
    """
    d2 = d**2
    D = d.shape[0]
    phi = np.zeros((D, D))
    for A in range(D):
        for B in range(D):
            if A == B:
                continue
            c = c_matrix[A, B]
            phi[A, B] = 0.5 * d2[B] * (1.0 / (1.0 + c + gamma2_ratio) - 1.0 / (1.0 + c))
    return phi


def phi_symmetrized(
    d: np.ndarray, c_matrix: np.ndarray, gamma2_ratio: float
) -> np.ndarray:
    """phi(A,B) + phi(B,A): the coupling carried by one inter-register ZZ coupler."""
    phi = phi_two_body(d, c_matrix, gamma2_ratio)
    return phi + phi.T


def gamma2_lower_bound(d: np.ndarray, gamma: float, safety: float = 10.0) -> float:
    """Practical floor for gamma2.

    Leakage through the collision channel scales as (omega^2/gamma)^2 / gamma2,
    so gamma2 must stay well above omega^2/gamma with omega ~ max|d|.
    """
    omega = float(np.max(np.abs(d)))
    return safety * omega**2 / gamma


def self_energy_report(d: np.ndarray, c_matrix: np.ndarray, gamma2_ratio: float = 1.0):
    """Print theta_1, its spread, and the size of the phi correction."""
    theta = theta_one_body(d, c_matrix)
    phi = phi_two_body(d, c_matrix, gamma2_ratio)
    print(f"theta_uniform          : {theta_uniform(d):.6f}")
    print(f"theta_1(A)             : {np.round(theta, 6)}")
    print(
        f"  spread (max-min)     : {theta.max() - theta.min():.3e}"
        "   (0 at the exact rank-1 optimum)"
    )
    print(
        f"phi (gamma2/gamma={gamma2_ratio}) : "
        f"min {phi.min():.4f}  max {phi.max():.4f}"
    )
    return theta, phi
