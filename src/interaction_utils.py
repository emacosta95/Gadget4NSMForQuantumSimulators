import numpy as np
from scipy.optimize import minimize


class EffectiveInteractionOptimizerTunableSelfEnergy:
    """
    Optimization for one-hot gadget drive parameters with distance-dependent coupling.

    The effective coupling is:
        g_ij ≈ -d_i * d_j / (1 + r_ij)

    where d_i are drive amplitudes and r_ij are distance parameters (r_ij >= 0).

    Parameters
    ----------
    nqubit : int
        Number of logical qubits.
    n_restarts : int
        Number of random restarts for global optimization.
    scale : float
        Scale of random initialization.
    ftol : float
        Tolerance for L-BFGS-B optimizer.
    gtol : float
        Gradient tolerance for L-BFGS-B optimizer.
    r_max : float or None
        Global upper bound on all r_ij values. None means no upper bound.
    """

    def __init__(self, nqubit, n_restarts=2000, scale=2.0, ftol=1e-15, gtol=1e-10,
                 r_max=None):
        self.n = nqubit
        self.n_restarts = n_restarts
        self.scale = scale
        self.ftol = ftol
        self.gtol = gtol
        self.r_max = r_max
        self.mask = np.triu(np.ones((nqubit, nqubit)), k=1)

        self.pairs = [(i, j) for i in range(nqubit) for j in range(i+1, nqubit)]
        self.n_pairs = len(self.pairs)
        self.r_bound_list = [(0, r_max)] * self.n_pairs

    # ------------------------------------------------------------------
    # Unpack / reshape
    # ------------------------------------------------------------------

    def _unpack(self, params):
        d = params[:self.n]
        r_vec = params[self.n:]
        r = np.zeros((self.n, self.n))
        for k, (i, j) in enumerate(self.pairs):
            r[i, j] = r_vec[k]
            r[j, i] = r_vec[k]
        return d, r

    def get_d(self, params):
        """Return drive amplitudes d of shape (n,)."""
        return params[:self.n].copy()

    def get_r_matrix(self, params):
        """Return symmetric distance matrix r of shape (n, n), zero diagonal."""
        _, r = self._unpack(params)
        return r

    # ------------------------------------------------------------------
    # Coupling matrix
    # ------------------------------------------------------------------

    def reconstructed(self, params):
        """Compute -d_i * d_j / (1 + r_ij) matrix (off-diagonal only)."""
        d, r = self._unpack(params)
        mat = -np.outer(d, d) / (1 + r)
        np.fill_diagonal(mat, 0)
        return mat

    # ------------------------------------------------------------------
    # Objective and gradient
    # ------------------------------------------------------------------

    def objective(self, params, g_matrix):
        diff = self.reconstructed(params) - g_matrix
        return np.sum((diff * self.mask) ** 2)

    def gradient(self, params, g_matrix):
        d, r = self._unpack(params)
        denom = 1 + r
        mat = -np.outer(d, d) / denom
        np.fill_diagonal(mat, 0)

        diff = (mat - g_matrix) * self.mask
        diff_sym = diff + diff.T

        grad_d = np.array([
            np.sum(diff_sym[k] * (-d / denom[k]))
            for k in range(self.n)
        ])

        grad_r = np.array([
            diff_sym[i, j] * d[i] * d[j] / denom[i, j] ** 2
            for i, j in self.pairs
        ])

        return np.concatenate([grad_d, grad_r])

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def _warm_start(self):
        d0 = np.random.randn(self.n) * self.scale
        hi = self.r_max if self.r_max is not None else self.scale
        r0 = np.random.uniform(0, hi, self.n_pairs)
        return np.concatenate([d0, r0])

    def optimize(self, g_matrix):
        """
        Find d and r_ij minimizing sum_{i<j} (-d_i*d_j/(1+r_ij) - g_ij)^2.

        Parameters
        ----------
        g_matrix : ndarray, shape (n, n)

        Returns
        -------
        params_opt : ndarray  [d_0,...,d_{n-1}, r_01, r_02, ...]
        result : OptimizeResult
        """
        bounds = [(None, None)] * self.n + self.r_bound_list

        opt_kwargs = dict(
            jac=self.gradient,
            args=(g_matrix,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': self.ftol, 'gtol': self.gtol, 'maxiter': 10000}
        )

        best = None
        for _ in range(self.n_restarts):
            p0 = self._warm_start()
            res = minimize(self.objective, p0, **opt_kwargs)
            if best is None or res.fun < best.fun:
                best = res

        return best.x, best

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self, g_matrix, params_opt):
        d = self.get_d(params_opt)
        r = self.get_r_matrix(params_opt)
        approx = self.reconstructed(params_opt)

        print("Target G:")
        print(g_matrix.round(4))
        print("\nReconstructed -d_i*d_j/(1+r_ij):")
        print(approx.round(4))
        print("\nOptimal drives d:")
        print(d.round(6))
        print("\nOptimal distance matrix r_ij:")
        print(r.round(6))
        print("\nPer-pair errors:")
        for i in range(self.n):
            for j in range(i + 1, self.n):
                t = g_matrix[i, j]
                a = approx[i, j]
                print(f"  g_{i}{j}: target={t:+.4f}  approx={a:+.4f}  "
                      f"error={abs(a - t):.2e}")

        eigvals = np.linalg.eigvalsh(g_matrix)
        print(f"\nEigenvalues of G: {eigvals.round(4)}")
        print(f"Rank of G: {np.linalg.matrix_rank(g_matrix, tol=1e-10)}")
        flux = g_matrix[0, 1] * g_matrix[0, 2] * g_matrix[1, 2]
        print(f"Flux g_01*g_02*g_12 = {flux:.4f} "
              f"({'pi-flux' if flux < 0 else 'zero-flux'})")


class EffectiveInteractionOptimizer:
    """
    Rank-1 and rank-r optimization for one-hot gadget drive parameters.

    Parameters
    ----------
    nqubit : int
        Number of logical qubits.
    n_restarts : int
        Number of random restarts for global optimization.
    scale : float
        Scale of random initialization.
    ftol : float
        Tolerance for L-BFGS-B optimizer.
    gtol : float
        Gradient tolerance for L-BFGS-B optimizer.
    """

    def __init__(self, nqubit, n_restarts=2000, scale=2.0, ftol=1e-15, gtol=1e-10):
        self.n = nqubit
        self.n_restarts = n_restarts
        self.scale = scale
        self.ftol = ftol
        self.gtol = gtol
        self.mask = np.triu(np.ones((nqubit, nqubit)), k=1)

    def reconstructed(self, d):
        outer = -np.outer(d, d)
        np.fill_diagonal(outer, 0)
        return outer

    def objective(self, d, g_matrix):
        diff = self.reconstructed(d) - g_matrix
        return np.sum((diff * self.mask) ** 2)

    def gradient(self, d, g_matrix):
        diff = self.reconstructed(d) - g_matrix
        diff_sym = (self.mask + self.mask.T) * diff
        return -2 * diff_sym @ d

    def optimize_rank1(self, g_matrix):
        eigvals, eigvecs = np.linalg.eigh(-g_matrix)
        idx = np.argmax(np.abs(eigvals))
        d0 = np.sqrt(max(abs(eigvals[idx]), 1e-8)) * eigvecs[:, idx]

        opt_kwargs = dict(
            jac=self.gradient,
            args=(g_matrix,),
            method='L-BFGS-B',
            options={'ftol': self.ftol, 'gtol': self.gtol, 'maxiter': 10000}
        )

        best = minimize(self.objective, d0, **opt_kwargs)
        for _ in range(self.n_restarts):
            d_init = np.random.randn(self.n) * self.scale
            res = minimize(self.objective, d_init, **opt_kwargs)
            if res.fun < best.fun:
                best = res

        return best.x, best

    def diagnostics(self, g_matrix, d_opt):
        d = np.atleast_2d(d_opt)
        approx = sum(-np.outer(d[k], d[k]) for k in range(len(d)))
        np.fill_diagonal(approx, 0)

        print("Target G:")
        print(g_matrix)
        print("\nReconstructed -d_i*d_j:")
        print(approx.round(4))
        print("\nPer-pair errors:")
        for i in range(self.n):
            for j in range(i + 1, self.n):
                t = g_matrix[i, j]
                a = approx[i, j]
                print(f"  g_{i}{j}: target={t:+.4f}  approx={a:+.4f}  "
                      f"error={abs(a - t):.2e}")

        eigvals = np.linalg.eigvalsh(g_matrix)
        print(f"\nEigenvalues of G: {eigvals.round(4)}")
        print(f"Rank of G: {np.linalg.matrix_rank(g_matrix, tol=1e-10)}")
        flux = g_matrix[0, 1] * g_matrix[0, 2] * g_matrix[1, 2]
        print(f"Flux g_01*g_02*g_12 = {flux:.4f} "
              f"({'pi-flux' if flux < 0 else 'zero-flux'})")