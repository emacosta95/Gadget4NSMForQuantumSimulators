import numpy as np
from scipy.optimize import minimize


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


