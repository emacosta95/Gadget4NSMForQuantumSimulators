import numpy as np

def computational_basis(n):
    """
    Returns matrix of shape (2**n, n).
    Row i is the binary representation of i as a length-n bit string.
    """
    N = 2**n
    basis = np.zeros((N, n), dtype=int)
    for i in range(N):
        basis[i] = np.array(list(np.binary_repr(i, width=n)), dtype=int)
    return basis