import numpy as np
import itertools
from itertools import combinations
import qutip as qt


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

def generate_particleconservation_basis(size_a:int,size_b:int,nparticles_a:int,nparticles_b:int):
    """Generate the basis of the number sector of the quasiparticle space

    Args:
        size_a (int): quasiparticle states particle a
        size_b (int): quasiparticle states particle b
        nparticles_a (int): number of particles a
        nparticles_b (int): number of particles b

    Returns:
        basis(np.ndarray): basis as numpy array 
    """
    combinations_list = []
    #print(combinations(range(self.nparticles_a), self.size_a))
    for indices_part1 in list(combinations(range(size_a), nparticles_a)):
        for indices_part2 in list(
            combinations(range(size_b), nparticles_b)
        ):
            base = [0] * (size_a + size_b)
            for idx in indices_part1:
                base[idx] = 1
            for idx in indices_part2:
                # because the second subsystem is related to the other species
                base[idx + size_a] = 1
            combinations_list.append(base)
            
    basis=np.asarray(combinations_list)
    
    return basis


def array_to_qutip(vec, n_qubits):
    """
    Convert a numpy array of shape (2**n_qubits,) into a QuTiP ket
    with tensor product structure [2, 2, ..., 2].
    """
    dims = [[2] * n_qubits, [1] * n_qubits]
    return qt.Qobj(vec.reshape(-1, 1), dims=dims)