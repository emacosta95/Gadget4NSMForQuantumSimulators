import numpy as np
import itertools
from itertools import combinations
import qutip as qt
from ManyBodyQutip.qutip_class import SpinOperator, SpinHamiltonian
from scipy.sparse import diags


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




def build_total_hamiltonian(n_qubits, d_opt, gamma, ntot,
                             links, 
                             coupling_dict=None):
    """
    Build the total Hamiltonian = longitudinal + transverse.

    Parameters
    ----------
    n_qubits : int
    d_opt : ndarray, shape (n_qubits,)
        Optimal drive amplitudes.
    gamma : float
        Constraint strength.
    ntot : int
        Target particle number.
    links : ndarray, shape (n_qubits,)
        Link weights for effective longitudinal field.
    SpinOperator : class
        SpinOperator class for building operators.
    coupling_dict : dict or None
        Additional ZZ couplings as {(i,j): value} in units of gamma.
        e.g. {(0,2): -2.2, (1,3): -1.5}
        Both (i,j) and (j,i) keys are accepted.
        If None, no additional couplings are added.

    Returns
    -------
    total_hamiltonian : qutip.Qobj
    longitudinal_hamiltonian : qutip.Qobj
    transverse_hamiltonian : qutip.Qobj
    """
    if coupling_dict is None:
        coupling_dict = {}

    # ------------------------------------------------------------------
    # Longitudinal Hamiltonian
    # ------------------------------------------------------------------

    # ZZ terms: constraint + additional couplings from dict
    hamiltonian_zz = 0.
    for i in range(n_qubits):
        for j in range(i+1, n_qubits):
            # constraint term for all pairs
            hamiltonian_zz += SpinOperator(
                [('qz', i, 'qz', j)],
                coupling=[2*gamma],
                size=n_qubits, verbose=0).qutip_op

            # check both (i,j) and (j,i) in dict
            key = (i, j) if (i, j) in coupling_dict else \
                  (j, i) if (j, i) in coupling_dict else None
            if key is not None:
                hamiltonian_zz += SpinOperator(
                    [('qz', i, 'qz', j)],
                    coupling=[coupling_dict[key] * gamma],
                    size=n_qubits, verbose=0).qutip_op

    # effective longitudinal field
    effective_longitudinal_field = links * (d_opt[0]**2) * (1/gamma)

    # linear Z terms
    hamiltonian_z = 0.
    for i in range(n_qubits):
        hamiltonian_z += SpinOperator(
            [('qz', i)],
            coupling=[effective_longitudinal_field[i] + gamma*(1 - 2*ntot)],
            size=n_qubits, verbose=0).qutip_op

    # identity shift
    identity_qubit_space = qt.tensor([qt.qeye(2)] * n_qubits)

    longitudinal_hamiltonian = (hamiltonian_zz + hamiltonian_z
                                + gamma*(ntot**2)*identity_qubit_space)

    # ------------------------------------------------------------------
    # Transverse Hamiltonian
    # ------------------------------------------------------------------

    transverse_hamiltonian = 0.
    for i in range(n_qubits):
        transverse_hamiltonian += SpinOperator(
            [('x', i)],
            coupling=[d_opt[i] / np.sqrt(2)],
            size=n_qubits, verbose=0).qutip_op

    # ------------------------------------------------------------------
    # Total
    # ------------------------------------------------------------------

    total_hamiltonian = longitudinal_hamiltonian + transverse_hamiltonian

    return total_hamiltonian, longitudinal_hamiltonian, transverse_hamiltonian

# --- Example usage ---
# if __name__ == "__main__":

#     gamma = 100
#     ntot = 1
#     n_qubits = 5

#     d_opt = np.ones(n_qubits)   # placeholder
#     links = np.array([0., 0.83, 0.83, 0.83, 0.])

#     total_H, long_H, trans_H = build_total_hamiltonian(
#         n_qubits=n_qubits,
#         d_opt=d_opt,
#         gamma=gamma,
#         ntot=ntot,
#         links=links,
#         SpinOperator=SpinOperator,
#     )

#     print("Total Hamiltonian shape:", total_H.shape)
#     print("Eigenvalues (lowest 6):", total_H.eigenenergies()[:6])


def build_effective_hamiltonian(total_hamiltonian, basis, gamma,
                                 low_energy_k=1,
                                 high_energy_k=None):
    """
    Build the effective Hamiltonian in the low-energy subspace
    via Brillouin-Wigner perturbation theory.
 
    H_eff = H_AA - H_AR * H_RR^{-1} * H_RA
 
    Parameters
    ----------
    total_hamiltonian : qutip.Qobj
        Full physical Hamiltonian.
    basis : ndarray, shape (2**n_qubits, n_qubits)
        Computational basis as bit strings.
    gamma : float
        Constraint strength (used for scaling output).
    low_energy_k : int
        Hamming weight of the low-energy (A) subspace. Default 1.
    high_energy_k : list or None
        Hamming weights of the high-energy (R) subspace.
        If None, uses k=0 and k=2.
 
    Returns
    -------
    new_hamiltonian : scipy sparse matrix
        Effective Hamiltonian in the low-energy subspace, shape (n_A, n_A).
    hamiltonian_aa : scipy sparse matrix
        Direct block H_AA.
    hamiltonian_delta : scipy sparse matrix
        Second-order correction H_AR * H_RR^{-1} * H_RA.
    idxs_single_spin : ndarray
        Indices of low-energy (A) subspace states.
    idxs_nn : ndarray
        Indices of high-energy (R) subspace states.
    """

 
    counts = basis.sum(axis=1)
 
    # low-energy subspace (A)
    idxs_single_spin = np.where(counts == low_energy_k)[0]
 
    # high-energy subspace (R)
    if high_energy_k is None:
        high_energy_k = [0, 2]
    mask = np.zeros(len(counts), dtype=bool)
    for k in high_energy_k:
        mask |= (counts == k)
    idxs_nn = np.where(mask)[0]
 
    # extract blocks
    H = total_hamiltonian.data.as_scipy()
 
    hamiltonian_ra = H[np.ix_(idxs_nn, idxs_single_spin)]
    hamiltonian_ar = H[np.ix_(idxs_single_spin, idxs_nn)]
    hamiltonian_rr = H[np.ix_(idxs_nn, idxs_nn)]
    hamiltonian_aa = H[np.ix_(idxs_single_spin, idxs_single_spin)]
 
    # invert H_RR (diagonal approximation)
    diag = hamiltonian_rr.diagonal()
    hamiltonian_rr_inv = diags(1.0 / diag)
 
    # second-order correction
    hamiltonian_delta = hamiltonian_ar @ hamiltonian_rr_inv @ hamiltonian_ra
 
    # effective Hamiltonian
    new_hamiltonian = hamiltonian_aa - hamiltonian_delta
 
    print("H_AA block:")
    print(hamiltonian_aa)
    print("\nH_RR diagonal:", hamiltonian_rr.diagonal())
    print("\nSecond-order correction (in units of 1/gamma):")
    print(hamiltonian_delta * gamma)
 
    return (new_hamiltonian, hamiltonian_aa, hamiltonian_delta,
            idxs_single_spin, idxs_nn)