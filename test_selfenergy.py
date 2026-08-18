import matplotlib
matplotlib.use('Agg')
import numpy as np
np.random.seed(0)
import numpy as np
import json
import matplotlib.pyplot as plt
from src.interaction_utils import (
    EffectiveInteractionOptimizer,
    EffectiveInteractionOptimizerTunableSelfEnergy,
    EffectiveInteractionOptimizer2ndVersion,
)
from ManyBodyQutip.qutip_class import SpinOperator, SpinHamiltonian
import qutip as qt
from src.utils import computational_basis
from qutip import fidelity
from NSMFermions.hamiltonian_utils import FermiHubbardHamiltonian
from NSMFermions.nuclear_physics_utils import (
    get_twobody_nuclearshell_model,
    SingleParticleState,
)
import numpy as np
import torch
from typing import Dict
import scipy
from NSMFermions.qml_models import AdaptVQEFermiHubbard
from NSMFermions.qml_utils.train import Fit
from NSMFermions.qml_utils.utils import configuration
from scipy.sparse.linalg import eigsh, expm_multiply
from tqdm import trange
import matplotlib.pyplot as plt
from NSMFermions.utils_quasiparticle_approximation import (
    QuasiParticlesConverter,
    HardcoreBosonsBasis,
    QuasiParticlesConverterOnlynnpp,
)
from src.utils import (
    generate_particleconservation_basis,
    array_to_qutip,
    build_total_hamiltonian,
    build_effective_hamiltonian,
    compute_particle_number,
)

data_onebody = np.load("data/matrix_elements_h_eff_2body/one_body_nn_sd.npz")
keys = data_onebody["keys"]
values = data_onebody["values"]
n_qubits = 6

g_onebody = {}
diagonal_elements = np.zeros(n_qubits)
g_matrix = np.zeros((n_qubits, n_qubits))
for a, key in enumerate(keys):
    i, j = key
    g_onebody[(i, j)] = values[a]
    if i != j:
        g_matrix[i, j] = values[a]
    if i == j:
        diagonal_elements[i] = values[a]

data_twobody = np.load("data/matrix_elements_h_eff_2body/twobody_nn_sd.npz")
keys = data_twobody["keys"]
values = data_twobody["values"]
g_twobody = {}
for a, key in enumerate(keys):
    i, j, k, l = key
    g_twobody[(i, j)] = values[a]
    print(i, j, k, l, values[a])
    if i != k and j != l:
        print(i, j, k, l)
        print(values[a])

# get the computational basis of the space
basis = computational_basis(n_qubits)

nparticles_a = 2
nparticles_b = 0

particle_conserved_basis = generate_particleconservation_basis(
    size_a=n_qubits, size_b=0, nparticles_a=nparticles_a, nparticles_b=nparticles_b
)

print(particle_conserved_basis)

# initialize the class in the number sector of the quasiparticle space (see NSMFermion library)
HBB = HardcoreBosonsBasis(basis=particle_conserved_basis)
quasiparticle_hamiltonian_particle_conserved = 0.0
for key, value in g_onebody.items():
    idx_a, idx_b = key
    if idx_a != idx_b:
        quasiparticle_hamiltonian_particle_conserved += value * HBB.adag_a_matrix(
            idx_a, idx_b
        )
    else:
        quasiparticle_hamiltonian_particle_conserved += value * HBB.adag_a_matrix(
            idx_a, idx_b
        )

for key, value in g_twobody.items():
    idx_a, idx_b = key
    quasiparticle_hamiltonian_particle_conserved += value * HBB.adag_adag_a_a_matrix(
        idx_a, idx_b, idx_a, idx_b
    )

print(quasiparticle_hamiltonian_particle_conserved)
value, eigenstates_particle_conserved = np.linalg.eigh(
    quasiparticle_hamiltonian_particle_conserved.todense()
)

print(value)

hamiltonian_xy = 0.0
for i in range(n_qubits):
    for j in range(i + 1, n_qubits):
        hamiltonian_xy += SpinOperator(
            [("x", i, "x", j)],
            coupling=[0.5 * g_matrix[i, j]],
            size=n_qubits,
            verbose=1,
        ).qutip_op
        hamiltonian_xy += SpinOperator(
            [("y", i, "y", j)],
            coupling=[0.5 * g_matrix[i, j]],
            size=n_qubits,
            verbose=1,
        ).qutip_op
hamiltonian_z = 0.0
for i in range(n_qubits):
    for j in range(i + 1, n_qubits):
        hamiltonian_z += SpinOperator(
            [("qz", i, "qz", j)], coupling=[g_twobody[(i, j)]], size=n_qubits, verbose=1
        ).qutip_op

for i in range(n_qubits):
    hamiltonian_z += SpinOperator(
        [("qz", i)], coupling=[diagonal_elements[i]], size=n_qubits, verbose=1
    ).qutip_op

nsm_quasiparticle_hamiltonian = hamiltonian_z + hamiltonian_xy

eigenvalues_fullspace_nsm, eigenstates_fullspace_nsm = (
    nsm_quasiparticle_hamiltonian.eigenstates()
)

print(eigenvalues_fullspace_nsm)

OptimalFieldBe6 = EffectiveInteractionOptimizer(
    nqubit=n_qubits, n_restarts=100, scale=2.0, ftol=1e-15, gtol=1e-10
)

d_opt, result = OptimalFieldBe6.optimize_rank1(g_matrix)
print("Optimal drive parameters:", d_opt)
print(
    "Optimized effective interaction matrix:\n",
    OptimalFieldBe6.reconstructed(d_opt) - g_matrix,
)
plt.imshow(
    OptimalFieldBe6.reconstructed(d_opt) - g_matrix,
    cmap="bwr",
    vmin=-np.max(np.abs(g_matrix)),
    vmax=np.max(np.abs(g_matrix)),
)
for i in range(n_qubits):
    for j in range(n_qubits):
        plt.text(
            j,
            i,
            f"{OptimalFieldBe6.reconstructed(d_opt)[i,j]-g_matrix[i,j]:.2f}",
            ha="center",
            va="center",
            color="black",
        )
plt.colorbar()
plt.title("Optimized effective interaction matrix - original interaction matrix")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

vmax = np.max(np.abs(g_matrix))
matrices = [OptimalFieldBe6.reconstructed(d_opt), g_matrix]
titles = [
    "Optimized effective interaction matrix",
    "One-quasiparticle interaction matrix",
]

for ax, mat, title in zip(axes, matrices, titles):
    im = ax.imshow(mat, cmap="bwr", vmin=-vmax, vmax=vmax)
    for i in range(n_qubits):
        for j in range(n_qubits):
            ax.text(
                j,
                i,
                f"{mat[i,j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=15,
            )
    plt.colorbar(im, ax=ax)
    ax.set_title(title)

plt.tight_layout()
plt.show()

opt = EffectiveInteractionOptimizer2ndVersion(n_qubits)

# Stage 1 (uncomment to refit d from scratch instead of using a fixed d_star):
d_opt, res1 = opt.optimize_rank1(g_matrix)

# Stage 2: alpha_AB and c_AB on top of the fixed d_star
alpha_matrix, c_matrix, report, g_corrected = opt.get_alpha_and_c(g_matrix, d_opt)
opt.print_report(g_matrix, d_opt, alpha_matrix, c_matrix, report)

c_matrix[np.abs(c_matrix) > 4] = np.abs(c_matrix[np.abs(c_matrix) > 4])

plt.imshow(
    g_corrected - g_matrix,
    cmap="bwr",
    vmin=-np.max(np.abs(g_matrix)),
    vmax=np.max(np.abs(g_matrix)),
)
for i in range(n_qubits):
    for j in range(n_qubits):
        plt.text(
            j,
            i,
            f"{g_corrected[i,j]-g_matrix[i,j]:.2f}",
            ha="center",
            va="center",
            color="black",
        )
plt.colorbar()
plt.title("Optimized effective interaction matrix - original interaction matrix")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

vmax = np.max(np.abs(g_matrix))
matrices = [g_corrected, g_matrix]
titles = [
    "Optimized effective interaction matrix",
    "One-quasiparticle interaction matrix",
]

for ax, mat, title in zip(axes, matrices, titles):
    im = ax.imshow(mat, cmap="bwr", vmin=-vmax, vmax=vmax)
    for i in range(n_qubits):
        for j in range(n_qubits):
            ax.text(
                j,
                i,
                f"{mat[i,j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=15,
            )
    plt.colorbar(im, ax=ax)
    ax.set_title(title)

plt.tight_layout()
plt.show()

n_qubits = 12
# get the computational basis of the space
basis = computational_basis(n_qubits)

#### particle constraint
gamma = 100
ntot = 1  # here we consider the first quantization encoding

effective_longitudinal_field = (diagonal_elements) / (gamma)

# lets start with the Z_A Z_B of the first particle constraint
hamiltonian_zz = 0.0
for i in range(n_qubits // 2):
    for j in range(i + 1, n_qubits // 2):
        hamiltonian_zz += SpinOperator(
            [("qz", i, "qz", j)],
            coupling=[(2 + c_matrix[i, j]) * gamma],
            size=n_qubits,
            verbose=1,
        ).qutip_op
# then the second particle constraint
for i in range(n_qubits // 2, n_qubits):
    for j in range(i + 1, n_qubits):
        hamiltonian_zz += SpinOperator(
            [("qz", i, "qz", j)],
            coupling=[(2 + c_matrix[i - n_qubits // 2, j - n_qubits // 2]) * gamma],
            size=n_qubits,
            verbose=1,
        ).qutip_op

# then the linear terms
hamiltonian_z = 0.0
# we split the constrain in the first particle sector and the second particle sector
for i in range(n_qubits // 2):
    # we add \gamma (1-ntot) since it's the linear part of the particle number constrain
    hamiltonian_z += SpinOperator(
        [("qz", i)],
        coupling=[effective_longitudinal_field[i] + gamma * (1 - 2 * ntot)],
        size=n_qubits,
        verbose=1,
    ).qutip_op
for i in range(n_qubits // 2, n_qubits):

    hamiltonian_z += SpinOperator(
        [("qz", i)],
        coupling=[
            effective_longitudinal_field[i - n_qubits // 2] + gamma * (1 - 2 * ntot)
        ],
        size=n_qubits,
        verbose=1,
    ).qutip_op


# hardcore boson constrain -> it forces the state to avoid 2 particles in the same site
gamma2 = 10 * gamma
for i in range(n_qubits // 2):
    hamiltonian_z += SpinOperator(
        [("qz", i, "qz", i + n_qubits // 2)],
        coupling=[gamma2],
        size=n_qubits,
        verbose=1,
    ).qutip_op

# we need to add the two-body interaction removing the self energy contribution
for i in range(0, n_qubits // 2):
    for j in range(i + 1, n_qubits // 2):
        print(i, j)
        hamiltonian_zz += SpinOperator(
            [("qz", i, "qz", j + n_qubits // 2)],
            coupling=[(g_twobody[(i, j)]) / gamma],
            size=n_qubits,
            verbose=1,
        ).qutip_op
        hamiltonian_zz += SpinOperator(
            [("qz", i + n_qubits // 2, "qz", j)],
            coupling=[(g_twobody[(i, j)]) / gamma],
            size=n_qubits,
            verbose=1,
        ).qutip_op

        print(i, j, g_twobody[(i, j)])
        print(i, j, "\n")
# finally add the identity such that the single quasiparticle ground state (without effective terms) centers in zero
identity_qubit_space = qt.tensor([qt.qeye(2)] * n_qubits)

longitudinal_hamiltonian = (
    hamiltonian_zz + hamiltonian_z + 2 * gamma * (ntot**2) * identity_qubit_space
)

# the transverse field
transverse_hamiltonian = 0.0
for i in range(n_qubits // 2):
    # we add \gamma (1-ntot) since it's the linear part of the particle number constrain
    transverse_hamiltonian += SpinOperator(
        [("x", i)], coupling=[d_opt[i] / np.sqrt(2)], size=n_qubits, verbose=1
    ).qutip_op
for i in range(n_qubits // 2, n_qubits):
    # we add \gamma (1-ntot) since it's the linear part of the particle number constrain
    transverse_hamiltonian += SpinOperator(
        [("x", i)],
        coupling=[d_opt[i - n_qubits // 2] / np.sqrt(2)],
        size=n_qubits,
        verbose=1,
    ).qutip_op

print("\n\n========== self-energy (diagonal 2nd-order shift) check ==========")

gamma = 5000.0
ntot = 1
n1 = n_qubits // 2

effective_longitudinal_field = diagonal_elements / gamma

hamiltonian_zz = 0.0
for i in range(n1):
    for j in range(i + 1, n1):
        hamiltonian_zz += SpinOperator([("qz", i, "qz", j)], coupling=[2 * gamma], size=n_qubits, verbose=1).qutip_op
for i in range(n1, n_qubits):
    for j in range(i + 1, n_qubits):
        hamiltonian_zz += SpinOperator([("qz", i, "qz", j)], coupling=[2 * gamma], size=n_qubits, verbose=1).qutip_op

hamiltonian_z = 0.0
for i in range(n1):
    hamiltonian_z += SpinOperator([("qz", i)], coupling=[effective_longitudinal_field[i] + gamma * (1 - 2 * ntot)], size=n_qubits, verbose=1).qutip_op
for i in range(n1, n_qubits):
    hamiltonian_z += SpinOperator([("qz", i)], coupling=[effective_longitudinal_field[i - n1] + gamma * (1 - 2 * ntot)], size=n_qubits, verbose=1).qutip_op

gamma2 = gamma
for i in range(n1):
    hamiltonian_z += SpinOperator([("qz", i, "qz", i + n1)], coupling=[gamma2], size=n_qubits, verbose=1).qutip_op

longitudinal_bare = hamiltonian_zz + hamiltonian_z + 2 * gamma * (ntot**2) * identity_qubit_space
total_bare = longitudinal_bare + transverse_hamiltonian

H = total_bare.data.as_scipy().tocsr()
counts = basis.sum(axis=1)

def state_index(a_site, b_site):
    bstr = np.zeros(n_qubits, dtype=int)
    bstr[a_site] = 1
    bstr[b_site + n1] = 1
    return int(np.nonzero((basis == bstr).all(axis=1))[0][0])

idxs_A = np.array([state_index(a, b) for a in range(n1) for b in range(n1) if a != b])
idxs_R = np.where((counts == 1) | (counts == 3))[0]

H_AA = H[np.ix_(idxs_A, idxs_A)].toarray()
H_AR = H[np.ix_(idxs_A, idxs_R)].toarray()
H_RA = H[np.ix_(idxs_R, idxs_A)].toarray()
H_RR_diag = H[np.ix_(idxs_R, idxs_R)].diagonal()

H_eff = H_AA - (H_AR * (1.0 / H_RR_diag)[None, :]) @ H_RA
H_eff_scaled = H_eff * gamma

pairs = [(a, b) for a in range(n1) for b in range(n1) if a != b]
idx_of_pair = {p: k for k, p in enumerate(pairs)}

print(f"{'a':>3} {'b':>3} {'H_eff diag*gamma':>18} {'target diag_a+diag_b':>22} {'residual':>12} {'-0.5*(d_a^2+d_b^2)':>20}")
for (a, b) in [(0,2),(0,3),(1,3),(2,4),(4,5),(0,4),(2,5)]:
    k = idx_of_pair[(a,b)]
    eff_diag = H_eff_scaled[k, k].real
    target_diag = diagonal_elements[a] + diagonal_elements[b]
    residual = eff_diag - target_diag
    guess = -0.5 * (d_opt[a]**2 + d_opt[b]**2)
    print(f"{a:>3} {b:>3} {eff_diag:>18.5f} {target_diag:>22.5f} {residual:>12.5f} {guess:>20.5f}")
