"""
He-6 adiabatic state preparation + comparison with exact QuTiP ground state.
Uses bloqade-analog (pip install bloqade-analog).

Correct Rydberg mapping derived from expanding Z_i in terms of n_i:
    Z_i = 1 - 2*n_i
    Z_i*Z_j = 1 - 2*n_i - 2*n_j + 4*n_i*n_j

So the notebook Hamiltonian:
    H_nb = 2*gamma * sum_{i<j} Z_i*Z_j + sum_i h_i*Z_i + gamma*ntot^2 + sum_i d_i/sqrt(2)*X_i

Expands to (dropping constants):
    H_nb = 8*gamma * sum_{i<j} n_i*n_j
         - sum_i (8*gamma*(n_sites-1) + 2*h_i) * n_i    [n_sites-1 pairs per site]
         + sum_i d_i/sqrt(2) * X_i
         + const

Matching with H_Ryd = J * sum_{i<j} n_i*n_j - sum_i Delta_i*n_i + sum_i Omega_i/2*X_i:
    J       = 8 * gamma                       (NOT 2*gamma)
    Delta_i = 8*gamma*(n_sites-1) + 2*h_i    (NOT -2*h_i)
    Omega_i = |d_i| * sqrt(2)                (phase handles sign of d_i)
"""

from cmath import pi

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import qutip as qt

from ManyBodyQutip.qutip_class import SpinOperator
from src.utils import computational_basis

from bloqade.analog import start
from bloqade.analog.emulate.ir.state_vector import AnalogGate, StateVector
from bloqade.analog.emulate.codegen.hamiltonian import RydbergHamiltonianCodeGen

# ═══════════════════════════════════════════════════════════════════════
# 1. NOTEBOOK PARAMETERS
# ═══════════════════════════════════════════════════════════════════════

n_qubits = 3
d_opt = np.array([-1.02132267, 1.02132267, -3.6852787])
diagonal_elements = np.array([-8.4321, -8.4321, -5.1203])
external_field = np.array([-18.4321 / 2.0, 0.0, 0.0])

r_spacing = 5.0  # µm
C6 = 862690.0  # rad/us * um^6

gamma_sim = 1 * C6 / (r_spacing**6)

ntot = 1
J_scale = 1.0  # 1 notebook unit = 1 rad/µs

# ─── Longitudinal fields ─────────────────────────────────────────────
eff_long = (diagonal_elements + 0.5 * np.sum(d_opt**2)) / gamma_sim
h_target = eff_long + gamma_sim * (1 - 2 * ntot)
h_driver = external_field / gamma_sim + gamma_sim * (1 - 2 * ntot)

# ═══════════════════════════════════════════════════════════════════════
# 2. EXACT GROUND STATE  (QuTiP)
# ═══════════════════════════════════════════════════════════════════════

basis = computational_basis(n_qubits)

H_zz = sum(
    SpinOperator(
        [("qz", i, "qz", j)], coupling=[2 * gamma_sim], size=n_qubits, verbose=0
    ).qutip_op
    for i in range(n_qubits)
    for j in range(i + 1, n_qubits)
)
H_z = sum(
    SpinOperator([("qz", i)], coupling=[h_target[i]], size=n_qubits, verbose=0).qutip_op
    for i in range(n_qubits)
)
H_x = sum(
    SpinOperator(
        [("x", i)], coupling=[d_opt[i] / np.sqrt(2)], size=n_qubits, verbose=0
    ).qutip_op
    for i in range(n_qubits)
)
Id = qt.tensor([qt.qeye(2)] * n_qubits)
H_total = H_zz + H_z + gamma_sim * ntot**2 * Id + H_x

evals, evecs = H_total.eigenstates()
gs_energy = evals[0]
psi_gs_vec = evecs[0].full().flatten()

print(f"=== Exact ground state (gamma_sim={gamma_sim}) ===")
print(f"Energy: {gs_energy:.6f}")
print("Probabilities:")
for idx, b in enumerate(basis):
    p = abs(psi_gs_vec[idx]) ** 2
    if p > 1e-3:
        print(f"  |{''.join(map(str,b))}> : {p:.4f}")

exact_probs = {
    "".join(map(str, b)): abs(psi_gs_vec[idx]) ** 2 for idx, b in enumerate(basis)
}

# ═══════════════════════════════════════════════════════════════════════
# 3. CORRECT RYDBERG MAPPING
#
# Expanding Z_i = 1 - 2*n_i in the notebook Hamiltonian gives:
#
#   J_ryd     = 8 * gamma_sim
#   Delta_i   = 8 * gamma_sim * (n_qubits - 1) + 2 * h_i
#   Omega_i   = |d_i| * sqrt(2)   (phase handles sign)
#   + const   (irrelevant for dynamics)
# ═══════════════════════════════════════════════════════════════════════

J_ryd = gamma_sim * J_scale  # rad/µs
Delta_target = (h_target) * J_scale
Delta_driver = (h_driver) * J_scale

print(f"\nJ_ryd (n_i n_j coupling) = {J_ryd:.3f} rad/µs")
print(f"Delta_target per site    = {np.round(Delta_target, 4)}")
print(f"Delta_driver per site    = {np.round(Delta_driver, 4)}")

# Sanity check: diagonal energies in Rydberg frame should match notebook
print("\nSanity check — diagonal energies:")
for idx, b in enumerate(basis):
    n_occ = list(b)
    e_ryd = 2 * J_ryd * sum(
        n_occ[i] * n_occ[j] for i in range(n_qubits) for j in range(i + 1, n_qubits)
    ) + sum(Delta_target[i] * n_occ[i] for i in range(n_qubits))
    # constant shift (doesn't affect eigenstates)
    if sum(b) == 1:  # 1-particle sector
        print(f"  |{''.join(map(str,b))}>  E_diag(Ryd) = {e_ryd:.4f}")

# ─── Geometry: equilateral triangle ──────────────────────────────────
# J_ryd = C6 / r^6  =>  r = (C6 / J_ryd)^(1/6)
# # C6_rad_us = 862690.0
# # r_spacing = (C6_rad_us / J_ryd) ** (1.0 / 6.0)

positions = [
    (0.0, 0.0),
    (r_spacing, 0.0),
    (r_spacing / 2.0, r_spacing * np.sqrt(3) / 2.0),
]
print(f"\nAtom spacing = {r_spacing:.3f} µm")

# ─── Rabi amplitude ───────────────────────────────────────────────────
Omega = np.abs(d_opt) * np.sqrt(2) * J_scale  # rad/µs per site
Omega_max = float(np.max(Omega))
rabi_scales = list(Omega / Omega_max)
print(f"Omega per site = {np.round(Omega,4)}")
print(f"Omega_max      = {Omega_max:.4f} rad/µs")
print(f"J_ryd/Omega    = {J_ryd/Omega_max:.2f}")

# ─── Rabi phase: pi where d_i < 0 ────────────────────────────────────
rabi_phases = [np.pi if d < 0 else 0.0 for d in d_opt]
print(f"Rabi phases    = {['pi' if p > 0 else '0' for p in rabi_phases]}")

# No decomposition needed — each site gets its own waveform directly

# ─── Adiabatic schedule ───────────────────────────────────────────────
T_total = 10.0 / J_scale
t_prep = 0.2
print(f"Total time = {T_total:.2f} µs")

# ═══════════════════════════════════════════════════════════════════════
# 4. BUILD BLOQADE PROGRAM
# ═══════════════════════════════════════════════════════════════════════


def pwl(v_start, v_end):
    return [v_start, v_start, v_end, v_end]


program = (
    start.add_position(positions[0])
    .add_position(positions[1])
    .add_position(positions[2])
    # ─────────────────────────────
    # MAIN ADIABATIC SCHEDULE
    # ─────────────────────────────
    .rydberg.rabi.amplitude.location(0)
    .piecewise_linear([T_total], [0.0, float(Omega[0])])
    .location(1)
    .piecewise_linear([T_total], [0.0, float(Omega[1])])
    .location(2)
    .piecewise_linear([T_total], [0.0, float(Omega[2])])
    .rydberg.rabi.phase.location(0)
    .constant(duration=T_total, value=rabi_phases[0])
    .location(1)
    .constant(duration=T_total, value=rabi_phases[1])
    .location(2)
    .constant(duration=T_total, value=rabi_phases[2])
    .detuning.location(0)
    .piecewise_linear([T_total], [Delta_driver[0], Delta_target[0]])
    .location(1)
    .piecewise_linear([T_total], [Delta_driver[1], Delta_target[1]])
    .location(2)
    .piecewise_linear([T_total], [Delta_driver[2], Delta_target[2]])
)


results = program.bloqade.python().run(1000)


report = results.report()
print(report.counts())
# ═══════════════════════════════════════════════════════════════════════
# 5. STATEVECTOR FIDELITY
# ═══════════════════════════════════════════════════════════════════════

# print("\nRunning emulator (lsoda) ...")
# sv_result = program.bloqade.python().run(1)
# task0 = list(sv_result.tasks.values())[0]
# hamiltonian = RydbergHamiltonianCodeGen(task0.compile_cache).emit(task0.emulator_ir)
# # Initial state: |100> = site 0 Rydberg, sites 1 and 2 ground
# # fock_state_to_index takes a string 'r'=Rydberg, 'g'=ground, one char per site
# # This matches the notebook: initial state is basis[4] = |1,0,0>
# init_data = np.zeros(hamiltonian.space.size, dtype=complex)
# init_data[hamiltonian.space.fock_state_to_index("rgg")] = 1.0
# init_state = StateVector(data=init_data, space=hamiltonian.space)

# gen = AnalogGate(hamiltonian)._apply(
#     init_state,
# )
# final_state = None
# for s in gen:
#     final_state = s

# psi_bloq = np.array(final_state.data)

# # Map from blockade subspace to full basis if needed
# if hamiltonian.space.size < 2**n_qubits:
#     psi_full = np.zeros(2**n_qubits, dtype=complex)
#     for amp, cfg in zip(psi_bloq, hamiltonian.space.configurations):
#         psi_full[cfg] = amp
#     psi_bloq = psi_full

# fidelity = abs(np.dot(psi_gs_vec.conj(), psi_bloq)) ** 2
# print(f"\n=== Statevector Fidelity ===")
# print(f"F = {fidelity:.6f}")

# print("\nBloqade state probabilities:")
# for idx, b in enumerate(basis):
#     p = abs(psi_bloq[idx]) ** 2
#     if p > 1e-3:
#         print(f"  |{''.join(map(str,b))}> : {p:.4f}")

# # ═══════════════════════════════════════════════════════════════════════
# # 6. SHOT HISTOGRAM
# # Sample directly from the statevector already computed above.
# # This guarantees shots and fidelity come from the same simulation,
# # unlike calling program.bloqade.python().run(n_shots) again which
# # uses dop853 internally and gives a different (inconsistent) result.
# # ═══════════════════════════════════════════════════════════════════════

# n_shots = 2000

# # sample() returns an array of shape (n_shots, n_qubits) with 0/1 entries
# # where 1 = Rydberg (excited) state
# raw_samples = final_state.sample(n_shots)  # shape (n_shots, n_qubits)

# counts_ryd = {}
# for shot in raw_samples:
#     bs = "".join(str(b) for b in shot)
#     counts_ryd[bs] = counts_ryd.get(bs, 0) + 1

# meas_probs = {bs: c / n_shots for bs, c in counts_ryd.items()}
# all_bs = sorted(set(list(exact_probs) + list(meas_probs)))

# print(f"\n=== Bitstring Probabilities ({n_shots} shots) ===")
# print(f"{'Bitstring':>12}  {'Exact':>8}  {'Measured':>10}  {'Diff':>8}")
# print("-" * 46)
# for bs in all_bs:
#     ep = exact_probs.get(bs, 0.0)
#     mp = meas_probs.get(bs, 0.0)
#     if ep > 1e-3 or mp > 5e-3:
#         print(f"  |{bs}>      {ep:.4f}    {mp:.4f}   {mp-ep:+.4f}")

# tvd = 0.5 * sum(
#     abs(meas_probs.get(bs, 0.0) - exact_probs.get(bs, 0.0)) for bs in all_bs
# )
# print(f"\nTVD      = {tvd:.4f}")
# print(f"Fidelity = {fidelity:.4f}")

# # ═══════════════════════════════════════════════════════════════════════
# # 7. PLOT
# # ═══════════════════════════════════════════════════════════════════════

# fig, ax = plt.subplots(figsize=(9, 4))
# x = np.arange(len(all_bs))
# ax.bar(
#     x - 0.2,
#     [exact_probs.get(bs, 0.0) for bs in all_bs],
#     width=0.4,
#     label=r"Exact $|\langle b|\psi_{gs}\rangle|^2$",
#     color="steelblue",
# )
# ax.bar(
#     x + 0.2,
#     [meas_probs.get(bs, 0.0) for bs in all_bs],
#     width=0.4,
#     label=f"Bloqade ({n_shots} shots)",
#     color="tomato",
#     alpha=0.85,
# )
# ax.set_xticks(x)
# ax.set_xticklabels([f"|{bs}>" for bs in all_bs], fontsize=10)
# ax.set_ylabel("Probability")
# ax.set_title(
#     rf"He-6  ($\gamma_{{sim}}={gamma_sim}$)  —  Fidelity={fidelity:.3f},  TVD={tvd:.3f}"
# )
# ax.legend()
# ax.set_ylim(0, 1)
# plt.tight_layout()
# plt.savefig("he6_comparison.png", dpi=150)
# print("Plot saved to he6_comparison.png")
