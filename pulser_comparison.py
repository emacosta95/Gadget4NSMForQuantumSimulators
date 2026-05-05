import matplotlib

matplotlib.use("Agg")  # fix Qt/display crash

import numpy as np
from scipy.sparse.linalg import eigsh, expm_multiply
from scipy.sparse import identity, kron, csr_matrix
from pulser import MockDevice, AnalogDevice, Register, Sequence, Pulse
from pulser.waveforms import RampWaveform, ConstantWaveform
from pulser_simulation import QutipEmulator
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Parameters ────────────────────────────────────────────────────────
gamma = 5.0
ntot = 1
n_qubits = 3
d_opt = np.array([-1.02132267, 1.02132267, -3.6852787])
diagonal_elements = np.array([-8.4321, -8.4321, -5.1203])
external_field = np.array([-18.4321 / 2, 0.0, 0.0])


# ── Operators ─────────────────────────────────────────────────────────
def make_ops(n):
    I2 = identity(2, format="csr", dtype=complex)
    X2 = csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
    N2 = csr_matrix(np.array([[0, 0], [0, 1]], dtype=complex))  # n_i = |1><1| = qz
    Xs, Ns = [], []
    for i in range(n):
        ox = identity(1, format="csr", dtype=complex)
        on = identity(1, format="csr", dtype=complex)
        for j in range(n):
            ox = kron(ox, X2 if j == i else I2, format="csr")
            on = kron(on, N2 if j == i else I2, format="csr")
        Xs.append(ox)
        Ns.append(on)
    return Xs, Ns


Xs, Ns = make_ops(n_qubits)

# ── Longitudinal coefficients h_i ─────────────────────────────────────
eff_long_total = (diagonal_elements + 0.5 * (d_opt**2).sum()) / gamma
h_total = eff_long_total + gamma * (1 - 2 * ntot)
eff_long_driver = external_field / gamma
h_driver = eff_long_driver + gamma * (1 - 2 * ntot)

# ── Hamiltonians ──────────────────────────────────────────────────────
# H = 2γ·Σ_{i<j} n_i n_j  +  Σ_i h_i·n_i  +  Σ_i (d_i/√2)·X_i
H_nn = sum(
    2 * gamma * Ns[i].dot(Ns[j])
    for i in range(n_qubits)
    for j in range(i + 1, n_qubits)
)


def build_Hzz(h_c):
    H = H_nn.copy()
    H += sum(h_c[i] * Ns[i] for i in range(n_qubits))
    return H


def build_Hx():
    H = sum((d_opt[i] / np.sqrt(2)) * Xs[i] for i in range(n_qubits))
    return H


H_total_sp = build_Hzz(h_total) + build_Hx()
H_driver_sp = build_Hzz(h_driver)

# ── Ground state of H_total ───────────────────────────────────────────
evals, evecs = eigsh(H_total_sp, k=3, which="SA")
idx = np.argsort(evals)
evals = evals[idx]
evecs = evecs[:, idx]
gs = evecs[:, 0]
gs_probs = np.abs(gs) ** 2
bitstrings = [format(i, "03b") for i in range(2**n_qubits)]

# ── Scipy exact evolution ─────────────────────────────────────────────
psi0 = np.zeros(2**n_qubits, dtype=complex)
psi0[4] = 1.0  # |100>
tau = 10 * gamma
n_steps = 5000
times = np.linspace(0, tau, n_steps)
dt = times[1] - times[0]
psi = psi0.copy()
for t in times:
    s = t / tau
    psi = expm_multiply(-1j * dt * ((1 - s) * H_driver_sp + s * H_total_sp), psi)
psi /= np.linalg.norm(psi)
exact_probs = np.abs(psi) ** 2
fid_exact = abs(np.dot(gs.conj(), psi)) ** 2

# ── Pulser mapping ────────────────────────────────────────────────────
# Direct (no extra prefactors):
#   U_ij    = 2γ          (all pairs equal → equilateral triangle)
#   δ_i     = −h_i        (Pulser: H −= δ·n, so δ = −h)
#   Ω_i     = √2·|d_i|    (from Ω/2·X = d/√2·X)
#   phase   = π if d_i<0
#
# Pulser basis: |r⟩=[1,0]ᵀ, |g⟩=[0,1]ᵀ
#   → state index i in Pulser ↔ state index (2ⁿ−1−i) in scipy
#   → initial |100⟩ (scipy idx 4) = |r,g,g⟩ (Pulser idx 3)
#     i.e. qt.basis(2,0)⊗qt.basis(2,1)⊗qt.basis(2,1)
U_pulser = 2 * gamma
delta_total_p = -h_total
delta_driver_p = -h_driver
Omega = np.sqrt(2) * np.abs(d_opt)
phases = np.where(d_opt < 0, np.pi, 0.0)
C6 = MockDevice.interaction_coeff
r = (C6 / U_pulser) ** (1 / 6)

# ── Pulser hyperparameter report ──────────────────────────────────────
ch_local = MockDevice.channels["rydberg_local"]
ch_real = AnalogDevice.channels["rydberg_global"]
sep = "═" * 62

print(sep)
print("  PULSER HYPERPARAMETER REPORT — He-6 Adiabatic Preparation")
print(sep)

print("\n── Physical system ─────────────────────────────────────────")
print(f"  gamma                        : {gamma}")
print(f"  ntot                         : {ntot}")
print(f"  d_opt                        : {d_opt}")
print(f"  GS energy                    : {evals[0]:.5f}")
print(f"  Spectral gap Δ               : {evals[1]-evals[0]:.5f}")

print("\n── Rydberg mapping ─────────────────────────────────────────")
print(f"  U_ij = 2·γ                   : {U_pulser:.4f}  rad/μs")
print(f"  Interatomic distance r        : {r:.4f}  μm")
print(
    f"  {'Qubit':<6} {'Ω (rad/μs)':<14} {'phase (rad)':<14} {'δ_driver':<14} {'δ_target'}"
)
print(f"  {'─'*62}")
for qi in range(n_qubits):
    print(
        f"  q{qi:<5} {Omega[qi]:<14.5f} {phases[qi]:<14.5f} "
        f"{delta_driver_p[qi]:<14.5f} {delta_total_p[qi]:.5f}"
    )

print("\n── Annealing schedule ──────────────────────────────────────")
print(f"  τ = 10·γ                     : {tau:.1f}  μs")
print(f"  Duration                     : {int(tau*1000)}  ns")
print(f"  H(s) = (1−s)·H_driver + s·H_total,  s ∈ [0,1]")

print("\n── MockDevice limits (used for simulation) ─────────────────")
print(f"  max_abs_detuning             : {ch_local.max_abs_detuning}  (unlimited)")
print(f"  max_amp                      : {ch_local.max_amp}  (unlimited)")
print(f"  clock_period                 : {ch_local.clock_period}  ns")

print("\n── AnalogDevice limits (real hardware target) ──────────────")
print(f"  max_abs_detuning             : {ch_real.max_abs_detuning:.4f}  rad/μs")
print(f"  max_amp (Ω_max)              : {ch_real.max_amp:.4f}  rad/μs")
print(f"  C6 coefficient               : {C6:.2f}  μm⁶·rad/μs")
print(f"  min_atom_distance            : {AnalogDevice.min_atom_distance}  μm")
print(f"  max_radial_distance          : {AnalogDevice.max_radial_distance}  μm")

print("\n── Feasibility vs AnalogDevice ─────────────────────────────")
all_ok = True
for qi in range(n_qubits):
    for val, name, limit in [
        (Omega[qi], f"Ω_q{qi}", ch_real.max_amp),
        (abs(delta_driver_p[qi]), f"|δ_driver_q{qi}|", ch_real.max_abs_detuning),
        (abs(delta_total_p[qi]), f"|δ_target_q{qi}|", ch_real.max_abs_detuning),
    ]:
        ok = val <= limit
        if not ok:
            all_ok = False
        print(f"  {name:<22}: {val:>8.4f}  ≤ {limit:.4f}  {'✓' if ok else '✗ EXCEEDS'}")

r_ok = r >= AnalogDevice.min_atom_distance
if not r_ok:
    all_ok = False
print(
    f"  r = {r:.4f} μm  ≥  min {AnalogDevice.min_atom_distance} μm  {'✓' if r_ok else '✗ TOO CLOSE'}"
)
print(
    f"\n  → {'✓ FEASIBLE on AnalogDevice' if all_ok else '✗ MockDevice only (limits exceeded)'}"
)
print(sep)

# ── Run Pulser ────────────────────────────────────────────────────────
reg = Register.from_coordinates(
    np.array([[0.0, 0.0], [r, 0.0], [r / 2, r * np.sqrt(3) / 2]]), prefix="q"
)
duration_ns = int(tau * 1000)
seq = Sequence(reg, MockDevice)
for qi in range(n_qubits):
    seq.declare_channel(f"ch{qi}", "rydberg_local", initial_target=f"q{qi}")
for qi in range(n_qubits):
    seq.add(
        Pulse(
            RampWaveform(duration_ns, 0, Omega[qi]),
            RampWaveform(duration_ns, delta_driver_p[qi], delta_total_p[qi]),
            phases[qi],
        ),
        f"ch{qi}",
    )

init_ket = qt.tensor(qt.basis(2, 0), qt.basis(2, 1), qt.basis(2, 1))
sim = QutipEmulator.from_sequence(seq, sampling_rate=0.005)
sim.set_initial_state(init_ket)
result = sim.run(progress_bar=False)
psi_raw = result.get_final_state().full().flatten()
N = 2**n_qubits
psi_pulser = np.array([psi_raw[N - 1 - i] for i in range(N)])
pulser_probs = np.abs(psi_pulser) ** 2
fid_pulser = abs(np.dot(gs.conj(), psi_pulser)) ** 2

print(f"\nScipy  fidelity: {fid_exact:.4f}")
print(f"Pulser fidelity: {fid_pulser:.4f}")
print(f"\n{'State':<8} {'Exact':>10} {'Pulser':>10} {'GS':>10}")
print("-" * 42)
for bs, ep, pp, gp in zip(bitstrings, exact_probs, pulser_probs, gs_probs):
    if ep + pp + gp > 1e-4:
        print(f"|{bs}>   {ep:>10.4f} {pp:>10.4f} {gp:>10.4f}")

# ── Plot ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor("#0f0f1a")
gs_grid = gridspec.GridSpec(2, 2, figure=fig, hspace=0.48, wspace=0.35)
C_EXACT = "#4fc3f7"
C_PULSER = "#ff8a65"
C_GS = "#a5d6a7"
bw = 0.25
x = np.arange(N)
labels = [f"|{bs}⟩" for bs in bitstrings]

ax_a = fig.add_subplot(gs_grid[0, :])
ax_a.set_facecolor("#12122a")
ax_a.bar(x - bw, exact_probs, bw, color=C_EXACT, alpha=0.9, label="Exact (scipy)")
ax_a.bar(x, pulser_probs, bw, color=C_PULSER, alpha=0.9, label="Pulser QutipEmulator")
ax_a.bar(x + bw, gs_probs, bw, color=C_GS, alpha=0.9, label="Ground state")
ax_a.set_xticks(x)
ax_a.set_xticklabels(labels, fontsize=10, color="#aaaacc")
ax_a.set_ylabel("Probability", fontsize=11, color="#ccccdd")
ax_a.set_title(
    f"Bitstring Probabilities — He-6  (γ={gamma}, τ=10γ={tau:.0f} μs)",
    fontsize=13,
    fontweight="bold",
    color="#e8e8ff",
    pad=10,
)
ax_a.legend(facecolor="#1a1a2e", edgecolor="#3a3a5e", labelcolor="white", fontsize=10)
ax_a.tick_params(colors="#888899")
ax_a.spines[:].set_color("#2a2a4a")
ax_a.yaxis.grid(True, color="#2a2a4a", linewidth=0.5)
ax_a.set_axisbelow(True)
ylim = max(np.max(exact_probs), np.max(pulser_probs), np.max(gs_probs)) * 1.15
ax_a.set_ylim(0, ylim)
for i in [1, 2, 4]:
    ax_a.axvspan(i - 0.45, i + 0.45, alpha=0.10, color="#2db666", zorder=1)
ax_a.text(
    2.3,
    ylim * 0.95,
    "one-hot sector (n=1)",
    ha="center",
    fontsize=9,
    color="#5dde99",
    style="italic",
)

oh = [1, 2, 4]
x_oh = np.arange(3)
ax_b = fig.add_subplot(gs_grid[1, 0])
ax_b.set_facecolor("#12122a")
ax_b.bar(x_oh - bw, exact_probs[oh], bw, color=C_EXACT, alpha=0.9, label="Exact")
ax_b.bar(x_oh, pulser_probs[oh], bw, color=C_PULSER, alpha=0.9, label="Pulser")
ax_b.bar(x_oh + bw, gs_probs[oh], bw, color=C_GS, alpha=0.9, label="GS")
ax_b.set_xticks(x_oh)
ax_b.set_xticklabels([labels[i] for i in oh], fontsize=11, color="#aaaacc")
ax_b.set_ylabel("Probability", fontsize=11, color="#ccccdd")
ax_b.set_title(
    "One-Hot Sector (n=1)", fontsize=13, fontweight="bold", color="#e8e8ff", pad=10
)
ax_b.legend(facecolor="#1a1a2e", edgecolor="#3a3a5e", labelcolor="white", fontsize=9)
ax_b.tick_params(colors="#888899")
ax_b.spines[:].set_color("#2a2a4a")
ax_b.yaxis.grid(True, color="#2a2a4a", linewidth=0.5)
ax_b.set_axisbelow(True)

ax_c = fig.add_subplot(gs_grid[1, 1])
ax_c.set_facecolor("#12122a")
diff = pulser_probs - exact_probs
ax_c.bar(x, diff, color=["#ef5350" if d < 0 else "#66bb6a" for d in diff], alpha=0.9)
ax_c.axhline(0, color="#aaaacc", linewidth=0.8, linestyle="--")
ax_c.set_xticks(x)
ax_c.set_xticklabels(labels, fontsize=9, color="#aaaacc", rotation=15)
ax_c.set_ylabel("Pulser − Exact", fontsize=11, color="#ccccdd")
ax_c.set_title("Residual", fontsize=13, fontweight="bold", color="#e8e8ff", pad=10)
ax_c.tick_params(colors="#888899")
ax_c.spines[:].set_color("#2a2a4a")
ax_c.yaxis.grid(True, color="#2a2a4a", linewidth=0.5)
ax_c.set_axisbelow(True)

param_str = (
    f"U=2γ={U_pulser:.0f} rad/μs  |  r={r:.2f} μm  |  "
    f"Ω=[{', '.join(f'{o:.3f}' for o in Omega)}] rad/μs  |  "
    f"δ_driver=[{', '.join(f'{d:.3f}' for d in delta_driver_p)}]  |  "
    f"δ_target=[{', '.join(f'{d:.3f}' for d in delta_total_p)}]  |  "
    f"F_scipy={fid_exact:.3f}  F_pulser={fid_pulser:.3f}"
)
fig.text(0.5, 0.003, param_str, ha="center", fontsize=7.5, color="#8888aa")

plt.savefig(
    "pulser_He6_corrected.png",
    dpi=150,
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
)
print("\nPlot saved → pulser_He6_corrected.png")
