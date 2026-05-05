"""
Adiabatic state preparation for the He-6 quasiparticle Ising Hamiltonian
using Bloqade (QuEra's neutral-atom SDK).

The notebook GadgetHe6.ipynb encodes the nuclear ground state of He-6 in
3 qubits (quasiparticle modes for the p-shell neutrons) via a gadget
Hamiltonian of the form:

  H_total = gamma * sum_{i<j} 2 Z_i Z_j          (particle-number constraint)
           + sum_i h_i Z_i                          (longitudinal fields)
           + sum_i (d_i / sqrt(2)) X_i             (transverse driver)

where:
  gamma  = 100  (constraint strength, in units of J)
  d_opt  = [-1.02132267,  1.02132267, -3.6852787]
  diagonal_elements (quasiparticle single-particle energies):
           [-8.4321, -8.4321, -5.1203]
  h_i    = (diagonal_elements[i] + 0.5 * sum(d_opt**2)) / gamma
           + gamma * (1 - 2*ntot)    with ntot=1

The adiabatic protocol interpolates from H_driver to H_total:
  H(s) = (1-s) * H_driver  +  s * H_total,   s in [0,1]

The driver Hamiltonian is the same ZZ constraint plus a single large
negative Z field on site 0:
  external_field = [-18.4321/2, 0, 0]

The initial computational state is |1,0,0> (basis[4] in the notebook,
i.e. qubit 0 occupied, qubits 1 and 2 empty).

----------------------------------------------------------------------
MAPPING TO RYDBERG PHYSICS
----------------------------------------------------------------------

The Rydberg Hamiltonian in Bloqade is (using n_i = (1 - Z_i)/2):

  H_Ryd = sum_i (Omega_i/2) X_i
        - sum_i Delta_i n_i
        + sum_{i<j} C6 / r_{ij}^6  n_i n_j

Expanding n_i:

  H_Ryd = (Omega_i/2) X_i
        + (Delta_i/2) Z_i  - Delta_i/2   (const)
        + C6/(4 r^6) Z_i Z_j + C6/(4 r^6) * (linear + const)

So the mapping is:
  Omega_i / 2   <-->  d_i / sqrt(2)         =>  Omega_i = d_i * sqrt(2)
  Delta_i / 2   <-->  h_i                   =>  Delta_i = 2 * h_i
  C6 / (4 r^6)  <-->  2 * gamma             =>  r = (C6 / (8*gamma))^(1/6)

Signs: d_i can be negative. Since X is symmetric, only |d_i| matters for
the magnitude of the Rabi drive. The sign can be absorbed into a local
Z rotation (i.e. a phase flip |0><->|1> on that site), which we handle
by flipping the detuning sign on that site.

For the geometry we place the 3 atoms in an equilateral triangle so all
pairwise distances are equal, giving uniform ZZ coupling.

Units: Bloqade uses rad/us for frequencies and um for distances.
We set an overall energy scale J_scale (rad/us) and express all
notebook quantities relative to it.

----------------------------------------------------------------------
NOTE ON HARDWARE vs EMULATOR
----------------------------------------------------------------------

- Local Rabi amplitude (site-dependent Omega):  supported in the Bloqade
  Python emulator via .rydberg.rabi.amplitude.scale([s0,s1,s2]).
  NOT available on Aquila hardware (global Omega only there).

- Local detuning: supported on both the emulator and Aquila hardware.

This script targets the Bloqade Python emulator.

----------------------------------------------------------------------
"""

import numpy as np
from bloqade import start

# ── Physical parameters from the notebook ────────────────────────────
gamma = 10.0  # constraint strength (notebook units)
ntot = 1  # target particle number
n_qubits = 3

d_opt = np.array([-1.02132267, 1.02132267, -3.6852787])  # from optimize_rank1

diagonal_elements = np.array([-8.4321, -8.4321, -5.1203])  # quasiparticle SP energies

# Effective longitudinal field for H_total
h_eff = (diagonal_elements + 0.5 * np.sum(d_opt**2)) / gamma + gamma * (1 - 2 * ntot)

# Driver longitudinal field (external_field = [-18.4321/2, 0, 0])
external_field = np.array([-18.4321 / 2.0, 0.0, 0.0])
h_driver = external_field / gamma + gamma * (1 - 2 * ntot)

print("h_eff   (target):", h_eff)
print("h_driver (driver):", h_driver)
print("d_opt:", d_opt)

# ── Energy-scale conversion ──────────────────────────────────────────
# We choose J_scale such that max(|d_i|/sqrt(2)) maps to a reasonable
# Rabi frequency (~10 rad/us is typical for Rydberg experiments).
# max(|d_i|) = 3.6852787  =>  Omega_max = 3.6852787 * sqrt(2) * J_scale
# We want Omega_max ~ 10 rad/us  =>  J_scale = 10 / (3.6852787 * sqrt(2))
J_scale = 10.0 / (np.max(np.abs(d_opt)) * np.sqrt(2))  # rad/us per notebook unit
print(f"\nJ_scale = {J_scale:.4f} rad/us per notebook unit")

# ── Rydberg C6 and atom spacing ──────────────────────────────────────
# For Rb |70S> in Bloqade's default: C6 ~ 862690 GHz·um^6
# = 862690e3 rad/us · um^6  (since 1 GHz = 2pi * 1e3 rad/us ... but
# Bloqade uses rad/us directly)
# We use the value from Bloqade documentation: C6 = 862690 (in rad/us · um^6)
# The ZZ coupling in notebook units is 2*gamma.
# In Rydberg units: C6 / (4 * r^6) = 2 * gamma * J_scale
# => r^6 = C6 / (8 * gamma * J_scale)
C6_rad_us = 862690.0  # rad/us * um^6  (Bloqade default for Rb 70S)
J_ZZ_ryd = 2.0 * gamma * J_scale  # rad/us
r_spacing = (C6_rad_us / (4.0 * J_ZZ_ryd)) ** (1.0 / 6.0)  # um

print(f"Target ZZ coupling: {J_ZZ_ryd:.3f} rad/us")
print(f"Required atom spacing (equilateral triangle): {r_spacing:.3f} um")

# ── Equilateral triangle positions ───────────────────────────────────
# Place 3 atoms at vertices of an equilateral triangle with side r_spacing
x0, y0 = 0.0, 0.0
x1, y1 = r_spacing, 0.0
x2, y2 = r_spacing / 2.0, r_spacing * np.sqrt(3) / 2.0

positions = [(x0, y0), (x1, y1), (x2, y2)]
print(f"\nAtom positions (um): {positions}")

# ── Rabi amplitudes ───────────────────────────────────────────────────
# Omega_i = |d_i| * sqrt(2) * J_scale   (magnitude; sign absorbed below)
Omega = np.abs(d_opt) * np.sqrt(2) * J_scale  # rad/us, per site
Omega_max = float(np.max(Omega))

# Scale factors s_i = Omega_i / Omega_max  (all positive, in [0, 1])
rabi_scales = (Omega / Omega_max).tolist()
print(f"\nOmega per site (rad/us): {Omega}")
print(f"Omega_max: {Omega_max:.4f} rad/us")
print(f"Rabi scale factors: {rabi_scales}")

# ── Detunings ─────────────────────────────────────────────────────────
# Delta_i = 2 * h_i * J_scale  (for the total Hamiltonian)
# Sign convention: In the notebook, the Z field is h_i * Z_i.
# In Rydberg: -Delta_i * n_i = (Delta_i/2) Z_i - Delta_i/2
# So Delta_i_ryd = -2 * h_i * J_scale
#
# Sites where d_i < 0: the sign flip of X is equivalent to flipping
# |0><->|1>, which sends Z -> -Z, so h_i -> -h_i for those sites.
sign_d = np.sign(d_opt)  # +1 or -1 per site

# Effective h after absorbing sign of d into basis flip
h_eff_signed = h_eff * sign_d
h_driver_signed = h_driver * sign_d

# Rydberg detuning = -2 * h * J_scale
Delta_target = -2.0 * h_eff_signed * J_scale  # rad/us, per site
Delta_driver = -2.0 * h_driver_signed * J_scale  # rad/us, per site

Delta_uniform_target = float(np.mean(Delta_target))
Delta_uniform_driver = float(np.mean(Delta_driver))

# Local detuning offsets (site-specific corrections on top of uniform)
delta_local_target = (Delta_target - Delta_uniform_target).tolist()
delta_local_driver = (Delta_driver - Delta_uniform_driver).tolist()

print(f"\nDelta per site, target (rad/us): {Delta_target}")
print(f"Delta per site, driver (rad/us): {Delta_driver}")
print(f"Uniform detuning, target: {Delta_uniform_target:.4f} rad/us")
print(f"Uniform detuning, driver: {Delta_uniform_driver:.4f} rad/us")
print(f"Local offsets, target: {delta_local_target}")
print(f"Local offsets, driver: {delta_local_driver}")

# ── Adiabatic schedule ────────────────────────────────────────────────
# The notebook uses tau = 200 (in units of 1/J).
# In physical time: T_total = tau / J_scale  (us)
tau_notebook = 200.0  # notebook time units
T_total = tau_notebook / J_scale  # us
print(f"\nTotal annealing time: {T_total:.2f} us")

# Ramp edges (rise/fall of Rabi amplitude): 0.05 us each
t_ramp = 0.05  # us
t_hold = T_total - 2 * t_ramp

# ── Build the Bloqade program ─────────────────────────────────────────
#
# Strategy:
#  - Rabi amplitude: ramp from 0 to Omega_max, hold, ramp back to 0.
#    Site-specific amplitudes via .scale([s0, s1, s2]).
#  - Global detuning: linearly sweep from Delta_driver to Delta_target.
#  - Local detuning: linearly sweep from delta_local_driver[i] to
#    delta_local_target[i] on each site, via .location(i, scale=...).
#    (The local detuning is a scale factor * the global waveform shape.)
#
# Note: Bloqade's local detuning API multiplies a site-specific scalar
# by a shared waveform. We use a waveform that sweeps [driver -> target]
# and set the per-site scale to 1.0 (then add individual corrections via
# separate .location() calls with their own waveforms).

durations = [t_ramp, t_hold, t_ramp]

# Global detuning waveform: linear ramp from driver value to target value
global_det_values = [
    Delta_uniform_driver,  # t=0
    Delta_uniform_driver,  # end of ramp-up
    Delta_uniform_target,  # end of hold (target reached)
    Delta_uniform_target,  # end of ramp-down
]

program = (
    start.add_position(positions[0])
    .add_position(positions[1])
    .add_position(positions[2])
    # ── Rabi amplitude: site-dependent via scale ──────────────────
    .rydberg.rabi.amplitude.scale(rabi_scales)  # [s0, s1, s2] in [0,1]
    .piecewise_linear(
        durations=durations,
        values=[0.0, Omega_max, Omega_max, 0.0],
    )
    # ── Global detuning: sweeps from driver to target ─────────────
    .detuning.uniform.piecewise_linear(
        durations=durations,
        values=global_det_values,
    )
    # ── Local detuning corrections per site ───────────────────────
    # Site 0
    .location(0, scale=1.0)
    .piecewise_linear(
        durations=durations,
        values=[
            delta_local_driver[0],
            delta_local_driver[0],
            delta_local_target[0],
            delta_local_target[0],
        ],
    )
    # Site 1
    .location(1, scale=1.0)
    .piecewise_linear(
        durations=durations,
        values=[
            delta_local_driver[1],
            delta_local_driver[1],
            delta_local_target[1],
            delta_local_target[1],
        ],
    )
    # Site 2
    .location(2, scale=1.0)
    .piecewise_linear(
        durations=durations,
        values=[
            delta_local_driver[2],
            delta_local_driver[2],
            delta_local_target[2],
            delta_local_target[2],
        ],
    )
)

# ── Run on the Bloqade Python emulator ───────────────────────────────
n_shots = 1000
result = program.bloqade.python().run(n_shots)

# ── Analyse results ───────────────────────────────────────────────────
# The ground state of H_total in the 1-particle sector is a superposition
# of |001>, |010>, |100> (one Rydberg excitation).
# After the sign-flip on sites with d_i < 0, the measurement basis is
# flipped on those sites: site 0 and site 2 have d<0, so |0>_ryd <-> |1>_phys.
#
# We report the raw shot counts and the probability of each bitstring.

counts = result.report().counts()
print("\n── Raw shot counts (Rydberg basis) ──")
for bitstring, count in sorted(counts[0].items(), key=lambda x: -x[1]):
    print(f"  |{bitstring}> : {count} ({100*count/n_shots:.1f}%)")

# The one-particle sector in the physical (notebook) basis:
#   |100> in physical = site 0 occupied
#   |010> in physical = site 1 occupied
#   |001> in physical = site 2 occupied
#
# After sign-flip: sites 0,2 flipped. So:
#   physical |100> -> rydberg |010>  (site 0 flipped: 1->0, site2 flipped 0->1... wait)
# More carefully: sign flip on site i means |0>_i <-> |1>_i in the measurement.
# Site 0: d<0 -> flip bit 0.  Site 2: d<0 -> flip bit 2.
#
# physical bitstring b -> rydberg bitstring r:
#   r[i] = 1 - b[i]  if sign_d[i] < 0
#   r[i] = b[i]      if sign_d[i] > 0

print("\n── Probabilities in the physical (notebook) basis ──")
total = sum(counts[0].values())
phys_counts = {}
for bitstring, count in counts[0].items():
    bits_ryd = [int(c) for c in bitstring]
    bits_phys = [
        (1 - bits_ryd[i]) if sign_d[i] < 0 else bits_ryd[i] for i in range(n_qubits)
    ]
    phys_str = "".join(str(b) for b in bits_phys)
    phys_counts[phys_str] = phys_counts.get(phys_str, 0) + count

for bitstring, count in sorted(phys_counts.items(), key=lambda x: -x[1]):
    print(f"  |{bitstring}> : {count} ({100*count/total:.1f}%)")

# The ground state of He-6 (one active neutron, lowest quasiparticle)
# should have dominant weight on the single-particle states.
one_particle_prob = sum(v for k, v in phys_counts.items() if k.count("1") == 1) / total
print(f"\nTotal probability in 1-particle sector: {100*one_particle_prob:.1f}%")
