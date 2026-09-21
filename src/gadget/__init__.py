"""Unified gadget-Hamiltonian toolbox.

Replaces the per-nucleus / per-encoding notebook code by four objects:

    QuasiparticleProblem   the physics  (eps, g1, g2, n_particles)
    Encoding               the qubit map (one-hot / first quantization / embedded)
    GadgetHamiltonian      H_long + H_trans, plus .driver()
    QuantumAdiabatic4Gadget  the sweep

Typical use
-----------
>>> from src.gadget import *
>>> problem = QuasiparticleProblem.from_npz(
...     "data/matrix_elements_h_eff_2body/one_body_nn_sd.npz",
...     "data/matrix_elements_h_eff_2body/twobody_nn_sd.npz",
...     n_levels=6, n_particles=2, label="O20")
>>> drive = fit_drive(problem.g1, n_restarts=100, cap=4.0)
>>> enc = FirstQuantizationEncoding(problem)
>>> H = GadgetHamiltonian(problem, enc, drive,
...                       GadgetParameters(gamma=300, gamma2=300, self_energy="exact"))
>>> H.check()
>>> ada = QuantumAdiabatic4Gadget(H)
>>> result = ada.run(tau=50, track_spectrum=True)

Same problem in the second-quantized encoding, or on D-Wave:

>>> enc2 = ParticleConservationEncoding(problem)
>>> emb = find_minor_embedding(problem.n_levels, pegasus_size=16)
>>> enc3 = MinorEmbeddedEncoding(enc2, emb)
>>> H3 = GadgetHamiltonian(problem, enc3, drive,
...                        GadgetParameters(gamma=200, J_F=suggested_J_F(200)))
"""

from .adiabatic import AdiabaticResult, QuantumAdiabatic4Gadget
from .embedding import (
    MinorEmbedding,
    chain_drive,
    find_minor_embedding,
    from_minorminer,
    physical_drive,
    suggested_J_F,
)
from .encodings import (
    Encoding,
    FirstQuantizationEncoding,
    MinorEmbeddedEncoding,
    ParticleConservationEncoding,
    state_index,
)
from .fit import DriveFit, alpha_from_c, fit_drive
from .hamiltonian import GadgetHamiltonian, GadgetParameters
from .problem import QuasiparticleProblem
from .selfenergy import (
    gamma2_lower_bound,
    phi_symmetrized,
    phi_two_body,
    self_energy_report,
    theta_one_body,
    theta_uniform,
)

__all__ = [
    "QuasiparticleProblem",
    "DriveFit",
    "fit_drive",
    "alpha_from_c",
    "Encoding",
    "ParticleConservationEncoding",
    "FirstQuantizationEncoding",
    "MinorEmbeddedEncoding",
    "state_index",
    "GadgetHamiltonian",
    "GadgetParameters",
    "MinorEmbedding",
    "find_minor_embedding",
    "from_minorminer",
    "chain_drive",
    "physical_drive",
    "suggested_J_F",
    "QuantumAdiabatic4Gadget",
    "AdiabaticResult",
    "theta_uniform",
    "theta_one_body",
    "phi_two_body",
    "phi_symmetrized",
    "gamma2_lower_bound",
    "self_energy_report",
]
