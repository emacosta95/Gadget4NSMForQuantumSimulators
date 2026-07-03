from doci_quantum_chemistry import extract_doci_dicts
import pickle

# H2: 1 pair, 2 orbitals → 1 g term, 1 v term
g, v, eps, E_core, meta = extract_doci_dicts("H 0 0 0; H 0 0 0.74")

with open("data/doci_couplings_h2.pkl", "wb") as f:
    pickle.dump((g, v, eps, E_core, meta), f)


# H4: 2 pairs, 4 orbitals → 6 g terms, 6 v terms

atom = "H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22"
g, v, eps, E_core, meta = extract_doci_dicts(atom, basis="sto-3g")

with open("data/doci_couplings_h4.pkl", "wb") as f:
    pickle.dump((g, v, eps, E_core, meta), f)

# # LiH: 2 active pairs, 6 orbitals → up to 15 g/v terms

g, v, eps, E_core, meta = extract_doci_dicts(
    "Li 0 0 0; H 0 0 1.595", basis="cc-pvdz", active_space=(6, 2)
)
with open("data/doci_couplings_lih.pkl", "wb") as f:
    pickle.dump((g, v, eps, E_core, meta), f)
    # # H2O: 5 active pairs, 7 orbitals (4e active space)
    g, v, eps, E_core, meta = extract_doci_dicts(
        "O 0 0 0; H 0.757 0.586 0; H -0.757 0.586 0", active_space=(4, 4)
    )
with open("data/doci_couplings_h2o.pkl", "wb") as f:
    pickle.dump((g, v, eps, E_core, meta), f)

g, v, eps, E_core, meta = extract_doci_dicts(
    "Be 0 0 0; Be 0 0 2.45", basis="sto-3g", active_space=(6, 4)
)
with open("data/doci_couplings_be2.pkl", "wb") as f:
    pickle.dump((g, v, eps, E_core, meta), f)
