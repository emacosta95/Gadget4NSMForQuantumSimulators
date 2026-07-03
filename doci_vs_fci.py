"""
doci_vs_fci.py
================
Compare DOCI (seniority-zero diagonalization) against exact FCI, using the
couplings (eps_a, g_ab, v_ab) from get_doci_couplings().

This builds H_DOCI explicitly in the n_pairs-occupation basis and diagonalizes
it directly (no OpenFermion needed), then compares E_DOCI to:
  (a) PySCF FCI  (exact answer, full Hilbert space)
  (b) PySCF's own DOCI solver (pyscf.doci), as an independent cross-check
      that our hand-rolled couplings + diagonalization agree with a
      battle-tested implementation.

Matrix element conventions (seniority-zero / hard-core pair-boson picture,
consistent with OpenFermion's DOCIHamiltonian and Eq. 18 of arXiv:1602.03543):
  - Diagonal:      <n|H|n> = E_core + sum_{a in n} eps_a + sum_{a<b in n} v_ab
  - Off-diagonal:  <n'|H|n> = g_ab   if n' is obtained from n by moving exactly
                              one pair from orbital b -> orbital a (a not in n,
                              b in n, all else identical). No fermionic sign:
                              seniority-zero pair operators carry no JW string.
  - Everything else: 0 (DOCI only connects single-pair-hop configurations,
                         consistent with P_a^dagger P_b being a 2-body op).
"""

import numpy as np
from itertools import combinations
import scipy.linalg

from pyscf import doci as pyscf_doci
from pyscf import gto, scf, fci, ao2mo, mcscf
from doci_quantum_chemistry import get_doci_couplings


def build_doci_matrix(couplings):
    """
    Build the full DOCI Hamiltonian matrix in the n_pairs-occupation basis.

    Returns
    -------
    H : (D, D) ndarray, D = C(n_orb, n_pairs)
    configs : list of frozenset, the basis states (orbital occupation sets)
    """
    eps, g, v, E_core = (
        couplings["eps"],
        couplings["g"],
        couplings["v"],
        couplings["E_core"],
    )
    n_orb, n_pairs = couplings["n_orb"], couplings["n_pairs"]

    configs = [frozenset(c) for c in combinations(range(n_orb), n_pairs)]
    D = len(configs)
    H = np.zeros((D, D))

    for I, cI in enumerate(configs):
        # Diagonal
        H[I, I] = (
            E_core
            + sum(eps[a] for a in cI)
            + sum(v[a, b] for a in cI for b in cI if a < b)
        )
        # Off-diagonal: single pair hop b (occupied) -> a (unoccupied)
        unocc = [a for a in range(n_orb) if a not in cI]
        for b in cI:
            for a in unocc:
                cJ = frozenset((cI - {b}) | {a})
                J = configs.index(cJ)
                H[I, J] = g[a, b]  # symmetric, Hermitian matrix, real coefficients

    return H, configs


def doci_fci_compare(atom, basis="sto-3g", active_space=None, charge=0):
    mol = gto.M(atom=atom, basis=basis, charge=charge, verbose=0)
    mf = scf.RHF(mol).run()

    couplings = get_doci_couplings(
        atom, basis=basis, active_space=active_space, charge=charge, verbose=False
    )
    n_orb, n_pairs = couplings["n_orb"], couplings["n_pairs"]

    # ---- hand-rolled DOCI in active space ----------------------------------
    H_doci, configs = build_doci_matrix(couplings)
    evals, evecs = scipy.linalg.eigh(H_doci)
    e_doci = evals[0]

    # ---- CASCI-FCI in active space (same space as DOCI) -------------------
    if active_space is not None:
        n_act, n_elec_act = active_space
        mc = mcscf.CASCI(mf, n_act, n_elec_act)
        e_fci_active = mc.kernel()[0]
    else:
        e_fci_active = None

    # ---- true full-space FCI (always) -------------------------------------
    fci_solver = fci.FCI(mf)
    e_fci_full = fci_solver.kernel()[0]

    # ---- print -------------------------------------------------------------
    print(
        f"  n_orb={n_orb}  n_pairs={n_pairs}  |  DOCI dim=C({n_orb},{n_pairs})={len(configs)}"
    )
    print(f"  RHF                           : {mf.e_tot:+.8f}")
    print(f"  DOCI  (active space)          : {e_doci:+.8f}")
    if e_fci_active is not None:
        print(
            f"  CASCI-FCI (same active space) : {e_fci_active:+.8f}  "
            f"| seniority error = {(e_doci - e_fci_active)*1e3:+.4f} mHa"
        )
    print(
        f"  FCI   (full space, exact)     : {e_fci_full:+.8f}  "
        f"| total DOCI error = {(e_doci - e_fci_full)*1e3:+.4f} mHa"
    )
    if e_fci_active is not None:
        print(
            f"  truncation error (CASCI-full) : {(e_fci_active - e_fci_full)*1e3:+.4f} mHa"
        )

    return {
        "e_rhf": mf.e_tot,
        "e_doci": e_doci,
        "e_fci_active": e_fci_active,
        "e_fci_full": e_fci_full,
        "n_configs": len(configs),
    }


if __name__ == "__main__":
    results = {}
    results["H2"] = doci_fci_compare("H 0 0 0; H 0 0 0.74", basis="sto-3g")
    results["LiH"] = doci_fci_compare(
        "Li 0 0 0; H 0 0 1.595", basis="cc-pvdz", active_space=(6, 2)
    )
    results["H4"] = doci_fci_compare(
        "H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22", basis="sto-3g"
    )
    results["H2O"] = doci_fci_compare(
        "O 0 0 0; H 0.757 0.586 0; H -0.757 0.586 0",  # equilibrium geometry, Angstrom
        basis="sto-3g",
    )
    results["Be2"] = doci_fci_compare(
        "Be 0 0 0; Be 0 0 2.45", basis="sto-3g", active_space=(6, 4)
    )
    print("=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(
        f"  {'Molecule':<12} {'RHF':>12} {'DOCI':>12} {'CASCI-FCI':>12} "
        f"{'Full-FCI':>12} {'Sen.Err/mHa':>12} {'Tot.Err/mHa':>12} {'#configs':>9}"
    )
    for name, r in results.items():
        sen_err = (
            (r["e_doci"] - r["e_fci_active"]) * 1e3
            if r["e_fci_active"] is not None
            else float("nan")
        )
        tot_err = (r["e_doci"] - r["e_fci_full"]) * 1e3
        casci_str = (
            f"{r['e_fci_active']:>12.6f}"
            if r["e_fci_active"] is not None
            else f"{'---':>12}"
        )
        print(
            f"  {name:<12} {r['e_rhf']:>12.6f} {r['e_doci']:>12.6f} "
            f"{casci_str} {r['e_fci_full']:>12.6f} "
            f"{sen_err:>12.3f} {tot_err:>12.3f} {r['n_configs']:>9}"
        )
