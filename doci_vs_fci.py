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

from pyscf import gto, scf, fci, ao2mo
from pyscf import doci as pyscf_doci

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


def doci_fci_compare(atom, basis="sto-3g", active_space=None):
    print("=" * 65)
    label = f"  atom={atom}  basis={basis}" + (
        f"  active_space={active_space}" if active_space else ""
    )
    print(label)
    print("=" * 65)

    couplings = get_doci_couplings(
        atom, basis=basis, active_space=active_space, verbose=False
    )
    mf = couplings["mf"]
    n_orb, n_pairs = couplings["n_orb"], couplings["n_pairs"]

    # ---- our hand-rolled DOCI diagonalization -----------------------------
    H_doci, configs = build_doci_matrix(couplings)
    evals, evecs = scipy.linalg.eigh(H_doci)
    e_doci_manual = evals[0]

    # ---- PySCF FCI (exact reference) --------------------------------------
    if active_space is None:
        fci_solver = fci.FCI(mf)
        e_fci, _ = fci_solver.kernel()
    else:
        from pyscf import mcscf

        n_act, n_elec_act = active_space
        mc = mcscf.CASCI(mf, n_act, n_elec_act)
        e_fci = mc.kernel()[0]

    # ---- PySCF's own DOCI solver (independent cross-check) ----------------
    norb_full = mf.mo_coeff.shape[1]
    nelec_full = mf.mol.nelectron
    h1e_full = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri_full = ao2mo.kernel(mf._eri, mf.mo_coeff, compact=False).reshape(
        norb_full, norb_full, norb_full, norb_full
    )
    nuc = mf.mol.energy_nuc()

    if active_space is None:
        doci_solver = pyscf_doci.DOCI(mf)
        e_doci_pyscf_elec, _ = doci_solver.kernel(
            h1e_full, eri_full, norb_full, nelec_full
        )
        e_doci_pyscf = e_doci_pyscf_elec + nuc
    else:
        e_doci_pyscf = (
            None  # skip cross-check for active-space case (different API path)
        )

    print(
        f"  n_orb={n_orb}  n_pairs={n_pairs}  |  DOCI space dim = C({n_orb},{n_pairs}) "
        f"= {len(configs)}"
    )
    print(f"  RHF                         : {mf.e_tot:+.8f}")
    print(f"  DOCI  (this script, hand-rolled diag) : {e_doci_manual:+.8f}")
    if e_doci_pyscf is not None:
        print(
            f"  DOCI  (pyscf.doci, cross-check)        : {e_doci_pyscf:+.8f}  "
            f"(diff vs hand-rolled: {abs(e_doci_pyscf-e_doci_manual):.2e})"
        )
    print(f"  FCI   (exact)               : {e_fci:+.8f}")
    print(f"  DOCI - FCI error            : {(e_doci_manual - e_fci)*1e3:+.4f} mHa")
    print()

    return {
        "e_rhf": mf.e_tot,
        "e_doci": e_doci_manual,
        "e_doci_pyscf": e_doci_pyscf,
        "e_fci": e_fci,
        "n_configs": len(configs),
    }


if __name__ == "__main__":
    results = {}
    results["H2"] = doci_fci_compare("H 0 0 0; H 0 0 0.74", basis="sto-3g")
    results["LiH"] = doci_fci_compare("Li 0 0 0; H 0 0 1.595", basis="sto-3g")
    results["H4"] = doci_fci_compare(
        "H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22", basis="sto-3g"
    )
    results["H2O"] = doci_fci_compare(
        "O 0 0 0; H 0.757 0.586 0; H -0.757 0.586 0",  # equilibrium geometry, Angstrom
        basis="sto-3g",
    )
    print("=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(
        f"  {'Molecule':<10} {'RHF':>12} {'DOCI':>12} {'FCI':>12} {'Err/mHa':>10} {'#configs':>9}"
    )
    for name, r in results.items():
        print(
            f"  {name:<10} {r['e_rhf']:>12.6f} {r['e_doci']:>12.6f}"
            f" {r['e_fci']:>12.6f} {(r['e_doci']-r['e_fci'])*1e3:>10.3f} {r['n_configs']:>9}"
        )
