"""
get_doci_couplings()
=====================
Standalone extractor for the DOCI / seniority-zero gadget Hamiltonian couplings,
for ANY molecule, in one shot, directly from PySCF MO-basis integrals.

Target operator form (qubit picture, orbital index a,b = logical qubit):

    H_DOCI = sum_a  eps_a * N_a
           + sum_{a<b} g_ab * (X_a X_b + Y_a Y_b)
           + sum_{a<b} v_ab * N_a N_b
           + E_core   (nuclear repulsion + frozen-core energy, if active space used)

Physical content (Bytautas/Henderson/Scuseria seniority formalism;
also Bender FCIQMC-DOCI paper, arXiv:1602.03543, Eq. 18):

    eps_a = 2*h_aa + (aa|aa)                  one-body + same-orbital Coulomb self-energy
    g_ab  = (ab|ab)                           pair-hopping matrix element (EXCHANGE integral)
    v_ab  = 2*[ 2*(aa|bb) - (ab|ab) ]          density-density (Coulomb minus exchange)

  NOTE ON A CORRECTED BUG: g_ab was originally (and incorrectly) coded as (aa|bb), the
  DIRECT Coulomb integral. The correct pair-hopping matrix element is the EXCHANGE
  integral (ab|ab). This was caught by comparing against pyscf.doci's own ground-state
  eigenvector for H2 (where DOCI = FCI exactly) and back-solving the required
  off-diagonal element of the 2x2 Hamiltonian -- it matched (ab|ab), not (aa|bb), to
  8 decimal places. The root cause was a physicist/chemist notation mix-up: the
  literature (e.g. Eq. 19b of arXiv:1602.03543) writes this coupling as PHYSICIST
  <pp|qq>, which converts to CHEMIST (pq|pq) -- i.e. exactly (ab|ab) -- NOT chemist
  (pp|qq)=(aa|bb) as a naive notation-blind reading would suggest. This is the same
  physicist/chemist trap flagged in the original validated script's hr1 docstring.

  Derivation check: closed-shell RHF energy is
      E = E_nuc + 2*sum_i h_ii + sum_{i,j in occ} [2(ii|jj) - (ij|ij)]   (i,j run over ALL
                                                                           occupied pairs,
                                                                           including i=j)
  Splitting the (i,j) double sum into i=j (-> eps_a, since 2(ii|ii)-(ii|ii)=(ii|ii)) and the
  i!=j cross terms (-> 2x the v_ab/2 i<j sum, since the double sum hits both (i,j) and (j,i))
  is exactly what fixes the original factor-of-2 bug found by the diagonal-energy check below
  (H4/LiH initially failed mf.e_tot by ~1.5 Ha with v_ab carrying no compensating factor of 2;
  H2 alone could not catch this since it has only 1 pair and no i!=j cross term at all).

  with (pq|rs) in CHEMIST notation (PySCF ao2mo convention):
      (pq|rs) = integral over r1,r2 of phi_p(1)phi_q(1) (1/r12) phi_r(2)phi_s(2)

IMPORTANT — this is NOT computed by running a separate 2-electron calculation
for g_ab and a separate 4-electron calculation for v_ab. These are matrix
elements of the FIXED one- and two-electron integral tensors in the chosen MO
basis; they do not depend on how many electrons / pairs are actually in the
molecule. The same g_ab, v_ab numbers appear whether you then diagonalize in
the 1-pair, 2-pair, or N-pair sector. One integral pass gives you everything.

This is exactly what the previous validation script's build_doci_hamiltonian()
already did under the hood (hc, hr1, hr2) -- this module just exposes it as a
clean, molecule-agnostic, dependency-light function (no OpenFermion required),
with active-space support carried over from get_doci_couplings() in the
validated DOCI toolkit.
"""

import numpy as np
from pyscf import gto, scf, ao2mo, mcscf


def get_doci_couplings(atom, basis="sto-3g", active_space=None, verbose=True):
    """
    Compute the DOCI gadget couplings (eps_a, g_ab, v_ab) for any molecule.

    Parameters
    ----------
    atom : str or list
        PySCF atom spec, e.g. "H 0 0 0; H 0 0 0.74" or a list of
        (symbol, (x,y,z)) tuples (Angstrom).
    basis : str
        Basis set name (default 'sto-3g').
    active_space : tuple(int, int) or None
        (n_act_orbitals, n_act_electrons). If given, runs RHF on the full
        molecule, then restricts to an active space of n_act_orbitals MOs
        around the Fermi level holding n_act_electrons, using PySCF's
        mcscf.CASCI machinery to get the active-space effective integrals
        (core orbitals frozen, their energy folded into E_core).
        If None, uses ALL MOs (no restriction) -- only sensible for small
        molecules / minimal basis sets, since the qubit count = n_orbitals.
    verbose : bool
        Print a short summary table.

    Returns
    -------
    dict with keys:
        'eps'      : (n,) array, eps_a
        'g'        : (n,n) symmetric array, g_ab (zero diagonal)
        'v'        : (n,n) symmetric array, v_ab (zero diagonal)
        'E_core'   : float, nuclear repulsion + frozen-core energy offset
        'n_orb'    : int, number of active spatial orbitals = number of qubits
        'n_pairs'  : int, number of electron pairs in the (active) space
        'mf'       : the PySCF RHF object (for reference, e.g. mf.e_tot)
    """
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol).run()

    nuc = mol.energy_nuc()
    n_elec_total = mol.nelectron

    if active_space is None:
        # Full MO space, no frozen core
        n_act, n_elec_act = mf.mo_coeff.shape[1], n_elec_total
        mo_coeff_act = mf.mo_coeff
        E_core = nuc
    else:
        n_act, n_elec_act = active_space
        if n_elec_act % 2 != 0:
            raise ValueError(
                f"n_elec_act={n_elec_act} is odd; DOCI requires a closed-shell "
                f"(all-paired) active space with an even electron count."
            )
        n_core_elec = n_elec_total - n_elec_act
        if n_core_elec % 2 != 0 or n_core_elec < 0:
            raise ValueError(
                f"Invalid active space: {n_core_elec} core electrons "
                f"(must be even and >= 0)."
            )
        ncore = n_core_elec // 2  # frozen doubly-occupied core orbitals

        # Use CASCI machinery purely to get the frozen-core-corrected
        # one-/two-electron integrals and core energy in the active space.
        # (We don't need to solve the CI problem here -- DOCI does that.)
        mc = mcscf.CASCI(mf, n_act, n_elec_act)
        mc.ncore = ncore
        h1e_act, E_core = mc.get_h1eff()
        eri_act = mc.get_h2eff()
        eri_act = ao2mo.restore(1, eri_act, n_act)  # unpack to full (n,n,n,n)
        h1e = h1e_act
        eri = eri_act

    if active_space is None:
        h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
        eri = ao2mo.kernel(mf._eri, mf.mo_coeff, compact=False).reshape(
            n_act, n_act, n_act, n_act
        )

    n_pairs = n_elec_act // 2

    # ---- Build couplings (chemist notation eri[p,q,r,s] = (pq|rs)) --------
    eps = np.zeros(n_act)
    g = np.zeros((n_act, n_act))
    v = np.zeros((n_act, n_act))

    for a in range(n_act):
        eps[a] = 2 * h1e[a, a] + eri[a, a, a, a]

    for a in range(n_act):
        for b in range(n_act):
            if a != b:
                g[a, b] = eri[a, b, a, b]  # (ab|ab) exchange
                v[a, b] = 2 * (
                    2 * eri[a, a, b, b] - eri[a, b, a, b]
                )  # 2*[2(aa|bb)-(ab|ab)]

    if verbose:
        print(
            f"DOCI couplings  |  basis={basis}  |  n_orb={n_act}  "
            f"n_pairs={n_pairs}  |  E_core={E_core:+.8f} Ha"
        )
        print(f"  eps_a : {np.array2string(eps, precision=6, suppress_small=True)}")
        if n_act <= 8:
            print("  g_ab (pair hopping):")
            print(
                " ", np.array2string(g, precision=5, suppress_small=True, prefix="  ")
            )
            print("  v_ab (density-density):")
            print(
                " ", np.array2string(v, precision=5, suppress_small=True, prefix="  ")
            )

    return {
        "eps": eps,
        "g": g,
        "v": v,
        "E_core": E_core,
        "n_orb": n_act,
        "n_pairs": n_pairs,
        "mf": mf,
    }


def doci_energy_check(couplings, occ_list):
    """
    Quick physical-consistency check: evaluate the DOCI diagonal energy
    (no XX+YY hopping, i.e. a single pair-configuration determinant) for a
    given list of occupied orbital indices (length = n_pairs), and compare
    to mf.e_tot when occ_list = [0, 1, ..., n_pairs-1] (the RHF reference).

    This is the same check that caught the missing-self-energy bug in the
    original validation: the RHF-occupation diagonal element of H_DOCI must
    equal mf.e_tot exactly.
    """
    eps, v, E_core = couplings["eps"], couplings["v"], couplings["E_core"]
    diag_energy = E_core + sum(eps[a] for a in occ_list)
    diag_energy += sum(
        v[a, b] for i, a in enumerate(occ_list) for b in occ_list[i + 1 :]
    )  # sum a<b once; the factor of 2 from the i!=j double sum is already
    # folded into v_ab's definition (see module docstring)
    return diag_energy


if __name__ == "__main__":
    # Sanity check against RHF reference for a few molecules
    cases = [
        ("H2", "H 0 0 0; H 0 0 0.74", "sto-3g", None),
        ("LiH", "Li 0 0 0; H 0 0 1.595", "sto-3g", None),
        ("H4", "H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22", "sto-3g", None),
        ("H2O", "O 0 0 0; H 0.757 0.586 0; H -0.757 0.586 0", "sto-3g", (4, 4)),
    ]
    for name, atom, basis, act in cases:
        print("=" * 60)
        print(name)
        res = get_doci_couplings(atom, basis=basis, active_space=act)
        occ = list(range(res["n_pairs"]))
        e_check = doci_energy_check(res, occ)
        print(f"  RHF reference (mf.e_tot)      : {res['mf'].e_tot:+.8f}")
        if act is None:
            print(
                f"  DOCI diag @ RHF occupation     : {e_check:+.8f}  "
                f"(should match mf.e_tot)"
            )
        else:
            print(
                f"  DOCI diag @ active occupation  : {e_check:+.8f}  "
                f"(active-space E_core absorbs frozen core; compare to "
                f"mc.kernel() e_tot, not bare mf.e_tot)"
            )
        print()
