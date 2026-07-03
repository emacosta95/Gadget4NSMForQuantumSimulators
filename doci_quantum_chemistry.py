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


def get_doci_couplings(
    atom,
    basis="sto-3g",
    active_space=None,
    charge=0,
    orb_type="rhf",  # "rhf" | "boys" | "pipek"
    verbose=True,
):
    """
    Compute the DOCI gadget couplings (eps_a, g_ab, v_ab) for any molecule.

    Parameters
    ----------
    atom : str
        PySCF atom spec.
    basis : str
        Basis set name (default 'sto-3g').
    active_space : tuple(int, int) or None
        (n_act_orbitals, n_act_electrons). If None, uses all MOs.
    charge : int
        Molecular charge (default 0). Use charge=2 for H4²⁺ etc.
    orb_type : str
        Orbital basis for the integral transform:
          "rhf"   — canonical RHF orbitals (default, original behaviour)
          "boys"  — Boys localized orbitals (Foster-Boys)
          "pipek" — Pipek-Mezey localized orbitals
        Localization is applied to ALL MOs when active_space=None,
        or to ACTIVE MOs only when active_space is given.
        NOTE: h1e from mc.get_h1eff() already encodes the frozen core;
        only the active MO columns are rotated, leaving the core untouched.
    verbose : bool
        Print a short summary table.

    Returns
    -------
    dict with keys:
        'eps', 'g', 'v', 'E_core', 'n_orb', 'n_pairs', 'mf', 'mo_coeff_used'
    """
    from pyscf import gto, scf, ao2mo, mcscf, lo

    mol = gto.M(atom=atom, basis=basis, verbose=0, charge=charge)
    mf = scf.RHF(mol).run()

    nuc = mol.energy_nuc()
    n_elec_total = mol.nelectron

    if active_space is None:
        n_act, n_elec_act = mf.mo_coeff.shape[1], n_elec_total
        E_core = nuc
        mo_act = mf.mo_coeff  # all MOs, shape (nao, n_act)
        h1e = mo_act.T @ mf.get_hcore() @ mo_act  # one-body in MO basis
    else:
        n_act, n_elec_act = active_space
        if n_elec_act % 2 != 0:
            raise ValueError(
                f"n_elec_act={n_elec_act} is odd; DOCI requires even electron count."
            )
        n_core_elec = n_elec_total - n_elec_act
        if n_core_elec % 2 != 0 or n_core_elec < 0:
            raise ValueError(
                f"Invalid active space: {n_core_elec} core electrons "
                f"(must be even and >= 0)."
            )
        ncore = n_core_elec // 2

        mc = mcscf.CASCI(mf, n_act, n_elec_act)
        mc.ncore = ncore
        h1e, E_core = mc.get_h1eff()  # core-corrected 1e integrals
        mo_act = mf.mo_coeff[:, ncore : ncore + n_act]  # active MO columns only

    # ---- Optional orbital localization (applied to active MOs only) --------
    if orb_type == "boys":
        localizer = lo.Boys(mol, mo_act)
        mo_loc = localizer.kernel()
        # rotate h1e into localized basis (only needed for active_space=None)
        if active_space is None:
            h1e = mo_loc.T @ mf.get_hcore() @ mo_loc
        else:
            # h1e from get_h1eff() is in the *original* active MO basis;
            # rotate it into the localized basis
            # U: rotation matrix among active orbitals, mo_loc = mo_act @ U
            U = mo_act.T @ mol.intor("int1e_ovlp") @ mo_loc  # overlap-based rotation
            h1e = U.T @ h1e @ U
        mo_act = mo_loc

    elif orb_type == "pipek":
        localizer = lo.PipekMezey(mol, mo_act)
        mo_loc = localizer.kernel()
        if active_space is None:
            h1e = mo_loc.T @ mf.get_hcore() @ mo_loc
        else:
            U = mo_act.T @ mol.intor("int1e_ovlp") @ mo_loc
            h1e = U.T @ h1e @ U
        mo_act = mo_loc

    elif orb_type != "rhf":
        raise ValueError(
            f"orb_type must be 'rhf', 'boys', or 'pipek', got {orb_type!r}"
        )

    # ---- Two-electron integrals in the (possibly localized) active MO basis -
    eri = ao2mo.kernel(mf._eri, mo_act, compact=False).reshape(
        n_act, n_act, n_act, n_act
    )

    n_pairs = n_elec_act // 2

    # ---- Build couplings (chemist notation eri[p,q,r,s] = (pq|rs)) ---------
    eps = np.zeros(n_act)
    g = np.zeros((n_act, n_act))
    v = np.zeros((n_act, n_act))

    for a in range(n_act):
        eps[a] = 2 * h1e[a, a] + eri[a, a, a, a]

    for a in range(n_act):
        for b in range(n_act):
            if a != b:
                g[a, b] = eri[a, b, a, b]
                v[a, b] = 2 * (2 * eri[a, a, b, b] - eri[a, b, a, b])

    if verbose:
        print(
            f"DOCI couplings  |  basis={basis}  |  orb_type={orb_type}  "
            f"|  n_orb={n_act}  n_pairs={n_pairs}  |  E_core={E_core:+.8f} Ha"
        )
        print(f"  eps_a : {np.array2string(eps, precision=6, suppress_small=True)}")
        if n_act <= 8:
            print("  g_ab (pair hopping):")
            print("  ", np.array2string(g, precision=5, suppress_small=True))
            print("  v_ab (density-density):")
            print("  ", np.array2string(v, precision=5, suppress_small=True))

    return {
        "eps": eps,
        "g": g,
        "v": v,
        "E_core": E_core,
        "n_orb": n_act,
        "n_pairs": n_pairs,
        "mf": mf,
        "mo_coeff_used": mo_act,  # useful for visualization
    }


def extract_doci_dicts(
    atom, basis="sto-3g", active_space=None, verbose=True, charge=0, orb_type="rhf"
):
    """
    Extract DOCI couplings for a molecule and return g_AB and v_AB as
    dictionaries keyed by orbital pair (a, b) with a < b.

    Returns
    -------
    g_dict : dict  { (a,b): g_ab }   pair-hopping (exchange integral)
    v_dict : dict  { (a,b): v_ab }   density-density coupling
    eps    : np.ndarray  shape (n_orb,)   on-site energies
    E_core : float                        nuclear repulsion + frozen-core energy
    meta   : dict   { 'n_orb', 'n_pairs', 'mf' }
    """
    res = get_doci_couplings(
        atom,
        basis=basis,
        active_space=active_space,
        verbose=verbose,
        charge=charge,
        orb_type=orb_type,
    )

    n = res["n_orb"]
    g_dict = {}
    v_dict = {}

    for a in range(n):
        for b in range(a + 1, n):
            g_val = res["g"][a, b]
            v_val = res["v"][a, b]
            if g_val != 0.0:
                g_dict[(a, b)] = g_val
            if v_val != 0.0:
                v_dict[(a, b)] = v_val

    if verbose:
        print(f"\ng_AB (pair hopping):")
        for k, v in g_dict.items():
            print(f"  g{k} = {v:+.8f}")
        print(f"\nv_AB (density-density):")
        for k, val in v_dict.items():
            print(f"  v{k} = {val:+.8f}")

    return (
        g_dict,
        v_dict,
        res["eps"],
        res["E_core"],
        {
            "n_orb": res["n_orb"],
            "n_pairs": res["n_pairs"],
            "mf": res["mf"],
        },
    )


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
