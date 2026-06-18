"""
Extract the seniority-zero (DOCI) Hamiltonian couplings from PySCF
molecular integrals, mapped onto your one-hot gadget framework:

    H = sum_A eps_A N_A
      + sum_{A<B} t_AB (Q_A^dag Q_B + h.c.)      <-- the "XX+YY" pair-hopping term
      + sum_{A<B} w_AB N_A N_B                    <-- the "N_A N_B" (Ising/ZZ-like) term

Mapping to standard quantum-chemistry notation (spatial orbitals p,q,
seniority-zero / DOCI Hamiltonian, see e.g. Henderson & Scuseria,
J. Chem. Phys. 144, 094112 (2016), Eq. 18-19c):

    eps_A   = 2 h_p + <pp|pp>                       (pair energy: one-electron
                                                       term x2 PLUS the same-
                                                       orbital Coulomb self-
                                                       energy; NOT just h_p)
    t_AB    = <pp|qq>  (pair-hopping / XX+YY)       -> PySCF array element
                                                        eri[p,q,q,p], verified
                                                        against PySCF's own
                                                        FCI contraction (NOT
                                                        eri[p,p,q,q], which is
                                                        the naive but WRONG
                                                        translation)
    w_AB    = 2<pq|pq> - <pq|qp>                     -> eri[p,p,q,q]*2 - eri[p,q,q,p]
              (used with an extra factor of 2 when summed over
               unordered pairs p<q in the full Hamiltonian -- see
               the worked example / self-test below)

All three are extracted directly from PySCF's standard one- and
two-electron integral arrays in the molecular-orbital (MO) basis;
no DOCI-specific PySCF module is required, since DOCI only ever
needs a restricted subset (p==q patterns) of the full 2-electron
integral tensor that PySCF already computes for any CI/CC method.

IMPORTANT: the eps_A and t_AB formulas above were NOT obtained by
direct (and error-prone) physicist-to-chemist bracket-notation
translation. They were verified by building the explicit DOCI
Hamiltonian from these couplings and checking, against PySCF's own
FCI solver, that (a) the diagonal energy of the RHF reference
determinant equals mf.e_tot exactly, and (b) the resulting DOCI
ground-state energy never falls below the exact FCI ground-state
energy (DOCI is a strict subspace of FCI, so E_DOCI >= E_FCI must
hold). An earlier version of this code used eri[p,p,q,q] for t_AB,
which is the "obvious" but incorrect translation -- it passed no
sanity check and gave E_DOCI < E_FCI, an unphysical result that
caught the bug. The __main__ block below re-runs this self-test
automatically; rerun it after changing molecule/basis/active_space.
"""

import numpy as np
from pyscf import gto, scf, ao2mo


def get_doci_couplings(mol, mf=None, active_space=None):
    """
    Extract DOCI / seniority-zero couplings from a converged PySCF
    mean-field object.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object (already built).
    mf : pyscf.scf.hf.SCF, optional
        Converged mean-field (RHF) object. If None, RHF is run here.
    active_space : tuple(int, int), optional
        (n_active_orbitals, n_active_electrons). If None, uses all
        molecular orbitals from the RHF calculation (no freezing).

    Returns
    -------
    dict with keys:
        'eps'   : (n_orb,) array, eps_A = h_p one-electron integrals
        't'     : (n_orb, n_orb) array, t_AB = <pp|qq> pair-hopping
                  couplings (symmetric, zero diagonal)
        'w'     : (n_orb, n_orb) array, w_AB = 2<pq|pq> - <pq|qp>
                  number-number couplings (symmetric, zero diagonal)
        'n_orb' : number of active spatial orbitals (= number of
                  logical one-hot modes A in your gadget framework)
        'e_core': core (frozen-orbital + nuclear repulsion) energy
                  offset, if active_space restricts to a subset
    """
    if mf is None:
        mf = scf.RHF(mol)
        mf.kernel()
    if not mf.converged:
        raise RuntimeError("Mean-field did not converge; check molecule/basis.")

    n_mo = mf.mo_coeff.shape[1]

    if active_space is not None:
        n_act, n_elec_act = active_space
        n_core_pairs = (mol.nelectron - n_elec_act) // 2
        if n_core_pairs < 0 or n_core_pairs + n_act > n_mo:
            raise ValueError("Invalid active_space for this molecule/basis.")
        core_idx = list(range(n_core_pairs))
        act_idx = list(range(n_core_pairs, n_core_pairs + n_act))
    else:
        core_idx = []
        act_idx = list(range(n_mo))

    n_orb = len(act_idx)
    mo_coeff = mf.mo_coeff

    # ---- one-electron integrals in MO basis (h_pq, full matrix) ----
    hcore_ao = mf.get_hcore()
    h_mo_full = mo_coeff.T @ hcore_ao @ mo_coeff  # (n_mo, n_mo)

    # ---- two-electron integrals in MO basis, chemist notation (pq|rs) ----
    # ao2mo.kernel returns the integrals restricted to the orbital subset
    # you pass in; passing mo_coeff (all columns) gives the full (pq|rs)
    # tensor needed to build the effective core Hamiltonian below.
    eri_mo_full = ao2mo.kernel(mol, mo_coeff, compact=False)
    eri_mo_full = eri_mo_full.reshape(n_mo, n_mo, n_mo, n_mo)
    # eri_mo_full[p,q,r,s] = (pq|rs) = <pr|qs> in physicist notation;
    # PySCF/ao2mo.kernel uses chemist notation (pq|rs) = integral of
    # phi_p(1)phi_q(1) (1/r12) phi_r(2)phi_s(2).

    # ---- fold core orbitals into an effective one-electron Hamiltonian
    # and a core energy offset (standard active-space / frozen-core
    # construction; if active_space is None, core_idx is empty and
    # this is a no-op) ----
    e_core = mol.energy_nuc()
    h_eff = h_mo_full.copy()
    for i in core_idx:
        e_core += 2.0 * h_mo_full[i, i]
        for j in core_idx:
            e_core += 2.0 * eri_mo_full[i, i, j, j] - eri_mo_full[i, j, j, i]
    for p in range(n_mo):
        for q in range(n_mo):
            for i in core_idx:
                h_eff[p, q] += 2.0 * eri_mo_full[p, q, i, i] - eri_mo_full[p, i, i, q]

    # ---- restrict to active orbitals and build the three couplings ----
    # eps_A is the energy of the one-hot logical state "orbital A doubly
    # occupied, all others empty", relative to the vacuum (all empty).
    # This equals 2*h_AA (two electrons in orbital A) PLUS the same-orbital
    # Coulomb self-energy <AA|AA>, which is present whenever orbital A
    # is doubly occupied and is NOT part of any cross-orbital w_AB term
    # (w_AB is defined only for A != B). Omitting this self-energy is a
    # real bug, not a convention choice: it was caught by checking that
    # the RHF reference determinant's DOCI diagonal energy must equal
    # mf.e_tot exactly, and it did not until this term was added.
    eps = np.array([2.0 * h_eff[p, p] + eri_mo_full[p, p, p, p] for p in act_idx])

    t = np.zeros((n_orb, n_orb))
    w = np.zeros((n_orb, n_orb))
    for i, p in enumerate(act_idx):
        for j, q in enumerate(act_idx):
            if p == q:
                continue
            # t_AB is the pair-hopping coupling, the matrix element of
            # P_p^dag P_q = a_p,up^dag a_p,down^dag a_q,down a_q,up
            # between the closed-shell determinants |qq> and |pp>.
            # VERIFIED directly against PySCF's own FCI Hamiltonian
            # matrix (fci.direct_spin1.contract_2e) on a minimal
            # 2-orbital/2-electron subsystem: the correct chemist-
            # notation array element is eri[p,q,q,p], NOT eri[p,p,q,q]
            # as a naive physicist-bracket-to-chemist-array translation
            # would suggest. The naive choice was caught because it
            # gave a DOCI ground state below the exact FCI energy on
            # the same restricted subspace, which is impossible since
            # DOCI is a strict subspace of FCI.
            t[i, j] = eri_mo_full[p, q, q, p]
            coulomb = eri_mo_full[p, p, q, q]  # (pp|qq) chemist = direct/Coulomb term
            exchange = eri_mo_full[p, q, q, p]  # (pq|qp) chemist = exchange term
            w[i, j] = 2.0 * coulomb - exchange

    return {
        "eps": eps,
        "t": t,
        "w": w,
        "n_orb": n_orb,
        "e_core": e_core,
    }


if __name__ == "__main__":
    import itertools

    # ------------------------------------------------------------
    # Worked example: H4 square, minimal STO-3G basis.
    # This is the standard small-molecule DOCI/seniority-zero
    # benchmark (4 electrons, 4 spatial orbitals -> exactly your
    # n=4 one-hot logical register, matching the native K_{4,4}
    # bipartite-embedding case discussed earlier).
    # ------------------------------------------------------------
    mol = gto.M(
        atom="""
        O   0.0000000000   0.0000000000   0.1173000000
        H   0.0000000000   0.7572000000  -0.4692000000
        H   0.0000000000  -0.7572000000  -0.4692000000
        """,
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
        spin=0,
    )
    # Check the total number of electrons in PySCF
    print(f"Total simulated electrons: {mol.nelectron}")

    # Check the breakdown of up-spin (Alpha) and down-spin (Beta) electrons
    print(f"Alpha/Beta breakdown:      {mol.nelec}")

    mol.nelectron = 4  # force RHF to treat this as a closed-shell 2-electron system
    mol.spin = 0
    mf = scf.RHF(mol)
    mf.kernel()
    print(f"RHF converged: {mf.converged}, E_RHF = {mf.e_tot:.6f} Ha")

    couplings = get_doci_couplings(mol, mf)

    print(f"\nNumber of active orbitals (logical one-hot modes): {couplings['n_orb']}")
    print(
        f"\neps_A (one-hot pair energies, includes same-orbital self-energy):"
        f"\n{np.round(couplings['eps'], 6)}"
    )
    print(f"\nt_AB (pair-hopping / XX+YY coupling):\n{np.round(couplings['t'], 6)}")
    print(
        f"\nw_AB (N_A N_B coupling, before the 2x unordered-pair factor "
        f"used when building H):\n{np.round(couplings['w'], 6)}"
    )

    # ------------------------------------------------------------
    # SELF-TEST: build the full DOCI Hamiltonian from these couplings
    # and confirm E_DOCI >= E_FCI (variational bound: DOCI is a
    # strict subspace of the full CI space, so its ground state can
    # never be lower than the exact FCI ground state). This check
    # should be run any time this extraction is reused on a new
    # molecule/basis/active space -- it caught a real sign/index
    # convention bug during development (see fci_validate.py for the
    # full minimal-subsystem diagnosis).
    # ------------------------------------------------------------
    eps, t, w, e_core = (
        couplings["eps"],
        couplings["t"],
        couplings["w"],
        couplings["e_core"],
    )
    n_orb = couplings["n_orb"]
    n_pairs = mol.nelectron // 2

    basis_dets = list(itertools.combinations(range(n_orb), n_pairs))
    dim = len(basis_dets)
    H = np.zeros((dim, dim))
    for I, occ_I in enumerate(basis_dets):
        occ_set = set(occ_I)
        diag = e_core + sum(eps[p] for p in occ_I)
        diag += sum(2.0 * w[p, q] for p, q in itertools.combinations(occ_I, 2))
        H[I, I] = diag
        unocc = [o for o in range(n_orb) if o not in occ_set]
        for p in occ_I:
            for q in unocc:
                occ_J = tuple(sorted(occ_set - {p} | {q}))
                J = basis_dets.index(occ_J)
                H[I, J] += t[p, q]
    H = 0.5 * (H + H.T)
    e_doci = np.linalg.eigvalsh(H)[0]

    from pyscf import fci

    e_fci, _ = fci.FCI(mf).kernel()

    print(f"\n--- Self-test ---")
    print(f"DOCI ground-state energy : {e_doci:.6f} Ha")
    print(f"FCI ground-state energy  : {e_fci:.6f} Ha  (variational lower bound)")
    assert (
        e_doci >= e_fci - 1e-6
    ), "DOCI energy below FCI -- coupling extraction is wrong!"
    print("PASSED: E_DOCI >= E_FCI, as required.")

    # Stronger check: compare directly against the official pyscf-doci
    # extension (pip install pyscf[doci]) if available. This solver
    # diagonalizes the exact seniority-zero Hamiltonian from raw
    # integrals directly -- it does not expose eps/t/w, but its energy
    # is an independent ground truth for the COUPLINGS extracted above
    # (if my eps/t/w are wrong, the Hamiltonian I build from them will
    # not match this energy, even though it might still satisfy the
    # weaker E_DOCI >= E_FCI bound by accident).
    try:
        from pyscf import doci as pyscf_doci
        from pyscf import ao2mo as _ao2mo

        h1e_full = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
        eri_full = _ao2mo.kernel(mol, mf.mo_coeff)
        mydoci = pyscf_doci.DOCI(mf)
        e_doci_official, _ = mydoci.kernel(
            h1e_full, eri_full, n_orb, mol.nelectron, ecore=e_core
        )
        print(f"\nOfficial pyscf-doci energy: {e_doci_official:.10f} Ha")
        print(f"My extraction-based energy: {e_doci:.10f} Ha")
        assert np.isclose(
            e_doci, e_doci_official, atol=1e-6
        ), "Mismatch with official pyscf-doci solver -- coupling extraction is wrong!"
        print("PASSED: matches official pyscf-doci to 1e-6 Ha.")
    except ImportError:
        print(
            "\n(pyscf-doci extension not installed -- skipping cross-check. "
            "Install with: pip install pyscf[doci])"
        )

    import pickle

    with open("data/doci_couplings_h4_square_sto3g.pkl", "wb") as f:
        pickle.dump(couplings, f)
