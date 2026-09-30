"""
molecular_doci.py
=================
Seniority-zero (DOCI) molecular Hamiltonian in the qubit / hard-core-boson form of

    Elfving, Millaruelo, Gamez, Gogolin, "Simulating quantum chemistry in the
    seniority-zero space on qubit-based quantum computers",
    PRA 103, 032605 (2021), arXiv:2002.00035

    H = E_core + sum_p h_p N_p + sum_{p!=q} h1_pq P^dag_p P_q + sum_{p!=q} h2_pq N_p N_q

    P^dag_p = c^dag_{p up} c^dag_{p dn}   (electron pair = hard-core boson = qubit)
    h_p    = 2 h_pp + (pp|pp)
    h1_pq  = (pq|pq)                       exchange integral  (chemist notation)
    h2_pq  = 2 (pp|qq) - (pq|qp)

Mapped onto the project convention (src/gadget/problem.py, same as the NSM case)

    H_log = sum_A eps_A N_A + sum_{A<B} g1_AB (a^dag_A a_B + h.c.) + sum_{A<B} g2_AB N_A N_B
    eps = h ,   g1 = h1 ,   g2 = 2 h2      (p!=q double sum -> A<B single sum)

Everything is in Hartree; E_core (nuclear repulsion + frozen core) is kept apart.
Qubit form: a^dag_A a_B + h.c. = (X_A X_B + Y_A Y_B)/2,  N_A = (1 - Z_A)/2.

Main entry points
-----------------
    doci_couplings(atom, ...)        -> dict(h, g1, g2, E_core, n_orb, n_pairs, ...)
    doci_problem(atom, ...)          -> QuasiparticleProblem  (plugs into src.gadget)
    doci_ground_energy(c)            -> exact DOCI energy in the pair sector
    fci_benchmarks(atom, ...)        -> RHF, CASCI (same active space), full FCI
    bond_scan(builder, R_list, ...)  -> arrays for energy vs interatomic distance
"""

from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Callable, Optional, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from pyscf import ao2mo, fci, gto, mcscf, scf

# --------------------------------------------------------------- geometries


def diatomic(a: str, b: str, R: float) -> str:
    """PySCF atom string for a diatomic A-B at distance R (Angstrom) along z."""
    return f"{a} 0 0 0; {b} 0 0 {R}"


def lih(R: float) -> str:
    return diatomic("Li", "H", R)


# ------------------------------------------------------------- SCF helpers


def run_rhf(atom: str, basis: str = "sto-3g", charge: int = 0, dm0=None):
    """RHF with an optional density-matrix guess (continuity along a scan)."""
    mol = gto.M(atom=atom, basis=basis, charge=charge, spin=0, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-11
    mf.kernel(dm0=dm0)
    if not mf.converged:
        mf = mf.newton().run()
    return mol, mf


def default_active_space(mol, max_pairs: int = 3, n_orb: Optional[int] = None):
    """(n_act_orb, n_act_elec) with at most `max_pairs` pairs.

    Freezes the lowest doubly occupied orbitals until n_pairs <= max_pairs,
    then keeps `n_orb` active orbitals (default: all remaining).
    """
    n_mo = mol.nao_nr()
    n_pairs_tot = mol.nelectron // 2
    n_core = max(0, n_pairs_tot - max_pairs)
    n_pairs = n_pairs_tot - n_core
    n_act = n_mo - n_core if n_orb is None else n_orb
    if n_act < n_pairs or n_core + n_act > n_mo:
        raise ValueError(f"invalid active space: n_core={n_core}, n_act={n_act}")
    return n_act, 2 * n_pairs


# ----------------------------------------------------------- the couplings


def doci_couplings(
    atom: str,
    basis: str = "sto-3g",
    active_space: Optional[tuple] = None,
    max_pairs: int = 3,
    orbitals: str = "rhf",
    charge: int = 0,
    dm0=None,
):
    """Elfving et al. seniority-zero couplings for one geometry.

    Parameters
    ----------
    active_space : (n_act_orb, n_act_elec) or None
        None -> `default_active_space(mol, max_pairs)` (freeze core pairs only
        if the molecule has more than `max_pairs` pairs).
    orbitals : "rhf" | "natural"
        "rhf"     canonical RHF orbitals (dense all-to-all g1, DOCI error larger)
        "natural" natural orbitals of the CASCI 1-RDM in the active space
                  (cheap proxy of orbital-optimised DOCI, much lower DOCI error)

    Returns dict with  h (=eps), g1, g2  [project convention], h1, h2
    [Elfving convention], E_core, n_orb, n_pairs, n_core, mf, mo_act.
    """
    mol, mf = run_rhf(atom, basis, charge, dm0)
    if active_space is None:
        active_space = default_active_space(mol, max_pairs)
    n_act, n_elec = active_space
    if n_elec % 2:
        raise ValueError("DOCI needs an even number of active electrons")
    n_core = (mol.nelectron - n_elec) // 2

    mc = mcscf.CASCI(mf, n_act, n_elec)
    mc.ncore = n_core
    mo = mf.mo_coeff.copy()
    if orbitals == "natural":
        mc.kernel()
        # sort_mo not needed: cas_natorb rotates only the active block
        mo, _, _ = mc.cas_natorb(sort=True)
        mc = mcscf.CASCI(mf, n_act, n_elec)
        mc.ncore = n_core
    elif orbitals != "rhf":
        raise ValueError("orbitals must be 'rhf' or 'natural'")
    mc.mo_coeff = mo

    h1e, E_core = mc.get_h1eff(mo)  # core-dressed one-body, E_nuc + E_frozen
    mo_act = mo[:, n_core : n_core + n_act]
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mo_act), n_act)  # (pq|rs)

    J = np.einsum("ppqq->pq", eri)  # (pp|qq)
    K = np.einsum("pqpq->pq", eri)  # (pq|pq) = (pq|qp) for real orbitals
    h = 2 * np.diag(h1e) + np.diag(J)
    h1 = K.copy()
    h2 = 2 * J - K
    np.fill_diagonal(h1, 0.0)
    np.fill_diagonal(h2, 0.0)

    return dict(
        h=h,
        g1=h1.copy(),
        g2=2 * h2,
        h1=h1,
        h2=h2,
        E_core=float(E_core),
        n_orb=n_act,
        n_pairs=n_elec // 2,
        n_core=n_core,
        basis=basis,
        orbitals=orbitals,
        mf=mf,
        mol=mol,
        mo_act=mo_act,
        e_rhf=float(mf.e_tot),
    )


def doci_problem(
    atom: str, label: str = "", flip_sign: bool = False, center_eps: bool = False, **kw
):
    """Same as `doci_couplings` but returns a `QuasiparticleProblem` (+ the dict).

    flip_sign  : build H' = -H_DOCI  (eps, g1, g2 -> -eps, -g1, -g2).
                 g1' = -(AB|AB) < 0 on every edge -> stoquastic, rank-1 fit -d_A d_B works.
                 Spectrum is mirrored: ground state of H_DOCI = HIGHEST state of H'.
    center_eps : subtract mean(eps) (only adds N*const in the fixed-N sector),
                 keeps the one-body fields small w.r.t. gamma.

    Energy map (stored in c['energy_map']):  E_DOCI = sign * E_problem + offset
    """
    from src.gadget import QuasiparticleProblem

    c = doci_couplings(atom, **kw)
    s = -1.0 if flip_sign else 1.0
    shift = float(np.mean(c["h"])) if center_eps else 0.0
    prob = QuasiparticleProblem(
        eps=s * (c["h"] - shift),
        g1=s * c["g1"],
        g2=s * c["g2"],
        n_particles=c["n_pairs"],
        label=label,
    )
    c["energy_map"] = dict(sign=s, offset=c["E_core"] + c["n_pairs"] * shift)
    return prob, c


def to_doci_energy(E_problem, c: dict):
    """Problem-space energy -> total DOCI energy (undo flip, eps shift, add E_core)."""
    m = c.get("energy_map", dict(sign=1.0, offset=c["E_core"]))
    return m["sign"] * np.asarray(E_problem) + m["offset"]


# ------------------------------------------------------- exact DOCI (pair sector)


def doci_matrix(c: dict, include_core: bool = True):
    """Sparse DOCI Hamiltonian in the C(n_orb, n_pairs) hard-core-boson basis.

    Basis ordering = itertools.combinations (same as QuasiparticleProblem).
    """
    n, N = c["n_orb"], c["n_pairs"]
    h, g1, g2 = c["h"], c["g1"], c["g2"]
    configs = list(combinations(range(n), N))
    index = {cf: i for i, cf in enumerate(configs)}
    rows, cols, vals = [], [], []
    for I, cf in enumerate(configs):
        occ = set(cf)
        diag = h[list(cf)].sum() + sum(g2[a, b] for a, b in combinations(cf, 2))
        rows.append(I), cols.append(I), vals.append(diag)
        for b in cf:  # hop a pair b -> a (no fermionic sign for pairs)
            for a in range(n):
                if a in occ:
                    continue
                J = index[tuple(sorted((occ - {b}) | {a}))]
                rows.append(J), cols.append(I), vals.append(g1[a, b])
    H = sp.csr_matrix((vals, (rows, cols)), shape=(len(configs),) * 2)
    if include_core:
        H = H + c["E_core"] * sp.identity(len(configs), format="csr")
    return H, configs


def doci_ground_energy(c: dict, k: int = 1, return_state: bool = False):
    """Lowest k DOCI eigenvalues (total energy incl. E_core)."""
    H, configs = doci_matrix(c)
    if H.shape[0] <= 400:
        w, v = np.linalg.eigh(H.toarray())
    else:
        w, v = eigsh(H, k=k, which="SA")
        o = np.argsort(w)
        w, v = w[o], v[:, o]
    if return_state:
        return w[:k], v[:, :k], configs
    return w[0] if k == 1 else w[:k]


def reference_energy(c: dict) -> float:
    """Diagonal element on the aufbau configuration (0..N-1). = RHF if orbitals='rhf'."""
    occ = list(range(c["n_pairs"]))
    return (
        c["E_core"]
        + c["h"][occ].sum()
        + sum(c["g2"][a, b] for a, b in combinations(occ, 2))
    )


# ------------------------------------------------------------ FCI benchmarks


def fci_benchmarks(c: dict, full_fci: bool = True) -> dict:
    """RHF, CASCI (FCI in the SAME active space as DOCI) and full-space FCI."""
    mf = c["mf"]
    n_act, n_elec = c["n_orb"], 2 * c["n_pairs"]
    mc = mcscf.CASCI(mf, n_act, n_elec)
    mc.ncore = c["n_core"]
    mc.verbose = 0
    e_cas = mc.kernel()[0]
    out = dict(e_rhf=float(mf.e_tot), e_casci=float(e_cas))
    if full_fci:
        solver = fci.FCI(mf)
        solver.conv_tol = 1e-10
        out["e_fci"] = float(solver.kernel()[0])
    return out


def seniority_zero_projection_check(c: dict) -> float:
    """Independent DOCI check: take PySCF's FCI Hamiltonian in the active space,
    project it on determinants with alpha string == beta string, diagonalise.
    Only for tiny spaces (dimension C(n,N)^2)."""
    from pyscf.fci import cistring, direct_spin1

    mc = mcscf.CASCI(c["mf"], c["n_orb"], 2 * c["n_pairs"])
    mc.ncore = c["n_core"]
    mc.mo_coeff = np.hstack(
        [
            c["mf"].mo_coeff[:, : c["n_core"]],
            c["mo_act"],
            c["mf"].mo_coeff[:, c["n_core"] + c["n_orb"] :],
        ]
    )
    h1e, ecore = mc.get_h1eff()
    eri = mc.get_h2eff()
    n, N = c["n_orb"], c["n_pairs"]
    na = cistring.num_strings(n, N)
    h2 = direct_spin1.absorb_h1e(h1e, eri, n, (N, N), 0.5)
    idx = [i * na + i for i in range(na)]  # alpha == beta  <=> seniority zero
    Hs = np.zeros((na, na))
    for j, J in enumerate(idx):
        e = np.zeros(na * na)
        e[J] = 1.0
        Hs[:, j] = direct_spin1.contract_2e(h2, e.reshape(na, na), n, (N, N)).ravel()[
            idx
        ]
    return float(np.linalg.eigvalsh(Hs)[0] + ecore)


# ------------------------------------------------------------------- scans


def bond_scan(
    builder: Callable[[float], str],
    R_list: Sequence[float],
    basis: str = "sto-3g",
    active_space: Optional[tuple] = None,
    max_pairs: int = 3,
    orbitals: str = "rhf",
    full_fci: bool = True,
    keep_couplings: bool = True,
    verbose: bool = True,
) -> dict:
    """Energy vs interatomic distance: RHF, DOCI, CASCI, FCI (+ couplings per R).

    RHF density of the previous point is used as a guess (smooth curve).
    """
    keys = ["e_rhf", "e_doci", "e_casci", "e_fci"]
    res = {k: [] for k in keys}
    res["R"] = np.asarray(R_list, float)
    res["couplings"] = []
    dm = None
    for R in R_list:
        c = doci_couplings(
            builder(R),
            basis=basis,
            active_space=active_space,
            max_pairs=max_pairs,
            orbitals=orbitals,
            dm0=dm,
        )
        dm = c["mf"].make_rdm1()
        b = fci_benchmarks(c, full_fci=full_fci)
        e_doci = doci_ground_energy(c)
        res["e_rhf"].append(b["e_rhf"])
        res["e_doci"].append(e_doci)
        res["e_casci"].append(b["e_casci"])
        res["e_fci"].append(b.get("e_fci", np.nan))
        if keep_couplings:
            res["couplings"].append(
                {k: c[k] for k in ("h", "g1", "g2", "E_core", "n_orb", "n_pairs")}
            )
        if verbose:
            print(
                f"R={R:5.2f}  RHF={b['e_rhf']:+.6f}  DOCI={e_doci:+.6f}  "
                f"CASCI={b['e_casci']:+.6f}  FCI={b.get('e_fci', np.nan):+.6f}  "
                f"dDOCI-FCI={1e3*(e_doci-b.get('e_fci', np.nan)):7.2f} mHa"
            )
    for k in keys:
        res[k] = np.asarray(res[k])
    res["meta"] = dict(
        basis=basis,
        active_space=active_space,
        orbitals=orbitals,
        n_orb=c["n_orb"],
        n_pairs=c["n_pairs"],
        n_core=c["n_core"],
    )
    return res


# ------------------------------------------------------- sign / gauge helper


def best_sign_gauge(g1: np.ndarray):
    """Gauge P_A -> s_A P_A (s_A = +-1, i.e. orbital phase i) maps g1 -> s g1 s.

    The gadget produces a hopping -d_A d_B (c=0), i.e. the *stoquastic* sign.
    DOCI has g1 = (AB|AB) > 0 on every edge, which is non-stoquastic and, on any
    triangle, frustrated: no gauge makes all edges negative. This brute-forces
    (2^(n-1), fine for n <= 20) the gauge minimising the leftover positive weight
    (a max-cut problem).

    Returns s, g1_gauged, frustration = sum_{A<B} max(g,0) / sum_{A<B} |g|.
    Spectrum is gauge invariant; apply the same s to the eigenvectors.
    """
    n = g1.shape[0]
    iu = np.triu_indices(n, 1)
    best, s_best = np.inf, None
    for m in range(2 ** (n - 1)):
        s = np.array([1] + [1 - 2 * ((m >> k) & 1) for k in range(n - 1)])
        pos = np.clip((np.outer(s, s) * g1)[iu], 0, None).sum()
        if pos < best:
            best, s_best = pos, s
    gg = np.outer(s_best, s_best) * g1
    return s_best, gg, best / np.abs(g1[iu]).sum()
