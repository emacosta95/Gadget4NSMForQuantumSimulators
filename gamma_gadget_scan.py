"""Gadget (no minor embedding) vs exact NSM: fidelity scan in gamma.

For every nucleus (shell x isospin channel x number of pairs) and every
encoding it saves

    gamma  ->  1 - F, 1 - F in code space, leakage, energy, energy error

and, at the gammas in --wf-gammas, the exact and gadget wavefunctions in the
orbital (logical) basis  (a, b, c, ...) -> psi(a, b, c, ...).

Encodings
---------
  pc   particle conservation (second quantization, D qubits,
       constraint gamma (sum N - N_tot)^2).  For N_tot > 1 the c_AB terms are
       switched off (they would sit on the logical diagonal); the second-order
       self-energy is then configuration-independent, sum_A d_A^2 / (2 gamma),
       so "exact" = "uniform" there.
  fq   first quantization (N registers x D qubits, hardcore penalty gamma2),
       with the exact second-order self-energy removal:
       theta_1(A) one-body + phi(A,B) inter-register two-body terms
       (src/gadget/selfenergy.py).  Exact at O(1/gamma) for any N.

Nuclei
------
  shell sd, isospin nn : O16 core  + N neutron pairs  -> O18, O20, ...
  shell pf, isospin nn : Ca40 core + N neutron pairs  -> Ca42, Ca44, ...
  isospin pp (proton pairs: Ne18..., Ti42...) is wired in but REFUSED by
  `check_isospin_data` until proper files exist: one_body_pp_sd.npz covers
  only levels 3-5, one_body_pp_pf.npz is identical to one_body_nn_pf.npz and
  twobody_pp_pf.npz does not exist.  Drop valid files in data/ and it runs.

Usage
-----
  python gadget_gamma_scan.py --shell sd pf --n-particles 1 2 3 \
         --encodings pc fq --gammas 1 1000 25 --wf-gammas 3 30 300
  python plot_gadget_gamma_scan.py            # figures from the saved .npz

Output: data/gamma_scan/<label>_<isospin>_<encoding>.npz
"""

from __future__ import annotations

import argparse
import os
import time
import warnings

import numpy as np

from src.gadget import (
    FirstQuantizationEncoding,
    GadgetHamiltonian,
    GadgetParameters,
    ParticleConservationEncoding,
    QuasiparticleProblem,
    alpha_from_c,
    fit_drive,
)

DATA = "data/matrix_elements_h_eff_2body"
SHELLS = {
    # shell: (n_levels, core mass, {isospin: core Z})
    "sd": (6, 16, {"nn": 8, "pp": 8}),
    "pf": (10, 40, {"nn": 20, "pp": 20}),
}
ELEMENTS = {
    8: "O",
    10: "Ne",
    12: "Mg",
    14: "Si",
    16: "S",
    18: "Ar",
    20: "Ca",
    22: "Ti",
    24: "Cr",
    26: "Fe",
    28: "Ni",
    30: "Zn",
    32: "Ge",
    34: "Se",
    36: "Kr",
    38: "Sr",
    40: "Zr",
}


# --------------------------------------------------------------------------
def nucleus_label(shell: str, isospin: str, n_pairs: int) -> str:
    D, A_core, Zc = SHELLS[shell]
    Z = Zc[isospin] + (2 * n_pairs if isospin == "pp" else 0)
    return f"{ELEMENTS.get(Z, f'Z{Z}')}{A_core + 2 * n_pairs}"


def check_isospin_data(shell: str, isospin: str) -> None:
    """Refuse channels whose matrix elements are missing or placeholders."""
    D = SHELLS[shell][0]
    one = f"{DATA}/one_body_{isospin}_{shell}.npz"
    two = f"{DATA}/twobody_{isospin}_{shell}.npz"
    problems = []
    if not os.path.exists(one):
        problems.append(f"{one} missing")
    else:
        keys = np.load(one)["keys"]
        diag = {int(i) for i, j in keys if i == j}
        if diag != set(range(D)):
            problems.append(f"{one}: diagonal only on levels {sorted(diag)}")
        if isospin != "nn":
            ref = np.load(f"{DATA}/one_body_nn_{shell}.npz")
            same = ref["keys"].shape == keys.shape and np.allclose(
                ref["values"], np.load(one)["values"]
            )
            if same:
                problems.append(f"{one} is identical to the nn file")
    if not os.path.exists(two):
        problems.append(f"{two} missing")
    elif np.load(two)["keys"][:, :2].max() >= D:
        problems.append(f"{two}: level indices >= {D} (offset convention?)")
    if problems:
        raise NotImplementedError(f"{shell}/{isospin}: " + "; ".join(problems))


def build_problem(shell: str, isospin: str, n_pairs: int) -> QuasiparticleProblem:
    D = SHELLS[shell][0]
    if not 1 <= n_pairs <= D - 1:
        raise ValueError(f"{shell}: n_pairs must be in 1..{D - 1}")
    check_isospin_data(shell, isospin)
    one = f"{DATA}/one_body_{isospin}_{shell}.npz"
    two = f"{DATA}/twobody_{isospin}_{shell}.npz"
    return QuasiparticleProblem.from_npz(
        one,
        two if n_pairs > 1 else None,
        n_levels=D,
        n_particles=n_pairs,
        label=nucleus_label(shell, isospin, n_pairs),
    )


def make_encoding(name: str, problem: QuasiparticleProblem):
    if name == "pc":
        return ParticleConservationEncoding(problem)
    if name == "fq":
        return FirstQuantizationEncoding(problem)
    raise ValueError(name)


def aligned(amps: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Real gadget amplitudes with the global phase fixed on the exact state."""
    ov = np.vdot(ref, amps)
    phase = ov / abs(ov) if abs(ov) > 0 else 1.0
    out = amps / phase
    if np.max(np.abs(out.imag)) > 1e-8:
        warnings.warn("gadget amplitudes are not real after phase fixing")
    return out.real


# --------------------------------------------------------------------------
def scan_one(problem, drive, encoding_name, gammas, wf_gammas, args):
    enc = make_encoding(encoding_name, problem)
    E_ex, psi_ex = problem.exact_groundstate()
    configs = np.array(problem.logical_basis(), dtype=int)

    keys = (
        "infidelity",
        "infidelity_code_space",
        "leakage",
        "energy",
        "rel_energy_error",
        "time_s",
    )
    res = {k: np.full(len(gammas), np.nan) for k in keys}
    psi_gad = np.full((len(wf_gammas), problem.dim), np.nan)
    psi_gad_raw = np.full_like(
        psi_gad, np.nan
    )  # not renormalised (norm^2 = 1 - leakage)

    for i, g in enumerate(gammas):
        prm = GadgetParameters(
            gamma=g,
            gamma2=None if args.gamma2_ratio is None else args.gamma2_ratio * g,
            self_energy=args.self_energy,
            use_c=not args.no_c,
        )
        H = GadgetHamiltonian(problem, enc, drive, prm)
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # pc N>1 exact->uniform
            try:
                E, psi = H.groundstate(solver=args.solver)
            except Exception as exc:  # noqa: BLE001
                print(f"    gamma={g:9.3f}  FAILED: {exc}")
                continue
        amps = enc.project(psi)
        energy = E * g
        F = abs(np.vdot(psi_ex, amps)) ** 2
        norm2 = float(np.sum(np.abs(amps) ** 2))
        res["infidelity"][i] = 1.0 - F
        res["infidelity_code_space"][i] = 1.0 - F / norm2
        res["leakage"][i] = 1.0 - norm2
        res["energy"][i] = energy
        res["rel_energy_error"][i] = abs(energy - E_ex) / abs(E_ex)
        res["time_s"][i] = time.time() - t0
        for k in np.nonzero(np.isclose(wf_gammas, g))[0]:
            a = aligned(amps, psi_ex)
            psi_gad_raw[k] = a
            psi_gad[k] = a / np.linalg.norm(a)
        print(
            f"    gamma={g:9.3f}  1-F={res['infidelity'][i]:.3e}  "
            f"leak={res['leakage'][i]:.2e}  dE/E={res['rel_energy_error'][i]:.2e}"
            f"  ({res['time_s'][i]:.1f}s)"
        )

    return dict(
        label=problem.label,
        encoding=encoding_name,
        shell=args._shell,
        isospin=args._isospin,
        n_pairs=problem.n_particles,
        n_levels=problem.n_levels,
        dim=problem.dim,
        n_qubits=enc.n_qubits,
        gammas=gammas,
        wf_gammas=wf_gammas,
        configs=configs,
        exact_energy=E_ex,
        psi_exact=psi_ex,
        psi_gadget=psi_gad,
        psi_gadget_raw=psi_gad_raw,
        d=drive.d,
        c_matrix=drive.c_matrix,
        self_energy=args.self_energy,
        cap=np.nan if drive.cap is None else drive.cap,
        gamma2_ratio=np.nan if args.gamma2_ratio is None else args.gamma2_ratio,
        **res,
    )


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--shell", nargs="+", default=["sd", "pf"], choices=list(SHELLS))
    ap.add_argument("--isospin", nargs="+", default=["nn"], choices=["nn", "pp"])
    ap.add_argument(
        "--n-particles",
        nargs="+",
        type=int,
        default=None,
        help="pairs; default: every N with n_qubits <= --max-qubits",
    )
    ap.add_argument(
        "--encodings", nargs="+", default=["pc", "fq"], choices=["pc", "fq"]
    )
    ap.add_argument(
        "--gammas",
        nargs=3,
        type=float,
        default=[1, 1000, 25],
        metavar=("MIN", "MAX", "NUM"),
        help="log-spaced grid",
    )
    ap.add_argument(
        "--wf-gammas",
        nargs="+",
        type=float,
        default=[3.0, 30.0, 300.0],
        help="gammas where the wavefunctions are stored (added to the grid)",
    )
    ap.add_argument(
        "--self-energy", default="exact", choices=["none", "uniform", "exact"]
    )
    ap.add_argument(
        "--gamma2-ratio",
        type=float,
        default=None,
        help="fq only: gamma2/gamma (default 1)",
    )
    ap.add_argument("--cap", default="4.0", help="|c_AB| cap of fit_drive, or 'none'")
    ap.add_argument("--no-c", action="store_true", help="pure rank-1 gadget (c_AB = 0)")
    ap.add_argument("--n-restarts", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-qubits", type=int, default=20)
    ap.add_argument("--solver", default="auto", choices=["auto", "scipy", "sparse"])
    ap.add_argument("--outdir", default="data/gamma_scan")
    args = ap.parse_args()

    cap = None if args.cap.lower() == "none" else float(args.cap)
    gmin, gmax, num = args.gammas
    wf_gammas = np.array(sorted(args.wf_gammas), dtype=float)
    gammas = np.union1d(
        np.logspace(np.log10(gmin), np.log10(gmax), int(num)), wf_gammas
    )
    os.makedirs(args.outdir, exist_ok=True)

    for shell in args.shell:
        D = SHELLS[shell][0]
        for iso in args.isospin:
            try:
                check_isospin_data(shell, iso)
            except NotImplementedError as exc:
                print(f"skip {exc}")
                continue
            args._shell, args._isospin = shell, iso
            drive = None  # g1 does not depend on N: fit once per (shell, isospin)
            pairs = args.n_particles or list(range(1, D))
            for N in pairs:
                if not 1 <= N <= D - 1:
                    continue
                p = build_problem(shell, iso, N)
                if drive is None:
                    drive = fit_drive(
                        p.g1,
                        n_restarts=args.n_restarts,
                        cap=cap,
                        use_c=not args.no_c,
                        seed=args.seed,
                    )
                    # coupling actually realised (capped c), not the fit's alpha
                    g_eff = -alpha_from_c(drive.c_matrix) * np.outer(drive.d, drive.d)
                    np.fill_diagonal(g_eff, 0.0)
                    res = np.abs(g_eff - p.g1).max()
                    print(
                        f"[{shell}/{iso}] drive fit (N=1 / fq): "
                        f"max |g_eff - g1| = {res:.2e}"
                    )
                for enc_name in args.encodings:
                    nq = D if enc_name == "pc" else N * D
                    if nq > args.max_qubits:
                        print(
                            f"  {p.label} {enc_name}: {nq} qubits > --max-qubits, skip"
                        )
                        continue
                    print(f"  {p.label} [{enc_name}]  N={N}  dim={p.dim}  qubits={nq}")
                    out = scan_one(p, drive, enc_name, gammas, wf_gammas, args)
                    path = os.path.join(args.outdir, f"{p.label}_{iso}_{enc_name}.npz")
                    np.savez(path, **out)
                    print(f"  -> {path}")


if __name__ == "__main__":
    main()
