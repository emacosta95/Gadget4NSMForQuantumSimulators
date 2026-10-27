"""LiH (DOCI) on the gadget: the two routes around the non-stoquastic sign.

DOCI has g1_AB = (AB|AB) > 0 on every edge, while the rank-1 gadget produces
the stoquastic sign -d_A d_B.  Two ways out:

  flip   H' = -H_DOCI.  Every edge becomes negative, the fit is exact, but the
         molecular ground state is the TOP state of the code block: fine for
         a coherent (top-state following) protocol, fragile for an open
         system / thermal annealer.
  trunc  stoquastic truncation.  Gauge the orbitals (P_A -> s_A P_A, max-cut,
         `best_sign_gauge`) so that as much g1 weight as possible is negative,
         then DROP the frustrated (still positive) edges.  The target is now a
         true ground state; the price is the truncation error of the dropped
         edges (which grows with the bond length).

Both routes in both encodings, particle conservation (pc, D qubits, c_AB = 0)
and first quantization (fq, N D qubits, bounded c_AB).  Fidelities and energy
errors are always measured against the EXACT DOCI ground state (of the full,
untruncated Hamiltonian) at the same geometry.

Usage
-----
  python lih_gadget_routes.py                 # gamma scan at R = 1.596 A
  python lih_gadget_routes.py --dissociation  # + energy vs R for both routes
  python lih_gadget_routes.py --plot-only     # figures from the saved .npz

Outputs: data/lih_routes/*.npz, figures/lih_routes/*.pdf
"""

from __future__ import annotations

import argparse
import os
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.gadget import (  # noqa: E402
    FirstQuantizationEncoding,
    GadgetHamiltonian,
    GadgetParameters,
    ParticleConservationEncoding,
    QuasiparticleProblem,
    fit_drive,
    fit_drive_bounded,
)
from src.molecular_doci import (  # noqa: E402
    best_sign_gauge,
    doci_couplings,
    doci_ground_energy,
    fci_benchmarks,
    lih,
)

CHEM_ACC = 1.6e-3  # Hartree
OUTDIR, FIGDIR = "data/lih_routes", "figures/lih_routes"
ROUTES = ["flip", "trunc"]
ENCODINGS = ["pc", "fq"]
STYLE = {  # (colour, linestyle) per (route, encoding)
    ("flip", "pc"): ("C0", "-"),
    ("flip", "fq"): ("C3", "-"),
    ("trunc", "pc"): ("C0", "--"),
    ("trunc", "fq"): ("C3", "--"),
}
LABEL = {
    ("flip", "pc"): "flip, PCE",
    ("flip", "fq"): "flip, FQE",
    ("trunc", "pc"): "stoquastic truncation, PCE",
    ("trunc", "fq"): "stoquastic truncation, FQE",
}


# ------------------------------------------------------------ the problems
def build_route(c: dict, route: str):
    """(QuasiparticleProblem, info) for one route from the DOCI couplings `c`.

    Energies of the problem map to DOCI energies as  E_DOCI = sign * E + offset.
    The exact DOCI ground state is returned in the problem's (gauged) basis.
    """
    h, g1, g2, N = c["h"], c["g1"], c["g2"], c["n_pairs"]
    shift = float(np.mean(h))  # only adds N*const in the fixed-N sector
    E_ex, V, configs = doci_ground_energy(c, k=1, return_state=True)
    E_ex = float(np.atleast_1d(E_ex)[0])
    psi_ex = V[:, 0]
    if route == "flip":
        prob = QuasiparticleProblem(
            eps=-(h - shift), g1=-g1, g2=-g2, n_particles=N, label="LiH"
        )
        info = dict(sign=-1.0, offset=c["E_core"] + N * shift, which="top")
        psi_target = psi_ex
        E_trunc = E_ex
    elif route == "trunc":
        s, g1g, frustration = best_sign_gauge(g1)
        g1t = np.minimum(g1g, 0.0)  # drop the frustrated edges
        prob = QuasiparticleProblem(
            eps=h - shift, g1=g1t, g2=g2, n_particles=N, label="LiH"
        )
        info = dict(
            sign=1.0,
            offset=c["E_core"] + N * shift,
            which="bottom",
            gauge=s,
            frustration=frustration,
            dropped=int(np.sum(g1g[np.triu_indices(len(h), 1)] > 0)),
        )
        # exact state in the gauged basis: amplitude of (A, B, ...) picks s_A s_B ...
        signs = np.array([np.prod(s[list(cf)]) for cf in configs])
        psi_target = signs * psi_ex
        # ground state of the truncated logical Hamiltonian (truncation error alone)
        w, _ = prob.exact_spectrum()
        E_trunc = info["sign"] * w[0] + info["offset"]
    else:
        raise ValueError(route)
    info.update(E_exact=float(E_ex), E_truncated_logical=float(E_trunc))
    return prob, psi_target, info


D_MAX = 1.0  # |d_A| bound of the drive fits (g1 ~ 0.01-0.1 Ha, so d ~ 0.1-0.3)


def make_gadget(prob, encoding, gamma, c_max=4.0, c_min=None, seed=42):
    if encoding == "pc":
        drive = fit_drive(prob.g1, n_restarts=50, use_c=False, seed=seed, d_max=D_MAX)
        enc = ParticleConservationEncoding(prob)
    else:
        drive = fit_drive_bounded(prob.g1, c_max=c_max, c_min=c_min, seed=seed, d_max=D_MAX)
        enc = FirstQuantizationEncoding(prob)
    return GadgetHamiltonian(prob, enc, drive, GadgetParameters(gamma=gamma)), drive


def code_block_state(H: GadgetHamiltonian, which: str, max_flips: int = 3):
    """(E, psi) of the lowest ('bottom') or highest ('top') eigenstate of the
    gadget that lives in the code space (weight > 1/2 on the symmetric code
    vectors).  Dense diagonalisation in the space within `max_flips` of the
    code space (exact to O((d/gamma)^(max_flips+1)))."""
    keep = np.asarray(H.truncated_space(max_flips))
    Ht = H.total().data.as_scipy().tocsr()[keep][:, keep].toarray()
    w, U = np.linalg.eigh(Ht)
    enc = H.encoding
    full = np.zeros((enc.dim, U.shape[1]))
    full[keep] = U
    weight = np.array([1.0 - enc.leakage(full[:, k]) for k in range(U.shape[1])])
    phys = np.nonzero(weight > 0.5)[0]
    k = phys[-1] if which == "top" else phys[0]
    return float(w[k]), full[:, k]


def evaluate(c, route, encoding, gamma, c_min=None):
    prob, psi_target, info = build_route(c, route)
    H, drive = make_gadget(prob, encoding, gamma, c_min=c_min)
    E, psi = code_block_state(H, info["which"])
    enc = H.encoding
    E_doci = info["sign"] * E * gamma + info["offset"]
    return dict(
        infidelity=1.0 - enc.fidelity(psi, psi_target, normalize=False),
        leakage=enc.leakage(psi),
        energy=E_doci,
        energy_error=E_doci - info["E_exact"],
        truncation_error=info["E_truncated_logical"] - info["E_exact"],
        n_qubits=enc.n_qubits,
        c_range=(
            (np.nan, np.nan)
            if encoding == "pc"
            else (
                drive.c_matrix[~np.eye(len(drive.d), dtype=bool)].min(),
                drive.c_matrix[~np.eye(len(drive.d), dtype=bool)].max(),
            )
        ),
        **{k: v for k, v in info.items() if k in ("frustration", "dropped")},
    )


# ------------------------------------------------------------------ scans
def gamma_scan(args):
    c = doci_couplings(lih(args.R), basis=args.basis, max_pairs=10)
    gammas = np.logspace(np.log10(args.gammas[0]), np.log10(args.gammas[1]), int(args.gammas[2]))
    print(
        f"LiH {args.basis} R={args.R} A: {c['n_orb']} orbitals, {c['n_pairs']} pairs, "
        f"E_DOCI={doci_ground_energy(c):.6f} Ha"
    )
    out = dict(R=args.R, basis=args.basis, gammas=gammas, E_exact=doci_ground_energy(c))
    for route in ROUTES:
        for enc_name in ENCODINGS:
            key = f"{route}_{enc_name}"
            inf, err, leak = [], [], []
            c_min = args.c_min_trunc if (route == "trunc" and enc_name == "fq") else None
            t0 = time.time()
            for g in gammas:
                r = evaluate(c, route, enc_name, g, c_min=c_min)
                inf.append(r["infidelity"]), err.append(r["energy_error"]), leak.append(r["leakage"])
            out[f"{key}_infidelity"] = np.array(inf)
            out[f"{key}_energy_error"] = np.array(err)
            out[f"{key}_leakage"] = np.array(leak)
            out[f"{key}_n_qubits"] = r["n_qubits"]
            if route == "trunc":
                out["truncation_error"] = r["truncation_error"]
                out["frustration"] = r["frustration"]
                out["dropped_edges"] = r["dropped"]
            extra = "" if enc_name == "pc" else f"  c in [{r['c_range'][0]:.3f}, {r['c_range'][1]:.3f}]"
            print(
                f"  {LABEL[(route, enc_name)]:32s} ({r['n_qubits']:2d} qubits): "
                f"1-F = {inf[np.argmin(abs(gammas-20))]:.2e} @20, {inf[-1]:.2e} @{gammas[-1]:g}; "
                f"dE = {1e3*err[np.argmin(abs(gammas-20))]:+.2f} mHa @20, {1e3*err[-1]:+.2f} mHa @{gammas[-1]:g}"
                f"{extra}  [{time.time()-t0:.0f}s]"
            )
    if "truncation_error" in out:
        print(
            f"  truncation alone: dropped {out['dropped_edges']} edges "
            f"(frustration {out['frustration']:.3f}), dE = {1e3*out['truncation_error']:+.2f} mHa"
        )
    path = f"{OUTDIR}/gamma_scan_R{args.R:g}.npz"
    np.savez(path, **out)
    print("->", path)
    return out


def dissociation(args):
    Rs = np.linspace(args.R_range[0], args.R_range[1], int(args.R_range[2]))
    out = dict(R=Rs, gammas=np.array(args.diss_gammas), basis=args.basis)
    keys = [(ro, en) for ro in args.routes for en in ENCODINGS]
    for k in ["e_rhf", "e_doci", "e_fci", "e_trunc_logical"]:
        out[k] = []
    for ro, en in keys:
        for g in args.diss_gammas:
            out[f"{ro}_{en}_g{g:g}_energy"] = []
            out[f"{ro}_{en}_g{g:g}_infidelity"] = []
    dm = None
    for R in Rs:
        t0 = time.time()
        c = doci_couplings(lih(R), basis=args.basis, max_pairs=10, dm0=dm)
        dm = c["mf"].make_rdm1()
        b = fci_benchmarks(c)
        out["e_rhf"].append(b["e_rhf"]), out["e_doci"].append(doci_ground_energy(c))
        out["e_fci"].append(b["e_fci"])
        line = f"R={R:5.3f}  DOCI={out['e_doci'][-1]:+.6f}  FCI={b['e_fci']:+.6f}"
        for ro, en in keys:
            for g in args.diss_gammas:
                c_min = args.c_min_trunc if (ro == "trunc" and en == "fq") else None
                r = evaluate(c, ro, en, g, c_min=c_min)
                out[f"{ro}_{en}_g{g:g}_energy"].append(r["energy"])
                out[f"{ro}_{en}_g{g:g}_infidelity"].append(r["infidelity"])
                if ro == "trunc":
                    e_tr = r["truncation_error"] + out["e_doci"][-1]
            line += f"  {ro}-{en}@{g:g}: {1e3*r['energy_error']:+.2f}mHa"
        out["e_trunc_logical"].append(e_tr)
        print(line + f"  [{time.time()-t0:.0f}s]", flush=True)
    for k, v in out.items():
        if isinstance(v, list):
            out[k] = np.array(v)
    path = f"{OUTDIR}/dissociation{args.tag}.npz"
    np.savez(path, **out)
    print("->", path)
    return out


# ---------------------------------------------------------------- figures
def plot_gamma_scan(d, path):
    g = d["gammas"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for (ro, en), (col, ls) in STYLE.items():
        k = f"{ro}_{en}"
        if f"{k}_infidelity" not in d:
            continue
        lab = f"{LABEL[(ro, en)]} ({int(d[f'{k}_n_qubits'])} qubits)"
        axes[0].plot(g, d[f"{k}_infidelity"], color=col, ls=ls, marker="o", ms=3.5, label=lab)
        axes[1].plot(g, np.abs(d[f"{k}_energy_error"]), color=col, ls=ls, marker="o", ms=3.5)
    axes[1].axhline(CHEM_ACC, color="gray", ls=":", lw=2, label="chemical accuracy")
    if "truncation_error" in d:
        axes[1].axhline(
            abs(float(d["truncation_error"])), color="k", ls="-.", lw=1,
            label="truncation alone",
        )
    for ax in axes:
        ax.set_xscale("log"), ax.set_yscale("log")
        ax.set_xlabel(r"$\gamma / \omega$", fontsize=13)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel(r"$1 - F$", fontsize=13)
    axes[1].set_ylabel(r"$|E_{\rm gadget} - E_{\rm DOCI}|$ [Ha]", fontsize=13)
    axes[0].set_title("(a)", loc="left", fontweight="bold")
    axes[1].set_title("(b)", loc="left", fontweight="bold")
    axes[0].legend(fontsize=8, framealpha=0.9)
    axes[1].legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("->", path)


def plot_dissociation(d, path):
    """2 x 3 grid: rows = encoding (PCE, FQE), columns = energy, |E - E_DOCI|,
    1 - F vs R.  Colour = gamma, solid = no flip (stoquastic truncation),
    dashed = flip (H -> -H, top state)."""
    R, gammas = d["R"], d["gammas"]
    colours = ["C0", "C3", "C2", "C1"]
    route_ls = {"trunc": "-", "flip": "--"}
    route_lab = {"trunc": "no flip (stoquastic truncation)", "flip": "flip ($H \\to -H$)"}
    enc_lab = {"pc": "PCE", "fq": "FQE"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharex=True)
    letters = iter("abcdef")
    for i, en in enumerate(ENCODINGS):
        ax_e, ax_err, ax_f = axes[i]
        ax_e.plot(R, d["e_fci"], color="k", lw=2, label="FCI")
        ax_e.plot(R, d["e_doci"], color="0.55", lw=2, label="DOCI")
        ax_err.axhline(CHEM_ACC, color="0.5", ls=":", lw=2, label="chemical accuracy")
        ax_err.plot(R, np.abs(d["e_trunc_logical"] - d["e_doci"]), color="k", ls="-.", lw=1.2,
                    label="truncation alone")
        for g, col in zip(gammas, colours):
            for ro in ROUTES:
                k = f"{ro}_{en}_g{g:g}"
                if f"{k}_energy" not in d:
                    continue
                kw = dict(color=col, ls=route_ls[ro], lw=1.6, marker="o", ms=3.5)
                ax_e.plot(R, d[f"{k}_energy"], **kw)
                ax_err.plot(R, np.abs(d[f"{k}_energy"] - d["e_doci"]), **kw)
                ax_f.plot(R, d[f"{k}_infidelity"], **kw)
        nq = int(d.get(f"{en}_n_qubits", 0)) if f"{en}_n_qubits" in d else None
        for ax in axes[i]:
            ax.set_title(f"({next(letters)})", loc="left", fontweight="bold", fontsize=13)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=11)
        ax_e.text(0.97, 0.95, enc_lab[en], transform=ax_e.transAxes, ha="right", va="top",
                  fontsize=14, fontweight="bold")
        ax_e.set_ylabel("energy [Ha]", fontsize=13)
        ax_err.set_ylabel(r"$|E_{\rm gadget} - E_{\rm DOCI}|$ [Ha]", fontsize=13)
        ax_f.set_ylabel(r"$1 - F$", fontsize=13)
        ax_err.set_yscale("log")
        ax_f.set_yscale("log")
    for ax in axes[1]:
        ax.set_xlabel(r"$R$ [$\AA$]", fontsize=13)
    handles = [Line2D([], [], color=col, lw=1.6, label=rf"$\gamma = {g:g}\,\omega$") for g, col in zip(gammas, colours)]
    handles += [Line2D([], [], color="k", ls=route_ls[ro], lw=1.6, label=route_lab[ro]) for ro in ROUTES]
    axes[0, 0].legend(fontsize=9, loc="lower right", framealpha=0.9)
    axes[0, 1].legend(fontsize=9, loc="lower right", framealpha=0.9)
    axes[0, 2].legend(handles=handles, fontsize=9, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("->", path)


def plot_dissociation_notebook(d, route, path):
    """The MolecularDOCI.ipynb 6-panel layout for ONE route: rows PCE / FQE,
    columns energy, |E_G - E_DOCI|, 1 - F (+ DOCI - FCI on a twin axis)."""
    R, gammas = d["R"], d["gammas"]
    colors = ["red", "green", "violet", "orange"][: len(gammas)]
    fig, axs = plt.subplots(2, 3, figsize=(24, 15), constrained_layout=True)
    for row, en in enumerate(ENCODINGS):
        ax_a, ax_b, ax_c = axs[row]
        ax_a.plot(R, d["e_fci"], color="black", lw=3, label="FCI")
        ax_a.plot(R, d["e_doci"], color="blue", lw=3, label="DOCI")
        if route == "trunc":
            ax_a.plot(R, d["e_trunc_logical"], color="blue", lw=2, ls=":", label="DOCI, truncated")
        ax_b.axhline(CHEM_ACC, color="gray", ls=":", lw=4, label="chemical accuracy")
        if route == "trunc":
            ax_b.plot(R, np.abs(d["e_trunc_logical"] - d["e_doci"]), color="black", lw=2, ls="-.",
                      label="truncation alone")
        for col, g in zip(colors, gammas):
            k = f"{route}_{en}_g{g:g}"
            if f"{k}_energy" not in d:
                continue
            E_g, F_g = d[f"{k}_energy"], 1.0 - d[f"{k}_infidelity"]
            ax_a.plot(R, E_g, ls="--", color=col)
            ax_a.scatter(R, E_g, color=col)
            err = np.abs(E_g - d["e_doci"])
            ax_b.plot(R, err, ls="--", color=col)
            ax_b.scatter(R, err, color=col)
            ax_c.plot(R, 1 - F_g, ls="--", color=col)
            ax_c.scatter(R, 1 - F_g, color=col)
        lo = float(np.min(d["e_fci"])) - 0.03
        hi = float(np.max(d["e_doci"][R >= 1.0])) + 0.03
        ax_a.set_ylim(lo, hi)
        ax_a.set_xlim(0.4, 4)
        ax_a.set_ylabel("Energy (Hartree)", fontsize=20)
        ax_a.legend(fontsize=18, loc="lower right")
        ax_b.set_yscale("log")
        ax_b.set_xlim(1, 4)
        ax_b.set_ylabel(r"$|E_{G}(R,\gamma) - E_{\mathrm{DOCI}}(R)|$ ", fontsize=20)
        ax_b.legend(fontsize=18, loc="lower right")
        ax_c.set_yscale("log")
        ax_c.set_xlim(0.5, 4)
        ax_c.set_ylabel(r"$1-F(R,\gamma)$", fontsize=20)
        ax_twin = ax_c.twinx()
        ax_twin.plot(R, np.abs(d["e_doci"] - d["e_fci"]), color="black", ls="-", lw=3)
        ax_twin.set_yscale("log")
        ax_twin.set_ylabel(r"$E_{\mathrm{DOCI}}(R) - E_{\mathrm{FCI}}(R)$ ", fontsize=20)
        ax_twin.tick_params(labelsize=15, which="both")
        for ax, letter in zip(axs[row], "abc"):
            ax.tick_params(labelsize=15, which="both")
            if row == 1:
                ax.set_xlabel("R (Angstrom)", fontsize=20)
            ax.text(0.03, 0.97, f"({letter}{row + 1})", transform=ax.transAxes, fontsize=24,
                    fontweight="bold", va="top", ha="left", zorder=10,
                    bbox=dict(facecolor="white", edgecolor="none", pad=2))
    handles = [plt.Line2D([], [], color=c, ls="--", marker="o", ms=8) for c in colors]
    labels = [rf"$\gamma={g:g}\,\omega$" for g in gammas]
    fig.legend(handles, labels, loc="outside upper center", ncol=len(gammas), fontsize=20, frameon=False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("->", path)


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--R", type=float, default=1.596, help="Li-H distance [A]")
    ap.add_argument("--basis", default="sto-6g")
    ap.add_argument("--gammas", nargs=3, type=float, default=[1, 200, 15], metavar=("MIN", "MAX", "NUM"))
    ap.add_argument("--c-min-trunc", type=float, default=0.0,
                    help="fq + truncation: lower bound of c_AB (the free bounded fit runs "
                         "into the c = -1 pole on the dropped edges)")
    ap.add_argument("--dissociation", action="store_true")
    ap.add_argument("--R-range", nargs=3, type=float, default=[0.8, 3.5, 12], metavar=("MIN", "MAX", "NUM"))
    ap.add_argument("--diss-gammas", nargs="+", type=float, default=[20.0, 200.0])
    ap.add_argument("--routes", nargs="+", default=ROUTES, choices=ROUTES,
                    help="routes included in the dissociation scan")
    ap.add_argument("--style", default="paper", choices=["paper", "notebook"],
                    help="dissociation figure: compact 2x3 with both routes (paper) or the "
                         "MolecularDOCI.ipynb 6-panel layout, one route per figure (notebook)")
    ap.add_argument("--tag", default="", help="suffix of the dissociation data / figure files")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(FIGDIR, exist_ok=True)

    if args.plot_only:
        d = dict(np.load(f"{OUTDIR}/gamma_scan_R{args.R:g}.npz"))
    else:
        d = gamma_scan(args)
    plot_gamma_scan(d, f"{FIGDIR}/lih_routes_gamma_scan_R{args.R:g}.pdf")

    dpath = f"{OUTDIR}/dissociation{args.tag}.npz"
    if args.dissociation or (args.plot_only and os.path.exists(dpath)):
        d = dict(np.load(dpath)) if args.plot_only else dissociation(args)
        if args.style == "paper":
            plot_dissociation(d, f"{FIGDIR}/lih_routes_dissociation{args.tag}.pdf")
        else:
            for ro in ROUTES:
                if any(k.startswith(f"{ro}_") for k in d):
                    plot_dissociation_notebook(d, ro, f"{FIGDIR}/lih_{ro}_6panel{args.tag}.pdf")


if __name__ == "__main__":
    main()
