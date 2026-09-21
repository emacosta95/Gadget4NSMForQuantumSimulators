"""D-Wave minor embedding: chains, couplers, and the chain rescaling of the drive.

`MinorEmbedding` holds everything the Hamiltonian builder needs, in *local*
indices 0..n_physical-1 (so the resulting Hamiltonian is directly diagonalizable
with qutip), together with the original hardware labels.

Chain rescaling of the transverse field.  A logical drive d_L realized on a
chain of length k must be produced at order k of perturbation theory in the
ferromagnetic coupling J_F, which gives the universal prefactor

    d_L = prod_i d_i / J_F^{k-1}   ->   |d_i| = |d_L|^{1/k} J_F^{(k-1)/k}

(the factorization holds on trees, one factor 1/J_F per bond; it breaks on
cycles).  The sign must make the *product* of the physical amplitudes carry the
sign of d_L: with k-1 positive amplitudes, the last one is -sign(d_L) for even k
and +... careful -- `chain_drive` below implements
`prod_i d_i = |d_L| * s` with `s = sign(d_L)` explicitly, which fixes the odd-k
sign bug present in the notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class MinorEmbedding:
    """Chains and couplers in local indices."""

    chains: Dict[int, List[int]]  # logical qubit -> physical qubits
    intra_edges: List[Tuple[int, int]]  # ferromagnetic chain bonds
    inter_edges: List[Tuple[int, int]]  # couplers between different chains
    hardware_of: Dict[int, int] = field(default_factory=dict)  # local -> hardware label

    # ------------------------------------------------------------------
    @property
    def n_physical(self) -> int:
        return 1 + max(max(c) for c in self.chains.values())

    @property
    def n_logical(self) -> int:
        return len(self.chains)

    def phys_to_logical(self) -> Dict[int, int]:
        return {p: l for l, chain in self.chains.items() for p in chain}

    def chain_lengths(self) -> Dict[int, int]:
        return {l: len(c) for l, c in self.chains.items()}

    def couplers(self) -> Dict[Tuple[int, int], Tuple[int, int]]:
        """Logical pair (l_u < l_v) -> ONE representative physical coupler.

        Deduplication matters: several hardware couplers can join the same pair
        of chains, and putting the logical coupling on each of them multiplies
        it by the number of couplers (the Ca42 multi-coupler bug).
        """
        p2l = self.phys_to_logical()
        out: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for u, v in self.inter_edges:
            lu, lv = p2l[u], p2l[v]
            if lu == lv:
                continue
            key = (min(lu, lv), max(lu, lv))
            out.setdefault(key, (u, v))
        return out

    def multiplicity_report(self) -> Dict[Tuple[int, int], int]:
        """How many hardware couplers join each logical pair (should be >= 1)."""
        p2l = self.phys_to_logical()
        counts: Dict[Tuple[int, int], int] = {}
        for u, v in self.inter_edges:
            lu, lv = p2l[u], p2l[v]
            if lu == lv:
                continue
            key = (min(lu, lv), max(lu, lv))
            counts[key] = counts.get(key, 0) + 1
        return counts

    def check(self, verbose: bool = True) -> bool:
        ok = True
        missing = [
            pair
            for pair in (
                (i, j)
                for i in range(self.n_logical)
                for j in range(i + 1, self.n_logical)
            )
            if pair not in self.couplers()
        ]
        if missing:
            ok = False
            if verbose:
                print(f"missing couplers for logical pairs: {missing}")
        if verbose:
            print(f"chains: {self.chain_lengths()}")
            print(f"couplers per logical pair: {self.multiplicity_report()}")
            print("OK" if ok else "INCOMPLETE")
        return ok


# --------------------------------------------------------------------------
def find_minor_embedding(
    n_logical: int,
    hardware=None,
    pegasus_size: int = 16,
    tries: int = 10,
    seed: Optional[int] = None,
) -> MinorEmbedding:
    """Embed the complete graph K_{n_logical} into a Pegasus graph.

    The gadget needs all-to-all connectivity (every pair carries a constraint
    coupler), so K_n is the right problem graph.
    """
    import dwave_networkx as dnx
    import minorminer
    import networkx as nx

    problem = nx.complete_graph(n_logical)
    hardware = hardware if hardware is not None else dnx.pegasus_graph(pegasus_size)

    best = None
    for t in range(tries):
        emb = minorminer.find_embedding(
            problem, hardware, random_seed=None if seed is None else seed + t
        )
        if not emb:
            continue
        cost = max(len(c) for c in emb.values())
        if best is None or cost < best[0]:
            best = (cost, emb)
    if best is None:
        raise RuntimeError("minorminer found no embedding")
    embedding = best[1]

    return from_minorminer(embedding, hardware, problem)


def from_minorminer(embedding, hardware, problem_graph) -> MinorEmbedding:
    """Convert a minorminer result into local-index `MinorEmbedding`."""
    import networkx as nx

    used_nodes = [p for chain in embedding.values() for p in chain]
    local = {p: i for i, p in enumerate(used_nodes)}

    chains = {int(l): [local[p] for p in chain] for l, chain in embedding.items()}

    intra_edges: List[Tuple[int, int]] = []
    for l, chain in embedding.items():
        sub = hardware.subgraph(chain)
        for u, v in nx.minimum_spanning_tree(sub).edges():
            intra_edges.append((local[u], local[v]))

    inter_edges: List[Tuple[int, int]] = []
    for u, v in problem_graph.edges():
        for a in embedding[u]:
            for b in embedding[v]:
                if hardware.has_edge(a, b):
                    inter_edges.append((local[a], local[b]))

    return MinorEmbedding(
        chains=chains,
        intra_edges=intra_edges,
        inter_edges=inter_edges,
        hardware_of={i: p for p, i in local.items()},
    )


# --------------------------------------------------------------------------
def chain_drive(
    d_logical: float, k: int, J_F: float, counterterm: bool = False
) -> np.ndarray:
    """Physical transverse amplitudes on a chain of length k.

    Returns an array of length k whose product is d_logical * J_F^(k-1),
    so that the k-th order effective drive equals d_logical.

    `counterterm` applies, for k = 2, the correction
    m^2 = |d_L| J_F (1 + 2|d_L|/J_F) which cancels the leading finite-J_F
    amplitude error.
    """
    if k == 1:
        return np.array([d_logical])

    s = np.sign(d_logical) if d_logical != 0 else 1.0
    magnitude = (abs(d_logical) ** (1.0 / k)) * (J_F ** ((k - 1.0) / k))

    if k == 2 and counterterm:
        magnitude = np.sqrt(abs(d_logical) * J_F * (1.0 + 2.0 * abs(d_logical) / J_F))

    amps = np.full(k, magnitude)
    # The k-th order process through the ferromagnetic bonds carries an extra
    # minus sign (the virtual domain-wall denominators are positive and the
    # perturbative shift is -|V|^2/dE), so prod(amps) must be -sign(d_L)|...|.
    amps[-1] *= -s
    return amps


def physical_drive(
    d_logical: np.ndarray,
    emb: MinorEmbedding,
    J_F: float,
    scale: float = 1.0 / np.sqrt(2.0),
    counterterm: bool = False,
) -> np.ndarray:
    """Transverse field on every physical qubit (local indices)."""
    d_phys = np.zeros(emb.n_physical)
    for l, chain in emb.chains.items():
        amps = chain_drive(
            scale * d_logical[l], len(chain), J_F, counterterm=counterterm
        )
        for q, a in zip(chain, amps):
            d_phys[q] = a
    return d_phys


def suggested_J_F(gamma: float, eps: float = 0.01) -> float:
    """The notebooks' heuristic J_F = gamma^2 / eps.

    The stability requirement from the notes is J_F > (|d|/sqrt2) eps^{-L}.
    """
    return gamma**2 / eps
