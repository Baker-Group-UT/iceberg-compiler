"""
profiles.py — per-qubit gate timelines.

A logical qubit's *profile* is everything about it that could matter to whether it
groups well with another qubit:

* its **frame** — what kind of gate it has, layer by layer. Two qubits merge well
  when their frames line up, because the merging pass fuses same-kind gates that
  occur at the same time: Hadamards into a whole-patch HH, X or Z pairs into an
  XX/ZZ, and same-type two-qubit gates into a transversal macro.
* its **cast** — the set of other qubits it interacts with, and when. Two qubits
  whose casts overlap generate cross-patch traffic that co-locating them removes;
  two qubits whose casts point at a *common third party* can have their two-qubit
  gates fused into one transversal macro, but only if they end up in the same patch
  and their partners do too.

Layers come from the same `layer_by_node` the merging pass uses, so "same layer"
here means what it means there.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from translator.merging.layer import layer_by_node

__all__ = ["GateEvent", "QubitProfile", "extract_profiles", "ONE_QUBIT_KINDS", "TWO_QUBIT_KINDS"]

# Gate kinds the merging pass can fuse across lanes of one patch.
ONE_QUBIT_KINDS = ("h", "x", "z")
# Two-qubit kinds that have a transversal macro.
TWO_QUBIT_KINDS = ("cx", "cz", "xcx")

# Kinds with no macro; they cost the same wherever the qubit sits, so they only
# enter the model as fixed cost.
_NEUTRAL_KINDS = ("s", "sx", "sdg", "sxdg", "rz", "rx", "y", "swap")

_SKIP = ("barrier", "measure", "delay", "id")


@dataclass(frozen=True)
class GateEvent:
    """One gate as seen from one qubit."""

    layer: int
    kind: str
    #: For a two-qubit gate, the other qubit. None for single-qubit gates.
    partner: Optional[int] = None
    #: True when this qubit is the control of an asymmetric two-qubit gate.
    is_control: bool = False

    @property
    def is_two_qubit(self) -> bool:
        return self.partner is not None


@dataclass
class QubitProfile:
    """The frame and cast of one logical qubit."""

    index: int
    events: List[GateEvent] = field(default_factory=list)

    # --- frame ---

    def frame(self) -> Dict[int, Tuple[str, ...]]:
        """layer -> the kinds of gate this qubit has in that layer."""
        out: Dict[int, List[str]] = defaultdict(list)
        for event in self.events:
            out[event.layer].append(event.kind)
        return {layer: tuple(sorted(kinds)) for layer, kinds in out.items()}

    def layers_with(self, kind: str) -> FrozenSet[int]:
        """Layers in which this qubit has a gate of the given kind."""
        return frozenset(event.layer for event in self.events if event.kind == kind)

    def count(self, kind: str) -> int:
        return sum(1 for event in self.events if event.kind == kind)

    @property
    def depth(self) -> int:
        return max((event.layer for event in self.events), default=-1) + 1

    # --- cast ---

    def cast(self) -> FrozenSet[int]:
        """The other qubits this one interacts with."""
        return frozenset(
            event.partner for event in self.events if event.partner is not None
        )

    def interactions_with(self, other: int) -> List[GateEvent]:
        return [event for event in self.events if event.partner == other]

    def two_qubit_events(self) -> List[GateEvent]:
        return [event for event in self.events if event.is_two_qubit]

    def two_qubit_count(self) -> int:
        return len(self.two_qubit_events())


def extract_profiles(circuit: QuantumCircuit) -> Dict[int, QubitProfile]:
    """
    Build a profile for every qubit of `circuit`.

    Gates on three or more qubits are ignored: the compiler has no macro for them
    and the translator would reject them anyway, so they cannot influence grouping.
    """
    dag = circuit_to_dag(circuit)
    layer_index, _ = layer_by_node(dag)
    profiles = {index: QubitProfile(index=index) for index in range(circuit.num_qubits)}

    for node in dag.topological_op_nodes():
        kind = getattr(node.op, "name", "").lower()
        if kind in _SKIP:
            continue
        indices = [circuit.find_bit(qubit).index for qubit in node.qargs]
        layer = layer_index.get(node, 0)

        if len(indices) == 1:
            profiles[indices[0]].events.append(GateEvent(layer=layer, kind=kind))
        elif len(indices) == 2:
            first, second = indices
            symmetric = kind in ("cz", "xcx", "swap")
            profiles[first].events.append(
                GateEvent(layer=layer, kind=kind, partner=second, is_control=not symmetric)
            )
            profiles[second].events.append(
                GateEvent(layer=layer, kind=kind, partner=first, is_control=False)
            )
    return profiles


def aligned_layers(
    profiles: Sequence[QubitProfile], kind: str, *, window: int = 0
) -> List[int]:
    """
    Layers at which *every* profile has a gate of `kind`.

    `window` allows a tolerance: the merging pass will delay a Hadamard by a few
    layers to make an HH fire, so exact layer equality understates the opportunity.
    Returns one representative layer per aligned occurrence, greedily and without
    reusing a gate, so the count is an achievable number of macros rather than an
    upper bound.
    """
    if not profiles:
        return []
    layer_sets = [set(profile.layers_with(kind)) for profile in profiles]
    if any(not layers for layers in layer_sets):
        return []

    matched: List[int] = []
    remaining = [set(layers) for layers in layer_sets]
    for anchor in sorted(remaining[0]):
        picks = [anchor]
        for others in remaining[1:]:
            candidates = [
                layer for layer in others if abs(layer - anchor) <= window
            ]
            if not candidates:
                picks = []
                break
            picks.append(min(candidates, key=lambda layer: (abs(layer - anchor), layer)))
        if not picks:
            continue
        for layers, pick in zip(remaining, picks):
            layers.discard(pick)
        matched.append(anchor)
    return matched


def pairwise_aligned_count(
    profiles: Sequence[QubitProfile], kind: str, *, window: int = 0
) -> int:
    """
    How many disjoint *pairs* within the group share a layer for `kind`.

    Used for XX/ZZ, which fuse two lanes at a time rather than the whole patch.
    """
    occurrences: List[Tuple[int, int]] = []
    for position, profile in enumerate(profiles):
        for layer in profile.layers_with(kind):
            occurrences.append((layer, position))
    occurrences.sort()

    used_positions_by_layer: Dict[int, List[int]] = defaultdict(list)
    for layer, position in occurrences:
        used_positions_by_layer[layer].append(position)

    pairs = 0
    consumed: set[Tuple[int, int]] = set()
    for layer in sorted(used_positions_by_layer):
        available = [
            (layer, position)
            for position in used_positions_by_layer[layer]
            if (layer, position) not in consumed
        ]
        # Also allow pairing with a gate within `window` layers.
        for offset in range(1, window + 1):
            for position in used_positions_by_layer.get(layer + offset, []):
                if (layer + offset, position) not in consumed:
                    available.append((layer + offset, position))
        seen_positions: set[int] = set()
        unique = []
        for entry in available:
            if entry[1] in seen_positions:
                continue
            seen_positions.add(entry[1])
            unique.append(entry)
        while len(unique) >= 2:
            first = unique.pop()
            second = unique.pop()
            consumed.add(first)
            consumed.add(second)
            pairs += 1
    return pairs


def describe(profiles: Iterable[QubitProfile]) -> str:
    """A compact human-readable dump, for the experiment log."""
    lines = []
    for profile in profiles:
        frame = profile.frame()
        rendered = " ".join(
            f"{layer}:{'+'.join(kinds)}" for layer, kinds in sorted(frame.items())
        )
        lines.append(
            f"  q{profile.index:<3d} cast={sorted(profile.cast())} "
            f"1q(h/x/z)={profile.count('h')}/{profile.count('x')}/{profile.count('z')} "
            f"2q={profile.two_qubit_count()}\n        {rendered}"
        )
    return "\n".join(lines)
