"""
logical_action.py — test support: extract the *logical* action of a physical
Iceberg circuit and compare it against a reference logical circuit.

This is the keystone verification tool for the [[n, n-2, 2]] family. It decodes
symplectically rather than simulating amplitudes, so it scales to any n and any
number of patches, and it is sign-exact: it tells S from S-dagger, and H from H
followed by a logical Pauli.

How it works
------------
`iceberg.layout.CodeLayout` fixes, per patch,

    stabilizers   S_X = X^{\\otimes n},  S_Z = Z^{\\otimes n}
    logical X     Xbar_i = X[x_anchor] X[lane_i]
    logical Z     Zbar_i = Z[z_anchor] Z[lane_i]

Every physical Pauli that commutes with the stabilizers factors uniquely as
(logical part) x (stabilizer part), and the factorization is a closed-form XOR
(`_decode_x_support` / `_decode_z_support`). So for a Clifford circuit U we
conjugate each logical generator, decode the image back into logical
coordinates, and assemble a `Clifford` on the logical qubits — which can then be
compared directly against `Clifford(reference_circuit)`.

Ancillas that must start and end in |0> — the two parity qubits, plus the flag
ancillas of the fault-tolerant gadgets — are handled by requiring that the
circuit map their Z operators back to +Z. That is exactly the statement that the
gadget returns them to |0>, i.e. that it is transparent in the fault-free case.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford, Pauli

from iceberg.layout import CodeLayout

__all__ = [
    "PatchFrame",
    "logical_clifford_of",
    "assert_logical_action",
    "physical_pauli",
    "unitary_part",
]


def physical_pauli(num_qubits: int, sites: dict[int, str]) -> Pauli:
    """Build a Pauli on `num_qubits` qubits from {qubit_index: 'X'|'Y'|'Z'}."""
    z = np.zeros(num_qubits, dtype=bool)
    x = np.zeros(num_qubits, dtype=bool)
    for qubit, letter in sites.items():
        if letter in "XY":
            x[qubit] = True
        if letter in "ZY":
            z[qubit] = True
    return Pauli((z, x))


def unitary_part(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Copy of `circuit` with measurements and barriers stripped.

    The flagged (fault-tolerant) gadgets measure their ancillas mid-circuit. In
    the fault-free case those measurements are deterministic, so the logical
    action is carried entirely by the unitary part; dropping the measurements
    lets the tableau machinery see it.
    """
    out = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    for instruction in circuit.data:
        if instruction.operation.name in ("measure", "barrier", "reset"):
            continue
        out.append(instruction.operation, instruction.qubits, instruction.clbits)
    return out


@dataclass
class PatchFrame:
    """One patch's logical/stabilizer operators, in *global* circuit indices."""

    num_qubits: int
    layout: CodeLayout
    data: list[int]

    @classmethod
    def of(cls, circuit: QuantumCircuit, module) -> "PatchFrame":
        return cls(
            num_qubits=circuit.num_qubits,
            layout=module.layout,
            data=[circuit.find_bit(q).index for q in module.data_qubits],
        )

    @property
    def k(self) -> int:
        return self.layout.k

    def x_bar(self, lane: int) -> Pauli:
        return physical_pauli(
            self.num_qubits, {self.data[q]: "X" for q in self.layout.x_support(lane)}
        )

    def z_bar(self, lane: int) -> Pauli:
        return physical_pauli(
            self.num_qubits, {self.data[q]: "Z" for q in self.layout.z_support(lane)}
        )

    def stab_x(self) -> Pauli:
        return physical_pauli(self.num_qubits, {q: "X" for q in self.data})

    def stab_z(self) -> Pauli:
        return physical_pauli(self.num_qubits, {q: "Z" for q in self.data})


def _decode_x_support(layout: CodeLayout, xs: list[bool]) -> tuple[list[bool], bool]:
    """
    Write the X-support `xs` as  sum_i a_i * supp(Xbar_i)  +  b * (all ones).

    supp(Xbar_i) = {x_anchor, lane_i}, so the z_anchor slot — the only data qubit
    in no Xbar support — reads off b, and the a_i follow by XOR. The x_anchor
    slot is then redundant, and checking it is precisely the S_Z parity check.
    """
    use_stab = xs[layout.z_anchor]
    a = [xs[layout.lane(i)] ^ use_stab for i in range(layout.k)]
    if xs[layout.x_anchor] != ((sum(a) % 2 == 1) ^ use_stab):
        raise ValueError("Pauli anticommutes with S_Z (odd X weight)")
    return a, use_stab


def _decode_z_support(layout: CodeLayout, zs: list[bool]) -> tuple[list[bool], bool]:
    """Dual of `_decode_x_support`: supp(Zbar_i) = {z_anchor, lane_i}."""
    use_stab = zs[layout.x_anchor]
    c = [zs[layout.lane(i)] ^ use_stab for i in range(layout.k)]
    if zs[layout.z_anchor] != ((sum(c) % 2 == 1) ^ use_stab):
        raise ValueError("Pauli anticommutes with S_X (odd Z weight)")
    return c, use_stab


class _Decoder:
    """Factors physical Paulis into logical and stabilizer coordinates."""

    def __init__(self, frames: list[PatchFrame], num_qubits: int):
        self.frames = frames
        self.num_qubits = num_qubits
        self.k_total = sum(f.k for f in frames)
        owned = {q for f in frames for q in f.data}
        # Everything else is a |0>-ancilla: parity qubits and FT flag ancillas.
        self.ancillas = [q for q in range(num_qubits) if q not in owned]
        self._offsets = np.cumsum([0] + [f.k for f in frames])

    def offset(self, patch: int) -> int:
        return int(self._offsets[patch])

    def locate(self, logical: int) -> tuple[int, int]:
        patch = int(np.searchsorted(self._offsets, logical, side="right")) - 1
        return patch, logical - self.offset(patch)

    def decode(self, p: Pauli) -> Pauli | None:
        """
        The logical Pauli that `p` represents, or None if `p` is a stabilizer.

        Returns a `k_total`-qubit Pauli with lanes numbered patch-major. Raises
        ValueError if `p` is not in the normalizer of the stabilizer group or has
        X support on a |0>-ancilla.
        """
        x_bits, z_bits = p.x.astype(bool), p.z.astype(bool)

        logical_x = np.zeros(self.k_total, dtype=bool)
        logical_z = np.zeros(self.k_total, dtype=bool)
        stabs: list[Pauli] = []

        for patch, frame in enumerate(self.frames):
            xs = [bool(x_bits[q]) for q in frame.data]
            zs = [bool(z_bits[q]) for q in frame.data]
            a, use_stab_x = _decode_x_support(frame.layout, xs)
            c, use_stab_z = _decode_z_support(frame.layout, zs)
            lo, hi = self.offset(patch), self.offset(patch) + frame.k
            logical_x[lo:hi] = a
            logical_z[lo:hi] = c
            if use_stab_x:
                stabs.append(frame.stab_x())
            if use_stab_z:
                stabs.append(frame.stab_z())

        for q in self.ancillas:
            if x_bits[q]:
                raise ValueError(f"Pauli has X support on |0>-ancilla qubit {q}: {p}")
            if z_bits[q]:
                stabs.append(physical_pauli(self.num_qubits, {q: "Z"}))

        # Rebuild a representative by composing the same factors in the same
        # order on both sides; the phase relating it to `p` then transfers to the
        # logical side unchanged, because Xbar/Zbar obey the same algebra as X/Z.
        def build(x_bar, z_bar, identity):
            out = identity
            for lane in np.flatnonzero(logical_x):
                out = out.compose(x_bar(int(lane)))
            for lane in np.flatnonzero(logical_z):
                out = out.compose(z_bar(int(lane)))
            return out

        physical = build(self._x_bar, self._z_bar, physical_pauli(self.num_qubits, {}))
        logical = build(
            lambda lane: physical_pauli(self.k_total, {lane: "X"}),
            lambda lane: physical_pauli(self.k_total, {lane: "Z"}),
            physical_pauli(self.k_total, {}),
        )
        for stab in stabs:
            physical = physical.compose(stab)

        return _apply_phase(logical, _relative_phase(p, physical))

    def _x_bar(self, logical: int) -> Pauli:
        patch, lane = self.locate(logical)
        return self.frames[patch].x_bar(lane)

    def _z_bar(self, logical: int) -> Pauli:
        patch, lane = self.locate(logical)
        return self.frames[patch].z_bar(lane)


_PHASES = (1 + 0j, -1 + 0j, 1j, -1j)


def _relative_phase(actual: Pauli, representative: Pauli) -> complex:
    for phase in _PHASES:
        if actual == _apply_phase(representative, phase):
            return phase
    raise AssertionError(f"{actual} is not a phase multiple of {representative}")


def _apply_phase(pauli: Pauli, phase: complex) -> Pauli:
    return pauli if phase == 1 else phase * pauli


def logical_clifford_of(circuit: QuantumCircuit, modules) -> Clifford:
    """
    The logical Clifford induced by `circuit` on the lanes of `modules`.

    Logical qubits are numbered patch-major: `sum(k of earlier patches) + lane`.
    Measurements and barriers are ignored (see `unitary_part`).

    Raises ValueError if `circuit` is not a valid logical operation: if it leaves
    the code space, fails to return an ancilla to |0>, or flips the sign of a
    stabilizer.
    """
    circuit = unitary_part(circuit)
    frames = [PatchFrame.of(circuit, m) for m in modules]
    decoder = _Decoder(frames, circuit.num_qubits)
    clifford = Clifford(circuit)

    # 1. The stabilizer group, extended by the ancilla |0> flags, must come back
    #    to itself with +1 signs, or this is not a logical operation at all.
    checks: list[tuple[str, Pauli]] = []
    for patch, frame in enumerate(frames):
        checks.append((f"S_X[{patch}]", frame.stab_x()))
        checks.append((f"S_Z[{patch}]", frame.stab_z()))
    for q in decoder.ancillas:
        checks.append((f"|0> ancilla {q}", physical_pauli(circuit.num_qubits, {q: "Z"})))

    for name, stabilizer in checks:
        image = decoder.decode(stabilizer.evolve(clifford, frame="s"))
        if image != physical_pauli(decoder.k_total, {}):
            raise ValueError(f"{name} is not preserved: mapped to {image}")

    # 2. Read off the logical action, generator by generator.
    destabilizers, stabilizers = [], []
    for logical in range(decoder.k_total):
        patch, lane = decoder.locate(logical)
        frame = frames[patch]
        destabilizers.append(decoder.decode(frame.x_bar(lane).evolve(clifford, frame="s")))
        stabilizers.append(decoder.decode(frame.z_bar(lane).evolve(clifford, frame="s")))

    for image in destabilizers + stabilizers:
        if image.phase % 2:
            raise ValueError(f"logical image is not Hermitian: {image}")

    return Clifford.from_dict(
        {
            "destabilizer": [p.to_label() for p in destabilizers],
            "stabilizer": [p.to_label() for p in stabilizers],
        }
    )


def assert_logical_action(
    circuit: QuantumCircuit,
    modules,
    reference: QuantumCircuit,
    *,
    label: str = "",
) -> None:
    """Assert `circuit` acts on the lanes of `modules` exactly as `reference`."""
    got = logical_clifford_of(circuit, modules)
    want = Clifford(reference)
    assert got == want, (
        f"{label or 'circuit'}: logical action mismatch\n"
        f"  got  destab={got.to_labels(mode='D')} stab={got.to_labels(mode='S')}\n"
        f"  want destab={want.to_labels(mode='D')} stab={want.to_labels(mode='S')}"
    )
