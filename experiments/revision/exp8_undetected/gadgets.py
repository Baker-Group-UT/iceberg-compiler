"""
gadgets.py -- Experiment8, step 1: how often a fault inside each Iceberg gadget goes undetected.

Every logical operation the translator emits is a fixed physical gadget. This builds each
one with the library's own constructors (`iceberg.iceberg_set.IcebergSet`, two patches),
converts it with the simulator's own `qiskit_to_stim` -- gate depolarising noise, idle
dephasing inside the gadget's layers, movement noise on the atoms it carries -- and
enumerates every single fault exactly. Each Pauli component of each noise channel is
propagated to the end of the gadget and classified against the code:

    detected   anticommutes with X^n or Z^n of some patch, so post-selection drops the shot
    logical    commutes with every stabilizer but is not one: a silent logical error
    trivial    identity or a stabilizer: harmless

Only faults *inside* a gadget matter for this. A gadget implementing a logical Clifford
maps stabilizers to stabilizers, so it maps every detectable incoming error to a
detectable one; an error turns silent only when it starts part-way through a gadget.

Rates are probabilities per execution at alpha = 1 (`NEUTRAL_ATOM`, distance and idle
scalars 1, Fig 6's machine). Every rate scales with alpha together, so the ratio to p2 --
the unit the cost model converts with -- is alpha-independent. Idle inside a gadget is
priced on the gadget's own layers; in a real circuit a layer can be longer because
something else is moving, so the idle share here is a lower bound.
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import itertools
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd

from experiments.revision.common import RESULTS, log, write_sidecar

NAME = "exp8-gadgets"
NOISE_GATES = {"DEPOLARIZE1", "DEPOLARIZE2", "PAULI_CHANNEL_1", "X_ERROR", "Y_ERROR", "Z_ERROR"}

#: Logical two-qubit kinds as `packing.tables.GATE_KEYS` spells them, and the transversal
#: macro the merging pass replaces a full lane bijection of each with.
from packing.silent import MACROS, TWO_QUBIT_KINDS  # noqa: E402  (one definition)


@dataclass
class Settings:
    n_codes: Sequence[int] = (4, 6, 8)
    idle_time_scalar: float = 1.0
    distance_scalar: float = 1.0
    out: str = str(RESULTS / "exp8")


def stim_overrides(params, *, idle_time_scalar: float, distance_scalar: float) -> Dict:
    """The parameter overrides `simulator.attribution.measured_noise_split` installs."""
    return {
        "T1": params.T1, "T2": params.T2,
        "t_1q": params.one_qubit_gate_time, "t_2q": params.two_qubit_gate_time,
        "t_meas": params.meas_time, "p1": params.one_qubit_error,
        "p2": params.two_qubit_error, "p_meas": params.readout_error,
        "atom_pitch": params.atom_pitch, "blockade_radius": params.blockade_radius,
        "movement_speed": params.movement_speed, "patch_gutter": params.patch_gutter,
        "move_error": params.move_error,
        "idle_time_scalar": idle_time_scalar, "distance_scalar": distance_scalar,
    }


def noisy_stim(circuit, params, *, idle_time_scalar: float = 1.0, distance_scalar: float = 1.0):
    """The Stim circuit the simulator would run for `circuit`, layout and all."""
    from simulator import stim_backend as sb
    from simulator.grid_constructor import positions_for_circuit

    with sb._active_params(stim_overrides(params, idle_time_scalar=idle_time_scalar,
                                          distance_scalar=distance_scalar)):
        layout = positions_for_circuit(circuit, geometry=sb._geometry())
        return sb.qiskit_to_stim(circuit, layout=layout, include_noise=True)


def data_qubits_by_patch(circuit) -> List[List[int]]:
    """Physical data-qubit indices of each patch, from the `data_*` registers."""
    from iceberg.layout import _data_registers

    return [[circuit.find_bit(q).index for q in register]
            for register in _data_registers(circuit)]


def classify(error, patches: Sequence[Sequence[int]]) -> str:
    """'detected', 'logical' or 'trivial' for a Pauli on the patches' data qubits."""
    xs, zs = error.to_numpy()
    silent = False
    for data in patches:
        x, z = xs[list(data)], zs[list(data)]
        # Odd X-support anticommutes with Z^n; odd Z-support anticommutes with X^n.
        if x.sum() % 2 or z.sum() % 2:
            return "detected"
        if not ((not x.any() or x.all()) and (not z.any() or z.all())):
            silent = True
    return "logical" if silent else "trivial"


def single_faults(circuit) -> Iterator[Tuple[float, Dict[int, str], object]]:
    """(probability, {qubit: Pauli}, noiseless remainder) for every single fault, in order."""
    import stim

    ops = list(circuit.flattened())
    for index, op in enumerate(ops):
        if op.name not in NOISE_GATES:
            continue
        rest = stim.Circuit()
        for later in ops[index + 1:]:
            if later.name not in NOISE_GATES:
                rest.append(later)
        targets = [t.value for t in op.targets_copy()]
        args = op.gate_args_copy()
        if op.name == "DEPOLARIZE1":
            for qubit in targets:
                for pauli in "XYZ":
                    yield args[0] / 3, {qubit: pauli}, rest
        elif op.name == "DEPOLARIZE2":
            for a, b in zip(targets[::2], targets[1::2]):
                for pa, pb in itertools.product("IXYZ", repeat=2):
                    if pa == pb == "I":
                        continue
                    yield args[0] / 15, {a: pa, b: pb}, rest
        elif op.name == "PAULI_CHANNEL_1":
            for qubit in targets:
                for pauli, p in zip("XYZ", args):
                    if p > 0:
                        yield p, {qubit: pauli}, rest
        else:  # X_ERROR / Y_ERROR / Z_ERROR
            for qubit in targets:
                yield args[0], {qubit: op.name[0]}, rest


def fault_split(circuit, patches: Sequence[Sequence[int]]) -> Dict[str, float]:
    """Exact first-order probability of each outcome class over a noisy Stim circuit."""
    import stim

    out = {"detected": 0.0, "logical": 0.0, "trivial": 0.0}
    width = circuit.num_qubits
    for p, components, rest in single_faults(circuit):
        error = stim.PauliString(width)
        for qubit, pauli in components.items():
            if pauli != "I":
                error[qubit] = pauli
        out[classify(error.after(rest) if len(rest) else error, patches)] += p
    return out


#: Rotations measured on their pi/2 stand-in. Pauli propagation needs a Clifford, and what
#: decides whether a fault is silent is the gadget's two-qubit structure, not the angle:
#: an X or Y entering `rz(theta)` leaves as a mix of X and Y, which a patch's checks
#: classify identically.
CLIFFORD_STAND_IN = {"rz": "s", "rx": "sx"}


def _gadget(n_code: int, build: Callable) -> Tuple[object, List[List[int]]]:
    """The gadget `build` emits on a fresh two-patch IcebergSet, alone in its circuit."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import SGate, SXGate

    from iceberg.iceberg_set import IcebergSet

    scratch = QuantumCircuit()
    patches = IcebergSet(2 * (n_code - 2), n_code, scratch)
    before = len(scratch.data)
    build(patches)
    gadget = scratch.copy_empty_like()
    stand_in = {"s": SGate(), "sx": SXGate()}
    for instruction in scratch.data[before:]:
        name = CLIFFORD_STAND_IN.get(instruction.operation.name)
        if name is None:
            gadget.append(instruction)
        else:
            gadget.append(stand_in[name], instruction.qubits)
    return gadget, data_qubits_by_patch(gadget)


def gadget_builders(n_code: int) -> Dict[str, List[Callable]]:
    """Every lane choice of every gadget; rates are averaged over the choices."""
    k = n_code - 2
    ordered = [(i, j) for i in range(k) for j in range(k) if i != j]
    across = [(i, j) for i in range(k) for j in range(k)]
    return {
        "CX_intra": [lambda s, i=i, j=j: s.CNOT(i, j) for i, j in ordered],
        "CX_cross": [lambda s, i=i, j=j: s.CNOT(i, k + j) for i, j in across],
        "CZ_intra": [lambda s, i=i, j=j: s.CZ(i, j) for i, j in ordered],
        "CZ_cross": [lambda s, i=i, j=j: s.CZ(i, k + j) for i, j in across],
        "XCX_intra": [lambda s, i=i, j=j: s.XCX(i, j) for i, j in ordered],
        "XCX_cross": [lambda s, i=i, j=j: s.XCX(i, k + j) for i, j in across],
        "CXT": [lambda s: s.CNOT_k_Transversal(0, 1)],
        "CZT": [lambda s: s.CZ_k_Transversal(0, 1)],
        "XCXT": [lambda s: s.XCX_k_Transversal(0, 1)],
        "H_lane": [lambda s, i=i: s.H_at(0, i) for i in range(k)],
        "HH": [lambda s: s.HH(0)],
        "S": [lambda s, i=i: s.S(i) for i in range(k)],
        "SX": [lambda s, i=i: s.SX(i) for i in range(k)],
        "RZ": [lambda s, i=i: s.RZ(i, 0.5) for i in range(k)],
        "RX": [lambda s, i=i: s.RX(i, 0.5) for i in range(k)],
        "X": [lambda s, i=i: s.X(i) for i in range(k)],
        "Z": [lambda s, i=i: s.Z(i) for i in range(k)],
        "SWAP_intra": [lambda s, i=i, j=j: s.SWAP([[0, i, j]]) for i, j in ordered if i < j],
    }


def measure_rates(settings: Settings) -> pd.DataFrame:
    from simulator.noise_models import StimNoiseParameters

    params = StimNoiseParameters.neutral_atom()
    rows = []
    for n_code in settings.n_codes:
        for gadget_name, builders in gadget_builders(n_code).items():
            splits = []
            for build in builders:
                circuit, patches = _gadget(n_code, build)
                noisy = noisy_stim(circuit, params, idle_time_scalar=settings.idle_time_scalar,
                                   distance_scalar=settings.distance_scalar)
                splits.append(fault_split(noisy, patches))
            mean = {key: float(np.mean([s[key] for s in splits])) for key in splits[0]}
            rows.append({
                "n_code": n_code, "gadget": gadget_name, "variants": len(builders),
                "p_detected": mean["detected"], "p_logical": mean["logical"],
                "p_trivial": mean["trivial"],
                "logical_per_p2": mean["logical"] / params.two_qubit_error,
                "detected_per_p2": mean["detected"] / params.two_qubit_error,
                "logical_spread": float(np.ptp([s["logical"] for s in splits])),
            })
            log(NAME, f"n={n_code} {gadget_name:11s} logical {mean['logical']:.2e} "
                      f"({mean['logical'] / params.two_qubit_error:.2f} p2), "
                      f"detected {mean['detected']:.2e}")
    return pd.DataFrame(rows)


# `UndetectedRates` lives in the library now (`packing.silent`), with the packaged tables.
from packing.silent import UndetectedRates  # noqa: E402,F401


def rates_path(out: Path | str = RESULTS / "exp8") -> Path:
    return Path(out) / "gadget_rates.csv"


def load_rates(n_code: int, out: Path | str = RESULTS / "exp8") -> UndetectedRates:
    """The measured rates for `n_code`; computes and caches the table on first use."""
    from simulator.noise_models import StimNoiseParameters

    path = rates_path(out)
    if not path.exists():
        run(Settings(out=str(out)))
    table = pd.read_csv(path)
    if n_code not in set(table["n_code"]):
        raise KeyError(f"no gadget rates for n_code={n_code} in {path}")
    return UndetectedRates.from_table(table, n_code, StimNoiseParameters.neutral_atom().two_qubit_error)


def run(settings: Settings) -> Path:
    began = time.time()
    out = Path(settings.out)
    out.mkdir(parents=True, exist_ok=True)
    table = measure_rates(settings)
    path = rates_path(out)
    table.to_csv(path, index=False)
    write_sidecar(path, experiment=NAME, settings=asdict(settings), started=began,
                  status="complete")
    print(table.round(6).to_string(index=False))
    return path


def parse_args() -> Settings:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--n-codes", nargs="+", type=int, default=[4, 6, 8])
    parser.add_argument("--out", default=str(RESULTS / "exp8"))
    args = parser.parse_args()
    return Settings(n_codes=tuple(args.n_codes), out=args.out)


if __name__ == "__main__":
    run(parse_args())
