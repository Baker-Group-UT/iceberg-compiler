"""
gadgets_412.py -- exact single-fault enumeration of the gadgets a solo patch uses, decoded
as [[4,2,2]] and as [[4,1,2]] (spare lane gauge-fixed).

Extends `experiments/revision/exp8_undetected/gadgets.py`: the same gadget constructions
(the library's `IcebergSet` on two patches), the same noisy Stim conversion (gate
depolarising, idle dephasing inside the gadget's layers, movement), the same exhaustive
list of single faults, each propagated to the end of the gadget. Patch 0 is the solo patch
(program lane p, spare lane s, both lane choices averaged); patch 1, where a gadget needs
a partner, holds two program qubits. A fault's Pauli at the gadget end is classified

    detected        anticommutes with S_X or S_Z of a patch (or X on a |0> ancilla)
    silent          commutes with them and acts on a *program* lane: an undetected error
    spare-only      commutes with them and acts only on the spare: harmless under
                    [[4,2,2]] (the spare is never read), discarded under [[4,1,2]] when it
                    anticommutes with the gauge
    trivial         stabilizer or identity

and, for [[4,1,2]], additionally against the gauge operator at the gadget end (Zbar_s,
or its image: after HH the spare is in |+bar> and the check that the final readout
measures, after an even number of HH, is Xbar_s at that time). exp8's table counted an
error on any lane, spare included, as logical; here only program lanes count.

First-order silent probability per p2 (the unit `UndetectedRates` uses) decides the
gadget's fault distance: zero means no single fault causes an undetected program error,
i.e. circuit-level distance 2 for that gadget; nonzero means distance 1.

The measurement block (stabilizer extraction + data readout) cannot be propagated past its
measurements, so it is evaluated with a Stim detector error model instead: noiseless
encoder, noisy readout, detectors on every syndrome bit and data parity (and the spare's
Zbar for [[4,1,2]]), observables on the program lanes' Zbar.

Run (from iceberg-compiler/):
    PYTHONPATH=. venv/bin/python -m experiments.solo_gauge.gadgets_412
    PYTHONPATH=. venv/bin/python -m experiments.solo_gauge.gadgets_412 \\
        --from-csv results/solo_gauge/gadgets_412.csv
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from experiments.revision.common import RESULTS, log, write_sidecar

NAME = "solo_gauge_gadgets"
OUT = RESULTS.parent / "solo_gauge"
EXP8_RATES = RESULTS / "exp8" / "gadget_rates.csv"
N_CODE = 4
K = N_CODE - 2


def _builders() -> Dict[str, List[Tuple[Callable, int, str, Tuple[int, ...]]]]:
    """
    gadget -> [(build, program lane p of patch 0, gauge frame at the gadget's start,
    extra program lanes)]. Global lanes: patch 0 = {0, 1}, patch 1 = {2, 3}.
    """
    out: Dict[str, List[Tuple[Callable, int, str, Tuple[int, ...]]]] = {}

    def each_lane(fn, frame="Z", partner=()):
        return [(lambda s, p=p: fn(s, p), p, frame, partner) for p in range(K)]

    out["S"] = each_lane(lambda s, p: s.S(p))
    out["SX"] = each_lane(lambda s, p: s.SX(p))
    out["RZ"] = each_lane(lambda s, p: s.RZ(p, 0.5))
    out["RX"] = each_lane(lambda s, p: s.RX(p, 0.5))
    out["X"] = each_lane(lambda s, p: s.X(p))
    out["Z"] = each_lane(lambda s, p: s.Z(p))
    out["HH"] = each_lane(lambda s, p: s.HH(0))
    out["H_lane (program)"] = each_lane(lambda s, p: s.H_at(0, p))
    out["H_lane (spare fix-up)"] = each_lane(lambda s, p: s.H_at(0, 1 - p), frame="X")
    partner = (2, 3)
    for kind, method in (("CX", "CNOT"), ("CZ", "CZ"), ("XCX", "XCX")):
        out[f"{kind} cross (solo control)"] = [
            (lambda s, p=p, j=j, m=method: getattr(s, m)(p, 2 + j), p, "Z", partner)
            for p in range(K) for j in range(K)]
        if kind != "CZ":  # CZ is symmetric
            out[f"{kind} cross (solo target)"] = [
                (lambda s, p=p, j=j, m=method: getattr(s, m)(2 + j, p), p, "Z", partner)
                for p in range(K) for j in range(K)]
    return out


def _gauge_operator(frame: str, spare: int, width: int, data: Sequence[int]):
    """Zbar_s or Xbar_s of patch 0 (lane `spare`), as a physical Pauli of `width` qubits."""
    import stim

    from iceberg.layout import CodeLayout

    layout = CodeLayout(N_CODE)
    support = layout.z_support(spare) if frame == "Z" else layout.x_support(spare)
    out = stim.PauliString(width)
    for q in support:
        out[data[q]] = frame
    return out


def classify_gadget(build: Callable, p: int, frame: str, partner: Tuple[int, ...],
                    params) -> Dict[str, float]:
    """Exact first-order probabilities of each outcome under both decodings."""
    import stim

    from experiments.revision.exp8_undetected.gadgets import _gadget, noisy_stim, single_faults
    from experiments.solo_gauge.gauge import decode_pauli, patch_map, to_stim

    circuit, patches = _gadget(N_CODE, build)
    pm = patch_map(circuit)
    noiseless = to_stim(circuit, skip_measure=True)
    noisy = noisy_stim(circuit, params)
    width = max(noisy.num_qubits, circuit.num_qubits)
    spare = 1 - p
    gauge_start = _gauge_operator(frame, spare, width, pm.data[0])
    gauge_end = gauge_start.after(noiseless) if len(noiseless) else gauge_start
    program = np.zeros(pm.lanes, dtype=bool)
    program[p] = True
    for lane in partner:
        program[lane] = True
    spare_mask = np.zeros(pm.lanes, dtype=bool)
    spare_mask[spare] = True
    ancillas = np.asarray(pm.ancillas)

    out = {key: 0.0 for key in (
        "detected_422", "silent_422", "spare_only_422", "trivial_422",
        "detected_412", "silent_412", "caught_harmful", "caught_harmless")}
    for prob, components, rest in single_faults(noisy):
        error = stim.PauliString(width)
        for qubit, pauli in components.items():
            if pauli != "I":
                error[qubit] = pauli
        if len(rest):
            error = error.after(rest)
        xs, zs = error.to_numpy()
        detected = any(xs[list(d)].sum() % 2 or zs[list(d)].sum() % 2 for d in pm.data)
        detected = detected or (len(ancillas) and bool(xs[ancillas].any()))
        if detected:
            out["detected_422"] += prob
            out["detected_412"] += prob
            continue
        image = decode_pauli(error, pm)
        acting = image.x | image.z
        on_program = bool((acting & program).any())
        on_spare = bool((acting & spare_mask).any())
        gauge_fires = not error.commutes(gauge_end)
        if on_program:
            out["silent_422"] += prob
        elif on_spare:
            out["spare_only_422"] += prob
        else:
            out["trivial_422"] += prob
        if gauge_fires:
            out["detected_412"] += prob
            out["caught_harmful" if on_program else "caught_harmless"] += prob
        elif on_program:
            out["silent_412"] += prob
    return out


def classify_measurement(p: int, params) -> Dict[str, float]:
    """
    The readout block via Stim's detector error model: noiseless encoder, then noisy
    stabilizer extraction and data measurement of both patches.
    """
    import stim
    from qiskit import QuantumCircuit

    from experiments.revision.exp8_undetected.gadgets import stim_overrides
    from iceberg.iceberg_set import IcebergSet
    from iceberg.layout import CodeLayout
    from simulator import stim_backend as sb
    from simulator.grid_constructor import positions_for_circuit
    from simulator.noise_markers import append_noise_begin

    qc = QuantumCircuit()
    patches = IcebergSet(2 * K, N_CODE, qc)
    patches.encode()
    append_noise_begin(qc)
    patches.decode()
    patches.measure()
    with sb._active_params(stim_overrides(params, idle_time_scalar=1.0, distance_scalar=1.0)):
        layout = positions_for_circuit(qc, geometry=sb._geometry())
        noisy = sb.qiskit_to_stim(qc, layout=layout, include_noise=True)

    # Record index of every measured bit.
    rows: Dict[Tuple[str, int], int] = {}
    for instruction in qc.data:
        if instruction.operation.name != "measure":
            continue
        for clbit in instruction.clbits:
            register, index = qc.find_bit(clbit).registers[0]
            rows[(register.name, index)] = len(rows)
    total = len(rows)
    layout_code = CodeLayout(N_CODE)

    def rec(name: str, index: int) -> stim.GateTarget:
        return stim.target_rec(rows[(name, index)] - total)

    def with_annotations(gauge: bool) -> stim.Circuit:
        c = noisy.copy()
        for patch in range(2):
            for bit in range(2):
                c.append("DETECTOR", [rec(f"m_parity_{patch}", bit)])
            c.append("DETECTOR", [rec(f"m_data_{patch}", q) for q in range(N_CODE)])
        spare = 1 - p
        if gauge:
            c.append("DETECTOR", [rec("m_data_0", layout_code.lane(spare)),
                                  rec("m_data_0", layout_code.z_anchor)])
        program = [(0, p), (1, 0), (1, 1)]
        for obs, (patch, slot) in enumerate(program):
            c.append("OBSERVABLE_INCLUDE", [rec(f"m_data_{patch}", layout_code.lane(slot)),
                                            rec(f"m_data_{patch}", layout_code.z_anchor)], obs)
        return c

    def silent(c: stim.Circuit) -> Tuple[float, float]:
        dem = c.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
        undetected_logical = detected = 0.0
        for inst in dem.flattened():
            if inst.type != "error":
                continue
            prob = inst.args_copy()[0]
            targets = inst.targets_copy()
            has_det = any(t.is_relative_detector_id() for t in targets)
            has_obs = any(t.is_logical_observable_id() for t in targets)
            if has_det:
                detected += prob
            elif has_obs:
                undetected_logical += prob
        return undetected_logical, detected

    s422, d422 = silent(with_annotations(False))
    s412, d412 = silent(with_annotations(True))
    return {"detected_422": d422, "silent_422": s422, "spare_only_422": float("nan"),
            "trivial_422": float("nan"), "detected_412": d412, "silent_412": s412,
            "caught_harmful": s422 - s412, "caught_harmless": d412 - d422 - (s422 - s412)}


def compute() -> pd.DataFrame:
    from simulator.noise_models import StimNoiseParameters

    params = StimNoiseParameters.neutral_atom()
    p2 = params.two_qubit_error
    rows = []
    for gadget, variants in _builders().items():
        splits = [classify_gadget(b, p, frame, partner, params)
                  for b, p, frame, partner in variants]
        mean = {key: float(np.mean([s[key] for s in splits])) for key in splits[0]}
        rows.append({"gadget": gadget, "variants": len(variants),
                     "gauge_frame_at_start": variants[0][2], **mean})
        log(NAME, f"{gadget:28s} silent/p2 {mean['silent_422'] / p2:.3f} -> "
                  f"{mean['silent_412'] / p2:.3f}")
    splits = [classify_measurement(p, params) for p in range(K)]
    mean = {key: float(np.mean([s[key] for s in splits])) for key in splits[0]}
    rows.append({"gadget": "measurement", "variants": K, "gauge_frame_at_start": "Z", **mean})
    log(NAME, f"{'measurement':28s} silent/p2 {mean['silent_422'] / p2:.3f} -> "
              f"{mean['silent_412'] / p2:.3f}")
    frame = pd.DataFrame(rows)
    for column in ("detected_422", "silent_422", "spare_only_422", "detected_412",
                   "silent_412", "caught_harmful", "caught_harmless"):
        frame[f"{column}_per_p2"] = frame[column] / p2
    tol = 1e-15
    frame["fault_distance_422"] = np.where(frame["silent_422"] > tol, 1, 2)
    frame["fault_distance_412"] = np.where(frame["silent_412"] > tol, 1, 2)
    if EXP8_RATES.exists():
        exp8 = pd.read_csv(EXP8_RATES)
        exp8 = exp8[exp8["n_code"] == N_CODE].set_index("gadget")["logical_per_p2"]
        mapping = {"S": "S", "SX": "SX", "RZ": "RZ", "RX": "RX", "X": "X", "Z": "Z",
                   "HH": "HH", "H_lane (program)": "H_lane", "H_lane (spare fix-up)": "H_lane",
                   "CX cross (solo control)": "CX_cross", "CX cross (solo target)": "CX_cross",
                   "CZ cross (solo control)": "CZ_cross",
                   "XCX cross (solo control)": "XCX_cross", "XCX cross (solo target)": "XCX_cross"}
        frame["exp8_any_lane_logical_per_p2"] = [float(exp8.get(mapping.get(g, ""), np.nan))
                                                 for g in frame["gadget"]]
    frame["p2"] = p2
    return frame


DISPLAY = {
    "S": "S", "SX": "SX", "RZ": "RZ", "RX": "RX",
    "H_lane (program)": "H, targeted (program lane)",
    "H_lane (spare fix-up)": "H, targeted (spare fix-up)",
    "measurement": "readout (2 patches)",
}


def plot(frame: pd.DataFrame, output_dir: Optional[Path],
         stem: str = "solo_gauge_gadgets") -> List[Path]:
    """
    Silent error per p2 on the program lanes under [[4,2,2]] and [[4,1,2]] decoding, for
    the solo-patch gadgets that leak any at first order; the silent-free ones are listed
    underneath. Grey: shots the gauge check additionally discards that carried no error.
    """
    import matplotlib.pyplot as plt

    from experiments.shared.figures import COLUMN_WIDTH, save_figure, use_style

    use_style()
    leaky = frame[frame["silent_422"] > 1e-15]
    free = frame[frame["silent_422"] <= 1e-15]["gadget"].tolist()
    data = leaky.set_index("gadget").iloc[::-1]
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.9))
    h = 0.27
    ax.barh(y + h, data["silent_422_per_p2"], height=h, color="#3b4c56",
            label="silent, [[4,2,2]] decoding")
    ax.barh(y, data["silent_412_per_p2"], height=h, color="#0097fd",
            label="silent, [[4,1,2]] decoding")
    ax.barh(y - h, data["caught_harmless_per_p2"], height=h, color="#cdc2c0",
            label="extra discards, no error")
    for yi, (a, b) in enumerate(zip(data["silent_422_per_p2"], data["silent_412_per_p2"])):
        change = 100 * (b - a) / a
        ax.text(max(a, b) + 0.01, yi + h / 2, f"{change:+.0f}%", va="center", ha="left",
                fontsize=8, color="#0097fd")
    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY.get(g, g) for g in data.index], fontsize=9)
    ax.set_xlabel(r"first-order probability per gadget / $p_2$")
    top = float(np.nanmax(data[["silent_422_per_p2", "caught_harmless_per_p2"]].to_numpy()))
    ax.set_xlim(0, top * 1.18)
    ax.grid(axis="x", linestyle="--", linewidth=0.4, color="#b0b0b0")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.42, 1.27), ncol=2,
              handlelength=1.2, columnspacing=0.9)
    short = {"CX cross (solo control)": "CX", "CX cross (solo target)": "CX",
             "CZ cross (solo control)": "CZ", "XCX cross (solo control)": "XCX",
             "XCX cross (solo target)": "XCX"}
    singles = [g for g in free if g not in short]
    cross = sorted({short[g] for g in free if g in short})
    note = ("silent-free under both: " + ", ".join(singles)
            + (", cross-patch " + "/".join(cross) if cross else ""))
    fig.text(0.02, 0.005, note, fontsize=8, ha="left", va="bottom")
    fig.get_layout_engine().set(rect=(0, 0.05, 1, 0.95))
    paths = save_figure(fig, stem, Path(__file__), override=output_dir, versioned=True)
    plt.close(fig)
    return paths


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--from-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--jobs", type=int, default=1, help="unused; the table takes seconds")
    args = parser.parse_args(argv)
    if args.from_csv:
        frame = pd.read_csv(args.from_csv)
    else:
        began = time.time()
        frame = compute()
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "gadgets_412.csv"
        if path.exists():
            from experiments.shared.figures import next_version

            raise SystemExit(f"{path} exists; refusing to overwrite (pass --out elsewhere)")
        frame.to_csv(path, index=False)
        write_sidecar(path, experiment=NAME, settings={"n_code": N_CODE}, started=began,
                      status="complete")
        log(NAME, f"wrote {path}")
    cols = ["gadget", "silent_422_per_p2", "silent_412_per_p2", "spare_only_422_per_p2",
            "caught_harmful_per_p2", "caught_harmless_per_p2", "fault_distance_422",
            "fault_distance_412", "exp8_any_lane_logical_per_p2"]
    pd.set_option("display.width", 200)
    print(frame[[c for c in cols if c in frame]].round(4).to_string(index=False))
    for path in plot(frame, Path(args.output_dir) if args.output_dir else None):
        print(f"[{NAME}] saved {path}")


if __name__ == "__main__":
    main()
