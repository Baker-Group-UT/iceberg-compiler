"""
Average CNOT implementation fidelity heatmaps over random logical stabilizer states.

This figure compares four implementations:
    1. Intra-patch logical CNOT.
    2. Inter-patch CNOT with aligned slots.
    3. Inter-patch CNOT with misaligned slots.
    4. Transversal CNOT between two patches.

Only the translated CNOT region is noisy. Encoding, random logical-state
preparation, decoding, and measurement are noiseless.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
MPL_CACHE_DIR = THIS_DIR / ".mplconfig"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

FONT_CACHE_DIR = THIS_DIR / ".cache"
(FONT_CACHE_DIR / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(FONT_CACHE_DIR))

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullLocator
from tqdm.auto import tqdm

from experiments.shared.random_circuits import random_gate_circuit
from pipeline import PipelineConfig, simulate_circuit, translate_permutation
from simulator import append_noise_begin, append_noise_end
from simulator.config_presets import StandardSimulator
from translator.config import TranslatorConfig
from translator.merging.gates import CXTGate


plt.rcParams["font.family"] = "Helvetica"

TITLE_FONTSIZE = 24
AXIS_LABEL_FONTSIZE = 21
TICK_LABEL_FONTSIZE = 16
COLORBAR_LABEL_FONTSIZE = 21
COLORBAR_TICK_FONTSIZE = 16
RANDOM_PREP_GATES = ("h", "x", "z", "s", "cz", "cx")
RANDOM_PREP_GATE_COUNT = 10
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "cnot_random_state_heatmaps",
    # ["#ffffff", "#b3cde0", "#005b96", "#011f4b"],
    ["#ffffff", "#b3cde0", "#6497b1", "#005b96", "#03396c", "#011f4b"],
)


@dataclass(frozen=True)
class Scenario:
    slug: str
    title: str
    num_logical_qubits: int
    permutation: tuple[tuple[int, int], ...]
    cnot_builder: callable


@dataclass(frozen=True)
class ExperimentConfig:
    repetitions: int = 100_000
    random_states: int = 25
    seed: int = 7
    idle_min: float = 1e1
    idle_max: float = 3e2
    idle_points: int = 9
    distance_min: float = 1e1
    distance_max: float = 1.5e3
    distance_points: int = 9
    metric: str = "logical_fidelity"
    output_dir: Path = THIS_DIR


def _intra_cnot_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2, name="intra_patch_cnot")
    qc.cx(0, 1)
    return qc


def _aligned_inter_patch_cnot_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(4, name="aligned_inter_patch_cnot")
    qc.cx(0, 1)
    return qc


def _misaligned_inter_patch_cnot_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(4, name="misaligned_inter_patch_cnot")
    qc.cx(0, 1)
    return qc


def _transversal_cnot_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(4, name="transversal_cnot")
    qc.append(
        CXTGate(
            control_patch=0,
            target_patch=1,
            control_count=2,
            logical_qubits=(0, 2, 1, 3),
        ),
        [0, 2, 1, 3],
    )
    return qc


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        slug="intra_patch",
        title="Intra-patch",
        num_logical_qubits=2,
        permutation=((0, 0), (0, 1)),
        cnot_builder=_intra_cnot_circuit,
    ),
    Scenario(
        slug="inter_patch_aligned",
        title="Inter-patch aligned",
        num_logical_qubits=4,
        permutation=((0, 0), (1, 0), (0, 1), (1, 1)),
        cnot_builder=_aligned_inter_patch_cnot_circuit,
    ),
    Scenario(
        slug="inter_patch_misaligned",
        title="Inter-patch misaligned",
        num_logical_qubits=4,
        permutation=((0, 0), (1, 1), (0, 1), (1, 0)),
        cnot_builder=_misaligned_inter_patch_cnot_circuit,
    ),
    Scenario(
        slug="transversal",
        title="Transversal",
        num_logical_qubits=4,
        permutation=((0, 0), (1, 0), (0, 1), (1, 1)),
        cnot_builder=_transversal_cnot_circuit,
    ),
)


def _metric_label(metric: str) -> str:
    if metric == "logical_fidelity":
        return "Logical fidelity"
    if metric == "logical_total_variation_distance":
        return "Logical TVD"
    raise ValueError(f"Unsupported metric: {metric}")


def _translate_segment(
    logical_circuit: QuantumCircuit,
    permutation: tuple[tuple[int, int], ...],
) -> QuantumCircuit:
    config = PipelineConfig(
        verbose=False,
        save=False,
        translator_config=TranslatorConfig(encode_override=True),
    )
    with StringIO() as stdout_buffer, StringIO() as stderr_buffer:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            translated = translate_permutation(logical_circuit, list(permutation), config)
    return translated.circuit


def _append_circuit_segment(target: QuantumCircuit, segment: QuantumCircuit) -> None:
    for inst in segment.data:
        qargs = [target.qubits[segment.find_bit(q).index] for q in inst.qubits]
        cargs = [target.clbits[segment.find_bit(c).index] for c in inst.clbits]
        target.append(inst.operation, qargs, cargs)


def _ensure_creg(circuit: QuantumCircuit, name: str, size: int) -> ClassicalRegister:
    for creg in circuit.cregs:
        if creg.name == name:
            if len(creg) != size:
                raise ValueError(f"Existing creg '{name}' has size {len(creg)} != {size}")
            return creg
    creg = ClassicalRegister(size, name=name)
    circuit.add_register(creg)
    return creg


def _module_registers(circuit: QuantumCircuit, module_index: int):
    data = next(reg for reg in circuit.qregs if reg.name == f"data_{module_index}")
    parity = next(reg for reg in circuit.qregs if reg.name == f"parity_{module_index}")
    return list(data), list(parity)


def _append_encode(circuit: QuantumCircuit, num_modules: int) -> None:
    for module_index in range(num_modules):
        data_qubits, _ = _module_registers(circuit, module_index)
        circuit.h(data_qubits[0])
        for idx in range(len(data_qubits) - 1):
            circuit.cx(data_qubits[idx], data_qubits[idx + 1])
        circuit.barrier(*(data_qubits + _module_registers(circuit, module_index)[1]))


def _append_decode_and_measure(circuit: QuantumCircuit, num_modules: int) -> None:
    for module_index in range(num_modules):
        data_qubits, parity_qubits = _module_registers(circuit, module_index)
        for data_qubit in data_qubits:
            circuit.cx(data_qubit, parity_qubits[0])
        circuit.barrier(*(data_qubits + parity_qubits))

        for data_qubit in data_qubits:
            circuit.h(data_qubit)
            circuit.cx(data_qubit, parity_qubits[1])
            circuit.h(data_qubit)
        circuit.barrier(*(data_qubits + parity_qubits))

        parity_creg = _ensure_creg(circuit, f"m_parity_{module_index}", len(parity_qubits))
        circuit.measure(parity_qubits, parity_creg)

    for module_index in range(num_modules):
        data_qubits, _ = _module_registers(circuit, module_index)
        data_creg = _ensure_creg(circuit, f"m_data_{module_index}", len(data_qubits))
        circuit.measure(data_qubits, data_creg)


def _build_random_prep_circuit(
    scenario: Scenario,
    state_index: int,
    base_seed: int,
) -> QuantumCircuit:
    return random_gate_circuit(
        num_qubits=scenario.num_logical_qubits,
        num_gates=RANDOM_PREP_GATE_COUNT,
        gates=RANDOM_PREP_GATES,
        seed=(base_seed + 1) * 1_000_003 + state_index * 97 + len(scenario.slug),
    )


def _num_modules_for(permutation: tuple[tuple[int, int], ...]) -> int:
    return max(patch for patch, _ in permutation) + 1


def _build_segmented_physical_circuit(
    scenario: Scenario,
    state_index: int,
    base_seed: int,
) -> QuantumCircuit:
    shell = _translate_segment(QuantumCircuit(scenario.num_logical_qubits), scenario.permutation)
    circuit = QuantumCircuit(*shell.qregs, name=f"{scenario.slug}_{state_index}")
    num_modules = _num_modules_for(scenario.permutation)

    _append_encode(circuit, num_modules)

    prep_segment = _translate_segment(
        _build_random_prep_circuit(scenario, state_index, base_seed),
        scenario.permutation,
    )
    _append_circuit_segment(circuit, prep_segment)

    append_noise_begin(circuit)
    cnot_segment = _translate_segment(scenario.cnot_builder(), scenario.permutation)
    _append_circuit_segment(circuit, cnot_segment)
    append_noise_end(circuit)

    _append_decode_and_measure(circuit, num_modules)
    return circuit


def _simulate_metric(
    circuit: QuantumCircuit,
    *,
    repetitions: int,
    seed: int,
    idle_scalar: float,
    distance_scalar: float,
    metric: str,
) -> float:
    sim_cfg = (
        StandardSimulator(repetitions=repetitions, seed=seed)
        .with_stim()
        .with_distance_noise(distance_scalar=distance_scalar)
        .with_idle_noise(idle_time_scalar=idle_scalar)
    )
    config = PipelineConfig(sim_config=sim_cfg, verbose=False, save=False, seed=seed)
    with StringIO() as stdout_buffer, StringIO() as stderr_buffer:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            result = simulate_circuit(circuit, config)
    return float(getattr(result, metric))


def _log_edges(values: np.ndarray) -> np.ndarray:
    if len(values) == 1:
        center = float(values[0])
        return np.array([center / np.sqrt(10.0), center * np.sqrt(10.0)])
    logs = np.log10(values)
    mids = 0.5 * (logs[:-1] + logs[1:])
    first = logs[0] - 0.5 * (logs[1] - logs[0])
    last = logs[-1] + 0.5 * (logs[-1] - logs[-2])
    return np.power(10.0, np.concatenate(([first], mids, [last])))


def _run_experiment(cfg: ExperimentConfig):
    idle_values = np.logspace(np.log10(cfg.idle_min), np.log10(cfg.idle_max), cfg.idle_points)
    distance_values = np.logspace(
        np.log10(cfg.distance_min),
        np.log10(cfg.distance_max),
        cfg.distance_points,
    )

    scenario_circuits = {
        scenario.slug: [
            _build_segmented_physical_circuit(scenario, state_index, cfg.seed)
            for state_index in range(cfg.random_states)
        ]
        for scenario in SCENARIOS
    }

    heatmaps = {
        scenario.slug: np.zeros((len(idle_values), len(distance_values)), dtype=float)
        for scenario in SCENARIOS
    }

    total_runs = len(SCENARIOS) * len(idle_values) * len(distance_values) * cfg.random_states
    with tqdm(total=total_runs, desc="Simulations", unit="run") as progress:
        for scenario in SCENARIOS:
            circuits = scenario_circuits[scenario.slug]
            for idle_index, idle_scalar in enumerate(idle_values):
                for distance_index, distance_scalar in enumerate(distance_values):
                    values = []
                    for state_index, circuit in enumerate(circuits):
                        values.append(
                            _simulate_metric(
                                circuit,
                                repetitions=cfg.repetitions,
                                seed=cfg.seed + state_index,
                                idle_scalar=float(idle_scalar),
                                distance_scalar=float(distance_scalar),
                                metric=cfg.metric,
                            )
                        )
                        progress.update(1)
                    heatmaps[scenario.slug][idle_index, distance_index] = float(np.mean(values))

    return idle_values, distance_values, heatmaps


def _write_csv(
    *,
    idle_values: np.ndarray,
    distance_values: np.ndarray,
    heatmaps: dict[str, np.ndarray],
    metric: str,
    output_path: Path,
) -> None:
    rows: list[dict[str, float | str]] = []
    for scenario in SCENARIOS:
        data = heatmaps[scenario.slug]
        for idle_index, idle_scalar in enumerate(idle_values):
            for distance_index, distance_scalar in enumerate(distance_values):
                rows.append(
                    {
                        "scenario": scenario.slug,
                        "scenario_title": scenario.title,
                        "idle_time_scalar": float(idle_scalar),
                        "distance_scalar": float(distance_scalar),
                        "metric": metric,
                        "value": float(data[idle_index, distance_index]),
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "scenario_title",
                "idle_time_scalar",
                "distance_scalar",
                "metric",
                "value",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _load_from_csv(csv_path: Path, metric: str):
    import pandas as pd

    df = pd.read_csv(csv_path)
    required = {
        "scenario",
        "idle_time_scalar",
        "distance_scalar",
        "metric",
        "value",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"CSV missing required columns: {', '.join(sorted(missing))}")

    df = df[df["metric"] == metric].copy()
    if df.empty:
        raise ValueError(f"No rows found in CSV for metric '{metric}'.")

    idle_values = np.array(sorted(df["idle_time_scalar"].astype(float).unique()))
    distance_values = np.array(sorted(df["distance_scalar"].astype(float).unique()))

    heatmaps: dict[str, np.ndarray] = {}
    for scenario in SCENARIOS:
        scenario_df = df[df["scenario"] == scenario.slug].copy()
        if scenario_df.empty:
            raise ValueError(f"No rows found in CSV for scenario '{scenario.slug}'.")
        table = (
            scenario_df.pivot(
                index="idle_time_scalar",
                columns="distance_scalar",
                values="value",
            )
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
        heatmaps[scenario.slug] = table.to_numpy(dtype=float)

    return idle_values, distance_values, heatmaps


def _plot_heatmaps(
    *,
    idle_values: np.ndarray,
    distance_values: np.ndarray,
    heatmaps: dict[str, np.ndarray],
    metric: str,
    output_path: Path,
) -> None:
    metric_label = _metric_label(metric)
    x_edges = _log_edges(distance_values)
    y_edges = _log_edges(idle_values)

    vmin = min(float(np.min(heatmaps[scenario.slug])) for scenario in SCENARIOS)
    vmax = max(float(np.max(heatmaps[scenario.slug])) for scenario in SCENARIOS)

    fig, axes = plt.subplots(1, 4, figsize=(22.0, 5.8), constrained_layout=True)
    axes_list = list(np.atleast_1d(axes))
    color_mesh = None

    for ax, scenario in zip(axes_list, SCENARIOS):
        color_mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            heatmaps[scenario.slug],
            vmin=vmin,
            vmax=vmax,
            cmap=HEATMAP_CMAP,
            shading="auto",
        )
        ax.set_title(scenario.title, fontsize=TITLE_FONTSIZE)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
        ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)

    if color_mesh is None:
        raise RuntimeError("Expected at least one subplot to be rendered.")

    colorbar = fig.colorbar(color_mesh, ax=axes_list, shrink=0.92)
    colorbar.set_label(metric_label, fontsize=COLORBAR_LABEL_FONTSIZE)
    colorbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)

    fig.supxlabel("Distance scalar", fontsize=AXIS_LABEL_FONTSIZE)
    fig.supylabel("Idle scalar", fontsize=AXIS_LABEL_FONTSIZE)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Average CNOT fidelity heatmaps over random logical stabilizer states."
    )
    parser.add_argument("--repetitions", type=int, default=ExperimentConfig.repetitions)
    parser.add_argument("--random-states", type=int, default=ExperimentConfig.random_states)
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--idle-min", type=float, default=ExperimentConfig.idle_min)
    parser.add_argument("--idle-max", type=float, default=ExperimentConfig.idle_max)
    parser.add_argument("--idle-points", type=int, default=ExperimentConfig.idle_points)
    parser.add_argument("--distance-min", type=float, default=ExperimentConfig.distance_min)
    parser.add_argument("--distance-max", type=float, default=ExperimentConfig.distance_max)
    parser.add_argument("--distance-points", type=int, default=ExperimentConfig.distance_points)
    parser.add_argument(
        "--metric",
        choices=("logical_fidelity", "logical_total_variation_distance"),
        default=ExperimentConfig.metric,
    )
    parser.add_argument("--output-dir", type=Path, default=ExperimentConfig.output_dir)
    parser.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        help="Reload a previously generated CSV and re-render the SVG without rerunning simulations.",
    )
    args = parser.parse_args(argv)

    cfg = ExperimentConfig(
        repetitions=args.repetitions,
        random_states=args.random_states,
        seed=args.seed,
        idle_min=args.idle_min,
        idle_max=args.idle_max,
        idle_points=args.idle_points,
        distance_min=args.distance_min,
        distance_max=args.distance_max,
        distance_points=args.distance_points,
        metric=args.metric,
        output_dir=args.output_dir,
    )

    output_path = cfg.output_dir / "cnot_random_state_heatmaps.svg"
    csv_path = cfg.output_dir / "cnot_random_state_heatmaps.csv"
    if args.from_csv is not None:
        idle_values, distance_values, heatmaps = _load_from_csv(args.from_csv, cfg.metric)
    else:
        idle_values, distance_values, heatmaps = _run_experiment(cfg)
    _plot_heatmaps(
        idle_values=idle_values,
        distance_values=distance_values,
        heatmaps=heatmaps,
        metric=cfg.metric,
        output_path=output_path,
    )
    if args.from_csv is None:
        _write_csv(
            idle_values=idle_values,
            distance_values=distance_values,
            heatmaps=heatmaps,
            metric=cfg.metric,
            output_path=csv_path,
        )

    print(f"Wrote: {output_path}")
    if args.from_csv is None:
        print(f"Wrote: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
