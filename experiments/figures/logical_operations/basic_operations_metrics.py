"""
Generate compact TVD / logical-selection-rate comparison plots for the two
permutations used in ``basic_operations.py``.

Outputs:
    - basic_operations_metrics_panel.svg
    - basic_operations_tvd.svg
    - basic_operations_depth.svg
    - basic_operations_one_qubit_gates.svg
    - basic_operations_two_qubit_gates.svg
    - basic_operations_metrics.csv
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

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
MPL_CACHE_DIR = THIS_DIR / ".mplconfig"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

FONT_CACHE_DIR = THIS_DIR / ".cache"
(FONT_CACHE_DIR / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(FONT_CACHE_DIR))

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit

from iceberg.iceberg_postselect import iceberg_postselect
from pipeline import PipelineConfig
from pipeline.analysis.metrics import total_variation_distance
from pipeline.analysis.postselection import reduce_distribution
from simulator.config_presets import StandardSimulator
from simulator.simulator import simulate_batch
from evaluator.gate_count_stats import one_two_qubit_gate_totals
from translator.config import TranslatorConfig
from translator.core import Translator

try:
    from .basic_operations import PERMUTATIONS, _build_program_circuit
except ImportError:
    from basic_operations import PERMUTATIONS, _build_program_circuit

plt.rcParams.update(
    {
        "font.family": "Helvetica",
        "font.sans-serif": "Helvetica",
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "figure.titlesize": 16,
    }
)


LABEL_ORDER = ("trivial", "optimal")
LABEL_TEXT = {
    "trivial": "Trivial",
    "optimal": "Optimized",
}
LABEL_COLORS = {
    "trivial": "#cfc5c3",
    "optimal": "#44a932",
}


@dataclass(frozen=True)
class PlotConfig:
    repetitions: int = 100_000
    seed: int = 7
    use_stim: bool = False
    distance_scalar: float = 1.0
    idle_time_scalar: float = 1.0
    output_dir: Path = THIS_DIR


def _build_sim_config(cfg: PlotConfig):
    sim_cfg = StandardSimulator(repetitions=cfg.repetitions, seed=cfg.seed)
    if cfg.use_stim:
        sim_cfg = sim_cfg.with_stim()
    if cfg.distance_scalar != 1.0:
        sim_cfg = sim_cfg.with_distance_noise(distance_scalar=cfg.distance_scalar)
    if cfg.idle_time_scalar != 1.0:
        sim_cfg = sim_cfg.with_idle_noise(idle_time_scalar=cfg.idle_time_scalar)
    return sim_cfg


def _collect_metrics(cfg: PlotConfig) -> list[dict[str, float | int | str]]:
    circuit: QuantumCircuit = _build_program_circuit()
    measured_circuit = circuit.copy()
    measured_circuit.measure_all()
    permutations = [PERMUTATIONS[label] for label in LABEL_ORDER]
    sim_cfg = _build_sim_config(cfg)
    pipeline_cfg = PipelineConfig(
        sim_config=sim_cfg,
        translator_config=TranslatorConfig(),
        verbose=False,
        save=False,
        seed=cfg.seed,
    )
    simulation_translator = Translator(config=pipeline_cfg.translator_config)
    summary_translator = Translator(config=TranslatorConfig(encode_override=True))

    # Note: `run_specific_scan` is the normal high-level entrypoint, but for this
    # figure the translated circuit contains `xcx`, which is unsupported by the
    # current Aer and Stim execution paths in this workspace. We therefore use the
    # lower-level translation + simulation path directly and extract the same two
    # metrics ourselves.
    with StringIO() as stdout_buffer, StringIO() as stderr_buffer:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            translated = simulation_translator.translate_batch(measured_circuit, permutations)
            ideal_result, noisy_results = simulate_batch(
                measured_circuit,
                translated,
                pipeline_cfg.sim_config,
            )

    rows: list[dict[str, float | int | str]] = []
    for label, translation, noisy_result in zip(LABEL_ORDER, translated, noisy_results):
        postselected = iceberg_postselect(noisy_result.noisy_dist, noisy_result.key_order)
        if isinstance(postselected, tuple):
            postselected_noisy_dist, _ = postselected
        else:
            postselected_noisy_dist = postselected.get("logical_dist", {})

        reduced_noisy_dist = reduce_distribution(postselected_noisy_dist, translation.context)
        tvd = total_variation_distance(reduced_noisy_dist, ideal_result.ideal_dist)
        with StringIO() as stdout_buffer, StringIO() as stderr_buffer:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                physical_circuit = summary_translator.translate(
                    circuit.copy(),
                    PERMUTATIONS[label],
                ).circuit
        one_q_gates, two_q_gates = one_two_qubit_gate_totals(physical_circuit)
        rows.append(
            {
                "label": label,
                "display_label": LABEL_TEXT[label],
                "logical_total_variation_distance": float(tvd),
                "physical_depth": int(physical_circuit.depth()),
                "physical_size": int(physical_circuit.size()),
                "physical_one_qubit_gates": int(one_q_gates),
                "physical_two_qubit_gates": int(two_q_gates),
            }
        )
    return rows


def _write_csv(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = _build_percent_summary(rows)
    summary_map = {str(row["metric"]): float(row["percent"]) for row in summary_rows}
    fieldnames = [
        "label",
        "display_label",
        "logical_total_variation_distance",
        "physical_depth",
        "physical_size",
        "physical_one_qubit_gates",
        "physical_two_qubit_gates",
        "optimized_tvd_reduction_percent",
        "optimized_depth_reduction_percent",
        "optimized_one_qubit_gate_reduction_percent",
        "optimized_two_qubit_gate_reduction_percent",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload["optimized_tvd_reduction_percent"] = summary_map["TVD reduction"]
            payload["optimized_depth_reduction_percent"] = summary_map["Depth reduction"]
            payload["optimized_one_qubit_gate_reduction_percent"] = summary_map[
                "1q-gate reduction"
            ]
            payload["optimized_two_qubit_gate_reduction_percent"] = summary_map[
                "2q-gate reduction"
            ]
            writer.writerow(payload)


def _pct_change(reference: float, challenger: float, *, lower_is_better: bool) -> float:
    if reference <= 0:
        return 0.0
    if lower_is_better:
        return 100.0 * (reference - challenger) / reference
    return 100.0 * (challenger - reference) / reference


def _build_percent_summary(rows: list[dict[str, float | int | str]]) -> list[dict[str, str | float]]:
    row_map = {str(row["label"]): row for row in rows}
    trivial = row_map["trivial"]
    optimal = row_map["optimal"]
    return [
        {
            "metric": "TVD reduction",
            "percent": _pct_change(
                float(trivial["logical_total_variation_distance"]),
                float(optimal["logical_total_variation_distance"]),
                lower_is_better=True,
            ),
            "detail": (
                f'{float(trivial["logical_total_variation_distance"]):.3f} -> '
                f'{float(optimal["logical_total_variation_distance"]):.3f}'
            ),
        },
        {
            "metric": "Depth reduction",
            "percent": _pct_change(
                float(trivial["physical_depth"]),
                float(optimal["physical_depth"]),
                lower_is_better=True,
            ),
            "detail": (
                f'{int(trivial["physical_depth"])} -> {int(optimal["physical_depth"])}'
            ),
        },
        {
            "metric": "1q-gate reduction",
            "percent": _pct_change(
                float(trivial["physical_one_qubit_gates"]),
                float(optimal["physical_one_qubit_gates"]),
                lower_is_better=True,
            ),
            "detail": (
                f'{int(trivial["physical_one_qubit_gates"])} -> '
                f'{int(optimal["physical_one_qubit_gates"])}'
            ),
        },
        {
            "metric": "2q-gate reduction",
            "percent": _pct_change(
                float(trivial["physical_two_qubit_gates"]),
                float(optimal["physical_two_qubit_gates"]),
                lower_is_better=True,
            ),
            "detail": (
                f'{int(trivial["physical_two_qubit_gates"])} -> '
                f'{int(optimal["physical_two_qubit_gates"])}'
            ),
        },
    ]


def _draw_percent_axis(
    ax,
    *,
    summary_rows: list[dict[str, str | float]],
    xlabel: str,
) -> None:
    labels = [str(row["metric"]) for row in summary_rows]
    values = np.array([float(row["percent"]) for row in summary_rows], dtype=float)
    positions = np.arange(len(summary_rows))

    bars = ax.bar(
        positions,
        values,
        width=0.68,
        color=LABEL_COLORS["optimal"],
        edgecolor="white",
        linewidth=1.5,
        alpha=0.92,
    )

    ymax = max(5.0, float(values.max()) * 1.32)
    ax.set_ylim(0.0, ymax)
    ax.set_ylabel(xlabel)
    ax.set_xticks(positions, labels)
    ax.axhline(0, color="#636363", linestyle="--", linewidth=1.0)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, row in zip(bars, summary_rows):
        height = float(bar.get_height())
        x_center = bar.get_x() + bar.get_width() / 2.0
        ax.text(
            x_center,
            height + ymax * 0.035,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=13,
            color="#1f1f1f",
            fontweight="bold",
        )
        ax.text(
            x_center,
            max(ymax * 0.06, height * 0.52),
            str(row["detail"]),
            ha="center",
            va="center",
            fontsize=11,
            color="white",
            fontweight="bold",
        )


def _save_panel(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    summary_rows = _build_percent_summary(rows)
    fig, ax = plt.subplots(figsize=(6.0, 3.1), constrained_layout=True)
    _draw_percent_axis(
        ax,
        summary_rows=summary_rows,
        xlabel="Percent improvement vs. trivial",
    )
    ax.set_title("Optimized permutation benefit", pad=8)
    fig.savefig(output_path, format="svg", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_single_metric(
    *,
    metric_label: str,
    percent_value: float,
    detail_text: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(2.85, 2.1), constrained_layout=True)
    _draw_percent_axis(
        ax,
        summary_rows=[
            {
                "metric": metric_label,
                "percent": percent_value,
                "detail": detail_text,
            }
        ],
        xlabel="Percent improvement",
    )
    fig.savefig(output_path, format="svg", dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> PlotConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=PlotConfig.repetitions)
    parser.add_argument("--seed", type=int, default=PlotConfig.seed)
    parser.add_argument(
        "--distance-scalar",
        type=float,
        default=PlotConfig.distance_scalar,
        help="Enable distance noise only when different from 1.0.",
    )
    parser.add_argument(
        "--idle-time-scalar",
        type=float,
        default=PlotConfig.idle_time_scalar,
        help="Enable idle noise only when different from 1.0.",
    )
    parser.add_argument(
        "--stim",
        action="store_true",
        help="Use the Stim backend when the translated gates are supported.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PlotConfig.output_dir,
    )
    args = parser.parse_args()
    return PlotConfig(
        repetitions=args.repetitions,
        seed=args.seed,
        use_stim=args.stim,
        distance_scalar=args.distance_scalar,
        idle_time_scalar=args.idle_time_scalar,
        output_dir=args.output_dir,
    )


def main() -> None:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _collect_metrics(cfg)
    summary_rows = _build_percent_summary(rows)

    _write_csv(rows, cfg.output_dir / "basic_operations_metrics.csv")
    _save_panel(rows, cfg.output_dir / "basic_operations_metrics_panel.svg")
    _save_single_metric(
        metric_label=str(summary_rows[0]["metric"]),
        percent_value=float(summary_rows[0]["percent"]),
        detail_text=str(summary_rows[0]["detail"]),
        output_path=cfg.output_dir / "basic_operations_tvd.svg",
    )
    _save_single_metric(
        metric_label=str(summary_rows[1]["metric"]),
        percent_value=float(summary_rows[1]["percent"]),
        detail_text=str(summary_rows[1]["detail"]),
        output_path=cfg.output_dir / "basic_operations_depth.svg",
    )
    _save_single_metric(
        metric_label=str(summary_rows[2]["metric"]),
        percent_value=float(summary_rows[2]["percent"]),
        detail_text=str(summary_rows[2]["detail"]),
        output_path=cfg.output_dir / "basic_operations_one_qubit_gates.svg",
    )
    _save_single_metric(
        metric_label=str(summary_rows[3]["metric"]),
        percent_value=float(summary_rows[3]["percent"]),
        detail_text=str(summary_rows[3]["detail"]),
        output_path=cfg.output_dir / "basic_operations_two_qubit_gates.svg",
    )


if __name__ == "__main__":
    main()
