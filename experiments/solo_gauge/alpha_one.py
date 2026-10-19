"""
alpha_one.py -- the [[4,1,2]]-on-solo-patches effect at a fixed, realistic noise level.

Fig 6 calibrates alpha per circuit to a baseline LSR of ~0.06, which puts 3-5 qubit circuits
at alpha 6-21 (p2 up to 6%), where most shots carry several faults. This repeats the
frame estimate of `circuit_level` (same arms, same translations, same gauge analysis) at
alpha = 1 for every arm: silent error per kept shot and kept fraction under both decodings.
No sampled distributions, so no TVD; frames are exact Monte Carlo over the noise.

Run (from iceberg-compiler/):
    PYTHONPATH=. venv/bin/python -m experiments.solo_gauge.alpha_one --jobs 3
    PYTHONPATH=. venv/bin/python -m experiments.solo_gauge.alpha_one \\
        --from-csv results/solo_gauge/alpha_one.csv
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from experiments.revision.common import ResultTable, log, stream_map
from experiments.solo_gauge.circuit_level import BASE_ARMS, OUT, arm_specs

NAME = "solo_gauge_alpha1"


def run_frames(job) -> Dict[str, object]:
    from experiments.figures.merging_improvement import (
        merging_improvement_benchmark_stylized_simulated as fig6,
    )
    from experiments.solo_gauge import gauge
    from experiments.tvd_explore.fastframes import at_alpha_one, rescale
    from pipeline.config import PipelineConfig
    from translator import explore_switches
    from translator.core import Translator

    began = time.time()
    circuit, arm, spec, settings = job
    cfg = PipelineConfig()
    cfg.translator_config.enable_h_commutation = spec["commute"]
    cfg.translator_config.enable_merging = spec["merging"]
    program = fig6.ensure_measurements(circuit.proxy)
    explore_switches.configure(**{name: True for name in spec["switches"]})
    try:
        tr = Translator(config=cfg.translator_config).translate_batch(
            program, [[tuple(p) for p in spec["placement"]]])[0]
    finally:
        explore_switches.configure(**{name: False for name in spec["switches"]})
    analysis = gauge.analyse_translation(tr)
    noisy = at_alpha_one(tr.circuit)
    row = {"name": circuit.name, "arm": arm, "num_qubits": program.num_qubits,
           "alpha": settings["alpha"], **analysis.summary()}
    row.update(gauge.frame_split(tr.circuit, rescale(noisy, settings["alpha"]),
                                 tr.context.state_positions, analysis.checks,
                                 shots=settings["frame_shots"], seed=settings["frame_seed"]))
    row["seconds"] = time.time() - began
    return row


def plot(frame: pd.DataFrame, main_csv: Optional[str], output_dir: Optional[Path],
         arms=BASE_ARMS, stem: str = "solo_gauge_alpha") -> List[Path]:
    """
    Change in silent error per kept shot from [[4,2,2]] to [[4,1,2]]-on-solo-patches
    decoding, per circuit with a valid check, against program width: at Fig 6's calibrated
    alpha (from `circuit_level`) and at alpha = 1 (this script). Frames, so every width.
    """
    import matplotlib.pyplot as plt

    from experiments.shared.figures import COLUMN_WIDTH, save_figure, use_style

    use_style()
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH, 2.3), sharey=True)
    sources = []
    if main_csv:
        sources.append(("Fig 6 alpha", pd.read_csv(main_csv)))
    sources.append((f"alpha = {frame['alpha'].iloc[0]:g}", frame))
    colors = ["#0097fd", "#3b4c56", "#cdc2c0", "#636363"]
    markers = ["o", "s", "^", "D"]
    for ax, (title, data) in zip(axes[-len(sources):], sources):
        for i, arm in enumerate(arms):
            block = data[(data["arm"] == arm) & (data["checks"] > 0)]
            rel = 100 * (block["silent_per_kept_412"] / block["silent_per_kept_422"] - 1)
            ax.plot(block["num_qubits"], rel, linestyle="none", marker=markers[i],
                    markersize=3.2, color=colors[i], label=arm,
                    markeredgecolor="#3b4c56" if colors[i] == "#cdc2c0" else colors[i],
                    markeredgewidth=0.4)
        ax.axhline(0, color="#636363", linewidth=1.0, linestyle="--")
        ax.set_xscale("log")
        ax.set_xticks([3, 10, 30, 100])
        ax.set_xticklabels(["3", "10", "30", "100"])
        ax.set_title(title, fontsize=9, pad=2)
        ax.set_xlabel("program qubits", fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("silent / kept change (%)", fontsize=9)
    axes[0].legend(fontsize=7, loc="lower right", handletextpad=0.2, borderaxespad=0.2)
    paths = save_figure(fig, stem, Path(__file__), override=output_dir, versioned=True)
    plt.close(fig)
    return paths


def main(argv=None) -> None:
    from experiments.revision.common import fig6_reference, load_fig6_circuits

    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--max-qubits", type=int, default=41)
    parser.add_argument("--arms", nargs="+", default=list(BASE_ARMS))
    parser.add_argument("--variants", nargs="*", default=[])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--frame-shots", type=int, default=400_000)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--from-csv", default=None)
    parser.add_argument("--main-csv", default=None,
                        help="circuit_level CSV, for the Fig 6-alpha panel of the figure")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    if args.from_csv:
        frame = pd.read_csv(args.from_csv)
    else:
        reference = fig6_reference()
        odd = reference[(reference["num_qubits"] % 2 == 1)
                        & (reference["num_qubits"] <= args.max_qubits)]
        names = sorted(odd["name"])
        specs = arm_specs(names, args.variants)
        circuits = {c.name: c for c in load_fig6_circuits(names=names)}
        settings = {"alpha": args.alpha, "frame_shots": args.frame_shots, "frame_seed": 11,
                    "arms": list(args.arms) + list(args.variants), "names": names}
        table = ResultTable(Path(args.out) / f"alpha_one_a{args.alpha:g}.csv",
                            key_columns=["name", "arm"], experiment=NAME, settings=settings)
        jobs = [(circuits[n], arm, specs[n][arm], settings) for n in names
                for arm in settings["arms"] if not table.is_done((n, arm))]
        jobs.sort(key=lambda j: -j[0].num_qubits)
        log(NAME, f"{len(jobs)} frame jobs at alpha {args.alpha:g}")
        counts = stream_map(run_frames, jobs, jobs=args.jobs, label=NAME,
                            on_result=lambda _i, row: table.append([row]))
        table.write_sidecar(status="complete" if counts["error"] == 0 else "partial",
                            extra={"counts": counts})
        frame = table.frame()
    rows = []
    for arm, block in frame.groupby("arm", sort=False):
        has = block[block["checks"] > 0]
        rel = 100 * (has["silent_per_kept_412"] / has["silent_per_kept_422"] - 1)
        kept = 100 * (block["frame_kept_412"] / block["frame_kept_422"] - 1)
        rows.append({"arm": arm, "circuits": len(block), "with_check": len(has),
                     "mean_rel_silent_pct_with_check": rel.mean(),
                     "median_rel_silent_pct_with_check": rel.median(),
                     "mean_rel_kept_pct": kept.mean(),
                     "n<=9_rel_silent_pct": rel[has["num_qubits"] <= 9].mean()})
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    if not args.no_plot:
        arms = [a for a in BASE_ARMS if a in set(frame["arm"])]
        for path in plot(frame, args.main_csv,
                         Path(args.output_dir) if args.output_dir else None, arms=arms):
            print(f"[{NAME}] saved {path}")


if __name__ == "__main__":
    main()
