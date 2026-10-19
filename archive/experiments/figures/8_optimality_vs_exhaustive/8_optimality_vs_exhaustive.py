"""
8_optimality_vs_exhaustive.py -- how much placement quality is available, and how much of it
AQORN takes.

**Claim.** Where the permutation space is small enough to enumerate exhaustively, AQORN's
placement is at or very near the true optimum, and the headroom it leaves is small next to
the headroom it closes against sequential packing. This is the figure that turns "AQORN
beats the baseline" into "AQORN is close to the best that any placement could do".

x = configuration, as `(program qubits, k)`. Panel (a) is regret -- how far above the
exhaustive optimum each placement lands, so the naive bar is the headroom that exists and
the AQORN bar is what is left after AQORN. Panel (b) is how often each lands exactly on the
optimum.

**The scope is small, and it is small for a countable reason.** The placement space is
`generate_permutations` over uniform code sets, and it grows past enumeration almost
immediately: 75 permutations at (4 qubits, k=2), 1,663 at (4, 4), 7,571 at (4, 6), 311 at
(5, 2), 1,383 at (6, 2) -- and then 1.86 million at (5, 4) and 17 million at (5, 6), which
the generator refuses outright. So exhaustive truth is available on 4-6 qubit programs and
nowhere else, and every number here is about programs that small. It is reported anyway
because "near-optimal where we can check" is a much stronger statement than "better than
the baseline", and because no larger experiment can supply it.

**Two slices, and only one of them survived contact with the data.** Holding `k` at 2 and
sweeping width asks whether the headroom grows with program size, and it does: sequential
packing sits 6.8%, 10.3% and 9.9% above the optimum at 4, 5 and 6 qubits. That is the slice
this figure is for.

The other slice -- holding qubits at 4 and sweeping `k`, to ask whether headroom grows with
the code size -- **is degenerate and is reported as such rather than quietly dropped.** At 4
qubits a `k = 4` or `k = 6` patch holds the entire program, so `ceil(n/k) = 1`: there is one
patch, no cross-patch gates, and therefore no packing decision. Every placement scores
identically, naive regret is exactly 0.00%, and both arms are trivially "optimal" 100% of
the time. That is not evidence that larger codes leave no headroom; it is the absence of a
question. Answering the real one would need `k = 4` at 5+ qubits, where the permutation
space is 1.86 million and the generator refuses outright. Single-patch configurations are
marked in the summary and hatched in the figure so the zero bars cannot be misread as a
measurement.

**Scored by translation, not simulation.** The quantity being optimised is the translated
circuit's cost, so every permutation is translated and counted rather than simulated. That
is what makes enumerating thousands of placements affordable (3.2 ms each), and it is also
the honest objective: the packer optimises weighted work, so weighted work is what "optimal"
should mean. Two-qubit gates and depth are scored alongside it and reported, because a
placement that wins on the packer's own objective and loses on depth would be worth knowing
about.

**Every arm translates under one commutation decision.** `plan_packing` decides whether H
commutation pays and prices its plan under that answer; the exhaustive enumeration and the
naive baseline are translated under the *same* decision, so the three arms differ in
placement and nothing else. Comparing against an enumeration run with the pass in the other
state would compare against circuits the pipeline would never build.

Run with:

    PYTHONPATH=. python experiments/figures/8_optimality_vs_exhaustive/8_optimality_vs_exhaustive.py
    PYTHONPATH=. python .../8_optimality_vs_exhaustive.py --from-csv <path>
"""

from __future__ import annotations

import argparse
import logging
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.gate_count_stats import one_two_qubit_gate_totals
from experiments.shared.figures import COLUMN_WIDTH, data_dir, save_figure, use_style
from experiments.shared.parallel import add_jobs_argument, parallel_map
from experiments.shared.random_circuits import plant_blocks, structured_program_circuit
from experiments.shared.simulation import naive_packing
from iceberg.costs import DEFAULT_WORK_WEIGHTS
from packing.plan import plan_packing
from permutations.code_sets import generate_uniform_code_sets
from permutations.generator import generate_permutations
from pipeline.config import PipelineConfig
from pipeline.tasks.translate_permutation import translate_permutation

NAME = "8_optimality_vs_exhaustive"
HERE = Path(__file__).resolve().parent
SCRIPT = Path(__file__).resolve()

Perm = List[Tuple[int, int]]

#: `(program qubits, k)` combinations whose permutation space can actually be enumerated.
#: Measured, not guessed -- see the module docstring for the counts and where the cliff is.
DEFAULT_CONFIGS: Tuple[Tuple[int, int], ...] = (
    (4, 2), (4, 4), (4, 6),   # headroom against code size, at fixed width
    (5, 2), (6, 2),           # headroom against width, at fixed code size
)

#: `(column, label)`. Weighted work first: it is what the packer optimises, so it is what
#: "optimal" means here. The other two are scored to catch a placement that wins the
#: packer's objective while losing elsewhere.
OBJECTIVES: Sequence[Tuple[str, str]] = (
    ("work", "weighted work"),
    ("two_qubit_gates", "2q gates"),
    ("circuit_depth", "depth"),
)

#: `(series key, legend label)`. AQORN takes palette position 1, naive position 2.
SERIES: Sequence[Tuple[str, str]] = (
    ("aqorn", "AQORN packing"),
    ("naive", "naive packing"),
)


@dataclass
class Settings:
    configs: Sequence[Tuple[int, int]] = DEFAULT_CONFIGS
    gates_per_qubit: int = 10
    locality: float = 0.85
    circuits: int = 20
    circuit_seed: int = 20260730
    objective: str = "work"
    csv_path: Path = field(default_factory=lambda: HERE / f"{NAME}.csv")
    from_csv: Optional[Path] = None
    output_override: Optional[Path] = None
    persist: bool = False
    jobs: Optional[int] = None


def _score(circuit, placement: Perm, config: PipelineConfig) -> Optional[Dict[str, float]]:
    """Translate one placement and return every objective, or None if it was rejected."""
    try:
        translated = translate_permutation(circuit, placement, config=config)
    except Exception:  # noqa: BLE001 - a rejected placement is a normal outcome
        return None
    one_qubit, two_qubit = one_two_qubit_gate_totals(translated.circuit)
    one_weight, two_weight = DEFAULT_WORK_WEIGHTS
    return {
        "one_qubit_gates": float(one_qubit),
        "two_qubit_gates": float(two_qubit),
        "circuit_depth": float(translated.circuit.depth()),
        "work": float(one_weight * one_qubit + two_weight * two_qubit),
    }


def _run_point(job) -> List[Dict]:
    """
    One (configuration, circuit): enumerate the whole space, then locate the two arms in it.

    Module-level and self-contained because `parallel_map` spawns rather than forks.
    """
    settings, qubits, k, index = job
    seed = settings.circuit_seed + index
    blocks = plant_blocks(qubits, min(2, qubits), seed=seed)
    circuit = structured_program_circuit(
        qubits, settings.gates_per_qubit, seed=seed, blocks=blocks,
        locality=settings.locality, immutable_ratio=0.0, clifford_only=True,
    )
    n_code = k + 2

    # The planner's commutation decision governs every arm, including the enumeration.
    try:
        plan = plan_packing(circuit, n_code, h_commutation="auto", verify=True)
        aqorn, commute = list(plan.placement), plan.h_commutation
    except Exception:  # noqa: BLE001
        aqorn, commute = naive_packing(qubits, k), True

    config = PipelineConfig(verbose=False, save=False)
    config.translator_config.n_code = n_code
    config.translator_config.enable_h_commutation = commute
    config.translator_config.enable_merging = True

    try:
        space = generate_permutations(generate_uniform_code_sets(qubits, n=n_code), qubits)
    except ValueError:
        return []

    best: Dict[str, float] = {}
    scored = 0
    for candidate in space:
        result = _score(circuit, [tuple(slot) for slot in candidate], config)
        if result is None:
            continue
        scored += 1
        for column, _ in OBJECTIVES:
            if column not in best or result[column] < best[column]:
                best[column] = result[column]
    if not scored:
        return []

    rows: List[Dict] = []
    arms = {"aqorn": [tuple(slot) for slot in aqorn], "naive": naive_packing(qubits, k)}
    for series, placement in arms.items():
        result = _score(circuit, placement, config)
        if result is None:
            continue
        row: Dict = {
            "circuit_index": index,
            "seed": seed,
            "num_qubits": qubits,
            "k": k,
            "n_code": n_code,
            "config": f"{qubits}q, $k$={k}",
            "series": series,
            "h_commutation": commute,
            "space_size": len(space),
            "placements_scored": scored,
        }
        for column, _ in OBJECTIVES:
            row[column] = result[column]
            row[f"{column}_optimum"] = best[column]
            row[f"{column}_regret"] = (
                (result[column] / best[column] - 1.0) * 100.0 if best[column] else float("nan")
            )
            row[f"{column}_optimal"] = bool(
                abs(result[column] - best[column]) < 1e-9
            )
        rows.append(row)
    return rows


def compute(settings: Settings) -> pd.DataFrame:
    points = [
        (settings, qubits, k, index)
        for qubits, k in settings.configs
        for index in range(settings.circuits)
    ]
    print(
        f"[{NAME}] {len(settings.configs)} configurations x {settings.circuits} circuits, "
        f"exhaustive enumeration per point"
    )
    produced = parallel_map(_run_point, points, jobs=settings.jobs, label="config x circuit")
    rows = [row for batch in produced if batch for row in batch]
    return pd.DataFrame(rows)


def summarise(frame: pd.DataFrame, objective: str) -> pd.DataFrame:
    """Mean regret and optimum hit rate per (configuration, placement)."""
    rows: List[Dict] = []
    for config in frame["config"].unique():
        for series, _ in SERIES:
            block = frame[(frame["config"] == config) & (frame["series"] == series)]
            if block.empty:
                continue
            regret = block[f"{objective}_regret"]
            qubits = int(block["num_qubits"].iloc[0])
            code = int(block["k"].iloc[0])
            rows.append({
                "config": config,
                "num_qubits": qubits,
                "k": code,
                # One patch holds the whole program, so there is no packing decision and
                # every placement scores alike. See the module docstring.
                "single_patch": bool(math.ceil(qubits / code) == 1),
                "series": series,
                "regret": float(regret.mean()),
                "regret_sd": float(regret.std(ddof=1)) if len(regret) > 1 else 0.0,
                "hit_rate": float(block[f"{objective}_optimal"].mean() * 100.0),
                "space_size": int(block["space_size"].iloc[0]),
                "circuits": int(len(block)),
            })
    return pd.DataFrame(rows)


def print_summary(frame: pd.DataFrame, settings: Settings) -> None:
    for objective, label in OBJECTIVES:
        table = summarise(frame, objective)
        print(f"\n[{NAME}] {label}: regret above the exhaustive optimum (%)")
        print("  config          space   AQORN regret   naive regret   AQORN optimal   naive optimal")
        for config in table["config"].unique():
            block = table[table["config"] == config]
            aqorn = block[block["series"] == "aqorn"]
            naive = block[block["series"] == "naive"]
            if aqorn.empty or naive.empty:
                continue
            aqorn, naive = aqorn.iloc[0], naive.iloc[0]
            mark = "   <- one patch: no packing decision" if aqorn["single_patch"] else ""
            print(
                f"  {config:<14} {aqorn['space_size']:>5}   "
                f"{aqorn['regret']:>11.2f}%   {naive['regret']:>11.2f}%   "
                f"{aqorn['hit_rate']:>12.0f}%   {naive['hit_rate']:>12.0f}%{mark}"
            )

    table = summarise(frame, settings.objective)
    # Degenerate configurations contribute 0/0 to a headroom-capture ratio, which would
    # drag the headline number toward a value describing configurations that pose no
    # packing problem at all.
    real = table[~table["single_patch"]]
    aqorn = real[real["series"] == "aqorn"]
    naive = real[real["series"] == "naive"]
    if not aqorn.empty and not naive.empty:
        captured = 1.0 - aqorn["regret"].mean() / max(naive["regret"].mean(), 1e-9)
        print(
            f"\n[{NAME}] on {settings.objective}, over the "
            f"{len(aqorn)} configurations that pose a real packing decision: AQORN "
            f"captures {captured * 100:.0f}% of the headroom sequential packing leaves, "
            f"and lands exactly on the optimum in {aqorn['hit_rate'].mean():.0f}% of "
            f"circuits"
        )

    width = real[real["series"] == "naive"].sort_values("num_qubits")
    if len(width) >= 2:
        print(f"\n[{NAME}] headroom against program width (naive regret above optimum)")
        for _, row in width.iterrows():
            print(f"    {int(row['num_qubits'])} qubits, k={int(row['k'])}: "
                  f"{row['regret']:.2f}%")

    degenerate = table[table["single_patch"]]["config"].unique()
    if len(degenerate):
        print(
            f"\n[{NAME}] excluded as degenerate (one patch holds the whole program, so "
            f"every placement is identical): {', '.join(sorted(degenerate))}. The "
            f"code-size question needs k=4 at 5+ qubits, whose space the generator "
            f"refuses at 1.86 million permutations."
        )


def plot(frame: pd.DataFrame, settings: Settings) -> List[Path]:
    use_style()
    table = summarise(frame, settings.objective)
    if table.empty:
        raise RuntimeError("nothing to plot")

    order = [f"{q}q, $k$={k}" for q, k in settings.configs
             if f"{q}q, $k$={k}" in set(table["config"])]
    positions = np.arange(len(order))
    width = 0.38

    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 1.05), sharex=True)
    top, bottom = axes

    for offset, (series, label) in zip((-width / 2, width / 2), SERIES):
        block = table[table["series"] == series].set_index("config").reindex(order)
        # Degenerate configurations are hatched rather than dropped: a zero bar that is
        # zero because no packing decision exists must not read as a measured zero.
        hatches = ["//" if flag else "" for flag in block["single_patch"].fillna(False)]
        # Thin error bars with caps: categorical x, per the house convention.
        bars = top.bar(
            positions + offset, block["regret"].to_numpy(), width,
            yerr=block["regret_sd"].to_numpy(), label=label,
            error_kw={"elinewidth": 0.8, "capsize": 2},
        )
        lower = bottom.bar(
            positions + offset, block["hit_rate"].to_numpy(), width, label=label
        )
        for group in (bars, lower):
            for patch, hatch in zip(group, hatches):
                if hatch:
                    patch.set_hatch(hatch)
                    patch.set_edgecolor("#636363")

    top.set_ylabel(f"regret above optimum (%)")
    bottom.set_ylabel("landed on optimum (%)")
    bottom.set_ylim(0, 100)
    bottom.set_xticks(positions)
    bottom.set_xticklabels(order)
    bottom.set_xlabel("program width and code size")
    top.legend(frameon=False, fontsize=8, loc="best")

    for axis, tag in zip(axes, "ab"):
        axis.annotate(
            f"({tag})", xy=(0.0, 1.0), xycoords="axes fraction",
            xytext=(-32, 2), textcoords="offset points",
            fontsize=10, fontweight="bold", va="bottom",
        )

    paths = save_figure(fig, NAME, SCRIPT, settings.output_override, versioned=True)
    plt.close(fig)
    for path in paths:
        print(f"[{NAME}] wrote {path}")
    return paths


def parse_args() -> Settings:
    defaults = Settings()
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Configurations as qubits:k, e.g. 4:2 4:4 5:2.")
    parser.add_argument("--gates-per-qubit", type=int, default=defaults.gates_per_qubit)
    parser.add_argument("--locality", type=float, default=defaults.locality)
    parser.add_argument("--circuits", type=int, default=defaults.circuits)
    parser.add_argument("--circuit-seed", type=int, default=defaults.circuit_seed)
    parser.add_argument("--objective", choices=[c for c, _ in OBJECTIVES],
                        default=defaults.objective,
                        help="Which objective the plotted panels use. All are reported.")
    add_jobs_argument(parser)
    parser.add_argument("--csv", type=Path, default=defaults.csv_path)
    parser.add_argument("--from-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--persist", action="store_true")
    args = parser.parse_args()
    configs = defaults.configs
    if args.configs:
        configs = tuple(
            (int(entry.split(":")[0]), int(entry.split(":")[1])) for entry in args.configs
        )
    return Settings(
        configs=configs,
        gates_per_qubit=args.gates_per_qubit,
        locality=args.locality,
        circuits=args.circuits,
        circuit_seed=args.circuit_seed,
        objective=args.objective,
        csv_path=args.csv,
        from_csv=args.from_csv,
        output_override=args.output_dir,
        persist=args.persist,
        jobs=args.jobs,
    )


def main() -> None:
    settings = parse_args()

    if settings.from_csv is not None:
        frame = pd.read_csv(settings.from_csv)
    else:
        frame = compute(settings)
        if frame.empty:
            raise RuntimeError("no data produced -- every enumeration failed")
        settings.csv_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(settings.csv_path, index=False)
        print(f"[{NAME}] wrote {len(frame)} rows to {settings.csv_path}")
        if settings.persist:
            kept = data_dir(SCRIPT) / f"{NAME}.csv"
            frame.to_csv(kept, index=False)
            print(f"[{NAME}] persisted to {kept}")

    print_summary(frame, settings)
    plot(frame, settings)


if __name__ == "__main__":
    main()
