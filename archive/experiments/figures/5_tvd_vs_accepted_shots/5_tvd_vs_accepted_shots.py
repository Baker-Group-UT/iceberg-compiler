"""
5_tvd_vs_accepted_shots.py -- is AQORN's advantage throughput, or a cleaner distribution?

**Claim.** Measured TVD decomposes as `T = a/sqrt(N) + b`, where `N` is the number of
post-selected shots that survive. The slope `a` is the sampling floor -- the error that
comes from estimating a distribution from finitely many samples -- and the intercept `b` is
the true per-shot distributional error, what is left when shots are free. AQORN's gain
lives almost entirely in `a`: it keeps more shots at the same raw budget and therefore
lands lower on the floor. The two placements' intercepts agree, which says the surviving
distribution is not cleaner, only more numerous -- and at a fixed shot budget that is the
gain that matters.

Stating this is what makes the paper's TVD framing legitimate rather than merely
defensible, and it removes the whole attack surface: a reader who suspects the TVD
improvement is a sampling artefact can be shown that it *is* a sampling effect, that this
is the mechanism being claimed, and that it is worth exactly what the shot budget says.

x = `1/sqrt(N)`, y = TVD. Panel (a) is the law itself, one fit per (circuit, placement).
Panel (b) is the answer: AQORN's fitted intercept against naive's, per circuit, with the
line of equality.

**This is a shot-budget sweep, not a re-plot of the existing CSVs, and that correction
cost a figure.** The obvious cheap version -- take LSR and TVD columns from the sweeps
already on disk and regress TVD on `1/sqrt(LSR)` -- does not work, and the reason is worth
recording so nobody tries it again. `a` depends on how widely the ideal distribution is
spread, which varies by orders of magnitude across circuits; regressing across circuits
therefore measures the between-circuit variation in `a` rather than the within-circuit
dependence on `N`. Measured on the three CSVs that carry both series, pooled fits gave
r = -0.15 to +0.30 with slopes that flip sign between sources and two physically
impossible negative intercepts. Within a single circuit, sweeping only the shot budget,
the same fit gives r = 0.999 and 0.989 with intercepts of 0.004 and 0.013 -- but slopes of
14.5 and 2.8, a factor of five apart, which is precisely what makes the pooled version
meaningless. So the sweep has to be over shots at a fixed circuit, and the fits have to be
per circuit.

**Both placements share alpha and share the raw budget at every rung.** The calibration is
solved once per circuit on the naive arm, so the two placements differ in placement and
nothing else, and any difference in accepted shots is the thing being measured rather than
an artefact of the two arms being run differently.

**n = 12 and k = 2**, where TVD is comfortably above its own sampling floor across the
whole ladder and the fit has room to resolve an intercept. This figure is about the shape
of the TVD-shot relationship, not about code size, so the code size is pinned.

Bands are omitted in panel (a) deliberately: each point is one circuit at one budget, and
the fit line through them is what carries the uncertainty. Panel (b) reports the fit
standard error on each intercept.

Run with:

    PYTHONPATH=. python experiments/figures/5_tvd_vs_accepted_shots/5_tvd_vs_accepted_shots.py
    PYTHONPATH=. python .../5_tvd_vs_accepted_shots.py --from-csv <path>
"""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.shared.figures import COLUMN_WIDTH, data_dir, save_figure, use_style
from experiments.shared.parallel import add_jobs_argument, parallel_map
from experiments.shared.random_circuits import plant_blocks, structured_program_circuit
from experiments.shared.simulation import (
    check_compile,
    SimSettings,
    add_sim_arguments,
    baseline_gate_count,
    build_config,
    calibrate,
    evaluate,
    naive_packing,
    noise_profile_for,
    settings_from_args,
)
from packing.plan import plan_packing

NAME = "5_tvd_vs_accepted_shots"
HERE = Path(__file__).resolve().parent
SCRIPT = Path(__file__).resolve()

#: Raw shot budgets, doubling. Six rungs over a 64x span give the fit enough leverage in
#: `1/sqrt(N)` to separate slope from intercept, which is the whole point of the figure.
DEFAULT_BUDGETS: Tuple[int, ...] = (10_000, 20_000, 40_000, 80_000, 160_000, 320_000)

#: `(series key, legend label)`. AQORN takes palette position 1 and naive position 2,
#: matching every other figure that compares the two.
SERIES: Sequence[Tuple[str, str]] = (
    ("aqorn", "AQORN packing"),
    ("naive", "naive packing"),
)


@dataclass
class Settings:
    budgets: Sequence[int] = DEFAULT_BUDGETS
    num_qubits: int = 12
    k: int = 2
    gates_per_qubit: int = 10
    block_size: int = 4
    locality: float = 0.85
    circuits: int = 8
    circuit_seed: int = 20260730
    sim: SimSettings = field(default_factory=SimSettings)
    csv_path: Path = field(default_factory=lambda: HERE / f"{NAME}.csv")
    from_csv: Optional[Path] = None
    output_override: Optional[Path] = None
    persist: bool = False
    jobs: Optional[int] = None


def _run_circuit(job) -> List[Dict]:
    """
    One circuit: calibrate once, plan once, then walk the shot ladder for both placements.

    The whole ladder is one work item so the calibration and the plan are paid once and
    shared by every rung -- which is also what makes the rungs comparable to each other.
    Module-level because `parallel_map` spawns rather than forks.
    """
    settings, index = job
    seed = settings.circuit_seed + index
    blocks = plant_blocks(settings.num_qubits, settings.block_size, seed=seed)
    circuit = structured_program_circuit(
        settings.num_qubits, settings.gates_per_qubit, seed=seed, blocks=blocks,
        locality=settings.locality, immutable_ratio=0.0, clifford_only=True,
    )
    n_code = settings.k + 2
    naive = naive_packing(settings.num_qubits, settings.k)

    calibration = calibrate(
        circuit, naive, settings.sim, n_code=n_code,
        baseline_gate_count=baseline_gate_count(circuit, naive, n_code),
    )

    # The planner decides whether H commutation pays, and both arms translate under that
    # same decision -- otherwise one arm describes a circuit the plan never costed.
    try:
        plan = plan_packing(
            circuit, n_code, h_commutation="auto", verify=True,
            noise=noise_profile_for(settings.sim),
        )
        aqorn, commute = plan.placement, plan.h_commutation
    except Exception:  # noqa: BLE001
        aqorn, commute = naive, True

    # The pipeline can silently miscompile. On `qpe_n9` at k = 2 the combination
    # (packed placement, merging on, H commutation off) computes the wrong distribution
    # while the selection rate stays at 1.0, so neither post-selection nor
    # `plan_packing(verify=True)` -- which compares cost, not correctness -- catches it.
    # Skip rather than fall back to naive: substituting the baseline for the arm under
    # test pulls the cell towards zero improvement and reads as a result.
    broken = check_compile(
        circuit, naive, aqorn, settings.sim, n_code=n_code, h_commutation=commute,
    )
    if broken:
        print(f"[{NAME}] skipping k {k}: {broken}")
        return []

    placements = {"aqorn": aqorn, "naive": naive}
    rows: List[Dict] = []
    for budget in settings.budgets:
        config = build_config(
            settings.sim, n_code=n_code, alpha=calibration.alpha,
            repetitions=budget, h_commutation=commute, merging=True,
        )
        for series, placement in placements.items():
            measured = evaluate(circuit, placement, config)
            if measured is None:
                continue
            rows.append({
                "seed": seed,
                "circuit_index": index,
                "num_qubits": settings.num_qubits,
                "k": settings.k,
                "n_code": n_code,
                "series": series,
                "raw_budget": budget,
                "alpha": calibration.alpha,
                "h_commutation": commute,
                "lsr": measured["lsr"],
                "tvd": measured["tvd"],
                "accepted_shots": budget * measured["lsr"],
                "one_qubit_gates": measured["one_qubit_gates"],
                "two_qubit_gates": measured["two_qubit_gates"],
                "circuit_depth": measured["circuit_depth"],
            })
    return rows


def compute(settings: Settings) -> pd.DataFrame:
    points = [(settings, index) for index in range(settings.circuits)]
    print(
        f"[{NAME}] {settings.circuits} circuits x {len(settings.budgets)} shot budgets "
        f"x 2 placements at n = {settings.num_qubits}, k = {settings.k}"
    )
    produced = parallel_map(_run_circuit, points, jobs=settings.jobs, label="circuit")
    rows = [row for batch in produced if batch for row in batch]
    return pd.DataFrame(rows)


def _fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Least squares of y on x, returning (slope, intercept, intercept standard error, r).

    The intercept's standard error is what decides whether two intercepts genuinely
    agree, which is the claim panel (b) makes, so it is computed rather than eyeballed.
    """
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual = y - predicted
    dof = len(x) - 2
    variance = float((residual ** 2).sum() / dof) if dof > 0 else float("nan")
    spread = float(((x - x.mean()) ** 2).sum())
    error = (
        float(np.sqrt(variance * (1.0 / len(x) + x.mean() ** 2 / spread)))
        if spread > 0 else float("nan")
    )
    correlation = float(np.corrcoef(x, y)[0, 1]) if len(set(x)) > 1 else float("nan")
    return float(slope), float(intercept), error, correlation


def fits(frame: pd.DataFrame) -> pd.DataFrame:
    """One fit of TVD on 1/sqrt(accepted shots) per (circuit, placement)."""
    rows: List[Dict] = []
    for index in sorted(frame["circuit_index"].unique()):
        for series, _ in SERIES:
            block = frame[
                (frame["circuit_index"] == index) & (frame["series"] == series)
            ].sort_values("accepted_shots")
            block = block[block["accepted_shots"] > 0]
            if len(block) < 3:
                continue
            x = 1.0 / np.sqrt(block["accepted_shots"].to_numpy())
            y = block["tvd"].to_numpy()
            slope, intercept, error, correlation = _fit(x, y)
            rows.append({
                "circuit_index": index,
                "series": series,
                "slope": slope,
                "intercept": intercept,
                "intercept_se": error,
                "r": correlation,
                "mean_lsr": float(block["lsr"].mean()),
                "points": len(block),
            })
    return pd.DataFrame(rows)


def print_summary(frame: pd.DataFrame, table: pd.DataFrame) -> None:
    print(f"\n[{NAME}] fit of TVD on 1/sqrt(accepted shots), per circuit")
    print("  circuit  series   slope   intercept   +-SE      r     mean LSR")
    for _, row in table.sort_values(["circuit_index", "series"]).iterrows():
        print(
            f"  {int(row['circuit_index']):>7}  {row['series']:<7} "
            f"{row['slope']:>6.2f}   {row['intercept']:>8.4f}  "
            f"{row['intercept_se']:>6.4f}  {row['r']:>6.3f}   {row['mean_lsr']:.4f}"
        )

    print(f"\n[{NAME}] the two questions this figure answers")
    for series, label in SERIES:
        block = table[table["series"] == series]
        if block.empty:
            continue
        print(
            f"  {label:<14} mean slope {block['slope'].mean():7.2f}   "
            f"mean intercept {block['intercept'].mean():.4f} "
            f"(sd {block['intercept'].std(ddof=1):.4f})"
        )

    paired = table.pivot(index="circuit_index", columns="series", values="intercept")
    if {"aqorn", "naive"}.issubset(paired.columns):
        difference = paired["aqorn"] - paired["naive"]
        errors = table.pivot(
            index="circuit_index", columns="series", values="intercept_se"
        )
        combined = np.sqrt(errors["aqorn"] ** 2 + errors["naive"] ** 2)
        agree = int((difference.abs() <= combined).sum())
        print(
            f"\n[{NAME}] intercepts: AQORN - naive = {difference.mean():+.4f} on average "
            f"({agree}/{len(difference)} circuits agree within one combined SE)"
        )
        if agree >= len(difference) * 0.6:
            print(
                f"[{NAME}] READING: the intercepts agree, so the surviving distribution "
                f"is not cleaner -- the gain is throughput. At a fixed shot budget that "
                f"is the gain that matters."
            )
        else:
            print(
                f"[{NAME}] READING: the intercepts differ, so the placements also change "
                f"the per-shot distributional error. Report this alongside the LSR gain."
            )

    slopes = table.pivot(index="circuit_index", columns="series", values="slope")
    if {"aqorn", "naive"}.issubset(slopes.columns):
        print(
            f"[{NAME}] slopes: AQORN {slopes['aqorn'].mean():.2f} against naive "
            f"{slopes['naive'].mean():.2f}; slope varies "
            f"{table['slope'].min():.2f}-{table['slope'].max():.2f} across circuits, "
            f"which is why a fit pooled over circuits is meaningless"
        )

    lsr = frame.pivot_table(index="circuit_index", columns="series", values="lsr")
    if {"aqorn", "naive"}.issubset(lsr.columns):
        gain = (lsr["aqorn"] / lsr["naive"] - 1.0) * 100.0
        print(
            f"[{NAME}] accepted-shot advantage: AQORN keeps {gain.mean():+.1f}% more "
            f"shots at the same raw budget, which moves it left along the fitted line"
        )


def plot(frame: pd.DataFrame, table: pd.DataFrame, settings: Settings) -> List[Path]:
    use_style()
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 1.15))
    top, bottom = axes

    # Panel (a): the law. Points are measurements; one fitted line per (circuit, series),
    # drawn thin because eight circuits' worth of lines is texture, not data to read off.
    colours: Dict[str, str] = {}
    for series, label in SERIES:
        block = frame[frame["series"] == series]
        block = block[block["accepted_shots"] > 0]
        if block.empty:
            continue
        x = 1.0 / np.sqrt(block["accepted_shots"].to_numpy())
        handle = top.scatter(
            x, block["tvd"].to_numpy(), s=9, alpha=0.65, label=label,
            edgecolors="none",
        )
        colours[series] = handle.get_facecolor()[0]

    for _, row in table.iterrows():
        block = frame[
            (frame["circuit_index"] == row["circuit_index"])
            & (frame["series"] == row["series"])
        ]
        block = block[block["accepted_shots"] > 0]
        if block.empty:
            continue
        x = 1.0 / np.sqrt(block["accepted_shots"].to_numpy())
        grid = np.linspace(0.0, x.max(), 20)
        top.plot(
            grid, row["slope"] * grid + row["intercept"],
            color=colours.get(row["series"], "#636363"),
            linewidth=0.6, alpha=0.45, zorder=1,
        )

    top.set_xlabel("$1/\\sqrt{N}$   ($N$ = accepted shots)")
    top.set_ylabel("total variation distance")
    top.set_xlim(left=0.0)
    top.set_ylim(bottom=0.0)
    top.legend(frameon=False, fontsize=8, loc="upper left")

    # Panel (b): the answer. Each point is one circuit; the line of equality is what the
    # claim "the gain is throughput, not per-shot fidelity" predicts the points sit on.
    paired = table.pivot(index="circuit_index", columns="series", values="intercept")
    errors = table.pivot(index="circuit_index", columns="series", values="intercept_se")
    if {"aqorn", "naive"}.issubset(paired.columns):
        bottom.errorbar(
            paired["naive"], paired["aqorn"],
            xerr=errors["naive"], yerr=errors["aqorn"],
            fmt="o", markersize=4, linewidth=0.8, capsize=2,
            color=colours.get("aqorn", "#0097fd"),
        )
        low = float(min(paired.min().min(), 0.0))
        high = float(paired.max().max()) * 1.15 or 1.0
        bottom.plot(
            [low, high], [low, high],
            color="#636363", linestyle="--", linewidth=1.0, zorder=1,
        )
        bottom.annotate(
            "equal per-shot error",
            xy=(high, high), xytext=(-4, -10), textcoords="offset points",
            fontsize=8, color="#636363", ha="right", va="top",
        )
        bottom.set_xlim(low, high)
        bottom.set_ylim(low, high)
    bottom.set_xlabel("naive intercept $b$")
    bottom.set_ylabel("AQORN intercept $b$")

    for axis, tag in zip(axes, "ab"):
        axis.annotate(
            f"({tag})", xy=(0.0, 1.0), xycoords="axes fraction",
            xytext=(-30, 2), textcoords="offset points",
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
    parser.add_argument("--budgets", nargs="+", type=int, default=list(defaults.budgets),
                        help="Raw shot budgets. Both placements run every rung.")
    parser.add_argument("--num-qubits", type=int, default=defaults.num_qubits)
    parser.add_argument("--k", type=int, default=defaults.k)
    parser.add_argument("--gates-per-qubit", type=int, default=defaults.gates_per_qubit)
    parser.add_argument("--block-size", type=int, default=defaults.block_size)
    parser.add_argument("--locality", type=float, default=defaults.locality)
    parser.add_argument("--circuits", type=int, default=defaults.circuits)
    parser.add_argument("--circuit-seed", type=int, default=defaults.circuit_seed)
    add_jobs_argument(parser)
    add_sim_arguments(parser, defaults.sim)
    parser.add_argument("--csv", type=Path, default=defaults.csv_path)
    parser.add_argument("--from-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--persist", action="store_true")
    args = parser.parse_args()
    return Settings(
        budgets=args.budgets,
        num_qubits=args.num_qubits,
        k=args.k,
        gates_per_qubit=args.gates_per_qubit,
        block_size=args.block_size,
        locality=args.locality,
        circuits=args.circuits,
        circuit_seed=args.circuit_seed,
        sim=settings_from_args(args, defaults.sim),
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
            raise RuntimeError("no data produced -- every simulation failed")
        settings.csv_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(settings.csv_path, index=False)
        print(f"[{NAME}] wrote {len(frame)} rows to {settings.csv_path}")
        if settings.persist:
            kept = data_dir(SCRIPT) / f"{NAME}.csv"
            frame.to_csv(kept, index=False)
            print(f"[{NAME}] persisted to {kept}")

    table = fits(frame)
    if table.empty:
        raise RuntimeError("no circuit produced enough rungs to fit")
    print_summary(frame, table)
    plot(frame, table, settings)


if __name__ == "__main__":
    main()
