"""
fig05_cnot_implementations.py — arXiv Figure 5, rebuilt.

"Logical fidelities of different CNOT implementations under varying idle- and
distance-induced noise." A heatmap per implementation over the two noise scalars.

This is the figure that establishes the placement trade-off the rest of the paper
rests on: intra-patch, inter-patch and transversal CNOTs respond differently to idle
versus distance noise, so which one is cheapest depends on the machine. Rebuilt here
because the transversal row now exists at every code size -- `--n-code` sweeps it,
and the transversal construction costs `(0, n)`, so its standing against the
targeted implementations shifts as the code grows.

Simulation-bound. Defaults are a coarse grid; raise `--grid` for a publication run.

    python -m experiments.figures.updated.fig05_cnot_implementations
    python -m experiments.figures.updated.fig05_cnot_implementations --n-code 8 --grid 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import qiskit
from tqdm.auto import tqdm

from experiments.figures.updated.common import OUTPUT_DIR, quiet, save, use_paper_style
from experiments.figures.updated.simulation import SimSettings
from iceberg.utils import logical_capacity
from pipeline.config import PipelineConfig
from pipeline.tasks.scan_specific import run_specific_scan
from simulator.config_presets import ScaledSimulator

#: label -> how to place two logical qubits so the CNOT takes that form.
PLACEMENTS = {
    "intra-patch": lambda k: [(0, 0), (0, 1)],
    "inter-patch": lambda k: [(0, 0), (1, 0)],
    "inter, misaligned": lambda k: [(0, 0), (1, min(1, k - 1))],
}


def program(num_gates: int, seed: int) -> qiskit.QuantumCircuit:
    """A two-qubit circuit whose cost is dominated by CNOTs."""
    import random

    rng = random.Random(seed)
    circuit = qiskit.QuantumCircuit(2)
    for _ in range(num_gates):
        if rng.random() < 0.7:
            circuit.cx(0, 1) if rng.random() < 0.5 else circuit.cx(1, 0)
        else:
            getattr(circuit, rng.choice(("h", "x", "z")))(rng.randrange(2))
    circuit.measure_all()
    return circuit


def simulate(
    circuit: qiskit.QuantumCircuit,
    placement,
    n_code: int,
    *,
    alpha: float,
    idle_scalar: float,
    distance_scalar: float,
    repetitions: int,
) -> Dict[str, float] | None:
    sim = (
        ScaledSimulator(alpha=alpha, repetitions=repetitions, seed=None)
        .with_stim()
        .with_distance_noise(distance_scalar=distance_scalar)
        .with_idle_noise(idle_time_scalar=idle_scalar)
    )
    config = PipelineConfig(verbose=False, save=False, sim_config=sim)
    config.translator_config.n_code = n_code
    try:
        record = run_specific_scan(
            program_circuit=circuit, permutations=[list(placement)], config=config
        )
    except Exception:  # noqa: BLE001
        return None
    if record.dataframe is None or record.dataframe.df.empty:
        return None
    row = record.dataframe.df.iloc[0]
    return {
        "fidelity": float(row["logical_fidelity"]),
        "lsr": float(row["logical_selection_rate"]),
    }


def collect(
    n_code: int, grid: int, circuits: int, repetitions: int, settings: SimSettings
) -> pd.DataFrame:
    k = logical_capacity(n_code)
    idles = np.linspace(0.0, 2.0, grid)
    distances = np.linspace(0.0, 2.0, grid)
    rows: List[Dict] = []

    total = len(PLACEMENTS) * grid * grid * circuits
    with tqdm(total=total, desc="  sweeping") as bar:
        for label, builder in PLACEMENTS.items():
            placement = builder(k)
            if len({p for p, _ in placement}) == 1 and label != "intra-patch":
                bar.update(grid * grid * circuits)
                continue
            for idle in idles:
                for distance in distances:
                    values = []
                    for trial in range(circuits):
                        result = simulate(
                            program(24, seed=trial + 7), placement, n_code,
                            alpha=settings.alpha_reference,
                            idle_scalar=float(idle),
                            distance_scalar=float(distance),
                            repetitions=repetitions,
                        )
                        bar.update(1)
                        if result is not None:
                            values.append(result["fidelity"])
                    if values:
                        rows.append({
                            "implementation": label, "idle_scalar": float(idle),
                            "distance_scalar": float(distance),
                            "fidelity": float(np.mean(values)), "n_code": n_code,
                        })
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, n_code: int, name: str) -> Path:
    import matplotlib.pyplot as plt

    use_paper_style()
    labels = [label for label in PLACEMENTS if (df.implementation == label).any()]
    fig, axes = plt.subplots(
        ncols=len(labels), figsize=(4.0 * len(labels), 4.0), squeeze=False
    )
    vmin = float(df.fidelity.min())
    vmax = float(df.fidelity.max())

    image = None
    for ax, label in zip(axes[0], labels):
        sub = df[df.implementation == label]
        grid = sub.pivot_table(
            index="idle_scalar", columns="distance_scalar", values="fidelity"
        ).sort_index(ascending=False)
        image = ax.imshow(
            grid.to_numpy(), cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto",
            extent=[
                float(grid.columns.min()), float(grid.columns.max()),
                float(grid.index.min()), float(grid.index.max()),
            ],
        )
        ax.set_title(label, fontsize=13)
        ax.set_xlabel("Distance noise scalar")
    axes[0][0].set_ylabel("Idle noise scalar")
    if image is not None:
        fig.colorbar(image, ax=axes[0].tolist(), label="Logical fidelity", shrink=0.85)
    k = logical_capacity(n_code)
    fig.suptitle(
        f"CNOT implementations under idle and distance noise, "
        f"$[[{n_code},\\,{k},\\,2]]$",
        fontsize=14,
    )
    return save(fig, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-code", type=int, default=4)
    parser.add_argument("--grid", type=int, default=4)
    parser.add_argument("--circuits", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20_000)
    parser.add_argument("--from-csv", type=Path, default=None)
    args = parser.parse_args()

    quiet()
    name = f"fig05_cnot_implementations_n{args.n_code}"
    if args.from_csv:
        df = pd.read_csv(args.from_csv)
    else:
        df = collect(
            args.n_code, args.grid, args.circuits, args.repetitions, SimSettings()
        )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)

    if df.empty:
        raise RuntimeError("no data collected")
    plot(df, args.n_code, name)
    print("\n[fig05] mean fidelity per implementation")
    for label, value in df.groupby("implementation").fidelity.mean().items():
        print(f"  {label:20s} {value:.4f}")


if __name__ == "__main__":
    main()
