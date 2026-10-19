"""
run_scan.py - high level test file
"""

import math
from collections import deque

from translator.config import TranslatorConfig

import qiskit
from experiments.shared.benchmark_circuits import load_proxy_circuits

from pipeline.config import PipelineConfig
from pipeline import run_specific_scan, translate_permutation
from simulator.config_presets import ScaledSimulator

# ---------------------------------------------------------------------------
# Grid placement helpers
# ---------------------------------------------------------------------------

def _grid_shape(n_codes: int) -> tuple[int, int]:
    """Return (rows, cols) for the smallest roughly-square grid that fits n_codes."""
    cols = math.ceil(math.sqrt(n_codes))
    rows = math.ceil(n_codes / cols)
    return rows, cols


def _perm_from_patch_order(n_qubits: int, patch_order: list[int]) -> list[tuple[int, int]]:
    """Assign logical qubit pairs to patches in the given order.

    Qubit 2i   → (patch_order[i], 0)
    Qubit 2i+1 → (patch_order[i], 1)
    Odd last qubit gets slot 0 in its patch.
    """
    perm = []
    for i in range(n_qubits):
        patch = patch_order[i // 2]
        slot = i % 2
        perm.append((patch, slot))
    return perm


def row_major_packing(n_qubits: int, n_codes: int) -> list[tuple[int, int]]:
    """Patches visited left-to-right, top-to-bottom."""
    patch_order = list(range(n_codes))
    return _perm_from_patch_order(n_qubits, patch_order)


def spiral_packing(n_qubits: int, n_codes: int) -> list[tuple[int, int]]:
    """Patches visited in clockwise spiral from top-left."""
    rows, cols = _grid_shape(n_codes)
    patch_order = []
    top, bottom, left, right = 0, rows - 1, 0, cols - 1
    while top <= bottom and left <= right:
        for c in range(left, right + 1):
            patch_order.append(top * cols + c)
        top += 1
        for r in range(top, bottom + 1):
            patch_order.append(r * cols + right)
        right -= 1
        if top <= bottom:
            for c in range(right, left - 1, -1):
                patch_order.append(bottom * cols + c)
            bottom -= 1
        if left <= right:
            for r in range(bottom, top - 1, -1):
                patch_order.append(r * cols + left)
            left += 1
    # Trim to n_codes in case grid has extra cells
    patch_order = [p for p in patch_order if p < n_codes]
    return _perm_from_patch_order(n_qubits, patch_order)


def bfs_packing(n_qubits: int, n_codes: int) -> list[tuple[int, int]]:
    """Patches visited in BFS order from top-left corner."""
    rows, cols = _grid_shape(n_codes)
    visited: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque([(0, 0)])
    patch_order = []
    while queue:
        r, c = queue.popleft()
        if (r, c) in visited or r >= rows or c >= cols:
            continue
        patch_id = r * cols + c
        if patch_id >= n_codes:
            continue
        visited.add((r, c))
        patch_order.append(patch_id)
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                queue.append((nr, nc))
    return _perm_from_patch_order(n_qubits, patch_order)

REPO_ROOT = "/Users/cordputer/Documents/Projects/School/Research/Infleqtion/infleqtion-qed-exploration-main"

# Representative medium circuits: one small-medium (~11q) and one larger (~18q)
PROBE_CIRCUITS = [
    f"{REPO_ROOT}/benchmarks/qasmbench/medium/sat_n11",
    f"{REPO_ROOT}/benchmarks/qasmbench/medium/qft_n18",
    f"{REPO_ROOT}/benchmarks/qasmbench/medium/ising_n26",
]

ALPHA = 0.05
PILOT_REPETITIONS = 4_000
TARGET_KEPT_SHOTS = 20_000
MIN_REPETITIONS   = 20_000
MAX_REPETITIONS   = 500_000

IDLE_SCALARS     = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
DISTANCE_SCALARS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


def _repetitions_from_lsr(lsr: float) -> int:
    denom = max(lsr, 1e-4)
    reps = int(math.ceil(TARGET_KEPT_SHOTS / denom))
    return int(max(MIN_REPETITIONS, min(MAX_REPETITIONS, reps)))


def _make_config(repetitions: int, idle_scalar: float, distance_scalar: float, seed: int) -> PipelineConfig:
    return PipelineConfig(
        sim_config=(
            ScaledSimulator(alpha=ALPHA, repetitions=repetitions, seed=seed)
            .with_stim()
            .with_distance_noise(distance_scalar=distance_scalar)
            .with_idle_noise(idle_time_scalar=idle_scalar)
        ),
        verbose=False,
        translator_config=TranslatorConfig(enable_h_commutation=False).disable_merging(),
    )


if __name__ == "__main__":
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.x(0)
    qc.z(0)
    qc.x(0)
    qc.h(0)

    qc.x(1)
    qc.z(1)
    qc.x(1)
    qc.h(1)


    config = PipelineConfig(
        sim_config=(
            ScaledSimulator(alpha=0.05, repetitions=250_000, seed=7)
            .with_stim()
            .with_distance_noise(distance_scalar=100.0)
            .with_idle_noise(idle_time_scalar=0.0)
        ),
        verbose=False,
        translator_config=TranslatorConfig(enable_h_commutation=False),
    )

    translate_permutation(program_circuit=qc, config=config, permutation=[(0,0),(0,1)])

    # circuits = []
    # for path in PROBE_CIRCUITS:
    #     loaded = load_proxy_circuits(path, replace_rz_rx_with_s=True)
    #     if loaded:
    #         name, qc = loaded[0]
    #         qc.measure_all()
    #         n = qc.num_qubits
    #         perm = tight_trivial_packing(n, patch_capacity=2)
    #         circuits.append((name, qc, perm))
    #         print(f"Loaded {name}: {n} qubits, {math.ceil(n/2)} codes, depth {qc.depth()}")

    # print()
    # header = f"{'circuit':<20} {'idle':>6} {'dist':>6} {'TVD':>8} {'LSR':>8} {'reps':>8}"
    # print(header)
    # print("-" * len(header))

    # --- Grid placement comparison: bigadder_n18 ---
    # name, qc = load_proxy_circuits(f"{REPO_ROOT}/benchmarks/qasmbench/medium/bigadder_n18")[0]
    # qc.measure_all()
    # n = qc.num_qubits
    # n_codes = math.ceil(n / 2)
    # print(f"Loaded {name}: {n} qubits, {n_codes} codes, grid {_grid_shape(n_codes)}, depth {qc.depth()}\n")

    # placements = [
    #     ("row_major", row_major_packing(n, n_codes)),
    #     ("spiral",    spiral_packing(n, n_codes)),
    #     ("bfs",       bfs_packing(n, n_codes)),
    # ]

    # config = PipelineConfig(
    #     sim_config=(
    #         ScaledSimulator(alpha=0.05, repetitions=250_000, seed=7)
    #         .with_stim()
    #         .with_distance_noise(distance_scalar=100.0)
    #         .with_idle_noise(idle_time_scalar=0.0)
    #     ),
    #     verbose=False,
    #     translator_config=TranslatorConfig(enable_h_commutation=False).disable_merging(),
    # )

    # header = f"{'placement':<12} {'TVD':>8} {'LSR':>8}"
    # print(header)
    # print("-" * len(header))
    # for label, perm in placements:
    #     record = run_specific_scan(program_circuit=qc, permutations=[perm], config=config)
    #     df = record.dataframe.df
    #     tvd = df["logical_total_variation_distance"].iloc[0]
    #     lsr = df["logical_selection_rate"].iloc[0]
    #     print(f"{label:<12} {tvd:>8.4f} {lsr:>8.4f}")
