# Iceberg Compiler

Open-source implementation for the paper **"Logical Compilation for Multi-Qubit Iceberg Patches"** by Cordell Mazzetti, Sayam Sethi, Rich Rines, Pranav Gokhale, and Jonathan Mark Baker.

> **Paper:** [arXiv preprint](https://arxiv.org/abs/2604.09956)

## Overview

This framework solves the **qubit packing problem** for the [[4, 2, 2]] Iceberg quantum error-detecting (QED) code: given a program circuit, which pairs of logical qubits should share a physical code patch, and how should those patches be laid out? The choice of mapping determines which gate implementations are available — intra-patch, inter-patch, or transversal — producing up to 4× variation in gate cost and circuit depth (see Table 1 and Figure 4 in the paper).

The compiler pipeline takes an arbitrary Clifford+Rz program circuit and produces a fully encoded physical circuit optimized for the [[4, 2, 2]] Iceberg code, using Hadamard commutation, gate merging, and a merge-maximizing packing heuristic. Across 71 QASMBench circuits, this reduces circuit depth by 34%, one- and two-qubit gate counts by 31% and 17%, and improves total variation distance by 1.75× relative to a naive compilation baseline.

## Compilation Pipeline

The pipeline follows three phases (Figure 9 in the paper):

**Phase 1 — Program circuit optimization**

1. Gate cancellation
2. Hadamard commutation: push all H gates to the end of the circuit, canceling pairs, leaving at most one H per qubit
3. Second gate cancellation pass

**Phase 2 — Logical circuit: packing and merging**

1. Pairwise merge scoring: run the gate merging algorithm on every candidate qubit pair and build a score matrix
2. Greedy packing: commit qubit pairs in order of net gain `w_ij - w_ii - w_jj` until the target patch count is reached
3. Grid alignment: map patches onto a 2D grid to minimize inter-patch communication distance
4. Circuit rewrite: substitute mergeable gate pairs with cheaper patch-wise or transversal primitives

**Phase 3 — Physical circuit**

1. Patch encoding circuit
2. Logical-to-physical gate translation (intra-patch / inter-patch / transversal based on qubit placement)
3. Decoding, syndrome measurement, and final gate cancellation

The translator selects the cheapest available implementation for each gate automatically. Intra-patch swap gates are treated as free qubit relabelings — this assumption holds for architectures with flexible connectivity such as neutral atom arrays.

## Repository Structure

```
.
├── pipeline/           # High-level orchestration: run_full_scan, run_random_scan, etc.
├── iceberg/            # [[4, 2, 2]] Iceberg code implementation (IcebergModule, IcebergSet, postselection)
├── translator/         # Logical-to-physical circuit translation with merging and H commutation
├── packer/             # Merge-maximizing greedy qubit packing heuristic
├── permutations/       # Permutation generation and canonical-form deduplication
├── simulator/          # Qiskit Aer and Stim simulation backends with configurable noise models
├── evaluator/          # Post-hoc analysis: efficiency scores, effectiveness metrics
├── repository/         # Persistence layer for RunRecord/RunManifest (JSON + gzip)
├── experiments/        # Research experiment scripts (see Experiments section)
├── benchmarks/
│   └── qasmbench/      # Git submodule: QASMBench benchmark circuits (71 circuits used in paper)
├── circuits/           # DAG visualizers and circuit graph tools
├── dag/                # DAG representation utilities
└── results/            # Output directory for saved run records
```

## Installation

Requires **Python 3.12+**.

```bash
# Clone the repo with submodules
git clone --recurse-submodules https://github.com/Baker-Group-UT/iceberg-compiler.git
cd iceberg-compiler

# Create and activate a virtual environment (venv/ is gitignored)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Key dependencies: `qiskit==2.2.1`, `qiskit-aer==0.17.0`, `stim==1.15.0`, `pandas==2.2.3`, `numpy==2.2.4`, `scipy==1.15.2`, and `qiskit-qasm3-import==0.6.0`, which is needed to read back the OpenQASM 3 circuits stored in saved run records and generated circuit sets.

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

## Quick Start

```python
from qiskit import QuantumCircuit
from pipeline import run_full_scan, PipelineConfig
from simulator.config_presets import StandardSimulator

# Build a 4-qubit GHZ state circuit
qc = QuantumCircuit(4)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
qc.cx(2, 3)
qc.measure_all()

# Configure the pipeline
config = PipelineConfig(
    sim_config=StandardSimulator(repetitions=10_000),
    verbose=True,
    save=False,
)

# Run: enumerate all permutations, translate, simulate, postselect
record = run_full_scan(qc, config)

# Inspect results
print(record.dataframe.df)
# DataFrame columns include: permutation, codes_used, one_qubit_gates, two_qubit_gates,
# logical_fidelity, logical_selection_rate, logical_total_variation_distance, ...
```

To search only a random subset of permutations (useful for larger circuits):

```python
from pipeline import run_random_scan

record = run_random_scan(qc, m=40, config=config)  # sample 40 random permutations
```

To analyze specific permutations:

```python
from pipeline import run_specific_scan

# Permutation format: list of (patch_id, slot_index) per logical qubit
perms = [[(0, 0), (0, 1), (1, 0), (1, 1)],   # qubits 0,1 share patch 0; 2,3 share patch 1
          [(0, 0), (1, 0), (2, 0), (3, 0)]]   # all qubits in separate patches (sparse)
record = run_specific_scan(qc, perms, config)
```

## Key Modules

### `pipeline/`

Top-level orchestration. Main entry points:

| Function                                       | Description                                                                 |
| ---------------------------------------------- | --------------------------------------------------------------------------- |
| `run_full_scan(circuit, config)`               | Enumerate all unique permutations, translate, simulate, and collect results |
| `run_random_scan(circuit, m, config)`          | Sample `m` random permutations                                              |
| `run_specific_scan(circuit, perms, config)`    | Analyze a provided list of permutations                                     |
| `run_extremal_scan(circuit, config)`           | Find and compare best/worst permutations                                    |
| `translate_permutation(circuit, perm, config)` | Translate a single permutation to a physical circuit                        |
| `simulate_circuit(circuit, config)`            | Run ideal+noisy simulation and return distributions                         |

### `iceberg/`

Implements the [[4, 2, 2]] Iceberg code (and the general [[n, n-2, 2]] family).

- `IcebergModule` — single code patch with `n` physical qubits and `n-2` logical qubits
- `IcebergSet` — collection of patches forming the full physical register
- `iceberg_postselect()` — filters measurement outcomes where any syndrome bit is non-zero

### `translator/`

Translates logical gates to their physical Iceberg implementations:

- **Single-qubit gates** (H, X, Z, S, RX, RZ, SX): targeted or patch-wise depending on permutation
- **Two-qubit gates** (CX, CZ, XCX): intra-patch, inter-patch, or transversal based on qubit placement
- Gate merging pass consolidates adjacent compatible operations into cheaper compiled primitives
- Hadamard commutation pass reduces targeted H gates (the most expensive operation; see Table 1 in the paper)

### `packer/`

`MergeMaximizingPacker` implements the greedy pair-selection algorithm (Section 4.3 in the paper):

1. Run the gate merging algorithm on every candidate qubit pair to build a score matrix
2. Greedily commit the pair with highest net gain `w_ij - w_ii - w_jj`
3. Optionally apply grid alignment to minimize inter-patch communication cost

Composite pair score: `w_ij = α·s¹_ij + β·s²_ij - γ·Δ_ij + δ·T_ij` where `s¹`, `s²` are 1q/2q gate savings, `Δ` is a depth-based delay penalty, and `T` is a gate-type preference bias.

### `permutations/`

Generates all unique qubit-to-patch assignments. Deduplicates using canonical forms — permutations that pair the same qubits but differ only in patch ordering are treated as equivalent.

### `simulator/`

Two backends:

- **Qiskit Aer** (default): full density-matrix simulation with configurable depolarizing, thermal relaxation, and readout noise
- **Stim** (faster): stabilizer simulation via Clifford proxies; better for circuits with >10 logical qubits

Noise parameters match Table 2 in the paper: `p1 = 1e-3`, `p2 = 5e-3`, `pm = 2e-2`, `T1 = 100 μs`, `T2 = 80 μs`. Distance-dependent inter-patch noise scales with patch separation on the grid.

### `evaluator/`

Post-hoc analysis utilities:

- `add_efficiency_score()` — combined metric: gate count + λ × codes used
- `attach_effectiveness_columns()` — merge fidelity metrics onto summary DataFrames

### `repository/`

Persists results as `RunRecord` objects (JSON, optionally gzipped) with an integrity hash. Use `RunRepository` to save/load records and `RunBuilder` to accumulate results during a pipeline run.

## Configuration

### `PipelineConfig`

```python
@dataclass
class PipelineConfig:
    fault_tolerant: bool = False       # Enable fault-tolerant gate encodings
    verbose: bool = True               # Print progress during execution
    save: bool = False                 # Persist results to disk
    save_dir: str = "results/pipeline" # Output directory when save=True
    sim_config: SimulatorConfig = ...  # See below
    translator_config: TranslatorConfig = ...
    seed: int = None                   # Random seed
```

### `SimulatorConfig` presets

```python
from simulator.config_presets import (
    TestSimulator,      # 1 shot, for unit tests
    QuickSimulator,     # 100 shots: smoke tests only, too few for meaningful metrics
    StandardSimulator,  # 1000 shots (default noise model)
    ScaledSimulator,    # Scaled noise, useful for noise sweeps
    FinalizedSimulator, # 200,000 shots for publication-quality results
)
```

### `TranslatorConfig`

| Option                          | Default | Effect                                         |
| ------------------------------- | ------- | ---------------------------------------------- |
| `enable_merging`                | `True`  | Gate consolidation pass                        |
| `enable_h_commutation`          | `True`  | Hadamard commutation pass                      |
| `allow_transversal_swap_merges` | `True`  | Include SWAP gates in merge candidates         |
| `fault_tolerant`                | `False` | Use fault-tolerant gate implementations        |
| `n_code`                        | `4`     | Physical qubits per patch (default: [[4,2,2]]) |

## Experiments

Scripts in [`experiments/`](experiments/) reproduce figures and analyses from the paper:

| Script                       | Description                                                          |
| ---------------------------- | -------------------------------------------------------------------- |
| `run_scan.py`                | Basic end-to-end scan demonstrating the pipeline on a GHZ state      |
| `packing_violins.py`         | Distribution of fidelity improvements by packing density (Figure 11) |
| `tvd_vs_gate_count.py`       | TVD vs. physical gate count across permutations                      |
| `tvd_vs_gate_breakdown.py`   | TVD improvement broken down by gate type                             |
| `tvd_vs_repetitions.py`      | LSR/TVD anticorrelation as repetitions increase (Figure 17c)         |
| `tvd_vs_work_depth.py`       | TVD vs. compiled circuit depth                                       |
| `random_perm_improvement.py` | Framework improvement over random permutation baselines (Figure 12)  |
| `codes_used_vs_sel_rate.py`  | Tradeoff between patch count and logical selection rate              |
| `violins_30.py`              | Violin plots of metric distributions over permutation space          |

Helper utilities in `experiments/shared/` provide pre-built circuits (`example_circuits.py`), benchmark circuit loaders (`benchmark_circuits.py`), and random circuit generators (`random_circuits.py`).

## Benchmarks

The [`benchmarks/qasmbench/`](benchmarks/qasmbench/) directory is a git submodule pointing to [QASMBench](https://github.com/pnnl/QASMBench). The paper evaluates on 71 circuits selected from this suite, covering circuits up to 150 logical qubits and 2,500 gates after transpilation to the gate set {H, X, Z, CX, CZ, S, Rx, Rz}.

## Metrics

Results DataFrames include the following columns, as defined in Section 5.1.1 of the paper:

| Metric                                   | Definition                                                                |
| ---------------------------------------- | ------------------------------------------------------------------------- |
| `logical_fidelity`                       | Bhattacharyya fidelity between ideal and noisy postselected distributions |
| `logical_total_variation_distance` (TVD) | `½ Σ_x \|P_noisy(x) - P_ideal(x)\|` over postselected outcomes            |
| `logical_selection_rate` (LSR)           | Fraction of shots surviving Iceberg postselection (syndrome = 0)          |
| `one_qubit_gates`, `two_qubit_gates`     | Physical gate counts of the compiled circuit                              |
| `circuit_depth`                          | Compiled circuit depth                                                    |
| `codes_used`                             | Number of distinct [[4, 2, 2]] patches in the permutation                 |
| `merge_savings_1q`, `merge_savings_2q`   | Gate reduction from the merging pass                                      |

LSR and TVD are strongly anticorrelated under the [[4, 2, 2]] code: permutations that retain more shots after postselection consistently produce output distributions closer to ideal (see Appendix A.2 and Figure 17 in the paper).

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{mazzetti2026iceberg,
  title         = {Logical Compilation for Multi-Qubit Iceberg Patches},
  author        = {Cordell Mazzetti and Sayam Sethi and Rich Rines and Pranav Gokhale and Jonathan Mark Baker},
  year          = {2026},
  eprint        = {2604.09956},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  url           = {https://arxiv.org/abs/2604.09956},
}
```
