# Iceberg Compiler

Open-source implementation for the paper **"Logical Compilation for Multi-Qubit Iceberg Patches"** by Cordell Mazzetti, Sayam Sethi, Rich Rines, Pranav Gokhale, and Jonathan Mark Baker.

> **Paper:** [arXiv preprint](https://arxiv.org/abs/2604.09956)

## Overview

This framework solves the **qubit packing problem** for the [[n, n−2, 2]] Iceberg quantum error-detecting (QED) code family: given a program circuit, which logical qubits should share a physical code patch, and how should those patches be laid out? The choice of mapping determines which gate implementations are available — intra-patch, inter-patch, or transversal — producing up to 4× variation in gate cost and circuit depth (see Table 1 and Figure 4 in the paper).

The compiler pipeline takes an arbitrary Clifford+Rz program circuit and produces a fully encoded physical circuit, using Hadamard commutation, gate merging, and qubit packing. Across 71 QASMBench circuits at [[4, 2, 2]], this reduces one- and two-qubit gate counts by 23% and 17% and circuit depth by 33% relative to a naive compilation baseline.

> **Beyond [[4, 2, 2]].** The paper evaluates the [[4, 2, 2]] code, where a patch holds two logical qubits. The compiler now supports the whole **[[n, n−2, 2]]** family at any even n ≥ 4, with gate merging and packing working at every code size.
>
> **Depth improves 7–8 points past k = 2** — 31.4% at k = 2 against 39.3% at k = 6 on the medium group, with every benchmark placing at every code size. Gate counts do not follow: they decline slowly with k, because merging's whole-patch macros need a gate on every occupied lane at once and fire less often as k grows. So the multi-qubit-patch argument holds for depth and not for gate count. See §3.3 of [STATUS.md](STATUS.md), which records the two wrong answers it took to get there.
>
> Two more things to know before comparing against published numbers. The one-qubit gate figure above (23%) is lower than the paper's 31% and the discrepancy is not yet explained. And the compiler on `main` mistranslates any program containing a `swap`, an `sx` or an `xcx`, and silently drops any gate it cannot place — ten pre-existing correctness bugs, since fixed — so data collected before this branch should be regenerated.
>
> **[STATUS.md](STATUS.md) is the current-state document:** capabilities, limitations, and what changed across the branches since `main`. Read it before quoting any number from this repository.

## Compilation Pipeline

The pipeline follows three phases (Figure 9 in the paper):

**Phase 1 — Program circuit optimization**

1. Gate cancellation
2. Hadamard commutation: push all H gates to the end of the circuit, canceling pairs, leaving at most one H per qubit
3. Second gate cancellation pass

**Phase 2 — Logical circuit: packing and merging**

1. Build a signed interaction graph over logical qubits, weighted by gate count *and* depth
2. Multilevel partition into patches, then exact Hungarian slot assignment within each patch pair
3. Local search on the analytic cost model, verified against the naive placement so the result is never worse
4. Circuit rewrite: substitute mergeable gate groups with cheaper whole-patch or transversal primitives

**Phase 3 — Physical circuit**

1. Patch encoding circuit
2. Logical-to-physical gate translation (intra-patch / inter-patch / transversal based on qubit placement)
3. Decoding, syndrome measurement, and final gate cancellation

The translator selects the cheapest available implementation for each gate automatically. Intra-patch swap gates are treated as free qubit relabelings — this assumption holds for architectures with flexible connectivity such as neutral atom arrays.

The paper's Phase 2 is the pairwise greedy heuristic in [`archive/packer/`](archive/packer/), which scores every qubit pair and commits the best. It is archived for reference but is superseded: it cannot express a placement at k > 2, since a patch then holds a *group* whose value is not the sum of its pairs. See [`packing/`](packing/) and §2.4 of [STATUS.md](STATUS.md).

## Repository Structure

```
.
├── pipeline/           # High-level orchestration: run_full_scan, run_random_scan, etc.
├── iceberg/            # [[n, n-2, 2]] code family: layout, physical gates, cost/depth tables
├── translator/         # Logical-to-physical translation with merging and H commutation
├── packing/            # Qubit packing: incremental cost objective, graph seed, descent (any k)
├── permutations/       # Permutation generation and canonical-form deduplication
├── simulator/          # Qiskit Aer and Stim simulation backends with configurable noise models
├── analysis/           # Post-hoc analysis: gate counts, postselection, fidelity/TVD metrics
├── experiments/        # Research experiment scripts (see Experiments section)
│   ├── shared/         # Circuit loaders and generators
│   ├── tests/          # testexp.py, the translator fuzz driver
│   └── figures/merging_improvement/  # The code-size sweep, ported to `packing`
├── benchmarks/
│   └── qasmbench/      # Git submodule: QASMBench benchmark circuits (71 circuits used in paper)
├── tests/              # Test suite; pytest.ini collects this directory only
└── archive/            # Superseded but still importable: grouping/, packer/, repository/,
                        #   evaluator/, circuits/, dag/, and most of experiments/.
                        #   See archive/README.md for the full mapping
```

## Installation

Requires **Python 3.13+**.

```bash
# Clone the repo with submodules
git clone --recurse-submodules https://github.com/Baker-Group-UT/iceberg-compiler.git
cd iceberg-compiler

# Install dependencies
pip install -r requirements.txt
```

Key dependencies: `qiskit==2.2.1`, `qiskit-aer==0.17.0`, `stim==1.15.0`, `pandas==2.2.3`, `numpy==2.2.4`, `scipy==1.15.2`.

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
print(record.dataframe)
# DataFrame columns include: permutation, codes_used, total_1q_gates, total_2q_gates,
# logical_fidelity, logical_selection_rate, logical_total_variation_distance, ...
```

To search only a random subset of permutations (useful for larger circuits):

```python
from pipeline import run_random_scan

record = run_random_scan(qc, config, m=40)  # sample 40 random permutations
```

To analyze specific permutations:

```python
from pipeline import run_specific_scan

# Permutation format: list of (patch_id, slot_index) per logical qubit
perms = [[(0, 0), (0, 1), (1, 0), (1, 1)],   # qubits 0,1 share patch 0; 2,3 share patch 1
          [(0, 0), (1, 0), (2, 0), (3, 0)]]   # all qubits in separate patches (sparse)
record = run_specific_scan(qc, config, perms)
```

## Key Modules

### `pipeline/`

Top-level orchestration. Main entry points:

| Function                                       | Description                                                                 |
| ---------------------------------------------- | --------------------------------------------------------------------------- |
| `run_full_scan(circuit, config)`               | Enumerate all unique permutations, translate, simulate, and collect results |
| `run_random_scan(circuit, config, m)`          | Sample `m` random permutations                                              |
| `run_specific_scan(circuit, config, perms)`    | Analyze a provided list of permutations                                     |
| `run_extremal_scan(circuit, config)`           | Find and compare best/worst permutations                                    |
| `translate_permutation(circuit, perm, config)` | Translate a single permutation to a physical circuit                        |
| `simulate_circuit(circuit, config)`            | Run ideal+noisy simulation and return distributions                         |

### `iceberg/`

Implements the [[n, n−2, 2]] Iceberg code family at any even n ≥ 4.

- `IcebergModule` — single code patch with `n` physical qubits and `n-2` logical qubits
- `IcebergSet` — collection of patches forming the full physical register
- `layout.py` — the code convention in one place (anchors, lanes, decode rule); gates, decoder and cost model all consult it
- `physical.py` — physical gate emission and the flagged fault-tolerant gadget
- `costs.py` — `work_table(n)` for (1q, 2q) gate counts and `depth_table(n)` for depth, both measured from the emitted circuits and pinned in tests
- `iceberg_postselect()` — filters measurement outcomes where any syndrome bit is non-zero
- `python -m iceberg.demo` — encodes a program at several code sizes and prints the circuits

The identity that makes the family work: **transversal H equals H^k composed with `C_A = I + J`, and `C_A` is a physical swap of the two anchors.** So the whole-patch Hadamard is transversal H plus one SWAP at any n, with no frame tracking and no cost penalty.

### `translator/`

Translates logical gates to their physical Iceberg implementations:

- **Single-qubit gates** (H, X, Z, S, RX, RZ, SX): targeted or patch-wise depending on permutation
- **Two-qubit gates** (CX, CZ, XCX): intra-patch, inter-patch, or transversal based on qubit placement
- Gate merging pass consolidates adjacent compatible operations into cheaper compiled primitives, at any k
- Hadamard commutation pass reduces targeted H gates (the most expensive operation; see Table 1 in the paper)

Merging distinguishes **whole-patch** macros (`hh`, `cxt`, `czt`, `xcxt`), which act on every lane whether asked to or not and so require a gate on every *occupied* lane, from **per-lane** macros (`xx`, `zz`), which fuse on any subset. At k = 2 the two rules coincide, which is why the original pairwise implementation never had to distinguish them.

### `packing/`

Qubit packing at any k. `plan_packing` is the entry point:

1. `cost` — `PackingCost`, the objective as an incremental function of the partition, so a move rescores only the terms it touches. It scores a placement in ~0.1 ms instead of the ~700 ms a translation costs at 140 qubits, and `tests/test_packing.py` pins it against the `AnalyticCost` it replaces (kept importable at `archive/grouping/analytic.py` for exactly that check)
2. `graph`, `seed` — a signed interaction graph: negative weight where a pair is cheaper apart, priced on gate count *and* depth, plus Hadamard-coincidence and transversal-affinity edges. It seeds the search, and nothing after the seed consults it again
3. `partition` — coarsen/solve/uncoarsen partitioning of that graph
4. `search` — `descend`: move / spread / swap sweeps on the real objective, since the graph is a pairwise proxy for a conjunctive objective it cannot express
5. `lanes` — exact slot assignment, against every already-placed patch
6. `plan` — chooses *how many* patches to spend under a `ResourceBudget` (`packing/budget.py`) and `NoiseProfile` (`packing/profiles.py`), rather than assuming the tightest packing

The final candidate is checked against the naive placement by exact measurement, so packing is never a regression.

```python
from packing import plan_packing
from packing.budget import ResourceBudget

plan = plan_packing(qc, n_code=6, budget=ResourceBudget(max_physical_qubits=256))
print(plan.summary())   # patches, occupancy, physical qubits, cost, moves
record = run_specific_scan(qc, config, [plan.placement])
```

`plan_packing` needs a code size. `compile_circuit` (`packing/compiler.py`) is the entry point that does not: it runs `packing.resource.select_resources` to choose the code size and patch count, runs the placement search at that code size, then translates the planner's placement, the selector's placement and the naive packing and keeps whichever measures best on expected detectable errors. The winner's translation is the returned circuit, so nothing is translated twice.

```python
from packing import compile_circuit
from packing.resource import NoiseKnobs

built = compile_circuit(qc, budget=ResourceBudget(max_physical_qubits=256))
print(built.summary())      # code size, patches, density, error split, which arm won
built.circuit               # the physical circuit, ready to simulate
built.resources.frontier    # every (code size, patch count) candidate considered
```

The `noise=NoiseKnobs(...)` argument is the one machine description: it ranks the resource candidates directly and, converted to weighted-gate units, tells the placement search what movement and idling cost. `NoiseKnobs(density_weight=...)` trades accepted-shot rate against how many logical qubits sit behind each detector pair; it has no defensible default other than 0, which optimises throughput alone and picks the tightest packing.

`python -m packing.bench` compares packing strategies by exact translation; `--techniques` reports the h-commutation / merging / packing decomposition instead. It replaces the archived `grouping.evaluate`; `grouping.tune`, which grid-searched `depth_weight`, has no live equivalent and is superseded (`archive/grouping/`).

### `archive/packer/` — archived

`MergeMaximizingPacker` implements the greedy pair-selection algorithm (Section 4.3 in the paper):

1. Run the gate merging algorithm on every candidate qubit pair to build a score matrix
2. Greedily commit the pair with highest net gain `w_ij - w_ii - w_jj`
3. Optionally apply grid alignment to minimize inter-patch communication cost

Composite pair score: `w_ij = α·s¹_ij + β·s²_ij - γ·Δ_ij + δ·T_ij` where `s¹`, `s²` are 1q/2q gate savings, `Δ` is a depth-based delay penalty, and `T` is a gate-type preference bias.

**Superseded by `packing/`, and archived to [`archive/packer/`](archive/packer/).** It is k = 2 only by construction, and a correlation study over 3,600 placements found that measured cost tracks cross-patch two-qubit gates at +0.80 while tracking the merge savings this packer maximises at **−0.17** — so the objective is the wrong one even at k = 2. `packing/` imports nothing from it; it is retained as a reference point only.

### `permutations/`

Generates all unique qubit-to-patch assignments. Deduplicates using canonical forms — permutations that pair the same qubits but differ only in patch ordering are treated as equivalent.

### `simulator/`

Two backends:

- **Qiskit Aer** (default): full density-matrix simulation with configurable depolarizing, thermal relaxation, and readout noise
- **Stim** (faster): stabilizer simulation via Clifford proxies; better for circuits with >10 logical qubits

Noise parameters match Table 2 in the paper: `p1 = 1e-3`, `p2 = 5e-3`, `pm = 2e-2`, `T1 = 100 μs`, `T2 = 80 μs`. Distance-dependent inter-patch noise scales with patch separation on the grid, computed from the circuit's own registers (which also gives fault-tolerant flag ancillas a real coordinate rather than attributing them to nonexistent patches).

> **Seeded Aer runs before commit `864d9f4` used 1% of their shots.** Aer's `seed_simulator` seeds a *run*, not a backend, and the simulator built one backend then called `run()` once per shot chunk — so every chunk replayed the first. Chunks now derive their own seeds from the configured one, pinned by `tests/test_simulator_seeding.py`. Noisy numbers collected before that commit carry roughly ten times the sampling error their shot count implies; Stim-backed results were never affected.

### `analysis/`

Post-hoc analysis utilities. The old `evaluator/` and `pipeline/analysis/` merged into this one package:

- `gate_count_stats` — one- and two-qubit gate totals; the metric every benchmark here is scored with
- `postselection` — Iceberg postselection and the selection rate that comes with it
- `metrics` — fidelity and TVD; `result_row` and `result_types` turn simulation results into DataFrame rows
- `extremal_selection` — pick the best and worst permutations out of a scan

`add_efficiency_score()` and `attach_effectiveness_columns()` were not carried over. They are archived at [`archive/evaluator/evaluator.py`](archive/evaluator/evaluator.py).

### `archive/repository/` — archived

Persisted results as `RunRecord` objects (JSON, optionally gzipped) with an integrity hash, via `RunRepository` and `RunBuilder`. Archived to [`archive/repository/`](archive/repository/): the scan tasks now return `analysis.result_types.ScanResult` and write nothing to disk, so `PipelineConfig.save` no longer has a store behind it.

## Configuration

### `PipelineConfig`

```python
@dataclass
class PipelineConfig:
    fault_tolerant: bool = False       # Enable fault-tolerant gate encodings
    verbose: bool = True               # Print progress during execution
    save: bool = False                 # Inert: the persistence layer is archived
    save_dir: str = "results/pipeline" # Inert, see `archive/repository/`
    sim_config: SimulatorConfig = ...  # See below
    translator_config: TranslatorConfig = ...
    seed: int = None                   # Random seed
```

### `SimulatorConfig` presets

```python
from simulator.config_presets import (
    TestSimulator,      # 1 shot, for unit tests
    QuickSimulator,     # 100 shots
    StandardSimulator,  # 1000 shots (default noise model)
    ScaledSimulator,    # Scaled noise, useful for noise sweeps
    FinalizedSimulator, # 200,000 shots for publication-quality results
)
```

### `TranslatorConfig`

| Option                          | Default            | Effect                                                                |
| ------------------------------- | ------------------ | --------------------------------------------------------------------- |
| `n_code`                        | `4`                | Physical qubits per patch; any even n ≥ 4, giving k = n−2 logical qubits |
| `enable_merging`                | `True`             | Gate consolidation pass                                               |
| `enable_h_commutation`          | `True`             | Hadamard commutation pass                                             |
| `h_commutation_stage`           | `"before_merging"` | When commutation runs relative to merging: `before_merging` (the paper's order), `after_merging`, or `never` |
| `allow_transversal_swap_merges` | `True`             | Include SWAP gates in merge candidates                                |
| `fault_tolerant`                | `False`            | Use fault-tolerant gate implementations                               |
| `encode_override`               | `False`            | Skip decode and measurement (for inspecting the encoded circuit)      |

`h_commutation_stage` exists so pass-order attribution can be measured rather than argued about. The paper notes that running commutation first hands it the largest share of the credit; the numbers in §3.2 of [STATUS.md](STATUS.md) came from `fig_pass_order.py`, now archived at `archive/experiments/figures/updated/`, and the nearest live measurement is `python -m packing.bench --techniques`, which decomposes h commutation, merging and packing but does not sweep the stage itself. All three orders are verified to preserve the logical action exactly at n ∈ {4, 6, 8, 10, 12}.

## Experiments

### Live — `experiments/figures/merging_improvement/`

The one figure directory still in the live tree, ported to `packing`. Six scripts, all of which run; `merging_improvement_code_size.py` is the code-size sweep with per-size-group bars ([writeup](experiments/figures/merging_improvement/CODE_SIZE_SWEEP.md)).

The same three-technique measurement, without the figure, is in `packing/bench.py`:

```bash
PYTHONPATH=. venv/bin/python -m packing.bench --techniques --code-sizes 4 6 8 10 12
```

`--code-sizes` takes n, not k, so that line is k = 2…10. Drop `--techniques` to compare packing strategies against each other instead, by exact translation.

### Archived — `archive/experiments/figures/updated/`

These recreated the paper's **data** figures on the placement and merging stack of the time, each script taking `--from-csv` to replot without recomputing and `--n-code` to change the code size, which the originals could not do. **They are archived and will not run as-is** — they import `grouping`, `packer` and the pre-strip `pipeline`. Kept as the record of how the numbers quoted in [STATUS.md](STATUS.md) were produced. Results and caveats: [RESULTS.md](archive/experiments/figures/updated/RESULTS.md).

| Script                        | Description                                                                    |
| ----------------------------- | ------------------------------------------------------------------------------ |
| `fig10_spacetime_reduction.py`| Gate and depth reduction over 71 QASMBench circuits (Figure 10)                 |
| `fig11_density_sweep.py`      | Improvement over random permutations vs packing density (Figure 11)            |
| `fig12_random_circuits.py`    | Reduction on random circuits by size group (Figure 12)                          |
| `fig13_ablation_sweep.py`     | Per-technique ablation over density (Figure 13)                                 |
| `fig05_cnot_implementations.py` | Fidelity of the three CNOT implementations (Figure 5)                        |
| `fig17_lsr_tvd_correlation.py`| LSR/TVD anticorrelation (Figure 17)                                             |
| `fig_k_sweep.py`              | **New** — full-pipeline reduction against code size, k = 2…10                    |
| `fig_pass_order.py`           | **New** — what merging and packing are worth without commutation first           |

### Archived — the original scripts, now under `archive/experiments/`

Written against the paper's compiler. They still import — self-references were rewritten to `archive.*` — but they predate the correctness fixes and the packing rewrite, so their numbers are not comparable to the figures above.

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

### Debugging harness

`python -m experiments.tests.testexp` generates a random Clifford program, simulates it as a reference, then translates it onto a random permutation at k = 2, 4 and 6 — printing the seed, permutation, logical circuit and translated circuit, and flagging any code size whose postselected distribution disagrees with the reference. Loops on the seed.

## Benchmarks

The [`benchmarks/qasmbench/`](benchmarks/qasmbench/) directory is a git submodule pointing to [QASMBench](https://github.com/pnnl/QASMBench). The paper evaluates on 71 circuits selected from this suite, covering circuits up to 150 logical qubits and 2,500 gates after transpilation to the gate set {H, X, Z, CX, CZ, S, Rx, Rz}.

## Metrics

Results DataFrames include the following columns, as defined in Section 5.1.1 of the paper:

| Metric                                   | Definition                                                                |
| ---------------------------------------- | ------------------------------------------------------------------------- |
| `logical_fidelity`                       | Bhattacharyya fidelity between ideal and noisy postselected distributions |
| `logical_total_variation_distance` (TVD) | `½ Σ_x \|P_noisy(x) - P_ideal(x)\|` over postselected outcomes            |
| `logical_selection_rate` (LSR)           | Fraction of shots surviving Iceberg postselection (syndrome = 0)          |
| `raw_fidelity`                           | Fidelity before postselection                                             |
| `total_1q_gates`, `total_2q_gates`       | Physical gate counts of the compiled circuit                              |
| `depth`                                  | Compiled circuit depth                                                    |
| `codes_used`                             | Number of distinct patches in the permutation                             |
| `merge_savings_1q`, `merge_savings_2q`   | Gate reduction from the merging pass                                      |

LSR and TVD are strongly anticorrelated under the [[4, 2, 2]] code: permutations that retain more shots after postselection consistently produce output distributions closer to ideal (see Appendix A.2 and Figure 17 in the paper). Reproduced on the current compiler at Pearson −0.913 / Spearman −0.906.

## Testing

```bash
PYTHONPATH=. venv/bin/python -m pytest -q     # 358 tests
```

The correctness tests are exact rather than statistical where possible: `test_iceberg_logical_action.py` and `test_pipeline_logical_action.py` compare the logical Clifford *induced by the translated physical circuit* against the program, via symplectic tableau, at n ∈ {4, 6, 8}. The round-trip tests cover what those skip — encode, decode, postselect — and derive their sampling bounds from multinomial error rather than fixing them, since TVD and Bhattacharyya deficit both grow with support size and a constant threshold goes flaky once a program spreads over enough outcomes.

`tests/random_programs.py` generates over the translator's own gate set rather than using `qiskit.random_clifford`, which transpiles to gates the translator has never seen.

See §2.6 of [STATUS.md](STATUS.md) for what each suite pins.

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
