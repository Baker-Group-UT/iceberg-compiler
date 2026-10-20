# Rebuilt arXiv data figures

The data figures from `AQORN_arXiv.pdf`, regenerated with the current compiler. The
draft's other figures (1–4, 6–9, 14–16) are schematics rather than measurements and
are not reproduced here.

| script | draft figure | needs simulation | runtime |
|---|---|---|---|
| `fig05_cnot_implementations.py` | 5 — CNOT implementations under noise | yes | ~1 h |
| `fig10_spacetime_reduction.py` | 10 — QASMBench resource reduction | no | ~15 min |
| `fig11_density_sweep.py` | 11 — improvement vs packing density | no | ~20 min |
| `fig12_random_circuits.py` | 12 — random-circuit technique benefits | no | ~2 min |
| `fig13_ablation_sweep.py` | 13 — ablation over density, 5 metrics | optional | min / hours |
| `fig17_lsr_tvd_correlation.py` | 17 — LSR vs TVD anticorrelation | yes | ~1 h |
| `fig_k_sweep.py` | **new** — everything as a function of *k* | no | ~20 min |
| `fig_pass_order.py` | **new** — merging/packing without commutation first | no | ~10 min |

Shared code lives in `common.py` (circuits, placement, gate metrics, plot style) and
`simulation.py` (noise calibration, LSR/TVD). Every script takes `--from-csv` to
replot without recomputing, and writes both `.svg` and `.pdf`.

## What changed underneath

Three differences from the scripts the paper was generated with, and they are why
numbers move:

**Merging runs at every code size.** It was [[4,2,2]]-only, so every figure was
implicitly a k=2 figure. `--n-code` now sweeps the code family. Whole-patch macros
(`hh`, `cxt`, `czt`, `xcxt`) need a gate on *every* occupied lane, so they get harder
to trigger as k grows; per-lane macros (`xx`, `zz`) fuse on any subset.

**Placement is the `grouping` stack**, not `packer/merge_packer.py`. A signed
interaction graph, multilevel partitioning, then exact Hungarian slot assignment per
patch pair. The old packer commits qubits in pairs and cannot express a placement at
k > 2 at all.

**Placements are scored analytically** (`grouping/analytic.py`) rather than by
translating the circuit. That is up to 4000× faster and is the only reason the large
benchmarks are runnable — the old search needed ~5.5 hours for a single 140-qubit
point. The model is exact for gate counts and approximates merging to 2–3%; regret
against the true optimum is 0.00–0.18% at k ≥ 4.

## Things to know before quoting a number

**Bars clip negative segments to zero,** matching the draft's stacked figures. A
technique that hurts on some circuit is drawn as zero, so the plotted total is an
upper bound. Every script prints unclipped means too — read those first.

**Depth is priced, and it has to be.** Intra- and inter-patch CNOT both cost `(0, 4)`
gates, so on gate counts alone a CNOT-only circuit (GHZ, cat, Bernstein–Vazirani) has
*no* grouping signal and the partition comes out arbitrary. Depth breaks the tie —
inter-patch CNOT is a layer shallower — so `depth_weight` is nonzero by default and set
from the grid search in `grouping/tune.py`. Placements are then refined on the analytic
cost itself, because the graph is a pairwise proxy for a conjunctive objective (macros
need *every* lane) that no edge weight can express.

**Simulated runs are unseeded, though no longer out of necessity.**
`Simulator._shot_chunk_size_for` caps a chunk at `repetitions // 100`, and one fixed
`seed_simulator` on the backend used to make every chunk a replay of the first — a
*seeded* run yielded `repetitions / 100` independent samples and about 10× the sampling
error its shot count implied. Chunks now derive their own seeds from the configured one
(`tests/test_simulator_seeding.py`), so seeding is safe; these scripts stay unseeded so
that a borderline result is not borderline identically on every rerun.

**Noise is calibrated per circuit.** Postselection is an exponentially strict filter,
so a fixed noise scale either keeps everything or nothing. Each circuit gets two
pilot runs and alpha is solved so the baseline selection rate lands near 12%.

## Reproducing

```bash
PYTHONPATH=. venv/bin/python -m experiments.figures.updated.fig10_spacetime_reduction
PYTHONPATH=. venv/bin/python -m experiments.figures.updated.fig_k_sweep --groups small medium large
PYTHONPATH=. venv/bin/python -m experiments.figures.updated.fig13_ablation_sweep --no-sim
```
