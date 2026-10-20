# Packers and partitioners against AQORN, under the current standard (2026-10-10)

Branch `v2-logical-opt-pass`, compiler frozen at 1f83c4f (standard w10T + solo-HH pricing +
logical optimisation pass with the no-new-rotations rule). Repeats the packer comparison; the Oct 1 Experiment1 figure used the submitted planner.

Method: `experiments/optimality/standard_packers.py` -- Experiment1's packers, the model-based ones
searching AQORN's own objective; every arm at AQORN's H-commutation decision. Each placement is
measured by `placement_measure.measure`: static counts of the original's translation, AQORN's own
criterion AH*, Pauli frames on the Clifford proxy at alpha = 1 and at the circuit's Fig 6 alpha
(LSR, occupied-lane logical error rate, exact TVD), and a noiseless correctness check (every
placement passed). Numbers below are at the Fig 6 alpha.

## 1. Corpus: 71 Fig 6 circuits x 11 packers (`packers_corpus.py`, figure `packers_corpus_v2.pdf`)

| packer | kept shots vs AQORN (geomean) | LER vs naive (median) | TVD vs naive (median, 59 circuits) |
|---|---|---|---|
| interaction matching | 1.15x | -0.5% | -1.0% |
| [24] stage 1 | 1.13x | -0.5% | -0.9% |
| interaction partition | 1.12x | -1.7% | -1.6% |
| naive (program order) | 1.01x | 0 | 0 |
| AQORN, verify off | 1.00x | -12.9% | -16.3% |
| **AQORN** | **1.00x** | **-15.7%** | **-16.2%** |
| interaction partition + AQORN's descent | 1.00x | -17.4% | -16.4% |
| AQORN descent from naive | 0.99x | -15.8% | -15.4% |
| AQORN seed only | 0.94x | -12.1% | -14.5% |
| pairwise greedy (earlier) | 0.94x | -4.2% | -4.2% |
| random, AQORN's budget (on AQORN's objective) | 0.87x | -15.7% | -14.3% |

- **Interaction-graph packers keep the most shots and make no fewer logical errors.** Matching,
  [24] stage 1 and the interaction partition keep 12-15% more shots than AQORN but cut logical
  error / TVD by under 2% against naive: they group the qubits that interact, which puts their
  gates in-patch, where they leak silent errors.
- **AQORN's objective is what buys accuracy, the seed matters little.** AQORN cuts logical error
  16% and TVD 16% against naive at naive's kept-shot rate. Its descent from naive, or from the
  interaction partition, lands in the same place; the signed-graph seed alone falls short
  (-12% LER, 0.94x shots). Random search on the same objective with AQORN's evaluation budget
  matches the accuracy but loses 13% of the shots.

## 2. Simulated annealing on AQORN's objective (`--arms aqorn annealing`, 120 s per circuit)

`annealing_summary.py`, 71 circuits, annealing vs AQORN (negative = annealing lower):

| | median | mean | annealing better on |
|---|---|---|---|
| search objective (model value) | 0.00% | -0.61% | 45% (equal on 35%) |
| AQORN's full criterion AH* | 0.00% | +0.42% | 18% (equal on 46%) |
| measured LSR | -0.27% | +1.13% | 48% |
| measured LER | +0.34% | +0.37% | 44% |

- **AQORN's search is at annealing's level on most circuits**: identical below 10 qubits; the
  measured outcomes do not favour annealing (LSR and LER within noise, either way about half the
  time).
- **Where annealing finds better optima of the model**: the adders and the Ising chains --
  adder_n64 -13.3%, adder_n28 -9.3%, ising_n26 -6.0%, ising_n42 -4.3%, ising_n34 -3.6% model
  value. These mostly do not carry over to AH* (adder_n64 and n28 +1%), except ising_n34 (AH*
  -9.6%, LSR +72%). A known weak spot of the descent (the Ising family also produced the
  ising_n98 depth outlier); fixing it is a compiler change, so not under the freeze.

## 3. One larger program in full: multiplier_n15 (`sampled_front.py`, figure `sampled_front_multiplier_n15.pdf`)

2,000 uniform random placements, 300 with AQORN's grouping and random lanes / patch order, 12
packers; all pass the noiseless check.

- AQORN beats **100%** of random placements on LSR and **97.9%** on logical error; none beats it
  on both. Against the random median: 0.120 vs 0.077 kept shots (+56%), 0.089 vs 0.102 logical
  error (-13%).
- The top cluster -- AQORN, its ablations, annealing (which at 300 s finds AQORN's grouping),
  the interaction partition with and without descent -- is within ~2% on logical error (about
  one standard error at 24k kept shots). Naive -24% LSR / +31% LER, pairwise greedy -38% / +19%,
  interaction matching -6% / +27%, [24] stage 1 -21% / +57%.
- AQORN's grouping with random lanes and patch order: identical gate counts and depth, logical
  error 0.084-0.095 (cross-validated worst vs best 4.2%) -- the which-atoms-move effect at 8
  patches.

## Incidents (fixed before the numbers above)

- A 64-bit pack of per-qubit bits wrapped from qubit 64 on: exact TVD was wrong for 11 of the 14
  >= 64-qubit circuits and the noiseless check failed every arm of adder_n64 / wstate_n76 /
  wstate_n118. Fixed (82bc36d, bit matrices; equal to the old code on 300 random small cases) and
  the 14 circuits re-measured. LER, LSR and everything below 64 qubits were unaffected.
- Committing that fix during the annealing run made 15 circuits' workers mix old and new modules
  (ImportError); their annealing rows were re-run.
