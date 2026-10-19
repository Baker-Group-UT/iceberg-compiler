# Concatenation under the current standard: simplified figures (A and C)

v2 revision, 2026-10-10. Branch `v2-concat-standard` (from `v2-logical-opt-pass`
9ac2dfc), worktree `.claude/worktrees/concat-standard`. Run: commit 84fce00 (clean),
`experiments/concatenation/concat_standard.py --jobs 12`, 00:52 - 01:05 (13 min), 26/26 circuits.
Replaces the figure `concat_results_v3` of [concatenation.md](concatenation.md) (2026-10-07).

## What was run

- **Configuration**: the current standard.
  - Logical optimisation first: `verified-routes-2`, on every program.
  - Outer plans from `plan_standard` (solo-HH aware). They equal Fig 6's plans under the pass on 26/26 circuits.
  - Transversal in-patch CZ / XCX on AQORN's arms, at both levels.
  - Every program layer-sorted (`asap_order`).
  - The inner level never runs the logical optimisation pass. Its program C1 is the outer code's own circuit (encoder, gadgets, syndrome extraction), and a unitary-preserving rewrite of it would change which faults the outer checks catch.
- **Arms** (7, down from 11):

  | arm | outer level | inner level |
  |---|---|---|
  | single_base | baseline | -- |
  | single_std | AQORN | -- |
  | concat_BB | baseline | baseline |
  | concat_AB | AQORN | baseline |
  | concat_BA | baseline | AQORN |
  | concat_AA | AQORN | AQORN |
  | concat_AA_index | AQORN | index layout, not spread |

  - Baseline: tight trivial placement, no AQORN passes.
  - Inner baseline: the stride layout, passes off.
  - Inner AQORN: spread-restricted placement search plus passes.
- **Noise**: each circuit's calibrated alpha, at which concat_BB keeps half its shots (median alpha 0.31, so p2 = 9.2e-4). Also fixed alpha 0.003, 0.01, 0.03, 0.1, 0.3, 1 (p2 = 3e-3 alpha). The grid now reaches the calibrated regime.
- **Measurement**:
  - 200k Pauli-frame shots per point for LSR and logical error rate.
  - 100k real samples for TVD.
  - Noiseless check: every arm keeps every shot (LSR 1.000), and TVD is at its sampling floor.
- **Circuits**: the 26 Fig 6 circuits with at most 8 qubits, as Clifford proxies of the optimised programs. That is 15 physical qubits on average at one level and 46 at two.

"Logical error rate" is the code's `postselect_silent`: among kept shots, the fraction with a wrong decoded output.

## Figures

![A](../../../../../results/figures/concat_gain.pdf)
`results/figures/concat_gain.pdf` (option A, one column)

![C](../../../../../results/figures/concat_tradeoff.pdf)
`results/figures/concat_tradeoff.pdf` (option C, one panel)

**Caption draft, A.**
*AQORN on a concatenated code: [[4,2,2]] inside [[4,2,2]], spread layout, [[16,4,4]] on the data.
26 benchmark circuits with at most 8 qubits, post-selection at both levels.*
- *(a) Reduction against the same compiler without AQORN, at one level and at two levels (AQORN at both). Measured at each circuit's calibrated noise, where the two-level baseline keeps half its shots. Bars show the mean over circuits with a bootstrap 95% CI; labels count the circuits that improve.*
- *(b) Logical error rate (wrong decoded output among kept shots), pooled over circuits, against the physical two-qubit error rate; all error rates scale together. The shaded band marks the calibrated noise of (a), interquartile range.*

**Caption draft, C.**
*Shots kept against logical error rate at the calibrated noise. Small markers are individual circuits; large markers show the mean LSR and geometric-mean logical error rate per configuration. AQORN improves both axes at one level and at two; concatenation trades shots for accuracy.*

## Numbers

Paired change against the arm's own baseline, at calibrated noise. Mean (median), with circuits better out of 26:

| | 1q | 2q | depth | LSR | logical error | its geo-mean ratio |
|---|---|---|---|---|---|---|
| 1 level: AQORN vs baseline | -4.1 (-9.1), 8 | +20.8 (+19.3), 24 | +16.8 (+13.1), 20 | +1.0, 23 | +48.6 (+47.1), 26 | 2.1x lower |
| 2 levels: AQORN both | +34.3 (+32.4), 25 | +41.8 (+40.0), 26 | +49.9 (+48.8), 26 | +30.0, 26 | +53.2 (+52.9), 25 | 2.4x lower |
| 2 levels: outer only | -6.6, 7 | +12.3, 22 | +10.8, 18 | +9.4, 23 | +36.6, 23 | 1.75x lower |
| 2 levels: inner only | +39.6, 25 | +33.8, 26 | +43.2, 26 | +22.4, 26 | +46.5, 26 | 2.2x lower |
| 2 levels: index order | +30.1, 23 | +40.0, 26 | +48.6, 26 | +30.8, 26 | -88.0, 4 | 1.74x **higher** |

Pooled logical error rate (frames), by p2:

| p2 | 9e-6 | 3e-5 | 9e-5 | 3e-4 | 9e-4 | 3e-3 | slope |
|---|---|---|---|---|---|---|---|
| 1 level, baseline | 1.9e-4 | 5.9e-4 | 1.7e-3 | 5.3e-3 | 1.3e-2 | 3.1e-2 | 0.88 |
| 1 level, AQORN | 1.1e-4 | 3.6e-4 | 1.1e-3 | 3.3e-3 | 7.9e-3 | 1.8e-2 | 0.89 |
| 2 levels, baseline | 2.2e-5 | 6.4e-5 | 2.0e-4 | 5.8e-4 | 1.5e-3 | 5.3e-3 | 0.93 |
| 2 levels, AQORN | 8.4e-6 | 2.5e-5 | 8.7e-5 | 2.8e-4 | 7.3e-4 | 2.3e-3 | 0.97 |
| 2 levels, index order | 4.3e-5 | 1.6e-4 | 4.3e-4 | 1.2e-3 | 2.7e-3 | 8.3e-3 | 0.89 |

Per-circuit geometric-mean factors at alpha 0.003-0.1:
- concatenation (one-level baseline / two-level baseline): 5.0-5.4x;
- AQORN at one level: 2.0-2.6x;
- AQORN at two levels: 2.0-2.2x;
- one-level baseline / two-level AQORN: 10-12x.

At the calibrated noise the same factors are 4.7x, 2.1x, 2.4x and 11.5x.

## Sentences the text can use

- **AQORN's gain holds one level up, and grows on gates and depth.** At two levels AQORN removes 42% of two-qubit gates and 50% of depth (one level: 21% / 17%), keeps 30% more shots, and lowers the logical error rate 2.4x (one level: 2.1x). Every circuit gets better LSR, and 25/26 get lower logical error.
- **Most of it comes from the inner level.** AQORN at the inner level alone gives two-qubit gates -34% and depth -43%; at the outer level alone, -12% and -11%. The outer code's encoding and syndrome extraction repeat on every outer patch, so with the spread layout the same gates sit on the two lanes of one inner patch, and merging fuses them into whole-patch and transversal macros.
- **The inner placement has to keep each outer patch spread.** The index layout uses the same qubits, with similar gate counts, but has 4.3x the logical error rate of the spread layout. That is 1.7x worse than even the two-level baseline: one inner patch holding two qubits of one outer patch drops the code distance from 4 to 2.
- **Figure A is concatenated error detection.** All checks are read once at the end. Correction, with mid-circuit inner checks, is in the next section.
  - Treating flagged inner patches as erasures, corrected offline by the outer code, keeps 85% of shots instead of 65%. It raises the logical error rate 75x (5.0e-2 vs 6.6e-4), because a flag at the end does not locate the fault in time.
- **Still first order.** The logical error rate grows linearly with p2 at both levels (slopes 0.88-0.97). The gadgets are not fault-tolerant, so the compiled circuit has distance 1. Concatenation lowers the coefficient about 5x, and AQORN lowers it about 2x more at either level.

## Error correction: mid-circuit inner checks and a correcting decoder (added 10-10)

Everything above is concatenated error **detection**: every check is read once, at the end, and
any failure discards the shot. Run 2 (`concat_correct.py`, commit 1e0071f, 23 min, 104/104 jobs)
adds what correction needs.

- **Mid-circuit syndrome rounds** (`TranslatorConfig.syndrome_rounds`; default 0, so existing
  translations are unchanged).
  - Each round checks both stabilizers of every inner patch (one level: of every patch), R times, at evenly spaced boundaries between program layers.
  - The round is fault-tolerant at n = 4 (`IcebergModule.syndrome_round`). An exhaustive search over the 840 CNOT interleavings found 72 in which every single fault either flips an outcome or leaves no logical error; the naive Z-then-X order has two hook faults that do neither.
  - Ancillas are reused without reset: noiseless, every check reads 0.
  - All two-level configurations got the requested rounds; four very shallow programs got fewer at one level.
- **A correcting decoder** (`lookup.py`).
  - Every single-fault mechanism (up to ~350k per circuit) is injected into a Pauli-frame simulation, giving the exact first-order map from syndrome (inner checks, mid-circuit and final, and outer checks) to the effect on the decoded output.
  - A shot is corrected when the single faults explaining its syndrome agree (confidence >= 0.99), kept unchanged when nothing fired, and discarded otherwise.
  - Validated: injection equals explicit single-fault circuits exactly, and the first-order detection rate equals the sampled one at small alpha.
  - Corrections are Pauli-frame XORs, so no feedback is needed: the programs are Clifford.

![correction](../../../../../results/figures/concat_correction.pdf)
`results/figures/concat_correction.pdf`: two levels, baseline and AQORN. Solid lines: detection
(Figure A's setup). Dashed lines: correction with 4 rounds. Shaded band: Figure A's calibrated
noise.

**AQORN at both levels, correction (4 rounds) against detection.** Per-circuit geometric-mean ratios:

| p2 (alpha) | shots discarded | logical error rate |
|---|---|---|
| 9e-6 (0.003) | x0.27 | x0.77 (14 circuits with errors in both) |
| 3e-5 (0.01) | x0.29 | x1.09 |
| 9e-5 (0.03) | x0.34 | x1.77 |
| 3e-4 (0.1) | x0.46 | x2.44 |
| ~9e-4 (calibrated) | x0.72 | x3.80 |

The baseline at both levels: discards x0.44 -> x0.79, logical error x1.28 -> x4.36.

What it shows:
- **Real correction at two levels.** At low noise the concatenated code corrects about 3/4 of the shots post-selection throws away, at the same logical error rate.
- **Mis-correction is second order.** It needs two faults, so the corrected logical error rate grows faster than linearly (pooled slope ~1.5 vs ~1). At the calibrated noise, where a kept shot often carries more than one fault, correction trades accuracy for shots. A higher-order decoder would move that crossover up.
- **Rounds are what make faults correctable.** The share of detected first-order fault probability that is uniquely correctable, for AQORN, is 60% / 69% / 74% / 79% at 0 / 1 / 2 / 4 rounds; for the baseline, 38% / 49% / 57% / 65%. Correction without rounds already beats the old erasure decoder by ~10x in logical error, and rounds cut it 2x more at the calibrated noise.
- **One level cannot correct** (distance 2). Only 6-14% of its detected faults are correctable, mostly readout errors on round ancillas; its LSR moves by about 1%.
- **AQORN's advantage holds under correction.** At the calibrated noise, against the baseline: logical error 0.31-0.34x (better on 26/26 circuits at every round count) and 24-28% more shots kept. Under detection the figures are 0.37-0.49x and +29-31%.
- **AQORN also makes faults more correctable** (79% vs 65% at 4 rounds): it removes the in-patch gadgets whose faults are ambiguous.
- The erasure decoder of the 10-07 report stays poor (logical error 5-7e-2 at the calibrated noise), as expected without the fault-to-syndrome map.

Sentence for the paper: *With mid-circuit checks on the inner patches and a decoder that corrects
the faults a single fault explains, the concatenated code corrects errors rather than only
detecting them. At low noise it keeps the logical error rate of post-selection while discarding
about a quarter as many shots. AQORN lowers the logical error rate about 3x in both modes.*

Files: `experiments/concatenation/{lookup,concat_correct}.py`, `tests/test_syndrome_rounds.py`,
data `results/concatenation/concat_correct.csv` (copied to the project's `results/data/`), log
`logs/concatenation/concat_correct.log`. Redraw:

    PYTHONPATH=. ../../../venv/bin/python -m experiments.concatenation.concat_figures.concat_figures \
        --correction-csv results/concatenation/concat_correct.csv --output-dir ../../../../results/figures

## Against the 2026-10-07 run

The older run had no logical optimisation, used Fig 6's recorded w10 / w10T plans, and had
transversal gadgets only in the "both, w10T" arm.

| | 10-07 (both w10T vs baseline) | now (both, standard) |
|---|---|---|
| 2 levels: 2q / depth / LSR / logical error | 43.4 / 48.7 / +32.0 / 58.6 (3.0x) | 41.8 / 49.9 / +30.0 / 53.2 (2.4x) |
| 1 level (w10T vs baseline) | 25.0 / 20.7 / +1.4 / 53.6 (2.4x) | 20.8 / 16.8 / +1.0 / 48.6 (2.1x) |
| index order vs spread, logical error | 3.7x (means) | 4.3x (geo-mean) |

As in Fig 6 under the pass, the gains shrink a little on optimised inputs; the shape is unchanged.

## Caveats

- Clifford proxies of the small group only (at most 8 program qubits; 36-72 physical qubits at two levels).
- TVD sits near its sampling floor (100k samples), so the logical error rate (200k frames) is the accuracy metric.
- Panel (b) pools over circuits, so the largest circuits dominate it. Panel (a) and the factors above are per circuit.
- LSR is left out of panel (a): at the shared noise one level keeps 92% of its shots, so its LSR has almost nothing to gain (+1%). LSR appears in C.
- The outer plans are made for one level (`plan_standard`); a planner aware of the inner level could do better.

## Files

- Code:
  - `experiments/concatenation/concat_standard.py` (the run);
  - `experiments/concatenation/concat_figures/concat_figures.py` (A, C, `--summary`);
  - `construct.py`, which gains a per-level `transversal` flag (default off: the old run is unchanged).
- Data: `results/concatenation/concat_standard.csv` (+ `.json`) in this worktree, copied to the project's `results/data/`.
- Figures: project `results/figures/concat_gain.{pdf,svg}`, `concat_tradeoff.{pdf,svg}`.
- Log: `logs/concatenation/concat_standard.log`.

Re-run and redraw:

    PYTHONPATH=. ../../../venv/bin/python -m experiments.concatenation.concat_standard --jobs 12    # ~13 min
    PYTHONPATH=. ../../../venv/bin/python -m experiments.concatenation.concat_figures.concat_figures \
        --from-csv results/concatenation/concat_standard.csv --output-dir ../../../../results/figures --summary
