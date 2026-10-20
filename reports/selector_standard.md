# The resource selector under the standard, 2026-10-07

Questions: how good is the resource selector, can it be improved, and had it changed this round?

**Had it changed?** `packing/resource/select.py` is untouched since the submission. What changed
was `compile_circuit`'s default (commit 4aff4c7: the standard planner in its middle stage), and
that change made it worse, below.

## Quality before this change

- Under v4 it is near-optimal for shots: `compile_circuit` keeps 0.985x the shots of AQORN at the
  best static k chosen with hindsight (Experiment6d), and within 0.3% of the best of the arms tried
  here. Fig 8's selector arm alone (its own greedy placement) keeps 0.825x; `compile_circuit`
  fixes that by placing with the planner.
- Its analytic error model is crude: on Fig 8's corpus the mean error in -ln(LSR) is 2.09 against
  0.29 for the schedule-based model the planner already uses (`measured_noise_split`), with
  per-circuit rank correlations down to -1.0 (0.65). Under v4 this rarely flipped the choice.
- **Under the standard (4aff4c7) it was worse than v4.** Its three stages optimised different
  things: the shape and the final pick on the analytic estimate of detected errors only, the
  placement on the silent-aware objective. The final pick discarded the w10T plan on 11 of 13
  Fig 8 circuits. Frame-simulated against v4's `compile_circuit`: 0.86x the shots for 0.97x the
  silent error per kept shot.

## The change

`compile_circuit(planner="standard")` now optimises one objective in every stage
(`packing/compiler.py::_compile_standard`, `packing/standard.py::standard_objective`):

- every code size is planned with the standard planner, at its own (tightest) patch count;
- the planned placement and the naive one are translated under the standard settings and priced
  on the standard objective: detected errors read off the schedule (on a Clifford stand-in, so
  rotations price too: `simulator/attribution.py::clifford_stand_in`) + 10 x silent errors;
- the lowest wins. The shape is judged on the placement that will be built, not a greedy seed.

`planner="v4"` is the submitted compiler, unchanged. Tests: tests/test_standard.py,
tests/test_packing_compiler.py (v4-contract tests pinned to `planner="v4"`); 712 pass.

## The planner's weight above k = 2

The weight 10 was tuned at k = 2, where the transversal in-patch CZ makes avoiding silent errors
nearly free. `experiments/tvd_explore/weight_sweep_k4.py`: 31 Fig 6 circuits (9-56 qubits) at
[[6,4,2]], weights 0-10, every arm layer-sorted, alpha calibrated on naive k = 4 to LSR 0.06,
frames (100k shots). Paired geometric means against weight 0 (95% bootstrap CI):

| weight | Fig 6 machine: shots | silent / kept | balanced machine (Fig 8 ratio 1): shots | silent / kept |
|---|---|---|---|---|
| 0.5 | 0.975 [0.93, 1.02] | 0.948 [0.90, 0.98] | 0.980 | 0.990 |
| 1 | 0.947 [0.89, 1.00] | 0.888 [0.84, 0.94] | 0.962 | 0.981 |
| 2 | 0.881 | 0.820 | 0.868 | 0.992 |
| 3 | 0.841 | 0.790 | 0.877 | 0.984 |
| 5 | 0.808 [0.74, 0.89] | 0.767 [0.72, 0.82] | 0.851 | 0.968 |
| 10 | 0.806 [0.73, 0.89] | 0.758 [0.71, 0.80] | 0.619 [0.52, 0.73] | 0.946 |

Scored per kept shot (-ln LSR + 10 x silent per kept): on Fig 6's machine every weight >= 3
improves by about 0.35 (plateau from 3-5); on the balanced machine weights <= 1 are neutral and
>= 2 hurt (10: +0.38). Scored per shot (-ln LSR + 10 x silent per shot) no weight beats 0 at k = 4:
there the shot loss outweighs silent errors that are mostly discarded with the shot anyway.

## End to end on Fig 8's corpus

`experiments/revision/exp6_selector/standard_selector.py`: 13 circuits x noise ratios 1/16, 1, 16
at Fig 8's alpha, frames (100k shots); every arm against v4's `compile_circuit` (geometric means):

| arm | shots | silent / kept | per-kept objective | k chosen (of 39) |
|---|---|---|---|---|
| standard, weight 5 above k = 2 | **0.82** | **0.75** | **-0.20** | k=2: 12, 4: 21, 6: 4, 8: 1, 10: 1 |
| standard, weight 1 | 0.77 | 0.73 | -0.17 | k=2: 19, 4: 18 |
| standard, weight 10 | 0.78 | 0.76 | -0.13 | k=2: 12, 4: 20 |
| w10T at k = 2 always | 0.56 | 0.62 | -0.05 | k=2: 39 |
| naive k = 2 | 0.46 | 0.71 | +0.29 | |

By noise ratio, standard with weight 5: movement-dominated (1/16) 1.02x shots, 0.90x silent per
kept -- better than v4 on both; balanced (1) 0.66x / 0.74x (neutral per kept shot); idle-dominated
(16) 0.81x / 0.63x.

**Decision: weight 5 above k = 2** (`AqornStandard.wide_silent_weight`), the same rule as at k = 2
(the first weight on the accuracy plateau) and the best end to end. The standard is now an accuracy
trade at k > 2 -- about 0.8x the shots for 0.75x the silent error per kept shot on this corpus --
where at k = 2 it is close to free.

Caveats: frames (silent error per kept shot is the TVD stand-in, TVD ~ 0.95 x it, rank corr
0.91-0.98), 13 circuits x 3 ratios, Fig 8's recorded alpha; the first-order silent model the
objective uses overstates per-kept differences at high noise (higher-order undetected errors are
not in it), which is why the per-shot score disagrees.

Re-run:

    PYTHONPATH=. venv/bin/python -m experiments.tvd_explore.weight_sweep_k4 --jobs 12
    PYTHONPATH=. venv/bin/python -m experiments.revision.exp6_selector.standard_selector --jobs 12
    ... --from-csv results/tvd_explore/weight_sweep_k4.csv | results/revision/exp6/standard_selector.csv

## The ranking weight (added 2026-10-08)

`AqornStandard.selection_weight` (default None = `silent_weight` = 10) prices a silent error when
`compile_circuit` *ranks* its candidates, separately from the planners' search weights (10 at
k = 2, 5 above), which it leaves unchanged. `STANDARD.with_(selection_weight=w)`; tests in
tests/test_standard.py (weight 0 keeps the fewest detected errors, a huge weight the fewest silent
errors, the same candidates either way).

Tested on Fig 8 under the standard with no new simulation
(`experiments/revision/exp6_selector/selection_weight.py`): every one of the 650 candidates the
selector considered (10 per column, 13 circuits x 5 noise ratios) carries its model components and
its measured LSR and silent error per kept shot, so each weight's pick can be read off. Geometric
means against naive [[4,2,2]] (figure: `results/figures/selection_weight_v3.pdf`):

| ranking weight | shots | silent / kept | picks k = 2 | pick is measured best (shots / accuracy, of 65) |
|---|---|---|---|---|
| 0 | 2.05 | 1.15 | 6 | 40 / 11 |
| 5 | 1.92 | 0.99 | 8 | 32 / 13 |
| **10 (default)** | **1.78** | **0.95** | 14 | 28 / 16 |
| 20 | 1.68 | 0.91 | 20 | 22 / 22 |
| 50 | 1.53 | 0.88 | 27 | 15 / 28 |
| 100 | 1.31 | 0.87 | 34 | 8 / 29 |
| 1000 | 1.12 | 0.90 | 34 | 2 / 25 |
| infinite (silent only) | 1.08 | 0.91 | 35 | 2 / 27 |

For reference: the measured best trade among the same candidates runs from 2.10x / 1.35x through
1.87x / 0.91x and 1.37x / 0.78x to 1.22x / 0.77x; the standard planner at k = 2 alone is
1.15x / 0.78x.

- **The knob works:** the weight moves the selector along a shots / accuracy trade, monotonically
  up to about 100.
- **The model limits it on the accuracy side.** The model-ranked curve sits inside the measured
  frontier, and above w ~ 100 accuracy gets *worse* (0.87x -> 0.91x) while shots keep falling:
  the first-order silent-error estimate ranks candidates less reliably than the detected-error
  one (its pick is the measured most accurate in at most 29 / 65 columns). No weight reaches the
  accuracy of simply planning at k = 2 (0.78x).
- **Choice:** 10 (default) for shots; 20-50 is the balanced range (e.g. 50: 1.53x shots, 0.88x
  silent error per kept shot). Closing the gap to the frontier needs a better accuracy estimate in
  the ranking -- e.g. frame-simulating the few best candidates -- not a different weight. The
  default is left at 10 pending your choice.
