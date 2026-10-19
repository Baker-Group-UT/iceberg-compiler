# How close AQORN gets to the best placement: its own objective, and what is measured

Run 2026-10-09 21:32-23:50, branch `v2-logical-opt-pass` (6e6b501 for the last parts),
`experiments/optimality/exhaustive_front.py`; figure
`experiments/figures/optimality_front/optimality_front.py` ->
`results/figures/optimality_front.pdf` (project root). Data: `results/optimality_front/` here,
copied to the project's `results/data/optimality_front_{space,plans,space_gaps}.csv`.

## Setup

- **75 programs**: the 15 small QASMBench circuits that build (9 / 4 / 2 at 4 / 5 / 6 qubits,
  Clifford proxies; `basis_trotter_n4` is the real 1888-gate file) and 20 random structured
  Clifford programs per width (Experiment7a's generator and seeds).
- **The current standard** (w10T, solo-HH pricing) **with the logical optimisation pass**: every
  program optimised once (Fig 6 small circuits: the study's verified choices), every plan,
  score and translation of the optimised program.
- **Every tightest [[4,2,2]] placement** (24 / 720 / 720), both H-commutation settings: 67,632
  placements, 66,048 distinct physical circuits.
- Per placement: AQORN's own objective (`plan_standard`'s choice criterion, recomputed: equal to
  the planner's recorded value for its own pick on **75/75**); 2q gates and depth under the
  standard translator; LSR, logical error rate (occupied lanes) and **exact TVD** from Pauli
  frames at alpha = 1 on Fig 6's machine, 500k shots, plus an independent 500k for choosing
  "best" (winner's curse).
- Exact TVD = P(the kept shot's logical flip lies outside the span of the ideal output's
  support): checked against the simulator's sampled TVD -- equal to 1e-4 on 1-4-outcome
  programs; sampled is higher by its sampling floor on 16-outcome ones (0.0036 on a uniform
  output whose true TVD is 0). TVD is 0 by construction on the 10 uniform-output programs,
  which the TVD row excludes.
- Arms: AQORN (`plan_standard`), naive (program order), interaction-count partition; the
  latter two at AQORN's commutation setting. AQORN fell back to naive on 3 programs (qft_n4,
  rand:4q:04, rand:4q:13).

## Results

Median gap to the best placement of the space, % (dominated = share of the space that beats the
pick on both LSR and LER, on both sets of shots):

| | AQORN cost | LSR | LER | TVD | 2q gates | depth | dominated |
|---|---|---|---|---|---|---|---|
| **AQORN** | **0.28** | 1.6 | **7.9** | **30** | 0.0 | 1.2 | **0.14** |
| naive | 11.5 | 3.1 | 32.9 | 71 | 11.9 | 8.7 | 6.1 |
| interaction partition | 13.3 | **0.18** | 52.6 | 97 | 0.0 | 6.8 | 1.1 |

By width (AQORN cost / LSR / LER / TVD):

| width | AQORN | naive | interaction |
|---|---|---|---|
| 4 | 0.04 / 0.5 / 3.3 / 6 | 7.2 / 0.8 / 12.7 / 50 | 7.2 / 0.1 / 32.3 / 57 |
| 5 | 0.40 / 1.7 / **22.3** / **88** | 11.7 / 3.8 / 39.3 / 144 | 14.3 / 0.2 / 77.8 / 212 |
| 6 | 0.58 / 3.0 / 7.3 / 20 | 11.6 / 5.4 / 28.5 / 45 | 14.3 / 0.2 / 50.7 / 96 |

- **Claim 1, search: AQORN finds the optimum of its own objective, near enough.** Median gap
  0.28%, mean 0.55%, worst 4.1%; exactly optimal on 34 / 8 / 5% of programs at 4 / 5 / 6
  qubits. Naive and interaction-partition placements are 11-14% off it.
- **Claim 2, trade-off: AQORN sits close to both measured peaks; the conventional packers sit at
  one end.** Its LSR is 1.6% below the best placement's and its logical error 7.9% above the
  best (median); a median 0.14% of the space beats it on both. The interaction partition is
  the best on LSR (0.18%) but 53% worse on logical error and 97% on TVD -- the survival end of
  the trade; naive packing is worse on both and beaten on both by 6% of the space.
- **Where AQORN is furthest from the measured best: 5 qubits** (one solo patch at the tightest
  packing): logical error 22% and TVD 88% above the best, while its own objective is within
  0.4% of optimal there. That is a model gap, not a search gap -- the cost model ranks those
  placements differently from the measurement -- and it is the place to look if this figure is
  pressed. At 4 and 6 qubits the logical-error gap is 3% and 7%.
- TVD gaps run larger than logical-error gaps because TVD only sees the flips that move the
  output off its support, so it is a smaller, noisier number (and is blind on uniform outputs).
- Across a program's placements the logical error rate varies 2.2x (median max/min), so these
  gaps are measured against a real spread.

## The pass defects this run found (fixed before the numbers above)

- **Cliffords written as rotations.** The optimiser emits Clifford gates as rz/rx at quarter
  turns (1,841 of 8,684 rotations in the study's Fig 6 circuits); the translator builds them as
  RZ gadgets and Stim cannot run them. The pass now names them (s, z, sx, x, ...).
- **Clifford programs gaining non-Clifford rotations.** O3's block resynthesis wrote 3 of the 60
  random Clifford programs with rz(pi/4)-type rotations (same unitary). The pass now keeps a
  Clifford program in Clifford gates; those 3 programs were re-planned and re-measured.
  Programs that already carry rotations may still trade 2q gates for more rotations (7 of the
  study's 56 Fig 6 circuits do, e.g. qugan_n71 519 -> 635) -- left as the optimiser's choice,
  an open question.
- **A measured copy getting a different answer** than the unmeasured program (the barrier
  `measure_all` adds); fixed, Fig 6's seeds re-validated 71/71.

## Notes for the paper (2026-10-10)

- **Which qubits are moved matters, though the circuit does not change.** Within one grouping
  (the same qubits share patches), the lane assignment and patch placement -- which atoms carry
  which logical qubit and which ones travel -- leave the compiled circuit's gate counts and
  depth unchanged in 85% of groupings (1,327 of 1,554), yet change its logical error rate by a
  median 7.7% (IQR 3.6-13.5%) and its TVD by 28% (12-56%), choosing best and worst on one set of
  shots and reading them on an independent set. LSR moves only 0.5%. Largest at 5 qubits (LER
  12%, TVD 39%). No static quantity we have predicts it: gate counts, depth, the schedule-based
  detected-error estimate (which does rank LSR within a grouping, Spearman -0.85 against a
  measurement ceiling of 0.80) and the silent-error model all correlate ~0 with it. The
  submitted section 4.3.1 says lane order "does not affect circuit quality"; that holds for the
  circuit, not for its logical error rate. Mechanism unknown (candidates: which atom ends up as
  the anchor after whole-patch HH exchanges; moves that carry both atoms of one logical
  operator). A lane pass chosen on measured LER would cut AQORN's LER by a median 3.2% (9.3% at
  5 qubits) at no LSR cost; one chosen on the schedule estimate gains only ~0.1% LSR, because
  `assign_lanes` already aligns lanes for the macros. Not adopted (compiler frozen 2026-10-10).
  At full size (`experiments/optimality/lane_refinement_screen.py`, all 71 Fig 6 circuits, a
  greedy pass over at most 16 patches scored by frames, results re-measured on fresh shots): LER
  -1.0% median / -4.5% mean below 10 qubits (toffoli_n3 -27%, wstate_n3 -25%,
  teleportation_n3 -18%), -0.5% / -1.0% at 10-30 qubits, ~0 above 30; LSR unchanged. Grid
  alignment changes the placement on only 4 of 71 circuits (it keeps the original order unless
  its own distance estimate improves) and moves nothing measurable.
- **Grid alignment (section 4.3.4) is described but never runs**: `packing.alignment.
  align_to_grid` exists and is tested, but no planner calls it, and the two sentences crediting
  packing's LSR / TVD gains to grid alignment are unsupported (the density work traced those to
  patch count).
- **The example program in (a) is typical, not chosen**: the 6-qubit program whose AQORN
  logical-error gap is the median of the 22 six-qubit programs (`rand:6q:17`).
- **What the clusters in (a) are**: one cluster per grouping (which qubits share a patch) and
  H-commutation setting -- 15 groupings x 2 settings = 30. The 48 points of a cluster are that
  grouping's lane assignments and patch orders: the same logical compilation, different physical
  arrangements. The grouping explains 99% of the LSR spread and 94% of the LER spread across all
  placements; lanes most of the rest; patch order about nothing at these sizes (2-3 patches).
- **AQORN sits at the accuracy end of the trade-off**: it gives up ~1.6% LSR for near-best LER,
  the intended effect of pricing silent errors; the interaction partition sits at the survival
  end (best LSR, +53% LER). Only 0.14% (median) of placements beat AQORN on both.
- **The search is near-optimal, the model is the limit**: 0.28% (median) from its objective's
  optimum, exactly optimal on 34 / 8 / 5% of programs at 4 / 5 / 6 qubits; the measured gap is
  model error, largest at odd widths.

## Caption draft

> **AQORN finds the placement its cost model ranks best, and that placement sits close to the
> best measured trade-off between kept shots and accuracy.** (a) Every tightest [[4,2,2]]
> placement of one 6-qubit program (grey), the placements no other beats on both LSR and
> logical error rate (line), the best placement on each alone (rings), and each packer's pick;
> the LSR axis is truncated. (b) Gap between each packer's placement and the best placement on
> each metric over 75 programs of 4-6 qubits (QASMBench proxies and random Clifford programs,
> logically optimised): dots are medians, whiskers the interquartile range. AQORN is within
> 0.3% of its own optimum and 1.6% / 7.9% of the best LSR / logical error rate; the
> interaction-count partition keeps the most shots but makes 53% more logical errors.

## Re-runs

    PYTHONPATH=. ../../../venv/bin/python -m experiments.optimality.exhaustive_front --jobs 12
    PYTHONPATH=. ../../../venv/bin/python -m experiments.figures.optimality_front.optimality_front \
        --from-csv results/optimality_front/space.csv

The first resumes by part file; `--redo PROGRAM ...` re-plans and re-measures given programs;
`--validate` re-runs the TVD check. Logs: `logs/optimality_front/`.
