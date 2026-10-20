# Optimality of AQORN's packing

v2 revision, overnight 2026-10-07. Branch `v2-optimality` (from `e5d5741`); code in
`experiments/optimality/`. Planners as you chose: **weight 10 (w10)** and **w10T** (weight 10 +
transversal in-patch CZ/XCX), v4 as reference; every arm and every enumerated placement translated
with every program layer-sorted first (`asap_order`, the fair baseline). The logical
pre-optimization study is on its own branch, `v2-logical-opt` (report `reports/logical_opt.md`
there).

**Bottom line.**
- **Small programs (4-6 qubits), against exhaustive search:** the weight-10 planner is 0.6 / 2.0 /
  2.2% above the optimum on the objective it optimises (w10T 1.0 / 1.2 / 1.8%), against 17-23% for
  naive packing and 43-52% for an interaction-count partitioner. Measured by simulation (Pauli frames, every
  placement, winner's-curse guarded), its silent error per kept shot -- the TVD stand-in -- is
  4-19% (median) above the best placement's, against ~2x for v4 and naive packing; it pays 3-8%
  of LSR for that.
- **Constructed circuits with a provably optimal packing:** AQORN finds it -- 100% of the planted
  pairs, exactly on the gate-wise lower bound -- at every size from 6 to 32 qubits, all three
  planners; the interaction-count partitioner finds none of the pairs (122-156% above the bound).
  This is the key experiment: packing for merging needs qubits that act *alike*,
  which an interaction graph cannot see.
- **Logical pre-optimization** (branch `v2-logical-opt`): the inputs are near-optimal in
  two-qubit structure on most circuits (2q unchanged by a strong optimizer on 41 of 69) but not
  all (-31 to -84% on 13); on optimized inputs AQORN's depth gain is unchanged (15.3%) while its
  2q, TVD and LSR gains shrink (15.5 -> 9.6%, 38.2 -> 29.7%, +25.9 -> +9.7%), mostly the
  Bernstein-Vazirani family; optimizer + AQORN compound (2q -22.8%, TVD -48.2%).
- **Caveat to keep in the paper:** near-optimality holds on the planner's objective, not on depth
  (1-15% depth regret), and one cost-model tie at n = 4 is never broken by the verify step.

## 1. Small programs against exhaustive search (4-6 qubits)

`exhaustive_silent.py`. Same programs as Experiment7a: the 4-6 qubit QASMBench originals that
build (15: 9 / 4 / 2 at 4 / 5 / 6 qubits) plus 20 random structured programs per size (7a's generator and seeds). The space is the
**complete tightest packing** (every partition x patch order x lane map: 24 / 720 / 720
placements -- the translator is not invariant under the relabellings the old canonical space
removed) **under both H-commutation settings**, since the planner chooses commutation too. Every
arm is scored on the weight-10 planner's own objective (expected detectable error, or weighted
gates + depth for programs with rotations, + 10 x model silent error), on the translator it runs
on; w10T's arm against w10T's own space. Checks: no arm beats the enumerated optimum (0 of 375);
the recomputed objective equals the planner's own recorded value on 150/150 plans.

Mean regret above the optimum (%), +-1 SD over circuits; P(opt) = share of circuits where the
arm is exactly optimal; "space better" = share of the space strictly better than the arm:

| program | arm | regret, w10 objective | P(opt) | space better | depth regret (fair) | 2q regret (fair) |
|---|---|---|---|---|---|---|
| 4 qubits (29) | **w10** | **0.59 +- 0.78** | 34% | 4.2% | 1.3 | 3.3 |
| | w10T | 0.99 +- 2.09 | 38% | 5.6% | 1.7 | 12.5 |
| | v4 | 28.2 +- 23.3 | 7% | 42.8% | 6.6 | 2.0 |
| | naive | 23.0 +- 21.8 | 7% | 38.2% | 9.1 | 6.8 |
| | interaction partition | 42.7 +- 23.6 | 3% | 54.3% | 10.3 | 4.4 |
| 5 qubits (24) | **w10** | **1.96 +- 2.27** | 12% | 4.4% | 12.4 | 5.8 |
| | w10T | 1.17 +- 1.76 | 12% | 2.5% | 8.4 | 15.6 |
| | v4 | 37.7 +- 20.2 | 4% | 55.8% | 17.3 | 2.5 |
| | naive | 17.0 +- 13.0 | 0% | 33.4% | 15.9 | 8.8 |
| | interaction partition | 43.7 +- 15.4 | 0% | 62.7% | 18.7 | 1.8 |
| 6 qubits (22) | **w10** | **2.20 +- 2.68** | 0% | 5.9% | 15.1 | 10.4 |
| | w10T | 1.75 +- 2.07 | 5% | 4.8% | 9.1 | 17.0 |
| | v4 | 25.2 +- 19.2 | 0% | 37.8% | 12.6 | 4.2 |
| | naive | 17.9 +- 10.3 | 0% | 33.2% | 20.2 | 9.7 |
| | interaction partition | 52.3 +- 17.0 | 0% | 70.1% | 14.6 | 5.8 |

- **The weight-10 planner is within 0.6-2.2% of the exhaustive optimum on the objective it
  optimises** (median 0.2-1.1%); w10T within 1.0-1.8% on its own. Naive packing is 17-23% away,
  the interaction-count partition 43-52%.
- v4 scores badly here only because this objective prices silent error, which v4 ignores; on its
  own objective (`measured`, expected detectable error) v4's regret is 1.5 / 1.8 / 3.8% (the 7a
  result, now on equal schedules and with commutation in the space).
- Model silent error of the w10 plans: 10.10 / 7.37 / 14.99 x 1e-3 against the space's minimum
  10.10 / 7.11 / 14.56 x 1e-3 -- essentially optimal on the quantity it is meant to minimise.
- **Depth is not optimised** and the w10 plans pay for silent error in gates and depth (5-15%
  depth regret at 5-6 qubits). Claim near-optimality on the planner's objective, not on depth.

## 2. The same small programs, measured (Pauli frames)

`exhaustive_frames.py`. What the paper claims is accuracy and kept shots, so every placement of the
complete tightest space (H commutation fixed to the weight-10 planner's choice) was measured on
Fig 6's machine at alpha = 1, every program layer-sorted, by Pauli frames (the noisy circuit the
simulator runs): LSR and silent error per kept shot (the TVD stand-in, TVD ~ 0.95 x silent per
kept). Against the winner's curse: every placement at 20k shots, then the 10 best by silent error,
the 10 best by LSR and every arm re-measured at 200k shots with fresh seeds; "best" is decided on
the fresh numbers (21 finalists per circuit on average). 27 programs: the QASMBench proxies that
build plus 4 random per size (13 / 8 / 6 at 4 / 5 / 6 qubits; fewer than section 1 because every
placement costs a Stim conversion, ~1 s).

Silent error per kept shot (x 1e-3, mean) against the best placement's, and the median regret;
LSR regret against the best-LSR placement:

| program | arm | silent (best) | silent regret, median | LSR regret, median |
|---|---|---|---|---|
| 4 qubits | **w10** | 13.1 (12.6) | **3.6%** | 3.3% |
| | w10T | 13.6 (12.6) | 3.9% | 1.0% |
| | v4 | 24.6 (12.6) | 92% | 0.2% |
| | naive | 24.5 (12.6) | 95% | 0.3% |
| | interaction partition | 24.7 (12.6) | 92% | 0.2% |
| 5 qubits | **w10** | 16.5 (14.3) | **19%** | 5.4% |
| | w10T | 14.9 (14.3) | 10% | 1.8% |
| | v4 | 24.1 (14.3) | 79% | 2.3% |
| | naive | 20.4 (14.3) | 69% | 5.8% |
| | interaction partition | 32.2 (14.3) | 128% | 0.2% |
| 6 qubits | **w10** | 31.0 (28.3) | **8.9%** | 8.4% |
| | w10T | 32.5 (28.3) | 19% | 1.5% |
| | v4 | 43.6 (28.3) | 82% | 1.5% |
| | naive | 39.0 (28.3) | 66% | 7.1% |
| | interaction partition | 54.3 (28.3) | 244% | 0.3% |

- **Measured, the weight-10 planner's silent error is within 4-19% (median) of the best placement
  of the whole space**, against roughly double for v4, naive packing and the interaction-count
  partition; it is below v4's on 22 of 27 programs. That is the quantity the paper's TVD claims
  rest on.
- **Survival and accuracy are different optima.** The placement that keeps the most shots is
  not the one with the least silent error: v4 and the interaction-count partition sit within
  0.2-2% of the best LSR and 80-240% above the best silent error; w10 gives up 3-8% LSR for its
  silent error. (Experiment7b saw the same split for v4.) w10T gets close on both (LSR regret
  1-2%, silent 4-19%) -- measured here against the best placement under the *shipped*
  translator, which w10T's gadget can beat.
- Absolute TVD is not reported here: at alpha = 1 the silent rates are 1-3%, and the frame
  estimate is the sharper statistic (the full-simulation version of this check, Experiment7b,
  exists for v4 only).


## 3. Planted packings: optimal by construction, at any size

`planted.py`. Hidden pairs (a_j, b_j); each layer applies, to disjoint pairs, **HH(j)** (H on both
members) or **CXT(i -> j)** (CX a_i -> a_j and b_i -> b_j); labels are then scrambled. Under the
planted packing every H is half a whole-patch HH and every CX half a lane-aligned transversal CXT.
From the cost table (`iceberg.costs.work_table(4)`, weights 1 : 13): an H costs 59 units alone and
2 as half an HH (plus the anchor swap's 2 carriages, movement not gates); a CX costs 52 alone (in-
or cross-patch) and 26 as half a CXT. Merging can only replace gates by macros, so under any packing
each H costs >= 2 and each CX >= 26: the sum is a **lower bound on the logical-gate work of every
placement, and the planted packing attains it** (checked by translation on every decoy-free circuit).
The macros leak no silent error, so it is also optimal on the silent model. CZ is excluded on
purpose: w10T's transversal in-patch CZ (6 units) undercuts CZT's 26 and would break the bound.
Commutation is off for every arm, so this isolates packing; patch *order* is outside the bound.

Your construction needed two fixes to be a test: (1) scrambling the labels, or naive program-order
packing finds the pairs for free; (2) avoiding repeats that cancel at the logical level (two
consecutive identical CXTs) -- a bug in my first generator made some circuits collapse, so the run
reported here is the corrected `planted_v2` (100% of gates survive cancellation). I also added
decoys (rate rho per layer: a lone H, or a CX between members of different pairs on crossed lanes),
with which the planted packing becomes a reference rather than a certified optimum.

126 circuits: n = 4, 6, 8, 12, 16, 24, 32 program qubits, 2n layers, decoy rate 0 / 0.1 / 0.3,
six seeds each. The planted packing attains the bound on **42/42 decoy-free circuits** (certified
optimal there).

Planted pairs recovered (mean over seeds) and logical-gate work above the bound:

| n | w10 (= w10T) | v4 | naive | interaction partition |
|---|---|---|---|---|
| | **pairs recovered, no decoys** | | | |
| 4 | 83% | 83% | 33% | 0% |
| 6-32 | **100%** at every size | 100% | 1-22% | 0-22% |
| | **gate work above the optimum, no decoys** | | | |
| 4 | 9.6% | 9.6% | 70% | 131% |
| 6-32 | **0.0%** at every size | 0.0% | 128-146% | 122-156% |
| | **pairs recovered, decoys 0.3** | | | |
| 6-32 | 92-100% | 100% | 4-33% | 0-33% |

- **AQORN finds the provably optimal packing at every size from 6 to 32 qubits**, all three
  planners, landing exactly on the gate-wise bound (weighted gate work above it: 0.0%). Its
  weight-10 score is within -0.7% to +0.6% of the planted packing's (it can be slightly better:
  the score also prices transport, and the bound says nothing about patch order). Planning takes
  1.4 s at 32 qubits.
- **The conventional interaction-count partition finds none of the pairs** and pays 122-156% more
  gate work than the optimum, naive packing 128-146%. This is the structural point of the paper in
  one experiment: packing for merging must group qubits that do the *same* operation at the same
  time, while an interaction graph groups qubits that interact -- here exactly the wrong pairs.
- With decoys (rho = 0.1, 0.3) the planted packing is no longer certified optimal; AQORN still
  recovers 92-100% of the pairs at n >= 6 and its score is within +-3% of the planted packing's.
- **One miss, explained**: at n = 4 one of six seeds (the same circuit at every decoy rate) is
  compiled with the naive pairing (+58% gates, +15% score). AQORN's cost model gives the naive and
  the planted partition *the same* value (812.0): it does not see that the planted pairing turns
  two opposite-direction cross-patch CXs into one CXT. The verify step only compares the result
  against naive -- here the same placement -- so the tie is never broken by translation. A cheap
  fix (break model ties among the top partitions by translating them) is possible; not applied.
- Exhaustive check at n <= 6 (complete space): the planted packing is optimal on weighted gates +
  depth (regret 0.00%); on the full weight-10 score it is 0.27% above the best placement (patch
  order), and AQORN 1.33%.

![optimality](../../../../../results/figures/optimality_gaps_v2.pdf)
`results/figures/optimality_gaps_v2.pdf`: (a) section 1, (b) recovery vs size, (c) gate work
above the bound.

## Files and re-runs

From the worktree (`iceberg-compiler/.claude/worktrees/optimality`):

    PYTHONPATH=. ../../../venv/bin/python -m experiments.optimality.exhaustive_silent --jobs 3
    PYTHONPATH=. ../../../venv/bin/python -m experiments.optimality.planted --jobs 2 --tag v2
    PYTHONPATH=. ../../../venv/bin/python -m experiments.optimality.exhaustive_frames --jobs 2
    PYTHONPATH=. ../../../venv/bin/python -m experiments.optimality.optimality_gaps.optimality_gaps \
        --exhaustive results/optimality/exhaustive_silent.csv --planted results/optimality/planted_v2.csv

Data (not committed): `results/optimality/{exhaustive_silent,exhaustive_silent_space,planted_v2,
exhaustive_frames}.csv`. Figure: `results/figures/optimality_gaps.pdf` (project root).
