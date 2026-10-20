# Code size at an equal physical-qubit budget

v2 revision, teddy, 2026-10-11. Branch `v2-teddy-final`. Compiler frozen; only experiment and
plotting code is new. Handoff: `handoffs/equal_qubit_code_size.md`.

- **Code:** `experiments/revision/equal_qubit/` — `equal_qubit_code_size.py` (the run and the
  gate), `report.py` (the tables below) and `figure.py`.
- **Data:** `results/data/teddy_final_2026-10-11/equal_qubit_code_size.csv`, 2600 rows. The two
  gate runs are `equal_qubit_code_size_gate{,_rep2}.csv`.
- **Figures:** `results/figures/teddy_final_2026-10-11/equal_qubit_code_size.{svg,pdf}` (budget
  Q2), and `equal_qubit_code_size_Q2_anc.{svg,pdf}` (ancillas counted).
- **Logs:** `logs/teddy_2026-10-10/final/equal_qubit_*.log` and `equal_qubit_report.txt`.

## Bottom line

**k = 2's accuracy advantage survives an equal-hardware comparison.** Every k gets the same
budget Q2 = 4 ceil(n/2) data qubits, so the larger codes get spare patches. Even so, AQORN at
k = 2 has the lowest logical error per kept shot at every idle/distance ratio:
- against the runner-up (k = 4 every time): x0.67-0.86, beyond one SE at all five ratios, k = 2
  better on 9-13 of 13 circuits;
- TVD on the 9 unsaturated circuits: x0.66-0.84, beyond one SE, 8-9 of 9.

The ranking holds with more hardware (1.25 Q2 and 1.5 Q2, where the margins grow) and with the
two parity ancillas per patch counted (x0.69-0.92; not significant at 1/256 only).

What a larger code buys at equal hardware is shots, not accuracy, and only where movement
dominates. k = 4 keeps x1.18-1.32 more shots than k = 2 at ratios 1/256-1. It ties k = 2 at 16
and 256. With ancillas counted the k = 4 shot advantage shrinks to x1.05-1.14, and k = 2 keeps
the most shots at 16 and 256.

Why: spare patches do not buy accuracy. Within one code size, 1.5x the patches costs a lot of
shots (LSR x0.34-0.58 at k = 2, x0.44-0.53 at k = 4) for little or no gain in logical error
(x0.84-0.95 at k = 2; x0.93-1.14 at k = 4). So at equal Q a larger code runs below full
occupancy, but its two checks per patch still guard more logical qubits each.

So, per the handoff's "how the paper will use it": the paper can say plainly that the accuracy
advantage of k = 2 survives an equal-hardware comparison. Larger codes save qubits, or buy shots
where movement dominates, only when the device is smaller than 4 ceil(n/2) data qubits.

## Setup

- **Corpus, ratios, noise.** Fig 8's: 13 circuits of 25-40 qubits; ratios 1/256, 1/16, 1, 16,
  256. Fig 8's per-(circuit, ratio) alpha is read from its standard CSV, not recalibrated, and
  Fig 8's simulation settings are used: 40k raw shots, 100k Pauli frames.
- **Budgets.** Q2 = 4 ceil(n/2) data qubits (the paper's and Fig 8's convention), 1.25 Q2 and
  1.5 Q2.
  - Plus `Q2+anc`, the sensitivity check. The emitted circuits do carry 2 parity ancillas per
    patch: the emitted width is P (n_code + 2) on every row checked. So Q2+anc = 6 ceil(n/2),
    with a patch costing n_code + 2.
- **Patch count.** P = floor(Q / cost per patch), clipped to [ceil(n/k), n]. Every k fits every
  budget for this corpus.
- **Arms.**
  - AQORN: `plan_standard(program, n_code = k + 2, patches = P, noise = the ratio's planner
    profile)`.
  - naive: program order spread evenly over P patches (sizes differ by at most one, larger
    first) under Fig 8's naive commutation rule. At k = 2 this equals `tight_trivial_packing`
    for every n and P (checked for n = 4..40).
- **Measured.** As Fig 8, via `evaluate(frames=)`:
  - LSR and TVD;
  - silent logical error per kept shot on occupied lanes only (the primary accuracy metric);
  - the all-lane count;
  - static counts and the emitted width;
  - the noiseless check on every arm.
- **TVD.** Reported on the 9 circuits whose naive k = 2 TVD is below 0.95 at some ratio. Dropped:
  ising_n26, ising_n34, qugan_n39, swap_test_n25.
- **Run.** Points shared by budgets are measured once. 1080 (k, P) points x 2 arms, 65 columns
  on 65 workers (a third of the machine), 10.0 min wall. The pilot on the two heaviest circuits
  took 10.7 min.
- **Sanity.** 0 miscompiled; P x cost per patch <= Q on every row; no unmeasured row.

## The gate

At Q2, k = 2 is its own tightest packing, so its AQORN arm must be Fig 8's static k = 2 AQORN
rung (`aqorn_fixed`).

- **Exact, in 65 of 65 (circuit, ratio) columns:** the plan (1q / 2q gates, depth, patches,
  commutation, final-check fallback) and the alpha.
- **Sampled values cannot match to the bit, because Fig 8 ran unseeded.** `SimSettings.seed` is
  None by design, and frames used `frame_seed=sim.seed`. So the gate tests the sampled values
  for consistency instead: LSR z SD 1.00 / 1.03 (two runs, chi-square p 0.25 / 0.35); silent
  error per kept shot z SD 1.02 / 0.99 (p 0.42 / 0.49).
- **One run-level note.** Run 1's LSR z had mean -0.35 (2.8 SE). Run 2's was -0.11, and run 1 vs
  run 2 alone gave -0.24, the same size. That is sampling, not a harness offset.
- **Verdict:** passed.

## Budget Q2: the winning code size (AQORN arms)

Margin = geometric mean over circuits of best / runner-up. "1 SE" means the paired log ratio's
mean exceeds its standard error.

| ratio | LSR: best (runner-up) | margin | 1 SE | logical error: best (runner-up) | margin | 1 SE | wins |
|---|---|---|---|---|---|---|---|
| 1/256 | k = 4 (2) | x1.315 | yes | k = 2 (4) | x0.838 | yes | 9/13 |
| 1/16 | k = 4 (2) | x1.322 | yes | k = 2 (4) | x0.783 | yes | 11/13 |
| 1 | k = 4 (2) | x1.179 | yes | k = 2 (4) | x0.668 | yes | 13/13 |
| 16 | k = 2 (4) | x1.069 | no | k = 2 (4) | x0.827 | yes | 13/13 |
| 256 | k = 10 (8) | x1.016 | no | k = 2 (4) | x0.860 | yes | 13/13 |

TVD (9 circuits): k = 2 wins at every ratio, by x0.835 / 0.796 / 0.664 / 0.808 / 0.828, all
beyond one SE.

AQORN at each k relative to naive k = 2 at Q2 (geometric mean; these are the figure's lines):

| ratio | LSR k = 2 | 4 | 6 | 8 | 10 | logical error k = 2 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1/256 | 1.74 | 2.29 | 0.40 | 0.89 | 0.26 | 0.98 | 1.17 | 3.32 | 3.23 | 5.88 |
| 1/16 | 1.47 | 1.95 | 0.55 | 1.00 | 0.44 | 0.93 | 1.19 | 2.71 | 2.58 | 4.47 |
| 1 | 1.13 | 1.33 | 0.78 | 1.13 | 0.74 | 0.89 | 1.33 | 1.83 | 2.12 | 2.94 |
| 16 | 1.01 | 0.94 | 0.85 | 0.90 | 0.87 | 0.78 | 0.95 | 1.11 | 1.28 | 1.50 |
| 256 | 0.92 | 0.97 | 0.97 | 0.99 | 1.01 | 0.75 | 0.88 | 1.03 | 1.18 | 1.27 |

AQORN / naive at the same k and patch count (the across-ratio packing gain the cut noise-ratio
figure showed):
- **Logical error:** x0.65-0.78 at ratios 16 and 256 for every k. At ratios 1/256 to 1 it is
  x0.76-1.00 for k = 2..8, and x1.03-1.10 at k = 10.
- **LSR:** x1.13-2.27 at ratios 1/256 to 1, x0.83-1.01 at ratios 16 and 256.
- **Reading:** where movement dominates AQORN buys shots at roughly unchanged logical error;
  where idling dominates it buys accuracy at slightly fewer shots. This is the
  detected + 10 x silent objective at work.
- **Caveat for k = 2 at ratio 1/256:** AQORN keeps x1.74 the shots of naive here. Fig 8's pooled
  figure was x1.20 over all ratios, which is consistent.

## Does the ranking move with the budget?

No: k = 2 has the lowest logical error at every ratio and every budget.

| budget | logical error: k = 2 vs runner-up (k = 4), by ratio | 1 SE | TVD: k = 2 vs k = 4 | LSR winner by ratio |
|---|---|---|---|---|
| Q2 | x0.84 / 0.78 / 0.67 / 0.83 / 0.86 | all | x0.84 / 0.80 / 0.66 / 0.81 / 0.83 | 4, 4, 4, 2, 10 |
| 1.25 Q2 | x0.77 / 0.74 / 0.65 / 0.74 / 0.76 | all | x0.62 / 0.71 / 0.66 / 0.77 / 0.74 | 4, 4, 4, 2, 10 |
| 1.5 Q2 | x0.62 / 0.73 / 0.62 / 0.75 / 0.75 | all | x0.55 / 0.70 / 0.64 / 0.70 / 0.68 | 4, 4, 4, 2, 8 |
| Q2+anc | x0.92 / 0.79 / 0.69 / 0.81 / 0.83 | 4 of 5 (not 1/256) | x0.96 / 0.81 / 0.69 / 0.75 / 0.78 | 4, 4, 4, 2, 2 |

Ratios are 1/256, 1/16, 1, 16, 256. Where the LSR winner is k = 4 (ratios <= 1) its margin is
beyond one SE, except at Q2+anc at ratio 1 (x1.05). At ratios 16 and 256 the LSR differences between codes are within a few percent
(not significant at Q2).

## Ancilla-counted sensitivity

Counting the 2 parity ancillas per patch makes a patch cost n_code + 2. That is relatively
cheaper for large codes, so k > 2 gets more patches (n = 28: k = 4 gets 10 patches instead of
9).
- **Logical error:** the result is unchanged. k = 2 is lowest everywhere, beyond one SE except
  at 1/256 (x0.92, 8 / 13).
- **Shots:** k = 4's advantage at ratios <= 1 falls from x1.18-1.32 to x1.05-1.14, and k = 2
  becomes the shots winner at 16 and 256 (x1.20, x1.08).
- **Larger codes:** with their extra patches, k = 6-10 lose shots at every ratio (LSR x0.14-0.78
  vs naive k = 2).

## Caveats

- **Two naive packings.** The naive arm is spread evenly. When AQORN's final check falls back,
  it falls back to its own naive at P patches, which is program order with full patches first.
  Fallback happens on 12-32% of AQORN rows by (budget, k), and 29% at k = 2 at Q2 (the same
  rows as Fig 8).
- **Clipping.** k = 2 at Q2 (and at Q2+anc) is at its tightest packing; every other k at every
  budget runs with spare patches (mean occupancy 0.64-0.77 at Q2). So the Q2 comparison is
  "k = 2 full vs k > 2 partly empty" by construction. The 1.25 / 1.5 Q2 rows give k = 2 spare
  patches too, and the ranking does not change.
- **Logical error is silent error per kept shot on occupied lanes** (Pauli frames, 100k
  frames). The all-lane count is in the CSV (`logical_error_all_lanes`). It is a median 3-5%
  higher, and single rows differ more because the two counts come from independent frame
  samples. Re-ranked on it, the winner is k = 2 at every ratio and budget, with margins
  x0.65-0.90.

## For the paper

The figure (k vs idle/distance ratio, AQORN at every k on budget Q2, naive k = 2 dashed) can
replace both the code-size phase diagram and the cut noise-ratio figure. Suggested sentence:

> At an equal budget of physical qubits (4 ceil(n/2) data qubits, k = 2's footprint), the
> smallest code remains the most accurate at every idle/distance balance: AQORN at k = 2 has
> 14-33% lower logical error per kept shot than the best larger code (k = 4), on 9-13 of 13
> circuits. The larger code keeps 18-32% more shots only where movement dominates. Spreading a
> code over spare patches costs shots without buying accuracy, so a larger code saves qubits
> only on a device too small for k = 2.

Caption must state: geometric mean over 13 circuits; 95% bootstrap intervals over circuits;
logical error on occupied lanes; budget Q2 counts data qubits; k = 2 at Q2 is its tightest
packing.
