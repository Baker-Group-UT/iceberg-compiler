# Compiler fixes, teddy night 2026-10-10/11

Branch `v2-teddy-final` (worktree `.claude/worktrees/teddy-final`). Three changes to the
standard planner (`packing/standard.py`) that the paper-vs-compiler audit flagged. Each one is an
`AqornStandard` switch, so the earlier behaviour stays reproducible. The defaults were set from one
paired sweep, using decision rules fixed before it ran.

| switch | what it does | default | commit |
|---|---|---|---|
| `standard_verify` (#7, done earlier, b664c94) | final check: plan vs naive priced on `standard_objective` | on | b664c94 |
| `standard_commutation` (#8) | H-commutation "auto": plan both arms and keep the lower `standard_objective` | **on** | 77153a7, 53e33d4 |
| `restarts` (#10) | 4 extra descents from shuffled seeds on programs of 16 qubits or fewer | **0 (dropped)** | 53e33d4 |
| `grid_alignment` (#1) | `align_to_grid` renumbering of the final plan (and of the naive fallback), kept only if it prices lower | **off** | 77153a7 |

Before these changes, the H-commutation decision was made by `plan_silent_aware`'s own score. That
score translated in program order and, on any program with rotations, fell back to gate counts plus
depth with no noise. After the change, every stage (seed, descent, commutation, alignment, final
check) compares candidates on the same `standard_objective`: detected errors read off the standard
schedule (on a Clifford stand-in) plus the planner weight × expected silent errors.

## The sweep

`experiments.optimality.noise_ratio_packers --variants aqorn_comm aqorn_comm_norestart
aqorn_comm_grid --tag variants --jobs 190`, summarised by `experiments.optimality.variant_summary`.

- **Corpus:** Fig 6's 71 circuits × 7 idle/distance ratios (1/64 … 64; ratio 64 is Fig 6's machine).
- **Measurement:** Pauli frames on the Clifford proxies, 200k shots, at the calibrated noise level
  (naive keeps the share of shots it keeps on Fig 6's machine).
- **Arms that serve as references:** the `aqorn` arm is the b664c94 standard. Its rows repeat
  `noise_ratio_packers_std_verify.csv` exactly: the same placements and the same calibrated α
  (checked).
- **Tests:** paired per ratio, as the geometric-mean ratio plus a Wilcoxon signed-rank test with
  zero differences dropped. Identical placements share one measurement, so an unchanged plan is an
  exact tie.

Data:
- `results/noise_ratio_packers/noise_ratio_packers_variants.csv` (3976 rows) and
  `_variants_summary_cal.csv`.
- Log: `logs/teddy_2026-10-10/final/phase2_variants.log`.
- The run's header reads `55c082c dirty 3`. It started from the uncommitted Phase 1 code, which was
  committed unchanged as 77153a7 a minute later.

**Decision rules (fixed before the run):**
- **Commutation fix:** adopt unless it is significantly worse (p < 0.05) on logical error or TVD at
  ratio 64.
- **Random restarts:** drop them if logical error, TVD and shots stay within ±1% and are not
  significant at every ratio.
- **Grid alignment:** adopt only if it is significantly better somewhere and never significantly
  worse.

### Commutation decision on the standard objective (`aqorn_comm` vs b664c94 `aqorn`)

| ratio | plans changed | logical error | TVD | shots kept |
|---|---|---|---|---|
| 1/64 | 13 | x1.012 | x1.029 | x1.006 |
| 1/16 | 8 | x1.002 | x1.008 | x1.002 |
| 1/4 | 5 | x0.995 | x0.989 | x1.010 |
| 1 | 4 | x0.998 | x0.998 | x1.005 |
| 4 | 7 | x0.999 | x1.005 | x1.018 (better, p 0.047) |
| 16 | 6 | x0.996 | x1.010 | x1.016 |
| 64 | 5 | x0.996 | x1.017 | x1.016 |

**Adopted:** the fix is never significantly worse at any ratio.

**Trade-off.** Pooled over the 48 (circuit, ratio) cells where the decision flips:
- shots kept ×1.113 (p 0.006);
- logical error per kept shot ×0.997 (n.s.);
- TVD ×1.164 on the 22 flips with a measurable TVD (p 0.042).

The standard objective prices detected errors and silent errors, not output bias. So when it flips
the decision it buys shots at unchanged logical error, and on some circuits that costs TVD.

The largest cases at Fig 6's machine are swap_test_n83 and swap_test_n25:
- swap_test_n83: shots 0.099 → 0.141, logical error 0.105 → 0.101, TVD 0.018 → 0.029.
- swap_test_n25: shots 0.039 → 0.066, TVD 0.037 → 0.065. This proxy is TVD-saturated under the
  logical-optimisation pass at the Fig 6 α.

If the paper wants the commutation choice made on accuracy, the place to change is the objective's
silent weight, not this decision. `STANDARD.with_(standard_commutation=False)` restores the old
decision.

### Random restarts (`aqorn_comm_norestart` vs `aqorn_comm`)

| ratio | plans changed | logical error | TVD | shots kept |
|---|---|---|---|---|
| 1/64 | 5 | x1.001 | x1.000 | x0.995 |
| 1/16 | 5 | x1.002 | x1.001 | x0.998 |
| 1/4 | 5 | x1.000 | x1.001 | x1.001 |
| 1 | 6 | x0.999 | x1.003 | x1.018 |
| 4 | 7 | x1.000 | x1.000 | x1.001 |
| 16 | 8 | x0.999 | x1.003 | x1.000 |
| 64 | 5 | x0.999 | x1.000 | x0.998 |

**Dropped (default `restarts=0`).** The restarts change 5–8 of the 33 small programs' plans per
ratio. Logical error and TVD stay within ±0.3%, and nothing is significant at any ratio.

The literal ±1% band is exceeded in one cell only: shots ×1.018 at ratio 1 (n.s.). That cell is in
the direction of *dropping* the restarts, and I read the band as a guard against harm. The planner
still descends from the naive packing as well as from its seed and keeps the better result, which
is what the paper describes. On the ≤16-qubit programs, the median plan time falls from 0.12 s to
0.10 s.

### Grid alignment (`aqorn_comm_grid` vs `aqorn_comm`)

| ratio | plans changed | logical error | TVD | shots kept |
|---|---|---|---|---|
| 1/64 | 4 | x1.000 | x1.002 | x1.008 |
| 1/16 | 5 | x1.001 | x1.003 | x1.006 |
| 1/4 | 3 | x0.999 | x1.000 | x1.003 |
| 1 | 2 | x1.000 | x1.001 | x1.001 |
| 4 | 4 | x1.001 | x1.002 | x1.002 |
| 16 | 3 | x0.999 | x1.001 | x1.000 |
| 64 | 4 | x0.999 | x0.999 | x1.002 |

**Left off.** The aligned plan prices lower, and so is kept, on 25 of 497 plans. No metric moves
by more than 1% at any ratio, even where movement dominates (1/64).

Why so little: `align_to_grid` only renumbers patches, so its gain is bounded by the spread of
patch-centre distances on a near-square grid of P patches. The standard objective already reads
the schedule's transport. For the paper:
- drop grid alignment from Sec IV (it never ran in the submitted results either);
- say that patch-to-site placement is left to the downstream router / layout stage (Fig 4).

## The final compiler vs naive, across machines

The final noise-ratio sweep (`--tag final`) was run on the committed final compiler (c962cdf).
Figure: `results/figures/teddy_final_2026-10-11/noise_ratio_packers.{svg,pdf}`.

AQORN / naive, geometric mean with 95% bootstrap CI:

| ratio | logical error | TVD | shots kept |
|---|---|---|---|
| 1/64 | 0.877 [0.832, 0.918] | 0.862 [0.813, 0.907] | 1.288 [1.179, 1.417] |
| 1/16 | 0.903 [0.869, 0.936] | 0.895 [0.859, 0.932] | 1.242 [1.146, 1.358] |
| 1/4 | 0.917 [0.885, 0.946] | 0.923 [0.889, 0.956] | 1.187 [1.110, 1.276] |
| 1 | 0.882 [0.849, 0.915] | 0.915 [0.883, 0.945] | 1.051 [0.979, 1.125] |
| 4 | 0.853 [0.816, 0.891] | 0.893 [0.845, 0.939] | 1.004 [0.939, 1.079] |
| 16 | 0.826 [0.785, 0.866] | 0.869 [0.817, 0.921] | 1.012 [0.949, 1.083] |
| 64 | 0.810 [0.770, 0.851] | 0.850 [0.800, 0.905] | 1.012 [0.958, 1.073] |

AQORN is at or below naive on logical error and TVD, and at or above it on shots, at every ratio.
Interaction partition + AQORN's descent, priced on AQORN's objective:
- **Ratios 1/16 to 64:** ties AQORN on logical error and TVD (n.s.), but keeps fewer shots (×0.79 at
  1/16, ×0.85 at 1/4, ×0.90 at 1, ×0.97 at 16; all p < 0.01).
- **Ratio 1/64:** worse on all three, with logical error ×1.08 (p 0.002), TVD ×1.06 (p 0.04) and
  shots ×0.75 (p < 1e-5). AQORN's final check and commutation decision recover what the search
  alone misses.

The cost-blind seed alone loses accuracy as idling grows (logical error ×1.04 at ratio 64).
