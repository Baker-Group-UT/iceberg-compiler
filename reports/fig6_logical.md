# Fig 6 under the standard with the logical optimisation pass

Run 2026-10-09 23:47 - 10-10 00:21 from 6e6b501 (clean), branch `v2-logical-opt-pass`:
`experiments/figures/ablation_standard/ablation_standard.py --jobs 12 --persist`, pass on
(`verified-routes-2`). Figure `results/figures/ablation_standard_logical.pdf` (project root),
CSV `results/data/ablation_standard_logical.csv` (+ the copy beside the script).

## What changed from the earlier standard run (reports/fig6_standard.md)

- **Every circuit is its logically optimised program**, all four arms -- the baseline included
  -- so the baseline compiler gets the same optimised input. The 71 programs are the
  logical-optimisation study's verified circuits with Clifford-angle rotations named
  (`experiments/logical_opt/seed_records.py`); proxies are made from them (rz/rx -> S).
- **Every circuit planned fresh** with `plan_standard` on the optimised program (71/71
  "planned"; the recorded w10T plans were made on the unoptimised programs). The standard now
  includes solo-HH pricing (`solo_hh_aware`, 2026-10-09), so **two things changed at once**
  against the earlier run: the inputs and, for odd widths, the planner.
- Alpha recalibrated per circuit on the (optimised) baseline with Fig 6's pilots and seeds: mean
  alpha 4.63, baseline LSR 0.073 (earlier 0.073), changed on 55/71 circuits (median x1.11).
- Gate: the submitted configuration, pass off, still reproduces Fig 6 exactly (adder_n10,
  bell_n4: calibration, baseline, all five panels).

## Caption numbers (the `values.tex` convention, cumulative after each stage)

| | + H commutation | + merging | + packing (= AQORN) | earlier standard | submitted v4 |
|---|---|---|---|---|---|
| 1q gates | 8.1 | 6.8 | **8.2** | 13.3 | 22.5 |
| 2q gates | 4.9 | 11.7 | **13.2** | 19.3 | 16.8 |
| depth | 6.8 | 13.9 | **17.3** | 16.4 | 28.9 |
| LSR | +8.9 | +23.5 | **+22.3** | +43.2 | +38.3 |
| TVD (59 circuits) | 11.6 | 20.3 | **32.0** | 40.7 (66) | 26.8 |

By size (AQORN): small (29) 1q 5.8, 2q 18.7, depth 22.6, LSR +27.5, TVD 28.1; medium (15) 10.8 /
10.9 / 19.6 / +20.3 / 35.1; large (27) 9.3 / 8.6 / 10.3 / +17.7 / 36.2.

Paired against the earlier standard run (same circuits, per-circuit difference, bootstrap 95%):

| | earlier -> now | paired difference |
|---|---|---|
| 1q | 13.3 -> 8.2 | -5.1 [-10.9, +0.2] |
| 2q | 19.3 -> 13.2 | -6.1 [-9.4, -3.2] |
| depth | 16.4 -> 17.3 | +0.9 [-2.2, +3.9] |
| LSR | +43.2 -> +22.3 | -20.9 [-29.9, -12.2] |
| TVD | 40.7 -> 32.0 | -8.6 [-15.3, -2.4] |

- **On optimised inputs AQORN's gains shrink but hold**: 13% fewer 2q gates, 17% less depth,
  22% more kept shots and 32% lower TVD than the same compiler without AQORN's passes on the
  same optimised program. Depth is the one gain that does not shrink. The size of the change
  matches what the logical-optimisation study measured for w10 (2q 15.5 -> 9.6, TVD 38.2 ->
  29.7, LSR +25.9 -> +9.7), which suggests the inputs rather than the solo-HH planner change
  (it only moves odd-width plans) account for most of it; no run separates the two.
- **With packing, LSR falls slightly (+23.5 -> +22.3) while TVD rises (20.3 -> 32.0)**: the
  placement trades shots for accuracy, as the standard's silent-error weight intends.

## Caveat: seven proxies lose a measurable TVD

dnn_n33, dnn_n51, qugan_n39, qugan_n71, swap_test_n25, swap_test_n41, swap_test_n83 had a
resolvable baseline TVD before (0.10-0.17) and now saturate (0.995-1.000), so Fig 6 withholds
their TVD (59 circuits instead of 66; the stars). The Clifford proxy maps every non-Clifford
rotation to S; for these optimised programs that proxy's output spreads over (almost) every
bitstring. Four of the seven are circuits where the optimiser traded two-qubit gates for more
rotations (dnn_n33 282 -> 323, dnn_n51 444 -> 501, qugan_n39 279 -> 344, qugan_n71 519 -> 635
non-Clifford rotations). The LSR, gate and depth panels are unaffected. Options, your call:

1. accept it (TVD on 59 circuits);
2. forbid the pass from adding non-Clifford rotations to any program (not just Clifford
   ones) -- keeps the four, re-seeds 7 of the 71, needs a Fig 6 re-run (~35 min);
3. a different proxy rule for these circuits.

## Files

- `experiments/figures/ablation_standard/ablation_standard_logical.csv` (+ `.json`), copied to
  the project's `results/data/`; figure `results/figures/ablation_standard_logical.{pdf,svg}`
  (Fig 6's own 22 in canvas and size ladder, as the earlier standard run).
- Log `logs/optimality_front/fig6_logical.log`.
- Redraw / compare without recomputing:

      PYTHONPATH=. ../../../venv/bin/python -m experiments.figures.ablation_standard.ablation_standard \
          --from-csv experiments/figures/ablation_standard/ablation_standard_logical.csv \
          --compare <earlier ablation_standard.csv>

## Addendum 2026-10-10: no new rotations (option 2)

You chose option 2: no program may come back with more non-Clifford rotations than it had
(commit 1f83c4f). The search itself is limited (`optimize(max_rotations=)`); filtering only the
final answers had left all seven unchanged, because both of the search's chains add rotations.

The seven answers that added rotations were re-seeded under the rule:
- basis_trotter_n4: 582 -> 199 two-qubit gates, 356 -> 232 rotations. The study's answer had 217
  and 461.
- variational_n4, qaoa_n6, dnn_n33, dnn_n51, qugan_n39, qugan_n71: no improvement exists without
  more rotations, so they run as they are, with their Clifford-angle rotations named.

Fig 6 re-run under the rule (10-10 01:37-02:25, `ablation_standard_logical_v2`, 71 circuits):

| | + H commutation | + merging | + packing (= AQORN) | before the rule |
|---|---|---|---|---|
| 1q gates | 9.6 | 8.8 | **10.4** | 8.2 |
| 2q gates | 5.7 | 11.9 | **14.0** | 13.2 |
| depth | 8.8 | 14.8 | **18.3** | 17.3 |
| LSR | +8.8 | +22.3 | **+24.0** | +22.3 |
| TVD | 12.6 | 21.4 | **33.3** (61 circuits) | 32.0 (59) |

By size (AQORN):

| | 1q | 2q | depth | LSR | TVD |
|---|---|---|---|---|---|
| small (29) | 9.5 | 19.1 | 24.0 | +26.7 | 29.8 |
| medium (15) | 10.8 | 10.9 | 19.6 | +20.3 | 35.1 |
| large (27) | 11.1 | 10.2 | 11.4 | +23.1 | 37.2 |

TVD on the seven proxies from the caveat:
- dnn_n33 and dnn_n51 are resolvable again (baseline TVD 0.22 and 0.16).
- swap_test_n25/41/83 saturate as before.
- qugan_n39 and qugan_n71 still saturate, and not because of rotations. The pass names their
  Clifford-angle `rx` as `sx` / `x` (qugan_n39: 76 `sx`, 38 `x`). The proxy rule maps only rz / rx
  to S, so it keeps those gates where the pre-pass proxy had S. That is a different Clifford
  circuit, and its output covers almost every bitstring.
