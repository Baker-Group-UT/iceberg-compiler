# AQORN vs interaction partition + descent across idle / distance noise ratios (2026-10-10, teddy)

Branch `v2-teddy-noise-ratio` (frozen compiler cc8a0e7 + the sorted QASMBench loader 20032e4).
Script `experiments/optimality/noise_ratio_packers.py`, summary `noise_ratio_summary.py`. Data
`results/noise_ratio_packers/noise_ratio_packers.csv` (also the main checkout's `results/data/`);
full summary `logs/teddy_2026-10-10/noise_ratio_summary_cal.txt` in the main checkout.

**Setup.** Fig 6's 71 circuits x the phase diagram's 7 ratios (`SimSettings.with_noise_ratio`,
1/64 = movement-dominated .. 64 = Fig 6's machine, idle and distance scalars 1). At each point the
planner gets that machine's profile (`noise_profile_for`); AQORN = `plan_standard` (which also
decides H commutation; every arm runs at its decision); "AQORN search" = the same with its final
check off (`verify=False`: fall back to naive when naive prices lower on the translated
schedule); interaction partition (IP, cost-model-blind) and IP + AQORN's descent search AQORN's
objective at that profile. Measured as `packers_corpus` measures (Pauli frames on the Clifford
proxy, occupied-lane LER, exact TVD, 200k shots; noiseless check) at alpha calibrated per
(circuit, ratio) so naive keeps the LSR it keeps on Fig 6's machine. 17.5 min on 190 workers.

**Checks.** Inputs are the Mac's: after restoring 13 logical-opt records and sorting the loader,
all 71 reproduce `ablation_standard_logical_v2`'s plans and static counts. Ratio 64 reproduces
`packers_corpus`'s placements, commutation and static counts exactly (16/16 in the smoke test);
there the calibrated alpha equals Fig 6's (median 1.66). Every placement passes the noiseless
check; no arm failed.

## Results (medians over circuits; paired ratios are geometric means, Wilcoxon p)

| ratio | AQORN fell back to naive | LER vs naive: IP+descent / AQORN search / AQORN | IP+descent / AQORN search, LER | IP+descent / AQORN, LER | AQORN search / AQORN, LER |
|---|---|---|---|---|---|
| 1/64 | 26/71 | -1.2% / -6.4% / 0.0% | x1.018 (p 0.37) | x0.974 (p 0.28) | x0.957 (p 0.046) |
| 1/16 | 25/71 | -6.6% / -7.3% / 0.0% | x1.002 (p 0.91) | x0.959 (p 0.034) | x0.956 (p 0.004) |
| 1/4 | 24/71 | -7.9% / -9.7% / 0.0% | x1.004 (p 0.35) | x0.960 (p 0.008) | x0.956 (p 0.0007) |
| 1 | 15/71 | -9.5% / -11.3% / -6.7% | x1.021 (p 0.003) | x0.997 (p 0.57) | x0.976 (p 0.005) |
| 4 | 6/71 | -9.6% / -12.5% / -12.5% | x1.005 (p 0.30) | x1.010 (p 0.15) | x1.005 (p 0.16) |
| 16 | 4/71 | -12.7% / -15.5% / -15.5% | x1.001 (p 0.64) | x1.004 (p 0.48) | x1.002 |
| 64 (Fig 6) | 3/71 | -18.0% / -18.0% / -18.0% | x0.997 (p 0.76) | x0.999 (p 0.84) | x1.002 |

TVD follows LER (59 circuits with a non-uniform ideal output). LSR: the searches keep 0.93-0.99x
naive's shots; IP alone keeps 1.12-1.32x but cuts LER by under 2.5% at every ratio.

1. **The search parity holds at every ratio.** IP + descent ties AQORN's own search on measured
   LER, TVD and LSR (p >= 0.27) everywhere except ratio 1, where AQORN's search is 2% better on
   LER and 3% on TVD (p 0.003 / 0.006), and ratio 16, where IP + descent keeps 1.5% fewer shots
   (p 0.003). On AQORN's objective AQORN's search is lower or equal on most circuits (IP + descent
   lower on 5-11 of 71, higher on 17-39). The seed still matters little.
2. **AQORN's final check is what fails when movement dominates.** At ratios 1/64-1/4 it falls back
   to naive on 24-26 of 71 circuits, and the fallback is worse than the plan it discards: AQORN's
   search beats shipped AQORN by 4.3-4.4% LER (p 0.0007-0.046) and 5.5% TVD (p 0.0001-0.015), with
   no shots lost (LSR x1.05, n.s.). So at 1/16 and 1/4 even IP + descent, which has no check, beats
   shipped AQORN (LER x0.96, p 0.034 / 0.008). From ratio 4 up the check rarely fires (3-6/71) and
   AQORN is its search. The check prices candidates on the translated schedule; at
   movement-dominated noise that estimate disagrees with the frames.
3. **Placement buys less as movement dominates.** AQORN's search cuts LER 18% vs naive on Fig 6's
   machine, 11% at ratio 1 and 6% at 1/64; shipped AQORN's median is 0 at ratios <= 1/4.

**Caveats.** One seed per point. The calibration (naive keeps its Fig 6 LSR at every ratio) is one
convention; the phase diagram calibrates on naive k = 2 with passes off. Ratio 1 is defined by
`MOVEMENT_BALANCE = 64` (mechanisms contribute equally on the k = 2 naive baseline).

## The fix (b664c94, compiler unfrozen 10-10 evening): the final check on the standard objective

`verify_diagnosis.py` priced AQORN's search result and naive on `standard_objective` (the
selector's objective: standard translation, Clifford stand-in so rotation programs are priced on
their schedule, + the planner's silent weight) at every point, with no new simulation. Where the
two checks disagree at ratio 1/64, the old one fell back on plans 1.2-4.3x better in shots
(knn, swap_test, adders, sat) and kept plans 2-6x worse (Ising, W-state, bell, vqe); the standard
objective sided with the frames on nearly all of them. Adopted as `AqornStandard.standard_verify`
(default on; off = the old check). The sweep re-run (`noise_ratio_packers_std_verify.csv`) changes
only AQORN's arm -- naive, IP + descent and AQORN's search are identical on 497/497 points:

| ratio | fallbacks old -> new | plans changed | new / old AQORN: LER | LSR | TVD |
|---|---|---|---|---|---|
| 1/64 | 26 -> 32 | 36 | x0.902 (p 3e-6) | x1.430 (p 3e-11) | x0.909 (p 4e-4) |
| 1/16 | 25 -> 32 | 37 | x0.941 (p 0.005) | x1.373 (p 2e-11) | x0.938 (p 0.007) |
| 1/4 | 24 -> 29 | 33 | x0.974 (p 0.07) | x1.266 (p 2e-10) | x0.962 (p 0.02) |
| 1 | 15 -> 15 | 18 | x0.993 (n.s.) | x1.088 (p 2e-5) | x0.993 (n.s.) |
| 4 | 6 -> 11 | 13 | x1.001 (n.s.) | x1.075 (p 5e-4) | x1.006 (n.s.) |
| 16 | 4 -> 6 | 8 | x1.006 (n.s.) | x1.036 (p 0.02) | x1.007 (n.s.) |
| 64 (Fig 6) | 3 -> 4 | 5 | x1.004 | x1.015 | x1.006 |

Never significantly worse on LER or TVD; more shots at every ratio; large gains where movement
dominates. On Fig 6's machine 5 of 71 plans change (linearsolver_n3, qram_n20, adder_n28, adder_n64,
wstate_n118; `logs/teddy_2026-10-10/validate_fig6_static_std_verify.log`), so every standard
figure must be re-run on this branch.

**For the paper / after the freeze.** Fig 6 and every number quoted for its machine sit at ratio
64, where none of this bites. If the text claims gains across noise regimes, add that the gains
shrink as movement dominates and that the shipped fallback check discards better plans there.
After the deadline: re-examine `verify` at movement-dominated noise (measure with frames, or
compare on the same estimate the search uses).
