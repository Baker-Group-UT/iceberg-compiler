# Teddy night 2026-10-10/11: summary for the morning

Branch **`v2-teddy-final`** (worktree `.claude/worktrees/teddy-final`), local only, nothing
pushed. Final compiler: commit 53e33d4 (defaults); everything after it is experiment code, reports,
and two merges -- `v2-teddy-concat` (the concatenation branch, opt-in `syndrome_rounds`) and
`v2-teddy-fig7` (an occupied-lane count in `evaluate(frames=)`). 862 tests pass on the merged head.
Figures (SVG + PDF, 24 pairs): `results/figures/teddy_final_2026-10-11/` in the main checkout.
Data (32 CSVs): `results/data/teddy_final_2026-10-11/` in the main checkout. Logs and queue scripts:
`logs/teddy_2026-10-10/final/` (one line per step in `STATUS.txt`). The b664c94 run of the same
figures ("before", `results/figures/teddy_2026-10-10/`) is kept for comparison. Also written
tonight: `reports/compiler_fixes_teddy.md` (the variant sweep). Nothing is running on teddy.

## 1. What changed in the compiler tonight

| # | change | default | evidence |
|---|---|---|---|
| 7 | final check prices plan vs naive on `standard_objective` (`standard_verify`) | on | b664c94; `reports/noise_ratio_packers.md`: the old check fell back on plans 1.2-4.3x better in shots where movement dominates |
| 8 | H-commutation decision on `standard_objective` (`standard_commutation`) | on | `reports/compiler_fixes_teddy.md`: never significantly worse at any of 7 ratios; Fig 6's machine LER x0.996, TVD x1.018, shots x1.016 (n.s.). The 48 flipped decisions trade TVD (x1.16, p 0.04) for shots (x1.11, p 0.006) at unchanged logical error |
| 10 | random restarts removed (`restarts=0`) | 0 | same sweep: LER / TVD within +-0.3%, never significant; the naive descent stays. Optimality-front search gap 0.28% -> 0.43% (median), measured gaps unchanged |
| 1 | grid alignment tested (`grid_alignment`) | off | kept on 25/497 plans, no metric moves > 1% at any ratio: leave it out, change the paper |

Every stage now compares candidates on one objective, `standard_objective` (detected errors off
the standard schedule on a Clifford stand-in + planner weight x expected silent errors): the
seed graph and descent use its analytic form (`SilentAwareCost`), the commutation decision, the
final check and the code-size selector use the scheduled form.

## 2. Final numbers per figure (final compiler)

**Fig 6 (ablation, 71 circuits, caption convention, TVD on 61)** --
`ablation_standard_logical.{svg,pdf}`; the paper version without the LSR row (10-11) is `ablation_standard_logical_1q_2q_depth_tvd.{svg,pdf}`

| stage (cumulative) | 1q | 2q | depth | LSR | TVD |
|---|---|---|---|---|---|
| +H commutation | 9.1 | 5.9 | 8.3 | +9.4 | 11.5 |
| +merging | 9.0 | 12.4 | 15.4 | +23.3 | 19.1 |
| +packing (AQORN) | **9.3** | **14.6** | **16.4** | **+28.5** | **30.6** |

By size: small (29) 1q 9.4 / 2q 19.2 / depth 24.1 / LSR 25.5 / TVD 25.6; medium (15) 8.6 / 11.6 /
18.9 / 27.2 / 33.5; large (27) 9.7 / 11.2 / 6.6 / 32.3 / 36.1. Plot averages (the bars): 13.8 /
14.7 / 21.5 / 32.9 / 32.5. Submitted: 22.5 / 16.8 / 28.9 / 38.3 / 26.8 (no logical optimisation,
baseline not layer-sorted). b664c94 tonight: 9.9 / 14.2 / 18.1 / +26.3 / 32.0. 10 plans differ
from b664c94 (7 commutation flips: ising_n10, ising_n42, qaoa_n3, bv_n14 and the three swap
tests; 3 restart changes: adder_n10, pea_n5, qaoa_n6); the swap tests' LSR gain rises (swap_test_n83 +52 ->
+121%) while their Fig 6 TVD is saturated either way. Calibration: alpha 0.21-21.9 (median 1.72,
mean 4.55), baseline LSR mean 0.072 (target 0.06 -- **the paper says ~0.2**), 296k raw shots mean.

**Fig 8 -> resource trade-off (replace)** -- `resource_tradeoff.{svg,pdf}`; Fig 8's 65
(circuit, ratio) columns less the TVD-saturated circuits (ising_n26, ising_n34, qugan_n39,
swap_test_n25) = 45 columns.

| | naive k2 | AQORN k2 | AQORN k4 | AQORN k6 | AQORN k8 | AQORN k10 | selector |
|---|---|---|---|---|---|---|---|
| shots kept / naive k2 | 1.00 | 1.20 | 1.82 | 1.05 | 1.39 | 0.98 | 1.88 |
| mean TVD | 0.115 | 0.105 | 0.129 | 0.230 | 0.236 | 0.373 | 0.136 |
| physical qubits vs naive k2 | 0 | 0 | -23% | -29% | -31% | -34% | -25% |

AQORN beats naive at every k on both axes (naive k4 1.55x / 0.155 ... k10 0.94x / 0.412). The
selector equals static k4 (1.88x / 0.136 vs 1.82x / 0.129); the measured hindsight frontier at
lambda = 4 is 1.37x / 0.102 (-11% TVD), lambda = 1 is 1.85x / 0.121. Selector weight 50: 1.59x /
0.127. Headroom panel (`resource_selection_headroom_standard`): the model's pick is the measured
best in 32/65 columns on shots, 14/65 on silent error per kept shot.

**Fig 7 (best k phase diagram)** -- `1_best_k_phase_diagram_standard{,_accuracy}.{svg,pdf}`:
fewest silent errors per kept shot is k = 2 in all 28 (size bin, ratio) cells (margin > 1 SE in
22); the shots winner varies (k4, k6, k10); the two agree in 2/28 cells. **Empty-lane bias
checked:** branch `v2-teddy-fig7` (= final + an occupied-lane count in `evaluate(frames=)`)
re-ran Fig 7 counting only lanes that hold a program qubit (`fig7_occupied/`): same winners, k = 2
in all 28 cells; the occupied-lane rate is within ~1% of the all-lane rate (median ratio
0.99-1.00 at every k), so the bias was negligible. Recommendation unchanged: fold into the text or
keep only as the shots-vs-accuracy disagreement.

**Fig 9 (density ablation)** -- `density_71/` (Fig 6's 71; the paper's version) and the 72-circuit
set beside it: `density_ablation_split_logical`, `density_vs_naive_{tight,sparse}_logical`
({svg,pdf}); data `experiments/figures/density_ablation_sweep/density_ablation_sweep_split_logical_final*.csv`
in teddy-final. Matched patch counts, merging split into 1q / 2q steps, logical error on occupied
lanes. Cumulative reduction vs naive at the same patch count (mean %, all stages / packing step),
by x = n/P (logical qubits per patch):

| n/P | 1.85 | 1.6 | 1.41 | 1.26 | 1.14 | 1.04 |
|---|---|---|---|---|---|---|
| logical error | 37.1 / 13.4 | 39.0 / 16.0 | 34.8 / 14.9 | 33.0 / 11.8 | 28.9 / 9.7 | 20.0 / 0.1 |
| TVD | 36.5 / 13.6 | 38.9 / 17.2 | 32.8 / 13.0 | 31.4 / 12.7 | 25.4 / 8.9 | 17.0 / 0.2 |
| LSR | +29.5 / +5.2 | +29.3 / +8.9 | +34.2 / +14.1 | +43.0 / +26.6 | +45.7 / +32.0 | +19.5 / +0.8 |
| depth | 15.5 / -0.6 | 22.1 / 7.3 | 20.4 / 5.0 | 18.0 / 3.9 | 14.2 / 2.9 | 13.3 / 0.1 |

Within 1-2 points of the b664c94 run everywhere (the compiler fixes barely touch it). Packing
contributes nothing at ~1 qubit per patch (n/P 1.04), as it should; the whole pipeline still gains
there through commutation and merging. Against the tight naive packing (`vs_tight`) the sparser
arms keep far fewer shots while gaining accuracy (logical error 33-52% lower): spreading qubits
over more patches trades shots for accuracy, the same trade as Fig 8. (Read the LSR magnitudes
off the figure; the summary CSV's LSR convention is not a plain percent change.)

**Packer comparison (C1/C2/D1)** -- `results/optimality_front_final/packers_corpus_final.csv`
(71 circuits, Fig 6's machine and alpha, frames, every arm under the same passes and AQORN's
commutation decision). Geometric mean vs naive (logical error / TVD on 59 unsaturated / shots):

| packer | logical error | TVD | shots | vs AQORN, logical error |
|---|---|---|---|---|
| **AQORN** | **0.815** | **0.863** | 1.011 | -- |
| AQORN search, no final check | 0.811 | 0.855 | 1.002 | x0.995 (p 0.27) |
| interaction partition + AQORN descent | 0.810 | 0.850 | 0.987 | x0.994 (p 0.29) |
| AQORN descent from naive only | 0.814 | 0.862 | 0.987 | x0.999 |
| random placements at AQORN's time budget, best on AQORN's objective | 0.818 | 0.844 | 0.856 | x1.003 |
| AQORN seed only (no descent) | 0.825 | 0.847 | 0.922 | x1.012 |
| greedy pairs | 0.881 | 0.935 | 0.925 | x1.080 (p < 1e-3) |
| hypergraph partitioning [24] | 1.029 | 1.071 | 1.112 | x1.263 (p < 1e-3) |
| interaction matching | 1.048 | 1.115 | 1.126 | x1.286 (p < 1e-3) |
| interaction partition (cost-blind) | 1.049 | 1.080 | 1.105 | x1.287 (p < 1e-3) |

Every arm that searches AQORN's objective lands within ~1% of AQORN on accuracy; every arm that
optimises connectivity alone ([24], interaction matching / partition) keeps 10-13% more shots but
is no more accurate than naive. The contribution is the cost model, not the search heuristic.

Figure: `packers_corpus.{svg,pdf}` (its panels use the median per-circuit reduction: AQORN
logical error -19.0%, TVD -14.4%; the table above is the geometric mean).

**Annealing (C3)** -- `packers_corpus_final_annealing.csv`: simulated annealing on AQORN's search
objective, 120 s per circuit, vs AQORN, 71 circuits. Search objective: median -0.03% (annealing
lower on 51%, equal on 31%); AQORN's full criterion: median 0.00% (annealing better on 23%, equal
41%); measured LSR median -0.4%, logical error +0.3% -- a wash. Largest objective gaps are on
Ising chains and wstate_n118 (annealing's plan there loses 54% LSR when measured).

**Sampled fronts (C3; four programs)** -- `sampled_front_<program>.{svg,pdf}` (paper versions `_v2`: interaction partition + descent not drawn, 10-11): 2000 random
placements + 300 random lanes/orders within AQORN's grouping + 12 packers each, frames at the
Fig 6 alpha.

| program | AQORN beats random on LSR | on logical error | random dominating AQORN | best other packer on logical error | connectivity packers' logical error |
|---|---|---|---|---|---|
| multiplier_n15 | 100% | 98.2% | 0% | IP + descent x0.95 (LSR x1.04) | x1.02-1.55 |
| qec9xz_n17 | 98.9% | 100% | 0% | IP + descent x0.95 (LSR x0.95) | x1.38-1.95 |
| adder_n10 | 95.2% | 100% | 0% | AQORN descent from naive x0.96 | x1.27-1.43 |
| ghz_state_n23 | 96.4% | 99.2% | 0% | random at AQORN's budget x0.98 (LSR x0.96) | x1.67-1.70 |

Labels in the dense cluster near AQORN overlap on every panel (interaction partition + descent,
annealing, greedy pairs, [24], random budget): fix in Inkscape. Annealing on AQORN's objective
lands at x1.03-1.08 logical error on all four. On single programs
the objective-driven searches trade places within ~5%; the connectivity packers ([24],
interaction matching / partition) are 27-95% worse on logical error.

**New: packing gain across idle/distance ratios (D5/F2)** -- `noise_ratio_packers.{svg,pdf}`.
AQORN / naive, geometric mean over 71 circuits [95% CI], calibrated noise:

| ratio | logical error | TVD | shots kept |
|---|---|---|---|
| 1/64 | 0.877 [0.832, 0.918] | 0.862 [0.813, 0.907] | 1.288 [1.179, 1.417] |
| 1/16 | 0.903 [0.869, 0.936] | 0.895 [0.859, 0.932] | 1.242 [1.146, 1.358] |
| 1/4 | 0.917 [0.885, 0.946] | 0.923 [0.889, 0.956] | 1.187 [1.110, 1.276] |
| 1 | 0.882 [0.849, 0.915] | 0.915 [0.883, 0.945] | 1.051 [0.979, 1.125] |
| 4 | 0.853 [0.816, 0.891] | 0.893 [0.845, 0.939] | 1.004 [0.939, 1.079] |
| 16 | 0.826 [0.785, 0.866] | 0.869 [0.817, 0.921] | 1.012 [0.949, 1.083] |
| 64 (Fig 6) | 0.810 [0.770, 0.851] | 0.850 [0.800, 0.905] | 1.012 [0.958, 1.073] |

Interaction partition + AQORN's descent: ties AQORN on LER/TVD from 1/16 up but keeps fewer shots
(x0.79 at 1/16 ... x0.97 at 16); at 1/64 it is worse on all three (LER x1.08, p 0.002). The
cost-blind seed alone: LER x1.04 at ratio 64.

**Optimality front (C3)** -- `results/optimality_front_final/final_front_gaps.csv` (75 programs,
4-6 qubits, every tightest placement, frames at alpha = 1, re-priced on `standard_objective`):
AQORN's pick is a median **0.43%** (mean 0.85%) from the exhaustive optimum of its own objective,
exactly optimal on 38 / 8 / 5% of programs at 4 / 5 / 6 qubits; naive is 11.6% off. Measured
median gaps to the best placement: LSR 1.8%, logical error 8.7%, TVD 27%, dominated on both LSR
and LER by 0.21% of the space (naive: 3.1 / 33 / 69%, 6.1%). With the restarts: 0.28% / 1.8 /
8.7 / 28%.

**Fixed noise, packing only (D2/E1)**, from the final noise sweep's alpha = 1 columns at Fig 6's
machine (71 circuits, Table II's machine for every circuit): absolute LSR median 0.228 naive
(IQR 0.025-0.73) vs 0.214 AQORN (0.029-0.71), ratio x1.07 (n.s.); logical error per kept shot
x0.751 (p 3e-10; median 0.079 -> 0.055); TVD x0.763 (p 1e-8; mean 0.084 -> 0.069, 54 circuits).
Naive keeps < 0.1% of shots on 5 circuits; AQORN rescues 1.

**Full pipeline at fixed noise (D2/E1/E3)** -- `results/revision/exp2_standard/fixed_noise_sweep.csv`
(71 circuits, the same physical machine for every circuit, 200k raw shots per point, Stim with
distance and idle noise, AQORN = final compiler vs the passes-off naive baseline):

| alpha | LSR median base -> AQORN | LSR geomean | TVD mean base -> AQORN (n) | at equal accepted shots | AQORN lower | baseline < 1000 kept / rescued |
|---|---|---|---|---|---|---|
| 0.5 | 0.435 -> 0.458 | x1.17 | 0.056 -> 0.033 (59) | -40.6% | 52/59 | 3 / 3 |
| **1 (Table II)** | **0.198 -> 0.214** | **x1.31** | **0.088 -> 0.054 (54)** | **-39.1%** | **50/54** | 11 / 2 |
| 2 | 0.049 -> 0.055 | x1.19 | 0.119 -> 0.082 (45) | -31.3% | 40/45 | 22 / 0 |
| 4 | 0.004 -> 0.004 | x1.22 | 0.169 -> 0.113 (34) | -33.3% | 30/34 | 36 / 0 |

TVD p < 1e-7 at every alpha (Wilcoxon). The noiseless finite-sample TVD floor at each point's
accepted count is ~0.004-0.009 for both arms; measured TVD is above twice the floor on 44-45 of
54 circuits at alpha = 1. Accepted shots at alpha = 1: median 54.7k (baseline) vs 60.9k (AQORN)
of 200k raw. Downsampling the larger arm to the smaller's accepted count leaves the TVD gain
unchanged (39.0% -> 39.1%), so it is not a sample-count effect.

**Non-Clifford originals vs proxies (D3/E2)** -- `results/revision/exp3/proxy_simulated_standard.csv`:
16 four-qubit circuits (6 QASMBench with non-Clifford angles + 10 random Clifford+Rz/Rx), exact
density-matrix simulation on Aer (gate noise only), the final compiler. AQORN vs the passes-off
baseline:

| | TVD reduction, original | TVD reduction, proxy | LSR gain, original | LSR gain, proxy | sign agreement (TVD) |
|---|---|---|---|---|---|
| alpha = 1 | 32.8% | 42.1% | +4.1% | +2.6% | 15/16 |
| calibrated | 27.4% | 32.4% | +21.4% | +13.9% | 15/16 |

The exception is qft_n4: the proxy predicts a TVD gain (35-43%) the original does not have (-24% /
-4%). Stim vs Aer on the proxies: mean |TVD diff| 0.012.

**Compile time (C3)** -- `results/scaling/standard_compile_time.csv`: the whole final planner
(`plan_standard`: both commutation arms, search, the standard-objective checks), one process per
circuit, CPU seconds on teddy (load ~33 of 384 threads). Median 0.57 s; 44/71 circuits under 1 s,
56/71 under 10 s; 51-98 qubits 10-94 s; wstate_n118 176 s, ghz_n127 292 s, cat_n130 297 s,
bv_n140 331 s. Growth ~n^3.3 (n >= 16). Translation of the plan adds < 0.2 s. (The exact
incremental evaluation on `v2-scaling` cuts n = 128 from ~200 s to ~18 s; not merged.)

**Table I rows (transversal in-patch gadgets at k = 2, [[4,2,2]])**: in-patch CZ (1q, 2q, depth)
(1, 3, 3) -> **(6, 0, 2)**; in-patch XCX (5, 3, 6) -> **(6, 0, 2)**; in-patch CX unchanged (0, 4,
4); cross-patch CXT / CZT (0, 4, 1), XCXT (8, 4, 3).

**Where the error goes (C2)** -- `results/revision/noise_budget.csv`: Fig 6's four arms priced on
the simulator's own schedule (alpha = 1). Baseline budget: gates 70.6%, idle 16.6%, movement
4.9% (the rest readout). Packing (AQORN vs +merging), geometric mean per circuit: gate error
-2.5%, idle error +5.6%, movement error +33%, total -2.9%. All stages vs baseline: gates -15%,
idle +2%, movement +41%, total -13%. **AQORN does not reduce movement or idling; it buys fewer and
cheaper gates (and fewer silent errors) with some extra transport.** This matches the 10-07
movement/idle study, and it means Sec V-B's attribution of packing's LSR gain to reduced movement
and idling has to go (paper item 14).

**Concatenation (B1/D4/F1)** -- branch `v2-teddy-concat` (= final compiler + `v2-concat-standard`,
862 tests pass), figures `results/figures/teddy_final_2026-10-11/concat/` (`concat_gain` =
option A, `concat_tradeoff`, `concat_correction`). 26 circuits <= 8 qubits, [[4,2,2]] inside
[[4,2,2]] ([[16,4,4]]), outer plans equal Fig 6's on 26/26. At calibrated noise: one level,
AQORN vs baseline 2q -21%, depth -17%, logical error 2.1x lower; two levels, AQORN at both
levels vs the concatenated baseline 2q -42%, depth -51%, LSR +30%, logical error **2.85x lower**
(25/26); concatenation alone 4.8x lower; spread vs index-order inner layout 4.8x. Error
correction (4 fault-tolerant inner syndrome rounds, first-order lookup decoder): AQORN vs the
concatenated baseline 0.30-0.50x logical error at every alpha (19-26 of 26 circuits), shots
x1.0-2.7; correction vs detection at alpha = 0.003: 0.28x the discards at 1.33x the logical
error.

**Fig 3 under the standard translation** -- `fig3_standard/4_single_patch_detection_vs_noise.{svg,pdf}`
(new `--standard` flag: layer-sorted, transversal in-patch CZ / XCX at k = 2; one patch, no idle or
movement noise). At alpha = 1: k = 2 LSR 0.933 / TVD 0.0055 (unencoded 0.038); k = 4 0.787 /
0.026; k = 6 0.761 / 0.038; k = 8 0.714 / 0.051; k = 10 0.715 / 0.051. Where each code stops
beating the unencoded circuit: k = 2 never (over alpha 0.1-20), k = 4 at alpha 3.5, k = 6 at 2.3,
k = 10 at 1.9 (k = 8's reported 0.08 is a low-noise crossing of two near-zero TVDs; read the
plot). Same message as the submitted figure, with the transversal gadget widening k = 2's lead.

**Figs 1, 5**: not re-run. Neither depends on the planner (Fig 1: two hand-picked permutations;
Fig 5: the merging pass on a toy circuit). Regenerate only if the text needs new numbers.

## 3. Paper text to change (code stays)

From the 10-10 audit (user decision: the paper follows the code), plus tonight:
1. Grid alignment (IV-C.4): remove; patch-to-site placement is left to the router / layout stage.
2. Objective: + 10 x expected silent logical errors (5 above k = 2); the seed graph subtracts the
   silent price of in-patch gates.
3. Table I / II-C: transversal in-patch CZ / XCX at k = 2 (rows above).
4. Every arm layer-sorted (fair baseline): depth 28.9% -> 16.4%.
5. Solo-qubit H = whole-patch HH in every arm, including the baseline.
6. A verified logical-optimisation round first (V-A2 "we do not re-optimize" is false now).
7. Final check and commutation decision on `standard_objective` (describe as one objective at
   every stage); the naive descent ("we also descend the naive packing"); no random restarts.
8. Sec IV-D: `compile_circuit` plans every code size and ranks on detected + 10 x silent (no
   simulation); drop "selector beats static"; Fig 8 -> trade-off figure.
9. Lane order changes LER (median 7.7%) / TVD (28%) at identical gates (IV-C.3 "does not affect
   circuit quality" holds for the circuit only).
10. Fig 9 axis = n/P (logical qubits per patch); the submitted packed arm sat at ceil(n/2) patches.
11. "Compiles in seconds" -> median 0.6 s, 56/71 under 10 s, up to 5.5 min at 140 qubits (n^3.3).
12. Stale captions ("merging dominates depth", "Ising up to 90.1%", "largest programs settle near
    k = 6").
13. **New:** calibration target is a baseline LSR of 0.06 (mean 0.072 measured), not ~0.2.
14. **New:** Sec V-B / IV-C: packing's LSR gain is not from less movement or idling (both rise
    slightly); it is from fewer / cheaper gates and fewer silent errors (noise budget above).
15. **New:** H-commutation trade: on the decisions the standard objective flips, shots rise and
    TVD can rise (swap tests); say the decision follows the objective (detected + 10 x silent).
16. **New:** state the machine (F2): one monolithic grid (storage and entangling together);
    patches = most-square rectangles of n + 2 atoms at 3 um pitch, 2-cell gutter, on a near-square
    grid of sites; Rydberg blockade 8 um (inside it, no movement); cross-patch operations move one
    group per logical operation at 1 um/us (linear), atoms return home, groups on disjoint patch
    pairs run concurrently; AOD row/column constraints and pick-up time are not modelled (arbitrary
    subsets move); handling infidelity 1e-3 per carried atom per trip; bystanders idle.

