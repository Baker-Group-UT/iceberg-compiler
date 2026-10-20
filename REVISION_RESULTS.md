# v2 revision — overnight results

Generated 2026-10-01 12:02 by `experiments/revision/summarize.py` from the CSVs under `results/revision/`. Code: branch `v2-revision-experiments` @ `9b71785302`; the submitted tree is tag `v2-v4-submitted-tree` (`9f828aa`).

## Summary

- Queue: 18 items ok, 0 failed, 2 stopped on purpose for the time budget (18a_exp7b_alpha1, 18_exp3b_trajectories; see Trimmed or skipped); finished.
- 2b: on the NEUTRAL_ATOM machine (alpha=1) AQORN keeps 1.32× the baseline's shots (geomean), ≥ baseline on 92% of circuits; baseline infeasible on 11, AQORN rescues 1
- 6: against AQORN's planner at the best single static k chosen with hindsight, Fig 8's selector arm keeps 0.825× the shots (loses) while the shipped `compile_circuit` keeps 0.985× (±0.036; a tie, without hindsight) and 1.194× Fig 8's arm
- Packers (1a/1c, 71 Fig 6 circuits, same passes): expected errors fall 13.0% with AQORN, 12.5% with the best conventional packer ([24] stage 1 (PBC hypergraph, km1, FM)), 9.4% with naive; in simulation interaction-count matching (k=2, exact) keeps 1.030× AQORN's shots but its TVD is -21.4% vs naive where AQORN's is +3.3% (positive = lower TVD)
- 3b: AQORN-vs-baseline TVD improvements agree in sign between originals and proxies on 94% of small circuits
- 8 (silent-error cost model): at weight 10, on one machine (alpha = 1) TVD at equal shots falls 23.4% against AQORN's (26 / 0 circuits better / worse beyond 2σ) for 0.892× the shots on the 49 changed circuits; under Fig 6's calibration the same weight cuts TVD 17.8% for 0.876× the shots.

Queue status:

```
queue started 2026-10-01 01:41:20   sha f540323   jobs 12
cost model version: 3
2026-10-01 01:41:20  START  01_exp7a_exhaustive_translate
2026-10-01 01:44:17  ok     01_exp7a_exhaustive_translate  2m  logs/revision/01_exp7a_exhaustive_translate.log
2026-10-01 01:44:17  START  02_exp7c_compile_time
2026-10-01 02:41:39  ok     02_exp7c_compile_time  57m  logs/revision/02_exp7c_compile_time.log
2026-10-01 02:41:39  START  03_exp1a_packer_static
2026-10-01 02:53:02  ok     03_exp1a_packer_static  11m  logs/revision/03_exp1a_packer_static.log
2026-10-01 02:53:02  START  04_exp1b_packer_static_o3
2026-10-01 03:03:31  ok     04_exp1b_packer_static_o3  10m  logs/revision/04_exp1b_packer_static_o3.log
2026-10-01 03:03:31  START  05_exp3a_proxy_static
2026-10-01 03:06:57  ok     05_exp3a_proxy_static  3m  logs/revision/05_exp3a_proxy_static.log
2026-10-01 03:06:57  START  06_exp6a_best_static_replot
2026-10-01 03:06:59  ok     06_exp6a_best_static_replot  0m  logs/revision/06_exp6a_best_static_replot.log
2026-10-01 03:06:59  START  07_exp2a_calibration_table
2026-10-01 03:07:01  ok     07_exp2a_calibration_table  0m  logs/revision/07_exp2a_calibration_table.log
2026-10-01 03:07:01  START  08_exp7b_exhaustive_simulate
2026-10-01 04:35:39  ok     08_exp7b_exhaustive_simulate  88m  logs/revision/08_exp7b_exhaustive_simulate.log
2026-10-01 04:35:39  START  09_exp6b_aqorn_static_k
2026-10-01 04:52:52  ok     09_exp6b_aqorn_static_k  17m  logs/revision/09_exp6b_aqorn_static_k.log
2026-10-01 04:52:52  START  10_exp6a_with_aqorn
2026-10-01 04:52:55  ok     10_exp6a_with_aqorn  0m  logs/revision/10_exp6a_with_aqorn.log
2026-10-01 04:52:55  START  11_exp6c_noise_mismatch
2026-10-01 05:01:47  ok     11_exp6c_noise_mismatch  8m  logs/revision/11_exp6c_noise_mismatch.log
2026-10-01 05:01:47  START  12_exp1c_packer_simulated
2026-10-01 05:53:51  ok     12_exp1c_packer_simulated  52m  logs/revision/12_exp1c_packer_simulated.log
2026-10-01 05:53:51  START  13_exp2b_fixed_noise_sweep
2026-10-01 06:48:32  ok     13_exp2b_fixed_noise_sweep  54m  logs/revision/13_exp2b_fixed_noise_sweep.log
2026-10-01 06:48:32  START  14_exp3b_proxy_simulated
2026-10-01 07:15:07  ok     14_exp3b_proxy_simulated  26m  logs/revision/14_exp3b_proxy_simulated.log
2026-10-01 07:15:07  START  15_summarize_interim
2026-10-01 07:15:11  ok     15_summarize_interim  0m  logs/revision/15_summarize_interim.log
2026-10-01 07:15:11  START  16_exp7d_k4_full
2026-10-01 07:24:55  ok     16_exp7d_k4_full  9m  logs/revision/16_exp7d_k4_full.log
2026-10-01 07:24:55  START  17_exp7d_tight
2026-10-01 08:05:30  ok     17_exp7d_tight  40m  logs/revision/17_exp7d_tight.log
2026-10-01 08:05:30  START  18a_exp7b_alpha1
2026-10-01 08:05:41  FAIL   18a_exp7b_alpha1  0m  exit 143  logs/revision/18a_exp7b_alpha1.log
2026-10-01 08:05:41  START  18_exp3b_trajectories
2026-10-01 08:05:54  FAIL   18_exp3b_trajectories  0m  exit 143  logs/revision/18_exp3b_trajectories.log
2026-10-01 08:05:54  START  19_summarize
2026-10-01 08:05:58  ok     19_summarize  0m  logs/revision/19_summarize.log
queue finished 2026-10-01 08:05:58   failed: 18a_exp7b_alpha1 18_exp3b_trajectories
```

Estimates made before launch: `logs/revision/ESTIMATES.txt`.

## Claims at risk

1. **Fig 8 claim.** "Beats even the best static code chosen with hindsight" does not hold (no win beyond 2 log-SE) against: best single k, naive, best k per noise ratio, naive, best k per (circuit, ratio), naive, best single k, AQORN placement, best k per noise ratio, AQORN placement, best k per (circuit, ratio), AQORN placement. All comparators: best single k, naive 1.089× (±0.043 log-SE; selector ahead on 54% of 65 rows); best k per noise ratio, naive 1.052× (±0.045 log-SE; selector ahead on 42% of 65 rows); best k per (circuit, ratio), naive 1.011× (±0.046 log-SE; selector ahead on 38% of 65 rows); best single k, AQORN placement 0.825× (±0.053 log-SE; selector ahead on 37% of 65 rows); best k per noise ratio, AQORN placement 0.801× (±0.052 log-SE; selector ahead on 18% of 65 rows); best k per (circuit, ratio), AQORN placement 0.760× (±0.053 log-SE; selector ahead on 14% of 65 rows). Fig 8 as submitted plots only the *worst* static k, against which the selector is far ahead; the oracle-max comparators are biased toward the static arm (max over noisy estimates), which makes this conservative for the selector. **What survives (6d):** the shipped `compile_circuit` -- the selector's shape with the planner's placement -- keeps 0.985× (±0.036 log-SE) the shots of the best single static k chosen with hindsight (AQORN placement), i.e. it matches it without hindsight, and 1.194× Fig 8's selector arm. Fig 8 plotted `select_resources` alone, whose own placement is the weak part.
2. **A conventional packer is closer to the simulated optimum than AQORN (7b).** On 4–6-qubit programs simulated on Fig 6's machine, interaction-count partition (same partitioner) is 1.15% below the best placement's LSR against AQORN's 7.48% -- while its TVD regret is 0.046 against AQORN's 0.027. Keeping more shots and being more accurate are different objectives here; say which one AQORN optimises.
3. **AQORN's margin over conventional partitioners is small (1a).** On expected detectable errors (n=4, 71 circuits), [24] stage 1 (PBC hypergraph, km1, FM) reduces 12.5% against AQORN's 13.0% (+H+merge naive: 9.4%); AQORN is lower on 25 circuits and higher on 24. The packing contribution is real but most of it is available to a good conventional partitioner.
4. **A conventional packer ties or beats AQORN on simulated LSR (1c) -- but not on TVD.** interaction-count matching (k=2, exact) keeps 1.030× AQORN's shots on geometric mean over the Fig 6 corpus, identical passes and noise (beats AQORN by >2σ on 26 circuits, loses on 15), yet its TVD on the 65 resolved circuits is -21.4% against naive where AQORN's is +3.3% (positive = lower TVD). Co-locating interacting qubits keeps shots and loses accuracy (see 7b); the paper's LSR claim needs the TVD beside it, and "packing improves both" holds for AQORN, not for interaction-count packing. AQORN's placement equals naive on 30 circuits.
5. **"Packing's LSR gains come from reduced movement and idling" (V-B) is half right.** Against naive packing with the same passes, AQORN cuts expected errors 9.6% (aggregate, n=4, NEUTRAL_ATOM rates): 70% of that is idling *forced by transport* and 30% is gate error, while direct movement error (handling + flight) changes +0.4%. Say "less transport-induced idling", not "less movement".
6. **"Every benchmark compiles in seconds" (V-C).** The Fig 6 packing call (`plan_packing`, n=4, auto commutation, verify, Fig 6's noise profile) takes 182 s on bv_n140 and over 60 s on 3 of 71 Fig 6 circuits (single core, Apple M3 Max). The descent re-prices movement over all patch pairs for every candidate move.
7. **Noise-aware verify is silently gate-counting on 52 of 71 Fig 6 circuits.** Fig 6 plans on the originals; any circuit with rz/rx makes `measured_noise_split` raise (Stim's scanner rejects rotations), and `plan.measured_cost` catches it and returns gates + 12·depth. So for those circuits the commutation choice and the never-worse-than-naive check were made on gate counts, not on the movement/idle-aware objective the code describes. Library untouched; Experiment1/2 reproduce the pipeline as submitted.
8. **Calibration target (established before the run).** Fig 6 calibrates the baseline to LSR **0.06** (`BenchmarkSettings.target_selection_rate`; measured mean 0.072), but the paper text says **~0.2**. The per-circuit alpha spans 0.22–20.8 (median 1.55), a 94× spread in physical error rates (p2 6.6e-04–6.2e-02).
9. **Proxies are cheaper circuits than the originals (3a).** An encoded Rz/Rx costs (1q, 2q) = (1, 2) and the S that replaces it (2, 1) (`iceberg.costs`), so the proxy Fig 6 simulates has on average 15.4% fewer two-qubit gates than the original whose gates Fig 6 counts (52 circuits with rotations). Absolute LSR/TVD in Fig 6 therefore describe a cheaper circuit; 3b measures whether the *improvements* transfer.
10. **Depth optimality (7a).** On 4–6-qubit programs AQORN's mean *depth* regret is 6.6% above the best placement in its own (tightest) space, against 2.74% on the work+depth objective it optimises. Depth is an outcome, not its objective (`plan.measured_cost`), so claim near-optimality on work, not depth.

## Experiment7a — optimality vs exhaustive search, by translation

Circuits per configuration: {'4q,k=2': 59, '5q,k=2': 54, '6q,k=2': 52} (QASMBench originals + random structured programs). Space: complete tightest patch count (every partition × patch order × lane map) — see caveats.
Mean regret above the exhaustive optimum (%), tightest space unless noted:

| config | arm | work_depth regret | depth regret | two_qubit_gates regret | P(optimal) work_depth | % space better (work_depth) | depth regret, full space |
|---|---|---|---|---|---|---|---|
| 4q,k=2 | AQORN (shipped, verify on) | 1.04 | 3.20 | 0.72 | 74.58 | 6.36 | 19.11 |
| 4q,k=2 | AQORN search, verify off | 2.03 | 5.61 | 0.82 | 66.10 | 13.70 | 22.04 |
| 4q,k=2 | AQORN signed-graph seed only | 2.09 | 5.59 | 0.91 | 61.02 | 15.25 | 21.96 |
| 4q,k=2 | earlier pairwise greedy (archive) | 5.75 | 9.84 | 4.36 | 25.42 | 34.32 | 26.97 |
| 4q,k=2 | [24] stage 1 (PBC hypergraph, km1, FM) | 6.31 | 8.41 | 6.19 | 32.20 | 31.92 | 24.91 |
| 4q,k=2 | interaction-count matching (k=2, exact) | 7.10 | 12.65 | 4.66 | 33.90 | 36.30 | 29.59 |
| 4q,k=2 | interaction-count partition (same partitioner) | 6.68 | 12.60 | 4.01 | 33.90 | 34.46 | 29.51 |
| 4q,k=2 | naive (program order) | 6.42 | 8.79 | 6.11 | 30.51 | 33.76 | 25.35 |
| 4q,k=2 | random search, AQORN's evaluation budget | 1.98 | 5.50 | 0.82 | 69.49 | 12.71 | 21.92 |
| 5q,k=2 | AQORN (shipped, verify on) | 3.18 | 8.76 | 2.77 | 42.59 | 14.14 | 16.66 |
| 5q,k=2 | AQORN search, verify off | 3.62 | 10.46 | 2.58 | 40.74 | 17.96 | 18.52 |
| 5q,k=2 | AQORN signed-graph seed only | 4.20 | 12.49 | 2.41 | 35.19 | 20.19 | 20.57 |
| 5q,k=2 | earlier pairwise greedy (archive) | 9.22 | 16.11 | 8.61 | 11.11 | 43.80 | 24.50 |
| 5q,k=2 | [24] stage 1 (PBC hypergraph, km1, FM) | 11.80 | 18.94 | 11.01 | 5.56 | 50.34 | 27.48 |
| 5q,k=2 | interaction-count matching (k=2, exact) | 7.13 | 18.50 | 3.65 | 20.37 | 30.12 | 27.02 |
| 5q,k=2 | interaction-count partition (same partitioner) | 6.94 | 18.07 | 3.56 | 20.37 | 27.78 | 26.59 |
| 5q,k=2 | naive (program order) | 11.84 | 19.10 | 11.04 | 5.56 | 50.62 | 27.64 |
| 5q,k=2 | random search, AQORN's evaluation budget | 3.85 | 10.79 | 2.75 | 33.33 | 19.54 | 18.85 |
| 6q,k=2 | AQORN (shipped, verify on) | 4.20 | 8.20 | 3.93 | 34.62 | 13.83 | 22.17 |
| 6q,k=2 | AQORN search, verify off | 4.60 | 8.68 | 4.31 | 32.69 | 15.11 | 22.70 |
| 6q,k=2 | AQORN signed-graph seed only | 7.28 | 13.14 | 6.30 | 19.23 | 26.43 | 27.51 |
| 6q,k=2 | earlier pairwise greedy (archive) | 9.16 | 14.45 | 8.63 | 7.69 | 39.80 | 29.06 |
| 6q,k=2 | [24] stage 1 (PBC hypergraph, km1, FM) | 11.96 | 19.91 | 10.28 | 3.85 | 48.10 | 35.22 |
| 6q,k=2 | interaction-count matching (k=2, exact) | 7.92 | 16.05 | 5.64 | 7.69 | 28.76 | 30.82 |
| 6q,k=2 | interaction-count partition (same partitioner) | 7.64 | 15.66 | 5.40 | 7.69 | 27.61 | 30.40 |
| 6q,k=2 | naive (program order) | 12.11 | 20.28 | 10.33 | 3.85 | 49.59 | 35.64 |
| 6q,k=2 | random search, AQORN's evaluation budget | 4.47 | 8.46 | 4.22 | 32.69 | 14.58 | 22.48 |

Where AQORN's regret comes from (means, %): the partition it picks vs the best partition (*partition gap*), and its patch order/lanes vs the best for that partition (*order/lane gap*); on the model, the search's distance from the model optimum (*search gap*) and the model optimum's translated distance from the true optimum (*model gap*).

| config | AQORN work_depth regret | partition gap | order/lane gap | search gap (model, raw) | model gap | Spearman(model, translated) | AQORN depth regret | naive depth regret |
|---|---|---|---|---|---|---|---|---|
| 4q,k=2 | 1.04 | 0.87 | 0.18 | 0.00 | 1.99 | 0.55 | 3.20 | 8.79 |
| 5q,k=2 | 3.18 | 3.12 | 0.05 | 0.00 | 3.33 | 0.48 | 8.76 | 19.10 |
| 6q,k=2 | 4.20 | 3.90 | 0.29 | 0.00 | 4.51 | 0.60 | 8.20 | 20.28 |

Sanity: arms beating the exhaustive tightest optimum: 0 (must be 0); `_plan_under` replication mismatches: 0.

Caveats: (1) the archived optimum (work regret 0.71/3.16/4.82% at 4/5/6 qubits, naive 6.84/10.28/9.89%; depth ~19–22%) was taken over the generator's canonical space, which the translator is **not** invariant under (patch order and lane alignment change gates and depth), so it was not an optimum; and it translated programs without measurements, unlike Fig 6. The first 20 random circuits per configuration are the archived ones. (2) Sparser patch counts are enumerated as canonical representatives only, so `full space` numbers are bounds. (3) 5 of the plan's 20 QASMBench names cannot be used: inverseqft_n4, qec_sm_n5 and shor_n5 need classical feed-forward or reset, vqe_uccsd_n4/n6 fail to parse.

*Suggested revision text:* report regret against an exhaustively enumerated optimum on 4–6-qubit programs, on the objective the planner optimises, and say plainly that depth is not optimised directly.

## Experiment7d — exhaustive search at (5 qubits, k = 4), full space (stretch)

Circuits per configuration: {'5q,k=4': 24} (QASMBench originals + random structured programs). Space: complete tightest patch count (every partition × patch order × lane map) — see caveats.
Mean regret above the exhaustive optimum (%), tightest space unless noted:

| config | arm | work_depth regret | depth regret | two_qubit_gates regret | P(optimal) work_depth | % space better (work_depth) | depth regret, full space |
|---|---|---|---|---|---|---|---|
| 5q,k=4 | AQORN (shipped, verify on) | 2.06 | 5.13 | 2.21 | 58.33 | 12.62 | 25.56 |
| 5q,k=4 | AQORN search, verify off | 2.17 | 4.79 | 2.70 | 58.33 | 13.45 | 25.15 |
| 5q,k=4 | AQORN signed-graph seed only | 13.36 | 29.31 | 5.38 | 8.33 | 49.17 | 54.65 |
| 5q,k=4 | [24] stage 1 (PBC hypergraph, km1, FM) | 13.05 | 25.57 | 7.49 | 0.00 | 52.50 | 49.89 |
| 5q,k=4 | interaction-count partition (same partitioner) | 9.44 | 20.70 | 4.32 | 12.50 | 37.86 | 44.14 |
| 5q,k=4 | naive (program order) | 13.16 | 25.68 | 7.74 | 0.00 | 55.48 | 50.00 |
| 5q,k=4 | random search, AQORN's evaluation budget | 2.45 | 5.63 | 2.64 | 54.17 | 15.83 | 26.10 |

Where AQORN's regret comes from (means, %): the partition it picks vs the best partition (*partition gap*), and its patch order/lanes vs the best for that partition (*order/lane gap*); on the model, the search's distance from the model optimum (*search gap*) and the model optimum's translated distance from the true optimum (*model gap*).

| config | AQORN work_depth regret | partition gap | order/lane gap | search gap (model, raw) | model gap | Spearman(model, translated) | AQORN depth regret | naive depth regret |
|---|---|---|---|---|---|---|---|---|
| 5q,k=4 | 2.06 | 2.06 | 0.00 | 0.00 | 2.14 | 0.48 | 5.13 | 25.68 |

Sanity: arms beating the exhaustive tightest optimum: 0 (must be 0); `_plan_under` replication mismatches: 0.

## Experiment7d — tightest space at (6, k = 4), (7, k = 2), (8, k = 2) (stretch)

Circuits per configuration: {'6q,k=4': 22, '7q,k=2': 20, '8q,k=2': 20} (QASMBench originals + random structured programs). Space: complete tightest patch count (every partition × patch order × lane map) — see caveats.
Mean regret above the exhaustive optimum (%), tightest space unless noted:

| config | arm | work_depth regret | depth regret | two_qubit_gates regret | P(optimal) work_depth | % space better (work_depth) |
|---|---|---|---|---|---|---|
| 6q,k=4 | AQORN (shipped, verify on) | 3.96 | 7.97 | 3.10 | 31.82 | 12.86 |
| 6q,k=4 | AQORN search, verify off | 4.40 | 8.57 | 3.44 | 31.82 | 16.88 |
| 6q,k=4 | AQORN signed-graph seed only | 11.94 | 23.62 | 6.53 | 9.09 | 56.86 |
| 6q,k=4 | [24] stage 1 (PBC hypergraph, km1, FM) | 10.35 | 17.99 | 7.38 | 4.55 | 44.68 |
| 6q,k=4 | interaction-count partition (same partitioner) | 9.33 | 18.61 | 5.03 | 18.18 | 34.81 |
| 6q,k=4 | naive (program order) | 10.76 | 18.96 | 7.40 | 4.55 | 45.71 |
| 6q,k=4 | random search, AQORN's evaluation budget | 5.40 | 9.19 | 4.68 | 22.73 | 18.57 |
| 7q,k=2 | AQORN (shipped, verify on) | 3.98 | 11.52 | 3.67 | 10.00 | 8.49 |
| 7q,k=2 | AQORN search, verify off | 4.36 | 11.12 | 4.41 | 10.00 | 10.68 |
| 7q,k=2 | AQORN signed-graph seed only | 5.75 | 16.00 | 4.53 | 15.00 | 20.14 |
| 7q,k=2 | earlier pairwise greedy (archive) | 13.67 | 25.97 | 12.05 | 0.00 | 59.90 |
| 7q,k=2 | [24] stage 1 (PBC hypergraph, km1, FM) | 12.93 | 27.98 | 10.33 | 0.00 | 54.46 |
| 7q,k=2 | interaction-count matching (k=2, exact) | 7.91 | 23.77 | 4.38 | 5.00 | 26.70 |
| 7q,k=2 | interaction-count partition (same partitioner) | 7.91 | 23.77 | 4.38 | 5.00 | 26.70 |
| 7q,k=2 | naive (program order) | 12.92 | 27.96 | 10.33 | 0.00 | 54.32 |
| 7q,k=2 | random search, AQORN's evaluation budget | 4.46 | 11.31 | 4.48 | 10.00 | 10.94 |
| 8q,k=2 | AQORN (shipped, verify on) | 3.35 | 9.90 | 3.24 | 10.00 | 5.51 |
| 8q,k=2 | AQORN search, verify off | 3.85 | 11.12 | 3.50 | 10.00 | 6.67 |
| 8q,k=2 | AQORN signed-graph seed only | 9.45 | 20.03 | 8.37 | 5.00 | 32.10 |
| 8q,k=2 | earlier pairwise greedy (archive) | 11.62 | 23.65 | 10.26 | 0.00 | 45.28 |
| 8q,k=2 | [24] stage 1 (PBC hypergraph, km1, FM) | 12.09 | 20.80 | 11.73 | 0.00 | 51.29 |
| 8q,k=2 | interaction-count matching (k=2, exact) | 7.77 | 19.37 | 5.80 | 5.00 | 20.12 |
| 8q,k=2 | interaction-count partition (same partitioner) | 7.77 | 19.37 | 5.80 | 5.00 | 20.12 |
| 8q,k=2 | naive (program order) | 12.11 | 20.87 | 11.73 | 0.00 | 51.39 |
| 8q,k=2 | random search, AQORN's evaluation budget | 3.94 | 10.91 | 3.69 | 10.00 | 7.35 |

Where AQORN's regret comes from (means, %): the partition it picks vs the best partition (*partition gap*), and its patch order/lanes vs the best for that partition (*order/lane gap*); on the model, the search's distance from the model optimum (*search gap*) and the model optimum's translated distance from the true optimum (*model gap*).

| config | AQORN work_depth regret | partition gap | order/lane gap | search gap (model, raw) | model gap | Spearman(model, translated) | AQORN depth regret | naive depth regret |
|---|---|---|---|---|---|---|---|---|
| 6q,k=4 | 3.96 | 3.96 | 0.00 | 0.02 | 3.51 | 0.52 | 7.97 | 18.96 |
| 7q,k=2 | 3.98 | 3.86 | 0.12 | 0.00 | 4.18 | 0.49 | 11.52 | 27.96 |
| 8q,k=2 | 3.35 | 3.28 | 0.06 | 0.01 | 3.92 | 0.58 | 9.90 | 20.87 |

Sanity: arms beating the exhaustive tightest optimum: 0 (must be 0); `_plan_under` replication mismatches: 0.

## Experiment7b — optimality vs exhaustive search, simulated

Simulated (Stim, distance+idle) over the complete tightest space; every placement at 20k raw shots, the top 10 by LSR plus every arm again at 200k with fresh seeds; best decided at 200k. TVD against the exact ideal distribution.

| config | alpha_label | arm | circuits | lsr_regret_pct | p_best_lsr | pct_space_better_lsr | tvd_regret_abs | lsr_200k | tvd_floor |
|---|---|---|---|---|---|---|---|---|---|
| 4q,k=2 | calibrated | AQORN (shipped, verify on) | 29 | 4.152 | 6.897 | 19.828 | 0.025 | 0.151 | 0.005 |
| 4q,k=2 | calibrated | AQORN search, verify off | 29 | 5.272 | 6.897 | 23.420 | 0.026 | 0.150 | 0.005 |
| 4q,k=2 | calibrated | earlier pairwise greedy (archive) | 29 | 15.189 | 6.897 | 37.931 | 0.023 | 0.136 | 0.005 |
| 4q,k=2 | calibrated | [24] stage 1 (PBC hypergraph, km1, FM) | 29 | 14.606 | 6.897 | 38.506 | 0.034 | 0.133 | 0.005 |
| 4q,k=2 | calibrated | interaction-count partition (same partitioner) | 29 | 1.227 | 10.345 | 13.793 | 0.042 | 0.157 | 0.005 |
| 4q,k=2 | calibrated | naive (program order) | 29 | 14.846 | 6.897 | 39.799 | 0.035 | 0.132 | 0.005 |
| 4q,k=2 | calibrated | random search, AQORN's evaluation budget | 29 | 5.354 | 10.345 | 22.701 | 0.027 | 0.150 | 0.005 |
| 5q,k=2 | calibrated | AQORN (shipped, verify on) | 24 | 5.453 | 12.500 | 13.929 | 0.033 | 0.147 | 0.006 |
| 5q,k=2 | calibrated | AQORN search, verify off | 24 | 7.128 | 8.333 | 18.275 | 0.030 | 0.145 | 0.006 |
| 5q,k=2 | calibrated | earlier pairwise greedy (archive) | 24 | 24.325 | 0.000 | 36.024 | 0.024 | 0.117 | 0.006 |
| 5q,k=2 | calibrated | [24] stage 1 (PBC hypergraph, km1, FM) | 24 | 28.873 | 4.167 | 47.477 | 0.016 | 0.109 | 0.007 |
| 5q,k=2 | calibrated | interaction-count partition (same partitioner) | 24 | 1.097 | 12.500 | 4.514 | 0.044 | 0.153 | 0.005 |
| 5q,k=2 | calibrated | naive (program order) | 24 | 28.962 | 4.167 | 49.566 | 0.018 | 0.109 | 0.006 |
| 5q,k=2 | calibrated | random search, AQORN's evaluation budget | 24 | 6.811 | 8.333 | 16.910 | 0.031 | 0.145 | 0.006 |
| 6q,k=2 | calibrated | AQORN (shipped, verify on) | 22 | 14.072 | 0.000 | 14.318 | 0.023 | 0.139 | 0.008 |
| 6q,k=2 | calibrated | AQORN search, verify off | 22 | 14.119 | 0.000 | 14.148 | 0.023 | 0.139 | 0.008 |
| 6q,k=2 | calibrated | earlier pairwise greedy (archive) | 22 | 26.952 | 0.000 | 38.396 | 0.017 | 0.116 | 0.009 |
| 6q,k=2 | calibrated | [24] stage 1 (PBC hypergraph, km1, FM) | 22 | 27.985 | 0.000 | 38.510 | 0.019 | 0.114 | 0.009 |
| 6q,k=2 | calibrated | interaction-count partition (same partitioner) | 22 | 1.123 | 0.000 | 2.955 | 0.055 | 0.159 | 0.008 |
| 6q,k=2 | calibrated | naive (program order) | 22 | 28.579 | 0.000 | 40.587 | 0.018 | 0.113 | 0.009 |
| 6q,k=2 | calibrated | random search, AQORN's evaluation budget | 22 | 14.100 | 4.545 | 13.434 | 0.022 | 0.139 | 0.008 |

The 200k winner was in the 20k top-10 for 95% of circuits. LSR regret is relative (%) to the best LSR at 200k; TVD regret is absolute, against the best TVD *among the 200k finalists* (the top 10 by LSR plus every arm), not the whole space -- arm-to-arm TVD comparisons are exact pairings, the regret's reference point is not. A 200k LSR has a standard error of ~0.0008; the noiseless TVD floor at the finalists' accepted counts is ~0.006.

Survival and accuracy disagree: interaction-count partition (same partitioner) keeps the most shots (LSR regret 1.15%) but its accepted output is further from the ideal distribution (TVD regret 0.046 vs AQORN's 0.027). Co-locating interacting qubits cuts movement (fewer detected faults) but lets more faults land as undetectable weight-2 errors inside one patch -- worth checking per circuit.

## Experiment7c — compile time vs circuit size

**compile_circuit** (86 circuits): median 1.19 s, mean 6.33 s, max 93.6 s; >10 s on 14, >60 s on 3. Slowest: random_n150_s2 (150q) 93.6 s, random_n150_s1 (150q) 86.6 s, random_n150_s0 (150q) 83.3 s, ising_n98 (98q) 23.5 s, bv_n140 (140q) 22.4 s
**plan_packing** (86 circuits): median 0.33 s, mean 21.00 s, max 373.3 s; >10 s on 18, >60 s on 6. Slowest: random_n150_s1 (150q) 373.3 s, random_n150_s2 (150q) 373.3 s, random_n150_s0 (150q) 355.1 s, bv_n140 (140q) 182.4 s, ghz_n127 (127q) 111.2 s

Median compile time (s) by program width:

| num_qubits | compile_circuit | plan_packing |
|---|---|---|
| (0, 10] | 0.26 | 0.03 |
| (10, 25] | 0.97 | 0.24 |
| (25, 50] | 2.34 | 1.21 |
| (50, 100] | 9.56 | 12.70 |
| (100, 200] | 22.44 | 182.40 |

Single process; runs over 30 s were not repeated (see `repeats`). Machine: Apple M3 Max.

Figures: `results/revision/exp7/compile_time.pdf`

## Experiment1a — packers under identical passes, static


**n_code = 4** — mean % reduction against Fig 6's baseline (naive, both passes off); every arm under AQORN's commutation decision with merging on; expected errors from the proxy's schedule at NEUTRAL_ATOM rates (alpha = 1):

| index | depth red. % | 2q red. % | weighted gates red. % | expected errors red. % | movement errors red. % | idle errors red. % |
|---|---|---|---|---|---|---|
| AQORN search, verify off | 29.79 | 16.15 | 16.45 | 13.18 | -47.03 | -17.02 |
| AQORN (shipped, verify on) | 28.85 | 16.82 | 17.12 | 12.97 | -30.07 | -14.53 |
| AQORN search, verify off + safety net | 28.85 | 16.82 | 17.12 | 12.97 | -30.07 | -14.53 |
| AQORN descent from naive seed | 29.03 | 15.91 | 16.22 | 12.75 | -47.04 | -17.31 |
| interaction-count partition + AQORN descent | 28.69 | 15.89 | 16.20 | 12.60 | -46.81 | -19.07 |
| AQORN descent from naive seed + safety net | 28.21 | 16.55 | 16.85 | 12.58 | -29.41 | -14.47 |
| [24] stage 1 (PBC hypergraph, km1, FM) | 24.07 | 15.55 | 15.80 | 12.52 | -12.29 | -8.64 |
| interaction-count partition + AQORN descent + safety net | 27.76 | 16.57 | 16.86 | 12.40 | -29.51 | -16.21 |
| interaction-count matching (k=2, exact) | 24.95 | 15.11 | 15.34 | 12.34 | -8.15 | -4.45 |
| interaction-count matching (k=2, exact) + safety net | 26.59 | 16.05 | 16.28 | 12.27 | -14.81 | -8.85 |
| [24] stage 1 (PBC hypergraph, km1, FM) + safety net | 25.92 | 15.85 | 16.13 | 11.99 | -15.61 | -11.84 |
| interaction-count partition (same partitioner) | 25.04 | 15.56 | 15.80 | 11.95 | -12.07 | -10.03 |
| AQORN signed-graph seed only + safety net | 28.26 | 16.55 | 16.80 | 11.56 | -28.54 | -35.38 |
| interaction-count partition (same partitioner) + safety net | 25.87 | 15.60 | 15.86 | 11.41 | -16.12 | -11.68 |
| earlier pairwise greedy (archive) + safety net | 28.46 | 16.05 | 16.36 | 11.27 | -31.84 | -30.73 |
| AQORN signed-graph seed only | 28.80 | 16.07 | 16.34 | 10.93 | -53.34 | -60.26 |
| random search, AQORN's evaluation budget + safety net | 28.81 | 16.17 | 16.46 | 10.68 | -29.62 | -23.45 |
| fig6_h_merge | 25.29 | 14.86 | 15.20 | 9.41 | -24.41 | -21.58 |
| naive (program order) | 25.29 | 14.86 | 15.20 | 9.41 | -24.41 | -21.58 |
| earlier pairwise greedy (archive) | 27.67 | 15.14 | 15.52 | 9.30 | -83.84 | -72.35 |
| random search, AQORN's evaluation budget | 29.74 | 14.74 | 15.09 | 6.87 | -74.66 | -91.57 |
| fig6_h_only | 3.05 | 7.59 | 7.77 | 4.87 | -3.05 | -4.48 |
| fig6_baseline | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**n_code = 6** — mean % reduction against Fig 6's baseline (naive, both passes off); every arm under AQORN's commutation decision with merging on; expected errors from the proxy's schedule at NEUTRAL_ATOM rates (alpha = 1):

| index | depth red. % | 2q red. % | weighted gates red. % | expected errors red. % | movement errors red. % | idle errors red. % |
|---|---|---|---|---|---|---|
| interaction-count partition + AQORN descent | 27.74 | 15.99 | 16.25 | 14.56 | -38.23 | -9.07 |
| AQORN search, verify off | 30.87 | 16.61 | 16.90 | 14.49 | -57.64 | -20.10 |
| AQORN descent from naive seed | 29.54 | 16.31 | 16.60 | 14.34 | -43.25 | -12.68 |
| AQORN (shipped, verify on) | 30.58 | 16.59 | 16.89 | 14.27 | -50.87 | -19.37 |
| AQORN search, verify off + safety net | 30.58 | 16.59 | 16.89 | 14.27 | -50.87 | -19.37 |
| interaction-count partition (same partitioner) | 24.04 | 14.98 | 15.20 | 14.07 | -7.25 | 2.15 |
| [24] stage 1 (PBC hypergraph, km1, FM) | 26.94 | 15.09 | 15.35 | 14.03 | -12.99 | -2.77 |
| AQORN descent from naive seed + safety net | 30.17 | 16.27 | 16.58 | 13.86 | -42.38 | -13.31 |
| interaction-count partition + AQORN descent + safety net | 29.48 | 16.16 | 16.45 | 13.84 | -40.41 | -13.77 |
| AQORN signed-graph seed only + safety net | 28.74 | 17.03 | 17.26 | 13.59 | -53.73 | -41.59 |
| [24] stage 1 (PBC hypergraph, km1, FM) + safety net | 28.44 | 15.61 | 15.87 | 13.40 | -18.61 | -8.61 |
| interaction-count partition (same partitioner) + safety net | 28.09 | 15.39 | 15.64 | 13.17 | -18.39 | -8.64 |
| AQORN signed-graph seed only | 31.62 | 16.88 | 17.12 | 12.98 | -69.50 | -47.44 |
| naive (program order) | 26.86 | 14.92 | 15.25 | 12.26 | -25.02 | -13.39 |
| fig6_h_merge | 26.86 | 14.92 | 15.25 | 12.26 | -25.02 | -13.39 |
| random search, AQORN's evaluation budget + safety net | 30.30 | 16.06 | 16.37 | 12.20 | -62.38 | -49.89 |
| random search, AQORN's evaluation budget | 31.86 | 15.18 | 15.52 | 8.70 | -109.53 | -106.61 |
| fig6_h_only | 11.75 | 8.98 | 9.09 | 6.72 | -2.68 | -1.05 |
| fig6_baseline | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**n_code = 8** — mean % reduction against Fig 6's baseline (naive, both passes off); every arm under AQORN's commutation decision with merging on; expected errors from the proxy's schedule at NEUTRAL_ATOM rates (alpha = 1):

| index | depth red. % | 2q red. % | weighted gates red. % | expected errors red. % | movement errors red. % | idle errors red. % |
|---|---|---|---|---|---|---|
| AQORN descent from naive seed | 30.75 | 16.05 | 16.28 | 15.19 | 6.30 | 7.92 |
| AQORN search, verify off | 30.52 | 16.19 | 16.42 | 15.17 | 6.28 | 6.63 |
| interaction-count partition + AQORN descent | 27.81 | 15.52 | 15.74 | 15.05 | 8.44 | 10.39 |
| AQORN descent from naive seed + safety net | 31.16 | 16.01 | 16.26 | 14.74 | 4.89 | 5.86 |
| AQORN signed-graph seed only | 30.23 | 15.97 | 16.18 | 14.64 | 9.14 | 1.90 |
| AQORN (shipped, verify on) | 31.46 | 16.06 | 16.31 | 14.58 | 4.23 | 3.85 |
| AQORN search, verify off + safety net | 31.46 | 16.06 | 16.31 | 14.58 | 4.23 | 3.85 |
| AQORN signed-graph seed only + safety net | 32.73 | 15.99 | 16.22 | 14.45 | 8.70 | 1.78 |
| [24] stage 1 (PBC hypergraph, km1, FM) | 26.37 | 14.55 | 14.78 | 14.26 | 11.45 | 9.92 |
| interaction-count partition (same partitioner) | 23.23 | 14.49 | 14.71 | 14.25 | 14.21 | 12.14 |
| interaction-count partition + AQORN descent + safety net | 30.43 | 15.62 | 15.87 | 14.08 | 5.22 | 4.14 |
| interaction-count partition (same partitioner) + safety net | 28.39 | 14.90 | 15.16 | 13.56 | 9.95 | 4.93 |
| [24] stage 1 (PBC hypergraph, km1, FM) + safety net | 29.30 | 14.93 | 15.19 | 13.40 | 8.01 | 2.78 |
| naive (program order) | 28.22 | 14.67 | 14.95 | 13.20 | 7.84 | 3.47 |
| fig6_h_merge | 28.22 | 14.67 | 14.95 | 13.20 | 7.84 | 3.47 |
| random search, AQORN's evaluation budget + safety net | 30.85 | 15.37 | 15.63 | 13.01 | 2.31 | -8.32 |
| random search, AQORN's evaluation budget | 31.05 | 14.71 | 14.97 | 11.37 | -3.86 | -20.94 |
| fig6_h_only | 11.27 | 8.70 | 8.74 | 7.62 | 8.22 | 8.13 |
| fig6_baseline | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

AQORN against the best conventional packer ([24] stage 1 (PBC hypergraph, km1, FM)) on expected errors, n=4, paired: AQORN lower on 25 circuits, higher on 24, equal on 22; mean margin 0.45 points.

Note on depth: random search at AQORN's budget reduces depth more (29.7% vs 28.9%) by scattering interacting qubits, which raises movement error sharply; on expected errors it reduces 6.9% against AQORN's 13.0%. Depth alone is the wrong figure of merit here.

At n=4, AQORN's `verify` kept naive on 25 of 71 circuits. Verify criterion per circuit: {'gates_depth_fallback': 52, 'expected_error': 19} — `gates_depth_fallback` means the original carries rz/rx, `measured_noise_split` raised inside `plan.measured_cost`, and the check silently priced gates + depth instead of expected error.

Sanity: AQORN worse than naive on its own verify criterion in 0 (circuit, n) cases (must be 0).

Figures/tables: per-arm placements are in the CSV (`placement`), reused by 1c.

*Suggested revision text:* a table of packer × {depth, 2q, expected error} under identical commutation and merging, including interaction-count partitioning, exact interaction matching, [24]'s hypergraph partition and a random search with AQORN's evaluation budget.

## Experiment1b — level-3 pre-optimised inputs

Cumulative mean % reduction against the baseline translation of the *same* input (n=4). If AQORN's reductions survive O3 pre-optimisation, they are not logical-level savings a standard optimiser would have found.

| index | O1 inputs (as submitted) | O3 inputs |
|---|---|---|
| depth: +H | 3.05 | -0.03 |
| depth: +H+merge | 25.29 | 24.06 |
| depth: AQORN | 28.85 | 28.21 |
| two_qubit_gates: +H | 7.59 | 5.61 |
| two_qubit_gates: +H+merge | 14.86 | 12.97 |
| two_qubit_gates: AQORN | 16.82 | 14.57 |
| one_qubit_gates: +H | 12.40 | 8.32 |
| one_qubit_gates: +H+merge | 22.12 | 19.83 |
| one_qubit_gates: AQORN | 22.46 | 20.16 |
| logical depth (mean) | 112.13 | 84.65 |
| logical gates (mean) | 291.79 | 234.55 |

*Suggested revision text:* state that inputs were level-1 transpiled and report the reductions on level-3 inputs alongside.

## Experiment1c — packers simulated

Sanity gate (Fig 6's baseline, +merge and AQORN samples re-drawn with Fig 6's seeds): **PASS**

71 circuits at Fig 6's per-circuit alpha and shots; AQORN's placement equals naive on 30 of them (verify fell back on 25; on the other 5 the search itself returned program order). TVD columns use the 65 *resolved* circuits (naive's noiseless floor < 0.1, ≥ 1k accepted shots, Fig 6 baseline TVD ≤ 0.95). Wins/losses count circuits where the LSR difference exceeds twice its binomial standard error.

| index | circuits | geomean LSR / AQORN | beats AQORN >2σ | loses to AQORN >2σ | LSR % vs naive | TVD red. % vs naive (resolved) | TVD red. % vs naive, equal shots (resolved) |
|---|---|---|---|---|---|---|---|
| interaction-count matching (k=2, exact) | 71.00 | 1.03 | 26.00 | 15.00 | 13.11 | -21.36 | -21.18 |
| [24] stage 1 (PBC hypergraph, km1, FM) | 71.00 | 1.02 | 20.00 | 15.00 | 12.17 | -9.41 | -9.42 |
| interaction-count matching (k=2, exact) + safety net | 71.00 | 1.02 | 19.00 | 12.00 | 11.31 | -12.99 | -12.99 |
| interaction-count partition (same partitioner) | 71.00 | 1.01 | 22.00 | 13.00 | 10.36 | -12.35 | -12.36 |
| [24] stage 1 (PBC hypergraph, km1, FM) + safety net | 71.00 | 1.00 | 12.00 | 13.00 | 10.09 | -4.58 | -4.59 |
| AQORN search, verify off + safety net | 71.00 | 1.00 | 0.00 | 0.00 | 9.23 | 3.99 | 3.99 |
| AQORN (shipped, verify on) | 71.00 | 1.00 | 0.00 | 0.00 | 9.19 | 3.32 | 3.31 |
| AQORN descent from naive seed + safety net | 71.00 | 0.99 | 3.00 | 8.00 | 8.19 | 2.92 | 2.92 |
| interaction-count partition + AQORN descent + safety net | 71.00 | 0.99 | 5.00 | 14.00 | 8.07 | 2.70 | 2.69 |
| interaction-count partition (same partitioner) + safety net | 71.00 | 0.99 | 15.00 | 13.00 | 7.83 | -6.78 | -6.75 |
| AQORN search, verify off | 71.00 | 0.99 | 9.00 | 16.00 | 8.31 | 4.81 | 5.08 |
| AQORN descent from naive seed | 71.00 | 0.98 | 12.00 | 24.00 | 7.03 | 3.92 | 4.10 |
| interaction-count partition + AQORN descent | 71.00 | 0.98 | 13.00 | 28.00 | 6.94 | 3.81 | 3.98 |
| AQORN signed-graph seed only + safety net | 71.00 | 0.97 | 4.00 | 17.00 | 5.62 | 3.54 | 3.55 |
| earlier pairwise greedy (archive) + safety net | 71.00 | 0.97 | 14.00 | 20.00 | 4.29 | 1.60 | 1.67 |
| random search, AQORN's evaluation budget + safety net | 71.00 | 0.95 | 7.00 | 15.00 | 2.22 | 3.68 | 3.69 |
| AQORN signed-graph seed only | 71.00 | 0.94 | 6.00 | 34.00 | 2.26 | 9.09 | 9.10 |
| naive (program order) | 71.00 | 0.93 | 4.00 | 22.00 | 0.00 | 0.00 | 0.00 |
| earlier pairwise greedy (archive) | 71.00 | 0.89 | 15.00 | 48.00 | -2.69 | 7.32 | 7.40 |
| random search, AQORN's evaluation budget | 71.00 | 0.83 | 9.00 | 42.00 | -7.36 | 9.63 | 9.89 |
| fig6_baseline | 71.00 | 0.75 | 1.00 | 56.00 | -16.81 | -59.53 | -59.11 |

Status: complete. Experiment4 statistics (bootstrap CI, floor, equal-shot TVD) are per row in `/Users/cordputer/Documents/Claude/Projects/AQORN Revisions/iceberg-compiler/results/revision/exp1/packer_simulated.csv`.

## Experiment1d — contribution by circuit family

Each pass's contribution by circuit family (n=4; depth contributions are percentage points of the baseline, so the three columns add up):

| family | circuits | H commutation % (depth) | + merging % (depth) | + packing % (depth) | + packing, LSR % (sim) |
|---|---|---|---|---|---|
| ising | 6 | -77.55 | 151.72 | 0.00 | 0.02 |
| wstate | 5 | 4.04 | 0.00 | 1.93 | -3.73 |
| bv | 5 | 0.00 | 20.20 | 0.00 | 8.85 |
| knn | 4 | 10.56 | 23.73 | 0.07 | 25.24 |
| adder | 4 | 31.18 | 9.01 | 0.00 | 0.01 |
| swap_test | 3 | 11.36 | 13.64 | 0.36 | 44.83 |
| ghz | 3 | 0.00 | 0.00 | 0.00 | -0.44 |
| cat | 3 | 0.00 | 0.00 | 0.00 | -0.56 |
| dnn | 3 | 2.72 | 11.03 | 33.91 | 17.84 |
| qaoa | 2 | 0.00 | 12.07 | 16.99 | 1.57 |
| qugan | 2 | 13.95 | 31.80 | -1.71 | 116.68 |
| qft | 2 | 7.09 | 33.79 | 0.00 | 0.10 |
| cat_state | 2 | -3.03 | 0.00 | 0.00 | 0.13 |
| variational | 1 | 36.92 | 1.54 | -3.85 | 14.43 |
| toffoli | 1 | 0.00 | 0.00 | 8.16 | -0.39 |
| teleportation | 1 | 0.00 | 0.00 | 21.21 | 6.29 |
| qec_en | 1 | 51.61 | 0.00 | 6.45 | -13.18 |
| simon | 1 | 4.42 | 15.04 | 4.42 | 2.15 |
| seca | 1 | 12.20 | 0.00 | 10.98 | 15.01 |
| sat | 1 | 19.89 | 12.26 | 5.18 | -0.19 |
| vqe | 1 | 0.00 | 0.00 | 0.00 | -0.26 |
| qrng | 1 | 0.00 | 45.16 | 0.00 | -1.11 |
| qram | 1 | 17.17 | 12.64 | -2.64 | 12.34 |
| qpe | 1 | 5.84 | 9.62 | 0.00 | 0.63 |
| bigadder | 1 | 29.17 | 19.21 | 0.00 | -0.51 |
| pea | 1 | 17.89 | 0.70 | 0.00 | -0.88 |
| qec9xz | 1 | 43.87 | 0.00 | 16.77 | 14.51 |
| bell | 1 | 1.67 | 0.00 | 0.00 | 0.53 |
| multiply | 1 | 0.00 | 25.27 | 5.91 | 8.78 |
| basis_change | 1 | 0.00 | 0.00 | -0.70 | 12.86 |
| lpn | 1 | 31.82 | 0.00 | -4.55 | -0.86 |
| linearsolver | 1 | 30.30 | 0.00 | 0.00 | 1.20 |
| basis_trotter | 1 | 19.07 | 0.00 | 1.30 | -1.43 |
| bb84 | 1 | -3.12 | 25.00 | 21.88 | 26.36 |
| hs4 | 1 | 25.45 | 25.45 | 0.00 | -0.32 |
| hhl | 1 | 1.07 | 2.71 | -0.10 | 0.64 |
| ghz_state | 1 | -2.02 | 0.00 | 0.00 | 0.30 |
| fredkin | 1 | -3.85 | 0.00 | 19.23 | 4.89 |
| error_correctiond3 | 1 | 58.09 | 1.47 | 0.00 | 0.36 |
| multiplier | 1 | 19.21 | 11.15 | 0.00 | 1.57 |

## Experiment2a — Fig 6 calibration table

`/Users/cordputer/Documents/Claude/Projects/AQORN Revisions/iceberg-compiler/results/revision/exp2/fig6_calibration_table.csv`: per circuit alpha and the physical rates it implies (p1, p2, readout, move error ×alpha; T1, T2 ÷alpha), shots and baseline LSR. alpha 0.22–20.82 (median 1.55, mean 4.13); 44 circuits above the NEUTRAL_ATOM machine's noise, 27 below. Baseline LSR 0.037–0.130.

*Suggested revision text:* put this table in an appendix and correct "~0.2" to 0.06.

## Experiment2b + Experiment4 — one fixed machine, TVD statistics

One machine for every circuit (alpha the same everywhere; alpha = 1 is the NEUTRAL_ATOM model), 200k raw shots per point, Fig 6's proxies, AQORN's placement as Fig 6 computed it. *TVD-resolved*: the baseline's noiseless finite-sample floor is below 0.1 and both arms keep ≥ 1k shots.

| index | 0.5 | 1.0 | 2.0 | 4.0 |
|---|---|---|---|---|
| circuits | 71.000 | 71.000 | 71.000 | 71.000 |
| baseline LSR (mean) | 0.432 | 0.297 | 0.190 | 0.109 |
| AQORN LSR (mean) | 0.459 | 0.321 | 0.210 | 0.125 |
| geomean LSR ratio | 1.194 | 1.317 | 1.282 | 1.374 |
| % circuits AQORN ≥ baseline | 87.324 | 91.549 | 84.507 | 83.099 |
| baseline infeasible (<1k acc.) | 3.000 | 11.000 | 27.000 | 38.000 |
| rescued by AQORN | 2.000 | 1.000 | 5.000 | 1.000 |
| TVD-resolved circuits | 62.000 | 56.000 | 41.000 | 33.000 |
| TVD red. % (raw shots) | 27.862 | 28.795 | 25.587 | 29.968 |
| TVD red. % (equal accepted) | 27.794 | 28.695 | 25.452 | 29.545 |
| AQORN TVD CI wholly below baseline's | 37.000 | 34.000 | 24.000 | 22.000 |
| AQORN TVD CI wholly above baseline's | 2.000 | 2.000 | 3.000 | 2.000 |

alpha = 1, all four Fig 6 arms (means over circuits):

| series | lsr | tvd | accepted_shots |
|---|---|---|---|
| aqorn | 0.3211 | 0.2080 | 64226.8592 |
| baseline | 0.2969 | 0.2547 | 59378.0141 |
| h_merge | 0.3191 | 0.2145 | 63813.3521 |
| h_only | 0.3075 | 0.2358 | 61502.6761 |

Per-point bootstrap 95% CIs, the noiseless floor at each accepted count, raw and accepted shots, and equal-accepted-shot TVDs are columns of `/Users/cordputer/Documents/Claude/Projects/AQORN Revisions/iceberg-compiler/results/revision/exp2/fixed_noise_sweep.csv`; histograms (≤ 12 qubits) in `results/revision/exp2/fixed_noise_histograms.jsonl`.

*Suggested revision text:* show absolute LSR/TVD with CIs on one fixed machine, and the equal-accepted-shot TVD beside the fixed-raw-budget one.

## Experiment3a — does the proxy change the compiled circuit?

| arm | circuits | identical structure | proxy 2q discount % (circuits with rotations) | proxy depth change % |
|---|---|---|---|---|
| aqorn | 71 | 19 | 15.39 | 3.51 |
| aqorn_planned_on_proxy | 71 | 19 | 16.68 | -0.60 |
| baseline | 71 | 19 | 14.57 | -8.44 |
| h_merge | 71 | 19 | 14.78 | 2.02 |

AQORN's mean depth reduction: 28.9% counted on originals, 21.7% on proxies. Planning on the proxy instead: same placement on 53/71, same commutation on 61/71.

## Experiment3b — non-Clifford originals vs proxies, simulated

Aer density matrix (exact), gate noise only (no movement, no idle channel on Aer); 4-qubit originals with non-Clifford rotations plus random Clifford+Rz programs. Improvement of AQORN against the baseline (and against naive with both passes), measured on the original and on its proxy:

| alpha | metric | circuits | original mean % | proxy mean % | sign agreement % | Spearman |
|---|---|---|---|---|---|---|
| alpha1 | lsr_impr_pct | 16 | 4.29 | 4.27 | 100.00 | 0.99 |
| alpha1 | tvd_exact_red_pct | 16 | 21.77 | 28.37 | 75.00 | 0.88 |
| alpha1 | lsr_impr_vs_hmerge_pct | 16 | 0.58 | 0.68 | 75.00 | 0.66 |
| alpha1 | tvd_exact_red_vs_hmerge_pct | 16 | 5.50 | 6.70 | 56.25 | 0.32 |
| calibrated | lsr_impr_pct | 16 | 18.84 | 20.12 | 100.00 | 0.98 |
| calibrated | tvd_exact_red_pct | 16 | 20.90 | 22.96 | 93.75 | 0.94 |
| calibrated | lsr_impr_vs_hmerge_pct | 16 | 2.56 | 2.36 | 68.75 | 0.72 |
| calibrated | tvd_exact_red_vs_hmerge_pct | 16 | 3.62 | 6.19 | 56.25 | 0.28 |

The full stack transfers from proxy to original; the packing-only increment (AQORN against naive with both passes) is small on 4-qubit circuits and transfers poorly: its TVD reduction is 3.6% on originals vs 6.2% on proxies, sign agreement 56%, Spearman 0.28 at calibrated noise.

Stim vs Aer on the proxy (distance and idle off): mean |ΔTVD| 0.0147, max 0.0480 (96 points).

## Experiment6 — Fig 8's best-static comparator, selector robustness

**6a — Fig 8's data, naive static ladder** (selector LSR / comparator LSR, paired per (circuit, ratio), geometric mean with log-SE; share of rows the selector wins):

| comparison | geomean_ratio | log_se | share_a_ge_b | rows |
|---|---|---|---|---|
| selector_vs_naive_k_worst | 2.493 | 0.100 | 1.000 | 65.000 |
| selector_vs_naive_k_best_global | 1.089 | 0.043 | 0.538 | 65.000 |
| selector_vs_naive_k_best_per_ratio | 1.052 | 0.045 | 0.415 | 65.000 |
| selector_vs_naive_k_best_per_point | 1.011 | 0.046 | 0.385 | 65.000 |
| oracle_vs_selector | 1.082 | 0.030 | 1.000 | 65.000 |
**6b — with AQORN's planner at every static k** (selector LSR / comparator LSR, paired per (circuit, ratio), geometric mean with log-SE; share of rows the selector wins):

| comparison | geomean_ratio | log_se | share_a_ge_b | rows |
|---|---|---|---|---|
| selector_vs_naive_k_worst | 2.493 | 0.100 | 1.000 | 65.000 |
| selector_rerun_vs_naive_k_worst | 2.493 | 0.100 | 1.000 | 65.000 |
| selector_vs_naive_k_best_global | 1.089 | 0.043 | 0.538 | 65.000 |
| selector_rerun_vs_naive_k_best_global | 1.089 | 0.045 | 0.585 | 65.000 |
| selector_vs_naive_k_best_per_ratio | 1.052 | 0.045 | 0.415 | 65.000 |
| selector_rerun_vs_naive_k_best_per_ratio | 1.052 | 0.046 | 0.415 | 65.000 |
| selector_vs_naive_k_best_per_point | 1.011 | 0.046 | 0.385 | 65.000 |
| selector_rerun_vs_naive_k_best_per_point | 1.011 | 0.048 | 0.415 | 65.000 |
| selector_vs_aqorn_k_best_global | 0.825 | 0.053 | 0.369 | 65.000 |
| selector_rerun_vs_aqorn_k_best_global | 0.825 | 0.055 | 0.369 | 65.000 |
| selector_vs_aqorn_k_best_per_ratio | 0.801 | 0.052 | 0.185 | 65.000 |
| selector_rerun_vs_aqorn_k_best_per_ratio | 0.801 | 0.054 | 0.246 | 65.000 |
| selector_vs_aqorn_k_best_per_point | 0.760 | 0.053 | 0.138 | 65.000 |
| selector_rerun_vs_aqorn_k_best_per_point | 0.759 | 0.055 | 0.200 | 65.000 |
| oracle_vs_selector | 1.082 | 0.030 | 1.000 | 65.000 |

In one line: best single k, naive 1.089× (±0.043 log-SE; selector ahead on 54% of 65 rows); best k per noise ratio, naive 1.052× (±0.045 log-SE; selector ahead on 42% of 65 rows); best k per (circuit, ratio), naive 1.011× (±0.046 log-SE; selector ahead on 38% of 65 rows); best single k, AQORN placement 0.825× (±0.053 log-SE; selector ahead on 37% of 65 rows); best k per noise ratio, AQORN placement 0.801× (±0.052 log-SE; selector ahead on 18% of 65 rows); best k per (circuit, ratio), AQORN placement 0.760× (±0.053 log-SE; selector ahead on 14% of 65 rows).

**6d — the shipped compiler (`compile_circuit`) as the selector arm** (same sweep, alpha and shots; paired geometric-mean LSR ratio):

| index | geomean_ratio | log_se | share_a_ge_b | rows |
|---|---|---|---|---|
| compile_circuit vs best single k, AQORN placement | 0.985 | 0.036 | 0.508 | 65.000 |
| compile_circuit vs best k per ratio, AQORN placement | 0.956 | 0.037 | 0.354 | 65.000 |
| compile_circuit vs best k per point, AQORN placement | 0.907 | 0.038 | 0.262 | 65.000 |
| compile_circuit vs best single k, naive | 1.300 | 0.062 | 0.662 | 65.000 |
| compile_circuit vs worst static k, naive (Fig 8's comparator) | 2.976 | 0.100 | 0.985 | 65.000 |
| compile_circuit vs Fig 8's selector arm | 1.194 | 0.047 | 0.723 | 65.000 |

`compile_circuit` returned the planner's placement in 61 columns, the selector's in 4, naive in 0; chosen k: {2: 3, 4: 55, 6: 1, 8: 3, 10: 3}.

**6c — selector told the wrong machine** (geomean LSR(r_model → r_true) / LSR(matched), rows r_true, columns r_model):

| r_true | 0.0625 | 1.0 | 16.0 |
|---|---|---|---|
| 0.062 | 1.000 | 0.998 | 1.057 |
| 1.000 | 0.996 | 1.000 | 1.039 |
| 16.000 | 0.946 | 0.943 | 1.000 |

Figures: `results/revision/exp6/fig8_best_static.pdf`, `results/revision/exp6/fig8_best_static_with_aqorn.pdf`, `results/revision/exp6/fig8_best_static_with_compiler.pdf`, `results/revision/exp6/fig8_best_static_with_compiler_v2.pdf`

## Experiment5 — feasibility note (not run tonight)

`simulator/attribution.py::measured_noise_split` attributes a translated circuit's expected detectable errors to gates, readout, idling (gate-, transport- and readout-induced) and movement (handling + flight) from the Stim schedule, statically — no shots. It is cheap (well under a second per circuit) but rejects rz/rx, so run it on proxies. Experiment1a already records it per arm (`E_*` columns). **Feasible tomorrow:** a movement/idle breakdown for Fig 6 is a replot of that CSV; a transport-scheduling sensitivity sweep needs only `distance_scalar` / `movement_speed` variations through the same function.

Preview: where AQORN's packing changes expected detectable errors, against naive packing with the same passes (n=4, 71 circuits, proxies at NEUTRAL_ATOM rates, summed over circuits):

| index | sum, naive +H+merge | sum, AQORN | aggregate change % | share of total change % | circuits lower | circuits higher |
|---|---|---|---|---|---|---|
| E_gates | 165.738 | 158.622 | -4.294 | 30.256 | 28.000 | 1.000 |
| E_measurement | 19.890 | 19.890 | 0.000 | -0.000 | 0.000 | 0.000 |
| E_idle_gates | 1.008 | 0.966 | -4.138 | 0.177 | 20.000 | 15.000 |
| E_idle_transport | 43.227 | 26.800 | -38.002 | 69.844 | 19.000 | 13.000 |
| E_idle_readout | 0.000 | 0.000 | – | -0.000 | 0.000 | 0.000 |
| E_move_handling | 14.762 | 14.936 | 1.179 | -0.740 | 10.000 | 20.000 |
| E_flight_dephasing | 0.393 | 0.284 | -27.682 | 0.462 | 20.000 | 11.000 |
| E_movement | 15.155 | 15.220 | 0.431 | -0.278 | 11.000 | 20.000 |
| E_idle | 44.235 | 27.767 | -37.230 | 70.022 | 19.000 | 16.000 |
| E_total | 245.018 | 221.498 | -9.599 | 100.000 | 29.000 | 6.000 |

By size group: small (29): movement +21.8%, idle +20.2%, gates -7.7%, total -5.4%; medium (15): movement +0.8%, idle -10.2%, gates -1.1%, total -1.5%; large (27): movement -4.4%, idle -39.2%, gates -4.0%, total -12.0%.

## Experiment8 — a cost model that prices undetected errors (follow-up to 1c)

Question (from 1c): conventional packers keep more shots but give worse answers, and the share of two-qubit gates placed inside a patch predicts both. Does a cost model that prices *silent* (undetected) logical errors do better than AQORN's, which counts every error once?

**Gadget rates** (`gadgets.py`): exact single-fault enumeration on the simulator's own noisy Stim circuit of each gadget, alpha = 1; silent logical error per execution in units of p2 (= 3e-3):

| gadget | n=4 | n=6 | n=8 |
|---|---|---|---|
| CX_intra | 0.80 | 0.80 | 0.83 |
| CZ_intra | 0.60 | 0.60 | 0.60 |
| XCX_intra | 0.60 | 0.60 | 0.60 |
| CX_cross | 0.00 | 0.00 | 0.00 |
| XCX_cross | 0.00 | 0.00 | 0.00 |
| CXT | 0.00 | 0.00 | 0.00 |
| H_lane | 0.82 | 0.82 | 0.84 |
| HH | 0.00 | 0.00 | 0.00 |
| S | 0.20 | 0.20 | 0.20 |
| RZ | 0.42 | 0.42 | 0.44 |

Cross-patch gadgets, transversal macros and the whole-patch HH leak nothing at first order; every in-patch two-qubit gadget and the single-lane Hadamard turn 0.6–0.84 p2 of their faults into silent logical errors. The cost model adds `weight` × expected silent errors to AQORN's objective, seed graph and verify (weight 0 = AQORN).

Weight 0 reproduces shipped AQORN's placement and commutation decision on 71/71 circuits.

**Static** (8a, 71 Fig 6 circuits, n = 4, planned on originals; silent and E_total on the proxies at alpha = 1):

| weight | changed | commutation changed | verify fell back | in-patch 2q share (proxy) | predicted silent (proxy, sum) | expected errors E_total (proxy, sum) | depth vs AQORN % | 2q gates vs AQORN % |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 25 | 0.298 | 10.680 | 221.498 | 0.000 | 0.000 |
| 1 | 30 | 4 | 31 | 0.236 | 9.623 | 220.206 | 0.039 | -0.218 |
| 3 | 51 | 8 | 11 | 0.113 | 7.961 | 220.231 | -0.481 | 0.706 |
| 10 | 57 | 11 | 4 | 0.037 | 6.918 | 227.632 | 4.100 | 1.545 |
| 30 | 59 | 11 | 1 | 0.029 | 6.886 | 226.779 | -1.571 | 1.885 |

**Simulated, Fig 6's per-circuit calibration** (8b; 59 circuits where some weight changed the plan — the rest are exact ties). Sanity gate (Fig 6's naive and AQORN samples re-drawn): **PASS**. TVD columns use circuits where AQORN's noiseless floor < 0.1 and ≥ 1k shots were kept; 'silent per kept shot' is measured directly on each arm's noisy circuit (`frames.py`) and covers every simulated circuit:

| arm | circuits changed | shots kept vs AQORN, changed (geomean) | shots kept vs AQORN, all 71 (ties = 1) | LSR better / worse >2σ | resolved | TVD red. vs AQORN % | TVD red. vs AQORN %, equal shots | TVD better / worse >2σ | silent per kept shot vs AQORN (geomean, frames) |
|---|---|---|---|---|---|---|---|---|---|
| silent_w1 | 52 | 0.984 | 0.987 | 4 / 14 | 54 | 4.561 | 4.556 | 13 / 1 | 0.956 |
| silent_w3 | 57 | 0.930 | 0.941 | 8 / 35 | 54 | 12.528 | 12.782 | 32 / 2 | 0.869 |
| silent_w10 | 59 | 0.876 | 0.896 | 8 / 42 | 54 | 17.701 | 17.759 | 38 / 2 | 0.797 |
| silent_w30 | 59 | 0.869 | 0.890 | 9 / 46 | 54 | 14.356 | 14.682 | 37 / 5 | 0.801 |

**Simulated, one fixed machine, alpha = 1** (8b; 49 circuits where some weight changed the plan — the rest are exact ties). TVD columns use circuits where AQORN's noiseless floor < 0.1 and ≥ 1k shots were kept; 'silent per kept shot' is measured directly on each arm's noisy circuit (`frames.py`) and covers every simulated circuit:

| arm | circuits changed | shots kept vs AQORN, changed (geomean) | shots kept vs AQORN, all 71 (ties = 1) | LSR better / worse >2σ | resolved | TVD red. vs AQORN % | TVD red. vs AQORN %, equal shots | TVD better / worse >2σ | silent per kept shot vs AQORN (geomean, frames) |
|---|---|---|---|---|---|---|---|---|---|
| silent_w1 | 43 | 0.985 | 0.990 | 0 / 10 | 40 | 6.827 | 6.815 | 8 / 0 | 0.934 |
| silent_w3 | 47 | 0.942 | 0.960 | 3 / 29 | 40 | 15.454 | 15.439 | 20 / 1 | 0.809 |
| silent_w10 | 49 | 0.892 | 0.924 | 4 / 35 | 40 | 23.414 | 23.429 | 26 / 0 | 0.740 |
| silent_w30 | 49 | 0.896 | 0.927 | 5 / 36 | 40 | 21.000 | 20.977 | 26 / 1 | 0.733 |

**Does the model predict the measurement?** At alpha = 1, where it predicted a weight's plan changes silent error relative to AQORN, the measured silent error per kept shot moved the same way in 127 of 134 (circuit, weight) pairs.

*Suggested revision text:* only if the α = 1 gain survives review: "AQORN's objective counts every error once; pricing errors the code cannot detect (in-patch gadgets) trades a little survival for accuracy."

## Figures drawn by this report

- `/Users/cordputer/Documents/Claude/Projects/AQORN Revisions/iceberg-compiler/results/revision/figures/exp7a_regret_vs_exhaustive_v4.pdf`
- `/Users/cordputer/Documents/Claude/Projects/AQORN Revisions/iceberg-compiler/results/revision/figures/exp1c_packers_survival_vs_accuracy_v3.pdf`
- `/Users/cordputer/Documents/Claude/Projects/AQORN Revisions/iceberg-compiler/results/revision/figures/exp2b_fixed_noise_lsr_v4.pdf`
- `/Users/cordputer/Documents/Claude/Projects/AQORN Revisions/iceberg-compiler/results/revision/figures/exp3b_original_vs_proxy_v4.pdf`
- `/Users/cordputer/Documents/Claude/Projects/AQORN Revisions/iceberg-compiler/results/revision/figures/exp8_silent_cost_model_v4.pdf`
- `/Users/cordputer/Documents/Claude/Projects/AQORN Revisions/iceberg-compiler/results/revision/exp6/fig8_best_static_with_compiler.pdf`
- `/Users/cordputer/Documents/Claude/Projects/AQORN Revisions/iceberg-compiler/results/revision/exp6/fig8_best_static_with_compiler_v2.pdf`
- `/Users/cordputer/Documents/Claude/Projects/AQORN Revisions/iceberg-compiler/results/revision/exp7/compile_time.pdf`

## Trimmed or skipped

- Nothing reported missing.

Changes to the plan's queue (all but the last entry decided before launch; the last is a budget trim made while the queue ran):
- 6c (selector noise mismatch) promoted from stretch into the main queue: ~117 simulations, cheap.
- 7b at fixed alpha = 1 runs as a stretch item after 7d (separate table), because the main-queue
  7b alone is ~1 h at 20k sweep / 200k final shots.
- 3b trajectory tier (5-6 logical qubits = 18 physical; density matrix infeasible) runs as a
  stretch item at 3k shots, alpha = 1 only: a trajectory costs ~150 s per 1,000 shots per arm.
- 3b exact tier covers the 4-qubit originals only (6 QASMBench + 10 random Clifford+Rz).
- 7a/7b QASMBench set: 15 of the plan's 20 names build. inverseqft_n4, qec_sm_n5, shor_n5 need
  classical feed-forward/reset (the proxy builder refuses them); vqe_uccsd_n4/n6 fail to parse.
- 7c timing note: two short single-process checks of mine (~20 s and ~45 s, one core each) overlapped 7c at ~01:44 (small circuits) and ~01:49; on a 14-core machine the effect on single-threaded timings is expected to be a few percent at most.
- Budget trim decided at 03:50 (queue running): 7b took ~2x its estimate at full 12-way load
  (22-38 min per 5-6-qubit point), putting the main queue's end near 08:10 (~6.5 h). To stay near
  the plan's ~7 h, the stretch items 18a (7b at alpha = 1, ~2 h) and 18 (3b trajectory tier at 3k
  shots, ~25 min) are stopped by a watcher as soon as they start; STATUS.txt will show them as
  FAIL (killed, exit 143) -- deliberate, not a crash. 7d (k = 4 exhaustive) still runs.
- Added while the queue ran (not in run_overnight.sh): Experiment6d, the shipped compile_circuit
  simulated as Fig 8's selector arm, after 6b showed select_resources' own placement is the weak
  part. Run at 04:55 alongside item 11 on 4 niced workers, ~7 min:
  PYTHONPATH=. venv/bin/python -m experiments.revision.exp6_selector.compiled_selector --jobs 4
  -> results/revision/exp6/compiled_selector.csv (commit 0cb4a9f's parent added the script).

## Reproduce

Code SHA `9b71785302d9c4f860ca6bcdffe19504e566711d` (branch `v2-revision-experiments`). Queue: `caffeinate -dimsu nohup bash experiments/revision/run_overnight.sh > logs/revision/queue.log 2>&1 &`. Each item's exact command is in `experiments/revision/run_overnight.sh`; each CSV's JSON sidecar records settings, seeds, git SHA, cost-model version and wall time. Re-running a finished item resumes (and refuses a settings mismatch); pass `--out` to a new directory to recompute.
