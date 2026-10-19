# Silent-error weight sweep with transversal in-patch CZ/XCX (wT), 2026-10-07

Question: with w10T as the standard, is 10 the right weight, or can a finer search do better
("3 was too little, 30 too much")?

**Method.** For each weight W in {3, 5, 7, 8, 9, 10, 12, 15, 20, 30}, the silent-aware planner
(`plan_silent_aware`, weight W, Fig 6's noise profile, transversal gadget rates) planned all 71
Fig 6 circuits with TRANSVERSAL_INTRA_TWO_QUBIT on (`plan_variant.py`), and Fig 6 was re-simulated
with Fig 6's own alpha, shot counts and seeds (`ablation_variant.py`; ~14 min per weight). Analysis:
`experiments/tvd_explore/weight_sweep.py` -- caption convention (per-circuit cumulative reduction,
unclipped; TVD on Fig 6's 66 resolved circuits), paired differences against w10T with bootstrap 95%
intervals, and a split-half check (pick a weight on a random half of the circuits, stratified by
size, score it on the other half; 2000 splits). The analysis reproduces the existing w10T numbers
exactly (TVD 40.43, LSR 41.31).

| W | TVD red. % | LSR impr. % | TVD vs w10T [95% CI] | LSR vs w10T [95% CI] |
|---|---|---|---|---|
| 3 | 35.27 | 52.52 | -5.17 [-7.76, -2.91] | +11.21 [+5.06, +17.53] |
| 5 | 38.91 | 46.09 | -1.53 [-2.54, -0.62] | +4.78 [+0.77, +8.75] |
| 7 | 39.33 | 45.97 | -1.11 [-2.13, -0.27] | +4.66 [+1.82, +7.88] |
| 8 | 39.48 | 43.73 | -0.96 [-1.99, -0.11] | +2.43 [-1.44, +6.16] |
| 9 | 40.27 | 41.38 | -0.16 [-0.36, +0.00] | +0.07 [-2.79, +2.44] |
| **10** | **40.43** | **41.31** | -- | -- |
| 12 | 40.70 | 39.70 | +0.27 [-0.09, +0.69] | -1.61 [-6.65, +2.22] |
| 15 | 40.62 | 37.52 | +0.18 [-0.26, +0.68] | -3.79 [-8.81, +0.02] |
| 20 | 40.40 | 36.24 | -0.03 [-0.96, +0.78] | -5.07 [-10.21, -1.02] |
| 30 | 40.33 | 34.65 | -0.10 [-1.06, +0.70] | -6.66 [-12.06, -2.22] |

Figure: `results/figures/weight_sweep_v2.pdf` (TVD and LSR against W, overall and per size group).

**Findings.**
- **TVD plateaus from W = 9-10.** Every weight from 9 to 30 is within 0.3 points of w10T, all
  intervals include zero. Above 10 the planner only trades movement and gates for silent error it
  can no longer find: LSR falls 1.6 (12) to 6.7 (30) points, significantly from 20.
- **The apparent peak at 12 is noise.** Picking the max-TVD weight on half the circuits and
  scoring the other half gains 0.0 TVD points over w10T and loses 2.4 LSR points (it picks 12, 15
  or 20 in 89% of splits). The knee rule (best LSR within 0.5 TVD points of the best) loses 0.9 LSR.
- **Below 9 TVD falls significantly while LSR rises**: 7 gives -1.1 TVD for +4.7 LSR (both
  significant), the only real alternative; 5 is dominated by 7 (lower TVD, same LSR); 3 gives up
  5.2 TVD points.
- **Per size group** the curves differ: small circuits' TVD is flat in W (about 34% for 3-12) while
  their LSR falls -- Fig 6 runs them at alpha 6-21, where the first-order silent model is weakest;
  medium circuits plateau at 10; large circuits keep trading LSR for TVD up to 12-20. A per-group
  weight chosen on half the data does not generalize (held-out vs w10T: TVD +0.15, LSR -0.8, both
  within noise); the only stable pattern is "low weight for small circuits" (in-sample: W = 3 for
  small, 10 otherwise gives TVD 40.50, LSR 43.37 vs 40.43 / 41.31).

**Decision.** Keep **W = 10** (w10T): it is the first weight on the TVD plateau and has the best
LSR there (9 is equivalent: its plans differ from 10's on 10 of 71 circuits). If kept shots matter more than ~1 TVD
point, W = 7 is the Pareto alternative. Weights above 12 are dominated.

Re-run (from the v2-tvd-explore worktree):

    zsh experiments/tvd_explore/weight_sweep_run.sh   # plans + Fig 6 per weight (skips existing files)
    PYTHONPATH=. ../../../venv/bin/python -m experiments.tvd_explore.weight_sweep \
        --weights 3 5 7 8 9 10 12 15 20 30 --output-dir ../../../../results/figures
