# Handoff: density ablation (Fig 9) under the logical-optimisation pass

2026-10-10, Claude session 4959efc1. For more detail, see `REVISION_SUMMARY_2026-10-08.md`
section 9 ("10-10").

## Branch and commits (local only, nothing pushed)
- **`v2-density-logical` @ `01b5083`** (worktree `.claude/worktrees/density-logical`, clean)
  holds all of it: shared simulations (697ef6a), plus the run scripts and logs (01b5083). It sits
  on `v2-logical-opt-pass` @ 1f83c4f (the no-new-rotations rule), which sits on
  `v2-exact-patches` @ 61ac6be.
- Another session has since moved the pass branch to `cc8a0e7` (optimality and packer work).
  It changes no file my branch touches, and a trial merge of the two is clean.

## Finished
| what | where (project `results/`) |
|---|---|
| Fig 9 replacement, Fig 6's 71 circuits | `figures/density_ablation_split_logical_71.{pdf,svg}` |
| reference figures (vs tight / vs all-solo) | `figures/density_vs_naive_{tight,sparse}_logical_71.*` |
| the same with all 72 circuits (adds gcm_h6) | `figures/*_logical.*` (no `_71`) |
| data | `data/density_ablation_sweep_split_logical{,_71}{,_recount,_floors,_summary}.csv` |
| Fig 6 under the rule | `figures/ablation_standard_logical_v2.*`, CSV in `data/` |
| re-seeded optimiser answers | `iceberg-compiler/results/cache/logical_optimization/` |

Checks: the recount matches 1590/1590 arm-points; 40% of the later arms shared a simulation;
849 tests pass.

## Unfinished or running
Nothing is running. What's left are the decisions below.

## Continuing on the lab server
1. Bring four things:
   - the branch: push it, or `git bundle create density.bundle v2-density-logical`;
   - the QASMBench submodule (`git submodule update --init benchmarks/qasmbench`);
   - **`results/cache/logical_optimization/`** (2.8 MB): place it in the new main checkout
     at the same path, or set `AQORN_LOGICAL_OPT_CACHE`. O3 is nondeterministic, so without
     it every optimised input changes;
   - the project's `results/data/`, since the CSVs are not in git.
2. Re-run from the repo root, in the project's Python 3.13 env. It took 6 h on 12 workers here.
   ```bash
   export PYTHONPATH=. MPLBACKEND=Agg OMP_NUM_THREADS=1
   D=experiments/figures/density_ablation_sweep; M=experiments.figures.density_ablation_sweep
   python -m $M.density_ablation_sweep --standard --matched-patches --split-merging \
       --share-identical --jobs 12 --csv $D/density_ablation_sweep_split_logical.csv --no-show
   C=$D/density_ablation_sweep_split_logical.csv   # the sweep prints _vN instead if taken
   for s in recount floors; do python -m $M.density_references $s --matched $C --jobs 12; done
   python -m $M.density_references plot --matched $C --output-dir <figures dir>
   ```
3. The 71-circuit set, written to an empty dir; then rename the outputs to `*_71`:
   ```bash
   python -c "import pandas as p; m=p.read_csv('$C'); m[m.circuit_name!='gcm_h6'].to_csv('${C%.csv}_71.csv', index=False)"
   python -m $M.density_references plot --matched ${C%.csv}_71.csv --recount ${C%.csv}_recount.csv \
       --floors ${C%.csv}_floors.csv --output-dir <empty dir>
   ```
4. To re-plot only, copy the CSVs from `results/data/` into `$D/` and run steps 2–3's plot
   commands.

## Decisions I need from you
1. **Averaging in the reference figures.** Switch to geometric means? As plotted, the
   vs-all-solo LSR bar reads +417%, driven by a few wide circuits. The draft is
   `logs/density_matched/full_run_analysis/geomean_refs.py`. It only re-plots, but it still reads
   the 10-09 CSV names, so point it at the `_logical_71` files.
2. **71 or 72 circuits** for Fig 9. I recommend 71, Fig 6's set; the two differ by at most 0.6
   points.
3. **Six programs get no logical optimisation under the rule**: variational_n4, qaoa_n6,
   dnn_n33/51 and qugan_n39/71. Every route that cuts their 2q gates adds rotations. Should the
   optimiser get a cancellation-only route? That would change their records, so Fig 6 and these
   circuits' density rows would need a re-run.
4. **Merge** `v2-density-logical` and the pass branch into `v2-revision-experiments`?
5. **Figures now stale** under the solo-HH planner and the pass: the best-k phase diagram, the
   selector reports, and the phase diagram's logical-error recount. Should they run on the lab
   server? Fig 8 is covered by `handoffs/fig8-resource-tradeoff.md`.
