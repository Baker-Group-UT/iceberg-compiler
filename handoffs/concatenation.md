# Handoff: concatenation ([[4,2,2]] in [[4,2,2]]), simplified figures and error correction

**Branch** `v2-concat-standard`, **HEAD `3d4c8ed`**. Worktree `.claude/worktrees/concat-standard`,
clean; nothing running. It is 5 commits on `v2-logical-opt-pass` @ 9ac2dfc (which is itself
unmerged), and nothing is pushed. Full report: `reports/concat_standard.md` on the branch.

## Finished

| what | result | files (project root unless noted) |
|---|---|---|
| Re-run under the current standard (logical-opt pass, solo-HH planner, transversal CZ/XCX at both levels), 26 circuits <= 8q, 7 arms | AQORN vs its own baseline: 1 level 2q -21%, depth -17%, logical error 2.1x lower; 2 levels -42% / -50% / 2.4x lower, LSR +30%. Concatenation alone ~5x lower logical error. Index layout 4.3x worse than spread | fig `results/figures/concat_gain.pdf` (**option A, your pick**), `concat_tradeoff.pdf` (option C); data `results/data/concat_standard.csv` |
| Error correction: fault-tolerant mid-circuit inner syndrome rounds (`TranslatorConfig.syndrome_rounds`, default 0) + exact first-order lookup decoder (`experiments/concatenation/lookup.py`) | AQORN at 2 levels, 4 rounds, vs detection: at low p2, 0.27x the discards at the same logical error. At Fig A's calibrated noise, discards 0.72x but logical error 3.8x (mis-correction is second order). AQORN vs baseline under correction: 0.31-0.34x logical error (26/26), +24-28% shots. One level cannot correct (d = 2) | fig `results/figures/concat_correction.pdf`; data `results/data/concat_correct.csv` |

Code (on the branch):
- `experiments/concatenation/{concat_standard,concat_correct,lookup}.py`
- `concat_figures/concat_figures.py`
- `iceberg/iceberg_module.py` (`syndrome_round`)
- `translator/{config,core}.py` (`syndrome_rounds`, `round_cuts`)
- `tests/test_syndrome_rounds.py`

All 845 tests pass. The CSVs are gitignored: copy them from `results/data/` (or from the worktree's `results/concatenation/`).

## Unfinished / not started

- No better decoder yet. A second-order or ML decoder would push the correction-vs-detection crossover to higher noise.
- Only small circuits (<= 8 qubits) were run.
- The high-rate [[24,8,4]] variant (inner [[6,4,2]]) was not built. It needs `syndrome_round` at n = 6, which currently raises.
- No paper text written. The report has caption drafts and the sentences to use.

## Continue (from the branch checkout, repo root; ~13 min and ~23 min on 12 cores)

Inputs a fresh checkout needs:
- the `benchmarks/qasmbench` submodule;
- `results/revision/_inputs/` (gitignored; copy it over);
- the logical-optimisation cache `results/cache/logical_optimization/`, from the main checkout or via `AQORN_LOGICAL_OPT_CACHE`. Without it O3 may optimise differently.

```
PYTHONPATH=. OMP_NUM_THREADS=1 venv/bin/python -m experiments.concatenation.concat_standard --jobs 12 --fig6-plans <path>/ablation_standard_logical.csv
PYTHONPATH=. OMP_NUM_THREADS=1 venv/bin/python -m experiments.concatenation.concat_correct --jobs 12   # reads alpha* from results/concatenation/concat_standard.csv
PYTHONPATH=. venv/bin/python -m experiments.concatenation.concat_figures.concat_figures --from-csv results/concatenation/concat_standard.csv --output-dir <figures dir> --summary
PYTHONPATH=. venv/bin/python -m experiments.concatenation.concat_figures.concat_figures --correction-csv results/concatenation/concat_correct.csv --output-dir <figures dir> --rounds 4
```

## Decisions needed from you

1. **Correction in the paper?** Options: add `concat_correction.pdf` (or one sentence) next to Figure A, or keep it out of the paper. Figure A itself is detection-only.
2. **The inner level is not logically optimised**: its input is the outer code's own circuit, and rewriting it changes which faults the checks catch. Confirm, or have it optimised too (a one-line change plus a re-run).
3. **Presentation settings**: 4 rounds and decoder confidence 0.99 (0.9 is nearly identical). 1-2 rounds keep more shots at slightly higher logical error.
4. **Merge order**: this branch sits on `v2-logical-opt-pass`. Merge it after that one, or rebase it if that branch changes.
5. **Scope**: run the medium circuits or the [[24,8,4]] variant before the deadline, or not.
