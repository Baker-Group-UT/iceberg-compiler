# Handoff: optimality figures, packer comparison, logical-optimisation pass (2026-10-10)

**Branch `v2-logical-opt-pass` @ `cc8a0e7`** (worktree `.claude/worktrees/logical-opt-pass`, clean, not
merged, not pushed; 24 commits on `v2-exact-patches` 61ac6be). **Compiler frozen** at `1f83c4f`
(logical-optimisation pass, no-new-rotations rule); later commits are experiments/figures/reports only.

## Finished (all placements pass a noiseless correctness check)
| what | report | figure (project `results/figures/`) | data |
|---|---|---|---|
| Logical-optimisation pass (every program optimised first; `AQORN_LOGICAL_OPT=0` = off) | `reports/logical_optimization_pass.md` | -- | records: main checkout `results/cache/logical_optimization/` |
| Optimality front: 75 programs, 4-6q, every tightest placement | `reports/optimality_front.md` (incl. "Notes for the paper") | `optimality_front_v2.pdf` | worktree `results/optimality_front/space.csv`; copy `results/data/optimality_front_space_rule.csv` |
| Packers vs AQORN, 71 Fig 6 circuits x 11 packers | `reports/packers_standard.md` | `packers_corpus_v2.pdf` (`packers_corpus.pdf` is stale) | `results/data/optimality_front_packers_corpus.csv` |
| Annealing vs AQORN's search, 71 circuits | same report, section 2 | -- | `results/data/optimality_front_packers_corpus_annealing.csv` |
| multiplier_n15 sampled front (2000 random + 300 + 12 packers) | same report, section 3 | `sampled_front_multiplier_n15.pdf` | `results/data/optimality_front_sampled_front_multiplier_n15.csv` |
| Lane-pass screen; sweep correctness check | `reports/optimality_front.md` notes | -- | `results/data/optimality_front_{lane_refinement_screen,check_space}.csv` |
| Fig 6 with the pass (other chat) | `reports/fig6_logical.md` | `ablation_standard_logical_v2.pdf` | `results/data/ablation_standard_logical_v2.csv` |

## Unfinished / not running
- Nothing is running.
- `sampled_front` panel: labels overlap in the top cluster (AQORN, annealing, interaction partition +/- descent).
  Needs an inset or manual offsets in `experiments/figures/sampled_front/sampled_front.py`.
- Branch not merged.

## To continue on the lab server
Check out `v2-logical-opt-pass` @ `cc8a0e7`. Copy two things:
- the worktree's `results/optimality_front/` (64 MB; or the `results/data/optimality_front_*` copies);
- the main checkout's `results/cache/logical_optimization/` (2.8 MB). Without it, O3 re-optimises differently. Point `AQORN_LOGICAL_OPT_CACHE` at it.

Then, with `PYTHONPATH=.`:

    python -m experiments.figures.optimality_front.optimality_front --from-csv results/optimality_front/space.csv
    python -m experiments.figures.packers_corpus.packers_corpus --from-csv results/optimality_front/packers_corpus.csv
    python -m experiments.figures.sampled_front.sampled_front --from-csv results/optimality_front/sampled_front_multiplier_n15.csv
    python -m experiments.optimality.annealing_summary --from-csv results/optimality_front/packers_corpus_annealing.csv
    # re-runs (all resume by part file): exhaustive_front | packers_corpus | sampled_front | check_space, each --jobs N
    python -m experiments.optimality.packers_corpus --arms aqorn annealing --tag annealing --annealing-seconds 120 --jobs N

Gotchas:
- Never edit or commit experiment modules while a spawn-pool run is using them; it caused an ImportError mid-run here.
- Set `OMP_NUM_THREADS=1`.

## Decisions needed from you
1. **Layout:** one full-width figure (a)+(b)+multiplier panel, or separate figures? Should annealing be a row in the packer figure?
2. **Merge:** merge `v2-logical-opt-pass`, and into which branch (`v2-exact-patches` / `v2-revision-experiments`)?
3. **Paper text** (details in the reports):
   - Section 4.3.1 says lane order "does not affect quality". In fact it changes LER by a median 7.7% and TVD by 28% with identical gates.
   - Section 4.3.4 grid alignment never runs in the compiler; the two sentences crediting gains to it are unsupported.
   - AQORN's search is at annealing's level on most circuits, but weaker on adders/Ising chains (model value up to -13%).
   - The 5-qubit model gap.
   - multiplier_n15 was chosen as a program "where placement matters".
4. **After the deadline (compiler is frozen):** a lane pass (LER -1% median below 10q, ~0 above 30q) and an Ising/adder search fix.
