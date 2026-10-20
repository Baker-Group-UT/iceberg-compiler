# Branches

What each branch holds and how they relate. Regenerate the facts with:

```bash
git for-each-ref --format='%(refname:short)' refs/heads/ | while read b; do \
  echo "$b: $(git rev-list --count main..$b) ahead"; done
```

Last updated: 2026-07-29, at `trim-codebase` = `1bdec94`.

---

## v2 revision (2026-10)

```
v2-revisions (9f828aa, = submitted tree)
v2-revision-experiments                  experiment code: experiments/revision, Fig 6 regenerations;
 │                                         was 1a5359f, fast-forwarded to 4aff4c7 on 2026-10-07 --
 │                                         everything below is on it; new work goes here
 └─ v2-tvd-explore (8c85802)             asap_order, explore switches (w10T gadget), weight sweep
     ├─ v2-optimality                    exhaustive + planted optimality
     ├─ v2-logical-opt                   logical pre-optimization of the Fig 6 inputs
     ├─ v2-scaling                       compile-time scaling, exact incremental evaluation
     ├─ v2-concatenation                 [[4,2,2]] x [[4,2,2]] by translating twice
     ├─ v2-movement-idle                 Fig 6 in atoms carried / idle time
     ├─ v2-solo-gauge                    [[4,1,2]] decoding of solo patches
     └─ v2-revision-integration (4aff4c7) all six merged, plus the standard configuration
```

**Standard configuration since 2026-10-07: w10T** (`packing/standard.py`): the silent-aware
planner (`packing/silent.py`, moved from Experiment8) at weight 10, transversal in-patch CZ / XCX
at k = 2 (`TranslatorConfig.transversal_intra_two_qubit`, `iceberg/gadget_options.py`), and
layer-sorted programs (`TranslatorConfig.asap_order`). `compile_circuit` uses it by default
(`planner="v4"` is the submitted compiler). Low-level defaults (`TranslatorConfig()`,
`plan_packing`, `plan_silent_aware`) are unchanged, so archived figure scripts reproduce.
Reports per experiment branch live in `reports/`. Fig 6 under the standard, every panel
re-simulated: `experiments/figures/ablation_standard/` (`reports/fig6_standard.md`). Under the
standard `compile_circuit` plans every code size and keeps the lowest standard objective; the planner
searches at weight 5 above k = 2 (`reports/selector_standard.md`). The paper's other planner-dependent
figures have opt-in `--standard` modes, run overnight by `experiments/figures/regenerate_standard.sh`
(`reports/figures_standard.md`).

---

## The local stack

Six local branches, and they are **strictly linear** — each contains all of the one
before it, and none has diverged. Verified with `git merge-base --is-ancestor`. So
`trim-codebase` alone contains every local change, and the earlier branches are
checkpoints you can diff against rather than parallel lines of work.

```
main (b1103b2, 2026-04-15)
 └─ generalize-iceberg-nn22        3 commits
     └─ generalize-pipeline-nn22   5
         └─ experiment-qubit-grouping   7
             └─ updated-figures         20
                 └─ simplified-packing  21
                     └─ trim-codebase   25   <- current, everything is here
```

| branch | new commits | diff vs. its parent branch |
|---|---|---|
| `generalize-iceberg-nn22` | 3 | 18 files, +3,241 / −816 |
| `generalize-pipeline-nn22` | 2 | 32 files, +1,606 / −135 |
| `experiment-qubit-grouping` | 2 | 36 files, +6,577 / −585 |
| `updated-figures` | 13 | 48 files, +6,054 / −671 |
| `simplified-packing` | 1 | 16 files, +3,596 / −2 |
| `trim-codebase` | 4 | 152 files, +1,452 / −352 |

### `main` — b1103b2, 2026-04-15

The published state. `[[4,2,2]]` only, pairwise packer, no code-size generalization.
Nothing since April; every branch below is ahead of it and none has been merged back.

### `generalize-iceberg-nn22`

Lifts the `iceberg/` layer off `[[4,2,2]]` and onto the `[[n, n-2, 2]]` family. Adds
`GENERALIZATION_PLAN.md` and the numerical verification behind the core claims, and
aligns stim `movement_speed` with `noise_models.py`.

### `generalize-pipeline-nn22`

Threads the code size through the translator, simulator and pipeline, so `n` is a
parameter rather than a constant. Adds random-circuit fuzz harnesses, which found six
correctness bugs that are fixed in the same commit.

### `experiment-qubit-grouping`

Generalizes gate merging past k=2 — whole-patch macros need a gate on every occupied
lane, which at k=2 was invisible because two lanes *are* the whole patch. Replaces
translate-per-candidate placement, and adds the agglomerative-clustering grouping
experiments and the analysis of what actually drives cost.

### `updated-figures`

The largest single segment. Rebuilds the arXiv figures on the new placement stack,
measures depth and fixes the packing objective, ports the code-size sweep to the
analytic packer, and writes `STATUS.md`. Also a run of correctness fixes: gates the
translator could not place were being silently dropped, seeded simulations were not
using every shot, `pytest` was opening figure windows. Ends with distance and idle
priced into the placement cost, and refinement able to spread a program into empty
patches.

### `simplified-packing`

One commit: `placement/resource`, the layer that chooses the code size and the patch
density rather than taking them as given. Prices expected detectable errors — gate
error plus space-time volume over T₂ — and exposes `density_weight` as the exchange
rate between accepted-shot rate and patch density.

### `trim-codebase` — current

Four commits, and the only segment that mostly *removes*:

1. **Archive code that nothing imports** — `circuits/`, `dag/`, and seven loose
   modules, 2,209 lines, moved to `archive/` with the directory structure mirrored.
2. **Make placement self-contained, archive grouping/packer/repository/figures** —
   `placement/` absorbed the ~1,000 lines it used from `grouping/`; `archive/` became
   an importable package so the `PackingCost == AnalyticCost` cross-check survives;
   `repository/` came out of `pipeline/tasks/scan_*`, which now return `ScanResult`.
3. **Rename placement → packing, merge evaluator and pipeline.analysis into analysis** —
   `packing/` is the paper's word; `analysis/` replaces a one-module package and a
   six-module subpackage doing the same job either side of the pipeline boundary.
4. **Skip H commutation on circuits it hurts** — decided by measurement, worth 0.63 to
   1.17 points of mean improvement across code sizes.

Live tree is ~14,100 lines across 8 packages, down from ~20,900 across 13.
`archive/` holds 24,778 lines, importable and reference-only.

---

## Remote branches

All five predate the current work by roughly a year and **none is merged into `main`**.
They are 51–54 commits behind it, so they are historical rather than pending.

| branch | tip | ahead | behind | own commits | what it was |
|---|---|---|---|---|---|
| `origin/webinar` | 2025-09-08 | 10 | 51 | 7 | webinar demo; the only one that contains another (`refactoring`) |
| `origin/refactoring` | 2025-08-26 | 8 | 52 | 6 | the package split, TVD, extremal task, configs and presets; renamed "faulty" to "fault tolerant" |
| `origin/fault_tolerant` | 2025-08-27 | 1 | 51 | 1 | Reichardt gadgets |
| `origin/permutation_data` | 2025-08-20 | 6 | 54 | 4 | permutation search, sqale sim integration, DAG visualization, param-sweep graphing |
| `origin/permutation_post_selection` | 2025-08-20 | 4 | 54 | 2 | Iceberg post-selection on logical error rate, for permutation search |

Unlike the local stack these **genuinely diverge**. They share ancestors — the common
base of the permutation and refactoring lines is `a79affe`, 2025-08-12 — and their
`git log` listings therefore overlap, but the only strict containment among them is
`origin/webinar` ⊇ `origin/refactoring`. `fault_tolerant` is in none of the others.

So if any of this is still wanted, it has to be picked out branch by branch; there is
no single tip that subsumes the rest.

---

## What is where, if you are looking for something

| you want | branch |
|---|---|
| the published `[[4,2,2]]` state | `main` |
| the arXiv figures, runnable | `updated-figures` (they move to `archive/` after) |
| the `grouping/` stack as the live packer | `updated-figures` |
| the new packer, before any deletion | `simplified-packing` |
| everything, current | `trim-codebase` |

Nothing has been pushed. `main` is untouched locally and matches `origin/main`.
