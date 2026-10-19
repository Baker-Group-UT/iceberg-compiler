# archive/

Superseded code, kept for reference. **This is an importable package** — `archive/` and
every subdirectory carry an `__init__.py`, so `from archive.grouping.analytic import
AnalyticCost` works. That is deliberate: `packing.cost.PackingCost` is pinned to the
old `AnalyticCost` by an equality test over random partitions, and that cross-check is
what makes the new model's numbers trustworthy. Deleting the reference implementation
would have deleted the guarantee.

Directory structure mirrors the original, so `archive/grouping/analytic.py` was
`grouping/analytic.py`. Absolute self-references inside archived code were rewritten to
`archive.*` so it still imports; nothing else was changed.

## The packages

| path | lines | why |
|---|---|---|
| `grouping/` | ~3,700 | superseded by `packing/`, which absorbed the four pieces it needed |
| `packer/` | 1,480 | the pairwise k=2 packer. `packing/` imports nothing from it |
| `repository/` | ~400 | run persistence. Removed from `pipeline/tasks/scan_*` — see below |
| `experiments/` | — | the loose drivers and three folders that were at the base of `experiments/`. Reference only; see below |
| `experiments/figures/` | ~14,000 | figure scripts. Reference only; several will not run, since they import `grouping`, `packer` and the pre-strip `pipeline` |
| `tests/` | — | `test_grouping.py`, `test_analytic_placement.py`. `pytest.ini` collects `tests/` only, so these no longer run |
| loose modules | ~1,600 | see the table below |

## What is left in `experiments/`

Everything at the base of `experiments/` was archived — the loose `tvd_vs_*`, `violins`
and scan drivers, plus `error_analysis/`, `noise_scaling/` and the folder that was
already called `experiments/archive/` (now `archive/experiments/archive/`). They ran
against a `pipeline` that persisted runs and a `packer` that no longer exists, and
nothing imported them.

Three things stayed live, because something still uses them:

| path | why |
|---|---|
| `experiments/shared/` | circuit loaders. `packing/bench.py` and the live figures import it |
| `experiments/tests/` | `testexp.py` is the translator fuzz driver |
| `experiments/figures/merging_improvement/` | brought back out of this archive and ported to `packing` — see below |

## What `packing/` absorbed rather than left behind

`packing/` is now self-contained. From `grouping/` it took:

| now | was | note |
|---|---|---|
| `packing/profiles.py` | `grouping/profiles.py` | whole file |
| `packing/graph.py` | `grouping/graph.py` | whole file |
| `packing/tables.py` | part of `grouping/analytic.py` | gate tables, `mean_patch_hops`, `translated_view`, the three timeline predicates. **Not** `AnalyticCost` |
| `packing/budget.py` | part of `grouping/plan.py` | just `ResourceBudget` |
| helpers in `packing/partition.py` | part of `grouping/multilevel.py` | `_contract`, `_heavy_edge_matching`, `_greedy_partition` |

Dropped rather than ported: the `legacy` seed partitioner, which called
`grouping.multilevel.partition`. It measured -4.88 / -5.20 / -5.18 / -4.25 against
naive at n = 4/6/8/12; `feasible` matched or beat it everywhere. The numbers are
recorded in `packing/seed.py` so the comparison outlives the code.

## repository, and what replaced it

`pipeline/tasks/scan_{all,extremals,random_m,specific}.py` used `RunBuilder` to
assemble a `RunRecord` and `RunRepository` to persist it. Both imports are gone. The
tasks now return `pipeline.analysis.result_types.ScanResult`, which keeps the one thing
every caller actually reached for — `record.dataframe.df` — and writes nothing to disk.
`PipelineConfig.save` no longer has a store behind it.

Experiment drivers that used the persistence layer (`run_scan.py`, `packing_violins.py`,
`violins_30.py`, `error_analysis/calibration_sweep.py`, and the still-live
`experiments/tests/converge_fidelity.py`) were repointed at `archive.repository` /
`archive.packer` so they still import. All but the last have since been archived
themselves.

## Loose modules archived earlier

| original path | lines | why |
|---|---|---|
| `circuits/` (4 files) | 499 | nothing outside the folder imported it |
| `dag/` (4 files) | 741 | imported only by `circuits/` and its own `__main__` scripts |
| `simulator/stim_backend_archive.py` | 376 | superseded by `stim_backend.py` |
| `simulator/stim_tuner.py` | 220 | no importers |
| `grouping/tune.py` | 218 | grid search for `depth_weight`; superseded |
| `evaluator/evaluator.py` | 67 | no importers. `gate_count_stats.py` moved to the live `analysis/` package — it is the metric everything measures with |
| `repository/circuit_repository.py` | 52 | no importers |
| `pipeline/analysis/window_print.py` | 36 | no importers |
| `pipeline/run_pipeline.py` | 0 | empty |

Two `circuits/` filenames were misspelled in the original (`dag_visaulizer.py`,
`graph_visualzier.py`) and are preserved as-is so they stay findable.

## The figures that came back out

`experiments/figures/merging_improvement/` was archived here and has since been restored
to the live tree and ported to `packing`. Six scripts, all of which run:

| script | what changed |
|---|---|
| `merging_improvement.py` | `MergeMaximizingPacker` → `common.pack_program` |
| `merging_improvement_benchmark.py` | same, plus the `merge_maximizing` series renamed `packed` |
| `merging_improvement_benchmark_stylized.py` | same |
| `merging_improvement_benchmark_stylized_simulated.py` | same |
| `merging_improvement_code_size.py` | the whole `grouping` packer path → `packing.plan_packing`; `--packer analytic/measured/legacy` → `packing/raw`; `--partitioner` added; `--no-refine`, `--refine-rounds`, `--refine-seconds` dropped with the code behind them |
| `merging_improvement_code_size_placement.py` | no longer substitutes anything, since the base script it wraps now calls `packing` too |

`common.py` is new and holds the two things they all needed: the packing call, and
`stack_heights`, which used to come from `experiments/figures/updated/common.py` and is
still here.

Three call-shape changes were forced by the move and are documented at
`common.pack_program`: the packer is handed the **raw** program rather than a
preprocessed one, `h_commutation` is pinned to `True` rather than left on `"auto"`, and
`depth_weight` matches what the plotted rows are scored with.

`merging_improvement_code_size.{csv,pdf,svg}` stayed here. They are the `grouping`-era
sweep and no live script can regenerate them; the `_placement` outputs went back with
the scripts, since `placement` is what `packing` was called when they were made.

The same measurement, without the figure, is also in `packing/bench.py`:

```bash
python -m packing.bench --techniques --code-sizes 2 4 6 8 10
```

Same three cumulative reductions, same exact-translation measurement, differenced from
clipped cumulative means for the same reason the figure does it that way.

## Renames that happened alongside

`placement/` became `packing/`. `evaluator/` and `pipeline/analysis/` merged into a
single top-level `analysis/`, since between them they were one module and six, doing
one job -- turning a translated circuit or a simulation result into a comparable
number. Archived code was rewritten to match, so it still imports.
