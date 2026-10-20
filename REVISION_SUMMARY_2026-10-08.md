# v2 revision: everything done, 2026-10-01 to 10-09

Paper source (`paper/`) untouched throughout; nothing pushed.
Links below are relative to this file (`iceberg-compiler/`); figures live one level up, in the
project's `results/figures/`.

> **Update 2026-10-09 -- read first.** Branch `v2-exact-patches` (worktree
> `.claude/worktrees/exact-patches`, 9 commits on `e84dc85`, not merged, not pushed; 822 tests
> pass there) fixes the density ablation and two translator / planner inconsistencies found on the
> way. Details in [section 9](#9-density-ablation-and-what-it-uncovered-2026-10-0809); items added
> elsewhere are marked [10-09]. What it changes for **other** experiments:
>
> - **The standard planner changed** (61ac6be): `AqornStandard.solo_hh_aware` is now on, so plans
>   price a solo patch's Hadamards as the whole-patch HH the translator actually emits. Plans for
>   every **odd-width** circuit (one solo patch at the tightest packing) and every fixed patch
>   count can change; even-width tightest plans cannot. Stale until re-run, or until pinned to
>   `STANDARD.with_(solo_hh_aware=False)` and said so: **Fig 6 under the standard** (its
>   recorded plans), **Fig 8** (resource selection headroom), the **best-k phase diagram**, the
>   **selector results** (reports/selector_standard.md), and anything else calling `plan_standard`
>   or `compile_circuit`.
> - **No "merging off" arm was merging-free**: the Hadamard handler turns a solo qubit's H into a
>   whole-patch HH in every arm, merging on or off. It matters wherever a qubit sits alone -- odd
>   widths at the tightest packing (one patch per circuit in Fig 6) and sparse layouts. Opt-out:
>   `TranslatorConfig.solo_hh_promotion=False`.
> - **Silent error per kept shot counted flips on empty lanes** in every frame measurement, the
>   `--standard` figures included. Only the density figures are recounted. Fig 8 and the phase
>   diagram compare code sizes whose patches can be partly empty (k > 2, n not a multiple of k),
>   so their accuracy maps are biased against those configurations until recounted.
> - **TVD estimability is per circuit, not "<= 18 qubits"**: the floor is set by the ideal output's
>   support. 52 of 71 circuits have one or two outcomes and are measurable at any width; only
>   uniform-output circuits (the Ising family, qaoa_n3, qft_n4, qrng_n4, qft_n18) are not.
> - **Terminology**: figures now say "logical error rate" for silent error per kept shot (your
>   decision); code and CSVs keep the old names.
> - **Paper text**: section 4 -- Fig 9's density-ablation claims are wrong as submitted.
> - **The full density run is in** (10-09 19:36, all 71 circuits, new default planner):
>   `density_ablation_split_v3` (the Fig 9 replacement) and `density_vs_naive_{tight,sparse}_v3`;
>   results in section 9. How the reference figures average is an open choice (section 5).
> - **[10-10] The density figure under the logical-optimisation pass is in** (10-10 08:33):
>   `density_ablation_split_logical_71` -- Fig 6's 71 circuits; the Fig 9 replacement -- and
>   `density_vs_naive_{tight,sparse}_logical_71`. They supersede the 10-09 v3 figures.
>   - Rule (your option 2): no program may gain non-Clifford rotations. Commit 1f83c4f on
>     `v2-logical-opt-pass`.
>   - The run simulated every arm, with identical arms sharing one simulation
>     (`--share-identical`). Commit 697ef6a on `v2-density-logical`.
>   - Fig 6 was re-run under the rule too: LSR +24.0%, TVD 33.3%.
>   - Results in section 9.

**State of the code.** Branch `v2-revision-experiments` in this checkout holds everything, at
`e84dc85`. Tags: `v2-v4-submitted-tree` (= 9f828aa, the tree behind the submitted figures),
`v2-fig6-standard` (6c8ddaf, Fig 6 under the standard). The `v2-revision-integration` branch
and the experiment worktrees under `.claude/worktrees/` are older and fully merged; all their data
was copied into this checkout's `results/`. Tests: 714 pass (`PYTHONPATH=. venv/bin/python -m pytest -q`).
[10-09] `v2-exact-patches` branches from `e84dc85` in its own worktree and is ahead of it; see
section 9.
Data (CSVs) is not in git: it lives in `results/`, beside the figure scripts, and in the project's
`results/data/`.

Earlier index documents, still accurate for their scope: [REVISION_RESULTS.md](REVISION_RESULTS.md)
(Oct 1 experiments), [OVERNIGHT_2026-10-07.md](OVERNIGHT_2026-10-07.md) (the six overnight
branches), [BRANCH_SUMMARY_2026-10-07.md](BRANCH_SUMMARY_2026-10-07.md) (files and objects per branch).

---

## 1. The standard configuration (what AQORN is now)

Defined in [packing/standard.py](packing/standard.py) (`STANDARD`, "w10T"), adopted 2026-10-07:

| setting | value | why |
|---|---|---|
| planner | `plan_silent_aware`, silent weight **10 at k = 2**, **5 above** | AQORN's objective + w x expected silent logical errors; weights from two sweeps |
| gadgets | in-patch CZ / XCX at k = 2 built transversally (S, SX on all four data qubits) | no two-qubit gate, no first-order silent error |
| scheduling | every program layer-sorted before translation (`asap_order`) | the fair baseline: no stage is credited with a re-sort |
| resource selection | `compile_circuit` plans every code size and keeps the lowest standard objective (detected errors off the schedule + 10 x silent) | one objective in every stage |
| ranking weight | `selection_weight` = None (= 10) | the knob that moves the selector along the shots / accuracy trade |
| solo patches [10-09] | the planner prices a solo patch's H as the whole-patch HH the translator emits (`solo_hh_aware`, **on**; branch v2-exact-patches) | the translator always emits it; the old model charged a leaky single-lane gadget |

`compile_circuit(planner="v4")` and the low-level defaults (`TranslatorConfig()`, `plan_packing`)
are the submitted compiler, unchanged, so archived figure scripts reproduce. [10-09]
`plan_standard(..., patches=P)` plans at exactly P patches (opt-in), and
`STANDARD.with_(solo_hh_aware=False)` reproduces every w10T plan recorded before 10-09.

## 2. Library changes

| change | where | default |
|---|---|---|
| layer-sort programs before translation | `translator/config.py` `asap_order`, `translator/core.py` | off |
| transversal in-patch CZ / XCX, scoped per call | `TranslatorConfig.transversal_intra_two_qubit`, [iceberg/gadget_options.py](iceberg/gadget_options.py), `iceberg/costs.py`, `iceberg/iceberg_module.py`; commutation cache keyed on it (`packing/passes.py`) | off |
| silent-aware planner + packaged silent-error rate tables | [packing/silent.py](packing/silent.py), `packing/data/gadget_rates*.csv` (n_code 4-12) | -- |
| the standard; `plan_standard`, `standard_objective`, configs | [packing/standard.py](packing/standard.py) | -- |
| standard resource selector (`_compile_standard`) | [packing/compiler.py](packing/compiler.py) | `compile_circuit` default |
| price circuits with rotations on the schedule | `simulator/attribution.py` `clifford_stand_in` | -- |
| standard translation flags + frame measurement for figures | `experiments/shared/simulation.py`: `SimSettings.asap_order / transversal_merging / with_standard()`, `evaluate(frames=)` | off |
| exploration switches (majority + undo merges, solo-patch H, ...) | `translator/explore_switches.py` | off |
| `--standard` modes | Fig 8, phase diagram, density ablation scripts; [experiments/figures/regenerate_standard.sh](experiments/figures/regenerate_standard.sh) | opt-in |
| [10-09] plan at a fixed patch count | `plan_standard` / `plan_silent_aware(patches=P)`; `packing/partition.py` `partition(exact=True)`, `packing/search.py` `descend(fixed_count=True)` and `program_order`, `packing/plan.py` `naive_packing(patches=)` | opt-in |
| [10-09] merging-free translation | `TranslatorConfig.solo_hh_promotion` (False: a solo qubit's H stays the targeted gadget), `merge_two_qubit` (False: HH / XX / ZZ macros only); `iceberg/gadget_options.py` `solo_h_targeted` | shipped behaviour |
| [10-09] solo-patch pricing in the planner | `packing/cost.py` `PackingCost(solo_hh=)`, `packing/silent.py` (`SilentAwareCost`, `silent_errors(solo_hh=)`), `AqornStandard.solo_hh_aware` | **on** in the standard; off in `PackingCost` / v4 / the resource selector |
| [10-09] density sweep modes | `density_ablation_sweep.py`: `--matched-patches`, `--reuse-arms`, `--split-merging`, `--simulate`, `--groups`, `--solo-aware-planner`, `--legacy-solo-pricing`; `density_references.py` (occupied-lane recount, TVD floors by stabilizer rank, the figures) | opt-in |

New tests: `tests/test_standard.py` (20), `test_asap_order.py`, `test_concatenation.py`,
`test_solo_gauge.py`; v4-contract tests in `test_packing_compiler.py` pinned to `planner="v4"`.
[10-09] `test_exact_patches.py` (82), `test_merging_split.py` (19), `test_solo_hh_cost.py` (7): 822
on v2-exact-patches.

## 3. Investigations and results

| # | question | headline | report |
|---|---|---|---|
| 1 | Oct 1 batch: packers, fixed noise, proxies, Fig 8 comparators, optimality vs exhaustive, compile time, silent-error cost model | conventional packers match AQORN on shots but lose on TVD; `compile_circuit` ties the best static k chosen with hindsight (0.985x); pricing silent errors at weight 10 cuts TVD 18-23% for ~0.88x shots | [REVISION_RESULTS.md](REVISION_RESULTS.md) |
| 2 | TVD exploration | weight 10 + transversal in-patch CZ / XCX ("w10T") wins; several alternatives rejected by measurement | (memory; branch v2-tvd-explore) |
| 3 | fair baseline | the merging rewriter re-sorted programs: with every arm layer-sorted Fig 6 depth falls 28.9% -> 15.5% (the baseline gets 24% shallower); fidelity unaffected | `experiments/figures/ablation_fair_depth/` |
| 4 | transversal weight sweep (k = 2) | TVD plateaus from W = 9-10; keep 10 (W = 7 the only Pareto alternative) | [reports/weight_sweep.md](reports/weight_sweep.md) |
| 5 | optimality | 0.6-2.2% above the exhaustive optimum on its own objective (4-6 qubits); planted circuits: 100% of pairs recovered, exactly on the bound (6-32 qubits); inputs near-optimal in 2q structure on 41/69 | [reports/optimality.md](reports/optimality.md), [reports/logical_opt.md](reports/logical_opt.md) |
| 6 | scaling | n^3.1-3.6 as implemented; exact incremental evaluation n^2.3-2.9 (128 qubits 18 s vs 203 s); quality flat in n; "compiles in seconds" is false | [reports/scaling.md](reports/scaling.md) |
| 7 | concatenation [[16,4,4]] | AQORN at both levels: 2q -38%, depth -47%, LSR +28%, silent error -60%; silent error stays first order (circuit distance 1) | [reports/concatenation.md](reports/concatenation.md) |
| 7b [10-10] | concatenation under the current standard (logical-opt pass, solo-HH planner, transversal at both levels), simplified figures | AQORN vs its own baseline: 1 level 2q -21%, depth -17%, logical error 2.1x lower; 2 levels 2q -42%, depth -50%, LSR +30%, logical error 2.4x lower; concatenation 5x; index layout 4.3x worse than spread. CORRECTION added (TranslatorConfig.syndrome_rounds + lookup decoder): at low noise 0.27x the discards at the same logical error; trade at calibrated noise; AQORN 0.31-0.34x logical error vs baseline under correction | branch `v2-concat-standard` (worktree `.claude/worktrees/concat-standard`), reports/concat_standard.md there |
| 8 | movement and idle | AQORN carries more atoms (+26% v4, +66% w10T); per-circuit idle worse, corpus idle -23..-28%; detected error still -11..-14% | [reports/movement_idle.md](reports/movement_idle.md) |
| 9 | [[4,1,2]] on solo patches | distance stays 2; TVD -3..-7%, LSR -3..-5% for n <= 18 | [reports/solo_gauge.md](reports/solo_gauge.md) |
| 10 | Fig 6 under the standard | 1q 13.3, 2q 19.3, depth 16.4, LSR +43.2, TVD 40.7 (submitted 22.5 / 16.8 / 28.9 / +38.3 / 26.8) | [reports/fig6_standard.md](reports/fig6_standard.md) |
| 11 | resource selector quality + fix | under v4 near-optimal for shots; the first standard default discarded the w10T plan (0.86x shots, 0.97x silent vs v4) -> fixed: one objective; k = 4 weight sweep -> 5; end to end 0.82x shots, 0.75x silent per kept vs v4 | [reports/selector_standard.md](reports/selector_standard.md) |
| 12 | paper figures under the standard | Fig 8: selector 1.78x shots, 0.95x silent per kept vs naive [[4,2,2]] (k = 2 standard 1.15x / 0.78x); phase diagram: most accurate k = 2 in 28/28 cells, shots map mostly k = 4; density: TVD improves at every density | [reports/figures_standard.md](reports/figures_standard.md) |
| 13 | why AQORN wins where it wins | single-lane H costs (7, 4) and leaks 0.82 p2 silent error; AQORN removes 97% of them; H density is the best predictor of every reduction; placement fuses CXT / CZT and builds CZ transversally; rotations are the floor | [reports/why_aqorn_works.md](reports/why_aqorn_works.md) |
| 14 | are LSR and TVD anticorrelated? | no: removing gadgets improves both (H commutation, merging: 100% of pairs), placement / silent weight / code size / density trade one for the other; only the noise level ties them | [reports/lsr_vs_tvd.md](reports/lsr_vs_tvd.md) |
| 15 | selector ranking weight | w = 0: 2.05x shots, 1.15x silent per kept; 10: 1.78x / 0.95x; 50: 1.53x / 0.88x; 100: 1.31x / 0.87x; never reaches the k = 2 planner's 0.78x (the silent model limits it) | [reports/selector_standard.md](reports/selector_standard.md) |
| 16 [10-09] | why does packing do so well near density 1? | it does not: since the Aug 1 regeneration (submitted Fig 9 included) the packed arm sat at ceil(n/2) patches at every density, so "+ Packing" was mostly fewer patches. At matched patch counts packing's LSR segment falls +226% -> +15% (density 1.0-1.15) and +133% -> +15% (1.15-1.3) | section 9 |
| 17 [10-09] | AQORN vs the naive tight / naive all-solo packing across density | spreading out trades shots for accuracy (full run, geometric means). Against all-solo, AQORN at density 1.85 gets 3.8x the shots with half the qubits, at 1.22x the logical error rate. Against the tight packing, spreading to one qubit per patch keeps 0.29x the shots and lowers the logical error rate 1.46x | section 9 |
| 18 [10-09] | measurement fixes | silent per kept counted empty lanes (all-solo arms ~25% high; differs by arm within a density too); TVD's floor is the output's support, not width | section 9 |
| 19 [10-09] | is merging off in the baseline? | not for solo qubits (H -> whole-patch HH in every arm). With a merging-free baseline (full run, 71 circuits), H commutation (logical error 14-17% lower) and 1q merging (10-12%) persist at every density. 2q merging (4% -> 0) and packing (15% -> -2%) vanish at one qubit per patch; there, 74 of 76 points translate identically with and without 2q merging | section 9 |
| 20 [10-09] | translator config audit (read-only) | the planner was blind to the solo-H promotion (fixed, default on); config scopes only turn switches on; 5 switches not in `to_dict`; three old bugs verified fixed | section 9 |
| 21 [10-09] | does AQORN's placement raise LSR at the tightest packing? | not on every circuit. On GHZ / cat / W-state / Ising it trades LSR (-11..-34%) for a 17-45% lower logical error rate. On VQE / QAOA it gives up 12-17% LSR for only 2-5%. On qugan / multiplier / swap_test / knn / dnn it gains both (LSR +20..+89%). Large-group median: -0.9%. Mean-of-percent bars against all-solo inflate the losses ~10x | section 9 |

## 4. Paper text that changes (your edits; nothing in `paper/` was touched)

- **Fig 6 / values.tex** (if the standard is adopted): `\OneQReduction` 13.3%, `\TwoQReduction`
  19.3%, `\DepthReduction` 16.4%, `\LSRImprovement` 43.2%, `\TVDImprovement` 40.7%; the figure
  is `results/figures/ablation_standard.pdf`. Caption: "merging dominates the depth reduction" ->
  H commutation and merging about equal; "Ising ... up to 90.1%" no longer holds (largest:
  error_correctiond3_n5 62.9%; ising_n98's plan is 3.5x deeper). Details: reports/fig6_standard.md.
- **Fig 8**: the selector now trades shots for accuracy on purpose; claim its shots (1.8x) at
  naive accuracy, and that the k = 2 standard is the most accurate configuration.
- **Phase diagram**: "largest programs settle near k = 6" does not hold (k = 4 -> 8 -> 2 as idling
  grows); the paper's figure is the Jul 30 v3 run. Shots and accuracy pick different k.
- **Movement (V-B)**: do not claim less movement; at most less transport-forced idling on wide circuits.
- **"Every benchmark compiles in seconds" (V-C)**: false as written (scaling report has text).
- **"Unused logical qubits act as gauge degrees of freedom" (formalizing.tex l.46)**: reword.
- **"Near-optimal inputs"**: qualify (2q structure unchanged on 41/69 circuits).
- **Depth vs time**: the fair baseline is fair on depth, not run time.
- **Candidates for new text**: planted optimality; concatenation; the Hadamard explanation of
  Fig 6; LSR vs TVD (GHZ / cat / W-state bars are the trade, not noise).
- [10-09] **Fig 9, the density ablation** (`evaluation.tex` ~l.227 and the caption ~l.281): "packing
  grows to dominate, with selection-rate improvements approaching 300% ... because at sparse
  densities the placement controls the movement and idling" and "the selection-rate improvements
  at sparse densities belong almost entirely to placement" are wrong: the submitted figure's
  packed arm had ceil(n/2) patches at every density, so the sparse bars were mostly fewer patches.
  Replacement [10-10]: the split-merging ablation under the logical-optimisation pass,
  `density_ablation_split_logical_71` (Fig 6's 71 circuits; section 9). What the data supports
  (means over circuits, densities 1.85 -> 1.04):
  - H commutation helps at every density: logical error rate 11-12% lower, TVD 14-17% lower,
    LSR +5..+8%.
  - Single-qubit merging (HH) helps at every density, more as patches empty: logical error 5% ->
    10% lower, LSR +7..+9%.
  - Two-qubit merging vanishes as density -> 1, because nothing shares a patch: logical error
    5.4% lower -> exactly 0, LSR +6.6% -> 0.
  - Packing vanishes the same way: logical error 15% lower -> 0.1%.
  - At equal patch counts the placement lowers the logical error rate 12-17% and TVD 11-17%
    (densities 1.85-1.14). Its LSR gain is +1% at the densest bin and +24..+30% at 1.26-1.14.
  - The densest bars add up to Fig 6's configuration: LSR +24.2%, against the new Fig 6's +24.0%.
  (Before the pass, the v3 numbers: H commutation 14-17%, 1q merging 10-12%.)
  Keep the tex's AXIS FLAG in mind: the x values are qubits per patch (n/P), not n/(kP).
- [10-09] **Placement and LSR**: do not claim placement raises LSR on every circuit. At the tightest
  packing it gives up LSR on GHZ / cat / W-state / Ising and gets a 17-45% lower logical error
  rate for it (the silent weight working as designed); describe that trade (section 9).
- [10-09] **Terminology**: "logical error rate" (post-selected: wrong outcome among kept shots) for
  what the code calls silent error per kept shot.
- [10-09] **Baseline descriptions**: if the text says the baseline has no merging, it is not quite
  so -- a solo qubit's H is applied as a whole-patch HH in every configuration. Either say so
  (only odd widths are affected at the tightest packing) or use the merging-free baseline
  (`solo_hh_promotion=False`), as the new density figure does.
- [10-09] **Planner description**: if the cost model is described as charging a qubit's single-qubit
  gates the same wherever it sits, add the solo-patch term (a solo qubit's Hadamards cost one
  whole-patch HH and leak no logical error) -- the standard prices it since 10-09.
- [10-09] **TVD estimability**: any claim or figure rule that TVD is only measurable at <= ~18 qubits
  should go: the floor depends on how many outcomes the ideal output has; 52/71 circuits are
  measurable at any width.

## 5. Open decisions

- `selection_weight` default: 10 (shots-leaning) or 20-50 (balanced).
- Adopt [[4,1,2]] decoding of solo patches (post-processing, small TVD gain)?
- Port the exact incremental evaluation (scaling) into `packing/search.py`?
- Tie-break-by-translation in the planner (fixes the one planted-circuit miss)?
- Under the transversal table, the merging pass prices CZT savings with the in-patch CZ cost
  (suppresses some CZT); kept as measured -- fixing it changes w10T's results.
- The planner's verify prices program-order translations while arms are layer-sorted (how the
  standard is defined; small effect seen on Fig 8).
- Housekeeping: superseded figure versions in `results/figures/`; the worktrees can be removed.
- [10-09] **Re-run the standard figures under `solo_hh_aware`** (odd widths change): Fig 6, Fig 8,
  the phase diagram, the selector reports -- or pin them to `STANDARD.with_(solo_hh_aware=False)`
  and state it.
- [10-09] **Recount silent error per kept shot on occupied lanes** for Fig 8 and the phase diagram,
  whose k > 2 configurations have partly empty patches (`density_references.split_by_lanes`;
  needs one Stim conversion per arm, no new shots).
- [10-09] **Fig 6's baseline**: keep and describe the solo-HH promotion, or re-run its baseline and
  +H arms merging-free (odd widths only).
- [10-09] The resource selector (`packing/resource/select.py`) still prices without `solo_hh`.
- [10-09] Merge `v2-exact-patches` into `v2-revision-experiments`: the full density run is in
  (10-09 19:36). Note that `v2-logical-opt-pass` branches from its head, 61ac6be.
- [10-09] **How the reference figures average** (`density_vs_naive_*`). v3 takes the mean of
  per-circuit percentages. Against all-solo, those bars are set by the few circuits whose all-solo
  LSR is ~10x below their tight LSR, and in five cells the placement bar changes sign between the
  two figures. A geometric-mean draft fixes both: log-ratio segments, with patch count x
  placement = net exactly for every circuit, so placement is the same bar in both figures. It is
  in `logs/density_matched/full_run_analysis/` (script + PDFs), not on the branch. Pick one.
  The ablation figure is unaffected, since every segment there is relative to the same baseline.
- [10-10] The density runs predated the logical-optimisation pass: re-run under it now (section 9).
- [10-10] Under the no-new-rotations rule, six of Fig 6's programs get no logical optimisation at
  all: variational_n4, qaoa_n6, dnn_n33, dnn_n51, qugan_n39 and qugan_n71. Every route that cuts
  their two-qubit gates adds rotations. A route limited to commutation and cancellation, with no
  1q resynthesis, might still help them; it would need a new route in the optimiser.
- [10-09] Provenance: five exploration switches change translations but are not in
  `TranslatorConfig.to_dict()`; config scopes only ever turn switches on.

## 6. Figure index

Paper-replacement candidates (project `results/figures/`):

| figure | shows | script | report |
|---|---|---|---|
| [ablation_standard.pdf](../results/figures/ablation_standard.pdf) | Fig 6 under the standard, every panel re-simulated | `experiments/figures/ablation_standard/` | fig6_standard |
| [resource_selection_headroom_standard.pdf](../results/figures/resource_selection_headroom_standard.pdf) | Fig 8 under the standard: shots and silent error per kept shot | `resource_selection_headroom --standard` | figures_standard |
| [1_best_k_phase_diagram_standard.pdf](../results/figures/1_best_k_phase_diagram_standard.pdf), [..._accuracy.pdf](../results/figures/1_best_k_phase_diagram_standard_accuracy.pdf) | best k for shots; most accurate k | `1_best_k_phase_diagram --standard` | figures_standard |
| [density_ablation_sweep_standard.pdf](../results/figures/density_ablation_sweep_standard.pdf) | the density ablation under the standard -- **superseded [10-09]**: packed arm at ceil(n/2) patches everywhere | `density_ablation_sweep --standard` | figures_standard |
| [10-10] [density_ablation_split_logical_71.pdf](../results/figures/density_ablation_split_logical_71.pdf) | **Fig 9 replacement** under the logical-optimisation pass (Fig 6's 71; `_logical` = all 72), shared simulations; same arms as v3 | `density_ablation_sweep --split-merging --share-identical`, `density_references plot` | section 9 |
| [10-09] [density_ablation_split_v3.pdf](../results/figures/density_ablation_split_v3.pdf) | superseded 10-10 (before the pass) -- Fig 9 replacement (full run, 71 circuits): merging-free baseline, H commutation / + 1q merging / + 2q merging / + Packing at equal patch counts, solo-aware planner | `density_ablation_sweep --split-merging`, `density_references plot` | section 9 |
| [10-09] [density_ablation_matched_v2.pdf](../results/figures/density_ablation_matched_v2.pdf) | the ablation at equal patch counts, one merging segment (old planner) | `--matched-patches`, `density_references plot` | section 9 |
| [10-10] [density_vs_naive_tight_logical_71.pdf](../results/figures/density_vs_naive_tight_logical_71.pdf), [density_vs_naive_sparse_logical_71.pdf](../results/figures/density_vs_naive_sparse_logical_71.pdf) | the reference figures under the pass (Fig 6's 71); averaging still open | `density_references plot` | section 9 |
| [10-09] [density_vs_naive_tight_v3.pdf](../results/figures/density_vs_naive_tight_v3.pdf), [density_vs_naive_sparse_v3.pdf](../results/figures/density_vs_naive_sparse_v3.pdf) | AQORN at each density against the naive tight / all-solo packing (+H +merge), split into patch count and placement (full run; averaging still open, section 5; geometric-mean drafts in `logs/density_matched/full_run_analysis/`) | `density_references plot` | section 9 |

Analyses (project `results/figures/`):

| figure | shows | report |
|---|---|---|
| [insights_mechanisms_v3.pdf](../results/figures/insights_mechanisms_v3.pdf) | where each stage's 1q / 2q savings come from | why_aqorn_works |
| [insights_error_paths.pdf](../results/figures/insights_error_paths.pdf) | LSR change vs detected errors; TVD change vs silent error per kept shot | why_aqorn_works |
| [insights_lsr_tvd.pdf](../results/figures/insights_lsr_tvd.pdf) | how LSR and error move together, by kind of change; the noise axis | lsr_vs_tvd |
| [selection_weight_v3.pdf](../results/figures/selection_weight_v3.pdf) | selector ranking weight: shots vs accuracy, against the measured frontier | selector_standard |
| [weight_sweep_v2.pdf](../results/figures/weight_sweep_v2.pdf) | silent weight W = 3..30 at k = 2 | weight_sweep |
| [ablation_fair_depth_v4.pdf](../results/figures/ablation_fair_depth_v4.pdf), [_w10](../results/figures/ablation_fair_depth_w10.pdf), [_w10T](../results/figures/ablation_fair_depth_w10T.pdf) | Fig 6 depth against the layer-sorted baseline | (fig6_standard) |
| [ablation_silent_aware_w10.pdf](../results/figures/ablation_silent_aware_w10.pdf), [_w3](../results/figures/ablation_silent_aware_w3.pdf); [silent_aware_vs_aqorn_w10.pdf](../results/figures/silent_aware_vs_aqorn_w10.pdf) | Fig 6 with the silent-aware planner (program order) | REVISION_RESULTS (Exp 8) |
| [optimality_gaps_v2.pdf](../results/figures/optimality_gaps_v2.pdf) | exhaustive and planted optimality | optimality |
| [logical_preoptimization_w10_v2.pdf](../results/figures/logical_preoptimization_w10_v2.pdf), [_w10T_v2](../results/figures/logical_preoptimization_w10T_v2.pdf) | AQORN on logically pre-optimized inputs | logical_opt |
| [compile_scaling_compact_w10i_vs_w10_random_v2.pdf](../results/figures/compile_scaling_compact_w10i_vs_w10_random_v2.pdf) (+ 6 more `compile_scaling_*_v2`) | compile time vs n and k | scaling |
| [concat_setup.pdf](../results/figures/concat_setup.pdf), [concat_c1.pdf](../results/figures/concat_c1.pdf), [concat_faults_v2.pdf](../results/figures/concat_faults_v2.pdf), [concat_results_v3.pdf](../results/figures/concat_results_v3.pdf) | concatenation setup, construction, fault analysis, results | concatenation |
| [10-10] [concat_gain.pdf](../results/figures/concat_gain.pdf), [concat_tradeoff.pdf](../results/figures/concat_tradeoff.pdf) | replace concat_results_v3: AQORN's gain at one and two levels + logical error vs p2 (A); shots vs logical error (C) | concat_standard (branch v2-concat-standard) |
| [10-10] [concat_correction.pdf](../results/figures/concat_correction.pdf) | error correction at two levels: mid-circuit inner syndrome rounds (FT) + first-order lookup decoder vs detection; discards and logical error vs p2 | concat_standard (correction section) |
| [movement_idle_v4.pdf](../results/figures/movement_idle_v4.pdf), [_w10](../results/figures/movement_idle_w10.pdf), [_w10T](../results/figures/movement_idle_w10T.pdf) | Fig 6 in atoms carried and idle time | movement_idle |
| [solo_gauge_paired.pdf](../results/figures/solo_gauge_paired.pdf), [_gadgets](../results/figures/solo_gauge_gadgets.pdf), [_availability_v2](../results/figures/solo_gauge_availability_v2.pdf), [_options](../results/figures/solo_gauge_options.pdf), [_density](../results/figures/solo_gauge_density.pdf), [_alpha](../results/figures/solo_gauge_alpha.pdf) | [[4,1,2]] decoding of solo patches | solo_gauge |

Oct 1 experiment figures (this checkout's `results/revision/`): [exp7a_regret_vs_exhaustive_v4.pdf](results/revision/figures/exp7a_regret_vs_exhaustive_v4.pdf),
[exp1c_packers_survival_vs_accuracy_v3.pdf](results/revision/figures/exp1c_packers_survival_vs_accuracy_v3.pdf),
[exp2b_fixed_noise_lsr_v4.pdf](results/revision/figures/exp2b_fixed_noise_lsr_v4.pdf),
[exp3b_original_vs_proxy_v4.pdf](results/revision/figures/exp3b_original_vs_proxy_v4.pdf),
[exp8_silent_cost_model_v4.pdf](results/revision/figures/exp8_silent_cost_model_v4.pdf),
[fig8_best_static_with_compiler_v2.pdf](results/revision/exp6/fig8_best_static_with_compiler_v2.pdf),
[compile_time.pdf](results/revision/exp7/compile_time.pdf).

Superseded (kept, never overwritten): `insights_attribution`, `insights_mechanisms` v1 / v2,
`selection_weight` v1 / v2, `weight_sweep` v1, un-versioned twins of every `_v2` / `_v3` above.
[10-09] Also `density_ablation_sweep_matched` (the sweep's own four-panel figure),
`density_ablation_matched` / `density_vs_naive_*` v1 (TVD cut at 18 qubits, "silent" label),
`density_ablation_split` v1 (old planner) and v2 (small group only), `density_vs_naive_*` v2
(matched run, old planner), and the untracked draft
`experiments/figures/density_ablation_sweep/density_ablation_matched.py` + `matched_draft/` in
this checkout.

## 7. Re-running

    PYTHONPATH=. venv/bin/python -m pytest -q                                              # 714 tests
    PYTHONPATH=. venv/bin/python -m experiments.figures.ablation_standard.ablation_standard --persist   # Fig 6, ~35 min
    PYTHONPATH=. nohup bash experiments/figures/regenerate_standard.sh > logs/regenerate_standard/RUN.txt 2>&1 &   # Fig 8 + phase + density, ~7 h
    PYTHONPATH=. venv/bin/python -m experiments.insights.census --jobs 12                 # ~25 min, then .structure / .figures
    PYTHONPATH=. venv/bin/python -m experiments.insights.lsr_tvd --alpha-sweep
    PYTHONPATH=. venv/bin/python -m experiments.revision.exp6_selector.selection_weight

    # [10-09] density ablation, on branch v2-exact-patches (run in .claude/worktrees/exact-patches)
    PYTHONPATH=. venv/bin/python -m experiments.figures.density_ablation_sweep.density_ablation_sweep --standard --matched-patches --reuse-arms <standard CSV> --jobs 12   # ~1.5 h
    PYTHONPATH=. venv/bin/python -m experiments.figures.density_ablation_sweep.density_ablation_sweep --standard --matched-patches --split-merging --reuse-arms <matched CSV> --simulate baseline h_only h_merge_1q packed --jobs 12   # ~6 h
    PYTHONPATH=. venv/bin/python -m experiments.figures.density_ablation_sweep.density_references {recount,floors,plot} --matched <CSV>   # recount ~1.5-2 h, the rest minutes
    bash logs/density_matched/full_split_std.sh                                          # the 10-09 full run, end to end, 8.5 h

Every figure script versions its outputs (`_v2`, ...) and never overwrites a CSV or figure.
Traps worth remembering: plan on originals, simulate Clifford proxies; Fig 6's alpha is extreme on
3-4 qubit circuits; `load_fig6_circuits(names=...)` can load a different program for
basis_trotter_n4; killing a parallel run leaves its spawn workers alive. [10-09] TVD's floor is
the ideal output's support, not the width -- test each circuit (`density_references.support_bits`)
instead of cutting at 18 qubits; silent error per kept shot counted empty lanes until 10-09
(recount with `density_references`); a solo qubit's H is a whole-patch HH in every arm unless
`solo_hh_promotion=False`; `--output-dir` puts figures where you say, so pass the project's
`../results/figures`.

---

## 8. Shortlist: code and experiments from your own requests

Grouped by the message that asked for it. Paths are in `iceberg-compiler/`.

**"Run overnight on their own branches" (optimality, scaling, concatenation, movement / idle)**
1. Logical pre-optimization of the Fig 6 inputs -- `experiments/logical_opt/` -> [reports/logical_opt.md](reports/logical_opt.md). Inputs near-optimal in 2q structure on 41/69; AQORN's depth gain holds, its gate and fidelity gains shrink.
2. Exhaustive optimality on 4-6 qubit programs -- `experiments/optimality/exhaustive_silent.py`, `exhaustive_frames.py` -> [reports/optimality.md](reports/optimality.md). 0.6-2.2% above the optimum.
3. Constructed circuits with a known optimal packing -- `experiments/optimality/planted.py`. 100% of planted pairs recovered, exactly on the bound.
4. Compile time vs qubits and k -- `experiments/scaling/compile_scaling.py`, plus an exact faster evaluation `incremental.py` (opt-in, not ported) -> [reports/scaling.md](reports/scaling.md).
5. Concatenated [[4,2,2]]-in-[[4,2,2]] ([[16,4,4]], your choice), spread inner placement, AQORN at both levels, both decoders (your choice) -- `experiments/concatenation/`, `tests/test_concatenation.py` -> [reports/concatenation.md](reports/concatenation.md) (extended, with setup figures).
6. Fig 6 for movement and idle time, under v4 / w10 / w10T (your choice) -- `experiments/figures/movement_idle/` -> [reports/movement_idle.md](reports/movement_idle.md).

**"Add a [[4,1,2]] mode ... solely if a patch is solo"**
7. [[4,1,2]] decoding of solo patches -- decoder `experiments/solo_gauge/gauge.py` (+ gadgets, circuit-level, options scripts), opt-in switches `SOLO_H_TARGETED` / `SOLO_GAUGE_PARITY_FIX` (`translator/explore_switches.py`, `translator/handlers/hadamard.py`, `translator/core.py`), `tests/test_solo_gauge.py` -> [reports/solo_gauge.md](reports/solo_gauge.md). Not on by default.

**"w10T should be the standard ... a more extended search for weights"**
8. Silent-weight sweep W = 3..30 with the transversal gadget -- `experiments/tvd_explore/weight_sweep.py`, `weight_sweep_run.sh` -> [reports/weight_sweep.md](reports/weight_sweep.md). Kept 10.

**"Go ahead with the merge, w10T as standard"**
9. The six branches merged; `v2-revision-experiments` fast-forwarded (after your "continue").
10. Transversal in-patch CZ / XCX as a real option -- `TranslatorConfig.transversal_intra_two_qubit`, `iceberg/gadget_options.py`, `packing/passes.py` (cache key).
11. The silent-aware planner in the library -- `packing/silent.py` + rate tables `packing/data/gadget_rates*.csv`.
12. The standard configuration -- `packing/standard.py` (`STANDARD`, `plan_standard`); `compile_circuit` uses it by default (`planner="v4"` = submitted); `tests/test_standard.py`.

**"Continue" (to my proposed next step)**
13. Fig 6 under the standard, every panel re-simulated -- `experiments/figures/ablation_standard/` -> [reports/fig6_standard.md](reports/fig6_standard.md), tag `v2-fig6-standard`.

**"Check the quality of the resource selector ... improve it?"; your answers "sweep k = 4 first", "focus on what the standard objective is doing"**
14. Standard resource selector, one objective in every stage -- `packing/compiler.py` (`_compile_standard`), `packing/standard.py` (`standard_objective`), `simulator/attribution.py` (`clifford_stand_in`); v4-contract tests pinned in `tests/test_packing_compiler.py`.
15. k = 4 weight sweep -- `experiments/tvd_explore/weight_sweep_k4.py` -> planner weight 5 above k = 2 (`AqornStandard.wide_silent_weight`).
16. End-to-end selector validation -- `experiments/revision/exp6_selector/standard_selector.py` -> [reports/selector_standard.md](reports/selector_standard.md).

**"Set them up to run overnight under the standard"; your answer "yes, add [silent error per kept shot]"**
17. `--standard` modes with frame-measured silent error per kept shot in `experiments/figures/{1_best_k_phase_diagram, resource_selection_headroom, density_ablation_sweep}/`, shared switches in `experiments/shared/simulation.py`, queue `experiments/figures/regenerate_standard.sh` -> [reports/figures_standard.md](reports/figures_standard.md). (The two figures commented out of the paper were not run.)

**"Why AQORN performs well on specific circuits"; "the relation between LSR and TVD"**
18. Gadget census and attribution -- `experiments/insights/census.py`, `structure.py`, `figures.py` -> [reports/why_aqorn_works.md](reports/why_aqorn_works.md).
19. Paired LSR / TVD comparisons and the noise sweep -- `experiments/insights/lsr_tvd.py` -> [reports/lsr_vs_tvd.md](reports/lsr_vs_tvd.md).

**"Add the selector ranking-weight knob and test it"**
20. `AqornStandard.selection_weight` / `ranking_weight()` (default = 10, unchanged), tests, and the sweep on Fig 8's candidates `experiments/revision/exp6_selector/selection_weight.py` -> section in [reports/selector_standard.md](reports/selector_standard.md).

**[10-09] "Why does packing perform so well near density 1?" and what followed (branch v2-exact-patches)**
21. The confound traced: the Aug 1 regeneration swapped the submitted packer (`MergeMaximizingPacker(num_codes=n_codes)`) for one that always packs tightest (investigation; section 9).
22. "The MLP can target a patch count" -> `plan_standard(patches=P)`: exact partition, count-preserving descent, verify at P (1ac0902; `tests/test_exact_patches.py`).
23. The matched-patches sweep with arm reuse, run overnight (e939d37).
24. "Go with your recommendations, do both metric fixes": occupied-lane logical error, TVD floors, the vs-tight / vs-sparse figures (3c9c02f).
25. "Use logical error"; "why is large missing from TVD?" -> label, per-circuit TVD floor test (e0f8880).
26. "Is the baseline merging-free?"; "split merging, small circuits first" -> `solo_hh_promotion`, `merge_two_qubit` (f90005d), `--split-merging`, `--groups` (eca44a8).
27. "Check the translator's config handling (read only)" -> audit, section 9.
28. "Fix the planner, see small first" -> `solo_hh` pricing (c1f9b00), `--solo-aware-planner`, `--simulate` (2bc97b0); "make solo_hh_aware default on" (61ac6be).
29. The full split run under the new default, 10-09 11:03-19:36 (`logs/density_matched/full_split_std.sh`): figures v3, section 9.

**Earlier decisions of yours that my notes record (before this conversation's span)**
- The Oct 1 revision experiments, including Experiment 8 (the silent-error cost model, your follow-up to 1c) -- `experiments/revision/` -> [REVISION_RESULTS.md](REVISION_RESULTS.md).
- Weight 10 for the silent-aware planner -- Fig 6 at weights 3 and 10, `experiments/figures/ablation_silent_aware/`, `silent_aware_vs_aqorn/`.
- Your "transversal + undo" idea -- majority + undo merges (`HH_MAJORITY_UNDO`, `TRANSVERSAL_MAJORITY_UNDO`, opt-in); cannot fire at k = 2.
- The fair baseline -- `TranslatorConfig.asap_order`, `tests/test_asap_order.py`, `experiments/figures/ablation_fair_depth/`.

---

## 9. Density ablation and what it uncovered (2026-10-08/09)

Branch `v2-exact-patches`, worktree `.claude/worktrees/exact-patches`, commits 1ac0902 ..
61ac6be on `e84dc85` (not merged, not pushed; 822 tests). Paper text implications are in section 4,
open items in section 5.

**The confound.** At submission (`submitted-v4`, Jul 31) the sweep's packed arm was
`MergeMaximizingPacker(num_codes=n_codes)`, at the trivial arm's patch count. The Aug 1
regeneration -- the tree behind the submitted PDF, committed later as 9f828aa -- swapped it for
`pack_program(ceil(n/n_codes) + 2)`, which at k = 2 always packs to ceil(n/2) patches. Below
density 2, "+ Packing" was mostly the patch-count difference. The `--standard` rerun inherited
it. The paper's Fig 9 text is built on it (section 4).

**Fixes, in commit order.**

| commit | what |
|---|---|
| 1ac0902 | `plan_standard(..., patches=P)`: `partition(exact=True)`, `descend(fixed_count=True)`, naive packing and verify at P; `patches=tightest` is the default plan (pinned) |
| e939d37 | `--matched-patches` (packed arm at each point's count), `--reuse-arms` (copy the unchanged arms) |
| 3c9c02f | `density_references.py`: occupied-lane recount, TVD floors, the vs-tight / vs-sparse figures |
| e0f8880 | "logical error rate" label; per-circuit TVD floor test by stabilizer rank |
| f90005d | `TranslatorConfig.solo_hh_promotion`, `merge_two_qubit` |
| eca44a8 | `--split-merging` (merging-free baseline; + 1q merging / + 2q merging), `--groups` |
| c1f9b00 | planner prices solo patches' Hadamards (`solo_hh`) |
| 2bc97b0 | `--solo-aware-planner`, `--simulate` |
| 61ac6be | `AqornStandard.solo_hh_aware` on by default; arm reuse per arm and circuit (`copied_arms`) |

**Measurement fixes.**
- *Logical error per kept shot* counted a flip on every lane, including lanes no program qubit
  occupies (frames.py), so it read high wherever patches have empty lanes, and by different
  amounts per arm (sat_n7 at 5 patches: 97% of the baseline's count on occupied lanes, 81% of
  AQORN's). The recount reruns each arm with one shot to rebuild its Stim circuit and counts
  occupied lanes only; its all-lane count reproduces the sweep exactly (1248/1248 matched,
  405/405 split). It resolved a contradiction: AQORN-tight vs all-solo looked tied on silent
  error (-0.1%) but -18% on TVD; on occupied lanes it is -34%.
- *TVD's floor* is set by how many outcomes the ideal output has. The simulated programs are
  Clifford, so the support is exact from the GF(2) rank of a noiseless sample at any width (it
  equals the state vector's on all 552 checked points). 52/71 circuits have 1-2 outcomes; the ten
  with uniform outputs read their floor and are excluded. TVD panels went from 11-31 to 40-61
  circuits per bar.

**Results before the full run** (means over circuits; logical error on occupied lanes; TVD above
its floor). The full run below supersedes the split-ablation and reference numbers.

- Matched ablation, "+ Packing" (placement alone; old planner), densities 1.85 -> 1.14: LSR
  +6..+17%, TVD +9..+14%, logical error +8..+15%; about 0 at one qubit per patch.
- AQORN vs the naive tight packing (+H +merge), 1.85 -> 1.14: LSR +4.5% -> -55% (-67% all-solo),
  TVD +16..+22%, logical error +18..+27%: spreading out costs shots and buys accuracy.
- AQORN vs naive all-solo (+H +merge), 1.85 -> 1.14: LSR +330% -> +54%, TVD -26% -> -2%, logical
  error -38% -> -11%: packing densely buys shots and halves the qubits and costs accuracy;
  placement wins back +31..+39 points of logical error at dense packings.
- The merging-free baseline: at one qubit per patch the old ("merging off") baseline was 18%
  cheaper in 1q gates and 12% in 2q gates (all 71 circuits, static) because of the solo-H
  promotion; merging-free it has +33% 1q, +19% 2q, +27% depth (small group).
- Split ablation, small group, solo-aware planner, logical error, density 1.85 -> 1.04: H
  commutation +13% -> +18%, + 1q merging +9% -> +10%, + 2q merging +6% -> +0.1%, + Packing +11% ->
  -0.5%. LSR tells the same story (2q merging +10.5% -> +0.3%).
- The planner fix, small group: 10 of 52 interior placements change; verify falls back to naive
  less often (density 1.0-1.15: 20% -> 0%); "+ Packing" on logical error +1..+2 points at sparse
  densities. Under the new default no small circuit's commutation decision changed.

**Translator config audit (read-only, 10-09).** Only the solo-H blind spot changed results (fixed
above). Also: config scopes (`transversal_intra`, `solo_h_targeted`, `asap_order` with
`LAYER_ORDER_ALL_ARMS`) only turn process-wide switches on, so a config `False` loses to a global
`True`; five exploration switches are not in `TranslatorConfig.to_dict()`; `enable_h_commutation=
True` with `h_commutation_stage="never"` runs none; every arm gets inverse-gate cancellation and
three peephole passes; the planner's verify and `commutation_pays` translate in program order;
`fault_tolerance_chance` is read by nothing; `verbose` writes into `fault_log` (cosmetic). Verified
fixed in code: `PipelineConfig(fault_tolerant=True)`, `contains_faults`, the Stim endianness bug.

**Files.** CSVs beside the worktree's script
(`experiments/figures/density_ablation_sweep/density_ablation_sweep_{matched, split, split_solo,
split_std_small, split_std}*.csv`, with `_recount` / `_floors` / `_summary` companions), copied to
`../results/data/`. Figures copied to `../results/figures/` (the runs wrote into this checkout's
`results/figures/`). Logs: `logs/density_matched/`.

**The full run** (10-09 11:03-19:36, 8.5 h; `logs/density_matched/full_split_std.sh`, status
`FULL_SPLIT_STD.txt`). Medium + large were simulated under the new default: baseline, +H,
+ 1q merging, packed. The + merge arm was copied from the matched run for 41 circuits, where the
commutation decision agrees. They were combined with the small group into
`density_ablation_sweep_split_std.csv` (71 circuits, 1392 rows).
- Checks: the recount (2 h) matches the sweep's all-lane count on 1560/1560 arm-points. TVD is at
  least 2x its floor at every point on 61/71 circuits.
- Outputs: `density_ablation_split_v3`, `density_vs_naive_{tight,sparse}_v3`, copied to
  `../results/figures/`; CSVs copied to `../results/data/`.
- Precondition: the run predates the logical-optimisation pass (section 5).

- Split ablation: means of per-circuit percentages, densities 1.85 -> 1.04. Bars have 47-71
  circuits each (TVD 40-61).

  | segment | LSR | TVD reduction | logical error reduction |
  |---|---|---|---|
  | H commutation | +6..+14% | 15-21% | 14-17% |
  | + 1q merging | +15..+19% | 7-9% | 10-12% |
  | + 2q merging | +7.4% -> +0.2% | 4.5% -> -0.4% | 3.9% -> 1.0% (1.14), noise at 1.04 |
  | + Packing | +5.9% (1.85), peaks +20.8% (1.26), +0.4% (1.04) | 13.7% -> 1.5% | 15.3% -> -1.8% |

  At one qubit per patch, 74/76 points translate identically with and without 2q merging. The
  +2.2-point logical-error residual there is seed noise (s.e. 1.1). Where 2q merging does change
  the translation (about two-thirds of the points at 1.14-1.85), it lowers the logical error rate
  2-6%.
- AQORN vs the naive tight packing (+H +merge), geometric means:
  - net LSR x1.00 (1.85) -> x0.39 (1.14) -> x0.29 (1.04);
  - logical error rate 1.25x -> 1.52x -> 1.46x lower;
  - TVD 1.22x -> 1.38x -> 1.39x lower.
  v3 shows arithmetic means: LSR +4% -> -55% -> -67%, logical error 18-29% lower, TVD 16-22% lower.
- AQORN vs naive all-solo, geometric means:
  - net LSR x3.79 (1.85) -> x1.48 (1.14) -> x1.01;
  - logical error rate 1.22x -> 1.05x -> 1.02x higher;
  - TVD 1.12x -> 1.01x higher.
  v3's arithmetic means: LSR +328% -> +53%, logical error 37% -> 8% higher, TVD 27% -> 3% higher.
- Placement alone (packed vs naive at the same P; the same quantity in both reference figures as
  a geometric mean):
  - LSR x1.01 (1.85), x1.13 (1.26), x1.00 (1.04);
  - logical error rate 1.25x lower at 1.85, 1.15x at 1.14, 0.98x at 1.04;
  - TVD 1.21x -> 1.13x -> 1.02x lower.

**LSR at the tightest packing is a per-family trade.** Placement here means packed vs naive at
the same P, at density ~1.85 (`full_run_analysis/placement_by_family.py`).
- GHZ / cat / W-state / Ising: LSR -11..-34%, logical error rate 17-45% lower. Same direction for
  TVD; GHZ / cat depth falls ~40%. This is the silent weight working as designed.
- VQE / variational / QAOA: LSR -12..-17% for only a 2-5% lower logical error rate, a poor trade
  worth a look.
- qugan / multiplier / swap_test / bigadder / multiply / knn / qram / dnn: LSR +20..+89%, logical
  error flat to 37% lower.
- Large group: 11/27 circuits gain LSR; the median is -0.9%; the mean is +5.6% (vs tight).

**Averaging (open, section 5).** The v3 reference figures average per-circuit percentages of a
fixed reference, so placement's bar is 100 (packed - naive at P) / reference.
- Against all-solo, the wide GHZ / cat / W-state circuits have a reference LSR ~10x below their
  tight one. Their losses get amplified about tenfold: the large group's 1.85 placement bar reads
  -24.6% vs sparse and +5.6% vs tight. Five cells change sign between the two figures, and the
  medians agree in every one.
- The draft in `logs/density_matched/full_run_analysis/` (`geomean_refs.py`,
  `density_vs_naive_{tight,sparse}_geomean.pdf`) stacks ln-ratios instead: geometric means on a
  log factor axis, net = patch count x placement exactly for every circuit. No point is dropped;
  every value is positive.

**10-10: the re-run under the logical-optimisation pass.** You chose to re-run the figure on
logically optimised inputs, as Fig 6 now uses ("option 2"). The aim, in your words: "logical
optimization we can optimize over again using AQORN".
- **The rule** (1f83c4f, `v2-logical-opt-pass`): no program comes back with more non-Clifford
  rotations than it had. It is applied inside the optimiser's search
  (`logical_optimizer.optimize(max_rotations=)`): neither chain steps to a circuit with more, so
  the rotation-free routes (commutation, CNOT-block resynthesis) keep going. Records written
  before the rule that add rotations are refused when read, and `seed_record` will not write one.
  No `OPTIMIZER_VERSION` bump: the rule only removes answers. 844 tests pass.
- **Fig 6's records.** 7 of the study's answers added rotations, and were re-seeded:
  - basis_trotter_n4 now goes 582 -> 199 two-qubit gates and 356 -> 232 rotations. The study's
    answer was 217 and 461.
  - variational_n4, qaoa_n6, dnn_n33, dnn_n51, qugan_n39, qugan_n71 have no improvement without
    more rotations, so they are used as they are (open item in section 5).
  - The loader check passes for 71/71.
  - A fallback-only version of the rule left all 7 unchanged, because both chains add rotations.
    That is why the rule has to be inside the search.
- **Fig 6 under the rule** ran 10-10 01:37-02:25 (`ablation_standard_logical_v2` in
  `../results/figures/`, CSV in `../results/data/`; report addendum in the pass branch's
  `reports/fig6_logical.md`, 5688b3f). AQORN, cumulative over 71 circuits:

  | | 1q | 2q | depth | LSR | TVD |
  |---|---|---|---|---|---|
  | under the rule | 10.4 | 14.0 | 18.3 | +24.0 | 33.3 (61 circuits) |
  | before the rule | 8.2 | 13.2 | 17.3 | +22.3 | 32.0 (59) |
  | earlier standard, no pass | 13.3 | 19.3 | 16.4 | +43.2 | 40.7 (66) |

  TVD on the seven proxies: dnn_n33 and dnn_n51 are resolvable again; swap_test x3 still
  saturate. qugan_n39/n71 saturate too, because the pass names their Clifford-angle rx as sx / x
  and the proxy keeps them -- not because of rotations.
- **Shared simulations** (697ef6a, `v2-density-logical`, `--share-identical`): an arm that
  compiles to exactly an earlier arm's circuit at the same point takes that arm's metrics. Matching
  means the same physical circuit, placement and lane each logical state is read from. A step that
  changes nothing then shows exactly zero instead of seed noise (the 10-09 run's +2.2 for 2q
  merging at one qubit per patch). Rows name the shared arms (`shared_arms`), and the recount
  draws a shared arm as its source.
  - `--split-merging` no longer needs `--reuse-arms`.
  - Rows record the pass (`logical_optimization`), and `--reuse-arms` refuses a CSV made under
    another setting.
  - Pass-run figures are named `*_logical`.
  - 6 new tests; 849 pass.
  - Smoke run on adder_n4, basis_trotter_n4 and qaoa_n6, end to end: the recount matched 35/35,
    shared arms included.
- **The run** (`logs/density_logical/full_split_logical.sh`, status `FULL_SPLIT_LOGICAL.txt`)
  ran 10-10 02:25-08:33: sweep 4 h 44 min, recount 1 h 23 min. It used v2-density-logical
  697ef6a, clean, pass verified-routes-2.
  - Inputs are Fig 6's exactly: the run read the same records and wrote no new one. On
    basis_trotter_n4 the baseline's 2q gates go 5148 -> 1642.
  - Shared simulations: 567 of the 1416 later-arm points (40%).
  - Recount: the all-lane count equals the sweep's on 1590/1590, shared arms included.
  - Floors: TVD is at least 2x its floor everywhere on 52/72 circuits (61/71 before the pass; its
    Clifford naming changes some proxies, as for qugan in Fig 6).
- **72 circuits, not 71**: gcm_h6 (medium, 13 qubits) joins. Optimised, its proxy has 1801 gates,
  under the sweep's 2500 cap; its record dates from 10-09 21:30 and obeys the rule. It sits at
  medium index 4, so the 11 medium circuits after it got new seeds (no effect: nothing is reused).
  - `*_logical` figures: all 72.
  - `*_logical_71` figures: Fig 6's 71, from `density_ablation_sweep_split_logical_71.csv`, plotted
    with the full run's recount and floors. Use these.
  - The two differ by at most 0.6 points.
- **Results, Fig 6's 71** (means; logical error on occupied lanes; TVD above its floor;
  densities 1.85 -> 1.04):

  | segment | logical error reduction | LSR | TVD reduction |
  |---|---|---|---|
  | H commutation | 12.0% -> 11.0% | +8.4% -> +8.1% | 16.9% -> 13.6% |
  | + 1q merging | 5.0% -> 10.0% | +8.0% -> +9.4% | 1.1% -> 4.2% |
  | + 2q merging | 5.4% -> 0.0% (1.14: -0.5) | +6.6% -> 0.0% | 5.4% -> 0.0% |
  | + Packing | 14.9%, 17.3% (1.60) -> 0.1% | +1.2%, +30.1% (1.14) -> +0.4% | 14.1% -> 0.2% |

  Against the v3 run (before the pass), 1q merging shrank most: LSR +15..+19% -> +7..+9%, TVD
  7-9% -> 1-4%. The optimiser now does part of what the HH / XX / ZZ macros did. H commutation
  shrank a little (logical error 14-17% -> 11-12%). Packing's logical-error gain is unchanged.
  Reference figures (arithmetic means, as plotted):
  - vs tight: net logical error 18% lower (1.85) -> 20% (1.04, peak 27% at 1.14); LSR +2% -> -69%.
  - vs all-solo: net logical error 21% higher -> 0; LSR +417% -> +1%.
  The averaging question for those two figures is still open (section 5).
- **Worktree trap:** QASMBench is a git submodule, so a new worktree gets an empty
  `benchmarks/qasmbench`. Copy it from this checkout (`cp -Rp`, then drop the `.git` file). The
  pass's loader test skips silently without it, and a sweep would load nothing.
