# Branch summary: v2-tvd-explore and the overnight branches (2026-10-07)

Lineage: `v2-revision-experiments` (1a5359f) → **`v2-tvd-explore`** (7 commits, e5d5741)
→ six overnight branches, each forked from e5d5741. Data (CSVs) is gitignored and lives only in
each worktree's `results/`; figures are in the project's `results/figures/`.

| branch | files | lines | touches library? | take |
|---|---|---|---|---|
| v2-tvd-explore | 30 | +2808 −26 | yes (opt-in) | `asap_order` + tests, fair-depth figure, fastframes; w10T switch only if adopted (promote to config); leave rejected switches |
| v2-optimality | 7 | +1231 | no | all |
| v2-logical-opt | 4 | +2044 | no | all |
| v2-scaling | 4 | +2267 | no (monkeypatch in its own process) | all; incremental evaluation is the promotion candidate |
| v2-concatenation | 18 | +2031 | no (adds a test) | all |
| v2-movement-idle | 2 | +1286 | no | all |
| v2-solo-gauge | 11 | +2806 −3 | yes (opt-in) | decoder if [[4,1,2]] is adopted; solo-H switches optional |

---

## v2-tvd-explore (vs 1a5359f)

**Library, all default off**

| file | added / changed |
|---|---|
| `translator/config.py` | `TranslatorConfig.asap_order: bool = False` (+ `to_dict`) |
| `translator/core.py` | `_layer_sorted(circuit)`; `Translator.translate` layer-sorts the program when `asap_order` (or `LAYER_ORDER_ALL_ARMS`) is on |
| `translator/explore_switches.py` (new) | module-level switches `H_AS_S_SX_S`, `TRANSVERSAL_INTRA_TWO_QUBIT`, `HH_MAJORITY_UNDO`, `TRANSVERSAL_MAJORITY_UNDO`, `MERGE_KEEP_PROGRAM_ORDER`, `LAYER_ORDER_ALL_ARMS`; `configure(**switches)` |
| `iceberg/iceberg_module.py` | in-patch CZ / XCX at k = 2 as S⊗4 / SX⊗4 + Pauli corrections (`TRANSVERSAL_INTRA_TWO_QUBIT`) |
| `iceberg/costs.py` | `_transversal_intra_work(n)`, `_transversal_intra_depth(n)`: cost-table entries under that switch |
| `translator/handlers/hadamard.py` | single-lane H as S·SX·S (`H_AS_S_SX_S`) |
| `translator/merging/models.py` | `MacroReplacement.undo_ops` |
| `translator/merging/single_qubit.py` | `_majority_undo_applies`, `_build_undo_record` (HH on a majority + undo H) |
| `translator/merging/transversal.py` | `allow_undo`, `_complete_alignment`, `_single_gate`, `_is_real_swap` (CXT/CZT/XCXT on a majority + undo) |
| `translator/merging/rewrite.py` | emits `undo_ops`; per-op emission keys (`_emit`); `_relinearize` (`MERGE_KEEP_PROGRAM_ORDER`) |
| `translator/merging/{conflicts,preprocessing,selection}.py` | merge savings net of undo gates |
| `tests/test_asap_order.py` (new) | 15 tests: default off, exact logical action n = 4/6/8, same DAG in layer order, plumbing, a shortening case |

**Experiments**
- `experiments/figures/ablation_fair_depth/ablation_fair_depth.py`: Fig 6 with the depth panel against the layer-sorted baseline (`_plans`, `_run_circuit` with the reproduction gate, `compute`, `plot`, `summarize`).
- `experiments/tvd_explore/`: `screen.py` (frame screening of variants), `plan_variant.py` (planning under switches), `ablation_variant.py` (Fig 6 under switches: the w10T figure), `budget.py`, `compose.py`, `exhaustive_small.py`, `fastframes.py` (`at_alpha_one`, `rescale`: convert once, rescale exactly), `check_noiseless.py` (miscompile check), `check_majority_undo.py`, `code_size_undo.py`, `code_size_undo_sim.py`, `order_effect.py`, `order_effect_frames.py`.
- `experiments/figures/silent_aware_vs_aqorn/silent_aware_vs_aqorn.py`: label options.

**Take:**
- **Keep:** `asap_order` + its tests, the fair-depth figure, `fastframes` (used by everything after it), `check_noiseless`.
- **Promote if w10T is adopted:** `TRANSVERSAL_INTRA_TWO_QUBIT` → a `TranslatorConfig` field. As a global it is unsafe: the commutation-decision cache is not keyed on it.
- **Leave on this branch:** `H_AS_S_SX_S`, `MERGE_KEEP_PROGRAM_ORDER`, both majority-undo paths (rejected, or no effect at k = 2), `LAYER_ORDER_ALL_ARMS` (duplicates `asap_order`); the screening and order scripts as a record.

---

## v2-optimality (vs e5d5741) — experiment only

| file | objects |
|---|---|
| `experiments/optimality/exhaustive_silent.py` | `Scorer` (the weight-10 planner's objective, recomputed exactly per placement), `fair_metrics` (`asap_order` translation counts), `switched` (w10T switch + cache clear), `_run_point` (complete tightest space × both commutation settings), `summarize` |
| `experiments/optimality/exhaustive_frames.py` | `_encoded`, `_frames` (Pauli-frame LSR / silent), two-stage finalists against the winner's curse |
| `experiments/optimality/planted.py` | `planted_circuit` (hidden HH/CXT pairs, scrambled labels, decoys), `pairs_recovered`, `logical_bound` (per-gate minimum work), `_run_point` (arms + exhaustive check for n ≤ 6) |
| `experiments/optimality/optimality_gaps/optimality_gaps.py` | the three-panel figure |
| `reports/optimality.md` | report |

**Take:** all.

## v2-logical-opt — experiment only

| file | objects |
|---|---|
| `experiments/logical_opt/optimizer.py` | routes `route_o3`, `route_o3_sx`, `route_clifford_ag`, `route_clifford_greedy`, `route_clifford_blocks`, `route_linear_blocks`, `route_commute`; `optimize` (best route, O3 iterated to a fixed point), `optimize_verified`, `check_equivalence` (operator / tableau / MPS and statevector miters), `Optimized`, `proxy_of` |
| `experiments/logical_opt/logical_opt.py` | `stage_a` (logical reduction), `stage_b` (Fig 6 static pipeline on both inputs, fair depth), `stage_c` (simulation at Fig 6's alpha and seeds), `combine` (optimizer + AQORN), `_boot_ci`, `plot` |
| `reports/logical_opt.md` | report |

**Take:** all.

## v2-scaling — experiment only

| file | objects |
|---|---|
| `experiments/scaling/compile_scaling.py` | workloads (`build_program`, `qb_program`), `make_planner` (w10 / w10i / v4 / w10T), `run_cell` (one timed cell: CPU and wall time, counts, RSS, speed probe), `CapExceeded`, `schedule` / `run_parallel` (non-timing pool), `fit_exponents`, `finalize`, five plot functions |
| `experiments/scaling/incremental.py` | `_State`, `install()` / `uninstall()`: exact incremental re-evaluation of `movement_hops` and `busiest_patch_depth`, patched into `packing.search.Packing.value_after` only inside the measuring process |
| `reports/scaling.md` | report |

Data includes new silent-rate tables for n_code 10 and 12 (`results/scaling/rates/`).

**Take:** all. **Promotion candidate:** the incremental evaluation into `packing/search.py`, with the bit-identity test (63 cells today). 11× faster at 128 qubits, k = 2, with no change to any result.

## v2-concatenation — experiment + one test file

| file | objects |
|---|---|
| `construct.py` | `Flattened`, `TwoLevel`, `outer_translate`, `flatten` (C1 SWAPs → relabels, measurements deferred, both checked), `is_spread`, `inner_layouts` (stride / role_aligned / index), `inner_translate`, `compile_two_level` |
| `decode.py` | `Maps`, `maps_for`, `decode` (post-select at both levels; inner flags as erasures), `SingleMaps`, `decode_single` |
| `inner_plan.py` | `Scorer`, `descend` (spread-preserving swaps), `plan_inner` (AQORN's model + lanes + verify over spread placements) |
| `noise.py` | `rescale_text` (exact alpha rescale on Stim text, 235× faster than `fastframes.rescale`), `check` |
| `concat_eval.py` | `compile_arm` (11 arms), `_calibrate`, `_measure` (frames + samples), `_run_circuit` (streams results) |
| `distance.py` | `deterministic_program`, `annotated` (checks as DETECTORs, outputs as observables), `analyse` (first-order silent mechanisms, distance search) |
| `summary.py`, `concat_setup/`, `concat_c1/`, `concat_faults/`, `concat_results/` | report tables and figures |
| `tests/test_concatenation.py` | 7 tests: spread claims, noiseless round trip, single flipped bit discarded / corrected |
| `reports/concatenation.md` | extended report |

**Take:** all. `noise.rescale_text` could replace `fastframes.rescale` everywhere.

## v2-movement-idle — experiment only

| file | objects |
|---|---|
| `experiments/figures/movement_idle/movement_idle.py` | `schedule_amounts` (atoms carried, flight, transport layers, idle by cause, makespan, from the schedule `measured_noise_split` prices), `with_move_overhead` (handling-time sensitivity), `expected_errors`, `_check_reproduction`, caption / stage / aggregate tables, Fig 6-style plot with hatched bars for circuits that got worse |
| `reports/movement_idle.md` | report |

**Take:** all.

## v2-solo-gauge — library-touching, opt-in

| file | objects |
|---|---|
| `translator/explore_switches.py` | `SOLO_H_TARGETED` (solo H as the targeted gadget), `SOLO_GAUGE_PARITY_FIX` (fix-up H on the spare after an odd number of HH) |
| `translator/handlers/hadamard.py` | honours both switches; counts HH per solo patch |
| `translator/core.py` | `_restore_solo_gauge_frames(context)` |
| `experiments/solo_gauge/gauge.py` | `GaugeAnalysis` / `GaugeCheck` (check validity by back-propagating the logical Clifford, `left_kernel` over GF(2)), `analyse_translation`, `decode_samples` ([[4,2,2]] vs [[4,1,2]] on the same samples), `frame_split`, `noiseless_gate`, `randomized_input_gate` |
| `experiments/solo_gauge/gadgets_412.py` | exact single-fault enumeration with the gauge check |
| `experiments/solo_gauge/circuit_level.py` | paired runs on the Fig 6 corpus, `_paired_bootstrap` |
| `experiments/solo_gauge/options.py`, `alpha_one.py` | forced-check variants, density panel, alpha = 1 frames |
| `tests/test_solo_gauge.py` | 19 tests |
| `reports/solo_gauge.md` | report |

**Take:** the decoder (post-processing; nothing changes unless you call it) if [[4,1,2]] decoding is adopted. The two solo-H switches measured as not paying; carry them only as opt-in, or leave them here.
