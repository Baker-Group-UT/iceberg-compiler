# The paper's planner-dependent figures under the standard, 2026-10-07/08

Overnight run (`experiments/figures/regenerate_standard.sh`, from clean commit 57779f0, 19:08-01:56):
the three figures `evaluation.tex` includes, each in its `--standard` mode -- the standard planner
(weight 10 at k = 2 with the transversal in-patch CZ / XCX, weight 5 above), every translation
layer-sorted, the standard resource selector, and Pauli frames on every simulated arm for the
silent logical error per kept shot (the accuracy measure that works where TVD saturates; TVD ~
0.95 x it, rank correlation 0.91-0.98 where both resolve). Fig 6 is separate
(`reports/fig6_standard.md`). `code_size_tradeoff` and `locality_and_k_quad` are commented out of
the paper and were not run.

| figure | script | rows | wall | CSV | figure |
|---|---|---|---|---|---|
| Fig 8, resource selection | resource_selection_headroom | 1105 | 31 min | `resource_selection_headroom_standard.csv` | `results/figures/resource_selection_headroom_standard.pdf` |
| best-k phase diagram | 1_best_k_phase_diagram | 560 | 57 min | `1_best_k_phase_diagram_standard.csv` | `..._standard.pdf`, `..._standard_accuracy.pdf` |
| density ablation | density_ablation_sweep | 1044 | 318 min | `density_ablation_sweep_standard.csv` | `results/figures/density_ablation_sweep_standard.pdf` |

CSVs sit beside their scripts and are copied to the project's `results/data/` (same names).

## Fig 8: resource selection (13 circuits x 5 noise ratios)

Relative to naive [[4,2,2]], geometric mean over circuits:

| series | shots kept | silent error per kept shot |
|---|---|---|
| oracle (best of the selector's candidates, per metric) | 2.10x | 0.77x |
| AQORN selector (standard objective) | 1.78x | 0.95x |
| AQORN, static k = 2 (w10T) | 1.15x | 0.78x |
| worst static k, naive | 0.70x | 3.32x |

- The selector captures 70% of the shots headroom (100% / 99% at the two movement-dominated
  ratios, 33-61% elsewhere) and picks k = 4 in 34 of 65 columns (k = 2: 14, 6: 10, 8: 3, 10: 4).
- On accuracy it is about naive k = 2 (0.95x): across code sizes the objective's detected-error
  term dominates (fewer, fuller patches save more detected errors than 10 x silent errors cost),
  so in effect it maximises shots. The accuracy winner is the standard planner at k = 2 (0.78x,
  next to the 0.77x oracle).
- The model's pick is the measured best candidate in 28/65 columns on shots, 16/65 on accuracy.
- **Caption implication:** the shots oracle is no longer the selector's ceiling (it now trades on
  purpose). An honest claim: the selector keeps 1.8x the shots of [[4,2,2]] at its accuracy; the
  k = 2 standard is the most accurate configuration. A separate ranking weight for the selector
  (today it shares the planner's 10) would move it along that trade; not yet tested.

## Best-k phase diagram (16 circuits, 20-56 qubits; 7 noise ratios x k in {2..10})

Shots map (k with the highest selection rate per (program size, ratio); * = margin > 1 SE):

    gates        1/64  1/16  1/4   1     4     16    64
    164-258      k4*   k4*   k4*   k4*   k4    k10   k10
    259-327      k4*   k4*   k4*   k4    k4    k6    k10
    327-510      k4*   k4*   k4*   k4    k8    k8    k10
    510-2059     k4    k4    k8    k8    k2    k2    k2

Accuracy map (fewest silent errors per kept shot): **k = 2 in all 28 cells** (26 with margin > 1 SE);
silent error per kept shot rises monotonically with k, to 2-6x the k = 2 value at k = 10. The two
maps agree in 3/28 cells.

- **Caption implications:** choosing k for shots and for accuracy are different decisions; k = 2
  (where the transversal CZ exists) is the accuracy choice everywhere. "The largest programs settle
  near k = 6" does not hold (they go k = 4 -> 8 -> 2 as idling grows) -- nor did it in the Aug 1 v4
  regeneration; the paper's figure is the Jul 30 v3 run.
- The planner's own verify kept the naive packing in 131 of 560 (circuit, ratio, k) points (k = 2:
  22/112, 4: 35, 6: 26, 8: 19, 10: 29); `plan_status` in the CSV records it per row.

## Density ablation (71 circuits, k = 2, 6 patch counts from ceil(n/2) to n)

Cumulative mean over circuits by packing density (logical qubits per patch); `_pack` is the
packing segment alone; submitted -> standard:

| density | 1q | 2q | depth | LSR | LSR_pack | TVD | TVD_pack | silent / kept (std) |
|---|---|---|---|---|---|---|---|---|
| 1.9 | 22.5 -> 13.3 | 16.8 -> 19.3 | 28.7 -> 16.4 | 38.4 -> 43.1 | 9.6 -> 6.6 | 25.6 -> 38.3 | 2.7 -> 13.0 | 42.8 |
| 1.5 | 24.1 -> 16.4 | 17.7 -> 19.5 | 20.6 -> 10.4 | 75.5 -> 77.3 | 57.4 -> 54.1 | 15.8 -> 29.0 | -3.0 -> 7.0 | 35.3 |
| 1.3 | 22.6 -> 17.1 | 15.9 -> 16.8 | 14.8 -> 5.0 | 88.6 -> 84.8 | 78.4 -> 71.1 | 7.7 -> 20.5 | -4.8 -> 4.5 | 29.9 |
| 1.2 | 27.4 -> 21.1 | 20.8 -> 21.8 | 10.2 -> 1.8 | 162.7 -> 150.1 | 157.1 -> 141.7 | 5.1 -> 21.0 | -7.2 -> 5.8 | 32.1 |
| 1.1 | 29.3 -> 22.2 | 22.3 -> 23.3 | 0.2 -> -2.9 | 236.4 -> 211.0 | 236.7 -> 208.8 | -7.7 -> 11.2 | -15.8 -> 1.9 | 28.2 |
| 1.0 | 2.0 -> 2.0 | 0.0 -> 0.0 | -3.5 -> -3.5 | -6.3 -> -3.9 | 0.1 -> 0.0 | 0.5 -> 0.5 | 1.8 -> 1.3 | -1.5 |

- **Check:** the tightest bin is Fig 6 by another route: 1q / 2q / depth equal Fig 6 under the
  standard exactly (13.3 / 19.3 / 16.4) and LSR to 0.1 (43.1 vs 43.2); TVD reads lower (38.3 vs
  40.7) because this figure does not withhold the five saturated Ising circuits.
- **TVD now improves at every density** (38 -> 11%); the submitted packing segment went negative
  from density 1.5 down (to -16 points), the standard's stays positive. Silent error per kept shot
  falls 28-43%.
- **The packing segment's large LSR gains at intermediate density are a patch-count change, in
  both versions:** the planner packs into ceil(n/2) patches whatever the density, so at density
  1.1 the trivial arms use 1.73x more patches than the packed one (`packed_patches` in the CSV).
  At one qubit per patch nothing can be packed: the packed arm is the trivial packing there
  (`packed_source = trivial`), as the submitted run's n_code-3 fallback made it.

## Caveats

Silent error per kept shot is a frame measurement (100k frames per arm) on the simulation's own
Stim circuit; the phase diagram and Fig 8 simulators are unseeded (as submitted); Fig 8 and the
phase diagram use the planner's program-order verify while the arms are layer-sorted (how the
standard is defined; `reports/selector_standard.md`).
