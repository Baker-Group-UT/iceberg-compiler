# Fig 6 under the standard configuration (w10T + fair baseline), 2026-10-07

Question: what does the paper's Fig 6 show when every panel is re-simulated with the standard
configuration (`packing.standard.STANDARD`) instead of the submitted compiler?

**Method.** `experiments/figures/ablation_standard/ablation_standard.py` reruns Fig 6's own
per-circuit procedure end to end on the 71 circuits: four translations, the two-pilot alpha
calibration, four simulated arms, Fig 6's settings, noise and seeds
(`seed*1_000_003 + circuit_idx*17 + offset`). Changes from the submitted figure:

| arm | placement | H commutation | merging | in-patch CZ / XCX | order |
|---|---|---|---|---|---|
| baseline | tight trivial | off | off | 3-gate gadget | layer-sorted |
| + H | tight trivial | plan's | off | 3-gate gadget | layer-sorted |
| + merging | tight trivial | plan's | on | transversal (S / SX) | layer-sorted |
| + packing | `plan_standard` | plan's | on | transversal (S / SX) | layer-sorted |

The transversal gadget is credited to merging, as it was measured. Alpha and the shot count
are recalibrated on the layer-sorted baseline with Fig 6's pilot seeds, so the caption's
"calibrated to a baseline LSR of 6%" stays true (the baseline circuit changed on 46/71
circuits; alpha moved by a median -2% there). Plans: the recorded w10T plans, which equal
`plan_standard` (71/71 earlier today; re-checked on the gate circuits each run).

**Checks.** Gate: the same code under the submitted configuration reproduces the submitted Fig 6
rows exactly (calibration, baseline, all five panels) on adder_n10 and bell_n4. The 1q / 2q /
depth panels equal `ablation_fair_depth_w10T` exactly (213/213 rows). TVD is withheld on the same
five Ising circuits as before (n >= 26), so TVD averages are over 66 circuits.

**Results** (caption convention: mean over circuits of each circuit's cumulative change,
unclipped; paired difference with bootstrap 95% CI over circuits):

| metric | submitted (v4) | w10T as measured | **standard** | standard vs v4 | standard vs w10T measured |
|---|---|---|---|---|---|
| 1q gates | 22.5 | 13.3 | **13.3** | -9.2 [-13.1, -5.6] | -0.03 (4 circuits move) |
| 2q gates | 16.8 | 19.3 | **19.3** | +2.5 [+1.3, +3.7] | -0.03 (4 circuits move) |
| depth | 28.9 | 32.7 | **16.4** | -12.5 [-24.6, -3.3] | -16.3 (fair baseline) |
| LSR | +38.3 | +41.3 | **+43.2** | +4.8 [-0.6, +10.8] | +1.9 [+0.5, +3.4] |
| TVD | 26.8 | 40.4 | **40.7** | +14.0 [+10.4, +17.6] | +0.3 [-0.5, +1.3] |

Per stage (cumulative): depth +H 8.2 -> +merging 15.5 -> +packing 16.4; 2q 7.0 -> 16.4 -> 19.3;
1q 13.1 -> 16.0 -> 13.3; LSR 12.8 -> 36.8 -> 43.2; TVD 16.8 -> 29.0 -> 40.7.

Figure: `results/figures/ablation_standard.pdf` (and `.svg`), Fig 6's canvas and plotting code,
so it can replace `figures/plots/merging_improvement_benchmark_stylized_simulated.pdf` one for one.

**Findings.**
- **Fidelity holds or improves.** TVD 40.7% vs the submitted 26.8%; LSR +43.2% vs +38.3%
  (interval includes zero). Against w10T as measured, the fair baseline moves TVD by only +0.3
  (noise) and LSR by +1.9: alpha recalibrates about 2% lower on the circuits whose baseline
  changed, consistent with the layer-sorted baseline idling longer (`reports/movement_idle.md`).
- **Depth is the number that falls**, 28.9% -> 16.4%, and it is the fair baseline, not w10T:
  the baseline gets shallower once it is layer-sorted. One circuit carries 4 points of it:
  ising_n98's plan is 216 layers against a 62-layer baseline (-248%). Without it the mean is
  20.2% (the clipped bars average 20.3%).
- **1q gates fall less** (13.3% vs 22.5%): the transversal CZ is six single-qubit gates where the
  gadget was (1, 3). 2q gates fall more (19.3% vs 16.8%).
- **Per-circuit regressions** (cumulative after packing): LSR worse on 17 circuits, 13 of them
  GHZ / cat / W-state circuits (-12 to -28% LSR; 12 of the 13 gain 24-47% TVD: w10T trades kept
  shots for accuracy there); TVD worse on 2 (qaoa_n3 -30%, ising_n10 -5%; submitted: 14); depth
  worse on 11 (ising_n98 -248%, the rest at most 7%); 1q worse on 21.

**Caption text that no longer holds** (`paper/text/evaluation.tex`, Fig 6; for the user to edit):
- values.tex: `\OneQReduction` 13.3%, `\TwoQReduction` 19.3%, `\DepthReduction` 16.4%,
  `\LSRImprovement` 43.2%, `\TVDImprovement` 40.7%.
- "merging dominates the depth reduction": H commutation and merging now contribute about
  equally (8.2 and 7.3 points), packing 0.9.
- "The Ising family shows the largest depth reductions (up to 90.1%) but only modest LSR gains":
  under the fair baseline Ising depth changes by -248% to +13% and the largest reductions are
  error_correctiond3_n5 62.9%, hs4_n4 59.3%, qec_en_n5 57.1%, qec9xz_n17 52.3%. Ising LSR gains
  are +20 to +51% for n >= 26 (ising_n10: -9%).
- Worth saying instead: the baseline is layer-sorted like every stage; merging includes the
  transversal in-patch CZ; the silent-aware packing trades kept shots for accuracy on the GHZ /
  cat / W-state circuits.

Re-run:

    PYTHONPATH=. venv/bin/python -m experiments.figures.ablation_standard.ablation_standard --persist
    PYTHONPATH=. venv/bin/python -m experiments.figures.ablation_standard.ablation_standard \
        --from-csv experiments/figures/ablation_standard/ablation_standard.csv --no-plot \
        --compare results/revision/_inputs/merging_improvement_benchmark_stylized_simulated.csv \
                  results/tvd_explore/ablation_variant_w10T.csv

About 35 min on 12 workers. The CSV sits next to the script and is persisted to the project's
`results/data/ablation_standard.csv` (with its `.json` sidecar: git state, settings, timing).
