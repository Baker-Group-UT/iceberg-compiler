# Handoff: code size at an equal physical-qubit budget (2026-10-11)

For a teddy session. Keep the run to about an hour of wall clock
and don't take the whole machine. **The compiler is frozen**: no changes under `translator/`,
`packing/` or the logical-optimisation pass. Experiment and plotting code is fine. Nothing gets
pushed.

## Why

Every code-size comparison in the paper puts each k at its *tightest* packing, ceil(n/k)
patches. On that footing k = 2 is the most accurate everywhere:
- the occupied-lane phase diagram (`fig7_occupied/`): fewest silent errors per kept shot in
  28 / 28 cells;
- Fig 8's trade-off: AQORN static k = 2 TVD 0.105 vs 0.129 at k = 4.

But k = 2 also uses the most hardware: 4 ceil(n/2) physical qubits, against 6 ceil(n/4) at
k = 4 (-23%). So "k = 2 is most accurate" partly just says "k = 2 has more patches, so more
detectors".

The fair question for a reader with a fixed device is: **given Q physical qubits, which code
gives the best accuracy and the most shots?**
- A larger code under the same Q has room to spare. It can run below full occupancy, which
  means more patches than ceil(n/k) and fewer logical qubits behind each pair of checks.
- k = 2 at the same Q is at (or near) its tightest packing.

This also compares against stronger comparators on equal
resources, which the current figure doesn't.

What we already know about empty lanes (`reports/solo_gauge.md`):
- An empty lane is free. The decoder never reads it, so errors confined to it are harmless.
- It adds no detection. Every patch has two checks whatever its occupancy.

So spreading a big code over more patches helps by adding patches (checks), not lanes. Don't
predict the winner; measure it.

The user would also like this figure to replace two earlier ones:
- the cut noise-ratio figure (AQORN vs naive across idle/distance ratios);
- the code-size phase diagram ("Fig 14").

So sweep the idle/distance ratios as well.

## Design

**Corpus and noise: Fig 8's.**
- 13 circuits, 25-40 qubits: adder_n28, bv_n30, cat_n35, dnn_n33, ghz_n40, ising_n26,
  ising_n34, knn_n25, knn_n31, qugan_n39, swap_test_n25, wstate_n27, wstate_n36.
- 5 idle/distance ratios: 1/256, 1/16, 1, 16, 256.
- The same per-(circuit, ratio) alpha as Fig 8 (column `alpha` in
  `results/data/teddy_final_2026-10-11/fig8_resource_selection_headroom_standard.csv`), so the
  numbers line up with Fig 8.
- If there's time, the 7 ratios of `noise_ratio_packers` instead.

**Budget.**
- Physical qubits Q = patches x n_code: data qubits only, the paper's and Fig 8's convention.
- Check whether the emitted circuits carry two parity qubits per patch. If they do, also report
  each point with them counted (n_code + 2 per patch), as a sensitivity check. It shifts the
  budget against small codes.
- Budgets per circuit, relative to Q2 = 4 ceil(n/2) (k = 2's tightest footprint):
  - **Q2**: the main comparison; every k fits;
  - **1.25 Q2** and **1.5 Q2**: does the ranking hold as hardware grows?
  - optionally **Q4** = 6 ceil(n/4): k = 2 does not fit, which shows its qubit price.

**Arms at each (circuit, ratio, budget, k).**
- Patch count P = floor(Q / n_code), clipped to [ceil(n/k), n]. If floor(Q / n_code) <
  ceil(n/k), the code does not fit; record that and skip.
- Record the qubits actually used: P x n_code is at most Q, and the slack differs by k.
- **AQORN**: `packing.standard.plan_standard(program, n_code=k + 2, patches=P, noise=profile)`.
  Use the ratio's noise profile, as `noise_ratio_packers.py` does. `patches=P` plans at exactly
  P patches; the density sweep's `--matched-patches` uses the same call.
- **naive**: program order, spread as evenly as possible over P patches (patch sizes differ by
  at most one). At k = 2 that reproduces `tight_trivial_packing(n, P)` in
  `experiments/figures/density_ablation_sweep/density_ablation_sweep.py`, which is k = 2-only;
  generalise it in the experiment code.
- k in {2, 4, 6, 8, 10}.
- Size: 13 x 5 x 3 budgets x 5 k x 2 arms, about 1950 points, fewer after skips. The Fig 8
  headroom run (1105 rows) took about 35 minutes.

**Measurement: Pauli frames, as Fig 8 and `noise_ratio_packers` do.**
- **LSR**.
- **Logical error per kept shot on occupied lanes only.** This is essential here: at equal Q
  the large codes have many empty lanes, and the all-lane count that biased the first Fig 7
  would bias this experiment far more. Use the occupied-lane count from `evaluate(frames=)`
  (merged from `v2-teddy-fig7`; `TranslationContext.state_positions`).
- **TVD** with the floor test. 4 of the 13 saturate (ising_n26, ising_n34, qugan_n39,
  swap_test_n25), so TVD is on the other 9 and logical error is the primary accuracy metric.
- Plus the static counts: 1q, 2q, depth, patches, physical qubits.
- Reuse what exists:
  - `experiments/figures/resource_selection_headroom/resource_selection_headroom.py
    --standard` already simulates every k with frames for this corpus. A `--qubit-budget`
    mode there is the shortest path.
  - Or `experiments/optimality/placement_measure.measure`, but note it is hard-wired to
    `standard_translator_config(4, ...)`, so pass n_code through.
- Run the noiseless check on every arm (0 miscompiled is what the paper quotes).

## Outputs

- CSV: `results/data/teddy_final_2026-10-11/equal_qubit_code_size.csv`, one row per
  (circuit, ratio, budget, k, arm), with:
  - n, k, n_code, P, physical qubits used and the budget, alpha;
  - lsr, logical error (occupied lanes), tvd, tvd floor;
  - 1q, 2q, depth;
  - the noiseless-check result.
- Figure, house style (`docs/figure-style.md`, `experiments/shared/aqorn.mplstyle`): single
  column, SVG + PDF, to `results/figures/teddy_final_2026-10-11/equal_qubit_code_size.{svg,pdf}`.
  Draw from the CSV with `--from-csv`.
- Suggested figure, adjust if the data says otherwise:
  - x = idle/distance ratio (log);
  - two stacked panels sharing x: (a) LSR relative to naive k = 2 at Q2, (b) logical error
    per kept shot (occupied lanes);
  - one line per k for AQORN at budget Q2;
  - colours in palette order with k = 2 first;
  - naive drawn once, dashed, at k = 2 only, so the AQORN gain over naive is visible (what the
    cut figure showed);
  - geometric mean over circuits with bootstrap 95% CIs (state it in the caption);
  - the 1.25 / 1.5 Q2 budgets go in a table in the report, not the figure.
- Report: `reports/equal_qubit_code_size.md`, with the headline numbers:
  - at Q2 and each ratio: the winning k on LSR and on logical error, its margin, and whether
    it clears one SE;
  - AQORN / naive per k;
  - whether the ranking moves with the budget;
  - the ancilla-counted sensitivity check.
- Commit the experiment and figure code on `v2-teddy-final` (local only), and add a line to
  `handoffs/laptop_writing_2026-10-11.md`'s figure map.

## Checks before reporting

- At budget Q2 and k = 2, the AQORN arm must reproduce Fig 8's static k = 2 AQORN numbers for
  the same (circuit, ratio): same plan, same alpha, same frame seed. That's the harness gate.
- Every arm compiles correctly (noiseless check), and P x n_code <= Q on every row.
- Don't compare a k that hits the clip ceil(n/k) against one that doesn't without saying so.

## How the paper will use it

- If a larger code beats or ties k = 2 on logical error at equal qubits, the code-size story
  becomes "at a fixed device, the larger code wins". That's a real result for E-C4.
- If k = 2 still wins at equal qubits, the paper says plainly that the accuracy advantage
  survives an equal-hardware comparison, and that larger codes save qubits only when the
  device is smaller than 4 ceil(n/2).
- Either way, the figure replaces the phase diagram and carries the across-ratio AQORN-vs-naive
  result.
