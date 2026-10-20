# Density ablation: matched-patch arm (draft spec)

## Problem
`+ Packing` in the density ablation is two things at once. The packed arm always uses
ceil(n/2) patches (`packed_patches` in the CSV), while the trivial arms get n_codes patches
from ceil(n/2) up to n. Below density 2 the packing segment therefore includes a
patch-count change. On the standard CSV, at equal patch counts the median LSR gain from
packing is about 0%; at ratio > 1.5 it is about +157%. Most of that is patch count.

## Decision: no new simulation
Every (circuit, n_codes) row already has all four arms' metrics. The h_merge arm at the
row with `n_codes == packed_patches` is exactly "trivial packing, merging on, same patches
as the packed arm". It was simulated with the circuit's own alpha and repetitions, so
nothing about the noise calibration changes (alpha stays per circuit, never per density).

Only caveat: the matched arm's seed differs from the density row's. Cross-row differences
are therefore subject to shot noise; the bootstrap CIs in the summary CSV cover
across-circuit spread, not per-run shot noise.

## Stages (stacked, deltas vs the same baseline)
1. H commutation = h_only vs baseline
2. + Merging = h_merge - h_only
3. + Patch count = h_merge@packed_patches - h_merge@this density
4. + Placement = packed - h_merge@packed_patches

The bar total is unchanged from the current figure; only the packing share is split.
Density-1 rows (packed = trivial at n patches) have matched = the row itself, so both
packing stages are about 0 there.

## Figure changes
- Stack split above; new light-blue `+ Patch count` segment.
- TVD panel restricted to n <= 18 qubits and baseline TVD < 0.95 (floor / saturation).
- Bars need >= 4 circuits per (bin, size group); otherwise not drawn.
- X label says the axis is the trivial arm's density.
- Companion outputs: summary CSV with mean / median / bootstrap CI; absolute LSR table.

## Not changed
alpha, repetitions, seeds, packing plan, planner, arm definitions, the CSV.

## Open items
- Silent error per kept shot as a replacement wide-circuit accuracy panel (data present as
  `*_silent_per_kept`; computed in the summary CSV, not yet plotted).
- Optional new run: re-simulate the matched arm with the same seed as the density row to
  remove seed noise from the patch-count stage. Not needed unless asked for.
- Caption text: "Below density 2 the trivial arms use more patches than the packed arm; the
  light-blue segment is that difference, placement is packing's own contribution."
