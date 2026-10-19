# `_add_cross_gate_conflicts` was cubic — fixed

**Status: done, uncommitted, on branch `na-packing-costs` (base `7d9d7ba`).** Touches only
`translator/merging/conflicts.py` plus a new test. Revert is:

```bash
git checkout -- translator/merging/conflicts.py && rm tests/test_merging_conflict_index.py
```

**Scope: speed only.** Behaviour-preserving. Which merges fire, the emitted circuits, gate
counts, LER and TVD are all unchanged, and **no figure needs regenerating**. Nothing in
`packing/`, `simulator/` or the noise model was touched. Read "Do not bundle" before
touching `selection.py`, which holds a separate finding that *would* change results.

Written 2026-07-31.

---

## What the routine is

[`_add_cross_gate_conflicts`](translator/merging/conflicts.py:226) detects one of four
classes of conflict between candidate merges. Merges A and B conflict via a bridging
two-qubit gate G when all of:

1. `pos(B.first) <= pos(G) < pos(B.second)` — G is inside B's blocking window
2. `pos(G) < pos(A.second)` — G precedes A's macro in the original circuit
3. `pos(B.second) > pos(A.second)` — B fires after A, so B's delay pushes G past A
4. `qargs(G) ∩ B.blocked_qubits ≠ ∅` — G is actually caught by B's block
5. `qargs(G) ∩ A.macro_qargs ≠ ∅` — G has an ordering relationship with A

`pos(n)` is `(layer_index[n], order_index[n])`. It was a triple loop over every candidate
A × every candidate B × every two-qubit gate, with `pos()` recomputed innermost:
**O(C² · G₂)**, cubic in circuit size, since the candidate count C grows with the
single-qubit gate count.

## Blast radius

Reached from exactly one place:

```
preprocess_program_for_merging_with_stats  (translator/merging/preprocessing.py:100)
  └── select_best_group                    (translator/merging/selection.py:45)
        └── compute_merge_conflicts        (translator/merging/conflicts.py:112)
              └── _add_cross_gate_conflicts
```

So the *code* is merging-only. The *cost* was not, because merging runs inside every
translation ([`translator/core.py:140`](translator/core.py:140)) and the planner
translates: [`packing/plan.py:145`](packing/plan.py:145) in the verify stage, and
[`packing/compiler.py:233`](packing/compiler.py:233) once per candidate arm.

**Never affected:** the packer's scoring loop. `packing.cost` and
`PairMergeAnalyzer` never call `compute_merge_conflicts` — `PairMergeAnalyzer` uses its
own `_select_non_overlapping` greedy. The noise model and simulator are downstream of the
emitted circuit and see nothing of this.

## What changed

The observation that removes the third factor: **the qualifying G depend only on B.**
Gather them once per B into `first_hit`, mapping each qubit to the earliest position at
which a gate caught by B's block touches it. The pair test then collapses to a lookup
over A's macro qargs:

```
conflict(A, B)  ⟺  pos(B.second) > pos(A.second)
                   and min(first_hit[q] for q in A.macro_qargs) < pos(A.second)
```

`first_hit` carries conditions 1 and 4; the comparison carries 2 and 5; the leading term
is 3. The window is bisected out of a position-sorted bridge array rather than found by
scanning, and `pos` is memoised. Cost: **O(C · window + C² · k)**.

`_add_first_in_window_conflicts` got its per-pair work hoisted out too — it was building
`set(merge_b.macro_qargs)` once per *pair* rather than once per merge. Still O(C²), which
is inherent, but with cheap comparisons.

### Why it is behaviour-preserving

The final `conflicts` dict is a symmetric edge set. Three things looked load-bearing and
are not:

- `if idx_b in conflicts.get(idx_a, set()): continue` only avoids re-adding an existing edge.
- `break` at the first witness G — existence is what the predicate asks; which witness
  supplied the edge was never recorded.
- Iterating B outer instead of A outer — `_record_conflict` is symmetric.

## Measured

Two passes of merging (default `num_merge_passes = 2`), 32 program qubits, H-dense random
circuits (45% H), valid packings (`k = n_code − 2`), same seeds before and after:

| gates | k=2 before | k=2 after | speedup | k=4 before | k=4 after | speedup |
|---|---|---|---|---|---|---|
| 250 | 58.5 ms | 22.9 ms | 2.6× | 13.8 ms | 13.4 ms | 1.0× |
| 500 | 448 ms | 47.4 ms | 9.5× | 54.0 ms | 25.2 ms | 2.1× |
| 1000 | 3.49 s | 114 ms | 31× | 395 ms | 55.4 ms | 7.1× |
| 2000 | 28.2 s | 234 ms | 120× | 2.58 s | 106 ms | 24× |
| 4000 | 231 s | 582 ms | **397×** | 21.3 s | 254 ms | **84×** |

The routine's share of the merging pass went from 66.7–99.7% to 2.0–3.5%. More
importantly the per-gate cost is now roughly flat — 92 → 145 µs/gate across a 16× range
in circuit size, against 234 → 57,691 µs/gate before. That is the cubic term gone, not
merely reduced.

On the paper's actual benchmark circuits it was already cheap (0.3–4.5% of one
`translate_permutation`), so **this was a latent scaling cliff, not a present-day
experiment cost** — do not expect existing figure runtimes to change much. `adder_n64`
went from 9.9 ms to 0.5 ms in the routine. What the fix buys is headroom: H-dense
circuits, high patch counts (small k ⇒ more patches ⇒ more candidates), and unrolled
circuits are no longer cliff edges. `vqe_n24` unrolls to 2.0 M gates and used to spend
~193 s in merging.

## Verification

- `tests/test_merging_conflict_index.py` — new, 39 cases. Keeps the enumerated form as an
  oracle and asserts edge-set identity across `n_code ∈ {4, 6, 8}` × 8 seeds; also checks
  that a pre-populated `conflicts` dict survives (the routine runs *after* the geometric
  and shared-endpoint passes, so it arrives populated, and a rewrite that built a fresh
  dict would pass the identity test and break here), and pins the full pass's savings
  triple and instruction list.
- Full suite: **642 passed** in 160 s, including `test_merging_generalized.py`,
  `test_pipeline_logical_action.py`, `test_packing*.py`.
- `ruff check` clean on both files.

## Do not bundle

Two further findings in the same subsystem. The first **would** change results; keep it
out of any perf commit:

1. **Dead tiebreak in the selector.**
   [`selection.py:175`](translator/merging/selection.py:175) hardcodes
   `conflict_weight=0.0` in every `MergePriority`, so the second comparison in
   [`_is_better`](translator/merging/selection.py:128) can never fire and ties fall
   through to program order. The real conflict weight *is* computed at
   [`selection.py:146`](translator/merging/selection.py:146) but only into
   `MergeConflictInfo`, never back into the priority. Wiring it up changes which of
   several equal-savings candidates wins, hence which merges fire — needs the merging
   figures regenerated, and should be argued on its own merits.

2. **The cost table cannot price per-lane groups wider than two lanes.** `X̄` on m lanes
   emits `m + (m mod 2)` physical gates —
   [`_apply_pauli`](iceberg/iceberg_module.py:185) fires the anchor only for odd m,
   because `X̄_i = X[x_anchor]·X[lane i]` and an even-sized product cancels the anchor
   outright. So the cost of `xx`/`zz` depends on the group **width**. `work_table` has one
   fixed `"XX": (2, 0)` entry with no width parameter, and
   [`_merge_savings_for_records`](translator/merging/preprocessing.py:207) computes
   `fused × X − XX`, exact at m = 2 and over-credited beyond it by `m − 2 + (m mod 2)`:

   | lanes m | emitted 1q | m separate X̄ | true saving | reported |
   |---|---|---|---|---|
   | 2 | 2 | 4 | 2 | 2 |
   | 4 | 4 | 8 | 4 | **6** |
   | 6 | 6 | 12 | 6 | **10** |
   | 10 | 10 | 20 | 10 | **18** |

   **The emitted circuit is correct** — this is a reporting and ranking error, not a
   compilation error. Anything measured by counting gates in the output circuit is
   unaffected; `merge_savings_*` reaches `analysis/result_row.py` and the CSVs but is
   plotted in no figure; and `packing/cost.py` models `_hh` and `_transversal` only, with
   no `xx` term, so the packer's objective never sees it. What it does affect is the
   selector's ranking magnitude in
   [`_merge_savings_weight`](translator/merging/selection.py:181) — wider is still better,
   so the direction is right, but a wide `xx` is over-valued against a rival candidate.
   Fixing it changes which merges fire, hence the placement in this section.

   Whole-patch macros are fine: `HH` is `(n, 0)` and `CXT` is `(0, n)` however many lanes
   they cover, both genuinely width-independent. Only the per-lane family has a
   width-dependent cost — which is exactly the family with no coverage condition. The same
   k = 2 collapse that hid the coverage rule hid this.

3. **Stale docstring.** [`merging/__init__.py:49`](translator/merging/__init__.py:49)
   still says routing swaps are "priced (0, 0) -- so alignment costs nothing". Wrong
   since `_macro_gain` landed:
   [`transversal.py:288`](translator/merging/transversal.py:288) charges
   `2 × transpositions × carriage_cost("SWAP")` and rejects non-positive-gain macros.
   `transversal.py`'s own docstring and `ALGORITHM_SUMMARY.md` are correct. Docs-only.

## Separate: a cheaper route for partial Hadamard coverage

Not a bug, and not part of this change, but it falls out of the same cost table and is a
genuine k > 2 phenomenon worth recording.

`hh` requires *full* coverage — a whole-patch Hadamard hits every lane, so the detector
refuses when only some lanes want one, and the translator emits targeted single-lane H
instead. But `HH` is `H̄^⊗k` exactly, so **HH followed by a targeted H on each unwanted
lane** also implements a partial Hadamard, and the table says it is cheaper well before
full coverage. Weighted work at `DEFAULT_WORK_WEIGHTS = (1, 13)`, targeted `H` = (7,4) =
59, `HH` = (n,0) + 2 carriages:

| | lanes wanting H | k × targeted | HH + undo | cheaper |
|---|---|---|---|---|
| k=2 | 1 of 2 | 59 | 73 | targeted |
| k=4 | 3 of 4 | 177 | **75** | HH + 1 undo (2.4×) |
| k=6 | 4 of 6 | 236 | **136** | HH + 2 undo |
| k=10 | 6 of 10 | 354 | **258** | HH + 4 undo |

Crossover is at roughly `want > k/2`. At k=2 targeted always wins for a partial group,
which is why the pairwise implementation never needed this and why it is new at k > 2.

**Unverified.** This is what the cost table says; whether it is emittable depends on the
anchor bookkeeping composing correctly across `HH` then `H_at` — `HH` emits transversal H
plus `SWAP(x_anchor, z_anchor)` ([`iceberg/iceberg_module.py:257`](iceberg/iceberg_module.py:257))
and the second H would act in that restored frame. Needs a logical-action test before
anyone believes the numbers.

## Also worth knowing while in these files

`MergeSettings.zero_delay_costs` defaults `True`
([`settings.py:30`](translator/merging/settings.py:30)), so `cost_for` returns 0.0 and
the delay budget never binds. The forward scan stops only at structural blockers, not on
a budget. Any prose describing merging as searching "within a delay budget" is describing
a disabled mechanism.
