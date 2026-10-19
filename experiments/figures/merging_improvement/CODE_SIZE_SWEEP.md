# Technique benefits across the `[[k+2, k, 2]]` code family

Figure: `merging_improvement_code_size.pdf` / `.svg`
Data: `merging_improvement_code_size.csv`
Script: `merging_improvement_code_size.py`

> **This document describes the sweep as the `grouping` stack ran it.** That package has
> since been archived and the script now calls `packing`, which absorbed the four pieces
> it needed. The experiment is unchanged — same circuits, same three cumulative
> techniques, same CSV schema — but the CLI is not: `--packer` now takes `packing`
> (verified, the default) or `raw` (verify off), and `--packer analytic/measured/legacy`,
> `--no-refine`, `--refine-rounds` and `--refine-seconds` are gone with the code behind
> them. `--partitioner` is new. The figure and CSV the run below produced are in
> `archive/experiments/figures/merging_improvement/`; nothing live can regenerate them.
> Everything from "Setup" down is the record of how the numbers in that CSV were
> arrived at, and is left as it was written.

Styled after paper Figure 12 (`technique_benefits_random_hg.pdf`): one row per metric,
stacked bars whose segments are the incremental benefit of each technique, one bar per
benchmark size group at every x position, Times New Roman to match.

Run that produced the committed figure:

```bash
PYTHONPATH=. venv/bin/python -m experiments.figures.merging_improvement.merging_improvement_code_size --no-show
```

29 small (3–10 qubits), 15 medium (11–27 qubits) and 27 large (28–140 qubits) QASMBench
proxy circuits, five code sizes, **116 seconds**. LSR and TVD are wired through the CSV
schema and the plot's row selection but need simulation, so they are absent until
`--metrics` asks for them.

## Setup

At each k the baseline is a naive index-order packing at the *same* k with merging and
H commutation off, so every bar is a within-k ablation. Packing density is held at its
tightest, `ceil(q/k)` patches — the direct analogue of the paper's `ceil(n/2)`.

The packer is the `grouping` stack: signed interaction graph, multilevel partitioning,
exact Hungarian slot alignment, then hill-climbing on the analytic cost. The objective
prices weighted gate count *and* depth, so all three plotted rows are optimised rather
than two of them plus a bystander. `packer/merge_packer.py` assigns qubits in pairs and
so cannot express a placement at k > 2; `--packer legacy` selects it anyway for the k=2
reference below.

### Why this used to take hours

The first version of this script was the slow one in the repo, and the cause was the
objective rather than the search. Every candidate placement was scored by *translating
the circuit*, so one sweep cost `candidates × translation`, and the swap neighbourhood
is `O(m²)`.

The symptom was worse than the average: runtime tracked neither qubit count nor the
progress bar. A point finished in 0.00s whenever the circuit collapsed to a single patch,
because local search is skipped when there is no placement decision to make — and then
jumped to seconds or minutes the moment there were two, scaled by *gate* count rather
than width. `dnn_n8` — 8 qubits, 900 gates — took 8.5s at k=6 while `qaoa_n6` took 0.00s,
and a 20-qubit `qram_n20` took 11.6s while a 23-qubit `ghz_state_n23` took 2.0s. Alternating
stalls on a sweep that looks uniform from the outside.

`--packer analytic`, now the default, scores with `grouping.analytic` instead: ~0.1ms per
candidate, no translation in the loop. Measured on these same circuits:

| point | old (`measured`) | new (`analytic`) |
|---|---|---|
| `dnn_n8`, k=6 | 8.47s | 0.14s |
| `hhl_n7`, k=6 | 5.11s | 0.14s |
| `ising_n10`, k=6 | 3.47s | 0.05s |
| `qram_n20`, k=6 | 13.25s | 0.18s |
| `ghz_state_n23`, k=6 | 10.94s | 0.03s |
| `qft_n18`, k=2 | 11.50s | 0.20s |

10–60× per point on small circuits, 50–100× on medium, and it needs no wall-clock cap —
so no point is truncated and every placement is a real local optimum of the stated
objective, where before the points that needed the search most were the ones that got a
truncated, order-biased fraction of it. The whole three-group sweep went from *not
finishing the medium group in 14.5 minutes* to 116 seconds for all 1080 points.

The final candidate is still checked against the naive placement by exact measurement —
two translations — because the analytic model over-counts merging by 2–3% and on a few
circuits that was enough to prefer a placement genuinely worse than sequential. That
makes the packing segment safe to read as a gain.

`--packer measured` restores the old path for comparison. It is the only one that needs
`--refine-seconds` and the only one whose `pack_note` can say `TRUNCATED`.

Correctness first: the exact logical-action check from `tests/test_pipeline_logical_action.py`
was extended to n=10 and n=12 (k=8 and k=10) before any measurement, since last night's
matrix stopped at n=8. Zero failures at every size.

## Result

> **This has been wrong twice, in opposite directions.** First it reported medium circuits
> best at k=4 (+6.3 depth) — measured on circuits whose rejected gates were being silently
> dropped, so they compiled cheaper *and* wrong. Then it retracted that in favour of k=2,
> correctly given the data, but that data was survivorship-selected: with the dropping fixed,
> a third of the benchmarks refused to place at k>2. Those refusals turned out to be a
> vestigial restriction (STATUS.md §4.3). **All 30 small, 15 medium and 27 large circuits now
> place at every code size**, and the numbers below are the first like-for-like set.

Full-pipeline reduction (stack height), same circuits at every k:

| k | medium 1q | medium 2q | medium depth | large 1q | large 2q | large depth |
|---|---|---|---|---|---|---|
| 2 | 26.5% | **18.4%** | 31.4% | 20.7% | **15.7%** | 37.7% |
| 4 | **26.7%** | 17.3% | 38.7% | 22.5% | 15.2% | **38.0%** |
| 6 | 24.8% | 17.1% | **39.3%** | 22.5% | 13.9% | 37.3% |
| 8 | 23.7% | 16.5% | 39.1% | **22.7%** | 15.0% | 37.2% |
| 10 | 23.2% | 16.3% | 38.2% | 21.8% | 13.7% | 35.0% |

**Depth on medium circuits gains 7–8 points past k=2** — 31.4% against 38.7% at k=4 and
39.3% at k=6, holding above 38% out to k=10. That is the direct evidence for the conjecture
in `text/evaluation.tex`, and it is a regime the paper's pairwise packer cannot reach at all.

Two-qubit gates move the other way, declining slowly at both widths; one-qubit gates are flat
to k=4 then fall. So the conjecture holds **for depth and not for gate count**, and the
merging table below is the mechanism for the gate-count half.

Large circuits are flat on depth (37.7% → 37.2% through k=8, dropping only at k=10) and gain
about two points on one-qubit gates.

Small circuits decline (24.0% → 12.7% on 1q) for a structural reason rather than an
algorithmic one: they run out of patches. 17 of 30 fit in a single patch at k=4 and all 30 at
k=10, so the packing segment is exactly 0.0% there. Medium's is the trend to quote.

## Merging past k=2

`translator/merging` used to be [[4,2,2]]-only, which made every k ≥ 4 bar a lower bound of
unknown tightness. It is now general in k. What that changed:

| k | merging 1q | 2q | depth | fired on |
|---|---|---|---|---|
| 2 | 10.1% | 7.5% | 25.9% | 58/99 |
| 4 | 10.3% | 5.7% | 15.2% | 45/99 |
| 6 | 9.9% | 4.8% | 13.1% | 40/99 |
| 8 | 9.4% | 4.3% | 12.0% | 39/99 |
| 10 | 8.7% | 4.1% | 11.1% | 41/99 |

Every one of those was 0.0% before. Merging is now the single largest contributor to depth
at every code size — 25.9 points at k=2, more than H commutation and packing combined.

**Merging decays gracefully with k rather than holding or vanishing**, and the reason is
structural. `hh` and the transversal macros are *whole-patch* operations, so a group needs
one program gate per occupied lane — k gates, not two — and the chance that a circuit
presents k of them adjacently falls as k grows. The firing rate tracks that: 58/99 circuits
at k=2 down to 41/99 at k=10. `xx`/`zz` are per-lane and fuse on any subset, so they are
unaffected; they are what keeps the segment from collapsing. Note that 1q *rises* slightly
from k=2 to k=4 before falling — a fired whole-patch macro subsumes k gates rather than 2,
so the wins that survive are individually larger.

Fusing a *subset* of lanes into a whole-patch macro is not available, and that is a fact
about the code rather than a gap in the implementation: partial transversal CX maps the
full-weight stabilizer `S_X = X^⊗n` to a weight-|T| operator outside the stabilizer group,
so it is not a logical operation at all. See the plan's §5.7 for the reachable middle
ground (transversal-then-correct) and why it is a placement question as much as a rewriting
one.

**Patch count falls as k rises**, so packing has progressively less to decide. Recorded per
row as `patches_packed`, counted in the summary, filterable with `--min-patches`.

## Three findings, two of them now fixed

**1. The patch budget is now honoured — was: a target, not a guarantee.** The clustering
version returned more patches than the tightest packing on 8 of 225 placements, 7 of them
at k=6, because greedy merging stalls when every remaining cluster sits just over half
capacity (at k=6, clusters of 4 and 4 cannot join and neither can 4 and 3). Those rows were
then penalised for carrying extra encode/decode, which is where the negative 2q entries for
`ghz_state_n23` and `cat_state_n22` came from. `multilevel.partition` has a `_consolidate`
bin-packing repair pass for exactly this. **0 of 495 placements exceed the budget now**, at
any k.

**2. The objective prices depth, so the 1q row no longer disagrees with it.** Under the old
gate-only objective `1×1q + 13×2q` the search would spend 1q gates to buy 2q gates —
`pea_n5` at k=4 read −98% on 1q while improving 2q by 6.7% — and the gap between the plotted
stack and the true reduction reached 11.4 points (medium k=6: 29.4% plotted vs 18.0% true).
The analytic objective prices `1×1q + 13×2q + 12×depth`, and the gap is now **at most 0.6
points** across all 15 group×k cells (only medium k=4 and large k=2 differ at all). The
summary still prints both; there is just no longer much to read.

The clipping convention also changed, and that matters independently of the objective. See
`aggregate`: clipping each *segment* at zero inflates the stack whenever one technique hurts
and a later one recovers, so this script now clips the cumulative series and differences it.

**3. The new packer beats the paper's at k=2 on the metrics that matter.** Packing segment,
pooled over all 72 circuits:

| packer | 1q | 2q | depth |
|---|---|---|---|
| `analytic` (signed graph + partition + analytic refinement) | 0.7% | **2.4%** | **7.7%** |
| `packer/merge_packer.py` (pairwise) | 0.8% | 0.6% | 2.0% |

4× on two-qubit gates and nearly 4× on depth, at a tenth of the runtime, and it is the only
one of the two that works at k > 2 — so it is a replacement rather than an extension. The
depth gap is the interesting one: the old comparison had the pairwise packer *ahead* on
depth (6.3% vs 5.5%), which was the gate-only objective leaving depth to chance.

Reproduce with:

```bash
PYTHONPATH=. venv/bin/python -m experiments.figures.merging_improvement.merging_improvement_code_size --packer legacy --k-values 2 --no-show --csv /tmp/legacy_k2.csv --output /tmp/legacy_k2.svg
```

## How merging was generalized

The whole change turns on one distinction the pairwise implementation never had to make.

*Whole-patch macros* — `hh`, `cxt`, `czt`, `xcxt` — are built from physical gates applied to
all n data qubits, so they act on every lane whether or not a program gate asked. A group is
therefore only a legal substitution when it covers every **occupied** lane: k program gates,
not two. *Per-lane macros* — `xx`, `zz` — need no such condition, because
`X̄ᵢ = X[x_anchor] X[lane i]` makes any subset legal and an even subset cancels the anchor
factor outright. At k=2 a group of two lanes *is* the whole patch, so the two rules coincide
and coverage never had to be named.

What that required:

- `MacroReplacement.first/second` → `MacroReplacement.nodes`, since a macro now subsumes up
  to k instructions. `first`/`second` survive as the window endpoints the conflict analysis
  reasons about. `rewrite.py` drops all but the last and arms each dropped node's block at
  *its own* position — arming them all at the earliest position would delay gates that
  legitimately precede a later lane's gate, reordering them against the macro.
- `HHGate`/`XXGate`/`ZZGate` take a width; the names stay `"hh"`/`"xx"`/`"zz"` so the
  handler tables and `MACRO_NAMES` are untouched.
- `single_qubit` collects one gate per uncovered lane within the delay budget, and takes a
  `require_full_coverage` flag that is True for `hh` and False for `xx`/`zz`.
- `transversal` accumulates up to k same-type cross-patch gates forming a slot bijection.
  Alignment generalizes from the hand-enumerated identity/cross pair to any permutation of
  k slots, decomposed by selection sort into at most k−1 transpositions and emitted as
  routing swaps around the macro. Verified exhaustively against all permutations up to k=6.
- Coverage is checked a second time in `handlers.utils.resolve_transversal_macro`, against
  the *live* slot map, because a SWAP can move a qubit between detection and translation.
  That also collapsed three near-identical resolvers into one.
- Savings scale with `len(record.nodes)` and with n (`conflicts.merge_work_table`). Fixing
  the count at two understates a k=10 merge fivefold, which would make the selector prefer
  the wrong candidates.

Idle lanes are treated as free to disturb, which is what makes the coverage rule practical:
a macro covering the two occupied slots of a four-slot patch is admissible, since nothing
reads the idle lanes and `apply_logical_hadamard` already fires a whole-patch HH for a patch
holding one qubit.

**Verification.** `tests/test_merging_generalized.py`, 62 cases at n ∈ {4,6,8,10,12}. Random
Clifford programs almost never place a gate on all k lanes at once, so the whole-patch
macros go unexercised in the random harness and it reports a vacuous pass at k > 2 — worse,
at n=10 and n=12 the fault-tolerance rules reject every random permutation, so `ok=0`. These
cases are built to trigger each macro and then checked with the same exact tableau
comparison. Negative cases are pinned too: 2-of-k H must not fuse at k > 2 while the same
input does fuse at k=2. Full suite 225 → 291 tests, all passing.

## Fixes made to support the sweep

- `grouping/refine.py`: `reorder` enumerated `k!` slot permutations per patch. At k=10
  that is 3.6M candidates for patch 0 alone, so any budgeted search would exhaust itself
  there and never look at patch 1. Now capped at `MAX_REORDER_SLOTS = 4`; transpositions
  remain reachable at any k because a `swap` of two qubits in the same patch is exactly
  one. Pinned by `test_reorder_moves_stay_bounded_at_large_k`.
- `refine_placement` docstring: notes that `max_evaluations` can cut a round short and
  bias the search toward whatever `_candidates` yields first, so bounding `max_rounds` is
  preferable when the objective is expensive.
- Two mismatches inside this experiment, caught before the measuring run: the refinement
  objective must translate the *raw* program, because `Translator.translate` applies
  `cancel_inverse_gates`/`commute_hadamards` itself and would otherwise run them twice;
  and it must not use `grouping.refine.measured_objective`, which counts SWAPs as
  two-qubit gates where `one_two_qubit_gate_totals` excludes them by the paper's
  atom-movement convention. Both would have had the search optimise a circuit or a cost
  that the figure does not report. The same distinction survives in `analytic_permutation`,
  where `translated_view` handles the preprocessing rather than the caller.
- Duplicate rows in the CSV. Rows are appended per group so an interrupted run keeps what
  it has measured, which means two runs sharing a `--csv` path merge instead of one
  overwriting the other — the startup `unlink` only fires for whichever started later. A
  collision during this revision produced a file holding the large group twice, which does
  not move the per-group means but weights that group twice in the pooled table. `read_csv`
  now de-duplicates on `ROW_KEY`, keeping the most recent measurement.

## Next

The binding constraint is no longer merging's existence but its *reach*: the whole-patch
macros need a gate on every occupied lane at once, and the firing rate falls from 58/99
circuits at k=2 to 41/99 at k=10. Recovering the near-misses means transversal-then-correct,
which the cost table says pays only within about k/3 lanes of full alignment, and which
therefore also asks the packer for a different objective — near-alignment of whole patch
pairs rather than the analytic cost it now hill-climbs on. Plan §5.7 has the costing
argument; §5.8 notes the cheaper adjacent win, which is that the pass flushes its candidate
group at a program SWAP instead of tracking the relabel through it.

Note also that `grouping/tune.py` selected `depth_weight = 12` *before* `best_placement`
added analytic refinement, so the two free parameters were fitted against an older
pipeline. Re-running the grid search is cheap now and has not been done.

Adding LSR and TVD rows needs the simulation harness from `density_ablation_sweep.py`, whose
alpha calibration and repetition scaling transfer directly. The simulator's per-chunk
reseeding bug, which gave seeded runs `repetitions/100` effective samples, is fixed as of
this revision (`tests/test_simulator_seeding.py`), so shot counts can now be taken at face
value.
