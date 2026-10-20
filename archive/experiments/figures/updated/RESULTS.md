# Rebuilt figures: what the numbers say

Everything here was produced by the scripts in this folder on the current compiler.
The originals are Figures 5, 10, 11, 12, 13 and 17 of `AQORN_arXiv.pdf`; `fig_k_sweep`
is new.

## The depth table, and why the packer was underperforming

Depths are now **measured** from the emitted circuits (`iceberg.costs.depth_table`,
pinned in `tests/test_iceberg_costs.py` exactly as the work table is). Every entry is
**constant in n**: HH is one transversal Hadamard layer plus the anchor swap, and the
transversal two-qubit macros are n gates in parallel.

| op | intra | cross | transversal (all k lanes) |
|---|---|---|---|
| CX | 4 | **3** | **1** |
| CZ | 3 | 3 | **1** |
| XCX | 6 | 5 | **3** |
| H | 8 | — | HH = **2** |

This exposed the bug. Gate counts and depth disagree about where a two-qubit gate
wants to live, and for CNOT **they disagree in opposite directions**:

    CX   gates: intra (0,4) == cross (0,4)  ->  0, no signal at all
         depth: intra 4     >  cross 3      -> -1, prefers to SPLIT

So on gate counts alone every CX edge weighed exactly zero. A CNOT-dominated circuit
produced a nearly empty interaction graph and an essentially arbitrary partition —
which is why the CX-heavy benchmarks (ghz, cat, bv, adder) showed negligible or
negative packing benefit. Cut minimisation, the natural reading of "put interacting
qubits together", was pushing the wrong way for every CNOT in the circuit.

Three fixes followed, each measured against exhaustive search over placements:

1. **Price depth in the edge weights.** CX now weighs −3 (splits), CZ +12 (groups).
2. **Add a transversal-affinity term.** A transversal macro is depth 1 and costs
   `(0, n)` while doing k gates' work, so it wins on both axes — but it needs *all* k
   lanes of two patches paired. What that rewards is neighbourhood similarity, not
   adjacency: two qubits belong together when their partners can also be together.
   Discounted by a confidence factor, since one coincident pair is only 2/k of a macro.
3. **Refine on the analytic cost, not the graph.** The graph is a pairwise proxy for a
   conjunctive objective and cannot express it. Against exhaustive search the seed
   alone captured half the available gain at k=2, and on `simon_n6` at k=4 it found
   **0% of a 40% improvement**. Hill-climbing on the analytic cost — affordable only
   because it is now ~0.1 ms — closed most of that gap.

Measured capture against exhaustive optimum at k=2: **5.58% of 7.23% available, 67%**,
up from 3.60%. `qaoa_n6` went 4.0% → 16.9% (100% of optimum); `simon_n6` at k=4 went
0% → 20.5%.

A fourth change is a safety property rather than an improvement: the model over-counts
merging by 2–3%, and on `vqe_n4` that was enough to make it pick a placement 8% *worse*
than naive. The final candidate is now checked against the naive one by exact
measurement and the better is kept — two translations, negligible, and it turns "usually
better" into **never worse**, which is what makes the packing bar safe to present.

## Figure 10 — QASMBench, 71 circuits, [[4,2,2]]

| | 1q gates | 2q gates | depth |
|---|---|---|---|
| small (29) | 24.0% | 18.1% | 28.4% |
| medium (15) | 26.5% | 18.4% | 31.4% |
| large (27) | 20.7% | 15.7% | 37.7% |
| **overall** | **23.2%** | **17.2%** | **32.5%** |

The draft reports 31% / 17% / 34–35%. **Two-qubit gates and depth now match**
(17.2 vs 17, 32.5 vs 34–35); one-qubit gates remain lower (23.2 vs 31).

The remaining 1q gap is not root-caused and should not be quoted as like-for-like.
Two candidates, in the order I would check:

1. **The stacking convention changed** (below). The old convention could overstate a
   total whenever one technique hurt and a later one recovered.
2. **The six correctness fixes** from the earlier branch changed what the compiler
   emits for any circuit containing a SWAP, an `sx`, or an `xcx`.

## A presentation bug worth knowing about

The draft's scripts clip each stack *segment* at zero. That breaks when one technique
hurts and a later one more than recovers. On `ising_n98`, H commutation costs 88% of
depth and merging returns 179%; the true full-pipeline reduction is 90%, but dropping
the negative segment leaves a **179% bar on a percent-reduction axis**. Several such
bars were visible in the first rebuild of Figure 10.

These scripts clip the *cumulative* series instead and difference that
(`common.stack_heights`). Segments stay non-negative, the stack height is the real
reduction, and a technique that hurts contributes zero instead of inflating the next
one. Raw per-segment values are still written to the CSV.

This affects the published Figure 10 and Figure 13, which use the old convention.

## fig_pass_order — the new attribution figure

The draft argues, of Figure 10: "If merging and packing were applied before
commutation, their relative contributions would increase accordingly." That is now
measured. All 71 benchmarks, cumulative against no techniques at all:

| condition | 1q | 2q | depth |
|---|---|---|---|
| Merging alone | 12.9% | 8.9% | 17.5% |
| + Packing | 13.4% | 10.9% | 22.4% |
| + Commutation **first** (paper order) | **23.5%** | **17.3%** | **31.9%** |
| + Commutation **last** | 18.4% | 15.2% | 18.1% |

Two findings:

**Merging and packing alone are worth 13.4% / 10.9% / 22.4%** — far more than the
5–10% the draft attributes to them when commutation runs first and takes the Hadamards
they would otherwise have fused. The draft's conjecture is right, and this quantifies
it.

**Commutation-last is worse than commutation-first overall** (18.4% vs 23.5% on 1q,
18.1% vs 31.9% on depth). The mechanism is concrete: once merging has run, the circuit
contains macro gates, and `commute_hadamards` flushes its frame at any instruction it
does not recognise — so macros act as barriers and commutation simply finds less to do.

All three orders are verified to preserve the logical action exactly, at n ∈
{4,6,8,10,12}, on programs built to trigger every macro (25/25, zero mismatches).

## fig_k_sweep — the new one

> **Rerun twice.** The first version reported medium best at k=4 on circuits that were
> silently dropping rejected gates; the second retracted it on a survivorship-selected set,
> because a third of the benchmarks then refused to place at k>2. Those refusals were a
> vestigial restriction on misaligned cross-patch gates (STATUS.md §4.3). Every circuit now
> places at every code size.

| k | medium 1q | medium 2q | medium depth | placed | merging fired |
|---|---|---|---|---|---|
| 2 | 26.5% | **18.4%** | 31.4% | 15/15 | 10/15 |
| 4 | **26.7%** | 17.3% | 38.7% | 15/15 | 11/15 |
| 6 | 24.8% | 17.1% | **39.3%** | 15/15 | 11/15 |
| 8 | 23.7% | 16.5% | 39.1% | 15/15 | 9/15 |
| 10 | 23.2% | 16.3% | 38.2% | 15/15 | 8/15 |

**Depth gains 7–8 points past k=2** and holds there. Gate counts do not follow: two-qubit
reduction declines steadily and one-qubit is flat to k=4 then falls. The merging column is the
mechanism for that half — whole-patch macros need a gate on every occupied lane at once, and
the firing rate goes 10/15 → 8/15.

Small circuits fall monotonically (24.0% → 12.7% on 1q) because they run out of patches: at
k=10 a 6-qubit circuit is a single patch and there is no placement decision left. Structural,
not algorithmic.

## Figure 12 — random circuits

| | 1q | 2q | depth |
|---|---|---|---|
| Small | 14.6% | 16.1% | 26.5% |
| Medium | 14.7% | 13.1% | 31.9% |
| Large | 13.8% | 12.5% | 32.4% |

Depth reduction is *larger* than on the benchmarks and gate reduction smaller,
which inverts the draft's framing. Worth a second look: the draft argues random
circuits have fewer immutable gates and so more merging opportunity. These runs use
`--immutable-ratio 0.08`; the conclusion is sensitive to that knob and I did not
calibrate it against the draft's generator.

## Figure 11 — improvement over random permutations

| sweep | density | 1q | 2q | depth |
|---|---|---|---|---|
| qubits | tightest | 4.6% | 4.1% | 3.6% |
| qubits | loosest | 0.1% | 0.9% | 5.4% |
| gates | tightest | 4.0% | 4.1% | 3.3% |
| gates | loosest | −6.2% | −4.4% | 8.1% |

Gate reductions are greatest at the tightest packing and vanish as density → 1,
reproducing the draft's shape. The **negative** gate numbers at the loosest density
are real and worth explaining: the placer minimises weighted gate count but the
comparison at low density is against random permutations that happen to spread work
differently, and with almost every qubit solo there is nothing left to optimise.

## Figure 13 — ablation over density (gate-only run)

| technique | 1q | 2q | depth |
|---|---|---|---|
| H commutation | 12.8% | 4.2% | 8.1% |
| + Merging | 3.3% | 3.2% | 8.0% |
| + Packing | **14.3%** | **13.6%** | 4.2% |

Packing now contributes *more than H commutation* on both gate metrics, which inverts
the draft's finding that commutation dominates everything. That is the depth-aware
placer plus analytic refinement. The LSR/TVD rows need the simulated run
(`fig13_ablation_sweep.py` without `--no-sim`), which was not completed in this
session.

## Figures 5 and 17 — simulated

Both scripts run and were smoke-tested, but only at reduced settings; the
publication-scale runs were not completed.

Figure 17 reproduces the headline claim even at the smoke-test size: LSR and TVD
anticorrelate at **Pearson −0.913 / Spearman −0.906** over the alpha sweep. Figure 5
produces the heatmaps; at the coarse default grid the three CNOT implementations sit
within 0.001 fidelity of each other, so the grid needs raising before the comparison
means anything.

## Not done

- Simulated runs of Figures 5, 13 and 17 at publication settings.
- Root-causing the Figure 10 gap against the draft.
- The `--n-code` sweeps of Figures 10, 12 and 13 (the machinery is there; only k=4
  was run).
