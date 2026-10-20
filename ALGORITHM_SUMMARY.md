# AQORN algorithm summary — handoff for the strategies section

Written to support outlining the paper section that describes what AQORN actually does.
Every step below is grouped by the module it lives in, and tagged with the figure panel
that will carry it. Numbers in *italics* are measured results already recorded in the
code's docstrings, so they are citable without re-running anything.

**The three figures do not exist yet.** They are at mockup stage; treat the tags as a
plan for what each panel must carry, and write the prose so it survives a panel moving.

---

## Read this first: three structural facts

1. **Two entry points answer two questions; a third composes them from above.**
   `packing.plan_packing(program, n_code)` answers *"given a code size and a patch
   count, which permutation?"*. `packing.resource.select_resources(program)` answers
   *"which code size and how many patches?"*. `select_resources` sits **above**
   `plan_packing` and **never calls it** — it runs its own thinner search internally.
   Do not write a sentence implying resource selection feeds the packer.

   `packing.compile_circuit(program)` is the composition and sits above both: resource
   stage, then the placement search at the code size that came back, then an exact
   translation of both placements plus naive, ranked on the resource objective. It is a
   caller, not a fourth algorithm. What earns it prose is *why* it has to measure rather
   than keep the packer's answer — the two objectives rank the same placements in nearly
   opposite orders — see the compiler subsection below.

2. **The packer contains a *prediction* of merging.** Merging is a later pass inside
   translation. `packing.cost` estimates what merging will fire using three timeline
   predicates (`_coincident_layers`, `_fusable_group_sizes`, `_complete_bijections`)
   without running the merging pass. It over-counts by *2–3%*, and the `verify` stage
   exists to catch the cases where that matters.

3. **Two objectives, two roles.** Cut weight on the signed interaction graph guides the
   *seed*; the true analytic cost drives everything *downstream* of the seed. This is
   the section's central design claim, and it is counterintuitive on purpose: a worse
   description of cost makes a better search guide, because cut weight is a pairwise
   relaxation that moves the partition coherently while the true objective is
   conjunctive (a macro needs *every* lane) and its landscape is plateaus and deep
   local optima.

---

## Figure map

| Figure | Panels | Carries |
|---|---|---|
| **Fig 1** — pipeline overview | single row | the three structural facts above |
| **Fig 2** — packing | (a) input conditioning, (b) signed graph, (c) multilevel V-cycle, (d) descent + lanes | the two-objectives claim; paired objective tags in (c)/(d) |
| **Fig 3** — merging | (a) coverage / two macro families, (b) alignment, (c) scheduling | why generalising past k=2 required a notion of coverage |

Steps marked **→ prose only** have no panel and must be carried by text.

---

## `packing/` — the placement search

Entry point: `plan_packing` in [plan.py](packing/plan.py). Runs in this order.

### 0. Commutation arm split — `plan.py:308`
With the default `h_commutation="auto"`, **the entire pipeline below runs twice**, once
with H commutation on and once off, and both arms are translated for real; the cheaper
placement wins. The arms are not comparable on model cost — each is priced against its
own commuted view, so they describe different circuits.

Why not decide first: `passes.commutation_pays` answers the same question on the *naive*
packing, and the two placements disagree. *On `dnn_n51` at k=2, naive prefers the pass
off (42,160 vs 42,768 weighted) while the packed placement prefers it on (41,918 vs
42,486)* — a 1.3% error, and it left `verify` falling back to naive on a circuit where
the packed arm was in fact better. The passes interact through the partition, so the
decision cannot precede it.

→ **Fig 1** (`×2` badge on the packing box)

### 1. Translated view — [tables.py:137](packing/tables.py:137)
Run the translator's *program-level* passes first: `cancel_inverse_gates`, then
optionally `commute_hadamards` + cancel again. These do not depend on placement, so they
apply once and are reused for every candidate.

Load-bearing because every cost table is keyed by gate **kind** and the translator
rewrites kinds — a CX with a commuted Hadamard on its control becomes an XCX. *Scoring
the raw program instead was worth 13–49% error.*

→ **Fig 2(a)**

### 2. Per-qubit profiles — [profiles.py:107](packing/profiles.py:107)
Each logical qubit gets a *frame* (which gate kind, layer by layer) and a *cast* (which
partners, in which layer). Layers come from `translator.merging.layer.layer_by_node` —
the merging pass's own layering — so "same layer" means here what it means there. Gates
on ≥3 qubits are dropped: no macro exists for them and the translator would reject them.

→ **Fig 2(a)** (the profile grid)

### 3. The objective — [cost.py:199](packing/cost.py:199)
`PackingCost` is a function of the **partition alone**, not the placement. Slot order
enters no term, which shrinks the search space by a factor of k!.

```
cost = solo                                    constant (single-qubit gates)
     + P × (ENCODE + DECODE)                   per live patch
     + Σ_patches  unary(A)                     intra-patch gates − HH/Pauli fusion
     + Σ_pairs    binary(A,B)                  cross-patch gates − transversal macros
     + movement(crossings, P) + idle(depth, P)  global, from the NoiseProfile
```

Three constants worth naming in the paper:

- **`DEFAULT_DEPTH_WEIGHT = 12.0`** weighted-gate units per layer. Not optional: intra-
  and cross-patch CNOT cost identical gates, so at zero the objective cannot see CNOT
  placement at all and a CX-dominated program carries no signal. *Beware:* `graph.py`
  has a **separate** `DEFAULT_DEPTH_WEIGHT = 3.0` for edge weights. Two different
  constants; do not conflate them in prose.
- **`PARALLEL_SHARE = 0.5`** — depth is estimated half from summed per-gate depth
  (nothing runs concurrently) and half from the busiest patch's load (everything that
  can, does). The max term is what makes load *balance* visible: a sum is unchanged by
  moving gates between patches. *Rank agreement against exact translation at fixed patch
  count — structured / QASMBench corpora: sum-only 0.02 / 0.19, half-and-half 0.42 /
  0.24, busiest-only 0.41 / 0.16.* Half-and-half is the only setting better than the sum
  on both.
- **`MOVEMENT_SHARE_CEILING = 0.25`** — movement is clamped to a share of the gate+depth
  work. Uncapped, at an idle/distance ratio of 1/16 the term becomes the entire
  objective, the search minimises crossings alone, and since program order already nearly
  minimises crossings the descent returns naive. *At ratio 1/16 uncapped, 60 of 76
  proxies came back metrically identical to naive, against 23 of 76 at ratio 16.* Note
  the clamp **never binds at the machine's own settings** — it is a guard for the
  stressed-distance end, not a tuned parameter.

Everything is cached per group membership and every candidate move is scored by exact
delta over the few terms it touches (`Packing._delta`), which is what makes the true
objective affordable everywhere instead of a graph proxy.

Agreement with the older `grouping.analytic.AnalyticCost` is asserted in
`tests/test_packing.py` on random partitions across code sizes — it is a faster spelling
of the same number, not a second opinion.

→ **prose only** (the cost equation belongs in text; no panel)

### 4. Slot count — `plan.py:208`
`slots = ceil(m/k) + spread`, clipped by `ResourceBudget.patch_ceiling`. `spread`
defaults to 0. `k = n_code − 2`; physical qubits = `patches × (n_code + 2)`.

**There is deliberately no patch-count scan.** The old stack scanned five candidates;
*41 of 63 QASMBench circuits returned one distinct count and the tightest won all 63*.
The count is not a separate decision — the `spread` move in the descent raises it when it
pays, and the objective charges `ENCODE + DECODE` per live patch, so the descent settles
it. Empty slots are free.

→ **prose only**

### 5. Signed interaction graph — [graph.py:179](packing/graph.py:179)
Four effects summed per edge: what co-location saves in gates, what it saves in depth,
Hadamard-coincidence affinity, and transversal affinity. Plus
`noise.movement_cost × distance_scale` on every two-qubit edge.

**Signed is the point.** On gate counts alone every CX edge weighs exactly zero, and on
depth it is −1 per gate — a CX edge wants to *split*. Counting every crossing equally
(as a plain interaction-count graph does) invents a preference the cost table does not
have and points it the wrong way; that is why cut minimisation produced negligible or
negative benefit on the CX-dominated benchmarks (ghz, cat, bv, adder).

**Transversal affinity is neighbourhood similarity, not adjacency** — two qubits belong
together when their gates, in the same layer and of the same kind, point at *different*
partners, because then they can be two lanes of one side of a macro. It is the term a
cut-minimising weight cannot express, and for CNOT it is the only term carrying signal.
Discounted by `DEFAULT_TRANSVERSAL_CONFIDENCE = 0.25` because a coincident pair is only
2/k of the way to a macro; at full confidence it pulled qubits *across* cleanly separated
CZ blocks and discarded a strictly cheaper all-intra-patch placement.

Honest limitations to state: the Hadamard-coincidence term makes the graph **complete**
on Hadamard-rich programs — *9730 of 9730 possible edges on `bv_n140`, every one
attracting*, which is no grouping signal at all. Both affinity terms are pairwise
stand-ins for conjunctive effects, so they over-reward; they only have to get the
*ordering* of candidate merges roughly right.

→ **Fig 2(b)** — the panel must show one negative edge and one positive edge between
qubits with **no gate between them**, or a reader will assume it is an adjacency graph.

### 6. Multilevel partition — [partition.py:468](packing/partition.py:468)
Coarsen → solve the small graph → uncoarsen with refinement at each level. Objective
throughout: **cut weight**. Defaults: `coarsen_to=24`, `max_levels=24`, `max_passes=4`,
and `DEFAULT_PARTITIONER = "feasible"` = `{matching: equal, base: merge, swap: False}`.

**Feasibility is an invariant, not a repair.** The coarsener matches only nodes of
*equal* size and caps that size at the largest power of two dividing k:

| k | 2 | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|----|----|
| cap | 2 | 4 | 2 | 8 | 2 | 4 |

Every node size is then a power of two dividing k, which makes `pack_bins` provably
overflow-free (proof in the module docstring). Why it matters: the old coarsener built
4-qubit super-nodes, and 4 does not tile 10 — *on `bv_n70` at k=10 the old base case
stranded at sizes [10,8,8,8,8,8,8,8,4], 9 groups against a target of 7*, and before a
repair pass existed *`bv_n140` came out 16% worse than naive*. Pairing legacy heavy-edge
matching with `pack_bins` produced *11 rejected placements* across the sweep; with the
invariant, zero.

Refinement is Fiduccia–Mattheyses-style with incremental gain bookkeeping, relocating
nodes (and optionally exchanging them). **Exchange measured worse and is off by
default** — *0.6 points at k=6 and 0.8 at k=10*. The reason is instructive and worth a
sentence in the paper: exchange minimises the *cut*, and the cut is a loose proxy
(*where FM fires it moves the cut 1.5% and the true cost 10–20%*), so a stronger cut
minimiser overfits the proxy rather than finding a better placement.

Seed partitioner arms, *mean improvement against naive over 64 QASMBench proxies at
n = 4/6/8/12, full pipeline with `verify` on*:

| arm | n=4 | n=6 | n=8 | n=12 |
|---|---|---|---|---|
| `feasible` (default) | −4.88 | −5.20 | −5.37 | −4.48 |
| `swapped` | −4.88 | −5.27 | −4.76 | −3.69 |
| `packed` | −4.77 | −4.13 | −3.86 | −3.23 |
| legacy multilevel | −4.88 | −5.20 | −5.18 | −4.25 |

`seed.graph_seed` then tidies: split anything over capacity, home any qubit the
partitioner dropped, pad out to `slots` empty patches. The padding is what lets the
descent's `spread` move raise the patch count later.

→ **Fig 2(c)**, objective tag `cut weight`

### 7. Descent on the true objective — [search.py:374](packing/search.py:374)
Sweep qubits in index order. For each, enumerate every admissible change of
**membership** and apply the best strictly-improving one; repeat until a sweep changes
nothing (`max_rounds=8`). Three move shapes:

- `move` — q joins a patch with room
- `spread` — q leaves for an empty slot, raising the patch count
- `swap` — q and r exchange patches; **the only move available once every patch is
  full, which is exactly what the tightest packing is**

**There is deliberately no lane move.** The objective cannot read lanes, so lane
permutations are provably zero-gain — *they were 72 of 109 candidate moves on `qpe_n9`,
scored on every round* of the old stack.

Per-qubit best-improvement rather than one globally-steepest move per round: a
steepest round costs a full O(m²) scan and buys a single move, so it needs as many
rounds as it makes moves — *at m=64 that was still improving when it hit a 12-round
cap*. Terminates because every accepted move strictly lowers a bounded objective.

This stage is where the improvement comes from: *on `knn_n31` at k=6 the seed alone is
5.3% better than naive and the descent takes it to 14.6%.*

Two negative results that justify the design and are worth citing:
- **A graph-free descent does not work.** *On `knn_n31` and `swap_test_n25` at k=10,
  plain descent from any cheap seed leaves the naive partition untouched — zero
  improving single moves — while a partition 13% cheaper by that same objective exists.*
- **Kernighan–Lin was removed.** Once the graph seeds the search, *KL changed the outcome
  on 1 of 8 hard circuits with a patience cutoff and 2 of 87 without, at 3–6× and 15× the
  cost*; and every circuit it helped was one where the model mis-ranks, so `verify` was
  already returning naive there. With the safety net on, KL changed nothing at all. A
  good seed removed the need for the escape hatch.

→ **Fig 2(d)** top half, objective tag `true cost` (paired with 2(c))

### 8. Restarts — `plan.py:232`
The **naive seed is always descended too**, at any width. The graph seed optimises cut
weight, which tracks the objective badly when noise prices movement: *at
`movement_cost=52` the descent from the graph seed lands above naive on its own
objective on `ising_n26` (18,460 against 17,758) and cannot climb out*, because a 1D
chain is a partition naive already nearly solves. Seeding from naive costs one descent
and makes "no worse than naive" hold **in the model** rather than only after `verify`
re-measures it (*`ising_n42` at ratio 1: 29,953 → 29,485*).

Shuffled restarts are size-gated: 4 of them when `num_qubits ≤ 16`, none above. *On
small circuits the basins are few, so which one the seed lands in decides the answer —
restarting recovered `sat_n7` from 1.6% better than naive to 6.7%, at 4 ms. On large
circuits they changed nothing measurable and cost 2–3×.*

→ **prose only**

### 9. Lane assignment — [lanes.py:32](packing/lanes.py:32)
Decided **once, at the end**, since the objective is a function of the partition. Not
free, though: a transversal macro whose control-slot-to-target-slot map is not the
identity is realigned by permuting the target patch, and those routing swaps displace
gates in time, which can stop a later macro firing.

One pass, patches in descending cross-patch traffic order, each solved exactly as a k×k
linear assignment (`scipy.optimize.linear_sum_assignment`) against **every already-placed
patch**. That last clause is the difference from the older `align_slots`, which built its
benefit matrix from gates to a single anchor patch — *on `sat_n11` that discarded 38 of
150 relevant gates*. Same exact solve, more of the information.

→ **Fig 2(d)** bottom half

### 10. Verify — `plan.py:325`
Translate the packed placement and the naive packing for real; keep the cheaper. Wired
in **by default**, unlike the old stack.

The justification is that the model is a model — it over-counts merging by 2–3%, which
*on `vqe_n4` at k=2 was enough to prefer a placement 12.9% worse than naive*. Two
translations turn "usually better" into "never worse", which is what makes the packing
contribution safe to quote as a gain. The plan records `verified` / `fell_back` and the
full `considered` list, so the fallback rate is reportable.

→ **Fig 1** (the `verify` diamond)

### Supporting modules
- **[budget.py](packing/budget.py)** — `ResourceBudget(max_physical_qubits, max_patches)`;
  converts a hardware limit into a patch ceiling. → prose only
- **[passes.py](packing/passes.py)** — `commutation_pays`, which decides H commutation by
  translating both ways on the naive packing, cached by circuit *content*. No longer used
  by `plan_packing` (superseded by the arm split in step 0) but still used by
  `resource/select.py` and the figure scripts. Its measurements are the clearest evidence
  that the commutation decision must be measured rather than estimated: *`adder_n64`,
  112 of 112 Hadamards absorbed, saves 11,272 units; `ising_n66`, 0 of 66 absorbed, costs
  422; `dnn_n51`, 52 of 52 absorbed and still costs 8,627, because the damage is in the
  schedule, not the gate list; `qft_n29`, 29 of 29 absorbed by count but none in effect,
  costs 856.* The cheap program-level estimate `commutation_gain` *agreed on only 21 of
  28 cases, and every disagreement was a false positive.* → prose only
- **[tables.py](packing/tables.py)** — cost tables, the `mean_patch_hops` geometry
  (reproducing the simulator's own layout convention so the model predicts the distances
  the noise model will charge), and the three conjunctive timeline predicates that stand
  in for the merging pass. `DEFAULT_WINDOW = 2` layers of tolerance, matching the merging
  pass's willingness to delay a gate. → **Fig 1** (the dashed prediction arrow)

---

## `packing/resource/` — choosing the code size and patch count

Entry point: `select_resources` in [resource/select.py](packing/resource/select.py:200).
**Not called by `plan_packing`** — it sits above it. `packing.compile_circuit` composes
the two (next subsection); the `code_size_crossover` and `resource_selection_headroom`
figures call `select_resources` directly.

### Why it cannot live inside the packer
Three mechanisms inside `plan_packing` are each independently monotone toward the
tightest packing: `pack_bins` scores affinity first and an empty bin scores zero;
`refine` minimises cut weight and spreading raises the cut; `descend` charges
`ENCODE + DECODE` per live patch with no counterweight when the noise coefficients are
zero. Each is right for its own objective, and together they mean *`spread=8` produced
an identical patch count on every circuit tried*. So patch count is an **output** of the
packer's objective, not a decision it makes.

The packing objective is also nearly flat in the two knobs that matter: *on `ising_n26`
gate counts move 6% across k = 2..10 while the space-time volume moves 30%.*

→ **Fig 1** (the dashed off-path branch)

### The objective — [resource/noise.py](packing/resource/noise.py)
One quantity: the expected number of **detectable** errors in a shot.

```
E = n1·p1 + n2·p2                             gates
  + V · t_layer · r,  V = P·(n+2)·depth       idling through the schedule
  + M · (moved·p_move + trip·P·(n+2)·r)       grouped cross-patch movements
  + I · (p_move + trip_in·P·(n+2)·r)          intra-patch beyond-blockade carries
```

`r` is the twirled thermal rate the simulator itself applies. Survival under
post-selection is `exp(−E)`, so minimising E maximises accepted shots. Defaults are
`simulator.noise_models.NEUTRAL_ATOM` verbatim.

Two points worth prose:
- **Movement is priced per grouped movement, not per physical gate** — one cross-patch
  logical gate is one round trip of its control side (two atoms), and the idling that
  trip forces on the whole array is the dominant cost, so a 16-patch layout pays more per
  trip than a 4-patch one twice over.
- **The intra-patch beyond-blockade term is what keeps large patches honest** — the
  fraction of a patch's own pairs beyond the blockade radius is fixed by the code size
  (*0.00 at k=2 rising to ~0.27 at k=10*), so it belongs exactly here, where code sizes
  are ranked; the placement search cannot change it and never sees it.

**Two objectives, not one.** Fewer, fuller patches win on E. More, emptier patches win
on *patch density* — more detectors per logical qubit, fewer logical qubits behind each,
so an error that slips through corrupts less. *On `ising_n26` at k=4, going from 7 patches
to 26 costs 12× the accepted shots and buys 4× fewer corrupted logical qubits.* There is
no objective-free way to combine them; `density_weight` is the exchange rate and is an
**application** property, not a machine one, so its only defensible default is 0.

### The sweep — [resource/select.py](packing/resource/select.py:200)
1. Decide commutation once per code size (`commutation_pays`) and extract profiles once
   per code size — the expensive step, independent of both knobs, so it is hoisted out.
2. For each of `DEFAULT_CODE_SIZES = (4, 6, 8, 10, 12)` × ~5 sampled patch counts
   (tightest, loosest, and quartiles between), build a partition with `_forced_groups` —
   affinity-greedy with a **balance cap of ceil(m/patches)**, seeded from the
   highest-degree qubits. The cap is the whole trick: without a floor on occupancy the
   greedy assignment piles everything into the first bins and the patch count silently
   collapses back to tightest.
3. Assign lanes, tally crossings and depth from the cost terms, score with `estimate`.
4. **Translate every admissible candidate for real** (`DEFAULT_MEASURE_TOP = None`).
5. `_refine_and_verify` on the winner: add a `descend` arm and a naive arm, translate all
   three under the same commutation decision, keep the best on the resource objective.

Two design notes the paper should state plainly:
- **The model generates candidates; it does not rank them.** An earlier version ranked on
  the packing model and translated only the top few, and it was wrong: *on `ising_n26` it
  returned [[6,4,2]]×7 when the best admissible point is [[4,2,2]]×13, which has 31%
  better survival.* The cause is that `PackingCost` tallies *summed* per-gate depth while
  the volume term needs the critical path, so it systematically penalises the spread
  placements that win here. It is wrong in both directions in absolute terms too
  (*`qft_n18` 918 against a true 768; `knn_n25` 336 against 380*).
- **`_refine_and_verify` is not optional.** Without it the returned placement had skipped
  both local search and the naive comparison, and *on QASMBench small at the chosen shape
  it was worse than naive on 4 of 8 circuits (`qpe_n9` +2.5%, `simon_n6` +7.5%,
  `adder_n10` +2.6%, `hhl_n7` +3.3%).*

### Stated trust boundaries — quote these, do not overclaim
- `exp(−E)` is a Poisson approximation assuming every error independent and detectable.
  *At the measured E of 3–5 on QASMBench proxies it is an ordering, not a rate.*
- `corrupted_qubits` uses `E²/(2P)` for the weight-2 failures a distance-2 code misses,
  and that expansion is only valid for E ≪ 1.
- Both exist to **rank** candidates. `iceberg_postselect` returns a real selection rate,
  and calibrating against it is what would make either number quotable.

→ **prose only** for the method; the *content* is already carried by the existing results
figures (`6_packing_density_tradeoff`, `code_size_crossover`,
`resource_selection_headroom`). No new method panel planned.

### `compile_circuit` — the two stages composed — [compiler.py](packing/compiler.py)
Resource stage, then `plan_packing` at the code size that came back, then an exact
translation of the planner's placement, the selector's placement and the naive packing,
ranked on the resource objective. The winner's translation is the returned circuit.

**Why the ranking is by measurement and not by "the packer ran last".** The two stages
minimise different things and disagree about the same placements. Measured on `knn_n25` at
the `[[8,6,2]]×5` shape the selector chose:

| placement | weighted gates + depth | expected detectable errors |
|---|---|---|
| planner | 13,304 | 2.2870 |
| naive | 13,584 | 2.2892 |
| selector | 13,700 | **2.2759** |

*Nearly opposite orders: the planner's placement is best on its own objective and middle
on the objective the code size was chosen for. On `ising_n26` at `[[6,4,2]]×7` the planner
wins gates-and-depth 6,668 to 6,716 and loses E 1.400 to 1.341, so keeping its answer
would have cost 4.4% on the objective the resources were chosen for.*

**The patch count is honoured by the arm comparison, not by the packer.** `spread` is
passed but does not bind: every term in `PackingCost.price` is non-decreasing in the patch
count except the `busiest` half of the depth estimate, and 234 weighted units per patch at
n=6 outbids it. *Measured at k=4 on `ising_n26`, `knn_n25` and `qft_n18`: the descent
returns the tightest packing under a zero profile, under the profile derived from
`NoiseKnobs`, and under that profile with idling priced 100× and 10,000× higher.* So at the
default `density_weight=0` both stages sit at the same (tightest) shape and the composition
is a placement contest; above zero the selector spreads, the planner cannot follow, and the
selector's arm wins on E. Closing that needs a density term in `PackingCost`. This is the
same wall §"Why it cannot live inside the packer" describes, seen from the other side.

**One machine, two unit systems.** `packing.resource.noise.packing_profile` converts
`NoiseKnobs` (error probabilities, seconds, metres) into a `NoiseProfile` (weighted-gate
units) so the planner is charged what the selector ranked on. Without it the planner prices
movement and idling at zero — `NoiseProfile`'s defaults — while the stage above it ranked
code sizes on exactly those terms. `experiments.shared.simulation.noise_profile_for` is now
the `SimSettings` adapter over that one conversion.

→ **prose only**, and only a paragraph: it is the composition, not a contribution.

---

## `translator/merging/` — what the placement is for

Runs inside translation, after placement. Given a permutation it knows which program
qubits share a patch and which lane each occupies, which is what makes macro
substitution decidable. Default `num_merge_passes = 2`.

### Two macro families, two admissibility rules
This is the section's cleanest technical point, and the reason generalising past k=2
required real work.

**Whole-patch** — `hh`, and transversal `cxt` / `czt` / `xcxt`. Built from physical gates
applied to all n data qubits, so they act on **every** lane whether or not a program gate
asked. A macro is a legal substitution only when it covers every *occupied* lane, so at k
slots it takes **k program gates to make one macro**. Fusing a subset is not merely
unprofitable but **inexpressible**: partial transversal CX sends the full-weight
stabilizer `S_X = X^⊗n` outside the stabilizer group, so it is not a logical operation.
Coverage is all-or-nothing, and is checked twice — at detection and again against the live
slot map at translation time, since a SWAP may have moved a qubit in between.

**Per-lane** — `xx`, `zz`. `X̄_i = X[x_anchor]·X[lane i]`, so any subset of lanes is
legal and an even-sized subset cancels the anchor factor outright. That cancellation *is*
the saving. No coverage condition.

*At k=2 the two rules coincide, since a group of two lanes **is** the whole patch.* That
is why the original pairwise implementation needed no notion of coverage, and why
generalising was mostly a matter of introducing one.

→ **Fig 3(a)**

### Alignment
A transversal group's control-slot → target-slot map is a permutation of k slots, and
only the identity can be emitted directly. Anything else is fixed by physically permuting
the target patch, decomposed into transpositions and emitted as `RoutingSwapGate`s around
the macro. A routing swap is an atom exchange — two carried atoms at `move_error` each —
so **realignment is not free**: `_macro_gain` prices the macro against the singles it
replaces minus its swaps, and a macro that does not pay for itself is not taken. Coverage
plus a positive gain plus surviving scheduling is the whole test.

Note for consistency: `packing.cost._transversal` charges realignment **in expectation
over a uniformly random alignment** (a random permutation of m lanes needs `m − H_m`
transpositions), which is what keeps the packing objective lane-order-invariant while
still pricing what realignment costs on average. The two modules agree on the mechanism
and differ only in that one knows the lanes and the other deliberately does not.

→ **Fig 3(b)**

### The pass pipeline — [preprocessing.py](translator/merging/preprocessing.py)
Per pass, in order:

| Step | Module | What it does |
|---|---|---|
| 1 | `layer.layer_by_node` | earliest layer + order per instruction |
| 2 | `_build_patch_lanes` | permutation → patch → occupied lanes, in slot order |
| 3 | `single_qubit.find_*` | propose `hh` / `xx` / `zz` groups within a delay budget |
| 4 | `selection.select_best_group` | pick a non-conflicting subset of those proposals |
| 5 | `transversal.find_*` | propose `cxt` / `czt` / `xcxt`, honouring slot alignment |
| 6 | `conflicts.filter_*` | drop single-qubit picks the transversal picks reorder |
| 7 | `rewrite.build_rewritten_circuit` | emit the circuit, delaying gates the macros displace |

Steps 4, 6 and 7 are the scheduling story, and they are where the packing model's
**+2–3% over-count** comes from: the predicates in `packing.tables` model coverage and
timing windows but none of the scheduler. Say this explicitly — it is what motivates
`verify`, and it closes the loop back to Fig 1.

→ **Fig 3(c)**

### One caveat if `PairMergeAnalyzer` comes up
It is a **k=2 estimator and deliberately still one** — it runs the detectors on a
hypothetical two-lane patch. Its counts are an upper bound at k > 2, usually a loose one,
because the detectors it calls require whole-patch coverage. Do not cite its numbers as
predictions for larger codes.

---

## Suggested section skeleton

1. **Problem and the two questions** — permutation given resources vs. choosing
   resources. Establishes the two entry points. → Fig 1
2. **Conditioning the input** — translated view and profiles; the 13–49% argument.
   → Fig 2(a)
3. **The objective** — partition-only state, the five terms, the three constants and
   their measurements. → prose
4. **Why a proxy seeds the search** — signed graph, the CX-weighs-zero argument,
   transversal affinity as neighbourhood similarity, then the two-objectives claim and
   the graph-free-descent negative result. → Fig 2(b), 2(c)
5. **The descent** — three move shapes, no lane move, per-qubit best-improvement, and
   why KL is gone. → Fig 2(d)
6. **Lanes and the safety net** — exact assignment once; `verify` and the never-worse
   guarantee. → Fig 2(d), Fig 1
7. **What the placement is for** — merging: coverage, alignment, scheduling, and the
   +2–3%. → Fig 3
8. **Choosing resources** — why it sits above the packer, the E objective, the
   density tradeoff, and the stated trust boundaries. → existing results figures
