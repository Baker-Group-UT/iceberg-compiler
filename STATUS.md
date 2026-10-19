# Project status

Branch `updated-figures`, 14 commits ahead of `main` (`b1103b2`). **358 tests pass.**

This is the current-state document: what the compiler can do now, what it cannot, and
what changed across the four branches since `main`. It supersedes the status sections of
[OVERNIGHT_REPORT.md](OVERNIGHT_REPORT.md) and [GENERALIZATION_PLAN.md](GENERALIZATION_PLAN.md),
which are point-in-time records and are retained as such.

Every number below was measured on this branch. Numbers attributed to the paper draft
come from `AQORN_arXiv.pdf` and are labelled as such.

---

## 1. What changed, in one paragraph

`main` compiles to the **[[4,2,2]]** code. It now compiles to the whole **[[n, n−2, 2]]**
family at any even n ≥ 4, with gate merging and qubit placement working at every code size
rather than only at k = 2. Along the way ten pre-existing correctness bugs were found and
fixed: six at n = 4, where the compiler on `main` mistranslates any program containing a
`swap`, an `sx` or an `xcx`, and four more in how the translator reports a gate it cannot
place (§4.0). Placement moved from a pairwise greedy heuristic to a partition-plus-local-
search stack scored by an analytic cost model roughly 4000× cheaper than translating the
circuit.

**Depth improves 7–8 points past k = 2**, on a circuit set where every benchmark places at
every code size — 31.4% at k = 2 against 38.7% at k = 4 and 39.3% at k = 6 for the medium
group. Gate counts do not follow; they decline slowly with k, because merging's whole-patch
macros need a gate on every occupied lane at once and fire less often as k grows. §3.3 has
the table and the history of getting there, which took two wrong answers first.

---

## 2. Current capabilities

### 2.1 The code family

Every logical operation is correct at any even n ≥ 4, verified by exact Clifford-tableau
comparison at n ∈ {4, 6, 8, 10, 12} in both bare and fault-tolerant modes.

One identity does most of the work: **transversal H equals H^k composed with the
patch-wide map `C_A = I + J`, and `C_A` is realised by physically swapping the two
anchors.** So the whole-patch Hadamard is transversal H plus one SWAP at every n, with no
frame tracking and no cost penalty. The tiered strategy the original plan proposed for
this is withdrawn as unnecessary.

`n_code` is a single knob (`PipelineConfig.translator_config.n_code`) that selects the
code size for the translator, simulator, evaluator and all five permutation-generating
pipeline tasks.

### 2.2 Cost and depth tables, measured

`iceberg.costs` is the single source for both, replacing three previously hand-synchronised
copies of an n = 4 table. `work_table(n)` gives (1q, 2q) gate counts; `depth_table(n)` gives
depth. Both are pinned against the emitted circuits in `tests/test_iceberg_costs.py`.

Every depth entry is **constant in n** — HH is one transversal Hadamard layer plus the
anchor swap, and the transversal two-qubit macros are n gates in parallel:

| operation | intra | cross-patch | transversal (all k lanes) |
|---|---|---|---|
| CX | 4 | **3** | **1** |
| CZ | 3 | 3 | **1** |
| XCX | 6 | 5 | **3** |
| H | 8 | — | HH = **2** |

This table is what exposed the placement bug in §2.4.

### 2.3 Gate merging at any k

`translator/merging` was written against the [[4,2,2]] macros and skipped entirely at
other code sizes. It is now general in k.

The organising distinction the original plan lacked: macros split into **whole-patch**
(`hh`, `cxt`, `czt`, `xcxt`) and **per-lane** (`xx`, `zz`), and only the first needs a
coverage condition. Whole-patch macros are built from physical gates on all n data qubits,
so they act on every lane whether asked to or not — a group must cover every *occupied*
lane, meaning k program gates rather than two. Per-lane macros need no condition, because
`X̄ᵢ = X[x_anchor] X[lane i]` makes any subset legal and an even subset cancels the anchor
factor. At k = 2 the two rules coincide, which is why the pairwise code never had to name
coverage.

Transversal alignment generalised from a hand-enumerated identity/cross pair to **any
permutation of k slots**, decomposed by selection sort into at most k−1 transpositions and
emitted as routing swaps. Verified exhaustively against all permutations up to k = 6.

### 2.4 Placement

Four components. They were written in `grouping/`, which is now archived at
`archive/grouping/`, and `packing/` absorbed them: `analytic` → `packing/cost.py`
(`PackingCost`), `graph` → `packing/graph.py` with `packing/seed.py` on top,
`multilevel` → `packing/partition.py`, `plan` → `packing/plan.py`, and the local search
below → `packing/search.py`. `archive/README.md` has the file-by-file mapping.

- **`analytic`** — decomposable cost model: `Σ patch_term + Σ pair_term`, cached by
  membership, so a swap recomputes two terms and reuses the rest. ~0.1 ms per placement
  against ~700 ms for a translation at 140 qubits. Exact for gate counts; approximate only
  for merging (2–3%, isolated by a test).
- **`graph`** — signed interaction graph, so a pair that is cheaper apart is pushed apart.
  Prices depth as well as gate count, plus Hadamard-coincidence and transversal-affinity
  edges that the gate table cannot see.
- **`multilevel`** — coarsen/solve/uncoarsen partitioning, boundary refinement, then exact
  Hungarian slot assignment per patch pair. A consolidation pass repairs budget overshoot.
- **`plan`** — chooses *how many* patches to spend under a `ResourceBudget` and
  `NoiseProfile`, rather than assuming the tightest packing.

**The bug this stack fixed.** Gate counts and depth disagree about where a two-qubit gate
belongs, and for CNOT they disagree in opposite directions: intra and cross cost identical
gates, but cross is a layer shallower. So on gate counts alone **every CX edge weighed
exactly zero** — a CNOT-dominated circuit produced a nearly empty interaction graph and an
essentially arbitrary partition. That is why `ghz`, `cat`, `bv` and `adder` showed
negligible or negative packing benefit. Cut minimisation was pushing the wrong way for
every CNOT in the circuit.

Three fixes, each measured against exhaustive search rather than asserted: price depth in
the edge weights (CX now −3, CZ +12); add a discounted transversal-affinity term, which
rewards *neighbourhood similarity* rather than adjacency because a macro needs all k lanes
paired; and **refine on the analytic cost, not the graph**, since the graph is a pairwise
proxy for a conjunctive objective it cannot express. Capture of the exhaustive optimum at
k = 2 went **50% → 67%**; `qaoa_n6` went 4.0% → 16.9% (the optimum), and `simon_n6` at
k = 4 went 0% → 20.5%.

A fourth change is a safety property rather than an improvement: the model over-counts
merging enough to have picked a placement **8% worse than sequential** on `vqe_n4`, so the
winner is now verified against the naive placement by exact measurement and the better one
kept. Two translations, and it turns "usually better" into **never worse** — which is what
makes the packing contribution safe to present as a gain.

### 2.5 Pass ordering is selectable

`TranslatorConfig.h_commutation_stage` takes `before_merging` (the paper's order, default),
`after_merging`, or `never`. The draft argues that commutation-first inflates its own share
of the credit; that can now be measured. All three orders are verified to preserve the
logical action exactly at n ∈ {4, 6, 8, 10, 12}.

### 2.6 Test suite

358 tests. The ones that matter for correctness:

| file | what it pins |
|---|---|
| `test_iceberg_logical_action.py` | exact symplectic logical action vs reference circuits, n ∈ {4,6,8} |
| `test_pipeline_logical_action.py` | the logical Clifford induced by the *translated physical* circuit equals the program |
| `test_pipeline_roundtrip.py` | translate → simulate → postselect → reduce matches the unencoded program |
| `test_iceberg_roundtrip.py` | encode → gates → decode → postselect, pinning conventions at n = 6 where they are observable |
| `test_merging_generalized.py` | every macro triggered deliberately at n ∈ {4,6,8,10,12}, negatives pinned too |
| `test_iceberg_costs.py` | work and depth tables match the emitted circuits |
| `test_packing.py` | `PackingCost` reproduces the archived `AnalyticCost` exactly, and an incremental rescore matches a full one. (`test_analytic_placement.py` — analytic cost vs measured, the 2–3% merging error isolated — is archived at `archive/tests/`, which `pytest.ini` does not collect) |
| `test_simulator_seeding.py` | a seeded run uses every shot, reproduces exactly, and neighbouring seeds are independent |
| `test_macro_rejection.py` | a handler that cannot place a gate is not mistaken for one that did; rejected macros expand rather than vanish; a misaligned cross-patch gate places at k > 2 |

Sampling bounds in the roundtrip tests are **derived** from multinomial sampling error
rather than fixed. A fixed bound is wrong in a specific way: TVD and Bhattacharyya deficit
both grow with support size, so any constant threshold becomes flaky once a program spreads
over enough outcomes. That was the cause of a ~1-in-70 flake on `main`.

---

## 3. Results

### 3.1 Full pipeline vs naive, 71 QASMBench circuits, [[4,2,2]]

| | 1q gates | 2q gates | depth |
|---|---|---|---|
| small (29) | 24.6% | 18.2% | 28.4% |
| medium (15) | 26.7% | 18.4% | 31.3% |
| large (27) | 20.9% | 15.7% | 36.1% |
| **overall** | **23.5%** | **17.3%** | **31.9%** |

The draft reports 31% / 17% / 34–35%. Two-qubit gates and depth match; **one-qubit gates
do not, and the gap is not root-caused** — see §4.1.

### 3.2 Pass-order attribution

All 71 benchmarks, cumulative against no techniques at all:

| condition | 1q | 2q | depth |
|---|---|---|---|
| Merging alone | 12.9% | 8.9% | 17.5% |
| + Packing | 13.4% | 10.9% | 22.4% |
| + Commutation **first** (paper order) | **23.5%** | **17.3%** | **31.9%** |
| + Commutation **last** | 18.4% | 15.2% | 18.1% |

Two findings. **Merging and packing unaided are worth 13.4% / 10.9% / 22.4%**, far more
than the 5–10% the draft attributes to them when commutation runs first and takes the
Hadamards they would otherwise have fused — the draft's conjecture is correct and this
quantifies it. And **commutation-last is worse overall**, with a concrete mechanism:
`commute_hadamards` flushes its frame at any instruction it does not recognise, so once
merging has run, macros act as barriers and commutation simply finds less to do.

### 3.3 Code size: depth improves past k = 2, on a full circuit set

This section has been wrong twice. The history matters, because the two errors pointed in
opposite directions:

> **First claim** — medium circuits best at k = 4, +6.3 points on depth. Rested on circuits
> whose rejected gates were being silently dropped, so they compiled cheaper *and* wrong.
>
> **Then retracted** — k = 2 best on nearly everything. Correct given the data then
> available, but that data was survivorship-selected: with the dropping fixed, a third of
> the benchmarks simply refused to place at k > 2, so the k > 2 columns covered a smaller,
> easier subset.
>
> **Now** — the refusals turned out to be a vestigial restriction, not a real constraint
> (§4.3). Every circuit places at every code size, and the comparison is finally
> like-for-like.

**All 15 medium and all 27 large circuits place at every code size**, so these columns share
a denominator:

| k | medium 1q | medium 2q | medium depth | large 1q | large 2q | large depth |
|---|---|---|---|---|---|---|
| 2 | 26.5% | **18.4%** | 31.4% | 20.7% | **15.7%** | 37.7% |
| 4 | **26.7%** | 17.3% | 38.7% | 22.5% | 15.2% | **38.0%** |
| 6 | 24.8% | 17.1% | **39.3%** | 22.5% | 13.9% | 37.3% |
| 8 | 23.7% | 16.5% | 39.1% | **22.7%** | 15.0% | 37.2% |
| 10 | 23.2% | 16.3% | 38.2% | 21.8% | 13.7% | 35.0% |

**Depth on medium circuits improves by 7–8 points past k = 2** — 31.4% at k = 2 against
38.7% at k = 4 and 39.3% at k = 6, holding above 38% out to k = 10. That is the original
claim's direction, now measured on a full circuit set rather than a contaminated one, and
larger than the +6.3 first reported. It is also a regime the paper's pairwise packer cannot
reach at all.

Two-qubit gates go the other way, declining slowly at every width, and one-qubit gates are
roughly flat to k = 4 then decline. Large circuits are flat on depth (37.7% → 37.2% through
k = 8) and gain about 2 points on one-qubit gates.

So the conjecture in `text/evaluation.tex` holds **for depth and not for gate count**. The
mechanism for the gate-count half is merging's reach: whole-patch macros need a gate on
every occupied lane at once, and the firing rate falls from 10/15 medium circuits at k = 2
to 8/15 at k = 10.

Small circuits decline for a structural reason that was never in question — they run out of
patches. At k = 10 all 30 fit in a single patch, so there is no placement decision left and
the packing segment is exactly zero.

### 3.4 Placement quality and cost

| | value |
|---|---|
| Analytic model vs translation | ~4000× faster; regret 0.00–0.18% against the true optimum at k ≥ 4 |
| Capture of exhaustive optimum, k = 2 | 67% (5.58% of 7.23% available) |
| vs `archive/packer/merge_packer.py` at k = 2 (packing segment) | 0.7% / **2.4%** / **7.7%** against 0.8% / 0.6% / 2.0% |
| Code-size sweep, 1080 points, three size groups | 112 s (was: had not finished the medium group in 14.5 min) |
| 140-qubit placement | 3.4 s (was 156 s, same answer) |

---

## 4. Limitations

Ordered by how much they constrain the work.

### 4.0 Rejected gates used to vanish silently — fixed, but it invalidated k > 2 data

`translate_and_apply` discarded the return value of every compound handler, so a handler
reporting "I could not place this and emitted nothing" was read as success. The guard in
`Translation.translate_circuit` that raises on that signal was **dead code for all ten
compound gates**, `cx`, `cz`, `xcx` and `h` among them. Two ways this bit:

- **Plain cross-patch gates.** `place_logical_cnot` rejects a misaligned cross-patch CNOT
  unless the control patch is all `|0>` or the target is alone in its patch. That rejection
  was swallowed, so the CNOT simply did not happen. `bv_n14` at k = 4 compiled to a circuit
  with a **verifiably wrong logical action** and reported success.
- **Merged macros.** Merging replaces up to k program gates with one macro and drops the
  originals, so a macro rejected at translation took those gates with it. Two of the
  rejections were themselves spurious (`realign_target_slots` and `active_slots_in_patch`
  both assumed a full patch), and `cxt`/`czt`/`xcxt` had no expansion fallback where `hh`
  did.

All four defects are fixed and pinned by `tests/test_macro_rejection.py`, whose 9
load-bearing cases were confirmed to fail against the previous code. Over 360 random
programs at k ∈ {2,4,6}: zero wrong, zero rejected, worst TVD 0.049 (sampling).

**k = 2 was immune to the gate-dropping itself** — 0 rejections over 120 random seeds, and
structurally so, since an occupied 2-lane patch is full and both full-patch assumptions hold.

It is *not* untouched overall, and an earlier revision of this document said it was. The §4.3
relaxation removes alignment SWAPs at every k, k = 2 included, so the k = 2 baseline moved
slightly: Figure 10 went 23.5% / 17.3% / 31.9% to **23.2% / 17.2% / 32.5%** — at most 0.6
points, and depth improved because a SWAP that bought nothing is no longer emitted. Small,
but real, and every k = 2 figure has been regenerated.

**Every k > 2 number collected before this fix should be discarded.**

### 4.1 The one-qubit gate gap is not explained

Figure 10 gives 23.5% against the draft's 31%. Two-qubit gates and depth line up; this does
not. **Do not quote it as like-for-like.** Two candidates, in the order worth checking:

1. The stacking convention changed (§4.2). The old one could overstate a total.
2. The six correctness fixes changed what the compiler emits for any circuit containing a
   `swap`, an `sx` or an `xcx`.

### 4.2 Published Figures 10 and 13 overstate some bars

The draft's scripts clip each stack *segment* at zero, which breaks when one technique hurts
and a later one more than recovers. On `ising_n98`, H commutation costs 88% of depth and
merging returns 179%; the true reduction is 90%, but dropping the negative segment leaves a
**179% bar on a percent-reduction axis**. The rebuilt scripts clip the *cumulative* series
and difference it instead (`common.stack_heights`), so segments stay non-negative and the
stack height is the real reduction. **The published Figures 10 and 13 use the old
convention.**

### 4.3 Cross-patch SWAP is still refused (the CNOT/CZ/XCX half is fixed)

**Fixed:** cross-patch CNOT/CZ/XCX used to be refused when the two slots were misaligned
unless the control patch was entirely `|0>`, the target was alone in its patch, or some lane
matched. At k = 2 that was nearly always satisfiable; at k > 2 with a partly-occupied patch
all three routinely missed — a control patch holding lanes {1,3} against a target holding
{0,2} has no matching lane, no solo target and no all-`|0>` partner — and it cost 3 of 15
medium and up to 3 of 27 large QASMBench circuits at every k > 2.

The restriction was vestigial. All three success branches ended in the same *targeted*
cross-patch call, which `test_cross_patch_targeted_ops_any_slot_pair` pins as correct at any
slot pair; the `*_k_Transversal` call the alignment existed to enable is commented out in
each handler. The bare path now emits the targeted gate directly
(`utils.emit_targeted_cross_patch`) and **every benchmark places at every code size**, which
is what made §3.3 answerable.

Fault-tolerant mode keeps the original rules. A logical-action test shows the ideal action is
right; it cannot show that a gate preserves the code's error-detecting property, and
`are_all_partners_zero` / `is_full_transversal` read like they are about not spreading errors
between patches. **Whether the FT rules can be relaxed the same way is open**, and it is the
one place where k > 2 placement is still constrained.

### 4.3b Program SWAPs across patches

5/20, 1/20 and 0/20 random placements are accepted at 4, 6 and 8 qubits when the program
contains a SWAP, against 20/20 without, because cross-patch SWAP is refused outright at
`translator/handlers/swap.py:70`. A program SWAP is a free relabel now, so a cross-patch one
would be correct as a relabel too — but the fault-tolerance predicates index *static* patch
membership, which a cross-patch relabel would invalidate.

The cheap route, worth trying before making those dynamic: rewrite a cross-patch `swap` into
three cross-patch CNOTs at the program level. `SWAP = CX(a,b)·CX(b,a)·CX(a,b)` is exact,
cross-patch CX works for any slot pair, and the cost is `(0, 12)` gates at depth 9, constant
in n. Nothing is relabelled, so the static-membership problem never arises. **Top follow-up.**

### 4.4 Placement headroom at k = 2 is single digits

Exhaustive search over placements gains only **7.2%** over sequential packing at k = 2, and
we capture 67% of it. The larger packing numbers in Figures 12 and 13 come substantially
from comparing against a deliberately naive baseline at each density — a legitimate framing
the draft already uses, but the *absolute* headroom for placement at k = 2 is small. The
k-sweep is where the stronger story is.

### 4.5 Merging reach falls with k

Whole-patch macros need a gate on every occupied lane simultaneously, and the firing rate
falls from 58/99 circuits at k = 2 to 41/99 at k = 10.

Fusing a *subset* of lanes into a whole-patch macro is **not available and cannot be**:
partial transversal CX sends the full-weight stabilizer `S_X = X^⊗n` outside the stabilizer
group, so it is not a logical operation at all. The reachable middle ground is
transversal-then-correct, which the cost table says pays only within about k/3 lanes of full
alignment — and which asks the packer for a different objective (near-alignment of whole
patch pairs) than the one it now optimises.

### 4.6 Simulated results are incomplete

- Publication-scale runs of Figures 5, 13 and 17 were not completed. Figure 13's LSR/TVD
  rows are absent; the gate-count rows are present.
- Figure 17 reproduces the LSR/TVD anticorrelation at **−0.91** even at smoke-test size.
- Figure 5 produces its heatmaps, but at the coarse default grid the three CNOT
  implementations sit within 0.001 fidelity of each other, so the grid needs raising before
  the comparison means anything.
- Figure 12's conclusion is sensitive to `--immutable-ratio`, which was not calibrated
  against the draft's generator.

### 4.7 Any pre-existing seeded Aer number is suspect

`AerSimulator`'s `seed_simulator` seeds a *run*, not a backend, and the simulator was
constructing one backend and calling `run()` once per shot chunk — so every chunk replayed
the first and a seeded run carried **1% of the shots it reported**. Fixed on this branch and
pinned by `test_simulator_seeding.py`, but **noisy numbers produced before `864d9f4` carry
roughly ten times the sampling error their shot count implies.** Stim-backed results were
never affected, since `run_stim_backend` compiles one sampler and `sample()` advances its
state.

### 4.8 Smaller items

- `archive/grouping/tune.py` selected `depth_weight = 12` *before* `best_placement` added
  analytic refinement, so both free parameters were fitted against an older pipeline. The
  value is still what `packing.cost.DEFAULT_DEPTH_WEIGHT` carries, but the script that fitted
  it is archived with no live replacement, so refitting now means rewriting it against
  `packing`.
- `PairMergeAnalyzer`'s `cx_merges` / `cz_merges` / `xcx_merges` have **always** been 0. It
  is fed one-lane patches, which cannot form a macro. Documented, not fixed.
- `archive/packer/merge_packer.py` is k = 2 only by construction — it assigns qubits in
  pairs — and is archived as a reference point, not a live path. `packing/` imports nothing
  from it, and the `--packer legacy` option went with the scripts that offered it.
- The `archive/grouping/cost_model.py` group model is **not** a usable surrogate (0.927
  against 0.879 achievable): it over-weights exactly the terms the correlation study found
  do not drive cost. `archive/grouping/analytic.py` replaced it, and `packing/cost.py`
  replaced that — `tests/test_packing.py` pins the two to each other over random
  partitions, which is why `archive/grouping/` is kept importable.
- Five scripts now under `archive/experiments/` still call `plt.show()` at module level
  (`noise_scaling/p_scaling.py`, `codes_used_vs_sel_rate.py`, `error_analysis/cnot_weight.py`,
  two under `archive/experiments/archive/`), so importing them opens a window. They are
  top-to-bottom analysis scripts rather than importable helpers, so guarding them means
  restructuring. None is reachable from `pytest` (see below) or from the compiler, so they
  only fire if you import them yourself.
- **`pytest` used to open four figure windows per run** and nothing in its output said so.
  `circuit_dag_shor_test.py`, then at `dag/` and now archived to
  `archive/dag/circuit_dag_shor_test.py`, matched pytest's default `*_test.py` glob, so
  collection *imported* it and ran the four `draw_*` calls that sat at module level; under
  an interactive backend `plt.show()` blocks, so a test run stopped and waited. The file
  contributes **zero test items**, so the collected count never changed and the windows
  looked like they came from nowhere. Fixed three ways: `pytest.ini` confines collection to
  `tests/` and drops the `*_test.py` glob, a root `conftest.py` forces `MPLBACKEND=Agg`
  before any test module is imported, and the two `archive/dag/` demo scripts are behind
  `__main__`. Worth knowing because the same trap catches any future file named `*_test.py`.
- Lint: 284 `E402` findings repo-wide are the `sys.path` bootstrap idiom used by every
  experiment and test file. Not addressed.

---

## 5. Changelog by branch

Four branches, `main` → `generalize-iceberg-nn22` → `generalize-pipeline-nn22` →
`experiment-qubit-grouping` → `updated-figures`.

### `ba77d0c`, `182ce27` — investigation

The [[n,n−2,2]] plan plus statevector verification of its core claims, no implementation.
Established that `iceberg/` was already ~80% n-general, that `transversal H = H^k · C_A`,
that the naive XOR decode rule yields `C_A·x` rather than `x`, and that three silent
n > 4 bugs existed. Also aligned stim `movement_speed` with `noise_models.py`.

### `generalize-iceberg-nn22` (`8a0380f`, `9343caa`) — the code layer, 163 → then threaded

Every logical operation generalised to any even n ≥ 4. **Four silent k = 2-only bugs**, all
traced to the same identity — an X-type logical operation built against the Z-anchor differs
from the intended gate by `C_A`, which is invisible at k = 2: `SX` was anchored wrongly with
an index reversal compensating; whole-patch H swapped *lanes* rather than anchors; cross-patch
`XCX` carried a k = 2 lane-swap compensation that ran anyway at k > 2; intra-patch `XCX` had
no general path. Two of the replacements are also *cheaper* — cross-patch XCX went (8,4) →
(4,4).

New modules `iceberg/layout.py`, `physical.py`, `costs.py`, `demo.py`. Three synchronised
copies of the cost table collapsed to one.

`9343caa` then made it reachable: `n_code` was dead config while `TranslationContext`
hardcoded `IcebergSet(n=4)`. Also fixed `update_states_on_swap`, which flipped 0↔1 for every
tracked state in the patch — coincidentally correct at k = 2 — and replaced `num_qubits // 6`
patch inference with layout helpers that read the registers, which was *already* wrong in
fault-tolerant mode at every code size because flag ancillas were attributed to nonexistent
patches.

### `generalize-pipeline-nn22` (`89aacdc`) — six pre-existing n = 4 bugs

Random-circuit harnesses through the full pipeline, compared against logical-level
simulation. Intended as plumbing; it found that the **existing n = 4 compiler mistranslates
several common gate patterns**:

1. Program SWAPs were never performed — `apply_logical_swap` emitted a physical SWAP *and*
   relabelled, which cancel. `h(0); swap(0,1)` came out at fidelity 0.4976.
2. `commute_hadamards` did not exchange the pending basis change across a SWAP.
3. Its substitution table was missing `sx`/`sxdg` and mapped `sdg` to S rather than SX†.
4. It could not consume the `xcx` gates it emits itself; `XCXGate` also had no definition,
   without which Clifford/Operator/Aer all fail on any circuit containing it.
5. `place_logical_cz` / `place_logical_xcx` addressed their intra-patch branch with static
   qubit ids where every other branch uses current slots.
6. `allow_transversal_swap_merges` paired the wrong lanes, two ways.

An overnight fuzz went 21 → 5 → 1 → 0 distinct failures as these landed, clean over 9,300
checks. **Every failure was at n = 4 with merging on**; n = 6 and n = 8 were clean
throughout, which is the evidence that the generalization is sound.

> Any previously collected data whose circuits contained a `swap`, an `sx` or an `xcx` — or
> a misaligned cross-patch gate followed by an intra-patch one — was measuring a circuit
> that did not compute the program, and should be regenerated.

### `experiment-qubit-grouping` (`432c9d4`, `fffcc7d`) — placement, and merging at k > 2

`432c9d4` measured what actually drives cost over 3,600 placements:

| correlation with measured cost | |
|---|---|
| cross-patch two-qubit gates | **+0.80** |
| physical SWAPs emitted | +0.56 |
| intra-patch two-qubit gates | +0.20 |
| `merge_savings_weighted` | **−0.17** |

So placement is capacity-constrained min-cut on the interaction graph, **not** the
merge-maximisation problem `archive/packer/merge_packer.py` is built around — which shows up as the
weakest non-trivial strategy in the comparison. Two experiments had to be corrected
mid-flight, both recorded: uniform-random circuits cannot distinguish grouping strategies
(no community structure, so min-cut has no signal), and "exhaustive over groupings" is not
exhaustive — the first version fixed slot order and was beaten by random slot shuffling,
which is what revealed that **slot order matters as much as grouping**.

`fffcc7d` generalised merging to k > 2 (§2.3) and replaced translate-per-candidate placement
with the analytic model (§2.4). Also fixed the ~1-in-70 flaky fidelity threshold.

### `updated-figures` (`b67d2e5` … `182445a`) — figures, depth, and two infrastructure bugs

- `b67d2e5` — `experiments/figures/updated/` rebuilt the six data figures from
  `AQORN_arXiv.pdf` plus a new code-size sweep, every script taking `--from-csv` and
  `--n-code`. Fixed the segment-clipping presentation bug (§4.2). **Since archived** to
  `archive/experiments/figures/updated/`, and not runnable as-is — see §6.
- `8a40bb1` — measured `depth_table(n)`; found and fixed the zero-weight CX bug (§2.4);
  added `h_commutation_stage` and the pass-order figure (§3.2).
- `5ac48d5` — ported the code-size sweep off the old placement path (14.5 min without
  finishing one group → 112 s for all 1080 points), and fixed a duplicate-row CSV hazard.
- `864d9f4` — the simulator seeding bug (§4.7).
- `182445a`, `7be9a99` — stopped `circuits/dag_visaulizer.py` and
  `dag/circuit_dag_shor_test.py` opening figure windows on import; the latter matched
  pytest's `*_test.py` glob, so every test run drew four of them. Both folders are since
  archived, to `archive/circuits/` and `archive/dag/`.
- `11b5bda` — this document, and the README brought up to date.
- **this commit** — a gate the translator could not place no longer disappears (§4.0).
  Found by `experiments/tests/testexp.py` at seed 32, k = 6, which is what the harness is
  for. Forced the §3.3 retraction: the k > 2 columns of every earlier sweep were partly
  measuring circuits that did not compute their program.

---

## 6. Reproducing the results

```bash
# Technique decomposition -- h commutation, then merging, then packing -- by exact
# translation. `--code-sizes` takes n, not k, so this is k = 2...10
PYTHONPATH=. venv/bin/python -m packing.bench --techniques --code-sizes 4 6 8 10 12

# Packing strategy comparison, same measurement, planners against each other
PYTHONPATH=. venv/bin/python -m packing.bench

# Code size sweep: three size groups, five code sizes, ~2 minutes
PYTHONPATH=. venv/bin/python -m experiments.figures.merging_improvement.merging_improvement_code_size --no-show

# Logical action at several code sizes, printed
PYTHONPATH=. venv/bin/python -m iceberg.demo

# Random-Clifford round-trip probe at k = 2, 4, 6, looping on the seed
PYTHONPATH=. venv/bin/python -m experiments.tests.testexp

PYTHONPATH=. venv/bin/python -m pytest -q
```

**The scripts that produced the figure numbers above are archived and will not run as-is.**
`fig10_spacetime_reduction.py` and `fig_pass_order.py` (§3.1, §3.2) are at
`archive/experiments/figures/updated/`, and `grouping.evaluate` / `grouping.tune` (§3.4) are
at `archive/grouping/`; all of them import `grouping`, `packer` or the pre-strip `pipeline`.
`packing.bench` above is the live replacement for the first three — on the QASMBench Clifford
proxy corpus rather than the exact 71-circuit set, so the numbers are close but not
like-for-like. Nothing replaces `tune`.

Further detail: [archive/experiments/figures/updated/RESULTS.md](archive/experiments/figures/updated/RESULTS.md)
for the archived figures, [experiments/figures/merging_improvement/CODE_SIZE_SWEEP.md](experiments/figures/merging_improvement/CODE_SIZE_SWEEP.md)
for the code-size sweep, [archive/README.md](archive/README.md) for where each archived
module went, [OVERNIGHT_REPORT.md](OVERNIGHT_REPORT.md) for the placement
correlation study, [GENERALIZATION_PLAN.md](GENERALIZATION_PLAN.md) for the original plan and
which of its predictions held.
