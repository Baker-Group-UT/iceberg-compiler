# Overnight run: [[n, n−2, 2]] pipeline generalization, testing, and grouping experiments

> **Point-in-time record, not current status.** This describes the state at
> `experiment-qubit-grouping` (`fffcc7d`), three commits before the branch tip. The
> agglomerative clustering it evaluates has since been replaced by the analytic /
> multilevel / refine stack, and the placement numbers here are superseded. See
> **[STATUS.md](STATUS.md)** for current capabilities and limitations. The correlation
> study in §"The headline measurement" is the part still worth reading — it is what
> established that placement is min-cut on the interaction graph rather than
> merge-maximisation.

Three branches, in sequence:

| Branch | Contents |
|---|---|
| `generalize-iceberg-nn22` | the `iceberg/` gate layer (from the previous session), committed |
| `generalize-pipeline-nn22` | **step 0 + 1**: code size threaded through the whole stack, plus the random-circuit test harnesses |
| `experiment-qubit-grouping` | **step 2**: agglomerative clustering for choosing which logical qubits share a patch |

---

## Headline: six pre-existing correctness bugs, all affecting [[4,2,2]]

The random-circuit harness was supposed to be plumbing. Instead it immediately found
that **the existing n=4 compiler mistranslates several common gate patterns**. Every
one of these is independent of the generalization work and would have corrupted any
experiment whose circuits contained the trigger.

Each was found by exact symplectic comparison (not sampling), shrunk to a minimal
reproduction, root-caused, fixed, and pinned with a regression test.

### 1. Program `SWAP` gates were never performed

`apply_logical_swap` emitted a physical SWAP **and** relabelled the two program
qubits in the state tracker. Those two cancel: the data moved and the labels chased
it, so the net logical operation was the identity. Postselection then read each
program qubit's value back out of its partner's lane.

Reproduction — `h(0); swap(0,1)` at n=4, tightest packing:

```
unencoded reference : {'0000': 0.496, '0100': 0.504}
encoded+postselected: {'0000': 0.499, '1000': 0.501}     fidelity 0.4976
```

The excitation should sit on qubit 1 and comes out on qubit 0.

**Fix.** A program SWAP is now a pure relabel — no physical gate, so a logical SWAP
is free. See bug 6 for why the physical-SWAP-and-no-relabel alternative is not
usable here.

### 2. `commute_hadamards` did not follow the qubit through a `SWAP`

`H(0); SWAP(0,1)` was rewritten to `SWAP(0,1); H(0)` instead of `SWAP(0,1); H(1)`.
The pass tracks a pending basis change per qubit and pushes Hadamards to the end of
the circuit; a SWAP exchanges the two qubits' states, so it has to exchange their
pending flags too. It fell through to the "unhandled gate" path and did neither.

This pass runs by default on every translation, so any circuit with an H before a
SWAP was silently rewritten into a different circuit.

**Fix.** `swap` exchanges the flags.

### 3. `commute_hadamards` had an incomplete and partly wrong substitution table

The table of "what does gate G look like through a pending Hadamard" was missing
`sx` and `sxdg` entirely, and mapped `sdg` to `S` rather than `SX†`:

| gate | H·G·H is | table said |
|---|---|---|
| `s` | `sx` | `sx` ✓ |
| `sdg` | `sxdg` | `s` ✗ |
| `sx` | `s` | *(missing — emitted unchanged)* ✗ |
| `sxdg` | `sdg` | *(missing)* ✗ |

`SXdgGate` and `SdgGate` were already imported but unused, which suggests the table
was half-written. Any program containing `sx` under an active frame — and `sx` is in
the translator's own gate set — came out wrong. A 300-circuit fuzz showed 32
unitary mismatches from this alone.

**Fix.** Completed the table, verified each entry against its matrix, and made
unknown single-qubit gates *flush* the frame (re-emit the explicit H) rather than be
reinterpreted in the wrong basis.

### 4. `commute_hadamards` could not consume the `xcx` gates it emits

The pass turns `cx` with an active control-side frame into `xcx`, but its two-qubit
branch only handled `cx` and `cz`. An `xcx` on the input fell through unchanged and
so was reinterpreted in the wrong basis. In production the pass runs once, so this
was latent — but it blocks fuzzing the pass against its own output, and it is wrong
on any hand-written program using `xcx`.

**Fix.** Added the `xcx` rules (`H_a XCX_ab H_a = CX(a→b)`, both-active = `CZ`).
Also gave `XCXGate` a `definition`, without which `Clifford`, `Operator` and Aer all
fail on any circuit containing it — that is why XCX circuits had never been checked.

### 5. Intra-patch `CZ` and `XCX` addressed stale slots

`place_logical_cz` and `place_logical_xcx` passed the *static* qubit ids to their
intra-patch branch, where every other branch — and all of `place_logical_cnot` —
passes the *current* slot positions. After an alignment SWAP has moved qubits within
a patch, the static ids point at whichever qubits moved into the original slots, so
the gate lands on the wrong pair.

Reproduction: a misaligned cross-patch `CX` (which inserts an alignment SWAP)
followed by an intra-patch `CZ` on the qubits that moved.

**Fix.** Both branches now use `patch * k_log + current_slot`.

### 6. `allow_transversal_swap_merges` produced macros that paired the wrong lanes

Two distinct problems in the same feature, which is on by default.

**6a.** The merging pass inserts program-level SWAPs around a transversal macro to
realign crossed lanes. Those are *routing* swaps: they have to physically move the
data, because the macro pairs lane *i* with lane *i*. But they were emitted as
ordinary `swap` gates, so they went through the same handler as program SWAPs — and
after fixing bug 1 to make program SWAPs free relabels, they moved no data and the
macro's pairing was left crossed.

The two callers genuinely need opposite implementations, and they differ *only* in
whether a physical SWAP is emitted (both update the tracker identically):

| | physical gate | tracker | net logical effect |
|---|---|---|---|
| program `SWAP` | no | relabel | the SWAP is performed |
| routing swap | yes | follows the data | identity |

**Fix.** A new `RoutingSwapGate` ("routing_swap"), emitted by the merging pass and
dispatched to `apply_routing_swap`. This is what the old code was doing for *both*
cases, which is why program SWAPs were broken; and it is what my first fix broke in
the other direction.

**6b.** The transversal scan fused two candidate gates *across* an intervening
program SWAP, moving the SWAP after the macro — while its own slot bookkeeping had
already applied the SWAP before it. So the pass's model of which qubit sat in which
slot at macro time disagreed with what the translator would see, and the macro
paired the wrong lanes.

Reproduction: `h(0); xcx(2,1); swap(2,3); cx(0,2)` at n=4.

**Fix.** A program SWAP now invalidates any pending transversal candidate. This is
conservative — it gives up some merges — but the alternative is making the pass's
bookkeeping survive its own reordering, which is a redesign.

### Effect on previously collected data

Any experiment whose circuits contained a `SWAP`, an `sx`, or an `xcx` — or a
misaligned cross-patch gate followed by an intra-patch one — was measuring a circuit
that did not compute the program. Fidelities from those runs are not trustworthy and
should be regenerated. Circuits built only from `h`/`x`/`z`/`s`/`cx`/`cz` with no
SWAPs are unaffected.

---

## Step 0 — code size threaded through the stack

Commit: *Thread the code size through the translator, simulator and pipeline*.

The gate layer already supported any even n, but nothing could reach it:
`TranslationContext` hardcoded `IcebergSet(n=4)` while `TranslatorConfig.n_code` was
dead config. One knob — `PipelineConfig.translator_config.n_code` — now selects the
code size for the whole stack.

**Translator.** `n_code` flows `entrypoints → Translator → Translation →
TranslationContext → mapping_for_perm + IcebergSet`, with `k_log` derived from it.
`update_states_on_swap` is now a true transposition of the two swapped slots (it used
to flip 0↔1 for every tracked state in the patch, which coincides with the
transposition only at k=2). The three copy-pasted `k_log == 2` realignment branches
in the cnot/cz/xcx handlers collapsed into one `realign_target_slots` that decomposes
the required slot permutation into transpositions, so it works at any k.

**Merging is gated, not half-generalized.** `translator.merging` is built around two
slots per patch. Rather than emit wrong macros at k>2, `Translator` skips merging at
other code sizes and warns once. Generalizing it is plan phase 3.

**Simulator / evaluator / pipeline.** The `num_qubits // 6` and `qubits_per_code=6`
patch inference is replaced by `iceberg.layout` helpers that read the layout off the
circuit's registers. This is not only n-general: **division was already wrong in
fault-tolerant mode at every code size**, because flag ancillas are appended after
the patches and were being attributed to nonexistent patches. `grid_constructor`
gained `positions_for_circuit`, which also gives flag ancillas the coordinate of a
patch they actually touch instead of leaving them out of the distance-noise layout.

The five permutation-generating pipeline tasks take n from the translator config;
`permutation_with_n_codes` and `create_permutations` take an `n` and use
`logical_capacity(n)` instead of a hardcoded two slots per patch.

**Verification.** An n=6 pipeline run reproduces the unencoded distribution exactly:

```
n=4  6 permutations   fidelity 1.0000   selection rate 1.000
n=6  6 permutations   fidelity 1.0000   selection rate 1.000
```

---

## Step 1 — random-circuit test harnesses

Two complementary harnesses, because each catches what the other cannot.

### `tests/test_pipeline_logical_action.py` — exact, sampling-free

For a random Clifford program, the logical Clifford induced by the *translated
physical circuit* must equal the program. No simulation, no shot noise, sign-exact
(it tells S from S†).

The comparison is per generator, which is what makes it usable: for every program
qubit i and each P ∈ {X, Z},

```
U (Pbar on lane start[i]) U†  ==  relabel(Program(P on qubit i), end)
```

where `start` comes from the permutation and `end` from the translator's final
`state_positions`. Comparing generator by generator avoids having to reconstruct the
induced permutation over *all* lanes, which is not recoverable from
`state_positions` once a qubit has been swapped into a lane that started idle.

Covers: placement under the FT rules, the alignment SWAPs, the merging macros (n=4),
the peephole passes, and every `iceberg/` construction — at n ∈ {4, 6, 8}, bare and
fault-tolerant, with merging on and off.

This is the harness that found bugs 1, 5, 6a and 6b.

### `tests/test_pipeline_roundtrip.py` — the distribution harness you described

Random circuits → translate → simulate → postselect → reduce → assert the logical
distribution matches the unencoded program. This is the path a real scan takes, and
it covers what the exact test cannot: encoding, syndrome extraction, the classical
decoder, and `reduce_distribution`'s use of the final state positions.

Fidelity is the primary assertion (it moves by ~1e-4 under shot noise but collapses
to ~0.5 for a single wrong gate). The TVD bound is derived per distribution from the
multinomial sampling error rather than fixed, because TVD grows with support size —
a random Clifford on 8 logical qubits spreads over up to 256 outcomes, where sampling
alone gives TVD ≈ 0.05 at 20k shots. A fixed 0.05 threshold was flaky at n=6.

Non-Clifford coverage is here rather than in the exact test, since `rz`/`rx` at
generic angles have no tableau: random programs with rotations, plus targeted
single-rotation cases that pin the angle end to end.

### `tests/test_translator_passes.py`

Fuzzes each peephole pass against the unitary it started from, and pins the
H-conjugation table against its matrices. This is where bugs 2, 3 and 4 lived.

### `tests/random_programs.py`

Shared generators over the translator's *own* gate set. Deliberately not
`qiskit.random_clifford`, which transpiles to gates the translator has never seen.

### Suite status

```
199 passed          (163 before this session's step-1 work)
runtime ~50 s
```

Plus an overnight fuzz harness (`scratchpad/heavy_fuzz.py`, not committed) that runs
the exact check over random programs × code sizes × packings × translator settings,
shrinking every distinct failure to a minimal reproduction. Progression across the
session:

| fuzz run | distinct failures | after |
|---|---|---|
| 1 | 21 | bugs 1–5 fixed |
| 2 | 5 | bug 6a fixed |
| 3 | 1 | bug 6b fixed |
| 4 | *(running)* | — |

All failures were at n=4 with merging enabled; n=6 and n=8 were clean across ~16,000
checks, which is the evidence that the generalization itself is sound.

---

## Step 2 — grouping logical qubits by agglomerative clustering

Branch `experiment-qubit-grouping`, module `grouping/`, runnable as
`python -m grouping.evaluate`. 25 tests in `tests/test_grouping.py`.

### The headline finding, which changes the objective

**The thing that drives placement cost is not merge opportunity. It is the number of
two-qubit gates that land across a patch boundary.** Measured over 3,600 placements
(all slot assignments of 5 random 6-qubit programs at n=4), correlation against the
measured weighted gate count of the translated circuit:

| signal | correlation with cost |
|---|---|
| cross-patch two-qubit gates | **+0.80** |
| physical SWAP gates emitted | +0.56 |
| intra-patch two-qubit gates | +0.20 |
| the merging pass's own `merge_savings_weighted` | **−0.17** |

So the problem is **capacity-constrained min-cut on the interaction graph**, not the
merge-maximisation problem that `packer/merge_packer.py` (and my first cost model)
are built around. That is worth knowing independently of anything else here: the
existing packer optimises the signal that explains the least variance, and it shows
up in the measurements below as the weakest non-trivial strategy.

A related surprise: cross-patch CX and intra-patch CX have *identical* entries in the
work table, (0, 4). Yet co-locating still helps, because a *misaligned* cross-patch
gate makes the translator insert an alignment SWAP. The cost table understates
cross-patch cost, and that is why slot order matters as much as grouping.

### What was built

`grouping/profiles.py`
: Per-qubit gate timelines — the **frame** (what kind of gate, layer by layer) and the
  **cast** (which other qubits it interacts with, and when), using the same layering
  the merging pass uses.

`grouping/cost_model.py`
: Scores an arbitrary *set* of qubits directly, in time linear in group size and
  depth, so subsets are never enumerated. This is the piece that answers "how do you
  score a group without enumerating"? It models the terms that genuinely are not
  sums over pairs: the whole-patch Hadamard is **conjunctive** (HH applies H to every
  lane, so at k=4 all four members must want one), and Pauli fusion is
  **super-additive** (X on m aligned lanes costs m + (m mod 2) rather than 2m).

`grouping/similarity.py`
: Your frame/cast idea as a standalone kernel: frame agreement (same gate kind at the
  same layer, with a layer tolerance), cast overlap (Jaccard of interaction
  partners), minus a schedule-conflict penalty for layers where both are busy with
  *different* kinds — the negative signal a pure affinity score misses.

`grouping/agglomerative.py`
: Capacity-constrained agglomerative clustering. Every qubit starts alone; repeatedly
  merge the two clusters with the best linkage score while the union still fits in k
  slots. **This is how the pairwise limitation is escaped**: the linkage is evaluated
  on the *union of two clusters*, which may be any size up to k, so the conjunctive
  and super-additive terms are seen exactly. It is pairwise only in that it merges two
  things at a time. Cost is O(m³) linkage evaluations rather than O(2^m).
  Three linkages: `cost`, `similarity`, `interaction` (cut weight).

`grouping/refine.py`
: Local search over the placement, with moves that can express what clustering cannot
  — `swap` (grouping *and* slot order), `relocate` (grouping), `reorder` (slot order
  within a patch). Runs against either the measured objective (translate and count)
  or the cost model as a cheap surrogate.

`grouping/evaluate.py`
: Translates every candidate placement and measures it, so every number below is an
  emitted gate count rather than a model's opinion.

### Measurements

n=4, 6 logical qubits, 10 circuits with community structure in their interaction
graph, relative to the sequential (tightest, in-order) packing. Lower is better;
`exhaustive` is the true optimum over all 720 slot assignments.

| strategy | × sequential | reached optimum |
|---|---|---|
| exhaustive (720 placements) | 0.879 | 10/10 |
| **refined←interaction** | **0.879** | **10/10** |
| refined←sequential | 0.883 | 7/10 |
| random-best of 30 | 0.898 | 1/10 |
| cost clustering | 0.923 | 1/10 |
| refined←model (cost model as surrogate) | 0.927 | 0/10 |
| interaction clustering | 0.934 | 1/10 |
| similarity clustering | 0.934 | 1/10 |
| `packer/merge_packer` | 0.970 | 1/10 |
| sequential | 1.000 | 0/10 |

n=6 (k=4), 8 logical qubits, 8 structured circuits — where the existing packer cannot
run at all, since it hard-requires two slots per patch:

| strategy | × sequential |
|---|---|
| refined←sequential | 0.936 |
| refined←interaction | 0.943 |
| random-best of 40 | 0.949 |
| refined←model | 0.949 |
| interaction clustering | 0.955 |
| similarity clustering | 0.986 |
| cost clustering | 0.995 |
| sequential | 1.000 |

### What worked, what did not

**Clustering alone beats the existing packer but not by much.** All three linkages
land at 0.92–0.93× sequential at n=4 versus the packer's 0.970×, and they work
unchanged at k=4. But they leave most of the available gain on the table.

**Clustering plus local refinement reaches the optimum.** `refined←interaction` found
the exhaustive optimum in 10/10 circuits using ~40 evaluations against 720 for
exhaustive search — an 18× saving that grows factorially with qubit count. It also
beat refinement from the sequential seed (10/10 versus 7/10), so the clustering seed
does earn its place; it just is not sufficient by itself.

**My cost model is not a usable surrogate.** Local search driven by it reaches only
0.927× where the measured objective reaches 0.879×. It over-weights the HH and
transversal terms, which the correlation table says barely matter, and under-weights
cross-patch traffic, which is what actually does. A surrogate built from the
correlation finding — cut weight plus a misalignment penalty — is the obvious next
attempt and is not yet written.

**The similarity kernel is the weakest of the three linkages** (0.986× at k=4). Its
frame-agreement term normalises by the *minimum* gate count of the pair, which
saturates at 1.0 easily and washes out the ordering. If it is worth pursuing, the
normalisation should change and the cast term should be weighted up, since cast
overlap is a proxy for cut weight and cut weight is what correlates.

### Two experiments that had to be corrected mid-flight

Both are worth recording because the first version of each produced a confident,
wrong answer.

**Uniform-random circuits cannot distinguish grouping strategies.** On uniformly
random programs every strategy sat within ±3% of sequential and all three clustering
linkages *lost* to it. The reason is that a uniformly random interaction graph has no
community structure, so min-cut has no signal and every grouping is about as good as
any other. Only after planting block structure — and then **shuffling the qubit
labels**, so that the planted blocks are not the sequential grouping — did the
comparison become meaningful. Without the shuffle the clustering recovered the
planted blocks perfectly and still showed 1.000× sequential, because sequential was
already the right answer.

**"Exhaustive" search over groupings is not exhaustive.** The first version enumerated
set partitions and assigned slots in sorted order, and was beaten by random slot
shuffling — which is impossible for a true optimum and was the clue that slot order
matters as much as grouping. It now enumerates assignments of qubits to *slots*.

### A separate finding: placement feasibility dominates everything

While setting up the experiment, most placements turned out to be **rejected** by the
translator's fault-tolerance rules. The cause is entirely cross-patch SWAP, which
`apply_logical_swap` refuses outright:

| circuit family | placeable at 4 / 6 / 8 qubits |
|---|---|
| Cliffords including SWAP | 5/20, 1/20, 0/20 |
| the same without SWAP | 20/20, 20/20, 20/20 |

At 6 qubits and up, essentially nothing containing a SWAP can be placed. Since step 1
made program SWAPs into free relabels, **a cross-patch SWAP is now also implementable
as a free relabel** — exchanging the two program qubits' labels is correct regardless
of which patches they sit in. Lifting the restriction would be a few lines and would
enlarge the feasible placement space enormously.

The catch, and the reason I did not do it tonight: the fault-tolerance predicates
(`are_all_partners_zero`, `is_alone_in_patch`) index `active_by_patch`, which is
*static* patch membership. Intra-patch relabels keep that valid because only slots
change; a cross-patch relabel would not, so those predicates would silently consult
stale membership. Making them dynamic is the prerequisite, and it is a real change to
the placement rules rather than plumbing.

**This is the highest-value follow-up I found** — larger than anything grouping can
win, because it changes which circuits can be compiled at all rather than how
cheaply.

### Suggested next steps, in order of expected value

1. **Make the FT predicates use dynamic patch membership, then allow cross-patch SWAP
   as a relabel.** Unblocks most circuits at 6+ qubits.
2. **Replace the packer's objective with cut weight plus a misalignment penalty.** The
   correlation table says this is what to optimise; the current merge-maximising score
   correlates *negatively* with cost.
3. **Ship `refined←interaction` as the packer's replacement.** It reaches the optimum
   on the cases where the optimum is computable, works at any k, and needs a cheap
   surrogate objective (item 2) to scale past ~10 qubits.
4. Generalize `translator/merging` to k > 2 (plan phase 3) so n=6 runs get merging at
   all — though note the correlation finding suggests the payoff is smaller than
   assumed.
5. ~~Fix the simulator's per-chunk reseeding (see Phase 5 note above) before trusting
   any noisy numbers.~~ **Done** — chunks now derive their own seeds from the configured
   one, so a seeded run uses every shot instead of `repetitions / 100`. Noisy numbers
   produced before that fix still carry ~10× the sampling error their shot count
   implies; Stim-backed ones were never affected. See `tests/test_simulator_seeding.py`.
