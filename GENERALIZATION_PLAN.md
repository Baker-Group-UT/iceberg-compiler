# Plan: Generalizing from [[4,2,2]] to the [[n, n−2, 2]] Iceberg family

> **The plan, retained for its reasoning and its record of which predictions held.** All
> five phases have since landed; **[STATUS.md](STATUS.md)** is the current-state document.
> Two of this plan's central predictions were **withdrawn** on contact with the code, and
> both are worth knowing: the tiered strategy proposed for the whole-patch Hadamard is
> unnecessary (transversal H plus one anchor swap works at any n, §2.1 below is
> superseded), and merging did not need the pairwise structure this plan assumed — the
> real distinction is whole-patch versus per-lane macros, which the plan does not mention.

**Branch:** `generalize-iceberg-nn22` · **Base:** `b1103b2` (main) · **Status:** Phases 0–5 implemented; see STATUS.md.

Motivation: evidence beyond [[4,2,2]], with [[6,4,2]] the natural next code. This plan targets uniform-n code sets ([[6,4,2]] as the concrete deliverable), with arbitrary even n falling out of the same changes.

## Status

**Done — the `iceberg/` layer is general.** Every logical operation is verified at n ∈ {4, 6, 8}, bare and fault-tolerant, by `tests/test_iceberg_logical_action.py` (exact symplectic comparison against a reference logical circuit), `tests/test_iceberg_rotations.py` (rotations as Pauli exponentials) and `tests/test_iceberg_roundtrip.py` (encode → gates → decode → postselect vs. an unencoded reference, through the production simulator). New: `iceberg/layout.py` (the code convention, one place), `iceberg/costs.py` (the n-parameterized work table, now the single source for `packer/work_table.py` and `translator/merging/conflicts.py`), `iceberg/physical.py` (gate emission and the flagged gadget).

n=4 regression: the translator emits **identical two-qubit counts and identical postselected distributions**; single-qubit counts drop by 0–4 per program (cheaper cross-patch XCX). Physical qubit choices changed for `SX`, `XCX` and `HH` — same logical action, same cost, see §2.2.

To look at the circuits: `python -m iceberg.demo` encodes a logical program at n = 4, 6, 8 and prints counts and drawings; `--per-gate` shows each logical operation's physical expansion on its own; `--translator` runs the production compiler alongside for comparison (n=4 only). Its `encode_program` is the n-general reference encoder — identity placement, no merging — and is what the Phase 2 n=6 end-to-end test should compare the translator against.

**Outstanding:** packer k-subsets (Phase 4 — partly superseded by the `grouping/` module), and the [[6,4,2]] simulation figure (Phase 5; the threading itself is done). Phases 0–3 are complete: the whole compile path from `PipelineConfig.translator_config.n_code` down to the physical circuit is general in k, merging included, and verified by exact logical-action comparison at n ∈ {4,6,8,10,12}.

## TL;DR difficulty verdict

The codebase is a layered onion. The `iceberg/` physical layer is now general. The translator is symbolic in `k_log` except for a few hard k=2 branches and one hardcoded `n=4` constructor call. The remaining work:

1. ~~**A physics problem:** the patch-wide Hadamard.~~ **Resolved, and it was not a research problem.** Transversal H equals H⊗ᵏ composed with the patch-wide map C_A, exactly as suspected — but C_A is realized by *physically swapping the two ancillas* (§2.1). So `HH` is transversal H plus one SWAP at every n, the merging pass needs no frame tracking, and the Tier 1/2/3 trade-off below is moot. Mid-circuit HH is as cheap at k=4 as at k=2.
2. **An algorithms problem:** the packer is intrinsically 2-way matching (`w_ij − w_ii − w_jj` over pairs); k>2 needs a k-subset grouping formulation. This is the §5.3 acknowledged limitation and the core of the P3.1 experiment. **This is now the only part with real design risk.**

Everything else is mechanical plumbing: threading `n` through configs and swapping hardcoded 6's (= n+2) in the simulator/evaluator. The three synchronized cost tables are down to one.

Revised effort: **1–2 weeks** for a working [[6,4,2]] end-to-end pipeline with the scoped greedy packer extension. Nothing left is severable for research risk; Phase 4 is the only judgement call.

---

## 1. What is already general (verified, safe to build on)

- `IcebergModule` / `IcebergSet` constructors, `encode` (|0⟩ᵏ GHZ chain), `decode` (Z⊗ⁿ and X⊗ⁿ parity via the 2 parity ancillas), `measure` — all parameterized by n.
- Logical Paulis X̄ᵢ = X₀X_{i+1}, Z̄ᵢ = Z_{n−1}Z_{i+1}: general.
- **Now covered by the test suite at n ∈ {4, 6, 8}, bare and fault-tolerant** (these were verified by hand at n=6 during the investigation; `tests/test_iceberg_logical_action.py` is the standing check):
  - Targeted inter-patch CNOT (4 physical CX regardless of n) — general, and **for any pair of slots**, not just `qi == qj`. Important: **inter-patch communication cost is constant in n**, which strengthens the paper's story at larger k.
  - Targeted inter-patch CZ — same, and likewise slot-free.
  - `CNOT_k_Transversal` (n physical CX) = lane-wise logical CNOT on all k lanes — general.
  - `CZ_k_Transversal` (0↔n−1 crossed, middle straight) = lane-wise logical CZ on all k lanes — general.
  - `XCX_k_Transversal` — general too. The crossed anchor pattern applies C_A a second time, cancelling the one the transversal Hadamards introduce (§2.1); that is *why* the crossing is there.
  - `S` (S·CZ·S against the Z-anchor), `RZ`/`RZZ` (CX–rz–CX on the Z̄ support), `RX` (CX–rx–CX against the X-anchor) — general.
  - Per-lane `H` gadget (4 CZ + singles against both anchors) — general, (7,4) at every n.
  - `SWAP` — general, and the cheap path generalizes: a bare physical swap of two lane qubits *is* a logical SWAP at any n, so the 3-logical-CNOT path is only needed in FT mode.
- The flagged FT gadget is transparent (returns every flag ancilla to |0⟩ and preserves the logical action) for every operation at every n tested.
- `permutations/`: `generate_permutations`, `random_permutation_from_circuit`, canonical-form dedup — already slot-count agnostic, even supports heterogeneous code sizes via `logical_capacity(n)`.
- The `(patch, slot)` permutation representation everywhere (translator, pipeline, repository, postselection) — slot indices unbounded in the data model.
- `phys = patch * k_log + slot` arithmetic in context/handlers/postselection — symbolic in `k_log`.
- All single-qubit translator handlers (h, x, z, rz, rx, s, sx, swap) and `handlers/utils.py` predicates — dict-based, k-agnostic.
- `merging/rewrite.py`, `layer.py`, selection machinery — operate on generic `MacroReplacement`s.
- Simulator noise stack (Aer + Stim), scheduler, markers, tuner — flat-circuit level, patch-blind.
- `repository/`, `dag/`, most `pipeline/` plumbing.

## 2. Physics-level findings (resolved)

### 2.1 Patch-wide Hadamard (HH) — resolved: C_A is an ancilla swap

For [[n, n−2, 2]], transversal H maps X̄ᵢ → ∏_{j≠i} Z̄ⱼ and Z̄ᵢ → ∏_{j≠i} X̄ⱼ. Equivalently:

> **transversal H = H⊗ᵏ ∘ C_A**, where C_A is the linear (CNOT-network) map A = J + I over GF(2) (each bit gets XORed with the total parity). For even k, A² = I.

The original investigation concluded that C_A is a free relabel only at k=2 and entangling beyond it, and proposed three implementation tiers. **That conclusion was wrong, and the tiers are unnecessary.** C_A is a relabel at *every* n:

> **C_A = SWAP(data[0], data[n−1])** — the physical exchange of the two ancillas.

Proof, using the identities in `iceberg/layout.py`: X̄ᵢ = X₀X_{i+1} and Z̄ᵢ = Z_{n−1}Z_{i+1}, while X₀X_{n−1} ≡ ∏ⱼ X̄ⱼ and Z₀Z_{n−1} ≡ ∏ⱼ Z̄ⱼ modulo the stabilizers. Exchanging the two ancilla positions therefore sends X₀X_{i+1} → X_{n−1}X_{i+1} = ∏_{j≠i} X̄ⱼ and Z_{n−1}Z_{i+1} → Z₀Z_{i+1} = ∏_{j≠i} Z̄ⱼ, which is C_A. Both stabilizers are permutation-invariant, so the code space is preserved.

Consequences:

- **`HH` = transversal H + one SWAP, at any n** — cost (n, 0) counting the swap as the free relabeling it is. Nothing about the merging pass's use of HH needs to change for k>2, mid-circuit or otherwise. No frame tracking, no classical postprocessing hook, no cost-model penalty.
- At k=2, A happens to *also* equal the lane permutation (0 1), which is why the old code's `swap(data[1], data[2])` worked there. That was the accident; the anchor swap is the general statement.
- The same identity explains the decode convention trap in §2.2 and both k=2-only gate hacks: the "wrong anchor" version of any X-type operation differs from the intended one by exactly C_A.

Tiers 1–3 as described in the original plan are withdrawn. Ignore any downstream note that refers to them.

### 2.2 Silent-wrongness bugs — all fixed

All five were confirmed by the tests before being fixed; the root cause of four of them is the same: an X-type logical operation built against the Z-anchor (or vice versa) differs from the intended gate by C_A, which is invisible at k=2.

- ~~`IcebergModule.H` full-patch branch~~ — now routes to `HH`, which applies the anchor swap (§2.1). General.
- ~~`IcebergSet.XCX` cross-patch (k=2 `i_swapped` compensation)~~ — replaced. Cross-patch XCX is now the X-type dual of cross-patch CZ: `x_basis_cz` over the four pairs drawn from the two X-supports, telescoping to 4 CNOTs + 4 Hadamards. Works at any n **and any pair of slots**, and is cheaper than the old n=4 path ((4,4) vs (8,4)).
- ~~`IcebergModule.XCX` intra-patch (fell back to `HH`)~~ — replaced by the X-type dual of the intra-patch CZ construction: 3 two-qubit gates + 5 single-qubit gates at any n, which is exactly the (5,3) the cost table already claimed.
- ~~`IcebergModule.SX`~~ — derived. The old construction rotated about X_{data[j]}X_{data[n−1]} = ∏_{l≠j−1} X̄ₗ, i.e. it used the **Z-anchor for an X-type gate**. At k=2 that product is the single other lane, which is why the `2−j` index reversal made it accidentally correct. The fix is to anchor on `data[0]` and drop the reversal — same gate count, general in n, and now cross-checked against `RX(π/2)`.
- ~~`iceberg_postselect._PHYS2LOGIC`~~ — replaced by `CodeLayout.decode_logical`. The rule is simpler than the plan proposed: **decode against the Z-anchor**, logicalᵢ = bit(lane i) ⊕ bit(data_{n−1}), after rejecting odd-parity strings. That is the direct eigenvalue of Z̄ᵢ and needs no parity-complement step. (The plan's "raw ⊕ parity(raw)·1" rule is equivalent, being C_A applied to the X-anchor reading.) One further trap the investigation missed: **n=4 cannot distinguish the two bit orders either** — reversing a 4-bit even-parity codeword leaves the decoded value unchanged — so the old table was silently reversal-invariant. The production order is fixed by `simulator.Simulator`, which inverts Aer's little-endian output so that position i *is* bit i; `tests/test_iceberg_roundtrip.py` pins this at n=6, where it matters.

### 2.3 Cost model as a function of n — implemented

`iceberg/costs.py:work_table(n)` is now the single source; `packer/work_table.py` and `translator/merging/conflicts.py` import from it, and `tests/test_iceberg_costs.py` asserts both that `work_table(4)` reproduces the old hardcoded values exactly and that every entry matches the gate counts actually emitted at n ∈ {4,6,8}. The third copy — the bias constants inside `packer/merge_packer.py:_score_pair` — is still separate (Phase 4).

| Logical op | n=4 | General n |
|---|---|---|
| CX targeted inter-patch | (0, 4) | (0, 4) — constant |
| CZ targeted inter-patch | (0, 4) | (0, 4) — constant |
| XCX targeted inter-patch | (8, 4) → **(4, 4)** | (4, 4) — constant (was 2n singles) |
| CXT transversal | (0, 4) | (0, n) |
| CZT transversal | (0, 4) | (0, n) |
| XCXT transversal | (8, 4) | (2n, n) |
| H per-qubit | (7, 4) | (7, 4) — constant |
| HH patch-wide | (4, 0) | (n, 0) — plus one free anchor SWAP, §2.1 |
| Intra-patch CX | (0, 4) | (0, 4) — constant |
| Intra-patch CZ | (1, 3) | (1, 3) — constant |
| Intra-patch XCX | (5, 3) | (5, 3) — constant |
| S / SX | (2, 1) / (4, 1) | same — constant |
| RZ / RX / RZZ | (1, 2) | same — constant |
| Encode | (1, 3) | (1, n−1) |
| Decode/syndrome | (8, 8) | (2n, 2n) + 2 ancillas |

Almost nothing scales with n, because logical operators stay weight-2 at every code size. That sharpens the packing story: **targeted cross-patch gates cost the same at any n while their transversal counterparts grow linearly**, so the larger the patch, the worse transversal merging looks relative to targeted placement — worth a sentence in §5 of the paper.

FT/flagged variants add 2 ancillas + (2 single, 5 two-qubit) gates per flagged two-qubit gate, n-independent (`costs.flagged_overhead`). Still to follow in Phase 4: `MergeWeights` calibration (λ=13, γ=0.2, solo divisor 8 ≈ (c(H)−c(HH))/8 — all tuned against the n=4 table), and the depth-bias constants in `_score_pair`.

### 2.4 Bonus: the slot-alignment restriction is not a property of the code

Targeted cross-patch CNOT/CZ/XCX only involve the two weight-2 supports, so they are correct for **any pair of slots** — the `qi == qj` guard and `NotImplementedError` are gone, verified at n ∈ {4,6,8} for every slot pair. The translator still inserts SWAPs to align slots before a cross-patch CNOT (`translator/handlers/controlled_not.py:91–123`); those SWAPs are now avoidable, which is free savings available in Phase 2. Genuine transversal (all-lane) macros still require alignment.

## 3. Compiler-level blockers by module

Ranked by depth of change (full file/line inventory in Appendix):

1. **`translator/merging/`** — broad but no longer deep, now that HH needs no frame (§2.1). `HHGate`/`XXGate`/`ZZGate` are fixed 2-qubit `Gate`s (→ k-qubit, parameterized); `_build_patch_pairs` keeps exactly 2 lanes and silently drops the rest; single-qubit scan hardcodes `lane_map = {qubits[0]: 0, qubits[1]: 1}`; `MacroReplacement.first/second` fuses exactly 2 gates (→ list of k); transversal matching requires exactly 2 CXs whose slot map is identity or the 2-cycle (→ k CXs + slot-permutation matching); savings formula `2*base − macro` (→ `len(originals)*base − macro`). ~~own copy of the n=4 work table~~ (now imports `iceberg.costs`).
2. **`packer/`** — algorithmic reformulation. Explicit `patch_capacity != 2` guard; `_pick_pairs_best_path` returns 2-tuples; pairs+solos dichotomy; `w_ij − w_ii − w_jj` net-gain defined over pairs; `_top_k_pairs` enumerates 2-subsets; solo-budget logic `num_qubits % 2`. Scoped plan (per P3.1): keep greedy, extend to k-subsets by **iterative pair-merging into 4-sets** — score pairs as today, then score pair-of-pairs unions with a generalized `GroupMergeAnalyzer`, commit best 4-sets by net gain. Avoids full hypergraph matching while directly answering Rev B.
3. **`translator/` core+context+handlers** — mostly mechanical. `TranslationContext.from_config` hardcodes `IcebergSet(n=4)` (the `TranslatorConfig.n_code` knob exists but is **never wired through** — dead config); `k_log: int = 2` default; `update_states_on_swap` ignores its slot args and flips 0↔1 for the whole patch (silently wrong for k>2 — must become a true transposition of the two swapped slots); three copy-pasted `k_log == 2` macro-realignment branches in cnot/cz/xcx handlers (→ general slot-permutation alignment via SWAP networks); `mapping_for_perm(n_code=4)` default; `control_count` fallback assumes equal-size halves.
4. **`simulator/` + `evaluator/` + pipeline metrics** — trivial once threaded. `PHYSICAL_QUBITS_PER_PATCH = 6` (= n+2); `simulator.py` infers patch count via `num_qubits // 6`; `qubits_per_code=6` defaults and the two call sites in `result_row.py`. Replace with a value derived from config (or an explicit layout object). Note: register width per module is actually n+2 **plus flag ancillas in FT mode** — inference-by-division is fragile; pass the layout explicitly.
5. **`permutations/`** — nearly done. `permutation_with_n_codes` caps 2 slots/patch; `entrypoints.py` and five pipeline tasks rely on the `n=4` default of `generate_uniform_code_sets` (which is itself properly parameterized).
6. **`tests/` + `experiments/`** — `test_cxt_dynamic_alignment.py` pins the k=2 slot-flip realignment semantics (will need parameterized rewrites, and is the safety net for Phase 2); experiments carry k=2 permutation literals and `// 6` groupings — non-blocking, migrate opportunistically.

## 4. Phased implementation plan

**Phase 0 — Single source of truth (S) — DONE except the `n_code` wiring.**
`iceberg/costs.py` holds the n-parameterized work table (§2.3); `packer/work_table.py` and `merging/conflicts.py` import from it, with a test asserting n=4 equality against the values they used to hardcode. No `NotImplementedError` guards were needed: the silent-wrong paths were generalized outright rather than fenced off. **Still to do:** wire `TranslatorConfig.n_code` → `Translation` → `TranslationContext` → `IcebergSet` (the knob exists and is ignored; `context.py:109` hardcodes `n=4`) — moved into Phase 2, where it belongs.

**Phase 1 — Physical layer correctness for general n (M) — DONE.**
`iceberg/` is general and tested at n ∈ {4,6,8}, bare and FT. No `if self.n == 4` fast paths were kept: in every case the general construction has the same gate count as the n=4 special case it replaced (and cross-patch XCX is cheaper), so the special cases were pure duplication. New modules: `layout.py` (the convention), `physical.py` (gate emission), `costs.py`. Tests: `test_iceberg_logical_action.py` (96 cases, exact symplectic logical-action comparison — the keystone suite), `test_iceberg_rotations.py`, `test_iceberg_roundtrip.py` (convention pinning end-to-end), `test_iceberg_costs.py`. Support tool: `tests/logical_action.py`.

**Phase 2 — Translator plumbing (M, ~3–4 days).**
Wire `n_code` through (from Phase 0); `k_log` from `n_code` everywhere; true-transposition `update_states_on_swap`; replace the three `k_log == 2` realignment branches with general slot-permutation alignment (decompose needed permutation into intra-patch SWAPs; cost via `iceberg/costs.py`); fix `control_count` fallback. **Free win available here:** targeted cross-patch CNOT/CZ/XCX no longer need aligned slots (§2.4), so the alignment SWAPs in `place_logical_cnot` can be dropped for non-transversal placements. **Test:** parameterized version of `test_cxt_dynamic_alignment` at k=2 and k=4; end-to-end translate-and-simulate a small circuit at n=6 with naive (no-merge) settings, compare logical distribution to unencoded reference (the round-trip helpers in `tests/test_iceberg_roundtrip.py` already do this against hand-built `IcebergSet` circuits — point them at the translator instead).

**Phase 3 — Merging generalization (L) — DONE.**
All of the planned work landed: k-qubit macro classes (`_MultiLaneMacroGate`), k-lane scanning in `single_qubit`/`preprocessing` (`_build_patch_pairs` → `_build_patch_lanes`), `MacroReplacement.first/second` → `MacroReplacement.nodes`, transversal matching over k slots with general permutation alignment, and savings formulas that scale with `len(record.nodes)` and with `n` (`conflicts.merge_work_table`). `Translator._merging_available` is gone; merging runs at every code size.

The organizing idea the plan did not have: **macros split into whole-patch and per-lane, and only the first needs a coverage condition.** `hh`/`cxt`/`czt`/`xcxt` are built from physical gates on all n data qubits, so they act on every lane whether or not a program gate asked; a group must therefore cover every *occupied* lane, which means k program gates rather than two. `xx`/`zz` are per-lane — `X̄ᵢ = X[x_anchor] X[lane i]`, so any subset is legal and an even subset cancels the anchor factor. At k=2 a group of two *is* the whole patch, which is exactly why the pairwise code needed no notion of coverage and why introducing one was most of the work. Coverage is enforced twice, at detection and again against the live slot map in `handlers.utils.resolve_transversal_macro`, since a SWAP can intervene.

**Fusing a subset of lanes is not available**, and this is a fact about the code rather than a limitation: partial transversal CX maps the full-weight stabilizer `S_X = X^⊗n` to a weight-|T| operator outside the stabilizer group, so it is not a logical operation. The reachable middle ground is transversal-then-correct (fire the full macro, undo unwanted lanes individually), which the cost table says pays only for |T| ≥ k − ⌊k/3⌋ — 3 of 4 at k=4, 7 of 10 at k=10. Not implemented; see §5.7.

Alignment turned out to be free rather than costly: a group's control→target slot map is any permutation of k slots, realigned by permuting the target patch via routing swaps, and routing swaps are atom movement priced (0, 0). The binding constraint on a merge at large k is scheduling — getting k gates adjacent — not cost.

Tests: `tests/test_merging_generalized.py` (62 cases at n ∈ {4,6,8,10,12}) builds the programs that trigger each macro, because random Clifford programs almost never put a gate on all k lanes at once and so report a vacuous pass at k > 2. It pins the negative cases too: 2-of-k H must *not* fuse at k > 2 while the same input does fuse at k=2. `test_merging_preserves_the_logical_action` widened from n=4 to all five sizes.

**Phase 4 — Packer k-subset extension (L, ~1 week).**
`GroupMergeAnalyzer` generalizing `PairMergeAnalyzer` to qubit groups; iterative pair-merge-into-4-sets greedy (§3.2); recalibrate `MergeWeights` against the n=6 cost table (`packer/calibrate.py` scaffolding exists); remainder/solo logic for `num_qubits mod k ≠ 0`.

**Phase 5 — Simulator/evaluator/pipeline threading + the [[6,4,2]] experiment (S–M, ~2–3 days).**
Thread patch size (n+2) from config; explicit layout instead of `// 6`; `n` parameter on the five pipeline tasks; rerun a QASMBench subset at n=6 with `ScaledNoiseModel` (per revision-plan footgun note) for the P3.1 figure.

**~~Fix before trusting any rerun~~ — FIXED.** `Simulator._shot_chunk_size_for` caps the chunk at `repetitions // 100`, and each chunk was run on an Aer backend constructed with one fixed `seed_simulator`. Aer's `seed_simulator` seeds a *run*, not a backend, so a seeded run produced 100 identical chunks — `repetitions / 100` worth of independent samples, not `repetitions`, and 10× the sampling error every shot count implied. It reproduced as probabilities quantised to 1/200 on a 20000-shot `h; h; measure`.

Chunks now draw their own seeds from a PRNG keyed on the configured seed (`Simulator._chunk_seed_stream`), which keeps a seeded run reproducible, uses every shot, and — because the derivation is not `seed + index` — leaves neighbouring base seeds independent rather than sharing a stream. Pinned by `tests/test_simulator_seeding.py`, whose two load-bearing cases were confirmed to fail against the old code. Any seeded noisy number produced before this revision still carries the inflated error; the Stim backend was never affected, since it reuses one sampler and `sample()` advances its state.

**Verification step (every phase):** n=4 golden-path regression (same permutations, same circuits, identical outputs) + the Phase 1 logical-action suite at n∈{4,6,8}. Sandbox is fine for unit tests; heavy sim reruns on the Mac (Python 3.13 requirement).

## 5. Open decisions

1. ~~**HH tier for k>2**~~ — moot. C_A is the anchor swap, so HH is (n, 0) + one free SWAP at every k (§2.1).
2. **Uniform n only, or mixed-n code sets?** `permutations/` already supports mixed; translator/evaluator assume uniform. Recommendation: uniform-n for the revision (matches P3.1), keep mixed-n out of scope. (`iceberg/` itself is per-patch, so `IcebergSet` is the only place that assumes uniformity — it builds `num_modules` patches of one size.)
3. **Packer scope:** greedy 4-set extension (recommended) vs. general k-subset hypergraph formulation (future work section). **The only remaining design decision.**
4. ~~**`SX` semantics**~~ — derived and fixed (§2.2): the `2−j` reversal was compensating for the use of the Z-anchor in an X-type gate.
5. **Paper tie-in:** §2.3's table *is* the P0.3 generalization table. Two things changed in the story: (a) the merge/packing trade-off now has a crisp statement — targeted cross-patch gates are constant in n, transversal ones linear, so transversal merging gets *relatively* worse as k grows; (b) the patch-wide Hadamard is cheap at every k, so there is no "HH only works at k=2" caveat to write. Both deserve a sentence in §5.
6. **New:** should the translator stop aligning slots for targeted cross-patch gates (§2.4)? It is a strict win in gate count but changes which permutations are equivalent, so it interacts with the permutation search space and the packer's model. Recommend doing it in Phase 2 and measuring.
7. **New (from Phase 3):** is transversal-then-correct worth building? A whole-patch macro needs a gate on all k lanes at once, and measurement says that gets rare fast — merging fires on 11/30 small circuits at k=6 against 14/30 at k=2, and its contribution decays with k rather than growing. Firing the full macro and undoing the unwanted lanes individually recovers the near-misses, and the cost table puts the break-even at |T| ≥ k − ⌊k/3⌋. That makes it a *placement* question as much as a rewriting one: the packer would need to aim for near-alignment of whole patch pairs, which is a different objective from the cut-weight one `grouping` currently optimizes. Recommend costing it against the measured near-miss distribution before building.
8. **New (from Phase 3):** the merging pass flushes its candidate group at a program SWAP rather than reasoning across it, so `HH; SWAP; HH` fuses nothing. Correct but conservative, and the cost is larger at k > 2 where a group spans more gates and so is likelier to contain a SWAP. Tracking the relabel through the scan instead is contained work; `test_program_swap_between_macros_stays_exact` already pins the correctness requirement.

---

## Appendix — Full inventory of hardcoded [[4,2,2]] sites

### iceberg/ — all resolved
| Site (pre-refactor line numbers) | Issue | Resolution |
|---|---|---|
| `iceberg_module.py:314–325` (`H`, full-patch) | H⊗ⁿ + swap(1,2) unconditional — silently wrong n>4 | routes to `HH` |
| `iceberg_module.py:345–359` (`HH`) | raised for n≠4 | transversal H + `swap(x_anchor, z_anchor)` (§2.1) |
| `iceberg_module.py:361–386` (`CNOT`) | n=4 non-FT shortcut | deleted — it emitted exactly the general path |
| `iceberg_module.py:405–439` (`XCX`) | n=4 path; fallback crashed n≠4 | X-type dual of `CZ`, (5,3) at any n |
| `iceberg_module.py:276–304` (`SX`) | `data_qubits[j]` / `2−j` reversal | anchored on `x_anchor`, reversal dropped |
| `iceberg_module.py:517–540` (`SWAP`) | n=4 non-FT shortcut | generalized: `swap(lane_i, lane_j)` at any n |
| `iceberg_set.py:375–406` (`XCX` cross-patch) | k=2 compensation, silently wrong k>2 | 4 × `x_basis_cz` over the X-supports, any slots |
| `iceberg_set.py` (`CNOT`/`CZ` cross-patch) | `NotImplementedError` unless `qi == qj` | restriction lifted (§2.4) |
| `iceberg_postselect.py:19–28` | 4-bit `_PHYS2LOGIC` lookup | `CodeLayout.decode_logical`, any n |
| 5 copy-pasted single-qubit dispatchers in `iceberg_set.py` | `X`/`Z`/`S`/`SX`/`H` each re-implemented the group-by-patch loop | one `_broadcast` / `_apply_at` pair |

Also of note in `iceberg/`, not in the original inventory:
- `HH(fault_tolerant=True)` emitted a bare SWAP while `H(all lanes, fault_tolerant=True)` emitted three *flagged* CNOTs — the same logical operation with two different FT policies. Unified on the bare SWAP (the path the translator actually takes, so no experiment changes); `SWAP_GATE_ALLOWED = False` still gets the flagged decomposition.
- `base.py`'s `AutoCheckEncodedMeta` / `mark_check_encoded` machinery is inert: `mark_check_encoded` returns the method unchanged, so `_check_encoded` is never set and `check_encoded` is never applied. Left alone (it is the abstract interface, orthogonal to this work) but it is dead code.

### translator/
| Site | Issue |
|---|---|
| `context.py:41` | `k_log: int = 2` default |
| `context.py:69–93` | `update_states_on_swap` flips 0↔1 patch-wide — wrong for k>2 |
| `context.py:108–114` | `IcebergSet(n=4)` hardcoded; `n_code` config never reaches here — **the one line that gates n=6 end-to-end** |
| `config.py:14`, `core.py:99`, `entrypoints.py:32–36` | dead `n_code` knob, re-hardcoded 4 |
| `utils.py:38` | `mapping_for_perm(n_code=4)`, uniform-size assumption |
| `handlers/controlled_not.py:192–203`, `controlled_z.py:199–208`, `controlled_x.py:173–182` | copy-pasted `k_log == 2` macro realignment |
| `handlers/controlled_{not,z,x}.py` (~150/~131) | `control_count` fallback `len(op.qubits) // 2` |
| `merging/gates.py:10–37` | 2-qubit HH/XX/ZZ `Gate` classes |
| `merging/preprocessing.py:141–157` | `_build_patch_pairs` keeps exactly 2 lanes |
| `merging/single_qubit.py:29,87–99` | 2-lane scans |
| `merging/models.py:21–51,87–97` | `MacroReplacement.first/second`, `PairMergeStats`, single-slot `_PendingTransversal` |
| `merging/transversal.py:145–173` | exactly-2-CX matching, identity/2-cycle slot maps only |
| ~~`merging/conflicts.py:12–27`~~ | ~~n=4 work-table copy #2~~ — now imports `iceberg.costs` |
| `conflicts.py:353`, `preprocessing.py:183`, `selection.py:193` | `2*base − macro` savings |
| `merging/analysis.py` | `PairMergeAnalyzer` pairwise by construction |

### packer/
| Site | Issue |
|---|---|
| `merge_packer.py:55,81–82` | `patch_capacity=2` default + hard guard |
| `merge_packer.py:102–120,239–316,319–359` | pairs+solos dichotomy, 2-way matching, pairwise scoring, n=4 bias constants |
| ~~`work_table.py:14–38`~~ | ~~n=4 work-table copy #1 + tuned weights~~ — now imports `iceberg.costs` |
| `work_table.py:110,145–202` | odd/even solo logic, 2-gate merges, pair edges |
| `calibrate.py:65–77,215,404–406,559–565` | k=2 calibration geometry |

### simulator/ evaluator/ pipeline/
| Site | Issue |
|---|---|
| `simulator/grid_constructor.py:15,53–55` | `PHYSICAL_QUBITS_PER_PATCH = 6` |
| `simulator/simulator.py:258` | `num_qubits // 6` patch inference |
| `evaluator/gate_count_stats.py:138,187` | `qubits_per_code=6` defaults |
| `pipeline/analysis/result_row.py:45,59` | `qubits_per_code=6` call sites |
| `pipeline/tasks/{scan_all,scan_random_m,scan_extremals,generate_permutations,generate_circuit_random}.py` | rely on n=4 default of `generate_uniform_code_sets` |

### permutations/ tests/
| Site | Issue |
|---|---|
| `generator.py:250–269` | `permutation_with_n_codes` caps 2 slots/patch |
| `entrypoints.py:3–45`, `code_sets.py:88` | n=4 defaults |
| `tests/test_cxt_dynamic_alignment.py` | pins k=2 slot-flip semantics (rewrite parameterized in Phase 2) |

*(experiments/ carries k=2 permutation literals, `control_count=2` macros, and `// 6` groupings throughout — non-blocking, migrate opportunistically.)*
