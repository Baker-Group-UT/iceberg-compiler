# Logical pre-optimization of the Fig 6 inputs (v2 revision)

Concern: AQORN's gains might be the clean-up of sloppy input circuits. Two questions:
(1) are the 71 Fig 6 circuits already near-optimal logical circuits, and (2) are AQORN's Fig 6
improvements the same when the input has first been through a strong logical optimizer?

## Answer

1. **Near-optimal for most circuits' two-qubit structure, not for the corpus as a whole.** The
   strongest Qiskit-only optimizer available leaves 15 circuits untouched and the 2q count of 41
   of the 69 circuits with 2q gates unchanged, but removes 31-84% of the 2q gates of 13
   circuits and shrinks single-qubit structure broadly (46% of all Hadamards). Per-circuit mean
   reduction: 1q 17.4%, 2q 10.8% (median 0), all gates 16.3% (median 7.0%), depth 11.3%
   (median 4.7%).
2. **AQORN's depth gain survives intact; 60-70% of its gate gains and about 80% of its TVD
   gain survive; most of its LSR gain does not.** w10, fair (every stage layer-sorted),
   original -> optimized input: depth 15.3% -> 15.3%, 2q 15.5% -> 9.6%, 1q 24.2% -> 17.5%,
   TVD reduction 38.2% -> 29.7%, LSR improvement 25.9% -> 9.7%. The losses are concentrated:
   over the 56 changed circuits the median per-circuit change is -0.9 points for 2q, +1.1 for
   depth, -0.4 for TVD and -2.9 for LSR, against mean changes of -7.5, 0.0, -11.4 and -20.5. The
   biggest are the five Bernstein-Vazirani circuits, whose Hadamard layers the optimizer's
   Clifford resynthesis deletes (279 -> 3 on bv_n140) -- the same redundancy AQORN's Hadamard
   commutation removes, so on those circuits the optimizer takes AQORN's gain -- and the small
   Clifford encoders and two-qubit-block-redundant circuits.
3. **Complementary, not redundant.** Against the original input's baseline (w10): optimizer
   alone 14.0% 2q / 12.5% depth / 21.9% TVD; AQORN alone 15.5 / 15.3 / 38.2; both 22.8 / 27.0 /
   48.2. The one metric where the optimizer alone beats AQORN alone is LSR (+47.2% vs +25.9%
   mean; medians +14.5% vs +21.8%), because at Fig 6's per-circuit alpha removing logical gates
   from small circuits raises the selection rate exponentially.

This undercuts any claim that AQORN's *gate-count* and *LSR* gains are independent of input
quality; it supports the claim for depth, and largely for TVD. Against plain Qiskit O3 (the
optimizer a skeptic would most likely try; static stages only) the picture is milder: w10's 2q
reduction 15.5% -> 12.8%, depth 15.3% -> 17.7%.

## Method

**Inputs.** Fig 6's 71 circuits exactly as Fig 6 loads them (`load_fig6_circuits()` over the
full corpus = the Fig 6 script's `_load_group_circuits`: QASMBench, level-1 transpile into
{H, X, Z, S, Rz, Rx, CX, CZ}, measurements dropped), with Fig 6's size groups and indices.
Translating each original at the baseline reproduces Fig 6's `baseline_gate_count` exactly on
71/71 circuits. See "Loader check" below for basis_trotter_n4.

**Optimizer** (`experiments/logical_opt/optimizer.py`, Qiskit 2.2.1 only; PyZX, BQSKit and
pytket are not installed). Barriers removed, then seven routes:

| route | what it does |
|---|---|
| `o3` | `transpile(optimization_level=3)` into the corpus basis {h, x, z, s, rz, rx, cx, cz} |
| `o3_sx` | the same into the translator's full single-qubit set (adds sx) |
| `clifford_ag`, `clifford_greedy` | Clifford circuits only: whole-circuit tableau resynthesis (Aaronson-Gottesman / greedy), then O3 |
| `clifford_blocks` | every maximal Clifford block resynthesized (greedy), then O3 |
| `linear_blocks` | every CX block resynthesized (Patel-Markov-Hayes), then O3 |
| `commute` | 1q runs commuted through 2q gates and resynthesized, commutative inverse cancellation, then O3 |

Every O3 call is the best of 3, because Qiskit 2.2.1's O3 is not deterministic (same circuit,
same `seed_transpiler`, `PYTHONHASHSEED=0`, one Rust thread: 1q counts and depth differ call to
call; 2q counts were stable in every probe). O3 alone is iterated to a fixed point (re-run on
its output until the score stops improving): one improving pass on 49 circuits, none on 22, and
a second pass never improved. The all-routes search is iterated the same way (every route on
the current circuit, keep the best; at most four improving rounds). Score: lexicographic (2q
gates, all gates, depth), O3's own priority. The result is the best of {original, O3 fixed
point, all-routes fixed point}; when nothing beats the original, the original circuit object
is kept, barriers and all. Program swaps are excluded from the target set (the translator
implements one as a relabel only inside a patch and rejects a cross-patch swap). No optimized
circuit contains sx. The chosen circuits are cached as QPY, so Stages B and C use exactly what
was verified.

**Equivalence**, all 56 changed circuits, up to global phase: `Operator` overlap
|tr(A^dag B)|/d >= 1 - 1e-9 for n <= 12 (23 circuits), Clifford-tableau equality (6), otherwise
a miter amplitude |<psi| U_opt^dag U_orig |psi>| >= 1 - 1e-7 for |0...0> and two Haar-random
product states, Aer statevector to 20 qubits (5) and Aer MPS beyond (22). All pass; the worst
is 1 - 1.3e-13. One genuine approximation was caught while building this and designed out:
Qiskit's matrix-based Clifford test accepts `rx(1.570068)` (7e-4 off pi/2), so
`CollectCliffords(matrix_based=True)` silently rounds near-Clifford rotations (3e-6 process
infidelity on basis_trotter_n4). The block route snaps rotations to named Cliffords at 1e-9 and
collects by name instead.

**Pipeline (Stage B).** Fig 6's stages on the original and on the optimized circuit: baseline
(tight trivial packing, both passes off), +H, +H+merge, packed, for v4 (`plan_packing`, the
submitted AQORN), w10 (weight-10 silent-aware planner, what the paper now uses) and w10T (w10
plus `TRANSVERSAL_INTRA_TWO_QUBIT` on its merging-enabled stages, planned with the switch on and
`gadget_rates_transversal.csv`). Each input is planned on itself with Fig 6's noise profile; the
planner's commutation decision drives +H and +H+merge, as in Fig 6. Every stage is translated
with `TranslatorConfig.asap_order = True` (the fair baseline) and, for reference, in shipped
order. All numbers are the caption convention: unclipped per-circuit mean of the cumulative %
reduction against the same input's baseline.

**Simulation (Stage C).** Baseline and packed arms (w10, w10T) of every changed circuit at its
Fig 6 alpha and shot count (the original's physical noise, not recalibrated), Fig 6's seeds
(`10 * 1_000_003 + idx * 17 + offset`, offsets 3 / 6) and Fig 6's shipped emission order, on
the Clifford proxy built by Fig 6's own rule (every rz/rx -> s) from the optimized circuit: 168
arms, 0 failures. The 15 unchanged circuits reuse their source figures' samples (not
resimulated). v4 was not simulated (static only).

**Guards, all passed.** (i) Source-figure reproduction: with the plans each figure was drawn
with, the recomputed shipped-order 1q/2q/depth percentages of every original arm equal Fig 6
(v4), `ablation_silent_aware_w10.csv` (w10) and `ablation_variant_w10T.csv` (w10T) to 1e-6,
213/213. (ii) Planner gate: re-planning four originals (adder_n10, qaoa_n6, bell_n4, qft_n4) with
each planner returns its cached plan exactly, 12/12. (iii) Simulation gate: bell_n4's baseline
arm and adder_n10's w10 packed arm, re-drawn with Fig 6's seeds, equal the figures' samples
exactly. (iv) Noiseless miscompile check (`miscompiled`, against the naive arm) on every packed
arm of both inputs and all three planners: 0 of 381 flagged.

## Results

### Stage A: how far from optimal are the inputs?

Logical reduction from optimization, per-circuit mean % (bb84_n8 and qrng_n4 have no 2q gates
and count as 0):

| group | circuits | changed | 1q | 2q | all gates | depth | 2q, O3 only | all gates, O3 only |
|---|---|---|---|---|---|---|---|---|
| small | 29 | 22 | 19.6 | 17.1 | 19.7 | 21.1 | 14.2 | 11.8 |
| medium | 15 | 13 | 19.2 | 5.1 | 14.0 | 6.3 | 2.3 | 5.8 |
| large | 27 | 21 | 14.0 | 7.4 | 13.8 | 3.5 | 6.9 | 9.6 |
| all | 71 | 56 | 17.4 | 10.8 | 16.3 | 11.3 | 8.9 | 9.7 |

Corpus totals: 1q -20.9%, 2q -18.4%, depth -20.3% (dominated by basis_trotter_n4, dnn, hhl_n7,
qugan). Hadamards 1,989 -> 1,082; rotations 10,238 -> 8,702. O3 alone finds most of the 2q
saving (8.9 of 10.8 points) but only about 60% of the all-gate saving (9.7 of 16.3); the
Clifford and block routes find the rest, including most of the Bernstein-Vazirani Hadamard
removal (O3 alone removes about half of those Hadamards, the tableau route nearly all).

What changed and why:

* **Untouched (15):** every cat/GHZ circuit (7), adder_n4, fredkin_n3, hs4_n4, qec_en_n5,
  qrng_n4, teleportation_n3.
* **Large 2q cuts, small circuits:** error_correctiond3_n5 -84% (a Clifford encoder, whole-
  tableau resynthesis), dnn_n8 -67%, basis_trotter_n4 -63%, pea_n5 -60%, hhl_n7 -53%,
  variational_n4 -50% (two-qubit block resynthesis), qec9xz_n17 -34% (Clifford), wstate_n3 and
  qaoa_n6 -33%, bell_n4 -29%.
* **Moderate 2q cuts, wide circuits:** dnn_n33/n51 and qugan_n39/n71 -31% (block resynthesis,
  paid for in rotations: their 1q count rises by up to 28%), knn and swap_test -12.5% (Clifford-
  block resynthesis of the controlled-swap ladders), adders and bigadder -6 to -8%.
* **1q only:** Bernstein-Vazirani -39 to -73% of 1q gates (the H layer around the CX fan-in is
  resynthesized away), wstate -33%, ising -21% (merged rotation chains), qft, qpe, lpn_n5,
  linearsolver_n3.
* **Trades:** six circuits (knn x4, adder_n28, adder_n64) get up to 11% deeper for fewer 2q gates;
  the score is 2q-first by design.

### Stage B: AQORN's static gains on each input

Cumulative % reduction against the same input's baseline, depth fair:

| planner | metric | original: small / medium / large / **all** | optimized: small / medium / large / **all** | delta all [95% CI] |
|---|---|---|---|---|
| w10 | 1q | 24.7 / 27.8 / 21.7 / **24.2** | 18.8 / 20.8 / 14.3 / **17.5** | -6.7 [-10.9, -2.9] |
| w10 | 2q | 16.5 / 17.2 / 13.6 / **15.5** | 12.4 / 8.5 / 7.3 / **9.6** | -5.9 [-9.2, -3.0] |
| w10 | depth | 23.2 / 17.9 / 5.4 / **15.3** | 18.9 / 18.3 / 9.8 / **15.3** | -0.0 [-3.4, +3.1] |
| w10T | 1q | 9.5 / 13.5 / 17.2 / **13.3** | 3.4 / 9.7 / 9.2 / **6.9** | -6.4 [-12.4, -0.9] |
| w10T | 2q | 22.0 / 20.5 / 15.7 / **19.3** | 18.3 / 11.1 / 8.6 / **13.1** | -6.2 [-9.4, -3.3] |
| w10T | depth | 24.8 / 19.6 / 5.5 / **16.4** | 21.4 / 19.6 / 10.0 / **16.7** | +0.3 [-2.9, +3.4] |
| v4 | 1q | 22.5 / 25.5 / 20.7 / **22.4** | 17.1 / 18.7 / 13.7 / **16.2** | -6.3 [-10.8, -2.1] |
| v4 | 2q | 17.5 / 17.3 / 15.8 / **16.8** | 13.6 / 9.9 / 7.9 / **10.7** | -6.1 [-9.4, -3.2] |
| v4 | depth | 21.9 / 15.2 / 8.7 / **15.5** | 18.0 / 13.1 / 11.9 / **14.6** | -0.9 [-4.0, +2.1] |

(CI: bootstrap over circuits of the per-circuit paired difference.) The original-input column
reproduces the known fair-depth numbers (v4 15.5, w10 15.3, w10T 16.4) and Fig 6's gate numbers
(v4 22.4 / 16.8). In shipped emission order (Fig 6 as drawn) depth goes v4 28.9 -> 26.8, w10
31.2 -> 30.9, w10T 32.7 -> 32.0; gate counts are identical in both orders.

Per stage (w10, cumulative, all circuits): 2q +H 6.9 -> 4.9, +H+merge 14.0 -> 8.6, packed
15.5 -> 9.6; depth +H 8.2 -> 6.8, +H+merge 13.4 -> 12.4, packed 15.3 -> 15.3. The optimizer
takes material from Hadamard commutation and from merging; packing's own 2q share is about
unchanged (1.5 -> 1.0 points) and its depth share grows (1.9 -> 2.9).

Where the 2q loss comes from (w10, per-circuit change in points): bv_n30..n140 -38 to -50
(AQORN's 40-51% becomes 0-6%), error_correctiond3_n5 -36, variational_n4 -28, lpn_n5 -26,
basis_trotter_n4 -23, qec9xz_n17 -20, dnn_n8 -16. The more the optimizer saves on its own, the
less AQORN adds (correlation -0.66 between the optimizer's physical-2q saving and the change in
AQORN's). Without the five Bernstein-Vazirani circuits, w10's 2q reduction goes 13.1 -> 10.2, 1q
21.4 -> 17.7, depth 14.8 -> 15.4. Some circuits gain: qaoa_n6 +14 points, bell_n4 +6, simon_n6
+5, ising +3 to +5.

### Stage C: fidelity

Packed vs the same input's baseline; TVD on the 59 circuits whose TVD both inputs resolve (the
original-input TVD numbers are therefore over 59 circuits, not the 66 of the source figures,
where w10 reads 38.7%):

| planner | metric | original | optimized | delta [95% CI] | median delta | changed circuits only: original -> optimized |
|---|---|---|---|---|---|---|
| w10 | LSR improvement % | 25.9 | 9.7 | -16.1 [-24.6, -8.9] | -0.9 | 32.2 -> 11.7 (n=56) |
| w10 | TVD reduction % | 38.2 | 29.7 | -8.5 [-16.2, -1.6] | 0.0 | 38.4 -> 27.0 (n=44) |
| w10T | LSR improvement % | 41.3 | 18.9 | -22.4 [-31.8, -13.7] | -1.3 | 49.3 -> 20.9 (n=56) |
| w10T | TVD reduction % | 40.3 | 32.7 | -7.7 [-14.4, -1.3] | 0.0 | 40.8 -> 30.5 (n=44) |

By size group (w10): LSR small 20.1 -> 10.3, medium 37.0 -> 8.0, large 25.9 -> 10.0; TVD small
29.5 -> 23.8, medium 45.7 -> 33.1, large 47.3 -> 37.0. Geometric-mean LSR ratio packed/baseline:
w10 1.186 -> 1.065, w10T 1.323 -> 1.148. w10's LSR gain drops by more than 5 points on 27
circuits and rises by more than 5 on 4; TVD drops by more than 5 on 16 and rises on 8. Without
the Bernstein-Vazirani family: LSR 16.6 -> 8.2, TVD 34.7 -> 31.3. The bv circuits carry the
largest single changes (LSR gain +113..+194% -> +3..+92%, TVD 70-78% -> 3-20%): once their
Hadamards are gone the baseline is already accurate (bv_n140 baseline TVD 0.213 -> 0.050).

TVD is withheld for 12 circuits: the five Ising circuits Fig 6 withholds (baseline TVD > 0.95),
and seven whose *optimized proxy* has a different output: for dnn_n33/n51, qugan_n39/n71 and
swap_test_n25/n41/n83 the original proxy's output is deterministic (noiseless TVD floor 0.00)
while the optimized proxy's is broad (floor 0.68-1.00 at 4,000 shots), because block
resynthesis scatters rotations that the proxy rule turns into S gates elsewhere. TVD there
measures the sampling floor, so these pairs are excluded by the same differential rule as
`miscompiled` (floor > 3x and > +0.10); their LSR is kept.

### Who does the work

Per-circuit mean % against the ORIGINAL input's baseline (fair depth; LSR as improvement, TVD
as reduction on the 59 resolved circuits; medians in brackets where they tell a different
story):

| planner | metric | optimizer alone | AQORN alone | optimizer + AQORN |
|---|---|---|---|---|
| w10 | 1q | 15.2 | 24.2 | 31.0 |
| w10 | 2q | 14.0 | 15.5 | 22.8 |
| w10 | depth | 12.5 | 15.3 | 27.0 |
| w10 | LSR | 47.2 (median 14.5) | 25.9 (21.8) | 59.6 (43.3) |
| w10 | TVD | 21.9 (10.0) | 38.2 (38.2) | 48.2 (45.4) |
| w10T | 2q | 14.0 | 19.3 | 25.7 |
| w10T | depth | 12.5 | 16.4 | 28.1 |
| w10T | LSR | 47.2 | 41.3 | 72.6 |
| w10T | TVD | 21.9 | 40.3 | 49.4 |
| v4 | 2q | 14.0 | 16.8 | 23.7 |
| v4 | depth | 12.5 | 15.5 | 26.1 |

Corpus totals (w10, fair): baseline physical 2q gates 70,699 -> 58,215 from the optimizer alone;
packed 60,611 (original) and 53,423 (optimized).

### Sensitivity: plain O3 (what a skeptic would run first)

Same Stages A and B with the optimizer restricted to `--routes o3` (O3 to a fixed point, best of
3 calls per pass; static only, no simulation): 49 circuits change (all equivalent), the 2q count
of 47 of 69 is untouched, and the logical reduction is 1q 9.4%, 2q 8.9%, all gates 10.1%, depth
8.4% (per-circuit means). It removes only about half of the Bernstein-Vazirani Hadamards
(bv_n30..n140 1q -37 to -48%; bv_n14/n19 unchanged), which the tableau route removes almost
entirely. AQORN on these inputs, fair depth:

| planner | metric | original | O3 input | delta [95% CI] |
|---|---|---|---|---|
| w10 | 1q | 24.2 | 21.9 | -2.3 [-4.5, -0.4] |
| w10 | 2q | 15.5 | 12.8 | -2.8 [-4.3, -1.4] |
| w10 | depth | 15.3 | 17.7 | +2.4 [-0.0, +4.7] |
| w10T | 2q | 19.3 | 16.4 | -2.8 [-4.4, -1.3] |
| w10T | depth | 16.4 | 18.9 | +2.5 [+0.3, +4.7] |
| v4 | 2q | 16.8 | 13.5 | -3.3 [-4.9, -1.9] |
| v4 | depth | 15.5 | 16.6 | +1.1 [-1.1, +3.4] |

Against the original baseline (w10): O3 alone 8.3% 2q / 7.0% depth, AQORN alone 15.5 / 15.3,
both 20.2 / 25.3. So against the optimizer a skeptic is most likely to try, AQORN keeps 83% of
its 2q gain and all of its depth gain; the larger losses above need the Clifford and block
routes. (Outputs: `results/logical_opt_o3only/`, figure by-product in its `fig/`.)

## Loader check (basis_trotter_n4)

Loading by name (`load_fig6_circuits(names=[...])` with <= 4 names per group, i.e.
`build_circuit`) gives a 92-gate basis_trotter_n4 where Fig 6's full-group load gives 1,888.
This is **not** the 1 s SIGALRM transpile timeout: every one of the 71 Fig 6 inputs is
gate-for-gate identical to a no-timeout level-1 transpile of the file it came from. The cause
is that `benchmarks/qasmbench/small/basis_trotter_n4/` holds two different programs under one
README name, `basis_test_n4.qasm` and `basis_trotter_n4.qasm` (unitary overlap 0.23, so not two
versions of one circuit). The full-group loader's dict keeps the last file, the real benchmark
(its 582 two-qubit gates match the README's dual-gate count); `_qasm_by_name` keeps the first,
the 92-gate test program. No other folder has two programs under one name. So Fig 6's
basis_trotter_n4 is the genuine QASMBench circuit, level-1 transpiled; its redundancy (2q -63%
here) is the circuit's own, not a loader artefact. Any experiment that loads it by name is
running a different program.

## Caveats

* **The optimizer is an upper bound on optimal, not optimal.** Qiskit only, no ZX-calculus,
  phase-polynomial or T-count passes; no measurement-aware simplification (diagonal gates before
  the final measurement). Example: generic 4-qubit unitary resynthesis (QSD) of basis_trotter_n4
  needs 95 CX against our 217 (original 582).
* **The optimizer does not know the code's cost model.** It counts a Hadamard as one gate, but a
  single-lane logical H on [[4,2,2]] costs 7 + 4 physical gates and an Rz 1 + 2 (`iceberg.costs`),
  so logical and physical savings differ; the 2q-first score also trades rotations and depth for
  2q gates on dnn/qugan (1q up to +28%) and knn (depth up to +11%).
* **LSR at Fig 6's alpha is exponentially sensitive on small circuits** (alpha 6-21 on 3-4-qubit
  circuits, see `aqorn-pipeline-traps`); the optimizer-alone LSR mean (+47%) is driven by a few
  small circuits (error_correctiond3_n5 +478%, dnn_n8 +327%, hhl_n7 +297%; median +14.5%). Not
  re-checked at a fixed alpha = 1.
* **TVD near the floor.** Where the optimizer makes the baseline very accurate, relative TVD
  changes are dominated by sampling (hhl_n7 optimized baseline 0.028, packed 0.048: -98 points;
  error_correctiond3_n5 0.016 vs 0.017).
* Single samples per arm, as in Fig 6; the intervals above are across circuits, not shots.
* Equivalence for the 27 wide non-Clifford circuits is a randomized miter (three states), not a
  proof; the other 29 changed circuits are checked exactly (unitary or tableau).
* O3 is nondeterministic, so rerunning Stage A gives slightly different 1q counts; Stages B and C
  read the cached QPY circuits.

## Suggested paper text

> We transpile QASMBench circuits at Qiskit's optimization level 1. To check that AQORN's gains
> are not the clean-up of redundant inputs, we also passed every benchmark through a strong
> logical optimizer (Qiskit O3 to a fixed point combined with Clifford-tableau, CNOT-block and
> commutation resynthesis; every output checked equivalent to its input). It leaves the
> two-qubit count of 41 of the 69 circuits with two-qubit gates unchanged and removes 11% of
> two-qubit gates on average, concentrated in a minority of circuits (13 lose 31-84%). On the
> optimized inputs AQORN's depth reduction is unchanged (15.3%), it still removes 9.6% of
> two-qubit gates (15.5% on the original inputs; 12.8% after plain Qiskit O3) and reduces TVD by
> 29.7% (38.2%); its LSR gain falls from 25.9% to 9.7%. Most of the difference is the
> Bernstein-Vazirani family, whose Hadamard layers the
> optimizer removes, the same redundancy AQORN's Hadamard commutation exploits. The two
> compose: optimizer plus AQORN reduces two-qubit gates by 22.8% and TVD by 48.2% against the
> original baseline, against 14.0% and 21.9% for the optimizer alone.

## Files and re-run

Code (committed): `experiments/logical_opt/optimizer.py`, `experiments/logical_opt/logical_opt.py`.

Data (gitignored), `results/logical_opt/`: `stage_a.csv`, `stage_b.csv`, `stage_c.csv` (each
with a `.json` sidecar), `circuits/*.qpy` (the 56 optimized circuits),
`logical_opt_20261007_024913.csv` + `.summary.txt` (the final per-circuit table and all tables
above), `run.log`. `logical_opt.csv` / `logical_opt.summary.txt` there come from the main run's
own last step, which still had the earlier TVD mask (without the proxy-floor guard) and figure
layout in memory: superseded.

Figures, `AQORN Revisions/results/figures/`: **`logical_preoptimization_w10_v2.pdf/.svg`** (main,
w10 in panel b) and **`logical_preoptimization_w10T_v2.pdf/.svg`** (w10T); full width, 7.14 in.
Superseded by-products of this session, safe to delete: `logical_opt.pdf/.svg` (six-circuit smoke
test), `logical_opt_w10_v4.pdf/.svg` (old layout), `logical_preoptimization_w10.pdf/.svg` and
`_w10T.pdf/.svg` (first layout, a label overlapping the diagonal).

Figure caption draft: *(a) Logical gate and depth reduction from a strong logical optimizer
(Qiskit O3 fixed point plus Clifford, CNOT-block and commutation resynthesis) on each Fig 6
circuit, sorted by two-qubit reduction. (b) AQORN's reduction (improvement for LSR) against the
same input's baseline, original input (x) vs optimized input (y); dashed: no change. Gate counts
and depth with every stage layer-sorted; LSR and TVD single samples at each circuit's Fig 6
noise; TVD on the 59 circuits both inputs resolve.*

Re-run, from the worktree root:

    PYTHONPATH=. ../../../venv/bin/python -m experiments.logical_opt.logical_opt --jobs 3 --output-dir results/logical_opt_rerun
    # summary + figure only, from the cached table (seconds):
    PYTHONPATH=. ../../../venv/bin/python -m experiments.logical_opt.logical_opt --from-csv results/logical_opt/logical_opt_20261007_024913.csv [--plot-variant w10T]
    # recombine the cached stage CSVs (no computation):
    PYTHONPATH=. ../../../venv/bin/python -m experiments.logical_opt.logical_opt --stages
    # the plain-O3 sensitivity run (static stages only):
    PYTHONPATH=. ../../../venv/bin/python -m experiments.logical_opt.logical_opt --stages A B --routes o3 --no-noiseless-check --output-dir results/logical_opt_o3only --figure-dir results/logical_opt_o3only/fig --figure-stem o3only_logical

Wall time with 3 workers on a loaded machine: Stage A 0.5 min, Stage B 19 min, Stage C 62 min.
