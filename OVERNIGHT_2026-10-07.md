# Overnight 2026-10-07: optimality, scaling, concatenation, movement/idle, [[4,1,2]]

v2 revision. Each experiment ran on its own branch, all from `e5d5741` (v2-tvd-explore:
`asap_order` + the w10T switch, every switch off by default). Each branch has a worktree in
`iceberg-compiler/.claude/worktrees/<name>/`, with its report at `reports/<name>.md`.
Figures are in the project's `results/figures/`.
Your choices applied throughout: weight 10 + w10T (v4 as reference), the fair baseline (every
arm layer-sorted), [[16,4,4]] concatenation, both decoders, [[4,1,2]] only for solo patches.
Nothing pushed; no library default changed; code committed, data not.

| experiment | branch | report | figures |
|---|---|---|---|
| Optimality: exhaustive + planted | `v2-optimality` | `optimality/reports/optimality.md` | `optimality_gaps_v2.pdf` |
| Optimality: logical pre-optimization | `v2-logical-opt` | `logical-opt/reports/logical_opt.md` | `logical_preoptimization_w10_v2.pdf`, `..._w10T_v2.pdf` |
| Scaling and wall clock | `v2-scaling` | `scaling/reports/scaling.md` | `compile_scaling_compact_w10i_vs_w10_random_v2.pdf` (paper candidate, single column) + 6 more |
| Concatenation (extended) | `v2-concatenation` | `concatenation/reports/concatenation.md` | `concat_setup.pdf`, `concat_c1.pdf`, `concat_faults_v2.pdf`, `concat_results_v3.pdf` |
| Movement and idle | `v2-movement-idle` | `movement-idle/reports/movement_idle.md` | `movement_idle_{v4,w10,w10T}.pdf` |
| [[4,1,2]] solo patches | `v2-solo-gauge` | `solo-gauge/reports/solo_gauge.md` | `solo_gauge_paired.pdf`, `solo_gauge_gadgets.pdf`, `solo_gauge_availability_v2.pdf`, `solo_gauge_options.pdf`, `solo_gauge_density.pdf`, `solo_gauge_alpha.pdf` |

(Report paths are relative to `iceberg-compiler/.claude/worktrees/`.)

## Headlines

**Optimality (small programs, exhaustive).** On 4-6 qubit programs, every placement under both
H-commutation settings: the weight-10 planner is **0.6 / 2.0 / 2.2% above the exhaustive optimum
on its own objective** (w10T 1.0 / 1.2 / 1.8%); naive packing 17-23%, the interaction-count
partition 43-52%. Its silent-error model value is essentially the space's minimum. Depth is not
its objective: 1-15% depth regret. Measured by simulation (every placement by Pauli frames, finalists re-measured with fresh seeds),
its silent error per kept shot (the TVD stand-in) is 4-19% (median) above the best placement's,
against ~2x for v4 and naive; it gives up 3-8% of LSR for that (survival and accuracy have
different optima).

**Optimality (constructed circuits).** Hidden pairs whose HH / transversal-CXT structure makes one
packing provably optimal (it attains a gate-by-gate lower bound; certified on 42/42 decoy-free
circuits), labels scrambled. **AQORN (v4, w10, w10T) recovers 100% of the pairs and lands exactly
on the bound at every size from 6 to 32 qubits**; the interaction-count partition recovers none
(122-156% above the bound), naive 128-146%. With decoys AQORN still recovers 92-100%. One miss at
n = 4: a cost-model tie that verify never breaks (explained in the report).

**Optimality (logical pre-optimization).** Strongest Qiskit-only optimizer (O3 to a fixed point +
Clifford / CNOT-block / commutation resynthesis; every result equivalence-checked) on all 71 Fig 6
inputs. **The inputs are near-optimal in two-qubit structure for most circuits, not all**: 2q
unchanged on 41 of 69, but -31 to -84% on 13 (error_correctiond3, dnn_n8, basis_trotter, pea, hhl,
variational ...); per-circuit mean 2q -10.8% (median 0), all gates -16.3%. **AQORN's depth gain is
unchanged on optimized inputs (15.3% -> 15.3%); its gate and fidelity gains shrink**: 2q 15.5 ->
9.6%, TVD reduction 38.2 -> 29.7%, LSR +25.9 -> +9.7% -- mostly the Bernstein-Vazirani family,
whose Hadamard layers the optimizer deletes (the same redundancy AQORN's commutation removes).
The two are complementary: optimizer + AQORN = 2q -22.8%, depth -27.0%, TVD -48.2%. At Fig 6's
alpha the optimizer alone beats AQORN alone on mean LSR (+47% vs +26%).

**Scaling.** Compile time is the placement search, not translation (three exact translations per
plan at every n; the descent is 97-99% of planner time from 64 qubits on). **As implemented the
weight-10 planner grows as n^3.1-3.6** in CPU time (1 s at 32 qubits, 10-16 s at 64, 89-207 s at
128), while its work (candidate moves priced) grows as n^2.1-2.5. An **exact incremental evaluation**
of the two whole-partition cost terms (`experiments/scaling/incremental.py`, opt-in, identical
placements bit for bit on all 63 cells compared) brings that to **n^2.3 (k = 2) - n^2.9 (k = 10)**: 128
qubits in 18 s instead of 203 s at k = 2, 256 qubits in 2-10 min, 512 at k = 2 in 10 min. **Quality
holds with n**: 2q reduction 26-31% (k = 2) and 18-25% (k = 4-10) from 32 to 512 qubits, no trend.
Memory (unbounded cost memo) is the next limit: 12 GB at n = 256, k = 10.

**Concatenation ([[16,4,4]] = [[4,2,2]] in [[4,2,2]]).** The translator run twice; spread inner
placement; two-level decoding; 26 Fig 6 circuits (<= 8 qubits, 36-72 physical qubits). **AQORN at
both levels vs the concatenated baseline: 2q gates -38%, depth -47%, LSR +28% (26/26 circuits),
silent error -60% (25/26).** Most of it is AQORN's merging at the inner level (the outer encoding
and syndrome extraction become whole-patch/transversal macros there). Against one level at the
same noise: 13-15x less silent error per kept shot, at 3x the qubits and LSR 0.64 vs 0.91. But
silent error stays **first order** (slope 1 in the noise): the code is distance 4, the compiled
circuit is circuit-distance 1 because neither level's gadgets are fault-tolerant (exact fault
analysis; spread + inner AQORN cuts first-order silent probability 34x vs one level). Index order
(not spread) has 3.7x the silent error. **Erasure correction** (inner flags as erasures) keeps 84%
of shots but has 90x the silent error: end-of-circuit flags do not locate errors well enough.

**Movement and idle (Fig 6 layout, fair ordering).** **AQORN does not reduce movement, it
increases it**: atoms carried, per-circuit mean, +26% (v4), +85% (w10), +66% (w10T), more on
41 / 66 / 56 of 71 circuits -- the whole-patch HH is transversal H plus an anchor-atom exchange,
transversal macros carry whole patches, and w10 splits interacting pairs across patches. Qubit
idle time is mixed: corpus total -23 to -28% (packing shortens flights on the few wide circuits
that hold most idle time) but per-circuit mean +12.5% (v4) / +58% (w10). Makespan ~flat. Expected
detectable error still falls 11-14% because gates are 75-82% of the budget. Paper: do not claim
less movement; claim less transport-forced idling on wide circuits only.

**[[4,1,2]] on solo patches.** It did not exist (a solo patch's spare lane was free to disturb: a
solo H became a whole-patch HH). Now: a post-processing decoder (gauge-fixed spare, a third check
read from the final Z readout; a validity criterion that refuses checks depending on the program's
input) plus two opt-in switches. **Distance stays 2 and no gadget gains fault distance** (Singleton
bound; silent error per p2 falls for S 0.13 -> 0.07, RZ 0.29 -> 0.16, targeted H 0.56 -> 0.42; SX,
RX, readout unchanged). On the 35 odd-width Fig 6 circuits (one solo patch each; check valid on
29-32, lost when the solo qubit sees an odd number of HH), same samples decoded both ways: **TVD
-3 to -7% on circuits with n <= 18 (CI below zero on 54/56), for LSR -3 to -5%**; the Fig 6
headline moves by under a point. Forcing the check to exist does not pay. Packing every qubit
solo cuts TVD 58% (LSR -51%) already under [[4,2,2]] decoding -- from silent-free cross-patch
gadgets, not from a gauge.

## Things you should know before writing

- **The fair baseline is fair on depth, not on time.** Layer-sorting the baseline cuts its depth
  24% (total) but its compute time *grows* 11% in total (median circuit: no change), atoms carried
  +6%; expected error +0.7%. Depth does not track run time on this machine (transport-paced).
- **The movement claim in V-B needs rewording** (above), and so does any "fewer moves" phrasing.
- **Concatenation answers "transfer to concatenated codes" positively for detection**, and
  quantifies why correction does not come for free with end-of-circuit syndromes.
- **"Near-optimal inputs" needs qualifying**: say "two-qubit structure unchanged by a strong optimizer
  on 41 of 69 circuits", and report AQORN on optimized inputs (depth holds; gate/LSR gains shrink).
- **"Every benchmark compiles in seconds" (V-C) is false as written** (bv_n140 182 s; n^3+ growth);
  the scaling report has replacement text. The incremental evaluation is an engineering fix you
  could adopt (exact, opt-in today).
- **The paper's "unused logical qubits act as gauge degrees of freedom" (formalizing.tex l.46)** is not
  what the decoder did; the solo-gauge report has replacement text.

## Decisions for you

- Whether to report the planted-packing experiment in the paper (it is the cleanest optimality
  evidence and the clearest picture of why interaction-graph packers fail at merging).
- Whether to add a tie-break-by-translation to the planner (fixes the one planted miss; untested
  elsewhere).
- Which concatenation numbers to quote (post-selection at both levels is the honest headline).
- Whether to adopt [[4,1,2]] decoding of solo patches (small TVD gain, small LSR cost; free to apply
  as post-processing) and/or rewrite the gauge sentence.
