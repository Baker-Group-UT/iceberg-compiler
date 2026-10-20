# Why AQORN wins where it wins, 2026-10-08

Question: which program structures make AQORN's reductions in 1q / 2q gates, depth, LSR and TVD
large, and why?

**Method.** Every logical gate becomes one gadget whose cost and error are fixed by the code, so
an arm's totals are a sum over a *gadget census*. `experiments/insights/census.py` rebuilds the
census of each of Fig 6's four arms under the standard (baseline, + H commutation, + merging,
+ packing; 71 circuits, the plans and decisions of `ablation_standard`) with the translator's
own passes, prices it, and measures each arm's detected errors off the schedule and its silent
error per kept shot by Pauli frames at the circuit's Fig 6 alpha. `structure.py` attributes the
reductions and correlates them with features of the logical programs. The census reproduces the
recorded gate counts exactly for the median arm (within 2% for 77% / 82% of arms on 1q / 2q;
the rest involve translator-level fusions the census does not model -- Hadamards folded into
readout on BV, CX.RZ.CX fused into an RZZ gadget on Ising -- so attribution there is approximate).

## The prices that decide everything ([[4,2,2]], one gadget each)

| gadget | (1q, 2q) | silent error (first order) |
|---|---|---|
| H on one lane | **(7, 4)** | 0.82 p2 |
| HH, both lanes | (4, 0) | 0 |
| RZ / RX on one lane | (1, 2) | 0.42 p2 |
| S | (2, 1) | 0.20 p2 |
| CX in a patch / across | (0, 4) / (0, 4) | **0.80 p2** / 0 |
| CZ in a patch / across | (1, 3) / (0, 4) | 0.60 p2 / 0 |
| CZ in a patch, transversal (standard) | (6, 0) | 0 |
| CXT / CZT: two CXs or CZs between two patches | (0, 4) | 0 |
| encode + decode, per patch | (9, 11) | -- |

A single-lane Hadamard costs more two-qubit gates than a CX, and in-patch two-qubit gadgets and
single-lane gates leak silent errors while cross-patch gadgets and whole-patch macros do not.

Where the baseline (program-order packing, no passes) spends its gates, corpus total:
two-qubit gadgets 42% of 2q (31% across patches, 11% in patch), rotations 29%, single-lane H 11%,
encode/decode 17%; of 1q, single-lane H 39%, rotations 28%, encode/decode 28%.

## What each stage does (corpus total, % of the baseline; `insights_mechanisms_v3.pdf`)

| stage | 2q saved | of which | 1q saved (net) | how |
|---|---|---|---|---|
| H commutation | 6.5 | single-lane H removed 6.1 | 12.9 | removes 54% of the Hadamards; H.CX.H becomes a reversed CX or a CZ (CZ costs 1 more 1q) |
| merging | 7.5 | H fused into HH 4.8, two-qubit fused / transversal 2.7 | 6.7 | fuses parallel Hadamards on a patch into HH (909 -> 59 single-lane H), two parallel CXs / CZs between two patches into CXT / CZT, builds in-patch CZ transversally |
| packing | 3.9 | two-qubit gadgets 3.9 | -0.5 | co-locates CZ partners (transversal, no 2q) and aligns patch pairs so parallel CXs fuse (CXT 61 -> 602), moving the rest across patches (in-patch CX 1584 -> 152) |

Gadget counts, corpus: single-lane H 1987 -> 909 -> 59 -> 51; HH 0 -> 0 -> 425 -> 429;
CXT + CZT 0 -> 0 -> 165 -> 788; in-patch CX 1872 -> 1584 -> 1584 -> 152. Rotations (10,238) and
encode/decode are untouched at k = 2: they are the floor of every reduction -- 35% of the 2q gates
left after AQORN are rotation gadgets.

## Which programs gain (Spearman correlation of a program feature with the reduction)

| feature | 1q | 2q | depth | LSR | TVD |
|---|---|---|---|---|---|
| fraction of gates that are H | **0.73** | **0.79** | 0.38 | **0.64** | 0.49 |
| fraction of H in parallel layers (HH-fusable) | 0.73 | 0.62 | 0.03 | 0.50 | 0.37 |
| fraction of H commutation removes | 0.02 | 0.26 | 0.00 | 0.34 | 0.36 |
| max interaction degree (hub qubits) | 0.14 | 0.24 | -0.32 | 0.49 | 0.35 |
| fraction of gates that are rotations | -0.22 | -0.26 | **-0.67** | -0.16 | -0.42 |
| fraction of 2q gates on program-order neighbours | -0.23 | -0.44 | 0.10 | **-0.63** | -0.22 |
| fraction of 2q gates already in a naive patch | -0.23 | -0.38 | 0.06 | -0.58 | -0.25 |

So the drivers are: **Hadamard density** (the most expensive gadget, and the one every pass can
remove), **parallel structure** (layers of the same gate on many qubits, which fuse into
whole-patch and patch-to-patch macros), and **interactions that program order packs badly**
(hubs, long-range pairs: what packing can improve). Rotation-heavy programs gain least: a
rotation gadget has no fused or transversal form at k = 2.

## LSR and TVD follow the census (`insights_error_paths.pdf`)

- **LSR tracks detected errors.** Across all 213 arms, the change in ln LSR against the baseline
  follows -alpha x the change in detected errors read off the schedule with r = 0.95 (slope 0.67:
  the first-order budget overstates large changes). Fewer 2q gates and shorter schedules are
  where the shots come from.
- **TVD tracks silent errors.** The change in TVD follows the change in silent error per kept shot
  with r = 0.90 (198 resolved arms), and that in turn follows the change in the census's
  first-order silent budget with r = 0.93.
- **Where the silent errors were, and where AQORN puts them** (first-order, corpus, alpha 1):
  baseline: rotations 56%, single-lane H 21%, in-patch two-qubit gadgets 21%, phase gates 2%.
  AQORN removes 97% of the Hadamard part (4.90 -> 0.13) and 92% of the in-patch part
  (4.74 -> 0.37, almost all at packing: transversal CZ and cross-patch placement). What is left is
  93% rotations -- the accuracy floor at k = 2.

## By family (means over the family's circuits; reductions in %)

| family | circuits | H share | rotation share | 1q | 2q | depth | LSR | TVD | mechanism |
|---|---|---|---|---|---|---|---|---|---|
| bv | 5 | 0.73 | 0 | 62 | 48 | 22 | +148 | 76 | layers of H on every qubit fuse into HH (all of it at merging) |
| error_correctiond3, qec_en, qec9xz, lpn, hs4 | 5 | 0.4-0.7 | ~0 | 25-73 | 32-69 | 39-63 | +54..+200 | 45-83 | H around CX: commutation removes 80-100% of the H (hs4_n4: half; the rest fuse into HH) |
| adder, multiply, qram, swap_test, knn | 15 | ~0.1 | 0.4-0.5 | -5..19 | 15-24 | -5..34 | +48..+73 | 37-53 | every H commutable; packing fuses / co-locates the 2q gates (adder_n64 LSR +68 points at packing) |
| qft, qpe | 3 | 0.07-0.14 | 0.5 | -18..30 | 20-23 | 18-22 | +51..+52 | 18-40 | commutation turns CX into CZ (+1q), merging fuses CZ pairs into CZT |
| qugan, dnn | 5 | <=0.06 | 0.5-0.6 | -7..9 | 11-23 | 4-6 | +23..+104 | 41-44 | 2q structure only; packing makes the gain (qugan_n39 +106 LSR points) |
| ising | 6 | 0.14 | 0.56 | 31 | 20 | **-35** | +28 | (saturated) | H layers fuse into HH, aligned CX ladders into CXT; ising_n98's plan is 3.5x deeper |
| ghz, cat, wstate | 14 | <=0.07 | 0-0.5 | -3..5 | ~1 | 5-37 | **-18..-21** | 30-41 | no gates to save: the standard moves in-patch CXs (0.8 p2 silent) across patches, trading shots for accuracy |
| vqe, hhl, bell, qaoa, basis_change | 6 | ~0 | 0.55-0.84 | -52..+12 | 0-13 | 1-13 | -18..+29 | -14..+24 | rotation-dominated: nothing to fuse |

Top circuits: by 2q -- hs4_n4 69%, error_correctiond3_n5 52%, bv_n140 51%, bv_n70 51%, bv_n30 49%;
by LSR -- hs4_n4 +200%, bv_n140 +194%, bv_n70 +174%, bv_n30 +130%; by TVD -- error_correctiond3_n5
83%, hs4_n4 79%, bv_n140 78%, bv_n70 78%, bv_n30 77%.

## For the paper

- One sentence explains most of Fig 6: *a Hadamard on one lane of an iceberg patch costs more
  two-qubit gates than a CNOT and leaks silent errors, and AQORN removes 97% of them* -- by
  commuting them away (54%) or fusing parallel pairs into whole-patch HH (most of the rest).
- The second lever is placement: fused patch-to-patch CXT / CZT and, under the standard, in-patch
  CZs built transversally -- which is also what removes the in-patch silent errors.
- The limits are structural: rotations (no fused form at k = 2; the residual accuracy floor) and
  programs whose interactions program order already packs well.
- GHZ / cat / W-state circuits improve TVD by giving up shots, not by saving gates -- worth saying
  where Fig 6 shows their LSR bars below zero.

Re-run: `PYTHONPATH=. venv/bin/python -m experiments.insights.census --jobs 12` (~25 min), then
`... -m experiments.insights.structure` and `... -m experiments.insights.figures` (seconds).
Data: results/insights/census.csv, structure_attribution.csv, structure_outcomes.csv. Figures:
results/figures/insights_mechanisms_v3.pdf, insights_error_paths.pdf.
