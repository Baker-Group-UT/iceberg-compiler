# AQORN on a concatenated code: [[4,2,2]] inside [[4,2,2]]

v2 revision, overnight 2026-10-07. Branch `v2-concatenation` (from `e5d5741`).
Code: `experiments/concatenation/`. Tests: `tests/test_concatenation.py` (7 pass).

Question: the main evaluation uses [[4,2,2]] with syndrome measurement only at circuit end.
Does AQORN carry over, quantitatively, to correcting or concatenated codes?

**Bottom line.**
- **AQORN's gains transfer to the concatenated code, and grow there.** Against the concatenated
  baseline (naive at both levels), AQORN at both levels cuts physical two-qubit gates 38% and
  depth 47%, keeps 28% more shots and has 60% less silent error per kept
  shot -- better LSR on every circuit, better silent error on all but one (26 circuits, at a
  noise level where the baseline keeps half its shots).
- **Most of it comes from the inner level, and from AQORN's passes there.** The outer code's
  encoding and syndrome extraction repeat across outer patches, so at the inner level AQORN's
  merging finds whole-patch and transversal macros everywhere (alone: -30% two-qubit gates, -40% depth).
  The spread-restricted placement search adds nothing measurable beyond the passes.
- **Concatenation pays in shots and buys accuracy**: at equal physical noise the concatenated
  circuits keep fewer shots (3x the qubits, ~7x the gates) but carry 4-5x (baseline) to 13-15x
  (AQORN at both levels) less silent error per kept shot than one level (per-circuit medians,
  alpha 0.03-0.1). **But it stays first order**: silent error grows linearly with the noise in
  every arm (log-log slopes 0.9-1.1), because neither level's gadgets are
  fault-tolerant -- the code has distance 4, the compiled circuit has circuit distance 1 (exact
  fault analysis, section 2).
- **Spread matters**: the index-order layout has nearly the same gate counts as spread but 3.7x the
  silent error of spread-with-AQORN, 37% more than even the concatenated baseline.
- **Correction does not come for free**: treating inner flags as erasures keeps 84% of shots
  instead of 64% but multiplies silent error 90x (TVD 0.037 vs 0.002). With detection only at
  the end, a flagged inner patch is often not where the error is: one fault in a gadget between
  two inner patches flags one and leaves an undetected error in the other, inside the same outer
  patch, so the outer code "corrects" the wrong qubit. Correction would need syndrome extraction
  during the circuit, which the paper's scheme does not have.

## 1. The construction

**Code.** Two levels of the paper's [[4,2,2]] iceberg code. An outer patch holds two program
qubits in 4 data qubits plus the 2 syndrome ancillas the paper's end-of-circuit check uses; each
of those 6 *outer qubits* is itself a logical qubit of an inner [[4,2,2]] patch (4 data + 2
ancillas, physical). Per pair of program qubits: 6 outer qubits in 3 inner patches, 18 physical
qubits; 9 physical qubits per program qubit.

**Spread placement (your note about index order).** For the concatenated distance, no inner patch
may hold two qubits of one outer patch. The undetected weight-2 inner error Xbar_1 Xbar_2 = X_1 X_2
acts on both lanes of one inner patch. If both lanes belong to outer patch A it is a weight-2 error
on A -- an outer logical operator -- and is missed at both levels (distance 2). If the two lanes
belong to two different outer patches, each sees a weight-1 error and detects it. Two inner
patches therefore cannot carry one outer patch (every pair of outer qubits would share a patch:
[[8,2,2]]); the smallest spread unit is the one you chose: **four inner patches carrying two outer
patches, each outer patch taking one lane of every inner patch -- [[16,4,4]] on the data**, plus
two inner patches for the two outer patches' ancillas, laid out the same way.

![setup](../../../../../results/figures/concat_setup.pdf)
`results/figures/concat_setup.pdf`: (a) spread, (b) index order, (c) the compilation path.

![c1](../../../../../results/figures/concat_c1.pdf)
`results/figures/concat_c1.pdf`: what the inner level is handed -- C1 for the program H(0); CX(0,1)
on one outer patch (35 gates): GHZ encoding of the four outer data qubits, the H and CX gadgets,
and the end-of-circuit syndrome extraction into the two outer ancillas (a0: Z check, a1: X check).
Each of the six wires is one logical qubit of an inner [[4,2,2]] patch; the inner level compiles
this circuit exactly as the outer level compiled the program.

**Compilation: the paper's translator, run twice** (`construct.py`).
1. *Outer level*: the logical program is compiled exactly as in the paper (placement into outer
   patches, H commutation, merging, encoding, end-of-circuit syndrome extraction, measurement).
   The result C1 is an ordinary translated circuit on 6 qubits per outer patch.
2. *Flatten*: every measurement in C1 follows its qubit's last gate (checked), so C1 is a unitary
   followed by Z measurements of all its qubits; measurements are deferred to the inner level's
   final readout. C1's SWAPs (the whole-patch HH's anchor exchange, lane realignment) are atom
   exchanges at the outer level but SWAPs of two *logical* qubits one level up, implemented as a
   free relabel (the translator refuses cross-patch SWAPs, and spread puts the two qubits of an
   outer patch in different inner patches; every C1 SWAP stays inside one outer patch -- checked).
3. *Inner level*: the same translator compiles C1 as a program onto inner patches under a spread
   placement. Every program is layer-sorted first at both levels (`asap_order`, the fair
   baseline).

Inner placements: **stride** (outer qubit q -> inner patch q mod 3P, slot q div 3P; the simplest
spread rule, which for an even number of outer patches pairs the same role of two outer patches),
**index** (q -> inner patch q div 2; not spread; the cautionary arm) and **inner AQORN**
(`inner_plan.py`: AQORN's own weight-10 cost model, steepest descent over swaps that keep the
placement spread, AQORN's lane assignment, then AQORN's verify step -- translate and score each
candidate, starts included -- so it is never worse than stride on AQORN's score).

**Decoding** (`decode.py`), both from the same samples:
- *post-select*: keep a shot only if every inner check (both syndrome bits and the data parity
  of every inner patch) and every outer check (the same three, computed from the inner-decoded
  values of the outer qubits) passes;
- *erasure correction*: a failed inner patch does not discard the shot; its two outer qubits
  are marked erased. An outer patch with no erasure is post-selected on its checks; with only an
  ancilla erased, the remaining checks are used; with one erased data qubit, its value is
  rebuilt as the parity of the other three (a distance-2 code corrects one located erasure) and
  that patch's checks are dropped; two erased data qubits discard the shot.

**Noise and measurement** (`concat_eval.py`). Fig 6's machine (ScaledNoiseModel, distance + idle
noise, transport between inner patches), converted to Stim once at alpha = 1 and rescaled exactly
(`tvd_explore/fastframes.py`'s rules, applied to the Stim text by `noise.py`: 235x faster, arguments
equal to 1e-10). LSR and silent error per kept shot from 200k Pauli-frame shots; TVD
against the exact ideal distribution from 100k real samples. Per circuit, alpha is calibrated so
the concatenated baseline keeps half its shots under post-selection; every arm is also run at
alpha = 0.003, 0.01, 0.03, 0.1. Circuits: the 26 Fig 6 circuits with at most 8 qubits (small
group), as Clifford proxies (what Fig 6 simulates); 36-72 physical qubits.

**Validation.**
- Noiseless, every arm on every circuit: all checks pass on every shot and the decoded output
  is the program's distribution to sampling accuracy (`noiseless_lsr`, `noiseless_tvd` columns).
- Unit tests: spread is what the layouts claim; one flipped data bit in any inner patch is
  discarded by post-selection and corrected exactly by the erasure decoder in the spread layout;
  in the index layout the same flip is uncorrectable where an inner patch holds two data qubits
  of one outer patch.

## 2. Which faults get through both levels (exact)

`distance.py`: deterministic 4-qubit test programs (random Clifford, then its inverse
re-synthesised so nothing cancels), every check a Stim DETECTOR, every output an observable. From
the detector error model, the total probability of the single fault mechanisms that flip an output
and fire no check -- the silent error post-selection keeps, to first order (alpha = 0.1, 8 programs,
mean):

| compilation | 2q gates | first-order silent mechanisms | their probability | circuit distance |
|---|---|---|---|---|
| 1 level | 99 | 9.4 | 3.1e-3 | 1 |
| 1 level + AQORN passes | 69 | 7.9 | 1.6e-3 | 1 |
| 2 levels, index order | 695 | 8.5 | 2.7e-3 | 1 |
| 2 levels, index order + passes | 471 | 7.9 | 1.2e-3 | 1 |
| 2 levels, spread | 722 | 5.8 | 5.0e-4 | 1 |
| **2 levels, spread + inner AQORN passes** | 471 | **1.5** | **9.0e-5** | 1 |

![faults](../../../../../results/figures/concat_faults_v2.pdf)
`results/figures/concat_faults_v2.pdf`

- **Index order buys almost nothing** over one level (2.7e-3 vs 3.1e-3), as the distance argument
  predicts; spread cuts first-order silent error 6x, and AQORN's passes at the inner level cut it
  another 5.6x (34x below one level).
- **The circuit distance stays 1 in every arm**, including spread. The surviving mechanisms are
  two-qubit gate faults inside one inner patch in gadgets that are not fault-tolerant (the
  GHZ-style encoder, in-patch gadgets): a single fault leaves an undetected inner logical error
  in the middle of an outer gadget, which the outer gadget then spreads within its outer patch. The
  code is distance 4; the compiled circuit is not, because neither level's gadgets are fault-
  tolerant (the paper's are not). Inner merging removes most of them because the whole-patch and
  transversal macros it fires leak no silent error at first order.

## 3. Results: 26 circuits, eleven compilations

At each circuit's calibrated noise (median alpha 0.30; the concatenated baseline keeps 50% of its
shots), post-selection decoding. Change against the concatenated baseline in %, mean (median) over
the 26 circuits, + = better; LSR, silent error per kept shot and TVD are means:

| compilation | phys. qubits | 2q gates | depth | LSR | LSR change | silent/kept | silent change | TVD |
|---|---|---|---|---|---|---|---|---|
| 1 level, baseline | 15 | +85.3 (+85.4) | +75.2 (+75.2) | 0.914 | +82.8 (+82.5) | 6.97e-03 | -592.4 (-424.2) | 0.0064 |
| 1 level, AQORN w10 | 15 | +87.9 (+87.5) | +79.4 (+78.1) | 0.922 | +84.4 (+83.9) | 3.38e-03 | -282.6 (-129.0) | 0.0037 |
| 1 level, AQORN w10T | 15 | +88.8 (+88.3) | +79.9 (+79.6) | 0.927 | +85.4 (+85.0) | 3.20e-03 | -296.1 (-91.5) | 0.0042 |
| 2 levels, baseline / baseline | 46 | +0.0 (+0.0) | +0.0 (+0.0) | 0.500 | +0.0 (+0.0) | 1.52e-03 | +0.0 (+0.0) | 0.0032 |
| outer AQORN / inner baseline | 46 | +14.6 (+12.4) | +12.6 (+1.5) | 0.555 | +11.1 (+8.2) | 6.85e-04 | +49.0 (+49.9) | 0.0028 |
| outer baseline / inner passes | 46 | +29.6 (+28.0) | +40.4 (+37.9) | 0.597 | +19.5 (+18.2) | 6.02e-04 | +54.8 (+51.3) | 0.0023 |
| outer baseline / inner AQORN | 46 | +30.1 (+27.6) | +41.7 (+38.3) | 0.599 | +19.9 (+17.2) | 5.91e-04 | +53.4 (+51.4) | 0.0023 |
| outer AQORN / inner passes | 46 | +38.5 (+37.6) | +46.5 (+47.1) | 0.640 | +28.1 (+26.1) | 5.44e-04 | +60.8 (+60.9) | 0.0023 |
| **outer AQORN / inner AQORN** | 46 | +38.2 (+38.0) | +46.5 (+47.4) | 0.639 | +27.8 (+27.0) | 5.64e-04 | +59.6 (+58.5) | 0.0022 |
| outer w10T / inner AQORN | 46 | +43.4 (+41.1) | +48.7 (+48.3) | 0.660 | +32.0 (+28.9) | 5.61e-04 | +58.6 (+59.3) | 0.0022 |
| outer AQORN / inner index (not spread) | 46 | +36.2 (+33.5) | +42.2 (+42.0) | 0.644 | +28.9 (+26.5) | 2.09e-03 | -50.2 (-55.3) | 0.0030 |

![results](../../../../../results/figures/concat_results_v3.pdf)
`results/figures/concat_results_v3.pdf`: (a) gates and depth, (b) LSR and silent error at the
calibrated noise, (c) silent error against the noise scale, (d) the two decoders.

Reading it:
- **AQORN at both levels: LSR better on 26/26 circuits, silent error lower on 25/26.** Outer AQORN
  alone (the paper's compiler, unchanged) already helps (LSR +11%, silent -49%; 22/26 and 24/26).
- **The inner level is where most of the gain is**, because the outer encoding and syndrome
  extraction are the same gate pattern on every outer patch: with the spread layout those
  identical gates sit on the two lanes of one inner patch, and AQORN's merging fuses them into
  whole-patch HH and transversal CXT/CZT macros (1q -44%, 2q -30%, depth -40% from the passes
  alone). Merging's macros also leak no silent error at first order, which is why silent error
  falls more than gate count.
- **The spread-restricted placement search adds nothing measurable beyond the passes** (outer
  baseline: inner passes LSR +19.5% vs inner AQORN +19.9%; with outer AQORN, +28.1% vs +27.8%).
  It moved the placement on 15/26 circuits, never by much: stride already puts the same role of
  two outer patches on one inner patch, the arrangement merging wants.
- **w10T at the outer level is the best arm on gates and LSR** (2q -43%, LSR +32%), same silent
  error as w10.
- **Index order** has gate counts close to spread (the passes do the same work) but 3.7x the
  silent error of spread with AQORN and 37% more than the concatenated baseline: the distance
  argument, measured.

**Against one level.** At the same physical noise one level keeps far more shots (LSR 0.91 vs 0.50
at the calibrated noise), since the concatenated circuits have 3x the qubits and ~7x the gates; what
concatenation buys is accuracy. Silent error per kept shot, per-circuit median ratio one-level
baseline / arm: one-level AQORN 2.0-2.1x lower, concatenated baseline 4.0-4.8x, **concatenated
AQORN 13-15x** (alpha 0.03-0.1; IQR 7.5-28). Pooled over circuits (dominated by the largest ones),
silent error per kept shot by fixed noise scale:

| compilation | alpha=0.003 | alpha=0.01 | alpha=0.03 | alpha=0.1 | log-log slope |
|---|---|---|---|---|---|
| 1 level, baseline | 3.23e-04 | 1.03e-03 | 2.89e-03 | 7.81e-03 | 0.91 |
| 1 level, AQORN w10 | 1.33e-04 | 4.50e-04 | 1.30e-03 | 3.82e-03 | 0.96 |
| 1 level, AQORN w10T | 1.36e-04 | 4.25e-04 | 1.28e-03 | 3.72e-03 | 0.95 |
| 2 levels, baseline / baseline | 3.13e-05 | 1.00e-04 | 2.79e-04 | 6.48e-04 | 0.87 |
| outer AQORN / inner baseline | 6.27e-06 | 3.22e-05 | 1.09e-04 | 3.08e-04 | 1.11 |
| outer baseline / inner passes | 9.40e-06 | 3.16e-05 | 1.01e-04 | 2.51e-04 | 0.95 |
| outer baseline / inner AQORN | 8.03e-06 | 3.28e-05 | 9.90e-05 | 2.47e-04 | 0.98 |
| outer AQORN / inner passes | 6.06e-06 | 2.28e-05 | 7.48e-05 | 2.40e-04 | 1.05 |
| **outer AQORN / inner AQORN** | 6.45e-06 | 2.35e-05 | 8.12e-05 | 2.43e-04 | 1.04 |
| outer w10T / inner AQORN | 5.08e-06 | 2.04e-05 | 8.07e-05 | 2.29e-04 | 1.10 |
| outer AQORN / inner index (not spread) | 3.12e-05 | 1.12e-04 | 3.05e-04 | 9.14e-04 | 0.96 |

Every slope is ~1: silent error is first order in the physical error rate at both levels -- the
fault analysis of section 2 explains why. Concatenation lowers the coefficient, not the order.

**The two decoders** (calibrated noise, mean over circuits):

| compilation | LSR post-select | LSR erasure | silent post-select | silent erasure | TVD post-select | TVD erasure |
|---|---|---|---|---|---|---|
| 2 levels, baseline / baseline | 0.500 | 0.792 | 1.52e-03 | 1.16e-01 | 0.0032 | 0.0838 |
| **outer AQORN / inner AQORN** | 0.639 | 0.841 | 5.64e-04 | 5.09e-02 | 0.0022 | 0.0373 |
| outer AQORN / inner index (not spread) | 0.644 | 0.669 | 2.09e-03 | 3.65e-03 | 0.0030 | 0.0039 |

Erasure correction recovers 20-30 points of LSR but multiplies silent error 75-90x, so TVD gets
17-26x worse. On the index layout it changes almost nothing: there a flagged inner patch holds two
qubits of one outer patch, which the outer code cannot correct, so the shot is discarded anyway.
Why erasure does badly on the spread layout: detection runs once at the end, so a flagged inner
patch says *that* something went wrong there, not *what else* it touched. One fault in a gadget
acting on two inner patches -- every outer two-qubit gate inside an outer patch is such a gadget,
by spread -- flags one patch and leaves an undetected logical error on the other, in the same outer
patch: the outer code then completes the erased qubit from a corrupted partner. A correcting
version would need syndrome extraction during the circuit (to locate faults in time), which is
outside the paper's scheme. Reported as measured, not tuned.

**Caveats.**
- Clifford proxies of the 26 Fig 6 circuits with <= 8 qubits (36-72 physical qubits); TVD is near
  its sampling floor (100k samples, small outputs), so silent error per kept shot (200k frames)
  is the sharper accuracy metric; the two agree in direction.
- Noise is Fig 6's machine at a per-circuit calibrated scale (median alpha 0.30, i.e. p2 about
  1e-3), plus fixed scales 0.003-0.1. Transport is modelled between inner patches exactly as for
  one level; nothing models the outer patches as separate hardware.
- The outer plans are Fig 6's weight-10 / w10T plans (made for one level); a planner aware of the
  inner level could do better. Inner AQORN is AQORN's own machinery restricted to spread layouts,
  not a new algorithm.
- Gadgets are the paper's non-fault-tolerant ones at both levels; `fault_tolerant=True` (flagged
  gadgets) was not tried.

## Files

- Code: `experiments/concatenation/{construct,decode,inner_plan,concat_eval,distance}.py`; figures
  `experiments/concatenation/{concat_setup,concat_results,concat_faults}/`.
- Data (not committed): `results/concatenation/concat_eval.csv`, `concat_distance.csv` (worktree).
- Figures: `results/figures/concat_setup.pdf`, `concat_faults_v2.pdf`, `concat_results.pdf` (project).

Re-run:

    PYTHONPATH=. ../../../venv/bin/python -m experiments.concatenation.concat_eval --jobs 3
    PYTHONPATH=. ../../../venv/bin/python -m experiments.concatenation.distance --seeds 8
    PYTHONPATH=. ../../../venv/bin/python -m experiments.concatenation.concat_results.concat_results --from-csv results/concatenation/concat_eval.csv
    PYTHONPATH=. ../../../venv/bin/python -m experiments.concatenation.concat_faults.concat_faults --from-csv results/concatenation/concat_distance.csv
    PYTHONPATH=. ../../../venv/bin/python -m experiments.concatenation.concat_setup.concat_setup
