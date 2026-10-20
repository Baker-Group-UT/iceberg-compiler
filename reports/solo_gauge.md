# Solo patches decoded as [[4,1,2]]

Branch `v2-solo-gauge` (from e5d5741), 2026-10-07. Question: is there a [[4,1,2]] mode in
which a patch holding one program qubit ("solo patch") has its other logical qubit reduced to
a gauge qubit, with a higher effective code distance? And: compare "every patch decoded as
[[4,2,2]]" with "[[4,1,2]] on solo patches, [[4,2,2]] elsewhere".

## Answer

* **No such mode existed.** A solo patch's spare lane was simply free: an H on a solo qubit is
  promoted to a whole-patch HH, which applies H to the spare too
  (`translator/handlers/hadamard.py`), and the decoder never reads the spare
  (`iceberg_postselect` + `reduce_distribution`). Errors confined to the spare were harmless
  but not detected. It now exists as post-processing (`experiments/solo_gauge/gauge.py`) plus
  two opt-in translator switches; no default changed.
* **The distance does not go up.** [[4,1,2]] is [[4,2,2]] with the spare logical gauge-*fixed*
  in |0̄⟩: its Z̄ = Z(z-anchor)·Z(spare lane) becomes a third stabilizer, read for free from the
  final Z-basis data readout. The code distance stays 2 (quantum Singleton bound
  n − k ≥ 2(d − 1) gives d ≤ 2 at n = 4, k = 1), and it corrects nothing. It *detects more*: of
  the 12 nontrivial logical error classes that touch the program lane, the 6 with an X̄ or Ȳ
  component on the spare now flip the check (e.g. X̄₁X̄₂ = X₁X₂, invisible to [[4,2,2]]).
* **The circuit-level (fault) distance does not go up either, for any gadget.** Exact
  single-fault enumeration: every gadget that leaks silent error at first order under
  [[4,2,2]] still leaks under [[4,1,2]] (S and RZ lose about half, the targeted H a quarter,
  SX/RX and the readout nothing); every other solo-patch gadget was already silent-free. A
  gadget with a two-qubit gate on the support of a weight-2 logical (S = exp(−iπ/4·Z_aZ_p))
  has a fault equal to that logical, which commutes with every check, gauge included.
* **What it buys at circuit level: a few percent of TVD on small circuits, for a similar
  share of LSR, and nothing on wide ones.** Fig 6 has exactly one solo patch per odd-width
  circuit (35 of 71, every arm), so the effect scales like 1/patches: TVD −3 to −7% (mean by
  arm, n ≤ 18, 54 of 56 per-circuit CIs below zero, none above) for LSR −3 to −5%; about
  −0.1% on n ≥ 19; Fig 6's corpus averages move by under one point. At alpha = 1 the gain and
  the cost both roughly halve. Forcing the check to exist (targeted solo H, or a fix-up H on
  the spare) costs more than it returns.

## Method

* **Decoder (post-processing, same samples).** `gauge.decode_samples` is `iceberg_postselect` +
  `reduce_distribution` vectorized, plus: drop a shot if a valid gauge check of a solo patch
  fires. Both rules decode the same noisy register distribution from `simulate_batch`, so every
  comparison is paired.
* **Validity criterion.** Let B be the translated circuit's body (after the encoder, measurements
  removed). A final Z̄ product P over spare lanes of solo patches is a check iff B†PB lies, up to
  sign, in ⟨S_X, S_Z of every patch; Z̄_j of every lane j that held no program qubit at the start;
  Z of every ancilla⟩ -- the stabilizer of "code space, spares in |0̄⟩, program lanes in *any*
  state". The sign is the expected bit. Conjugation is physical (Stim `PauliString.before`), then
  factored into logical coordinates (closed form, as in `tests/logical_action.py`), so routing
  SWAPs, HH frame flips and entangling macros are all accounted for; products of several spares
  are solved by GF(2) elimination (the user's rule: candidates are spare lanes of solo patches
  only). Requiring determinism for every program input is what keeps this from being
  post-selection on the known ideal output.
* **Gates, all passed.** (1) The [[4,2,2]] path reproduces the source CSVs on all 140 arms
  (Fig 6 v4, `ablation_silent_aware_w10.csv`, `ablation_variant_w10T.csv`; max |ΔLSR| 2.5e-14,
  |ΔTVD| 6e-16) and the library's own post-selection on the 108 arms with n ≤ 40. (2) Noiseless
  forward simulation, 2 000 shots per arm: no syndrome and no valid check ever fired (so the
  logical distribution with noise off is identical under both rules). (3) Random stabilizer
  states on the program lanes (6 inputs × 400 shots, 92 arms with n ≤ 30): no valid check fired.
  Tests in `tests/test_solo_gauge.py` pin the criterion on hand-built cases: clean spare (valid),
  odd HH (rejected), even HH (valid), spare as target of a transversal CNOT from a program lane
  (rejected although deterministic at the |0⟩ input -- the trap), spare as transversal control
  (valid), routing SWAP, |1̄⟩ sign, and a Bell-paired product of two spares (valid as a product,
  neither alone).
* **Arms.** Fig 6 baseline (tight trivial packing, both passes off, seed offset 3) and the packed
  arms of v4, w10 and w10T (plans and commutation from the source CSVs; w10T with
  `TRANSVERSAL_INTRA_TWO_QUBIT`), offset 6, at each circuit's Fig 6 alpha and shot count. Every
  odd-width circuit (35): the only ones with a solo patch, in every arm (no plan is sparse).
* **Metrics.** LSR; TVD against the sampled ideal (Fig 6 convention), quoted only for n ≤ 18
  (estimability limit); silent error per kept shot from Pauli frames (`gauge.frame_split`,
  100 k shots, program lanes only -- unlike `exp8_undetected/frames.py`, a flipped spare is not
  an output error), at every width. Uncertainty: paired bootstrap of ΔTVD (200 draws, noisy and
  ideal samples resampled); LSR change is an exact count of the shots only the gauge drops.

## Availability of the check

Of the 35 solo patches per arm, the spare survives translation as a valid check in

| arm | valid | lost (X frame) | lost (entangled) |
|---|---|---|---|
| baseline | 29 | 6 | 0 |
| v4 | 29 | 6 | 0 |
| w10 | 32 | 3 | 0 |
| w10T | 30 | 5 | 0 |

Every loss is the same mechanism: an odd number of H on the solo qubit, each promoted to HH,
leaves the spare in |+̄⟩ at readout. No transversal macro ever touches a solo patch's spare
(merging skips one-qubit patches), so "entangled" never occurs in this corpus. HH promotion
(cheap, silent-free) and gauge fixing pull against each other exactly here.

## Results: Fig 6 corpus, paired

Mean over circuits with a valid check (per-circuit 95% bootstrap CI of ΔTVD below zero /
above zero in brackets):

| arm | TVD, n ≤ 18 | LSR, n ≤ 18 | silent/kept, n ≤ 18 | LSR, n ≥ 19 | silent/kept, n ≥ 19 |
|---|---|---|---|---|---|
| baseline (13 + 16 circuits) | −2.9% (13/0) | −3.2% | −2.8% | −0.06% | −0.05% |
| v4 (13 + 16) | −5.6% (12/0) | −4.6% | −5.7% | −0.08% | −0.13% |
| w10 (16 + 16) | −6.6% (16/0) | −4.6% | −6.0% | −0.09% | −0.16% |
| w10T (14 + 16) | −5.2% (13/0) | −3.8% | −5.3% | −0.12% | −0.20% |

Largest single-circuit gains: TVD −13.7% (pea_n5, v4), −13.0% (hhl_n7, w10T), at LSR −6.6% and
−15.3%. About half of the shots the gauge check drops carried an output error (frame estimate,
0.47–0.58 of them per arm); the rest are harmless spare-only errors. TVD is compared at equal
raw shots, so the [[4,1,2]] arm pays a slightly higher sampling floor and still wins.

Fig 6's headline (caption convention: mean per-circuit change of AQORN against the baseline,
TVD over the 66 resolved circuits, LSR over 71) with every arm decoded the same way:

| | TVD reduction, [[4,2,2]] → [[4,1,2]] on solo | LSR gain, [[4,2,2]] → [[4,1,2]] on solo |
|---|---|---|
| v4 | 26.8% → 27.2% | +38.3% → +38.1% |
| w10 | 38.7% → 39.3% | +25.9% → +25.4% |
| w10T | 40.4% → 40.8% | +41.3% → +41.2% |

Baseline and AQORN gain about equally, so the comparison barely moves.

## Gadget level (exact single-fault enumeration, alpha = 1)

Probability per gadget execution / p2, silent = undetected error on a *program* lane (exp8's
table counted any lane, which is why its S/RZ/H numbers are higher):

| gadget (solo lane) | silent, [[4,2,2]] | silent, [[4,1,2]] | extra discards, no error | fault distance |
|---|---|---|---|---|
| S | 0.133 | 0.067 | 0.067 | 1 → 1 |
| RZ | 0.289 | 0.156 | 0.133 | 1 → 1 |
| SX | 0.133 | 0.133 | 0 | 1 → 1 |
| RX | 0.289 | 0.289 | 0 | 1 → 1 |
| H, targeted (program lane) | 0.556 | 0.422 | 0.133 | 1 → 1 |
| H, targeted on the spare (fix-up) | 0.533 | 0.333 | 0.222 | 1 → 1 |
| readout (S_Z, S_X extraction + data, 2 patches) | 0.400 | 0.400 | 0.163 | 1 → 1 |
| X, Z, HH, cross-patch CX/CZ/XCX (solo as control or target) | 0 | 0 | 0 | 2 → 2 |

The Z̄ gauge only sees faults with an X̄_s component, so it helps Z-type gadgets (whose weight-2
faults on (lane, z-anchor) include X_pX_za ≅ X̄_s and Y_pY_za ≅ X̄_sZ̄_p) and not X-type ones.
Side observation, not pursued: the final S_X extraction leaks first-order silent error (a Z
fault on the parity ancilla mid-extraction becomes a weight-2 X error on data that the Z
readout cannot see), 0.4 p2 per two patches; no decoding rule fixes it.

## Making the check available: two opt-in variants

Both are translator switches in `translator/explore_switches.py`, off by default (shipped
behaviour unchanged, tested), run on the 23 odd-width circuits with n ≤ 27:

* `SOLO_H_TARGETED` (the suggested arm): a solo H is the targeted single-lane gadget (7, 4)
  instead of HH, so the spare is never rotated.
* `SOLO_GAUGE_PARITY_FIX` (added because it is cheaper): keep the HH promotion, and only when a
  solo patch received an odd number of HH apply one targeted H to its *spare* lane before
  readout. One gadget per affected patch instead of one per H, and the gadget table shows it
  leaks less silent error onto the program lane than a targeted program H (0.33 vs 0.42 p2).

Both make the check valid on 23/23 solo patches. Neither pays. On the circuits they actually
change (relative to the shipped arm decoded as [[4,2,2]]; frames for silent/kept, TVD for
n ≤ 18 excluding the floor-level qaoa_n3):

| arm | circuits changed | 2q gates | LSR | silent/kept | TVD |
|---|---|---|---|---|---|
| baseline + fix-up | 6 | +4.6% | −10.5% | −1.2% | +0.3% (4) |
| baseline + targeted H | 13 | +7.0% | −12.5% | +1.0% | +0.9% (9) |
| w10T + fix-up | 5 | +4.8% | −9.3% | −3.1% | −2.2% (3) |
| w10T + targeted H | 5 | +4.8% | −7.4% | +3.6% | +0.4% (3) |

Decoding alone moves along LSR ≈ silent/kept (1 : 1); buying availability with a gadget costs
about 10% LSR for at most 3% silent/kept, because the extra gadget's own silent leakage and
detected faults outweigh the one check it enables (`solo_gauge_options.pdf`). Use the check
where the frame survives; do not add gates to create it.

## The paper's gauge sentence, tested: one qubit per patch

Fig 6's baseline translation of all 36 circuits with n ≤ 18, at the tight packing and at one
program qubit per patch (every patch solo, 254 spares, of which 183 valid checks and 71 lost
to odd HH counts), each decoded both ways; same Fig 6 alpha and shots; medians over circuits:

| | TVD | LSR | silent/kept |
|---|---|---|---|
| 1 qubit/patch vs tight, both [[4,2,2]] | −58% | −51% | −57% |
| 1 qubit/patch: [[4,1,2]] vs [[4,2,2]] (paired) | −17% | −5% | −16% |
| 1 qubit/patch + fix-up, [[4,1,2]], vs 1 qubit/patch [[4,2,2]] | −16% | −16% | n/a |

So "more patches improve error detection" holds strongly, but under the decoder the paper
uses it is *not* the gauge: the spares are never read. The −58% comes from every two-qubit
gate becoming a cross-patch gadget (silent-free at first order, against 0.6–0.83 p2 for
in-patch ones) and from errors on unused lanes being harmless. Reading the spares as
[[4,1,2]] checks adds a further −17% TVD for −5% LSR. Near-floor circuits (qft_n4, qrng_n4,
qaoa_n3, TVD ≈ 0.01) get *worse* TVD with sparse packing, because halving the kept shots
raises the sampling floor; qft_n18 is saturated (TVD 0.93).

## Caveats

* One solo patch per odd-width circuit is all Fig 6 offers, so the corpus-level effect is
  small by construction; the per-patch effect is what the small circuits show.
* Fig 6's alpha is extreme on small circuits (6–21 for n = 3–5): there most kept shots carry
  several faults, and pairs of single-qubit faults spanning both lanes are exactly what the
  gauge catches. At alpha = 1 the gain is smaller (below).
* Clifford proxies, as in Fig 6. Silent/kept is the frame estimate (flip of a program bit
  relative to the noiseless reference), not TVD.
* The gain is post-selection: TVD improves by discarding shots, about one point of TVD per
  point of LSR given up on the small circuits (0.9–1.5 by arm). For comparison, the
  silent-aware planner traded roughly two points of TVD per point of shots (Experiment8;
  different circuits and alpha, so only indicative).
* The comparison of variants with the shipped arm is unpaired (a different circuit gets its
  own sample), so their TVD differences carry sampling noise of the order of the TVD floor;
  silent/kept from frames is the cleaner measure there.

## At alpha = 1 (frames, odd-width circuits up to 41 qubits)

Same arms and translations, Fig 6's machine at alpha = 1 instead of the per-circuit
calibration (`alpha_one.py`, 400 k frame shots). Circuits with n ≤ 18 and a valid check:

| arm | silent/kept change, Fig 6 alpha → alpha = 1 | kept-fraction change, Fig 6 alpha → alpha = 1 |
|---|---|---|
| baseline (13) | −2.8% → −0.9% | −3.1% → −0.6% |
| v4 (13) | −5.7% → −2.6% | −4.7% → −2.3% |
| w10 (16) | −6.0% → −3.2% | −4.6% → −2.2% |
| w10T (14) | −5.3% → −2.2% | −3.9% → −2.2% |

At realistic noise both the gain and the cost roughly halve: Fig 6's small-circuit alphas
(6–21) inflate the gauge's benefit because they make pairs of single-qubit faults across the
two lanes common, and those are what the check catches. The variants get worse, not better,
at alpha = 1: on the circuits they change, silent/kept *rises* by 2.4% (baseline + fix-up),
4.1% (baseline + targeted H), 2.1% (w10T + fix-up) and 8.1% (w10T + targeted H), for
−1.1 to −2.6% kept shots; the added gadget leaks more than the check recovers.

## Suggested paper text (formalizing.tex l.46, the gauge sentence)

Current: "Conversely, more patches improve error detection as unused logical qubits act as
gauge degrees of freedom and more errors per program qubit can be detected, but increases
spatial and qubit resource overheads."

Suggested (accurate for the shipped compiler):

> Conversely, more patches improve error detection: each program qubit shares a patch's two
> checks with fewer others, two-qubit gates between patches use constructions in which every
> single fault is detected, and errors confined to an unused logical qubit leave the program
> untouched. An unused logical qubit can additionally be fixed in $|\bar{0}\rangle$ and read
> out as a third check, turning a patch that holds a single program qubit into the
> $[[4,1,2]]$ code; this detects more errors but leaves the code distance at two. The price
> is spatial and qubit overhead.

If the [[4,1,2]] decoding is adopted in the evaluation, one sentence for Section VI: "Decoding
patches that hold a single program qubit as $[[4,1,2]]$ lowers TVD by 3–7% on the circuits of
up to 18 qubits where the check survives compilation, for a similar loss in selection rate,
and moves the Fig. 6 averages by under one point." Do not claim a distance increase.

## Files

Code (committed): `experiments/solo_gauge/{gauge,circuit_level,gadgets_412,options,alpha_one}.py`,
`tests/test_solo_gauge.py`, opt-in switches `SOLO_H_TARGETED`, `SOLO_GAUGE_PARITY_FIX` in
`translator/explore_switches.py` (hooks in `translator/handlers/hadamard.py`,
`translator/core.py`; defaults reproduce the shipped translator, tested).

Data (gitignored): `results/solo_gauge/{circuit_level,circuit_level_density,circuit_level_variants,
alpha_one_a1,gadgets_412}.csv` with JSON sidecars; logs beside them.

Figures (PDF + SVG with real text, in `AQORN Revisions/results/figures/`):

* `solo_gauge_paired.pdf` (full width): the requested comparison. Per circuit, paired change
  from [[4,2,2]] everywhere to [[4,1,2]] on solo patches, four arms: (a) TVD (n ≤ 18, ±1 sd
  bootstrap), (b) silent error per kept shot (frames, every width), (c) LSR. A cross marks a
  circuit whose solo patch has no valid check (change exactly 0).
* `solo_gauge_gadgets.pdf` (column): silent error per p2 per gadget under both decodings, and
  the harmless extra discards; the silent-free gadgets listed underneath.
* `solo_gauge_availability_v2.pdf` (column): how often the check survives, per arm and
  variant. (`solo_gauge_availability.pdf` is a superseded, crowded draft from this run.)
* `solo_gauge_options.pdf` (column): LSR vs silent/kept change for decoding only, + spare
  fix-up, targeted solo H (baseline and w10T).
* `solo_gauge_density.pdf` (full width): tight vs one qubit per patch, both decodings, TVD and
  LSR per circuit (n ≤ 18).
* `solo_gauge_alpha.pdf` (column): silent/kept change at Fig 6's alpha and at alpha = 1.

Re-run (from `iceberg-compiler/`, Python `venv/bin/python`, `PYTHONPATH=.`; figures are saved
under the next free `_vN` name; drop `--from-csv` to recompute):

```
python -m experiments.solo_gauge.gadgets_412 --from-csv results/solo_gauge/gadgets_412.csv
python -m experiments.solo_gauge.circuit_level --from-csv results/solo_gauge/circuit_level.csv \
    --skip-availability
python -m experiments.solo_gauge.options --main-csv results/solo_gauge/circuit_level.csv \
    --variants-csv results/solo_gauge/circuit_level_variants.csv \
    --density-csv results/solo_gauge/circuit_level_density.csv
python -m experiments.solo_gauge.alpha_one --from-csv results/solo_gauge/alpha_one_a1.csv \
    --main-csv results/solo_gauge/circuit_level.csv
# recompute (3 workers): circuit_level --jobs 3 (55 min); density:
#   circuit_level --jobs 3 --max-qubits 18 --arms baseline sparse --variants sparse_fix \
#       --stem circuit_level_density
# variants: circuit_level --jobs 3 --arms --variants baseline_fix baseline_tH w10T_fix w10T_tH \
#       --stem circuit_level_variants --names <the 23 odd-width circuits with n <= 27>
# alpha = 1: alpha_one --jobs 3 --max-qubits 41 --variants baseline_fix baseline_tH w10T_fix w10T_tH
python -m pytest tests/test_solo_gauge.py
```
