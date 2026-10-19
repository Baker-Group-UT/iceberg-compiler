# The immutable-fraction axis, measured and retired

This figure originally swept locality × immutable fraction. The y axis was replaced by
`k` after measuring it. The sweep is kept at `results/data/improvement_vs_structure.csv`
(30 cells, 8 circuits each, 480 rows) because the numbers are quotable in prose and
because re-deriving them costs 45 minutes.

Recorded here rather than in the script docstring so the script describes what it does
now, not what it used to do.

## The axis works

The mechanism is real. Raising the immutable share blocks merging, and the naive
baseline's translated circuit inflates accordingly:

| immutable ratio | 1q | 2q | depth | realised fraction |
|---|---|---|---|---|
| 0.0 | 169.4 | 253.9 | 124.6 | 0.000 |
| 0.1 | 194.4 | 266.1 | 135.2 | 0.106 |
| 0.2 | 206.3 | 271.0 | 136.0 | 0.191 |
| 0.3 | 219.6 | 288.2 | 151.6 | 0.293 |
| 0.4 | 226.3 | 298.8 | 169.4 | 0.382 |

The realised fraction tracks the requested one to three decimals, so the knob does what
it claims.

## But it is second-order, and structurally so

| metric | locality marginal | immutable marginal | r(locality) | r(immutable) |
|---|---|---|---|---|
| selection rate | 9.5 → 46.0 | 17.2 → 27.5 | 0.91 | 0.85 |
| 2q gates | 2.7 → 7.4 | 3.7 → 5.0 | 0.93 | **−0.70** |
| depth | 6.2 → 19.5 | 7.4 → 12.3 | 0.86 | 0.69 |
| 1q gates | −6.5 → 0.6 | −4.4 → −1.2 | −0.96 | −0.83 |
| TVD | −10.4 → 1.2 | −5.4 → −0.8 | −0.89 | **−0.09** |

Locality carries roughly three times the leverage on every metric, and the immutable axis
carries none at all on TVD.

The reason is in the generator. `structured_program_circuit` emits exactly
`num_qubits * gates_per_qubit` gates with `two_qubit_ratio = 0.40` held fixed, so raising
`immutable_ratio` swaps single-qubit H/X/Z for S and **never changes the interaction
graph**. The packing problem is therefore identical along that axis: it modulates how much
the baseline bloats, not what the placer decides. For a figure about placement, that is
the wrong knob — which is why `k`, where patch capacity changes the problem itself,
replaced it.

Note the sign flip on 2q: more immutables means slightly *more* LSR gain but *less* 2q
gain. Packing removes a roughly fixed number of cross-patch CNOTs, so as the surrounding
circuit inflates that fixed saving is a smaller share of a bigger total, while the
absolute error reduction it buys still shows up in selection rate.

## Caveat on what "immutable" means here

`clifford_only=True` is forced by the Stim backend, which rejects Rx and Rz. So every
immutable gate in this sweep is an S, standing proxy for the rotations that make
benchmark circuits hard to merge. The axis is a Clifford stand-in for a non-Clifford
reality — fine for asking how merging's reach affects packing, not a substitute for
measuring rotation-bearing circuits directly.
