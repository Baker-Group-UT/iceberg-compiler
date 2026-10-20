# Review: neutral-atom noise model (`na-noise-model` branch)

Reviewed 2026-07-30. Scope: `simulator/transport.py`, `scheduler.py`, `grid_constructor.py`,
`stim_backend.py`, `noise_models.py`, `simulator.py`, `config.py`,
`experiments/shared/simulation.py`, `experiments/tests/test_model.py`,
`tests/test_transport_model.py`, plus the `iceberg/` emission code the model's assumptions
lean on. All 62 transport tests pass. Findings below were verified empirically where
possible (inspector runs at n=4 and n=12, targeted probes of the emitted Stim circuits);
each one says what the evidence was and which direction it biases results.

Not covered here: the three `TODO(na-noise-model)` markers in `packing/` — the
search-vs-simulator geometry disagreement is a known, documented gap. One addition to it
is recorded as finding 13.

---

## Status after the packing-costs round (`na-packing-costs` branch, same day)

The five tagged `TODO(na-noise-model)` sites — the cost models that still priced the
old machine — are **resolved**; no live-code markers remain. 569 tests pass, including
new pins for every change. What each site got:

1. **`packing/tables.mean_patch_hops`** prices real cells: most-square rectangle plus
   gutter strides, matching `simulator.grid_constructor`'s centres exactly (pinned by
   `test_mean_patch_hops_matches_the_simulators_centres`). Signature gained
   `qubits_per_patch`; all call sites wired.
2. **`MOVEMENT_SHARE_CEILING`** re-swept on the corrected pricing
   (`experiments/tests/movement_cap_sweep.py`, table in the docstring). At the
   machine's own settings the derived movement term never trips the clamp; the guard
   is needed only at the 32x-distance stress end, where uncapped movement still
   strands 9/12 circuits at naive. Value stays 0.25 — mid-band, a guard not a tune.
3. **`NoiseKnobs`** carries NA rates imported from `simulator.noise_models` (1 µs
   binding layer, twirl idle rate with T1), and `estimate()` prices movement per
   grouped trip: two carried atoms pay `move_error`, the array pays the flight's
   idling, and the beyond-blockade fraction of intra-patch gates pays a carry each —
   the k-honesty mechanism, priced where code sizes are ranked. Attribution follows
   `measured_noise_split` (movement = handling; forced idle is idle).
4. **`SWAP_WEIGHT` stays (0,0) as a *pulse count*** — correct, and right for program
   swaps, which are relabels. The price lives in a new carriage column:
   `CARRIAGES = {SWAP: 2, HH: 2}` with `carriage_weight()` = `move_error/p1` = 5
   units/atom, derived not tuned. `packing.cost` charges HH's anchor swap in the
   HH-vs-singles comparison, charges macros the *expected* routing swaps of a random
   alignment (2·(m−H_m), keeping the objective lane-order-invariant), and credits a
   fired macro (m−1) crossings of movement (one grouped trip instead of m).
   `CROSS_PATCH_COST` moved to `iceberg.costs` as the single source of truth.
5. **`transversal.py` has a real accept/reject gate** (`_macro_gain`): singles minus
   macro minus actual routing swaps, at the caller's code size. Two bugs fell out of
   the boolean it replaces: misaligned macros were taken however many swaps they
   cost, and an n-gate macro fired over two occupied lanes of a large patch where
   two targeted singles are cheaper.

Also done because site 2's re-measurement was meaningless without it:
`noise_profile_for` now **derives** `movement_cost` from the knobs (per-cell trip
idling on a k=2 reference array plus amortised handling; 1.45 units/crossing-cell,
≈ 0.97 two-qubit gates per typical crossing against the old hand-set ~4), and
`distance_scalar` enters once instead of quadratically (`distance_scale` is 1.0, true
cells). The archived-`AnalyticCost` agreement test pins the shared gate accounting
with carriage pricing neutralized; the carriage arithmetic has its own pins.

Not done (documented in place): `selection.py` still excludes transversal macros from
conflict *ranking* (they are gain-gated now but not compared against rival merges),
and `graph.py`'s seed heuristic still multiplies `movement_cost × distance_scale`
flat per crossing with `idle_cost` unused — both noted by the consumer-map review as
pre-existing seed-quality items, not correctness.

---

## Status after the fix round (same day)

Everything below except the FT items (finding 16, out of scope by decision) and the
packing-gap items (13, known TODO territory) has been **fixed on this branch**. The
full suite passes: 562 tests, including 8 new regressions pinning the fixes
(`tests/test_transport_model.py`, "noise-accounting regressions" section). Decisions
taken with the fixes, so the model has one story:

- **Bystanders dephase (1).** `ScheduleUnit` carries per-qubit `busy` seconds; a
  transport group's occupancy no longer exempts the held patches' stationary atoms from
  idle. The mover works flight + its gates; a gate partner works its gates and idles
  through the flight; everyone else in the patch idles the whole unit.
- **Every carriage is charged (2).** `ScheduleUnit.moved` unifies the three kinds of
  carriage — group control side, intra-patch beyond-blockade mover (first qubit, the
  control-side convention), swap exchange — and all of them pay `move_error` plus
  flight dephasing.
- **A swap is an atom exchange (3).** No pulse: duration is one one-way flight at the
  separation, both atoms carried, no DEPOLARIZE2. With no layout to price the exchange
  (distance noise off / non-encoded circuit) it falls back to gate pricing instead of
  being free. Aer charges swaps p2 (`twoq_ops` now includes `"swap"`), having no
  movement channel. The **remaining swap-free assumptions** live in the cost models and
  are documented in place rather than fixed (integer tables have no movement column):
  `iceberg.costs.SWAP_WEIGHT = (0, 0)` (also inside `HH`),
  `translator/merging/transversal.py`'s routing-swap realignment decision, and
  `analysis.gate_count_stats`'s #2q metric (excluding swaps is *correct* — not a gate —
  but it is not a complete error proxy). All three carry `TODO(na-noise-model)`-style
  notes now; they are the same class of gap as the packing TODOs.
- **`--superconducting` selects the superconducting preset (4).**
- **Everything error-based scales with alpha (5).** `ScaledNoiseModel` scales
  `move_error` (`BASE_MOVE_ERROR × alpha`, zero at alpha=0); `StochasticNoiseModel`
  carries the 10× stress value. Times and geometry stay fixed — machine shape, not
  error.
- **The split prices idle at the simulator's own rate (6).**
  `stim_backend.thermal_error_rate` = 1/(2·T2) + 1/(4·T1); used by
  `measured_noise_split` and the inspector's budget.
- **XCX is one motion (7).** The sandwich Hadamards are hoisted outside the contiguous
  four-CX run in `iceberg_set.XCX` (same unitary; pinned by the logical-action tests).
  CNOT/CZ were already contiguous; the transversal macros already had their H's outside.
- **Fenced batches cannot reorder (8).** `build_layers` keeps a fence floor; a later
  measurement batch can never land in an earlier slot than one already flushed.
- **Attribution convention (9, by decision):** `movement` = distance-based noise only
  (move handling + flight dephasing); *all* idling — including what transport forces on
  the array — rolls up under `idle`, split by cause (`idle_transport`, `idle_gates`,
  `idle_readout`). Docstrings in `simulation.py` updated; numbers quoted in old
  docstrings predate the convention and are labelled as such.
- Plus the robustness round (10-12, 14-18): `SimulatorConfig` fields annotated,
  `clone()` copies everything and keeps `None` seeds, layout built with the run's
  geometry, missing patch centre raises, gutter-vs-blockade warning in
  `Geometry.__post_init__`, idle-attribution residual goes to the actual runner-up
  cause, stale Table-2/`DefaultNoiseModel` docs corrected, split counts only
  noise-enabled records.
- **Physics notes added to `transport.py`'s "Deliberately absent" list** (per
  decision): fixed per-move overhead (the first-order one — comparable to the flights
  themselves), blockade exclusivity of concurrent Rydberg gates, syndrome-ancilla
  corner placement. Movement speed stays 1 m/s.

### Why the simulated benchmark figure's LSR improvements collapsed

Answered from the cached run (`merging_improvement_benchmark_stylized_simulated.csv`)
and reproduced: **the baselines were sitting far above the calibration target, not
below it.** Median baseline LSR across the 71 circuits was 0.449 against the 0.12
target; 26 of 71 sat above 0.5 (worst: dnn_n8 at 0.644). Relative LSR improvement
compresses mathematically near the ceiling — from LSR 0.58 the largest possible gain
is +72%, and a 30% cut in expected error buys exp(0.18)−1 ≈ +19% where the same cut
at the intended operating point buys ~+90%. That is the whole "LSR way worse" effect;
TVD compresses less (it is not ceiling-limited the same way), hence "a bit worse".

Root cause is finding 5: the calibration corrects alpha assuming selection rate is
exponential in alpha, and the constant (unscaled) `move_error` broke that on exactly
the large circuits — at their small initial alphas the movement term dominated the
pilot's error, the log-space correction stalled, and no achievable correction reached
the target (small circuits pushed alpha to ~10-17 where movement was negligible, and
calibrated fine — which is why the effect looked size-dependent). With `move_error`
now scaling, the same two-pilot calibration lands on target:

    circuit        cached pilot_final -> fixed pilot_final   (target 0.12)
    adder_n10           0.554         ->      0.169
    dnn_n8              0.626         ->      0.152
    bigadder_n18        0.578         ->      0.126

An end-to-end smoke of the figure's own worker on adder_n10 (reduced shots) shows the
improvements re-expanding as expected: H-commutation LSR gain +5.6% (cached, broken)
-> +24.3% (fixed). **No target adjustment is needed** — the 0.12 target is right and
now reachable; the figure needs regeneration, which is running.

**Regeneration result (same day).** All 71 baselines now land on target: baseline LSR
median 0.127, range 0.082-0.202, none above 0.3 (was median 0.449, 26 of 71 above 0.5).
Mean LSR improvement per component, broken -> fixed: H commutation 2.0% -> 9.1%,
merging 6.6% -> 13.0%, packing **-0.8% -> +3.9%** — packing was net *negative* under
the broken calibration and is positive under the fixed one, helped also by the model
now charging the cross-patch traffic packing removes (bystander idle, carriage on
intra-patch moves). Median stacked LSR improvement per circuit 3.5% -> 19.7%, max
101.6%. As the caveat above predicted, the absolute LSR gains sit below the
superconducting-era paper figure (which ran to +350% on a machine whose idle dominated)
— that is the honest NA answer, not a calibration artifact: with T2 = 1.5 s the budget
is gate-dominated, so gate-count wins buy less selection rate than they did against a
341 us T2. TVD reductions are comparable across the two runs (means 13.2 / 8.8 / 7.8). One honest caveat for
reading the regenerated figure: at a calibrated alpha the mechanism mix now equals the
machine's own (alpha scales everything uniformly), and under NA-with-decoupling
movement+idle are a few percent of the budget — so the packing arm, which buys
movement and idle, may legitimately show smaller LSR gains than it did under the
superconducting model where idle dominated. That would be the physics of the machine,
not a calibration artifact; the gate-count-driven arms (H commutation, merging) are
insensitive to it.

---

## High — affects simulated numbers in a k-dependent direction

### 1. Atoms inside transport-locked patches are exempt from idle dephasing

A transport unit occupies **every qubit of both patches** ([stim_backend.py:288-295](simulator/stim_backend.py:288))
so that the scheduler blocks the patches until the atoms return. But
`idle_times_for_layer` treats "in a unit" as "working" ([scheduler.py:165-169](simulator/scheduler.py:165)):
every qubit of the unit gets `duration − unit.duration = 0` idle. So during the longest
layers in the schedule, the stationary bystanders *inside* the two participating patches
are charged **zero** dephasing, while every atom outside the patches is charged the full
layer.

Verified on the k=2 inspector circuit: a 55 µs transport layer charges 55 µs to the 36
atoms outside patches C/E, and 0.0 to the 9 stationary atoms inside them (the moved atom
and the 3 gate partners are genuinely working; the other 9 are not).

- The occupancy rule is a *scheduling* constraint (addressability); it is not a reason
  those atoms stop decohering. The model's own rule — "every qubit not working dephases
  for the layer" — is violated for exactly the qubits nearest the movement.
- Direction of bias: **flatters transport-heavy schedules, and flatters large k more**
  (a k=10 patch shields 14 atoms per side per trip; a k=2 patch shields 6). This works
  directly against the model's stated goal of keeping large patches honest.
- Magnitude: per trip, ~(atoms per patch × 2 − movers − partners) × trip duration / T2.
  Small at T2 = 1.5 s, but it scales with exactly the knobs the figures sweep
  (`distance_scalar` slows trips; T2 sweeps shrink the denominator).

Suggested treatment: keep patch-level occupancy for scheduling, but compute idle for
transport units per-qubit — moved atoms are busy for the flight (they already get the
flight charge), gate partners for their gates, everyone else in the unit idles the full
duration.

### 2. Intra-patch beyond-blockade moves pay time but no error

`_duration` adds a round trip to an intra-patch two-qubit gate whose atoms sit beyond the
blockade radius ([stim_backend.py:237-246](simulator/stim_backend.py:237)) — but that is
*all* it adds. The moved atom pays no `move_error` and no flight dephasing; both charges
exist only on the cross-patch transport-group path ([stim_backend.py:374-387](simulator/stim_backend.py:374)).

Verified at n=12 (k=10): intra-patch units of 24.5 µs (t_2q + 24 µs round trip) carry
`transport=None`, hence no `DEPOLARIZE1(move_error)` and no in-flight `PAULI_CHANNEL_1`.

- Picking an atom up and putting it down costs heating/handoff regardless of whether the
  destination is inside or outside the patch — that is the whole argument for
  `move_error` being distance-independent. Charging it on 30 µm hops but not 12 µm hops
  is inconsistent.
- Direction of bias: **flatters large k**. The fraction of intra-patch pairs beyond
  blockade is the mechanism that "keeps large patches honest" (0.00 at k=2 → 0.27 at
  k=10), and every one of those gates is currently missing a 1e-3-class error that a
  cross-patch gate would pay. At k=10 the decode round alone does 2n = 24 serialized
  data–ancilla CXs per patch, many beyond blockade.
- Also worth noting: consecutive beyond-blockade CXs each pay their own round trip in
  time (no grouping on the intra-patch path), so time is if anything over-charged while
  error is under-charged.

### 3. Physical `swap` gates carry zero error in both backends

`_emit_record` skips depolarizing noise for `swap` ([stim_backend.py:325](simulator/stim_backend.py:325)),
and Aer's `ScaledNoiseModel` attaches two-qubit noise to `["cx", "cz"]` only
([noise_models.py:269](simulator/noise_models.py:269)). Physical swaps are not rare:
every patch-wide logical H (`HH`) emits one anchor swap
([iceberg_module.py:277](iceberg/iceberg_module.py:277)), and routing swaps are inserted
before transversal macros ([iceberg_module.py:339](iceberg/iceberg_module.py:339)).

The rationale in [physical.py:115-123](iceberg/physical.py:115) is "on hardware that has
one it is a relabeling". But the simulation does not implement a relabeling — the layout
maps *qubit index → cell* for the whole circuit, so a Stim SWAP with fixed cells models a
**physical exchange of two atoms**, which on this architecture is two transports
(2 × `move_error`, plus real time). A true relabel would be free but would have to move
the *labels'* cells for all subsequent geometry, which nothing does.

- Currently a swap costs t_2q of time (plus an intra-patch round trip if beyond
  blockade — time again with no error) and exactly zero error anywhere.
- Direction of bias: flatters H-heavy circuits and merged/transversal arms (the ones
  that buy their savings with routing swaps). The H-commutation pass gets its swaps
  free.
- Pick one story and charge it: relabel (0 gates, 0 time, but then it should not appear
  as a scheduled unit at all — and note the geometry caveat), 3-CX decomposition
  (3 × p2), or atom exchange (2 × move_error + transport time).

### 4. `test_model.py --superconducting` is a no-op

[test_model.py:561-565](experiments/tests/test_model.py:561) selects
`StimNoiseParameters()` for `--superconducting` — but the dataclass defaults *are*
`NEUTRAL_ATOM` now, so both branches produce identical parameters. Verified: the flag
prints "superconducting parameters" over T2 = 1.5 s, t_1q = 1 µs, movement enabled.
Should be `StimNoiseParameters.superconducting()`. Any comparison someone has run with
this flag was NA-vs-NA.

---

## Medium — accounting and sweep-design issues

### 5. `move_error` does not scale with `alpha`, and alpha=0 is not noiseless

`ScaledNoiseModel(alpha)` scales p1/p2/p_meas linearly and divides T1/T2 by alpha, but
the `StimNoiseParameters` it attaches never sets `move_error`
([noise_models.py:307-316](simulator/noise_models.py:307)), so it stays at the 1e-3
default at every alpha. Two consequences:

- **alpha=0 leaks noise.** The "explicitly noiseless" branch
  ([noise_models.py:250-264](simulator/noise_models.py:250)) zeroes everything except
  `move_error`. Verified: a distance-noise run under `ScaledNoiseModel(0)` emits
  `DEPOLARIZE1(0.001)` on every moved atom. Any noiseless-reference run that passes a
  layout is contaminated. (The pipeline's ideal reference runs with
  `include_noise=False`, so the main path dodges it — but the model object lies about
  what it is.)
- **Calibrated runs quietly change the mechanism mix.** `calibrate()` solves alpha per
  circuit (`experiments/shared/simulation.py:423`), typically alpha ≪ 1 for large
  circuits. At alpha = 0.1, gates/readout/idle are 10× weaker than the machine's, while
  move handling is unchanged — so the movement share of the budget at the *calibrated*
  point is inflated by ~1/alpha relative to the machine at alpha = 1. Every figure that
  compares mechanisms at a calibrated alpha is measuring a different balance than the
  hardware description implies. Decide explicitly whether `move_error` is part of "noise
  scale" (then scale it with alpha) or part of "geometry" (then document that alpha
  sweeps shift the balance — and zero it at alpha=0 either way).

### 6. The analytic split double-counts idle relative to the simulation (~2×)

`measured_noise_split` and `test_model.budget` price idle and flight as
`seconds / T2` ([simulation.py:226-231](experiments/shared/simulation.py:226),
[test_model.py:386-390](experiments/tests/test_model.py:386)), but the simulator's
channel for the same seconds is the twirl
([stim_backend.py:110-114](simulator/stim_backend.py:110)), whose total error
probability is ≈ t/(2·T2) + t/(4·T1) — about **half** of t/T2 at T2 ≪ T1. Gates,
measurement and move handling use exact channel probabilities. So every split, budget
table, and "movement share" annotation overstates the idle-family terms by ~2× relative
to the gate-family terms it is compared against. The relative ordering of idle sources is
unaffected; the idle-vs-gates and movement-vs-gates ratios are not. Either convert idle
seconds through the same twirl formula, or state the convention on the figure.

### 7. Transport groups split inside single logical operations

The docstring justifies the group-ending rule with "the gates within one logical
operation are emitted back to back with nothing in between"
([transport.py:55-58](simulator/transport.py:55)). Not true for X-type constructions:
cross-patch `XCX` interleaves Hadamards on the control patch between its CX pairs
([iceberg_set.py:340-348](iceberg/iceberg_set.py:340)), so one logical XCX is charged
**two** round trips where CNOT/CZ pay one. `HH`'s anchor swap and neighbouring
constructions' 1q fix-ups split groups the same way. Measured on the k=2 inspector
circuit: 12 crossing gates group into 5 movements (2.4×), not the documented 4×; groups
were ended by `z`, `sx`, `h`, and `swap` on a participating patch.

- Conservative direction (movement over-charged), but the asymmetry is arbitrary: XCX
  pays 2× the transport of CZ for the same logical work, which quietly penalizes
  whichever compiler choices route through X-type gates — and the packing cost model
  can't see it.
- Cheap fix at the emission layer: hoist the sandwich Hadamards outside the run of
  two-qubit gates (they act on the moved atoms, which are addressable at the target
  patch), or teach the grouper that a 1q gate on a *moved* atom doesn't end the group.

### 8. Latent measurement-reordering hazard in `build_layers`

`flush()` places a fenced batch at `max(earliest(u))` with no floor from previous
flushes ([scheduler.py:120-127](simulator/scheduler.py:120)). Constructed failure: gate
on q0 → measure q0 (pending) → gate on q0 (flushes batch 1 into slot 1) → measure q1
(q1 untouched, pends) → end flush places batch 2 in slot **0**. Emission order is then
M(q1), M(q0) while `run_stim_backend`'s positional `meas_map`
([stim_backend.py:476-484](simulator/stim_backend.py:476)) expects M(q0) first — the
bits are silently relabelled, the exact failure mode
`test_measurement_order_survives_scheduling` was written for (its scenario stays in one
flush and passes).

Not reachable today: the pipeline does one detection round at the end, and FT flag
qubits are never touched after their measurement, so everything lands in a single flush.
But the comment at [scheduler.py:136-140](simulator/scheduler.py:136) ("a mid-circuit
round would split here rather than silently reordering") claims a safety the code does
not have. A one-line fix (track the last flushed slot and clamp `start` to it) or an
assert would close it.

### 9. The noise-ratio axis leaves the biggest movement term invariant

Under the transport model, `with_noise_ratio` sets `idle_time_scalar = r`,
`distance_scalar = 1/r` ([simulation.py:295-300](experiments/shared/simulation.py:295)).
But idle-from-transport error ∝ `idle_time_scalar × (transport seconds)` and transport
seconds ∝ `distance_scalar` (it divides movement speed), so the product — the term the
module docstrings call the dominant cost of movement — is **constant along the entire
ratio sweep**. The sweep actually trades gate/readout idle (∝ r) against move handling
and flight (∝ 1/r). At NA defaults those two halves of the movement budget are
comparable (measured: move handling 1.2%, idle-from-transport 1.2% of total), so the
axis moves only about half of "movement". The `MOVEMENT_BALANCE` deprecation note
covers the wrong-constant problem but not this structural invariance; the four figures
still on this axis (`depth_vs_gates_tradeoff`, `code_size_crossover`,
`1_best_k_phase_diagram`, `10_winning_patch_count`, `2_tvd_vs_noise_ratio_by_k`) inherit
it.

Related trap: `with_idle_noise(idle_noise=False)` zeroes `idle_time_scalar`, which also
kills idle-from-transport — so "idle off, distance on" silently removes the largest
movement channel rather than isolating movement. Worth a docstring warning at minimum.

---

## Low — robustness, stale docs, edge cases

### 10. `SimulatorConfig`'s new knobs are not dataclass fields

`distance_noise`, `distance_scalar`, `idle_noise`, `idle_time_scalar` have no type
annotations ([config.py:23-26](simulator/config.py:23)), so they are class attributes,
not fields: `SimulatorConfig(distance_noise=True)` raises `TypeError`,
`dataclasses.replace`/`asdict` silently drop them, and a careless
`SimulatorConfig.distance_noise = True` would mutate every instance. The fluent setters
work, and nothing in-tree hits the broken paths today — but it is a trap. Add the
annotations.

### 11. `Simulator.clone()` loses settings and mints a seed

[simulator.py:98-108](simulator/simulator.py:98) copies neither the distance/idle flags
nor the scalars, and turns an unseeded simulator (`seed=None`, fresh entropy) into
`seed=0` (fixed stream). No in-tree callers currently, but both behaviours are wrong if
anyone picks it up.

### 12. Layout is built with default geometry, not the run's overrides

`_stimulate` passes `layout=positions_for_circuit(circuit)` with no `geometry`
([simulator.py:297](simulator/simulator.py:297)), while `qiskit_to_stim` computes
centres with the overridden geometry. Benign today — intra-patch relative positions
don't depend on the gutter, and the layout is only consulted for same-patch blockade
checks — but the two geometry sources will diverge the moment someone sweeps
`patch_gutter` and uses absolute positions for anything.

### 13. Packing gap (known): the constants are stale too, not just the geometry

Beyond the three TODOs, `NoiseKnobs` defaults are superconducting-era —
`layer_time=88e-9`, `t2=340.61e-6`, `two_qubit_error=3.5e-3`
([packing/resource/noise.py:52-63](packing/resource/noise.py:52)) — and
`noise_profile_for` derives the planner's idle exchange rate from those defaults
([simulation.py:557-560](experiments/shared/simulation.py:557)) even when the simulator
it feeds runs NA parameters. So the planner and simulator disagree on rates as well as
geometry. Recording it here so the TODO sweep includes it.

### 14. Stale docs contradicting the new defaults

- [config.py:35](simulator/config.py:35): "Table 2, not the drifted hybrid" — the
  default `ScaledNoiseModel(alpha=1.0)` is now NEUTRAL_ATOM, not Table 2.
- [config_presets.py:4-13](simulator/config_presets.py:4): "`ScaledNoiseModel` at
  alpha=1 *is* Table 2 exactly" — same drift. Anyone reading these picks the wrong
  mental model for every preset.
- [testexp.py:330](experiments/tests/testexp.py:330): help text references the deleted
  `DefaultNoiseModel`.
- `MOVEMENT_BALANCE` docstring ([simulation.py:138-143](experiments/shared/simulation.py:138)):
  "movement share of the error budget: k=2 98-99%" reads as share of *total*; measured
  total-share at defaults is ~2.4% (movement/(movement+idle) is ~92%, which is
  presumably what was meant). Ambiguous denominator; worth one clarifying word,
  since nearby text quotes 7% for the same k.

### 15. Analytic split ignores noise markers

`measured_noise_split` and `budget()` count every record and every layer
([simulation.py:219-231](experiments/shared/simulation.py:219)); `qiskit_to_stim`
honours `record.noise` per instruction and per layer. Currently equivalent — no code in
`translator/`/`pipeline/` emits markers, so translated circuits are noisy end-to-end —
but the two will silently diverge if markers come back. Same for the layer-level `noisy`
flag ([stim_backend.py:367-369](simulator/stim_backend.py:367)): ASAP scheduling can mix
marked and unmarked instructions in one layer, charging idle during nominally noise-free
prep. Only relevant if markers return.

### 16. FT mode has no geometry

Flag ancillas are placed **on the same cell** as the patch's first qubit
([grid_constructor.py:178-184](simulator/grid_constructor.py:178)) — two atoms in one
cell, distance zero — and flag-gadget CXs skip both the blockade check and transport
grouping (the flag owns no patch, so the gate is neither same-patch nor crossing). FT
mode therefore simulates gate noise but essentially no movement for the gadgets. Fine
while `fault_tolerant=False` everywhere in the figures; should be stated somewhere if FT
numbers are ever quoted. (One pleasant surprise: flag measurements pend to the final
global image, which is exactly right for this architecture.)

### 17. Cross-patch pairs are never checked against the blockade

A crossing gate is always grouped into transport regardless of separation; only
same-patch pairs consult `needs_transport`
([stim_backend.py:239-245](simulator/stim_backend.py:239)). At default geometry the
nearest cross-patch atoms are 9 µm > 8 µm blockade, so this is correct — but the margin
is 1 µm and nothing asserts it. A swept `patch_gutter=1` (6 µm) or `blockade_radius`
> 9 µm silently charges round trips for pairs that could gate in place. Cheap guard:
assert `(gutter+1) × pitch > blockade_radius` when building geometry. Relatedly, the
`PATCH_GUTTER` comment ([transport.py:122-125](simulator/transport.py:122)) says "2
cells is 6 µm against an 8 µm radius", which describes the gutter as *failing* its
stated purpose; the operative number is nearest-atom spacing, gutter+1 = 3 cells = 9 µm.

### 18. Small items

- `derive_transport_groups` silently uses distance 0.0 when a patch centre is missing
  ([transport.py:265-267](simulator/transport.py:265)) — free transport instead of a
  loud failure. Unreachable via `centres_for_circuit`; an assert would keep it that way.
- `idle_attribution` credits the residual to `"gate"` unconditionally
  ([scheduler.py:216-218](simulator/scheduler.py:216)) even if the counterfactual unit
  is readout. Cosmetic today (readout never shares a layer in practice).
- Gates inside a transport group are charged **serially**
  (`sum` at [stim_backend.py:287](simulator/stim_backend.py:287)); the 4 CX of a logical
  CNOT land on disjoint pairs and could fire in parallel once the atoms arrive. ~1.5 µs
  against a 54 µs trip, so noise-level only — but it is the same "capable machine"
  question the concurrency assumption already answers the other way.
- The Stim gate map ([stim_backend.py:24-38](simulator/stim_backend.py:24)) lacks
  `sdg`/`sxdg`/`rx`/`rz`, which `TRANSLATED_ONE_QUBIT_OPS` says translated circuits can
  contain and which `RZ`/`RX`/`initialize_arbitrary_state` do emit
  ([iceberg_module.py:385-407](iceberg/iceberg_module.py:385)). Failure is a loud
  `NotImplementedError`, and the Clifford pipeline never hits it — fine, just noting the
  asymmetry between the two backends' gate coverage.
- `_run` maps at most one clbit per measure record; qiskit's standard 1:1 measures are
  fine, a hand-built multi-clbit measure instruction would misalign. Not produced
  in-tree.

---

## Physics / feasibility questions (not bugs — the "is this a machine" checklist)

- **No fixed per-move overhead.** Transport time is `2 × distance / speed` with zero
  pickup/handoff/settle time. Demonstrated systems spend tens of µs per
  pickup/drop-off, comparable to or exceeding the 27–54 µs flights modelled here, and
  the docstring's "Deliberately absent" list ([transport.py:68-81](simulator/transport.py:68))
  names acceleration but not this constant term. Since the model's headline mechanism is
  "transport idles everyone", a missing constant that could double trip durations is
  first-order; recommend adding a `move_overhead` term or listing it as deliberately
  absent.
- **1 m/s is on the optimistic edge.** Published moves are ~0.25–0.55 m/s mean
  (acceleration-limited). The docstring honestly records why 0.2 m/s was abandoned
  (trips dominated everything); worth saying "optimistic by ~2×" rather than "the figure
  quoted for current hardware".
- **Concurrent Rydberg gates inside each other's blockade.** The scheduler happily runs
  two disjoint same-patch pairs in one layer 3 µm apart; two simultaneous Rydberg gates
  within a blockade radius interfere. On the model's own numbers, most of a patch is
  inside one blockade sphere, so true intra-patch gate parallelism is ~1. This flatters
  dense intra-patch schedules (mostly decode rounds). Cheap partial fix: serialize
  same-patch two-qubit gates.
- **Gate quality vs separation.** p2 is uniform from 3 µm out to the 8 µm threshold;
  real blockade-gate fidelity degrades steeply with separation (V ∝ 1/r⁶). The
  threshold-not-a-slope simplification is documented and defensible — just noting it is
  generous precisely for the widest intra-patch pairs of large k.
- **Parity atoms sit in the worst cells.** Row-major fill puts the two syndrome ancillas
  in the rectangle's last corner; every data qubit CXes with them in decode, so large-k
  decode pays near-maximal intra-patch distances. A machine designer would put the
  ancillas centrally. Free accuracy win if the fill order ever gets revisited; currently
  it penalizes large k (conservative, but by layout accident rather than by argument).

## What was checked and looks right

- 62/62 tests pass. Geometry: rectangle selection (2×3/3×3/3×4/4×4 as documented),
  one-atom-per-cell, gutter spacing (nearest cross-patch atoms 9 µm > 8 µm blockade),
  centre-to-centre Manhattan distances and round-trip times all verified by hand against
  the inspector output.
- Grouping matches the documented rules on real circuits (4-CX logical CNOT → one trip,
  2 movers; groups end on any touch of either patch; disjoint pairs stay concurrent).
  The control-side inference (`qubits[0]`) is sound for every cross-patch construction
  in `iceberg_set.py` — CNOT/CZ/XCX and all three transversal macros emit control-patch
  first.
- The one-global-image story holds end to end: all data + parity measurements pend to a
  single fenced layer, it is the last layer, every qubit is in it, so it charges
  **zero idle** — matching the stated Iceberg readout model. Measurement/creg alignment
  is safe for this single-flush pattern (see finding 8 for the latent general case).
- ASAP reordering only ever commutes disjoint-qubit units, so the sampled distribution
  is unchanged; barriers are dropped deliberately and only affect timing.
- Thread-local parameter scoping, seeded chunked sampling, `alpha=0` → T=∞ handling,
  Pauli-twirl normalization (sums ≤ 1, safe for `PAULI_CHANNEL_1`), and the
  `min(1.0, ·)` clamp on `move_error` (Stim accepts DEPOLARIZE1 up to 1.0) are all fine.
- The patch→index contiguity assumption in `grid_constructor` (`base = patch × (n+2)`)
  matches `IcebergModule`'s data-then-parity register creation order, including with
  flag registers appended later.
- Parameter values (T1 10 s, T2 1.5 s w/ DD, 1 µs 1q > 0.5 µs 2q inversion, 3 ms image,
  1e-3/trip handling error) are individually defensible and the unusual ones are
  documented with reasons.

## Questions for you

1. Is `move_error` conceptually part of the alpha noise scale or part of the geometry?
   (Finding 5 — it currently scales with `distance_scalar` but not `alpha`, which makes
   calibrated-alpha comparisons movement-heavy.)
2. What is the intended physical story for `swap` — relabel, 3 CX, or atom exchange?
   (Finding 3 — currently it's none of them.)
3. Should XCX's sandwich Hadamards be hoisted so X-type and Z-type cross-patch gates pay
   the same transport? (Finding 7 — currently 2 trips vs 1.)
4. Is FT mode in scope for any quoted numbers? If yes, findings 2 and 16 get bigger.
