# Movement and idle time across Fig 6's stages

v2 revision, 2026-10-07. Branch `v2-movement-idle` (from `e5d5741`). Code:
`experiments/figures/movement_idle/movement_idle.py`.

**Bottom line.** AQORN does not reduce atom movement. Against a baseline scheduled the same
way, the full pipeline carries more atoms on most circuits. The per-circuit mean increase
is 26% with the submitted planner (v4), 85% with the weight-10 planner the paper now uses
(w10) and 66% with w10T. Qubit idle time is mixed. The corpus total falls by 23-28%,
because packing shortens transport flights on the few long, wide circuits that hold most
of the idle time. But most small and medium circuits idle more, so the per-circuit mean
rises (v4 +12.5%, w10 +57.6%, w10T +32.1%). Expected detectable error still falls, by 11-14% per circuit, because gates
make up 75-82% of the error budget and movement and idle only 2-13%. So the paper can
claim fewer and cheaper gates. It cannot claim less movement, and it can claim less
idling only for wide circuits.

## What was measured

- **Same setup as Fig 6:** the 71 circuits and four stages: baseline (tight trivial
  packing, both passes off), + H commutation, + merging, + packing (the planner's
  placement). Each circuit uses its plan (placement plus commutation decision) from
  `ablation_fair_depth`, for v4, w10 and w10T. The w10T transversal switch is on in the
  merging stages only.
- **Ordering.** Every stage, the baseline included, is translated with
  `TranslatorConfig.asap_order = True`. These are the primary numbers. The shipped
  ordering (False) is kept as a reference.
- **Schedule.** This is the simulator's own schedule (Stim backend, transport groups,
  ASAP layers, `idle_attribution`), built as `measured_noise_split` builds it, but the
  script returns amounts instead of errors:
  - **Movement: atoms carried**, one count per atom per carriage (out and back), over
    transport groups, intra-patch moves and atom-exchange swaps. This is the panel because
    the simulator charges `move_error` per carried atom, and that charge is 97-98% of the
    movement error. Flight dephasing (atom-seconds of flight) is the other 2-3% and is kept
    in the CSV.
  - **Idle: qubit-seconds** over every physical qubit and layer. The physical qubit count
    never changes between stages, so idle per qubit gives the same percentages.
  - **Makespan:** wall-clock length, including the final 3 ms readout.
  - The amounts do not depend on alpha. The E_* columns are at alpha = 1; multiply by
    `alpha_final` to get Fig 6's calibration.
- **Originals and proxies.** The figure uses originals. After translation, each physical
  rz/rx is replaced by a Clifford of the same duration. The scheduler reads only an
  instruction's kind and qubits: every 1q gate lasts `t_1q`, none is virtual, and the name
  matters only for `swap`. Clifford proxies (what Fig 6 simulates) are reported as well.
- **Checks, run before anything is written (all pass for all three variants):**
  - Shipped 1q, 2q and depth reproduce the source CSV on 71/71 circuits.
  - Layer-sorted depth matches `ablation_fair_depth` on 71/71.
  - On every proxy schedule, amount × rate equals `measured_noise_split` exactly
    (maximum deviation 0.0).
  - All 416 rotation-bearing original schedules are identical under a second set of
    stand-ins (rz→z, rx→x).
  - The packed v4 and w10 proxy E_* match Experiment8a's `planner_static.csv` (relative
    deviation 8e-13).

## Headline: originals, fair ordering

All values use the caption convention: the unclipped per-circuit mean of each circuit's
cumulative % reduction against the baseline. **Negative means more than the baseline.**
"Corpus total" is the % reduction of the sum over all 71 circuits.

| amount | variant | +H | +merging | +packing | median | worse / better / same | corpus total |
|---|---|---|---|---|---|---|---|
| atoms carried | v4 | -0.9 | -20.2 | -25.7 | -6.7 | 41 / 12 / 18 | -17.0 |
| atoms carried | w10 | -0.9 | -18.5 | -85.2 | -59.4 | 66 / 5 / 0 | -50.6 |
| atoms carried | w10T | -0.2 | -18.2 | -66.1 | -33.3 | 56 / 14 / 1 | -35.5 |
| qubit idle time | v4 | -4.9 | -20.6 | -12.5 | +0.0 | 26 / 34 / 11 | +28.2 |
| qubit idle time | w10 | +3.3 | -6.1 | -57.6 | -36.8 | 44 / 27 / 0 | +23.4 |
| qubit idle time | w10T | +3.2 | -6.6 | -32.1 | -13.7 | 44 / 27 / 0 | +27.8 |
| makespan | v4 | -3.0 | -8.7 | -1.6 | +0.0 | 26 / 34 / 11 | +17.0 |
| makespan | w10 | -0.2 | -3.8 | -10.8 | -4.1 | 44 / 27 / 0 | +14.1 |
| makespan | w10T | -0.4 | -4.0 | -7.9 | -3.2 | 43 / 27 / 1 | +17.4 |
| flight atom-time | v4 | -3.0 | -5.9 | -1.0 | +0.0 | 31 / 23 / 17 | +18.5 |
| flight atom-time | w10 | -2.8 | -4.1 | -53.4 | -29.4 | 47 / 24 / 0 | +11.3 |
| E_total (context) | v4 | +5.8 | +10.8 | +13.8 | +12.7 | 1 / 59 / 11 | +15.5 |
| E_total (context) | w10 | +5.4 | +10.5 | +10.7 | +10.2 | 17 / 54 / 0 | +12.0 |
| E_total (context) | w10T | +5.5 | +12.2 | +14.3 | +12.6 | 15 / 56 / 0 | +15.5 |

The worse/better/same counts compare the full pipeline to the baseline, using the amounts
themselves. E_movement and E_idle track atoms carried and idle time within 1 point (for
example, v4 E_movement is -24.9).

**By size group** (+packing, the full pipeline; w = worse, b = better):

| amount | variant | small (29) | medium (15) | large (27) |
|---|---|---|---|---|
| atoms carried | v4 | -9.7 (17w/7b) | -33.4 (10w/1b) | -38.6 (14w/4b) |
| atoms carried | w10 | -76.9 (24w/5b) | -66.8 (15w/0b) | -104.4 (27w/0b) |
| atoms carried | w10T | -42.4 (20w/8b) | -54.6 (11w/4b) | -98.1 (25w/2b) |
| qubit idle time | v4 | +3.1 (10w/18b) | -24.4 (6w/8b) | -22.7 (10w/8b) |
| qubit idle time | w10 | -69.2 (20w/9b) | -34.3 (6w/9b) | -58.1 (18w/9b) |
| qubit idle time | w10T | -13.9 (16w/13b) | -26.9 (9w/6b) | -54.4 (19w/8b) |
| makespan | v4 | -1.2 (10w/18b) | +0.5 (6w/8b) | -3.3 (10w/8b) |
| makespan | w10 | -9.0 (20w/9b) | -2.6 (6w/9b) | -17.3 (18w/9b) |
| makespan | w10T | -5.0 (15w/13b) | +1.0 (9w/6b) | -15.8 (19w/8b) |

**Reference columns.** Each cell is the +packing value, then how many circuits got worse,
then the corpus total.

| amount | variant | originals, fair | originals, shipped | proxies, fair | proxies, shipped |
|---|---|---|---|---|---|
| atoms carried | v4 | -25.7 (41; -17.0) | -29.3 (41; -23.6) | -26.8 (41; -17.0) | -29.6 (41; -23.8) |
| atoms carried | w10 | -85.2 (66; -50.6) | -95.3 (66; -60.1) | -90.5 (67; -50.8) | -92.6 (67; -59.6) |
| atoms carried | w10T | -66.1 (56; -35.5) | -71.0 (57; -44.3) | -68.8 (56; -36.7) | -71.0 (58; -44.8) |
| qubit idle time | v4 | -12.5 (26; +28.2) | -9.8 (31; +19.0) | -17.5 (33; +27.6) | -14.5 (33; +17.1) |
| qubit idle time | w10 | -57.6 (44; +23.4) | -52.2 (47; +15.6) | -64.3 (47; +22.1) | -56.0 (49; +12.6) |
| qubit idle time | w10T | -32.1 (44; +27.8) | -27.7 (46; +20.4) | -39.1 (46; +26.4) | -33.2 (46; +17.3) |
| makespan | v4 | -1.6 (26; +17.0) | -2.2 (31; +10.2) | -3.3 (33; +15.9) | -4.5 (33; +7.8) |
| makespan | w10 | -10.8 (44; +14.1) | -10.0 (46; +8.0) | -12.4 (47; +12.8) | -12.2 (49; +5.3) |
| makespan | w10T | -7.9 (43; +17.4) | -7.2 (45; +11.4) | -9.7 (46; +15.9) | -9.6 (46; +8.6) |

The shipped ordering makes movement look 4-10 points worse. Per-circuit idle looks 3-5
points better, but the corpus-total idle gain is 7-9 points smaller. No conclusion
changes.

**This reproduces Experiment1a.** v4 proxies with the shipped ordering show movement error
-28.8% per circuit. That is -30.1% over the 68 circuits that move atoms at baseline, which
is Experiment1a's convention and its number (-30.07). Idle error is -14.5% (Experiment1a:
-14.53). The fair baseline does not change the sign of either.

## Which stage moves movement and idle, and why

All figures here are v4, originals, fair ordering, against the stage before.

- **+H commutation.**
  - Net per-circuit effect is small: atoms -0.9 and idle -4.9 points. It raises atoms
    carried on 22 circuits and lowers it on 13.
  - It raises corpus totals by about 8% for both movement and idle.
  - On the Ising family it splits and serializes cross-patch transport. In ising_n34,
    transport groups go from 32 to 49 and transport-paced layers from 18 to 48, so idle
    rises 94%. This happens although only 2 more atoms are carried.
- **+merging: most of the added movement.** It raises atoms carried on 30 circuits and
  lowers it on 6 (mean segment -19.3 points).
  - The whole-patch HH macro is transversal H plus an exchange of the two anchor atoms
    (`IcebergPatch.HH`). Two lane-Hadamard gadgets that moved nothing become one macro
    that carries 2 atoms.
  - After merging, 429 of the 549 physical atom exchanges are HH anchor exchanges. Most
    of the rest come with transversal two-patch macros.
  - Transversal macros carry the whole control patch: 4 atoms instead of about 3. When the
    lane pairs are crossed, they also need lane exchanges. In a two-gate toy,
    cx(0,3) + cx(1,2) goes from 3 to 8 carried atoms and from 43 to 77 µs.
  - Merging's selection prices gates, not movement.
  - Its idle effect is mixed: up on 22 circuits, down on 17. HH is faster than two lane-H
    gadgets (14 vs 24 µs), but on bv and ising the extra exchange and macro layers
    lengthen the schedule (bv_n30: 29 → 84 atoms, idle +128%).
- **+packing, v4: less idling, not less movement.**
  - Total carriages barely change (+154 atoms carried over the corpus, +1.0%; atoms up on
    20 circuits, down on 11).
  - Interacting patches sit closer, so the mean flight per carriage falls from 77 to
    55 µs, transport-paced time by 30%, and corpus idle by 38% against +merging.
  - This is concentrated on the widest circuits. swap_test_n83 goes from 166 to 63 µs per
    carriage, and its idle from 26.5 to 8.3 qubit-s; knn_n67 goes from 14.1 to 4.8.
  - The five circuits with the most idle time (swap_test_n83, qugan_n71, adder_n64,
    knn_n67, dnn_n51) hold 67% of baseline idle. Their idle falls from 71.7 to 38.2
    qubit-s, while the other 66 rise from 35.7 to 38.9. That is why the mean and the total
    disagree.
- **+packing, w10 / w10T: much more movement.**
  - The silent-aware planner splits interacting pairs across patches, because cross-patch
    gadgets leak no silent error.
  - Over the corpus, carriages rise 31% and atoms carried 32% (v4: about 1%). Atoms
    carried rise on 57 circuits (w10). Each carriage is shorter (mean 79 → 47 µs).
  - Per circuit, idle rises on 40. The corpus total still falls, because of the wide
    circuits.
  - cat, ghz and wstate circuits, which v4 leaves untouched, about double their movement.
    For example, ghz_n127 goes from 63 to 148 carriages and its idle from 1.19 to 2.22
    qubit-s.
  - w10T moves less than w10. Presumably this is because in-patch CZ/XCX become
    transversal and leak no silent error, so the planner has less reason to split those
    pairs.

## Caveats

- **The two averages disagree for idle.** The caption mean weights circuits equally. The
  corpus total is dominated by the five circuits with the most idle time. Quote both.
- **Zero-baseline circuits.** bb84_n8, hs4_n4 and qrng_n4 move no atoms at baseline and
  4 after merging (HH). Fig 6's `pct_reduction` counts them as 0, and the figure marks
  them ▼. Excluding them gives v4 -26.8, w10 -89.0 and w10T -69.1.
- **What "idle" means on this machine.** 98% of idling is attributed to transport, and
  97% of pre-readout time is paced by transport. Idle ≈ physical qubits × pre-readout
  time (corpus ratio 0.99), so the idle panel is effectively array size × time spent
  waiting for transport. Makespan adds the fixed 3 ms readout, which is 53% of the median
  circuit's makespan. That is why its percentages are small.
- **The transport model favours shorter moves.** Time is linear in distance at 1 m/s, and
  picking up and putting down an atom costs no time. Groups on disjoint patch pairs run
  concurrently, and atoms return home. The idle gains come from shorter flights, so a
  fixed handling time per carriage erodes them. This is a sensitivity only
  (`with_move_overhead`), not the simulator's model:

  | idle, fair originals | v4 mean / total | w10 mean / total | w10T mean / total |
  |---|---|---|---|
  | model (0 µs) | -12.5 / +28.2 | -57.6 / +23.4 | -32.1 / +27.8 |
  | +20 µs per carriage | -20.8 / +22.1 | -75.7 / +13.3 | -41.2 / +18.8 |
  | +50 µs per carriage | -30.2 / +16.0 | -97.7 / +3.1 | -50.8 / +9.8 |

- **Error budget.** At baseline, movement is 3.2% and idle 1.8% of the median circuit's
  expected detectable error (corpus: 4.5% and 13.2%). Gates are 75-82% and readout
  7-10%. So these increases do not undo Fig 6's gains. Even so, w10 and w10T raise E_total
  on 17 and 15 circuits respectively (v4: 1).
- **Layer-sorting is not neutral for time.** Compared with the shipped baseline, the
  `asap_order` baseline (v4, originals, corpus totals):
  - is 24% shallower in gate depth;
  - takes 11% longer before readout (slower on 30 circuits, faster on 17);
  - carries 6% more atoms and idles 13% more.

  The cause is that layer order interleaves logical gates across patches, which splits
  transport groups (8,655 vs 7,807). The merged stages are already layer-sorted, so their
  numbers barely change (under 0.3%). Two consequences follow. Depth is not a proxy for
  run time on this machine. And the fair baseline is fair for depth, but for movement and
  idle it is the more transport-heavy of the two baselines.
- **The baselines differ by variant.** The w10 and w10T +H and +merging numbers differ
  slightly from v4's because their planners made different commutation decisions.

## Suggested paper text

> *Movement and idling.* Figure X repeats the ablation of Fig. 6 on the scheduled physical
> circuit, with every stage (baseline included) emitted in the same ASAP-layer order.
> AQORN does not reduce atom transport. Averaged per circuit, the full pipeline carries
> 26% more atoms than the naive baseline (41 of 71 circuits carry more, 12 fewer). Per
> circuit, most of this comes from merging: the whole-patch Hadamard exchanges two anchor atoms, and the
> transversal two-patch gates carry the whole control patch. Total qubit idle time, which
> on this machine is almost entirely the array waiting for transport, falls 28% over the
> corpus. Packing places interacting patches close together and shortens flights on the
> widest circuits (up to 66% less idling). It does not do so on small circuits, so the
> per-circuit mean is 12.5% above the baseline. Fig. 6's gains therefore come from fewer
> and cheaper gates, not from less movement.

If the paper ships the weight-10 planner, replace the numbers and add: "the silent-aware
packing trades in-patch gates, which leak undetectable errors, for cross-patch gates,
which do not. It carries 85% more atoms per circuit (66 of 71 circuits carry more) and
idles 58% more per circuit, while total idle over the corpus still falls 23%."

Avoid "packing reduces movement" anywhere. Use "less transport-forced idling on wide
circuits", which is the same wording REVISION_RESULTS item 5 recommended.

## Files and re-runs

- **Figures:** `results/figures/movement_idle_{v4,w10,w10T}.{pdf,svg}` at the project root
  (`AQORN Revisions/results/figures/`), with SVG copies beside the script. Panels are
  #atom moves, qubit idle time and makespan. Fig 6's stacks are kept for circuits that
  improved. Hatched bars below zero are circuits that got worse; they are cut at -200% and
  labelled. The average bar is a waterfall of the per-stage means (caption convention).
- **Data:** `experiments/figures/movement_idle/movement_idle_{v4,w10,w10T}.csv`, plus a
  `.csv.json` sidecar with the checks and a `_segments.csv` in Fig 6's frame. These are
  gitignored and not committed. The long CSV has one row per (circuit, program, ordering,
  stage): every amount, E_* at alpha = 1, gate counts and depth.
- **Compute (about 1 minute per variant at `--jobs 4`), run from the worktree root:**
  `MPLBACKEND=Agg PYTHONPATH=. ../../../venv/bin/python experiments/figures/movement_idle/movement_idle.py --source v4 --jobs 4`
  (and `w10`, `w10T`). The QASMBench submodule must be populated.
- **Redraw and re-summarize without recomputing:**
  `... movement_idle.py --from-csv experiments/figures/movement_idle/movement_idle_v4.csv`.
  Add `--ordering shipped` or `--program proxy` for the other views. Saves are versioned,
  so a redraw writes `_v2` and never replaces an existing file.
