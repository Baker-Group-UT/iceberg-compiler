# Logical optimisation as the compiler's first pass

Branch `v2-logical-opt-pass` (from `v2-exact-patches` 61ac6be), 2026-10-09. Every program
now gets one verified round of logical optimisation before anything else sees it.

## What a round is

The optimiser of the logical-optimisation study ([logical_opt.md](logical_opt.md)), unchanged,
moved from `experiments/logical_opt/optimizer.py` to `translator/logical_optimizer.py` (the old
path re-exports it): Qiskit O3 to a fixed point, plus whole-circuit and block Clifford
resynthesis, CNOT-block resynthesis and a commutation route; the best by (2q gates, total gates,
depth); never worse than the input; kept only if it is the same unitary up to global phase
(operator up to 12 qubits, tableau for Clifford circuits, a three-state miter otherwise). Final
measurements are split off and put back on the same clbits; qubits are never permuted.

**Then Clifford-angle rotations become named gates** (`translator_cliffords`; added in the
second version of the pass, `verified-routes-2`). The optimiser works in the corpus basis and
writes Cliffords as rotations: 1,841 of the 8,684 rotations in the study's 56 optimised Fig 6
circuits (in 36 of them) sit at multiples of pi/2. The translator builds a rotation as the RZ
gadget -- more two-qubit gates than the S gadget and about twice its silent error -- and Stim
cannot simulate one at all, so every rz / rx at a quarter turn is written as s, z, s+z, sx, x or
sx+x (exact up to global phase). The study's Stage B / C numbers were therefore measured on
circuits slightly costlier than the pass now produces.

**A barrier ending the program belongs to the measurement block** (`measure_all` puts one
there), so a program and its measured copy share one record. Before this, the planner (often
handed the unmeasured program) and the translator (handed `ensure_measurements(program)`)
could each get a different circuit.

## Where it runs

`translator/logical_optimization.py::logically_optimized`, in the three places every program
passes through, so planner and translator cannot see different programs:

| place | what it covers |
|---|---|
| `Translator.translate` (first, before `cancel_inverse_gates`) | every translation: experiments, the planner's verify and commutation decisions, the selector |
| `packing.tables.translated_view` | the planner's view of the program: every cost model, interaction graph, partitioner |
| QASMBench loaders (`create_proxy_from_qasm`, `build_circuit`, via `finish_loaded`) | the measurement-free original is optimised *before* its Clifford proxy is made; the returned circuit is registered as already optimised, so the translator never re-simplifies a proxy |

## Switch, determinism, cost

- **On by default.** `AQORN_LOGICAL_OPT=0` (or `with logical_optimization.disabled():`) turns it
  off everywhere; the switch is an environment variable so spawn workers inherit it.
  `regenerate_submission.sh` sets it off; the logical-opt study forces it off (it compares
  originals with optimised circuits itself); the test suite runs with it off (`conftest.py`).
- **Deterministic by record.** O3 is not deterministic, so the first answer for each program is
  recorded (content-addressed QPY, first writer wins) in the main checkout's
  `results/cache/logical_optimization/`, shared by all worktrees. **Keep that folder with the
  results**: deleting it re-optimises and O3 may choose differently.
- **Fig 6's 71 circuits are seeded with the study's own choices** (`experiments/logical_opt/
  seed_records.py`): 56 optimised circuits, each re-verified, and 15 unchanged, each then with
  its Clifford-angle rotations named. Validated 71/71 through the loader (original and proxy)
  and through the measured path the translator sees. `--replace NAME ...` (with
  `drop_record`) fixes a record written by mistake -- needed once, when a smoke run recorded
  fresh answers for basis_trotter_n4, pea_n5, qaoa_n6 and variational_n4 before the seed got
  to them.
- **Time budget.** A program not on record is optimised in a child process with a 120 s budget
  (`AQORN_LOGICAL_OPT_TIMEOUT`); on overrun it is killed and recorded unchanged. Needed because
  the miter check can blow up on wide non-Clifford circuits and cannot be interrupted from
  Python. A cache miss costs ~2 s of start-up; small programs optimise in < 1 s, gcm_h6
  (13 qubits, 2640 gates) in 6 s. With the records warm, loading Fig 6's 71 circuits takes
  11.8 s with the pass on against 8.6 s off; their logical gates fall 20,717 -> 16,579 (-20%).
  The first load also optimised the folders' other circuits (5.5 min, once); 220 records, 1.1 MB.
- **Provenance.** `TranslatorConfig.to_dict()`, revision sidecars and settings fingerprints, and
  `stamp_provenance` record the optimiser version when the pass is on (absent when off, so
  older records are unchanged). `ResultTable` refuses to resume a table across the switch;
  `check_provenance` warns on a CSV made with the other setting.

## What it changes downstream (not yet re-run)

- **Every number** in a figure made since the submission was made without the pass. The study
  measured the size of the change on Fig 6 (w10, Fig 6 alpha): AQORN's 2q reduction
  15.5 -> 9.6%, TVD 38.2 -> 29.7%, LSR +25.9 -> +9.7%, depth unchanged (15.3%).
- **`ablation_standard` (Fig 6 under the standard) is updated**: with the pass on it plans every
  circuit on its optimised program (the recorded w10T plans are not used) and writes
  `ablation_standard_logical*`; the submitted-figure gate runs with the pass off and still
  reproduces Fig 6 exactly (adder_n10, bell_n4).
- **Scripts that still assume the old programs**: anything that reuses recorded plans or arms
  (`density_ablation_sweep --reuse-arms`, the selector's recorded candidates, the older
  `exhaustive_*` plans) -- re-run, do not resume.
- **Fig 6's own loader** keeps circuits of at most 2,500 gates after loading; optimised
  circuits are smaller, so it may admit circuits it used to drop. `load_fig6_circuits` pins the
  71 by name and is unaffected.
- **Illustration scripts** that translate a hand-built example circuit (the merging examples)
  now show the optimised circuit; run them with `AQORN_LOGICAL_OPT=0` if the example's gates
  are the point.

## Tests

`tests/test_logical_optimization.py` (17, pass on, temporary records): smaller and same
unitary; Clifford-angle rotations come back as named gates; a program and its measured copy
share a record; measurements kept; minimal programs and mid-circuit measurement left alone; one
answer in memory, from disk and across two spawn workers; outputs and registered proxies are
fixed points; the budget path in a child process, and an overrun recorded unchanged; the switch
and its propagation; provenance; planner view, translation and `plan_standard` all of the
optimised program; the loader optimises before the proxy. Full suite with the pass off: 839
pass. With the pass off, the new loaders return exactly the old circuits on all 186 QASMBench
loads (93 files, original and proxy).
