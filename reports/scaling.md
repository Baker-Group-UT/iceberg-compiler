# Compile-time scaling (v2 revision)

Question: does the method scale as claimed?

Measured 2026-10-07 01:00-04:21 on branch `v2-scaling` (base e5d5741), one core of an Apple
M3 Max (14 cores, 36 GB) that other jobs were also using. Code:
`experiments/scaling/compile_scaling.py` (sweep, analysis, figures) and
`experiments/scaling/incremental.py` (exact incremental evaluation). Data (gitignored):
`results/scaling/`. Logs: `logs/scaling/`.

## Bottom line

1. **Compile time is the placement search, not translation.** Every plan makes exactly three
   exact translations (two commutation arms and the verify baseline), whatever n is. The descent
   on the analytic model takes 85-90% of planner CPU at 32 qubits and 97-99% from 64 qubits on.
   The paper's sentence "the wall clock is dominated by the handful of exact translations the
   pipeline performs, and every benchmark compiles in seconds" is wrong on both counts (it was
   already false for the largest Fig 6 circuits: Experiment7c timed bv_n140 at 182 s). It needs
   replacing; suggested text below.
2. **As implemented, the weight-10 planner grows as n^3.1-3.6** (CPU time, n = 32-128, every
   k): 1.0-1.4 s at 32 qubits, 10-16 s at 64, 89-207 s at 128 (k = 2 slowest). The search's
   work, candidate moves priced, grows only as n^2.1-2.5. The rest is per-candidate cost: two
   objective terms, movement over every patch pair and busiest-patch depth over every gate pair,
   are re-summed over the whole partition for every candidate.
3. **An exact incremental evaluation of those two terms grows like the work count:** n^2.3 at
   k = 2 and n^2.5-2.9 at k = 4-10. Same placement, same descent counts and same model cost bit
   for bit on all 63 cells compared (26 random programs at n = 8-256 and every k, 37 QASMBench
   bv/knn cells at n = 14-140). Quiet re-timing at n = 128: 18 s against 203 s at k = 2 (11x),
   55 s against 95 s at k = 6 (1.7x), 77 s against 92 s at k = 10 (1.2x). It compiled 256 qubits
   in 2-10 min (k = 2-10) and 512 qubits at k = 2 in 10 min.
4. **Quality holds as n grows.** Random programs against the fair (layer-sorted) naive
   baseline: 2q-gate reduction 26-31% at k = 2 and 18-25% at k = 4-10 for n = 32-512, depth
   26-47%, 1q 25-37%, no trend in n. At k >= 4 nearly all of the 2q gain is the passes (naive
   placement with both passes: 18-21%); the weight-10 placement adds -2.3 to +4.5 points there
   and 6-9 points at k = 2. Real and structured circuits keep their per-family levels as n
   grows (2q: bv 40-57%, knn 11-18%, adder 13-17%; none for GHZ and banded QFT), but depth can
   be negative on chain-like circuits at k = 2 (QASMBench ising, knn).
5. **Memory is the next wall.** Peak RSS 1.5-3 GB at n = 128 as implemented; with the
   incremental evaluation 1.8 GB (k = 2) to 12 GB (k = 10) at n = 256 and 7.6 GB at k = 2,
   n = 512; k = 4 at n = 512 passed the 16 GB cap and was stopped. `PackingCost` memoises every
   patch-pair term the search meets (`_binary`, `_unary`) and never evicts.
6. **Where each series stopped** (30-min cap per planner run, 16 GB memory cap): as implemented,
   128 for every k, plus one run at k = 10, n = 256 (870 s, 15.4 GB: just under the memory cap;
   k = 2 at 256 extrapolates to about 43 min). Incremental: 512 at k = 2 (1024 extrapolates to
   about 46 min), 256 at k = 4 (512 hit the memory cap), 256 at k = 6-10 (512 predicted 33-44
   min, not run).

## What was run

| arm | what | k | n |
|---|---|---|---|
| **w10** | the paper's planner: `plan_silent_aware(..., silent_weight=10, noise=fig6_noise_profile())`, gadget silent rates per n_code (4-8 from Experiment8's table, 10 and 12 computed for this run into `results/scaling/rates/`) | 2-10 | 8-128, and 256 at k = 10 |
| **w10i** | w10 with `incremental.py` installed: identical algorithm and results, faster evaluation | 2-10 | 8-512 |
| v4 | submitted planner `plan_packing(..., h_commutation="auto", verify=True, noise=fig6_noise_profile())`, reference | 2-10 | 8-128 |
| w10T | w10 + `TRANSVERSAL_INTRA_TWO_QUBIT` (transversal rate table); k = 2 only, because the switch is gated on n_code == 4 in `iceberg/costs.py` and `iceberg_module.py` | 2 | 8-128 |

Workloads: **random** = Experiment7c's generator (planted 2-blocks on shuffled labels, locality
0.85, 10 logical gates per qubit, 40% CX/CZ, Clifford only; seed 20261001 + 1000 n), sizes 8,
16, ..., doubling. Families with fixed gates per qubit: generated GHZ chain, Bernstein-Vazirani
(all-ones secret), banded QFT (bandwidth 4, CP as rz/cx), 1D transverse-field Ising (3 Trotter
steps, rz/rx); and QASMBench ghz, bv, ising, knn, swap_test, adder, wstate at their corpus widths
(up to 255-433 qubits, loaded exactly as Fig 6 loads originals).

Per cell (planner, family, k, n), one fresh process, strictly one timing cell at a time:
switches first, a warm-up plan, Experiment7c's cache clearing before every repeat, 3 repeats and
the median below 60 s, else one. Recorded: CPU (`process_time`, primary) and wall seconds of the
planner and of translating its plan; `os.getloadavg()` and a fixed speed probe beside every
repeat; descent evaluations (all descents, both commutation arms), sweeps, accepted moves,
cost-term computations/lookups, exact translator calls, schedule pricings; an exclusive-time
phase breakdown; peak RSS; and quality: 1q/2q/depth of the planned circuit against the naive
packing with both passes off, every arm layer-sorted (`asap_order`, the fair baseline), plus the
naive packing with both passes on. Library code is untouched; instrumentation wraps functions
inside the child only.

## Results

### Compile CPU time (s), random programs

Compile = planner + translating its plan, as measured. Load average in brackets where other
jobs' load was high, and for the one late w10 cell; unbracketed w10 cells ran at load 4-15. See
Timing conditions: cells measured at load 40-55 read about 30% high.

| k | arm | n=32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|
| 2 | w10 (as implemented) | 1.4 | 16.1 | 207 | - | - |
| 2 | w10i (incremental) | 1.1 [44] | 5.2 [47] | 24.7 [51] | 129 [42] | 620 [16] |
| 4 | w10 | 1.1 | 10.4 | 123 | - | - |
| 4 | w10i | 1.4 [43] | 9.3 [46] | 58 [54] | 269 [25] | memory > 16 GB |
| 6 | w10 | 1.4 | 10.7 | 110 | - | - |
| 6 | w10i | 1.9 [43] | 11.6 [46] | 71 [47] | 425 [19] | predicted 33 min |
| 8 | w10 | 1.1 | 10.7 | 89 | - | - |
| 8 | w10i | 1.5 [43] | 12.8 [53] | 75 [38] | 456 [18] | predicted 35 min |
| 10 | w10 | 1.0 | 10.9 | 108 | 870 [3.5] | - |
| 10 | w10i | 1.4 [44] | 13.6 [45] | 100 [38] | 587 [17] | predicted 44 min |

Quiet re-timing (load ~5, nothing else of ours running), CPU s:

| n | k | w10 | w10i | v4 | w10 / w10i |
|---|---|---|---|---|---|
| 64 | 2 | 15.6 | 3.8 | 11.0 | 4.1x |
| 64 | 6 | 10.4 | 8.9 | 6.8 | 1.2x |
| 64 | 10 | 10.6 | 10.4 | 8.6 | 1.0x |
| 128 | 2 | 203 | 18.2 | 202 | 11.2x |
| 128 | 6 | 95 | 55 | 84 | 1.7x |
| 128 | 10 | 92 | 77 | 62 | 1.2x |

v4 is 10-35% cheaper than w10 (the silent-error terms cost per candidate) except at k = 2,
n = 128, where they tie. w10T (k = 2) read 313 s at n = 128 under load 29, about 250 s once the
25-30% load inflation is taken off, against 203 s for w10.

### Exponents (log-log least squares over n >= 32; stderr from 3-5 points)

| k | w10 CPU | w10 evaluations | w10i CPU | w10i evaluations | v4 CPU | v4 evaluations | us/evaluation, w10 / w10i |
|---|---|---|---|---|---|---|---|
| 2 | 3.62 | 2.13 | 2.30 (n = 32-512) | 2.21 | 3.88 | 2.45 | 1.62 / -0.01 |
| 4 | 3.38 | 2.44 | 2.54 (32-256) | 2.40 | 3.34 | 2.41 | 1.05 / 0.10 |
| 6 | 3.14 | 2.22 | 2.62 (32-256) | 2.29 | 3.24 | 2.38 | 0.99 / 0.28 |
| 8 | 3.17 | 2.41 | 2.74 (32-256) | 2.39 | 3.22 | 2.56 | 0.87 / 0.31 |
| 10 | 3.24 (32-256) | 2.43 | 2.90 (32-256) | 2.43 | 3.22 | 2.48 | 0.89 / 0.41 |

w10 and v4 fits use n = 32-128 (3 points, stderr 0.02-0.22; w10 at k = 10 adds n = 256); w10i
4-5 points (stderr <= 0.11).
Peak memory grows as n^1.3-1.9 for every arm.

Local slopes between successive doublings are stable (w10 2.9-3.7; w10i 2.2-2.7 at k <= 6). The
n <= 16 points sit low because the planner restarts its descent from 4 shuffled seeds at
n <= 16 (`SMALL_ENOUGH_TO_RESTART`); fits start at 32. The w10i exponents are, if anything,
slightly low: its small-n cells ran under heavier load (read ~30% high) than its large-n cells.

### Where the time goes (median run, random)

| arm | n | descent % of planner | exact translation % | translator calls | us per evaluation |
|---|---|---|---|---|---|
| w10 | 32 | 85-90 | 5-9 | 3 | 83-102 |
| w10 | 64 | 96.5-97.4 | 1.3-2.7 | 3 | 157-225 |
| w10 | 128 | 98.8-99.1 | 0.5-1.1 | 3 | 339-788 |
| w10i, k = 2 | 128 / 512 | 86 / 80 | 12 / 19 | 3 | 106 / 75 |
| w10i, k = 10 | 128 / 256 | 99.2 / 99.6 | 0.4 / 0.3 | 3 | 431 / 356 |

Profiled (cProfile, w10, k = 2, n = 64): `movement_hops` is 76% of the descent's time,
`busiest_patch_depth` 12%, the unary/binary delta 12%. Translating the chosen plan is cheap until
n is large: 0.7 s at 128 and 34 s at 512 (k = 2).

### Quality, random programs (% reduction against the fair baseline; w10 = w10i)

| k | metric | n=32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|
| 2 | 2q | 30.3 | 30.9 | 26.6 | 26.3 | 25.9 |
| 2 | 2q, passes only (naive placement) | 22.9 | 22.3 | 20.1 | 19.8 | 19.8 |
| 2 | depth | 38.8 | 32.4 | 39.0 | 45.8 | 44.1 |
| 4 | 2q | 20.0 | 25.1 | 19.2 | 20.5 | |
| 4 | depth | 44.5 | 28.6 | 31.0 | 35.3 | |
| 6 | 2q | 20.6 | 21.6 | 18.3 | 19.4 | |
| 6 | depth | 27.8 | 35.2 | 26.0 | 32.1 | |
| 10 | 2q | 18.1 | 20.0 | 18.2 | 18.9 | |
| 10 | depth | 43.1 | 39.8 | 35.3 | 26.7 | |

One seed per size, so single points move by a few percent; the level does not drift with n.

### Structured and real circuits

**Timed at k = 2** (incremental evaluation, sequential, machine quiet: load 5-6):

| family | n | compile CPU at largest n | peak RSS | CPU exponent (evaluations) | 2q / depth reduction at ~128 | at largest n |
|---|---|---|---|---|---|---|
| QASMBench adder | 10-433 | 521 s | 8.2 GB | 2.45 (2.38) | 14.9 / 42.2 | 14.3 / 44.2 |
| QASMBench ising | 26-420 | 199 s | 2.8 GB | 2.36 (2.33) | 23.4 / -248 at 98 | 22.2 / -19.4 |
| QASMBench knn | 25-341 | 112 s | 1.1 GB | 2.19 (2.13) | 15.8 / -9.9 | 15.8 / -10.8 |
| QASMBench bv | 14-280 | 34 s | 0.7 GB | 2.19 (2.20) | 51.1 / 10.6 | 50.9 / 5.7 |
| QASMBench ghz | 23-255 | 47 s | 2.1 GB | 2.17 (2.10) | 0.0 / 40.6 | 0.0 / 41.9 |
| GHZ chain | 8-512 | 183 s | 3.9 GB | 2.14 (2.07) | 0.0 / 40.7 | 0.0 / 0.0 (fell back to naive) |
| Bernstein-Vazirani | 8-256 | 24 s | 1.0 GB | 2.13 (2.03) | 45.5 / 64.8 | 45.6 / 65.7 |
| banded QFT | 8-256 | 117 s | 1.8 GB | 2.27 (2.13) | 0.0 / 37.0 | 0.0 / 37.2 |
| Ising chain | 8-256 | 82 s | 1.1 GB | 2.27 (2.22) | 13.1 / 15.2 | 13.6 / 1.9 |
| random (for reference) | 8-512 | 620 s | 7.6 GB | 2.30 (2.21) | 26.6 / 39.0 | 25.9 / 44.1 |

The largest QASMBench circuits, which Fig 6 left out (255-433 qubits), compile in 1-9 minutes at
k = 2. bv/qft/ising at 512 were not started (the invocation's budget ran out).

**All k, n <= 140** (non-timing pool, 255 cells, 11 families): the descent's work grows as
n^2.0-2.7 for every family and code size -- the cost is set by the move set, not by circuit
structure. Quality per family is flat in n but not uniformly positive:

- 2q reduction: bv 40-57%, ising (QASMBench) 17-30%, knn 11-18%, swap_test 12-18%, adder 13-17%;
  GHZ chains and banded QFT get none (0%); wstate gets -2 to -4% at every k and n (the passes
  alone give 0%, and the weight-10 placement accepts a few more 2q gates for fewer in-patch,
  silent-error-prone ones).
- Depth: positive and growing with n for bv, ghz, adder, qft (16-76%); negative at k = 2 on
  QASMBench ising (-248% at n = 98, the known ising_n98 case: a 216-layer plan against a
  62-layer fair baseline; -19% at n = 420), knn (-3 to -11%) and swap_test (-9%); those turn
  positive by k = 6-10.
- One verify fallback in the timed cells at scale: GHZ at n = 512 (the packed placement priced
  4% more expected error than naive, so naive was kept and the reduction is 0; at n = 256 the
  packed placement won with 42% depth reduction). The safety net works; the model and the
  schedule disagree more at that width.


### Timing conditions (read before quoting seconds)

The machine was shared. Load average was 4-15 during the as-implemented w10 series (wall/CPU
<= 1.01) and 16-55 during the v4, w10T and w10i series, when other experiments ran 13-17 worker
processes. A non-timing pool of mine (3 processes under `taskpolicy -b`, efficiency cores) ran
the family sweep at all k from 01:17 to 02:54, alongside the w10 n = 128 cells, all of v4/w10T
and the w10i series; the family timings, the re-times and the w10 run at n = 256 ran with nothing
else of ours. Every timing row carries
`pool_cells_overlapping` (computed from the pool's own timestamps), and rows from 01:50 on (w10i,
families, re-times, w10 at 256) also `concurrent_cells_max/mean` (other measurement processes
seen while the cell ran), `probe_*` and `load1_during_mean`. Re-timing 18 representative cells
at the end with nothing else of ours running (table above): cells first measured at load 40-55
read 24-31% high in CPU time (wall up to 43%);
the as-implemented w10 cells read 1-3% high at load 5-7 and 15-17% high at load 12-15. So CPU
time is the number to quote, the w10 table is close to quiet, and the w10i/v4 tables are upper
bounds; work counts (evaluations, cost terms, translator calls) are exact and load-free, and
were identical across every repeat.

## What this means for the paper

Replace the last sentences of Scaling and Limitations (`evaluation.tex`, "Lastly, the
evaluation's ceiling is practical rather than algorithmic ... compiles in seconds") with
measured statements. If the incremental evaluation is adopted into `packing/search.py` (it
changes no placement, so no other result moves):

> Compile time is set by the placement search, not by translation: a plan makes three exact
> translations whatever the program's width, while the descent prices O(n^2) candidate moves per
> sweep on the analytic model (measured n^2.1-2.5). On one laptop core a 128-qubit program
> compiles in 18 s at k = 2 and 55-77 s at k = 6-10, a 256-qubit program in 2-10 minutes, and a
> 512-qubit program in 10 minutes at k = 2; time grows as n^2.3-2.9. The largest QASMBench
> circuits (255-433 qubits) compile in 1-9 minutes at k = 2, and on random programs the gate and
> depth reductions do not change with width (Fig. X).

If it is not adopted, the honest version is:

> ... as implemented, each candidate also re-sums the movement and depth terms over the whole
> partition, so time grows as n^3.1-3.6: about 1 s at 32 qubits, 10-16 s at 64 and 1.5-3.5 min
> at 128 on one laptop core.

Either way, drop "every benchmark compiles in seconds", and keep the 150-qubit / 2,500-gate
statement as a property of the benchmark set (the sweep goes to 512 qubits / 5,120 gates). In
short: the cost is polynomial with a measured exponent, quality is flat in n, and
the next limit is memory (memo caches), not time. Figure candidate for a single column:
`compile_scaling_compact_w10i_vs_w10_random_v2` (time and 2q reduction against n, one line per
k, as-implemented dotted).

## Caveats

- Laptop, single-threaded Python, shared machine: absolute seconds are machine- and load-specific;
  exponents, counts and quality are not.
- One random seed per size; exponents from 3 points for w10/v4 (n = 32-128) and 4-5 for w10i.
- `incremental.py` is an experiment-side patch (installed in the measurement process only).
  Adopting it means porting it into `packing/search.py`. Its exactness rests on every summand of
  the movement and depth terms being an integer-valued double (integer hop distances, counts and
  depths), and is checked cell by cell, not assumed.
- The weight-10 planner trades gate count for silent-error reduction by design, so at k >= 4 its
  2q reduction is close to the passes alone; that is a property of the objective, not of size.
- Depth against the fair baseline can be negative on chain-like real circuits: QASMBench ising at
  k = 2, n = 98 reads -248% (the known ising_n98 case: 216 layers against a 62-layer fair
  baseline), knn reads -2 to -11% at k = 2. The random family shows nothing similar.
- Not reached: n = 1024 by any arm, n = 256 as implemented except at k = 10, n = 512 with the
  incremental evaluation except at k = 2 (time or memory caps, above). The cap rule skipped a
  size when the series' last two sizes extrapolated past 30 min, so those limits are predictions,
  recorded with their predicted seconds (`skipped_predicted`), except k = 4 at 512, which ran and
  was killed at 17.5 GB.
- The QASMBench files were missing from this worktree until the coordinator copied them in at
  about 01:20; nothing ran against the empty corpus (the QASMBench families were added after).

## Files and re-running

Run from the repository root (in this worktree the venv is `../../../venv/bin/python`).

| what | path |
|---|---|
| sweep, analysis, figures | `experiments/scaling/compile_scaling.py` |
| exact incremental evaluation (opt-in, experiment-side) | `experiments/scaling/incremental.py` |
| timing cells, raw (append-only) | `results/scaling/compile_scaling_runs.jsonl` |
| timing cells + concurrency flags + re-times, what the figures read | `results/scaling/compile_scaling_final.csv` |
| exponents | `results/scaling/compile_scaling_final_fits.csv` |
| non-timing family sweep (all k, n <= 140; `timing_valid = False`) | `results/scaling/compile_scaling_pool.csv` (+ `_pool_runs.jsonl`) |
| re-timed cells | `results/scaling/compile_scaling_retime.csv` (+ `_retime_runs.jsonl`) |
| silent rates for n_code 10 and 12 | `results/scaling/rates/gadget_rates.csv` |
| logs, queue of exact commands run | `logs/scaling/` (`queue.log`, `tier*.log`, `pool.log`, `retime.log`) |
| figures (versioned PDF + SVG) | `/Users/cordputer/Documents/Claude/Projects/AQORN Revisions/results/figures/compile_scaling_*` |

Figures (`results/figures/` at the project root; PDF + SVG, and an SVG copy beside the script).
The final set is the `_v2` files; the unsuffixed files are the first render of the same CSV
(superseded: tick-label thinning and one legend moved, no data change).

| figure | width | content |
|---|---|---|
| `compile_scaling_compact_w10i_vs_w10_random_v2` | column | **paper candidate**: (a) compile CPU time against n, one line per k, incremental solid and as implemented dotted, fitted exponents in the legend; (b) 2q-gate reduction against n |
| `compile_scaling_compact_w10_random_v2` | column | the same for the as-implemented planner alone |
| `compile_scaling_w10_random_v2` | full | as implemented: (a) compile CPU time, (b) descent evaluations with an n^2 guide, (c) 2q and (d) depth reduction against n, per k |
| `compile_scaling_w10i_random_v2` | full | the same with the incremental evaluation, to n = 512 |
| `compile_scaling_breakdown_w10_random_v2`, `compile_scaling_breakdown_w10i_random_v2` | column | (a) the k-axis view, compile CPU time against k at fixed n; (b) descent share of planner time against n |
| `compile_scaling_planners_k2_random_v2` | column | w10, w10i, v4 and w10T at k = 2: time and 2q reduction |
| `compile_scaling_families_k2_v2` | full | generated (top) and QASMBench (bottom) families at k = 2: compile CPU time, 2q and depth reduction; the ising_n98 depth point (-248%) sits on the axis floor at -60% with its value written beside it |


Redraw everything and print every table here, no compute (seconds):

    MPLBACKEND=Agg PYTHONPATH=. venv/bin/python experiments/scaling/compile_scaling.py \
        --from-csv results/scaling/compile_scaling_final.csv

Recompute (each line is one sequential timing invocation; it resumes by cell, so re-running a
line only fills what is missing; budgets and caps as used tonight):

    PYTHONPATH=. venv/bin/python experiments/revision/exp8_undetected/gadgets.py --n-codes 10 12 --out results/scaling/rates
    PYTHONPATH=. venv/bin/python experiments/scaling/compile_scaling.py --planners w10 --families random --sizes 8 16 32 64 128
    PYTHONPATH=. venv/bin/python experiments/scaling/compile_scaling.py --planners v4 w10T --families random --sizes 8 16 32 64 128
    PYTHONPATH=. venv/bin/python experiments/scaling/compile_scaling.py --planners w10i --families random --sizes 8 16 32 64 128 256 512 --predict-factor 1.0 --mem-cap-gb 16
    PYTHONPATH=. venv/bin/python experiments/scaling/compile_scaling.py --planners w10i --families ghz bv qft ising qb_ghz qb_bv qb_ising qb_knn qb_adder --ks 2 --sizes 8 16 32 64 128 256 512 --predict-factor 1.0
    PYTHONPATH=. venv/bin/python experiments/scaling/compile_scaling.py --planners w10 --families random --ks 10 --sizes 8 16 32 64 128 256 --predict-factor 1.0 --mem-cap-gb 16
    PYTHONPATH=. venv/bin/python experiments/scaling/compile_scaling.py --planners w10 v4 w10i --families random --ks 2 6 10 --sizes 64 128 --runs results/scaling/compile_scaling_retime_runs.jsonl --csv results/scaling/compile_scaling_retime.csv
    PYTHONPATH=. venv/bin/python experiments/scaling/compile_scaling.py --planners w10i --families ghz bv qft ising qb_ghz qb_bv qb_ising qb_knn qb_swap_test qb_adder qb_wstate --max-n 140 --workers 3 --background --runs results/scaling/compile_scaling_pool_runs.jsonl --csv results/scaling/compile_scaling_pool.csv
    PYTHONPATH=. venv/bin/python experiments/scaling/compile_scaling.py --finalize --no-plot

The pool line is non-timing (three efficiency-core workers); never run it beside a timing line
if the seconds matter, and never run two timing lines at once.

