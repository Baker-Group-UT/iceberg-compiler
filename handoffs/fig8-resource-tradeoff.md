# Handoff: Fig 8 redrawn as a resource trade (code size k vs shots kept and TVD)

**Branch:** `v2-fig8-tradeoff`, **HEAD `c25e009`**, one commit on `v2-logical-opt-pass` @ `c002567` (the frozen compiler).
- That branch has since moved to `cc8a0e7`, with no changes to `translator/`, `packing/` or `iceberg/`.
- Worktree: `.claude/worktrees/fig8-tradeoff`. It is clean and nothing is running.
- Nothing is pushed. The paper is untouched.

**Why:** the current Fig 8 (`paper/figures/plots/resource_selection_headroom.pdf`, caption at `paper/text/evaluation.tex:283`) ranks by shots kept alone. That rewards weak detection: an oracle choosing on shots keeps 2.10x the shots of naive k=2, but its TVD is 44% worse. Larger k saves physical qubits but costs accuracy, so the figure has to show all three.

## Finished

| What | Result | Files |
|---|---|---|
| Figure: two stacked panels over evenly spaced k. Shots kept / naive k=2 (geometric mean) and mean TVD. Naive and AQORN static k, the selector in its own slot. Tick labels show the qubit saving (patches x n_code). Error bars are a 95% bootstrap over circuits; Ising is dropped (TVD saturates). | AQORN beats naive at every k on both axes. Qubits saved: k4 -23%, k6 -29%, k8 -31%, k10 -34%. TVD: k2 0.099, k4 0.128, k6-10 0.23-0.37. The selector sits on static k=4 (1.80x shots, TVD 0.127, -21% qubits). | Draft `results/revision/exp6/resource_tradeoff_draft_v2.{pdf,png,csv}` (main checkout) |
| Claim numbers: the selector re-ranked at weights 0-1000, and the measured shots/TVD frontier in hindsight | At no weight does the selector beat TVD -11% vs naive k2; static k2 reaches -24%. The frontier beats static k2 and k4 on both axes: 1.33x shots at -26%, 1.83x at -9%. | Printed by the script |
| Script, committed | Reads the CSV only, no simulation | `experiments/revision/exp6_selector/resource_tradeoff.py` |

**Caveats:**
- **Stale input.** The source CSV is the 10-07 standard run (`experiments/figures/resource_selection_headroom/resource_selection_headroom_standard.csv`, main checkout). It predates the logical-opt pass and the `solo_hh_aware` default, so the numbers will move.
- **Gitignored files.** All results and CSVs are gitignored. Copy them to the server if you want them there.
- **Qubits, not resources.** Space-time volume is flat (0.90-0.96x), because depth grows to 1.46x by k=10. Say "qubits" in the caption, not "resources".
- **Shots zigzag in k.** k6 and k10 come out low; this has not been investigated.

## Unfinished

- **Re-run of the headroom experiment under the frozen compiler.** Not queued, as you asked. About 35 min on 12 jobs on the laptop (10-07).
- **Not done yet:** the final figure (no draft banner), the caption, and swapping it into the paper.

## Continue (on the server, repo root, branch checked out)

```bash
git checkout v2-fig8-tradeoff
# Gitignored inputs: benchmarks/qasmbench, plus the logical-opt records (O3 is nondeterministic,
# so without them the programs differ; 2.8 MB, 619 files; or point AQORN_LOGICAL_OPT_CACHE at a copy):
rsync -a <laptop>:"…/iceberg-compiler/benchmarks/qasmbench/" benchmarks/qasmbench/
rsync -a <laptop>:"…/iceberg-compiler/results/cache/logical_optimization/" results/cache/logical_optimization/
PYTHONPATH=. OMP_NUM_THREADS=1 MPLBACKEND=Agg venv/bin/python -m experiments.figures.resource_selection_headroom.resource_selection_headroom --standard --jobs 12 --persist
PYTHONPATH=. venv/bin/python -m experiments.revision.exp6_selector.resource_tradeoff --source experiments/figures/resource_selection_headroom/resource_selection_headroom_standard.csv
```

- The headroom run writes `resource_selection_headroom_standard[_vN].csv` beside its script, so pass the version it printed.
- `resource_tradeoff` writes `results/figures/resource_tradeoff[_vN].{svg,pdf}` and `results/revision/exp6/resource_tradeoff[_vN].csv`.
- Draft banner: `--note "..."`.

## Decisions needed from you

1. **When to run the re-run**, and whether it goes on the server queue.
2. **Claim wording.** Proposed: "Larger codes trade accuracy for physical qubits; most of the saving comes by k=4; AQORN beats naive at every code size." The current caption ("dynamic resource selection beats static code size") does not hold for the selector on TVD. The hindsight frontier supports "choosing k per program beats any static k", but only in the text, since a three-metric plot has no single frontier.
3. **Replacement.** Should this figure replace `resource_selection_headroom.pdf` in the paper?
