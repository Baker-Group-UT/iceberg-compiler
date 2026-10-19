#!/usr/bin/env bash
# After overnight_annealing.sh: re-measure the >= 64-qubit circuits with the overflow fix (82bc36d)
# in both corpus runs, recombine, summarise, redraw, and replace the results/data copies.
set -uo pipefail
W="/Users/cordputer/Documents/Claude/Projects/AQORN Revisions/iceberg-compiler/.claude/worktrees/logical-opt-pass"
PY="$W/../../../venv/bin/python"; L="$W/logs/optimality_front"
D="/Users/cordputer/Documents/Claude/Projects/AQORN Revisions/results/data"
cd "$W"
export OMP_NUM_THREADS=1 MPLBACKEND=Agg TQDM_DISABLE=1 PYTHONPATH=.
until grep -q ' done$' "$L/overnight_annealing.log"; do sleep 30; done
echo "$(date '+%m-%d %H:%M') start at $(git rev-parse --short HEAD)"
WIDE="adder_n64 cat_n65 ising_n66 knn_n67 bv_n70 qugan_n71 wstate_n76 ghz_n78 swap_test_n83 ising_n98 wstate_n118 ghz_n127 cat_n130 bv_n140"
for name in $WIDE; do
  rm -f "results/optimality_front/packers_corpus_parts/$name.csv" "results/optimality_front/packers_corpus_annealing_parts/$name.csv"
done
nice -n 5 "$PY" -u -m experiments.optimality.packers_corpus --jobs 12 > "$L/packers_corpus_rerun.log" 2>&1
echo "$(date '+%H:%M') corpus exit $?"; grep "^\[packers_corpus\]" "$L/packers_corpus_rerun.log" | tail -1
nice -n 5 "$PY" -u -m experiments.optimality.packers_corpus --arms aqorn annealing --annealing-seconds 120 \
    --tag annealing --jobs 12 > "$L/packers_annealing_rerun.log" 2>&1
echo "$(date '+%H:%M') annealing exit $?"; grep "^\[packers_corpus\]" "$L/packers_annealing_rerun.log" | tail -1
"$PY" -m experiments.optimality.annealing_summary --from-csv results/optimality_front/packers_corpus_annealing.csv > "$L/annealing_summary.txt" 2>&1
cat "$L/annealing_summary.txt"
"$PY" -m experiments.figures.packers_corpus.packers_corpus --from-csv results/optimality_front/packers_corpus.csv > "$L/figure_packers_rerun.log" 2>&1
grep "saved\|circuits" "$L/figure_packers_rerun.log"
for f in packers_corpus.csv packers_corpus.csv.json packers_corpus_annealing.csv packers_corpus_annealing.csv.json; do
  cp -f "results/optimality_front/$f" "$D/optimality_front_$f"
done
echo "$(date '+%m-%d %H:%M') done"
