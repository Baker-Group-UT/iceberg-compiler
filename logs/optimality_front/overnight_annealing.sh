#!/usr/bin/env bash
# After overnight.sh: AQORN and simulated annealing (120 s per circuit) on all 71 Fig 6 circuits.
set -uo pipefail
W="/Users/cordputer/Documents/Claude/Projects/AQORN Revisions/iceberg-compiler/.claude/worktrees/logical-opt-pass"
PY="$W/../../../venv/bin/python"; L="$W/logs/optimality_front"
cd "$W"
export OMP_NUM_THREADS=1 MPLBACKEND=Agg TQDM_DISABLE=1 PYTHONPATH=.
until grep -q ' done$' "$L/overnight.log"; do sleep 30; done
echo "$(date '+%m-%d %H:%M') start at $(git rev-parse --short HEAD)"
nice -n 5 "$PY" -u -m experiments.optimality.packers_corpus --arms aqorn annealing \
    --annealing-seconds 120 --tag annealing --jobs 12 > "$L/packers_annealing.log" 2>&1
echo "$(date '+%H:%M') annealing corpus exit $?"
"$PY" -m experiments.optimality.annealing_summary \
    --from-csv results/optimality_front/packers_corpus_annealing.csv > "$L/annealing_summary.txt" 2>&1
cat "$L/annealing_summary.txt"
D="/Users/cordputer/Documents/Claude/Projects/AQORN Revisions/results/data"
[ -e "$D/optimality_front_packers_corpus_annealing.csv" ] || cp results/optimality_front/packers_corpus_annealing.csv* "$D/" 2>/dev/null
echo "$(date '+%m-%d %H:%M') done"
