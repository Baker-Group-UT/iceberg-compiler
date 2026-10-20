#!/usr/bin/env bash
# After the main sweep and the redo: one combine over every part (no new measurement), then the figure.
set -uo pipefail
W="/Users/cordputer/Documents/Claude/Projects/AQORN Revisions/iceberg-compiler/.claude/worktrees/logical-opt-pass"
PY="$W/../../../venv/bin/python"
cd "$W"
export OMP_NUM_THREADS=1 MPLBACKEND=Agg TQDM_DISABLE=1 PYTHONPATH=.
until grep -q '^exit' logs/optimality_front/run.log && grep -q '^exit' logs/optimality_front/redo.log; do sleep 30; done
echo "$(date '+%m-%d %H:%M') sweep $(grep '^exit' logs/optimality_front/run.log), redo $(grep '^exit' logs/optimality_front/redo.log)"
nice -n 5 "$PY" -u -m experiments.optimality.exhaustive_front --jobs 4 > logs/optimality_front/final.log 2>&1
echo "$(date '+%m-%d %H:%M') combine exit $?"
grep "^\[exhaustive_front\]" logs/optimality_front/final.log | tail -3
nice -n 5 "$PY" -u -m experiments.figures.optimality_front.optimality_front \
    --from-csv results/optimality_front/space.csv > logs/optimality_front/figure.log 2>&1
echo "$(date '+%m-%d %H:%M') figure exit $?"
cat logs/optimality_front/figure.log | grep -v Warn
