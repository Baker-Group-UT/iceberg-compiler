#!/usr/bin/env bash
# After the exhaustive-front sweep: Fig 6 under the standard + logical optimisation pass.
set -uo pipefail
W="/Users/cordputer/Documents/Claude/Projects/AQORN Revisions/iceberg-compiler/.claude/worktrees/logical-opt-pass"
PY="$W/../../../venv/bin/python"
cd "$W"
export OMP_NUM_THREADS=1 MPLBACKEND=Agg TQDM_DISABLE=1 PYTHONPATH=.
until grep -q '^exit' logs/optimality_front/run.log; do sleep 30; done
echo "$(date '+%m-%d %H:%M') sweep finished: $(grep '^exit' logs/optimality_front/run.log); Fig 6 starts at $(git rev-parse --short HEAD), $(git status --porcelain --untracked-files=no | wc -l | tr -d ' ') dirty tracked files"
nice -n 5 "$PY" -u -m experiments.figures.ablation_standard.ablation_standard --jobs 12 --persist \
    > logs/optimality_front/fig6_logical.log 2>&1
echo "$(date '+%m-%d %H:%M') Fig 6 exit $?"
