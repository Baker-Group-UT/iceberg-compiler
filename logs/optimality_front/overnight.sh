#!/usr/bin/env bash
# Overnight 2026-10-10: packer comparison, multiplier_n15 sampled front, sweep check, figures.
# Compiler frozen; every step logs and a failure does not stop the next.
set -uo pipefail
W="/Users/cordputer/Documents/Claude/Projects/AQORN Revisions/iceberg-compiler/.claude/worktrees/logical-opt-pass"
PY="$W/../../../venv/bin/python"; L="$W/logs/optimality_front"
D="/Users/cordputer/Documents/Claude/Projects/AQORN Revisions/results/data"
cd "$W"
export OMP_NUM_THREADS=1 MPLBACKEND=Agg TQDM_DISABLE=1 PYTHONPATH=.
until grep -q '^exit' "$L/lane_screen.log"; do sleep 30; done
echo "$(date '+%m-%d %H:%M') start at $(git rev-parse --short HEAD), $(git status --porcelain --untracked-files=no | wc -l | tr -d ' ') dirty tracked files"
step() {  # name, command...
  local name=$1; shift
  echo "$(date '+%H:%M') $name"
  nice -n 5 "$@" > "$L/$name.log" 2>&1; local code=$?
  echo "$(date '+%H:%M') $name exit $code"
  grep -E "^\[(packers_corpus|sampled_front|check_space)\]" "$L/$name.log" | tail -3
}
step packers_corpus "$PY" -u -m experiments.optimality.packers_corpus --jobs 12
step sampled_front "$PY" -u -m experiments.optimality.sampled_front --jobs 12
step check_space "$PY" -u -m experiments.optimality.check_space --jobs 12
step figure_packers "$PY" -m experiments.figures.packers_corpus.packers_corpus --from-csv results/optimality_front/packers_corpus.csv
step figure_sampled "$PY" -m experiments.figures.sampled_front.sampled_front --from-csv results/optimality_front/sampled_front_multiplier_n15.csv
for f in packers_corpus.csv packers_corpus.csv.json sampled_front_multiplier_n15.csv sampled_front_multiplier_n15.csv.json check_space.csv lane_refinement_screen.csv; do
  [ -e "results/optimality_front/$f" ] && [ ! -e "$D/optimality_front_$f" ] && cp "results/optimality_front/$f" "$D/optimality_front_$f"
done
echo "$(date '+%m-%d %H:%M') done"
