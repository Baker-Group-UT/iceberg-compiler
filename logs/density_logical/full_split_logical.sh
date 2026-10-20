#!/usr/bin/env bash
# The density ablation (split merging, matched patch counts) under the logical optimisation pass
# with the no-new-rotations rule, every arm simulated (no reuse: earlier CSVs predate the pass),
# identical arms sharing one simulation; then recount, floors and the figures (*_logical).
set -uo pipefail
P="/Users/cordputer/Documents/Claude/Projects/AQORN Revisions"; MAIN="$P/iceberg-compiler"
W="$MAIN/.claude/worktrees/density-logical"; LOG="$MAIN/logs/density_logical"; PY="$MAIN/venv/bin/python"
D=experiments/figures/density_ablation_sweep; M=experiments.figures.density_ablation_sweep
export OMP_NUM_THREADS=1 MPLBACKEND=Agg TQDM_DISABLE=1 PYTHONPATH=.
unset AQORN_LOGICAL_OPT
cd "$W"
echo "$(date '+%m-%d %H:%M') start  $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD), $(git status --porcelain --untracked-files=no | wc -l | tr -d ' ') dirty tracked files; logical optimisation $("$PY" -c 'from translator.logical_optimization import state; print(state())')"
"$PY" -m $M.density_ablation_sweep --standard --matched-patches --split-merging --share-identical \
    --jobs 12 --csv "$D/density_ablation_sweep_split_logical.csv" --no-show > "$LOG/sweep.log" 2>&1 \
  || { echo "$(date '+%H:%M') sweep FAILED, see sweep.log"; exit 1; }
grep -E "Wrote|Appended|failed|dropped" "$LOG/sweep.log"
CSV=$(grep -oE "Wrote [0-9]+ rows to new .*\.csv" "$LOG/sweep.log" | head -1 | sed -E 's/^Wrote [0-9]+ rows to new //')
[ -n "$CSV" ] && [ -f "$CSV" ] || { echo "no sweep CSV found in sweep.log"; exit 1; }
echo "$(date '+%H:%M') sweep done: $CSV"
"$PY" - "$CSV" <<'EOF' || { echo "summary FAILED"; exit 1; }
import sys, pandas as pd
m = pd.read_csv(sys.argv[1]); p = m[m.component == "packing"]
shared = p.shared_arms.fillna("").str.split(";").map(lambda x: len([s for s in x if s]))
print(f"{m.circuit_name.nunique()} circuits, {len(p)} density points; logical optimisation "
      f"{sorted(m.logical_optimization.astype(str).unique())}; shared arm-points {int(shared.sum())} "
      f"of {4 * len(p)} (the four arms after the baseline)")
EOF
for step in recount floors; do
  echo "$(date '+%H:%M') $step"
  "$PY" -m $M.density_references $step --matched "$CSV" --jobs 12 > "$LOG/$step.log" 2>&1 \
    || { echo "$(date '+%H:%M') $step FAILED"; exit 1; }
  grep "density_references\]" "$LOG/$step.log"
done
echo "$(date '+%H:%M') plot"
"$PY" -m $M.density_references plot --matched "$CSV" --output-dir "$P/results/figures" > "$LOG/plot.log" 2>&1 \
  || { echo "$(date '+%H:%M') plot FAILED"; exit 1; }
grep "density_references\]" "$LOG/plot.log"
# exactly the CSVs this run wrote: the sweep's, and each step's "wrote <path>.csv"
{ echo "$CSV"; grep -h "density_references\] wrote .*\.csv" "$LOG"/recount.log "$LOG"/floors.log \
    "$LOG"/plot.log | sed 's/.*wrote //'; } | while IFS= read -r src; do
  cp -n "$src" "$P/results/data/" && echo "copied $(basename "$src") to results/data/"
done
echo "$(date '+%m-%d %H:%M') done"
