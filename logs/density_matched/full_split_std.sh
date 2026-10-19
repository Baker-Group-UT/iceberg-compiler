#!/usr/bin/env bash
# Full split-merging density ablation under the standard (solo_hh_aware on): medium + large, then
# combine with the finished small group, recount, floors, and every figure.
set -uo pipefail
MAIN="/Users/cordputer/Documents/Claude/Projects/AQORN Revisions/iceberg-compiler"
W="$MAIN/.claude/worktrees/exact-patches"; LOG="$MAIN/logs/density_matched"; PY="$MAIN/venv/bin/python"
D=experiments/figures/density_ablation_sweep; M=experiments.figures.density_ablation_sweep.density_references
export OMP_NUM_THREADS=1 MPLBACKEND=Agg TQDM_DISABLE=1 PYTHONPATH=.
cd "$W"
echo "$(date '+%m-%d %H:%M') start  $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD), $(git status --porcelain --untracked-files=no | wc -l | tr -d ' ') dirty tracked files"
"$PY" -m experiments.figures.density_ablation_sweep.density_ablation_sweep --standard --matched-patches \
    --split-merging --reuse-arms "$D/density_ablation_sweep_matched.csv" \
    --simulate baseline h_only h_merge_1q packed --groups medium large --jobs 12 \
    --csv "$D/density_ablation_sweep_split_std_rest.csv" --no-show > "$LOG/split_std_rest.log" 2>&1 \
  || { echo "$(date '+%H:%M') sweep FAILED, see split_std_rest.log"; exit 1; }
grep -E "Wrote|Appended|failed|dropped" "$LOG/split_std_rest.log"
echo "$(date '+%H:%M') sweep done"
"$PY" - <<'EOF' || { echo "combine FAILED"; exit 1; }
import pandas as pd
D = "experiments/figures/density_ablation_sweep/"
small = pd.read_csv(D + "density_ablation_sweep_split_std_small.csv")
rest = pd.read_csv(D + "density_ablation_sweep_split_std_rest.csv")
assert list(small.columns) == list(rest.columns), "schemas differ"
both = pd.concat([small, rest], ignore_index=True)
n = both.circuit_name.nunique()
print(f"combined: {n} circuits, {len(both)} rows; copied arms {both.drop_duplicates('circuit_name').copied_arms.value_counts().to_dict()}")
assert n == 71, n
both.to_csv(D + "density_ablation_sweep_split_std.csv", index=False)
EOF
CSV="$D/density_ablation_sweep_split_std.csv"
for step in recount floors; do
  echo "$(date '+%H:%M') $step"
  "$PY" -m $M $step --matched "$CSV" --jobs 12 > "$LOG/split_std_$step.log" 2>&1 \
    || { echo "$(date '+%H:%M') $step FAILED"; exit 1; }
  grep "density_references\]" "$LOG/split_std_$step.log"
done
echo "$(date '+%H:%M') plot"
"$PY" -m $M plot --matched "$CSV" --output-dir "$MAIN/results/figures" > "$LOG/split_std_plot.log" 2>&1 \
  || { echo "$(date '+%H:%M') plot FAILED"; exit 1; }
grep "density_references\]" "$LOG/split_std_plot.log"
for f in density_ablation_sweep_split_std density_ablation_sweep_split_std_recount \
         density_ablation_sweep_split_std_floors density_ablation_sweep_split_std_summary; do
  [ -e "$MAIN/results/data/$f.csv" ] || cp "$D/$f.csv" "$MAIN/results/data/$f.csv"
done
echo "$(date '+%m-%d %H:%M') done"
