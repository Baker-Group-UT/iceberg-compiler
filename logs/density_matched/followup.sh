#!/usr/bin/env bash
# After the matched density sweep: occupied-lane recount, TVD floors, the three figures.
set -uo pipefail
MAIN="/Users/cordputer/Documents/Claude/Projects/AQORN Revisions/iceberg-compiler"
W="$MAIN/.claude/worktrees/exact-patches"; LOG="$MAIN/logs/density_matched"
PY="$MAIN/venv/bin/python"; M=experiments.figures.density_ablation_sweep.density_references
export OMP_NUM_THREADS=1 MPLBACKEND=Agg TQDM_DISABLE=1 PYTHONPATH=.
cd "$W"
until grep -q "^finished:" "$LOG/RUN.txt"; do sleep 60; done
echo "$(date '+%H:%M') sweep finished"
if ! grep -q "density_ablation_sweep_matched.pdf" "$LOG/RUN.txt"; then
  echo "sweep did not write its figure; stopping"; exit 1
fi
"$PY" - <<'EOF' || { echo "matched CSV incomplete; stopping"; exit 1; }
import pandas as pd
d = pd.read_csv("experiments/figures/density_ablation_sweep/density_ablation_sweep_matched.csv")
n = d.circuit_name.nunique(); print(f"matched CSV: {n} circuits, {len(d)} rows")
assert n == 71, n
assert (d.packed_patches == d.n_codes).all()
EOF
run() { local step="$1"; shift; echo "$(date '+%H:%M') $step"; "$PY" -m $M "$step" "$@" > "$LOG/$step.log" 2>&1 \
  || { echo "$(date '+%H:%M') $step FAILED, see $LOG/$step.log"; exit 1; }; grep "density_references\]" "$LOG/$step.log"; }
run recount --jobs 12
run floors --jobs 12
run plot --output-dir "$MAIN/results/figures"
echo "$(date '+%H:%M') done"
