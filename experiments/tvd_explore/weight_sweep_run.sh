#!/bin/zsh
# Transversal (T) weight sweep: plan, then Fig 6 with Fig 6's seeds, per weight. Skips files that exist.
cd "/Users/cordputer/Documents/Claude/Projects/AQORN Revisions/iceberg-compiler/.claude/worktrees/tvd-explore"
PY=../../../venv/bin/python
export PYTHONPATH=. TQDM_DISABLE=1 MPLBACKEND=Agg
for w in 3 5 7 8 9 12 15 20; do
  if [ -f results/tvd_explore/plans_w${w}T.csv ]; then echo "plans_w${w}T exists, skip"; continue; fi
  echo "$(date +%T) plan w${w}T"
  $PY -m experiments.tvd_explore.plan_variant w${w}T --weight $w --switch TRANSVERSAL_INTRA_TWO_QUBIT \
      --rates results/tvd_explore/gadget_rates_transversal.csv --jobs 12 > logs/tvd_explore/plan_w${w}T.log 2>&1 || echo "plan w${w}T FAILED"
done
for w in 3 5 7 8 9 12 15 20 30; do
  if [ -f results/tvd_explore/ablation_variant_w${w}T.csv ]; then echo "ablation w${w}T exists, skip"; continue; fi
  echo "$(date +%T) fig6 w${w}T"
  $PY experiments/tvd_explore/ablation_variant.py --silent-weight $w --variant-csv results/tvd_explore/plans_w${w}T.csv \
      --label w${w}T --switch TRANSVERSAL_INTRA_TWO_QUBIT --jobs 12 --output-dir results/tvd_explore/figures \
      > logs/tvd_explore/ablation_w${w}T.log 2>&1 || echo "fig6 w${w}T FAILED"
done
echo "$(date +%T) sweep done"
