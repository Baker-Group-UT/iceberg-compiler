#!/bin/zsh
# Regenerate every figure that moved from the nominal idle/distance ratio axis to the
# measured-regime axis (measured_regime / measured_noise_split), in dependency order:
# 10_winning_patch_count reads 1_best_k_phase_diagram's CSV, so it runs after it.
#
# Run from the repo root:  ./experiments/figures/regenerate_measured_axis.sh
set -e
export PYTHONPATH=.
PY=./venv/bin/python

echo "=== depth_vs_gates_tradeoff (translation-only) ==="
$PY experiments/figures/depth_vs_gates_tradeoff/depth_vs_gates_tradeoff.py --refresh

echo "=== 2_tvd_vs_noise_ratio_by_k ==="
$PY experiments/figures/2_tvd_vs_noise_ratio_by_k/2_tvd_vs_noise_ratio_by_k.py

echo "=== 1_best_k_phase_diagram ==="
$PY experiments/figures/1_best_k_phase_diagram/1_best_k_phase_diagram.py

echo "=== 10_winning_patch_count_vs_noise_ratio (replots figure 1) ==="
$PY experiments/figures/10_winning_patch_count_vs_noise_ratio/10_winning_patch_count_vs_noise_ratio.py

echo "=== code_size_crossover ==="
$PY experiments/figures/code_size_crossover/code_size_crossover.py

echo "=== all measured-axis figures regenerated ==="
