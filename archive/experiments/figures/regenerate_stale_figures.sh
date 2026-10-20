#!/bin/zsh
# Regenerate every figure whose cached CSV predates the transport-model fix round
# (abe4831) and/or the cost-model round (2ce3355), excluding the five measured-axis
# figures handled by regenerate_measured_axis.sh.
#
# Two phases, shaped around core usage on a 14-core machine:
#   A  serial figures (no parallel_map) run as a small concurrent pool -- they use the
#      cores the measured-axis chain's single-threaded stages leave idle.
#   B  parallel_map figures each saturate ~12 workers, so they wait for the
#      measured-axis chain to finish and then run one at a time.
#
# One figure failing must not kill the night: failures are recorded and reported at
# the end rather than aborting (no `set -e`).
#
# Run from the repo root:  ./experiments/figures/regenerate_stale_figures.sh <logdir>
set -u
export PYTHONPATH=.
PY=./venv/bin/python
LOGDIR=${1:-/tmp/regen_stale_figures}
mkdir -p "$LOGDIR"
FAILED=()

run() {
  local name=$1; shift
  echo "=== $name start $(date '+%H:%M:%S')"
  if $PY "$@" > "$LOGDIR/$name.log" 2>&1; then
    echo "=== $name done  $(date '+%H:%M:%S')"
  else
    echo "=== $name FAILED (see $LOGDIR/$name.log)"
    FAILED+=("$name")
  fi
}

echo "--- phase A: serial figures, pooled ---"
# Cheap, translation-only: minutes each.
run ablation_vs_k experiments/figures/ablation_vs_k/ablation_vs_k.py &
run merging_improvement_code_size experiments/figures/merging_improvement/merging_improvement_code_size.py &
run merging_improvement_code_size_placement experiments/figures/merging_improvement/merging_improvement_code_size_placement.py &
run 7_improvement_vs_immutable_fraction experiments/figures/7_improvement_vs_immutable_fraction/7_improvement_vs_immutable_fraction.py &
wait

# Simulation-heavy but serial: run the three concurrently so they share cores.
# --refresh: this one silently reuses its cached CSV otherwise.
run improvement_vs_random experiments/figures/improvement_vs_random/improvement_vs_random.py --refresh &
run improvement_vs_benchmark_locality experiments/figures/improvement_vs_benchmark_locality/improvement_vs_benchmark_locality.py &
run improvement_vs_locality_and_k experiments/figures/improvement_vs_locality_and_k/improvement_vs_locality_and_k.py &
wait

echo "--- waiting for the measured-axis chain to release the cores ---"
# NOTE: phase A's heavy serial batch above and this gate both block phase B. If the
# measured-axis chain finishes before the serial figures do, run phase B by hand --
# the sequencing here cost idle cores the first night it ran.
while pgrep -f "regenerate_measured_axis.sh" > /dev/null 2>&1; do
  sleep 60
done

echo "--- phase B: parallel figures, one at a time ---"
# The paper's headline benchmark first: its packing arm predates the cost-model round.
run merging_improvement_benchmark_stylized_simulated -m experiments.figures.merging_improvement.merging_improvement_benchmark_stylized_simulated --no-show
run lsr_tvd_vs_k experiments/figures/lsr_tvd_vs_k/lsr_tvd_vs_k.py
run 5_tvd_vs_accepted_shots experiments/figures/5_tvd_vs_accepted_shots/5_tvd_vs_accepted_shots.py
run 6_packing_density_tradeoff experiments/figures/6_packing_density_tradeoff/6_packing_density_tradeoff.py
run 4_single_patch_detection_vs_noise experiments/figures/4_single_patch_detection_vs_noise/4_single_patch_detection_vs_noise.py
run 8_optimality_vs_exhaustive experiments/figures/8_optimality_vs_exhaustive/8_optimality_vs_exhaustive.py

if (( ${#FAILED[@]} > 0 )); then
  echo "=== finished with failures: ${FAILED[*]}"
  exit 1
fi
echo "=== all stale figures regenerated"
