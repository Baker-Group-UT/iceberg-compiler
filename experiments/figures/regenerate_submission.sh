#!/usr/bin/env bash
#
# regenerate_submission.sh -- rerun every figure the paper actually uses, in order.
#
# Replaces regenerate_stale_figures.sh and regenerate_measured_axis.sh, both of which
# referenced figures now under archive/ and would fail on the first line.
#
# Ordered cheapest first, so a script-level breakage surfaces in seconds rather than after
# the overnight run. Each figure logs to its own file and a failure is recorded rather than
# fatal -- one bad figure should not cost the six that would have worked.
#
# Everything here is regenerated for two reasons. Cost model version 3 changed what the
# planner chooses (see packing/version.py), and target_selection_rate moved 0.12 -> 0.06
# (see SimSettings), which moves alpha and therefore every simulated number.
#
# One entry needs neither and is kept only so its CSV carries a provenance stamp:
#
#   4_single_patch_detection_vs_noise   never calls the planner, and never calls
#                                       `calibrate` either -- its shot budget is a fixed
#                                       raw count because the x axis *is* the noise scale.
#                                       Both changes are provable no-ops. Already at v3;
#                                       safe to skip if the queue needs the hours.
#
# 9_code_size_tradeoff simulates nothing at all -- it replots 6's CSV -- so it costs
# seconds but must run after 6.
#
#   PYTHONPATH=. bash experiments/figures/regenerate_submission.sh
#   JOBS=8 bash experiments/figures/regenerate_submission.sh
#
set -uo pipefail
# The submitted figures predate the logical optimisation pass (translator/
# logical_optimization.py, on by default since 2026-10-09); reproduce them without it.
export AQORN_LOGICAL_OPT=0

JOBS="${JOBS:-12}"
LOGS="${LOGS:-logs/regenerate}"
PY="${PY:-./venv/bin/python}"
mkdir -p "$LOGS"

run() {
  local label="$1"; shift
  local log="$LOGS/${label}.log"
  printf '%-46s ' "$label"
  local started=$SECONDS
  if PYTHONPATH=. "$PY" "$@" > "$log" 2>&1; then
    printf 'ok    %5dm  %s\n' $(( (SECONDS - started) / 60 )) "$log"
  else
    printf 'FAIL  %5dm  %s\n' $(( (SECONDS - started) / 60 )) "$log"
    FAILED="${FAILED:-} $label"
  fi
}

echo "cost model version: $(PYTHONPATH=. "$PY" -c 'from packing.version import COST_MODEL_VERSION; print(COST_MODEL_VERSION)')"
echo "jobs: $JOBS   logs: $LOGS"
echo

# NOT RUN: compiler_stack. The paper switched to the hand-drawn Inkscape figure at
# paper/figures/diagrams/compiler_pipeline.pdf on 31 July, so regenerating the
# matplotlib version would produce a file nothing includes. The script is kept under
# experiments/figures/compiler_stack/ in case the drawn version needs replacing.

# --- analytic, seconds. 9 reads 6's CSV, so 6 runs first even though its own figure is
# --- not in the paper. 6 never calls the planner, so its numbers are unchanged by version 3.
run 6_packing_density_tradeoff \
    -m experiments.figures.6_packing_density_tradeoff.6_packing_density_tradeoff \
    --jobs "$JOBS" --persist

run 9_code_size_tradeoff \
    -m experiments.figures.9_code_size_tradeoff.9_code_size_tradeoff

# --- simulated, no planner ---------------------------------------------------------
# Off by default: see the header. It calibrates nothing and plans nothing, so neither cost
# model version 3 nor the selection-rate change can move it, and its CSV is already at v3.
# SKIP_SINGLE_PATCH=0 puts it back if you want the whole set rebuilt from one run.
if [ "${SKIP_SINGLE_PATCH:-1}" = "0" ]; then
  run 4_single_patch_detection_vs_noise \
      -m experiments.figures.4_single_patch_detection_vs_noise.4_single_patch_detection_vs_noise \
      --jobs "$JOBS" --persist
else
  printf '%-46s %s\n' 4_single_patch_detection_vs_noise "skipped (unaffected; SKIP_SINGLE_PATCH=0 to run)"
fi

# --- planner, cheap to moderate ----------------------------------------------------
run merging_improvement_benchmark_stylized_simulated \
    -m experiments.figures.merging_improvement.merging_improvement_benchmark_stylized_simulated \
    --jobs "$JOBS"

# `--refresh` is load-bearing: without it this figure silently reuses its cached CSV and
# reports success in zero seconds, replotting pre-fix data. It is one of the figures whose
# placements DO move with the cost model, so reusing the cache defeats the whole rerun.
run improvement_vs_random \
    -m experiments.figures.improvement_vs_random.improvement_vs_random \
    --jobs "$JOBS" --persist --refresh

run 1_best_k_phase_diagram \
    -m experiments.figures.1_best_k_phase_diagram.1_best_k_phase_diagram \
    --jobs "$JOBS" --persist

run resource_selection_headroom \
    -m experiments.figures.resource_selection_headroom.resource_selection_headroom \
    --jobs "$JOBS" --persist

# --- the long one. Checkpointed, so a kill resumes rather than restarting. ----------
run improvement_vs_noise_and_k \
    -m experiments.figures.improvement_vs_noise_and_k.improvement_vs_noise_and_k \
    --jobs "$JOBS" --persist --fresh

# --- last because it is the longest, not because it is serial: its per-circuit loop is now
# --- a module-level worker (`_run_one_circuit`) that `parallel_map` can reach. Verified
# --- identical to the serial path row for row on a 6-circuit corpus. It appends to CSV
# --- after each suite (small, then medium, then large), so a kill leaves the completed
# --- suites on disk and `--from-csv` replots them without resimulating.
run density_ablation_sweep \
    -m experiments.figures.density_ablation_sweep.density_ablation_sweep \
    --jobs "$JOBS"

echo
if [ -n "${FAILED:-}" ]; then
  echo "FAILED:${FAILED}"
  echo "Read the log named above before rerunning -- a figure that fails on its own script"
  echo "will fail the same way again."
  exit 1
fi
echo "all figures regenerated. Confirm with:"
echo "  PYTHONPATH=. $PY experiments/check_stale_data.py --stale-only"
