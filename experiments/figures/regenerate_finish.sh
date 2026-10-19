#!/usr/bin/env bash
#
# regenerate_finish.sh -- the tail of regenerate_submission.sh, re-ordered under deadline.
#
# `regenerate_submission.sh` ran 6, 9, merging and improvement_vs_random to completion and
# was killed partway through 1_best_k_phase_diagram, whose python was left alive. This picks
# up from there with two changes:
#
#   * improvement_vs_noise_and_k moves from third to LAST. It is 420 work items and 4200
#     simulations against roughly 1850 for density_ablation_sweep, so on the old order it
#     would have consumed the whole remaining budget and left the two cheaper figures
#     unbuilt.
#   * it runs at reduced granularity -- see IVNK_ARGS. It is also the only figure here that
#     is checkpointed, so if the deadline arrives mid-run its completed items survive and
#     `--from-checkpoint` will plot whatever finished.
#
# Order is now cheapest-first among what matters, so a figure that overruns costs only
# itself.
#
#   PYTHONPATH=. bash experiments/figures/regenerate_finish.sh
#
set -uo pipefail

JOBS="${JOBS:-12}"
LOGS="${LOGS:-logs/regenerate}"
PY="${PY:-./venv/bin/python}"
WAIT_PID="${WAIT_PID:-}"
mkdir -p "$LOGS"

run() {
  local label="$1"; shift
  local log="$LOGS/${label}.log"
  printf '%-40s ' "$label"
  local started=$SECONDS
  if PYTHONPATH=. "$PY" "$@" > "$log" 2>&1; then
    printf 'ok    %5dm  %s\n' $(( (SECONDS - started) / 60 )) "$log"
  else
    printf 'FAIL  %5dm  %s\n' $(( (SECONDS - started) / 60 )) "$log"
    FAILED="${FAILED:-} $label"
  fi
}

# The box is saturated while 1_best_k finishes; starting here would only timeshare with it.
if [ -n "$WAIT_PID" ]; then
  echo "waiting for 1_best_k_phase_diagram (pid $WAIT_PID) to finish..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
  echo "1_best_k_phase_diagram exited at $(date '+%H:%M')"
fi

echo "jobs: $JOBS   logs: $LOGS   started $(date '+%H:%M')"
echo

run resource_selection_headroom \
    -m experiments.figures.resource_selection_headroom.resource_selection_headroom \
    --jobs "$JOBS" --persist

# Newly parallel: its per-circuit loop is a module-level worker now, verified row-for-row
# identical to the serial path. Writes CSV after each size group, so a kill keeps what
# finished and `--from-csv` replots it.
run density_ablation_sweep \
    -m experiments.figures.density_ablation_sweep.density_ablation_sweep \
    --jobs "$JOBS"

# Reduced granularity. The grid *shape* is preserved -- all five regime columns and all
# five yield columns -- because the columns are the claim. What shrinks is the corpus
# (42 circuits -> 12) and the k rows (5 -> 3), which cuts points from 2100 to about 360
# while leaving both axes at full resolution. Drop --max-circuits first if there is time.
IVNK_ARGS=(--max-circuits 12 --k-values 2 4 8)
run improvement_vs_noise_and_k \
    -m experiments.figures.improvement_vs_noise_and_k.improvement_vs_noise_and_k \
    --jobs "$JOBS" --persist --fresh "${IVNK_ARGS[@]}"

echo
if [ -n "${FAILED:-}" ]; then
  echo "FAILED:${FAILED}"
  exit 1
fi
echo "done at $(date '+%H:%M')"
