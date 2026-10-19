#!/usr/bin/env bash
#
# run_ivnk_after_density.sh -- the last figure, launched once density_ablation_sweep exits.
#
# Split out of regenerate_finish.sh rather than left inside it. Bash reads a script lazily
# from a byte offset while executing it, so editing the runner mid-flight can make it
# resume at the wrong place; `density_ablation_sweep`'s `large` group turned out slower
# than projected and its budget needed retuning, which meant either editing a running
# script or moving the step out. This is the safe half of that choice.
#
# Granularity is cut further than regenerate_finish.sh had it -- 8 circuits rather than 12
# -- because `large` is now expected to finish between 04:45 and 06:20, and the deadline is
# 07:50. The grid *shape* is untouched: all five regime columns and all five yield columns
# survive, because the columns are the claim. What shrinks is the corpus and the k rows.
#
# This figure is checkpointed, so if the deadline arrives mid-run the completed items are on
# disk and `--from-checkpoint` plots whatever finished.
#
#   PYTHONPATH=. WAIT_PID=<density pid> bash experiments/figures/run_ivnk_after_density.sh
#
set -uo pipefail

JOBS="${JOBS:-12}"
LOG="${LOG:-logs/regenerate/improvement_vs_noise_and_k.log}"
PY="${PY:-./venv/bin/python}"
WAIT_PID="${WAIT_PID:-}"
mkdir -p "$(dirname "$LOG")"

if [ -n "$WAIT_PID" ]; then
  echo "waiting for density_ablation_sweep (pid $WAIT_PID)..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  echo "density_ablation_sweep exited at $(date '+%H:%M')"
fi

# density writes its own figure at the end of its process, so give it a moment to flush
# before starting something that will take every core.
sleep 5

echo "improvement_vs_noise_and_k starting $(date '+%H:%M')  jobs=$JOBS"
started=$SECONDS
if PYTHONPATH=. "$PY" \
    -m experiments.figures.improvement_vs_noise_and_k.improvement_vs_noise_and_k \
    --jobs "$JOBS" --persist --fresh \
    --max-circuits 8 --k-values 2 4 8 \
    > "$LOG" 2>&1; then
  printf 'improvement_vs_noise_and_k  ok    %5dm  %s\n' $(( (SECONDS - started) / 60 )) "$LOG"
else
  printf 'improvement_vs_noise_and_k  FAIL  %5dm  %s\n' $(( (SECONDS - started) / 60 )) "$LOG"
  echo "checkpoints survive; replot with --from-checkpoint"
  exit 1
fi
echo "done at $(date '+%H:%M')"
