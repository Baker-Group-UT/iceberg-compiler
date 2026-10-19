#!/usr/bin/env bash
#
# run_overnight.sh -- the v2 revision queue, unattended.
#
# Modelled on experiments/figures/regenerate_submission.sh: one log per item, a failure is
# recorded and the queue moves on, items run one after another and each uses all JOBS
# workers. Order follows OVERNIGHT_EXPERIMENT_PLAN.md section 3: no-simulation items first,
# so they finish even if a later one breaks; 7c alone and single-process before any
# simulation, because it measures wall-clock time. The last item writes REVISION_RESULTS.md.
#
# Every simulated script appends rows as circuits finish and resumes by key (refusing a
# settings mismatch), so re-running this script after a crash continues where it stopped.
#
#   caffeinate -dimsu nohup bash experiments/revision/run_overnight.sh > logs/revision/queue.log 2>&1 &
#
set -uo pipefail
cd "$(dirname "$0")/../.."

JOBS="${JOBS:-12}"
LOGS="${LOGS:-logs/revision}"
PY="${PY:-./venv/bin/python}"
STATUS="$LOGS/STATUS.txt"
mkdir -p "$LOGS"

# One thread per worker: Aer's OpenMP would otherwise multiply into ~12 threads per
# process (measured 1,226% CPU at 3 workers), and process-level parallelism is the
# efficient one here (single-threaded density matrix: 35 s CPU vs 60 s multithreaded).
export OMP_NUM_THREADS=1
export TQDM_DISABLE=1
export MPLBACKEND=Agg
export PYTHONPATH=.

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

{
  echo "queue started $(stamp)   sha $(git rev-parse --short HEAD)   jobs $JOBS"
  echo "cost model version: $("$PY" -c 'from packing.version import COST_MODEL_VERSION; print(COST_MODEL_VERSION)')"
} >> "$STATUS"

run() {
  local label="$1"; shift
  local log="$LOGS/${label}.log"
  local started=$SECONDS
  echo "$(stamp)  START  $label" >> "$STATUS"
  if "$PY" "$@" > "$log" 2>&1; then
    echo "$(stamp)  ok     $label  $(( (SECONDS - started) / 60 ))m  $log" >> "$STATUS"
  else
    local code=$?
    echo "$(stamp)  FAIL   $label  $(( (SECONDS - started) / 60 ))m  exit $code  $log" >> "$STATUS"
    FAILED="${FAILED:-} $label"
  fi
}

# 1. Experiment7a: exhaustive optimality, translation-scored.
run 01_exp7a_exhaustive_translate -m experiments.revision.exp7_optimality.exhaustive_translate \
    --jobs "$JOBS"

# 2. Experiment7c: compile time. Single process, nothing else running.
run 02_exp7c_compile_time -m experiments.revision.exp7_optimality.compile_time

# 3. Experiment1a/1b: packer baselines, static; level-3 pre-optimised inputs.
run 03_exp1a_packer_static -m experiments.revision.exp1_packers.packer_static \
    --preopt o1 --n-codes 4 6 8 --jobs "$JOBS"
run 04_exp1b_packer_static_o3 -m experiments.revision.exp1_packers.packer_static \
    --preopt o3 --n-codes 4 --jobs "$JOBS"

# 4. Experiment3a: proxy vs original, static.
run 05_exp3a_proxy_static -m experiments.revision.exp3_proxy.proxy_static --jobs "$JOBS"

# 5. Experiment6a and 2a: no simulation.
run 06_exp6a_best_static_replot -m experiments.revision.exp6_selector.best_static_replot
run 07_exp2a_calibration_table -m experiments.revision.exp2_fixed_noise.calibration_table

# 6. Experiment7b: exhaustive optimality, simulated.
run 08_exp7b_exhaustive_simulate -m experiments.revision.exp7_optimality.exhaustive_simulate \
    --jobs "$JOBS" ${EXP7B_ARGS:-}

# 7. Experiment6b: AQORN at every static k; then the comparison with it; then 6c.
run 09_exp6b_aqorn_static_k -m experiments.revision.exp6_selector.aqorn_static_k --jobs "$JOBS"
run 10_exp6a_with_aqorn -m experiments.revision.exp6_selector.best_static_replot \
    --aqorn-csv results/revision/exp6/aqorn_static_k.csv
run 11_exp6c_noise_mismatch -m experiments.revision.exp6_selector.noise_mismatch --jobs "$JOBS"

# 8. Experiment1c: packers, simulated (sanity gate first; exits 3 if Fig 6 does not reproduce).
run 12_exp1c_packer_simulated -m experiments.revision.exp1_packers.packer_simulated \
    --jobs "$JOBS" ${EXP1C_ARGS:-}

# 9. Experiment2b + Experiment4: fixed physical noise, TVD statistics inline.
run 13_exp2b_fixed_noise_sweep -m experiments.revision.exp2_fixed_noise.fixed_noise_sweep \
    --jobs "$JOBS" ${EXP2B_ARGS:-}

# 10. Experiment3b: non-Clifford originals vs proxies (Aer density matrix).
run 14_exp3b_proxy_simulated -m experiments.revision.exp3_proxy.proxy_simulated \
    --tiers exact --jobs "$JOBS"

# Interim report, so something is on disk before the stretch items.
run 15_summarize_interim -m experiments.revision.summarize

# 11. Stretch: 7d (k = 4, and k = 2 at 7-8 qubits), 3b trajectories.
if [ "${SKIP_STRETCH:-0}" != "1" ]; then
  run 16_exp7d_k4_full -m experiments.revision.exp7_optimality.exhaustive_translate \
      --configs 5:4 --space full --random 20 --name exhaustive_translate_k4 --jobs "$JOBS"
  run 17_exp7d_tight -m experiments.revision.exp7_optimality.exhaustive_translate \
      --configs 6:4 7:2 8:2 --space tightest --random 20 --name exhaustive_translate_tight \
      --jobs "$JOBS"
  run 18a_exp7b_alpha1 -m experiments.revision.exp7_optimality.exhaustive_simulate \
      --alphas 1 --name exhaustive_simulate_alpha1 --jobs "$JOBS"
  run 18_exp3b_trajectories -m experiments.revision.exp3_proxy.proxy_simulated \
      --tiers trajectory --table proxy_simulated_trajectory --trajectory-shots 3000 \
      --jobs "$JOBS"
fi

# 12. The morning report.
run 19_summarize -m experiments.revision.summarize

echo "queue finished $(stamp)   failed:${FAILED:- none}" >> "$STATUS"
