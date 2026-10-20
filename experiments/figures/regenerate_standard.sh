#!/usr/bin/env bash
#
# regenerate_standard.sh -- the paper's planner-dependent figures under the standard configuration.
#
# Every figure here runs in its opt-in `--standard` mode (`packing.standard`, since 2026-10-07):
# the silent-aware planner (weight 10 at k = 2 with the transversal in-patch CZ / XCX, weight 5
# above), every arm layer-sorted, the standard resource selector, and the silent logical error
# per kept shot recorded next to every selection rate (frames; the accuracy measure that works
# at widths where TVD saturates). Each writes `<name>_standard[_vN]` CSVs and figures, so
# nothing from the submitted runs is overwritten. Fig 6 under the standard is separate
# (`experiments/figures/ablation_standard`, done).
#
# Only the three figures `paper/text/evaluation.tex` includes are run. `code_size_tradeoff`
# and `locality_and_k_quad` are commented out of the paper (and the latter's live analogue,
# improvement_vs_noise_and_k, is a ~14 h sweep that has never finished).
#
# Cheapest first, so a script-level breakage surfaces in minutes rather than after the long one;
# a failure is recorded, not fatal.
#
#   PYTHONPATH=. nohup bash experiments/figures/regenerate_standard.sh > logs/regenerate_standard/RUN.txt 2>&1 &
#
set -uo pipefail

JOBS="${JOBS:-12}"
LOGS="${LOGS:-logs/regenerate_standard}"
PY="${PY:-./venv/bin/python}"
mkdir -p "$LOGS"
export OMP_NUM_THREADS=1 MPLBACKEND=Agg TQDM_DISABLE=1

run() {
  local label="$1"; shift
  local log="$LOGS/${label}.log"
  printf '%s  %-38s ' "$(date '+%H:%M')" "$label"
  local started=$SECONDS
  if PYTHONPATH=. "$PY" "$@" > "$log" 2>&1; then
    printf 'ok    %5dm  %s\n' $(( (SECONDS - started) / 60 )) "$log"
  else
    printf 'FAIL  %5dm  %s\n' $(( (SECONDS - started) / 60 )) "$log"
    FAILED="${FAILED:-} $label"
  fi
}

echo "git: $(git rev-parse --short HEAD) $(git status --porcelain --untracked-files=no | wc -l | tr -d ' ') dirty tracked files"
echo "standard: $(PYTHONPATH=. "$PY" -c 'from packing.standard import STANDARD; print(STANDARD)')"
echo "jobs: $JOBS   logs: $LOGS"
echo

run resource_selection_headroom_standard \
    -m experiments.figures.resource_selection_headroom.resource_selection_headroom \
    --standard --jobs "$JOBS" --persist

run 1_best_k_phase_diagram_standard \
    -m experiments.figures.1_best_k_phase_diagram.1_best_k_phase_diagram \
    --standard --jobs "$JOBS" --persist

run density_ablation_sweep_standard \
    -m experiments.figures.density_ablation_sweep.density_ablation_sweep \
    --standard --jobs "$JOBS"

echo
date '+%H:%M  done'
if [ -n "${FAILED:-}" ]; then
  echo "FAILED:${FAILED}"
  exit 1
fi
