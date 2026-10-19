# Are LSR and TVD always anticorrelated? 2026-10-08

**No.** They are anticorrelated along the noise axis and for any change that removes errors, and
they trade against each other for any change that moves work between in-patch and cross-patch
gadgets -- which is what placement, the silent-error weight, the code size and the patch density
all do.

## Why: two error channels

- **LSR counts shots with no detected error:** LSR ~ exp(-alpha E_detected). Across Fig 6's 213
  arms the change in ln LSR follows the schedule's detected-error change with r = 0.95
  (`reports/why_aqorn_works.md`).
- **TVD measures the errors the code did not detect,** in the kept shots: within each circuit,
  TVD ~ 0.82 x silent error per kept shot (R^2 = 0.87 over Fig 6's arms). The sampling floor
  (fewer kept shots, noisier histogram) adds nothing at Fig 6's shot counts: of the mean
  baseline -> AQORN TVD change (-0.096), the floor term accounts for -0.001.
- **Silent errors come from specific gadgets:** in-patch CX / CZ (0.6-0.8 p2 per gadget),
  single-lane H (0.82 p2), rotations (0.42 p2) and phase gates (0.2 p2), at first order; cross-patch
  gadgets and whole-patch macros leak none. At high noise, coincidences of detected errors add a
  second-order part that grows with E_detected.

So a change that **removes** gadgets lowers both channels (LSR up, TVD down); **scaling the
noise** moves both together (LSR down, TVD up); but a change that **replaces** an in-patch gadget
with a cross-patch one (more movement, more detected error, no silent error) lowers TVD *and*
LSR, and the reverse raises both.

## Evidence: 6,025 paired comparisons (same circuit, same noise, same shots)

`experiments/insights/lsr_tvd.py`; a pair counts when both LSR and the error moved by more than 3%
(relative). Error = TVD where it resolves, else silent error per kept shot (frames).

| change | pairs moved | both better | shots for accuracy | accuracy for shots | both worse | Spearman |
|---|---|---|---|---|---|---|
| H commutation (Fig 6, standard) | 35 | **100%** | 0 | 0 | 0 | -0.89 |
| merging (Fig 6, standard) | 37 | **100%** | 0 | 0 | 0 | -0.74 |
| packing, submitted planner (Fig 6) | 19 | 68% | 16% | 16% | 0 | -0.07 |
| packing, standard planner (Fig 6) | 43 | 49% | **42%** | 5% | 5% | +0.11 |
| all three stages, standard | 61 | 74% | 23% | 2% | 2% | -0.59 |
| v4 -> w10T planner (packed arm) | 47 | 51% | **45%** | 0 | 4% | +0.05 |
| silent weight up, k = 2 (W 3 -> 30) | 45 | 11% | **64%** | 2% | 22% | +0.10 |
| silent weight up, k = 4 (frames) | 93 | 22% | **57%** | 4% | 17% | +0.17 |
| AQORN vs naive packing, k = 4 (frames) | 366 | 71% | 26% | 0 | 3% | -0.43 |
| larger code, AQORN at each k (phase diagram) | 380 | 6% | 2% | **52%** | 41% | -0.30 |
| larger code, naive (Fig 8) | 254 | 3% | 2% | **65%** | 29% | -0.52 |
| AQORN selector vs naive [[4,2,2]] (Fig 8) | 58 | 55% | 5% | 36% | 3% | +0.05 |
| more, emptier patches (density, baseline) | 204 | 1% | **86%** | 0 | 13% | -0.14 |
| noise scale alpha up (frames, 6 circuits x 2 arms) | all | -- | -- | -- | **100%** | (monotone) |

(Spearman is between the changes in ln LSR and ln error across a row's pairs: negative means
anticorrelated. "Both better" is the anticorrelated case in level terms -- LSR up, TVD down.)

Reading it:
- **The passes that remove gadgets** -- H commutation and merging -- improve both on every pair:
  a removed Hadamard or a fused macro takes its detected and its silent error with it.
- **Placement trades.** Under the submitted planner, packing mostly improved both; under the
  standard (silent-aware) planner 42% of moving circuits trade shots for accuracy -- by design: it
  moves in-patch CXs across patches. Raising the silent weight is almost purely that trade.
- **Code size trades the other way:** larger patches mean fewer patches (less movement, idling,
  encode / decode: more shots) but more in-patch gadgets (more silent error per kept shot).
- **Spreading over more, emptier patches** is the mirror image: fewer in-patch gadgets (more
  accurate) at the cost of more patches to encode, idle and move between (fewer shots).
- **The noise axis is the only place they are tied:** for one placement, raising alpha lowers LSR
  and raises silent error per kept shot monotonically (`insights_lsr_tvd.pdf`, right). On the six
  circuits swept, AQORN's placement is better on both at every alpha (0.25x to 4x Fig 6's).

## For the paper

- Report both metrics; an LSR gain does not certify accuracy, and vice versa. Fig 6's GHZ, cat
  and W-state bars (LSR below zero, TVD well above it) are this trade, not noise.
- The intuition "post-selection keeps the good shots, so more shots means a better answer" holds
  only when the change removes errors. Changes that convert silent errors into detected ones (or
  back) move the two in the same direction.
- The resource-selection results follow: the k = 2 standard is the accuracy winner, larger codes
  the shots winner (`reports/figures_standard.md`).

Data: results/insights/lsr_tvd_pairs.csv (one row per pair, with its quadrant),
lsr_tvd_alpha_sweep.csv. Figure: results/figures/insights_lsr_tvd.pdf. Re-run:
`PYTHONPATH=. venv/bin/python -m experiments.insights.lsr_tvd --alpha-sweep` (a few minutes; the
pairs come from the CSVs already on disk).
