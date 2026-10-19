"""
version.py -- one integer saying "results computed before this are not comparable".

Every figure in this repo caches its simulation output to CSV, because the runs are hours
long. That cache is only safe if a change to the *planner* invalidates it, and nothing
made that happen: `improvement_vs_noise_and_k`'s checkpoint fingerprint covered the shot
budget and the noise flags but not the objective, so a rerun after a cost-model fix would
have silently reused placements the fix was meant to change and produced a CSV mixing two
code versions.

**Bump `COST_MODEL_VERSION` whenever a change can alter a chosen placement.** That is a
judgement, which is why this is a hand-maintained integer rather than a hash of the source:
hashing churns on comment edits and would invalidate a fortnight of simulation over a typo,
while a human can tell the difference between rewording a docstring and rewriting `price`.
The cost of forgetting is a stale figure, so err towards bumping.

Not the same thing as a git commit. The commit says *what* the code was; this says *whether
the numbers still mean anything*. Most commits do not move it.

Changelog
---------

3   2026-07-31  Movement pricing made placement-aware, and `verify` given a real criterion.
                Four changes, each of which can move a placement:
                  * `plan.measured_cost` prices movement and idling on the translated
                    circuit's real schedule when given a noise profile, where it used to
                    count gates and depth only -- so `verify`'s "never worse than naive"
                    guarantee stopped being vacuous. It ranks on expected error alone,
                    which makes `depth_weight` inert under a profile.
                  * `cost.price` no longer clamps movement to `MOVEMENT_SHARE_CEILING *
                    work`. The clamp removed the term's *gradient* wherever it bound (100%
                    of calls at `distance_scalar = 128`), leaving the search blind to
                    movement.
                  * `cost.price` charges per-pair patch distance via
                    `tables.patch_hop_matrix` instead of the placement-blind
                    `mean_patch_hops`, so a placement that keeps interacting qubits in
                    neighbouring patches is priced differently from one that scatters them.
                  * `plan_packing` verifies against the profile the caller *handed* it
                    rather than the `NoiseProfile()` default, so search and check agree.

2   (implicit)  The `na-packing-costs` pricing: real-cell `mean_patch_hops`, grouped-trip
                movement, carriage-column swap pricing, `distance_scalar` applied once.
                Never recorded as a version; every CSV predating version 3 without a
                `cost_model_version` column is some state at or before this.

1   (implicit)  Pre-transport pricing: unit-pitch hops, `distance_scalar` applied twice,
                four round trips per crossing.
"""

from __future__ import annotations

#: Bump on any change that can alter a chosen placement. See the changelog above.
COST_MODEL_VERSION = 3

__all__ = ["COST_MODEL_VERSION"]
