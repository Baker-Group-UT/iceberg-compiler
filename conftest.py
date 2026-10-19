"""
Session-wide test setup.

Forces Matplotlib to a non-interactive backend before any test module is imported. A
test run must never open a window: under an interactive backend `plt.show()` blocks
until the window is dismissed, so a stray module-level plot turns `pytest` into a
prompt. That is not hypothetical -- `dag/circuit_dag_shor_test.py` matched pytest's
default `*_test.py` glob, drew four figures at import, and contributed zero tests, so
nothing in the collected count hinted at it.

`pytest.ini` now confines collection to `tests/`, which is the real fix. This is the
backstop for anyone running `pytest` against a path directly.
"""

from __future__ import annotations

import os

os.environ["MPLBACKEND"] = "Agg"

# The suite pins the compiler's behaviour before the logical optimisation pass
# (translator/logical_optimization.py): exact gate counts, recorded plans, hand-built
# programs whose gates are the point of the test. So it runs with the pass off unless
# asked (AQORN_LOGICAL_OPT=1 pytest ...); tests/test_logical_optimization.py turns it on.
os.environ.setdefault("AQORN_LOGICAL_OPT", "0")
