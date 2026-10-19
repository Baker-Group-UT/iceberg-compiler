---
name: figure
description: Build a plotting script for a new paper figure — one directory per figure under experiments/figures/, computing its data, caching it to CSV, and emitting SVG plus PDF in the project's house style. Use when the user types /figure, or describes a plot they want by naming its axes, the comparison it should make, or the claim it needs to support. Also use when they ask for a new plot, chart, or figure for the paper, or want an existing figure regenerated or restyled. Prefer this over writing an ad-hoc matplotlib snippet, since figures in a paper need consistent styling, a reproducible script, and a cached data path.
---

# Paper figure

Create a script that produces one figure, in the house style, reproducibly. The deliverable is a
committed script that can be re-run — not a one-off plot — because every figure in a paper gets
regenerated at least once, usually the week before a deadline when someone asks for one more
series.

## Gather the spec

Get these before writing anything. Ask for whatever's missing in one message:

- **What the figure has to show** — the claim it supports, not just the quantities. This decides
  chart type, and a plot that doesn't land a claim is wasted space in a paper.
- **x axis** — quantity, units, scale (linear/log), and range if it matters.
- **y axis** — same.
- **Series** — what's being compared, and which is the baseline.
- **Where the data comes from** — an existing CSV, an experiment in this repo to run, or something
  that doesn't exist yet. If it needs a new experiment, say so before writing the plotting code.
- **Deviations from house style** — anything this figure needs that the shared style doesn't give.
- **Single column or full width** — determines the canvas.

## Load the style

Read two files at the project root before writing the script: `docs/figure-style.md` for the visual
spec, and `docs/notation.md` for symbols and preferred terms. Axis labels, legend entries, and
annotations are notation, so a figure that writes a quantity differently from the prose creates
exactly the mismatch readers notice. Check the open-questions section too — don't pick a side on
terminology the project has deliberately parked.

Have the script load the shared matplotlib style rather than setting rcParams inline:

```python
from pathlib import Path
import matplotlib.pyplot as plt

STYLE = Path(__file__).resolve().parents[2] / "shared" / "aqorn.mplstyle"
plt.style.use(STYLE)
```

Inline `rcParams` in a figure script is how the figures in a paper end up with three different
tick sizes. If this figure genuinely needs to deviate, override the specific key after loading the
style and leave a comment saying why.

Older scripts in `experiments/figures/` still carry their own `rcParams` blocks with Helvetica at
16 pt. Don't copy them — they predate the shared style, which is Times at a 10/9/8 ladder.

The style handles two things automatically, so don't reimplement them:

- **Redundant encoding.** Colour, marker, and line style advance together, so series stay
  distinguishable in greyscale. Just plot series in order and don't pass explicit colours.
- **Final-size canvas.** The default figsize is exactly one column (3.487 in), with constrained
  layout and a standard bounding box. For a full-width `figure*`, set `figsize` to 7.14 in wide and
  leave the rest alone.

## Conventions the figure has to follow

These come from `docs/figure-style.md`; the ones that get forgotten most often:

- **y starts at zero** unless the user asks otherwise or the axis is log. Log excludes zero, so it's
  the exception rather than a contradiction.
- **Error bars are ±1 standard deviation**, and the caption says what the spread is over and how
  many samples. Shaded band at `alpha=0.15` for continuous x; thin error bars with `capsize=2` for
  categorical. No violins.
- **Percent is the default** for improvements, and the figure must agree with the prose — don't put
  `1.75×` in a plot whose caption says 75%.
- **Units in the axis label** for a single quantity; in the legend entries when series carry
  different units. Dimensionless quantities (LER, selection rate, TVD, ratios) get no units at all.

## Circuits

A figure of a quantum circuit is drawn by qiskit, not by hand:

```python
import matplotlib.pyplot as plt
plt.style.use(STYLE)          # before the drawer — this is load-bearing
circuit.draw(output="mpl", style="bw", filename=out / f"{name}.svg")
```

Loading the style first is required, not tidiness: matplotlib's default `svg.fonttype` outlines text
into paths, so drawing without it yields an SVG with no editable text and every gate label arrives in
Inkscape as polygons. Pass `style={"name": "bw", "fontsize": 9, "subfontsize": 8}` when a circuit
needs smaller type to fit a column.

## Naming

Name everything after what the figure captures, not how it was made: `tvd_vs_merge_savings`,
`work_per_code`. That name is the directory, the script, and both output files.

```
experiments/figures/<name>/<name>.py
```

One directory per figure, matching the existing layout.

## Script requirements

Build around a compute → cache → plot split, so restyling never means recomputing. This matters
in practice: the simulation runs are slow, and figure iteration is fast.

- `argparse` CLI with `--from-csv PATH` to redraw from cached data without recomputing, and
  `--output-dir` to override where output lands.
- Write the raw and, where it applies, aggregated data to CSV. Keep the cache next to the script
  by default. Only copy it into the project's `results/data/` when the computation is expensive
  enough to be worth keeping — storing a CSV that takes two seconds to regenerate is just clutter.
  Put a `--persist` flag on it and mention the tradeoff to the user.
- Save **both** SVG and PDF to `results/figures/<name>.svg` and `.pdf`. The paper includes PDFs
  (there's no `svg` package and no `-shell-escape`), and the SVG is what gets promoted to
  `figures-wip/` when a figure needs hand work in Inkscape.
- Seed any randomness explicitly and record the seed, so the figure in the paper corresponds to a
  run that can be reproduced.
- Guard the entry point with `if __name__ == "__main__":`.

**Resolve the output directory without hardcoding a path out of the repo.** This repo is its own
git repo with its own remote and has to stay runnable on its own, so a hardcoded `../../results`
would break it for anyone who clones just the code. Walk up for a project-root marker and fall
back to a local directory:

```python
def project_root() -> Path | None:
    """Nearest ancestor containing CLAUDE.md, or None when the repo stands alone."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "CLAUDE.md").exists():
            return parent
    return None

def output_dir(override: Path | None = None) -> Path:
    if override:
        return override
    root = project_root()
    target = (root / "results" / "figures") if root else Path(__file__).parent / "out"
    target.mkdir(parents=True, exist_ok=True)
    return target
```

## Run it

Run the script and confirm both files exist and are non-empty before reporting success. A figure
script that imports cleanly but produces an empty axes is the common failure, and it looks like
success from the outside.

If the repo has a virtualenv (`venv/`), use it. If a dependency is missing, say so rather than
silently installing into the system environment.

## Check the output

Look at the rendered figure, don't just confirm the file was written. Specifically:

- **Check the true output width**, because this is what guarantees the type renders at the size it
  was specified: `pdfinfo <name>.pdf` should report ~251.06 bp for a single-column figure or
  ~514.3 bp for full width. Any other value means LaTeX will rescale it and every label with it.
- Do series follow the palette order, and does the baseline read as the baseline?
- Does y start at zero, and do the axis labels carry units only where units exist?
- Does anything overlap, or does constrained layout squeeze an axis to nothing?
- Would the figure still be readable printed in greyscale?
- For an SVG: does it contain real `<text>` elements? `grep -c '<text' <name>.svg` returning 0 means
  the style wasn't loaded and the file is useless for Inkscape editing.

Fix what you find before handing it over.

## Finish

Report the script path, both output paths, and how to re-run it — including the `--from-csv`
invocation, which is what they'll actually use next. Offer the LaTeX include block, matching the
paper's caption convention of a bold lead-in sentence followed by the explanation:

```latex
\begin{figure}          % figure* for full width
  \centering
  \includegraphics[width=\linewidth]{figures/plots/<name>.pdf}
  \caption{\textbf{<the claim in one sentence>.} <what the reader should notice>}
  \label{fig:<name>}
\end{figure}
```

Note that the PDF still has to be copied into `paper/figures/plots/` — either directly, or
via `figures-wip/` if it needs hand editing first. The script writes to `results/figures/`, which
the paper doesn't read from.
