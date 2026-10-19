"""
compiler_stack.py -- Fig. 5: where AQORN sits in the compiler stack.

A hand-style diagram rather than a plot: there is no data and no CSV cache. It lives
under experiments/figures/ anyway so the paper's art is regenerable from one command
and inherits the house style file rather than a second set of hardcoded sizes.

WHAT THE FIGURE HAS TO ARGUE
----------------------------
Why are placement and routing from other compilers not the baseline? The answer is that they decide a different variable, so this figure labels
every stage with the variable it owns:

    circuit compiler  -> which gates
    AQORN             -> which encoding
    device mapping    -> which hardware qubits
    post-selection    -> which shots to keep

Read across, the four are disjoint. That is the whole argument, and it is carried by the
sub-labels, not by the boxes.

The greyed dashed box above AQORN is the second half of the claim. The encoding step is
not absent from today's stack -- it is present and naive (one program qubit per patch,
gates encoded one at a time). AQORN replaces that default rather than displacing a
neighbour, which is why the gain composes with whatever the stages on either side do.
Drawing the naive step explicitly is what makes "free" visible instead of asserted; a
plain four-box row would leave the reader to take it on the caption's word.

The post-selection tail is drawn faded because it is runtime, not compilation. It is
included only because decoding and scheduling overhead is a natural question,
and the figure should show that AQORN does not reach into that stage either.

GEOMETRY
--------
Drawn at exactly one IEEEtran column (3.487in) so \\includegraphics[width=\\linewidth]
scales 1:1 and 8pt type in the figure renders as 8pt on the page. All coordinates below
are in inches inside an equal-aspect axes filling the whole canvas, so the numbers here
are the numbers on paper. Do not change figsize without changing the limits with it.

Usage:
    PYTHONPATH=. python -m experiments.figures.compiler_stack.compiler_stack
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parents[3]
STYLE = REPO / "experiments" / "shared" / "aqorn.mplstyle"
OUT = REPO / "results" / "figures"

# --- palette, from docs/figure-style.md ------------------------------------------
ACCENT = "#0097fd"  # series 1: AQORN, the one thing being proposed
DARK = "#3b4c56"    # series 2: the stages that already exist
MID = "#636363"     # reserved for reference material -- here, the sub-labels
LIGHT = "#bdbdbd"   # separators and the faded runtime stage

# --- canvas ----------------------------------------------------------------------
W, H = 3.487, 1.45

MARGIN = 0.06
GAP = 0.15            # horizontal space for the arrow between two boxes
BOX_W = (W - 2 * MARGIN - 3 * GAP) / 4  # = 0.729in
BOX_H = 0.44
BOX_BOT = 0.33
BOX_TOP = BOX_BOT + BOX_H

SUB_TOP = BOX_BOT - 0.05   # sub-labels hang below their box
ANNOT_BOT = 0.96           # the dashed "today" box floats above the row
ANNOT_TOP = 1.42

#: (label, sub-label, role). Role drives colour: "live" stages exist today and are
#: drawn in the neutral dark, "aqorn" is the proposal, "runtime" is faded because it
#: happens after compilation and is shown only for completeness.
#:
#: Box labels are kept to at most nine characters a line so 9pt Times fits inside a
#: 0.729in box without touching the border. The precision they give up -- "device
#: routing" is really mapping, routing and scheduling; "decoding" is really decoding
#: and post-selection -- is recovered by the sub-label and the caption. Widening the
#: labels instead would mean dropping below the 8pt floor.
STAGES = [
    ("circuit\ncompiler", "which gates", "live"),
    ("AQORN", "which encoding", "aqorn"),
    ("device\nrouting", "which hardware\nqubits", "live"),
    ("decoding", "which shots\nto keep", "runtime"),
]


def box_left(i: int) -> float:
    return MARGIN + i * (BOX_W + GAP)


def box_center(i: int) -> float:
    return box_left(i) + BOX_W / 2


def draw_box(ax, i: int, label: str, role: str) -> None:
    """One stage box. Only AQORN is filled; the rest are outlines so the eye lands
    on the proposed step first and reads the existing stack as context."""
    if role == "aqorn":
        edge, face, text_c, weight, size = ACCENT, ACCENT, "white", "bold", 10
    elif role == "runtime":
        edge, face, text_c, weight, size = LIGHT, "white", LIGHT, "normal", 9
    else:
        edge, face, text_c, weight, size = DARK, "white", DARK, "normal", 9

    ax.add_patch(
        FancyBboxPatch(
            (box_left(i), BOX_BOT),
            BOX_W,
            BOX_H,
            boxstyle="round,pad=0,rounding_size=0.035",
            linewidth=0.8,
            edgecolor=edge,
            facecolor=face,
            zorder=2,
        )
    )
    ax.text(
        box_center(i),
        BOX_BOT + BOX_H / 2,
        label,
        ha="center",
        va="center",
        fontsize=size,
        color=text_c,
        fontweight=weight,
        linespacing=1.25,
        zorder=3,
    )


def draw_arrow(ax, x0: float, y0: float, x1: float, y1: float, color: str) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=6,
            linewidth=0.8,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=1,
        )
    )


def build() -> plt.Figure:
    plt.style.use(STYLE)
    # Explicit layout=None: the style file turns constrained_layout on for plots, which
    # would refit an axes we are deliberately pinning to the full canvas.
    fig = plt.figure(figsize=(W, H), layout=None)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    for i, (label, sub, role) in enumerate(STAGES):
        draw_box(ax, i, label, role)
        # The sub-label is the load-bearing part of the figure: four disjoint decision
        # variables, read across. Italic and grey so it reads as annotation, not as a
        # second row of stages.
        ax.text(
            box_center(i),
            SUB_TOP,
            sub,
            ha="center",
            va="top",
            fontsize=8,
            style="italic",
            color=LIGHT if role == "runtime" else MID,
            linespacing=1.2,
        )

    # Flow arrows. The last one fades because the stage it enters is runtime.
    for i in range(3):
        draw_arrow(
            ax,
            box_left(i) + BOX_W,
            BOX_BOT + BOX_H / 2,
            box_left(i + 1),
            BOX_BOT + BOX_H / 2,
            LIGHT if i == 2 else DARK,
        )

    # --- input ------------------------------------------------------------------
    # Sits above the first box rather than to its left: at column width there is no
    # horizontal room for a fifth element, and stacking it mirrors the dashed
    # annotation on the right so the two read as one annotation layer.
    ax.text(
        box_center(0),
        ANNOT_TOP,
        "program +\nnoise model",
        ha="center",
        va="top",
        fontsize=8,
        style="italic",
        color=MID,
        linespacing=1.2,
    )
    draw_arrow(ax, box_center(0), ANNOT_TOP - 0.30, box_center(0), BOX_TOP, MID)

    # --- the step AQORN replaces ------------------------------------------------
    # Narrow enough to clear the input label on its left: three short lines rather
    # than two long ones is what buys that clearance.
    annot_w = 1.12
    # Nudged right of centre so the widest line clears the border on both sides while
    # the box still clears the input label on its left. The arrow stays on AQORN's
    # centre line, which is what the reader tracks.
    annot_x = box_center(1) - annot_w / 2 + 0.05
    ax.add_patch(
        FancyBboxPatch(
            (annot_x, ANNOT_BOT),
            annot_w,
            ANNOT_TOP - ANNOT_BOT,
            boxstyle="round,pad=0,rounding_size=0.035",
            linewidth=0.8,
            linestyle=(0, (3, 2)),
            edgecolor=LIGHT,
            facecolor="white",
            zorder=2,
        )
    )
    ax.text(
        annot_x + annot_w / 2,
        (ANNOT_BOT + ANNOT_TOP) / 2,
        "today: naive encoding\none qubit per patch,\none gate at a time",
        ha="center",
        va="center",
        fontsize=8,
        style="italic",
        color=MID,
        linespacing=1.25,
        zorder=3,
    )
    draw_arrow(ax, box_center(1), ANNOT_BOT, box_center(1), BOX_TOP, LIGHT)
    ax.text(
        box_center(1) + 0.06,
        (ANNOT_BOT + BOX_TOP) / 2,
        "replaced by",
        ha="left",
        va="center",
        fontsize=8,
        style="italic",
        color=LIGHT,
    )

    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig = build()
    for ext in ("svg", "pdf"):
        path = OUT / f"compiler_stack.{ext}"
        fig.savefig(path)
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
