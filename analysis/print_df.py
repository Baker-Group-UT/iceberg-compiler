"""
Concise DataFrame printing utilities.
"""

from __future__ import annotations
from typing import TextIO, Any, Mapping
import sys
import pandas as pd


def _to_list(maybe_iter: Any) -> list:
    try:
        return list(maybe_iter)
    except Exception:
        return [str(maybe_iter)]

def print_df_keys(df: Any, *, header: str | None = None, stream: TextIO = sys.stdout) -> None:
    """
    Print only the column names ("keys") of a pandas DataFrame.

    Accepts any object exposing `.columns` (preferred) or `.keys()`.

    Args:
        df: pandas.DataFrame or similar object with .columns / .keys().
        header: Optional label printed above the list (default: "DataFrame keys").
        stream: Output stream (default: sys.stdout).
    """
    # Prefer .columns; fall back to .keys() if needed.
    cols: list = []
    if hasattr(df, "columns"):
        cols = _to_list(getattr(df, "columns"))
    elif hasattr(df, "keys"):
        keys_attr = getattr(df, "keys")
        cols = _to_list(keys_attr() if callable(keys_attr) else keys_attr)

    title = header or "DataFrame keys"
    print(f"{title} (n={len(cols)})", file=stream)
    for i, c in enumerate(cols):
        print(f"  {i:>2}: {c}", file=stream)

def _format_gate_count_table(gc: Any, max_len: int = 80) -> str:
    """Compact 'gate: count' string, truncated to max_len characters."""
    if isinstance(gc, Mapping):
        parts = [f"{k}:{v}" for k, v in sorted(gc.items(), key=lambda kv: kv[0])]
        s = " ".join(parts)
    else:
        s = str(gc)
    return (s[: max_len - 1] + "…") if len(s) > max_len else s


def print_df_table(
    df: pd.DataFrame,
    *,
    header: str = "Run table",
    float_precision: int = 6,
    max_rows: int | None = None,
    sort_by: str | None = None,
    hide_permutation: bool = False,
    stream: TextIO = sys.stdout,
    show_index: bool = True,
) -> None:
    """
    Print a compact table (rows = runs) with your selected columns:

      permutation |
      # codes     (codes_used)
      fidelity
      logical_error_rate
      logical_fidelity
      selection rate        (logical_selection_rate)
      TVD                   (logical_total_variation_distance)
      gate count            (gate_count)
      contains faults       (contains_faults)
      fault tolerant        (fault_tolerant)
    """
    wanted = [
        "permutation",
        "codes_used",
        # "fidelity",
        # "logical_error_rate",
        "logical_fidelity",
        "logical_selection_rate",
        "logical_total_variation_distance",
        "code_equality_entropy",
        "depth_equality_entropy",
        # "total_gates",
        "one_qubit_gates",
        "two_qubit_gates",
        "circuit_depth",
        # "contains_faults",
        # "fault_tolerant",
        # "postselected_dist",
    ]
    if hide_permutation and "permutation" in wanted:
        wanted.remove("permutation")
    rename = {
        "codes_used": "# codes",
        "logical_selection_rate": "selection rate",
        "logical_total_variation_distance": "TVD",
        "code_equality_entropy": "code equality",
        "depth_equality_entropy": "depth equality",
        "one_qubit_gates": "#1q",
        "two_qubit_gates": "#2q",
        "total_gates": "gate count",
        "circuit_depth": "depth",
        "contains_faults": "contains faults",
        # "fault_tolerant": "fault tolerant",
    }

    cols = [c for c in wanted if c in df.columns]
    dfx = df.loc[:, cols].copy()
    if sort_by is not None and sort_by in dfx.columns:
        dfx = dfx.sort_values(sort_by, ascending=True)
    if max_rows is not None:
        dfx = dfx.head(max_rows)

    # Numeric formatting
    for c in (
        "fidelity",
        "logical_error_rate",
        "logical_fidelity",
        "logical_selection_rate",
        "logical_total_variation_distance",
        "code_equality_entropy",
        "depth_equality_entropy",
    ):
        if c in dfx.columns:
            dfx[c] = pd.to_numeric(dfx[c], errors="coerce").round(float_precision)

    # Gate count formatting
    if "gate_count" in dfx.columns:
        dfx["gate_count"] = dfx["gate_count"].apply(_format_gate_count_table)

    # Rename headers for readability
    dfx = dfx.rename(columns=rename)

    # Print
    n = len(dfx)
    print(f"{header} (n={n})", file=stream)
    print(dfx.to_string(index=show_index), file=stream)

def _fmt_float(x: Any, prec: int = 6) -> str:
    try:
        return f"{float(x):.{prec}f}"
    except Exception:
        return str(x)
    
def _format_gate_count(gc: Any) -> str:
    """
    Pretty-print a dict-like gate count (sorted by key for stability).
    Accepts dict, Mapping, or string.
    """
    if isinstance(gc, Mapping):
        # Stable alphabetical sort; flip to sort by count if you prefer.
        items = sorted(gc.items(), key=lambda kv: kv[0])
        return ", ".join(f"{k}: {v}" for k, v in items)
    if isinstance(gc, str):
        return gc
    # Fallback
    return str(gc)

def print_df_verbose(
    df,
    *,
    header: str = "Run summaries",
    float_precision: int = 6,
    stream: TextIO = sys.stdout,
    max_rows: int | None = None,
    sort_by: str | None = None,
    hide_permutation: bool = False,
) -> None:
    """
    Print selected fields per row, with custom labels:

      permutation
      codes_used               -> "# codes"
      fidelity
      logical_error_rate
      logical_fidelity
      logical_selection_rate   -> "selection rate"
      logical_total_variation_distance -> "TVD"
      gate_count               (pretty-printed mapping)
      contains_faults          -> "contains faults"
      fault_tolerant           -> "fault tolerant"
    """
    # Column name -> printed label
    cols = [
        ("permutation", "permutation"),
        ("codes_used", "# codes"),
        ("one_qubit_gates", "#1q"),
        ("two_qubit_gates", "#2q"),
        ("circuit_depth", "depth"),
        ("fidelity", "fidelity"),
        ("logical_error_rate", "logical_error_rate"),
        ("logical_fidelity", "logical_fidelity"),
        ("logical_selection_rate", "selection rate"),
        ("logical_total_variation_distance", "TVD"),
        ("code_equality_entropy", "code equality"),
        ("depth_equality_entropy", "depth equality"),
        ("gate_count", "gate count"),
        ("contains_faults", "contains faults"),
        ("fault_tolerant", "fault tolerant"),
    ]
    if hide_permutation:
        cols = [entry for entry in cols if entry[0] != "permutation"]

    n = len(df) if hasattr(df, "__len__") else None
    hdr = f"{header}" + (f" (n={n})" if n is not None else "")
    print(hdr, file=stream)

    if sort_by is not None and hasattr(df, "sort_values") and sort_by in getattr(df, "columns", []):
        df = df.sort_values(sort_by, ascending=True)
    iterable = df.iterrows() if hasattr(df, "iterrows") else enumerate(_to_list(df))
    for idx, row in iterable:
        if max_rows is not None and idx >= max_rows:
            print(f"... ({n - max_rows} more rows)", file=stream)
            break

        # Pull values safely
        def get(col):
            try:
                return row[col] if hasattr(row, "__getitem__") else getattr(row, col)
            except Exception:
                return None

        # Build lines
        perm = get("permutation")
        codes = get("codes_used")
        one_q = get("one_qubit_gates")
        two_q = get("two_qubit_gates")
        depth = get("circuit_depth")
        fid = _fmt_float(get("fidelity"), float_precision)
        l_err = _fmt_float(get("logical_error_rate"), float_precision)
        l_fid = _fmt_float(get("logical_fidelity"), float_precision)
        sel = _fmt_float(get("logical_selection_rate"), float_precision)
        tvd = _fmt_float(get("logical_total_variation_distance"), float_precision)
        work_conc = _fmt_float(get("code_equality_entropy"), float_precision)
        depth_eq = _fmt_float(get("depth_equality_entropy"), float_precision)
        gcount = _format_gate_count(get("gate_count"))
        faults = get("contains_faults")
        ft = get("fault_tolerant")

        if not hide_permutation:
            print(f"[{idx:>3}] permutation: {perm}", file=stream)
        print(f"      # codes: {codes}", file=stream)
        print(f"      #1q: {one_q} | #2q: {two_q} | depth: {depth}", file=stream)
        print(f"      fidelity: {fid} | logical_fidelity: {l_fid} | logical_error_rate: {l_err}", file=stream)
        print(
            f"      selection rate: {sel} | TVD: {tvd} | code equality: {work_conc} | depth equality: {depth_eq}",
            file=stream,
        )
        print(f"      gate count: {gcount}", file=stream)
        print(f"      contains faults: {faults} | fault tolerant: {ft}", file=stream)
