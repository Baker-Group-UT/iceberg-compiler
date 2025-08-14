# permutations/results_manager.py
from __future__ import annotations

import json, gzip, hashlib, os, re, datetime as _dt
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import cirq


# ---------- paths & filenames ----------

def _results_dir(base_dir: str = "results/permutations") -> str:
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def _sanitize(name: str) -> str:
    # keep simple, filesystem-safe names
    name = name.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_.-]", "", name) or "circuit"

def _now_stamp() -> str:
    # UTC stamp without colons for safe filenames
    return _dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")

def _make_filename(circuit_name: str, repetitions: int, ts: Optional[str] = None) -> str:
    ts = ts or _now_stamp()
    return f"{ts}_{_sanitize(circuit_name)}_R{int(repetitions)}.json.gz"

# ---------- circuit serialization ----------

def _circuit_to_openqasm(circuit: cirq.Circuit) -> Tuple[str, str]:
    """
    Try to export to OpenQASM 2.0; fall back to Cirq JSON if QASM not supported.
    Returns (payload, format), where format is 'openqasm2' or 'cirq_json'.
    """
    try:
        # Cirq has a qasm() exporter in modern versions.
        qasm = cirq.qasm(circuit)  # type: ignore[attr-defined]
        if isinstance(qasm, bytes):
            qasm = qasm.decode("utf-8")
        return qasm, "openqasm2"
    except Exception:
        # Fallback: Cirq JSON is lossless for Cirq circuits
        return cirq.to_json(circuit), "cirq_json"

def _hash_payload(s: str) -> str:
    h = hashlib.sha256()
    h.update(s.encode("utf-8"))
    return h.hexdigest()

# ---------- saving & loading ----------

def save_permutation_run(
    *,
    circuit: cirq.Circuit,
    circuit_name: str,
    df: pd.DataFrame,
    repetitions: int,
    base_dir: str = "results/permutations",
    extra_meta: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save a complete run to results/permutations as a single .json.gz.

    Args:
        circuit: the *program* (logical) circuit used to generate permutations.
        circuit_name: friendly name for the circuit (used in filename).
        df: the final analysis DataFrame (will be stored losslessly).
        repetitions: number of MC shots used in effectiveness simulation.
        base_dir: base folder to write into (default: results/permutations).
        extra_meta: any additional metadata to persist.
        options: optional dict of run options (n_code, faulty, metric, etc).

    Returns:
        Full path to the written file.
    """
    base = _results_dir(base_dir)
    filename = _make_filename(circuit_name, repetitions)
    path = os.path.join(base, filename)

    # circuit payload
    circ_payload, circ_fmt = _circuit_to_openqasm(circuit)

    # dataframe payload: use records for simplicity & portability
    df_payload = df.to_dict(orient="records")

    meta: Dict[str, Any] = {
        "schema": 1,
        "created_utc": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "circuit_name": circuit_name,
        "repetitions": int(repetitions),
        "circuit_format": circ_fmt,
        "options": options or {},
        "extra_meta": extra_meta or {},
    }

    # integrity hash across the essential parts
    integrity_src = json.dumps(
        {
            "circuit": circ_payload,
            "circuit_format": circ_fmt,
            "df_columns": list(df.columns),
            "df_len": len(df_payload),
            "meta": meta,
        },
        sort_keys=True,
    )
    meta["integrity_sha256"] = _hash_payload(integrity_src)

    packed = {
        "meta": meta,
        "circuit": circ_payload,
        "dataframe": {
            "columns": list(df.columns),
            "records": df_payload,
        },
    }

    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(packed, f, ensure_ascii=False)

    return path

def load_run(path: str) -> Dict[str, Any]:
    """
    Load a saved run (.json or .json.gz). Returns:
      {
        'meta': dict,
        'circuit': {'format': 'openqasm2'|'cirq_json', 'text': str},
        'df': pandas.DataFrame,
      }
    """
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        obj = json.load(f)

    meta = obj.get("meta", {})
    circ_text = obj.get("circuit", "")
    circ_fmt = meta.get("circuit_format", "openqasm2")

    df_obj = obj.get("dataframe", {})
    columns = df_obj.get("columns", [])
    records = df_obj.get("records", [])
    df = pd.DataFrame.from_records(records, columns=columns if columns else None)

    return {
        "meta": meta,
        "circuit": {"format": circ_fmt, "text": circ_text},
        "df": df,
    }

def list_runs(
    base_dir: str = "results/permutations",
    circuit_name: Optional[str] = None,
) -> List[str]:
    """
    Return full paths to saved runs (sorted newest→oldest).
    Optionally filter by circuit_name.
    """
    base = _results_dir(base_dir)
    files = [os.path.join(base, f) for f in os.listdir(base) if f.endswith((".json", ".json.gz"))]
    if circuit_name:
        sn = _sanitize(circuit_name)
        files = [p for p in files if f"_{sn}_" in os.path.basename(p)]
    return sorted(files, key=os.path.getmtime, reverse=True)

def load_latest_run(
    base_dir: str = "results/permutations",
    circuit_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Convenience: load most recent run (optionally filtered by circuit_name).
    Returns None if no runs found.
    """
    paths = list_runs(base_dir=base_dir, circuit_name=circuit_name)
    if not paths:
        return None
    return load_run(paths[0])

# ---------- tiny utility: rebuild cirq.Circuit if you stored cirq_json ----------

def circuit_from_saved(circ_info: Dict[str, str]) -> Optional[cirq.Circuit]:
    """
    If you stored the circuit in 'cirq_json' format, reconstruct a Cirq Circuit.
    For 'openqasm2', we return None (left to the caller to import/convert if needed).
    """
    fmt = circ_info.get("format")
    text = circ_info.get("text", "")
    if fmt == "cirq_json":
        try:
            return cirq.read_json(json_text=text)
        except Exception:
            return None
    return None