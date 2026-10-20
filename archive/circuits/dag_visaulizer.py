"""
Plot a stored DAG with `dag.circuit_dag.draw_qubit_graph`.

Guarded behind `__main__` on purpose. `draw_qubit_graph` ends in a bare `plt.show()`,
and the default Matplotlib backend on macOS is interactive, so while this ran at module
level *importing* the file was enough to open a figure window -- an editor "run file",
a language server, a REPL or a stray import would all pop one, with nothing on screen
to say where it came from. `graph_visualzier.py` has the same call commented out, which
looks like the same thing being worked around rather than fixed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from networkx.readwrite import json_graph

from archive.dag.circuit_dag import draw_qubit_graph

DEFAULT_DAG = Path("circuit_sims/12_100_29372102/original_dag.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dag-json", type=Path, default=DEFAULT_DAG)
    parser.add_argument(
        "--orient", choices=("horiz", "laneless", "tower"), default="laneless"
    )
    args = parser.parse_args()

    payload = json.loads(args.dag_json.read_text())
    graph = json_graph.node_link_graph(
        payload, edges="links", multigraph=True, directed=True
    )
    draw_qubit_graph(graph, orient=args.orient)


if __name__ == "__main__":
    main()
