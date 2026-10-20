"""Utility for visualising stored DAGs and exporting them to GML."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph

from archive.dag.circuit_dag import draw_qubit_graph


def _load_dag(path: Path) -> nx.MultiDiGraph:
    payload = json.loads(path.read_text())
    return json_graph.node_link_graph(
        payload,
        edges="links",
        multigraph=True,
        directed=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise a stored DAG JSON and export it to GML."
    )
    parser.add_argument(
        "--dag_json",
        default="circuit_sims/12_100_29372102/original_dag.json",
        type=Path,
        help="Path to the networkx.node_link_data JSON file.",
    )
    parser.add_argument(
        "--orient",
        choices=("horiz", "laneless", "tower"),
        default="laneless",
        help="Layout to use when plotting the DAG.",
    )
    parser.add_argument(
        "--gml-path",
        default="custom.gml",
        type=Path,
        help="Optional explicit destination for the exported .gml file.",
    )
    parser.add_argument(
        "--skip-plot",
        default=True,
        action="store_true",
        help="Skip Matplotlib rendering and only export GML.",
    )
    args = parser.parse_args()

    graph = _load_dag(args.dag_json)

    # if args.skip_plot:
    #     draw_qubit_graph(graph, orient=args.orient)

    gml_path = args.gml_path or args.dag_json.with_suffix(".gml")
    gml_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gml(graph, gml_path)
    print(f"Saved DAG to {gml_path}")


if __name__ == "__main__":
    main()
