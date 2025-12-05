from pathlib import Path
import json
from networkx.readwrite import json_graph
from dag.circuit_dag import draw_qubit_graph

dag_path = Path("circuit_sims/12_100_29372102/original_dag.json")
payload = json.loads(dag_path.read_text())
graph = json_graph.node_link_graph(payload, edges="links", multigraph=True, directed=True)

draw_qubit_graph(graph, orient="laneless")   # or "horiz"/"tower"
