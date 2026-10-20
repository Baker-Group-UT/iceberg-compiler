"""Placement (packed vs naive at the same P) at the tightest bin, per circuit and family.
Run from the exact-patches worktree with PYTHONPATH=.
"""
import numpy as np, pandas as pd
from pathlib import Path
from experiments.figures.density_ablation_sweep import density_references as dr
D = Path("experiments/figures/density_ablation_sweep")
matched = pd.read_csv(D / "density_ablation_sweep_split_std.csv")
table = dr.arm_values(matched, pd.read_csv(D / "density_ablation_sweep_split_std_recount.csv"),
                      pd.read_csv(D / "density_ablation_sweep_split_std_floors.csv"))
wide, info = dr._wide(table)
pd.set_option("display.width", 250)
f = dr._with_bins(info.reset_index())
f = f[np.isclose(f.plot_bin, 0.541667)].set_index(["circuit_name", "n_codes"])
def rel(q, lower):
    a, b = wide[(q, "packed")], wide[(q, "h_merge")]
    return (100 * ((b - a) / b if lower else (a - b) / b)).reindex(f.index)
out = pd.DataFrame({"group": f.size_group, "n": f.num_qubits,
                    "LSR": rel("lsr", False), "logical_err_reduction": rel("logical_error", True),
                    "TVD_reduction": rel("tvd", True), "2q_gates_reduction": rel("two_qubit_gates", True),
                    "depth_reduction": rel("depth", True)})
# expected logical errors per *shot* = LSR x logical error per kept shot: does the trade pay?
le_kept = lambda arm: wide[("logical_error", arm)].reindex(f.index)
lsr = lambda arm: wide[("lsr", arm)].reindex(f.index)
out["good_shots_ratio"] = (lsr("packed") * (1 - le_kept("packed"))) / (lsr("h_merge") * (1 - le_kept("h_merge")))
fam = out.index.get_level_values(0).str.extract(r"^([a-z_]+?)_n\d+$")[0].to_numpy()
out["family"] = fam
big = out[out.group == "large"].sort_values("LSR")
print(big.round(2).to_string())
print("\nby family (all groups, density ~1.85): mean of per-circuit % (packed vs naive at the same P)")
print(out.groupby("family")[["LSR", "logical_err_reduction", "TVD_reduction", "2q_gates_reduction", "depth_reduction"]].mean().round(1).sort_values("LSR").to_string())
# which planner path produced the packed arm? (fallback to naive, etc.)
cols = [c for c in matched.columns if any(k in c for k in ("fallback", "verif", "planner", "packed_commute", "commute"))]
print("\nplanner columns:", cols)
