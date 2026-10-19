"""Is the + 2q merging segment at one qubit per patch seed noise? Run from the worktree, PYTHONPATH=.
"""
import numpy as np, pandas as pd
from pathlib import Path
from experiments.figures.density_ablation_sweep import density_references as dr
D = Path("experiments/figures/density_ablation_sweep")
table = dr.arm_values(pd.read_csv(D / "density_ablation_sweep_split_std.csv"),
                      pd.read_csv(D / "density_ablation_sweep_split_std_recount.csv"),
                      pd.read_csv(D / "density_ablation_sweep_split_std_floors.csv"))
seg = dr._with_bins(dr.ablation_segments(table))
wide, info = dr._wide(table)
f = dr._with_bins(info.reset_index()).set_index(["circuit_name", "n_codes"])
for center in sorted(f.plot_bin.unique()):
    idx = f.index[np.isclose(f.plot_bin, center)]
    same = np.ones(len(idx), bool)
    for q in ("one_qubit_gates", "two_qubit_gates", "depth"):
        same &= (wide.loc[idx, (q, "h_merge_1q")].to_numpy() == wide.loc[idx, (q, "h_merge")].to_numpy())
    le = seg[(seg.quantity == "logical_error") & np.isclose(seg.plot_bin, center)].set_index(["circuit_name", "n_codes"])["merging_2q"].reindex(idx)
    a, b = le[same], le[~same]
    sem = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else np.nan
    print(f"density {1/center:.2f}: identical translations {same.sum():3d}/{len(idx):3d}; 2q-merging logical-error segment "
          f"on identical: mean {a.mean():+.2f} (s.e. {sem:.2f}); on changed: mean {b.mean():+.2f} (n={len(b)})")
