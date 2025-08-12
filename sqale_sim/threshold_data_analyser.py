import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
print("Matplotlib backend:", matplotlib.get_backend())

def load_data(path):
    with open(path, "r") as f:
        content = json.load(f)

    # Check for manifest
    if isinstance(content, dict):
        manifest = content.get("_manifest", {})
        data = content.get("data", [])
    else:
        manifest = {}
        data = content

    return manifest, data

def plot_fidelity_surface(data, title="Post-Selected Fidelity vs Noise Scalars"):
    rz_vals = sorted(set(d["rz_scalar"] for d in data))
    cz_vals = sorted(set(d["cz_scalar"] for d in data))

    rz_grid, cz_grid = np.meshgrid(rz_vals, cz_vals)
    fidelity_grid = np.zeros_like(rz_grid, dtype=float)

    for d in data:
        i = cz_vals.index(d["cz_scalar"])
        j = rz_vals.index(d["rz_scalar"])
        fidelity_grid[i, j] = d["fidelity"]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(rz_grid, cz_grid, fidelity_grid, cmap="viridis")

    ax.set_xlabel("RZ Scalar")
    ax.set_ylabel("CZ Scalar")
    ax.set_zlabel("Post-Selected Fidelity")
    ax.set_title(title)

    plt.tight_layout()
    # plt.ion() # causin problems rn
    plt.show()

def build_fidelity_grid(data):
    rz_vals = sorted(set(d["rz_scalar"] for d in data))
    cz_vals = sorted(set(d["cz_scalar"] for d in data))

    rz_grid, cz_grid = np.meshgrid(rz_vals, cz_vals)
    fidelity_grid = np.zeros_like(rz_grid, dtype=float)

    for d in data:
        i = cz_vals.index(d["cz_scalar"])
        j = rz_vals.index(d["rz_scalar"])
        fidelity_grid[i, j] = d["fidelity"]

    return rz_vals, cz_vals, rz_grid, cz_grid, fidelity_grid


def plot_fidelity_comparison_surface(data1, data2, label1="Faulty", label2="FT"):
    rz_vals1, cz_vals1, rz_grid1, cz_grid1, grid1 = build_fidelity_grid(data1)
    rz_vals2, cz_vals2, rz_grid2, cz_grid2, grid2 = build_fidelity_grid(data2)

    assert rz_vals1 == rz_vals2 and cz_vals1 == cz_vals2, "Grids must align for overlay"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    surf1 = ax.plot_surface(rz_grid1, cz_grid1, grid1, alpha=0.6, cmap="viridis", label=label1)
    surf2 = ax.plot_surface(rz_grid2, cz_grid2, grid2, alpha=0.6, cmap="plasma", label=label2)

    ax.set_xlabel("RZ Scalar")
    ax.set_ylabel("CZ Scalar")
    ax.set_zlabel("Post-Selected Fidelity")
    ax.set_title(f"{label1} vs {label2} Fidelity Surfaces")

    # Create custom legend
    ax.legend([surf1, surf2], [label1, label2], loc="upper right")

    plt.tight_layout()
    plt.show()


def get_intersection_points(data1, data2, atol=1e-3):
    by_coords = lambda data: {
        (d["rz_scalar"], d["cz_scalar"]): d["fidelity"] for d in data
    }

    map1 = by_coords(data1)
    map2 = by_coords(data2)

    shared_coords = set(map1.keys()) & set(map2.keys())

    intersections = [
        (rz, cz, map1[(rz, cz)])
        for (rz, cz) in shared_coords
        if abs(map1[(rz, cz)] - map2[(rz, cz)]) <= atol
    ]

    return intersections


# --------------------------------------------------------- #
#                     Difference Grid                       #
# --------------------------------------------------------- #
# ---------- Add these to your threshold_data_analyser.py ----------

def _build_coord_map(data):
    """(rz, cz) -> fidelity"""
    return {(d["rz_scalar"], d["cz_scalar"]): d["fidelity"] for d in data}

def build_diff_grid(data_faulty, data_ft):
    """
    Returns aligned grids and the difference surface:
    Δ = fidelity_ft - fidelity_faulty
    """
    rz_vals_f = sorted(set(d["rz_scalar"] for d in data_faulty))
    cz_vals_f = sorted(set(d["cz_scalar"] for d in data_faulty))
    rz_vals_t = sorted(set(d["rz_scalar"] for d in data_ft))
    cz_vals_t = sorted(set(d["cz_scalar"] for d in data_ft))

    if rz_vals_f != rz_vals_t or cz_vals_f != cz_vals_t:
        raise ValueError("Faulty and FT sweeps must share identical rz/cz grids.")

    rz_vals, cz_vals = rz_vals_f, cz_vals_f
    rz_grid, cz_grid = np.meshgrid(rz_vals, cz_vals)

    map_f  = _build_coord_map(data_faulty)
    map_ft = _build_coord_map(data_ft)

    grid_faulty = np.zeros_like(rz_grid, dtype=float)
    grid_ft     = np.zeros_like(rz_grid, dtype=float)
    diff_grid   = np.zeros_like(rz_grid, dtype=float)

    for i, cz in enumerate(cz_vals):
        for j, rz in enumerate(rz_vals):
            f_f  = map_f[(rz, cz)]
            f_ft = map_ft[(rz, cz)]
            grid_faulty[i, j] = f_f
            grid_ft[i, j]     = f_ft
            diff_grid[i, j]   = f_ft - f_f

    return rz_vals, cz_vals, rz_grid, cz_grid, grid_faulty, grid_ft, diff_grid

def plot_fidelity_difference_surface(
    data_faulty,
    data_ft,
    *,
    title="Δ Fidelity (FT − Faulty) vs Noise Scalars",
    zero_plane=True,
):
    rz_vals, cz_vals, rz_grid, cz_grid, _, _, diff_grid = build_diff_grid(data_faulty, data_ft)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Diverging colormap: positive (FT better), negative (Faulty better)
    surf = ax.plot_surface(rz_grid, cz_grid, diff_grid, cmap="seismic", alpha=0.9)

    # Optional zero reference plane
    if zero_plane:
        z0 = np.zeros_like(diff_grid)
        ax.plot_surface(rz_grid, cz_grid, z0, color="gray", alpha=0.15, linewidth=0)

    ax.set_xlabel("RZ Scalar")
    ax.set_ylabel("CZ Scalar")
    ax.set_zlabel("Δ Fidelity (FT − Faulty)")
    ax.set_title(title)

    # Colorbar
    fig.colorbar(surf, shrink=0.6, aspect=12, pad=0.08, label="Δ Fidelity")

    plt.tight_layout()
    plt.show()

def print_diff_extrema(data_faulty, data_ft, top=5):
    """Print top improvements and regressions (largest |Δ|)."""
    rz_vals, cz_vals, _, _, _, _, diff = build_diff_grid(data_faulty, data_ft)

    rows = []
    for i, cz in enumerate(cz_vals):
        for j, rz in enumerate(rz_vals):
            rows.append((rz, cz, float(diff[i, j])))

    rows.sort(key=lambda x: x[2], reverse=True)
    gains = rows[:top]
    rows.sort(key=lambda x: x[2])  # ascending → most negative
    losses = rows[:top]

    print(f"\nTop {top} improvements (FT better):")
    for rz, cz, d in gains:
        print(f"  rz={rz}, cz={cz}  Δ={d:+.4f}")

    print(f"\nTop {top} regressions (Faulty better):")
    for rz, cz, d in losses:
        print(f"  rz={rz}, cz={cz}  Δ={d:+.4f}")


# if __name__ == "__main__":
#     results_dir = Path("results/test")

#     # Load both runs
#     manifest1, data1 = load_data(results_dir / "faulty_run.json")
#     manifest2, data2 = load_data(results_dir / "ft_run.json")

#     print("Faulty Manifest:", manifest1)
#     print("FT Manifest:", manifest2)

#     plot_fidelity_comparison_surface(data1, data2, label1="Faulty", label2="FT")

#     intersect = get_intersection_points(data1, data2, atol=1e-3)
#     print(f"Intersection points (fidelity difference ≤ 1e-3):")
#     for rz, cz, fid in intersect:
#         print(f"  rz={rz}, cz={cz}  →  fidelity ≈ {fid:.4f}")

if __name__ == "__main__":
    results_dir = Path("results/test")
    _, data_faulty = load_data(results_dir / "faulty_run.json")
    _, data_ft     = load_data(results_dir / "ft_run.json")

    plot_fidelity_difference_surface(data_faulty, data_ft)
    print_diff_extrema(data_faulty, data_ft, top=5)