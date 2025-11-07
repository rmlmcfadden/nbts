import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from simulation.quantities import J_c


base_dir = Path("experiments/2025-05-15_const_2a1de1d/results_const_temp")
os.makedirs(base_dir, exist_ok=True)

all_dirs = [d for d in base_dir.glob("sim_t*_T*")]

for d in all_dirs:
    data_dir = os.path.join(d, "data")
    plot_dir = os.path.join(d, "plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Processing directory: {d}")

    # Load penetration_depth_corrected.csv from the data directory into a numpy array
    penetration_depth_file = os.path.join(data_dir, "penetration_depth_corrected.csv")

    # Read both columns and split them into two NumPy arrays
    df = pd.read_csv(
        penetration_depth_file, usecols=["x", "penetration_depth_corrected"]
    )

    x_vals = df["x"].to_numpy()  # 1‑D array of x positions
    penetration_depth_data = df[
        "penetration_depth_corrected"
    ].to_numpy()  # 1‑D array of depths

    # Calculate the critical current density using J_c function
    J_c_values = J_c(penetration_depth_data)
    # Save x_vals and J_c_values to a CSV file
    output_file = os.path.join(data_dir, "critical_current_density.csv")
    output_df = pd.DataFrame({"x": x_vals, "critical_current_density": J_c_values})
    output_df.to_csv(output_file, index=False)
    print(f"Critical current density saved to: {output_file}")

    current_density_file = os.path.join(data_dir, "current_density_corrected.csv")
    # Read both columns and split them into two NumPy arrays
    df_current = pd.read_csv(
        current_density_file,
        usecols=["x", "current_density_corrected", "J_clean_corr", "J_dirty_corr"],
    )
    x_vals_current = df_current["x"].to_numpy()  # 1‑D array of x positions
    current_density_data = df_current[
        "current_density_corrected"
    ].to_numpy()  # 1‑D array of current density
    J_clean_corr = df_current[
        "J_clean_corr"
    ].to_numpy()  # 1‑D array of clean current density
    J_dirty_corr = df_current[
        "J_dirty_corr"
    ].to_numpy()  # 1‑D array of dirty current density

    # ─── set up a 2‑row layout ──────────────────────────────────────────────
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(4.8, 4.8), constrained_layout=True
    )

    # ─── upper panel: measured / clean / dirty current densities ───────────
    ax_top.plot(x_vals_current, current_density_data / 1e11, label="Current density")
    ax_top.plot(x_vals_current, J_clean_corr / 1e11, label="J clean", linestyle="--")
    ax_top.plot(x_vals_current, J_dirty_corr / 1e11, label="J dirty", linestyle=":")
    ax_top.set_ylabel("J  [10^11 A m⁻²]")
    ax_top.set_xlim(0, 150)  # set x limits to the last x value
    ax_top.set_title("Current density profiles")
    ax_top.legend(frameon=False)
    ax_top.grid(alpha=0.3)

    # ─── lower panel: critical current density ─────────────────────────────
    ax_bot.plot(x_vals, J_c_values / 1e11, color="tab:red")
    ax_bot.set_xlabel("Depth x  [nm]")
    ax_bot.set_ylabel("J_c  [10^11 A m⁻²]")
    ax_bot.set_xlim(0, 150)  # set x limits to the last x value
    ax_bot.set_title("Critical current density")
    ax_bot.grid(alpha=0.3)

    # ─── save to the plots directory ───────────────────────────────────────
    plot_path = os.path.join(plot_dir, "critical_current.pdf")
    fig.savefig(plot_path)
    plt.close(fig)  # free memory when looping
    print(f"Figure saved → {plot_path}")
